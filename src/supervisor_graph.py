"""
Supervisor 多 Agent 主图。

图结构：
  START → analyze → route → [worker1, worker2, ...]（并行）
                ↑                       │
                └───────────────────────┘
                │ (DONE)
                ▼
           aggregate → quality_check → END
"""
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from src.config import get_config
from src.system_prompts_multi import SUPERVISOR_PROMPT
from src.workers import WORKERS


# ====== 状态定义 ======

class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    analyze_count: int  # 防止无限循环


# ====== 模型 ======

def _build_model(temp: float = 0.3):
    config = get_config()
    return ChatOpenAI(
        model=config.deepseek_model,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        temperature=temp,
    )


# ====== 质检规则引擎 ======

DANGEROUS_PHRASES = [
    "不用加热", "变质", "发霉", "生鸡肉直接吃", "不用洗手",
]

HIGH_RISK_TERMS = ["鸡翅", "鸡肉", "禽肉", "五花肉", "肉类", "海鲜"]


def _quality_check(answer: str) -> str:
    """规则引擎质检，给分析节点参考。"""
    issues = []
    for phrase in DANGEROUS_PHRASES:
        if phrase in answer:
            negations = ["不要", "不能", "禁止", "切勿", "请勿", "避免"]
            idx = answer.index(phrase)
            before = answer[max(0, idx - 8):idx]
            if not any(n in before for n in negations):
                issues.append(f"包含危险用语「{phrase}」")

    for term in HIGH_RISK_TERMS:
        if term in answer and "熟" not in answer and "加热" not in answer:
            issues.append(f"涉及{term}但未提醒充分加热")

    if issues:
        return "安全检查不通过：" + "；".join(issues)
    return "安全检查通过"


# ====== 构建 Supervisor 图 ======

def build_supervisor():
    model = _build_model()

    # ------ 分析 Prompt ------

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("human",
         "请分析以上对话，输出需要派遣哪些 Worker。\n"
         "可选: recipe_qa_worker, recommend_worker, safety_worker\n"
         "只输出 Worker 名称（逗号分隔）或 'DONE'。\n"
         "例如：'recipe_qa_worker' 或 'recipe_qa_worker, recommend_worker' 或 'DONE'"),
    ])

    # ------ 1. 分析节点 ------

    def analyze_node(state: SupervisorState) -> dict:
        """Supervisor 分析节点：决定需要哪些 Worker。"""
        count = state.get("analyze_count", 0)

        # 防止无限循环：最多 4 轮分析
        if count >= 4:
            return {
                "messages": [AIMessage(content="DONE")],
                "analyze_count": 0,
            }

        # 检查上轮是否已有安全检查结果
        last_content = str(state["messages"][-1].content) if state["messages"] else ""
        if last_content.startswith("安全检查不通过"):
            # 安全不通过，退回给 recipe_qa_worker 重新回答
            return {
                "messages": [AIMessage(content="recipe_qa_worker")],
                "analyze_count": count + 1,
            }

        # 首次分析或 Worker 返回后的再分析
        result = (analysis_prompt | model).invoke({"messages": state["messages"]})
        return {
            "messages": [result],
            "analyze_count": count + 1,
        }

    # ------ 2. 路由函数 ------

    def route_after_analyze(state: SupervisorState) -> list[Send] | str:
        """解析分析结果，决定并行发送到哪些 Worker。"""
        last_msg = state["messages"][-1]
        content = str(last_msg.content).strip().lower()

        if "done" in content:
            return "aggregate"

        selected = []
        if "recipe_qa_worker" in content:
            selected.append("recipe_qa_worker")
        if "recommend_worker" in content:
            selected.append("recommend_worker")
        if "safety_worker" in content:
            selected.append("safety_worker")

        if not selected:
            # 默认至少派遣 recipe_qa_worker
            return [Send("recipe_qa_worker", {"messages": state["messages"]})]

        # 并行分发
        return [
            Send(node=name, arg={"messages": state["messages"]})
            for name in selected
        ]

    # ------ 3. 汇总节点 ------

    def aggregate_node(state: SupervisorState) -> dict:
        """汇总 Worker 结果，运行规则质检，生成最终回答。"""
        # 收集 Worker 的输出
        parts: list[str] = []
        for msg in state["messages"]:
            if msg.type == "ai" and msg.content:
                content = str(msg.content)
                # 跳过 Supervisor 的分析消息和 Worker 内部工具调用
                if any(kw in content for kw in ["recipe_qa_worker", "recommend_worker",
                                                 "safety_worker", "DONE"]):
                    continue
                if content.startswith("安全检查"):
                    continue
                parts.append(content)

        # 找到最后一个安全的非分析 AI 回答
        final_answer = ""
        for msg in reversed(state["messages"]):
            if msg.type == "ai" and msg.content:
                content = str(msg.content)
                skip_keywords = ["recipe_qa_worker", "recommend_worker",
                                 "safety_worker", "DONE", "安全检查"]
                if not any(kw in content for kw in skip_keywords):
                    final_answer = content
                    break

        if not final_answer:
            final_answer = "抱歉，处理过程中出现了问题。请重新提问。"

        # 运行质检
        qc_result = _quality_check(final_answer)

        if qc_result != "安全检查通过":
            final_answer = f"{final_answer}\n\n系统提示：{qc_result}"

        return {"messages": [AIMessage(content=final_answer)]}

    # ------ 构建图 ------

    wf = StateGraph(SupervisorState)

    wf.add_node("analyze", analyze_node)

    # 注册三个 Worker 子图
    for name, worker_graph in WORKERS.items():
        wf.add_node(name, worker_graph)

    wf.add_node("aggregate", aggregate_node)

    wf.add_edge(START, "analyze")
    wf.add_conditional_edges("analyze", route_after_analyze)

    # Worker 执行完 → 回到 analyze 再判断
    for name in WORKERS:
        wf.add_edge(name, "analyze")

    wf.add_edge("aggregate", END)

    checkpointer = MemorySaver()
    return wf.compile(checkpointer=checkpointer)


# ====== 全局单例 ======

app = build_supervisor()
