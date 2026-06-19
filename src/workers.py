"""
Worker Agent 子图构建模块。

每个 Worker 是一个独立的 StateGraph（chat_bot → tools → chat_bot → END），
由 Supervisor 根据任务需求并行或串行调度。
"""
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from src.config import get_config
from src.tools import search_recipes, recommend_by_ingredients, get_user_profile


# ====== 通用 Worker 状态 ======

class WorkerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ====== Worker 工厂函数 ======

def create_worker(
    tools: list,
    system_prompt: str,
    worker_name: str,
):
    """
    创建一个 Worker Agent 子图。

    参数:
        tools: 该 Worker 可以使用的工具列表
        system_prompt: 系统提示词
        worker_name: 节点名称（如 "recipe_qa_worker"）

    返回:
        编译好的 StateGraph 子图
    """
    config = get_config()
    model = ChatOpenAI(
        model=config.deepseek_model,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        temperature=0.3,
    ).bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    def chat_node(state: WorkerState) -> dict:
        """LLM 调用节点。"""
        response = (prompt | model).invoke({"messages": state["messages"]})
        return {"messages": [response]}

    def route(state: WorkerState) -> str:
        """判断下一步：调用工具还是结束。"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "worker_tools"
        return END

    wf = StateGraph(WorkerState)

    wf.add_node("worker_chat", chat_node)
    wf.add_node("worker_tools", ToolNode(tools))

    wf.add_edge(START, "worker_chat")
    wf.add_conditional_edges("worker_chat", route)
    wf.add_edge("worker_tools", "worker_chat")

    # 子图不需要自己的 checkpointer，由主图统一管理
    return wf.compile()


# ====== 创建三个 Worker 实例 ======

from src.system_prompts_multi import RECIPE_QA_PROMPT, RECOMMEND_PROMPT, SAFETY_PROMPT

# Recipe QA Worker: 只有 search_recipes 工具
recipe_qa_worker = create_worker(
    tools=[search_recipes],
    system_prompt=RECIPE_QA_PROMPT,
    worker_name="recipe_qa_worker",
)

# Recommend Worker: 有 recommend_by_ingredients + get_user_profile 工具
recommend_worker = create_worker(
    tools=[recommend_by_ingredients, get_user_profile],
    system_prompt=RECOMMEND_PROMPT,
    worker_name="recommend_worker",
)

# Safety Worker: 无工具，纯 LLM 审查
safety_worker = create_worker(
    tools=[],
    system_prompt=SAFETY_PROMPT,
    worker_name="safety_worker",
)

WORKERS = {
    "recipe_qa_worker": recipe_qa_worker,
    "recommend_worker": recommend_worker,
    "safety_worker": safety_worker,
}
