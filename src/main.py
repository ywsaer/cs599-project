"""
KitchenPilot LangChain 版 — FastAPI 入口。

采用 LangChain 生态（langchain_core + langchain_openai + langgraph），
多 Agent 协作：Supervisor 编排 + 3 个 Worker 并行 + MemorySaver + SSE Token 级流式。
"""
import json
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from src.config import get_config
from src.supervisor_graph import app as agent_app
from src.rag.retriever import get_retriever
from src.recipe_service import load_recipes, Recipe

# ── 应用初始化 ────────────────────────────────────────────────────────────

config = get_config()
server = FastAPI(title="KitchenPilot LangChain", version="0.1.0")

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静态文件 ──────────────────────────────────────────────────────────────

server.mount("/static", StaticFiles(directory="static"), name="static")


# ── 首页 ──────────────────────────────────────────────────────────────────

@server.get("/")
def index():
    """返回前端聊天页面。"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


# ── 启动事件 ──────────────────────────────────────────────────────────────

@server.on_event("startup")
def on_startup() -> None:
    """启动时构建 BM25 检索索引。"""
    print("正在构建菜谱检索索引...")
    retriever = get_retriever()
    count = retriever.ensure_indexed()
    print(f"已索引 {count} 个分块。")


# ── 数据模型 ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str = Field(default="")
    user_id: str = Field(default="demo_user")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    quality_passed: bool = True
    active_recipe: str | None = None


# ── 健康检查 ──────────────────────────────────────────────────────────────

@server.get("/health")
def health():
    return {"status": "ok", "app": "KitchenPilot LangChain"}


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def _extract_result_info(result: dict) -> dict:
    """
    从 LangGraph 返回的消息列表中提取：
    - 最终回答
    - 工具调用记录
    - 执行追踪
    - 质检状态
    """
    answer = ""
    tool_calls: list[dict] = []
    trace: list[str] = []
    quality_passed = True
    active_recipe = None

    for msg in result["messages"]:
        if msg.type == "human":
            trace.append(f"用户输入: {str(msg.content)[:50]}")
        elif msg.type == "ai":
            content = str(msg.content) if msg.content else ""
            # 过滤 Supervisor/Worker 内部控制消息
            skip_control = ["recipe_qa_worker", "recommend_worker", "safety_worker", "DONE",
                           "安全检查通过", "安全检查不通过", "Worker", "worker"]
            if any(kw in content for kw in skip_control):
                continue
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                    })
                    trace.append(f"调用工具: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
            elif "[质检通过]" in content:
                trace.append("质检: 通过")
                quality_passed = True
            elif "[质检标记-需修复]" in content:
                trace.append("质检: 发现风险，已添加安全提示")
                quality_passed = False
            elif content and content != answer:
                answer = content
        elif msg.type == "tool":
            trace.append(f"工具返回: {str(msg.content)[:80]}...")
            # 尝试提取菜名作为活跃菜谱
            tool_content = str(msg.content) if msg.content else ""
            if "菜谱：" in tool_content:
                lines = tool_content.split("\n")
                for line in lines:
                    if line.startswith("菜谱："):
                        active_recipe = line.replace("菜谱：", "").strip()
                        break

    # 取最后一条有效 AI 消息
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            content = str(msg.content)
            if "[质检" not in content:
                answer = content
                break

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "trace": trace,
        "quality_passed": quality_passed,
        "active_recipe": active_recipe,
    }


# ── 对话接口（非流式）────────────────────────────────────────────────────

@server.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    运行 ReAct Agent 处理用户消息。

    通过 thread_id 区分会话：
    同一 session_id 的多轮对话会被 MemorySaver 自动拼接历史。
    """
    session_id = request.session_id or f"session_{uuid.uuid4().hex}"
    config_dict = {"configurable": {"thread_id": session_id}}

    # 把 user_id 注入到消息中，让 LLM 知道该查哪个用户的画像
    user_id = request.user_id or "demo_user"
    enriched_query = f"[当前用户ID: {user_id}]\n{request.query}"

    result = agent_app.invoke(
        {"messages": [HumanMessage(content=enriched_query)]},
        config=config_dict,
    )

    info = _extract_result_info(result)

    return ChatResponse(
        session_id=session_id,
        answer=info["answer"],
        tool_calls=info["tool_calls"],
        trace=info["trace"],
        quality_passed=info["quality_passed"],
        active_recipe=info["active_recipe"],
    )


# ── 对话接口（SSE 流式）──────────────────────────────────────────────────

async def _stream_chat(request: ChatRequest) -> AsyncIterator[dict]:
    """
    两阶段流式输出：
      阶段 1：LangGraph 多 Agent 协作（快速完成）→ 发送工具调用事件
      阶段 2：LLM token 级流式生成最终回答（逐字打字效果）
    """
    import asyncio

    session_id = request.session_id or f"session_{uuid.uuid4().hex}"
    config_dict = {"configurable": {"thread_id": session_id}}
    user_id = request.user_id or "demo_user"

    yield {"event": "start", "data": json.dumps({"query": request.query}, ensure_ascii=False)}
    await asyncio.sleep(0)

    # ── 阶段 1：运行多 Agent 图，收集工具调用 ──

    enriched_query = f"[当前用户ID: {user_id}]\n{request.query}"
    result = agent_app.invoke(
        {"messages": [HumanMessage(content=enriched_query)]},
        config=config_dict,
    )

    control_kw = ["recipe_qa_worker", "recommend_worker", "safety_worker",
                  "DONE", "安全检查通过", "安全检查不通过"]

    # 发送工具调用事件
    for msg in result["messages"]:
        content_check = str(msg.content) if hasattr(msg, "content") and msg.content else ""
        if any(kw in content_check for kw in control_kw):
            continue
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                yield {
                    "event": "tool_call",
                    "data": json.dumps({"tool": tc["name"], "args": tc["args"]}, ensure_ascii=False),
                }
                await asyncio.sleep(0.05)
        elif msg.type == "tool":
            yield {
                "event": "tool_result",
                "data": json.dumps({"content": str(msg.content)[:200]}, ensure_ascii=False),
            }
            await asyncio.sleep(0.05)

    # ── 收集工具输出作为 LLM 上下文 ──

    tool_outputs: list[str] = []
    for msg in result["messages"]:
        if msg.type == "tool" and msg.content:
            c = str(msg.content)
            if not any(kw in c for kw in control_kw):
                tool_outputs.append(c)

    context_text = "\n\n".join(tool_outputs[-6:]) if tool_outputs else "知识库中未找到相关资料。"

    # ── 阶段 2：用 httpx 直接调 DeepSeek 流式 API，逐 token SSE 输出 ──

    import httpx

    config = get_config()
    system_msg = (
        "你是 KitchenPilot 烹饪助手。请根据以下资料回答用户问题。\n"
        "用朴素中文和编号列表，不要用 Markdown 加粗符号。"
    )
    user_msg = (
        f"用户问题：{request.query}\n\n"
        f"参考资料：\n{context_text}\n\n"
        f"请用中文直接回答，不要展开思考过程。"
    )

    yield {"event": "generating", "data": "{}"}
    await asyncio.sleep(0)

    full_answer = ""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{config.deepseek_base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.deepseek_api_key}",
            },
            json={
                "model": config.deepseek_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": True,
                "temperature": 0.3,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_answer += content
                        yield {
                            "event": "token",
                            "data": json.dumps({"text": content}, ensure_ascii=False),
                        }
                except json.JSONDecodeError:
                    continue

    # 发送完整答案用于前端兜底
    yield {
        "event": "answer",
        "data": json.dumps({"answer": full_answer}, ensure_ascii=False),
    }
    yield {"event": "done", "data": json.dumps({"session_id": session_id})}


@server.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式输出——手动格式化，确保逐字刷新。"""
    async def event_generator():
        async for evt in _stream_chat(request):
            # evt["data"] 已经是 JSON 字符串，不要再 json.dumps
            data_str = evt["data"] if isinstance(evt["data"], str) else json.dumps(evt["data"], ensure_ascii=False)
            yield f"event: {evt['event']}\ndata: {data_str}\n\n".encode("utf-8")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ── 菜谱接口 ──────────────────────────────────────────────────────────────

@server.get("/api/recipes")
def list_recipes() -> list[dict]:
    """返回全部可用菜谱的摘要列表。"""
    return [
        {
            "id": r.id,
            "name": r.name,
            "difficulty": r.difficulty,
            "time_minutes": r.time_minutes,
            "beginner_friendly": r.beginner_friendly,
            "cuisine": r.cuisine,
            "ingredient_count": len(r.ingredients),
        }
        for r in load_recipes()
    ]


@server.get("/api/recipes/{recipe_id}")
def get_recipe(recipe_id: int) -> dict:
    """按 ID 返回完整菜谱详情。"""
    for r in load_recipes():
        if r.id == recipe_id:
            return r.model_dump()
    raise HTTPException(status_code=404, detail="Recipe not found")


# ── 删除会话 ──────────────────────────────────────────────────────────────

@server.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """删除指定会话（MemorySaver 不支持按 key 删除，此处为说明接口）。"""
    return {
        "session_id": session_id,
        "deleted": False,
        "note": "MemorySaver 为内存存储，重启自动清空，无需手动删除",
    }


# ── 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:server", host=config.app_host, port=config.app_port, reload=True)
