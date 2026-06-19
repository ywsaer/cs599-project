"""KitchenPilot LangChain 版 — 智能体评估基准测试。

评估维度：
- 工具调用正确率（是否调用了正确的工具）
- RAG 检索召回率（目标菜谱是否在 top-5 中）
- 端到端延迟（含 LLM 调用）
"""
import json
import time
from pathlib import Path

# ── 测试用例 ──────────────────────────────────────────────────────────────
# (query, expected_tool, expected_recipe)

TEST_CASES = [
    # 菜谱问答 → 应调用 search_recipes
    ("番茄炒蛋怎么做？", "search_recipes", "番茄炒蛋"),
    ("酸辣土豆丝怎么炒才脆？", "search_recipes", "酸辣土豆丝"),
    ("可乐鸡翅为什么太甜？", "search_recipes", "可乐鸡翅"),
    ("清蒸鱼有什么注意事项？", "search_recipes", "清蒸鱼"),
    ("这个菜失败了怎么补救？", "search_recipes", None),

    # 食材推荐 → 应调用 recommend_by_ingredients
    ("我有鸡蛋和番茄，能做什么？", "recommend_by_ingredients", None),
    ("今天吃什么？", "recommend_by_ingredients", None),
    ("家里只有米饭和鸡蛋，推荐一道简单菜。", "recommend_by_ingredients", None),
    ("冰箱里有土豆和青椒，今晚做什么？", "recommend_by_ingredients", None),

    # 边界情况
    ("你好", None, None),
    ("今天天气怎么样", None, None),
]


def run_tool_call_benchmark() -> dict:
    """
    评估：LLM 是否正确选择合适的工具。

    注意：此测试需要 DEEPSEEK_API_KEY 环境变量。
    """
    from src.agent_graph import app
    from langchain_core.messages import HumanMessage

    correct = 0
    total = 0
    details: list[dict] = []

    for query, expected_tool, _ in TEST_CASES:
        if expected_tool is None:
            continue  # 跳过边界情况（无需工具调用）
        total += 1

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"bench_{total}"}},
            )

            # 检查消息中是否包含对应工具调用
            tools_called: list[str] = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_called.append(tc["name"])

            is_correct = expected_tool in tools_called
            if is_correct:
                correct += 1

            details.append({
                "query": query,
                "expected_tool": expected_tool,
                "actual_tools": tools_called,
                "correct": is_correct,
            })

        except Exception as e:
            details.append({
                "query": query,
                "expected_tool": expected_tool,
                "error": str(e),
                "correct": False,
            })

    return {
        "test": "工具调用正确率",
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "details": details,
    }


def run_retrieval_benchmark() -> dict:
    """
    评估：RAG 检索引擎的 Recall@5。
    不依赖 LLM，只测检索模块。
    """
    from src.rag.retriever import get_retriever

    retriever = get_retriever()
    retriever.ensure_indexed()

    hits = 0
    total = 0
    details: list[dict] = []

    for query, _, recipe_hint in TEST_CASES:
        if recipe_hint is None:
            continue
        total += 1

        chunks = retriever.search(query, top_k=5)
        names = [c.recipe_name for c in chunks]
        is_hit = any(recipe_hint in name or name in recipe_hint for name in names)
        if is_hit:
            hits += 1

        details.append({
            "query": query,
            "expected": recipe_hint,
            "retrieved_top3": names[:3],
            "hit": is_hit,
        })

    return {
        "test": "RAG 检索 Recall@5",
        "recall": hits / total if total > 0 else 0,
        "hits": hits,
        "total": total,
        "details": details,
    }


def run_latency_benchmark(samples: int = 2) -> dict:
    """
    评估：端到端延迟（含 LLM 调用）。

    注意：此测试会消耗 API 配额。
    """
    from src.agent_graph import app
    from langchain_core.messages import HumanMessage

    queries = [
        "番茄炒蛋怎么做？",
        "我有鸡蛋和土豆，推荐一道菜",
    ]

    latencies: list[float] = []
    for query in queries * samples:
        try:
            start = time.perf_counter()
            app.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"latency_{time.time_ns()}"}},
            )
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
        except Exception:
            continue

    if not latencies:
        return {"test": "端到端延迟", "error": "全部失败，请检查 DEEPSEEK_API_KEY"}

    latencies.sort()
    return {
        "test": "端到端延迟",
        "avg_ms": sum(latencies) / len(latencies) * 1000,
        "p50_ms": latencies[len(latencies) // 2] * 1000,
        "min_ms": latencies[0] * 1000,
        "max_ms": latencies[-1] * 1000,
        "samples": len(latencies),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("KitchenPilot LangChain — Benchmark")
    print("=" * 60)

    # [1] RAG 检索（不调 LLM）
    print("\n[1/3] RAG 检索 Recall@5（不调用 LLM）")
    print("-" * 40)
    rag_result = run_retrieval_benchmark()
    print(f"Recall@5: {rag_result['recall']:.1%} ({rag_result['hits']}/{rag_result['total']})")
    for r in rag_result["details"]:
        status = "HIT" if r["hit"] else "MISS"
        print(f"  [{status}] {r['query']}")
        print(f"         期望: {r['expected']}, 命中: {r['retrieved_top3']}")

    # [2] 工具调用正确率（调 LLM）
    print("\n[2/3] 工具调用正确率（需调用 LLM）")
    print("-" * 40)
    tool_result = run_tool_call_benchmark()
    if tool_result["total"] > 0:
        print(f"正确率: {tool_result['accuracy']:.1%} ({tool_result['correct']}/{tool_result['total']})")
        for r in tool_result["details"]:
            status = "PASS" if r["correct"] else "FAIL"
            print(f"  [{status}] {r['query']} -> {r.get('actual_tools', r.get('error', 'unknown'))}")

    # [3] 端到端延迟（调 LLM）
    print("\n[3/3] 端到端延迟（需调用 LLM，会消耗 API 配额）")
    print("-" * 40)
    latency_result = run_latency_benchmark()
    if "error" in latency_result:
        print(f"错误: {latency_result['error']}")
    else:
        print(f"样本数: {latency_result['samples']}")
        print(f"平均: {latency_result['avg_ms']:.0f}ms, P50: {latency_result['p50_ms']:.0f}ms, "
              f"最小: {latency_result['min_ms']:.0f}ms, 最大: {latency_result['max_ms']:.0f}ms")

    # 保存结果
    output = {
        "retrieval": {k: v for k, v in rag_result.items() if k != "details"},
        "tool_calling": {k: v for k, v in tool_result.items() if k != "details"},
        "latency": {k: v for k, v in latency_result.items()},
    }
    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至 {out_path}")
