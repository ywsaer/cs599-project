# KitchenPilot 架构说明

## 1. 整体架构

```
┌──────────┐     HTTP/SSE      ┌──────────────┐     LangGraph     ┌──────────────────┐
│  前端     │ ◄──────────────► │   FastAPI     │ ◄─────────────► │  Supervisor Graph │
│ index.html│                   │   main.py     │                  │  + 3 Workers      │
└──────────┘                   └──────────────┘                  └────────┬─────────┘
                                                                         │
                                                                  ┌──────▼──────┐
                                                                  │    BM25     │
                                                                  │  Retriever  │
                                                                  └──────┬──────┘
                                                                         │
                                                                  ┌──────▼──────┐
                                                                  │  recipes_   │
                                                                  │  full.json  │
                                                                  └─────────────┘
```

## 2. Supervisor 图流程

```
START
  │
  ▼
analyze ──(派遣 Worker)──► [recipe_qa_worker, recommend_worker, safety_worker] (并行)
  │         (Send API)         │
  │                           │ (Worker 返回)
  │◄──────────────────────────┘
  │
  │ (分析结果: "DONE")
  ▼
aggregate ──► END
```

## 3. Worker 内部流程

```
worker_chat ──(LLM 决定调工具)──► worker_tools
     │                                  │
     │◄─────────────────────────────────┘
     │
     │ (LLM 决定不调工具 → 任务完成)
     ▼
   return
```

## 4. 关键设计决策

### 4.1 为什么选 BM25 而不是向量检索

- DeepSeek 不提供嵌入 API（`text-embedding-3-small` 返回 404）
- ChromaDB 默认嵌入模型是英文的，中文效果极差
- BM25 + 中文 bigram 对菜名匹配效果已经足够好
- 零外部依赖，启动快，无需 GPU

### 4.2 为什么 SSE 流式用 httpx 而不是 LangChain

- `ChatOpenAI(streaming=True).astream()` 经过多层封装，存在缓冲问题
- `httpx.AsyncClient.stream()` 直连 DeepSeek SSE 端点，每个 token 立即 yield
- 完全控制 SSE 格式（`StreamingResponse` + 手动编码 `event: token\ndata: ...`）

### 4.3 为什么用 Supervisor 模式而不是直接 ReAct

- 菜谱问答和食材推荐是**可并行**的独立任务
- Supervisor 分析后通过 `Send` API 并行分发，减少串行等待
- 安全检查 Worker 独立审查，职责分离更清晰

## 5. 数据流

```
用户 query
  → 注入 [当前用户ID: xxx]
  → HumanMessage → AgentState.messages
  → Supervisor analyze → "recipe_qa_worker, recommend_worker"
  → Send 并行分发
  → Worker chat → tool_call → ToolNode 执行 → ToolMessage
  → Worker chat → AIMessage（最终回答）
  → analyze again → "DONE"
  → aggregate → 提取最终回答 → 规则引擎质检
  → httpx stream → SSE token → 前端
```

## 6. 模块依赖关系

```
main.py
  ├── config.py              (环境变量)
  ├── supervisor_graph.py    (图编排)
  │     ├── workers.py       (Worker 子图)
  │     │     └── tools.py   (@tool 工具)
  │     │           └── rag/retriever.py  (检索)
  │     │                 └── rag/chunks.py (分块)
  │     │                       └── recipe_service.py (数据)
  │     └── system_prompts_multi.py (提示词)
  └── static/index.html       (前端)
```

## 7. 部署架构

```
docker-compose.yml:
  app (Python 3.12-slim)
    ├── FastAPI + uvicorn
    ├── 读取 .env 环境变量
    ├── 启动时构建 BM25 索引
    └── 端口 8000
```
