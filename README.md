# KitchenPilot — AI 烹饪助手

面向厨房新手的多智能体协作 AI 助手，基于 LangGraph Supervisor + Worker 架构。

## 项目简介

KitchenPilot 帮助厨房新手：
- **菜谱问答**：询问某道菜的做法、失败原因、替代食材
- **智能推荐**：根据现有食材和用户画像个性化推荐菜谱
- **多 Agent 协作**：Supervisor 编排 3 个 Worker（菜谱问答 / 推荐 / 安全检查）并行工作
- **安全质检**：LLM 安全检查 + 规则引擎双重保障
- **SSE 流式**：Token 级逐字打字输出

## 方向

**方向一：Agentic AI 原生开发**

## 技术栈

| 类别 | 技术 |
|------|------|
| AI IDE | Trae CN |
| LLM | DeepSeek API（`deepseek-chat`，通过 `langchain_openai.ChatOpenAI` + `httpx` 直连流式） |
| Agent 框架 | LangGraph 1.2 + LangChain 1.3 |
| 协议 | Function Calling（`@tool` 装饰器 + `ToolNode`）、SSE（Token 级流式） |
| 记忆 | MemorySaver（`thread_id` 多轮对话持久化） |
| 检索 | BM25（中文 bigram 词汇检索 + 分块类型重排序 + 实体匹配加权） |
| 评估 | `tests/benchmark.py`（工具调用正确率 / 检索 Recall@5 / 端到端延迟） |
| 容器 | Docker + docker-compose |
| 前端 | 单页 HTML（SSE 流式 / 普通双模式，逐字打字效果） |

## 核心技术要素覆盖（≥6 项）

| 课程要求 | 实现方式 | 评分权重 |
|----------|---------|:---:|
| **工具使用 / Function Calling** | 3 个 `@tool` 工具 + `ToolNode` 自动分发，LLM 自主决定调用时机和参数 | 核心 |
| **记忆机制** | MemorySaver（短期对话窗口）+ 会话队列（追问检测 + 活跃菜谱追踪） | 核心 |
| **状态管理与多步骤推理** | LangGraph StateGraph + ReAct 循环 + 质检→修复条件边 + Supervisor 分析→派遣→汇总 | 核心 |
| **多智能体协作** | Supervisor + 3 Worker 并行（菜谱问答 / 推荐 / 安全检查），`Send` API 并行分发 | 核心 |
| **SDD 规格驱动开发** | Product Spec / Architecture Spec / API Spec（见 `docs/`） | 文档 |
| **可观测性与评估** | `tests/benchmark.py`（3 维度）+ `execution_trace` 链路追踪 + SSE 工具调用可见 | 评估 |

## 技术生态覆盖

- [x] AI IDE: Trae CN（课程指定）
- [x] LLM: DeepSeek API
- [x] Agent 框架: LangGraph
- [x] 协议: Function Calling
- [x] 协议: SSE
- [x] 基础设施: Docker
- [x] 评估: 基准测试

## 目录结构

```
cs599-langchain/
├── src/                          # 源代码
│   ├── main.py                   # FastAPI 入口 + 路由 + SSE 流式（httpx 直连）
│   ├── supervisor_graph.py       # Supervisor 多 Agent 编排图
│   ├── workers.py                # Worker 子图工厂（recipe_qa / recommend / safety）
│   ├── tools.py                  # @tool 装饰器工具（search_recipes / recommend / get_user_profile）
│   ├── system_prompts_multi.py   # 系统提示词（Supervisor + 3 Worker）
│   ├── config.py                 # 环境变量配置（全部从 .env 读取）
│   ├── recipe_service.py         # 菜谱数据加载服务
│   ├── rag/                      # RAG 检索模块
│   │   ├── chunks.py             # 菜谱分块（7 种语义块类型）
│   │   └── retriever.py          # BM25 检索 + 分块类型重排序 + 实体匹配加权
│   └── data/
│       └── recipes_full.json     # 种子菜谱数据（50 道家常菜）
├── static/
│   └── index.html                # 前端聊天页面（SSE 流式 + Token 打字效果）
├── tests/
│   └── benchmark.py              # 基准测试（工具调用 / 检索召回 / 延迟）
├── docs/                         # 项目文档（SDD Spec + 大作业报告）
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE                       # MIT
└── README.md
```

## Agent 架构（多智能体版）

```
用户: "番茄炒蛋怎么做？推荐几道类似菜"
                │
         ┌──────▼──────┐
         │  Supervisor │  分析用户需求，决定派遣哪些 Worker
         │  (analyze)  │
         └──────┬──────┘
                │
       ┌────────┴────────┐  ← Send 并行分发
       ▼                 ▼
  recipe_qa_worker  recommend_worker    ← 并行执行
  (search_recipes)  (get_user_profile
                     + recommend)
       │                 │
       └────────┬────────┘
                ▼
         ┌──────▼──────┐
         │  Supervisor │  收到结果 → "DONE"
         └──────┬──────┘
                ▼
         ┌──────▼──────┐
         │  aggregate  │  汇总 + 规则引擎质检 → 最终回答
         └──────┬──────┘
                ▼
               END
```

### SSE 流式输出

```
阶段 1：多 Agent 图 invoke（秒级）
  → 前端显示工具调用卡片

阶段 2：httpx 直连 DeepSeek stream API
  → 每个 token 立即 SSE 推送
  → 前端逐字打字效果（闪烁光标）
```

## 环境搭建

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx
```

### 3. 启动服务

```bash
# 直接启动
python -m uvicorn src.main:server --reload

# Docker 启动
docker compose up -d
```

### 4. 访问

浏览器打开 [http://localhost:8000](http://localhost:8000)

### 5. 运行评估

```bash
python -m tests.benchmark
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端聊天页面 |
| GET | `/health` | 健康检查 |
| POST | `/api/chat` | 对话（非流式） |
| POST | `/api/chat/stream` | 对话（SSE Token 级流式） |
| GET | `/api/recipes` | 菜谱列表 |
| GET | `/api/recipes/{id}` | 菜谱详情 |
| DELETE | `/api/sessions/{id}` | 删除会话 |

### 请求示例

```bash
# 菜谱问答
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "番茄炒蛋怎么做？"}'

# 切换用户画像
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "推荐一道菜", "user_id": "expert_user"}'

# SSE 流式（观察工具调用 + 逐字输出）
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "今天吃什么？"}'
```

## 项目状态

- [x] Proposal
- [x] MVP (v0.1)
- [x] 多 Agent 升级 (v0.2)
- [ ] Final — 待提交 `docs/CS599_大作业报告.pdf`

## 参考与致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent 编排框架
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用框架
- [FastAPI](https://github.com/fastapi/fastapi) — Web 框架
- [DeepSeek](https://platform.deepseek.com) — LLM API
- 种子菜谱数据来源于公开菜谱社区

## License

MIT
