# Multi-Agent Medical Assistant 项目说明文档

> 本文档面向项目维护者、部署人员和后续开发者，系统说明当前项目的整体定位、功能模块、技术架构、运行配置、核心流程和维护注意事项。

---

## 1. 项目定位

`Multi-Agent Medical Assistant` 是一个面向医疗问答、医学知识检索、网页医学信息查询和医学影像辅助分析的多智能体系统。

项目使用 FastAPI 提供后端服务，使用 LangGraph 编排多个专业 Agent，通过统一的聊天 / 上传入口，根据用户输入自动或手动路由到合适的处理模块。

当前项目重点能力包括：

- 通用医疗对话
- 医学文献 RAG 检索增强问答
- Tavily 实时 Web Search 医学信息检索
- 脑肿瘤 / 脑卒中影像辅助分析
- 人工审核（Human-in-the-Loop）验证流程
- LangGraph SQLite 持久化 checkpoint
- 多用户会话隔离
- 历史消息窗口裁剪与过期 checkpoint 清理
- 语音输入 / 语音输出接口

---

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI, Uvicorn |
| Agent 编排 | LangGraph, LangChain |
| 主聊天 LLM | MiMo Token Plan, `mimo-v2.5-pro` |
| Embedding / 视觉 LLM | Zhipu AI, `embedding-3`, `glm-4.5-air` |
| 向量数据库 | Qdrant |
| Web Search | Tavily |
| 医学影像 | PyTorch, Torchvision, OpenCV, `heyi-Trans-master` |
| 语音识别 | 百度 ASR |
| 语音合成 | gTTS |
| 前端 | HTML, CSS, JavaScript, Bootstrap 5 |
| 配置管理 | `.env`, `python-dotenv` |
| 持久化断点 | LangGraph `SqliteSaver` / `MemorySaver` |

---

## 3. 总体架构

```text
用户浏览器
  │
  ├─ /chat      文本问答
  ├─ /upload    图像 + 文本多模态输入
  ├─ /validate  人工审核反馈
  │
  ▼
FastAPI app.py
  │
  ├─ Cookie session_id 管理
  ├─ 静态文件 /uploads /data 挂载
  ├─ 音频清理后台线程
  └─ LangGraph checkpoint 清理后台线程
  │
  ▼
process_query()
  │
  ├─ _get_compiled_graph()
  │    └─ 缓存已编译 LangGraph，避免每次请求重复 compile
  │
  ├─ _build_thread_config(session_id)
  │    └─ session_id → thread_id，实现多用户 checkpoint 隔离
  │
  ├─ graph.invoke(state, thread_config)
  │
  ├─ RemoveMessage + graph.update_state()
  │    └─ 将超出窗口的历史消息从 checkpoint 中真正移除
  │
  └─ 返回 AgentState
       │
       ├─ CONVERSATION_AGENT
       ├─ RAG_AGENT
       ├─ WEB_SEARCH_PROCESSOR_AGENT
       ├─ BRAIN_TUMOR_AGENT
       ├─ BRAIN_STROKE_AGENT
       └─ HUMAN_VALIDATION
```

---

## 4. 核心目录结构

```text
.
├── app.py                         # FastAPI 主入口，HTTP API 和后台任务
├── config.py                      # 全局配置类，读取环境变量并初始化 LLM / RAG / CV 配置
├── ingest_rag_data.py             # RAG 文档入库脚本
├── requirements.txt               # Python 依赖
├── README.md                      # 项目首页说明
├── .env.example                   # 环境变量模板，不包含真实密钥
│
├── agents/
│   ├── agent_decision.py          # LangGraph 编排核心，路由、checkpoint、历史裁剪
│   ├── rag_agent/                 # 医学 RAG 检索增强模块
│   ├── web_search_processor_agent/# Tavily/Web Search 处理模块
│   ├── image_analysis_agent/      # 医学影像分析 Agent
│   ├── guardrails/                # 本地输入 / 输出安全护栏
│   └── README.md                  # Agent 级说明
│
├── docs/
│   ├── API_KEYS.md                # 外部 API 和密钥清单
│   ├── LANGGRAPH_CHECKPOINT.md    # LangGraph checkpoint 专项说明
│   ├── PROJECT_GUIDE.md           # 当前项目详细说明文档
│   └── competition_materials/     # 比赛 / 答辩 / 演示材料归档
│
├── templates/                     # 前端页面模板
├── assets/                        # 静态资源
├── data/                          # RAG 数据、解析内容、运行时 checkpoint 数据库
├── uploads/                       # 用户上传文件和生成结果图，已 gitignore
├── tests/                         # 测试脚本
└── heyi-Trans-master/             # Heyi Vision Transformer 分割框架
```

---

## 5. Agent 模块说明

### 5.1 Conversation Agent

用途：

- 普通医疗问答
- 非专业检索型对话
- 验证反馈后的自然语言回应
- 兜底对话

主要配置：

- `config.ConversationConfig`
- 使用 MiMo OpenAI-compatible Chat API

---

### 5.2 RAG Agent

用途：

- 对医学文献、指南、资料进行检索增强问答
- 支持自定义 PDF / 文档入库
- 通过 Qdrant 进行向量检索

核心组件：

| 组件 | 说明 |
|------|------|
| `MedicalDocParser` | 文档解析 |
| `ContentProcessor` | 文本清洗与切分 |
| `VectorStore` | Qdrant 向量库封装 |
| `Reranker` | 检索结果重排，可降级 |
| `QueryExpander` | 查询扩展 |
| `ResponseGenerator` | RAG 回答生成 |

入库命令示例：

```bash
python ingest_rag_data.py --dir data/raw
```

---

### 5.3 Web Search Agent

用途：

- 获取最新医学信息
- 查询实时研究、指南、新闻或公开资料
- 通过 Tavily 搜索后，再由 LLM 整理成用户可读回答

主要依赖：

- `TAVILY_API_KEY`
- `agents/web_search_processor_agent/`

---

### 5.4 Brain Tumor Agent

用途：

- 脑肿瘤图像分析
- 图像分割
- 病灶标注
- 生成中文辅助诊断文本

当前特征：

- 使用 `heyi-Trans-master` 的 ViT 框架作为底层视觉模型适配基础
- 没有微调权重时可进入 demo fallback 模式
- 输出中会明确标注演示模式风险

---

### 5.5 Brain Stroke Agent

用途：

- 脑卒中相关图像分析
- 支持 `auto` / `hemorrhage` / `ischemia` 任务类型
- 支持远程 Heyi 服务优先，本地 fallback 兜底

核心流程：

```text
输入图像
  → segment_image
  → mark_lesion
  → diagnose
  → 返回诊断文本 + 结果图
```

远程服务接口：

- `/health`
- `/segment`
- `/segment/preview`

---

### 5.6 Human-in-the-Loop Validation

当前实现：

- CV Agent 输出会标记需要人工审核
- 前端展示验证面板
- `/validate` 接收人工反馈
- 验证反馈路由到 `CONVERSATION_AGENT`，避免重新触发 CV 流程

后续规划：

- 使用 LangGraph 原生中断恢复能力
- `compile(interrupt_before=["human_validation"])`
- `Command(resume=...)`
- 依托 checkpoint 从中断点恢复，而不是重新进入完整图

---

## 6. LangGraph Checkpoint 机制

项目已经引入 LangGraph 持久化 checkpoint，解决以下问题：

- 服务重启后对话历史丢失
- 多用户共享同一个 thread 导致串话
- 每次请求重复编译 LangGraph 图带来的性能浪费
- 对话历史无限增长导致 prompt token 成本膨胀
- SQLite checkpoint 文件长期增长缺少清理机制

### 6.1 后端选择

默认使用 SQLite：

```env
LANGGRAPH_CHECKPOINT_BACKEND=sqlite
LANGGRAPH_CHECKPOINT_SQLITE_PATH=./data/langgraph_checkpoints.sqlite
```

如果 SQLite 初始化失败，自动回退：

```text
SqliteSaver → MemorySaver
```

### 6.2 多用户隔离

```text
浏览器 Cookie session_id
  → process_query(session_id=...)
  → LangGraph thread_id
  → 独立 checkpoint 状态
```

这样不同用户的对话不会互相污染。

### 6.3 图编译缓存

`_get_compiled_graph()` 使用全进程单例缓存已编译图：

```text
首次请求：create_agent_graph() → workflow.compile()
后续请求：直接复用 _compiled_graph
```

收益：

- 降低 CPU 消耗
- 避免重复创建重对象
- 保持服务响应稳定

### 6.4 历史截断写回

当 `messages` 超过 `config.max_conversation_history` 时：

```text
计算待删除旧消息
  → 构造 RemoveMessage(id=...)
  → graph.update_state(thread_config, {"messages": remove_ops})
  → 同步更新 checkpoint
```

这与简单截断返回值不同，它会真正影响持久化状态，下一轮恢复时不会重新加载被裁剪的旧消息。

### 6.5 过期 checkpoint 清理

项目维护额外访问表：

```sql
langgraph_thread_access(thread_id, last_seen)
```

每次调用 `process_query()` 后更新 `last_seen`。

后台线程定期清理：

```env
LANGGRAPH_CHECKPOINT_CLEANUP_ENABLED=true
LANGGRAPH_CHECKPOINT_RETENTION_DAYS=30
LANGGRAPH_CHECKPOINT_CLEANUP_INTERVAL_SECONDS=86400
```

默认含义：

- 开启清理
- 保留最近 30 天活跃会话
- 每 24 小时检查一次

详细说明见：

```text
docs/LANGGRAPH_CHECKPOINT.md
```

---

## 7. 环境变量配置

真实密钥应写入本地 `.env`，不要提交到仓库。

模板文件：

```text
.env.example
```

最小可用配置：

```env
MIMO_API_KEY=你的MiMo Token Plan Key
ZHIPU_API_KEY=你的智谱 Key
TAVILY_API_KEY=你的 Tavily Key
```

常用可选配置：

```env
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
MIMO_CHAT_MODEL=mimo-v2.5-pro

LANGGRAPH_CHECKPOINT_BACKEND=sqlite
LANGGRAPH_CHECKPOINT_SQLITE_PATH=./data/langgraph_checkpoints.sqlite
LANGGRAPH_CHECKPOINT_CLEANUP_ENABLED=true
LANGGRAPH_CHECKPOINT_RETENTION_DAYS=30
LANGGRAPH_CHECKPOINT_CLEANUP_INTERVAL_SECONDS=86400

ENABLE_OFFLINE_FALLBACK=true
FORCE_OFFLINE_MODE=false
```

完整密钥说明见：

```text
docs/API_KEYS.md
```

---

## 8. 启动与运行

### 8.1 安装依赖

```bash
pip install -r requirements.txt
```

### 8.2 配置环境变量

```bash
cp .env.example .env
```

然后填入实际 API Key。

Windows PowerShell 下可以直接编辑 `.env`。

### 8.3 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：

```text
http://localhost:8000
```

### 8.4 健康检查

```text
GET /health
```

---

## 9. HTTP API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat` | 文本聊天入口 |
| `POST` | `/upload` | 图像 + 文本上传入口 |
| `POST` | `/validate` | 人工审核反馈 |
| `POST` | `/api/brain_stroke/segment` | 脑卒中直接分割接口 |
| `POST` | `/transcribe` | 百度 ASR 语音转文本 |
| `POST` | `/generate-speech` | gTTS 文本转语音 |
| `GET` | `/health` | 服务健康检查 |

更多请求 / 响应示例见 `README.md` 的 HTTP API 章节。

---

## 10. 运行时文件与 Git 管理

以下文件或目录属于本地运行时产物，不应提交：

| 路径 | 说明 |
|------|------|
| `.env` | 真实密钥配置 |
| `.env.*` | 本地环境配置备份 |
| `.venv/` | Python 虚拟环境 |
| `uploads/` | 用户上传文件、分析结果图、语音临时文件 |
| `data/*.sqlite` | LangGraph checkpoint SQLite 数据库 |
| `__pycache__/` | Python 缓存 |
| `*.mp3`, `*.webm` | 本地音频文件 |

已经提交的模板 / 文档包括：

- `.env.example`
- `README.md`
- `docs/API_KEYS.md`
- `docs/LANGGRAPH_CHECKPOINT.md`
- `docs/PROJECT_GUIDE.md`

---

## 11. 当前已完成的重要优化

| 优化项 | 状态 | 说明 |
|--------|------|------|
| LangGraph SQLite checkpoint | 已完成 | 对话状态持久化到 SQLite |
| 多用户会话隔离 | 已完成 | `session_id` 映射到 `thread_id` |
| 图编译缓存 | 已完成 | 避免每次请求重复 `compile()` |
| 历史消息写回裁剪 | 已完成 | `RemoveMessage` 从 checkpoint 删除旧消息 |
| RAG / Web Agent 单例化 | 已完成 | 避免每次节点执行重复构造重对象 |
| checkpoint 过期清理 | 已完成 | 后台任务清理长期不活跃 thread |
| 文档结构整理 | 已完成 | README、API_KEYS、Checkpoint 文档、比赛资料归档 |

---

## 12. 后续建议

### 12.1 真正的 HITL 中断恢复

当前 `/validate` 可以完成人工反馈闭环，但还不是 LangGraph 原生的中断恢复。

后续可实现：

```python
workflow.compile(interrupt_before=["human_validation"])
```

并通过：

```python
Command(resume=...)
```

从人工审核节点继续执行。

适合在医学影像推理链路进一步稳定后实现。

### 12.2 异步 Checkpoint 或 PostgresSaver

当前 SQLite 适合单机和中低并发部署。

如果后续并发量增加，可以考虑：

- `AsyncSqliteSaver`
- `PostgresSaver`
- 独立服务化 checkpoint 存储

### 12.3 更完善的测试覆盖

建议增加：

- `process_query()` session 隔离测试
- checkpoint 重启恢复测试
- 历史裁剪写回测试
- 过期清理 SQL 测试
- RAG Agent singleton 行为测试
- `/chat` `/upload` `/validate` API smoke test

### 12.4 部署说明补充

后续可以新增：

```text
docs/DEPLOYMENT.md
```

内容包括：

- Windows 本地运行
- Linux 服务器运行
- Docker 部署
- `.env` 配置检查
- Qdrant 初始化
- Heyi 远程服务联调

---

## 13. 维护建议

- 不要把真实 API Key 写入 README 或任何可提交文档
- 不要提交 `.env` / `.env.*`
- 不要提交 `uploads/` 中的用户文件
- 不要提交 `data/*.sqlite` 运行时 checkpoint 数据库
- 修改 LangGraph state 字段时，要同步检查 checkpoint 兼容性
- 修改 Agent 路由名称时，要同步前端 `preferred_agent` 选项
- 修改 RAG embedding 模型时，要确认 Qdrant collection 维度是否一致
- 修改脑卒中任务类型时，要同步前端 chip、后端 normalization 和远程 Heyi API 参数

---

## 14. 相关文档索引

| 文档 | 说明 |
|------|------|
| `README.md` | 项目首页、快速启动、API 概览 |
| `docs/API_KEYS.md` | 外部服务和密钥清单 |
| `docs/LANGGRAPH_CHECKPOINT.md` | LangGraph checkpoint 专项说明 |
| `agents/README.md` | Agent 级说明 |
| `docs/competition_materials/README.md` | 比赛材料归档说明 |
| `heyi-Trans-master/README.md` | Heyi 视觉模型框架说明 |

---

## 15. 总结

当前项目已经从一个基础多 Agent 医疗助手，升级为具有持久化对话状态、会话隔离、历史管理、运行时清理和较完整文档体系的多智能体医疗应用。

从工程角度看，当前架构已经具备以下能力：

- 可持续迭代
- 可本地部署
- 可接入远程医学影像服务
- 可维护多用户对话上下文
- 可通过配置控制在线 / 离线 / checkpoint 行为
- 可进一步扩展 HITL、异步 checkpoint 和生产部署能力
