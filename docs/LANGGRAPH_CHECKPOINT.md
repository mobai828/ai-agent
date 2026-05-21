# LangGraph 持久化断点（Checkpoint）模块说明

> 本文档描述项目中 LangGraph 图编排层的 Checkpoint 持久化机制——它的功能、架构设计、配置方式与使用方式。

---

## 1. 功能概览

| 功能 | 说明 |
|------|------|
| **对话历史持久化** | 基于 LangGraph `SqliteSaver`，将每轮对话的完整 `AgentState`（含 `messages`）写入本地 SQLite 文件。进程重启、崩溃恢复后对话不丢失。 |
| **多用户会话隔离** | 每个用户的浏览器 Cookie 中携带 `session_id`，在 LangGraph 层映射为独立的 `thread_id`，不同用户的对话互不干扰。 |
| **图编译缓存** | `StateGraph.compile()` 的结果全进程单例缓存（`_get_compiled_graph()`），避免每次请求都重复编译图结构。 |
| **历史窗口截断（写回 checkpoint）** | 当消息数超出 `max_conversation_history`（默认 20 条），使用 `RemoveMessage` + `graph.update_state()` 将旧消息从持久化层真正移除，控制 prompt token 成本。 |
| **重对象单例化** | `MedicalRAG`、`WebSearchProcessorAgent` 仅在图首次编译时创建一次，节点闭包复用，避免每次请求重连 Qdrant / 重载 reranker 模型。 |
| **优雅降级** | 若 `langgraph-checkpoint-sqlite` 未安装或 SQLite 初始化失败，自动回退到内存级 `MemorySaver`，保证服务可用（仅丢失跨重启持久化能力）。 |
| **过期 checkpoint 清理** | 维护 `langgraph_thread_access` 访问时间表，后台线程定期删除长期未活跃的 `thread_id` 对应 checkpoint，防止 SQLite 文件无限增长。 |

---

## 2. 架构设计

### 2.1 整体数据流

```
浏览器 Cookie (session_id)
        │
        ▼
  FastAPI 端点 (/chat, /upload, /validate)
        │  透传 session_id
        ▼
  process_query(query, session_id=...)
        │
        ├─ _get_compiled_graph()       ← 全进程单例，首次调用时编译
        │       └─ create_agent_graph()
        │               └─ workflow.compile(checkpointer=_get_checkpointer())
        │                                       │
        │                          ┌────────────┴────────────┐
        │                     SqliteSaver              MemorySaver
        │                   (持久化，默认)            (内存，降级)
        │
        ├─ _build_thread_config(session_id)  ← session_id → thread_id
        │
        ▼
  graph.invoke(state, thread_config)
        │
        ▼
  历史截断：RemoveMessage → graph.update_state()
        │
        ▼
  返回 AgentState (result)
```

### 2.2 核心组件

| 组件 | 位置 | 职责 |
|------|------|------|
| `CheckpointConfig` | `config.py` | 读取环境变量，决定使用 `sqlite` 还是 `memory` 后端以及 SQLite 文件路径 |
| `_get_checkpointer()` | `agents/agent_decision.py` | 全进程单例工厂：优先构造 `SqliteSaver`，失败回退 `MemorySaver` |
| `_build_thread_config()` | `agents/agent_decision.py` | 将 `session_id` 包装为 `{"configurable": {"thread_id": ...}}` |
| `_get_compiled_graph()` | `agents/agent_decision.py` | 懒加载 + 线程锁，编译后的图全进程缓存 |
| `process_query()` | `agents/agent_decision.py` | 入口函数，组装 state → invoke → 截断 → 返回 |

### 2.3 状态结构 (`AgentState`)

继承自 LangGraph 的 `MessagesState`，自带 `add_messages` reducer（追加合并语义）。

```python
class AgentState(MessagesState):
    agent_name: Optional[str]
    current_input: Optional[Union[str, Dict]]
    has_image: bool
    image_type: Optional[str]
    output: Optional[str]
    needs_human_validation: bool
    retrieval_confidence: float
    bypass_routing: bool
    insufficient_info: bool
    language: str
    preferred_agent: str
```

---

## 3. 配置方式

### 3.1 环境变量

在 `.env` 中配置（模板见 `.env.example`）：

```dotenv
# 后端选择：sqlite（默认，持久化）或 memory（仅进程内有效）
LANGGRAPH_CHECKPOINT_BACKEND=sqlite

# SQLite 文件路径（仅 backend=sqlite 时生效）
LANGGRAPH_CHECKPOINT_SQLITE_PATH=./data/langgraph_checkpoints.sqlite

# 是否启用 checkpoint 过期清理
LANGGRAPH_CHECKPOINT_CLEANUP_ENABLED=true

# 保留最近 N 天活跃会话
LANGGRAPH_CHECKPOINT_RETENTION_DAYS=30

# 清理任务运行间隔，默认每天一次
LANGGRAPH_CHECKPOINT_CLEANUP_INTERVAL_SECONDS=86400
```

### 3.2 配置类

`config.py` 中的 `CheckpointConfig`：

```python
class CheckpointConfig:
    def __init__(self):
        self.backend = os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "sqlite")
        self.sqlite_path = os.getenv(
            "LANGGRAPH_CHECKPOINT_SQLITE_PATH",
            "./data/langgraph_checkpoints.sqlite",
        )
        self.cleanup_enabled = _to_bool(os.getenv("LANGGRAPH_CHECKPOINT_CLEANUP_ENABLED"), default=True)
        self.retention_days = int(os.getenv("LANGGRAPH_CHECKPOINT_RETENTION_DAYS", "30"))
        self.cleanup_interval_seconds = int(os.getenv("LANGGRAPH_CHECKPOINT_CLEANUP_INTERVAL_SECONDS", "86400"))
```

通过 `Config().checkpoint` 访问。

### 3.3 历史窗口大小

```python
# config.py → Config
self.max_conversation_history = 20  # 保留最近 20 条消息（10 轮 Q&A）
```

超出此窗口的旧消息会被 `RemoveMessage` 从 checkpoint 中真正删除。

---

## 4. 依赖

| 包名 | 用途 | 必装 |
|------|------|------|
| `langgraph` | 图编排核心 | ✅ |
| `langgraph-checkpoint` | Checkpoint 基础抽象 | ✅ |
| `langgraph-checkpoint-sqlite` | SQLite 持久化实现 | 推荐（未装时自动降级） |

安装：

```bash
pip install langgraph-checkpoint-sqlite
```

---

## 5. 会话隔离机制

```
用户 A (Cookie: session_id=abc-123)  →  thread_id="abc-123"  →  独立的 checkpoint 记录
用户 B (Cookie: session_id=def-456)  →  thread_id="def-456"  →  独立的 checkpoint 记录
无 Cookie 的请求                     →  thread_id="default"   →  共享兜底线程
```

FastAPI 端点（`/chat`、`/upload`、`/validate`）在首次请求时自动生成 `session_id`（UUID），通过 Cookie 返回给浏览器，后续请求自动携带。

---

## 6. 历史截断策略

```
invoke 结束后:
  messages 长度 > max_conversation_history?
    ├─ 是 → 计算 to_drop = messages[:-N]
    │       生成 RemoveMessage(id=m.id) 列表
    │       调用 graph.update_state(thread_config, {"messages": remove_ops})
    │       截断返回值 result["messages"] = result["messages"][-N:]
    └─ 否 → 不做处理
```

**关键点**：`RemoveMessage` 通过 `MessagesState` 的 `add_messages` reducer 在持久化层生效，下一轮从 checkpoint 恢复时也只剩窗口内的消息，避免 token 成本线性膨胀。

---

## 7. 过期清理策略

SQLite checkpoint 默认包含 `checkpoints` 和 `writes` 两张 LangGraph 内部表，但它们本身没有可靠的 `updated_at` 字段。因此项目额外维护一张轻量级访问表：

```sql
CREATE TABLE IF NOT EXISTS langgraph_thread_access (
    thread_id TEXT PRIMARY KEY,
    last_seen INTEGER NOT NULL
)
```

清理流程：

```
每次 process_query 成功 invoke 后:
  _mark_thread_seen(session_id)
    └─ 更新 langgraph_thread_access.last_seen

后台线程按 LANGGRAPH_CHECKPOINT_CLEANUP_INTERVAL_SECONDS 定期运行:
  1. 计算 cutoff = 当前时间 - retention_days
  2. 找出 last_seen < cutoff 的 thread_id
  3. 删除 writes 中对应 thread_id
  4. 删除 checkpoints 中对应 thread_id
  5. 删除 langgraph_thread_access 中对应记录
```

默认策略：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LANGGRAPH_CHECKPOINT_CLEANUP_ENABLED` | `true` | 是否启用后台清理线程 |
| `LANGGRAPH_CHECKPOINT_RETENTION_DAYS` | `30` | 保留最近 30 天活跃会话 |
| `LANGGRAPH_CHECKPOINT_CLEANUP_INTERVAL_SECONDS` | `86400` | 每 24 小时检查一次 |

---

## 8. 降级与容错

| 场景 | 行为 |
|------|------|
| `langgraph-checkpoint-sqlite` 未安装 | 打印 warning，回退 `MemorySaver`（仅进程内有效） |
| SQLite 文件创建 / 连接失败 | 同上 |
| `LANGGRAPH_CHECKPOINT_BACKEND=memory` | 直接使用 `MemorySaver` |
| `graph.update_state()` 截断失败 | 打印 warning，仅截断当前返回值，不影响本次响应 |
| 无 `session_id` Cookie | 使用 `thread_id="default"`，功能正常但不隔离 |
| 清理功能关闭 | `LANGGRAPH_CHECKPOINT_CLEANUP_ENABLED=false` 时不启动后台清理线程 |
| 访问表不存在 | 自动创建 `langgraph_thread_access` |
| 清理失败 | 打印 warning，不影响在线请求 |

---

## 9. 文件清单

| 文件 | 改动内容 |
|------|----------|
| `config.py` | 新增 `CheckpointConfig` 类 |
| `agents/agent_decision.py` | checkpoint 基础设施、图编译缓存、单例化、历史截断、过期清理 |
| `app.py` | `/chat`、`/upload`、`/validate` 透传 `session_id`，启动 checkpoint 后台清理线程 |
| `requirements.txt` | 新增 `langgraph-checkpoint-sqlite` |
| `.env.example` | 新增 checkpoint 相关环境变量说明 |

---

## 10. 后续演进（TODO）

- [ ] **HITL 中断恢复**：使用 `compile(interrupt_before=["human_validation"])` + `Command(resume=...)` 实现真正的人工验证断点续跑，避免验证回包重跑全图前置节点。适合在影像推理模型真正接入后落地。
- [ ] **异步 Checkpoint**：当并发量上升时，可切换为 `AsyncSqliteSaver` 或 `PostgresSaver`，避免 SQLite 写锁成为瓶颈。
