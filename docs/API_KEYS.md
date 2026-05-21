# Multi-Agent Medical Assistant — API 调用与密钥清单

> 本文档汇总项目中所有外部 / 远程 API 调用、对应的环境变量（API Key）、调用位置和用途，方便部署与排查。
> 所有真实密钥请配置在本地 `.env`（已被 `.gitignore` 忽略），切勿提交。模板见 `.env.example`。

---

## 1. 总览

| # | 服务 | 必填 | 环境变量 | 主要调用位置 |
|---|------|------|----------|--------------|
| 1 | MiMo（小米）| ✅ | `MIMO_API_KEY`, `MIMO_BASE_URL`, `MIMO_CHAT_MODEL` | `config.py`（所有聊天 LLM：决策/对话/RAG/Web Search）|
| 1b | 智谱 AI（GLM）| ✅ | `ZHIPU_API_KEY`, `ZHIPU_BASE_URL` | `config.py`（Embedding + 视觉 LLM）|
| 2 | Tavily Web Search | ✅（启用 Web 搜索时） | `TAVILY_API_KEY` | `agents/web_search_processor_agent/tavily_search.py` |
| 3 | Qdrant Cloud（向量库） | ⚠️ 仅云端 RAG 时 | `QDRANT_URL`, `QDRANT_API_KEY` | `agents/rag_agent/vectorstore_qdrant.py` |
| 4 | HuggingFace Hub | ⚠️ 受限模型 / reranker | `HUGGINGFACE_TOKEN` | `config.RAGConfig.huggingface_token` |
| 5 | ElevenLabs TTS | 可选 | `ELEVEN_LABS_API_KEY` | `config.SpeechConfig`（当前 `/generate-speech` 实际走 gTTS）|
| 6 | 百度语音 ASR | ⚠️ 使用 `/transcribe` 时 | `BAIDU_API_KEY`, `BAIDU_SECRET_KEY` | `app.py::get_baidu_access_token`, `/transcribe` |
| 7 | Heyi 远程分割服务 | 可选（无 Key，仅 URL） | `HEYI_REMOTE_URL` 等 | `agents/image_analysis_agent/heyi_remote_client.py` |
| 8 | PubMed E-utilities | 可选（无 Key） | — | `agents/web_search_processor_agent/pubmed_search.py`（当前已注释）|
| 9 | Google TTS（gTTS） | 无需 Key | — | `app.py::/generate-speech` |

---

## 2. 详细说明

### 2.1 MiMo（小米 Token Plan）

- **环境变量**
  - `MIMO_API_KEY`（必填，启动时强校验，缺失会抛 `ValueError`）
  - `MIMO_BASE_URL`（默认 `https://token-plan-cn.xiaomimimo.com/v1`）
  - `MIMO_CHAT_MODEL`（默认 `mimo-v2.5-pro`）
- **调用方式**：通过 `langchain_openai.ChatOpenAI`（OpenAI 兼容协议）。
- **使用模型**
  - `mimo-v2.5-pro`：决策、对话、Web Search、RAG 总结/分块/响应生成
- **代码位置**：`@d:\ai agent\agentgithub\config.py:38-42`（变量定义）、各 Config 类中的 `self.llm`
- **申请**：<https://mimo.xiaomi.com/>

### 2.1b 智谱 AI（GLM 系列，Embedding + 视觉）

- **环境变量**
  - `ZHIPU_API_KEY`（必填，启动时强校验，缺失会抛 `ValueError`）
  - `ZHIPU_BASE_URL`（默认 `https://open.bigmodel.cn/api/paas/v4/`）
- **调用方式**：通过 `langchain_openai.ChatOpenAI` / `OpenAIEmbeddings` 兼容接口。
- **使用模型**
  - `glm-4.5-air`：医学影像视觉分析（`MedicalCVConfig`）
  - `embedding-3`（2048 维）：RAG 向量化
- **代码位置**：`@d:\ai agent\agentgithub\config.py:86-90`（Embedding）、`@d:\ai agent\agentgithub\config.py:170-175`（视觉 LLM）
- **申请**：<https://open.bigmodel.cn/>

### 2.2 Tavily Web 搜索

- **环境变量**：`TAVILY_API_KEY`（必填，`config.py` 启动时强校验）
- **调用方式**：`langchain_community.tools.tavily_search.TavilySearchResults`，SDK 自动从环境变量读取 Key。
- **代码位置**：`@d:\ai agent\agentgithub\agents\web_search_processor_agent\tavily_search.py:17-44`
- **申请**：<https://tavily.com/>

### 2.3 Qdrant 向量数据库

- **环境变量**
  - `QDRANT_URL`：Qdrant Cloud 实例地址
  - `QDRANT_API_KEY`：实例访问密钥
- **本地模式**：当 `RAGConfig.use_local=True` 时，使用 `./data/qdrant_db` 本地存储，无需上述变量。
- **代码位置**：`@d:\ai agent\agentgithub\agents\rag_agent\vectorstore_qdrant.py:32-36`
- **申请**：<https://cloud.qdrant.io/>

### 2.4 HuggingFace Hub Token

- **环境变量**：`HUGGINGFACE_TOKEN`
- **用途**：下载受限模型 / reranker（`cross-encoder/ms-marco-TinyBERT-L-6`）。
- **代码位置**：`@d:\ai agent\agentgithub\config.py:113`
- **申请**：<https://huggingface.co/settings/tokens>

### 2.5 ElevenLabs（文本转语音，当前未启用）

- **环境变量**：`ELEVEN_LABS_API_KEY`
- **状态**：`config.SpeechConfig` 中已读取，但 `/generate-speech` 端点实际改用 `gTTS`，本 Key 当前不会被请求。如要切回 ElevenLabs 需改造 `app.py`。
- **代码位置**：`@d:\ai agent\agentgithub\config.py:173-174`、`@d:\ai agent\agentgithub\app.py:716-738`
- **申请**：<https://elevenlabs.io/>

### 2.6 百度语音 ASR

- **环境变量**
  - `BAIDU_API_KEY`
  - `BAIDU_SECRET_KEY`
- **调用流程**
  1. `GET https://aip.baidubce.com/oauth/2.0/token`（client_credentials 换 access_token）
  2. `POST https://vop.baidu.com/server_api?dev_pid=1537&...&token=...`（音频 base64 上传识别）
- **代码位置**：`@d:\ai agent\agentgithub\app.py:65-74`、`@d:\ai agent\agentgithub\app.py:610-700`
- **要求**：音频需为 16k/16bit 单声道 PCM/WAV/AMR/M4A。
- **申请**：<https://console.bce.baidu.com/ai/#/ai/speech/overview/index>

### 2.7 Heyi 远程医疗分割服务（同学部署）

- **不需要 API Key**，但需配置访问地址。
- **环境变量**
  - `HEYI_REMOTE_URL`（默认示例 `http://222.198.105.83:8000`，留空则只走本地降级）
  - `HEYI_REMOTE_ENABLED`（`true`/`false` 总开关）
  - `HEYI_REMOTE_TIMEOUT`（默认 `120` 秒，3D NIfTI 推理用）
  - `HEYI_REMOTE_HEALTH_TIMEOUT`（默认 `3` 秒）
  - `HEYI_REMOTE_TASK`（`auto` / `hemorrhage` / `ischemia`）
- **端点**：`/health`（探活）、`/segment`（分割）
- **代码位置**：`@d:\ai agent\agentgithub\agents\image_analysis_agent\heyi_remote_client.py:1-100`、`@d:\ai agent\agentgithub\config.py:147-162`
- **降级**：远程不可达时自动切到本地 `HeyiVisionAdapter` 演示模式。

### 2.8 PubMed E-utilities（当前已注释）

- **无需 Key**（NCBI 公共接口，建议传 `tool` 与 `email` 参数；高频访问可申请 `api_key`）。
- **代码位置**：`@d:\ai agent\agentgithub\agents\web_search_processor_agent\pubmed_search.py:16-35`、`@d:\ai agent\agentgithub\agents\web_search_processor_agent\web_search_agent.py:15-26`（调用已注释）。
- **如需启用**：在 `WebSearchAgent` 中取消注释并在 `config` 中加入 `pubmed_api_url`（如 `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`）。

### 2.9 Google TTS（gTTS）

- **无需 Key**，使用 `gtts` Python 包合成 mp3。
- **代码位置**：`@d:\ai agent\agentgithub\app.py:716-755`

---

## 3. 非外部 API 的运行时开关

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `FORCE_OFFLINE_MODE` | `false` | 全部请求只走本地兜底回应器 |
| `ENABLE_OFFLINE_FALLBACK` | `true` | 在线流水线异常时自动降级 |
| `LANGGRAPH_CHECKPOINT_BACKEND` | `sqlite` | LangGraph 断点持久化后端（`sqlite` / `memory`）|
| `LANGGRAPH_CHECKPOINT_SQLITE_PATH` | `./data/langgraph_checkpoints.sqlite` | SQLite 文件路径 |

---

## 4. 最少可用配置（Minimal `.env`）

仅运行核心对话 + RAG（本地 Qdrant）+ Web Search 的最小集合：

```env
MIMO_API_KEY=你的小米Token-Plan-key
ZHIPU_API_KEY=你的智谱key
TAVILY_API_KEY=你的tavily-key
```

启用云端 RAG / 影像分割 / 语音输入，按需追加：

```env
QDRANT_URL=...
QDRANT_API_KEY=...
HUGGINGFACE_TOKEN=...
HEYI_REMOTE_URL=http://222.198.105.83:8000
BAIDU_API_KEY=...
BAIDU_SECRET_KEY=...
ELEVEN_LABS_API_KEY=...
```

---

## 5. 安全提醒

- **永远不要**把真实 Key 写进 `config.py`、`.env.example` 或提交到 Git。
- 若怀疑泄露，立即在对应平台轮换（regenerate）密钥。
- 部署到服务器时，建议通过容器/系统级环境变量注入，而不是 `.env` 文件落盘。
