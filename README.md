<div align="center">

# ⚕️ Multi-Agent Medical Assistant

**AI-powered multi-agent system for medical diagnosis & assistance**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.13+-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-Apache_2.0-4B8BBE?style=for-the-badge)](LICENSE)

[Overview](#-overview) ·
[Features](#-features) ·
[Architecture](#-architecture) ·
[Quick Start](#-quick-start) ·
[Project Structure](#-project-structure) ·
[HTTP API](#-http-api) ·
[Brain Imaging Pipeline](#-brain-imaging-pipeline) ·
[Docs](#-docs) ·
[Changelog](#-changelog)

</div>

---

## 📌 Overview

The **Multi-Agent Medical Assistant** is an AI-powered chatbot for **medical diagnosis,
research and patient interactions**, optimized for lightweight deployment, fast iteration
and an enhanced front-end experience.

It orchestrates several specialized agents:

| Agent | Role |
|-------|------|
| 🤖 **Conversation Agent** | General health Q&A and small talk |
| 📚 **RAG Agent** | Retrieval over ingested medical literature (Qdrant) |
| 🌐 **Web Search Agent** | Up-to-date medical information via Tavily |
| 🧠 **Brain Tumor Agent** | MRI tumor segmentation + AI-assisted diagnosis |
| 🩸 **Brain Stroke Agent** | CT/MRI stroke segmentation (ischemia / hemorrhage) with **remote inference** |
| 👨‍⚕️ **Human-in-the-Loop** | Mandatory review for any CV agent output |

> Brain CV agents are powered by the in-repo **`heyi-Trans-master`** general-purpose
> Vision Transformer framework (ViT-B/16 backbone + lightweight binary segmentation head).

---

## ✨ Features

- 🤖 **Multi-Agent Orchestration** — LangGraph drives task routing; users may rely on
  automatic routing or **manually force-select** a specific agent.
- 💾 **Persistent LangGraph Checkpoints** — SQLite-backed conversation state with
  per-session isolation, history-window pruning and automatic expired-thread cleanup.
- 🔍 **Agentic RAG** — `PyPDFLoader` + semantic chunking + Qdrant Cloud hybrid search,
  with input/output guardrails.
- 🏥 **3-Stage Imaging Pipeline** — `segment_image → mark_lesion → diagnose`, returning
  Chinese AI-assisted diagnosis text and a side-by-side visualization.
- 🌐 **Real-time Web Search** — Tavily integration for the latest medical research.
- 👩‍⚕️ **Expert Oversight** — Human validation gate before any CV diagnosis is finalized.
- 🌏 **Bilingual UI** — Fully localized EN/ZH front-end, including dynamic agent capability
  descriptions.
- 🛡️ **Resilient Remote Inference** — Stroke agent prefers a remote Heyi service; transient
  failures no longer permanently lock the agent into local demo mode.

---

## 🧠 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Frontend (Bootstrap 5)                     │
│  /chat  ─┐                                                        │
│  /upload ┤  multipart  ──►  FastAPI                               │
│          │  + stroke_task                                         │
└──────────┴────────────────────┬──────────────────────────────────┘
                                │
            ┌───────────────────▼────────────────────┐
            │   LangGraph Agent Decision             │
            │   auto/force-select + checkpoint       │
            └─┬─────────┬─────────┬─────────┬─────┬──┘
              │         │         │         │     │
        ┌─────▼──┐ ┌────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌▼──────┐
        │ Convo  │ │  RAG   │ │  Web   │ │ Tumor│ │Stroke │
        │ Agent  │ │ Agent  │ │ Search │ │ CV   │ │ CV    │
        └────────┘ └────────┘ └────────┘ └──────┘ └───┬───┘
                                                     │
                                       ┌─────────────▼─────────────┐
                                       │ Heyi Remote API (v2.0)    │
                                       │  /health  /segment        │
                                       │           /segment/preview│
                                       │  + local ViT fallback     │
                                       └───────────────────────────┘
```

LangGraph state is checkpointed per browser session:

```text
session_id cookie → LangGraph thread_id → SqliteSaver checkpoint
```

By default, checkpoints are stored at `./data/langgraph_checkpoints.sqlite`.
The system keeps only the recent conversation window in the prompt and runs a
background cleanup job for long-inactive threads.

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- API keys for **MiMo**, **Zhipu AI**, **Tavily**, **Qdrant Cloud** (RAG only)
- Optional: a reachable Heyi remote segmentation service URL

### 2. Install

```bash
git clone https://github.com/mobai828/agentgithub.git
cd agentgithub
pip install -r requirements.txt
```

### 3. Configure

Copy the template and fill in your secrets:

```bash
cp .env.example .env
```

| Env var | Required | Default | Notes |
|---------|:---:|---------|-------|
| `MIMO_API_KEY` | ✅ | — | Main chat LLM provider for decision/conversation/RAG/Web Search |
| `MIMO_BASE_URL` | ⛔ | `https://token-plan-cn.xiaomimimo.com/v1` | MiMo Token Plan OpenAI-compatible endpoint |
| `MIMO_CHAT_MODEL` | ⛔ | `mimo-v2.5-pro` | Main chat model |
| `ZHIPU_API_KEY` | ✅ | — | Embedding and medical vision LLM |
| `ZHIPU_BASE_URL` | ⛔ | `https://open.bigmodel.cn/api/paas/v4/` | Zhipu OpenAI-compatible gateway |
| `TAVILY_API_KEY` | ✅ | — | Web Search Agent |
| `QDRANT_URL` | ⚠️ | — | Required only when using RAG |
| `QDRANT_API_KEY` | ⚠️ | — | Required only when using RAG |
| `BAIDU_API_KEY` | ⚠️ | — | Required only when using `/transcribe` |
| `BAIDU_SECRET_KEY` | ⚠️ | — | Required only when using `/transcribe` |
| `HEYI_REMOTE_URL` | ⛔ | _empty_ | Empty → skip remote, use local fallback |
| `HEYI_REMOTE_ENABLED` | ⛔ | `true` | Master switch |
| `HEYI_REMOTE_TIMEOUT` | ⛔ | `120` | `/segment` total timeout (s) |
| `HEYI_REMOTE_HEALTH_TIMEOUT` | ⛔ | `3` | `/health` timeout (s) |
| `HEYI_REMOTE_TASK` | ⛔ | `auto` | Default `task`: `auto` / `hemorrhage` / `ischemia` |
| `FORCE_OFFLINE_MODE` | ⛔ | `false` | Skip all online calls; use offline fallback agent |
| `ENABLE_OFFLINE_FALLBACK` | ⛔ | `true` | Auto-fallback when online pipeline fails |
| `LANGGRAPH_CHECKPOINT_BACKEND` | ⛔ | `sqlite` | `sqlite` for persistent checkpoint, `memory` for test-only state |
| `LANGGRAPH_CHECKPOINT_SQLITE_PATH` | ⛔ | `./data/langgraph_checkpoints.sqlite` | SQLite checkpoint file |
| `LANGGRAPH_CHECKPOINT_CLEANUP_ENABLED` | ⛔ | `true` | Enable expired checkpoint cleanup |
| `LANGGRAPH_CHECKPOINT_RETENTION_DAYS` | ⛔ | `30` | Keep active threads for N days |
| `LANGGRAPH_CHECKPOINT_CLEANUP_INTERVAL_SECONDS` | ⛔ | `86400` | Cleanup interval |

Local secret storage best practice:

- Keep all personal keys in local `.env` only (already ignored by `.gitignore`).
- Commit only `.env.example` with blank placeholders.
- If you need a personal backup, create another local file such as `.env.private.backup` (also ignored by `.gitignore` due to `.env.*`) and never commit it.

### 4. Run

```bash
python app.py
```

Visit [`http://localhost:8000`](http://localhost:8000) — upload a medical image or ask a
health question, and the system will route to the right agent automatically (or follow
your manual selection in the sidebar).

### 5. (Optional) Ingest your own medical PDFs into RAG

```bash
python ingest_rag_data.py --dir data/raw
```

---

## 🗂️ Project Structure

```text
.
├── app.py                         # FastAPI app, API endpoints, static mounts, background cleanup
├── config.py                      # Centralized runtime configuration
├── ingest_rag_data.py             # RAG document ingestion helper
├── requirements.txt               # Python dependencies
├── agents/
│   ├── agent_decision.py          # LangGraph routing, checkpointing, history pruning
│   ├── rag_agent/                 # Medical RAG pipeline
│   ├── web_search_processor_agent/# Tavily/Web search processing
│   ├── image_analysis_agent/      # Brain tumor/stroke CV agents
│   └── guardrails/                # Local input/output guardrails
├── docs/
│   ├── API_KEYS.md                # External API and secret checklist
│   └── LANGGRAPH_CHECKPOINT.md    # Checkpoint architecture and operations
├── templates/                     # Bootstrap front-end template
├── assets/                        # Static project assets
├── data/                          # RAG data, parsed content, local checkpoint db
├── uploads/                       # Runtime uploads and generated result images (gitignored)
├── tests/                         # Integration and smoke tests
└── heyi-Trans-master/             # Vision Transformer segmentation framework
```

Runtime/generated folders such as `uploads/`, `.venv/`, `.env`, `.env.*`,
`__pycache__/` and local audio files are ignored by Git.

---

## 🔌 HTTP API

> Base URL: `http://<host>:8000` · Content negotiation: all responses are
> `application/json` unless noted. Static assets (result images) are served under
> `/uploads/...` with a `?v=<mtime>` cache-buster.

### Endpoint summary

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Text-only conversation |
| `POST` | `/upload` | Multimodal: image (+ optional text) |
| `POST` | `/api/brain_stroke/segment` | **Direct** stroke pipeline (for external "ischemia/hemorrhage classifier" modules) |
| `POST` | `/validate` | Submit human validation result |
| `POST` | `/transcribe` | Speech-to-text (Baidu ASR) |
| `POST` | `/generate-speech` | Text-to-speech (gTTS) |
| `GET`  | `/health` | Container health probe |

---

### `POST /chat`

JSON body:

```json
{
  "query": "What is ischemic stroke?",
  "language": "en",
  "preferred_agent": "AUTO"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `query` | string | User input (required) |
| `language` | string | `"en"` (default) or `"zh"` |
| `preferred_agent` | string | `AUTO` / `CONVERSATION_AGENT` / `RAG_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT` |

Response:

```json
{
  "status": "success",
  "response": "…markdown text…",
  "agent": "CONVERSATION_AGENT"
}
```

---

### `POST /upload`

`multipart/form-data`:

| Field | Type | Notes |
|-------|------|-------|
| `image` | file | PNG / JPG / JPEG (required) |
| `text` | string | Optional question alongside the image |
| `language` | string | `"en"` / `"zh"` |
| `preferred_agent` | string | `BRAIN_TUMOR_AGENT` / `BRAIN_STROKE_AGENT` recommended |
| `stroke_task` | string | **Stroke agent only**: `auto` / `hemorrhage` / `ischemia` (also accepts `缺血` / `出血` / `梗死`) |

Response (stroke example):

```json
{
  "status": "success",
  "agent": "BRAIN_STROKE_AGENT, HUMAN_VALIDATION",
  "stroke_task": "ischemia",
  "response": "**AI 初步结论（脑卒中检测）**\n\n- 任务类型：**缺血灶（ischemia）**\n- 共检测到 **1** 个可疑区域；主病灶位于 **右脑上部**。\n- 病灶面积约 **11438** 像素，占切片面积 **2.94%**，属于 **中等范围病灶**。\n- 主病灶外接矩形：x=277, y=91, w=101, h=200\n- ✅ 本次分割由 **远程 Heyi 分割服务 (v2.0)** 完成。",
  "result_image": "/uploads/brain_stroke_output/brain_stroke_plot.png?v=1714032123456"
}
```

---

### ⭐ `POST /api/brain_stroke/segment`

A **headless** entry point for external classifier modules — the team that decides
between *ischemia* and *hemorrhage* upstream can call this directly and receive the
structured pipeline result without going through the chatbot routing.

`multipart/form-data`:

| Field | Type | Notes |
|-------|------|-------|
| `image` | file | PNG / JPG / JPEG (required) |
| `task`  | string | `auto` / `hemorrhage` / `ischemia` (default `auto`) |

Response schema:

```json
{
  "status": "success",
  "task":   "ischemia",
  "source": "remote",
  "stages": {
    "segmentation":  true,
    "lesion_marking": true,
    "ai_diagnosis":   true
  },
  "diagnosis":    "**AI 初步结论（脑卒中检测）**\n…",
  "message":      "脑卒中检测完成（远程 Heyi 分割服务，task=ischemia）。",
  "result_image": "/uploads/brain_stroke_output/brain_stroke_plot.png?v=1714032123456"
}
```

| Field | Meaning |
|-------|---------|
| `status` | `"success"` / `"error"` / `"not_implemented"` |
| `task` | Task that actually ran (`auto`/`hemorrhage`/`ischemia`) |
| `source` | `"remote"` = real model · `"local"` = demo fallback (untrained ViT head, **not clinical**) |
| `stages` | Which pipeline stages completed |
| `diagnosis` | Markdown AI-assisted diagnosis text — same body as `/upload` `response` |
| `result_image` | Path to the visualization PNG (cache-busted via `?v=<ms>`) |
| `message` | One-line status line for logs |

Python client example:

```python
import requests

with open("ct_axial.png", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/brain_stroke/segment",
        files={"image": ("ct_axial.png", f, "image/png")},
        data={"task": "ischemia"},   # from upstream classifier
        timeout=180,
    )
result = r.json()
print(result["status"], result["source"], result["task"])
print(result["diagnosis"])
print("Visualization:", result.get("result_image"))
```

---

### `POST /validate`

`multipart/form-data`:

| Field | Type | Notes |
|-------|------|-------|
| `validation_result` | string | `"yes"` or `"no"` |
| `comments` | string | Optional reviewer comment |
| `language` | string | `"en"` / `"zh"` |

Response:

```json
{
  "status":   "validated",
  "message":  "**Output confirmed by human validator:**",
  "response": "…follow-up text from the conversation agent…"
}
```

> Validation replies are explicitly routed to `CONVERSATION_AGENT` so the LLM never
> mis-routes back into the CV pipeline (which would re-run on a missing image and emit
> a stale "no image received" message).

---

### `POST /transcribe`, `POST /generate-speech`, `GET /health`

| Endpoint | Body | Returns |
|----------|------|---------|
| `/transcribe` | `multipart` `audio` (webm/wav) | `{"transcript": "…"}` |
| `/generate-speech` | JSON `{"text": "…", "language": "en"\|"zh"}` | `audio/mpeg` stream |
| `/health` | — | `{"status":"healthy"}` |

---

## 🩸 Brain Imaging Pipeline

The two brain CV agents share a 3-stage skeleton, both wired through the in-repo
[`heyi-Trans-master`](./heyi-Trans-master/README.md) framework:

```
segment_image  ─►  mark_lesion  ─►  diagnose
   (mask)         (overlay PNG)     (markdown text)
```

### Stroke agent: remote-first

```
            ┌────────────────────────────────────────┐
            │  POST /upload (or /api/brain_stroke…)  │
            └──────────────────┬─────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │  BrainStrokeAgent       │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │  /health  ✓?            │   ──► ❌ go local
                  │  /segment       (mask)  │
                  │  /segment/preview (PNG) │   ──► soft-fail OK
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ mark_lesion             │
                  │  • prefer preview PNG   │
                  │  • else local overlay   │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ diagnose (mask stats)   │
                  │ + 任务类型 / 来源标注  │
                  └─────────────────────────┘
```

Resilience (added after observing "one transient error → permanent demo mode"):

| Rule | Why |
|------|-----|
| `/segment` failure does **not** flip `_remote_alive` to permanent False | Allows the next request to retry the remote service |
| Consecutive failures (`>= 3`) trigger a **60 s cooldown** before re-checking `/health` | Avoids hammering an unhealthy service |
| `/segment/preview` failure is **non-fatal** — local overlay is rebuilt from the (still valid) remote mask | Keeps high-quality results when only the visualization endpoint hiccups |
| `/segment/preview` returning `4xx` retries **once without `task`** | Tolerates older server builds that only accept `task` on `/segment` |
| Both PNG and JPEG responses are accepted | The visualization endpoint may switch encoders without breaking us |

### Demo vs Production

| Mode | Trigger | Quality | Diagnosis text |
|------|---------|---------|----------------|
| **Production (remote)** | `/health` ✓ on configured `HEYI_REMOTE_URL` | Real fine-tuned segmentation | Marked `✅ 远程 Heyi 分割服务 (v2.0)` |
| **Production (local)** | A compatible `.pth` weight file under `models/` | Real fine-tuned segmentation | No demo warning |
| **Demo** | No remote, no weights | Noisy high-frequency response from an untrained head | `⚠️ 演示模式` — explicit warning, **not for clinical use** |

Drop your fine-tuned weights here:

```
agents/image_analysis_agent/
├── brain_tumor_agent/models/brain_tumor_segmentation.pth
└── brain_stroke_agent/models/brain_stroke_segmentation.pth
```

The adapter accepts these state-dict layouts:

1. `{"encoder": …, "decoder": …}` — **recommended**
2. Decoder-only `state_dict`
3. Flat dict with `encoder.*` / `decoder.*` prefixes
4. Encoder-only `state_dict`

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| Backend | FastAPI · Uvicorn |
| Orchestration | LangGraph · LangChain |
| LLM | MiMo Token Plan (`mimo-v2.5-pro`) for chat/routing/RAG/Web · Zhipu AI for embeddings and medical vision |
| Vector DB | Qdrant Cloud |
| Web Search | Tavily |
| Vision | PyTorch · Torchvision · OpenCV (headless) · `heyi-Trans-master` ViT |
| Speech | Baidu ASR · gTTS |
| Frontend | HTML · CSS · JavaScript · Bootstrap 5 · Marked.js |

---

## 📚 Docs

| Document | Purpose |
|----------|---------|
| [`docs/API_KEYS.md`](docs/API_KEYS.md) | External API providers, required environment variables and call sites |
| [`docs/LANGGRAPH_CHECKPOINT.md`](docs/LANGGRAPH_CHECKPOINT.md) | LangGraph checkpoint architecture, persistence, cleanup and operations |
| [`agents/README.md`](agents/README.md) | Agent-level module overview |
| [`heyi-Trans-master/README.md`](heyi-Trans-master/README.md) | Heyi vision transformer framework details |

Competition/demo-only materials are kept under `docs/competition_materials/` when present, so the repository root stays focused on runtime files.

---

## ✅ Testing

```bash
pytest -q
```

End-to-end Heyi remote integration test (real network, auto-skips when offline):

```bash
python tests/test_heyi_remote.py
```

Verifies:

- `/health` availability
- `/segment` upload/download contract (NIfTI mask)
- `/segment/preview` PNG/JPEG visualization
- `auto` / `hemorrhage` / `ischemia` task switching
- Graceful error handling via `HeyiRemoteError`

---

## 📝 Changelog

> Most recent first.

### `v2026.05` — LangGraph checkpoint persistence and repository cleanup

- 💾 **Persistent LangGraph checkpoints** — SQLite-backed `SqliteSaver` with `MemorySaver`
  fallback.
- 🧵 **Per-session isolation** — browser `session_id` is mapped to LangGraph `thread_id`.
- ✂️ **Checkpoint-aware history pruning** — old messages are removed from persisted state
  via `RemoveMessage` + `graph.update_state()`.
- ♻️ **Expired checkpoint cleanup** — background task removes long-inactive threads using
  the `langgraph_thread_access` table.
- ⚡ **Runtime optimization** — compiled graph, `MedicalRAG`, and `WebSearchProcessorAgent`
  are reused instead of rebuilt on every request.
- 📚 **Documentation update** — README, `.env.example`, API key inventory and checkpoint
  operations docs now reflect MiMo + LangGraph checkpoint configuration.

### `v2026.04` — Stroke pipeline overhaul

- 🩸 **Explicit `task` parameter end-to-end** — frontend chip selector → `/upload`
  `stroke_task` → `BrainStrokeAgent.predict(task=…)` → `HeyiRemoteClient.segment(task=…)`.
  Chinese aliases (`缺血` / `出血` / `梗死`) auto-normalized on both ends.
- ⭐ **New direct endpoint** `POST /api/brain_stroke/segment` for external classifiers
  to call the pipeline without traversing the chatbot routing.
- 🖼️ **Result image now uses `/segment/preview`** — the served-by-the-trained-model
  visualization is written verbatim to `brain_stroke_plot.png`. Local overlay is only
  used when the preview endpoint hiccups.
- 💧 **Cache-busting** — every `result_image` URL carries `?v=<file_mtime_ms>` so
  uploading a second image never shows the first one's segmentation.
- 🔁 **Stale-image safety** — old `brain_*_plot.png` is removed before each new
  prediction; a failed run no longer surfaces the previous result.

### Resilience fixes

- 🩹 **`_remote_alive` is no longer sticky** — a single transient `/segment` failure
  used to lock the agent into permanent local demo mode. Now uses
  `consecutive_failures + cooldown_window` (3 fails ⇒ 60 s cooldown ⇒ re-probe).
- 🩹 **`/segment/preview` is more lenient** — accepts JPEG, retries once without
  `task` on a 4xx, never demotes the agent on its own.

### UI polish

- 🪟 **Manual collapse for the CV upload panel** — header chevron button to fold the
  dropzone into a slim "Upload another image" pill at any time.
- 🗨️ **Collapsible "Add comments" textarea** in the human-validation panel — keeps
  the segmentation result above the fold.
- 🎯 **Stroke subtype chip selector** — auto / 缺血 / 出血; only shown for
  `BRAIN_STROKE_AGENT`.

### Earlier

- ⚙️ Lightweight RAG (`PyPDFLoader`), Tavily web search, dynamic agent routing,
  bilingual UI, validation flow fix (validation replies route back to
  `CONVERSATION_AGENT`).

---

## 📜 License

This project is licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).

