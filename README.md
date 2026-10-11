<div align="center">
  <h1>🧠 NoteWise AI</h1>
  <p><strong>Production-Grade Multi-Stage RAG & Document Intelligence Platform</strong></p>

  <p>
    <a href="https://notewise-ai-sarkar.streamlit.app" target="_blank">
      <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
    </a>
    <img src="https://github.com/LogicKatanaX/NoteWise-AI/actions/workflows/ci.yml/badge.svg" alt="CI">
    <img src="https://img.shields.io/badge/Model-Google%20Gemini%203.8%20%7C%203.7%20%7C%203.5-blue?logo=google" alt="Gemini">
    <img src="https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en--v1.5-orange" alt="BAAI Embeddings">
    <img src="https://img.shields.io/badge/Re--Ranker-ms--marco--MiniLM--L--6-teal" alt="CrossEncoder">
    <img src="https://img.shields.io/badge/Orchestration-LlamaIndex%20Vector%20Core-purple" alt="LlamaIndex">
    <img src="https://img.shields.io/badge/Retrieval_Latency-~40ms-brightgreen" alt="Latency">
    <img src="https://img.shields.io/badge/Evaluation-RAGAS--Inspired-red" alt="RAGAS">
    <img src="https://img.shields.io/badge/Tests-pytest-yellow" alt="Tests">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </p>

  <h3>
    🌐 <a href="https://notewise-ai-sarkar.streamlit.app">Experience the Live Application Here</a>
  </h3>
</div>

---

## 📌 Executive Overview

**NoteWise AI** is a production-quality, multi-stage Retrieval-Augmented Generation (RAG) platform for deep PDF analysis, resume intelligence, and interview preparation. It goes beyond basic vector search with a **bi-encoder + cross-encoder hybrid retrieval pipeline**, **multi-turn conversation memory**, and **RAGAS-inspired response evaluation** — all deployed on Streamlit Community Cloud with zero-GPU constraints.

---

## ⚡ Key Architecture & Features

### 1. 🔍 Multi-Stage Hybrid Retrieval (~40–60ms Total)
- **Stage 1 — Dense Retrieval**: `BAAI/bge-small-en-v1.5` bi-encoder retrieves top-5 candidate chunks via cosine similarity in ~40ms.
- **Stage 2 — Cross-Encoder Re-ranking**: `ms-marco-MiniLM-L-6-v2` cross-encoder re-scores all 5 candidates against the query and selects top-2 by semantic alignment.
- This two-stage approach mirrors production RAG at scale (Cohere, Pinecone, Anthropic) — bi-encoder for speed, cross-encoder for precision.

### 2. 🧠 Multi-Turn Conversation Memory
- Maintains a **4-turn sliding window** of conversation history.
- Every Gemini generation prompt includes the full prior Q&A context, enabling coherent follow-up questions across a document session.
- Stateful memory is preserved per-session in Streamlit session state.

### 3. 📊 RAGAS-Inspired Response Evaluation
- After every RAG response, NoteWise auto-evaluates itself using a second Gemini call as an evaluator.
- Scores displayed inline as color-coded badges:
  - 🎯 **Faithfulness** — Is the answer grounded in the retrieved context?
  - 💡 **Relevance** — Does the answer address the user's question?
  - 📋 **Completeness** — Is the full available context utilized effectively?
- Green (≥0.75) / Yellow (0.45–0.74) / Red (<0.45) visual indicators.

### 4. 🛡️ High-Availability LLM Cascade (Zero-Downtime)
- **Primary**: `gemini-3.8-flash` → **Failover**: `gemini-3.7-flash` → `gemini-3.5-flash-lite`
- Dynamic runtime model discovery via Google GenAI SDK — adapts to API quota and server capacity.

### 5. 📡 Per-Query Telemetry Dashboard
- Every response displays: **model used**, **retrieval latency (ms)**, **generation latency (ms)**, **chunks retrieved/used**, and **estimated token count**.
- Sidebar panel shows the last 5 queries' telemetry for session-level observability.

### 6. 🔒 Anti-Abuse Security
- 1.0s query cooldown throttle + 35-query session cap.
- PyTorch/OpenMP thread locks for container stability on shared cloud infrastructure.

### 7. ⬇ Export & Portability
- Download the full conversation as a structured Markdown file — preserving model metadata, retrieval latency, and source citations.

---

## 🏗️ System Architecture Pipeline

```text
[ User Document (PDF / TXT / MD) ]
             │
             ▼
   pypdf Page Parser
             │
             ▼
  TokenTextSplitter (512 tokens / 50 overlap)
             │
             ▼
  Dense Embedding (BAAI/bge-small-en-v1.5 · CPU · ~40ms)
             │
             ▼
    VectorStoreIndex (In-Memory Semantic Graph)
             │
    [ User Query + Conversation Memory ]
             │
             ▼
  Stage 1: Cosine Retriever (top_k=5)  ──►  ~40ms
             │
             ▼
  Stage 2: CrossEncoder Re-ranker (ms-marco-MiniLM-L-6-v2)
             │  Selects top 2 chunks by semantic alignment
             ▼
  Context Builder + Conversation History Window (4 turns)
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│             Multi-Tier Gemini Inference Cascade             │
│                                                             │
│   Gemini 3.8 Flash  ──(503/429)──► Gemini 3.7 Flash        │
│                                           │                 │
│                                        (503/429)            │
│                                           ▼                 │
│                                  Gemini 3.5 Flash-Lite      │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
  RAGAS Evaluator (Faithfulness · Relevance · Completeness)
             │
             ▼
  Typewriter Streaming UI + Citations + Telemetry Badges
```

---

## 📊 Performance Benchmarks

| Metric | Value |
|---|---|
| Dense retrieval latency | ~40ms (BAAI/bge-small, CPU) |
| Cross-encoder re-ranking | ~15–30ms (MiniLM-L-6, CPU) |
| Total retrieval pipeline | ~55–70ms |
| Average generation latency | ~1.5–3s (Gemini 3.8 Flash) |
| Embedding model size | ~130MB (bge-small-en-v1.5) |
| Re-ranker model size | ~67MB (ms-marco-MiniLM-L-6-v2) |
| Session query budget | 35 queries |
| Test coverage | 4 test modules, 20 tests |

---

## 🧪 Testing & CI

```bash
# Install dependencies
pip install -r requirements.txt pytest

# Run the full test suite
pytest tests/ -v
```

**Test modules:**
| File | What It Tests |
|---|---|
| `tests/test_chunking.py` | TokenTextSplitter chunk count, size limits, metadata, overlap |
| `tests/test_rate_limit.py` | Counter increment, cap enforcement, cooldown logic, session reset |
| `tests/test_export.py` | Chat export header, content, model badge, source citations |
| `tests/test_eval_scores.py` | RAGAS JSON parsing, score clamping, missing keys, badge classes |

CI runs automatically on every push via **GitHub Actions** — see badge above.

---

## 💻 Local Setup & Quickstart

### Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get one free at Google AI Studio](https://aistudio.google.com/))

### Installation
```bash
git clone https://github.com/LogicKatanaX/NoteWise-AI.git
cd NoteWise-AI

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Running Locally
```bash
streamlit run streamlit_app.py
```

Add your key to `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

---

## 📝 Resume Bullet Points (Copy-Paste Ready)

> **NoteWise AI** | Production Document Intelligence Platform  
> `Python · LlamaIndex · Google Gemini 3 · BAAI/bge · CrossEncoder · Streamlit · pytest · GitHub Actions`

- Built a **multi-stage RAG pipeline** — bi-encoder (BAAI/bge-small-en-v1.5) dense retrieval + cross-encoder (ms-marco-MiniLM-L-6-v2) re-ranking + 4-turn conversation memory — achieving sub-70ms total retrieval latency on CPU-only Streamlit Community Cloud
- Integrated **RAGAS-inspired self-evaluation loop** using Gemini as an evaluator, scoring Faithfulness, Relevance, and Completeness as color-coded inline badges on every RAG response
- Implemented **per-query telemetry observability** (retrieval/generation latency, chunk utilization, token estimates) and designed automated **GitHub Actions CI** with 20 pytest unit tests across chunking, rate limiting, export, and eval modules

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
