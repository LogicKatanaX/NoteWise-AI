import os

# Prevent PyTorch & Tokenizer deadlocks on Streamlit Community Cloud
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import shutil
import tempfile
import torch

# Force single-threaded PyTorch to prevent OpenMP container freeze
torch.set_num_threads(1)

import streamlit as st
import pypdf
from google import genai

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

import base64

# ---------------------------------------------------------------------------
# Page Configuration & Assets
# ---------------------------------------------------------------------------
LOGO_ICON_PATH = "assets/logo_icon.png"

def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

LOGO_B64 = get_base64_image(LOGO_ICON_PATH)

st.set_page_config(
    page_title="NoteWise AI | Production Document Intelligence Platform",
    page_icon=LOGO_ICON_PATH if os.path.exists(LOGO_ICON_PATH) else "🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Claude.ai Dark Theme Exact Aesthetic Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #f4f4f5;
    }

    /* Claude Deep Obsidian Backgrounds */
    .stApp {
        background-color: #18181b !important;
    }

    [data-testid="stSidebar"] {
        background-color: #121214 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 5rem;
        max-width: 820px;
    }

    /* Claude's Exact Center Heading */
    .claude-hero-container {
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 2.2rem;
    }
    
    .claude-hero-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.75rem;
        font-weight: 400;
        letter-spacing: -0.02em;
        color: #f4f4f5;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        margin: 0;
    }

    .claude-flower {
        color: #d97757;
        font-size: 2.5rem;
        line-height: 1;
        display: inline-block;
        font-style: normal;
    }

    .claude-hero-subtitle {
        font-size: 0.95rem;
        color: #71717a;
        margin-top: 8px;
        font-weight: 400;
    }

    /* Claude Floating Prompt Suggestion Pills */
    .claude-pills-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
        margin-bottom: 24px;
    }

    .claude-pill-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #242220;
        border: 1px solid #36332e;
        color: #d4d4d8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        text-decoration: none;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .claude-pill-btn:hover {
        background-color: #2e2b27;
        border-color: #d97757;
        color: #f4f4f5;
    }

    /* Claude Chat Messages */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 16px 0 !important;
    }

    /* User Message - Claude Style Dark Pill Card */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #272522 !important;
        border: 1px solid #383531 !important;
        border-radius: 16px !important;
        padding: 14px 20px !important;
        margin-bottom: 16px !important;
    }

    /* Claude Artifact / Source Box */
    .claude-artifact-box {
        background-color: #1f1d1b;
        border: 1px solid #33302c;
        border-left: 3px solid #d97757;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.83rem;
        color: #d4d4d8;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.55;
    }

    /* Claude Sidebar Cards */
    .sidebar-card {
        background-color: #1a1918;
        border: 1px solid #2e2c29;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 12px;
        font-size: 0.83rem;
        color: #a1a1aa;
        line-height: 1.6;
    }

    /* Claude Expander Drawer */
    div[data-testid="stExpander"] {
        border: 1px solid #2e2c29 !important;
        border-radius: 12px !important;
        background-color: #1a1918 !important;
    }

    /* Claude Floating Input Bar */
    div[data-testid="stChatInput"] {
        border-radius: 18px !important;
        border: 1px solid #383531 !important;
        background-color: #22201e !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
        padding: 4px !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #d97757 !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #f4f4f5 !important;
    }

    /* Claude Style Rounded Action Buttons */
    .stButton > button {
        border-radius: 9999px !important;
        border: 1px solid #383531 !important;
        background-color: #22201e !important;
        color: #e4e4e7 !important;
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        padding: 6px 12px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: #d97757 !important;
        color: #d97757 !important;
        background-color: #2b2824 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #d97757 !important;
        color: #ffffff !important;
        border-color: #d97757 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #c26547 !important;
        border-color: #c26547 !important;
    }

    /* Model Pill Tag */
    .model-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(217, 119, 87, 0.12);
        border: 1px solid rgba(217, 119, 87, 0.35);
        color: #d97757;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* Eval metric badges */
    .eval-badge-row {
        display: flex;
        gap: 8px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .eval-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid;
    }
    .eval-badge.high {
        background-color: rgba(34, 197, 94, 0.1);
        border-color: rgba(34, 197, 94, 0.35);
        color: #22c55e;
    }
    .eval-badge.medium {
        background-color: rgba(234, 179, 8, 0.1);
        border-color: rgba(234, 179, 8, 0.35);
        color: #eab308;
    }
    .eval-badge.low {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.35);
        color: #ef4444;
    }

    /* Telemetry row */
    .telemetry-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 6px;
        font-size: 0.72rem;
        color: #71717a;
        font-family: 'JetBrains Mono', monospace;
    }
    .telemetry-item {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .telemetry-val {
        color: #a1a1aa;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Rate Limiting & Anti-Abuse Guard
# ---------------------------------------------------------------------------
RATE_LIMIT_COOLDOWN = 1.0
MAX_QUERIES_PER_SESSION = 35

def get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Local Embeddings Ingestion
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.transformations = [TokenTextSplitter(chunk_size=512, chunk_overlap=50)]
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu"
    )

@st.cache_resource(show_spinner=False)
def load_reranker():
    """Load cross-encoder for re-ranking retrieved chunks."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

Settings.embed_model = load_embedding_model()

# ---------------------------------------------------------------------------
# Multi-Tier Gemini 3 Cascade Engine
# ---------------------------------------------------------------------------
def generate_gemini_content(prompt: str, api_key: str) -> tuple[str, str]:
    """Returns (answer_text, model_name_used)."""
    client = genai.Client(api_key=api_key)
    
    preferred = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
    ]
    
    candidate_models = list(preferred)
    try:
        discovered = []
        for m in client.models.list():
            m_id = m.name.replace("models/", "")
            if ("flash" in m_id.lower() or "pro" in m_id.lower()) and m_id not in candidate_models:
                discovered.append(m_id)
        candidate_models.extend(discovered)
    except Exception:
        pass

    last_error = None
    for model_name in candidate_models:
        try:
            print(f"[NoteWise] Querying {model_name}...", flush=True)
            t_start = time.time()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            elapsed = time.time() - t_start
            print(f"[NoteWise] ✅ {model_name} succeeded in {elapsed:.2f}s!", flush=True)
            if response and response.text:
                return response.text.strip(), model_name
        except Exception as e:
            err_msg = str(e)
            last_error = e
            print(f"[NoteWise ⚠️] {model_name} unavailable: {err_msg[:80]}... Rolling over to next model...", flush=True)
            continue

    if last_error:
        err_str = str(last_error)
        if "429" in err_str or "resource_exhausted" in err_str:
            return "⚠️ **Daily API quota reached on free tier.** Please wait or configure your own key in settings.", "none"
        raise last_error

    return "⚠️ Could not generate an answer from the document. Please try asking in a different way.", "none"

def evaluate_rag_response(context: str, question: str, answer: str, api_key: str) -> dict:
    """Run Gemini-based RAGAS-inspired evaluation. Returns dict with scores."""
    eval_prompt = (
        "You are an objective evaluator of AI-generated answers from a RAG system. "
        "Score the following response on 3 dimensions (0.0 to 1.0 each). Be strict and precise.\n\n"
        f"Context (retrieved document chunks):\n{context[:600]}\n\n"
        f"User Question: {question}\n\n"
        f"AI Answer: {answer[:600]}\n\n"
        "Return ONLY a JSON object with exactly these keys:\n"
        '{"faithfulness": <float>, "relevance": <float>, "completeness": <float>}\n\n'
        "faithfulness = Is the answer grounded in context only? (1.0 = fully grounded, 0.0 = hallucinated)\n"
        "relevance = Does the answer address the user's question? (1.0 = fully relevant)\n"
        "completeness = Does the answer use the available context effectively? (1.0 = comprehensive)"
    )
    try:
        eval_text, _ = generate_gemini_content(eval_prompt, api_key)
        # Extract JSON from the response
        start = eval_text.find("{")
        end = eval_text.rfind("}") + 1
        if start != -1 and end > start:
            scores = json.loads(eval_text[start:end])
            return {
                "faithfulness": max(0.0, min(1.0, float(scores.get("faithfulness", 0.5)))),
                "relevance": max(0.0, min(1.0, float(scores.get("relevance", 0.5)))),
                "completeness": max(0.0, min(1.0, float(scores.get("completeness", 0.5)))),
            }
    except Exception as e:
        print(f"[NoteWise] Eval error: {e}", flush=True)
    return {}

def score_badge_class(score: float) -> str:
    if score >= 0.75:
        return "high"
    elif score >= 0.45:
        return "medium"
    return "low"

def stream_typewriter(text: str, delay: float = 0.008):
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)

# ---------------------------------------------------------------------------
# Document Ingestion Logic
# ---------------------------------------------------------------------------
def process_uploaded_files(uploaded_files):
    try:
        documents = []
        total_pages = 0
        progress_bar = st.progress(0, text="📄 Parsing documents...")

        for file_idx, file in enumerate(uploaded_files):
            if file.name.lower().endswith(".pdf"):
                reader = pypdf.PdfReader(file)
                total_pages += len(reader.pages)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n\n".join(pages_text).strip()
                if full_text:
                    documents.append(Document(text=full_text, metadata={"file_name": file.name}))
            else:
                raw_text = file.getvalue().decode("utf-8", errors="ignore").strip()
                if raw_text:
                    documents.append(Document(text=raw_text, metadata={"file_name": file.name}))

        progress_bar.progress(33, text="✂️ Chunking text (TokenTextSplitter 512/50)...")

        if not documents:
            progress_bar.empty()
            return None, "No readable text detected in uploaded files (is the PDF a scanned image?).", 0

        splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        progress_bar.progress(55, text="🧠 Generating embeddings (BAAI/bge-small-en-v1.5)...")
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter])

        progress_bar.progress(90, text="📦 Building vector index...")
        time.sleep(0.3)
        progress_bar.progress(100, text="✅ Index ready!")
        time.sleep(0.4)
        progress_bar.empty()

        return index, f"Indexed {len(documents)} document(s) successfully!", total_pages
    except Exception as e:
        return None, f"Document parsing error: {str(e)}", 0

# ---------------------------------------------------------------------------
# Chat Export Helper
# ---------------------------------------------------------------------------
def generate_markdown_export(messages: list) -> str:
    lines = [
        "# NoteWise AI — Session Export",
        f"*Exported on {time.strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]
    for m in messages:
        role_label = "**You**" if m["role"] == "user" else "**NoteWise**"
        meta = ""
        if m["role"] == "assistant":
            if m.get("model_used"):
                meta += f" ·  `{m['model_used']}`"
            if m.get("retrieval_ms") is not None:
                meta += f" · ⚡ {m['retrieval_ms']}ms retrieval"
        lines.append(f"### {role_label}{meta}")
        lines.append(m["content"])
        if m.get("sources"):
            lines.append("\n<details><summary>📚 Source Citations</summary>\n")
            for idx, src in enumerate(m["sources"], 1):
                score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                lines.append(f"**Source #{idx}** · Relevance: {score_str}")
                lines.append(f"> {src['text']}\n")
            lines.append("</details>")
        lines.append("\n---\n")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"files": 0, "pages": 0}

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0.0

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "telemetry" not in st.session_state:
    st.session_state.telemetry = []

api_key = get_api_key()

# ---------------------------------------------------------------------------
# Core RAG Execution with Re-ranking + Memory
# ---------------------------------------------------------------------------
def execute_rag(prompt_text: str):
    """
    Full pipeline:
      1. Build conversation memory (last 4 turns)
      2. Dense vector retrieval (top_k=5)
      3. Cross-encoder re-ranking → select top 2
      4. Gemini generation with memory + context
      5. Return answer, sources, telemetry metadata
    """
    if not api_key:
        return "❌ **GOOGLE_API_KEY is missing!** Configure it in Streamlit Cloud Secrets.", [], {}

    telemetry = {}

    # --- Build conversation memory ---
    history_turns = st.session_state.messages[-8:]  # last 4 Q&A pairs
    if history_turns:
        history_lines = []
        for m in history_turns:
            if m["role"] == "user":
                history_lines.append(f"User: {m['content']}")
            else:
                history_lines.append(f"Assistant: {m['content'][:250]}")
        history_str = "\n".join(history_lines)
    else:
        history_str = ""

    if st.session_state.index is not None:
        # --- Dense Retrieval ---
        t_ret_start = time.time()
        retriever = st.session_state.index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(prompt_text)
        t_ret_end = time.time()
        retrieval_ms = round((t_ret_end - t_ret_start) * 1000)
        print(f"[NoteWise] Retrieved {len(nodes)} nodes in {retrieval_ms}ms", flush=True)

        # --- Cross-Encoder Re-ranking ---
        reranked_nodes = nodes  # fallback if reranker fails
        try:
            reranker = load_reranker()
            pairs = [(prompt_text, node.node.text) for node in nodes]
            t_rerank_start = time.time()
            rerank_scores = reranker.predict(pairs)
            t_rerank_end = time.time()
            rerank_ms = round((t_rerank_end - t_rerank_start) * 1000)
            # Sort by rerank score descending, take top 2
            ranked = sorted(zip(rerank_scores, nodes), key=lambda x: x[0], reverse=True)
            reranked_nodes = [n for _, n in ranked[:2]]
            print(f"[NoteWise] Re-ranked in {rerank_ms}ms, top score: {ranked[0][0]:.3f}", flush=True)
            telemetry["rerank_ms"] = rerank_ms
        except Exception as e:
            rerank_ms = 0
            print(f"[NoteWise] Re-ranker skipped: {e}", flush=True)
            reranked_nodes = nodes[:2]

        # --- Build context ---
        context_blocks = []
        sources = []
        for i, node in enumerate(reranked_nodes, 1):
            clean_text = node.node.text.strip().replace("\n", " ")
            context_blocks.append(f"[Source {i}]: {clean_text}")
            sources.append({
                "score": node.score,
                "text": clean_text[:280] + ("..." if len(clean_text) > 280 else "")
            })

        context_str = "\n\n".join(context_blocks)

        # --- Build RAG Prompt with Memory ---
        memory_section = f"\n[CONVERSATION HISTORY]\n{history_str}\n" if history_str else ""
        rag_prompt = (
            "You are NoteWise, an executive document intelligence assistant and senior technical interview coach. "
            "Use the following retrieved context to answer the user's question accurately, deeply, and clearly. "
            "Maintain an authentic, professional tone with crisp formatting, bold terms, and structured bullet points."
            f"{memory_section}\n"
            f"[RETRIEVED CONTEXT]\n{context_str}\n\n"
            f"[CURRENT QUESTION]\n{prompt_text}"
        )

        # --- Generate ---
        t_gen_start = time.time()
        answer, model_used = generate_gemini_content(rag_prompt, api_key)
        t_gen_end = time.time()
        generation_ms = round((t_gen_end - t_gen_start) * 1000)

        telemetry.update({
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "model_used": model_used,
            "chunks_retrieved": len(nodes),
            "chunks_used": len(reranked_nodes),
            "token_estimate": round(len(answer.split()) * 1.3),
        })

        return answer, sources, telemetry, context_str

    else:
        # General mode (no document)
        memory_section = f"\n[CONVERSATION HISTORY]\n{history_str}\n" if history_str else ""
        general_prompt = (
            "You are NoteWise, an intelligent AI copilot. "
            "The user has not uploaded any documents yet. Answer politely and concisely, "
            "and encourage them to upload a PDF or resume in the sidebar to unlock vector search."
            f"{memory_section}\n"
            f"User Query: {prompt_text}"
        )
        t_gen_start = time.time()
        answer, model_used = generate_gemini_content(general_prompt, api_key)
        generation_ms = round((time.time() - t_gen_start) * 1000)
        telemetry = {
            "retrieval_ms": 0,
            "generation_ms": generation_ms,
            "model_used": model_used,
            "chunks_retrieved": 0,
            "chunks_used": 0,
            "token_estimate": round(len(answer.split()) * 1.3),
        }
        return answer, [], telemetry, ""

# ---------------------------------------------------------------------------
# Sidebar UI (Claude Style Sidebar)
# ---------------------------------------------------------------------------
with st.sidebar:
    if LOGO_B64:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 18px;">
            <img src="data:image/png;base64,{LOGO_B64}" width="34" style="vertical-align: middle;" />
            <span style="font-family: 'Newsreader', serif; font-size: 1.45rem; color: #f4f4f5; font-weight: 500;">NoteWise</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 18px;">
            <span style="font-size: 1.4rem;">🧠</span>
            <span style="font-family: 'Newsreader', serif; font-size: 1.4rem; color: #f4f4f5; font-weight: 500;">NoteWise</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.telemetry = []
        st.rerun()

    st.markdown("---")
    st.markdown("##### 📁 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF / Resume",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload your document to activate semantic search, citations, and interview preparation."
    )

    if st.button("Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            index, msg, pages = process_uploaded_files(uploaded_files)
            if index:
                st.session_state.index = index
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.session_state.doc_stats = {
                    "files": len(uploaded_files),
                    "pages": pages
                }
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    if st.session_state.index is not None:
        st.markdown(f"""
        <div class="sidebar-card">
            <span style="color: #22c55e;">●</span> <b>Knowledge Base Active</b><br>
            <b>File:</b> {', '.join(st.session_state.indexed_files)}<br>
            <b>Pages:</b> {st.session_state.doc_stats['pages']}<br>
            <b>Retrieval:</b> BAAI/bge-small (~40ms)<br>
            <b>Re-ranker:</b> ms-marco-MiniLM-L-6<br>
            <div style="margin-top: 8px;">
                <span class="model-tag">Gemini 3 Cascade</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-card">
            <span style="color: #a1a1aa;">○</span> <b>General Mode</b><br>
            Upload a document to enable citations and interview tools.
        </div>
        """, unsafe_allow_html=True)

    # --- Session Telemetry Panel ---
    if st.session_state.telemetry:
        st.markdown("---")
        with st.expander("📡 Session Telemetry", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.telemetry[-5:]), 1):
                st.markdown(f"""
                <div style="font-size: 0.75rem; color: #71717a; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #272522;">
                    <span style="color: #a1a1aa; font-weight: 600;">Q{len(st.session_state.telemetry) - i + 1}</span>
                    &nbsp;·&nbsp;<span style="color: #d97757;">{entry.get('model_used','?')}</span><br>
                    ⚡ Retrieval: <b>{entry.get('retrieval_ms',0)}ms</b>
                    &nbsp;|&nbsp; 🧠 Gen: <b>{entry.get('generation_ms',0)}ms</b><br>
                    📦 Chunks: <b>{entry.get('chunks_used',0)}/{entry.get('chunks_retrieved',0)}</b>
                    &nbsp;|&nbsp; ~<b>{entry.get('token_estimate',0)}</b> tokens
                </div>
                """, unsafe_allow_html=True)

    # --- Export Chat ---
    if st.session_state.messages:
        st.markdown("---")
        md_content = generate_markdown_export(st.session_state.messages)
        st.download_button(
            "⬇ Export Chat",
            data=md_content,
            file_name=f"notewise_session_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    remaining_queries = MAX_QUERIES_PER_SESSION - st.session_state.query_count
    st.caption(f"Session Quota: {remaining_queries}/{MAX_QUERIES_PER_SESSION} queries")

    st.markdown("""
    <div style="margin-top: 30px; font-size: 0.82rem; color: #71717a; border-top: 1px solid #272522; padding-top: 14px;">
        👤 User · Free Plan
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Content Area (Claude Hero & Interaction)
# ---------------------------------------------------------------------------
active_prompt = None
bot_avatar = LOGO_ICON_PATH if os.path.exists(LOGO_ICON_PATH) else "✳"

# If no messages yet, show centered hero greeting
if not st.session_state.messages:
    hero_logo = f'<img src="data:image/png;base64,{LOGO_B64}" width="46" style="vertical-align: middle; margin-right: 8px;" />' if LOGO_B64 else '<span class="claude-flower">✳</span>'
    st.markdown(f"""
    <div class="claude-hero-container">
        <h1 class="claude-hero-title">
            {hero_logo} Back at it, User
        </h1>
        <p class="claude-hero-subtitle">
            What document or interview challenge would you like to explore today?
        </p>
    </div>
    """, unsafe_allow_html=True)

# Render Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(msg["content"])

            # Telemetry + model badge row
            if msg.get("model_used") or msg.get("retrieval_ms") is not None:
                parts = []
                if msg.get("model_used") and msg["model_used"] != "none":
                    parts.append(f'<span class="model-tag">✦ {msg["model_used"]}</span>')
                if msg.get("retrieval_ms") is not None and msg["retrieval_ms"] > 0:
                    parts.append(f'<span style="font-size:0.72rem;color:#52525b;">⚡ {msg["retrieval_ms"]}ms retrieval · {msg.get("chunks_used",0)}/{msg.get("chunks_retrieved",0)} chunks · ~{msg.get("token_estimate",0)} tokens</span>')
                if parts:
                    st.markdown(f'<div class="telemetry-row" style="margin-top:8px;">{"&nbsp;&nbsp;".join(parts)}</div>', unsafe_allow_html=True)

            # RAGAS Eval Badges
            if msg.get("eval_scores"):
                ev = msg["eval_scores"]
                badges_html = '<div class="eval-badge-row">'
                for label, key in [("🎯 Faithfulness", "faithfulness"), ("💡 Relevance", "relevance"), ("📋 Completeness", "completeness")]:
                    score = ev.get(key)
                    if score is not None:
                        cls = score_badge_class(score)
                        badges_html += f'<span class="eval-badge {cls}">{label} {score:.2f}</span>'
                badges_html += "</div>"
                st.markdown(badges_html, unsafe_allow_html=True)

            # Source Citations
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Retrieved Source Citations"):
                    for idx, src in enumerate(msg["sources"], 1):
                        score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                        st.markdown(f"**Source #{idx}** &nbsp;·&nbsp; `Relevance: {score_str}`")
                        st.markdown(f"""<div class="claude-artifact-box">{src['text']}</div>""", unsafe_allow_html=True)

# --- ALWAYS-VISIBLE SUGGESTIVE ACTION PILLS ---
if not st.session_state.messages:
    st.markdown("<div style='margin-top: 24px; margin-bottom: 6px;'></div>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("📄 Executive Summary", use_container_width=True):
            active_prompt = "Provide a high-level executive summary of this document, highlighting its core message, background, and key takeaways."
    with col_p2:
        if st.button("🛠️ Skills & Tech Stack", use_container_width=True):
            active_prompt = "Extract and categorize all key technical skills, tools, frameworks, and achievements mentioned in this document."
    with col_p3:
        if st.button("🎙️ 60s Spoken Pitch", use_container_width=True):
            active_prompt = "Write a crisp, powerful 60 to 90-second spoken 'Tell Me About Yourself' introduction for this candidate based on their achievements."
    with col_p4:
        if st.button("🎯 Technical Questions", use_container_width=True):
            active_prompt = "Generate 3 challenging technical interview questions based directly on the achievements and projects in this document."

# Chat Input Box
chat_val = st.chat_input("Ask NoteWise anything...")
if chat_val:
    active_prompt = chat_val

# Process Query
if active_prompt:
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_query_time
    if time_since_last < RATE_LIMIT_COOLDOWN:
        st.toast("⏳ Please wait 1s before submitting another question.", icon="⚠️")
        st.stop()

    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        st.error("🚫 You have reached the maximum session query budget (35 queries). Please click 'New Chat' in the sidebar.")
        st.stop()

    st.session_state.last_query_time = current_time
    st.session_state.query_count += 1

    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant", avatar=bot_avatar):
        with st.spinner(""):
            answer, sources, telemetry, context_str = execute_rag(active_prompt)

        st.write_stream(stream_typewriter(answer))

        # Telemetry + model badge
        model_used = telemetry.get("model_used", "")
        retrieval_ms = telemetry.get("retrieval_ms", 0)
        chunks_used = telemetry.get("chunks_used", 0)
        chunks_retrieved = telemetry.get("chunks_retrieved", 0)
        token_estimate = telemetry.get("token_estimate", 0)

        parts = []
        if model_used and model_used != "none":
            parts.append(f'<span class="model-tag">✦ {model_used}</span>')
        if retrieval_ms > 0:
            parts.append(f'<span style="font-size:0.72rem;color:#52525b;">⚡ {retrieval_ms}ms retrieval · {chunks_used}/{chunks_retrieved} chunks · ~{token_estimate} tokens</span>')
        if parts:
            st.markdown(f'<div class="telemetry-row" style="margin-top:8px;">{"&nbsp;&nbsp;".join(parts)}</div>', unsafe_allow_html=True)

        # RAGAS Eval (only if we have context = document mode)
        eval_scores = {}
        if context_str and sources:
            with st.spinner("Evaluating response quality..."):
                eval_scores = evaluate_rag_response(context_str, active_prompt, answer, api_key)
            if eval_scores:
                badges_html = '<div class="eval-badge-row">'
                for label, key in [("🎯 Faithfulness", "faithfulness"), ("💡 Relevance", "relevance"), ("📋 Completeness", "completeness")]:
                    score = eval_scores.get(key)
                    if score is not None:
                        cls = score_badge_class(score)
                        badges_html += f'<span class="eval-badge {cls}">{label} {score:.2f}</span>'
                badges_html += "</div>"
                st.markdown(badges_html, unsafe_allow_html=True)

        # Source Citations
        if sources:
            with st.expander("📚 Retrieved Source Citations"):
                for idx, src in enumerate(sources, 1):
                    score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                    st.markdown(f"**Source #{idx}** &nbsp;·&nbsp; `Relevance: {score_str}`")
                    st.markdown(f"""<div class="claude-artifact-box">{src['text']}</div>""", unsafe_allow_html=True)

        # Log telemetry
        st.session_state.telemetry.append(telemetry)

        # Store message with all metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "model_used": model_used,
            "retrieval_ms": retrieval_ms,
            "chunks_used": chunks_used,
            "chunks_retrieved": chunks_retrieved,
            "token_estimate": token_estimate,
            "eval_scores": eval_scores,
        })
