import os

# Prevent PyTorch & Tokenizer deadlocks on Streamlit Community Cloud
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
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

# ---------------------------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NoteWise-AI | Executive Document & Interview Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium ChatGPT / Claude / Gemini Aesthetic Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1160px;
    }

    /* Claude & Gemini Inspired Radiant Header Card */
    .app-header {
        background: radial-gradient(circle at 0% 0%, rgba(99, 102, 241, 0.12) 0%, rgba(15, 23, 42, 0) 50%),
                    radial-gradient(circle at 100% 100%, rgba(236, 72, 153, 0.08) 0%, rgba(15, 23, 42, 0) 50%),
                    linear-gradient(180deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .app-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 20%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .app-subtitle {
        font-size: 0.96rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    .badges-strip {
        margin-top: 14px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #E2E8F0;
    }

    .badge-pill-accent {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        color: #818CF8;
    }

    .badge-pill-gemini {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.35);
        color: #38BDF8;
    }

    /* Clean Card Container */
    .pro-card {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .pro-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
    }

    /* Citation Box */
    .source-snippet {
        background: rgba(15, 23, 42, 0.7);
        border-left: 3px solid #818CF8;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-top: 6px;
        font-size: 0.84rem;
        color: #CBD5E1;
        line-height: 1.5;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Sidebar Stat Block */
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 10px;
        font-size: 0.84rem;
        color: #CBD5E1;
        line-height: 1.6;
    }

    /* Rating Pills */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.9rem;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #4ADE80;
        margin-bottom: 12px;
    }

    /* Streamlit UI Tweaks */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        background: rgba(15, 23, 42, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Rate Limiting & Anti-Abuse Guard
# ---------------------------------------------------------------------------
RATE_LIMIT_COOLDOWN = 1.0         # Seconds between submissions
MAX_QUERIES_PER_SESSION = 35      # Session cap to prevent bot abuse

def get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Local Embeddings Ingestion
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Cache the local BGE embedding model across sessions."""
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.transformations = [TokenTextSplitter(chunk_size=512, chunk_overlap=50)]
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu"
    )

Settings.embed_model = load_embedding_model()

# ---------------------------------------------------------------------------
# Multi-Tier Gemini 3 Cascade Engine
# ---------------------------------------------------------------------------
def generate_gemini_content(prompt: str, api_key: str) -> str:
    """Generates text prioritizing Gemini 3.8 Flash with automatic cascade to 3.7 & 3.5 on 429/503."""
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
                return response.text.strip()
        except Exception as e:
            err_msg = str(e)
            last_error = e
            print(f"[NoteWise ⚠️] {model_name} unavailable: {err_msg[:80]}... Rolling over to next model...", flush=True)
            continue

    if last_error:
        err_str = str(last_error)
        if "429" in err_str or "resource_exhausted" in err_str:
            return "⚠️ **Daily API quota reached on free tier.** Please try again tomorrow or configure your own key in settings."
        raise last_error

    return "⚠️ Could not generate an answer from the document. Please try asking in a different way."

def stream_typewriter(text: str, delay: float = 0.008):
    """Smooth word-by-word typewriter animation."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)

# ---------------------------------------------------------------------------
# Document Ingestion Logic
# ---------------------------------------------------------------------------
def process_uploaded_files(uploaded_files):
    """Parses clean text from PDFs and text files and builds the vector index."""
    try:
        documents = []
        total_pages = 0
        
        for file in uploaded_files:
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

        if not documents:
            return None, "No readable text detected in uploaded files (is the PDF a scanned image?).", 0

        splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
        
        return index, f"Indexed {len(documents)} document(s) successfully!", total_pages
    except Exception as e:
        return None, f"Document parsing error: {str(e)}", 0

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

if "generated_eval" not in st.session_state:
    st.session_state.generated_eval = None

# ---------------------------------------------------------------------------
# Sidebar UI (Control Center)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Resume / PDF Document",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload your resume or any technical PDF to activate AI analysis and interview coaching."
    )

    if st.button("📥 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Parsing & embedding documents..."):
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

    st.divider()

    st.markdown("### 📊 Engine Status")
    if st.session_state.index is not None:
        st.success("● RAG Knowledge Base Active")
        st.markdown(f"""
        <div class="sidebar-metric">
            <b>Active Document:</b> {', '.join(st.session_state.indexed_files)}<br>
            <b>Pages Read:</b> {st.session_state.doc_stats['pages']}<br>
            <b>Retrieval Engine:</b> BAAI/bge-small (~40ms)<br>
            <b>Inference Cascade:</b> Gemini 3.8 ➔ 3.7 ➔ 3.5
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("○ General Mode (No documents loaded)")
        st.caption("Upload a resume or PDF to unlock interview prep, architecture breakdowns, and verifiable citations.")

    st.divider()

    remaining_queries = MAX_QUERIES_PER_SESSION - st.session_state.query_count
    st.caption(f"🛡️ Session Budget: {remaining_queries}/{MAX_QUERIES_PER_SESSION} queries remaining")

    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.generated_eval = None
        st.rerun()

# ---------------------------------------------------------------------------
# Main Content Header (Claude / Gemini Aesthetic)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">
        <span>🧠 NoteWise-AI</span>
    </div>
    <p class="app-subtitle">
        Production-grade Retrieval-Augmented Generation & Technical Interview Readiness Copilot
    </p>
    <div class="badges-strip">
        <span class="badge-pill badge-pill-accent">⚡ Sub-50ms Dense Retrieval</span>
        <span class="badge-pill badge-pill-gemini">● Gemini 3 High-Availability Cascade</span>
        <span class="badge-pill">LlamaIndex Vector Core</span>
        <span class="badge-pill">Verifiable Citations</span>
        <span class="badge-pill">Anti-Abuse Guard</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Tabs: Copilot Chat, Prep Kit, and Mock Interview
# ---------------------------------------------------------------------------
tab_chat, tab_prep, tab_mock = st.tabs([
    "💬 Document Copilot & Chat",
    "🎯 Interview Readiness Kit",
    "🧠 Mock Interview & Self-Evaluation"
])

api_key = get_api_key()

# Helper for executing RAG retrieval
def execute_rag(prompt_text: str):
    if not api_key:
        return "❌ **GOOGLE_API_KEY is missing!** Configure it in Streamlit Cloud Secrets.", []

    if st.session_state.index is not None:
        t_ret_start = time.time()
        retriever = st.session_state.index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(prompt_text)
        print(f"[NoteWise] Retrieved {len(nodes)} nodes in {time.time() - t_ret_start:.2f}s", flush=True)

        context_blocks = []
        sources = []
        for i, node in enumerate(nodes, 1):
            clean_text = node.node.text.strip().replace("\n", " ")
            context_blocks.append(f"[Source {i}]: {clean_text}")
            sources.append({
                "score": node.score,
                "text": clean_text[:280] + ("..." if len(clean_text) > 280 else "")
            })

        context_str = "\n\n".join(context_blocks)
        rag_prompt = (
            "You are NoteWise-AI, an executive technical interview coach and document intelligence assistant. "
            "Use the following retrieved context to answer the user's question accurately, professionally, and concisely. "
            "Provide crisp, structured formatting with bold headings and clean bullet points. "
            "If the answer is not present in the context, state that clearly.\n\n"
            f"--- CONTEXT ---\n{context_str}\n\n"
            f"--- USER QUESTION ---\n{prompt_text}"
        )
        answer = generate_gemini_content(rag_prompt, api_key)
        return answer, sources
    else:
        general_prompt = (
            "You are NoteWise-AI, an intelligent document intelligence assistant. "
            "The user has not uploaded any documents yet. Answer their query politely, direct and concisely, "
            "and encourage them to upload a PDF or text file in the sidebar to activate knowledge search.\n\n"
            f"User Query: {prompt_text}"
        )
        answer = generate_gemini_content(general_prompt, api_key)
        return answer, []

# ===========================================================================
# TAB 1: Document Copilot & Chat
# ===========================================================================
with tab_chat:
    preset_prompt = None

    if st.session_state.index is not None:
        st.markdown("##### ⚡ Quick Recruiter Actions")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📄 Executive Summary", use_container_width=True):
                preset_prompt = "Provide a high-level executive summary of this document, highlighting its core message, background, and key takeaways."
        with c2:
            if st.button("🛠️ Categorize Skills & Stack", use_container_width=True):
                preset_prompt = "Extract and categorize all key technical skills, tools, frameworks, programming languages, and achievements mentioned in this document."
        with c3:
            if st.button("🎯 3 Core Technical Questions", use_container_width=True):
                preset_prompt = "Generate 3 challenging technical interview questions based directly on the achievements and projects in this document."

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Retrieved Source Citations"):
                    for idx, src in enumerate(msg["sources"], 1):
                        score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                        st.markdown(f"**Source #{idx}** &nbsp;·&nbsp; `Relevance: {score_str}`")
                        st.markdown(f"""<div class="source-snippet">{src['text']}</div>""", unsafe_allow_html=True)

    # Chat Input Box
    chat_input_val = st.chat_input("Ask any question about your document or resume...")
    active_prompt = preset_prompt or chat_input_val

    if active_prompt:
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_query_time
        if time_since_last < RATE_LIMIT_COOLDOWN:
            st.toast("⏳ Please wait 1s before submitting another question.", icon="⚠️")
            st.stop()

        if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
            st.error("🚫 You have reached the maximum session query budget (35 queries). Please click 'Reset Chat' in the sidebar.")
            st.stop()

        st.session_state.last_query_time = current_time
        st.session_state.query_count += 1

        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document with Gemini..."):
                answer, sources = execute_rag(active_prompt)

            st.write_stream(stream_typewriter(answer))

            if sources:
                with st.expander("📚 Retrieved Source Citations"):
                    for idx, src in enumerate(sources, 1):
                        score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                        st.markdown(f"**Source #{idx}** &nbsp;·&nbsp; `Relevance: {score_str}`")
                        st.markdown(f"""<div class="source-snippet">{src['text']}</div>""", unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

# ===========================================================================
# TAB 2: Interview Readiness Kit
# ===========================================================================
with tab_prep:
    if st.session_state.index is None:
        st.info("💡 Please upload and index a resume or document in the sidebar to activate the Interview Readiness Kit.")
    else:
        st.markdown("### 🎯 Career & Technical Interview Preparation Kit")
        st.caption("Arm yourself with tailored spoken introductions, technical stack defense strategies, and project architecture breakdowns.")

        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            if st.button("🎙️ 60s Elevator Pitch ('Tell Me About Yourself')", use_container_width=True, type="primary"):
                pitch_prompt = (
                    "Write a crisp, powerful 60 to 90-second spoken 'Tell Me About Yourself' introduction for this candidate. "
                    "Structure it for a high-stakes technical interview: "
                    "1) Hook & Background (Degree, College, CGPA highlights); "
                    "2) Technical Superpowers (Languages & Core competencies); "
                    "3) Flagship Achievement Highlight (SIH finalist, top project, or major architecture); "
                    "4) Forward-Looking Conclusion (What they are excited to build next). "
                    "Keep it confident, natural, and conversational."
                )
                with st.spinner("Synthesizing your executive pitch..."):
                    pitch_answer, _ = execute_rag(pitch_prompt)
                    st.session_state["active_pitch"] = pitch_answer

        with col_p2:
            if st.button("🛠️ Tech Stack Defense & Trap Questions", use_container_width=True):
                stack_prompt = (
                    "Analyze the technologies and tools listed in this resume (e.g. C++, Python, Flask, React, SQLite, Linux, etc.). "
                    "For the top 3-4 most prominent technologies, provide: "
                    "1) The Exact Defense Question a Senior Engineer will ask (e.g., 'Why SQLite over Postgres?', 'How did you handle state?'); "
                    "2) Common Traps (What beginners say wrong); "
                    "3) The Gold Standard Answer (The architectural rationale the candidate should give based on their projects)."
                )
                with st.spinner("Extracting tech stack defense questions..."):
                    stack_answer, _ = execute_rag(stack_prompt)
                    st.session_state["active_stack"] = stack_answer

        with col_p3:
            if st.button("📂 Project Architecture & STAR Breakdown", use_container_width=True):
                project_prompt = (
                    "Select the key projects in this resume (such as the Waste-Segregation system, YouMe OS, or RPG learning platform). "
                    "Break each down into the STAR framework for an engineering interview: "
                    "- Situation & Problem: What real problem did it solve? "
                    "- Task & Architecture: The tech stack, concurrency, zero-backend, or hardware-software pipeline. "
                    "- Action & Implementation: Key technical choices made. "
                    "- Results & Challenges: Scalability limits, latency, or edge cases overcome."
                )
                with st.spinner("Breaking down project architectures..."):
                    project_answer, _ = execute_rag(project_prompt)
                    st.session_state["active_projects"] = project_answer

        # Display results in clean cards
        if "active_pitch" in st.session_state:
            with st.expander("🎙️ Your 60-Second Spoken Elevator Pitch", expanded=True):
                st.markdown(st.session_state["active_pitch"])

        if "active_stack" in st.session_state:
            with st.expander("🛠️ Tech Stack Defense & Engineering Trade-Offs", expanded=True):
                st.markdown(st.session_state["active_stack"])

        if "active_projects" in st.session_state:
            with st.expander("📂 Deep Project Breakdown (STAR Method)", expanded=True):
                st.markdown(st.session_state["active_projects"])

# ===========================================================================
# TAB 3: Mock Interview & Self-Evaluation Simulator
# ===========================================================================
with tab_mock:
    if st.session_state.index is None:
        st.info("💡 Please upload and index your resume in the sidebar to simulate an interactive mock interview.")
    else:
        st.markdown("### 🧠 Interactive Mock Interview & Self-Evaluation")
        st.caption("Practice answering challenging technical questions derived directly from your resume. Receive immediate scoring, blindspot detection, and dynamic senior follow-ups.")

        # Question Selection / Generation
        if "mock_question" not in st.session_state:
            st.session_state.mock_question = "How did you manage game state persistence, user progress tracking, and data security in your RPG-style learning platform without a traditional backend server?"

        if st.button("🔄 Generate New Interview Challenge", use_container_width=False):
            with st.spinner("Formulating technical challenge from your projects..."):
                q_prompt = (
                    "Generate ONE single challenging technical interview question based specifically on the candidate's projects or architecture claims in the document. "
                    "Make it deep and specific (e.g., about concurrency, sensor sync, state persistence, or database choices). "
                    "Output ONLY the question text."
                )
                new_q, _ = execute_rag(q_prompt)
                st.session_state.mock_question = new_q
                st.session_state.generated_eval = None

        st.markdown(f"""
        <div class="pro-card" style="border-left: 4px solid #6366F1;">
            <b>Interviewer's Question:</b><br>
            <span style="font-size: 1.05rem; color: #FFFFFF; font-weight: 600;">{st.session_state.mock_question}</span>
        </div>
        """, unsafe_allow_html=True)

        user_answer = st.text_area(
            "Your Answer (Type how you would respond in the interview):",
            placeholder="Explain your architectural choices, state management, edge cases, and why you designed it this way...",
            height=160
        )

        if st.button("🚀 Evaluate My Answer & Give Follow-up", type="primary", use_container_width=True):
            if not user_answer or len(user_answer.strip()) < 15:
                st.warning("Please write a detailed technical response so NoteWise can accurately evaluate your readiness.")
            else:
                eval_prompt = (
                    "You are a Staff Software Engineer and Senior Technical Hiring Manager conducting an interview. "
                    f"Question Asked: '{st.session_state.mock_question}'\n\n"
                    f"Candidate's Answer: '{user_answer}'\n\n"
                    "Evaluate the candidate's response rigorously. Format your assessment with: "
                    "1. **Readiness Score (out of 10)**: Give a realistic rating (e.g., 7.5/10 - Solid Concept, Light on Edge Cases). "
                    "2. **What Was Strong**: Specific good points mentioned. "
                    "3. **Critical Blindspots & Missing Points**: What the candidate neglected (e.g., latency, failure modes, data tampering, security, scale). "
                    "4. **The Model Answer**: How a Senior/Staff Engineer would phrase this answer in 3 concise bullet points. "
                    "5. **Dynamic Follow-Up Question**: Formulate the exact tough follow-up question the interviewer would ask next to test them further."
                )
                with st.spinner("Analyzing your response & formulating feedback..."):
                    eval_result, _ = execute_rag(eval_prompt)
                    st.session_state.generated_eval = eval_result

        # Display Evaluation
        if st.session_state.generated_eval:
            st.markdown("---")
            st.markdown("### 📋 Evaluation & Interview Feedback")
            st.markdown(st.session_state.generated_eval)
