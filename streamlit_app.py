import os
import time
import shutil
import tempfile
import streamlit as st
import pypdf
from google import genai
from google.genai import types

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
    page_title="NoteWise-AI | Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Executive Professional Styling (Modern Dark Aesthetic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    .hero-card {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.08) 0%, rgba(30, 30, 30, 0.4) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #B0B0B0;
        margin: 0;
    }
    .hero-badges {
        margin-top: 12px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #E0E0E0;
    }
    .badge-accent {
        background: rgba(255, 75, 75, 0.15);
        border-color: rgba(255, 75, 75, 0.4);
        color: #FF6B6B;
    }
    .badge-turbo {
        background: rgba(0, 200, 83, 0.15);
        border-color: rgba(0, 200, 83, 0.4);
        color: #00E676;
    }
    .badge-deep {
        background: rgba(156, 39, 176, 0.15);
        border-color: rgba(156, 39, 176, 0.4);
        color: #CE93D8;
    }

    .stat-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
    
    .source-box {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #FF4B4B;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Rate Limiting & Anti-Abuse
# ---------------------------------------------------------------------------
RATE_LIMIT_COOLDOWN = 1.0         # Seconds between submissions
MAX_QUERIES_PER_SESSION = 35      # Session cap to prevent bot abuse

def get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Embedding Model Cache
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
# Streaming Generator (Interactive Thinking Toggle)
# ---------------------------------------------------------------------------
def stream_gemini_response(prompt: str, api_key: str, enable_thinking: bool = False):
    """
    Streams response chunks with customizable reasoning budget:
    - Thinking OFF: thinking_budget=0 (Sub-second instant Turbo response)
    - Thinking ON:  thinking_budget=2048 (In-depth deliberate reasoning)
    """
    client = genai.Client(api_key=api_key)
    
    # Configure reasoning budget based on user toggle
    budget = 2048 if enable_thinking else 0
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=budget),
        temperature=0.7 if enable_thinking else 0.2,
    )
    
    try:
        stream = client.models.generate_content_stream(
            model="gemini-3.8-flash",
            contents=prompt,
            config=config,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "resource_exhausted" in err_msg:
            yield "\n\n⚠️ **API Rate limit reached.** Please wait 10 seconds before asking another question."
        else:
            try:
                stream = client.models.generate_content_stream(
                    model="gemini-3.8-flash",
                    contents=prompt,
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as e2:
                yield f"\n\n❌ Error: {str(e2)}"

# ---------------------------------------------------------------------------
# Clean Document Ingestion Logic
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

# ---------------------------------------------------------------------------
# Sidebar UI (Control Center)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Upload PDF or text files to index into the private knowledge base."
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

    # --- INFERENCE MODE TOGGLE ---
    st.markdown("### ⚙️ Inference Controls")
    enable_thinking = st.toggle(
        "🧠 Deep Thinking Mode",
        value=False,
        help="Turn OFF for instant sub-second answers (Turbo Mode). Turn ON for in-depth deliberate reasoning on complex inquiries."
    )
    if enable_thinking:
        st.caption("🟣 *Deep Reasoning active (higher analysis depth).*")
    else:
        st.caption("🟢 *Turbo Mode active (sub-second instant streaming).*")

    st.divider()

    st.markdown("### 📊 Engine Status")
    if st.session_state.index is not None:
        st.success("● RAG Knowledge Base Active")
        mode_label = "Deep Reasoning" if enable_thinking else "Turbo Stream"
        st.markdown(f"""
        <div class="stat-box">
            <b>Active Documents:</b> {st.session_state.doc_stats['files']}<br>
            <b>Total Pages Read:</b> {st.session_state.doc_stats['pages']}<br>
            <b>Embedding Model:</b> BAAI/bge-small-en-v1.5<br>
            <b>Inference Mode:</b> {mode_label}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("○ General Mode (No documents loaded)")
        st.caption("Upload documents above to enable semantic retrieval & source citations.")

    st.divider()

    remaining_queries = MAX_QUERIES_PER_SESSION - st.session_state.query_count
    st.caption(f"Session Quota: {remaining_queries}/{MAX_QUERIES_PER_SESSION} queries remaining")

    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
active_badge_class = "badge-deep" if enable_thinking else "badge-turbo"
active_badge_text = "● Deep Reasoning ON" if enable_thinking else "⚡ Turbo Mode (Instant)"

st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">🧠 NoteWise-AI</div>
    <p class="hero-subtitle">Production-grade Retrieval-Augmented Generation & Document Intelligence</p>
    <div class="hero-badges">
        <span class="badge {active_badge_class}">{active_badge_text}</span>
        <span class="badge">Gemini 3.8 Flash</span>
        <span class="badge">LlamaIndex Vector Core</span>
        <span class="badge">Dense Embeddings</span>
        <span class="badge">Anti-Abuse Guard</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1-Click Starter Pills
# ---------------------------------------------------------------------------
preset_prompt = None
if st.session_state.index is not None:
    st.markdown("##### ⚡ Quick Actions")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("📄 Executive Summary", use_container_width=True):
            preset_prompt = "Provide a high-level executive summary of this document, highlighting its core message and key takeaways."
    with c2:
        if st.button("🛠️ Extract Skills & Tech", use_container_width=True):
            preset_prompt = "Extract and categorize all key technical skills, tools, frameworks, and achievements mentioned in this document."
    with c3:
        if st.button("🎯 Interview Questions", use_container_width=True):
            preset_prompt = "Generate 3 challenging technical interview questions based directly on the achievements and projects in this document."

# Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Retrieved Source Citations"):
                for idx, src in enumerate(msg["sources"], 1):
                    score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                    st.markdown(f"**Source #{idx}** &nbsp;·&nbsp; `Relevance: {score_str}`")
                    st.markdown(f"""<div class="source-box">{src['text']}</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Query Processing (Handles Text Input or Quick Prompts)
# ---------------------------------------------------------------------------
chat_input_val = st.chat_input("Ask a question about the document...")
active_prompt = preset_prompt or chat_input_val

if active_prompt:
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_query_time
    if time_since_last < RATE_LIMIT_COOLDOWN:
        st.toast("⏳ Please wait 1s before sending another question.", icon="⚠️")
        st.stop()

    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        st.error("🚫 You have reached the maximum session query budget (35 queries). Please click 'Reset Chat' in the sidebar.")
        st.stop()

    st.session_state.last_query_time = current_time
    st.session_state.query_count += 1

    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    api_key = get_api_key()
    if not api_key:
        with st.chat_message("assistant"):
            st.error("❌ **GOOGLE_API_KEY is missing!** Configure it in Streamlit Cloud Secrets.")
        st.stop()

    with st.chat_message("assistant"):
        sources = []
        
        if st.session_state.index is not None:
            # Laser-focused retrieval
            retriever = st.session_state.index.as_retriever(similarity_top_k=2)
            nodes = retriever.retrieve(active_prompt)

            context_blocks = []
            for i, node in enumerate(nodes, 1):
                clean_text = node.node.text.strip().replace("\n", " ")
                context_blocks.append(f"[Source {i}]: {clean_text}")
                sources.append({
                    "score": node.score,
                    "text": clean_text[:280] + ("..." if len(clean_text) > 280 else "")
                })

            context_str = "\n\n".join(context_blocks)
            rag_prompt = (
                "You are NoteWise-AI, an executive document intelligence assistant. "
                "Use the following retrieved context to answer the user's question accurately, professionally, and concisely. "
                "Be direct. Use clean markdown formatting. If the answer cannot be found in the context, state that clearly.\n\n"
                f"--- CONTEXT ---\n{context_str}\n\n"
                f"--- USER QUESTION ---\n{active_prompt}"
            )

            # Stream with user's chosen reasoning mode
            answer = st.write_stream(stream_gemini_response(rag_prompt, api_key, enable_thinking=enable_thinking))

        else:
            general_prompt = (
                "You are NoteWise-AI, an intelligent document intelligence assistant. "
                "The user has not uploaded any documents yet. Answer their query politely, direct and concisely, "
                "and encourage them to upload a PDF or text file in the sidebar to activate knowledge search.\n\n"
                f"User Query: {active_prompt}"
            )
            answer = st.write_stream(stream_gemini_response(general_prompt, api_key, enable_thinking=enable_thinking))

        # Collapsible Sources
        if sources:
            with st.expander("📚 Retrieved Source Citations"):
                for idx, src in enumerate(sources, 1):
                    score_str = f"{src['score']:.3f}" if src.get("score") is not None else "N/A"
                    st.markdown(f"**Source #{idx}** &nbsp;·&nbsp; `Relevance: {score_str}`")
                    st.markdown(f"""<div class="source-box">{src['text']}</div>""", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
