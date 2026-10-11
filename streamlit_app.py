import os
import shutil
import tempfile
import streamlit as st
from google import genai

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NoteWise-AI | Advanced RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# API Key Management (Supports st.secrets and environment variables)
# ---------------------------------------------------------------------------
def get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.environ.get("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Cache Models for Speed & Instant Startup
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

def generate_with_gemini(prompt: str, api_key: str):
    """Generate response using Google's modern google-genai SDK (supports AQ. keys)."""
    client = genai.Client(api_key=api_key)
    
    for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text
        except Exception:
            continue

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    return response.text if response and response.text else "No response generated."

# ---------------------------------------------------------------------------
# Ingestion Logic
# ---------------------------------------------------------------------------
import pypdf
from llama_index.core import Document

def process_uploaded_files(uploaded_files):
    """Processes uploaded files and returns a VectorStoreIndex with clean extracted text."""
    try:
        documents = []
        for file in uploaded_files:
            if file.name.lower().endswith(".pdf"):
                reader = pypdf.PdfReader(file)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n\n".join(pages_text).strip()
                if full_text:
                    documents.append(Document(text=full_text, metadata={"file_name": file.name}))
            else:
                raw_text = file.getvalue().decode("utf-8", errors="ignore").strip()
                if raw_text:
                    documents.append(Document(text=raw_text, metadata={"file_name": file.name}))

        if not documents:
            return None, "No readable text found in uploaded files (is the PDF a scanned image?)."

        splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
        return index, f"Successfully indexed {len(documents)} document(s) from {len(uploaded_files)} file(s)!"
    except Exception as e:
        return None, f"Error reading documents: {str(e)}"

# ---------------------------------------------------------------------------
# App State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 NoteWise-AI")
    st.caption("Advanced Document RAG & Agentic Assistant")
    st.divider()

    st.subheader("📁 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Upload files to search and chat with your private documents."
    )

    if st.button("📥 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing & indexing documents..."):
                index, message = process_uploaded_files(uploaded_files)
                if index:
                    st.session_state.index = index
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(message)
                else:
                    st.error(message)

    st.divider()

    st.subheader("📊 System Status")
    if st.session_state.index is not None:
        st.success("✅ Knowledge Index Active")
        st.info(f"📄 Files: {', '.join(st.session_state.indexed_files)}")
    else:
        st.info("ℹ️ General Chat Mode (No docs uploaded)")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------
st.title("💬 Chat with NoteWise-AI")
st.markdown("Ask anything about your uploaded documents, or chat generally with Google Gemini!")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    score_str = f"`{src['score']:.3f}`" if src.get("score") is not None else "N/A"
                    st.markdown(f"**Source {idx}** (Similarity Score: {score_str})")
                    st.write(f"_{src['text']}_")
                    if idx < len(msg["sources"]):
                        st.divider()

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_key = get_api_key()
    if not api_key:
        with st.chat_message("assistant"):
            st.error("❌ **GOOGLE_API_KEY is missing!** Please add it in Streamlit Secrets (`.streamlit/secrets.toml`).")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                sources = []
                if st.session_state.index is not None:
                    # 1. Retrieve top 3 relevant chunks
                    retriever = st.session_state.index.as_retriever(similarity_top_k=3)
                    nodes = retriever.retrieve(prompt)

                    context_blocks = []
                    for i, node in enumerate(nodes, 1):
                        clean_text = node.node.text.strip().replace("\n", " ")
                        context_blocks.append(f"[Source {i}]: {clean_text}")
                        snippet = clean_text[:250] + ("..." if len(clean_text) > 250 else "")
                        sources.append({
                            "score": node.score,
                            "text": snippet
                        })

                    context_str = "\n\n".join(context_blocks)
                    rag_prompt = (
                        "You are NoteWise-AI, an intelligent document analysis assistant. "
                        "Use the following retrieved context to answer the user's question accurately and concisely. "
                        "If the answer cannot be found in the context, state that clearly.\n\n"
                        f"--- CONTEXT ---\n{context_str}\n\n"
                        f"--- QUESTION ---\n{prompt}"
                    )
                    answer = generate_with_gemini(rag_prompt, api_key)
                else:
                    general_prompt = (
                        "You are NoteWise-AI, an intelligent document analysis assistant. "
                        "The user has not uploaded any documents yet. Answer their question politely, "
                        "and remind them they can upload PDFs or text files via the sidebar to chat with their documents.\n\n"
                        f"User Question: {prompt}"
                    )
                    answer = generate_with_gemini(general_prompt, api_key)

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View Retrieved Sources"):
                        for idx, src in enumerate(sources, 1):
                            score_str = f"`{src['score']:.3f}`" if src.get("score") is not None else "N/A"
                            st.markdown(f"**Source {idx}** (Similarity Score: {score_str})")
                            st.write(f"_{src['text']}_")
                            if idx < len(sources):
                                st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

