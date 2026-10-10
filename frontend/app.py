import streamlit as st
import requests

# Constants
API_URL = "http://localhost:8000"

def get_status() -> dict | None:
    """Fetch backend status to check if the index is ready."""
    try:
        response = requests.get(f"{API_URL}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def trigger_ingestion() -> None:
    """Trigger the document ingestion process on the backend."""
    try:
        with st.spinner("Ingesting documents (this may take a few minutes)..."):
            response = requests.post(f"{API_URL}/ingest")
            response.raise_for_status()
            st.success("Documents ingested successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to ingest documents: {e}")

def render_sidebar():
    """Render the sidebar with app info and admin controls."""
    with st.sidebar:
        st.title("NoteWise-AI 🧠")
        st.markdown("Your advanced RAG and Agentic Chatbot.")
        st.divider()

        # Display System Status
        st.subheader("System Status")
        status = get_status()
        if status:
            is_ready = status.get("index_ready", False)
            doc_count = status.get("collection_count", 0)
            
            if is_ready:
                st.success(f"✅ Index Ready")
                st.info(f"📄 Documents indexed: {doc_count}")
            else:
                st.warning("⚠️ Index Not Ready (Empty)")
        else:
            st.error("❌ Backend Offline")
            st.caption("Please ensure FastAPI is running on port 8000.")

        st.divider()
        
        # Admin Controls
        st.subheader("Admin")
        if st.button("📥 Ingest Documents", use_container_width=True):
            if status is None:
                st.error("Cannot ingest: Backend is offline.")
            else:
                trigger_ingestion()

def render_chat():
    """Render the main chat interface."""
    st.title("Chat with NoteWise")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show tools used if applicable
            if msg.get("tools_used"):
                st.caption(f"🛠️ **Tools used:** {', '.join(msg['tools_used'])}")
                
            # Show sources if applicable
            if msg.get("sources"):
                with st.expander("📄 View Sources"):
                    for idx, source in enumerate(msg["sources"]):
                        score = source.get('score')
                        score_text = f"{score:.4f}" if score else "N/A"
                        st.markdown(f"**Source {idx + 1}** (Relevance Score: `{score_text}`)")
                        st.write(f"_{source.get('text', '').strip()}_")
                        if idx < len(msg["sources"]) - 1:
                            st.divider()

    # Chat input box
    if prompt := st.chat_input("Ask a question about your documents, or try some math..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(f"{API_URL}/chat", json={"question": prompt})
                    response.raise_for_status()
                    data = response.json()

                    answer = data.get("answer", "No answer provided.")
                    tools = data.get("tools_used", [])
                    sources = data.get("source_nodes", [])

                    # Display the answer
                    st.markdown(answer)
                    
                    # Display metadata
                    if tools:
                        st.caption(f"🛠️ **Tools used:** {', '.join(tools)}")
                    
                    if sources:
                        with st.expander("📄 View Sources"):
                            for idx, source in enumerate(sources):
                                score = source.get('score')
                                score_text = f"{score:.4f}" if score else "N/A"
                                st.markdown(f"**Source {idx + 1}** (Relevance Score: `{score_text}`)")
                                st.write(f"_{source.get('text', '').strip()}_")
                                if idx < len(sources) - 1:
                                    st.divider()

                    # Save response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tools_used": tools,
                        "sources": sources
                    })

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the backend. Is FastAPI running? Error: {e}")

def main():
    st.set_page_config(
        page_title="NoteWise-AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()

