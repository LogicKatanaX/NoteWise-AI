<div align="center">
  <h1>🧠 NoteWise-AI</h1>
  <p><strong>Advanced RAG & Agentic Chatbot for Private Knowledge Bases</strong></p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/LlamaIndex-0.10+-blueviolet.svg" alt="LlamaIndex">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

## 📖 Overview
NoteWise-AI is an intelligent document assistant that leverages advanced Retrieval-Augmented Generation (RAG) to accurately answer questions based on your private knowledge base. Outfitted with a lightweight routing agent, NoteWise dynamically decides when to search your documents, when to evaluate math expressions safely, and when to converse directly.

---

## ✨ Key Features
- 🔍 **Advanced Hybrid Search:** Fuses traditional Keyword Search (BM25) with semantic Vector Search (ChromaDB) via Reciprocal Rank Fusion for unparalleled retrieval accuracy.
- 🎯 **CrossEncoder Re-ranking:** Uses HuggingFace's CrossEncoders to scrutinize and re-rank candidate documents before generating answers, severely reducing hallucinations.
- 🤖 **Agentic Routing:** A lightweight custom zero-shot router that evaluates user intent to intelligently trigger internal tools (RAG Engine, Calculator, Summarizer).
- ⚡ **Pluggable LLM Backends:** Switch effortlessly between Groq (for ultra-fast free inference) and OpenAI via simple environment variables.
- 🎨 **Modern UI & API:** Built entirely on a decoupled architecture using FastAPI for the robust backend and Streamlit for a sleek, responsive interactive chat interface.

## 🛠️ Tech Stack
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **RAG Orchestration**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (Local HuggingFace model)
- **Re-ranker**: `BAAI/bge-reranker-base` (Local CrossEncoder model)
- **LLMs**: Groq (Llama 3) / OpenAI (GPT-4o)

## 🏗️ Architecture
1. **Ingestion**: Documents are chunked and embedded via local HuggingFace models, then indexed into ChromaDB and an in-memory persistent document store.
2. **Retrieval**: User queries pass through a Hybrid Retriever (Vector + BM25), returning the top 10 candidates.
3. **Re-ranking**: Candidates are aggressively scored by a CrossEncoder to extract the top 3 most contextually relevant nodes.
4. **Generation**: The selected LLM provider synthesizes the final response, securely tracking citations and metadata for transparency.

---

## 🚀 Getting Started (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/NoteWise-AI.git
cd NoteWise-AI
```

### 2. Set up the Environment
We highly recommend using a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Copy the example environment file and add your API keys (Groq is set as the default for ultra-fast generation).
```bash
cp .env.example .env
```
**Inside your `.env` file:**
```env
LLM_PROVIDER=groq
LLM_MODEL=llama3-8b-8192
GROQ_API_KEY=gsk_your_key_here
```

### 4. Run the Application
Start the **FastAPI backend** (runs on port `8000`):
```bash
uvicorn app.main:app --reload
```

In a new terminal window (with the virtual environment activated), start the **Streamlit frontend**:
```bash
streamlit run frontend/app.py
```

---

## 🐳 Deployment (Docker)

NoteWise-AI includes a Dockerfile designed for easy deployment to platforms like Hugging Face Spaces, Render, or Railway. It runs both the backend and frontend concurrently.

```bash
# Build the Docker image
docker build -t notewise-ai .

# Run the container (exposing both Backend and Frontend ports)
docker run -d -p 8000:8000 -p 8501:8501 --env-file .env notewise-ai
```
- **API Docs:** `http://localhost:8000/docs`
- **Chat UI:** `http://localhost:8501`

---

## 📸 Screenshots
*(Placeholder for application screenshots: Include a picture of the sidebar status, document ingestion, and a rich chat message showing source citations and tools used)*

## 🔮 Future Improvements
- [ ] Native multi-user authentication.
- [ ] Deep integration with specialized agent frameworks (e.g., LangGraph).
- [ ] External connectors for Confluence, Notion, and Google Drive.
- [ ] GraphRAG integration for complex entity relationships.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
