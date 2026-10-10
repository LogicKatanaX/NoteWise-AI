from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.engine.rag_engine import NoteWiseRAG
from app.engine.agent import NoteWiseAgent
from app.models.schemas import QueryRequest, QueryResponse, StatusResponse, SourceNode

# Global instances
rag_engine: NoteWiseRAG | None = None
agent: NoteWiseAgent | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager to initialize resources on startup and clean up on shutdown."""
    global rag_engine, agent
    try:
        rag_engine = NoteWiseRAG()
        agent = NoteWiseAgent(rag_engine=rag_engine)
        print("NoteWiseRAG and NoteWiseAgent initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize engines: {e}")
        
    yield
    print("NoteWise-AI is shutting down.")

app = FastAPI(
    title="NoteWise-AI",
    description="Backend API for NoteWise-AI RAG and Agentic Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# New schema for Chat Endpoint
class ChatResponse(BaseModel):
    answer: str
    source_nodes: list[SourceNode] = []
    tools_used: list[str] = []

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to the NoteWise-AI API!"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Check application health status."""
    return {"status": "healthy"}

@app.get("/status", response_model=StatusResponse)
async def get_status() -> Any:
    """Return RAG index status and document count in the Chroma DB."""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized.")
    try:
        count = rag_engine.chroma_collection.count()
        index_ready = rag_engine.index is not None
        return StatusResponse(
            status="ready" if index_ready else "not_ready",
            index_ready=index_ready,
            collection_count=count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents() -> Dict[str, str]:
    """Trigger document ingestion from the configured data folder."""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized.")
    try:
        rag_engine.create_index()
        return {"message": "Documents ingested and index created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> Any:
    """Standard RAG query endpoint."""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized.")
    try:
        result = rag_engine.query(request.question)
        formatted_nodes = [
            SourceNode(text=node.node.text, score=node.score, metadata=node.node.metadata)
            for node in result.get("source_nodes", [])
        ]
        return QueryResponse(answer=result["answer"], source_nodes=formatted_nodes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_agent(request: QueryRequest) -> Any:
    """Agentic chat endpoint that dynamically routes to tools or RAG."""
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized.")
    try:
        result = agent.chat(request.question)
        formatted_nodes = [
            SourceNode(text=node.node.text, score=node.score, metadata=node.node.metadata)
            for node in result.get("source_nodes", [])
        ]
        return ChatResponse(
            answer=result["answer"],
            source_nodes=formatted_nodes,
            tools_used=result.get("tools_used", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
