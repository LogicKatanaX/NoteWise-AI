from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Schema for incoming RAG queries."""
    question: str

class SourceNode(BaseModel):
    """Schema for individual document source nodes retrieved by the engine."""
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    """Schema for RAG query results including answer and source context."""
    answer: str
    source_nodes: List[SourceNode] = []

class StatusResponse(BaseModel):
    """Schema for API and Vector Index status."""
    status: str
    index_ready: bool
    collection_count: int

