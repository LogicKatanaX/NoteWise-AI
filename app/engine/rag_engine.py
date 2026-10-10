import os
import chromadb
from typing import List, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

# Advanced RAG components (wrapped in try-except for graceful fallback)
try:
    from llama_index.retrievers.bm25 import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False

from app.core.config import settings

class NoteWiseRAG:
    """Advanced RAG engine with Hybrid Search and CrossEncoder Re-ranking."""
    
    def __init__(self) -> None:
        """Initialize the RAG engine, configure settings, and setup models."""
        # 1. Global Settings & LLM setup
        self._configure_llm()
        
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL
        )
        Settings.chunk_size = settings.CHUNK_SIZE
        Settings.chunk_overlap = settings.CHUNK_OVERLAP
        
        # 2. Setup Vector Store (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="notewise_docs"
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # 3. Setup Docstore Persistence (Required to keep docs around for BM25)
        self.persist_dir = os.path.join(os.path.dirname(settings.CHROMA_DB_DIR), "storage")
        
        if os.path.exists(self.persist_dir):
            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.persist_dir,
                vector_store=self.vector_store
            )
        else:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
        # 4. Initialize Re-ranker model (Loaded once to save time across queries)
        self.reranker = None
        if RERANK_AVAILABLE:
            self.reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-base",
                top_n=3
            )
            
        self.index = None
        self.query_engine = None
        
        # 5. Load existing index if data is present
        if self.chroma_collection.count() > 0:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
            )
            self._setup_query_engine()

    def _configure_llm(self) -> None:
        """Configure the global LLM based on environment settings."""
        if settings.LLM_PROVIDER == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")
            try:
                from llama_index.llms.groq import Groq
                Settings.llm = Groq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY)
            except ImportError:
                raise ImportError("Missing package. Please run: pip install llama-index-llms-groq")
                
        elif settings.LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")
            try:
                from llama_index.llms.openai import OpenAI
                Settings.llm = OpenAI(model=settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY)
            except ImportError:
                raise ImportError("Missing package. Please run: pip install llama-index-llms-openai")
                
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. Use 'groq' or 'openai'.")

    def _setup_query_engine(self) -> None:
        """Configure Hybrid Search (Vector + BM25) and Re-ranking pipeline."""
        if not self.index:
            return
            
        # --- HYBRID SEARCH SETUP ---
        # Retrieve top 10 candidates using Vector Search
        vector_retriever = self.index.as_retriever(similarity_top_k=10)
        
        nodes = list(self.index.docstore.docs.values())
        if nodes and BM25_AVAILABLE:
            # Retrieve top 10 candidates using Keyword Search (BM25)
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=10
            )
            
            # Fuse the results using Reciprocal Rank Fusion
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=10,
                num_queries=1,
                mode="reciprocal_rerank"
            )
        else:
            # Fallback to standard vector search if BM25 is missing or docstore is empty
            retriever = vector_retriever
            
        # --- RE-RANKING SETUP ---
        node_postprocessors = []
        if self.reranker:
            # Re-rank the fused top 10 candidates and keep the best 3
            node_postprocessors.append(self.reranker)
            
        # Create final query engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=node_postprocessors
        )

    def load_documents(self, directory: str = settings.DATA_DIR) -> List[Document]:
        """Load all documents from the specified directory."""
        reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
        return reader.load_data()

    def create_index(self, directory: str = settings.DATA_DIR) -> None:
        """Ingest documents, build vector/BM25 indexes, and setup query engine."""
        docs = self.load_documents(directory)
        if not docs:
            print("No documents found to index.")
            return
            
        # Build index and populate docstore
        self.index = VectorStoreIndex.from_documents(
            docs,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        # Persist docstore for BM25 so it can be reloaded on restart
        os.makedirs(self.persist_dir, exist_ok=True)
        self.storage_context.persist(persist_dir=self.persist_dir)
        
        # Initialize query engine with new data
        self._setup_query_engine()

    def query(self, question: str) -> Dict[str, Any]:
        """Execute advanced RAG query returning answer and re-ranked source nodes."""
        if not self.query_engine:
            raise ValueError("The knowledge base is currently empty or uninitialized. Please ingest documents first via the UI or /ingest.")
        
        # The query engine will now automatically:
        # 1. Retrieve top 10 (Vector + BM25)
        # 2. Re-rank top 10 -> top 3
        # 3. Generate answer using the refined context
        response = self.query_engine.query(question)
        
        return {
            "answer": str(response),
            "source_nodes": response.source_nodes
        }
