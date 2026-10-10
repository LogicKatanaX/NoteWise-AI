import os
import sys

# Ensure Python can find the app module if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.engine.rag_engine import NoteWiseRAG
from app.core.config import settings

def main() -> None:
    """Script to ingest documents into the ChromaDB vector index."""
    print("=========================================")
    print("NoteWise-AI: Starting Document Ingestion")
    print("=========================================")
    
    try:
        # Ensure the data directory exists
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        print(f"Data Directory: {settings.DATA_DIR}")
        print(f"ChromaDB Directory: {settings.CHROMA_DB_DIR}")
        print("\nInitializing RAG Engine (this might download embedding models on first run)...")
        rag = NoteWiseRAG()
        
        print(f"\nLoading and indexing documents from {settings.DATA_DIR}...")
        rag.create_index(directory=settings.DATA_DIR)
        
        print("\n✅ Ingestion completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

