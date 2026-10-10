import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Settings:
    """Application configuration and environment variables."""
    
    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chromadb")
    
    # RAG Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    # LLM Provider Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower() # Options: 'groq', 'openai'
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")     # Use 'gpt-4o' if using openai
    
    # Vector Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Instantiate a global settings object to be imported across the app
settings = Settings()
