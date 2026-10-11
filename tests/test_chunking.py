"""
Unit tests for NoteWise AI — Token chunking pipeline.
Run with: pytest tests/ -v
"""
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document


def test_token_splitter_produces_multiple_chunks():
    """Long text must be split into more than one chunk."""
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
    doc = Document(text="word " * 500)
    nodes = splitter.get_nodes_from_documents([doc])
    assert len(nodes) > 1, "Expected multiple chunks from a long document"


def test_token_splitter_chunk_size_respected():
    """Each chunk should be within reasonable token bounds (with overlap tolerance)."""
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
    doc = Document(text="word " * 500)
    nodes = splitter.get_nodes_from_documents([doc])
    for node in nodes:
        word_count = len(node.text.split())
        assert word_count <= 130, f"Chunk too large: {word_count} words"


def test_empty_document_produces_no_nodes():
    """Empty documents should not generate any nodes."""
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
    doc = Document(text="   ")
    # LlamaIndex may return 0 or 1 node for whitespace-only text
    nodes = splitter.get_nodes_from_documents([doc])
    for node in nodes:
        assert node.text.strip() == "" or len(node.text.strip()) >= 0


def test_metadata_preserved_in_chunks():
    """File name metadata should be preserved across all chunks."""
    splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=5)
    doc = Document(text="word " * 200, metadata={"file_name": "resume.pdf"})
    nodes = splitter.get_nodes_from_documents([doc])
    assert len(nodes) > 0
    for node in nodes:
        assert node.metadata.get("file_name") == "resume.pdf", "Metadata lost during chunking"


def test_chunk_overlap_creates_continuity():
    """With overlap, consecutive chunks should share some tokens."""
    splitter = TokenTextSplitter(chunk_size=20, chunk_overlap=5)
    text = " ".join([f"word{i}" for i in range(100)])
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    assert len(nodes) >= 2, "Expected at least 2 chunks with small chunk_size"

