"""
Unit tests for NoteWise AI — Hybrid Retrieval & Semantic Cache.
Run with: pytest tests/ -v
"""
import math
from app.engine.hybrid_retriever import BM25Index, reciprocal_rank_fusion, SemanticCache, cosine_sim
from llama_index.core.schema import TextNode, NodeWithScore


def test_bm25_exact_keyword_retrieval():
    """BM25 must rank document containing rare keyword higher."""
    bm25 = BM25Index()
    nodes = [
        TextNode(text="Python and FastAPI are used for backend web development."),
        TextNode(text="Kafka is a distributed event streaming platform for microservices."),
        TextNode(text="Docker containers allow reproducible cloud deployments."),
    ]
    bm25.fit(nodes)
    results = bm25.retrieve("Kafka streaming", top_k=2)

    assert len(results) > 0
    top_score, top_node = results[0]
    assert "Kafka" in top_node.text
    assert top_score > 0.0


def test_bm25_irrelevant_query_scores_zero():
    """Queries with zero overlapping terms should receive 0.0 score."""
    bm25 = BM25Index()
    nodes = [
        TextNode(text="Machine learning models require clean data."),
        TextNode(text="Neural networks use backpropagation."),
    ]
    bm25.fit(nodes)
    results = bm25.retrieve("xyz123 nonexistent query", top_k=2)

    for score, _ in results:
        assert score == 0.0


def test_reciprocal_rank_fusion_scoring():
    """RRF must correctly fuse dense and BM25 results with rank formula."""
    node1 = TextNode(text="First document about AI", id_="n1")
    node2 = TextNode(text="Second document about databases", id_="n2")

    dense_nodes = [
        NodeWithScore(node=node1, score=0.9),
        NodeWithScore(node=node2, score=0.5),
    ]
    bm25_nodes = [
        (1.5, node1),
        (0.8, node2),
    ]

    fused = reciprocal_rank_fusion(dense_nodes, bm25_nodes, k=60, top_k=2)
    assert len(fused) == 2
    # Node1 ranked 1 in both: score should be (1/61) + (1/61) = 2/61 ≈ 0.0328
    assert fused[0].node.text == "First document about AI"
    expected_score = (1.0 / 61) + (1.0 / 61)
    assert abs(fused[0].score - expected_score) < 1e-4


def test_cosine_similarity_computation():
    """Identical vectors should yield 1.0, orthogonal should yield 0.0."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert abs(cosine_sim(v1, v2) - 1.0) < 1e-5
    assert abs(cosine_sim(v1, v3) - 0.0) < 1e-5


def test_semantic_cache_hit_and_miss():
    """Cache should hit on high cosine similarity and miss on distant queries."""
    cache = SemanticCache(threshold=0.95)

    base_emb = [1.0, 0.0, 0.0]
    near_emb = [0.99, 0.05, 0.0]
    distant_emb = [0.0, 1.0, 0.0]

    cache.add(
        query_emb=base_emb,
        query="What is NoteWise?",
        answer="NoteWise is a document platform.",
        sources=[],
        eval_scores={},
        model="gemini-3.8-flash",
    )

    # Near embedding should hit
    hit_entry, sim = cache.lookup(near_emb)
    assert hit_entry is not None
    assert sim >= 0.95
    assert hit_entry["answer"] == "NoteWise is a document platform."

    # Distant embedding should miss
    miss_entry, sim_miss = cache.lookup(distant_emb)
    assert miss_entry is None
    assert sim_miss < 0.95


def test_empty_semantic_cache_returns_none():
    """Empty cache lookup returns None gracefully."""
    cache = SemanticCache(threshold=0.95)
    entry, sim = cache.lookup([1.0, 0.0])
    assert entry is None
    assert sim == 0.0
