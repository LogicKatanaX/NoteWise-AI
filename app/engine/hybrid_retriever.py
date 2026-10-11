"""
NoteWise AI — Hybrid Retrieval, Lexical BM25, Semantic Cache & Guardrails Engine.
"""
import math
import re
from typing import Any, List, Dict, Tuple, Optional
from llama_index.core.schema import NodeWithScore


# ---------------------------------------------------------------------------
# Security Guardrails — Prompt Injection & Adversarial Filter
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"system\s+prompt\s*(reveal|leak|show)",
    r"you\s+are\s+now\s+(dan|developer\s+mode|unrestricted)",
    r"disregard\s+all\s+(rules|guidelines)",
    r"reveal\s+your\s+(secret|initial|developer)\s+prompt",
]

def check_prompt_guardrails(prompt: str) -> Tuple[bool, str]:
    """Inspect query against adversarial patterns before LLM execution."""
    p_lower = prompt.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, p_lower):
            return False, "⚠️ **Security Guardrail Active**: Adversarial prompt pattern detected. Query quarantined for safety compliance."
    return True, ""


# ---------------------------------------------------------------------------
# Okapi BM25 Lexical Keyword Engine (Zero-Dependency Pure Python)
# ---------------------------------------------------------------------------
class BM25Index:
    """Exact Okapi BM25 index for sparse lexical keyword retrieval."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.nodes = []
        self.doc_len = []
        self.doc_freqs = []
        self.avgdl = 0.0
        self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text)]

    def fit(self, nodes: List[Any]) -> None:
        self.nodes = nodes
        self.doc_len = []
        self.doc_freqs = []
        df = {}
        for node in nodes:
            text = node.text if hasattr(node, "text") else str(node)
            tokens = self._tokenize(text)
            self.doc_len.append(len(tokens))
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)
            for t in freq:
                df[t] = df.get(t, 0) + 1

        n_docs = len(nodes)
        self.avgdl = sum(self.doc_len) / max(1, n_docs)
        self.idf = {}
        for t, count in df.items():
            self.idf[t] = math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, Any]]:
        if not self.nodes:
            return []
        query_tokens = self._tokenize(query)
        scored = []
        for i, (freqs, dl) in enumerate(zip(self.doc_freqs, self.doc_len)):
            score = 0.0
            for t in query_tokens:
                if t in freqs:
                    f = freqs[t]
                    idf = self.idf.get(t, 0.0)
                    denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, self.avgdl)))
                    score += idf * (f * (self.k1 + 1.0)) / denom
            scored.append((score, self.nodes[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(dense_nodes: List[Any], bm25_scored_nodes: List[Tuple[float, Any]], k: int = 60, top_k: int = 5) -> List[Any]:
    """
    Combines dense vector retrieval ranks with BM25 lexical ranks.
    RRF Score(d) = sum(1 / (k + rank))
    """
    scores = {}
    node_map = {}

    for rank, node in enumerate(dense_nodes, 1):
        nid = getattr(node.node, "node_id", str(id(node.node)))
        scores[nid] = scores.get(nid, 0.0) + (1.0 / (k + rank))
        node_map[nid] = node

    for rank, (bm_score, raw_node) in enumerate(bm25_scored_nodes, 1):
        nid = getattr(raw_node, "node_id", str(id(raw_node)))
        scores[nid] = scores.get(nid, 0.0) + (1.0 / (k + rank))
        if nid not in node_map:
            node_map[nid] = NodeWithScore(node=raw_node, score=bm_score)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for nid in sorted_ids[:top_k]:
        wrapped = node_map[nid]
        wrapped.score = scores[nid]
        fused.append(wrapped)
    return fused


# ---------------------------------------------------------------------------
# Semantic Vector Caching Layer (<5ms Latency, $0 LLM Cost)
# ---------------------------------------------------------------------------
def cosine_sim(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

class SemanticCache:
    """In-memory cosine vector cache for sub-5ms repeated queries."""
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.entries = []  # list of dicts

    def lookup(self, query_emb: List[float]) -> Tuple[Optional[Dict[str, Any]], float]:
        if not self.entries:
            return None, 0.0
        best_entry = None
        best_sim = -1.0
        for entry in self.entries:
            sim = cosine_sim(query_emb, entry["emb"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_sim >= self.threshold:
            return best_entry, best_sim
        return None, best_sim

    def add(self, query_emb: List[float], query: str, answer: str, sources: List[Any], eval_scores: Dict[str, Any], model: str) -> None:
        self.entries.append({
            "emb": query_emb,
            "query": query,
            "answer": answer,
            "sources": sources,
            "eval_scores": eval_scores,
            "model": model,
        })

