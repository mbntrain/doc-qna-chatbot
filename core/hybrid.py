"""Hybrid BM25 + SBERT retriever with Reciprocal Rank Fusion (RRF).

RRF formula:
    RRF(d) = Σ  1 / (k + r(d))   for each ranked list r in R
    k = 60  (standard constant — dampens top-rank dominance)

Why RRF instead of score blending:
- BM25 scores are unbounded; SBERT scores are [0,1]. Different scales can't be added.
- RRF only uses ranks, so scale never matters.
- A chunk ranked #3 by BOTH methods beats a chunk ranked #1 by one method only.
  Consensus = confidence.
"""
from __future__ import annotations

from core.index import Index

_RRF_K = 60


class HybridRetriever:
    """Wraps an SBERT Index, adds BM25, fuses results via RRF."""

    def __init__(self, sbert_index: Index):
        self.sbert_index = sbert_index
        self._bm25 = None

    def build_bm25(self) -> None:
        """Build a BM25Okapi index over the same corpus as the SBERT index."""
        from rank_bm25 import BM25Okapi

        corpus = self.sbert_index.chunks_meta  # [{"text": str, "doc_name": str}]
        tokenized = [chunk["text"].lower().split() for chunk in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Run BM25 + SBERT, fuse with RRF, return top-k chunks.

        Each result dict:
            text, doc_name, score (RRF), source (e.g. "BM25 #1 · SBERT #3")
        """
        if self._bm25 is None:
            raise ValueError("Call build_bm25() before searching.")

        corpus = self.sbert_index.chunks_meta
        n = len(corpus)
        # fetch more than k so fusion has a wide candidate pool
        fetch_n = min(n, max(k * 5, 30))

        # ── BM25 ranking ─────────────────────────────────────────────────────
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:fetch_n]
        bm25_rank: dict[int, int] = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}

        # ── SBERT ranking ────────────────────────────────────────────────────
        sbert_results = self.sbert_index.search(query, k=fetch_n, return_indices=True)
        sbert_rank: dict[int, int] = {
            r["_idx"]: rank + 1 for rank, r in enumerate(sbert_results)
        }

        # ── RRF fusion ───────────────────────────────────────────────────────
        candidates = set(bm25_order) | set(sbert_rank.keys())
        rrf: dict[int, float] = {}
        for idx in candidates:
            score = 0.0
            if idx in bm25_rank:
                score += 1.0 / (_RRF_K + bm25_rank[idx])
            if idx in sbert_rank:
                score += 1.0 / (_RRF_K + sbert_rank[idx])
            rrf[idx] = score

        top_k_indices = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]

        results = []
        for idx in top_k_indices:
            br = bm25_rank.get(idx)
            sr = sbert_rank.get(idx)

            if br and sr:
                source_label = f"BM25 #{br} · SBERT #{sr}"
            elif br:
                source_label = f"BM25 #{br} only"
            else:
                source_label = f"SBERT #{sr} only"

            results.append({
                "text": corpus[idx]["text"],
                "doc_name": corpus[idx]["doc_name"],
                "score": round(rrf[idx], 6),
                "source": source_label,
            })

        return results
