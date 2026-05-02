"""Sentence-window retrieval — SBERT + BM25 + RRF.

Why this beats fixed-size chunks:
  - Index unit = individual sentence  → SBERT/BM25 match precisely
  - Retrieval unit = matched sentence ± window  → LLM gets enough context

A 300-token chunk might contain a formula buried under 10 unrelated sentences.
Sentence-window finds the exact sentence, then gives the LLM the 2 sentences
before and after it — enough to understand the formula without the noise.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re

import numpy as np

_WINDOW = 2   # sentences before + after the matched sentence
_RRF_K  = 60

_sbert_model = None


def _load_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering very short fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 15]


class SentenceWindowRetriever:
    """Indexes individual sentences; retrieves them with surrounding context."""

    def __init__(self, window: int = _WINDOW, cache_dir: str = ".faiss_cache"):
        self.window = window
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # flat list of every indexed sentence
        self._records: list[dict] = []        # {text, doc_name, sent_idx}
        # per-doc full sentence list (for window expansion)
        self._doc_sents: dict[str, list[str]] = {}

        self._faiss_index = None
        self._bm25 = None

    # ── hashing (same pattern as Index) ──────────────────────────────────────

    @staticmethod
    def content_hash(texts: list[str]) -> str:
        hashes = sorted(hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts)
        return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]

    def _cache_paths(self, key: str) -> tuple[str, str]:
        base = os.path.join(self.cache_dir, f"{key}_sw_v1")
        return f"{base}.index", f"{base}.pkl"

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, docs: list[tuple[str, str]]) -> None:
        """Build index from raw document texts.

        docs: [(doc_name, full_text), ...]
        """
        self._records = []
        self._doc_sents = {}

        for doc_name, text in docs:
            sents = _split_sentences(text)
            self._doc_sents[doc_name] = sents
            for i, sent in enumerate(sents):
                self._records.append({
                    "text": sent,
                    "doc_name": doc_name,
                    "sent_idx": i,
                })

        self._build_sbert()
        self._build_bm25()

    def _build_sbert(self) -> None:
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss-cpu required: pip install faiss-cpu") from e

        model = _load_sbert()
        texts = [r["text"] for r in self._records]
        vecs = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )
        vecs = np.array(vecs, dtype=np.float32)
        self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])
        self._faiss_index.add(vecs)

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        tokenized = [r["text"].lower().split() for r in self._records]
        self._bm25 = BM25Okapi(tokenized)

    # ── persist ───────────────────────────────────────────────────────────────

    def save(self, key: str) -> None:
        import faiss
        idx_path, meta_path = self._cache_paths(key)
        faiss.write_index(self._faiss_index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"records": self._records, "doc_sents": self._doc_sents}, f
            )

    def load(self, key: str) -> bool:
        import faiss
        idx_path, meta_path = self._cache_paths(key)
        if not os.path.exists(idx_path):
            return False
        try:
            self._faiss_index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            self._records = data["records"]
            self._doc_sents = data["doc_sents"]
            self._build_bm25()   # fast to rebuild, not worth persisting
            return True
        except Exception:
            return False

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict]:
        """RRF over sentence-level BM25 + SBERT, expand to window on return.

        Each result:
            text             — window context (matched sentence ± N)
            matched_sentence — the exact sentence that triggered the match
            doc_name         — source document
            score            — RRF score
            source           — e.g. "BM25 #2 · SBERT #1"
        """
        n = len(self._records)
        fetch_n = min(n, max(k * 5, 30))

        # ── SBERT ─────────────────────────────────────────────────────────
        model = _load_sbert()
        q_vec = np.array(
            model.encode([query], normalize_embeddings=True),
            dtype=np.float32,
        )
        dists, idxs = self._faiss_index.search(q_vec, fetch_n)
        sbert_rank: dict[int, int] = {
            int(idx): rank + 1
            for rank, idx in enumerate(idxs[0])
            if 0 <= idx < n
        }

        # ── BM25 ──────────────────────────────────────────────────────────
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:fetch_n]
        bm25_rank: dict[int, int] = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}

        # ── RRF ───────────────────────────────────────────────────────────
        candidates = set(sbert_rank) | set(bm25_rank)
        rrf: dict[int, float] = {}
        for idx in candidates:
            score = 0.0
            if idx in bm25_rank:
                score += 1.0 / (_RRF_K + bm25_rank[idx])
            if idx in sbert_rank:
                score += 1.0 / (_RRF_K + sbert_rank[idx])
            rrf[idx] = score

        top_k = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]

        # ── expand to window, deduplicate overlapping windows ─────────────
        results: list[dict] = []
        seen: set[tuple] = set()

        for idx in top_k:
            rec = self._records[idx]
            doc_sents = self._doc_sents[rec["doc_name"]]
            si = rec["sent_idx"]

            start = max(0, si - self.window)
            end   = min(len(doc_sents), si + self.window + 1)
            key   = (rec["doc_name"], start, end)

            if key in seen:
                continue
            seen.add(key)

            window_text = " ".join(doc_sents[start:end])

            br = bm25_rank.get(idx)
            sr = sbert_rank.get(idx)
            if br and sr:
                source_label = f"BM25 #{br} · SBERT #{sr}"
            elif br:
                source_label = f"BM25 #{br} only"
            else:
                source_label = f"SBERT #{sr} only"

            results.append({
                "text": window_text,
                "matched_sentence": rec["text"],
                "doc_name": rec["doc_name"],
                "score": round(rrf[idx], 6),
                "source": source_label,
            })

        return results

    # ── info ──────────────────────────────────────────────────────────────────

    @property
    def num_sentences(self) -> int:
        return len(self._records)

    @property
    def doc_names(self) -> list[str]:
        seen: list[str] = []
        for r in self._records:
            if r["doc_name"] not in seen:
                seen.append(r["doc_name"])
        return seen
