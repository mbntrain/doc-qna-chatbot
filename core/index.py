"""SBERT-backed FAISS index.

Embeds text units with all-MiniLM-L6-v2 (384-dim, normalized).
Uses IndexFlatIP so search = cosine similarity.
Cache keyed by content hash so re-uploads are instant.
"""
import hashlib
import os
import pickle

import numpy as np

_faiss_mod = None


def _faiss():
    """Lazy import: faiss-cpu wheels can fail under NumPy 2.x; defer so `Index` still imports."""
    global _faiss_mod
    if _faiss_mod is None:
        try:
            import faiss as _f

            _faiss_mod = _f
        except Exception as e:
            raise ImportError(
                "faiss-cpu required: pip install faiss-cpu. "
                "If you see NumPy / _ARRAY_API errors, use: pip install 'numpy<2'"
            ) from e
    return _faiss_mod


_sbert_model = None


def _load_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


class Index:
    def __init__(self, cache_dir: str = ".faiss_cache"):
        self.cache_dir = cache_dir
        self.faiss_index = None
        self.chunks_meta: list[dict] = []  # [{"text": str, "doc_name": str}]
        os.makedirs(cache_dir, exist_ok=True)

    # --- hashing ---

    @staticmethod
    def content_hash(texts: list[str]) -> str:
        """Stable cache key for a set of document units (order-independent)."""
        hashes = sorted(
            hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts
        )
        return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]

    def _cache_paths(self, key: str) -> tuple[str, str]:
        base = os.path.join(self.cache_dir, f"{key}_sbert_v2")
        return f"{base}.index", f"{base}.pkl"

    # --- build ---

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = _load_sbert()
        vecs = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,  # required for IndexFlatIP = cosine similarity
        )
        return np.array(vecs, dtype=np.float32)

    def build(self, doc_chunks: list[tuple[str, list[str]]]) -> None:
        """Build index from multiple docs.
        doc_chunks: [(doc_name, [unit_text, ...]), ...]
        """
        self.chunks_meta = [
            {"text": unit, "doc_name": doc_name}
            for doc_name, units in doc_chunks
            for unit in units
        ]
        embeddings = self._embed([cm["text"] for cm in self.chunks_meta])
        f = _faiss()
        self.faiss_index = f.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

    # --- persist ---

    def save(self, key: str) -> None:
        idx_path, meta_path = self._cache_paths(key)
        _faiss().write_index(self.faiss_index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks_meta, f)

    def load(self, key: str) -> bool:
        idx_path, meta_path = self._cache_paths(key)
        if not os.path.exists(idx_path):
            return False
        try:
            self.faiss_index = _faiss().read_index(idx_path)
            with open(meta_path, "rb") as f:
                self.chunks_meta = pickle.load(f)
            return True
        except Exception:
            return False

    # --- search ---

    def search(
        self, query: str, k: int = 3, return_indices: bool = False
    ) -> list[dict]:
        """Return top-k results as [{"text": str, "doc_name": str, "score": float}].

        return_indices=True adds "_idx" (corpus position) to each result —
        used by HybridRetriever for RRF fusion.
        """
        if self.faiss_index is None:
            raise ValueError("Index not loaded.")

        model = _load_sbert()
        q_vec = np.array(
            model.encode([query], normalize_embeddings=True),
            dtype=np.float32,
        )
        distances, indices = self.faiss_index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks_meta):
                # IP on normalized vectors is in [-1, 1]; map to [0, 1]
                score = round((float(dist) + 1.0) / 2.0, 4)
                result = {**self.chunks_meta[idx], "score": score}
                if return_indices:
                    result["_idx"] = int(idx)
                results.append(result)
        return results

    # --- info ---

    @property
    def num_chunks(self) -> int:
        return len(self.chunks_meta)

    @property
    def doc_names(self) -> list[str]:
        seen: list[str] = []
        for cm in self.chunks_meta:
            if cm["doc_name"] not in seen:
                seen.append(cm["doc_name"])
        return seen
