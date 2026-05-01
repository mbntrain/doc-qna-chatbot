import hashlib
import os
import pickle
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu required. Run: pip install faiss-cpu")

from ai_engine import _load_word2vec, _sentence_vector_w2v, _tokenize

_STOP = {"what", "who", "where", "when", "why", "how", "is", "are"}


class FAISSIndex:
    def __init__(self, cache_dir: str = ".faiss_cache"):
        self.cache_dir = cache_dir
        self.index = None
        # each entry: {"text": str, "doc_name": str}
        self.chunks_meta: list[dict] = []
        os.makedirs(cache_dir, exist_ok=True)

    # --- hashing ---

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def multi_hash(texts: list[str]) -> str:
        """Stable cache key for a set of documents (order-independent)."""
        hashes = sorted(FAISSIndex._hash(t) for t in texts)
        return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]

    def _cache_paths(self, doc_hash: str) -> tuple[str, str]:
        base = os.path.join(self.cache_dir, doc_hash)
        return f"{base}.index", f"{base}.pkl"

    # --- build ---

    def _embed(self, chunks_meta: list[dict]) -> np.ndarray:
        model = _load_word2vec()
        vecs = [
            _sentence_vector_w2v(
                [w for w in _tokenize(cm["text"]) if w.lower() not in _STOP],
                model,
            )
            for cm in chunks_meta
        ]
        return np.array(vecs, dtype=np.float32)

    def build(self, chunks: list[str], doc_name: str = "doc") -> None:
        """Single-doc convenience wrapper."""
        self.build_multi([(doc_name, chunks)])

    def build_multi(self, doc_chunks: list[tuple[str, list[str]]]) -> None:
        """Build one index from multiple docs.
        doc_chunks: [(doc_name, [chunk_text, ...]), ...]
        """
        self.chunks_meta = [
            {"text": chunk, "doc_name": doc_name}
            for doc_name, chunks in doc_chunks
            for chunk in chunks
        ]
        embeddings = self._embed(self.chunks_meta)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    # --- persist ---

    def save(self, doc_hash: str) -> None:
        idx_path, meta_path = self._cache_paths(doc_hash)
        faiss.write_index(self.index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks_meta, f)

    def load(self, doc_hash: str) -> bool:
        idx_path, meta_path = self._cache_paths(doc_hash)
        if not os.path.exists(idx_path):
            return False
        try:
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                self.chunks_meta = pickle.load(f)
            return True
        except Exception:
            return False

    # --- search ---

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        """Return top-k results as [(chunk_text, doc_name, score), ...]."""
        if self.index is None:
            raise ValueError("No index loaded")

        model = _load_word2vec()
        q_tokens = [w for w in _tokenize(query) if w.lower() not in _STOP]
        q_vec = np.array([_sentence_vector_w2v(q_tokens, model)], dtype=np.float32)

        distances, indices = self.index.search(q_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks_meta):
                score = round(1.0 / (1.0 + float(dist)), 4)
                cm = self.chunks_meta[idx]
                results.append((cm["text"], cm["doc_name"], score))
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
