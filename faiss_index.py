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

# Lazy SBERT model cache
_sbert_model = None


def _load_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


class FAISSIndex:
    def __init__(self, cache_dir: str = ".faiss_cache", model_type: str = "glove"):
        """
        model_type: "glove" uses GloVe-50 word averaging (50-dim vectors).
                    "sbert" uses all-MiniLM-L6-v2 sentence embeddings (384-dim vectors).
        """
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.index = None
        self.chunks_meta: list[dict] = []  # [{"text": str, "doc_name": str}]
        os.makedirs(cache_dir, exist_ok=True)

    # --- hashing ---

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def multi_hash(texts: list[str]) -> str:
        hashes = sorted(FAISSIndex._hash(t) for t in texts)
        return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16]

    def _cache_paths(self, doc_hash: str) -> tuple[str, str]:
        # model_type in filename so glove and sbert caches don't collide
        base = os.path.join(self.cache_dir, f"{doc_hash}_{self.model_type}")
        return f"{base}.index", f"{base}.pkl"

    # --- embedding ---

    def _embed(self, chunks_meta: list[dict]) -> np.ndarray:
        if self.model_type == "sbert":
            return self._embed_sbert(chunks_meta)
        return self._embed_glove(chunks_meta)

    def _embed_glove(self, chunks_meta: list[dict]) -> np.ndarray:
        model = _load_word2vec()
        vecs = [
            _sentence_vector_w2v(
                [w for w in _tokenize(cm["text"]) if w.lower() not in _STOP],
                model,
            )
            for cm in chunks_meta
        ]
        return np.array(vecs, dtype=np.float32)

    def _embed_sbert(self, chunks_meta: list[dict]) -> np.ndarray:
        model = _load_sbert()
        texts = [cm["text"] for cm in chunks_meta]
        vecs = model.encode(texts, show_progress_bar=False, batch_size=64)
        return np.array(vecs, dtype=np.float32)

    # --- build ---

    def build(self, chunks: list[str], doc_name: str = "doc") -> None:
        self.build_multi([(doc_name, chunks)])

    def build_multi(self, doc_chunks: list[tuple[str, list[str]]]) -> None:
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
        """Return [(chunk_text, doc_name, score), ...]"""
        if self.index is None:
            raise ValueError("No index loaded")

        if self.model_type == "sbert":
            model = _load_sbert()
            q_vec = np.array(model.encode([query]), dtype=np.float32)
        else:
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
