import hashlib
import json
import os
import pickle
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu required. Run: pip install faiss-cpu")

from ai_engine import _load_word2vec, _sentence_vector_w2v, _tokenize


class FAISSIndex:
    def __init__(self, cache_dir: str = ".faiss_cache"):
        self.cache_dir = cache_dir
        self.index = None
        self.chunks = None
        self.doc_hash = None
        os.makedirs(cache_dir, exist_ok=True)
    
    def _compute_hash(self, text: str) -> str:
        """Hash of document content for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _get_cache_paths(self, doc_hash: str) -> tuple[str, str, str]:
        """Return paths for index, chunks, metadata."""
        index_path = os.path.join(self.cache_dir, f"{doc_hash}.index")
        chunks_path = os.path.join(self.cache_dir, f"{doc_hash}.pkl")
        meta_path = os.path.join(self.cache_dir, f"{doc_hash}.meta")
        return index_path, chunks_path, meta_path
    
    def build(self, chunks: list[str]) -> None:
        """Build FAISS index from chunks using GloVe embeddings."""
        model = _load_word2vec()
        embeddings = []
        
        for chunk in chunks:
            tokens = [w for w in _tokenize(chunk) if w.lower() not in 
                     {"what", "who", "where", "when", "why", "how", "is", "are"}]
            vec = _sentence_vector_w2v(tokens, model)
            embeddings.append(vec)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index (flat L2 distance)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = chunks
    
    def save(self, doc_hash: str) -> None:
        """Persist index and chunks to disk."""
        if self.index is None or self.chunks is None:
            raise ValueError("No index built yet")
        
        index_path, chunks_path, meta_path = self._get_cache_paths(doc_hash)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(meta_path, 'w') as f:
            json.dump({"doc_hash": doc_hash, "num_chunks": len(self.chunks)}, f)
    
    def load(self, doc_hash: str) -> bool:
        """Load index from cache. Return True if successful."""
        index_path, chunks_path, meta_path = self._get_cache_paths(doc_hash)
        
        if not os.path.exists(index_path):
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            self.doc_hash = doc_hash
            return True
        except Exception:
            return False
    
    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Search for top-k most similar chunks. Return [(chunk, distance), ...]."""
        if self.index is None or self.chunks is None:
            raise ValueError("No index loaded")
        
        model = _load_word2vec()
        q_tokens = [w for w in _tokenize(query) if w.lower() not in 
                   {"what", "who", "where", "when", "why", "how", "is", "are"}]
        q_vec = _sentence_vector_w2v(q_tokens, model)
        q_vec = np.array([q_vec], dtype=np.float32)
        
        distances, indices = self.index.search(q_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                # Convert L2 distance to similarity (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + dist)
                results.append((self.chunks[idx], round(similarity, 4)))
        return results
