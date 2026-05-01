import re
import numpy as np

STOP_WORDS = {"what", "who", "where", "when", "why", "how", "is", "are", "was",
              "were", "do", "does", "did", "the", "a", "an", "in", "on", "of",
              "and", "or", "to", "it", "for", "with", "this", "that", "be", "has", "have"}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r'\w+', text)]


def _get_keywords(question: str) -> list[str]:
    return [w for w in _tokenize(question) if w not in STOP_WORDS]


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 10]


# --- Word2Vec / GloVe-50 ---

_w2v_model = None


def _load_word2vec():
    global _w2v_model
    if _w2v_model is None:
        import gensim.downloader as api
        _w2v_model = api.load("glove-wiki-gigaword-50")
    return _w2v_model


def _sentence_vector_w2v(tokens: list[str], model) -> np.ndarray:
    vectors = [model[w] for w in tokens if w in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def _cosine_sim_dense(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    mag = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / mag) if mag > 0 else 0.0


def _search_word2vec(question: str, sentences: list[str],
                     tokenized_sentences: list[list[str]]) -> tuple[int, float]:
    model = _load_word2vec()
    q_tokens = [w for w in _tokenize(question) if w not in STOP_WORDS]
    q_vec = _sentence_vector_w2v(q_tokens, model)

    best_idx, best_score = -1, 0.0
    for i, tokens in enumerate(tokenized_sentences):
        s_vec = _sentence_vector_w2v(tokens, model)
        score = _cosine_sim_dense(q_vec, s_vec)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, round(best_score, 4)


# --- Dispatch ---

def answer_question(context: str, question: str,
                    method: str = "word2vec") -> tuple[str, float, dict]:
    """Sentence-level search via Word2Vec. FAISS and SBERT are handled directly in app.py."""
    keywords = _get_keywords(question)
    if not keywords:
        return "Could not understand the question.", 0.0, {}

    sentences = _split_sentences(context)
    if not sentences:
        return "No content to search.", 0.0, {}

    tokenized_sentences = [_tokenize(s) for s in sentences]
    best_idx, raw_score = _search_word2vec(question, sentences, tokenized_sentences)

    if best_idx == -1 or raw_score == 0:
        return "No relevant answer found in the document.", 0.0, {}

    matched = sum(1 for kw in keywords if kw in tokenized_sentences[best_idx])
    confidence = round(matched / len(keywords), 2)
    return sentences[best_idx], confidence, {"method": "word2vec", "raw_score": raw_score}
