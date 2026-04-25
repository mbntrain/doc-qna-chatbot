import re
import math
from collections import Counter

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


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse vectors (word count dicts).
    Measures the angle between two vectors — 1.0 = identical direction, 0.0 = unrelated.
    """
    # dot product: sum of (a[word] * b[word]) for shared words
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[w] * vec_b[w] for w in common)

    # magnitudes: sqrt(sum of squares)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _bow_vector(tokens: list[str]) -> dict[str, int]:
    """Convert token list into a Bag of Words vector (word → count).
    Each unique word is a dimension. The count is the value.
    """
    return dict(Counter(t for t in tokens if t not in STOP_WORDS))


# --- SEARCH METHOD 1: BM25 (Era 1 — industry standard, used by Elasticsearch) ---

def _search_bm25(keywords: list[str], sentences: list[str],
                  tokenized_sentences: list[list[str]]) -> tuple[int, float]:
    """BM25 improves TF-IDF with two fixes:
    1. TF saturation: 10 occurrences of 'java' isn't 10x better than 1. Diminishing returns.
    2. Document length normalization: short sentences aren't unfairly penalized.
    k1 = 1.5 controls TF saturation. b = 0.75 controls length normalization.
    """
    k1, b = 1.5, 0.75
    num = len(sentences)
    avg_len = sum(len(t) for t in tokenized_sentences) / num if num else 1

    # IDF per keyword (same concept, BM25 variant)
    idf = {}
    for kw in keywords:
        doc_count = sum(1 for tokens in tokenized_sentences if kw in tokens)
        idf[kw] = math.log((num - doc_count + 0.5) / (doc_count + 0.5) + 1)

    best_idx, best_score = -1, 0.0
    for i, tokens in enumerate(tokenized_sentences):
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for kw in keywords:
            freq = tf.get(kw, 0)
            # TF with saturation: freq / (freq + k1 * length_adjustment)
            tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_len))
            score += idf.get(kw, 0) * tf_norm
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, round(best_score, 4)


# --- SEARCH METHOD 2: Bag of Words + Cosine Similarity (Era 2) ---

def _search_bow_cosine(question: str, sentences: list[str],
                        tokenized_sentences: list[list[str]]) -> tuple[int, float]:
    """Represent question and each sentence as word-count vectors,
    then pick the sentence whose vector points in the most similar direction.
    """
    q_vec = _bow_vector(_tokenize(question))

    best_idx, best_score = -1, 0.0
    for i, tokens in enumerate(tokenized_sentences):
        s_vec = _bow_vector(tokens)
        score = _cosine_similarity(q_vec, s_vec)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, round(best_score, 4)


def answer_question(context: str, question: str,
                    method: str = "bow") -> tuple[str, float, dict]:
    """Find the most relevant sentence. method='bm25' or 'bow' (bag of words)."""
    keywords = _get_keywords(question)
    if not keywords:
        return "Could not understand the question.", 0.0, {}

    sentences = _split_sentences(context)
    if not sentences:
        return "No content to search.", 0.0, {}

    tokenized_sentences = [_tokenize(s) for s in sentences]

    if method == "bm25":
        best_idx, raw_score = _search_bm25(keywords, sentences, tokenized_sentences)
    else:
        best_idx, raw_score = _search_bow_cosine(question, sentences, tokenized_sentences)

    if best_idx == -1 or raw_score == 0:
        return "No relevant answer found in the document.", 0.0, {}

    matched = sum(1 for kw in keywords if kw in tokenized_sentences[best_idx])
    confidence = round(matched / len(keywords), 2)

    # return debug info so you can compare
    debug = {"method": method, "raw_score": raw_score, "sentence_idx": best_idx}
    return sentences[best_idx], confidence, debug
