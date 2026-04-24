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


def answer_question(context: str, question: str) -> tuple[str, float]:
    """Find the most relevant sentence using TF-IDF scoring."""
    keywords = _get_keywords(question)
    if not keywords:
        return "Could not understand the question.", 0.0

    sentences = _split_sentences(context)
    if not sentences:
        return "No content to search.", 0.0

    num_sentences = len(sentences)
    tokenized_sentences = [_tokenize(s) for s in sentences]

    # IDF: how rare is each keyword across all sentences
    idf = {}
    for kw in keywords:
        doc_count = sum(1 for tokens in tokenized_sentences if kw in tokens)
        idf[kw] = math.log((num_sentences + 1) / (doc_count + 1)) + 1  # smoothed IDF

    # score each sentence by TF-IDF
    best_idx = -1
    best_score = 0.0
    for i, tokens in enumerate(tokenized_sentences):
        tf = Counter(tokens)
        total = len(tokens) or 1
        score = sum((tf[kw] / total) * idf.get(kw, 0) for kw in keywords)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1 or best_score == 0:
        return "No relevant answer found in the document.", 0.0

    # normalize confidence to 0-1 range based on keyword coverage
    matched = sum(1 for kw in keywords if kw in tokenized_sentences[best_idx])
    confidence = round(matched / len(keywords), 2)

    return sentences[best_idx], confidence
