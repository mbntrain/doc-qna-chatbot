import re


def answer_question(context: str, question: str) -> tuple[str, float]:
    """Find the most relevant sentence based on keyword overlap."""
    stop_words = {"what", "who", "where", "when", "why", "how", "is", "are", "was",
                  "were", "do", "does", "did", "the", "a", "an", "in", "on", "of", "and", "or", "to"}
    keywords = [w.lower() for w in re.findall(r'\w+', question) if w.lower() not in stop_words]

    if not keywords:
        return "Could not understand the question.", 0.0

    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    if not sentences:
        return "No content to search.", 0.0

    best_sentence = ""
    best_score = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        score = sum(1 for kw in keywords if kw in words)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score == 0:
        return "No relevant answer found in the document.", 0.0

    # confidence = fraction of keywords matched
    confidence = round(best_score / len(keywords), 2)
    return best_sentence, confidence
