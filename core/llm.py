"""LLM interface — Phase 4.

Thin wrapper around Groq so the provider is swappable.
Reads GROQ_API_KEY from environment (.env file).
"""
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.test")
load_dotenv(_root / ".env")


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Send question + retrieved context to the LLM, return generated answer."""
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Put GROQ_API_KEY=... in .env.test or .env "
            f"(project root: {_root})."
        )

    from groq import Groq

    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()
