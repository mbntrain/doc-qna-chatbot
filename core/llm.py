"""LLM interface — Phase 5 (Conversational RAG).

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

_HISTORY_TURNS = 6  # how many prior (user, assistant) pairs to send


def generate_answer(
    question: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
) -> str:
    """Send question + retrieved context + prior turns to the LLM.

    history: list of {"role": "user"|"assistant", "content": str}
    """
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Put GROQ_API_KEY=... in .env.test or .env "
            f"(project root: {_root})."
        )

    from groq import Groq

    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided document context. If the answer is not in the context, "
        "say so clearly. Do not fabricate information."
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # inject retrieved context as a system-level note before history
    messages.append({
        "role": "system",
        "content": f"Relevant document context for the current question:\n\n{context}",
    })

    # last N turns — strip any extra keys (e.g. "sources") Groq doesn't accept
    if history:
        clean = [
            {"role": m["role"], "content": m["content"]}
            for m in history[-(_HISTORY_TURNS * 2):]
        ]
        messages.extend(clean)

    messages.append({"role": "user", "content": question})

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()
