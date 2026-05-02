import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
load_dotenv(_root / ".env.test")
load_dotenv(_root / ".env")

from core.parser import parse_file
from core.chunker import chunk_text
from core.index import Index
from core.llm import generate_answer

st.title("Jay — AI Doc Q&A Chatbot")

# --- Upload ---
col1, col_sep, col2 = st.columns([5, 1, 5])

with col1:
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

with col_sep:
    st.markdown(
        "<div style='text-align:center; padding-top:40px; font-weight:bold;'>OR</div>",
        unsafe_allow_html=True,
    )

with col2:
    pasted_text = st.text_area("Paste your text here", height=150)

_, btn_col, _ = st.columns([4, 2, 4])
with btn_col:
    load_clicked = st.button("Load", use_container_width=True)


def _units(text: str, doc_name: str) -> list[str]:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if len(s.strip()) > 10
    ]
    if len(sentences) >= 2:
        return sentences
    chunks = [c.text for c in chunk_text(text, doc_name, chunk_size=120, overlap=30)]
    return chunks or [text.strip()]


if load_clicked:
    docs: list[dict] = []

    if uploaded_files:
        for f in uploaded_files:
            try:
                docs.append({"name": f.name, "text": parse_file(f)})
            except ValueError as e:
                st.error(f"{f.name}: {e}")
    elif pasted_text.strip():
        docs = [{"name": "Pasted text", "text": pasted_text.strip()}]
    else:
        st.error("Upload a file or paste text first.")

    if docs:
        doc_chunks = [(d["name"], _units(d["text"], d["name"])) for d in docs]
        cache_material = [
            f"{name}\n" + "\n".join(units) for name, units in doc_chunks
        ]
        cache_key = Index.content_hash(cache_material + ["sbert-v1"])

        with st.spinner("Building index (first run downloads ~90MB)..."):
            idx = Index()
            if not idx.load(cache_key):
                idx.build(doc_chunks)
                idx.save(cache_key)
                idx_status = "built"
            else:
                idx_status = "cached ✓"

        st.session_state.update({
            "docs": docs,
            "combined_text": "\n\n".join(d["text"] for d in docs),
            "idx": idx,
            "idx_info": {
                "num_docs": len(idx.doc_names),
                "num_chunks": idx.num_chunks,
                "status": idx_status,
            },
        })

# --- Status bar ---
if info := st.session_state.get("idx_info"):
    st.caption(
        f"📦 {info['num_docs']} doc(s) | {info['num_chunks']} units | {info['status']}"
    )

idx: Index | None = st.session_state.get("idx")
docs = st.session_state.get("docs", [])
combined_text = st.session_state.get("combined_text", "")

# --- Q&A ---
if combined_text:
    st.caption(
        f"📄 {', '.join(d['name'] for d in docs)} ({len(combined_text):,} chars)"
    )
    st.text_area("Preview", docs[0]["text"], height=180, disabled=True)

    with st.form("qa", clear_on_submit=False):
        question = st.text_input("Ask a question:")
        ask_clicked = st.form_submit_button("Get Answer")

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
        elif not idx:
            st.warning("Load a document first.")
        else:
            with st.spinner("Searching..."):
                results = idx.search(question, k=3)

            if not results:
                st.warning("No relevant content found.")
            else:
                context_chunks = [r["text"] for r in results]

                # --- LLM answer ---
                try:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(question, context_chunks)
                    st.markdown("### Answer")
                    st.success(answer)
                except EnvironmentError as e:
                    st.info(str(e))
                except Exception as e:
                    st.error(f"LLM error: {e}")

                # --- Supporting chunks ---
                top = results[0]
                with st.expander(
                    f"Sources — top relevance: {int(top['score'] * 100)}%",
                    expanded=False,
                ):
                    for r in results:
                        st.write(r["text"])
                        st.caption(
                            f"{r['doc_name']} | Relevance: {int(r['score'] * 100)}%"
                        )
                        st.divider()
else:
    st.info("Upload a file or paste text, then click Load.")
