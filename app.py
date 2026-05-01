import streamlit as st
from file_parser import parse_file
from ai_engine import answer_question
from chunker import chunk_text
from faiss_index import FAISSIndex

st.title("Jay — AI Doc Q&A Chatbot")

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
        combined_text = "\n\n".join(d["text"] for d in docs)
        st.session_state["docs"] = docs
        st.session_state["combined_text"] = combined_text

        # Pre-compute chunks once for both indexes
        doc_chunks = [
            (d["name"], [c.text for c in chunk_text(d["text"], d["name"])])
            for d in docs
        ]
        cache_key = FAISSIndex.multi_hash([d["text"] for d in docs])

        # Build GloVe FAISS index
        with st.spinner("Building GloVe index..."):
            glove_idx = FAISSIndex(model_type="glove")
            if not glove_idx.load(cache_key):
                glove_idx.build_multi(doc_chunks)
                glove_idx.save(cache_key)
                glove_status = "built"
            else:
                glove_status = "cached ✓"

        # Build SBERT index
        with st.spinner("Building SBERT index (first run downloads ~90MB)..."):
            sbert_idx = FAISSIndex(model_type="sbert")
            if not sbert_idx.load(cache_key):
                sbert_idx.build_multi(doc_chunks)
                sbert_idx.save(cache_key)
                sbert_status = "built"
            else:
                sbert_status = "cached ✓"

        st.session_state["glove_idx"] = glove_idx
        st.session_state["sbert_idx"] = sbert_idx
        st.session_state["index_info"] = {
            "num_docs": len(glove_idx.doc_names),
            "num_chunks": glove_idx.num_chunks,
            "glove": glove_status,
            "sbert": sbert_status,
        }

# --- index status bar ---
if "index_info" in st.session_state:
    info = st.session_state["index_info"]
    st.caption(
        f"📦 {info['num_docs']} doc(s) | {info['num_chunks']} chunks | "
        f"GloVe: {info['glove']} | SBERT: {info['sbert']}"
    )

combined_text = st.session_state.get("combined_text", "")
glove_idx: FAISSIndex | None = st.session_state.get("glove_idx")
sbert_idx: FAISSIndex | None = st.session_state.get("sbert_idx")


def _render_faiss_result(results: list, label: str) -> None:
    """Shared renderer for FAISS/SBERT results."""
    if not results:
        st.warning("No result.")
        return
    top_text, top_doc, top_score = results[0]
    st.success(top_text)
    st.progress(min(top_score, 1.0), text=f"Score: {top_score:.4f}")
    st.caption(f"Source: **{top_doc}**")
    if len(results) > 1:
        with st.expander(f"Neighbor chunks ({len(results) - 1})"):
            for nb_text, nb_doc, nb_score in results[1:]:
                st.write(nb_text)
                st.caption(f"{nb_doc} | {nb_score:.4f}")
                st.divider()


if combined_text:
    docs = st.session_state.get("docs", [])
    st.caption(
        f"📄 Loaded: **{', '.join(d['name'] for d in docs)}** ({len(combined_text):,} characters)"
    )
    st.text_area("Document Preview", docs[0]["text"], height=200, disabled=True)

    question = st.text_input("Ask a question about the document:")
    method = st.radio(
        "Search method:",
        ["word2vec", "faiss", "sbert"],
        horizontal=True,
        help="Word2Vec = GloVe sentence avg | FAISS = GloVe + vector index | SBERT = sentence embeddings",
    )
    compare = st.checkbox("Compare all methods side by side")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")

        elif compare:
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**🧠 Word2Vec**")
                with st.spinner("Loading model..."):
                    ans, conf, dbg = answer_question(combined_text, question)
                st.success(ans)
                if conf > 0:
                    st.progress(conf, text=f"Confidence: {int(conf * 100)}%")
                st.caption(f"Cosine: {dbg.get('raw_score', 0):.4f}")

            with col_b:
                st.markdown("**⚡ FAISS (GloVe)**")
                if glove_idx:
                    results = glove_idx.search(question, k=2)
                    _render_faiss_result(results, "FAISS")
                else:
                    st.warning("Load a document first.")

            with col_c:
                st.markdown("**🚀 SBERT**")
                if sbert_idx:
                    with st.spinner("Searching..."):
                        results = sbert_idx.search(question, k=2)
                    _render_faiss_result(results, "SBERT")
                else:
                    st.warning("Load a document first.")

        else:
            if method == "word2vec":
                with st.spinner("Finding answer..."):
                    answer, confidence, debug = answer_question(combined_text, question)
                st.success(f"**Answer:** {answer}")
                if confidence > 0:
                    st.progress(confidence, text=f"Confidence: {int(confidence * 100)}%")
                else:
                    st.warning("Low relevance — try rephrasing.")
                st.caption(f"Method: word2vec | Score: {debug.get('raw_score', 0):.4f}")

            elif method == "faiss":
                if glove_idx:
                    with st.spinner("Searching..."):
                        results = glove_idx.search(question, k=3)
                    st.markdown("**⚡ FAISS (GloVe)**")
                    _render_faiss_result(results, "FAISS")
                else:
                    st.warning("Load a document first.")

            elif method == "sbert":
                if sbert_idx:
                    with st.spinner("Searching..."):
                        results = sbert_idx.search(question, k=3)
                    st.markdown("**🚀 SBERT**")
                    _render_faiss_result(results, "SBERT")
                else:
                    st.warning("Load a document first.")
else:
    st.info("Upload a file or paste text, then click Load.")
