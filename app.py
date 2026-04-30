import streamlit as st
from file_parser import parse_file
from ai_engine import answer_question

st.title("Jay — AI Doc Q&A Chatbot")

col1, col_sep, col2 = st.columns([5, 1, 5])

with col1:
    uploaded = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

with col_sep:
    st.markdown("<div style='text-align:center; padding-top:40px; font-weight:bold;'>OR</div>", unsafe_allow_html=True)

with col2:
    pasted_text = st.text_area("Paste your text here", height=150)

_, btn_col, _ = st.columns([4, 2, 4])
with btn_col:
    load_clicked = st.button("Load", use_container_width=True)

if load_clicked:
    if uploaded:
        try:
            st.session_state["doc_text"] = parse_file(uploaded)
            st.session_state["doc_name"] = uploaded.name
        except ValueError as e:
            st.error(str(e))
    elif pasted_text.strip():
        st.session_state["doc_text"] = pasted_text.strip()
        st.session_state["doc_name"] = "Pasted text"
    else:
        st.error("Upload a file or paste text first.")

doc_text = st.session_state.get("doc_text", "")

if doc_text:
    doc_name = st.session_state.get("doc_name", "Document")
    st.caption(f"📄 Loaded: **{doc_name}** ({len(doc_text):,} characters)")
    st.text_area("Document Preview", doc_text, height=200, disabled=True)
    question = st.text_input("Ask a question about the document:")
    method = st.radio("Search method:", ["bow", "bm25", "word2vec", "faiss"], horizontal=True,
                      help="BoW = Bag of Words | BM25 = Elasticsearch | Word2Vec = Embeddings | FAISS = Vector index (fast for long docs)")

    compare = st.checkbox("Compare all methods side by side")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif compare:
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown("**🔤 BM25**")
                ans_t, conf_t, dbg_t = answer_question(doc_text, question, method="bm25")
                st.success(ans_t)
                if conf_t > 0:
                    st.progress(conf_t, text=f"Confidence: {int(conf_t * 100)}%")
                st.caption(f"Score: {dbg_t.get('raw_score', 0):.4f}")
            with col_b:
                st.markdown("**📐 Bag of Words**")
                ans_b, conf_b, dbg_b = answer_question(doc_text, question, method="bow")
                st.success(ans_b)
                if conf_b > 0:
                    st.progress(conf_b, text=f"Confidence: {int(conf_b * 100)}%")
                st.caption(f"Cosine: {dbg_b.get('raw_score', 0):.4f}")
            with col_c:
                st.markdown("**🧠 Word2Vec**")
                with st.spinner("Loading model..."):
                    ans_w, conf_w, dbg_w = answer_question(doc_text, question, method="word2vec")
                st.success(ans_w)
                if conf_w > 0:
                    st.progress(conf_w, text=f"Confidence: {int(conf_w * 100)}%")
                st.caption(f"Cosine: {dbg_w.get('raw_score', 0):.4f}")
            with col_d:
                st.markdown("**⚡ FAISS**")
                with st.spinner("Building index..."):
                    ans_f, conf_f, dbg_f = answer_question(doc_text, question, method="faiss")
                st.success(ans_f)
                if conf_f > 0:
                    st.progress(conf_f, text=f"Confidence: {int(conf_f * 100)}%")
                st.caption(f"Score: {dbg_f.get('score', 0):.4f}")
        else:
            with st.spinner("Finding answer..."):
                answer, confidence, debug = answer_question(doc_text, question, method=method)
            st.success(f"**Answer:** {answer}")
            if confidence > 0:
                st.progress(confidence, text=f"Confidence: {int(confidence * 100)}%")
            else:
                st.warning("Low relevance — try rephrasing your question.")
            st.caption(f"Method: {debug.get('method', '')} | Score: {debug.get('raw_score', debug.get('score', 0)):.4f}")
else:
    st.info("Upload a file or paste text, then click Load.")
