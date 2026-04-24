import streamlit as st
from file_parser import parse_file
from ai_engine import answer_question

st.title("Document Q&A Chatbot")

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

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding answer..."):
                answer, confidence = answer_question(doc_text, question)
            st.success(f"**Answer:** {answer}")
            if confidence > 0:
                st.progress(confidence, text=f"Confidence: {int(confidence * 100)}%")
            else:
                st.warning("Low relevance — try rephrasing your question.")
else:
    st.info("Upload a file or paste text, then click Load.")
