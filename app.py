import re
import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

st.set_page_config(
    page_title="Jay — AI Doc Q&A",
    page_icon="🤖",
    initial_sidebar_state="collapsed",
)

_root = Path(__file__).resolve().parent
load_dotenv(_root / ".env.test")
load_dotenv(_root / ".env")

from core.parser import parse_file
from core.chunker import chunk_text
from core.index import Index
from core.hybrid import HybridRetriever
from core.llm import generate_answer


# ── Markdown → HTML for AI bubble rendering ───────────────────────────────────
def _md_to_html(text: str) -> str:
    try:
        import markdown as _md
        return _md.markdown(
            text,
            extensions=["fenced_code", "tables", "nl2br"],
        )
    except Exception:
        return html.escape(text).replace("\n", "<br>")


# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Page */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #f0f2f8 !important;
}
.block-container {
    max-width: 800px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 5rem !important;
}

/* ── Title ── */
.jay-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.jay-logo-icon {
    width: 44px; height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #4f46e5 0%, #a855f7 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 14px rgba(79,70,229,0.4);
}
.jay-logo-text {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.8px;
    background: linear-gradient(135deg, #4f46e5, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.jay-tagline {
    font-size: 13px;
    color: #94a3b8;
    margin: 0 0 24px 56px;
    font-weight: 400;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: white !important;
    border: 1.5px dashed #c7d2fe !important;
    border-radius: 14px !important;
    padding: 6px 10px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
}
[data-testid="stFileUploader"] label { color: #374151 !important; font-weight: 500 !important; }
[data-testid="stFileUploader"] button {
    border-radius: 8px !important;
    border-color: #e0e7ff !important;
    color: #4f46e5 !important;
    font-weight: 500 !important;
}

/* Paste textarea */
[data-testid="stTextArea"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #e0e7ff !important;
    background: white !important;
    font-size: 14px !important;
    color: #374151 !important;
    transition: border-color 0.2s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* OR separator */
.or-sep {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding-top: 44px;
}
.or-pill {
    background: #e0e7ff;
    color: #4f46e5;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ── Load button ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 0.55rem 2rem !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.38) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 7px 22px rgba(79,70,229,0.48) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
.chat-divider {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 32px 0 22px;
}
.chat-divider-line { flex: 1; height: 1px; background: #dde3f0; }
.chat-divider-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #94a3b8;
    white-space: nowrap;
}

/* ── Status bar ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 13px;
    color: #475569;
    margin-bottom: 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.2);
}
.status-tag {
    background: #ede9fe;
    color: #6d28d9;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    margin-left: auto;
}

/* ── Chat message bubbles ── */
.chat-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 6px;
    gap: 10px;
}
.chat-row.user { justify-content: flex-end; }
.chat-row.ai   { justify-content: flex-start; }

.ai-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
    margin-top: 3px;
    box-shadow: 0 2px 8px rgba(79,70,229,0.35);
}

.bubble-user {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    border-radius: 20px 20px 4px 20px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 15px;
    line-height: 1.65;
    box-shadow: 0 4px 18px rgba(79,70,229,0.3);
    word-wrap: break-word;
}

.bubble-ai {
    background: white;
    border: 1px solid #e8edf5;
    border-radius: 4px 20px 20px 20px;
    padding: 14px 20px;
    max-width: 78%;
    font-size: 15px;
    line-height: 1.72;
    color: #1e293b;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    word-wrap: break-word;
}
.bubble-ai p  { margin: 0 0 10px; }
.bubble-ai p:last-child { margin: 0; }
.bubble-ai ul, .bubble-ai ol { padding-left: 18px; margin: 8px 0; }
.bubble-ai li { margin-bottom: 4px; }
.bubble-ai strong { color: #1e1b4b; font-weight: 600; }
.bubble-ai code {
    background: #f0f2f8;
    color: #4f46e5;
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 13px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
.bubble-ai pre {
    background: #0f172a;
    border-radius: 10px;
    padding: 14px 16px;
    overflow-x: auto;
    margin: 10px 0;
}
.bubble-ai pre code {
    background: none !important;
    color: #e2e8f0;
    padding: 0;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-radius: 26px !important;
    border: 2px solid #6366f1 !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.16) !important;
    overflow: hidden;
    background: white !important;
    transition: box-shadow 0.2s, border-color 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4f46e5 !important;
    box-shadow: 0 6px 30px rgba(99,102,241,0.26) !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    color: #1e293b !important;
    background: white !important;
}

/* ── Sources expander ── */
details[data-testid="stExpander"] {
    background: #fafbff !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 12px !important;
    overflow: hidden;
    margin: 4px 0 16px 42px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 52px 0 32px;
    color: #94a3b8;
}
.empty-state .icon { font-size: 46px; margin-bottom: 14px; }
.empty-state h3 { font-size: 17px; font-weight: 600; color: #64748b; margin: 0 0 6px; }
.empty-state p  { font-size: 13px; margin: 0; line-height: 1.65; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def _user_bubble(content: str) -> None:
    safe = html.escape(content)
    st.markdown(
        f'<div class="chat-row user"><div class="bubble-user">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def _ai_bubble(content: str) -> None:
    body = _md_to_html(content)
    st.markdown(f"""
    <div class="chat-row ai">
      <div class="ai-avatar">🤖</div>
      <div class="bubble-ai">{body}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jay-logo">
  <div class="jay-logo-icon">🤖</div>
  <div class="jay-logo-text">Jay</div>
</div>
<div class="jay-tagline">AI-powered doc Q&amp;A · Upload, ask, understand.</div>
""", unsafe_allow_html=True)


# ── Upload section (original layout kept exactly) ────────────────────────────
col1, col_sep, col2 = st.columns([5, 1, 5])

with col1:
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

with col_sep:
    st.markdown(
        '<div class="or-sep"><div class="or-pill">OR</div></div>',
        unsafe_allow_html=True,
    )

with col2:
    pasted_text = st.text_area("Paste your text here", height=150)

_, btn_col, _ = st.columns([4, 2, 4])
with btn_col:
    load_clicked = st.button("⚡ Load", use_container_width=True)


# ── Load logic ────────────────────────────────────────────────────────────────
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

        with st.spinner("Building index… (first run downloads ~90 MB)"):
            idx = Index()
            if not idx.load(cache_key):
                idx.build(doc_chunks)
                idx.save(cache_key)
                idx_status = "built"
            else:
                idx_status = "cached ✓"

        with st.spinner("Building BM25 index…"):
            retriever = HybridRetriever(idx)
            retriever.build_bm25()

        st.session_state.update({
            "docs": docs,
            "combined_text": "\n\n".join(d["text"] for d in docs),
            "idx": idx,
            "retriever": retriever,
            "idx_info": {
                "num_docs": len(idx.doc_names),
                "num_chunks": idx.num_chunks,
                "status": idx_status,
            },
            "messages": [],
            "_scroll_to_chat": True,
        })
        st.rerun()


# ── State ─────────────────────────────────────────────────────────────────────
idx: Index | None = st.session_state.get("idx")
retriever: HybridRetriever | None = st.session_state.get("retriever")
docs = st.session_state.get("docs", [])
combined_text = st.session_state.get("combined_text", "")


# ── Chat section ──────────────────────────────────────────────────────────────
if combined_text and idx and retriever:

    # auto-scroll to here after fresh load
    if st.session_state.get("_scroll_to_chat"):
        st.session_state._scroll_to_chat = False
        components.html("""<script>
        setTimeout(function() {
            var main = window.parent.document.querySelector('section.main')
                    || window.parent.document.querySelector('.main');
            if (main) main.scrollTo({top: 99999, behavior: 'smooth'});
        }, 300);
        </script>""", height=0)

    # section divider
    info = st.session_state.get("idx_info", {})
    doc_names = ", ".join(d["name"] for d in docs)
    st.markdown(f"""
    <div class="chat-divider">
      <div class="chat-divider-line"></div>
      <div class="chat-divider-label">Conversation</div>
      <div class="chat-divider-line"></div>
    </div>
    <div class="status-bar">
      <div class="status-dot"></div>
      <span><strong>{html.escape(doc_names)}</strong>
        &nbsp;·&nbsp; {info.get('num_docs', 0)} doc(s)
        &nbsp;·&nbsp; {info.get('num_chunks', 0)} chunks
      </span>
      <div class="status-tag">{info.get('status', 'ready')}</div>
    </div>
    """, unsafe_allow_html=True)

    # clear button — right-aligned, minimal
    _, clr = st.columns([9, 1])
    with clr:
        if st.button("✕", help="Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # render history + sources stored alongside messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            _user_bubble(msg["content"])
        else:
            _ai_bubble(msg["content"])
            # render stored sources for this turn if available
            if msg.get("sources"):
                top_score = int(msg["sources"][0]["score"] * 100)
                with st.expander(f"📚 Sources — top match {top_score}%", expanded=False):
                    for r in msg["sources"]:
                        st.markdown(f"> {r['text']}")
                        st.caption(
                            f"**{r['doc_name']}** · {r.get('source', '')} · "
                            f"RRF {r['score']:.4f}"
                        )
                        st.divider()

    # new question
    if question := st.chat_input("Ask anything about your document…"):
        st.session_state.messages.append({"role": "user", "content": question})
        _user_bubble(question)

        with st.spinner("Thinking…"):
            results = retriever.search(question, k=5)

            if not results:
                answer = "I couldn't find relevant content in the document for that question."
                sources = []
            else:
                context_chunks = [r["text"] for r in results]
                sources = results
                try:
                    answer = generate_answer(
                        question,
                        context_chunks,
                        history=st.session_state.messages[:-1],
                    )
                except EnvironmentError as e:
                    answer = str(e)
                    sources = []
                except Exception as e:
                    answer = f"LLM error: {e}"
                    sources = []

        _ai_bubble(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        if sources:
            top_score = int(sources[0]["score"] * 100)
            with st.expander(f"📚 Sources — top match {top_score}%", expanded=False):
                for r in sources:
                    st.markdown(f"> {r['text']}")
                    st.caption(
                        f"**{r['doc_name']}** · {r.get('source', '')} · "
                        f"RRF {r['score']:.4f}"
                    )
                    st.divider()

else:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">📄</div>
      <h3>No document loaded yet</h3>
      <p>Upload a PDF, DOCX, or TXT above — or paste text —<br>then click <strong>Load</strong> to start the conversation.</p>
    </div>
    """, unsafe_allow_html=True)
