import base64
import html
from pathlib import Path

from PIL import Image as _PILImage
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ── Paths & assets ────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
_assets = _root / "assets"
_sift_icon_path = _assets / "sift_icon.png"
_sift_icon = _PILImage.open(_sift_icon_path)


def _asset_data_uri(filename: str) -> str:
    raw = (_assets / filename).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"


_ICON_DATA_URI = _asset_data_uri("sift_icon.png")
_LOGO_DATA_URI = _asset_data_uri("sift_logo.png")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sift — ask questions, get grounded answers",
    page_icon=_sift_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_dotenv(_root / ".env.test")
load_dotenv(_root / ".env")

from core.parser import parse_file
from core.sentence_window import SentenceWindowRetriever
from core.llm import generate_answer

# streamlit-extras: a real wrapper around widgets (raw HTML divs can't contain them)
try:
    from streamlit_extras.stylable_container import stylable_container
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def stylable_container(key, css_styles):  # graceful fallback
        yield


def _md_to_html(text: str) -> str:
    try:
        import markdown as _md
        return _md.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    except Exception:
        return html.escape(text).replace("\n", "<br>")


# ── CSS ───────────────────────────────────────────────────────────────────────
# Palette — locked, used everywhere:
#   ink     #0F1117   navy text / dark surfaces (matches logo background)
#   paper   #FAF7F0   page background (cream)
#   paper2  #F2EDDF   inset surfaces
#   line    #E8E2D2   borders
#   amber   #F5A524   primary accent (matches logo bars)
#   amber-d #D88A0E   accent hover
#   cream   #FFFDF7   cards
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink: #0F1117;
  --ink-2: #1B1F2A;
  --paper: #FAF7F0;
  --paper-2: #F2EDDF;
  --line: #E8E2D2;
  --line-2: #D9D2BD;
  --amber: #F5A524;
  --amber-d: #D88A0E;
  --amber-soft: #FCEBC4;
  --cream: #FFFDF7;
  --muted: #6B6F7C;
  --muted-2: #8B8F9C;
  --ok: #2F8F5C;
}

html, body, [class*="css"], [class*="st-"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--ink);
}

/* ── Page surface ───────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp {
  background:
    radial-gradient(ellipse at 50% -10%, rgba(245,165,36,0.06) 0%, transparent 55%),
    var(--paper) !important;
}
.block-container {
  max-width: 880px !important;
  padding-top: 2.4rem !important;
  padding-bottom: 6rem !important;
}

/* ── Top accent rail (single warm hairline) ─────────────────────────────── */
.top-rail {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent 0%, var(--amber) 35%, var(--amber-d) 65%, transparent 100%);
  z-index: 9999;
  opacity: 0.85;
}

/* ── Brand header ───────────────────────────────────────────────────────── */
.sift-head { display: flex; align-items: flex-start; gap: 18px; margin: 4px 0 6px; }
.sift-head img.logo { height: 64px; width: auto; }
.lede {
  font-size: 14px;
  color: var(--muted);
  line-height: 1.55;
  max-width: 640px;
  margin: 6px 0 0 0;
  font-weight: 400;
}
.lede b { color: var(--ink); font-weight: 600; }
.sift-chips { display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; }
.sift-chip {
  background: var(--cream);
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.3px;
}
.sift-chip.warm { background: var(--amber-soft); border-color: #f3d98c; color: #6a4a07; }
.sift-chip.dim  { color: var(--muted); }

/* ── Section caption (small uppercase) ──────────────────────────────────── */
.sec-cap {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: var(--muted-2);
  margin: 22px 0 10px 2px;
}

/* ──────────────────────────────────────────────────────────────────────────
   FILE UPLOADER — single border, no nested rectangles.
   Strip the outer wrapper border, style only the inner dropzone.
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
  background: var(--paper-2) !important;
  border: 1.5px dashed var(--line-2) !important;
  border-radius: 12px !important;
  padding: 18px 16px !important;
  transition: border-color 0.18s ease, background 0.18s ease;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--amber) !important;
  background: var(--amber-soft) !important;
}
[data-testid="stFileUploader"] label {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--muted) !important;
}
[data-testid="stFileUploader"] button {
  background: var(--cream) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 12.5px !important;
  padding: 6px 14px !important;
  box-shadow: none !important;
}
[data-testid="stFileUploader"] button:hover {
  border-color: var(--amber) !important;
  color: var(--amber-d) !important;
  background: #FFF8E6 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
  background: var(--cream) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  margin-top: 8px !important;
}

/* ──────────────────────────────────────────────────────────────────────────
   TEXT AREA — same logic. One border, no doubles.
   Outer wrapper: no border. Inner BaseWeb input: the only border.
   ────────────────────────────────────────────────────────────────────────── */
[data-testid="stTextArea"] { background: transparent !important; }
[data-testid="stTextArea"] label {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
}
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextArea"] [data-baseweb="base-input"] {
  background: var(--paper-2) !important;
  border: 1.5px solid var(--line) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  transition: border-color 0.18s ease, background 0.18s ease;
}
[data-testid="stTextArea"] textarea {
  background: transparent !important;
  border: none !important;
  outline: none !important;
  color: var(--ink) !important;
  font-size: 14px !important;
  line-height: 1.6 !important;
  padding: 12px 14px !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
[data-testid="stTextArea"] [data-baseweb="base-input"]:focus-within {
  border-color: var(--amber) !important;
  background: var(--cream) !important;
  box-shadow: 0 0 0 3px rgba(245,165,36,0.16) !important;
}

/* ── OR separator ───────────────────────────────────────────────────────── */
.or-sep {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding-top: 42px;
}
.or-pill {
  background: var(--paper);
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 5px 11px;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 1px;
}

/* ── Primary button (Load) ─────────────────────────────────────────────── */
.stButton > button {
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: 1px solid var(--ink) !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-size: 14.5px !important;
  padding: 0.6rem 1.6rem !important;
  letter-spacing: 0.15px;
  box-shadow: 0 2px 0 rgba(15,17,23,0.06) !important;
  transition: transform 0.08s ease, background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
}
.stButton > button:hover {
  background: var(--amber) !important;
  border-color: var(--amber) !important;
  color: var(--ink) !important;
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button:focus { box-shadow: 0 0 0 3px rgba(245,165,36,0.30) !important; }

/* small clear-history button */
.clear-btn .stButton > button {
  background: transparent !important;
  border: 1px solid var(--line) !important;
  color: var(--muted) !important;
  padding: 4px 10px !important;
  font-size: 12px !important;
  border-radius: 8px !important;
}
.clear-btn .stButton > button:hover {
  border-color: var(--amber) !important;
  color: var(--amber-d) !important;
  background: var(--amber-soft) !important;
}

/* ── Conversation rule ─────────────────────────────────────────────────── */
.conv-rule {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 30px 0 18px;
}
.conv-rule .line { flex: 1; height: 1px; background: var(--line); }
.conv-rule .lbl {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 1.3px;
  text-transform: uppercase;
  color: var(--muted-2);
}

/* ── Status bar ─────────────────────────────────────────────────────────── */
.status-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--cream);
  border: 1px solid var(--line);
  border-left: 3px solid var(--amber);
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px;
  color: var(--ink);
  margin-bottom: 14px;
}
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--ok);
  box-shadow: 0 0 0 3px rgba(47,143,92,0.15);
  animation: pulse-dot 2.4s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { box-shadow: 0 0 0 3px rgba(47,143,92,0.15); }
  50%     { box-shadow: 0 0 0 5px rgba(47,143,92,0.06); }
}
.status-name {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 300px;
}
.status-meta { color: var(--muted); font-size: 12.5px; }
.status-tag {
  background: var(--amber-soft);
  color: #6a4a07;
  border: 1px solid #f3d98c;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 600;
  margin-left: auto;
  white-space: nowrap;
}

/* ── Chat bubbles ───────────────────────────────────────────────────────── */
.chat-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.chat-row.user { justify-content: flex-end; }
.chat-row.ai   { justify-content: flex-start; }

.ai-avatar {
  width: 32px; height: 32px;
  border-radius: 9px;
  flex-shrink: 0;
  margin-top: 4px;
  overflow: hidden;
  background: var(--ink);
  border: 1px solid var(--line);
  box-shadow: 0 1px 2px rgba(15,17,23,0.05);
}
.ai-avatar img { width: 100%; height: 100%; object-fit: cover; display: block; }

.bubble-user {
  background: var(--ink);
  color: var(--paper);
  border-radius: 18px 18px 4px 18px;
  padding: 10px 16px;
  max-width: 72%;
  font-size: 14.5px;
  line-height: 1.6;
  box-shadow: 0 2px 6px rgba(15,17,23,0.08);
  word-wrap: break-word;
}

.bubble-ai {
  background: var(--cream);
  border: 1px solid var(--line);
  border-radius: 4px 18px 18px 18px;
  padding: 12px 16px;
  max-width: 78%;
  font-size: 14.5px;
  line-height: 1.7;
  color: var(--ink);
  box-shadow: 0 1px 2px rgba(15,17,23,0.04);
  word-wrap: break-word;
}
.bubble-ai p  { margin: 0 0 8px; }
.bubble-ai p:last-child { margin: 0; }
.bubble-ai ul, .bubble-ai ol { padding-left: 18px; margin: 6px 0; }
.bubble-ai li { margin-bottom: 3px; }
.bubble-ai strong { color: var(--ink); font-weight: 700; }
.bubble-ai code {
  background: var(--paper-2);
  color: #6a4a07;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 1px 6px;
  font-size: 12.5px;
  font-family: 'JetBrains Mono', monospace !important;
}
.bubble-ai pre {
  background: var(--ink);
  border-radius: 10px;
  padding: 12px 14px;
  overflow-x: auto;
  margin: 10px 0;
  border: 1px solid var(--ink-2);
}
.bubble-ai pre code {
  background: none !important;
  color: #f1ead8 !important;
  padding: 0;
  border: none;
  font-size: 12.5px;
}

/* ── Chat input ─────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: var(--cream) !important;
  border: 1.5px solid var(--line) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 3px rgba(15,17,23,0.04) !important;
  overflow: hidden;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 3px rgba(245,165,36,0.18) !important;
}
[data-testid="stChatInput"] textarea {
  font-size: 14.5px !important;
  color: var(--ink) !important;
  background: transparent !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: var(--amber-d) !important; }

/* ── Sources expander ──────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
  background: var(--paper-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  overflow: hidden;
  margin: 4px 0 16px 42px;
  box-shadow: none !important;
}
details[data-testid="stExpander"] summary {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
  padding: 9px 14px !important;
}
details[data-testid="stExpander"] summary:hover { color: var(--amber-d) !important; }

/* ── Alerts (used inside sources) ──────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  border-left: 3px solid var(--amber) !important;
  background: var(--cream) !important;
  font-size: 13.5px !important;
  color: var(--ink) !important;
}

/* ── Empty state ───────────────────────────────────────────────────────── */
.empty {
  text-align: center;
  padding: 56px 24px 30px;
  color: var(--muted);
  border: 1px dashed var(--line-2);
  border-radius: 18px;
  background: var(--cream);
  margin-top: 20px;
}
.empty .ico {
  margin: 0 auto 14px;
  width: 64px; height: 64px;
  background: var(--ink);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 14px rgba(15,17,23,0.10);
}
.empty .ico img { width: 38px; height: 38px; object-fit: contain; }
.empty h3 {
  font-size: 17px; font-weight: 700; color: var(--ink);
  margin: 0 0 6px; letter-spacing: -0.01em;
}
.empty p { font-size: 13.5px; margin: 0; line-height: 1.7; color: var(--muted); }
.empty .hints { display: flex; gap: 8px; justify-content: center; margin-top: 18px; flex-wrap: wrap; }
.empty .hint {
  background: var(--paper-2);
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 5px 12px;
  font-size: 12px;
  font-weight: 500;
}

/* ── Spinner ───────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
  border-color: var(--amber) !important;
  border-top-color: transparent !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--cream) !important;
  border-right: 1px solid var(--line) !important;
}

/* ── Hide Streamlit chrome we don't need ───────────────────────────────── */
#MainMenu, footer, header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Top accent rail ───────────────────────────────────────────────────────────
st.markdown('<div class="top-rail"></div>', unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
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
      <div class="ai-avatar"><img src="{_ICON_DATA_URI}" alt="" /></div>
      <div class="bubble-ai">{body}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="sift-head">
  <img class="logo" src="{_LOGO_DATA_URI}" alt="Sift" />
</div>
<p class="lede">
Drop in a <b>PDF, DOCX, or TXT</b> &mdash; or paste any text &mdash; then ask questions in plain English.
Every answer is grounded in <b>retrieved passages from your document</b>, with sources you can verify.
</p>
<div class="sift-chips">
  <span class="sift-chip warm">SBERT &middot; BM25 &middot; RRF</span>
  <span class="sift-chip">Sentence Window</span>
  <span class="sift-chip dim">Groq &middot; Llama-3</span>
</div>
""", unsafe_allow_html=True)


# ── Upload section ────────────────────────────────────────────────────────────
st.markdown('<p class="sec-cap">Your documents</p>', unsafe_allow_html=True)

_CARD_CSS = (
    "{ background: #FFFDF7; border: 1px solid #E8E2D2; border-radius: 16px; "
    "padding: 18px 18px 14px; box-shadow: 0 1px 2px rgba(15,17,23,0.03); "
    "margin-bottom: 14px; }"
)

with stylable_container(key="upload_card", css_styles=_CARD_CSS):
    col1, col_sep, col2 = st.columns([5, 1, 5], gap="small")

    with col1:
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="visible",
        )

    with col_sep:
        st.markdown(
            '<div class="or-sep"><div class="or-pill">OR</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        pasted_text = st.text_area(
            "Paste text",
            height=148,
            placeholder="Paste any text here — an article, a contract, your notes…",
        )

_, btn_col, _ = st.columns([4, 2, 4])
with btn_col:
    load_clicked = st.button("Load and index", use_container_width=True)


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
        raw_docs = [(d["name"], d["text"]) for d in docs]
        cache_key = SentenceWindowRetriever.content_hash(
            [f"{name}\n{text}" for name, text in raw_docs] + ["sw-v1"]
        )

        with st.spinner("Building index… (first run downloads ~90 MB)"):
            retriever = SentenceWindowRetriever()
            if not retriever.load(cache_key):
                retriever.build(raw_docs)
                retriever.save(cache_key)
                idx_status = "built"
            else:
                idx_status = "cached"

        st.session_state.update({
            "docs": docs,
            "combined_text": "\n\n".join(d["text"] for d in docs),
            "retriever": retriever,
            "idx_info": {
                "num_docs": len(retriever.doc_names),
                "num_chunks": retriever.num_sentences,
                "status": idx_status,
            },
            "messages": [],
            "_scroll_to_chat": True,
        })
        st.rerun()


# ── State ─────────────────────────────────────────────────────────────────────
retriever: SentenceWindowRetriever | None = st.session_state.get("retriever")
docs = st.session_state.get("docs", [])
combined_text = st.session_state.get("combined_text", "")


# ── Chat section ──────────────────────────────────────────────────────────────
if combined_text and retriever:

    if st.session_state.get("_scroll_to_chat"):
        st.session_state._scroll_to_chat = False
        components.html("""<script>
        setTimeout(function() {
            var main = window.parent.document.querySelector('section.main')
                    || window.parent.document.querySelector('.main');
            if (main) main.scrollTo({top: 99999, behavior: 'smooth'});
        }, 280);
        </script>""", height=0)

    info = st.session_state.get("idx_info", {})
    doc_names = ", ".join(d["name"] for d in docs)
    short_names = doc_names if len(doc_names) < 48 else doc_names[:45] + "…"

    st.markdown(f"""
    <div class="conv-rule">
      <div class="line"></div>
      <div class="lbl">Conversation</div>
      <div class="line"></div>
    </div>
    <div class="status-bar">
      <div class="status-dot"></div>
      <span class="status-name">{html.escape(short_names)}</span>
      <span class="status-meta">&nbsp;·&nbsp;{info.get('num_docs', 0)} doc &nbsp;·&nbsp; {info.get('num_chunks', 0):,} sentences</span>
      <div class="status-tag">{info.get('status', 'ready')}</div>
    </div>
    """, unsafe_allow_html=True)

    _, clr = st.columns([9, 1])
    with clr:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("Clear", help="Clear chat history"):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _user_bubble(msg["content"])
        else:
            _ai_bubble(msg["content"])
            if msg.get("sources"):
                best = msg["sources"][0]["score"] or 1
                with st.expander("Sources", expanded=False):
                    for j, r in enumerate(msg["sources"], 1):
                        pct = int((r["score"] / best) * 100)
                        matched = r.get("matched_sentence") or r.get("text", "")
                        st.caption(f"#{j} · **{r['doc_name']}** · {r.get('source', '')} · {pct}% relevance")
                        st.info(matched)

    if question := st.chat_input("Ask anything about your document…"):
        st.session_state.messages.append({"role": "user", "content": question})
        _user_bubble(question)

        with st.spinner("Thinking…"):
            results = retriever.search(question, k=3)

            if not results:
                answer = "I couldn't find relevant content in the document for that question."
                sources = []
            else:
                context_chunks = [r["text"][:600] for r in results]
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
            best = sources[0]["score"] or 1
            with st.expander("Sources", expanded=False):
                for r in sources:
                    pct = int((r["score"] / best) * 100)
                    matched = r.get("matched_sentence") or r.get("text", "")
                    st.caption(
                        f"**{r['doc_name']}** · {r.get('source', '')} · "
                        f"relevance {pct}%"
                    )
                    st.info(matched)

else:
    st.markdown(f"""
    <div class="empty">
      <div class="ico"><img src="{_ICON_DATA_URI}" alt="" /></div>
      <h3>No document loaded yet</h3>
      <p>Upload a PDF, DOCX, or TXT — or paste text — then click <b>Load and index</b>.<br>
      Sift will read it once and let you ask questions afterwards.</p>
      <div class="hints">
        <span class="hint">Research papers</span>
        <span class="hint">Contracts</span>
        <span class="hint">Meeting notes</span>
        <span class="hint">Long articles</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
