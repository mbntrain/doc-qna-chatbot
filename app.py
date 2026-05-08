import base64
import html
from pathlib import Path

from PIL import Image as _PILImage
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ── Paths & assets ────────────────────────────────────────────────────────────
_root   = Path(__file__).resolve().parent
_assets = _root / "assets"
_sift_icon = _PILImage.open(_assets / "sift_icon.png")


def _asset_b64(filename: str, mime: str = "png") -> str:
    raw = (_assets / filename).read_bytes()
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode()}"


_ICON_URI    = _asset_b64("sift_icon.png")
_LOGO_URI    = _asset_b64("sift_logo.png")
_LOADING_URI = _asset_b64("sift_loading.gif", "gif")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sift — ask questions, get grounded answers",
    page_icon=_sift_icon,
    layout="centered",
    initial_sidebar_state="collapsed",
)
load_dotenv(_root / ".env.test")
load_dotenv(_root / ".env")

from core.parser import parse_file
from core.sentence_window import SentenceWindowRetriever
from core.llm import generate_answer


def _md_to_html(text: str) -> str:
    try:
        import markdown as _md
        return _md.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    except Exception:
        return html.escape(text).replace("\n", "<br>")


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Manrope:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ─────────────────────────────────────────────────────────────── */
:root {
  --bg:           #faf9f6;
  --border:       #f1f5f9;
  --border-2:     #e2e8f0;
  --primary:      #221a12;
  --secondary:    #64748b;
  --muted:        #94a3b8;
  --amber:        #f59e0b;
  --amber-dark:   #855300;
  --amber-soft:   rgba(255,221,184,0.5);
  --amber-border: rgba(255,185,95,0.3);
  --card:         #ffffff;
  --chip-bg:      #f1f5f9;
  --chip-text:    #475569;
  --ok:           #2f8f5c;
  --nav-active:   #fffbeb;
  --nav-text:     #b45309;
}

/* Safer global styling: avoid broad [class*="css"] / [class*="st-"] selectors */
html, body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--primary);
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--primary);
}

/* ── Page & container ────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp { background: var(--bg) !important; }

.block-container {
  max-width: 780px !important;
  padding-top: 0 !important;
  padding-bottom: 7rem !important;
}

/* Hide sidebar toggle completely */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
#MainMenu  { visibility: hidden !important; }
footer     { visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Main header (when doc loaded) ──────────────────────────────────────── */
.main-header {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 18px 0 16px;
  border-bottom: 1px solid var(--border-2);
  margin-bottom: 28px;
  background: var(--bg);
  position: sticky;
  top: 0;
  z-index: 100;
}
.mh-logo {
  display: flex;
  align-items: center;
  gap: 8px;
}
.mh-logo img { width: 32px; height: 32px; display: block; }
.mh-logo-name {
  font-family: 'Manrope', sans-serif;
  font-size: 20px;
  font-weight: 700;
  color: #0f172a;
  letter-spacing: -1px;
  line-height: 1;
}
.mh-divider {
  width: 1px; height: 16px;
  background: #cbd5e1;
  margin: 0 14px;
  flex-shrink: 0;
}
.mh-doc {
  font-size: 14px;
  color: var(--secondary);
  letter-spacing: -0.3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Welcome state (doc loaded, no messages yet) ─────────────────────────── */
.welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 56px 0 40px;
  text-align: center;
}
.welcome-icon {
  width: 64px; height: 64px;
  background: #fff;
  border: 1px solid var(--border-2);
  border-radius: 16px;
  box-shadow: 0 1px 1px rgba(0,0,0,0.05);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 16px;
}
.welcome-icon img { width: 40px; height: 40px; display: block; }
.welcome h2 {
  font-family: 'Manrope', sans-serif;
  font-size: 24px;
  font-weight: 600;
  color: #221a12;
  letter-spacing: -0.3px;
  margin: 0 0 12px;
  line-height: 1.4;
}
.welcome p {
  font-size: 16px;
  color: var(--secondary);
  line-height: 1.6;
  max-width: 520px;
  margin: 0;
}

/* ── Upload welcome (no doc) ─────────────────────────────────────────────── */
.upload-welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 48px 0 28px;
  text-align: center;
}
.upload-welcome-icon {
  width: 64px; height: 64px;
  background: #0f172a;
  border-radius: 16px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 16px;
  box-shadow: 0 4px 14px rgba(15,17,23,0.12);
}
.upload-welcome-icon img { width: 40px; height: 40px; display: block; }
.upload-welcome h2 {
  font-family: 'Manrope', sans-serif;
  font-size: 22px;
  font-weight: 700;
  color: #221a12;
  margin: 0 0 8px;
  letter-spacing: -0.3px;
}
.upload-welcome p {
  font-size: 14px;
  color: var(--secondary);
  line-height: 1.65;
  max-width: 460px;
  margin: 0 0 28px;
}

/* ── Chat rows ───────────────────────────────────────────────────────────── */
.chat-row { display: flex; align-items: flex-start; margin-bottom: 28px; }
.chat-row.user { justify-content: flex-end; }
.chat-row.ai   { justify-content: flex-start; }

/* ── User bubble ─────────────────────────────────────────────────────────── */
.bubble-user {
  background: var(--amber-soft);
  border: 1px solid var(--amber-border);
  color: #221a12;
  border-radius: 16px;
  padding: 14px 22px;
  width: fit-content;
  max-width: 78%;
  font-size: 15px;
  line-height: 1.6;
  word-wrap: break-word;
}

/* ── AI card ─────────────────────────────────────────────────────────────── */
.ai-card {
  background: var(--card);
  border: 1px solid var(--border-2);
  border-radius: 16px;
  box-shadow: 0 1px 1px rgba(0,0,0,0.05);
  max-width: 70%;
  overflow: hidden;
}
.ai-card-body {
  padding: 20px 22px 14px;
  font-size: 15px;
  line-height: 1.65;
  color: #221a12;
}
.ai-card-body p  { margin: 0 0 10px; }
.ai-card-body p:last-child { margin: 0; }
.ai-card-body ul, .ai-card-body ol { padding-left: 20px; margin: 6px 0 10px; }
.ai-card-body li { margin-bottom: 6px; line-height: 1.6; }
.ai-card-body li::marker { color: var(--amber); }
.ai-card-body strong { color: #221a12; font-weight: 600; }
.ai-card-body code {
  background: #f8fafc;
  color: #b45309;
  border: 1px solid var(--border-2);
  border-radius: 5px;
  padding: 1px 6px;
  font-size: 12.5px;
  font-family: 'JetBrains Mono', monospace !important;
}
.ai-card-body pre {
  background: #1e293b;
  border-radius: 10px;
  padding: 12px 14px;
  overflow-x: auto;
  margin: 10px 0;
}
.ai-card-body pre code {
  background: none !important;
  color: #f1f5f9 !important;
  padding: 0; border: none;
  font-size: 12.5px;
}

/* AI card sources strip */
.ai-card-sources {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 22px 16px;
  border-top: 1px solid var(--border);
  flex-wrap: wrap;
}
.sources-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
  flex-shrink: 0;
}
.source-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.source-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--chip-bg);
  color: var(--chip-text);
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 11.5px;
  font-weight: 600;
  white-space: nowrap;
}

/* ── Thinking indicator ──────────────────────────────────────────────────── */
.thinking {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 20px 0 8px;
}
.thinking img { width: 22px; height: 22px; opacity: 0.7; }
.thinking span {
  font-size: 12.5px;
  font-weight: 700;
  color: var(--secondary);
  letter-spacing: 0.8px;
  text-transform: uppercase;
  animation: blink 1.5s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Status bar ──────────────────────────────────────────────────────────── */
.status-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--card);
  border: 1px solid var(--border-2);
  border-left: 3px solid var(--amber);
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px;
  margin-bottom: 20px;
}
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--ok);
  box-shadow: 0 0 0 3px rgba(47,143,92,0.15);
  animation: pulse 2.4s ease-in-out infinite;
  flex-shrink: 0;
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 0 3px rgba(47,143,92,0.15); }
  50%     { box-shadow: 0 0 0 5px rgba(47,143,92,0.06); }
}
.status-name {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 280px;
}
.status-meta { color: var(--secondary); font-size: 12px; }
.status-tag {
  background: var(--nav-active);
  color: var(--nav-text);
  border: 1px solid #fde68a;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 600;
  margin-left: auto;
  white-space: nowrap;
}

/* ── Conversation divider ────────────────────────────────────────────────── */
.conv-rule {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 8px 0 24px;
}
.conv-rule .r { flex: 1; height: 1px; background: var(--border-2); }
.conv-rule .l {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.3px;
  text-transform: uppercase;
  color: var(--muted);
}

/* ── OR separator ────────────────────────────────────────────────────────── */
.or-sep {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding-top: 40px;
}
.or-pill {
  background: var(--bg);
  color: var(--muted);
  border: 1px solid var(--border-2);
  border-radius: 999px;
  padding: 5px 11px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
}

.section-label {
  font-size: 13px;
  font-weight: 700;
  color: var(--primary);
  margin: 0 0 10px;
  letter-spacing: -0.1px;
}

/* ── Primary button ──────────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(180deg, #f59e0b 0%, #ea8c00 100%) !important;
  color: #ffffff !important;
  border: 1px solid #ea8c00 !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
  font-size: 14px !important;
  padding: 0.55rem 1.5rem !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase !important;
  box-shadow: 0 8px 20px rgba(245,158,11,0.28) !important;
  transition: background 0.15s ease, transform 0.08s ease;
}
.stButton > button:hover  { background: linear-gradient(180deg, #ffab1f 0%, #f59e0b 100%) !important; border-color: #f59e0b !important; }
.stButton > button:active { transform: translateY(1px); }
.stButton > button:focus  { box-shadow: 0 0 0 3px rgba(245,158,11,0.20), 0 8px 20px rgba(245,158,11,0.28) !important; }

/* Clear button override */
.clear-btn .stButton > button {
  background: transparent !important;
  border: 1px solid var(--border-2) !important;
  color: var(--secondary) !important;
  padding: 4px 10px !important;
  font-size: 12px !important;
  border-radius: 8px !important;
  box-shadow: none !important;
}
.clear-btn .stButton > button:hover {
  border-color: var(--amber) !important;
  color: var(--amber-dark) !important;
  background: var(--nav-active) !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
/* Keep uploader styling visual only; do not force internal layout */
[data-testid="stFileUploader"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
  background: #fff !important;
  border: 1.5px dashed var(--border-2) !important;
  border-radius: 12px !important;
  padding: 18px 16px !important;
  transition: border-color .18s, background .18s;
  min-height: 118px !important;
}

[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--amber) !important;
  background: var(--nav-active) !important;
}

[data-testid="stFileUploader"] label {
  color: var(--primary) !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--secondary) !important;
}

/* Style button appearance only; avoid layout/child overrides */
[data-testid="stFileUploader"] button {
  background: #fff !important;
  border: 1px solid rgba(245,158,11,0.32) !important;
  border-radius: 12px !important;
  color: var(--amber-dark) !important;
  font-weight: 800 !important;
  font-size: 13px !important;
  letter-spacing: 0.6px !important;
  text-transform: uppercase !important;
  padding: 12px 22px !important;
  box-shadow: 0 4px 12px rgba(245,158,11,0.10) !important;
}

[data-testid="stFileUploader"] button:hover {
  border-color: var(--amber) !important;
  color: var(--amber-dark) !important;
  background: #fffaf0 !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
  background: #fff !important;
  border: 1px solid var(--border-2) !important;
  border-radius: 10px !important;
  margin-top: 8px !important;
}

/* ── Text area ───────────────────────────────────────────────────────────── */
[data-testid="stTextArea"] { background: transparent !important; }
[data-testid="stTextArea"] label { color: var(--primary) !important; font-weight: 600 !important; font-size: 13.5px !important; }
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextArea"] [data-baseweb="base-input"] { background: #fff !important; border: 1.5px solid var(--border-2) !important; border-radius: 12px !important; box-shadow: none !important; transition: border-color .18s; }
[data-testid="stTextArea"] textarea { background: transparent !important; border: none !important; outline: none !important; color: var(--primary) !important; font-size: 14px !important; line-height: 1.6 !important; padding: 12px 14px !important; font-family: 'Inter', sans-serif !important; }
[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
[data-testid="stTextArea"] [data-baseweb="base-input"]:focus-within { border-color: var(--amber) !important; box-shadow: 0 0 0 3px rgba(245,158,11,0.14) !important; }

/* ── Chat input ──────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: #fff !important;
  border: 1.5px solid var(--border-2) !important;
  border-radius: 24px !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
  transition: border-color .18s, box-shadow .18s;
  overflow: visible !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 3px rgba(245,158,11,0.13), 0 0 12px rgba(245,158,11,0.06) !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus,
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
  outline: none !important;
  box-shadow: none !important;
  border: none !important;
  background: transparent !important;
}
[data-testid="stChatInput"] textarea {
  font-size: 15px !important;
  color: var(--primary) !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; font-style: italic; }

[data-testid="stChatInputSubmitButton"] {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
[data-testid="stChatInputSubmitButton"] button {
  background: #855300 !important;
  border-radius: 50% !important;
  border: none !important;
  width: 36px !important;
  height: 36px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: 0 2px 8px rgba(133,83,0,0.25) !important;
  cursor: pointer !important;
}
[data-testid="stChatInputSubmitButton"] button:hover { background: #6b4400 !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #fff !important; color: #fff !important; }

/* ── Alerts & expanders ──────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid var(--border-2) !important; border-left: 3px solid var(--amber) !important; background: #fff !important; font-size: 13px !important; color: var(--primary) !important; }
details[data-testid="stExpander"] { background: #f8fafc !important; border: 1px solid var(--border-2) !important; border-radius: 12px !important; overflow: hidden; margin: -16px 0 20px; box-shadow: none !important; }
details[data-testid="stExpander"] summary { font-size: 13px !important; font-weight: 600 !important; color: var(--secondary) !important; padding: 9px 14px !important; }
details[data-testid="stExpander"] summary:hover { color: var(--amber-dark) !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

/* ── Input meta strip ────────────────────────────────────────────────────── */
.input-meta {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  padding: 8px 0 0;
}
.input-meta-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
}

</style>
""", unsafe_allow_html=True)


# ── State ─────────────────────────────────────────────────────────────────────
retriever: SentenceWindowRetriever | None = st.session_state.get("retriever")
docs          = st.session_state.get("docs", [])
combined_text = st.session_state.get("combined_text", "")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _user_bubble(content: str) -> None:
    st.markdown(
        f'<div class="chat-row user">'
        f'<div class="bubble-user">{html.escape(content)}</div></div>',
        unsafe_allow_html=True,
    )


def _ai_bubble(content: str, sources: list | None = None) -> None:
    body = _md_to_html(content)

    sources_html = ""
    if sources:
        chips = "".join(
            f'<span class="source-chip">'
            f'<svg width="8" height="11" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
            f'stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
            f'<polyline points="14 2 14 8 20 8"/></svg>'
            f'{html.escape(s.get("doc_name",""))}'
            f'{"  ·  " + html.escape(s["source"]) if s.get("source") else ""}'
            f'</span>'
            for s in sources[:3]
        )
        sources_html = (
            f'<div class="ai-card-sources">'
            f'<span class="sources-label">Sources Found</span>'
            f'<div class="source-chips">{chips}</div></div>'
        )

    st.markdown(
        f'<div class="chat-row ai"><div class="ai-card">'
        f'<div class="ai-card-body">{body}</div>'
        f'{sources_html}</div></div>',
        unsafe_allow_html=True,
    )


# ── Main header (doc loaded) ──────────────────────────────────────────────────
if combined_text and retriever:
    doc_names  = ", ".join(d["name"] for d in docs)
    short_name = doc_names if len(doc_names) < 55 else doc_names[:52] + "…"
    st.markdown(
        f'<div class="main-header">'
        f'<div class="mh-logo">'
        f'<img src="{_ICON_URI}" alt="" />'
        f'<span class="mh-logo-name">Sift</span>'
        f'</div>'
        f'<div class="mh-divider"></div>'
        f'<span class="mh-doc">Document Analysis: {html.escape(short_name)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Upload section (no doc loaded) ────────────────────────────────────────────
if not (combined_text and retriever):
    st.markdown(
        f'<div class="upload-welcome">'
        f'<div class="upload-welcome-icon"><img src="{_ICON_URI}" alt="" /></div>'
        f'<h2>Welcome to Sift</h2>'
        f'<p>Deeply analyze documents, extract complex data points, or synthesize '
        f'new insights from your library with human-centric AI precision.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        col1, col_sep, col2 = st.columns([5, 1, 5], gap="small")
        with col1:
            st.markdown('<div class="section-label">Upload documents</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload one or more files",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
        with col_sep:
            st.markdown('<div class="or-sep"><div class="or-pill">OR</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-label">Paste text</div>', unsafe_allow_html=True)
            pasted_text = st.text_area(
                "Paste text",
                height=148,
                placeholder="Paste any text here — an article, a contract, your notes…",
                label_visibility="collapsed",
            )

    _, btn_col, _ = st.columns([4, 2, 4])
    with btn_col:
      load_clicked = st.button("LOAD", use_container_width=True)

else:
    load_clicked   = False
    uploaded_files = []
    pasted_text    = ""


# ── Load logic ────────────────────────────────────────────────────────────────
if load_clicked:
    new_docs: list[dict] = []

    if uploaded_files:
        for f in uploaded_files:
            try:
                new_docs.append({"name": f.name, "text": parse_file(f)})
            except ValueError as e:
                st.error(f"{f.name}: {e}")
    elif pasted_text.strip():
        new_docs = [{"name": "Pasted text", "text": pasted_text.strip()}]
    else:
        st.error("Upload a file or paste text first.")

    if new_docs:
        raw_docs  = [(d["name"], d["text"]) for d in new_docs]
        cache_key = SentenceWindowRetriever.content_hash(
            [f"{name}\n{text}" for name, text in raw_docs] + ["sw-v1"]
        )

        st.markdown('<div id="loading-anchor"></div>', unsafe_allow_html=True)
        components.html("""<script>
        setTimeout(function(){
          var anchor = window.parent.document.getElementById('loading-anchor');
          if (anchor) {
            anchor.scrollIntoView({behavior: 'smooth', block: 'center'});
          } else {
            var el = window.parent.document.querySelector('section.main')
                  || window.parent.document.querySelector('.main');
            if (el) el.scrollTo({top: 99999, behavior: 'smooth'});
          }
        }, 80);
        </script>""", height=0)

        _build_ph = st.empty()
        _build_ph.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'justify-content:center;padding:72px 0;text-align:center;">'
            f'<img src="{_LOADING_URI}" '
            f'style="width:80px;height:80px;margin-bottom:18px;" />'
            f'<div style="font-size:13px;font-weight:700;color:#64748b;'
            f'letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">'
            f'Building Your Index…</div>'
            f'<div style="font-size:13px;color:#94a3b8;">'
            f'First run downloads ~90 MB of models</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        ret = SentenceWindowRetriever()
        if not ret.load(cache_key):
            ret.build(raw_docs)
            ret.save(cache_key)
            idx_status = "built"
        else:
            idx_status = "cached"

        _build_ph.empty()

        st.session_state.update({
            "docs":          new_docs,
            "combined_text": "\n\n".join(d["text"] for d in new_docs),
            "retriever":     ret,
            "idx_info": {
                "num_docs":   len(ret.doc_names),
                "num_chunks": ret.num_sentences,
                "status":     idx_status,
            },
            "messages":        [],
            "_scroll_to_chat": True,
        })
        st.rerun()


# ── Chat section ──────────────────────────────────────────────────────────────
if combined_text and retriever:

    if st.session_state.get("_scroll_to_chat"):
        st.session_state._scroll_to_chat = False
        components.html("""<script>
        setTimeout(function(){
          var el = window.parent.document.querySelector('section.main')
                || window.parent.document.querySelector('.main');
          if(el) el.scrollTo({top:99999,behavior:'smooth'});
        }, 300);
        </script>""", height=0)

    info       = st.session_state.get("idx_info", {})
    doc_names  = ", ".join(d["name"] for d in docs)
    short_name = doc_names if len(doc_names) < 48 else doc_names[:45] + "…"

    st.markdown(
        f'<div class="status-bar">'
        f'<div class="status-dot"></div>'
        f'<span class="status-name">{html.escape(short_name)}</span>'
        f'<span class="status-meta">'
        f'&nbsp;·&nbsp;{info.get("num_docs",0)} doc'
        f'&nbsp;·&nbsp;{info.get("num_chunks",0):,} sentences</span>'
        f'<div class="status-tag">{info.get("status","ready")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _, clr = st.columns([9, 1])
    with clr:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("Clear", help="Clear chat history"):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            f'<div class="welcome">'
            f'<div class="welcome-icon"><img src="{_ICON_URI}" alt="" /></div>'
            f'<h2>How can Sift help your research today?</h2>'
            f'<p>Deeply analyze documents, extract complex data points, or synthesize '
            f'new insights from your library with human-centric AI precision.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="conv-rule">'
            '<div class="r"></div><div class="l">Conversation</div><div class="r"></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _user_bubble(msg["content"])
        else:
            _ai_bubble(msg["content"], sources=msg.get("sources"))

    st.markdown(
        '<div class="input-meta">'
        '<span class="input-meta-item">'
        '<svg width="9" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'
        'FAISS indexing</span>'
        '<span class="input-meta-item">'
        '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
        'Document Caching</span>'
        '<span class="input-meta-item">'
        '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'
        'Llama-3 Enhanced</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if question := st.chat_input("Sift through your documents…"):
        st.session_state.messages.append({"role": "user", "content": question})
        _user_bubble(question)

        _think_ph = st.empty()
        _think_ph.markdown(
            f'<div class="thinking">'
            f'<img src="{_ICON_URI}" />'
            f'<span>Sifting Through Data…</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        results = retriever.search(question, k=3)

        if not results:
            answer  = "I couldn't find relevant content in the document for that question."
            sources = []
        else:
            context_chunks = [r["text"][:600] for r in results]
            sources        = results
            try:
                answer = generate_answer(
                    question,
                    context_chunks,
                    history=st.session_state.messages[:-1],
                )
            except EnvironmentError as e:
                answer  = str(e)
                sources = []
            except Exception as e:
                answer  = f"LLM error: {e}"
                sources = []

        _think_ph.empty()
        _ai_bubble(answer, sources=sources)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
        })

        if sources:
            best = sources[0]["score"] or 1
            with st.expander("Full source context", expanded=False):
                for r in sources:
                    pct     = int((r["score"] / best) * 100)
                    matched = r.get("matched_sentence") or r.get("text", "")
                    st.caption(f"**{r['doc_name']}** · {r.get('source','')} · relevance {pct}%")
                    st.info(matched)