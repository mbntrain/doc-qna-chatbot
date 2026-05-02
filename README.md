# Sift

**Retrieval that reasons.**

Upload PDF, DOCX, or TXT (or paste text), then chat over your documents. Each answer uses **retrieved context** (sentence-window + hybrid search) and an **LLM** (Groq) so follow-ups stay coherent.

## How it works

1. Documents are split into sentences and indexed with **SBERT** and **BM25**; results are fused with **RRF**.
2. The best-matching sentence is expanded to a **local window** (neighboring sentences) for LLM context.
3. **Conversational RAG**: prior chat turns are sent with the latest question so short follow-ups still make sense.

## Assets

Brand files live in `assets/`:

| File | Use |
|------|-----|
| `sift_logo.png` | Main header logo |
| `sift_logo_dark.png` | Alternate / dark surfaces |
| `sift_icon.png` | Favicon, assistant avatar, empty state |

## Tech stack

| Layer | Choice |
|--------|--------|
| UI | Streamlit |
| Retrieval | Sentence-window hybrid (SBERT + BM25 + RRF), cached index |
| LLM | Groq (`llama-3.1-8b-instant`) via `GROQ_API_KEY` |
| Parsing | PyPDF2, python-docx |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Create `.env` in the project root (see `.env.example`):

```env
GROQ_API_KEY=your_key_here
```

Run:

```bash
streamlit run app.py
```
