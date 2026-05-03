<div align="center">
  <img src="assets/sift_demo.gif" width="720" alt="Sift — particles sifting through a retrieval filter" />
</div>

<br />

<div align="center">
  <img src="assets/sift_logo.png" width="240" alt="Sift" />
  <p><strong>Ask questions of your documents. Get answers grounded in the source.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20Llama--3-7C3AED?style=flat-square" alt="LLM" />
    <img src="https://img.shields.io/badge/retrieval-SBERT%20%2B%20BM25%20%2B%20RRF-6366F1?style=flat-square" alt="Retrieval" />
  </p>
</div>

---

Upload a PDF, DOCX, or TXT — or paste any text — then ask questions in plain English. Every answer is grounded in **retrieved context** from your document, not hallucinated from model weights. Follow-up questions stay coherent because prior turns travel with each prompt.

## How it works

```
Document → sentences → SBERT index + BM25 index
                              ↓
              Query hits both indexes in parallel
                              ↓
                   RRF fuses the ranked lists
                              ↓
          Matched sentence → expand ±2 neighbours
                              ↓
              LLM generates a grounded answer
```

1. **Index** — each document is split into individual sentences and indexed with SBERT embeddings (FAISS) and BM25 in parallel. The index is cached to disk by content hash so subsequent loads are instant.

2. **Retrieve** — a query hits both indexes simultaneously. Reciprocal Rank Fusion (RRF) merges the two ranked lists, capturing exact-keyword hits from BM25 and semantic matches from SBERT.

3. **Expand** — the best-matching sentence is padded with ±2 neighbouring sentences (sentence-window). The LLM gets enough context without drowning in noise.

4. **Generate** — Groq + Llama-3 produces a direct, grounded answer. The last 6 chat turns are included so short follow-ups like "why?" still make sense.

## Tech stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| Retrieval | Sentence-window hybrid — SBERT + BM25 + RRF |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector index | FAISS (`faiss-cpu`), persisted by content hash |
| LLM | Groq — `llama-3.1-8b-instant` |
| Parsing | PyPDF2 · python-docx |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

Create `.env` in the project root (copy from `.env.example`):

```env
GROQ_API_KEY=your_key_here
```

Run:

```bash
streamlit run app.py
```