# Document Q&A Chatbot

Upload a document (PDF, DOCX, TXT) or paste text — ask questions, get answers.

## How It Works

Uses **TF-IDF scoring** to find the most relevant sentence for your question:
1. Extracts keywords from your question (removes stop words)
2. Splits the document into sentences
3. Scores each sentence by Term Frequency × Inverse Document Frequency
4. Returns the highest-scoring sentence with a confidence percentage

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Search | TF-IDF (custom, no ML) |
| File Parsing | PyPDF2, python-docx |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

