from PyPDF2 import PdfReader
from docx import Document


def parse_file(file) -> str:
    """Extract text from an uploaded Streamlit file object."""
    name = file.name.lower()

    try:
        if name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif name.endswith(".docx"):
            doc = Document(file)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".txt"):
            text = file.read().decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file.name}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse {file.name}: {e}")

    text = text.strip()
    if not text:
        raise ValueError(f"No readable text in {file.name} — may be empty or scanned.")

    return text
