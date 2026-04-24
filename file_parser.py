from PyPDF2 import PdfReader
from docx import Document


def parse_file(file) -> str:
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith(".txt"):
        return file.read().decode("utf-8")

    raise ValueError(f"Unsupported file type: {file.name}")
