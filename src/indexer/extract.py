# src/indexer/extract.py
from pypdf import PdfReader
import os

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text page-wise from a PDF.
    Returns: [{'page_no': 1, 'content': '...'}, ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_no": i + 1, "content": text.strip()})
    return pages
