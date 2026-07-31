"""
chatbot/pdf_reader.py

Loads uploaded files (PDF, TXT), extracts text, and indexes it into the
vector store (chatbot/vectorstore.py) for RAG retrieval. This file was a
stub (`...` placeholders) in the starter template.
"""

import os
import fitz  # PyMuPDF

from chatbot import vectorstore

# Tracks whichever document was most recently uploaded/loaded (shown in the UI)
active_document = {
    "filename": None,
    "chunks_indexed": 0,
}


def _extract_pdf_text(filepath: str) -> str:
    text_parts = []
    with fitz.open(filepath) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_txt_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(filepath: str):
    """Extract text from an uploaded PDF (or .txt) and index it for RAG.
    Despite the name (kept for compatibility with app.py's existing call),
    this also handles .txt files so the preloaded EU AI Act corpus works."""
    filename = os.path.basename(filepath)

    if filepath.lower().endswith(".pdf"):
        text = _extract_pdf_text(filepath)
    elif filepath.lower().endswith(".txt"):
        text = _extract_txt_text(filepath)
    else:
        # Unsupported for text extraction (docx/csv/xlsx) -- stored but not indexed
        active_document["filename"] = filename
        active_document["chunks_indexed"] = 0
        return

    n_chunks = vectorstore.index_document(doc_id=filename, text=text)

    active_document["filename"] = filename
    active_document["chunks_indexed"] = n_chunks


def get_document_text():
    """Kept for backward compatibility with any code expecting the old
    'dump full document into the prompt' behaviour. Prefer vectorstore.retrieve()
    for actual grounding -- see chatbot/llm.py."""
    return ""


def get_active_document():
    return active_document


def load_default_corpus():
    """Preload the EU AI Act grounding corpus at startup so the assistant
    has something to retrieve from even before a user uploads a file."""
    from config import DOCUMENT_FOLDER

    default_path = os.path.join(DOCUMENT_FOLDER, "EU_AI_Act_Recitals.txt")
    if os.path.exists(default_path):
        load_pdf(default_path)  # handles .txt too, see above
