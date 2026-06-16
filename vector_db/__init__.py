"""
vector_db/ — Everything related to the vector database
========================================================
  embeddings.py  → Google Gemini embedding model (text → vectors)
  chunker.py     → text splitting + PDF/Excel quotation extraction
  database.py    → Qdrant: store, search, and manage documents
"""
from vector_db.database import add_documents, add_chunks, search_with_scores, get_document_count
from vector_db.chunker import (
    Chunk,
    extract_chunks_from_pdf,
    extract_chunks_from_excel,
    extract_chunks_from_csv,
)
