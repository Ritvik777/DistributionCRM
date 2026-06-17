"""
vector_db/ — Everything related to the vector database
========================================================
  chunker.py     → text splitting + PDF/Excel quotation extraction
  database.py    → Qdrant: server-side hybrid (dense + BM25) embeddings, store, search
"""
from vector_db.database import (
    add_documents,
    add_chunks,
    search_with_scores,
    get_document_count,
    list_sources,
    delete_by_source,
    reindex_source,
)
from vector_db.chunker import (
    Chunk,
    extract_chunks_from_pdf,
    extract_chunks_from_excel,
    extract_chunks_from_csv,
)
