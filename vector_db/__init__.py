"""
vector_db/ — Everything related to the vector database
========================================================
  chunker.py        → text splitting + PDF/Excel quotation extraction
  database.py       → Qdrant: server-side hybrid (dense + BM25) embeddings, store, search
  vision.py         → Claude vision captioning and re-ranking
  component_store.py → CLIP image catalog + hybrid component matching
"""
from vector_db.database import (
    add_documents,
    add_chunks,
    search_kb_hits,
    get_document_count,
    delete_by_source,
)
from vector_db.chunker import (
    Chunk,
    extract_chunks_from_pdf,
    extract_chunks_from_excel,
    extract_chunks_from_csv,
)
from vector_db.component_store import (
    delete_component_image,
    get_component_count,
    index_component_image,
    list_component_images,
    match_component_image,
)
