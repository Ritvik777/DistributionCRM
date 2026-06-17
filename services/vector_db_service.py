from vector_db import add_documents, add_chunks, get_document_count, list_sources, delete_by_source, reindex_source
from vector_db.chunker import extract_chunks_from_pdf, extract_chunks_from_excel, extract_chunks_from_csv


def get_doc_count() -> int:
    return get_document_count()


def get_kb_sources() -> list[dict]:
    return list_sources()


def remove_kb_source(source: str) -> int:
    return delete_by_source(source)


def reindex_kb_source(source: str) -> int:
    return reindex_source(source)


def add_text_documents(raw_text: str) -> int:
    text = raw_text.strip()
    if not text:
        return 0
    return add_documents([text], source="pasted text", doc_type="text")


def add_pdf_document(pdf_file) -> int:
    source = getattr(pdf_file, "name", "PDF")
    return add_chunks(extract_chunks_from_pdf(pdf_file, source=source))


def add_excel_document(excel_file) -> int:
    source = getattr(excel_file, "name", "Excel")
    return add_chunks(extract_chunks_from_excel(excel_file, source=source))


def add_csv_document(csv_file) -> int:
    source = getattr(csv_file, "name", "CSV")
    return add_chunks(extract_chunks_from_csv(csv_file, source=source))
