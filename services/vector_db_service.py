from vector_db import add_documents, add_chunks, get_document_count, search_kb_hits
from vector_db.component_store import (
    delete_component_image,
    get_component_count,
    index_component_image,
    list_component_images,
    match_component_image,
)
from vector_db.chunker import extract_chunks_from_pdf, extract_chunks_from_excel, extract_chunks_from_csv


def get_doc_count() -> int:
    return get_document_count()


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


def add_component_image(image_file, sku: str = "", name: str = "") -> dict:
    raw = image_file.read()
    filename = getattr(image_file, "name", "component.jpg")
    return index_component_image(raw, filename=filename, sku=sku.strip(), name=name.strip())


def match_component_image_file(image_file, top_k: int = 5) -> list[dict]:
    raw = image_file.read()
    filename = getattr(image_file, "name", "query.jpg")
    return match_component_image(raw, filename=filename, top_k=top_k)


def add_component_images_batch(image_files, sku: str = "", name: str = "") -> list[dict]:
    results: list[dict] = []
    for image_file in image_files:
        results.append(add_component_image(image_file, sku=sku, name=name))
    return results


def match_component_image_bytes(image_bytes: bytes, filename: str = "query.jpg", top_k: int = 5) -> list[dict]:
    return match_component_image(image_bytes, filename=filename, top_k=top_k)


def get_component_catalog() -> list[dict]:
    return list_component_images()


def get_component_image_count() -> int:
    return get_component_count()


def remove_component_image(image_id: str) -> bool:
    return delete_component_image(image_id)
