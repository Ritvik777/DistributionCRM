import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_SIZE
from vector_db.embeddings import get_embedding_model
from vector_db.chunker import chunk_text, Chunk

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

CONTENT_KEY = "page_content"
_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
_text_index_ready = False


def _ensure_text_index():
    # Full-text index powers the keyword half of hybrid search (e.g. exact SKU lookups).
    global _text_index_ready
    if _text_index_ready:
        return
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=CONTENT_KEY,
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=30,
                lowercase=True,
            ),
        )
    except Exception:
        pass
    _text_index_ready = True


def setup_collection():
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
    except Exception as exc:
        raise RuntimeError(f"Qdrant connection failed: {exc}") from exc

    if COLLECTION_NAME in collections:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        if info.config.params.vectors.size != EMBEDDING_SIZE:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' vector size mismatch. "
                f"Expected {EMBEDDING_SIZE}, found {info.config.params.vectors.size}. "
                "Use a new COLLECTION_NAME or recreate this collection manually."
            )
    if COLLECTION_NAME not in collections:
        try:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to create Qdrant collection: {exc}") from exc

    _ensure_text_index()


def get_vector_store():
    setup_collection()
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=get_embedding_model(),
    )


def _chunk_id(text: str, metadata: dict) -> str:
    # Deterministic ID so re-uploading the same content upserts instead of duplicating.
    key = text + "||" + repr(sorted(metadata.items()))
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def add_chunks(chunks: list[Chunk]) -> int:
    items = [c for c in chunks if c.text and c.text.strip()]
    if not items:
        return 0

    texts, metadatas, ids, seen = [], [], [], set()
    for chunk in items:
        text = chunk.text.strip()
        chunk_id = _chunk_id(text, chunk.metadata)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        texts.append(text)
        metadatas.append(chunk.metadata)
        ids.append(chunk_id)

    try:
        store = get_vector_store()
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return len(texts)
    except Exception as exc:
        raise RuntimeError(f"Failed to add documents to Qdrant: {exc}") from exc


def add_documents(texts: list[str], source: str = "", doc_type: str = "text") -> int:
    chunks: list[Chunk] = []
    for text in texts:
        for piece in chunk_text(text):
            chunks.append(Chunk(piece, {"source": source, "type": doc_type}))
    return add_chunks(chunks)


def _keyword_search(query: str, limit: int) -> list[str]:
    # Token AND-match: only fires when the doc contains all query tokens (great for SKUs/codes).
    try:
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key=CONTENT_KEY, match=models.MatchText(text=query))]
            ),
            limit=limit,
            with_payload=True,
        )
        return [p.payload.get(CONTENT_KEY, "") for p in points if p.payload]
    except Exception:
        return []


def search_with_scores(query: str, top_k: int = 8) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    seen: set[str] = set()

    # Keyword (exact-ish) matches first so SKU / code lookups always surface.
    for text in _keyword_search(query, top_k):
        if text and text not in seen:
            seen.add(text)
            results.append((text, 1.0))

    try:
        store = get_vector_store()
        for doc, score in store.similarity_search_with_score(query, k=top_k):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                results.append((doc.page_content, score))
    except Exception:
        pass

    return results[:top_k]


def get_document_count() -> int:
    try:
        return qdrant_client.get_collection(COLLECTION_NAME).points_count
    except Exception:
        return 0
