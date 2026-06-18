import logging
import uuid

from qdrant_client import QdrantClient, models
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_SIZE,
    DENSE_MODEL,
    SPARSE_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)
from vector_db.chunker import chunk_text, Chunk

logger = logging.getLogger(__name__)

# cloud_inference=True makes Qdrant generate embeddings server-side from raw text
# (models.Document), so we don't run any local embedding model or external API.
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60, cloud_inference=True)

CONTENT_KEY = "page_content"
META_KEY = "metadata"
_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")
_collection_ready = False


def _has_hybrid_config(info) -> bool:
    vectors = info.config.params.vectors or {}
    sparse = info.config.params.sparse_vectors or {}
    dense_ok = isinstance(vectors, dict) and DENSE_VECTOR_NAME in vectors and vectors[DENSE_VECTOR_NAME].size == EMBEDDING_SIZE
    return bool(dense_ok and SPARSE_VECTOR_NAME in sparse)


def _create_collection():
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def setup_collection():
    global _collection_ready
    if _collection_ready:
        return
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
    except Exception as exc:
        raise RuntimeError(f"Qdrant connection failed: {exc}") from exc

    if COLLECTION_NAME not in collections:
        _create_collection()
    else:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        if not _has_hybrid_config(info):
            # Old/incompatible layout (e.g. dense-only). Safe to recreate only if empty.
            if (info.points_count or 0) == 0:
                qdrant_client.delete_collection(COLLECTION_NAME)
                _create_collection()
            else:
                raise RuntimeError(
                    f"Collection '{COLLECTION_NAME}' exists with an incompatible (non-hybrid) layout and "
                    f"holds {info.points_count} points. Recreate it or set a new COLLECTION_NAME."
                )
    _collection_ready = True


def _chunk_id(text: str, metadata: dict) -> str:
    key = text + "||" + repr(sorted(metadata.items()))
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def _to_point(text: str, metadata: dict) -> models.PointStruct:
    return models.PointStruct(
        id=_chunk_id(text, metadata),
        vector={
            DENSE_VECTOR_NAME: models.Document(text=text, model=DENSE_MODEL),
            SPARSE_VECTOR_NAME: models.Document(text=text, model=SPARSE_MODEL),
        },
        payload={CONTENT_KEY: text, META_KEY: metadata},
    )


def add_chunks(chunks: list[Chunk]) -> int:
    items = [c for c in chunks if c.text and c.text.strip()]
    if not items:
        return 0

    points, seen = [], set()
    for chunk in items:
        text = chunk.text.strip()
        pid = _chunk_id(text, chunk.metadata)
        if pid in seen:
            continue
        seen.add(pid)
        points.append(_to_point(text, chunk.metadata))

    try:
        setup_collection()
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        return len(points)
    except Exception as exc:
        raise RuntimeError(f"Failed to add documents to Qdrant: {exc}") from exc


def add_documents(texts: list[str], source: str = "", doc_type: str = "text") -> int:
    chunks: list[Chunk] = []
    for text in texts:
        for piece in chunk_text(text):
            chunks.append(Chunk(piece, {"source": source, "type": doc_type}))
    return add_chunks(chunks)


def search_kb_hits(query: str, top_k: int = 8, excerpt_len: int = 240) -> list[dict]:
    """Hybrid search returning content plus source metadata for citations."""
    try:
        setup_collection()
        result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=DENSE_MODEL),
                    using=DENSE_VECTOR_NAME,
                    limit=top_k,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=SPARSE_MODEL),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        hits: list[dict] = []
        for point in result.points:
            if not point.payload:
                continue
            text = (point.payload.get(CONTENT_KEY) or "").strip()
            if not text:
                continue
            meta = point.payload.get(META_KEY) or {}
            excerpt = text
            if len(excerpt) > excerpt_len:
                excerpt = excerpt[:excerpt_len].rsplit(" ", 1)[0] + "…"
            hits.append(
                {
                    "source": meta.get("source") or "(unknown)",
                    "type": meta.get("type") or "text",
                    "score": round(float(point.score), 3),
                    "content": text,
                    "excerpt": excerpt,
                    "image_id": meta.get("image_id") or "",
                    "sku": meta.get("sku") or "",
                }
            )
        return hits
    except Exception as exc:
        logger.warning("Qdrant search failed: %s", exc)
        return []


def get_document_count() -> int:
    try:
        return qdrant_client.get_collection(COLLECTION_NAME).points_count
    except Exception:
        return 0


def _scroll_all_points(batch_size: int = 100):
    setup_collection()
    offset = None
    while True:
        records, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        yield from records
        if offset is None:
            break


def list_sources() -> list[dict]:
    counts: dict[str, int] = {}
    try:
        for record in _scroll_all_points():
            meta = (record.payload or {}).get(META_KEY) or {}
            source = meta.get("source") or "(unknown)"
            counts[source] = counts.get(source, 0) + 1
    except Exception as exc:
        logger.warning("Failed to list KB sources: %s", exc)
        return []
    return [{"source": name, "chunks": count} for name, count in sorted(counts.items())]


def _chunks_for_source(source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in _scroll_all_points():
        meta = (record.payload or {}).get(META_KEY) or {}
        if meta.get("source") != source:
            continue
        text = (record.payload or {}).get(CONTENT_KEY, "")
        if text:
            chunks.append(Chunk(text, dict(meta)))
    return chunks


def delete_by_source(source: str) -> int:
    try:
        setup_collection()
        ids = [
            record.id
            for record in _scroll_all_points()
            if ((record.payload or {}).get(META_KEY) or {}).get("source") == source
        ]
        if not ids:
            return 0
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=models.PointIdsList(points=ids), wait=True)
        return len(ids)
    except Exception as exc:
        raise RuntimeError(f"Failed to delete source '{source}': {exc}") from exc


def reindex_source(source: str) -> int:
    chunks = _chunks_for_source(source)
    if not chunks:
        return 0
    delete_by_source(source)
    return add_chunks(chunks)
