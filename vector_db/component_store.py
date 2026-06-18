"""Component image catalog: CLIP visual vectors + Claude captions in Qdrant."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from qdrant_client import models

from config import (
    CLIP_MODEL_NAME,
    CLIP_VECTOR_DIM,
    COMPONENT_CLIP_CANDIDATES,
    COMPONENT_COLLECTION_NAME,
    COMPONENT_IMAGES_DIR,
    COMPONENT_RERANK_POOL,
)
from vector_db.chunker import Chunk
from vector_db.database import add_chunks, delete_by_source, qdrant_client, search_kb_hits
from vector_db.vision import (
    caption_component_image,
    caption_to_catalog_text,
    prepare_image_bytes,
    rerank_component_candidates,
)

logger = logging.getLogger(__name__)

META_KEY = "metadata"
IMAGE_ID_KEY = "image_id"
_collection_ready = False
_clip_model = None


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", (name or "component").strip())
    return cleaned[:80] or "component"


def _ensure_image_dir() -> Path:
    path = Path(COMPONENT_IMAGES_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_clip_model():
    global _clip_model
    if _clip_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for component image matching. "
                "Run: pip install sentence-transformers torch pillow"
            ) from exc
        logger.info("Loading CLIP model %s (first run may download weights)...", CLIP_MODEL_NAME)
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME)
    return _clip_model


def embed_image(image_bytes: bytes) -> list[float]:
    from PIL import Image
    import io

    prepared = prepare_image_bytes(image_bytes)
    image = Image.open(io.BytesIO(prepared)).convert("RGB")
    vector = get_clip_model().encode(image, normalize_embeddings=True)
    return vector.tolist()


def setup_component_collection() -> None:
    global _collection_ready
    if _collection_ready:
        return

    collections = [item.name for item in qdrant_client.get_collections().collections]
    if COMPONENT_COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COMPONENT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=CLIP_VECTOR_DIM, distance=models.Distance.COSINE),
        )
    else:
        info = qdrant_client.get_collection(COMPONENT_COLLECTION_NAME)
        params = info.config.params.vectors
        size = params.size if hasattr(params, "size") else None
        if isinstance(params, dict):
            size = next(iter(params.values())).size
        if size != CLIP_VECTOR_DIM:
            raise RuntimeError(
                f"Collection '{COMPONENT_COLLECTION_NAME}' has vector size {size}, expected {CLIP_VECTOR_DIM}."
            )
    _collection_ready = True


def _point_id(image_id: str) -> str:
    return str(uuid.uuid5(uuid.UUID("8f4e2a1c-9b3d-4e5f-a6b7-c8d9e0f1a2b3"), image_id))


def index_component_image(
    image_bytes: bytes,
    *,
    filename: str = "component.jpg",
    sku: str = "",
    name: str = "",
) -> dict:
    """Save a catalog photo, embed with CLIP, caption with Claude, index for hybrid search."""
    prepared = prepare_image_bytes(image_bytes)
    image_id = uuid.uuid4().hex
    source = filename or "component.jpg"
    catalog_source = f"component_image/{image_id}/{source}"
    stored_name = f"{image_id}_{_safe_filename(source)}.jpg"
    image_path = _ensure_image_dir() / stored_name
    image_path.write_bytes(prepared)

    caption = caption_component_image(prepared, filename=source, sku=sku, name=name)
    if caption is None:
        raise RuntimeError("Vision model could not caption this component image.")

    catalog_text = caption_to_catalog_text(caption, sku=sku, name=name)
    vector = embed_image(prepared)
    setup_component_collection()

    metadata = {
        "source": catalog_source,
        "type": "component_image",
        "image_id": image_id,
        "image_path": str(image_path),
        "sku": sku.strip(),
        "name": name.strip(),
        "caption": caption.summary,
        "category": caption.category,
        "package": caption.package,
    }
    qdrant_client.upsert(
        collection_name=COMPONENT_COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=_point_id(image_id),
                vector=vector,
                payload={META_KEY: metadata, "caption_full": catalog_text},
            )
        ],
        wait=True,
    )

    text_meta = dict(metadata)
    text_meta["doc_type"] = "component_image"
    add_chunks([Chunk(catalog_text, text_meta)])

    return {
        "image_id": image_id,
        "source": catalog_source,
        "sku": sku,
        "name": name,
        "caption": caption.summary,
        "image_path": str(image_path),
    }


def _search_clip(image_bytes: bytes, top_k: int = 8) -> list[dict]:
    setup_component_collection()
    vector = embed_image(image_bytes)
    results = qdrant_client.query_points(
        collection_name=COMPONENT_COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    hits: list[dict] = []
    for point in results.points:
        meta = (point.payload or {}).get(META_KEY) or {}
        hits.append(
            {
                "image_id": meta.get("image_id") or "",
                "source": meta.get("source") or "(unknown)",
                "sku": meta.get("sku") or "",
                "name": meta.get("name") or "",
                "caption": meta.get("caption") or "",
                "category": meta.get("category") or "",
                "package": meta.get("package") or "",
                "image_path": meta.get("image_path") or "",
                "clip_score": round(float(point.score), 4),
            }
        )
    return hits


def _text_match_scores(query_text: str, candidates: list[dict], top_k: int = 12) -> dict[str, float]:
    if not query_text.strip():
        return {}
    hits = search_kb_hits(query_text, top_k=top_k)
    scores: dict[str, float] = {}
    for hit in hits:
        image_id = hit.get("image_id") or ""
        if image_id:
            scores[image_id] = max(scores.get(image_id, 0.0), float(hit.get("score", 0)))
            continue
        if hit.get("type") != "component_image":
            continue
        content = hit.get("content") or ""
        for candidate in candidates:
            cid = candidate.get("image_id") or ""
            sku = candidate.get("sku") or ""
            if not cid:
                continue
            if sku and f"SKU: {sku}" in content:
                scores[cid] = max(scores.get(cid, 0.0), float(hit.get("score", 0)))
    return scores


def _aggregate_by_sku(matches: list[dict]) -> list[dict]:
    """Keep the best-scoring photo per SKU so multi-photo catalogs dedupe cleanly."""
    best_by_sku: dict[str, dict] = {}
    unsorted: list[dict] = []
    for match in matches:
        sku = (match.get("sku") or "").strip()
        if not sku:
            unsorted.append(match)
            continue
        current = best_by_sku.get(sku)
        if current is None or match.get("match_score", 0) > current.get("match_score", 0):
            best_by_sku[sku] = match
    combined = list(best_by_sku.values()) + unsorted
    combined.sort(key=lambda item: item.get("match_score", 0), reverse=True)
    return combined


def _part_number_boost(query_summary: str, candidate: dict) -> float:
    """Small boost when visible part numbers in the query appear in catalog metadata."""
    if not query_summary:
        return 0.0
    query_lower = query_summary.lower()
    caption = (candidate.get("caption") or "").lower()
    sku = (candidate.get("sku") or "").lower()
    boost = 0.0
    if sku and sku in query_lower:
        boost += 0.08
    for token in query_lower.replace(",", " ").split():
        if len(token) >= 4 and token in caption:
            boost += 0.03
    return min(boost, 0.15)


def _fuse_score(clip_score: float, vision_score: float, text_score: float, extra: float = 0.0) -> float:
    # CLIP cosine for normalized embeddings is typically ~0.15–0.40 for good matches.
    clip_norm = min(max((clip_score - 0.12) / 0.28, 0.0), 1.0)
    vision_norm = min(max(vision_score / 100.0, 0.0), 1.0)
    text_norm = min(max(text_score / 10.0, 0.0), 1.0) if text_score else 0.0
    fused = round(0.38 * clip_norm + 0.47 * vision_norm + 0.15 * text_norm + extra, 4)
    return min(fused, 1.0)


def match_component_image(image_bytes: bytes, *, filename: str = "", top_k: int = 5) -> list[dict]:
    """Hybrid match: CLIP retrieval → Claude vision re-rank → text KB corroboration."""
    prepared = prepare_image_bytes(image_bytes)
    clip_hits = _search_clip(prepared, top_k=max(top_k * 2, COMPONENT_CLIP_CANDIDATES))
    if not clip_hits:
        return []

    pool_size = min(COMPONENT_RERANK_POOL, len(clip_hits))
    rerank_pool = clip_hits[:pool_size]
    vision = rerank_component_candidates(prepared, rerank_pool, filename=filename)
    vision_by_id = {}
    query_summary = ""
    if vision:
        query_summary = vision.query_summary
        vision_by_id = {item.image_id: item for item in vision.candidates}

    if not query_summary:
        caption = caption_component_image(prepared, filename=filename)
        query_summary = caption.summary if caption else ""

    text_scores = _text_match_scores(query_summary, clip_hits)

    matches: list[dict] = []
    for hit in clip_hits:
        image_id = hit["image_id"]
        vision_item = vision_by_id.get(image_id)
        vision_score = vision_item.score if vision_item else int(min(max(hit["clip_score"] * 200, 0), 85))
        reasoning = vision_item.reasoning if vision_item else "CLIP visual similarity only (vision re-rank unavailable)."
        text_score = text_scores.get(image_id, 0.0)
        boost = _part_number_boost(query_summary, hit)
        fused = _fuse_score(hit["clip_score"], vision_score, text_score, extra=boost)
        matches.append(
            {
                **hit,
                "vision_score": vision_score,
                "text_score": round(text_score, 3),
                "match_score": fused,
                "match_percent": int(round(fused * 100)),
                "match_tier": (
                    "high" if fused >= 0.8 else "medium" if fused >= 0.6 else "low" if fused >= 0.4 else "none"
                ),
                "reasoning": reasoning,
                "query_summary": query_summary,
            }
        )

    matches.sort(key=lambda item: item["match_score"], reverse=True)
    return _aggregate_by_sku(matches)[:top_k]


def list_component_images() -> list[dict]:
    setup_component_collection()
    items: list[dict] = []
    offset = None
    while True:
        records, offset = qdrant_client.scroll(
            collection_name=COMPONENT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            meta = (record.payload or {}).get(META_KEY) or {}
            items.append(
                {
                    "image_id": meta.get("image_id") or "",
                    "source": meta.get("source") or "(unknown)",
                    "sku": meta.get("sku") or "",
                    "name": meta.get("name") or "",
                    "caption": meta.get("caption") or "",
                    "category": meta.get("category") or "",
                    "package": meta.get("package") or "",
                    "image_path": meta.get("image_path") or "",
                }
            )
        if offset is None:
            break
    return sorted(items, key=lambda item: (item.get("sku") or item.get("source") or "").lower())


def find_catalog_match_by_sku(sku: str) -> dict | None:
    """Look up a catalog photo record by SKU (case-insensitive)."""
    needle = (sku or "").strip().lower()
    if not needle:
        return None
    for item in list_component_images():
        if (item.get("sku") or "").strip().lower() == needle:
            path = item.get("image_path") or ""
            if path and Path(path).is_file():
                return item
    return None


def get_component_count() -> int:
    try:
        setup_component_collection()
        return qdrant_client.get_collection(COMPONENT_COLLECTION_NAME).points_count or 0
    except Exception:
        return 0


def delete_component_image(image_id: str) -> bool:
    setup_component_collection()
    records, _ = qdrant_client.scroll(
        collection_name=COMPONENT_COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key=f"{META_KEY}.image_id", match=models.MatchValue(value=image_id))]
        ),
        limit=1,
        with_payload=True,
    )
    if not records:
        return False

    meta = (records[0].payload or {}).get(META_KEY) or {}
    catalog_source = meta.get("source") or ""
    image_path = meta.get("image_path") or ""
    qdrant_client.delete(
        collection_name=COMPONENT_COLLECTION_NAME,
        points_selector=models.PointIdsList(points=[records[0].id]),
        wait=True,
    )
    if catalog_source:
        try:
            delete_by_source(catalog_source)
        except Exception as exc:
            logger.warning("Could not delete linked text chunks for %s: %s", catalog_source, exc)
    if image_path:
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete image file %s: %s", image_path, exc)
    return True
