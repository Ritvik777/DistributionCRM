"""Shared component image matching for GTM, outreach, and UI."""

from __future__ import annotations

import base64

from services.vector_db_service import match_component_image_bytes, search_kb_hits


def format_component_match_context(matches: list[dict]) -> str:
    if not matches:
        return (
            "Component image matching found no catalog matches. "
            "The component image catalog may be empty, or this part is not indexed yet."
        )
    query_summary = matches[0].get("query_summary") or ""
    lines = [
        "Component image hybrid match results (CLIP visual search + Claude vision re-rank + text KB):",
    ]
    if query_summary:
        lines.append(f"Query image understood as: {query_summary}")
    for index, match in enumerate(matches, start=1):
        label = match.get("name") or match.get("sku") or match.get("source") or f"Candidate {index}"
        lines.append(
            f"{index}. {label} — {match.get('match_percent', 0)}% combined confidence\n"
            f"   SKU: {match.get('sku') or 'n/a'} | Category: {match.get('category') or 'n/a'} | "
            f"Package: {match.get('package') or 'n/a'}\n"
            f"   CLIP: {match.get('clip_score')} | Vision: {match.get('vision_score')}/100 | "
            f"Text KB: {match.get('text_score')}\n"
            f"   Reason: {match.get('reasoning')}\n"
            f"   Catalog caption: {match.get('caption') or 'n/a'}"
        )
    return "\n\n".join(lines)


def kb_sources_from_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "source": hit.get("source") or "(unknown)",
            "type": hit.get("type") or "text",
            "score": hit.get("score", 0),
            "excerpt": hit.get("excerpt") or "",
        }
        for hit in hits
    ]


def kb_sources_for_matches(matches: list[dict], *, top_k: int = 6) -> list[dict]:
    if not matches:
        return []
    summary = matches[0].get("query_summary") or ""
    if not summary:
        return []
    return kb_sources_from_hits(search_kb_hits(summary, top_k=top_k))


def run_component_image_match(image_b64: str) -> tuple[str, list[dict], list[dict]]:
    raw = base64.standard_b64decode(image_b64)
    matches = match_component_image_bytes(raw, filename="query.jpg")
    context = format_component_match_context(matches)
    kb_sources = kb_sources_for_matches(matches)
    return context, matches, kb_sources


def run_component_image_match_bytes(
    image_bytes: bytes,
    *,
    filename: str = "query.jpg",
) -> tuple[str, list[dict], list[dict]]:
    matches = match_component_image_bytes(image_bytes, filename=filename)
    context = format_component_match_context(matches)
    kb_sources = kb_sources_for_matches(matches)
    return context, matches, kb_sources


def resolve_component_context(
    *,
    component_matches: list[dict] | None,
    query_image_b64: str = "",
) -> tuple[str, list[dict], list[dict]]:
    """Use precomputed matches, or run hybrid match from base64 image."""
    matches = component_matches or []
    if matches:
        return format_component_match_context(matches), matches, kb_sources_for_matches(matches)
    image_b64 = (query_image_b64 or "").strip()
    if image_b64:
        return run_component_image_match(image_b64)
    return "", [], []
