from contextvars import ContextVar

from langchain_core.tools import tool

from vector_db import search_kb_hits

_kb_hits_var: ContextVar[list[dict] | None] = ContextVar("_kb_hits_collector", default=None)


def begin_kb_collection() -> None:
    """Start collecting KB source metadata for the current tool-calling window."""
    _kb_hits_var.set([])


def _record_hits(hits: list[dict]) -> None:
    collector = _kb_hits_var.get()
    if collector is not None:
        collector.extend(hits)


def consume_kb_sources() -> list[dict]:
    """Return deduped KB citations collected since begin_kb_collection()."""
    collector = _kb_hits_var.get() or []
    _kb_hits_var.set(None)
    return _dedupe_sources(collector)


def _dedupe_sources(hits: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for hit in sorted(hits, key=lambda item: -item.get("score", 0)):
        key = (hit.get("source", ""), (hit.get("excerpt") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "source": hit.get("source") or "(unknown)",
                "type": hit.get("type") or "text",
                "score": hit.get("score", 0),
                "excerpt": hit.get("excerpt") or "",
            }
        )
    return unique


def _format_hit(hit: dict) -> str:
    label = hit.get("source") or "(unknown)"
    doc_type = hit.get("type") or "text"
    score = hit.get("score", 0)
    return f"[Source: {label} ({doc_type}) | relevance: {score:.3f}]\n{hit['content']}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search internal product docs stored in Qdrant."""
    hits = search_kb_hits(query, top_k=8)
    _record_hits(hits)
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(_format_hit(hit) for hit in hits)
