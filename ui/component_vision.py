"""Component Vision workspace — fast CLIP + Claude matching UI."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st

from services.vector_db_service import (
    get_component_image_count,
    match_component_image_bytes,
)


def tier_label(tier: str, percent: int) -> tuple[str, str]:
    mapping = {
        "high": ("Strong match", "#059669"),
        "medium": ("Likely match", "#d97706"),
        "low": ("Weak match", "#dc2626"),
        "none": ("No match", "#6b7280"),
    }
    label, color = mapping.get(tier, ("Unknown", "#6b7280"))
    if percent >= 80:
        return "Strong match", "#059669"
    if percent >= 60:
        return "Likely match", "#d97706"
    if percent >= 40:
        return "Weak match", "#dc2626"
    return label, color


def _tier_label(tier: str, percent: int) -> tuple[str, str]:
    return tier_label(tier, percent)


def confidence_bar_html(percent: int, color: str) -> str:
    return (
        f'<div class="conf-track">'
        f'<div class="conf-fill" style="width:{min(percent, 100)}%;background:{color}"></div>'
        f"</div>"
    )


def _confidence_bar(percent: int, color: str) -> str:
    return confidence_bar_html(percent, color)


def match_card_html(match: dict, rank: int) -> str:
    tier = match.get("match_tier") or "none"
    percent = match.get("match_percent", 0)
    tier_text, tier_color = tier_label(tier, percent)
    label = html.escape(match.get("name") or match.get("sku") or match.get("source") or f"Candidate {rank}")
    sku = html.escape(match.get("sku") or "—")
    reasoning = html.escape(match.get("reasoning") or "")
    category = html.escape(match.get("category") or "—")
    package = html.escape(match.get("package") or "—")
    clip = match.get("clip_score", "—")
    vision = match.get("vision_score", "—")
    text = match.get("text_score", "—")
    reasoning_block = f'<div class="vision-reason">{reasoning}</div>' if reasoning else ""
    return (
        f'<div class="vision-match-card">'
        f'<div class="vision-match-rank">#{rank}</div>'
        f'<div class="vision-match-title">{label}</div>'
        f'<div class="vision-match-tier" style="color:{tier_color}">{tier_text} · {percent}%</div>'
        f"{confidence_bar_html(percent, tier_color)}"
        f'<div class="vision-match-meta">SKU <code>{sku}</code> · {category} · {package}</div>'
        f'<div class="vision-scores">CLIP {clip} · Vision {vision}/100 · Text {text}</div>'
        f"{reasoning_block}"
        f"</div>"
    )


def _image_mime(path_or_bytes: bytes | str, *, from_path: bool = False) -> str:
    if from_path and isinstance(path_or_bytes, str):
        suffix = Path(path_or_bytes).suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        return "image/jpeg"
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else b""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _bytes_to_data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.standard_b64encode(data).decode('ascii')}"


def compare_images_html(
    query_image_bytes: bytes | None,
    catalog_image_path: str | None,
    *,
    max_height: int = 240,
) -> str:
    """Side-by-side Your photo vs Catalog match (HTML for consistent sizing)."""
    query_block = '<div class="compare-placeholder">No photo</div>'
    if query_image_bytes:
        mime = _image_mime(query_image_bytes)
        src = _bytes_to_data_uri(query_image_bytes, mime)
        query_block = f'<img src="{src}" alt="Your photo" style="max-height:{max_height}px" />'

    catalog_block = '<div class="compare-placeholder">No catalog image</div>'
    if catalog_image_path and Path(catalog_image_path).exists():
        raw = Path(catalog_image_path).read_bytes()
        mime = _image_mime(catalog_image_path, from_path=True)
        src = _bytes_to_data_uri(raw, mime)
        catalog_block = f'<img src="{src}" alt="Catalog match" style="max-height:{max_height}px" />'

    return (
        '<div class="catalog-compare">'
        '<div class="compare-side">'
        '<div class="compare-label">Your photo</div>'
        f'<div class="compare-frame">{query_block}</div>'
        "</div>"
        '<div class="compare-vs" aria-hidden="true">'
        '<span class="compare-vs-icon">↔</span>'
        "</div>"
        '<div class="compare-side">'
        '<div class="compare-label">Catalog match</div>'
        f'<div class="compare-frame">{catalog_block}</div>'
        "</div>"
        "</div>"
    )


def render_catalog_matches_panel(
    matches: list[dict],
    query_image_bytes: bytes | None = None,
    *,
    max_results: int = 5,
) -> None:
    """Rich aside-style catalog match panel for chat and vision workspace."""
    if not matches and not query_image_bytes:
        return

    st.markdown('<div class="catalog-match-panel">', unsafe_allow_html=True)
    st.markdown('<div class="catalog-match-header">Catalog matches</div>', unsafe_allow_html=True)

    if not matches:
        if query_image_bytes:
            st.image(query_image_bytes, width=140)
        st.caption("No catalog match — add images in the sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    best = matches[0]
    best_percent = best.get("match_percent", 0)
    best_tier, best_color = tier_label(best.get("match_tier") or "none", best_percent)
    best_label = html.escape(best.get("name") or best.get("sku") or "Product")
    st.markdown(
        f'<div class="catalog-match-best" style="border-color:{best_color}">'
        f'<span class="catalog-match-best-label">Best match</span> '
        f'<strong style="color:{best_color}">{best_tier} · {best_percent}%</strong> '
        f"— {best_label}"
        f"</div>",
        unsafe_allow_html=True,
    )

    query_summary = (best.get("query_summary") or "").strip()
    if query_summary:
        st.caption(f"Query understood as: {query_summary}")

    best_path = best.get("image_path") or ""
    st.markdown(
        compare_images_html(query_image_bytes, best_path, max_height=260),
        unsafe_allow_html=True,
    )
    st.markdown(match_card_html(best, 1), unsafe_allow_html=True)

    for rank, match in enumerate(matches[1:max_results], start=2):
        st.markdown('<div class="catalog-match-divider"></div>', unsafe_allow_html=True)
        image_path = match.get("image_path") or ""
        if image_path and Path(image_path).exists():
            st.image(image_path, width=120)
        st.markdown(match_card_html(match, rank), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_match_card(match: dict, rank: int, query_bytes: bytes | None) -> None:
    query_summary = (match.get("query_summary") or "").strip()

    left, right = st.columns([1, 1.2])
    with left:
        if query_bytes:
            st.caption("Your photo")
            st.image(query_bytes, width=140)
        image_path = match.get("image_path") or ""
        if image_path and Path(image_path).exists():
            st.caption("Catalog reference")
            st.image(image_path, width=140)
    with right:
        st.markdown(match_card_html(match, rank), unsafe_allow_html=True)
    if rank == 1 and query_summary:
        st.caption(f"Query understood as: {query_summary}")


def render_component_vision_page(on_explain_match) -> None:
    """Fast visual match workspace (skips full agent graph for speed)."""
    catalog_count = get_component_image_count()

    st.markdown(
        '<div class="vision-hero">'
        "<h3>Component Vision</h3>"
        "<p>Hybrid matching: CLIP visual search + Claude vision re-rank + text KB. "
        f"<strong>{catalog_count}</strong> photo(s) in catalog.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if catalog_count == 0:
        st.warning("Add component photos in the sidebar first (**Add Component Photos**).")

    col_in, col_out = st.columns([1, 1.35], gap="large")

    with col_in:
        st.markdown("**1. Upload query photo**")
        query_file = st.file_uploader(
            "Query component photo",
            type=["png", "jpg", "jpeg", "webp"],
            key="vision_query_photo",
            label_visibility="collapsed",
        )
        query_bytes = query_file.getvalue() if query_file is not None else None
        if query_bytes:
            st.image(query_bytes, caption=getattr(query_file, "name", "Query"), width=200)

        run_match = st.button(
            "⚡ Run visual match",
            type="primary",
            disabled=not query_bytes or catalog_count == 0,
            use_container_width=True,
        )

        if run_match and query_bytes:
            with st.spinner("CLIP search → Claude re-rank…"):
                results = match_component_image_bytes(
                    query_bytes,
                    filename=getattr(query_file, "name", "query.jpg"),
                )
            st.session_state.last_match_results = results
            st.session_state.last_query_image = query_bytes
            st.session_state.last_query_name = getattr(query_file, "name", "")

        st.caption("Typical time: 3–8s depending on vision model.")

    with col_out:
        st.markdown("**2. Match results**")
        results = st.session_state.get("last_match_results") or []
        stored_query = st.session_state.get("last_query_image")

        if not results:
            st.markdown(
                '<div class="vision-empty">Upload a photo and run visual match to see ranked catalog hits.</div>',
                unsafe_allow_html=True,
            )
        else:
            best = results[0].get("match_percent", 0)
            st.success(f"Best match: **{best}%** — {results[0].get('name') or results[0].get('sku') or 'catalog hit'}")
            for index, match in enumerate(results[:5], start=1):
                _render_match_card(match, index, stored_query if index == 1 else None)
                st.divider()

            if st.button("💬 Explain best match with AI", use_container_width=True):
                on_explain_match(results, stored_query)
                st.info("Switch to the **Chat** tab to see the AI explanation.")
