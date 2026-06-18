import html
import base64
import streamlit as st
import os
from uuid import uuid4
from config import GALILEO_DEBUG_URLS
from agents.constants import SEND_CONFIRM_PHRASES
from services.conversation_service import (
    apply_agent_result_to_session,
    draft_has_recipient,
    get_chat_history,
    pending_component_matches,
    question_for_agent,
)
from services.component_match_service import kb_sources_for_matches
from services.vector_db_service import (
    add_text_documents,
    add_pdf_document,
    add_excel_document,
    add_csv_document,
    add_component_image,
    get_kb_sources,
    get_component_catalog,
    get_component_image_count,
    match_component_image_bytes,
    remove_component_image,
    remove_kb_source,
    reindex_kb_source,
)
from services.agent_service import ask_agent, confirm_send_email
from ui.component_vision import render_catalog_matches_panel
from observability import start_chat_session, get_console_links

APP_NAME = "Product Distribution Agent"

STYLE_BLOCK = """
<style>
    :root {
        --app-bg: #f4f6f5;
        --panel-bg: #ffffff;
        --card-bg: #ffffff;
        --border: #dfe6e3;
        --text-main: #111827;
        --text-muted: #6b7280;
        --brand: #0d9488;
        --brand-dark: #0f766e;
        --brand-light: #ecfdf5;
        --gtm: #0369a1;      --gtm-bg: #e0f2fe;
        --outreach: #0d9488; --outreach-bg: #ccfbf1;
        --crm: #6d28d9;      --crm-bg: #ede9fe;
    }
    .stApp { background: var(--app-bg) !important; color: var(--text-main) !important; }
    [data-testid="stAppViewContainer"] { background: var(--app-bg) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: var(--panel-bg) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text-main) !important; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 760px !important;
    }
    .stMarkdown, .stCaption, .stText, p, label, h1, h2, h3 { color: var(--text-main) !important; }
    .stTextInput input, .stTextArea textarea, .stChatInput textarea, [data-testid="stChatInput"] textarea {
        background: var(--card-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,.04) !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background: linear-gradient(to top, var(--app-bg) 80%, transparent) !important;
        border-top: none !important;
        padding: 0.75rem 0 1.25rem !important;
    }
    [data-testid="stChatInput"] { max-width: 100% !important; margin: 0 !important; }
    [data-testid="stChatInputContainer"] { background: transparent !important; }
    .stButton button {
        background: #ffffff !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    .stButton button:hover { border-color: var(--brand) !important; color: var(--brand-dark) !important; }
    .stButton button[kind="primary"] {
        background: var(--brand-dark) !important;
        color: #ffffff !important;
        border: 1px solid var(--brand-dark) !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.35rem 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        margin-bottom: 0.5rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
    }
    [data-testid="stExpander"] details {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: #fff !important;
    }
    [data-testid="stFileUploader"] {
        padding: 0 !important;
    }

    .composer-attach { margin-bottom: 0.5rem; }
    .attach-chip {
        display: inline-block;
        background: var(--brand-light); border: 1px solid #99f6e4;
        border-radius: 999px; padding: 4px 12px;
        font-size: 0.75rem; color: var(--brand-dark);
        margin-bottom: 0.35rem;
    }

    .app-hero { text-align: center; margin: 0.5rem 0 1.25rem; }
    .app-hero-title {
        font-size: 1.5rem; font-weight: 650; letter-spacing: -0.035em;
        color: var(--text-main); margin-bottom: 0.3rem;
    }
    .app-hero-sub { font-size: 0.875rem; color: var(--text-muted); font-weight: 400; }
    .welcome-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        margin: 0 0 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }
    .welcome-card p { margin: 0; color: var(--text-muted); font-size: 0.9rem; line-height: 1.55; }
    .sidebar-brand { font-size: 1rem; font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
    .sidebar-tag { font-size: 0.72rem; color: var(--text-muted); margin-bottom: 0.65rem; line-height: 1.4; }

    .agent-badge {
        display: inline-block; font-size: 10px; font-weight: 600;
        padding: 2px 8px; border-radius: 5px; margin-bottom: 8px;
        letter-spacing: 0.04em; text-transform: uppercase;
    }
    .badge-gtm { background: var(--gtm-bg); color: var(--gtm); }
    .badge-outreach { background: var(--outreach-bg); color: var(--outreach); }
    .badge-crm { background: var(--crm-bg); color: var(--crm); }

    .trace-step {
        background: #f9fafb; color: var(--text-muted);
        border-left: 2px solid var(--border);
        padding: 4px 10px; margin: 2px 0; border-radius: 0 6px 6px 0;
        font-family: ui-monospace, monospace; font-size: 11px;
    }
    .kb-source {
        background: #f9fafb; border: 1px solid var(--border);
        border-radius: 8px; padding: 8px 10px; margin: 4px 0; font-size: 12px;
    }
    .kb-source-title { font-weight: 600; margin-bottom: 2px; font-size: 12px; }
    .kb-source-meta { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
    .kb-source-excerpt { font-size: 12px; line-height: 1.45; color: var(--text-muted); }

    .stat-card {
        background: #f9fafb; border: 1px solid var(--border);
        border-radius: 10px; padding: 9px 6px; text-align: center;
    }
    .stat-number { font-size: 1.25rem; font-weight: 700; color: var(--brand-dark); line-height: 1.1; }
    .stat-label { font-size: 9px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.07em; }

    .vision-match-card {
        background: #ffffff; border: 1px solid var(--border);
        border-radius: 10px; padding: 10px 12px;
    }
    .vision-match-rank {
        font-size: 10px; font-weight: 700; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px;
    }
    .vision-match-title { font-size: 14px; font-weight: 600; margin: 2px 0; }
    .vision-match-tier { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
    .vision-match-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
    .vision-scores {
        font-size: 11px; color: var(--text-muted); font-family: ui-monospace, monospace;
        margin: 5px 0 2px;
    }
    .vision-reason {
        font-size: 12px; color: var(--text-main); line-height: 1.45;
        margin-top: 6px; padding-top: 6px; border-top: 1px dashed var(--border);
    }
    .conf-track {
        height: 5px; background: #e5e7eb; border-radius: 999px;
        overflow: hidden; margin: 6px 0 4px;
    }
    .conf-fill { height: 100%; border-radius: 999px; }
    .catalog-match-panel {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 14px;
        margin: 0.65rem 0 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }
    .catalog-match-header {
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: var(--brand-dark); margin-bottom: 8px;
    }
    .catalog-match-best {
        background: #ffffff; border: 1px solid var(--border);
        border-left-width: 3px; border-radius: 8px;
        padding: 8px 10px; margin-bottom: 10px; font-size: 0.84rem;
    }
    .catalog-match-best-label {
        color: var(--text-muted); font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .catalog-match-divider {
        height: 1px; background: var(--border); margin: 14px 0 12px;
    }
    .catalog-compare {
        display: flex;
        align-items: stretch;
        justify-content: center;
        gap: 14px;
        margin: 14px 0 16px;
    }
    .compare-side {
        flex: 1 1 0;
        min-width: 0;
        max-width: 320px;
        text-align: center;
    }
    .compare-label {
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.07em; color: var(--text-muted); margin-bottom: 8px;
    }
    .compare-frame {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        min-height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }
    .compare-frame img {
        max-width: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
        border-radius: 6px;
    }
    .compare-placeholder {
        color: var(--text-muted);
        font-size: 0.8rem;
        padding: 2rem 1rem;
    }
    .compare-vs {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 36px;
        padding-top: 1.6rem;
    }
    .compare-vs-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px; height: 32px;
        border-radius: 999px;
        background: var(--brand-light);
        border: 1px solid #99f6e4;
        color: var(--brand-dark);
        font-size: 0.85rem;
        font-weight: 700;
    }

    .send-banner {
        background: var(--brand-light); border: 1px solid #99f6e4;
        border-radius: 10px; padding: 10px 12px; margin: 6px 0;
        font-size: 0.82rem; color: var(--brand-dark);
    }
</style>
"""

BADGES = {
    "gtm": ("Catalog & pricing", "badge-gtm"),
    "outreach": ("Customer outreach", "badge-outreach"),
    "crm": ("Salesforce CRM", "badge-crm"),
}
SEND_WORDS = list(SEND_CONFIRM_PHRASES)


def _draft_has_recipient(text: str) -> bool:
    return draft_has_recipient(text)


def _get_chat_history() -> list[dict]:
    return get_chat_history(st.session_state.messages)


def _pending_component_matches() -> list[dict]:
    return pending_component_matches(
        st.session_state.messages,
        st.session_state.get("pending_component_matches"),
    )


def _question_for_agent(prompt: str) -> str:
    return question_for_agent(
        prompt,
        awaiting_email=st.session_state.awaiting_email,
        pricing_question=st.session_state.pricing_question,
        pending_drafts=st.session_state.pending_drafts,
    )


def _update_session_from_result(prompt: str, result: dict) -> None:
    apply_agent_result_to_session(st.session_state, prompt, result)


def _kb_sources_for_matches(matches: list[dict]) -> list[dict]:
    return kb_sources_for_matches(matches)


def apply_styles() -> None:
    st.markdown(STYLE_BLOCK, unsafe_allow_html=True)


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("awaiting_email", False)
    st.session_state.setdefault("pricing_question", "")
    st.session_state.setdefault("pending_drafts", "")
    st.session_state.setdefault("galileo_session_started", False)
    st.session_state.setdefault("galileo_debug_links_shown", False)
    st.session_state.setdefault("ui_streamlit_session_id", uuid4().hex[:8])
    st.session_state.setdefault("last_match_results", [])
    st.session_state.setdefault("last_query_image", None)
    st.session_state.setdefault("last_query_name", "")
    st.session_state.setdefault("chat_attach_key", 0)
    st.session_state.setdefault("pending_component_matches", [])
    if not st.session_state.messages:
        st.session_state.galileo_session_started = False
        st.session_state.galileo_debug_links_shown = False


def _reset_chat_state() -> None:
    st.session_state.messages = []
    st.session_state.awaiting_email = False
    st.session_state.pricing_question = ""
    st.session_state.pending_drafts = ""
    st.session_state.galileo_session_started = False
    st.session_state.galileo_debug_links_shown = False
    st.session_state.ui_streamlit_session_id = uuid4().hex[:8]
    st.session_state.last_match_results = []
    st.session_state.last_query_image = None
    st.session_state.last_query_name = ""
    st.session_state.chat_attach_key = 0
    st.session_state.pending_component_matches = []


def _show_galileo_debug_links_once() -> None:
    if not GALILEO_DEBUG_URLS or st.session_state.galileo_debug_links_shown:
        return
    links = get_console_links()
    if not links:
        return
    st.info(
        "Galileo Links\n\n"
        f"- Project: {links['project_url']}\n"
        f"- Log Stream: {links['log_stream_url']}"
    )
    st.session_state.galileo_debug_links_shown = True


def _render_trace(steps: list[str]) -> None:
    if not steps:
        return
    with st.expander("Details", expanded=False):
        for index, step in enumerate(steps, start=1):
            st.markdown(
                f'<div class="trace-step">{step}</div>',
                unsafe_allow_html=True,
            )


def _render_component_matches(matches: list[dict], query_image_bytes: bytes | None = None) -> None:
    render_catalog_matches_panel(matches, query_image_bytes)


def _render_kb_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            name = html.escape(source.get("source") or "(unknown)")
            doc_type = html.escape(source.get("type") or "text")
            score = source.get("score", 0)
            excerpt = html.escape(source.get("excerpt") or "")
            st.markdown(
                f'<div class="kb-source">'
                f'<div class="kb-source-title">{index}. {name}</div>'
                f'<div class="kb-source-meta">{doc_type} · relevance score {score:.3f}</div>'
                f'<div class="kb-source-excerpt">{excerpt}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _salesforce_ready() -> bool:
    try:
        from services.salesforce_client import is_salesforce_configured

        return is_salesforce_configured()
    except Exception:
        return False


def _env_set(name: str, placeholder_prefix: str = "your-") -> bool:
    value = (os.getenv(name) or "").strip()
    return bool(value) and not value.startswith(placeholder_prefix)


def _stat_card(number, label: str) -> str:
    return (
        f'<div class="stat-card"><div class="stat-number">{number}</div>'
        f'<div class="stat-label">{label}</div></div>'
    )


def _render_stats(doc_count: int) -> None:
    component_count = get_component_image_count()
    col1, col2 = st.columns(2)
    col1.markdown(_stat_card(doc_count, "Docs"), unsafe_allow_html=True)
    col2.markdown(_stat_card(component_count, "Images"), unsafe_allow_html=True)


def _render_data_panel() -> None:
    with st.expander("Add catalog data", expanded=False):
        tab_docs, tab_images = st.tabs(["Documents", "Images"])

        with tab_docs:
            text_input = st.text_area(
                "Paste product info",
                height=100,
                placeholder="Specs, pricing, MOQ, lead times…",
                label_visibility="collapsed",
            )
            if st.button("Add text", use_container_width=True, key="add_text_btn"):
                if text_input.strip():
                    try:
                        with st.spinner("Indexing…"):
                            count = add_text_documents(text_input)
                        st.success(f"Added {count} chunks")
                        st.rerun()
                    except Exception as error:
                        st.error(str(error))
            quotation_file = st.file_uploader(
                "PDF / Excel quotation",
                type=["pdf", "xlsx", "csv"],
                label_visibility="collapsed",
                key="quotation_upload",
            )
            if quotation_file and st.button("Upload quotation", use_container_width=True, key="add_quote_btn"):
                try:
                    with st.spinner("Processing…"):
                        suffix = quotation_file.name.rsplit(".", 1)[-1].lower()
                        if suffix == "pdf":
                            count = add_pdf_document(quotation_file)
                        elif suffix == "csv":
                            count = add_csv_document(quotation_file)
                        else:
                            count = add_excel_document(quotation_file)
                    st.success(f"Added {count} chunks") if count else st.warning("No rows found.")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

        with tab_images:
            col1, col2 = st.columns(2)
            sku = col1.text_input("SKU", key="catalog_component_sku", placeholder="SKU")
            name = col2.text_input("Name", key="catalog_component_name", placeholder="Product name")
            photos = st.file_uploader(
                "Images",
                type=["png", "jpg", "jpeg", "webp"],
                key="catalog_component_photo",
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if photos and st.button("Add to catalog", use_container_width=True, type="primary", key="add_catalog_btn"):
                try:
                    for photo in photos:
                        add_component_image(photo, sku=sku, name=name)
                    st.success(f"Added {len(photos)} photo(s)")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
            try:
                catalog = get_component_catalog()
            except Exception:
                catalog = []
            if catalog:
                st.caption(f"{len(catalog)} in catalog")
                for index, item in enumerate(catalog[:5]):
                    label = item.get("sku") or item.get("name") or "Item"
                    c1, c2 = st.columns([3, 1])
                    c1.caption(label)
                    if c2.button("Remove", key=f"del_part_{index}"):
                        remove_component_image(item.get("image_id") or "")
                        st.rerun()
                if len(catalog) > 5:
                    st.caption(f"+ {len(catalog) - 5} more")


def _render_manage_panel() -> None:
    with st.expander("Manage indexed data", expanded=False):
        try:
            sources = get_kb_sources()
        except Exception as error:
            st.error(f"Could not load sources: {error}")
            return
        if not sources:
            st.caption("No indexed sources yet. Upload docs above.")
            return
        for index, item in enumerate(sources):
            source = item["source"]
            chunks = item["chunks"]
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.caption(f"**{source}** — {chunks} chunks")
            if col2.button("Re-index", key=f"reindex_{index}", use_container_width=True):
                try:
                    with st.spinner(f"Re-indexing {source}..."):
                        count = reindex_kb_source(source)
                    st.success(f"Re-indexed {count} chunks for {source}")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
            if col3.button("Delete", key=f"delete_{index}", use_container_width=True):
                try:
                    with st.spinner(f"Deleting {source}..."):
                        removed = remove_kb_source(source)
                    st.success(f"Removed {removed} chunks from {source}")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))


def _render_how_it_works() -> None:
    with st.expander("About", expanded=False):
        st.markdown("""
**Product Distribution Agent** handles catalog lookup, customer emails, and Salesforce CRM in one conversation.

Upload documents and catalog images from the sidebar to get started.
""")


def render_empty_welcome(doc_count: int, catalog_count: int) -> None:
    if doc_count > 0 or catalog_count > 0:
        st.markdown(
            '<div class="welcome-card"><p>Ask about products, inventory, customers, or CRM.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="welcome-card"><p>Upload product documents or catalog images in the sidebar, '
            "then start a conversation below.</p></div>",
            unsafe_allow_html=True,
        )


def render_sidebar(doc_count: int) -> None:
    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand">{APP_NAME}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-tag">Distribution · catalog · CRM</div>',
            unsafe_allow_html=True,
        )
        if st.button("New conversation", use_container_width=True):
            _reset_chat_state()
            st.rerun()
        st.divider()
        _render_stats(doc_count)
        if st.session_state.get("pending_drafts") and _draft_has_recipient(st.session_state.pending_drafts):
            st.markdown(
                '<div class="send-banner">Email draft ready — confirm below the chat.</div>',
                unsafe_allow_html=True,
            )
        _render_data_panel()
        _render_manage_panel()
        _render_how_it_works()
        if _salesforce_ready() or _env_set("BREVO_API_KEY"):
            st.caption("Connected: " + ", ".join(
                name for name, ok in [
                    ("Salesforce", _salesforce_ready()),
                    ("Email", _env_set("BREVO_API_KEY")),
                    ("Catalog", _env_set("QDRANT_URL")),
                ] if ok
            ))


def _execute_confirm_send() -> None:
    draft = st.session_state.get("pending_drafts", "")
    if not draft or not _draft_has_recipient(draft):
        return
    try:
        with st.spinner("Sending via Brevo..."):
            result = confirm_send_email(
                draft,
                chat_history=_get_chat_history(),
                component_matches=_pending_component_matches() or None,
            )
        st.session_state.pending_drafts = ""
        st.session_state.pending_component_matches = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "agent": "outreach",
            "trace": result.get("steps", []),
        })
        st.rerun()
    except Exception as error:
        st.error(f"Send failed: {error}")


def render_pending_send() -> None:
    draft = st.session_state.get("pending_drafts", "")
    if not draft or not _draft_has_recipient(draft):
        return
    if st.button("Send email", type="primary", key="confirm_send_email", use_container_width=True):
        _execute_confirm_send()


def render_chat_history() -> None:
    last_query_image_b64 = ""
    for message in st.session_state.messages:
        if message.get("role") == "user" and message.get("query_image_b64"):
            last_query_image_b64 = message["query_image_b64"]

        with st.chat_message(message["role"]):
            agent_type = message.get("agent")
            if agent_type:
                label, css_class = BADGES.get(agent_type, (agent_type, "badge-gtm"))
                st.markdown(f'<span class="agent-badge {css_class}">{label}</span>', unsafe_allow_html=True)
            st.markdown(message["content"])
            if message.get("query_image_b64"):
                try:
                    st.image(base64.standard_b64decode(message["query_image_b64"]), width=200)
                except Exception:
                    pass
            if message.get("component_matches"):
                query_bytes = None
                if last_query_image_b64:
                    try:
                        query_bytes = base64.standard_b64decode(last_query_image_b64)
                    except Exception:
                        query_bytes = None
                _render_component_matches(message["component_matches"], query_bytes)
            _render_kb_sources(message.get("kb_sources", []))
            _render_trace(message.get("trace", []))


def _push_assistant_message(result: dict) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "agent": result.get("agent_type", "gtm"),
            "trace": result.get("steps", []),
            "kb_sources": result.get("kb_sources", []),
            "component_matches": result.get("component_matches", []),
        }
    )


def render_chat_composer() -> tuple[str, bytes | None, str]:
    attach_key = st.session_state.get("chat_attach_key", 0)
    image_bytes: bytes | None = None
    image_name = ""

    with st.container():
        st.markdown('<div class="composer-attach">', unsafe_allow_html=True)
        with st.expander("Attach image", expanded=False):
            attach = st.file_uploader(
                "Upload",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"chat_attach_photo_{attach_key}",
                label_visibility="collapsed",
            )
            if attach is not None:
                image_bytes = attach.getvalue()
                image_name = getattr(attach, "name", "") or ""
                if image_bytes:
                    c1, c2 = st.columns([1, 3])
                    c1.image(image_bytes, width=64)
                    if c2.button("Remove", key="clear_chat_attach"):
                        st.session_state.chat_attach_key = attach_key + 1
                        st.rerun()
        if image_bytes:
            st.markdown(
                f'<div class="attach-chip">Image ready · {html.escape(image_name[:28])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    typed = st.chat_input("Message Product Distribution Agent…")
    prompt = typed or st.session_state.pop("queued_prompt", "")
    return prompt, image_bytes, image_name


def handle_new_prompt(
    prompt: str,
    *,
    query_image_bytes: bytes | None = None,
    query_image_name: str = "",
) -> None:
    if not st.session_state.messages and not st.session_state.galileo_session_started:
        session_name = f"UI Streamlit {st.session_state.ui_streamlit_session_id}"
        # Start one Galileo session per fresh chat thread from the UI.
        st.session_state.galileo_session_started = start_chat_session(session_name)

    query_image_b64 = ""
    if query_image_bytes:
        query_image_b64 = base64.standard_b64encode(query_image_bytes).decode("ascii")

    user_message = {"role": "user", "content": prompt}
    if query_image_b64:
        user_message["query_image_b64"] = query_image_b64
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)
        if query_image_bytes:
            st.image(query_image_bytes, width=160)

    with st.chat_message("assistant"):
        question = _question_for_agent(prompt)
        precomputed_matches: list[dict] | None = None
        precomputed_kb: list[dict] | None = None
        if query_image_bytes:
            with st.spinner("Matching image…"):
                precomputed_matches = match_component_image_bytes(
                    query_image_bytes,
                    filename=query_image_name or "query.jpg",
                )
                precomputed_kb = _kb_sources_for_matches(precomputed_matches)

        try:
            with st.spinner("Thinking…"):
                result = ask_agent(
                    question,
                    chat_history=_get_chat_history(),
                    query_image_b64=query_image_b64 or None,
                    component_matches=precomputed_matches,
                    kb_sources=precomputed_kb,
                )
        except Exception as error:
            error_text = str(error)
            st.error(f"Setup issue: {error_text}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Setup issue: {error_text}",
                "agent": "gtm",
                "trace": ["App Error → configuration needed"],
            })
            return

        _update_session_from_result(prompt, result)

        label, css_class = BADGES.get(result.get("agent_type", "gtm"), ("gtm", "badge-gtm"))
        st.markdown(f'<span class="agent-badge {css_class}">{label}</span>', unsafe_allow_html=True)
        _show_galileo_debug_links_once()
        st.markdown(result.get("answer", ""))
        matches = result.get("component_matches") or precomputed_matches or []
        _render_component_matches(matches, query_image_bytes)
        _render_kb_sources(result.get("kb_sources") or precomputed_kb or [])
        if result.get("send_intent") and not result.get("send_confirmed"):
            st.caption("Review the draft, then click **Send email**.")
        _render_trace(result.get("steps", []))

    assistant_message = dict(result)
    if precomputed_matches and not assistant_message.get("component_matches"):
        assistant_message["component_matches"] = precomputed_matches
    if precomputed_kb and not assistant_message.get("kb_sources"):
        assistant_message["kb_sources"] = precomputed_kb
    _push_assistant_message(assistant_message)
