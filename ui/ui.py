import streamlit as st
from datetime import datetime, UTC
import os
from uuid import uuid4
from services.vector_db_service import add_text_documents, add_pdf_document, add_excel_document, add_csv_document
from services.agent_service import ask_agent, load_graph_image, load_graph_ascii
from observability import start_chat_session, get_console_links


STYLE_BLOCK = """
<style>
    :root {
        --app-bg: #ffffff;
        --panel-bg: #f7f7f7;
        --card-bg: #ffffff;
        --border: #d9d9d9;
        --text-main: #111111;
        --text-muted: #4b4b4b;
        --brand: #111111;
    }
    .stApp {
        background: var(--app-bg) !important;
        color: var(--text-main) !important;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--app-bg) !important;
    }
    [data-testid="stHeader"] {
        background: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background: var(--panel-bg) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }
    .stMarkdown, .stCaption, .stText, p, label, h1, h2, h3 {
        color: var(--text-main) !important;
    }
    .stTextInput input, .stTextArea textarea, .stChatInput textarea, [data-testid="stChatInput"] textarea {
        background: var(--card-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background: #ffffff !important;
        border-top: 1px solid var(--border);
    }
    [data-testid="stChatInput"] {
        max-width: 100% !important;
        margin: 0 !important;
    }
    [data-testid="stChatInputContainer"] {
        background: #ffffff !important;
    }
    .stButton button {
        background: #ffffff !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    .stAlert {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] {
        background: var(--panel-bg);
    }
    .agent-badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 8px;
    }
    .badge-gtm { background: #eeeeee; color: #111111; }
    .badge-outreach { background: #eeeeee; color: #111111; }
    .trace-step {
        background: var(--card-bg);
        color: var(--text-main);
        border-left: 3px solid var(--brand);
        border: 1px solid var(--border);
        padding: 8px 14px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
        font-size: 13px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: var(--text-muted);
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: none;
    }
    .stat-number { font-size: 28px; font-weight: 700; color: var(--brand); }
    .stat-label {
        font-size: 12px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
"""

BADGES = {
    "gtm": ("🎯 GTM Agent", "badge-gtm"),
    "outreach": ("📝 Outreach Agent", "badge-outreach"),
}
SEND_WORDS = ["send", "market", "deliver", "mail them", "email them"]


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


def _show_galileo_debug_links_once() -> None:
    debug_enabled = os.getenv("GALILEO_DEBUG_URLS", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not debug_enabled or st.session_state.galileo_debug_links_shown:
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
    with st.expander("🔍 Pipeline Trace"):
        for index, step in enumerate(steps, start=1):
            st.markdown(
                f'<div class="trace-step">Step {index}: {step}</div>',
                unsafe_allow_html=True,
            )


def _render_stats(doc_count: int) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{doc_count}</div>
                <div class="stat-label">Docs in DB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">3</div>
                <div class="stat-label">Agents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_doc_upload() -> None:
    with st.expander("📄 Add Knowledge Base Docs", expanded=False):
        text_input = st.text_area(
            "Paste product docs or quotation notes:",
            height=140,
            placeholder="Paste product descriptions, pricing notes, or policy text...",
        )
        if st.button("➕ Add Text", use_container_width=True):
            if text_input.strip():
                try:
                    with st.spinner("Embedding..."):
                        count = add_text_documents(text_input)
                    st.success(f"Added {count} chunks!")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not add text documents: {error}")
            else:
                st.warning("Paste some text first.")

        st.caption("Upload price quotations — each product row becomes its own searchable chunk.")
        quotation_file = st.file_uploader(
            "Upload PDF or Excel quotation",
            type=["pdf", "xlsx", "csv"],
            label_visibility="collapsed",
        )
        if quotation_file and st.button("📎 Add Quotation", use_container_width=True):
            try:
                with st.spinner("Processing quotation..."):
                    suffix = quotation_file.name.rsplit(".", 1)[-1].lower()
                    if suffix == "pdf":
                        count = add_pdf_document(quotation_file)
                    elif suffix == "csv":
                        count = add_csv_document(quotation_file)
                    else:
                        count = add_excel_document(quotation_file)
                if count:
                    st.success(f"Added {count} chunks!")
                else:
                    st.warning("No product rows found in that file.")
                st.rerun()
            except Exception as error:
                st.error(f"Could not add quotation: {error}")


def _render_graph() -> None:
    with st.expander("🗺️ Agent Graph", expanded=False):
        img_bytes = load_graph_image()
        if img_bytes:
            st.image(img_bytes)
        else:
            try:
                st.caption("PNG unavailable, showing graph structure:")
                st.code(load_graph_ascii(), language="text")
            except Exception:
                st.caption("Graph structure:")
                st.code(
                    "classify ─┬─ gtm_retrieve → pricing_gate ─┬─ gtm_generate → END\n"
                    "          │                              └─ collect_email → gtm_generate / END\n"
                    "          └─ outreach_research → outreach_generate → send_gate ─┬─ END (review)\n"
                    "                                                                 └─ outreach_send → END",
                    language="text",
                )


def _render_how_it_works() -> None:
    with st.expander("🤖 How it works", expanded=False):
        st.markdown("""
**Supervisor Routing Agent** classifies your message:
- Product / pricing question → **GTM Agent**
- Content creation / lead research request → **Outreach Agent**

**GTM Agent** 🎯
- Searches product docs + live web for answers
- Gates pricing details behind email verification
- Uses: `search_knowledge_base`, `web_search`

**Outreach Agent** 📝
- Researches product context and audience
- Finds leads via Apollo (job title, industry)
- Creates LinkedIn posts, emails, marketing copy
- Can send emails via Brevo when you ask to send
- Uses: `search_knowledge_base`, `web_search`, `apollo_search`, `send_email`

**All 4 tools:**
- `search_knowledge_base` (Qdrant) — product docs from your knowledge base
- `web_search` (DuckDuckGo) — live web for market/competitor info
- `apollo_search` (Apollo.io) — find leads by job title, location, industry
- `send_email` (Brevo) — deliver personalized emails
""")


def render_sidebar(doc_count: int) -> None:
    with st.sidebar:
        st.markdown("## 🚀 Galileo Marketing AI")
        st.caption("Multi-Agent RAG for GTM & Outreach")
        if st.button("🆕 New Chat", use_container_width=True):
            _reset_chat_state()
            st.rerun()
        st.divider()
        _render_stats(doc_count)
        st.divider()
        _render_doc_upload()
        _render_graph()
        _render_how_it_works()
        st.divider()
        st.caption("Galileo AI · LangGraph · Qdrant · OpenAI")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            agent_type = message.get("agent")
            if agent_type:
                label, css_class = BADGES.get(agent_type, (agent_type, "badge-gtm"))
                st.markdown(f'<span class="agent-badge {css_class}">{label}</span>', unsafe_allow_html=True)
            st.markdown(message["content"])
            _render_trace(message.get("trace", []))


def _build_agent_question(prompt: str) -> str:
    if st.session_state.awaiting_email:
        return f"{st.session_state.pricing_question} My email is {prompt}"
    if st.session_state.pending_drafts and any(word in prompt.lower() for word in SEND_WORDS):
        return (
            f"{prompt}\n\n"
            "Here are the drafted emails to send:\n"
            f"{st.session_state.pending_drafts}"
        )
    return prompt


def _update_session_from_result(prompt: str, result: dict) -> None:
    if result.get("is_pricing") and not result.get("user_email"):
        st.session_state.awaiting_email = True
        if not st.session_state.pricing_question:
            st.session_state.pricing_question = prompt
    else:
        st.session_state.awaiting_email = False
        st.session_state.pricing_question = ""

    if result.get("agent_type") == "outreach" and result.get("answer"):
        st.session_state.pending_drafts = result["answer"]


def _push_assistant_message(result: dict) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "agent": result.get("agent_type", "gtm"),
            "trace": result.get("steps", []),
        }
    )


def handle_new_prompt(prompt: str) -> None:
    if not st.session_state.messages and not st.session_state.galileo_session_started:
        session_name = f"UI Streamlit {st.session_state.ui_streamlit_session_id}"
        # Start one Galileo session per fresh chat thread from the UI.
        st.session_state.galileo_session_started = start_chat_session(session_name)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        question_for_agent = _build_agent_question(prompt)
        try:
            with st.spinner("🔄 Routing to the right agent..."):
                result = ask_agent(question_for_agent)
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
        _render_trace(result.get("steps", []))

    _push_assistant_message(result)
