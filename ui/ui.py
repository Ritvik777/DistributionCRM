import re
import streamlit as st
import os
from uuid import uuid4
from services.vector_db_service import (
    add_text_documents,
    add_pdf_document,
    add_excel_document,
    add_csv_document,
    get_kb_sources,
    remove_kb_source,
    reindex_kb_source,
)
from services.agent_service import ask_agent, confirm_send_email, load_graph_image, load_graph_ascii
from observability import start_chat_session, get_console_links

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


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
    .badge-crm { background: #eeeeee; color: #111111; }
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
    "crm": ("🗂️ CRM Agent", "badge-crm"),
}
SEND_WORDS = ["send it", "send now", "go ahead and send", "yes send", "please send"]
MAX_HISTORY_MESSAGES = 10


def _draft_has_recipient(text: str) -> bool:
    return bool(EMAIL_PATTERN.search(text or ""))


def _get_chat_history() -> list[dict]:
    history: list[dict] = []
    for msg in st.session_state.messages[:-1][-MAX_HISTORY_MESSAGES:]:
        entry: dict = {"role": msg["role"], "content": msg.get("content", "")}
        if msg.get("agent"):
            entry["agent"] = msg["agent"]
        history.append(entry)
    return history


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
                <div class="stat-number">4</div>
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


def _render_kb_manage() -> None:
    with st.expander("📚 Manage Knowledge Base", expanded=False):
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
- Write/send content or find new prospects → **Outreach Agent**
- Anything in Salesforce/CRM → **CRM Agent**

**GTM Agent** 🎯
- Searches product docs + live web for answers
- Gates pricing details behind email verification
- Uses: `search_knowledge_base`, `web_search`

**Outreach Agent** 📝
- Researches product context and audience
- Finds new leads via Apollo (job title, industry)
- Creates LinkedIn posts, emails, marketing copy
- Can send emails via Brevo after you **confirm** in the UI (review draft first)
- Auto-logs a Task in Salesforce after a successful send
- Uses: `search_knowledge_base`, `web_search`, `apollo_search`, `salesforce_search_leads`, `send_email`

**CRM Agent** 🗂️
- Fetch / list / search Leads, Contacts, Accounts, Opportunities
- SOQL + aggregate (GROUP BY / COUNT) queries
- Create / update / delete records; upsert leads
- Describe objects, search objects
- Read / write / execute **Apex** (via your TypeScript MCP server)
- Uses: `salesforce_query_records`, `salesforce_aggregate_query`, `salesforce_dml_records`, `salesforce_describe_object`, `salesforce_read_apex`, `salesforce_write_apex`, `salesforce_execute_anonymous`
""")


def render_sidebar(doc_count: int) -> None:
    with st.sidebar:
        st.markdown("## 🚀 Product Marketing")
        st.caption("Multi-Agent RAG for GTM & Outreach")
        if st.button("🆕 New Chat", use_container_width=True):
            _reset_chat_state()
            st.rerun()
        st.divider()
        _render_stats(doc_count)
        if st.session_state.get("pending_drafts") and _draft_has_recipient(st.session_state.pending_drafts):
            st.warning("📧 Email draft ready — use **Confirm & Send Email** in the chat to send via Brevo.")
        st.divider()
        _render_doc_upload()
        _render_kb_manage()
        _render_graph()
        _render_how_it_works()
        st.divider()
        st.caption("LangGraph · Qdrant · Anthropic · Brevo · Salesforce")


def _execute_confirm_send() -> None:
    draft = st.session_state.get("pending_drafts", "")
    if not draft or not _draft_has_recipient(draft):
        return
    try:
        with st.spinner("Sending via Brevo..."):
            result = confirm_send_email(draft, chat_history=_get_chat_history())
        st.session_state.pending_drafts = ""
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "agent": "outreach",
            "trace": result.get("steps", []),
        })
        st.rerun()
    except Exception as error:
        st.error(f"Send failed: {error}")


def _render_confirm_send_button(key_suffix: str) -> None:
    draft = st.session_state.get("pending_drafts", "")
    if not draft or not _draft_has_recipient(draft):
        return
    if st.button(
        "✅ Confirm & Send Email",
        type="primary",
        key=f"confirm_send_email_{key_suffix}",
        use_container_width=True,
    ):
        _execute_confirm_send()


def render_pending_send() -> None:
    draft = st.session_state.get("pending_drafts", "")
    if not draft or not _draft_has_recipient(draft):
        return
    st.info("📧 **Outreach draft ready.** Review the email above, then confirm to send via Brevo.")
    _render_confirm_send_button("main")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            agent_type = message.get("agent")
            if agent_type:
                label, css_class = BADGES.get(agent_type, (agent_type, "badge-gtm"))
                st.markdown(f'<span class="agent-badge {css_class}">{label}</span>', unsafe_allow_html=True)
            st.markdown(message["content"])
            _render_trace(message.get("trace", []))


def _question_for_agent(prompt: str) -> str:
    if st.session_state.awaiting_email:
        return f"{st.session_state.pricing_question} My email is {prompt}"
    if st.session_state.pending_drafts and any(word in prompt.lower() for word in SEND_WORDS):
        return (
            f"{prompt}\n\nUse this draft:\n{st.session_state.pending_drafts}"
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

    if result.get("send_confirmed"):
        st.session_state.pending_drafts = ""
    elif result.get("agent_type") == "outreach" and result.get("answer"):
        answer = result["answer"]
        if _draft_has_recipient(answer) and not answer.strip().startswith("✅ **Sent to:**"):
            st.session_state.pending_drafts = answer


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
        question = _question_for_agent(prompt)
        try:
            with st.spinner("🔄 Routing to the right agent..."):
                result = ask_agent(question, chat_history=_get_chat_history())
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
        if result.get("send_intent") and not result.get("send_confirmed"):
            st.caption("💡 When you're happy with the draft, click **Confirm & Send Email** below.")
        _render_trace(result.get("steps", []))

    _push_assistant_message(result)
