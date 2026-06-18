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
from observability import start_chat_session, get_console_links, is_galileo_enabled

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


STYLE_BLOCK = """
<style>
    :root {
        --app-bg: #ffffff;
        --panel-bg: #fbfbfd;
        --card-bg: #ffffff;
        --border: #e6e6ec;
        --text-main: #14141a;
        --text-muted: #6b6b78;
        --brand: #4f46e5;
        --gtm: #2563eb;     --gtm-bg: #e7efff;
        --outreach: #047857; --outreach-bg: #def5ea;
        --crm: #7c3aed;      --crm-bg: #efe6fe;
    }
    .stApp { background: var(--app-bg) !important; color: var(--text-main) !important; }
    [data-testid="stAppViewContainer"] { background: var(--app-bg) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: var(--panel-bg) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text-main) !important; }
    .stMarkdown, .stCaption, .stText, p, label, h1, h2, h3 { color: var(--text-main) !important; }
    .stTextInput input, .stTextArea textarea, .stChatInput textarea, [data-testid="stChatInput"] textarea {
        background: var(--card-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stBottomBlockContainer"] { background: #ffffff !important; border-top: 1px solid var(--border); }
    [data-testid="stChatInput"] { max-width: 100% !important; margin: 0 !important; }
    [data-testid="stChatInputContainer"] { background: #ffffff !important; }
    .stButton button {
        background: #ffffff !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        transition: border-color .15s ease, box-shadow .15s ease;
    }
    .stButton button:hover { border-color: var(--brand) !important; box-shadow: 0 1px 6px rgba(79,70,229,.12) !important; }
    .stButton button[kind="primary"] {
        background: var(--brand) !important;
        color: #ffffff !important;
        border: 1px solid var(--brand) !important;
    }
    .stAlert { border: 1px solid var(--border) !important; border-radius: 12px !important; }
    [data-testid="stChatMessage"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 6px 14px;
        box-shadow: 0 1px 2px rgba(20,20,26,.03);
    }

    .agent-badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 8px;
        letter-spacing: .2px;
    }
    .badge-gtm { background: var(--gtm-bg); color: var(--gtm); }
    .badge-outreach { background: var(--outreach-bg); color: var(--outreach); }
    .badge-crm { background: var(--crm-bg); color: var(--crm); }

    .trace-step {
        background: #fafafe;
        color: var(--text-main);
        border-left: 3px solid var(--brand);
        padding: 7px 12px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 12.5px;
    }
    .hero-title { font-size: 30px; font-weight: 800; letter-spacing: -.5px; margin-bottom: 2px; }
    .hero-subtitle { font-size: 14px; color: var(--text-muted); margin-top: 0; margin-bottom: 18px; }

    .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 10px;
        text-align: center;
    }
    .stat-number { font-size: 24px; font-weight: 800; color: var(--brand); line-height: 1.1; }
    .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .8px; }

    .chip {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 12px; font-weight: 600;
        padding: 4px 10px; margin: 3px 4px 3px 0;
        border-radius: 999px; border: 1px solid var(--border); background: #fff;
    }
    .chip .dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
    .dot-on { background: #10b981; }
    .dot-off { background: #c4c4cf; }
    .chip-off { color: var(--text-muted); }
</style>
"""

BADGES = {
    "gtm": ("🎯 GTM Agent", "badge-gtm"),
    "outreach": ("📝 Outreach Agent", "badge-outreach"),
    "crm": ("🗂️ CRM Agent", "badge-crm"),
}

EXAMPLE_PROMPTS = [
    "Do we have LED Red 5mm?",
    "Draft a cold email to CTOs at Series B SaaS companies",
    "Fetch the latest leads from Salesforce",
    "Count opportunities by stage",
]
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
    crm = "ON" if _salesforce_ready() else "—"
    col1, col2, col3 = st.columns(3)
    col1.markdown(_stat_card(doc_count, "Docs in DB"), unsafe_allow_html=True)
    col2.markdown(_stat_card(4, "Agents"), unsafe_allow_html=True)
    col3.markdown(_stat_card(crm, "CRM"), unsafe_allow_html=True)


def _render_integration_status() -> None:
    integrations = [
        ("Qdrant", _env_set("QDRANT_URL")),
        ("Anthropic", _env_set("ANTHROPIC_API_KEY")),
        ("Salesforce", _salesforce_ready()),
        ("Brevo", _env_set("BREVO_API_KEY")),
        ("Apollo", _env_set("APOLLO_API_KEY")),
        ("Galileo", is_galileo_enabled()),
    ]
    chips = "".join(
        f'<span class="chip {"" if on else "chip-off"}">'
        f'<span class="dot {"dot-on" if on else "dot-off"}"></span>{name}</span>'
        for name, on in integrations
    )
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


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
                    "          ├─ outreach_research → outreach_generate → send_gate ─┬─ END (review)\n"
                    "          │                                                     └─ outreach_send → END\n"
                    "          └─ crm_research → crm_generate → END",
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


def render_examples() -> None:
    """Clickable example prompts shown on an empty chat."""
    st.caption("Try one of these:")
    cols = st.columns(2)
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        if cols[index % 2].button(prompt, key=f"example_{index}", use_container_width=True):
            st.session_state["queued_prompt"] = prompt
            st.rerun()


def render_sidebar(doc_count: int) -> None:
    with st.sidebar:
        st.markdown("## 🚀 Product Marketing")
        st.caption("Multi-Agent RAG · GTM · Outreach · CRM")
        if st.button("🆕 New Chat", use_container_width=True):
            _reset_chat_state()
            st.rerun()
        st.divider()
        _render_stats(doc_count)
        st.markdown("**Integrations**")
        _render_integration_status()
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


def render_pending_send() -> None:
    draft = st.session_state.get("pending_drafts", "")
    if not draft or not _draft_has_recipient(draft):
        return
    st.info("📧 **Outreach draft ready.** Review the email above, then confirm to send via Brevo.")
    if st.button("✅ Confirm & Send Email", type="primary", key="confirm_send_email", use_container_width=True):
        _execute_confirm_send()


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
