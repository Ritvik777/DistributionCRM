import streamlit as st
from services.vector_db_service import get_doc_count
from ui.ui import (
    apply_styles,
    render_sidebar,
    initialize_session_state,
    render_chat_history,
    render_pending_send,
    render_examples,
    handle_new_prompt,
)


def main() -> None:
    st.set_page_config(page_title="Product Marketing", page_icon="🚀", layout="wide")
    apply_styles()
    initialize_session_state()

    doc_count = get_doc_count()
    render_sidebar(doc_count)

    st.markdown('<div class="hero-title">🚀 Product Marketing</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">One assistant, three specialists — ask about products, generate outreach, '
        'or work your Salesforce CRM. The right agent handles it.</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        if doc_count == 0:
            st.info("👈 Add product docs to the knowledge base first — expand **Add Knowledge Base Docs** in the sidebar.")
        render_examples()

    render_chat_history()

    typed_prompt = st.chat_input("Ask about products, draft outreach, or query your CRM...")
    prompt = typed_prompt or st.session_state.pop("queued_prompt", "")
    if prompt:
        handle_new_prompt(prompt)

    render_pending_send()


main()
