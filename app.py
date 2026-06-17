import streamlit as st
from services.vector_db_service import get_doc_count
from ui.ui import (
    apply_styles,
    render_sidebar,
    initialize_session_state,
    render_chat_history,
    render_pending_send,
    handle_new_prompt,
)


def main() -> None:
    st.set_page_config(page_title="Product Marketing", page_icon="🚀", layout="wide")
    apply_styles()
    initialize_session_state()

    doc_count = get_doc_count()
    render_sidebar(doc_count)

    st.markdown("# 🚀 Product Marketing")
    st.markdown(
        '<p class="hero-subtitle">Ask about your products, compare with competitors, or generate marketing content — the right agent handles it.</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages and doc_count == 0:
        st.info("👈 Add product docs to the knowledge base first — expand **Add Knowledge Base Docs** in the sidebar.")

    render_chat_history()

    new_prompt = st.chat_input("Ask about your products, or request marketing content...")
    if new_prompt:
        handle_new_prompt(new_prompt)

    render_pending_send()


main()
