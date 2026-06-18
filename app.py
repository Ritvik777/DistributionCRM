import streamlit as st
from services.vector_db_service import get_doc_count, get_component_image_count
from ui.ui import (
    apply_styles,
    render_sidebar,
    initialize_session_state,
    render_chat_history,
    render_pending_send,
    render_empty_welcome,
    render_chat_composer,
    handle_new_prompt,
)


def main() -> None:
    st.set_page_config(page_title="Product Distribution Agent", page_icon="📦", layout="wide")
    apply_styles()
    initialize_session_state()

    doc_count = get_doc_count()
    component_count = get_component_image_count()
    render_sidebar(doc_count)

    if not st.session_state.messages:
        st.markdown(
            '<div class="app-hero">'
            '<div class="app-hero-title">Product Distribution Agent</div>'
            '<div class="app-hero-sub">Catalog · inventory · outreach · CRM</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        render_empty_welcome(doc_count, component_count)

    render_chat_history()

    prompt, image_bytes, image_name = render_chat_composer()

    if prompt:
        handle_new_prompt(
            prompt,
            query_image_bytes=image_bytes,
            query_image_name=image_name,
        )
        if image_bytes:
            st.session_state.chat_attach_key = st.session_state.get("chat_attach_key", 0) + 1
        st.rerun()

    render_pending_send()


main()
