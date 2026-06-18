from agents import ask, get_graph_image, get_graph_ascii, confirm_send


def ask_agent(
    question: str,
    chat_history: list[dict] | None = None,
    send_confirmed: bool = False,
    pending_draft: str = "",
    query_image_b64: str | None = None,
    component_matches: list[dict] | None = None,
    kb_sources: list[dict] | None = None,
) -> dict:
    return ask(
        question,
        chat_history=chat_history,
        send_confirmed=send_confirmed,
        pending_draft=pending_draft,
        query_image_b64=query_image_b64,
        component_matches=component_matches,
        kb_sources=kb_sources,
    )


def confirm_send_email(
    pending_draft: str,
    chat_history: list[dict] | None = None,
    component_matches: list[dict] | None = None,
) -> dict:
    return confirm_send(pending_draft, chat_history=chat_history, component_matches=component_matches)


def load_graph_image() -> bytes | None:
    return get_graph_image()


def load_graph_ascii() -> str:
    return get_graph_ascii()
