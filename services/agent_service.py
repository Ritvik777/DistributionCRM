from agents import ask, get_graph_image, get_graph_ascii, confirm_send


def ask_agent(
    question: str,
    chat_history: list[dict] | None = None,
    send_confirmed: bool = False,
    pending_draft: str = "",
) -> dict:
    return ask(
        question,
        chat_history=chat_history,
        send_confirmed=send_confirmed,
        pending_draft=pending_draft,
    )


def confirm_send_email(pending_draft: str, chat_history: list[dict] | None = None) -> dict:
    return confirm_send(pending_draft, chat_history=chat_history)


def load_graph_image() -> bytes | None:
    return get_graph_image()


def load_graph_ascii() -> str:
    return get_graph_ascii()
