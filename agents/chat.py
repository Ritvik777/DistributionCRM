MAX_HISTORY_MESSAGES = 10


def trim_chat_history(history: list[dict] | None, max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    if not history:
        return []
    return history[-max_messages:]


def format_chat_history(history: list[dict] | None, max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    lines: list[str] = []
    for msg in trim_chat_history(history, max_messages):
        role = msg.get("role", "user")
        label = "User" if role == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if len(content) > 2000:
            content = content[:2000] + "..."
        if not content:
            continue
        agent = msg.get("agent")
        if agent and role == "assistant":
            label = f"{label} ({agent})"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def build_turn_context(state: dict) -> str:
    """Structured conversation context for classifier, gates, and generators."""
    history = format_chat_history(state.get("chat_history"))
    question = (state.get("question") or "").strip()
    parts: list[str] = []
    if history:
        parts.append(f"Conversation history:\n{history}")
    parts.append(f"Current user message:\n{question}")
    return "\n\n".join(parts)
