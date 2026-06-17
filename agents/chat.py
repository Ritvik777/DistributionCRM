import re

MAX_HISTORY_MESSAGES = 10

_CRM_FETCH_RE = re.compile(r"\b(salesforce|sfdc|crm)\b", re.IGNORECASE)
_CRM_LIST_RE = re.compile(
    r"\b(fetch|get|show|list|latest|recent|pull|retrieve|give me|display)\b",
    re.IGNORECASE,
)
_CRM_ACTION_RE = re.compile(
    r"\b(fetch|get|show|list|latest|recent|pull|retrieve|find|search|lookup|give me|display)\b",
    re.IGNORECASE,
)
_CRM_OBJECT_RE = re.compile(
    r"\b(leads?|contacts?|accounts?|prospects?)\b",
    re.IGNORECASE,
)
_CRM_HISTORY_RE = re.compile(
    r"(latest salesforce leads|salesforce lead|crm fetch|salesforce_query)",
    re.IGNORECASE,
)


def wants_crm_fetch(text: str) -> bool:
    """True when the user wants to read/list lead or contact data (Salesforce/CRM)."""
    has_object = bool(_CRM_OBJECT_RE.search(text))
    has_action = bool(_CRM_ACTION_RE.search(text))
    has_platform = bool(_CRM_FETCH_RE.search(text))
    has_list_verb = bool(_CRM_LIST_RE.search(text))
    has_crm_history = bool(_CRM_HISTORY_RE.search(text))

    if has_platform and (has_action or has_object):
        return True
    # "fetch latest leads", "show contacts", "get leads" — no platform word required
    if has_list_verb and has_object:
        return True
    # Follow-up after a prior Salesforce leads response in chat history
    if has_crm_history and has_action and has_object:
        return True
    return False


def wants_crm_list_fetch(text: str) -> bool:
    """True when we should run a direct Salesforce query for latest leads/contacts."""
    has_object = bool(_CRM_OBJECT_RE.search(text))
    if _CRM_FETCH_RE.search(text) and has_object:
        return True
    if _CRM_LIST_RE.search(text) and has_object:
        return True
    if _CRM_HISTORY_RE.search(text) and _CRM_LIST_RE.search(text):
        return True
    return False


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
