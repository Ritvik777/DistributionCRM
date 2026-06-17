import re

MAX_HISTORY_MESSAGES = 10

_CRM_FETCH_RE = re.compile(r"\b(salesforce|sfdc|crm)\b", re.IGNORECASE)
_CRM_LIST_RE = re.compile(
    r"\b(fetch|get|show|list|latest|recent|pull|retrieve|give me|display)\b",
    re.IGNORECASE,
)
_CRM_OBJECT_RE = re.compile(
    r"\b(leads?|contacts?|accounts?|prospects?|opportunit(?:y|ies)|cases?|deals?)\b",
    re.IGNORECASE,
)
# CRM operations that, combined with a CRM object, indicate a Salesforce data/metadata task.
_CRM_OP_RE = re.compile(
    r"\b(count|how many|aggregate|group by|sum|average|avg|describe|schema|fields?|"
    r"update|create|delete|insert|upsert|modify|status|save|add|log)\b",
    re.IGNORECASE,
)
# Strong CRM signals: Apex/SOQL/metadata operations always belong to the CRM agent.
_CRM_APEX_RE = re.compile(
    r"\b(apex|soql|sosl|execute anonymous|trigger|aggregate query|describe (the )?object)\b",
    re.IGNORECASE,
)
# Outreach verbs: when present (without an explicit CRM platform word), prefer the outreach agent.
_OUTREACH_VERB_RE = re.compile(
    r"\b(email|e-?mail|draft|compose|send|outreach|reach out|cold (email|message)|"
    r"write (a|an) (email|post|message|linkedin)|linkedin post)\b",
    re.IGNORECASE,
)


def is_crm_request(text: str) -> bool:
    """True when the message should be handled by the dedicated CRM (Salesforce) agent.

    Covers reads (leads/contacts/objects), aggregates, record DML, and Apex/SOQL work.
    """
    # Apex / SOQL / metadata work is unambiguously CRM.
    if _CRM_APEX_RE.search(text):
        return True
    # Explicit platform mention (salesforce/sfdc/crm) → CRM agent.
    if _CRM_FETCH_RE.search(text):
        return True
    # Operations on a CRM object (count/update/describe leads/opps/etc.) with no outreach intent.
    if _OUTREACH_VERB_RE.search(text):
        return False
    if _CRM_OBJECT_RE.search(text) and _CRM_OP_RE.search(text):
        return True
    # "fetch/list/show leads/contacts" with no outreach intent → CRM data lookup.
    if wants_crm_list_fetch(text):
        return True
    return False


def wants_crm_list_fetch(text: str) -> bool:
    """True when we should run a direct Salesforce query for latest leads/contacts.

    Intended to run on the CURRENT user message only (not chat history).
    """
    has_object = bool(_CRM_OBJECT_RE.search(text))
    if not has_object:
        return False
    return bool(_CRM_FETCH_RE.search(text) or _CRM_LIST_RE.search(text))


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
