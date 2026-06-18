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
# Strong CRM signals: SOQL/aggregate/metadata operations always belong to the CRM agent.
_CRM_SIGNAL_RE = re.compile(
    r"\b(soql|aggregate query|describe (the )?object|group by)\b",
    re.IGNORECASE,
)
# Outreach verbs: emailing/messaging intent. "send" alone is excluded (ambiguous, e.g.
# "send me the latest leads"); we require an email/draft/outreach signal.
_OUTREACH_VERB_RE = re.compile(
    r"\b(e-?mail|draft|compose|outreach|reach out|cold (email|message)|"
    r"send (an? )?(email|message|note)|write (a|an) (email|post|message|linkedin)|linkedin post)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_OUTREACH_RECIPIENT_RE = re.compile(
    r"\b(send|mail|notify|tell|inform|contact|reach out)\b.*@|@\S+\s+(about|regarding|on|for)\b",
    re.IGNORECASE,
)


def is_crm_request(text: str) -> bool:
    """True when the message should be handled by the dedicated CRM (Salesforce) agent.

    Covers reads (leads/contacts/objects), aggregates, record DML, and SOQL/metadata work.
    A send/email intent always wins (CRM agent cannot send email) — except pure SOQL/metadata.
    """
    # SOQL / aggregate / metadata work is unambiguously CRM, even if it mentions email.
    if _CRM_SIGNAL_RE.search(text):
        return True
    # Email/outreach intent → outreach agent (it can also look up CRM during research),
    # even if the message mentions Salesforce/leads.
    if _OUTREACH_VERB_RE.search(text):
        return False
    # Explicit platform mention (salesforce/sfdc/crm) → CRM agent.
    if _CRM_FETCH_RE.search(text):
        return True
    # Operations on a CRM object (count/update/describe leads/opps/etc.).
    if _CRM_OBJECT_RE.search(text) and _CRM_OP_RE.search(text):
        return True
    # "fetch/list/show leads/contacts" → CRM data lookup.
    if wants_crm_list_fetch(text):
        return True
    return False


def is_outreach_request(text: str) -> bool:
    """True when the message has an email/outreach send intent (which the CRM agent cannot do).

    SOQL/metadata work still belongs to CRM, so those are excluded.
    """
    if _CRM_SIGNAL_RE.search(text):
        return False
    if _OUTREACH_VERB_RE.search(text):
        return True
    if _EMAIL_RE.search(text) and _OUTREACH_RECIPIENT_RE.search(text):
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
