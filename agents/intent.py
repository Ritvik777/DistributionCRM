"""Unified intent classification and CRM fast-path eligibility."""

from __future__ import annotations

import re

from agents.constants import EMAIL_PATTERN, MAX_HISTORY_MESSAGES

_CRM_FETCH_RE = re.compile(r"\b(salesforce|sfdc|crm)\b", re.IGNORECASE)
_CRM_LIST_RE = re.compile(
    r"\b(fetch|get|show|list|latest|recent|pull|retrieve|give me|display)\b",
    re.IGNORECASE,
)
_CRM_OBJECT_RE = re.compile(
    r"\b(leads?|contacts?|accounts?|prospects?|opportunit(?:y|ies)|cases?|deals?)\b",
    re.IGNORECASE,
)
_CRM_OP_RE = re.compile(
    r"\b(count|how many|aggregate|group by|sum|average|avg|describe|schema|fields?|"
    r"update|create|delete|insert|upsert|modify|status|save|add|log)\b",
    re.IGNORECASE,
)
_CRM_SIGNAL_RE = re.compile(
    r"\b(soql|aggregate query|describe (the )?object|group by)\b",
    re.IGNORECASE,
)
_OUTREACH_VERB_RE = re.compile(
    r"\b(e-?mail|draft|compose|outreach|reach out|cold (email|message)|"
    r"send (an? )?(email|message|note)|write (a|an) (email|post|message|linkedin)|linkedin post)\b",
    re.IGNORECASE,
)
_OUTREACH_RECIPIENT_RE = re.compile(
    r"\b(send|mail|notify|tell|inform|contact|reach out)\b.*@|@\S+\s+(about|regarding|on|for)\b",
    re.IGNORECASE,
)
_SF_PLATFORM_RE = re.compile(r"\b(salesforce|sfdc|crm)\b", re.IGNORECASE)

_READ_VERB_RE = re.compile(
    r"\b(fetch|get|show|list|latest|recent|last|pull|retrieve|display|view|see|give me)\b",
    re.IGNORECASE,
)
_WRITE_OR_ADVANCED_RE = re.compile(
    r"\b(soql|aggregate|count|sum|average|group by|describe|object|"
    r"contact|account|opportunity|case|update|delete|insert|create|upsert|save|add|log|new)\b",
    re.IGNORECASE,
)
_FILTERED_LEADS_RE = re.compile(
    r"\b(about|for|who|with|where|enquir|inquir|requested|asked|interested in|"
    r"part|sku|product|component|item|model|number|no\.?)\b",
    re.IGNORECASE,
)
_CRM_MODIFIER_RE = r"(?:\s+(?:crm|salesforce|sfdc))*"
_SINGULAR_LEAD_RE = re.compile(
    rf"\b(?:the\s+)?(?:last|latest|newest|most recent){_CRM_MODIFIER_RE}\s+lead\b",
    re.IGNORECASE,
)
_COUNT_LEADS_RE = re.compile(
    r"\b(?:last|latest|recent|top|first)\s+(\d+)\s+leads?\b",
    re.IGNORECASE,
)
_TIME_WINDOW_RE = re.compile(
    r"\b(?:last|past|in the last|from the last|from last)\s+(\d+)\s*"
    r"(minute|minutes|min|hour|hours|hr|day|days)\b",
    re.IGNORECASE,
)
_PART_REF_PATTERNS = [
    re.compile(r"part\s*(?:no\.?|number|#)\s*[:.]?\s*([A-Za-z0-9][\w\s\-./]+)", re.IGNORECASE),
    re.compile(r"sku\s*[:.]?\s*([A-Za-z0-9][\w\s\-./]+)", re.IGNORECASE),
    re.compile(
        r"(?:enquir(?:ed|y|ing)?|inquir(?:ed|y|ing)?|asked|requested|interested)\s+"
        r"(?:about|for|in)\s+(?:part\s*(?:no\.?|number)?\s*)?([A-Za-z0-9][\w\s\-./]+)",
        re.IGNORECASE,
    ),
]


def is_crm_request(text: str) -> bool:
    if _CRM_SIGNAL_RE.search(text):
        return True
    if _OUTREACH_VERB_RE.search(text):
        return False
    if _CRM_FETCH_RE.search(text):
        return True
    if _CRM_OBJECT_RE.search(text) and _CRM_OP_RE.search(text):
        return True
    if wants_crm_list_fetch(text):
        return True
    return False


def is_outreach_request(text: str) -> bool:
    if _CRM_SIGNAL_RE.search(text):
        return False
    if _OUTREACH_VERB_RE.search(text):
        return True
    if EMAIL_PATTERN.search(text) and _OUTREACH_RECIPIENT_RE.search(text):
        return True
    return False


def wants_crm_list_fetch(text: str) -> bool:
    if not _CRM_OBJECT_RE.search(text):
        return False
    return bool(_CRM_FETCH_RE.search(text) or _CRM_LIST_RE.search(text))


def wants_salesforce_leads_for_outreach(text: str) -> bool:
    """Email recipients sourced from Salesforce (handled in outreach research)."""
    return bool(_SF_PLATFORM_RE.search(text) and re.search(r"\blead", text, re.IGNORECASE))


def is_simple_leads_fetch(text: str) -> bool:
    if _WRITE_OR_ADVANCED_RE.search(text):
        return False
    if _FILTERED_LEADS_RE.search(text):
        return False
    if not re.search(r"\blead", text, re.IGNORECASE):
        return False
    return bool(_READ_VERB_RE.search(text))


def is_leads_by_part_enquiry(text: str) -> bool:
    if not re.search(r"\bleads?\b", text, re.IGNORECASE):
        return False
    if extract_part_reference(text):
        return True
    return bool(
        re.search(
            r"\b(enquir|inquir|asked about|requested|interested in)\b.*"
            r"\b(part|sku|product|component)\b",
            text,
            re.IGNORECASE,
        )
    )


def extract_part_reference(text: str) -> str | None:
    for pattern in _PART_REF_PATTERNS:
        match = pattern.search(text)
        if match:
            part = match.group(1).strip().strip(".,;")
            if part and len(part) >= 3:
                return part
    return None


def parse_leads_time_window_minutes(text: str) -> int | None:
    match = _TIME_WINDOW_RE.search(text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("min"):
        return amount
    if unit.startswith("hour") or unit == "hr":
        return amount * 60
    if unit.startswith("day"):
        return amount * 24 * 60
    return None


def is_leads_time_window_fetch(text: str) -> bool:
    return parse_leads_time_window_minutes(text) is not None and bool(
        re.search(r"\bleads?\b", text, re.IGNORECASE)
    )


def is_recent_leads_list_fetch(text: str) -> bool:
    """Plural lead list where user wants recency, not oldest-first or arbitrary CreatedDate."""
    if is_leads_time_window_fetch(text) or is_singular_lead_fetch(text):
        return False
    if not re.search(r"\bleads\b", text, re.IGNORECASE):
        return False
    if re.search(r"\b(recent|latest|newest|all)\b", text, re.IGNORECASE):
        return True
    return bool(re.search(r"\blast\b.*\bleads\b", text, re.IGNORECASE))


def is_singular_lead_fetch(text: str) -> bool:
    """User wants one lead — prefer last outreach recipient over newest CreatedDate."""
    if _SINGULAR_LEAD_RE.search(text):
        return True
    if infer_leads_limit(text) != 1:
        return False
    if not re.search(r"\blead\b", text, re.IGNORECASE):
        return False
    return bool(_CRM_FETCH_RE.search(text) and _READ_VERB_RE.search(text))


def infer_leads_limit(text: str) -> int:
    if _SINGULAR_LEAD_RE.search(text):
        return 1
    if re.search(r"\ball\b", text, re.IGNORECASE) and re.search(r"\bleads\b", text, re.IGNORECASE):
        return 50
    count_match = _COUNT_LEADS_RE.search(text) or re.search(r"\b(\d+)\s+leads?\b", text, re.IGNORECASE)
    if count_match:
        return max(1, min(int(count_match.group(1)), 50))
    if re.search(r"\bleads\b", text, re.IGNORECASE):
        return 10
    return 1


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
    history = format_chat_history(state.get("chat_history"))
    question = (state.get("question") or "").strip()
    parts: list[str] = []
    if history:
        parts.append(f"Conversation history:\n{history}")
    parts.append(f"Current user message:\n{question}")
    return "\n\n".join(parts)
