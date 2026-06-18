"""Streamlit-agnostic conversation/session helpers for the chat UI."""

from __future__ import annotations

from agents.constants import EMAIL_PATTERN, MAX_HISTORY_MESSAGES, SEND_CONFIRM_PHRASES


def draft_has_recipient(text: str) -> bool:
    return bool(EMAIL_PATTERN.search(text or ""))


def get_chat_history(messages: list[dict], *, exclude_last: bool = True) -> list[dict]:
    history: list[dict] = []
    source = messages[:-1] if exclude_last and messages else messages
    for msg in source[-MAX_HISTORY_MESSAGES:]:
        entry: dict = {"role": msg["role"], "content": msg.get("content", "")}
        if msg.get("agent"):
            entry["agent"] = msg["agent"]
        history.append(entry)
    return history


def question_for_agent(
    prompt: str,
    *,
    awaiting_email: bool,
    pricing_question: str,
    pending_drafts: str,
) -> str:
    if awaiting_email:
        return f"{pricing_question} My email is {prompt}"
    if pending_drafts and any(word in prompt.lower() for word in SEND_CONFIRM_PHRASES):
        return f"{prompt}\n\nUse this draft:\n{pending_drafts}"
    return prompt


def pending_component_matches(
    messages: list[dict],
    stored: list[dict] | None = None,
) -> list[dict]:
    if stored:
        return stored
    for message in reversed(messages):
        matches = message.get("component_matches") or []
        if matches:
            return matches
    return []


def apply_agent_result_to_session(
    session: dict,
    prompt: str,
    result: dict,
) -> None:
    """Update Streamlit session_state keys from an agent result."""
    if result.get("is_pricing") and not result.get("user_email"):
        session["awaiting_email"] = True
        if not session.get("pricing_question"):
            session["pricing_question"] = prompt
    else:
        session["awaiting_email"] = False
        session["pricing_question"] = ""

    if result.get("send_confirmed"):
        session["pending_drafts"] = ""
        session["pending_component_matches"] = []
    elif result.get("agent_type") == "outreach" and result.get("answer"):
        answer = result["answer"]
        if draft_has_recipient(answer) and not answer.strip().startswith("✅ **Sent to:**"):
            session["pending_drafts"] = answer
            if result.get("component_matches"):
                session["pending_component_matches"] = result["component_matches"]
            elif not session.get("pending_component_matches"):
                for message in reversed(session.get("messages") or []):
                    if message.get("component_matches"):
                        session["pending_component_matches"] = message["component_matches"]
                        break
