"""
agents/ — The multi-agent system (lazy imports to keep lightweight submodule loading).
"""

from observability import ensure_galileo_initialized, get_langchain_config, get_logger_instance, is_galileo_enabled

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from agents.graph import graph as compiled_graph
        _graph = compiled_graph
    return _graph


def _base_state(
    question: str,
    chat_history: list[dict] | None = None,
    **overrides,
) -> dict:
    state = {
        "question": question,
        "chat_history": chat_history or [],
        "agent_type": "",
        "context": "",
        "answer": "",
        "kb_sources": [],
        "query_image_b64": "",
        "component_matches": [],
        "is_pricing": False,
        "user_email": "",
        "send_intent": False,
        "send_requested": False,
        "send_confirmed": False,
        "steps": [],
    }
    state.update(overrides)
    return state


def _invoke_with_tracing(state: dict, *, trace_name: str = "ask_agent") -> dict:
    ensure_galileo_initialized()
    question = state.get("question", "")

    if not is_galileo_enabled():
        config = get_langchain_config(metadata={"question": question})
        return _get_graph().invoke(state, config=config)

    logger = get_logger_instance()
    if logger is None:
        config = get_langchain_config(metadata={"question": question})
        return _get_graph().invoke(state, config=config)

    in_existing_trace = logger.current_parent() is not None
    if not in_existing_trace:
        logger.start_trace(input={"question": question}, name=trace_name)
    config = get_langchain_config(metadata={"question": question})
    try:
        result = _get_graph().invoke(state, config=config)
        if not in_existing_trace:
            logger.conclude(result.get("answer", ""))
            logger.flush()
        return result
    except Exception:
        if not in_existing_trace:
            try:
                logger.flush()
            except Exception:
                pass
        raise


def ask(
    question: str,
    chat_history: list[dict] | None = None,
    send_confirmed: bool = False,
    pending_draft: str = "",
    query_image_b64: str | None = None,
    component_matches: list[dict] | None = None,
    kb_sources: list[dict] | None = None,
) -> dict:
    if send_confirmed and pending_draft.strip():
        return confirm_send(pending_draft, chat_history=chat_history)

    overrides: dict = {"query_image_b64": query_image_b64 or ""}
    if component_matches is not None:
        overrides["component_matches"] = component_matches
    if kb_sources is not None:
        overrides["kb_sources"] = kb_sources
    state = _base_state(question, chat_history, **overrides)
    return _invoke_with_tracing(state)


def confirm_send(
    pending_draft: str,
    chat_history: list[dict] | None = None,
    component_matches: list[dict] | None = None,
) -> dict:
    """UI-confirmed Brevo send — bypasses draft generation, runs outreach_send only."""
    from agents.outreach_agent.nodes import outreach_send

    ensure_galileo_initialized()
    question = "User confirmed send via UI"
    overrides: dict = {
        "agent_type": "outreach",
        "answer": pending_draft,
        "send_confirmed": True,
    }
    if component_matches:
        overrides["component_matches"] = component_matches
    state = _base_state(question, chat_history, **overrides)
    config = get_langchain_config(metadata={"question": question, "send_confirmed": True})

    logger = get_logger_instance()
    in_existing_trace = logger is not None and logger.current_parent() is not None
    if logger and not in_existing_trace:
        logger.start_trace(input={"question": question, "action": "confirm_send"}, name="confirm_send")

    try:
        result = outreach_send(state, config=config)
        result["agent_type"] = "outreach"
        result["send_confirmed"] = True
        result["send_requested"] = True
        if logger and not in_existing_trace:
            logger.conclude(result.get("answer", ""))
            logger.flush()
        return result
    except Exception:
        if logger and not in_existing_trace:
            try:
                logger.flush()
            except Exception:
                pass
        raise


def get_graph_image() -> bytes | None:
    try:
        return _get_graph().get_graph().draw_mermaid_png()
    except Exception:
        return None


def get_graph_ascii() -> str:
    try:
        return _get_graph().get_graph().draw_ascii()
    except ImportError:
        return _get_graph().get_graph().draw_mermaid()
