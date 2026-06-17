"""
Simple Galileo SDK helper layer.

Think of Galileo like a "flight recorder" for the AI app:
- it records what happened
- in what order
- and with which inputs/outputs
"""

import os
from functools import wraps
from typing import Any, Callable

from config import GALILEO_API_KEY, GALILEO_LOG_STREAM, GALILEO_PROJECT

try:
    # Galileo SDK imports (real recorder).
    from galileo import galileo_context, log as galileo_log
    from galileo.handlers.langchain import GalileoCallback
except Exception:
    # Safe fallback: app still runs even if Galileo SDK is missing.
    galileo_context = None
    galileo_log = None
    GalileoCallback = None

_INITIALIZED = False


def is_galileo_enabled() -> bool:
    # Galileo is ON only when credentials + SDK are available.
    return bool(GALILEO_API_KEY and GALILEO_PROJECT and GALILEO_LOG_STREAM and galileo_context and GalileoCallback)


def ensure_galileo_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED or not is_galileo_enabled():
        return
    # Step 1: connect this app process to one Galileo project + log stream.
    galileo_context.init(project=GALILEO_PROJECT, log_stream=GALILEO_LOG_STREAM)
    _INITIALIZED = True


def _get_config_from_context() -> dict[str, Any] | None:
    """Get current RunnableConfig from LangChain context (set when running inside graph.invoke)."""
    try:
        from langchain_core.runnables.config import var_child_runnable_config

        ctx_config = var_child_runnable_config.get()
        return dict(ctx_config) if ctx_config else None
    except Exception:
        return None


def merge_node_config(
    config: dict[str, Any] | None,
    *,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Merge node metadata/tags into config. Preserves parent run tree for correct span nesting.
    Uses LangChain context when config is None (nodes may not receive config as param).
    Falls back to get_langchain_config() when no config with callbacks is available."""
    if config is None:
        config = _get_config_from_context()
    if not config or not config.get("callbacks"):
        return get_langchain_config(metadata=metadata, tags=tags)
    merged: dict[str, Any] = dict(config)
    if metadata:
        merged["metadata"] = {**(merged.get("metadata") or {}), **metadata}
    if tags:
        merged["tags"] = list(merged.get("tags") or []) + list(tags)
    return merged


def get_langchain_config(*, metadata: dict[str, Any] | None = None, tags: list[str] | None = None) -> dict[str, Any]:
    # If Galileo is OFF, return empty config so LangChain calls stay normal.
    if not is_galileo_enabled():
        return {}
    ensure_galileo_initialized()

    # Step 2: get a Galileo logger used to attach callbacks to LLM/tool calls.
    logger = galileo_context.get_logger_instance(project=GALILEO_PROJECT, log_stream=GALILEO_LOG_STREAM)
    in_experiment = logger.current_parent() is not None
    # Step 3: callback lets Galileo listen to each LangChain invoke event.
    callback = GalileoCallback(
        galileo_logger=logger,
        start_new_trace=not in_experiment,
        flush_on_chain_end=not in_experiment,
    )
    config: dict[str, Any] = {"callbacks": [callback]}
    if metadata:
        config["metadata"] = metadata
    if tags:
        config["tags"] = tags
    return config


def log_span(*, span_type: str = "workflow", name: str | None = None) -> Callable:
    # Decorator factory: wraps a function as one Galileo span.
    if galileo_log:
        kwargs = {"span_type": span_type}
        if name:
            kwargs["name"] = name
        base_decorator = galileo_log(**kwargs)

        def _decorator(fn: Callable) -> Callable:
            decorated = base_decorator(fn)

            @wraps(fn)
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                # Always ensure Galileo connection exists before logging spans.
                ensure_galileo_initialized()
                return decorated(*args, **kwargs)

            return _wrapped

        return _decorator

    def _noop_decorator(fn: Callable) -> Callable:
        # Galileo OFF path: no-op decorator keeps behavior unchanged.
        @wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return _wrapped

    return _noop_decorator


def get_eval_project(default: str = "product-marketing-evals") -> str:
    # Evals can log into a separate Galileo project if desired.
    return os.getenv("GALILEO_EVAL_PROJECT") or GALILEO_PROJECT or default


def start_chat_session(session_name: str) -> bool:
    # UI uses sessions so each chat thread is grouped in Galileo.
    if not is_galileo_enabled():
        return False
    ensure_galileo_initialized()
    try:
        logger = galileo_context.get_logger_instance(project=GALILEO_PROJECT, log_stream=GALILEO_LOG_STREAM)
        # "Session" = one conversation thread in Galileo UI.
        logger.start_session(name=session_name)
        return True
    except Exception:
        return False


def get_logger_instance():
    # Shared helper so other modules can start traces, sessions, and flush logs.
    if not is_galileo_enabled():
        return None
    ensure_galileo_initialized()
    try:
        return galileo_context.get_logger_instance(project=GALILEO_PROJECT, log_stream=GALILEO_LOG_STREAM)
    except Exception:
        return None


def get_console_links() -> dict[str, str]:
    # Convenience links for debugging (project + log stream pages).
    logger = get_logger_instance()
    if logger is None:
        return {}
    try:
        console_url = os.getenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai").rstrip("/")
        project_id = getattr(logger, "project_id", None)
        log_stream_id = getattr(logger, "log_stream_id", None)
        if not project_id or not log_stream_id:
            return {}
        project_url = f"{console_url}/project/{project_id}"
        log_stream_url = f"{project_url}/log-streams/{log_stream_id}"
        return {"project_url": project_url, "log_stream_url": log_stream_url}
    except Exception:
        return {}
