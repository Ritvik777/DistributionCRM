"""Live health checks for TradeFlow Agent integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import requests

from config import (
    ANTHROPIC_API_KEY,
    APOLLO_API_KEY,
    BREVO_API_KEY,
    BREVO_FROM_EMAIL,
    QDRANT_API_KEY,
    QDRANT_URL,
)

HealthStatus = Literal["up", "down", "off"]


@dataclass(frozen=True)
class SystemHealth:
    name: str
    status: HealthStatus
    detail: str
    required: bool = False


def _env_configured(name: str, *, placeholder_prefix: str = "your-") -> bool:
    value = (os.getenv(name) or "").strip()
    return bool(value) and not value.startswith(placeholder_prefix)


def _check_anthropic() -> SystemHealth:
    if not _env_configured("ANTHROPIC_API_KEY"):
        return SystemHealth("Anthropic LLM", "down", "Missing ANTHROPIC_API_KEY", required=True)
    try:
        from llm import get_llm

        get_llm()
        return SystemHealth("Anthropic LLM", "up", "API key configured", required=True)
    except Exception as exc:
        return SystemHealth("Anthropic LLM", "down", str(exc)[:140], required=True)


def _check_qdrant_text_kb() -> SystemHealth:
    if not QDRANT_URL or not QDRANT_API_KEY:
        return SystemHealth("Qdrant (text KB)", "down", "Missing QDRANT_URL or QDRANT_API_KEY", required=True)
    try:
        from vector_db.database import COLLECTION_NAME, qdrant_client

        qdrant_client.get_collections()
        try:
            info = qdrant_client.get_collection(COLLECTION_NAME)
            count = info.points_count or 0
            detail = f"Connected · {count} chunks"
        except Exception:
            detail = "Connected · text KB empty (upload docs to index)"
        return SystemHealth("Qdrant (text KB)", "up", detail, required=True)
    except Exception as exc:
        return SystemHealth("Qdrant (text KB)", "down", str(exc)[:140], required=True)


def _check_qdrant_components() -> SystemHealth:
    if not QDRANT_URL or not QDRANT_API_KEY:
        return SystemHealth("Component catalog", "off", "Qdrant not configured", required=False)
    try:
        from vector_db.component_store import get_component_count

        count = get_component_count()
        return SystemHealth("Component catalog", "up", f"{count} indexed photo(s)", required=False)
    except Exception as exc:
        return SystemHealth("Component catalog", "down", str(exc)[:140], required=False)


def _check_clip() -> SystemHealth:
    try:
        import sentence_transformers  # noqa: F401

        return SystemHealth("CLIP (local vision)", "up", "Ready (loads on first photo match)", required=False)
    except ImportError:
        return SystemHealth(
            "CLIP (local vision)",
            "down",
            "Install sentence-transformers + torch for photo match",
            required=False,
        )


def _check_galileo() -> SystemHealth:
    try:
        from observability import get_logger_instance, is_galileo_enabled

        if not is_galileo_enabled():
            return SystemHealth("Galileo", "off", "Not configured (optional)", required=False)
        if get_logger_instance() is None:
            return SystemHealth("Galileo", "down", "Logger unavailable", required=False)
        return SystemHealth("Galileo", "up", "Tracing enabled", required=False)
    except Exception as exc:
        return SystemHealth("Galileo", "down", str(exc)[:140], required=False)


def _check_salesforce() -> SystemHealth:
    try:
        from services.salesforce_client import is_salesforce_configured, ping_salesforce

        if not is_salesforce_configured():
            return SystemHealth("Salesforce CRM", "off", "Not configured (optional)", required=False)
        ok, detail = ping_salesforce()
        if ok:
            return SystemHealth("Salesforce CRM", "up", detail, required=False)
        return SystemHealth("Salesforce CRM", "down", detail, required=False)
    except Exception as exc:
        return SystemHealth("Salesforce CRM", "down", str(exc)[:140], required=False)


def _check_brevo() -> SystemHealth:
    if not BREVO_API_KEY or BREVO_API_KEY == "your-brevo-api-key-here":
        return SystemHealth("Brevo email", "off", "Not configured (optional)", required=False)
    if not BREVO_FROM_EMAIL or BREVO_FROM_EMAIL == "you@yourcompany.com":
        return SystemHealth("Brevo email", "down", "Set BREVO_FROM_EMAIL", required=False)
    try:
        resp = requests.get(
            "https://api.brevo.com/v3/account",
            headers={"api-key": BREVO_API_KEY, "accept": "application/json"},
            timeout=12,
        )
        if resp.status_code == 200:
            return SystemHealth("Brevo email", "up", "API reachable", required=False)
        return SystemHealth("Brevo email", "down", f"HTTP {resp.status_code}: {resp.text[:80]}", required=False)
    except Exception as exc:
        return SystemHealth("Brevo email", "down", str(exc)[:140], required=False)


def _check_apollo() -> SystemHealth:
    if not APOLLO_API_KEY or APOLLO_API_KEY == "your-apollo-api-key-here":
        return SystemHealth("Apollo leads", "off", "Not configured (optional)", required=False)
    return SystemHealth("Apollo leads", "up", "API key configured", required=False)


def check_all_systems() -> list[SystemHealth]:
    """Run all integration health probes (may take a few seconds)."""
    return [
        _check_anthropic(),
        _check_qdrant_text_kb(),
        _check_qdrant_components(),
        _check_clip(),
        _check_galileo(),
        _check_salesforce(),
        _check_brevo(),
        _check_apollo(),
    ]


def summarize_health(report: list[SystemHealth]) -> tuple[int, int, int]:
    up = sum(1 for item in report if item.status == "up")
    down = sum(1 for item in report if item.status == "down")
    off = sum(1 for item in report if item.status == "off")
    return up, down, off


def core_systems_healthy(report: list[SystemHealth]) -> bool:
    return all(item.status != "down" for item in report if item.required)
