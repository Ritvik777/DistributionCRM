"""Single Salesforce read/format layer for agents and tools."""

from __future__ import annotations

from typing import Any

from services.salesforce_client import (
    _run_query,
    fetch_leads_from_recent_outreach,
    fetch_leads_in_time_window,
    find_leads_by_part_enquiry,
    is_salesforce_configured,
)
from services.salesforce_mcp import parse_query_records

DEFAULT_LEAD_FIELDS = ["Id", "Name", "Email", "Company", "Title", "Status", "LastModifiedDate", "CreatedDate"]
OUTREACH_LEAD_FIELDS = ["Id", "Name", "Email", "Company", "Title", "Status"]


def fetch_latest_leads(limit: int = 10) -> list[dict[str, Any]]:
    """Most recently modified leads (datetime precision — avoids same-day LastActivityDate ties)."""
    return _run_query(
        "Lead",
        DEFAULT_LEAD_FIELDS,
        order_by="LastModifiedDate DESC",
        limit=max(1, min(limit, 50)),
    )


def fetch_recent_outreach_recipients(limit: int = 10) -> list[dict[str, Any]]:
    """People most recently emailed by the agent (from logged Salesforce Tasks)."""
    return fetch_leads_from_recent_outreach(limit=limit)


def fetch_leads_for_time_window(minutes: int, limit: int = 50) -> list[dict[str, Any]]:
    return fetch_leads_in_time_window(minutes, limit=limit)


def fetch_leads_with_email(limit: int = 10) -> list[dict[str, Any]]:
    """Leads that have a usable email address (for outreach batch drafts)."""
    rows = fetch_latest_leads(limit=limit)
    results: list[dict[str, Any]] = []
    for row in rows:
        email = (row.get("Email") or "").strip()
        if email and email.lower() not in ("null", "none"):
            results.append(row)
    return results


def format_leads_markdown(
    rows: list[dict[str, Any]] | str,
    *,
    limit: int = 10,
    heading: str | None = None,
) -> str:
    if isinstance(rows, str):
        parsed = parse_query_records(rows)
    else:
        parsed = rows

    if not parsed:
        return "No leads found in Salesforce."

    if heading is None:
        heading = "## Latest Salesforce Lead" if limit == 1 else "## Latest Salesforce Leads"

    show_type = any((row.get("_ObjectType") or "Lead") == "Contact" for row in parsed)
    lines = [
        heading,
        "",
        "| Name | Email | Company | Title | Type | Status |" if show_type else "| Name | Email | Company | Title | Status |",
        "| --- | --- | --- | --- | --- | --- |" if show_type else "| --- | --- | --- | --- | --- |",
    ]
    for row in parsed:
        def cell(key: str) -> str:
            val = row.get(key, "")
            return "" if val in (None, "null", "None") else str(val)

        object_type = row.get("_ObjectType") or "Lead"
        if show_type:
            lines.append(
                f"| {cell('Name')} | {cell('Email')} | {cell('Company')} | {cell('Title')} | "
                f"{object_type} | {cell('Status') if object_type == 'Lead' else '—'} |"
            )
        else:
            lines.append(
                f"| {cell('Name')} | {cell('Email')} | {cell('Company')} | {cell('Title')} | {cell('Status')} |"
            )

    first = parsed[0]
    subject = (first.get("_LastEmailSubject") or "").strip()
    if subject:
        lines.extend(["", f"*Last outreach email:* {subject}"])
    return "\n".join(lines)


def format_leads_for_outreach(rows: list[dict[str, Any]]) -> list[str]:
    blocks: list[str] = []
    for row in rows:
        name = row.get("Name") or "there"
        title = row.get("Title") if row.get("Title") not in (None, "null", "None") else "N/A"
        company = row.get("Company") if row.get("Company") not in (None, "null", "None") else "N/A"
        email = (row.get("Email") or "").strip()
        blocks.append(f"**{name}** — {title} at {company}\n  Email: {email}")
    return blocks


def query_records_as_text(
    object_name: str,
    fields: list[str],
    where_clause: str = "",
    order_by: str = "",
    limit: int = 10,
) -> str:
    """Format query results as tool-style text (MCP-compatible layout)."""
    rows = _run_query(object_name, fields, where_clause, order_by, limit)
    if not rows:
        return "Query returned 0 records."
    lines = []
    for index, row in enumerate(rows, 1):
        parts = [f" {field}: {row.get(field)}" for field in fields if field in row]
        lines.append(f"Record {index}:\n" + "\n".join(parts))
    return f"Query returned {len(rows)} records:\n\n" + "\n\n".join(lines)


__all__ = [
    "DEFAULT_LEAD_FIELDS",
    "OUTREACH_LEAD_FIELDS",
    "fetch_latest_leads",
    "fetch_recent_outreach_recipients",
    "fetch_leads_for_time_window",
    "fetch_leads_with_email",
    "find_leads_by_part_enquiry",
    "format_leads_for_outreach",
    "format_leads_markdown",
    "is_salesforce_configured",
    "query_records_as_text",
]
