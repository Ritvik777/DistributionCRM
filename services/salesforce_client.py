"""Salesforce integration — uses your TypeScript MCP server by default, Python REST as fallback.

All operations are written once on top of two backend primitives:
  _run_query(...)  → list[dict] rows   (MCP salesforce_query_records  OR  simple-salesforce SOQL)
  _run_dml(...)    → None, raises      (MCP salesforce_dml_records     OR  simple-salesforce DML)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import requests

from services.salesforce_mcp import (
    mcp_dml_records,
    mcp_query_records,
    parse_query_records,
    salesforce_backend,
)

_LEAD_FIELDS = ["Id", "Name", "Email", "Company", "Title", "Status", "LastActivityDate", "LastModifiedDate", "CreatedDate"]
_CONTACT_FIELDS = ["Id", "Name", "Email", "Title"]
_OUTREACH_EMAIL_TASK_PREFIX = "Email sent:"


def is_salesforce_configured() -> bool:
    conn = (os.getenv("SALESFORCE_CONNECTION_TYPE") or "User_Password").strip()
    if conn == "OAuth_2.0_Client_Credentials":
        return bool(
            os.getenv("SALESFORCE_CLIENT_ID")
            and os.getenv("SALESFORCE_CLIENT_SECRET")
            and os.getenv("SALESFORCE_INSTANCE_URL")
        )
    if conn == "Salesforce_CLI":
        return salesforce_backend() == "mcp"
    return bool(os.getenv("SALESFORCE_USERNAME") and os.getenv("SALESFORCE_PASSWORD"))


def uses_mcp_server() -> bool:
    return is_salesforce_configured() and salesforce_backend() == "mcp"


def _escape_soql(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


# --- Connection (Python REST fallback) ---------------------------------------

def _oauth_client_credentials_token() -> tuple[str, str]:
    instance_url = os.getenv("SALESFORCE_INSTANCE_URL", "").rstrip("/")
    resp = requests.post(
        f"{instance_url}/services/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": os.getenv("SALESFORCE_CLIENT_ID", ""),
            "client_secret": os.getenv("SALESFORCE_CLIENT_SECRET", ""),
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Salesforce OAuth failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    return data["access_token"], data.get("instance_url", instance_url)


@lru_cache(maxsize=1)
def get_salesforce_client():
    from simple_salesforce import Salesforce

    if not is_salesforce_configured():
        raise RuntimeError("Salesforce is not configured. Add SALESFORCE_* vars to .env")

    conn = (os.getenv("SALESFORCE_CONNECTION_TYPE") or "User_Password").strip()
    if conn == "OAuth_2.0_Client_Credentials":
        token, instance_url = _oauth_client_credentials_token()
        return Salesforce(instance=urlparse(instance_url).netloc, session_id=token)

    instance_url = os.getenv("SALESFORCE_INSTANCE_URL", "").strip()
    kwargs: dict[str, Any] = {
        "username": os.getenv("SALESFORCE_USERNAME", ""),
        "password": os.getenv("SALESFORCE_PASSWORD", ""),
        "security_token": os.getenv("SALESFORCE_TOKEN", ""),
    }
    if instance_url:
        kwargs["instance"] = urlparse(instance_url).netloc or instance_url.replace("https://", "")
    return Salesforce(**kwargs)


def _is_auth_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(k in msg for k in ("INVALID_SESSION_ID", "SESSION EXPIRED", "EXPIRED", "AUTHENTICATION", "401"))


def _with_python_client(fn):
    """Run fn(salesforce_client); on an expired/invalid session, re-auth once and retry.

    OAuth/CLI session ids cached by get_salesforce_client() can expire; this transparently
    refreshes the connection instead of failing the request.
    """
    try:
        return fn(get_salesforce_client())
    except Exception as exc:
        if not _is_auth_error(exc):
            raise
        get_salesforce_client.cache_clear()
        return fn(get_salesforce_client())


# --- Backend primitives (the only place that branches MCP vs Python) ----------

def _assert_dml_succeeded(output: str) -> None:
    match = re.search(r"Successful:\s*(\d+)", output)
    if match and int(match.group(1)) >= 1:
        return
    raise RuntimeError(output.strip() or "Salesforce DML reported no successful records")


def _run_query(
    object_name: str,
    fields: list[str],
    where: str = "",
    order_by: str = "",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return matching records as a list of {field: value} dicts (backend-agnostic)."""
    if uses_mcp_server():
        text = mcp_query_records(object_name, fields, where, order_by, limit)
        return parse_query_records(text)

    soql = f"SELECT {', '.join(fields)} FROM {object_name}"
    if where:
        soql += f" WHERE {where}"
    if order_by:
        soql += f" ORDER BY {order_by}"
    if limit:
        soql += f" LIMIT {limit}"
    result = _with_python_client(lambda sf: sf.query(soql))
    return [{f: row.get(f) for f in fields} for row in result.get("records", [])]


def _run_dml(operation: str, object_name: str, record: dict[str, Any]) -> None:
    """Insert/update/delete a single record; raises on failure."""
    op = operation.lower()
    if uses_mcp_server():
        _assert_dml_succeeded(mcp_dml_records(op, object_name, [record]))
        return

    def _do(sf):
        sobject = getattr(sf, object_name)
        if op == "insert":
            sobject.create(dict(record))
        elif op == "update":
            sobject.update(record["Id"], {k: v for k, v in record.items() if k != "Id"})
        elif op == "delete":
            sobject.delete(record["Id"])
        else:
            raise ValueError(f"Unsupported DML operation: {operation}")

    _with_python_client(_do)


# --- High-level CRM operations (written once) --------------------------------

def fetch_record_by_id(record_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """Load a Lead or Contact by Salesforce Id."""
    safe_id = _escape_soql(record_id.strip())
    if not safe_id:
        return None, None
    for obj, fields in (("Lead", _LEAD_FIELDS), ("Contact", _CONTACT_FIELDS)):
        rows = _run_query(obj, fields, f"Id = '{safe_id}'", limit=1)
        if rows:
            rows[0]["_ObjectType"] = obj
            return rows[0], obj
    return None, None


def _soql_datetime(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _attach_outreach_task(record: dict[str, Any], task: dict[str, Any], obj: str) -> dict[str, Any]:
    record["_ObjectType"] = obj
    record["_LastEmailSubject"] = (task.get("Subject") or "").replace(
        _OUTREACH_EMAIL_TASK_PREFIX, "", 1
    ).strip()
    record["_LastEmailAt"] = task.get("CreatedDate") or ""
    return record


def fetch_leads_from_recent_outreach(limit: int = 10) -> list[dict[str, Any]]:
    """Leads or Contacts tied to the most recent agent-logged outreach email Tasks."""
    limit = max(1, min(limit, 50))
    prefix = _escape_soql(_OUTREACH_EMAIL_TASK_PREFIX)
    tasks = _run_query(
        "Task",
        ["WhoId", "Subject", "CreatedDate"],
        where=f"Subject LIKE '{prefix}%' AND WhoId != null",
        order_by="CreatedDate DESC",
        limit=min(limit * 8, 50),
    )

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        who_id = (task.get("WhoId") or "").strip()
        if not who_id or who_id in seen:
            continue
        seen.add(who_id)
        record, obj = fetch_record_by_id(who_id)
        if not record:
            continue
        results.append(_attach_outreach_task(record, task, obj))
        if len(results) >= limit:
            break
    return results


def fetch_leads_in_time_window(minutes: int, limit: int = 50) -> list[dict[str, Any]]:
    """Leads with outreach email Tasks or record changes within the last N minutes."""
    minutes = max(1, min(minutes, 7 * 24 * 60))
    limit = max(1, min(limit, 50))
    since = _soql_datetime(datetime.now(timezone.utc) - timedelta(minutes=minutes))
    prefix = _escape_soql(_OUTREACH_EMAIL_TASK_PREFIX)

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    tasks = _run_query(
        "Task",
        ["WhoId", "Subject", "CreatedDate"],
        where=f"CreatedDate >= {since} AND Subject LIKE '{prefix}%' AND WhoId != null",
        order_by="CreatedDate DESC",
        limit=min(limit * 8, 50),
    )
    for task in tasks:
        who_id = (task.get("WhoId") or "").strip()
        if not who_id or who_id in seen:
            continue
        record, obj = fetch_record_by_id(who_id)
        if not record or obj != "Lead":
            continue
        seen.add(who_id)
        results.append(_attach_outreach_task(record, task, obj))
        if len(results) >= limit:
            return results

    for row in _run_query(
        "Lead",
        _LEAD_FIELDS,
        where=f"(CreatedDate >= {since} OR LastModifiedDate >= {since})",
        order_by="LastModifiedDate DESC",
        limit=limit,
    ):
        lead_id = (row.get("Id") or "").strip()
        if lead_id and lead_id not in seen:
            seen.add(lead_id)
            results.append(row)
        if len(results) >= limit:
            break
    return results[:limit]


def find_person_by_email(email: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return (record, 'Lead'|'Contact') for the most recent match, or (None, None)."""
    where = f"Email = '{_escape_soql(email.strip())}'"
    for obj, fields in (("Lead", _LEAD_FIELDS), ("Contact", _CONTACT_FIELDS)):
        rows = _run_query(obj, fields, where, "LastModifiedDate DESC", 1)
        if rows:
            rows[0]["_ObjectType"] = obj
            return rows[0], obj
    return None, None


def _part_like_patterns(part_reference: str) -> list[str]:
    ref = part_reference.strip().strip(".,;")
    if not ref:
        return []
    patterns: list[str] = []
    tokens = [t for t in re.split(r"[\s\-_/]+", ref) if t]
    if len(tokens) >= 2:
        patterns.append("%" + "%".join(_escape_soql(t) for t in tokens) + "%")
    for variant in {ref, ref.replace("-", " "), ref.upper(), ref.replace("-", " ").upper()}:
        collapsed = re.sub(r"\s+", " ", variant.strip())
        if collapsed:
            patterns.append(f"%{_escape_soql(collapsed)}%")
    return list(dict.fromkeys(patterns))


def find_leads_by_part_enquiry(part_reference: str, limit: int = 50) -> list[dict[str, Any]]:
    """Leads linked to Tasks (or Lead.Description) mentioning a part / SKU enquiry."""
    limit = max(1, min(limit, 50))
    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    for pattern in _part_like_patterns(part_reference):
        where = (
            f"Id IN (SELECT WhoId FROM Task WHERE Description LIKE '{pattern}' AND WhoId != null)"
        )
        for row in _run_query("Lead", _LEAD_FIELDS, where, "LastModifiedDate DESC", limit):
            lead_id = row.get("Id")
            if lead_id and lead_id not in seen:
                seen.add(lead_id)
                results.append(row)
            if len(results) >= limit:
                return results

    for pattern in _part_like_patterns(part_reference):
        for row in _run_query(
            "Lead",
            _LEAD_FIELDS,
            f"Description LIKE '{pattern}'",
            "LastModifiedDate DESC",
            limit,
        ):
            lead_id = row.get("Id")
            if lead_id and lead_id not in seen:
                seen.add(lead_id)
                results.append(row)
            if len(results) >= limit:
                return results

    return results[:limit]


def search_leads_and_contacts(
    search_text: str = "",
    email: str = "",
    company: str = "",
    limit: int = 10,
) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 25))

    if email.strip():
        rec, _ = find_person_by_email(email.strip())
        return [rec] if rec else []

    records: list[dict[str, Any]] = []
    if company.strip():
        where = f"Company LIKE '%{_escape_soql(company.strip())}%'"
        for row in _run_query("Lead", _LEAD_FIELDS, where, "LastModifiedDate DESC", limit):
            row["_ObjectType"] = "Lead"
            records.append(row)

    if search_text.strip():
        where = f"Name LIKE '%{_escape_soql(search_text.strip())}%'"
        for obj, fields in (("Lead", _LEAD_FIELDS), ("Contact", _CONTACT_FIELDS)):
            for row in _run_query(obj, fields, where, "LastModifiedDate DESC", limit):
                row["_ObjectType"] = obj
                records.append(row)
            if len(records) >= limit:
                break

    return records[:limit]


def upsert_lead(
    email: str,
    last_name: str,
    first_name: str = "",
    company: str = "Unknown",
    title: str = "",
    description: str = "",
) -> dict[str, Any]:
    existing, obj = find_person_by_email(email)
    if existing and obj == "Contact":
        return {"action": "found_contact", "id": existing["Id"], "object": obj}

    payload: dict[str, Any] = {
        "Email": email,
        "LastName": last_name or email.split("@")[0],
        "Company": company or "Unknown",
        "LeadSource": "Product Marketing Agent",
    }
    if first_name:
        payload["FirstName"] = first_name
    if title:
        payload["Title"] = title
    if description:
        payload["Description"] = description[:32000]

    if existing and obj == "Lead":
        _run_dml("update", "Lead", {**payload, "Id": existing["Id"]})
        return {"action": "updated", "id": existing["Id"], "object": "Lead"}

    _run_dml("insert", "Lead", payload)
    created, _ = find_person_by_email(email)
    return {"action": "created", "id": created["Id"] if created else "", "object": "Lead"}


def _build_email_task(email: str, subject: str, body: str, who_id: str, details: dict[str, Any]) -> dict[str, Any]:
    """Assemble a rich, fully-detailed Salesforce Task for a sent outreach email."""
    sent_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    description = (
        "=== Outreach Email Logged by Product Marketing Agent ===\n"
        f"Recipient: {details.get('recipient_name') or '-'} <{email}>\n"
        f"Company: {details.get('company') or '-'}\n"
        f"Sent By: {details.get('sent_by') or '-'}\n"
        f"Sent At: {sent_at}\n"
        f"Reason / Topic: {details.get('reason') or subject}\n"
        f"Products / Parts: {details.get('products') or '-'}\n\n"
        f"--- Subject ---\n{subject}\n\n"
        f"--- Email Body ---\n{body}"
    )
    return {
        "Subject": f"Email sent: {subject[:180]}",
        "Description": description[:32000],
        "Status": "Completed",
        "Priority": "Normal",
        "ActivityDate": datetime.now(timezone.utc).date().isoformat(),
        "WhoId": who_id,
    }


def log_email_activity(
    email: str,
    subject: str,
    body: str,
    recipient_name: str = "",
    company: str = "",
    products: str = "",
    reason: str = "",
    sent_by: str = "",
) -> dict[str, Any]:
    """Find/create the Lead or Contact, then log a completed Task with full email detail."""
    record, obj = find_person_by_email(email)
    if record:
        who_id = record["Id"]
    else:
        parts = recipient_name.strip().split(None, 1)
        first = parts[0] if len(parts) > 1 else ""
        last = parts[1] if len(parts) > 1 else (parts[0] if parts else email.split("@")[0])
        upsert = upsert_lead(
            email=email,
            first_name=first,
            last_name=last,
            company=company,
            description=f"Outreach email subject: {subject}",
        )
        who_id, obj = upsert["id"], upsert["object"]

    details = {
        "recipient_name": recipient_name,
        "company": company,
        "products": products,
        "reason": reason,
        "sent_by": sent_by,
    }
    _run_dml("insert", "Task", _build_email_task(email, subject, body, who_id, details))
    return {"task_id": "logged", "who_id": who_id, "object": obj, "email": email}
