"""Salesforce integration — uses your TypeScript MCP server by default, Python REST as fallback."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import requests

from services.salesforce_mcp import (
    call_mcp_tools_batch,
    mcp_dml_records,
    mcp_query_records,
    parse_query_records,
    salesforce_backend,
)


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


# --- Python REST fallback (simple-salesforce) --------------------------------

def _oauth_client_credentials_token() -> tuple[str, str]:
    instance_url = os.getenv("SALESFORCE_INSTANCE_URL", "").rstrip("/")
    client_id = os.getenv("SALESFORCE_CLIENT_ID", "")
    client_secret = os.getenv("SALESFORCE_CLIENT_SECRET", "")
    resp = requests.post(
        f"{instance_url}/services/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
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
        host = urlparse(instance_url).netloc
        return Salesforce(instance=host, session_id=token)

    username = os.getenv("SALESFORCE_USERNAME", "")
    password = os.getenv("SALESFORCE_PASSWORD", "")
    token = os.getenv("SALESFORCE_TOKEN", "")
    instance_url = os.getenv("SALESFORCE_INSTANCE_URL", "").strip()

    kwargs: dict[str, Any] = {
        "username": username,
        "password": password,
        "security_token": token,
    }
    if instance_url:
        kwargs["instance"] = urlparse(instance_url).netloc or instance_url.replace("https://", "")
    return Salesforce(**kwargs)


def _python_find_person_by_email(email: str) -> tuple[dict[str, Any] | None, str | None]:
    sf = get_salesforce_client()
    safe = _escape_soql(email.strip())
    lead_result = sf.query(
        f"SELECT Id, Name, Email, Company, Title, Status FROM Lead "
        f"WHERE Email = '{safe}' ORDER BY LastModifiedDate DESC LIMIT 1"
    )
    if lead_result.get("totalSize", 0):
        return lead_result["records"][0], "Lead"

    contact_result = sf.query(
        f"SELECT Id, Name, Email, Title FROM Contact "
        f"WHERE Email = '{safe}' ORDER BY LastModifiedDate DESC LIMIT 1"
    )
    if contact_result.get("totalSize", 0):
        return contact_result["records"][0], "Contact"
    return None, None


def _mcp_find_person_by_email(email: str) -> tuple[dict[str, str] | None, str | None]:
    safe = _escape_soql(email.strip())
    for obj, fields in (
        ("Lead", ["Id", "Name", "Email", "Company", "Title", "Status"]),
        ("Contact", ["Id", "Name", "Email", "Title"]),
    ):
        text = mcp_query_records(
            object_name=obj,
            fields=fields,
            where_clause=f"Email = '{safe}'",
            order_by="LastModifiedDate DESC",
            limit=1,
        )
        rows = parse_query_records(text)
        if rows:
            rows[0]["_ObjectType"] = obj
            return rows[0], obj
    return None, None


def find_person_by_email(email: str) -> tuple[dict[str, Any] | None, str | None]:
    if uses_mcp_server():
        return _mcp_find_person_by_email(email)
    return _python_find_person_by_email(email)


def search_leads_and_contacts(
    search_text: str = "",
    email: str = "",
    company: str = "",
    limit: int = 10,
) -> list[dict[str, Any]]:
    if uses_mcp_server():
        return _mcp_search_leads_and_contacts(search_text, email, company, limit)
    return _python_search_leads_and_contacts(search_text, email, company, limit)


def _mcp_search_leads_and_contacts(
    search_text: str,
    email: str,
    company: str,
    limit: int,
) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 25))
    records: list[dict[str, Any]] = []

    if email.strip():
        rec, obj = _mcp_find_person_by_email(email.strip())
        if rec:
            return [rec]

    if company.strip():
        c = _escape_soql(company.strip())
        text = mcp_query_records(
            object_name="Lead",
            fields=["Id", "Name", "Email", "Company", "Title", "Status"],
            where_clause=f"Company LIKE '%{c}%'",
            order_by="LastModifiedDate DESC",
            limit=limit,
        )
        for row in parse_query_records(text):
            row["_ObjectType"] = "Lead"
            records.append(row)

    if search_text.strip():
        t = _escape_soql(search_text.strip())
        for obj, fields in (
            ("Lead", ["Id", "Name", "Email", "Company", "Title", "Status"]),
            ("Contact", ["Id", "Name", "Email", "Title"]),
        ):
            text = mcp_query_records(
                object_name=obj,
                fields=fields,
                where_clause=f"Name LIKE '%{t}%'",
                order_by="LastModifiedDate DESC",
                limit=limit,
            )
            for row in parse_query_records(text):
                row["_ObjectType"] = obj
                records.append(row)
            if len(records) >= limit:
                break

    return records[:limit]


def _python_search_leads_and_contacts(
    search_text: str,
    email: str,
    company: str,
    limit: int,
) -> list[dict[str, Any]]:
    sf = get_salesforce_client()
    limit = max(1, min(limit, 25))
    records: list[dict[str, Any]] = []

    if email.strip():
        rec, obj = _python_find_person_by_email(email.strip())
        if rec:
            rec["_ObjectType"] = obj
            return [rec]

    if company.strip():
        c = _escape_soql(company.strip())
        result = sf.query(
            f"SELECT Id, Name, Email, Company, Title, Status FROM Lead "
            f"WHERE Company LIKE '%{c}%' ORDER BY LastModifiedDate DESC LIMIT {limit}"
        )
        for row in result.get("records", []):
            row["_ObjectType"] = "Lead"
            records.append(row)

    if search_text.strip():
        t = _escape_soql(search_text.strip())
        for obj, fields in (
            ("Lead", "Id, Name, Email, Company, Title, Status"),
            ("Contact", "Id, Name, Email, Title"),
        ):
            result = sf.query(
                f"SELECT {fields} FROM {obj} WHERE Name LIKE '%{t}%' "
                f"ORDER BY LastModifiedDate DESC LIMIT {limit}"
            )
            for row in result.get("records", []):
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
    if uses_mcp_server():
        return _mcp_upsert_lead(email, last_name, first_name, company, title, description)
    return _python_upsert_lead(email, last_name, first_name, company, title, description)


def _mcp_upsert_lead(
    email: str,
    last_name: str,
    first_name: str,
    company: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    existing, obj = _mcp_find_person_by_email(email)
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
        payload["Id"] = existing["Id"]
        mcp_dml_records("update", "Lead", [payload])
        return {"action": "updated", "id": existing["Id"], "object": "Lead"}

    mcp_dml_records("insert", "Lead", [payload])
    created, _ = _mcp_find_person_by_email(email)
    return {
        "action": "created",
        "id": created["Id"] if created else "",
        "object": "Lead",
    }


def _python_upsert_lead(
    email: str,
    last_name: str,
    first_name: str,
    company: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    sf = get_salesforce_client()
    existing, obj = _python_find_person_by_email(email)
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
        sf.Lead.update(existing["Id"], payload)
        return {"action": "updated", "id": existing["Id"], "object": "Lead"}

    created = sf.Lead.create(payload)
    return {"action": "created", "id": created["id"], "object": "Lead"}


def log_email_activity(
    email: str,
    subject: str,
    body: str,
    recipient_name: str = "",
    company: str = "",
) -> dict[str, Any]:
    if uses_mcp_server():
        return _mcp_log_email_activity(email, subject, body, recipient_name, company)
    return _python_log_email_activity(email, subject, body, recipient_name, company)


def _mcp_log_email_activity(
    email: str,
    subject: str,
    body: str,
    recipient_name: str,
    company: str,
) -> dict[str, Any]:
    safe = _escape_soql(email.strip())
    calls: list[tuple[str, dict[str, Any]]] = [
        (
            "salesforce_query_records",
            {
                "objectName": "Lead",
                "fields": ["Id", "Name", "Email"],
                "whereClause": f"Email = '{safe}'",
                "limit": 1,
            },
        ),
        (
            "salesforce_query_records",
            {
                "objectName": "Contact",
                "fields": ["Id", "Name", "Email"],
                "whereClause": f"Email = '{safe}'",
                "limit": 1,
            },
        ),
    ]
    outputs = call_mcp_tools_batch(calls)
    who_id = ""
    obj = "Lead"
    for text, object_type in zip(outputs, ("Lead", "Contact")):
        rows = parse_query_records(text)
        if rows:
            who_id = rows[0].get("Id", "")
            obj = object_type
            break

    if not who_id:
        parts = recipient_name.strip().split(None, 1) if recipient_name.strip() else []
        first = parts[0] if len(parts) > 1 else ""
        last = parts[1] if len(parts) > 1 else (parts[0] if parts else email.split("@")[0])
        upsert = _mcp_upsert_lead(
            email=email,
            first_name=first,
            last_name=last,
            company=company,
            title="",
            description=f"Outreach email subject: {subject}",
        )
        who_id = upsert["id"]
        obj = upsert["object"]

    task_payload = {
        "Subject": f"Email sent: {subject[:200]}",
        "Description": body[:32000],
        "Status": "Completed",
        "Priority": "Normal",
        "WhoId": who_id,
    }
    call_mcp_tools_batch([("salesforce_dml_records", {
        "operation": "insert",
        "objectName": "Task",
        "records": [task_payload],
    })])
    return {"task_id": "logged", "who_id": who_id, "object": obj, "email": email}


def _python_log_email_activity(
    email: str,
    subject: str,
    body: str,
    recipient_name: str,
    company: str,
) -> dict[str, Any]:
    sf = get_salesforce_client()
    record, obj = _python_find_person_by_email(email)

    if not record:
        parts = recipient_name.strip().split(None, 1) if recipient_name.strip() else []
        first = parts[0] if parts else ""
        last = parts[1] if len(parts) > 1 else (parts[0] if parts else email.split("@")[0])
        upsert = _python_upsert_lead(
            email=email,
            first_name=first if len(parts) > 1 else "",
            last_name=last,
            company=company,
            description=f"Outreach email subject: {subject}",
        )
        who_id = upsert["id"]
        obj = upsert["object"]
    else:
        who_id = record["Id"]

    result = sf.Task.create({
        "Subject": f"Email sent: {subject[:200]}",
        "Description": body[:32000],
        "Status": "Completed",
        "Priority": "Normal",
        "WhoId": who_id,
    })
    return {"task_id": result["id"], "who_id": who_id, "object": obj, "email": email}
