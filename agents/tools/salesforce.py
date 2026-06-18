from langchain_core.tools import tool


def _require_salesforce() -> str | None:
    from services.salesforce_client import is_salesforce_configured

    if not is_salesforce_configured():
        return "ERROR: Salesforce not configured. Add SALESFORCE_* vars to .env (see README)."
    return None


def _require_mcp() -> str | None:
    """Advanced CRM tools (aggregate, describe, object search) require the MCP backend."""
    from services.salesforce_client import is_salesforce_configured, uses_mcp_server

    if not is_salesforce_configured():
        return "ERROR: Salesforce not configured. Add SALESFORCE_* vars to .env (see README)."
    if not uses_mcp_server():
        return "ERROR: This CRM operation needs the MCP backend. Set SALESFORCE_BACKEND=mcp (Node/npx required)."
    return None


def _call_mcp(tool_name: str, args: dict) -> str:
    from services.salesforce_mcp import call_mcp_tool

    return call_mcp_tool(tool_name, {k: v for k, v in args.items() if v not in ("", None)})


@tool
def salesforce_query_records(
    object_name: str,
    fields: list[str],
    where_clause: str = "",
    order_by: str = "",
    limit: int = 10,
) -> str:
    """Query Salesforce records (same tool as mcp-server-salesforce). Use for Leads, Contacts, Tasks, etc."""
    from services.salesforce_client import uses_mcp_server
    from services.salesforce_mcp import call_mcp_tool

    err = _require_salesforce()
    if err:
        return err

    args = {
        "objectName": object_name,
        "fields": fields,
        "limit": limit,
    }
    if where_clause:
        args["whereClause"] = where_clause
    if order_by:
        args["orderBy"] = order_by

    try:
        if uses_mcp_server():
            return call_mcp_tool("salesforce_query_records", args)
        return _python_query_records(object_name, fields, where_clause, order_by, limit)
    except Exception as e:
        return f"Salesforce query failed: {e}"


def _python_query_records(
    object_name: str,
    fields: list[str],
    where_clause: str,
    order_by: str,
    limit: int,
) -> str:
    from services.salesforce_repository import query_records_as_text

    return query_records_as_text(object_name, fields, where_clause, order_by, limit)


@tool
def salesforce_dml_records(
    operation: str,
    object_name: str,
    records: list[dict],
    external_id_field: str = "",
) -> str:
    """Insert/update/delete Salesforce records (same tool as mcp-server-salesforce)."""
    from services.salesforce_client import uses_mcp_server
    from services.salesforce_mcp import call_mcp_tool

    err = _require_salesforce()
    if err:
        return err

    try:
        if uses_mcp_server():
            args = {
                "operation": operation,
                "objectName": object_name,
                "records": records,
            }
            if external_id_field:
                args["externalIdField"] = external_id_field
            return call_mcp_tool("salesforce_dml_records", args)
        return _python_dml_records(operation, object_name, records, external_id_field)
    except Exception as e:
        return f"Salesforce DML failed: {e}"


def _python_dml_records(
    operation: str,
    object_name: str,
    records: list[dict],
    external_id_field: str,
) -> str:
    from services.salesforce_client import get_salesforce_client

    sf = get_salesforce_client().sobject(object_name)
    op = operation.lower()
    if op == "insert":
        result = sf.create(records if len(records) > 1 else records[0])
    elif op == "update":
        result = sf.update(records if len(records) > 1 else records[0])
    elif op == "delete":
        ids = [r["Id"] for r in records]
        result = sf.delete(ids if len(ids) > 1 else ids[0])
    elif op == "upsert":
        if not external_id_field:
            return "ERROR: external_id_field required for upsert"
        result = sf.upsert(records if len(records) > 1 else records[0], external_id_field)
    else:
        return f"ERROR: unsupported operation {operation}"

    results = result if isinstance(result, list) else [result]
    ok = sum(1 for r in results if r.get("success", True))
    return f"{operation.upper()} operation completed.\nProcessed {len(results)} records:\n- Successful: {ok}\n- Failed: {len(results) - ok}"


@tool
def salesforce_search_leads(
    search_text: str = "",
    email: str = "",
    company: str = "",
    limit: int = 10,
) -> str:
    """Search Salesforce CRM for existing Leads and Contacts by name, email, or company."""
    from services.salesforce_client import search_leads_and_contacts

    err = _require_salesforce()
    if err:
        return err

    try:
        records = search_leads_and_contacts(
            search_text=search_text,
            email=email,
            company=company,
            limit=limit,
        )
    except Exception as e:
        return f"Salesforce search failed: {e}"

    if not records:
        return "No matching Leads or Contacts found in Salesforce."

    lines = []
    for r in records:
        obj = r.get("_ObjectType", "Lead")
        name = r.get("Name", "Unknown")
        em = r.get("Email") or "no email"
        company_val = r.get("Company") or r.get("Title") or ""
        status = r.get("Status", "")
        line = f"**{name}** ({obj}) — {em}"
        if company_val:
            line += f" | {company_val}"
        if status:
            line += f" | Status: {status}"
        lines.append(line)
    return f"Found {len(lines)} CRM record(s):\n\n" + "\n\n".join(lines)


@tool
def salesforce_upsert_lead(
    email: str,
    last_name: str,
    first_name: str = "",
    company: str = "Unknown",
    title: str = "",
    notes: str = "",
) -> str:
    """Create or update a Lead in Salesforce CRM. Use when the user asks to add someone to the CRM."""
    from services.salesforce_client import upsert_lead

    err = _require_salesforce()
    if err:
        return err

    try:
        result = upsert_lead(
            email=email,
            last_name=last_name,
            first_name=first_name,
            company=company,
            title=title,
            description=notes,
        )
    except Exception as e:
        return f"Salesforce upsert failed: {e}"

    action = result.get("action", "updated")
    return f"CRM {action}: {result.get('object')} Id={result.get('id')} for {email}"


@tool
def salesforce_aggregate_query(
    object_name: str,
    select_fields: list[str],
    group_by_fields: list[str],
    where_clause: str = "",
    having_clause: str = "",
    order_by: str = "",
    limit: int = 0,
) -> str:
    """Run a Salesforce aggregate/GROUP BY query (COUNT, SUM, AVG, etc.). E.g. count opportunities by stage."""
    err = _require_mcp()
    if err:
        return err
    try:
        return _call_mcp("salesforce_aggregate_query", {
            "objectName": object_name,
            "selectFields": select_fields,
            "groupByFields": group_by_fields,
            "whereClause": where_clause,
            "havingClause": having_clause,
            "orderBy": order_by,
            "limit": limit or None,
        })
    except Exception as e:
        return f"Salesforce aggregate query failed: {e}"


@tool
def salesforce_search_objects(search_pattern: str) -> str:
    """Find Salesforce objects (standard or custom) by partial name. E.g. 'Account' finds Account, AccountHistory."""
    err = _require_mcp()
    if err:
        return err
    try:
        return _call_mcp("salesforce_search_objects", {"searchPattern": search_pattern})
    except Exception as e:
        return f"Salesforce object search failed: {e}"


@tool
def salesforce_describe_object(object_name: str) -> str:
    """Describe a Salesforce object's schema: fields, types, picklist values, and relationships."""
    err = _require_mcp()
    if err:
        return err
    try:
        return _call_mcp("salesforce_describe_object", {"objectName": object_name})
    except Exception as e:
        return f"Salesforce describe failed: {e}"
