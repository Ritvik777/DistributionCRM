from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from vector_db import search_with_scores
from llm import get_llm
from observability import merge_node_config, log_span
import json


@tool
def search_knowledge_base(query: str) -> str:
    """Search internal product docs stored in Qdrant."""
    results = search_with_scores(query, top_k=8)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(f"[{score:.3f}] {text}" for text, score in results)


@tool
def web_search(query: str) -> str:
    """Search the live web via DuckDuckGo."""
    from duckduckgo_search import DDGS
    errors = []
    for backend in ("html", "lite"):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, backend=backend))
            if not results:
                continue
            return "\n\n".join(f"**{r['title']}**\n{r['body']}" for r in results)
        except Exception as e:
            errors.append(f"{backend}: {e}")
    if errors:
        return "WEB_SEARCH_UNAVAILABLE: " + " | ".join(errors)
    return "No web results found."


@tool
def apollo_search(job_titles: str, location: str = "", industry: str = "", limit: int = 5) -> str:
    """Search Apollo.io for leads by job title, location, and industry. Returns names, titles, companies, and verified emails for outreach."""
    import os
    import requests

    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key or api_key == "your-apollo-api-key-here":
        return "ERROR: APOLLO_API_KEY not configured in .env"

    headers = {"Content-Type": "application/json", "Cache-Control": "no-cache", "X-Api-Key": api_key}
    titles = [t.strip() for t in job_titles.split(",")]

    # Step 1: Search Apollo for people matching criteria
    search_payload = {
        "person_titles": titles,
        "page": 1,
        "per_page": min(limit, 10),
    }
    if location:
        search_payload["person_locations"] = [location]
    if industry:
        search_payload["organization_industries"] = [industry]

    try:
        resp = requests.post(
            "https://api.apollo.io/api/v1/mixed_people/api_search",
            headers=headers, json=search_payload, timeout=15,
        )
        if resp.status_code != 200:
            return f"Apollo search error (status {resp.status_code}): {resp.text[:200]}"

        people = resp.json().get("people", [])
        if not people:
            return "No leads found matching your criteria."

        # Step 2: Enrich each person by ID to get email + full details
        results = []
        for p in people[:limit]:
            pid = p.get("id", "")
            if not pid:
                continue

            try:
                enrich = requests.post(
                    "https://api.apollo.io/api/v1/people/match",
                    headers=headers, timeout=10,
                    json={"id": pid, "reveal_personal_emails": True},
                )
                if enrich.status_code != 200:
                    continue
                ep = enrich.json().get("person", {})
            except Exception:
                continue

            name = ep.get("name") or p.get("first_name", "Unknown")
            email = ep.get("email", "")
            title = ep.get("title") or p.get("title", "N/A")
            org = ep.get("organization", {})
            company = org.get("name") or p.get("organization", {}).get("name", "N/A")
            industry_val = org.get("industry", "N/A")
            emp_count = org.get("estimated_num_employees", "N/A")
            city = ep.get("city", "")
            linkedin = ep.get("linkedin_url", "")

            lead = f"**{name}** — {title} at {company}"
            lead += f"\n  Industry: {industry_val} | Size: {emp_count} employees"
            if city:
                lead += f" | Location: {city}"
            lead += f"\n  Email: {email}" if email else "\n  Email: not found"
            if linkedin:
                lead += f"\n  LinkedIn: {linkedin}"
            results.append(lead)

        if not results:
            return "Found leads but could not enrich any with email data."

        enriched = sum(1 for r in results if "Email: not found" not in r)
        return f"Found {len(results)} leads ({enriched} with verified emails):\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Apollo search failed: {e}"


@tool
def salesforce_query_records(
    object_name: str,
    fields: list[str],
    where_clause: str = "",
    order_by: str = "",
    limit: int = 10,
) -> str:
    """Query Salesforce records (same tool as mcp-server-salesforce). Use for Leads, Contacts, Tasks, etc."""
    from services.salesforce_client import is_salesforce_configured, uses_mcp_server
    from services.salesforce_mcp import call_mcp_tool, mcp_query_records

    if not is_salesforce_configured():
        return "ERROR: Salesforce not configured. Add SALESFORCE_* vars to .env (see README)."

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
    from services.salesforce_client import get_salesforce_client

    soql = f"SELECT {', '.join(fields)} FROM {object_name}"
    if where_clause:
        soql += f" WHERE {where_clause}"
    if order_by:
        soql += f" ORDER BY {order_by}"
    if limit:
        soql += f" LIMIT {limit}"
    result = get_salesforce_client().query(soql)
    rows = result.get("records", [])
    if not rows:
        return "Query returned 0 records."
    lines = []
    for i, row in enumerate(rows, 1):
        parts = [f" {f}: {row.get(f)}" for f in fields if f in row]
        lines.append(f"Record {i}:\n" + "\n".join(parts))
    return f"Query returned {len(rows)} records:\n\n" + "\n\n".join(lines)


@tool
def salesforce_dml_records(
    operation: str,
    object_name: str,
    records: list[dict],
    external_id_field: str = "",
) -> str:
    """Insert/update/delete Salesforce records (same tool as mcp-server-salesforce)."""
    from services.salesforce_client import is_salesforce_configured, uses_mcp_server
    from services.salesforce_mcp import call_mcp_tool, mcp_dml_records

    if not is_salesforce_configured():
        return "ERROR: Salesforce not configured. Add SALESFORCE_* vars to .env (see README)."

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
    from services.salesforce_client import is_salesforce_configured, search_leads_and_contacts

    if not is_salesforce_configured():
        return "ERROR: Salesforce not configured. Add SALESFORCE_* vars to .env (see README)."

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
    from services.salesforce_client import is_salesforce_configured, upsert_lead

    if not is_salesforce_configured():
        return "ERROR: Salesforce not configured. Add SALESFORCE_* vars to .env (see README)."

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
@log_span(span_type="tool", name="send_email")
def send_email(to_email: str, subject: str, html_body: str) -> str:
    """Send a personalized marketing email via Brevo. Provide recipient email, subject line, and HTML body."""
    import os

    import requests

    api_key = os.getenv("BREVO_API_KEY")
    if not api_key or api_key == "your-brevo-api-key-here":
        return "ERROR: BREVO_API_KEY not configured in .env"

    from_email = os.getenv("BREVO_FROM_EMAIL")
    if not from_email or from_email == "you@yourcompany.com":
        return "ERROR: BREVO_FROM_EMAIL not configured in .env"

    from_name = os.getenv("BREVO_FROM_NAME", "Product Marketing")
    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_body,
    }

    try:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": api_key, "Content-Type": "application/json", "accept": "application/json"},
            json=payload,
            timeout=15,
        )
        if resp.status_code in (200, 201, 202):
            return f"SENT to {to_email} (status {resp.status_code})"
        return f"FAILED (status {resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"


# GalileoCallback (via gtm_retrieve/outreach_research nodes) logs LLM + tool calls;
# no @log_span to avoid duplicate call_tools span (same work as gtm_retrieve).
def call_tools(question, tools, system_prompt, config=None):
    """LLM picks which tools to call, runs them, returns results.
    Pass config from the graph so LLM/tool spans nest under the parent node."""
    tool_map = {t.name: t for t in tools}
    try:
        llm = get_llm().bind_tools(tools)
    except Exception as exc:
        return f"LLM_UNAVAILABLE: {exc}", []
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    log = []
    seen_calls = set()
    invoke_config = merge_node_config(
        config,
        metadata={"component": "tools", "question": question},
        tags=["agent:shared", "phase:tool-routing"],
    )
    for _ in range(3):
        try:
            resp = llm.invoke(msgs, config=invoke_config or None)
        except Exception as exc:
            return f"LLM_ERROR: {exc}", log
        msgs.append(resp)
        if not resp.tool_calls:
            break
        for tc in resp.tool_calls:
            signature = f"{tc['name']}::{json.dumps(tc.get('args', {}), sort_keys=True, default=str)}"
            if signature in seen_calls:
                msgs.append(ToolMessage(content="Skipped duplicate tool call.", tool_call_id=tc["id"]))
                continue
            seen_calls.add(signature)
            # Per-tool config so Galileo shows which tool was used + args in span metadata
            args_str = json.dumps(tc.get("args", {}), default=str)[:200]
            tool_config = merge_node_config(
                invoke_config,
                metadata={
                    "tool_name": tc["name"],
                    "tool_args": args_str,
                },
                tags=["tool", f"tool:{tc['name']}"],
            )
            try:
                out = tool_map[tc["name"]].invoke(tc["args"], config=tool_config)
            except Exception as exc:
                out = f"TOOL_ERROR[{tc['name']}]: {exc}"
            log.append(tc["name"])
            msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

    context = "\n\n".join(m.content for m in msgs if isinstance(m, ToolMessage))
    return context or "No context found.", log
