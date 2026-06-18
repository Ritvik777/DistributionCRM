"""CRM Agent — owns Salesforce data operations (reads, aggregates, record DML, schema)."""

import logging

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from agents.chat import (
    build_turn_context,
    extract_part_reference,
    infer_leads_limit,
    is_leads_by_part_enquiry,
    is_leads_time_window_fetch,
    is_recent_leads_list_fetch,
    is_simple_leads_fetch,
    is_singular_lead_fetch,
    parse_leads_time_window_minutes,
)
from agents.tools import (
    call_tools,
    salesforce_query_records,
    salesforce_aggregate_query,
    salesforce_dml_records,
    salesforce_search_leads,
    salesforce_upsert_lead,
    salesforce_search_objects,
    salesforce_describe_object,
)
from llm import get_llm
from observability import merge_node_config
from services.salesforce_repository import (
    fetch_latest_leads,
    fetch_leads_for_time_window,
    fetch_recent_outreach_recipients,
    find_leads_by_part_enquiry,
    format_leads_markdown,
    is_salesforce_configured,
)

logger = logging.getLogger(__name__)

_CRM_TOOLS = [
    salesforce_query_records,
    salesforce_aggregate_query,
    salesforce_search_leads,
    salesforce_upsert_lead,
    salesforce_dml_records,
    salesforce_search_objects,
    salesforce_describe_object,
]


def crm_research(state: AgentState, config: RunnableConfig | None = None) -> dict:
    turn = build_turn_context(state)
    question = (state.get("question") or "").strip()

    if not is_salesforce_configured():
        return {
            "context": "",
            "answer": "Salesforce is not configured. Add SALESFORCE_* variables to `.env` (see README).",
            "steps": ["CRM Research → blocked (not configured)"],
        }

    if is_leads_by_part_enquiry(question):
        part_ref = extract_part_reference(question)
        if not part_ref:
            return {
                "context": "",
                "answer": (
                    "Please include the part number or SKU (e.g. `Part No. LED-RED-5MM`) "
                    "so I can search lead enquiry history in Salesforce."
                ),
                "steps": ["CRM Research → part enquiry query missing part reference"],
            }
        try:
            rows = find_leads_by_part_enquiry(part_ref, limit=50)
            heading = f"## Leads who enquired about {part_ref}"
            if not rows:
                answer = (
                    f"No Salesforce leads found with enquiry activity for **{part_ref}**.\n\n"
                    "Enquiries are logged on **Task** records when outreach emails mention a product/part. "
                    "Try a slightly different spelling (e.g. `LED 5MM RED` vs `LED-RED-5MM`)."
                )
            else:
                answer = format_leads_markdown(rows, limit=len(rows), heading=heading)
            return {
                "context": str(rows),
                "answer": answer,
                "steps": [f"CRM Research → part enquiry leads ({part_ref}, {len(rows)} found)"],
            }
        except Exception as exc:
            return {
                "context": "",
                "answer": f"Could not search leads by part enquiry: {exc}",
                "steps": [f"CRM Research → part enquiry failed ({exc})"],
            }

    if is_leads_time_window_fetch(question) or is_simple_leads_fetch(question):
        try:
            limit = infer_leads_limit(question)
            window_minutes = parse_leads_time_window_minutes(question)

            if window_minutes is not None:
                rows = fetch_leads_for_time_window(window_minutes, limit=limit)
                unit = (
                    f"{window_minutes} minutes"
                    if window_minutes < 60
                    else f"{window_minutes // 60} hour(s)"
                    if window_minutes < 24 * 60
                    else f"{window_minutes // (24 * 60)} day(s)"
                )
                heading = f"## CRM leads (last {unit})"
                step = f"CRM Research → leads in time window ({unit}, limit={limit})"
            elif is_singular_lead_fetch(question):
                rows = fetch_recent_outreach_recipients(limit=limit)
                if not rows:
                    rows = fetch_latest_leads(limit=limit)
                heading = (
                    "## Last outreach recipient"
                    if limit == 1
                    else "## Recent outreach recipients"
                )
                step = f"CRM Research → last outreach recipient (limit={limit})"
            elif is_recent_leads_list_fetch(question):
                rows = fetch_recent_outreach_recipients(limit=limit)
                if not rows:
                    rows = fetch_latest_leads(limit=limit)
                heading = "## Recent CRM leads (by last outreach activity)"
                step = f"CRM Research → recent outreach leads (limit={limit})"
            else:
                rows = fetch_latest_leads(limit=limit)
                heading = None
                step = f"CRM Research → fast-path leads fetch (limit={limit})"
            return {
                "context": str(rows),
                "answer": format_leads_markdown(rows, limit=limit, heading=heading),
                "steps": [step],
            }
        except Exception as exc:
            return {
                "context": "",
                "answer": f"Could not fetch leads from Salesforce: {exc}",
                "steps": [f"CRM Research → fast-path failed ({exc})"],
            }

    ctx, log, _kb_sources = call_tools(
        turn,
        tools=_CRM_TOOLS,
        config=config,
        system_prompt=(
            "You are a Salesforce CRM specialist. Use the Salesforce tools to fulfill the user's request.\n"
            "- Read/list records: salesforce_query_records (for latest leads use objectName Lead, orderBy LastModifiedDate DESC; for recent outreach use logged Email sent Tasks)\n"
            "- Product/part enquiries are stored on Task.Description (field 'Products / Parts') linked via WhoId to Lead/Contact. "
            "For 'leads who enquired about part X', query Task with whereClause Description LIKE '%part tokens%' "
            "then query Lead WHERE Id IN (those WhoIds), or use Lead Description LIKE as a fallback.\n"
            "- Counts/grouping: salesforce_aggregate_query\n"
            "- Create/update/delete records: salesforce_dml_records (or salesforce_upsert_lead for leads)\n"
            "- Inspect schema: salesforce_describe_object, salesforce_search_objects\n"
            "Call the minimum tools needed, then stop. Do not invent record IDs."
        ),
    )
    return {"context": ctx, "steps": [f"CRM Research → {', '.join(log) or 'no tools called'}"]}


def crm_generate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    if state.get("answer"):
        return {"steps": ["CRM Generate → passthrough"]}

    llm = get_llm(temperature=0)
    turn = build_turn_context(state)
    ctx = state.get("context", "")
    resp = llm.invoke(
        "You are a Salesforce CRM specialist. Using ONLY the tool results below, answer the user's request.\n"
        "- Present records or query results as a clean Markdown table when appropriate.\n"
        "- For aggregate/count results, summarize the grouping clearly.\n"
        "- If the tool results contain an error, explain it plainly and suggest the fix.\n\n"
        f"Tool results:\n{ctx}\n\n{turn}\nAnswer:",
        config=merge_node_config(
            config,
            metadata={"node": "crm_generate", "agent_type": "crm"},
            tags=["agent:crm", "phase:generate"],
        ) or None,
    )
    return {"answer": resp.content, "steps": [f"CRM Generate → {len(resp.content)} chars"]}
