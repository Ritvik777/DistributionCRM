"""CRM Agent — owns all Salesforce operations (reads, aggregates, record DML, Apex).

Routed to whenever the supervisor classifies a message as CRM/Salesforce work.

  crm_research  → fast-path table for simple lead fetches, else an LLM tool loop
  crm_generate  → formats tool output into a final answer (passthrough for fast path)
"""

import logging
import re

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from agents.chat import build_turn_context
from agents.tools import (
    call_tools,
    salesforce_query_records,
    salesforce_aggregate_query,
    salesforce_dml_records,
    salesforce_search_leads,
    salesforce_upsert_lead,
    salesforce_search_objects,
    salesforce_describe_object,
    salesforce_read_apex,
    salesforce_write_apex,
    salesforce_execute_anonymous,
)
from llm import get_llm
from observability import merge_node_config

try:
    from services.salesforce_client import is_salesforce_configured
except ImportError:
    def is_salesforce_configured() -> bool:
        return False

logger = logging.getLogger(__name__)

_CRM_TOOLS = [
    salesforce_query_records,
    salesforce_aggregate_query,
    salesforce_search_leads,
    salesforce_upsert_lead,
    salesforce_dml_records,
    salesforce_search_objects,
    salesforce_describe_object,
    salesforce_read_apex,
    salesforce_write_apex,
    salesforce_execute_anonymous,
]

# Read verbs that justify the deterministic "latest leads" table fast-path.
_READ_VERB_RE = re.compile(
    r"\b(fetch|get|show|list|latest|recent|pull|retrieve|display|view|see)\b",
    re.IGNORECASE,
)
# Write/advanced signals: if present, do NOT use the fast-path — run the tool loop instead.
_WRITE_OR_ADVANCED_RE = re.compile(
    r"\b(apex|soql|sosl|aggregate|count|sum|average|group by|describe|object|trigger|"
    r"contact|account|opportunity|case|update|delete|insert|create|upsert|save|add|log|new)\b",
    re.IGNORECASE,
)


def _is_simple_leads_fetch(text: str) -> bool:
    """Only a read-style 'list leads' request (no writes/advanced ops) uses the table fast-path."""
    if _WRITE_OR_ADVANCED_RE.search(text):
        return False
    if not re.search(r"\blead", text, re.IGNORECASE):
        return False
    return bool(_READ_VERB_RE.search(text))


def _fetch_latest_leads(limit: int = 10) -> str:
    return salesforce_query_records.invoke({
        "object_name": "Lead",
        "fields": ["Id", "Name", "Email", "Company", "Title", "Status", "CreatedDate"],
        "order_by": "CreatedDate DESC",
        "limit": limit,
    })


def _format_leads_markdown(raw: str) -> str:
    from services.salesforce_mcp import parse_query_records

    rows = parse_query_records(raw)
    if not rows:
        return "No leads found in Salesforce."

    lines = [
        "## Latest Salesforce Leads",
        "",
        "| Name | Email | Company | Title | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        def cell(key: str) -> str:
            val = row.get(key, "")
            return "" if val in (None, "null", "None") else str(val)

        lines.append(
            f"| {cell('Name')} | {cell('Email')} | {cell('Company')} | {cell('Title')} | {cell('Status')} |"
        )
    return "\n".join(lines)


def crm_research(state: AgentState, config: RunnableConfig | None = None) -> dict:
    turn = build_turn_context(state)
    question = (state.get("question") or "").strip()

    if not is_salesforce_configured():
        return {
            "context": "",
            "answer": "Salesforce is not configured. Add SALESFORCE_* variables to `.env` (see README).",
            "steps": ["CRM Research → blocked (not configured)"],
        }

    # Fast-path decision uses the CURRENT message only (history would falsely trigger it).
    if _is_simple_leads_fetch(question):
        try:
            raw = _fetch_latest_leads(limit=10)
            return {
                "context": raw,
                "answer": _format_leads_markdown(raw),
                "steps": ["CRM Research → fast-path leads fetch (salesforce_query_records)"],
            }
        except Exception as exc:
            return {
                "context": "",
                "answer": f"Could not fetch leads from Salesforce: {exc}",
                "steps": [f"CRM Research → fast-path failed ({exc})"],
            }

    ctx, log = call_tools(
        turn,
        tools=_CRM_TOOLS,
        config=config,
        system_prompt=(
            "You are a Salesforce CRM specialist. Use the Salesforce tools to fulfill the user's request.\n"
            "- Read/list records: salesforce_query_records (for latest leads use objectName Lead, orderBy CreatedDate DESC)\n"
            "- Counts/grouping: salesforce_aggregate_query\n"
            "- Create/update/delete records: salesforce_dml_records (or salesforce_upsert_lead for leads)\n"
            "- Inspect schema: salesforce_describe_object, salesforce_search_objects\n"
            "- Apex: salesforce_read_apex, salesforce_write_apex, salesforce_execute_anonymous\n"
            "Call the minimum tools needed, then stop. Do not invent record IDs."
        ),
    )
    return {"context": ctx, "steps": [f"CRM Research → {', '.join(log) or 'no tools called'}"]}


def crm_generate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    # Fast path already produced a final answer (e.g. the leads table).
    if state.get("answer"):
        return {"steps": ["CRM Generate → passthrough"]}

    llm = get_llm(temperature=0)
    turn = build_turn_context(state)
    ctx = state.get("context", "")
    resp = llm.invoke(
        "You are a Salesforce CRM specialist. Using ONLY the tool results below, answer the user's request.\n"
        "- Present records or query results as a clean Markdown table when appropriate.\n"
        "- For Apex or execution results, show the relevant output and a one-line summary.\n"
        "- If the tool results contain an error, explain it plainly and suggest the fix.\n\n"
        f"Tool results:\n{ctx}\n\n{turn}\nAnswer:",
        config=merge_node_config(
            config,
            metadata={"node": "crm_generate", "agent_type": "crm"},
            tags=["agent:crm", "phase:generate"],
        ) or None,
    )
    return {"answer": resp.content, "steps": [f"CRM Generate → {len(resp.content)} chars"]}
