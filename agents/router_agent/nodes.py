import logging

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from agents.chat import build_turn_context, is_crm_request, is_outreach_request
from agents.schemas import RouteDecision
from agents.structured import invoke_structured
from llm import get_llm
from observability import merge_node_config

logger = logging.getLogger(__name__)

_ROUTING_PROMPT = """You are the routing layer for a Product Marketing assistant with THREE specialist agents: GTM, OUTREACH, and CRM.

RULES — Route to GTM for:
- Product questions, features, pricing, cost, plans, tiers, subscriptions
- Stock/availability lookups when the user is asking YOU for info (e.g. 'do we have LED-RED-5MM?')
- Company news and announcements about our products
- Competitor, market, and industry research tied to positioning
- User wants product/pricing info formatted as copy for their team to send themselves

RULES — Route to OUTREACH for:
- User asks to EMAIL, SEND, MARKET, or WRITE outreach TO a recipient (person or email address)
- Product availability or specs should be communicated TO someone else via email/outreach
- Draft or send cold emails, LinkedIn posts, or marketing copy
- Finding NEW leads/prospects to contact (Apollo / net-new research)
- Follow-ups that refine a prior draft (e.g. 'tell them we have this in stock', 'include the specs')
- If conversation history shows outreach context, keep routing outreach on follow-ups

RULES — Route to CRM for (anything operating on Salesforce/CRM data or code):
- Fetch, list, show, or search existing Leads/Contacts/Accounts/Opportunities in Salesforce/CRM
- SOQL queries, aggregate/GROUP BY queries (count opportunities by stage, etc.)
- Create / update / delete CRM records; upsert leads; log CRM activity
- Describe objects/fields, search objects, inspect schema

IMPORTANT: 'email X about product Y' or 'market product Y to X@email.com' → outreach (not gtm).
IMPORTANT: 'fetch leads from Salesforce', 'show CRM contacts', 'count opportunities by stage' → crm.
IMPORTANT: 'find VP Sales leads in fintech to email' → outreach (net-new prospecting + send), NOT crm.

If unrelated to product marketing, route gtm.

Examples:
- 'Do you have led-red-5mm?' → gtm
- 'Can you email rgaur@company.com about availability of LED Red 5mm?' → outreach
- 'Find VP Marketing leads at Series B SaaS to reach out to' → outreach
- 'Fetch latest leads from Salesforce' → crm
- 'Show me contacts in CRM for Acme Corp' → crm
- 'Count opportunities by stage' → crm
- 'Update the status of lead 00Q... to Working' → crm
"""


def classify(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """LLM reads the message and picks: 'gtm', 'outreach', or 'crm'. Falls back to gtm on failure."""
    turn = build_turn_context(state)
    # Keyword routing must look at the CURRENT message only — using full history would
    # let a prior CRM turn (e.g. a leads table) hijack every later message.
    question = (state.get("question") or "").strip()
    if is_crm_request(question):
        return {
            "agent_type": "crm",
            "steps": ["Supervisor Routing Agent(keyword) → CRM"],
        }

    invoke_config = merge_node_config(
        config,
        metadata={"node": "classify", "question": state.get("question", "")},
        tags=["agent:supervisor_routing"],
    )

    llm = get_llm(temperature=0)
    decision = invoke_structured(
        RouteDecision,
        llm,
        f"{_ROUTING_PROMPT}\n\n{turn}",
        invoke_config,
    )
    if decision is None:
        agent = "gtm"
        source = "fallback"
    else:
        agent = decision.agent_type
        source = "structured"

    if agent != "crm" and is_crm_request(question):
        agent = "crm"
        source = "crm_override"
    # A send/email intent must go to Outreach — the CRM agent cannot send email.
    elif agent == "crm" and is_outreach_request(question):
        agent = "outreach"
        source = "outreach_override"

    return {"agent_type": agent, "steps": [f"Supervisor Routing Agent({source}) → {agent.upper()}"]}


def route(state: AgentState, config: RunnableConfig | None = None) -> str:
    return state["agent_type"]
