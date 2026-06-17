import logging

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from agents.chat import build_turn_context
from agents.schemas import RouteDecision
from agents.structured import invoke_structured
from llm import get_llm
from observability import merge_node_config

logger = logging.getLogger(__name__)

_ROUTING_PROMPT = """You are the routing layer for a Product Marketing assistant (GTM product knowledge + outreach content).

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
- Finding leads/prospects to contact
- Follow-ups that refine a prior draft (e.g. 'tell them we have this in stock', 'include the specs')
- If conversation history shows outreach context, keep routing outreach on follow-ups

IMPORTANT: 'email X about product Y' or 'market product Y to X@email.com' → outreach (not gtm).

If unrelated to product marketing, route gtm.

Examples:
- 'Do you have led-red-5mm?' → gtm
- 'Can you email rgaur@company.com about availability of LED Red 5mm?' → outreach
- 'Market LED Red 5mm to buyer@example.com' → outreach
- 'Tell them we have 40k in stock' (after prior product discussion) → outreach
"""


def classify(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """LLM reads the message and picks: 'gtm' or 'outreach'. Falls back to gtm on failure."""
    turn = build_turn_context(state)
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

    return {"agent_type": agent, "steps": [f"Supervisor Routing Agent({source}) → {agent.upper()}"]}


def route(state: AgentState, config: RunnableConfig | None = None) -> str:
    return state["agent_type"]
