import logging
import re

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from agents.chat import build_turn_context
from agents.schemas import PricingGateDecision
from agents.structured import invoke_structured
from llm import get_llm
from agents.tools import search_knowledge_base, web_search, call_tools
from observability import merge_node_config

logger = logging.getLogger(__name__)
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"


def gtm_retrieve(state: AgentState, config: RunnableConfig | None = None) -> dict:
    turn = build_turn_context(state)
    ctx, log = call_tools(
        turn,
        tools=[search_knowledge_base, web_search],
        config=config,
        system_prompt=(
            "You are a product marketing specialist. Find product info and competitor data for the user's question from the knowledge base. "
            "Use conversation history to resolve references like 'this product'. "
            "If using search_knowledge_base, treat the data from it as ground truth. "
            "Use web_search for competitor/market data. "
            "Do not call the same tool with the same arguments more than once."
        ),
    )
    return {"context": ctx, "steps": [f"GTM Retrieve → {', '.join(log) or 'none'}"]}


def pricing_gate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    turn = build_turn_context(state)
    invoke_config = merge_node_config(
        config,
        metadata={"node": "pricing_gate", "agent_type": "gtm"},
        tags=["agent:gtm", "gate:pricing"],
    )
    decision = invoke_structured(
        PricingGateDecision,
        get_llm(temperature=0),
        (
            "You are a pricing gate for a Product Marketing assistant.\n"
            "Set is_pricing true if the user wants pricing, cost, or plan information about our product(s), "
            "regardless of output format (email template, table, etc.).\n\n"
            f"{turn}"
        ),
        invoke_config,
    )
    if decision is None:
        is_pricing = False
        source = "fallback"
    else:
        is_pricing = decision.is_pricing
        source = "structured"

    label = "🔒 pricing — email needed" if is_pricing else "✅ not pricing"
    return {"is_pricing": is_pricing, "steps": [f"Pricing Gate({source}) → {label}"]}


def route_pricing(state: AgentState, config: RunnableConfig | None = None) -> str:
    return "pricing" if state.get("is_pricing") else "not_pricing"


def collect_email(state: AgentState, config: RunnableConfig | None = None) -> dict:
    match = re.search(EMAIL_PATTERN, build_turn_context(state))
    if not match:
        return {
            "user_email": "",
            "answer": "💰 **Pricing requires a verified email.** Please reply with your work email.",
            "steps": ["Collect Email → ❌ no email found"],
        }
    return {"user_email": match.group(), "steps": [f"Collect Email → ✅ {match.group()}"]}


def route_email(state: AgentState, config: RunnableConfig | None = None) -> str:
    return "valid" if state.get("user_email") else "no_email"


def gtm_generate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    llm = get_llm()
    extra = ""
    if state.get("user_email"):
        extra = f"\nUser email verified ({state['user_email']}). Include full pricing details.\n"
    turn = build_turn_context(state)
    resp = llm.invoke(
        f"You are a product marketing specialist for our company. Answer using the context below.{extra}\n"
        f"Product, pricing, competitor, and market questions are all IN SCOPE for GTM.\n"
        f"If the user asks to email, send, or market a product TO someone, reply briefly that outreach can handle that.\n"
        f"If the context lacks the needed market/competitor data (e.g. live web search returned nothing), "
        f"say you couldn't retrieve current market data right now — do NOT claim the question is unrelated.\n"
        f"Only say a question is out of scope if it has nothing to do with products, pricing, market, or GTM.\n\n"
        f"Context:\n{state['context']}\n\n"
        f"{turn}\nAnswer:",
        config=merge_node_config(
            config,
            metadata={
                "node": "gtm_generate",
                "agent_type": "gtm",
                "has_user_email": bool(state.get("user_email")),
            },
            tags=["agent:gtm", "phase:generate"],
        ) or None,
    )
    return {"answer": resp.content, "steps": [f"GTM Generate → {len(resp.content)} chars"]}
