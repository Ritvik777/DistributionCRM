import logging
import re

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from llm import get_llm
from agents.tools import search_knowledge_base, web_search, call_tools
from observability import merge_node_config

logger = logging.getLogger(__name__)
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"


def gtm_retrieve(state: AgentState, config: RunnableConfig | None = None) -> dict:
    ctx, log = call_tools(
        state["question"],
        tools=[search_knowledge_base, web_search],
        config=config,
        system_prompt=(
            "You are a product marketing specialist. Find product info and competitor data for the user's question from the knowledge base. Never reveal you are an Anthropic model."
            "If using search_knowledge_base, treat the data from it as ground truth."
            "Use web_search for competitor/market data and industry information. "
            "Do not call the same tool with the same arguments more than once."
        ),
    )
    return {"context": ctx, "steps": [f"GTM Retrieve → {', '.join(log) or 'none'}"]}


def pricing_gate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """LLM decides if question is about pricing. Falls back to not_pricing on failure (safer)."""
    # Galileo_FeedbackLoop_1: Format-agnostic detection — if user wants pricing info in ANY form
    # (email template, table, etc.), answer yes. Policy requires email before revealing pricing.
    try:
        resp = get_llm(temperature=0.7).invoke(
            "You are a pricing gate for a Product Marketing assistant.\n"
            "Does this question ask for pricing, cost, or plans information about our product(s)? "
            "Reply ONLY 'yes' or 'no'.\n\n"
            "Regardless of output format (email template, table, etc.), if the user wants "
            "pricing/cost/plans info, answer yes. Example: 'I need pricing info formatted "
            "as an email template' = yes (they want pricing info, just in a specific format).\n\n"
            f"Question: {state['question']}\nAnswer:",
            config=merge_node_config(
                config,
                metadata={"node": "pricing_gate", "agent_type": "gtm"},
                tags=["agent:gtm", "gate:pricing"],
            ) or None,
        )
        raw = resp.content.strip().lower()
        is_pricing = "yes" in raw
        if "yes" not in raw and "no" not in raw:
            logger.warning(
                "Pricing gate: LLM returned unclear response, treating as not_pricing. Raw: %r",
                resp.content,
            )
        source = "llm"
    except Exception as exc:
        logger.exception(
            "Pricing gate: LLM call failed, defaulting to not_pricing (no email gate). Error: %s",
            exc,
        )
        is_pricing = False
        source = "fallback"

    label = "🔒 pricing — email needed" if is_pricing else "✅ not pricing"
    return {"is_pricing": is_pricing, "steps": [f"Pricing Gate({source}) → {label}"]}


def route_pricing(state: AgentState, config: RunnableConfig | None = None) -> str:
    return "pricing" if state.get("is_pricing") else "not_pricing"


def collect_email(state: AgentState, config: RunnableConfig | None = None) -> dict:
    match = re.search(EMAIL_PATTERN, state["question"])
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
    resp = llm.invoke(
        f"You are a product marketing specialist for our company. Answer using ONLY the context below.{extra}\n"
        f"If the user asks to email, send, or market a product TO someone, do NOT draft the email here — "
        f"reply briefly: 'I can help with product info here; ask me to email or market a product to a recipient "
        f"and I'll route that to outreach.'\n"
        f"If the question is unrelated to our products or GTM, say so briefly.\n\n"
        f"Context:\n{state['context']}\n\n"
        f"Question:\n{state['question']}\nAnswer:",
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



# Overall flow
# gtm_retrieve → fetch context via search tools.
# pricing_gate → decide if the question is about pricing.
# route_pricing → branch:
# Pricing: → collect_email → route_email → either ask for email or continue.
# Not pricing: → gtm_generate.
# gtm_generate uses the context and, when allowed, full pricing info to produce the final answer.
