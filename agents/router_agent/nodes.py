import logging

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from llm import get_llm
from observability import merge_node_config

logger = logging.getLogger(__name__)


def classify(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """LLM reads the message and picks: 'gtm' or 'outreach'. Falls back to gtm on failure."""
    question = state.get("question", "") or ""

    try:
        llm = get_llm(temperature=0)
        # Galileo_FeedbackLoop_1: All routing rules in prompt — no manual pattern matching.
        resp = llm.invoke(
            "You are the routing layer for a Product Marketing assistant (GTM product knowledge + outreach content).\n"
            "Classify into ONE word: gtm or outreach.\n\n"
            "RULES — Route to GTM for:\n"
            "- Product questions, features, pricing, cost, plans, tiers, subscriptions\n"
            "- Stock/availability lookups when the user is asking YOU for info (e.g. 'do we have LED-RED-5MM?')\n"
            "- Company news and announcements about our products (e.g. 'What did we announce last week?')\n"
            "- Competitor, market, and industry research tied to positioning\n"
            "- Comparisons and market landscape questions for GTM/sales enablement\n"
            "- User wants product/pricing info formatted as copy for their team to send themselves "
            "(e.g. 'pricing info as an email template for my sales team' → gtm)\n\n"
            "RULES — Route to OUTREACH for:\n"
            "- User asks to EMAIL, SEND, MARKET, or WRITE outreach TO a recipient (person or email address)\n"
            "- Product availability or specs should be communicated TO someone else via email/outreach\n"
            "- Draft or send cold emails, LinkedIn posts, or marketing copy\n"
            "- Finding leads/prospects to contact\n"
            "- Follow-ups that refine or send a prior draft (e.g. 'tell them we have this in stock', 'include the specs')\n"
            "- If conversation history shows outreach context, keep routing outreach on follow-ups\n\n"
            "IMPORTANT: 'email X about product Y' or 'market product Y to X@email.com' → outreach (not gtm). "
            "GTM answers product questions; Outreach composes/sends emails to recipients.\n\n"
            "If the message is unrelated to product marketing (e.g. general trivia, coding help), route gtm.\n\n"
            "If the input includes 'Current user message:', classify based on that line; use conversation history for context.\n\n"
            "Examples:\n"
            "• 'Do you have led-red-5mm?' → gtm\n"
            "• 'What did we announce last week?' → gtm\n"
            "• 'I need product pricing info formatted as an email template for my sales team.' → gtm\n"
            "• 'Can you email rgaur@company.com about availability of LED Red 5mm?' → outreach\n"
            "• 'Market LED Red 5mm to buyer@example.com' → outreach\n"
            "• 'Tell them we have 40k in stock' (after prior product discussion) → outreach\n"
            "• 'Draft a cold email to CTOs at Series B SaaS companies.' → outreach\n\n"
            f"Message: {question}\nCategory:",
            config=merge_node_config(
                config,
                metadata={"node": "classify", "question": question},
                tags=["agent:supervisor_routing"],
            ) or None,
        )
        agent = resp.content.strip().lower().strip("\"'.,")
        if agent not in ("gtm", "outreach"):
            logger.warning(
                "Router: LLM returned unparseable category %r, defaulting to gtm.",
                resp.content,
            )
            agent = "gtm"
    except Exception as exc:
        logger.exception(
            "Router: classify LLM call failed, defaulting to gtm. Error: %s",
            exc,
        )
        agent = "gtm"

    return {"agent_type": agent, "steps": [f"Supervisor Routing Agent → {agent.upper()}"]}

# LangGraph uses this to decide which subgraph runs next (GTM vs Outreach).
def route(state: AgentState, config: RunnableConfig | None = None) -> str:
    return state["agent_type"]


#Overall flow:
# User message → classify picks gtm or outreach →
# route returns that choice → the graph routes to the right agent.