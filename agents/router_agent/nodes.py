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
            "Classify into ONE word: gtm or outreach.\n\n"
            "RULES — Route to GTM for:\n"
            "- Product questions, features, pricing, cost, plans, tiers, subscriptions\n"
            "- Company news and announcements (e.g. 'What did X announce?', 'What did we announce last week?')\n"
            "- Market and competitor lists (e.g. 'List the top 10 AI observability startups', 'top competitors')\n"
            "- Comparisons, industry/startup lists\n"
            "- ANY request for pricing info, even if formatted as email (e.g. 'I need product pricing info formatted "
            "as an email template for my sales team' → gtm — they want pricing data, just in email form)\n\n"
            "RULES — Route to OUTREACH only for:\n"
            "- PRIMARY intent is creating content from scratch: draft an email, write a LinkedIn post, create marketing copy\n"
            "- User wants you to WRITE outreach content, NOT retrieve and format existing product/pricing info\n"
            "- If they want info (pricing, news, comparisons, lists) even 'formatted as email' → gtm, not outreach\n\n"
            "Examples:\n"
            "• 'What did we announce last week?' → gtm\n"
            "• 'List the top 10 AI observability startups.' → gtm\n"
            "• 'I need product pricing info formatted as an email template for my sales team.' → gtm\n"
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