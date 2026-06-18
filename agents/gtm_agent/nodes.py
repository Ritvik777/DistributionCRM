import base64
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
from vector_db import match_component_image, search_kb_hits

logger = logging.getLogger(__name__)
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"


def _format_component_match_context(matches: list[dict]) -> str:
    if not matches:
        return (
            "Component image matching found no catalog matches. "
            "The component image catalog may be empty, or this part is not indexed yet."
        )
    query_summary = matches[0].get("query_summary") or ""
    lines = [
        "Component image hybrid match results (CLIP visual search + Claude vision re-rank + text KB):",
    ]
    if query_summary:
        lines.append(f"Query image understood as: {query_summary}")
    for index, match in enumerate(matches, start=1):
        label = match.get("name") or match.get("sku") or match.get("source") or f"Candidate {index}"
        lines.append(
            f"{index}. {label} — {match.get('match_percent', 0)}% combined confidence\n"
            f"   SKU: {match.get('sku') or 'n/a'} | Category: {match.get('category') or 'n/a'} | "
            f"Package: {match.get('package') or 'n/a'}\n"
            f"   CLIP: {match.get('clip_score')} | Vision: {match.get('vision_score')}/100 | "
            f"Text KB: {match.get('text_score')}\n"
            f"   Reason: {match.get('reasoning')}\n"
            f"   Catalog caption: {match.get('caption') or 'n/a'}"
        )
    return "\n\n".join(lines)


def _kb_sources_from_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "source": hit.get("source") or "(unknown)",
            "type": hit.get("type") or "text",
            "score": hit.get("score", 0),
            "excerpt": hit.get("excerpt") or "",
        }
        for hit in hits
    ]


def _run_component_image_match(image_b64: str) -> tuple[str, list[dict], list[dict]]:
    raw = base64.standard_b64decode(image_b64)
    matches = match_component_image(raw)
    context = _format_component_match_context(matches)
    kb_sources: list[dict] = []
    query_summary = matches[0].get("query_summary") if matches else ""
    if query_summary:
        kb_sources = _kb_sources_from_hits(search_kb_hits(query_summary, top_k=6))
    return context, matches, kb_sources


def gtm_retrieve(state: AgentState, config: RunnableConfig | None = None) -> dict:
    image_b64 = (state.get("query_image_b64") or "").strip()
    if image_b64:
        existing = state.get("component_matches") or []
        if existing:
            ctx = _format_component_match_context(existing)
            kb_sources = state.get("kb_sources") or []
            if not kb_sources:
                query_summary = existing[0].get("query_summary") if existing else ""
                if query_summary:
                    kb_sources = _kb_sources_from_hits(search_kb_hits(query_summary, top_k=6))
            best = existing[0].get("match_percent", 0) if existing else 0
            return {
                "context": ctx,
                "component_matches": existing,
                "kb_sources": kb_sources,
                "steps": [f"GTM Retrieve → component image match ({len(existing)} hits, best {best}%)"],
            }
        ctx, matches, kb_sources = _run_component_image_match(image_b64)
        best = matches[0]["match_percent"] if matches else 0
        return {
            "context": ctx,
            "component_matches": matches,
            "kb_sources": kb_sources,
            "steps": [f"GTM Retrieve → component image match ({len(matches)} hits, best {best}%)"],
        }

    turn = build_turn_context(state)
    ctx, log, kb_sources = call_tools(
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
    return {
        "context": ctx,
        "kb_sources": kb_sources,
        "steps": [f"GTM Retrieve → {', '.join(log) or 'none'}"],
    }


def pricing_gate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    if (state.get("query_image_b64") or "").strip() or state.get("component_matches"):
        return {"is_pricing": False, "steps": ["Pricing Gate(image) → skipped for component ID"]}
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
    if state.get("query_image_b64") or state.get("component_matches"):
        extra += (
            "\nThe user uploaded a component photo for catalog verification.\n"
            "Use the hybrid match results in Context. Lead with whether we stock this part, "
            "the best match confidence %, and SKU/name if available.\n"
            "≥80%: confident match. 60–79%: likely match, note uncertainty. <60%: weak/no match — "
            "say it may not be in the catalog and suggest adding it.\n"
            "Do not invent specs not supported by the match data.\n"
        )
    turn = build_turn_context(state)
    resp = llm.invoke(
        f"You are a product marketing specialist for our company. Answer using the context below.{extra}\n"
        f"Product, pricing, competitor, and market questions are all IN SCOPE for GTM.\n"
        f"If the user asks to email, send, or market a product TO someone, reply briefly that outreach can handle that.\n"
        f"If the context lacks the needed market/competitor data (e.g. live web search returned nothing), "
        f"say you couldn't retrieve current market data right now — do NOT claim the question is unrelated.\n"
        f"Only say a question is out of scope if it has nothing to do with products, pricing, market, or GTM.\n"
        f"Ground every product fact in the Context below. If the Context does not contain the answer, "
        f"say the knowledge base does not have that information — do not invent specs, prices, or stock levels.\n\n"
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
    return {"answer": resp.content, "steps": [f"GTM Generate → {len(resp.content)} chars"], "component_matches": state.get("component_matches", [])}
