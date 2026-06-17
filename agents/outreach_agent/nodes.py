import logging
import re

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from agents.chat import build_turn_context
from agents.schemas import LeadsGateDecision, SendIntentDecision
from agents.structured import invoke_structured
from llm import get_llm
from agents.tools import search_knowledge_base, web_search, apollo_search, send_email, call_tools
from observability import merge_node_config

logger = logging.getLogger(__name__)

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"


def _leads_gate_decision(state: AgentState, config: RunnableConfig | None = None) -> tuple[str, str]:
    turn = build_turn_context(state)
    invoke_config = merge_node_config(
        config,
        metadata={"node": "leads_gate_decision", "agent_type": "outreach"},
        tags=["agent:outreach", "gate:leads"],
    )
    decision = invoke_structured(
        LeadsGateDecision,
        get_llm(temperature=0),
        (
            "You are a gate in a Product Marketing outreach assistant.\n"
            "Choose leads if the user wants to FIND prospects/people/companies to contact.\n"
            "Choose content if the user wants to write emails, posts, or marketing copy.\n"
            "If uncertain, choose content.\n\n"
            f"{turn}"
        ),
        invoke_config,
    )
    if decision is None:
        return "content", "fallback"
    return decision.path, "structured"


def _send_intent_decision(state: AgentState, config: RunnableConfig | None = None) -> tuple[str, str]:
    turn = build_turn_context(state)
    draft = state.get("answer", "")
    invoke_config = merge_node_config(
        config,
        metadata={"node": "send_gate_decision", "agent_type": "outreach"},
        tags=["agent:outreach", "gate:send"],
    )
    decision = invoke_structured(
        SendIntentDecision,
        get_llm(temperature=0),
        (
            "You are a gate in a Product Marketing outreach assistant.\n"
            "Choose send ONLY if the user explicitly wants immediate delivery of an existing draft "
            "(e.g. 'send it', 'send now', 'go ahead and send').\n"
            "Choose review for compose/draft/can-you-email requests or follow-up edits.\n"
            "If uncertain, choose review.\n\n"
            f"Draft preview:\n{draft[:400]}\n\n"
            f"{turn}"
        ),
        invoke_config,
    )
    if decision is None:
        return "review", "fallback"
    return decision.intent, "structured"


def _extract_emails(text: str) -> list[str]:
    return re.findall(EMAIL_PATTERN, text)


def _body_to_html(body: str) -> str:
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    body_html = "".join(
        f'<p style="margin: 0 0 12px 0;">{p.strip()}</p>'
        for p in body.split("\n\n")
        if p.strip()
    )
    if not body_html:
        body_html = body.replace("\n", "<br>")
    return body_html


def _parse_email_drafts(content: str) -> list[dict[str, str]]:
    """Parse one or more email drafts (split by ---) with per-recipient subject/body."""
    blocks = re.split(r"\n---+\n", content)
    drafts: list[dict[str, str]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        emails = _extract_emails(block)
        if not emails:
            continue
        subject_match = re.search(r"\*{0,2}Subject:?\*{0,2}\s*(.+)", block, re.IGNORECASE)
        subject = (
            subject_match.group(1).strip()
            if subject_match
            else "A personalized product introduction"
        )
        body = re.sub(r"^\*{0,2}To:?\*{0,2}.*\n?", "", block, flags=re.MULTILINE)
        body = re.sub(r"^\*{0,2}Subject:?\*{0,2}.*\n?", "", body, flags=re.MULTILINE)
        body = body.strip().strip("-").strip()
        if body:
            drafts.append({"to_email": emails[0], "subject": subject, "body": body})
    return drafts

def outreach_research(state: AgentState, config: RunnableConfig | None = None) -> dict:
    turn = build_turn_context(state)
    path, source = _leads_gate_decision(state, config)
    wants_leads = path == "leads"

    if wants_leads:
        ctx, log = call_tools(
            turn,
            tools=[apollo_search, search_knowledge_base],
            config=config,
            system_prompt=(
                "You are a product marketing research assistant. The user wants to find leads/prospects for outreach. You MUST:\n"
                "1. Call apollo_search with relevant job titles\n"
                "2. Call search_knowledge_base to get our product info for personalization\n\n"
                "ALWAYS call apollo_search. Do NOT skip it."
            ),
        )
        path_label = "leads (apollo)"
    else:
        ctx, log = call_tools(
            turn,
            tools=[search_knowledge_base, web_search],
            config=config,
            system_prompt=(
                "You are a product marketing research assistant preparing outreach content.\n"
                "Use search_knowledge_base for product specs, SKU/part numbers, stock, price, MOQ, and lead time. "
                "If conversation history or the current message mentions a product, search the KB for it.\n"
                "Use web_search only for target company/industry info to personalize."
            ),
        )
        path_label = "content"

    return {"context": ctx, "steps": [f"Outreach Research({source}) → {path_label}, {', '.join(log) or 'none'}"]}


def outreach_generate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    llm = get_llm(temperature=0.7)

    ctx = state.get("context", "")
    has_leads = "leads" in ctx.lower() and "Email:" in ctx

    turn = build_turn_context(state)

    if has_leads:
        prompt = (
            "You are a product marketing outreach specialist. "
            "You found REAL leads from Apollo with their emails. "
            "For EACH lead that has an email, write a personalized product outreach email.\n\n"
            "Rules:\n"
            "- Write ONLY emails, NOT LinkedIn posts\n"
            "- Use their actual name, title, company, and industry\n"
            "- Connect their likely pain points to our product benefits\n"
            "- Keep each email 2-3 short paragraphs\n"
            "- Sign off as 'The Product Marketing Team'\n"
            "- NO placeholder text like [Your Name] — use real data only\n\n"
            "Format EACH email as:\n"
            "**To: FirstName LastName** (their@email.com)\n"
            "**Subject:** <personalized subject>\n"
            "<email body>\n"
            "---\n\n"
            f"Leads + product info:\n{ctx}\n\n"
            f"{turn}\nContent:"
        )
    else:
        recipient_emails = _extract_emails(build_turn_context(state))
        recipient_hint = ""
        if recipient_emails:
            recipient_hint = (
                f"\nThe user specified recipient(s): {', '.join(recipient_emails)}. "
                "Address the email to them and include their email in the output "
                "using the format: **To:** name (email@example.com)\n"
            )

        # Galileo_FeedbackLoop_1: Defense in depth — never output full pricing in Outreach.
        # Pricing requests should route to GTM; if one slips through, refuse to reveal pricing.
        pricing_safety = (
            "\nNEVER reveal specific pricing tiers, dollar amounts, or plan names. "
            "If the user asks for pricing information, respond: "
            "'Pricing questions require email verification. Please ask for pricing directly and provide your work email.'"
        )
        prompt = (
            "You are a product marketing content specialist. Create EXACTLY the marketing content the user asks for.\n"
            "- Ground copy in our product context from the Context section and conversation history below\n"
            "- If emailing about product availability, include ALL specs from Context: part number, description, "
            "manufacturer, price, stock quantity, MOQ, lead time, package, voltage — never generic 'reply for details'\n"
            "- If they ask for a LinkedIn post: write ONLY a LinkedIn post\n"
            "- If they ask for an email: write ONLY an email with **To:** and **Subject:** lines\n"
            "- Do NOT create multiple content types\n"
            "- No placeholder text like [Your Name]\n"
            "- Sign off as 'The Product Marketing Team'\n"
            f"{pricing_safety}\n"
            f"{recipient_hint}\n"
            f"Context:\n{ctx}\n\n"
            f"{turn}\nContent:"
        )

    resp = llm.invoke(
        prompt,
        config=merge_node_config(
            config,
            metadata={
                "node": "outreach_generate",
                "agent_type": "outreach",
                "send_requested": bool(state.get("send_requested")),
            },
            tags=["agent:outreach", "phase:generate"],
        ) or None,
    )
    return {"answer": resp.content, "steps": [f"Outreach Generate → {len(resp.content)} chars"]}


def send_gate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    intent, source = _send_intent_decision(state, config)
    wants_send = intent == "send"
    label = "📤 send intent — confirm in UI" if wants_send else "👀 review only"
    return {
        "send_intent": wants_send,
        "send_requested": False,
        "steps": [f"Send Gate({source}) → {label}"],
    }


def route_send(state: AgentState, config: RunnableConfig | None = None) -> str:
    return "send" if state.get("send_confirmed") else "review"


def outreach_send(state: AgentState, config: RunnableConfig | None = None) -> dict:
    content = state["answer"]
    drafts = _parse_email_drafts(content)

    if not drafts:
        emails_found = _extract_emails(content) or _extract_emails(state["question"])
        if not emails_found:
            return {
                "answer": state["answer"] + "\n\n---\n⚠️ *No email addresses found in drafts to send.*",
                "steps": ["Outreach Send → ❌ no emails found in generated content"],
            }
        subject_match = re.search(r"\*{0,2}Subject:?\*{0,2}\s*(.+)", content, re.IGNORECASE)
        subject = (
            subject_match.group(1).strip()
            if subject_match
            else "A personalized product introduction"
        )
        body = re.sub(r"^\*{0,2}To:?\*{0,2}.*\n?", "", content, flags=re.MULTILINE)
        body = re.sub(r"^\*{0,2}Subject:?\*{0,2}.*\n?", "", body, flags=re.MULTILINE)
        body = body.strip().strip("-").strip()
        drafts = [{"to_email": email, "subject": subject, "body": body} for email in emails_found]

    sent = []
    failed = []

    invoke_config = merge_node_config(
        config,
        metadata={"node": "outreach_send"},
        tags=["agent:outreach", "tool:send_email"],
    )
    for draft in drafts:
        body_html = _body_to_html(draft["body"])
        html = f"""
        <div style="font-family: -apple-system, Arial, sans-serif; max-width: 600px; margin: 0 auto; color: #1a1a1a; line-height: 1.6;">
            {body_html}
            <hr style="border: none; border-top: 1px solid #e5e5e5; margin: 24px 0;">
            <p style="font-size: 12px; color: #999;">Sent via Product Marketing</p>
        </div>
        """

        result = send_email.invoke(
            {
                "to_email": draft["to_email"],
                "subject": draft["subject"],
                "html_body": html,
            },
            config=invoke_config,
        )
        if "SENT" in result:
            sent.append(draft["to_email"])
        else:
            failed.append(f"{draft['to_email']} ({result})")

    summary = ""
    if sent:
        summary += f"✅ **Sent to:** {', '.join(sent)}\n\n"
    if failed:
        summary += f"❌ **Failed:** {', '.join(failed)}\n\n"

    return {
        "answer": f"{summary}---\n\n{state['answer']}",
        "steps": [f"Outreach Send → ✅ {len(sent)} sent, ❌ {len(failed)} failed"],
    }



# User question
#      ↓
# [outreach_research]
#   - leads vs content gate (LLM)
#   - Leads → Apollo + KB
#   - Content → KB + web search
#      ↓
# [outreach_generate]
#   - Leads → one personalized email per lead
#   - Content → single email or LinkedIn post
#      ↓
# [send_gate]
#   - send vs review gate (LLM)
#      ↓
# route_send
#   - "review" → END (review only)
#   - "send" → outreach_send
#      ↓
# [outreach_send]
#   - Extract emails from draft/question
#   - Parse subject and body
#   - Send each via send_email (Brevo)
#   - Append summary to answer
#      ↓
# END