import logging
import os
import re

from langchain_core.runnables import RunnableConfig

from agents.state import AgentState
from agents.chat import build_turn_context
from agents.schemas import LeadsGateDecision, SendIntentDecision
from agents.structured import invoke_structured
from llm import get_llm
from agents.tools import (
    search_knowledge_base,
    web_search,
    apollo_search,
    call_tools,
    salesforce_search_leads,
    salesforce_query_records,
)
from agents.tools.knowledge_base import begin_kb_collection, consume_kb_sources
from observability import merge_node_config

try:
    from services.salesforce_client import is_salesforce_configured, log_email_activity
except ImportError:
    def is_salesforce_configured() -> bool:
        return False

    def log_email_activity(*args, **kwargs):
        raise RuntimeError("Salesforce client unavailable")

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
            "Choose crm if the user wants to fetch, list, or search existing leads/contacts in Salesforce/CRM.\n"
            "Choose leads if the user wants to FIND new prospects/people/companies to contact (Apollo/net-new).\n"
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


from agents.outreach_agent.email_html import build_formal_email_html
from agents.tools.email import deliver_brevo_email
from services.catalog_image_host import (
    catalog_image_attachment_path,
    resolve_hosted_catalog_image_url,
)
from vector_db.component_store import find_catalog_match_by_sku

_FORMAL_EMAIL_RULES = (
    "Write a formal, professional B2B email suitable for an electronic components distributor.\n"
    "- Open with 'Dear [First Name],' — use the recipient's name if known, otherwise 'Dear Customer,'\n"
    "- Use complete sentences and a courteous, business-appropriate tone (no emoji, slang, or casual phrasing)\n"
    "- Structure: brief introduction → product/availability details → clear next step → formal closing\n"
    "- Include a 'Product Details' bullet list with SKU, description, package, and stock/availability status\n"
    "- Keep the body to 2–3 short paragraphs plus the bullet list\n"
    "- Do NOT repeat the signature block — the HTML template adds 'Best regards' automatically\n"
    "- End the body with a single closing line such as 'Please let us know if you would like a formal quote.'\n"
)

_LEAD_EMAIL_FORMAT = (
    "Format EACH email as:\n"
    "**To: FirstName LastName** (their@email.com)\n"
    "**Subject:** <formal, specific subject line mentioning the product or topic>\n"
    "<formal email body — paragraphs and bullet lists only, no Subject/To lines in body>\n"
    "---\n"
)


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
            drafts.append({
                "to_email": emails[0],
                "subject": subject,
                "body": body,
                "recipient_name": _parse_recipient_name(block),
            })
    return drafts


def _parse_recipient_name(block: str) -> str:
    match = re.search(r"\*{0,2}To:?\*{0,2}\s*(.+?)\s*\(", block, re.IGNORECASE)
    return match.group(1).strip() if match else ""


# Part/SKU tokens like LED-RED-5MM, ABC-123, MPN1234-A
_PART_TOKEN_RE = re.compile(r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]+){1,}\b")
# Labeled fields in KB/quotation context: "Part Number: X", "SKU: Y", "MPN # Z"
_PART_LABEL_RE = re.compile(
    r"(?:part\s*(?:no|number|#)?|sku|mpn|model)\s*[:#]?\s*([A-Za-z0-9][\w\-./]{2,})",
    re.IGNORECASE,
)


def _extract_products(*texts: str) -> str:
    """Collect part numbers / SKUs mentioned in the email + retrieved context for the CRM record."""
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _PART_LABEL_RE.findall(text):
            token = match.strip().strip(".,;")
            if token and token.lower() not in seen:
                seen.add(token.lower())
                found.append(token)
        for token in _PART_TOKEN_RE.findall(text):
            if token.lower() not in seen:
                seen.add(token.lower())
                found.append(token)
    return ", ".join(found[:15])


def _sender_label() -> str:
    name = os.getenv("BREVO_FROM_NAME", "Product Marketing")
    email = os.getenv("BREVO_FROM_EMAIL", "")
    return f"{name} <{email}>" if email else name


def _salesforce_tools():
    """CRM tools available to outreach for lead de-duplication during research."""
    if not is_salesforce_configured():
        return []
    return [salesforce_search_leads, salesforce_query_records]


_SF_PLATFORM_RE = re.compile(r"\b(salesforce|sfdc|crm)\b", re.IGNORECASE)


def _wants_salesforce_leads(text: str) -> bool:
    """User wants to email recipients sourced from Salesforce (e.g. 'email the leads from salesforce')."""
    return bool(_SF_PLATFORM_RE.search(text) and re.search(r"\blead", text, re.IGNORECASE))


def _salesforce_leads_context(turn: str, limit: int = 10) -> tuple[str, int, list[dict]]:
    """Pull leads (with emails) from Salesforce + product info, formatted for per-lead email drafting."""
    from services.salesforce_mcp import parse_query_records

    raw = salesforce_query_records.invoke({
        "object_name": "Lead",
        "fields": ["Id", "Name", "Email", "Company", "Title", "Status"],
        "order_by": "CreatedDate DESC",
        "limit": limit,
    })
    leads = []
    for row in parse_query_records(raw):
        email = (row.get("Email") or "").strip()
        if not email or email.lower() in ("null", "none"):
            continue
        name = row.get("Name") or "there"
        title = row.get("Title") if row.get("Title") not in (None, "null", "None") else "N/A"
        company = row.get("Company") if row.get("Company") not in (None, "null", "None") else "N/A"
        leads.append(f"**{name}** — {title} at {company}\n  Email: {email}")

    if not leads:
        return "", 0

    begin_kb_collection()
    product_info = search_knowledge_base.invoke({"query": turn[:500]})
    kb_sources = consume_kb_sources()
    block = (
        f"Found {len(leads)} leads from Salesforce CRM:\n\n"
        + "\n\n".join(leads)
        + f"\n\nProduct info:\n{product_info}"
    )
    return block, len(leads), kb_sources


def _component_match_context(state: AgentState) -> tuple[str, list[dict], list[dict]]:
    """Reuse precomputed matches or run hybrid match for outreach email drafts."""
    matches = state.get("component_matches") or []
    image_b64 = (state.get("query_image_b64") or "").strip()
    if matches:
        from agents.gtm_agent.nodes import _format_component_match_context

        return _format_component_match_context(matches), matches, []
    if image_b64:
        from agents.gtm_agent.nodes import _run_component_image_match

        ctx, matches, kb_sources = _run_component_image_match(image_b64)
        return ctx, matches, kb_sources
    return "", [], []


def outreach_research(state: AgentState, config: RunnableConfig | None = None) -> dict:
    turn = build_turn_context(state)
    question = (state.get("question") or "").strip()
    component_ctx, component_matches, match_kb_sources = _component_match_context(state)

    # Cross-agent flow: "email the leads from Salesforce" → source recipients from CRM, then draft per lead.
    if is_salesforce_configured() and _wants_salesforce_leads(question):
        block, count, kb_sources = _salesforce_leads_context(turn)
        if count:
            return {
                "context": block,
                "kb_sources": kb_sources,
                "steps": [f"Outreach Research → {count} leads from Salesforce + KB"],
            }

    path, source = _leads_gate_decision(state, config)
    wants_leads = path == "leads"

    if wants_leads:
        tools = [apollo_search, search_knowledge_base, *_salesforce_tools()]
        sf_hint = ""
        if is_salesforce_configured():
            sf_hint = (
                "3. Call salesforce_search_leads to check CRM and avoid duplicating existing contacts\n\n"
            )
        ctx, log, kb_sources = call_tools(
            turn,
            tools=tools,
            config=config,
            system_prompt=(
                "You are a product marketing research assistant. The user wants to find leads/prospects for outreach. You MUST:\n"
                "1. Call apollo_search with relevant job titles\n"
                "2. Call search_knowledge_base to get our product info for personalization\n"
                f"{sf_hint}"
                "ALWAYS call apollo_search when finding new prospects. "
                "Use salesforce_search_leads to avoid duplicating contacts already in CRM."
            ),
        )
        path_label = "leads (apollo+crm)" if is_salesforce_configured() else "leads (apollo)"
    else:
        content_tools = [search_knowledge_base, web_search, *_salesforce_tools()]
        sf_content = ""
        if is_salesforce_configured():
            sf_content = (
                "If the user mentions a recipient email or company, call salesforce_query_records or "
                "salesforce_search_leads first to pull existing Lead/Contact context from CRM.\n"
            )
        ctx, log, kb_sources = call_tools(
            turn,
            tools=content_tools,
            config=config,
            system_prompt=(
                "You are a product marketing research assistant preparing outreach content.\n"
                "Use search_knowledge_base for product specs, SKU/part numbers, stock, price, MOQ, and lead time. "
                "If conversation history or the current message mentions a product, search the KB for it.\n"
                "Use web_search only for target company/industry info to personalize.\n"
                f"{sf_content}"
            ),
        )
        path_label = "content"

    if component_ctx:
        ctx = f"{component_ctx}\n\n{ctx}"
        if not log:
            log = ["component_image_match"]

    result: dict = {
        "context": ctx,
        "kb_sources": kb_sources,
        "steps": [f"Outreach Research({source}) → {path_label}, {', '.join(log) or 'none'}"],
    }
    if component_matches:
        result["component_matches"] = component_matches
    if match_kb_sources:
        result["kb_sources"] = match_kb_sources + (kb_sources or [])
    return result


def outreach_generate(state: AgentState, config: RunnableConfig | None = None) -> dict:
    llm = get_llm(temperature=0.7)

    ctx = state.get("context", "")
    has_leads = "leads" in ctx.lower() and "Email:" in ctx

    turn = build_turn_context(state)

    if has_leads:
        prompt = (
            "You are a product marketing outreach specialist. "
            "You have REAL leads (from Apollo or your Salesforce CRM) with their emails. "
            "For EACH lead that has an email, write a personalized product outreach email.\n\n"
            "Rules:\n"
            "- Write ONLY emails, NOT LinkedIn posts\n"
            "- Use their actual name, title, company, and industry\n"
            "- Connect their likely pain points to our product benefits\n"
            f"- {_FORMAL_EMAIL_RULES}\n"
            "- NO placeholder text like [Your Name] — use real data only\n"
            "- Do NOT refuse and do NOT add commentary about whether a lead fits the product. "
            "Always write one email per lead and output ONLY the emails.\n\n"
            f"{_LEAD_EMAIL_FORMAT}\n\n"
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
        component_hint = ""
        if state.get("component_matches"):
            best = state["component_matches"][0]
            sku = best.get("sku") or "see Context"
            component_hint = (
                "\nThe user attached a component photo with catalog match results in Context. "
                f"Mention SKU {sku}, stock availability, and relevant specs in a formal tone. "
                "The sent HTML email will include the catalog reference photo automatically — "
                "you do not need to describe the image itself.\n"
            )

        pricing_safety = (
            "\nNEVER reveal specific pricing tiers, dollar amounts, or plan names. "
            "If the user asks for pricing information, respond: "
            "'Pricing questions require email verification. Please ask for pricing directly and provide your work email.'"
        )
        prompt = (
            "You are a product marketing content specialist. Create EXACTLY the marketing content the user asks for.\n"
            f"- {_FORMAL_EMAIL_RULES}\n"
            "- Ground copy in our product context from the Context section and conversation history below\n"
            "- If emailing about product availability, include ALL specs from Context: part number, description, "
            "manufacturer, stock quantity, MOQ, lead time, package, voltage — never generic 'reply for details'\n"
            "- If they ask for a LinkedIn post: write ONLY a LinkedIn post\n"
            "- If they ask for an email: write ONLY an email with **To:** and **Subject:** lines\n"
            "- Do NOT create multiple content types\n"
            "- No placeholder text like [Your Name]\n"
            f"{pricing_safety}\n"
            f"{recipient_hint}\n"
            f"{component_hint}\n"
            f"Context:\n{ctx}\n\n"
            f"{turn}\n"
            f"{_LEAD_EMAIL_FORMAT}\n"
            "Content:"
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
    return {"answer": resp.content, "steps": [f"Outreach Generate → {len(resp.content)} chars"], "component_matches": state.get("component_matches", [])}


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


def _resolve_component_matches_for_send(state: AgentState, draft_text: str) -> list[dict]:
    """Best-effort catalog match for email image — state, then SKU in draft/context."""
    matches = state.get("component_matches") or []
    if matches and catalog_image_attachment_path(matches[0]):
        return matches

    haystack = " ".join(
        filter(None, [draft_text, state.get("context", ""), state.get("question", "")])
    )
    for pattern in (
        r"\bSKU[:\s]+([A-Za-z0-9][\w \-./]{2,})",
        r"\b(?:part|item)\s*(?:#|number|no\.?)?[:\s]+([A-Za-z0-9][\w \-./]{2,})",
    ):
        for hit in re.findall(pattern, haystack, re.IGNORECASE):
            sku = hit.strip().strip(".,;")
            found = find_catalog_match_by_sku(sku)
            if found:
                return [found]

    for token in _PART_TOKEN_RE.findall(haystack):
        found = find_catalog_match_by_sku(token)
        if found:
            return [found]

    if matches:
        return matches
    return []


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
        drafts = [{"to_email": email, "subject": subject, "body": body, "recipient_name": ""} for email in emails_found]

    sent = []
    failed = []
    crm_logged = []
    crm_failed = []

    invoke_config = merge_node_config(
        config,
        metadata={"node": "outreach_send"},
        tags=["agent:outreach", "tool:send_email"],
    )
    image_notes: list[str] = []

    for draft in drafts:
        draft_blob = " ".join([draft.get("body", ""), draft.get("subject", "")])
        component_matches = _resolve_component_matches_for_send(state, draft_blob)

        catalog_image_url: str | None = None
        attachment_paths: list[str] = []

        if component_matches:
            best = component_matches[0]
            local_path = catalog_image_attachment_path(best)
            catalog_image_url = resolve_hosted_catalog_image_url(best)
            if catalog_image_url:
                # Inline <img> only — do not also attach (was showing 2 copies in inbox).
                image_notes.append(f"📷 {best.get('sku') or 'catalog'} → inline photo")
            elif local_path:
                attachment_paths.append(local_path)
                image_notes.append(f"📎 {best.get('sku') or 'catalog'} → attachment")

        html = build_formal_email_html(
            draft["body"],
            component_matches=component_matches,
            catalog_image_url=catalog_image_url,
            recipient_name=draft.get("recipient_name", ""),
        )

        result = deliver_brevo_email(
            draft["to_email"],
            draft["subject"],
            html,
            attachment_paths=attachment_paths or None,
        )
        if "SENT" in result:
            sent.append(draft["to_email"])
            if is_salesforce_configured():
                try:
                    products = _extract_products(draft["body"], draft["subject"], state.get("context", ""))
                    crm = log_email_activity(
                        email=draft["to_email"],
                        subject=draft["subject"],
                        body=draft["body"],
                        recipient_name=draft.get("recipient_name", ""),
                        products=products,
                        reason=draft["subject"],
                        sent_by=_sender_label(),
                    )
                    crm_logged.append(f"{draft['to_email']} (Task {crm['task_id']})")
                except Exception as exc:
                    logger.warning("Salesforce CRM update failed for %s: %s", draft["to_email"], exc)
                    crm_failed.append(f"{draft['to_email']} ({exc})")
        else:
            failed.append(f"{draft['to_email']} ({result})")

    summary = ""
    if image_notes:
        summary += " ".join(dict.fromkeys(image_notes)) + "\n\n"
    if sent:
        summary += f"✅ **Sent to:** {', '.join(sent)}\n\n"
    if crm_logged:
        summary += f"📋 **CRM updated:** {', '.join(crm_logged)}\n\n"
    if failed:
        summary += f"❌ **Failed:** {', '.join(failed)}\n\n"
    if crm_failed:
        summary += f"⚠️ **CRM update failed:** {', '.join(crm_failed)}\n\n"

    step_parts = [f"✅ {len(sent)} sent", f"❌ {len(failed)} failed"]
    if is_salesforce_configured():
        step_parts.append(f"📋 {len(crm_logged)} CRM logged")
    return {
        "answer": f"{summary}---\n\n{state['answer']}",
        "steps": [f"Outreach Send → {', '.join(step_parts)}"],
    }