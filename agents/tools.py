from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from vector_db import search_with_scores
from llm import get_llm
from observability import get_langchain_config, merge_node_config, log_span
import json


@tool
def search_knowledge_base(query: str) -> str:
    """Search internal product docs stored in Qdrant."""
    results = search_with_scores(query, top_k=8)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(f"[{score:.3f}] {text}" for text, score in results)


@tool
def web_search(query: str) -> str:
    """Search the live web via DuckDuckGo."""
    from duckduckgo_search import DDGS
    errors = []
    for backend in ("html", "lite"):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, backend=backend))
            if not results:
                continue
            return "\n\n".join(f"**{r['title']}**\n{r['body']}" for r in results)
        except Exception as e:
            errors.append(f"{backend}: {e}")
    if errors:
        return "WEB_SEARCH_UNAVAILABLE: " + " | ".join(errors)
    return "No web results found."


@tool
def apollo_search(job_titles: str, location: str = "", industry: str = "", limit: int = 5) -> str:
    """Search Apollo.io for leads by job title, location, and industry. Returns names, titles, companies, and verified emails for outreach."""
    import os
    import requests

    api_key = os.getenv("APOLLO_API_KEY")
    if not api_key or api_key == "your-apollo-api-key-here":
        return "ERROR: APOLLO_API_KEY not configured in .env"

    headers = {"Content-Type": "application/json", "Cache-Control": "no-cache", "X-Api-Key": api_key}
    titles = [t.strip() for t in job_titles.split(",")]

    # Step 1: Search Apollo for people matching criteria
    search_payload = {
        "person_titles": titles,
        "page": 1,
        "per_page": min(limit, 10),
    }
    if location:
        search_payload["person_locations"] = [location]
    if industry:
        search_payload["organization_industries"] = [industry]

    try:
        resp = requests.post(
            "https://api.apollo.io/api/v1/mixed_people/api_search",
            headers=headers, json=search_payload, timeout=15,
        )
        if resp.status_code != 200:
            return f"Apollo search error (status {resp.status_code}): {resp.text[:200]}"

        people = resp.json().get("people", [])
        if not people:
            return "No leads found matching your criteria."

        # Step 2: Enrich each person by ID to get email + full details
        results = []
        for p in people[:limit]:
            pid = p.get("id", "")
            if not pid:
                continue

            try:
                enrich = requests.post(
                    "https://api.apollo.io/api/v1/people/match",
                    headers=headers, timeout=10,
                    json={"id": pid, "reveal_personal_emails": True},
                )
                if enrich.status_code != 200:
                    continue
                ep = enrich.json().get("person", {})
            except Exception:
                continue

            name = ep.get("name") or p.get("first_name", "Unknown")
            email = ep.get("email", "")
            title = ep.get("title") or p.get("title", "N/A")
            org = ep.get("organization", {})
            company = org.get("name") or p.get("organization", {}).get("name", "N/A")
            industry_val = org.get("industry", "N/A")
            emp_count = org.get("estimated_num_employees", "N/A")
            city = ep.get("city", "")
            linkedin = ep.get("linkedin_url", "")

            lead = f"**{name}** — {title} at {company}"
            lead += f"\n  Industry: {industry_val} | Size: {emp_count} employees"
            if city:
                lead += f" | Location: {city}"
            lead += f"\n  Email: {email}" if email else "\n  Email: not found"
            if linkedin:
                lead += f"\n  LinkedIn: {linkedin}"
            results.append(lead)

        if not results:
            return "Found leads but could not enrich any with email data."

        enriched = sum(1 for r in results if "Email: not found" not in r)
        return f"Found {len(results)} leads ({enriched} with verified emails):\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Apollo search failed: {e}"


@tool
@log_span(span_type="tool", name="send_email")
def send_email(to_email: str, subject: str, html_body: str) -> str:
    """Send a personalized marketing email via Brevo. Provide recipient email, subject line, and HTML body."""
    import os

    import requests

    api_key = os.getenv("BREVO_API_KEY")
    if not api_key or api_key == "your-brevo-api-key-here":
        return "ERROR: BREVO_API_KEY not configured in .env"

    from_email = os.getenv("BREVO_FROM_EMAIL")
    if not from_email or from_email == "you@yourcompany.com":
        return "ERROR: BREVO_FROM_EMAIL not configured in .env"

    from_name = os.getenv("BREVO_FROM_NAME", "Product Marketing")
    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_body,
    }

    try:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": api_key, "Content-Type": "application/json", "accept": "application/json"},
            json=payload,
            timeout=15,
        )
        if resp.status_code in (200, 201, 202):
            return f"SENT to {to_email} (status {resp.status_code})"
        return f"FAILED (status {resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"


# GalileoCallback (via gtm_retrieve/outreach_research nodes) logs LLM + tool calls;
# no @log_span to avoid duplicate call_tools span (same work as gtm_retrieve).
def call_tools(question, tools, system_prompt, config=None):
    """LLM picks which tools to call, runs them, returns results.
    Pass config from the graph so LLM/tool spans nest under the parent node."""
    tool_map = {t.name: t for t in tools}
    try:
        llm = get_llm().bind_tools(tools)
    except Exception as exc:
        return f"LLM_UNAVAILABLE: {exc}", []
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    log = []
    seen_calls = set()
    invoke_config = merge_node_config(
        config,
        metadata={"component": "tools", "question": question},
        tags=["agent:shared", "phase:tool-routing"],
    )
    for _ in range(3):
        try:
            resp = llm.invoke(msgs, config=invoke_config or None)
        except Exception as exc:
            return f"LLM_ERROR: {exc}", log
        msgs.append(resp)
        if not resp.tool_calls:
            break
        for tc in resp.tool_calls:
            signature = f"{tc['name']}::{json.dumps(tc.get('args', {}), sort_keys=True, default=str)}"
            if signature in seen_calls:
                msgs.append(ToolMessage(content="Skipped duplicate tool call.", tool_call_id=tc["id"]))
                continue
            seen_calls.add(signature)
            # Per-tool config so Galileo shows which tool was used + args in span metadata
            args_str = json.dumps(tc.get("args", {}), default=str)[:200]
            tool_config = merge_node_config(
                invoke_config,
                metadata={
                    "tool_name": tc["name"],
                    "tool_args": args_str,
                },
                tags=["tool", f"tool:{tc['name']}"],
            )
            try:
                out = tool_map[tc["name"]].invoke(tc["args"], config=tool_config)
            except Exception as exc:
                out = f"TOOL_ERROR[{tc['name']}]: {exc}"
            log.append(tc["name"])
            msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

    context = "\n\n".join(m.content for m in msgs if isinstance(m, ToolMessage))
    return context or "No context found.", log
