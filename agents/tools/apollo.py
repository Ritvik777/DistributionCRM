import os

import requests
from langchain_core.tools import tool


@tool
def apollo_search(job_titles: str, location: str = "", industry: str = "", limit: int = 5) -> str:
    """Search Apollo.io for leads by job title, location, and industry. Returns names, titles, companies, and verified emails for outreach."""
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
