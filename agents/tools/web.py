from langchain_core.tools import tool


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
