from langchain_core.tools import tool


def _get_ddgs():
    """Return a DuckDuckGo client class from whichever package is installed."""
    try:
        from ddgs import DDGS  # current package
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # legacy package name
        return DDGS


@tool
def web_search(query: str) -> str:
    """Search the live web via DuckDuckGo."""
    try:
        ddgs_cls = _get_ddgs()
    except ImportError as exc:
        return f"WEB_SEARCH_UNAVAILABLE: {exc}"

    errors = []
    for backend in ("html", "lite"):
        try:
            with ddgs_cls() as ddgs:
                results = list(ddgs.text(query, max_results=3, backend=backend))
            if not results:
                continue
            return "\n\n".join(f"**{r['title']}**\n{r['body']}" for r in results)
        except Exception as e:
            errors.append(f"{backend}: {e}")
    if errors:
        return "WEB_SEARCH_UNAVAILABLE: " + " | ".join(errors)
    return "No web results found."
