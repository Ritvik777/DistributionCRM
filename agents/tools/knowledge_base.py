from langchain_core.tools import tool

from vector_db import search_with_scores


@tool
def search_knowledge_base(query: str) -> str:
    """Search internal product docs stored in Qdrant."""
    results = search_with_scores(query, top_k=8)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(f"[{score:.3f}] {text}" for text, score in results)
