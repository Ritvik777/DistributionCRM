from typing import TypedDict, Annotated


def _merge(a: list, b: list) -> list:
    return a + b


class AgentState(TypedDict):
    question: str
    chat_history: list[dict]
    agent_type: str
    context: str
    answer: str
    kb_sources: list[dict]
    query_image_b64: str
    component_matches: list[dict]
    is_pricing: bool
    user_email: str
    send_intent: bool
    send_requested: bool
    send_confirmed: bool
    steps: Annotated[list[str], _merge]
