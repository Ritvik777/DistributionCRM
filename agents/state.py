from typing import TypedDict, Annotated


def _merge(a: list, b: list) -> list:
    return a + b


class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    agent: str


class AgentState(TypedDict):
    question: str
    chat_history: list[dict]
    agent_type: str
    context: str
    answer: str
    is_pricing: bool
    user_email: str
    send_intent: bool
    send_requested: bool
    send_confirmed: bool
    steps: Annotated[list[str], _merge]
