import logging
from typing import TypeVar

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def invoke_structured(
    model: type[T],
    llm,
    prompt: str,
    config: RunnableConfig | None = None,
) -> T | None:
    try:
        structured_llm = llm.with_structured_output(model)
        return structured_llm.invoke(prompt, config=config or None)
    except Exception as exc:
        logger.exception("Structured output failed for %s: %s", model.__name__, exc)
        return None
