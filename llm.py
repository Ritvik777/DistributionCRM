"""
llm.py — Language Model Router
The "brain" that reads text and generates answers.
"""
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    VISION_CAPTION_MODEL,
    VISION_RERANK_MODEL,
    VISION_MODEL,
)


def _chat_anthropic(model: str, temperature: float):
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to your .env file and restart Streamlit."
        )
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "langchain-anthropic is not installed. Run: pip install langchain-anthropic"
        ) from exc
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )


def get_llm(temperature=0.7):
    return _chat_anthropic(ANTHROPIC_MODEL, temperature)


def get_vision_llm(temperature=0):
    """Default vision model (backward compatible)."""
    return _chat_anthropic(VISION_MODEL, temperature)


def get_vision_caption_llm(temperature=0):
    """Fast/cheap model for indexing captions (defaults to VISION_CAPTION_MODEL)."""
    return _chat_anthropic(VISION_CAPTION_MODEL, temperature)


def get_vision_rerank_llm(temperature=0):
    """Accurate model for visual re-ranking (defaults to VISION_RERANK_MODEL)."""
    return _chat_anthropic(VISION_RERANK_MODEL, temperature)
