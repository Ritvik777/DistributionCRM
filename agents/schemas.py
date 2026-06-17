from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    agent_type: Literal["gtm", "outreach"] = Field(
        description="gtm for product/pricing research; outreach for emails, content, leads"
    )


class PricingGateDecision(BaseModel):
    is_pricing: bool = Field(
        description="True if the user wants pricing, cost, or plan information"
    )


class SendIntentDecision(BaseModel):
    intent: Literal["send", "review"] = Field(
        description="send only if user explicitly wants immediate delivery of an existing draft"
    )


class LeadsGateDecision(BaseModel):
    path: Literal["leads", "content"] = Field(
        description="leads to find prospects; content to write marketing copy"
    )
