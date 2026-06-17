from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    agent_type: Literal["gtm", "outreach", "crm"] = Field(
        description=(
            "gtm for product/pricing research; "
            "outreach for writing/sending emails, content, or finding NEW prospects; "
            "crm for Salesforce operations: fetch/list leads or contacts, SOQL/aggregate "
            "queries, record create/update/delete, describe objects, and Apex read/write/execute"
        )
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
        description="leads to find new prospects (Apollo); content to write marketing copy"
    )
