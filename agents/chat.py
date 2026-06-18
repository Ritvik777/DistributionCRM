"""Conversation helpers — re-exports from agents.intent for backward compatibility."""

from agents.constants import EMAIL_PATTERN, MAX_HISTORY_MESSAGES, SEND_CONFIRM_PHRASES
from agents.intent import (
    build_turn_context,
    extract_part_reference,
    format_chat_history,
    infer_leads_limit,
    is_crm_request,
    is_leads_by_part_enquiry,
    is_leads_time_window_fetch,
    is_outreach_request,
    is_recent_leads_list_fetch,
    is_simple_leads_fetch,
    is_singular_lead_fetch,
    parse_leads_time_window_minutes,
    trim_chat_history,
    wants_crm_list_fetch,
    wants_salesforce_leads_for_outreach,
)

__all__ = [
    "EMAIL_PATTERN",
    "MAX_HISTORY_MESSAGES",
    "SEND_CONFIRM_PHRASES",
    "build_turn_context",
    "extract_part_reference",
    "format_chat_history",
    "infer_leads_limit",
    "is_crm_request",
    "is_leads_by_part_enquiry",
    "is_leads_time_window_fetch",
    "is_outreach_request",
    "is_recent_leads_list_fetch",
    "is_simple_leads_fetch",
    "is_singular_lead_fetch",
    "parse_leads_time_window_minutes",
    "trim_chat_history",
    "wants_crm_list_fetch",
    "wants_salesforce_leads_for_outreach",
]
