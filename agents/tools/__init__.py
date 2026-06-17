"""
agents/tools/ — Shared agent tools, grouped by concern.

  knowledge_base.py → search_knowledge_base (Qdrant RAG)
  web.py            → web_search (DuckDuckGo)
  apollo.py         → apollo_search (Apollo.io leads)
  salesforce.py     → Salesforce CRM tools (query/dml/search/upsert)
  email.py          → send_email (Brevo)
  runner.py         → call_tools (LLM tool-routing loop)
"""

from agents.tools.knowledge_base import search_knowledge_base
from agents.tools.web import web_search
from agents.tools.apollo import apollo_search
from agents.tools.salesforce import (
    salesforce_query_records,
    salesforce_dml_records,
    salesforce_search_leads,
    salesforce_upsert_lead,
    salesforce_aggregate_query,
    salesforce_search_objects,
    salesforce_describe_object,
    salesforce_read_apex,
    salesforce_write_apex,
    salesforce_execute_anonymous,
)
from agents.tools.email import send_email
from agents.tools.runner import call_tools

__all__ = [
    "search_knowledge_base",
    "web_search",
    "apollo_search",
    "salesforce_query_records",
    "salesforce_dml_records",
    "salesforce_search_leads",
    "salesforce_upsert_lead",
    "salesforce_aggregate_query",
    "salesforce_search_objects",
    "salesforce_describe_object",
    "salesforce_read_apex",
    "salesforce_write_apex",
    "salesforce_execute_anonymous",
    "send_email",
    "call_tools",
]
