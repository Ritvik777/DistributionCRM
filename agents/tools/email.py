import os

import requests
from langchain_core.tools import tool

from observability import log_span


@tool
@log_span(span_type="tool", name="send_email")
def send_email(to_email: str, subject: str, html_body: str) -> str:
    """Send a personalized marketing email via Brevo. Provide recipient email, subject line, and HTML body."""
    api_key = os.getenv("BREVO_API_KEY")
    if not api_key or api_key == "your-brevo-api-key-here":
        return "ERROR: BREVO_API_KEY not configured in .env"

    from_email = os.getenv("BREVO_FROM_EMAIL")
    if not from_email or from_email == "you@yourcompany.com":
        return "ERROR: BREVO_FROM_EMAIL not configured in .env"

    from_name = os.getenv("BREVO_FROM_NAME", "Product Marketing")
    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_body,
    }

    try:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": api_key, "Content-Type": "application/json", "accept": "application/json"},
            json=payload,
            timeout=15,
        )
        if resp.status_code in (200, 201, 202):
            return f"SENT to {to_email} (status {resp.status_code})"
        return f"FAILED (status {resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"
