import base64
import logging
from pathlib import Path

import requests
from langchain_core.tools import tool

from config import BREVO_API_KEY, BREVO_FROM_EMAIL, BREVO_FROM_NAME
from observability import log_span

logger = logging.getLogger(__name__)


def deliver_brevo_email(
    to_email: str,
    subject: str,
    html_body: str,
    *,
    attachment_paths: list[str] | None = None,
    attachment_urls: list[tuple[str, str]] | None = None,
) -> str:
    """Send via Brevo REST API. attachment_urls: list of (https_url, filename)."""
    api_key = BREVO_API_KEY
    if not api_key or api_key == "your-brevo-api-key-here":
        return "ERROR: BREVO_API_KEY not configured in .env"

    from_email = BREVO_FROM_EMAIL
    if not from_email or from_email == "you@yourcompany.com":
        return "ERROR: BREVO_FROM_EMAIL not configured in .env"

    from_name = BREVO_FROM_NAME
    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_body,
    }

    attachments: list[dict] = []
    seen_names: set[str] = set()

    for path_str in attachment_paths or []:
        path = Path(path_str)
        if not path.is_file():
            logger.warning("Attachment missing: %s", path_str)
            continue
        name = path.name if path.suffix else f"{path.name}.jpg"
        if name in seen_names:
            continue
        seen_names.add(name)
        attachments.append({
            "content": base64.standard_b64encode(path.read_bytes()).decode("ascii"),
            "name": name,
        })

    for url, name in attachment_urls or []:
        if not url.startswith("https://") or name in seen_names:
            continue
        seen_names.add(name)
        attachments.append({"url": url, "name": name})

    if attachments:
        payload["attachment"] = attachments
        logger.info("Brevo send to %s with %d attachment(s)", to_email, len(attachments))

    try:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": api_key, "Content-Type": "application/json", "accept": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code in (200, 201, 202):
            return f"SENT to {to_email} (status {resp.status_code})"
        return f"FAILED (status {resp.status_code}): {resp.text[:300]}"
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
@log_span(span_type="tool", name="send_email")
def send_email(
    to_email: str,
    subject: str,
    html_body: str,
    attachment_paths: list[str] | None = None,
) -> str:
    """Send a personalized marketing email via Brevo. Provide recipient email, subject line, and HTML body."""
    return deliver_brevo_email(
        to_email,
        subject,
        html_body,
        attachment_paths=attachment_paths,
    )
