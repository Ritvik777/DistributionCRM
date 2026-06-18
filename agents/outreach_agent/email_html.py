"""Formal HTML email layout for Brevo outreach sends."""

from __future__ import annotations

import html
import os
import re


def _paragraphs_to_html(text: str) -> str:
    """Convert plain-text email body to HTML paragraphs and bullet lists."""
    blocks = re.split(r"\n\s*\n", text.strip())
    parts: list[str] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if all(re.match(r"^[\-\*•]\s+", line) for line in lines):
            items = "".join(
                f"<li style=\"margin: 4px 0;\">{html.escape(re.sub(r'^[\-\*•]\s+', '', line))}</li>"
                for line in lines
            )
            parts.append(
                f'<ul style="margin: 0 0 16px 0; padding-left: 22px; color: #374151;">{items}</ul>'
            )
            continue
        paragraph = "<br>".join(html.escape(line) for line in lines)
        paragraph = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", paragraph)
        parts.append(f'<p style="margin: 0 0 16px 0; color: #374151;">{paragraph}</p>')
    return "".join(parts) or f'<p style="margin: 0; color: #374151;">{html.escape(text)}</p>'


def _product_details_block(match: dict) -> str:
    sku = html.escape(match.get("sku") or "N/A")
    name = html.escape(match.get("name") or match.get("caption") or "Component")
    package = html.escape(match.get("package") or "—")
    category = html.escape(match.get("category") or "—")
    confidence = match.get("match_percent")
    confidence_row = ""
    if confidence is not None:
        confidence_row = (
            f'<tr><td style="padding: 8px 12px; color: #6b7280; width: 140px;">Match</td>'
            f'<td style="padding: 8px 12px; color: #111827;">{confidence}% catalog confidence</td></tr>'
        )
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="margin: 24px 0; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;">'
        '<tr><td colspan="2" style="background: #f9fafb; padding: 12px 16px; '
        'font-size: 13px; font-weight: 600; color: #111827; letter-spacing: 0.02em;">'
        "PRODUCT DETAILS</td></tr>"
        f'<tr><td style="padding: 8px 12px; color: #6b7280; border-top: 1px solid #e5e7eb;">SKU</td>'
        f'<td style="padding: 8px 12px; color: #111827; border-top: 1px solid #e5e7eb;"><strong>{sku}</strong></td></tr>'
        f'<tr><td style="padding: 8px 12px; color: #6b7280;">Description</td>'
        f'<td style="padding: 8px 12px; color: #111827;">{name}</td></tr>'
        f'<tr><td style="padding: 8px 12px; color: #6b7280;">Package</td>'
        f'<td style="padding: 8px 12px; color: #111827;">{package}</td></tr>'
        f'<tr><td style="padding: 8px 12px; color: #6b7280;">Category</td>'
        f'<td style="padding: 8px 12px; color: #111827;">{category}</td></tr>'
        f"{confidence_row}"
        "</table>"
    )


def _catalog_image_block(match: dict, hosted_url: str | None = None) -> str:
    """Brevo/Gmail need a public https URL — base64 inline is stripped by most clients."""
    sku = html.escape(match.get("sku") or "product")
    alt = html.escape(match.get("name") or match.get("sku") or "Catalog product")

    if not hosted_url:
        return (
            '<div style="margin: 24px 0; padding: 16px; background: #fff7ed; border-radius: 8px; '
            'border: 1px solid #fed7aa; text-align: center;">'
            '<p style="margin: 0; font-size: 14px; color: #9a3412;">'
            f"📎 Catalog photo for SKU <strong>{sku}</strong> is attached to this email.</p>"
            "</div>"
        )

    safe_url = html.escape(hosted_url)
    return (
        '<div style="margin: 24px 0; padding: 20px; background: #f9fafb; border-radius: 8px; '
        'border: 1px solid #e5e7eb; text-align: center;">'
        '<p style="margin: 0 0 12px 0; font-size: 13px; font-weight: 600; color: #374151;">'
        "CATALOG REFERENCE</p>"
        f'<img src="{safe_url}" alt="{alt}" width="280" '
        'style="display: block; max-width: 100%; height: auto; border-radius: 6px; '
        'border: 1px solid #e5e7eb; margin: 0 auto 12px auto;">'
        f'<p style="margin: 0; text-align: center;">'
        f'<a href="{safe_url}" style="color: #2563eb; text-decoration: none; font-size: 14px;">'
        f"View catalog photo ({sku}) →</a></p>"
        "</div>"
    )


def build_formal_email_html(
    body: str,
    *,
    component_matches: list[dict] | None = None,
    catalog_image_url: str | None = None,
    recipient_name: str = "",
) -> str:
    """Build a formal B2B HTML email with optional catalog product image."""
    from_name = os.getenv("BREVO_FROM_NAME", "Product Distribution Team")
    from_email = os.getenv("BREVO_FROM_EMAIL", "")

    body_html = _paragraphs_to_html(body)
    product_block = ""
    image_block = ""
    if component_matches:
        best = component_matches[0]
        product_block = _product_details_block(best)
        image_block = _catalog_image_block(best, hosted_url=catalog_image_url)

    signature_email = (
        f'<span style="color: #6b7280; font-size: 14px;">{html.escape(from_email)}</span>'
        if from_email
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="margin: 0; padding: 0; background-color: #f3f4f6; font-family: Georgia, 'Times New Roman', serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f3f4f6; padding: 32px 16px;">
    <tr>
      <td align="center">
        <table role="presentation" width="600" cellpadding="0" cellspacing="0"
               style="max-width: 600px; width: 100%; background-color: #ffffff; border-radius: 8px;
                      border: 1px solid #e5e7eb; overflow: hidden;">
          <tr>
            <td style="padding: 28px 40px 20px; border-bottom: 1px solid #e5e7eb; background: #ffffff;">
              <p style="margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                         font-size: 18px; font-weight: 600; color: #111827; letter-spacing: -0.01em;">
                {html.escape(from_name)}
              </p>
              <p style="margin: 4px 0 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                         font-size: 13px; color: #6b7280;">Product availability &amp; catalog inquiry</p>
            </td>
          </tr>
          <tr>
            <td style="padding: 32px 40px; font-size: 16px; line-height: 1.65;">
              {body_html}
              {product_block}
              {image_block}
              <p style="margin: 24px 0 4px; color: #111827;">Best regards,</p>
              <p style="margin: 0; color: #111827;"><strong>{html.escape(from_name)}</strong></p>
              {f'<p style="margin: 4px 0 0;">{signature_email}</p>' if signature_email else ''}
            </td>
          </tr>
          <tr>
            <td style="padding: 16px 40px 24px; border-top: 1px solid #e5e7eb; background: #f9fafb;">
              <p style="margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                         font-size: 11px; color: #9ca3af; line-height: 1.5;">
                This message was sent regarding a product catalog inquiry. Please reply directly if you have questions.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
