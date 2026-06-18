"""Make catalog photos reachable in Brevo emails (public HTTPS URL required for inline img)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import requests

from config import COMPONENT_IMAGE_PUBLIC_BASE_URL

logger = logging.getLogger(__name__)

_TEMP_UPLOAD_DEFAULT = os.getenv("CATALOG_IMAGE_TEMP_UPLOAD", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def catalog_image_public_url(match: dict) -> str | None:
    """Static CDN/base URL from env, if configured."""
    base = (COMPONENT_IMAGE_PUBLIC_BASE_URL or "").strip().rstrip("/")
    if not base:
        return None
    image_id = (match.get("image_id") or "").strip()
    image_path = match.get("image_path") or ""
    filename = Path(image_path).name if image_path else ""
    if image_id and filename:
        return f"{base}/{image_id}/{filename}"
    return None


def _upload_temp_public_url(image_path: Path) -> str | None:
    """Upload to catbox.moe so Brevo/Gmail can load <img src='https://...'> (dev/demo hack)."""
    try:
        mime = "image/jpeg"
        if image_path.suffix.lower() == ".png":
            mime = "image/png"
        with image_path.open("rb") as handle:
            resp = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (image_path.name, handle, mime)},
                timeout=60,
            )
        if resp.status_code == 200:
            url = resp.text.strip()
            if url.startswith("https://"):
                logger.info("Catalog image temp-hosted at %s", url)
                return url
    except Exception as exc:
        logger.warning("Temp catalog image upload failed: %s", exc)
    return None


def resolve_hosted_catalog_image_url(match: dict) -> str | None:
    """
    Return a public HTTPS URL for the catalog photo.
    Order: env CDN base → temp upload (0x0.st) when enabled.
    """
    static = catalog_image_public_url(match)
    if static:
        return static

    image_path = match.get("image_path") or ""
    path = Path(image_path)
    if not path.is_file():
        return None

    if not _TEMP_UPLOAD_DEFAULT:
        return None

    return _upload_temp_public_url(path)


def catalog_image_attachment_path(match: dict) -> str | None:
    image_path = match.get("image_path") or ""
    if image_path and Path(image_path).is_file():
        return image_path
    return None


def attachment_filename_for_match(match: dict) -> str:
    sku = re.sub(r"[^\w.\-]+", "_", (match.get("sku") or "catalog").strip())[:40]
    return f"catalog_{sku or 'product'}.jpg"
