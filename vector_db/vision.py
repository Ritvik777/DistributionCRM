"""Claude vision helpers for component captioning and visual re-ranking."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from config import MAX_COMPONENT_IMAGE_UPLOAD_BYTES
from llm import get_vision_caption_llm, get_vision_rerank_llm

logger = logging.getLogger(__name__)


class ImageUploadTooLargeError(ValueError):
    """Raised when an uploaded image exceeds MAX_COMPONENT_IMAGE_UPLOAD_BYTES."""


def max_image_upload_label() -> str:
    mb = MAX_COMPONENT_IMAGE_UPLOAD_BYTES // (1024 * 1024)
    return f"{mb} MB"


def assert_image_upload_size(raw: bytes) -> None:
    if len(raw) > MAX_COMPONENT_IMAGE_UPLOAD_BYTES:
        raise ImageUploadTooLargeError(
            f"Image is too large ({len(raw) / (1024 * 1024):.1f} MB). Maximum upload size is {max_image_upload_label()}."
        )

_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}


class ComponentCaption(BaseModel):
    summary: str = Field(description="One-sentence description of the component")
    category: str = Field(description="Component category, e.g. LED, resistor, capacitor, connector")
    package: str = Field(description="Package or form factor, e.g. through-hole 5mm, SMD 0805")
    color: str = Field(default="", description="Dominant color if visible")
    size: str = Field(default="", description="Visible size or dimensions if inferable")
    part_numbers: list[str] = Field(default_factory=list, description="Any visible part numbers or markings")
    visible_text: str = Field(default="", description="All readable text on the component or label")
    distinguishing_features: str = Field(
        default="",
        description="Unique visual traits: lead count, lens shape, marking codes, pin layout",
    )


class CandidateVisionScore(BaseModel):
    image_id: str
    score: int = Field(ge=0, le=100, description="Visual match confidence 0-100")
    reasoning: str


class VisionRerankResult(BaseModel):
    query_summary: str
    candidates: list[CandidateVisionScore]


def media_type_for_filename(filename: str, fallback: str = "image/jpeg") -> str:
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    return _MEDIA_TYPES.get(ext, fallback)


def detect_media_type(image_bytes: bytes, filename: str = "") -> str:
    """Detect MIME type from actual image bytes (filename is fallback only)."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            fmt = (img.format or "").upper()
            mapping = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
                "GIF": "image/gif",
            }
            if fmt in mapping:
                return mapping[fmt]
    except Exception:
        pass

    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if filename:
        return media_type_for_filename(filename)
    return "image/jpeg"


def _image_block(image_bytes: bytes, media_type: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(image_bytes).decode("ascii"),
        },
    }


def _invoke_vision_structured(
    model: type[BaseModel],
    content: list[dict[str, Any]],
    *,
    use_rerank_model: bool = False,
) -> BaseModel | None:
    try:
        llm_factory = get_vision_rerank_llm if use_rerank_model else get_vision_caption_llm
        llm = llm_factory(temperature=0).with_structured_output(model)
        return llm.invoke([HumanMessage(content=content)])
    except Exception as exc:
        logger.exception("Vision structured call failed: %s", exc)
        return None


def _bytes_for_vision_api(raw: bytes) -> tuple[bytes, str]:
    """Normalize for Claude vision; media type always matches payload bytes."""
    prepared = prepare_image_bytes(raw)
    return prepared, detect_media_type(prepared)


def caption_component_image(
    image_bytes: bytes,
    *,
    filename: str = "",
    sku: str = "",
    name: str = "",
) -> ComponentCaption | None:
    vision_bytes, media_type = _bytes_for_vision_api(image_bytes)
    hints = []
    if sku:
        hints.append(f"Catalog SKU: {sku}")
    if name:
        hints.append(f"Catalog name: {name}")
    hint_text = "\n".join(hints)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are cataloging electronic components from product photos for a parts distributor.\n"
                "Describe the component precisely for later visual matching.\n"
                "CRITICAL: transcribe ALL visible text, part numbers, date codes, and marking lines exactly.\n"
                "Focus on category, package (through-hole/SMD/size), color, lead/pin count, lens or body shape.\n"
                "Examples: '5mm red LED 2-lead' vs '3mm red LED' are DIFFERENT parts — capture size/package.\n"
                f"{hint_text}\n"
                "If unsure about a field, leave it empty rather than guessing."
            ),
        },
        _image_block(vision_bytes, media_type),
    ]
    return _invoke_vision_structured(ComponentCaption, content)


def caption_to_catalog_text(caption: ComponentCaption, *, sku: str = "", name: str = "") -> str:
    parts = ["Component image catalog entry"]
    if sku:
        parts.append(f"SKU: {sku}")
    if name:
        parts.append(f"Name: {name}")
    parts.append(f"Summary: {caption.summary}")
    if caption.category:
        parts.append(f"Category: {caption.category}")
    if caption.package:
        parts.append(f"Package: {caption.package}")
    if caption.color:
        parts.append(f"Color: {caption.color}")
    if caption.size:
        parts.append(f"Size: {caption.size}")
    if caption.part_numbers:
        parts.append(f"Part numbers: {', '.join(caption.part_numbers)}")
    if caption.visible_text:
        parts.append(f"Visible text: {caption.visible_text}")
    if caption.distinguishing_features:
        parts.append(f"Features: {caption.distinguishing_features}")
    return "\n".join(parts)


def rerank_component_candidates(
    query_bytes: bytes,
    candidates: list[dict],
    *,
    filename: str = "",
) -> VisionRerankResult | None:
    if not candidates:
        return None

    query_bytes, query_media = _bytes_for_vision_api(query_bytes)
    lines = [
        "Image 1 is the QUERY component a user photographed.",
        "Images 2 onward are CATALOG reference photos already in our database.",
        "Score each catalog candidate 0-100 for whether it is the SAME component (not just same category).",
        "Penalize candidates that differ in size, package, lead count, or marking code.",
        "90-100 = same part; 70-89 = very likely; 50-69 = similar category only; below 50 = different part.",
        "Use shape, package, color, markings, lead count, and visible text.",
        "Candidates:",
    ]
    for index, candidate in enumerate(candidates, start=2):
        lines.append(
            f"- image_id={candidate['image_id']} (Image {index}): "
            f"SKU={candidate.get('sku') or 'n/a'}, name={candidate.get('name') or 'n/a'}, "
            f"caption={candidate.get('caption') or ''}"
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}, _image_block(query_bytes, query_media)]
    for candidate in candidates:
        path = candidate.get("image_path")
        if not path:
            continue
        try:
            catalog_bytes = open(path, "rb").read()
        except OSError:
            continue
        catalog_bytes, catalog_media = _bytes_for_vision_api(catalog_bytes)
        content.append(_image_block(catalog_bytes, catalog_media))

    return _invoke_vision_structured(VisionRerankResult, content, use_rerank_model=True)


def prepare_image_bytes(raw: bytes, *, max_edge: int = 1600) -> bytes:
    """Normalize orientation/size so CLIP and Claude see a consistent RGB JPEG."""
    assert_image_upload_size(raw)
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / longest
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=92)
        return out.getvalue()
    except Exception as exc:
        logger.warning("Image normalize failed, using original bytes: %s", exc)
        return raw
