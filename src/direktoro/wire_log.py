"""Redaction and serialisation for logging what went over the wire.

A consumer keeping a verbatim audit log of its API calls needs two things
only this package can know: which content-block shapes carry inline image
bytes on each wire it speaks, and how to flatten a provider SDK's response
object to a plain dict. Both live here, so the wire vocabulary stays in one
repository. The consumer's log itself — file layout, entry envelope,
locking — is its own business; every `NormalisedResponse` carries a
`wire_request` ready to be passed through `redact_wire_request` and written.

Redaction replaces an inline base64 image block with an `image_ref` stub
carrying the bytes' `media_type`, `sha256` and `byte_length` — enough for a
log reader to match the image against a separately stored copy and to detect
drift, without duplicating megabytes of base64 per call. It is the ONLY
redaction: text, tool_use and tool_result blocks pass through verbatim,
because their content is what an audit log is for.

Three block shapes carry inline image bytes, one per wire:

  - canonical / Anthropic: `{"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": <base64>}}`;
  - OpenAI Responses: `{"type": "input_image", "image_url":
    "data:<media>;base64,<base64>"}`;
  - OpenAI Chat Completions: `{"type": "image_url", "image_url":
    {"url": "data:<media>;base64,<base64>"}}`.

A plain (non-data) image URL is a small reference and passes through.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any


def _image_ref(media_type: Any, data: str) -> dict:
    """An `image_ref` stub from base64 `data`, or a decode-error stub."""
    try:
        raw = base64.b64decode(data) if data else b""
    except Exception:
        # Log what is in hand rather than crash the caller's audit log.
        return {
            "type": "image_ref",
            "media_type": media_type,
            "sha256": None,
            "byte_length": None,
            "_decode_error": True,
        }
    return {
        "type": "image_ref",
        "media_type": media_type,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
    }


def _redact_content_block(block: Any) -> Any:
    """One block: an inline-image shape becomes an `image_ref` stub; every
    other block (and a non-data image reference) passes through unchanged."""
    if not isinstance(block, dict):
        return block
    if block.get("type") == "image":
        src = block.get("source") or {}
        if src.get("type") != "base64":
            return block
        return _image_ref(src.get("media_type"), src.get("data") or "")
    if block.get("type") == "input_image":
        url = block.get("image_url")
        if isinstance(url, str) and url.startswith("data:") \
                and ";base64," in url:
            header, b64 = url.split(";base64,", 1)
            media_type = header[len("data:"):] or None
            return _image_ref(media_type, b64)
        return block
    if block.get("type") == "image_url":
        url = block.get("image_url")
        # Chat Completions nests the URL: {"image_url": {"url": ...}}.
        inner = url.get("url") if isinstance(url, dict) else None
        if isinstance(inner, str) and inner.startswith("data:") \
                and ";base64," in inner:
            header, b64 = inner.split(";base64,", 1)
            media_type = header[len("data:"):] or None
            return _image_ref(media_type, b64)
        return block
    return block


def _redact_content_list(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    return [_redact_content_block(b) for b in content]


def redact_messages(messages: Any) -> Any:
    """Walk a canonical `messages` list and stub inline image blocks.

    Text, tool_use and tool_result blocks pass through verbatim; their
    content is what an audit log is for.
    """
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, list):
            out.append({**m, "content": _redact_content_list(content)})
        else:
            out.append(m)
    return out


def redact_system(system: Any) -> Any:
    """`system` may be a string OR a list of content blocks (the canonical
    request supports both). Images are not expected there; redacted
    defensively in case a caller attaches reference images."""
    if isinstance(system, list):
        return _redact_content_list(system)
    return system


def redact_wire_request(wire: Any) -> Any:
    """Redact image bytes from a wire request before logging it.

    Every wire this package speaks is covered: the canonical / Anthropic
    request carries its conversation under `messages` (with `system`
    alongside), the OpenAI Responses request under a top-level `input` list,
    Chat Completions under `messages`. Message items hold content arrays that
    may include inline image bytes; those are walked and stubbed. Non-message
    items (function_call / function_call_output, `role:"tool"` results) and
    string-valued content pass through unchanged.
    """
    if not isinstance(wire, dict):
        return wire
    out = wire
    if isinstance(out.get("system"), list):
        out = {**out, "system": redact_system(out["system"])}
    # EVERY conversation-carrying key is walked, not just the first found: no
    # wire this package emits carries more than one, but this function is
    # public and a hand-built or merged wire can, and a redactor that stopped
    # at the first would fail OPEN on the second — the one failure mode a
    # redactor must not have.
    for key in ("input", "messages"):
        items = out.get(key)
        if not isinstance(items, list):
            continue
        redacted = []
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("content"),
                                                     list):
                redacted.append(
                    {**item, "content": _redact_content_list(item["content"])})
            else:
                redacted.append(item)
        out = {**out, key: redacted}
    return out


def response_to_dict(response: Any) -> Any:
    """Flatten a provider SDK response object to a plain dict.

    A dict passes through verbatim (the OpenAI paths already normalise to
    one), None stays None. An SDK object is flattened via pydantic v2's
    `.model_dump(mode="json")` — leaf types render as JSON natives, so the
    result survives a consumer's plain json.dumps — falling back to plain
    `.model_dump()`, the v1 `.dict()`, or an attribute walk over the fields
    an audit reader needs. Never raises: on any failure it returns
    `{"_serialisation_error": ...}`, so an append-only log is never the
    thing that kills a call that already succeeded — and that stub is also
    what an object the walk cannot faithfully represent produces, so "the
    model emitted nothing" and "we could not read the response" stay
    distinguishable in the record.
    """
    if response is None or isinstance(response, dict):
        return response
    try:
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump(mode="json")
            except TypeError:
                return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        return {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "stop_sequence": getattr(response, "stop_sequence", None),
            "role": getattr(response, "role", None),
            "type": getattr(response, "type", None),
            "usage": (response.usage.__dict__
                      if getattr(response, "usage", None) else None),
            # `.content` is read WITHOUT a default: an object with no content
            # attribute lands in the error stub below rather than reading as
            # an empty answer.
            "content": [
                (b if isinstance(b, dict)
                 else (b.model_dump() if hasattr(b, "model_dump")
                       else getattr(b, "__dict__", str(b))))
                for b in (response.content or [])
            ],
        }
    except Exception as exc:
        return {"_serialisation_error": repr(exc)}
