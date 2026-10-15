from __future__ import annotations

import json
import re
from typing import Any

from ie643.schema import coerce_canonical

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_ZERO_PAD = re.compile(r'(:\s*"[^"\n]*?)0{12,}')
_BOX_RUN = re.compile(r'(:\s*"[^"\n]*?)([ㅁ□■▪▫�])\2{7,}')


def _repair_truncated_json(raw: str) -> str:
    """Close a generation that ran away repeating glyphs or hit max_tokens."""
    s = _ZERO_PAD.sub(r'\1"', raw)
    s = _BOX_RUN.sub(r'\1\2\2\2"', s)
    s = s.rstrip()
    if s.endswith("\\"):
        s = s[:-1]
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif stack and ch == stack[-1]:
            stack.pop()
    if in_str:
        s += '"'
    if s.rstrip().endswith(","):
        s = s.rstrip()[:-1]
    s += "".join(reversed(stack))
    return s


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    raw = _FENCE.sub("", raw).strip()
    if not raw:
        raise ValueError("empty model output")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        snippet = raw[i : j + 1] if i >= 0 and j > i else raw
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError:
            payload = json.loads(_repair_truncated_json(raw if i < 0 else raw[i:]))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object, got {type(payload).__name__}")
    return payload


def parse_teacher_output(text: str, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    return coerce_canonical(extract_json_object(text), fallback=fallback)
