"""Compatibility re-export. Canonical schema lives in ie643.schema."""

from ie643.schema import (
    CANONICAL_SCHEMA,
    FIELD_ORDER,
    TEACHER_SYSTEM_PROMPT,
    coerce_canonical,
    empty_canonical,
    teacher_prompt,
)

__all__ = [
    "CANONICAL_SCHEMA",
    "FIELD_ORDER",
    "TEACHER_SYSTEM_PROMPT",
    "coerce_canonical",
    "empty_canonical",
    "teacher_prompt",
]
