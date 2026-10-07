"""Stable review labels for text already produced by the vector reader.

These labels are advisory.  They must never decide whether a non-empty vector
read reaches review.
"""

from __future__ import annotations

import re
from typing import Any

from r2p_dimension_format_validator import validate_dimension_text_format


SYNTAX_OK = "ok"
SYNTAX_INCOMPLETE_SEPARATOR = "incomplete_separator"
SYNTAX_UNREAD_GLYPH_PRESENT = "unread_glyph_present"
SYNTAX_INVALID_DIMENSION = "invalid_dimension_syntax"
DIMENSION_SYNTAX_LABELS = frozenset({
    SYNTAX_OK,
    SYNTAX_INCOMPLETE_SEPARATOR,
    SYNTAX_UNREAD_GLYPH_PRESENT,
    SYNTAX_INVALID_DIMENSION,
})

_UNREAD_DROP_REASONS = frozenset({
    "unrecognized_slot",
    "grouped_window_authority_read_failed",
})
_SEPARATOR_FREE_PARTIAL = re.compile(
    r"^(?:\d+[xX×])?[ØR]?[+\-]?\d+(?:\.\d+){2,}$"
)


def classify_dimension_syntax(
    text: Any,
    *,
    drops: Any = None,
    format_result: Any = None,
) -> str:
    """Classify a vector read without suppressing it.

    Unread glyph evidence takes precedence because a syntactically valid
    partial numeric string can still have occupied trailing ink.
    """
    source_text = text if isinstance(text, str) else ""
    if _has_unread_glyph_drop(drops):
        return SYNTAX_UNREAD_GLYPH_PRESENT
    formal = (
        format_result
        if isinstance(format_result, dict)
        else validate_dimension_text_format(source_text)
    )
    if formal.get("valid") is True:
        return SYNTAX_OK
    if _SEPARATOR_FREE_PARTIAL.fullmatch(source_text):
        return SYNTAX_INCOMPLETE_SEPARATOR
    return SYNTAX_INVALID_DIMENSION


def _has_unread_glyph_drop(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and any(
            isinstance(row, dict)
            and row.get("reason") in _UNREAD_DROP_REASONS
            for row in value
        )
    )
