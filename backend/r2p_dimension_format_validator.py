"""Dump-only dimension text format validator for R2p readers."""

from __future__ import annotations

import re
from typing import Any


SCHEMA_VERSION = "r2p_dimension_format_validator_v1"
CONSUMER_ALLOWED = False

_NUMBER_RE = re.compile(r"^(?P<sign>[+\-]?)(?P<int>\d+)(?:\.(?P<frac>\d+))?$")
_MAIN_PREFIX_RE = re.compile(r"^(?:(?P<qty>\d+)[*xX×])?(?P<prefix>Ra|R|Ø)?(?P<rest>.*)$")
_CHAMFER_RE = re.compile(
    r"^(?P<size>[+\-]?\d{1,3}(?:\.\d{1,3})?)[*xX×]"
    r"(?P<angle>\d{1,3}(?:\.\d{1,3})?)°$"
)
_DIAMETER_EQUIVALENTS = frozenset({"ø", "⌀", "∅", "Φ", "φ"})


def validate_dimension_text_format(text: str) -> dict[str, Any]:
    source_text = str(text or "")
    normalized_text = re.sub(r"\s+", "", source_text)
    canonical_text, normalization_ledger = _canonicalize_dimension_tokens(
        normalized_text
    )
    valid, reason = _validate(canonical_text)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_text": source_text,
        "normalized_text": normalized_text,
        "canonical_text": canonical_text,
        "normalization_ledger": normalization_ledger,
        "text": canonical_text,
        "valid": valid,
        "reason": reason,
        "block_output": _should_block_output(
            canonical_text,
            valid,
            reason,
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _canonicalize_dimension_tokens(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    chars: list[str] = []
    ledger: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        canonical = "Ø" if char in _DIAMETER_EQUIVALENTS else char
        chars.append(canonical)
        if canonical == char:
            continue
        ledger.append({
            "source_index": index,
            "source_token": char,
            "canonical_token": canonical,
            "reason": "diameter_symbol_equivalent",
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return "".join(chars), ledger


def _validate(text: str) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    if re.fullmatch(r"[.\-+±/]+", text):
        return False, "symbol_only"
    if not any(ch.isdigit() for ch in text):
        return False, "no_digits"
    if "---" in text:
        return False, "long_minus_run"

    qualifier = ""
    for candidate in ("MAX", "MIN"):
        if text.endswith(candidate):
            qualifier = candidate
            text = text[:-len(candidate)]
            break
    if not text:
        return False, "missing_main"
    if _CHAMFER_RE.fullmatch(text):
        return (False, "invalid_qualifier") if qualifier else (True, "ok")

    match = _MAIN_PREFIX_RE.match(text)
    if match is None:
        return False, "invalid_prefix"
    quantity = match.group("qty")
    if quantity is not None and (
        len(quantity) > 3
        or not quantity.isascii()
        or not quantity.isdigit()
        or int(quantity) < 1
    ):
        return False, "invalid_quantity"
    rest = match.group("rest")
    main, tolerance = _split_main_and_tolerance(rest)
    if not main:
        return False, "missing_main"
    main_has_degree = main.endswith("°")
    if tolerance and "°" in tolerance and not main_has_degree:
        return False, "invalid_degree_unit"
    if not _valid_main_number(main):
        return False, "invalid_main_number"
    if tolerance and not _valid_tolerance(tolerance):
        return False, "invalid_tolerance_number"
    if qualifier and tolerance:
        return False, "invalid_qualifier"
    if not tolerance and rest != main:
        return False, "invalid_suffix"
    return True, "ok"


def _split_main_and_tolerance(rest: str) -> tuple[str, str]:
    for idx, char in enumerate(rest):
        if idx == 0:
            continue
        if char in "+-±":
            return rest[:idx], rest[idx:]
    return rest, ""


def _valid_main_number(text: str) -> bool:
    if text.endswith("°"):
        text = text[:-1]
    match = _NUMBER_RE.match(text)
    if match is None:
        return False
    frac = match.group("frac") or ""
    return 1 <= len(match.group("int")) <= 3 and len(frac) <= 3


def _valid_tolerance(text: str) -> bool:
    if text.startswith("±"):
        return _valid_tolerance_number(text[1:])
    match = re.match(r"^([+\-][^+\-/]+)(?:/?([+\-][^+\-/]+))?$", text)
    if match is None:
        return False
    return all(
        _valid_tolerance_number(part[1:])
        for part in match.groups()
        if part
    )


def _valid_tolerance_number(text: str) -> bool:
    if text.endswith("°"):
        text = text[:-1]
    match = _NUMBER_RE.match(text)
    # The tolerance separator already carries the sign.  Accepting another
    # sign here would silently turn a negative tolerance magnitude into
    # valid symmetric tolerances while negative main values still need the
    # signed number grammar above.
    if match is None or match.group("sign"):
        return False
    frac = match.group("frac") or ""
    return 1 <= len(match.group("int")) <= 1 and len(frac) <= 3


def _should_block_output(text: str, valid: bool, reason: str) -> bool:
    if valid:
        return False
    if reason in {"empty"}:
        return False
    if reason in {"symbol_only", "long_minus_run"}:
        return True
    if reason == "invalid_tolerance_number":
        return True
    return False
