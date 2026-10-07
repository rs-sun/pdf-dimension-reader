"""Vector hypothesis conversion plus its server-approved product entrypoint.

The R2 vector chain still emits dump rows with ``consumer_allowed=false`` for
audit compatibility. Product code must call
``consume_approved_vector_dimension_hypotheses`` with a decision from the
server resolver. The lower-level converter remains available to offline proof
scripts and unit tests, but is not an authorization boundary.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from assembler_dedup import _dedup_dimensions
from assembler_scoring import _post_filter_all
from assembler_utils import (
    _clean_dimension_text,
    _extract_prefix,
    _fixup_ocr_text,
    _nxfont_correction,
    _parse_nominal,
    _validate_engineering,
)
from final_consume_approval import FinalConsumeApproval


VECTOR_FINAL_SOURCE = "vector_glyph"
SUPPORTED_FINAL_PHRASE_KINDS = frozenset({
    "plain_number",
    "nominal_plus_minus_tolerance",
    "diameter_number",
    "angle_number",
})


def consume_approved_vector_dimension_hypotheses(
    *,
    approval: FinalConsumeApproval,
    legacy_dimensions: list[dict[str, Any]] | None,
    vector_dimension_hypotheses: list[dict[str, Any]] | None,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> dict[str, Any]:
    """Product entrypoint: require authority minted by the server resolver."""
    if (
        not isinstance(approval, FinalConsumeApproval)
        or approval.consumer_allowed is not True
        or approval.display_allowed is not True
        or approval.release_allowed is not False
    ):
        raise PermissionError("final_consume_not_approved")
    return consume_vector_dimension_hypotheses(
        enabled=True,
        legacy_dimensions=legacy_dimensions,
        vector_dimension_hypotheses=vector_dimension_hypotheses,
        page_width=page_width,
        page_height=page_height,
    )


def consume_vector_dimension_hypotheses(
    *,
    enabled: bool,
    legacy_dimensions: list[dict[str, Any]] | None,
    vector_dimension_hypotheses: list[dict[str, Any]] | None,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> dict[str, Any]:
    """Return merged dimensions plus an audit ledger.

    When disabled, this is a no-op. When enabled, only passed, geometry-bound
    vector hypotheses are converted to dimension dicts, post-filtered, then
    placed before legacy dimensions so equal-confidence de-dup prefers vector
    evidence for overlapping text.
    """
    legacy = list(legacy_dimensions or [])
    hypotheses = list(vector_dimension_hypotheses or [])
    ledger: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []

    if not enabled:
        return {
            "dimensions": legacy,
            "vector_dimensions": [],
            "ledger": [],
            "stats": {
                "schema_version": "vector_final_consume_stats_v1",
                "enabled": False,
                "input_count": len(hypotheses),
                "converted_count": 0,
                "consumed_count": 0,
                "legacy_count": len(legacy),
                "output_count": len(legacy),
                "dropped_by_reason": {},
            },
        }

    for row in hypotheses:
        dim, reason = _dimension_from_hypothesis(row)
        if dim is None:
            ledger.append(_ledger_row(row, reason))
            continue
        converted.append(dim)

    pre_filter_count = len(converted)
    vector_debug: dict[str, Any] = {}
    if converted:
        converted = _post_filter_all(
            converted,
            page_width=page_width,
            page_height=page_height,
            debug=vector_debug,
        )
    if pre_filter_count != len(converted):
        ledger.append({
            "reason": "post_filter_drop",
            "count": pre_filter_count - len(converted),
            "post_filter_debug_stats": vector_debug.get("post_filter_debug_stats", {}),
        })

    merged = _dedup_dimensions(converted + legacy) if converted else legacy
    dropped = Counter(str(item.get("reason") or "unknown") for item in ledger)
    return {
        "dimensions": merged,
        "vector_dimensions": converted,
        "ledger": ledger,
        "stats": {
            "schema_version": "vector_final_consume_stats_v1",
            "enabled": True,
            "input_count": len(hypotheses),
            "converted_count": pre_filter_count,
            "consumed_count": len(converted),
            "legacy_count": len(legacy),
            "output_count": len(merged),
            "dropped_by_reason": dict(sorted(dropped.items())),
            "post_filter_debug_stats": vector_debug.get("post_filter_debug_stats", {}),
        },
    }


def _dimension_from_hypothesis(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(row, dict):
        return None, "invalid_row"
    if str(row.get("schema_version") or "") != "vector_dimension_hypotheses_v1":
        return None, "unsupported_schema"
    gate = row.get("evidence_gate") or {}
    if gate.get("passed") is not True:
        return None, str(row.get("drop_reason") or "evidence_gate_failed")
    checks = gate.get("checks") or {}
    if checks.get("geometry_bound") is not True:
        return None, "not_geometry_bound"
    if checks.get("no_ocr_source") is not True:
        return None, "ocr_sourced_phrase"

    phrase_kind = str(row.get("phrase_kind") or "")
    if phrase_kind not in SUPPORTED_FINAL_PHRASE_KINDS:
        return None, "unsupported_phrase_kind"

    bbox = _bbox_from_any(row)
    if bbox is None:
        return None, "invalid_bbox"

    text = _normalize_text(str(row.get("text") or ""))
    if not text:
        return None, "empty_text"
    nominal_part, upper_tol, lower_tol = _split_nominal_and_tolerance(text)
    nominal_part, prefix, dim_type = _extract_prefix(nominal_part)
    nominal_part = nominal_part.strip()
    if phrase_kind == "angle_number":
        dim_type = "angle"
    elif phrase_kind == "diameter_number" and not prefix:
        prefix = "⌀"
        dim_type = "diameter"
    if not nominal_part:
        return None, "empty_nominal"
    if _contains_unsupported_array_nominal(nominal_part):
        return None, "engineering_invalid"

    nominal_val = _parse_nominal(nominal_part)
    validity = _validate_engineering(nominal_val, upper_tol, lower_tol)
    if validity == "low":
        return None, "engineering_invalid"

    confidence = _confidence_label(row)
    full_text = _display_text(
        nominal=nominal_part,
        prefix=prefix,
        upper_tol=upper_tol,
        lower_tol=lower_tol,
    )
    dim = {
        "text": full_text,
        "nominal": nominal_part,
        "upper_tol": upper_tol,
        "lower_tol": lower_tol,
        "type": dim_type,
        "prefix": prefix,
        "bbox": bbox,
        "confidence": confidence,
        "source": VECTOR_FINAL_SOURCE,
        "orientation": _orientation_from_row(row),
        "vector_hypothesis_id": str(row.get("hypothesis_id") or ""),
        "vector_confidence": row.get("confidence"),
        "source_phrase_id": str(row.get("source_phrase_id") or ""),
        "source_candidate_ids": list(row.get("source_candidate_ids") or []),
        "source_anchor_ids": list(row.get("source_anchor_ids") or []),
    }
    return dim, None


def _normalize_text(text: str) -> str:
    text = _clean_dimension_text(text)
    text = _nxfont_correction(text)
    text = _fixup_ocr_text(text)
    text = text.replace("∅", "⌀").replace("Ø", "⌀").replace("φ", "⌀")
    return re.sub(r"\s+", " ", text).strip()


def _split_nominal_and_tolerance(text: str) -> tuple[str, str | None, str | None]:
    m = re.search(r"[±]\s*([Oo0-9.]+)", text)
    if m:
        value = _clean_tolerance_value(m.group(1))
        return text[:m.start()].strip(), f"+{value}", f"-{value}"

    m = re.search(r"\+\s*([Oo0-9.]+)\s*/?\s*-\s*([Oo0-9.]+)", text)
    if m and _signed_tolerance_match_allowed(text, m.start(), m.group(1)):
        return (
            text[:m.start()].strip(),
            f"+{_clean_tolerance_value(m.group(1))}",
            f"-{_clean_tolerance_value(m.group(2))}",
        )

    m = re.search(r"\+\s*([Oo0-9.]+)\s+(0?\.\d+)(?!\d)", text)
    if m and _signed_tolerance_match_allowed(text, m.start(), m.group(1)):
        return (
            text[:m.start()].strip(),
            f"+{_clean_tolerance_value(m.group(1))}",
            f"-{_clean_tolerance_value(m.group(2))}",
        )

    m = re.search(r"\+\s*([Oo0-9.]+)(?!\d)", text)
    if m and _signed_tolerance_match_allowed(text, m.start(), m.group(1)):
        return text[:m.start()].strip(), f"+{_clean_tolerance_value(m.group(1))}", None

    m = re.search(r"-\s*([Oo0-9.]+)(?!\d)", text)
    if m and _signed_tolerance_match_allowed(text, m.start(), m.group(1)):
        return text[:m.start()].strip(), None, f"-{_clean_tolerance_value(m.group(1))}"

    return text.strip(), None, None


def _signed_tolerance_match_allowed(text: str, sign_index: int, raw_value: str) -> bool:
    if sign_index <= 0:
        return True
    if text[sign_index - 1].isspace():
        return True
    return _is_sub_one_tolerance(raw_value)


def _clean_tolerance_value(raw: str) -> str:
    return raw.replace(" ", "").replace("O", "0").replace("o", "0")


def _is_sub_one_tolerance(raw: str) -> bool:
    try:
        value = float(_clean_tolerance_value(raw))
    except ValueError:
        return False
    return 0.0 <= value < 1.0


def _contains_unsupported_array_nominal(text: str) -> bool:
    return bool(re.search(r"\d\s*[×*]\s*\d", text))


def _display_text(
    *,
    nominal: str,
    prefix: str | None,
    upper_tol: str | None,
    lower_tol: str | None,
) -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(nominal)
    if upper_tol and lower_tol:
        if upper_tol.lstrip("+") == lower_tol.lstrip("-"):
            parts.append(f"±{upper_tol.lstrip('+')}")
        else:
            parts.append(f"{upper_tol}/{lower_tol}")
    elif upper_tol:
        parts.append(upper_tol)
    elif lower_tol:
        parts.append(lower_tol)
    return " ".join(part for part in parts if part).strip()


def _confidence_label(row: dict[str, Any]) -> str:
    try:
        score = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    gate_score = (row.get("evidence_gate") or {}).get("score")
    try:
        score = max(score, float(gate_score or 0.0))
    except (TypeError, ValueError):
        pass
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _orientation_from_row(row: dict[str, Any]) -> float:
    quad = row.get("oriented_quad") or []
    if isinstance(quad, list) and len(quad) >= 2:
        p0 = quad[0]
        p1 = quad[1]
        if isinstance(p0, dict) and isinstance(p1, dict):
            x0, y0 = p0.get("x"), p0.get("y")
            x1, y1 = p1.get("x"), p1.get("y")
        else:
            try:
                x0, y0 = p0[0], p0[1]
                x1, y1 = p1[0], p1[1]
            except (TypeError, IndexError):
                return 0.0
        try:
            return round(math.degrees(math.atan2(float(y1) - float(y0), float(x1) - float(x0))), 3)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _bbox_from_any(row: dict[str, Any]) -> dict[str, float] | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": round(x, 3), "y": round(y, 3), "w": round(w, 3), "h": round(h, 3)}


def _ledger_row(row: dict[str, Any], reason: str | None) -> dict[str, Any]:
    return {
        "reason": reason or "unknown",
        "hypothesis_id": str(row.get("hypothesis_id") or ""),
        "text": str(row.get("text") or ""),
        "phrase_kind": str(row.get("phrase_kind") or ""),
        "bbox": row.get("bbox"),
    }
