"""R2b OCR pass necessity dump layer.

See docs/restart_plans/R2b.md for the dump-only contract. This module only
attributes existing OCR evidence to final dimensions; it never disables an OCR
pass and treats unlinked OCR rows as diagnostic survivors, not claimed output.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


SCHEMA_VERSION = "ocr_pass_necessity_v1"
PASS_LINK_IOU = 0.10
SHORT_TEXT_LINK_IOU = 0.30
VECTOR_SUPPORT_IOU = 0.01

PASS_NAMES = (
    "candidate_guided",
    "dimline",
    "angle_label",
    "radius_label",
    "directed",
    "gdt_comp",
    "capsule",
    "strip",
    "hires_retry",
    "phase2_90",
    "phase2_270",
)

OFF_FLAG_NAMES = {
    "candidate_guided": "CANDIDATE_GUIDED_OCR",
    "dimline": "OCR_PASS_DIMLINE_ENABLED",
    "angle_label": "OCR_ANGLE_LABEL_ENABLED",
    "radius_label": "OCR_RADIUS_LABEL_ENABLED",
    "directed": "OCR_PASS_DIRECTED_ENABLED",
    "gdt_comp": "OCR_PASS_GDT_COMP_ENABLED",
    "capsule": "OCR_PASS_CAPSULE_ENABLED",
    "strip": "OCR_PASS_STRIP_ENABLED",
    "hires_retry": "OCR_PASS_HIRES_RETRY_ENABLED",
    "phase2_90": "OCR_PASS_PHASE2_90_ENABLED",
    "phase2_270": "OCR_PASS_PHASE2_270_ENABLED",
}

SOURCE_TO_PASS = {
    "candidate_guided_ocr": "candidate_guided",
    "dimline_strip_ocr": "dimline",
    "angle_label_ocr": "angle_label",
    "radius_label_ocr": "radius_label",
    "directed_ocr": "directed",
    "gdt_compartment_ocr": "gdt_comp",
    "capsule_ocr": "capsule",
    "strip_ocr": "strip",
    "hires_retry": "hires_retry",
}

BREAKDOWN_KEYS = {
    "candidate_guided": "candidate_guided",
    "dimline": "dimline",
    "angle_label": "angle_label",
    "radius_label": "radius_label",
    "directed": "base",
    "gdt_comp": "gdt_comp",
    "capsule": "capsule",
    "strip": "strip",
    "hires_retry": "hires",
    "phase2_90": "phase2_90",
    "phase2_270": "phase2_270",
}

REQUIRED_ROW_FIELDS = frozenset({
    "schema_version",
    "pass_name",
    "raw_count",
    "dedup_survivor_count",
    "claimed_dimension_count",
    "exclusive_claimed_dimension_count",
    "vector_replaced_count",
    "fallback_trigger",
    "off_flag_name",
})

_NON_TOKEN_RE = re.compile(r"[^0-9A-Z.+\-±°⌀Ø∅ΦΦφX/]")
_DIAMETER_RE = re.compile(r"[Ø∅Φφ]")


def build_ocr_pass_necessity_v1(
    *,
    dimensions: list[dict[str, Any]] | None = None,
    ocr_results: list[dict[str, Any]] | None = None,
    ocr_breakdown: dict[str, Any] | None = None,
    ocr_pass_counters: dict[str, Any] | None = None,
    pass_results_by_name: dict[str, list[dict[str, Any]] | None] | None = None,
    vector_glyph_tokens: list[dict[str, Any]] | None = None,
    iou_threshold: float = PASS_LINK_IOU,
) -> dict[str, Any]:
    """Build pass-level diagnostic necessity counters.

    The only rows counted as claimed contributions are final OCR survivors that
    can be linked to a final dimension by pass provenance plus text/bbox
    evidence. Rows with unknown provenance remain in survivor/unlinked stats.
    """

    dims = list(dimensions or [])
    survivors = list(ocr_results or [])
    breakdown = dict(ocr_breakdown or {})
    counters = dict(ocr_pass_counters or {})
    raw_by_pass = dict(pass_results_by_name or {})
    glyphs = list(vector_glyph_tokens or [])
    identity_pass = _identity_pass_index(raw_by_pass)

    survivor_by_pass: Counter[str] = Counter()
    unlinked_survivor_by_pass: Counter[str] = Counter()
    unknown_survivor_count = 0
    linked_passes_by_dim: dict[str, set[str]] = {
        _dimension_id(dim, idx): set()
        for idx, dim in enumerate(dims)
    }
    linked_rows_by_dim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dim_by_id = {
        _dimension_id(dim, idx): dim
        for idx, dim in enumerate(dims)
    }

    for row in survivors:
        pass_name = _row_pass_name(row, identity_pass)
        if pass_name not in PASS_NAMES:
            unknown_survivor_count += 1
            continue
        survivor_by_pass[pass_name] += 1
        matched_dimension_ids = [
            _dimension_id(dim, dim_idx)
            for dim_idx, dim in enumerate(dims)
            if _ocr_row_supports_dimension(
                row,
                dim,
                iou_threshold=iou_threshold,
            )
        ]
        if not matched_dimension_ids:
            unlinked_survivor_by_pass[pass_name] += 1
            continue
        for dim_id in matched_dimension_ids:
            linked_passes_by_dim[dim_id].add(pass_name)
            linked_rows_by_dim[dim_id].append(row)

    claimed_by_pass: Counter[str] = Counter()
    exclusive_by_pass: Counter[str] = Counter()
    vector_replaced_by_pass: Counter[str] = Counter()
    claimed_dimension_ids_by_pass: dict[str, list[str]] = defaultdict(list)
    exclusive_dimension_ids_by_pass: dict[str, list[str]] = defaultdict(list)
    vector_dimension_ids_by_pass: dict[str, list[str]] = defaultdict(list)

    for dim_id, pass_set in linked_passes_by_dim.items():
        if not pass_set:
            continue
        dim = dim_by_id[dim_id]
        vector_supported = _dimension_has_vector_support(dim, glyphs)
        for pass_name in sorted(pass_set):
            claimed_by_pass[pass_name] += 1
            claimed_dimension_ids_by_pass[pass_name].append(dim_id)
            if vector_supported:
                vector_replaced_by_pass[pass_name] += 1
                vector_dimension_ids_by_pass[pass_name].append(dim_id)
        if len(pass_set) == 1:
            only_pass = next(iter(pass_set))
            exclusive_by_pass[only_pass] += 1
            exclusive_dimension_ids_by_pass[only_pass].append(dim_id)

    rows: list[dict[str, Any]] = []
    for pass_name in PASS_NAMES:
        raw_count = _raw_count_for_pass(
            pass_name,
            pass_results_by_name=raw_by_pass,
            ocr_breakdown=breakdown,
        )
        ocr_calls = _counter_int(counters.get(pass_name), "ocr_calls")
        row = {
            "schema_version": SCHEMA_VERSION,
            "pass_name": pass_name,
            "raw_count": raw_count,
            "dedup_survivor_count": int(survivor_by_pass.get(pass_name, 0)),
            "claimed_dimension_count": int(claimed_by_pass.get(pass_name, 0)),
            "exclusive_claimed_dimension_count": int(exclusive_by_pass.get(pass_name, 0)),
            "vector_replaced_count": int(vector_replaced_by_pass.get(pass_name, 0)),
            "fallback_trigger": _fallback_trigger(
                raw_count=raw_count,
                dedup_survivor_count=survivor_by_pass.get(pass_name, 0),
                claimed_dimension_count=claimed_by_pass.get(pass_name, 0),
                exclusive_claimed_dimension_count=exclusive_by_pass.get(pass_name, 0),
                vector_replaced_count=vector_replaced_by_pass.get(pass_name, 0),
            ),
            "off_flag_name": OFF_FLAG_NAMES[pass_name],
            "ocr_calls": ocr_calls,
            "unlinked_survivor_count": int(unlinked_survivor_by_pass.get(pass_name, 0)),
            "claimed_dimension_ids": sorted(claimed_dimension_ids_by_pass.get(pass_name, [])),
            "exclusive_dimension_ids": sorted(exclusive_dimension_ids_by_pass.get(pass_name, [])),
            "vector_replaced_dimension_ids": sorted(vector_dimension_ids_by_pass.get(pass_name, [])),
        }
        rows.append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "ocr_pass_necessity_v1": rows,
        "ocr_pass_necessity_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "pass_count": len(rows),
            "final_dimension_count": len(dims),
            "total_ocr_calls": sum(
                _counter_int(counters.get(pass_name), "ocr_calls")
                for pass_name in PASS_NAMES
            ),
            "total_raw_count": sum(row["raw_count"] for row in rows),
            "total_dedup_survivor_count": sum(
                row["dedup_survivor_count"] for row in rows
            ),
            "total_claimed_dimension_count": sum(
                row["claimed_dimension_count"] for row in rows
            ),
            "total_exclusive_claimed_dimension_count": sum(
                row["exclusive_claimed_dimension_count"] for row in rows
            ),
            "total_vector_replaced_count": sum(
                row["vector_replaced_count"] for row in rows
            ),
            "unlinked_ocr_row_count": sum(
                row["unlinked_survivor_count"] for row in rows
            ),
            "unknown_pass_survivor_count": unknown_survivor_count,
            "schema_field_coverage": _schema_field_coverage(rows),
        },
    }


def _identity_pass_index(
    pass_results_by_name: dict[str, list[dict[str, Any]] | None],
) -> dict[int, str]:
    identity_pass: dict[int, str] = {}
    for pass_name, rows in pass_results_by_name.items():
        if pass_name not in PASS_NAMES:
            continue
        for row in rows or []:
            identity_pass[id(row)] = pass_name
    return identity_pass


def _row_pass_name(row: dict[str, Any], identity_pass: dict[int, str]) -> str | None:
    identity = identity_pass.get(id(row))
    if identity:
        return identity
    for key in ("pass_name", "ocr_pass", "source", "source_stage"):
        value = row.get(key)
        normalized = _normalize_pass_name(value)
        if normalized:
            return normalized
    return None


def _normalize_pass_name(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in PASS_NAMES:
        return text
    if text in SOURCE_TO_PASS:
        return SOURCE_TO_PASS[text]
    if text == "rotated_ocr":
        return None
    return SOURCE_TO_PASS.get(text.replace("-", "_"))


def _raw_count_for_pass(
    pass_name: str,
    *,
    pass_results_by_name: dict[str, list[dict[str, Any]] | None],
    ocr_breakdown: dict[str, Any],
) -> int:
    rows = pass_results_by_name.get(pass_name)
    if rows is not None:
        return len(rows)
    key = BREAKDOWN_KEYS.get(pass_name)
    return _safe_int(ocr_breakdown.get(key), default=0)


def _counter_int(counter: Any, key: str) -> int:
    if not isinstance(counter, dict):
        return 0
    return _safe_int(counter.get(key), default=0)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ocr_row_supports_dimension(
    row: dict[str, Any],
    dim: dict[str, Any],
    *,
    iou_threshold: float,
) -> bool:
    dim_bbox = _bbox_from_any(dim)
    row_bbox = _bbox_from_any(row)
    if not dim_bbox or not row_bbox:
        return False
    dim_text = _dimension_text(dim)
    row_text = _normalize_text(row.get("text"))
    if not dim_text or not row_text:
        return False
    text_match = _text_matches_dimension(dim_text, row_text)
    if not text_match:
        return False
    threshold = (
        max(iou_threshold, SHORT_TEXT_LINK_IOU)
        if min(len(dim_text), len(row_text)) <= 2
        else iou_threshold
    )
    return _bbox_iou(dim_bbox, row_bbox) >= threshold


def _text_matches_dimension(dim_text: str, row_text: str) -> bool:
    if dim_text == row_text:
        return True
    if min(len(dim_text), len(row_text)) <= 2:
        return False
    return _token_contains(row_text, dim_text) or _token_contains(dim_text, row_text)


def _token_contains(haystack: str, needle: str) -> bool:
    if not haystack or not needle or needle not in haystack:
        return False
    start = haystack.find(needle)
    end = start + len(needle)
    before = haystack[start - 1] if start > 0 else ""
    after = haystack[end] if end < len(haystack) else ""
    return not before.isdigit() and not after.isdigit()


def _dimension_has_vector_support(
    dim: dict[str, Any],
    vector_glyph_tokens: list[dict[str, Any]],
) -> bool:
    if not vector_glyph_tokens:
        return False
    dim_bbox = _bbox_from_any(dim)
    dim_text = _dimension_text(dim)
    if not dim_bbox or not dim_text:
        return False
    for token in vector_glyph_tokens:
        token_bbox = _bbox_from_any(token)
        token_text = _normalize_text(token.get("text"))
        glyph_type = str(token.get("glyph_type") or "")
        if not token_bbox:
            continue
        if _bbox_iou(dim_bbox, token_bbox) < VECTOR_SUPPORT_IOU and not _bbox_contains_center(
            dim_bbox,
            token_bbox,
        ):
            continue
        if token_text and token_text in dim_text:
            return True
        if glyph_type == "dot" and "." in dim_text:
            return True
        if glyph_type == "plus_minus" and "±" in dim_text:
            return True
        if glyph_type == "degree" and "°" in dim_text:
            return True
        if glyph_type == "diameter" and "⌀" in dim_text:
            return True
    return False


def _fallback_trigger(
    *,
    raw_count: int,
    dedup_survivor_count: int,
    claimed_dimension_count: int,
    exclusive_claimed_dimension_count: int,
    vector_replaced_count: int,
) -> str:
    if exclusive_claimed_dimension_count > 0:
        return "exclusive_claims_present"
    if claimed_dimension_count > 0 and vector_replaced_count < claimed_dimension_count:
        return "claimed_dimensions_without_vector_replacement"
    if claimed_dimension_count > 0:
        return "vector_replacement_candidate"
    if dedup_survivor_count > 0:
        return "unclaimed_survivors_present"
    if raw_count > 0:
        return "raw_rows_deduped_or_unclaimed"
    return "no_ocr_observed"


def _dimension_id(dim: dict[str, Any], idx: int) -> str:
    for key in ("dimension_id", "id", "candidate_id", "source_candidate_id"):
        value = dim.get(key)
        if value not in (None, ""):
            return str(value)
    return f"dim_{idx:06d}"


def _dimension_text(dim: dict[str, Any]) -> str:
    for key in ("text", "nominal", "value", "raw_text", "display_text"):
        value = _normalize_text(dim.get(key))
        if value:
            return value
    return ""


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = _DIAMETER_RE.sub("⌀", text)
    return _NON_TOKEN_RE.sub("", text)


def _bbox_from_any(item: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    nested = item.get("bbox")
    if isinstance(nested, dict):
        return _bbox_from_any(nested)
    nested = item.get("bbox_in_pdf")
    if isinstance(nested, dict):
        return _bbox_from_any(nested)
    try:
        if all(key in item for key in ("x", "y", "w", "h")):
            x = float(item.get("x") or 0.0)
            y = float(item.get("y") or 0.0)
            w = float(item.get("w") or 0.0)
            h = float(item.get("h") or 0.0)
            if w <= 0.0 or h <= 0.0:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
        if all(key in item for key in ("x0", "y0", "x1", "y1")):
            x0 = float(item.get("x0"))
            y0 = float(item.get("y0"))
            x1 = float(item.get("x1"))
            y1 = float(item.get("y1"))
            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            if w <= 0.0 or h <= 0.0:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
    except (TypeError, ValueError):
        return None
    return None


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1 = a["x"]
    ay1 = a["y"]
    ax2 = ax1 + a["w"]
    ay2 = ay1 + a["h"]
    bx1 = b["x"]
    by1 = b["y"]
    bx2 = bx1 + b["w"]
    by2 = by1 + b["h"]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0.0 else 0.0


def _bbox_contains_center(container: dict[str, float], item: dict[str, float]) -> bool:
    cx = item["x"] + item["w"] / 2.0
    cy = item["y"] + item["h"] / 2.0
    return (
        container["x"] <= cx <= container["x"] + container["w"]
        and container["y"] <= cy <= container["y"] + container["h"]
    )


def _schema_field_coverage(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    present = 0
    total = len(rows) * len(REQUIRED_ROW_FIELDS)
    for row in rows:
        present += sum(1 for key in REQUIRED_ROW_FIELDS if key in row)
    return round(present / total, 6) if total else 1.0
