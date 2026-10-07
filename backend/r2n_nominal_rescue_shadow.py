"""R2n nominal-rescue shadow rows.

This module turns vector numeric phrase evidence into auditable rows for the
integer/nominal rescue path. It is dump-only: rows are never allowed to feed
OCR skip, assembler decisions, or final dimensions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


SCHEMA_VERSION = "r2n_nominal_rescue_shadow_v1"
STATS_SCHEMA_VERSION = "r2n_nominal_rescue_shadow_stats_v1"
CONSUMER_ALLOWED = False
FINAL_CONSUME_ALLOWED = False

_SUPPORTED_INLINE_KINDS = frozenset({
    "nominal_plus_minus_tolerance",
    "diameter_number",
})
_SUPPORTED_PAIRED_NOMINAL_KINDS = frozenset({
    "plain_number",
    "diameter_number",
})
_TOLERANCE_KIND = "plus_minus_tolerance"
_NOMINAL_RE = re.compile(
    r"^(?:[\u2300\u2205\u00d8\u00f8\u03a6\u03c6R])?\d+(?:\.\d+)?$"
)
_MIN_CANDIDATE_IOU = 0.02
_MAX_CANDIDATE_CENTER_DISTANCE_PT = 32.0
_MIN_ANCHOR_IOU = 0.02
_MAX_ANCHOR_CENTER_DISTANCE_PT = 36.0
_MAX_PAIRED_GAP_PT = 48.0
_MAX_PAIRED_GAP_FACTOR = 5.0


def build_r2n_nominal_rescue_shadow_v1(
    *,
    vector_numeric_phrases: list[dict[str, Any]] | None = None,
    vector_numeric_phrase_stats: dict[str, Any] | None = None,
    dimension_candidates: list[dict[str, Any]] | None = None,
    anchor_corridor_candidates: list[dict[str, Any]] | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    """Build default-off shadow rows from vector numeric phrase evidence."""
    phrases = [
        phrase for phrase in (vector_numeric_phrases or [])
        if _bbox_from_any(phrase) is not None
        and int(phrase.get("page_index", page_index)) == int(page_index)
    ]
    candidates = list(dimension_candidates or [])
    anchors = list(anchor_corridor_candidates or [])

    rows: list[dict[str, Any]] = []
    for phrase in phrases:
        nominal = _inline_nominal(phrase)
        if not nominal:
            continue
        rows.append(_row_from_parts(
            page_index=page_index,
            order=len(rows),
            source_pattern="inline_plus_minus_tolerance",
            nominal_text=nominal,
            tolerance_text=_inline_tolerance_text(phrase),
            phrases=[phrase],
            candidates=candidates,
            anchors=anchors,
        ))

    for tolerance_phrase in phrases:
        if str(tolerance_phrase.get("phrase_kind") or "") != _TOLERANCE_KIND:
            continue
        nominal_phrase = _best_left_nominal_phrase(tolerance_phrase, phrases)
        rows.append(_row_from_parts(
            page_index=page_index,
            order=len(rows),
            source_pattern="paired_plus_minus_tolerance",
            nominal_text=str(nominal_phrase.get("text") or "") if nominal_phrase else "",
            tolerance_text=str(tolerance_phrase.get("text") or ""),
            phrases=(
                [nominal_phrase, tolerance_phrase]
                if nominal_phrase else [tolerance_phrase]
            ),
            candidates=candidates,
            anchors=anchors,
            forced_drop_reason=None if nominal_phrase else "no_left_nominal_phrase",
        ))

    stats = _build_stats(
        rows,
        phrases=phrases,
        upstream_phrase_stats=vector_numeric_phrase_stats,
        candidates=candidates,
        anchors=anchors,
        page_index=page_index,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "final_consume_allowed": FINAL_CONSUME_ALLOWED,
        "r2n_nominal_rescue_shadow_v1": rows,
        "r2n_nominal_rescue_shadow_stats_v1": stats,
    }


def _row_from_parts(
    *,
    page_index: int,
    order: int,
    source_pattern: str,
    nominal_text: str,
    tolerance_text: str,
    phrases: list[dict[str, Any] | None],
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    forced_drop_reason: str | None = None,
) -> dict[str, Any]:
    source_phrases = [phrase for phrase in phrases if isinstance(phrase, dict)]
    bbox = _union_many(_bbox_from_any(phrase) for phrase in source_phrases)
    nominal_bbox = _bbox_from_any(source_phrases[0]) if source_phrases else None
    tolerance_bbox = _bbox_from_any(source_phrases[-1]) if source_phrases else None
    candidate_binding = _best_candidate_binding(bbox, candidates) if bbox else None
    anchor_binding = _best_anchor_binding(bbox, anchors) if bbox else None
    geometry_bound = candidate_binding is not None or anchor_binding is not None
    source_allowed = all(not bool(phrase.get("consumer_allowed")) for phrase in source_phrases)
    nominal_ok = bool(nominal_text) and _nominal_looks_valid(nominal_text)
    tolerance_ok = bool(tolerance_text)
    if forced_drop_reason:
        drop_reason = forced_drop_reason
    elif not source_allowed:
        drop_reason = "input_consumer_allowed"
    elif not nominal_ok:
        drop_reason = "invalid_nominal_text"
    elif not tolerance_ok:
        drop_reason = "missing_tolerance_text"
    elif not geometry_bound:
        drop_reason = "no_geometry_binding"
    else:
        drop_reason = None

    confidence_values = [
        _float_or(phrase.get("confidence"), 0.0)
        for phrase in source_phrases
    ]
    if candidate_binding:
        confidence_values.append(float(candidate_binding["score"]))
    if anchor_binding:
        confidence_values.append(float(anchor_binding["score"]))
    confidence = min(confidence_values) if confidence_values else 0.0
    passed = drop_reason is None
    return {
        "schema_version": SCHEMA_VERSION,
        "rescue_id": f"p{int(page_index) + 1:03d}_r2n_nominal_{int(order):06d}",
        "page_index": int(page_index),
        "source_pattern": source_pattern,
        "nominal_text": str(nominal_text or ""),
        "tolerance_text": str(tolerance_text or ""),
        "full_text": _full_text(nominal_text, tolerance_text),
        "bbox": _round_bbox(bbox),
        "nominal_bbox": _round_bbox(nominal_bbox),
        "tolerance_bbox": _round_bbox(tolerance_bbox),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "source_phrase_ids": [
            str(phrase.get("phrase_id") or "") for phrase in source_phrases
        ],
        "source_candidate_ids": (
            [candidate_binding["candidate_id"]] if candidate_binding else []
        ),
        "source_anchor_ids": (
            [anchor_binding["candidate_id"]] if anchor_binding else []
        ),
        "geometry_binding": {
            "candidate": candidate_binding,
            "anchor_corridor": anchor_binding,
        },
        "evidence_gate": {
            "passed": passed,
            "score": round(confidence, 4),
            "checks": {
                "source_phrase_schema_ok": all(
                    str(phrase.get("schema_version") or "")
                    == "vector_numeric_phrases_v1"
                    for phrase in source_phrases
                ),
                "source_phrase_orientation_horizontal": all(
                    str(phrase.get("orientation") or "H").upper() == "H"
                    for phrase in source_phrases
                ),
                "source_phrase_consumer_allowed_false": source_allowed,
                "nominal_present": bool(nominal_text),
                "nominal_text_valid": nominal_ok,
                "tolerance_present": tolerance_ok,
                "geometry_bound": geometry_bound,
                "candidate_bound": candidate_binding is not None,
                "anchor_bound": anchor_binding is not None,
                "consumer_allowed": CONSUMER_ALLOWED,
                "final_consume_allowed": FINAL_CONSUME_ALLOWED,
            },
        },
        "source_segments": [
            {
                "kind": "vector_numeric_phrase",
                "phrase_id": str(phrase.get("phrase_id") or ""),
                "phrase_kind": str(phrase.get("phrase_kind") or ""),
                "text": str(phrase.get("text") or ""),
            }
            for phrase in source_phrases
        ],
        "drop_reason": drop_reason,
        "consumer_allowed": CONSUMER_ALLOWED,
        "final_consume_allowed": FINAL_CONSUME_ALLOWED,
    }


def _inline_nominal(phrase: dict[str, Any]) -> str:
    kind = str(phrase.get("phrase_kind") or "")
    text = str(phrase.get("text") or "").strip()
    if kind not in _SUPPORTED_INLINE_KINDS or "\u00b1" not in text:
        return ""
    nominal = text.split("\u00b1", 1)[0].strip()
    return nominal if _nominal_looks_valid(nominal) else ""


def _inline_tolerance_text(phrase: dict[str, Any]) -> str:
    text = str(phrase.get("text") or "").strip()
    if "\u00b1" not in text:
        return ""
    return "\u00b1" + text.split("\u00b1", 1)[1].strip()


def _best_left_nominal_phrase(
    tolerance_phrase: dict[str, Any],
    phrases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    tol_bbox = _bbox_from_any(tolerance_phrase)
    if tol_bbox is None:
        return None
    tol_cx, _ = _bbox_center(tol_bbox)
    best: tuple[float, dict[str, Any]] | None = None
    for phrase in phrases:
        if phrase is tolerance_phrase:
            continue
        if str(phrase.get("phrase_kind") or "") not in _SUPPORTED_PAIRED_NOMINAL_KINDS:
            continue
        text = str(phrase.get("text") or "").strip()
        if "\u00b1" in text or not _nominal_looks_valid(text):
            continue
        bbox = _bbox_from_any(phrase)
        if bbox is None:
            continue
        cx, _ = _bbox_center(bbox)
        if cx >= tol_cx:
            continue
        if not _same_text_line(bbox, tol_bbox):
            continue
        gap = tol_bbox["x"] - (bbox["x"] + bbox["w"])
        max_gap = min(
            _MAX_PAIRED_GAP_PT,
            max(bbox["h"], tol_bbox["h"]) * _MAX_PAIRED_GAP_FACTOR,
        )
        if gap > max_gap:
            continue
        score = max(0.0, 1.0 - max(gap, 0.0) / max(max_gap, 1e-6))
        score += _vertical_overlap_ratio(bbox, tol_bbox)
        item = (score, phrase)
        if best is None or item[0] > best[0]:
            best = item
    return best[1] if best else None


def _best_candidate_binding(
    bbox: dict[str, float] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if bbox is None:
        return None
    best: dict[str, Any] | None = None
    for candidate in candidates:
        if str(candidate.get("type") or "dimension") != "dimension":
            continue
        if candidate.get("reject_reason"):
            continue
        candidate_bbox = _bbox_from_any(candidate)
        if candidate_bbox is None:
            continue
        iou = _iou(bbox, candidate_bbox)
        dist = _center_distance(bbox, candidate_bbox)
        if iou < _MIN_CANDIDATE_IOU and dist > _MAX_CANDIDATE_CENTER_DISTANCE_PT:
            continue
        priority_boost = {
            "high": 0.10,
            "normal": 0.04,
            "low": 0.0,
        }.get(str(candidate.get("priority_hint") or "normal"), 0.0)
        score = min(
            1.0,
            iou
            + max(0.0, 1.0 - dist / _MAX_CANDIDATE_CENTER_DISTANCE_PT) * 0.35
            + priority_boost,
        )
        item = {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "subtype": str(candidate.get("subtype") or ""),
            "priority_hint": str(candidate.get("priority_hint") or ""),
            "bbox": _round_bbox(candidate_bbox),
            "iou": round(iou, 4),
            "center_distance_pt": round(dist, 3),
            "score": round(score, 4),
        }
        if best is None or (item["score"], item["iou"]) > (best["score"], best["iou"]):
            best = item
    return best


def _best_anchor_binding(
    bbox: dict[str, float] | None,
    anchors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if bbox is None:
        return None
    best: dict[str, Any] | None = None
    for anchor in anchors:
        gate = anchor.get("evidence_gate") or {}
        if gate and gate.get("passed") is False:
            continue
        anchor_bbox = _bbox_from_any(anchor.get("corridor_bbox") or anchor)
        if anchor_bbox is None:
            continue
        iou = _iou(bbox, anchor_bbox)
        dist = _center_distance(bbox, anchor_bbox)
        if iou < _MIN_ANCHOR_IOU and dist > _MAX_ANCHOR_CENTER_DISTANCE_PT:
            continue
        score = min(1.0, iou + max(0.0, 1.0 - dist / _MAX_ANCHOR_CENTER_DISTANCE_PT) * 0.30)
        item = {
            "candidate_id": str(anchor.get("candidate_id") or ""),
            "anchor_type": str(anchor.get("anchor_type") or ""),
            "bbox": _round_bbox(anchor_bbox),
            "iou": round(iou, 4),
            "center_distance_pt": round(dist, 3),
            "score": round(score, 4),
        }
        if best is None or (item["score"], item["iou"]) > (best["score"], best["iou"]):
            best = item
    return best


def _build_stats(
    rows: list[dict[str, Any]],
    *,
    phrases: list[dict[str, Any]],
    upstream_phrase_stats: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    page_index: int,
) -> dict[str, Any]:
    dropped = [
        row for row in rows
        if not (row.get("evidence_gate") or {}).get("passed")
    ]
    phrase_kind_counts = Counter(
        str(phrase.get("phrase_kind") or "unknown") for phrase in phrases
    )
    supported_inline_count = sum(
        1 for phrase in phrases if _inline_nominal(phrase)
    )
    supported_tolerance_count = sum(
        1
        for phrase in phrases
        if str(phrase.get("phrase_kind") or "") == _TOLERANCE_KIND
    )
    supported_paired_nominal_count = sum(
        1
        for phrase in phrases
        if str(phrase.get("phrase_kind") or "") in _SUPPORTED_PAIRED_NOMINAL_KINDS
        and _nominal_looks_valid(str(phrase.get("text") or ""))
    )
    zero_row_reason = ""
    if not rows:
        zero_row_reason = (
            "no_vector_numeric_phrases"
            if not phrases
            else "no_supported_tolerance_or_inline_phrases"
        )
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "page_index": int(page_index),
        "row_count": len(rows),
        "active_count": len(rows) - len(dropped),
        "drop_count": len(dropped),
        "zero_row_reason": zero_row_reason,
        "input_phrase_count": len(phrases),
        "input_phrase_kind_counts": dict(sorted(phrase_kind_counts.items())),
        "input_consumer_allowed_count": sum(
            1 for phrase in phrases if phrase.get("consumer_allowed")
        ),
        "supported_inline_phrase_count": supported_inline_count,
        "supported_tolerance_phrase_count": supported_tolerance_count,
        "supported_paired_nominal_phrase_count": supported_paired_nominal_count,
        "dimension_candidate_count": len(candidates),
        "anchor_corridor_candidate_count": len(anchors),
        "upstream_vector_numeric_phrase_stats": _safe_upstream_phrase_stats(
            upstream_phrase_stats,
        ),
        "counts_by_source_pattern": dict(sorted(Counter(
            str(row.get("source_pattern") or "unknown") for row in rows
        ).items())),
        "drop_reason_counts": dict(sorted(Counter(
            str(row.get("drop_reason") or "") for row in dropped
        ).items())),
        "consumer_allowed_count": sum(
            1 for row in rows if row.get("consumer_allowed")
        ),
        "final_consume_allowed_count": sum(
            1 for row in rows if row.get("final_consume_allowed")
        ),
        "shadow_dump_ready": True,
        "consumer_allowed": CONSUMER_ALLOWED,
        "final_consume_allowed": FINAL_CONSUME_ALLOWED,
    }


def _safe_upstream_phrase_stats(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    keys = (
        "schema_version",
        "page_index",
        "row_count",
        "character_count",
        "input_digit_match_count",
        "accepted_digit_count",
        "symbol_char_count",
        "segment_count",
        "rejected_segment_count",
        "segment_rescue_count",
        "counts_by_phrase_kind",
        "counts_by_segment_rescue_kind",
        "counts_by_char_kind",
        "counts_by_accepted_digit_decision",
        "counts_by_accepted_digit_source",
        "counts_by_input_digit_decision",
        "counts_by_input_digit_source",
        "counts_by_symbol_glyph_type",
        "counts_by_rejected_segment_reason",
        "symbol_adjacency_by_type",
        "consumer_allowed_count",
    )
    return {key: raw[key] for key in keys if key in raw}


def _nominal_looks_valid(text: str) -> bool:
    return bool(_NOMINAL_RE.match(str(text or "").strip()))


def _full_text(nominal_text: str, tolerance_text: str) -> str:
    return f"{nominal_text}{tolerance_text}" if tolerance_text else str(nominal_text or "")


def _same_text_line(a: dict[str, float], b: dict[str, float]) -> bool:
    if _vertical_overlap_ratio(a, b) >= 0.30:
        return True
    return abs(_bbox_center(a)[1] - _bbox_center(b)[1]) <= max(a["h"], b["h"]) * 0.75


def _vertical_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    ay0 = float(a["y"])
    ay1 = ay0 + float(a["h"])
    by0 = float(b["y"])
    by1 = by0 + float(b["h"])
    overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
    return overlap / max(1e-6, min(float(a["h"]), float(b["h"])))


def _union_many(boxes: Any) -> dict[str, float] | None:
    out = None
    for bbox in boxes:
        if bbox is None:
            continue
        out = _union_bbox(out, bbox) if out else dict(bbox)
    return out


def _union_bbox(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    x0 = min(a["x"], b["x"])
    y0 = min(a["y"], b["y"])
    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
    y1 = max(a["y"] + a["h"], b["y"] + b["h"])
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _bbox_from_any(item: Any) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("bbox"), dict):
        return _bbox_from_any(item["bbox"])
    try:
        if all(key in item for key in ("x", "y", "w", "h")):
            x = float(item["x"])
            y = float(item["y"])
            w = float(item["w"])
            h = float(item["h"])
            if w <= 0.0 or h <= 0.0:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
        if all(key in item for key in ("x0", "y0", "x1", "y1")):
            return _bbox_from_xyxy(
                float(item["x0"]),
                float(item["y0"]),
                float(item["x1"]),
                float(item["y1"]),
            )
        if all(key in item for key in ("x0", "top", "x1", "bottom")):
            return _bbox_from_xyxy(
                float(item["x0"]),
                float(item["top"]),
                float(item["x1"]),
                float(item["bottom"]),
            )
    except (TypeError, ValueError):
        return None
    return None


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> dict[str, float] | None:
    x_min = min(x0, x1)
    y_min = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x_min, "y": y_min, "w": w, "h": h}


def _round_bbox(bbox: dict[str, float] | None) -> dict[str, float]:
    if not bbox:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    return {
        "x": round(float(bbox["x"]), 4),
        "y": round(float(bbox["y"]), 4),
        "w": round(float(bbox["w"]), 4),
        "h": round(float(bbox["h"]), 4),
    }


def _bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return (
        float(bbox["x"]) + float(bbox["w"]) / 2.0,
        float(bbox["y"]) + float(bbox["h"]) / 2.0,
    )


def _center_distance(a: dict[str, float], b: dict[str, float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax0 = float(a["x"])
    ay0 = float(a["y"])
    ax1 = ax0 + float(a["w"])
    ay1 = ay0 + float(a["h"])
    bx0 = float(b["x"])
    by0 = float(b["y"])
    bx1 = bx0 + float(b["w"])
    by1 = by0 + float(b["h"])
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a["w"]) * float(a["h"]))
    area_b = max(0.0, float(b["w"]) * float(b["h"]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _float_or(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)
