"""Vector-only dimension hypothesis shadow layer.

This is the first bridge from R2a vector numeric phrases toward a full vector
recognition path. It binds vector phrase text to existing vector/geometric
candidate evidence, but still produces dump-only rows and never changes final
dimensions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


SCHEMA_VERSION = "vector_dimension_hypotheses_v1"
STATS_SCHEMA_VERSION = "vector_dimension_hypothesis_stats_v1"
CONSUMER_ALLOWED = False

_SUPPORTED_PHRASE_KINDS = frozenset({
    "plain_number",
    "nominal_plus_minus_tolerance",
    "plus_minus_tolerance",
    "diameter_number",
    "angle_number",
})

# Initial dump thresholds. These are intentionally permissive and must be
# calibrated by gold metrics before any consumer flag is introduced.
_MIN_CANDIDATE_IOU = 0.02
_MAX_CANDIDATE_CENTER_DISTANCE_PT = 28.0
_MIN_ANCHOR_IOU = 0.02
_MAX_ANCHOR_CENTER_DISTANCE_PT = 32.0


def build_vector_dimension_hypotheses_v1(
    *,
    vector_numeric_phrases: list[dict[str, Any]] | None,
    dimension_candidates: list[dict[str, Any]] | None = None,
    anchor_corridor_candidates: list[dict[str, Any]] | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidates = list(dimension_candidates or [])
    anchors = list(anchor_corridor_candidates or [])
    for phrase in vector_numeric_phrases or []:
        rows.append(_row_for_phrase(
            phrase=phrase,
            candidates=candidates,
            anchors=anchors,
            page_index=page_index,
            order=len(rows),
        ))

    active_count = sum(1 for row in rows if row.get("evidence_gate", {}).get("passed"))
    dropped = Counter(
        str(row.get("drop_reason") or "active")
        for row in rows
        if not row.get("evidence_gate", {}).get("passed")
    )
    kind_counts = Counter(str(row.get("phrase_kind") or "unknown") for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_dimension_hypotheses_v1": rows,
        "vector_dimension_hypothesis_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": int(page_index),
            "row_count": len(rows),
            "active_count": active_count,
            "drop_count": len(rows) - active_count,
            "counts_by_phrase_kind": dict(sorted(kind_counts.items())),
            "dropped_by_reason": dict(sorted(dropped.items())),
            "consumer_allowed_count": sum(
                1 for row in rows if row.get("consumer_allowed")
            ),
        },
    }


def _row_for_phrase(
    *,
    phrase: dict[str, Any],
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    page_index: int,
    order: int,
) -> dict[str, Any]:
    phrase_kind = str(phrase.get("phrase_kind") or "")
    phrase_bbox = _bbox_from_any(phrase)
    phrase_ok = (
        str(phrase.get("schema_version") or "") == "vector_numeric_phrases_v1"
        and phrase_kind in _SUPPORTED_PHRASE_KINDS
        and phrase_bbox is not None
        and not bool(phrase.get("consumer_allowed"))
    )
    candidate_binding = _best_candidate_binding(phrase_bbox, candidates) if phrase_bbox else None
    anchor_binding = _best_anchor_binding(phrase_bbox, anchors) if phrase_bbox else None
    geometry_bound = candidate_binding is not None or anchor_binding is not None

    if bool(phrase.get("consumer_allowed")):
        drop_reason = "input_consumer_allowed"
    elif phrase_bbox is None:
        drop_reason = "invalid_phrase_bbox"
    elif phrase_kind not in _SUPPORTED_PHRASE_KINDS:
        drop_reason = "unsupported_phrase_kind"
    elif not geometry_bound:
        drop_reason = "no_geometry_binding"
    else:
        drop_reason = None
    passed = drop_reason is None and phrase_ok
    confidence_values = [_float_or(phrase.get("confidence"), 0.0)]
    if candidate_binding:
        confidence_values.append(float(candidate_binding["score"]))
    if anchor_binding:
        confidence_values.append(float(anchor_binding["score"]))
    confidence = min(confidence_values) if confidence_values else 0.0
    geometry_bbox = _geometry_bbox(candidate_binding, anchor_binding)
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": f"p{int(page_index) + 1:03d}_vdh_{int(order):06d}",
        "page_index": int(page_index),
        "text": str(phrase.get("text") or ""),
        "phrase_kind": phrase_kind,
        "bbox": _round_bbox(phrase_bbox) if phrase_bbox else None,
        "oriented_quad": _quad_from_bbox(phrase_bbox) if phrase_bbox else None,
        "text_bbox": _round_bbox(phrase_bbox) if phrase_bbox else None,
        "geometry_bbox": _round_bbox(geometry_bbox) if geometry_bbox else None,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "source_phrase_id": str(phrase.get("phrase_id") or ""),
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
                "phrase_schema_ok": str(phrase.get("schema_version") or "")
                == "vector_numeric_phrases_v1",
                "phrase_kind_supported": phrase_kind in _SUPPORTED_PHRASE_KINDS,
                "phrase_bbox_valid": phrase_bbox is not None,
                "phrase_consumer_allowed_false": not bool(phrase.get("consumer_allowed")),
                "geometry_bound": geometry_bound,
                "candidate_bound": candidate_binding is not None,
                "anchor_bound": anchor_binding is not None,
                "no_ocr_source": _no_ocr_source(phrase),
                "consumer_allowed": CONSUMER_ALLOWED,
            },
        },
        "source_segments": [{
            "kind": "vector_numeric_phrase",
            "phrase_id": str(phrase.get("phrase_id") or ""),
            "source_match_ids": list(phrase.get("source_match_ids") or []),
            "source_token_ids": list(phrase.get("source_token_ids") or []),
        }],
        "drop_reason": drop_reason,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _best_candidate_binding(
    phrase_bbox: dict[str, float],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for candidate in candidates:
        if str(candidate.get("type") or "") != "dimension":
            continue
        if candidate.get("reject_reason"):
            continue
        bbox = _bbox_from_any(candidate)
        if bbox is None:
            continue
        iou = _iou(phrase_bbox, bbox)
        dist = _center_distance(phrase_bbox, bbox)
        if iou < _MIN_CANDIDATE_IOU and dist > _MAX_CANDIDATE_CENTER_DISTANCE_PT:
            continue
        priority_boost = {
            "high": 0.10,
            "normal": 0.04,
            "low": 0.0,
        }.get(str(candidate.get("priority_hint") or "normal"), 0.0)
        score = min(1.0, iou + max(0.0, 1.0 - dist / _MAX_CANDIDATE_CENTER_DISTANCE_PT) * 0.35 + priority_boost)
        item = {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "subtype": str(candidate.get("subtype") or ""),
            "priority_hint": str(candidate.get("priority_hint") or ""),
            "bbox": _round_bbox(bbox),
            "iou": round(iou, 4),
            "center_distance_pt": round(dist, 3),
            "score": round(score, 4),
        }
        if best is None or (item["score"], item["iou"]) > (best["score"], best["iou"]):
            best = item
    return best


def _best_anchor_binding(
    phrase_bbox: dict[str, float],
    anchors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for anchor in anchors:
        gate = anchor.get("evidence_gate") or {}
        if gate and gate.get("passed") is False:
            continue
        bbox = _bbox_from_any(anchor.get("corridor_bbox") or anchor)
        if bbox is None:
            continue
        iou = _iou(phrase_bbox, bbox)
        dist = _center_distance(phrase_bbox, bbox)
        if iou < _MIN_ANCHOR_IOU and dist > _MAX_ANCHOR_CENTER_DISTANCE_PT:
            continue
        score = min(1.0, iou + max(0.0, 1.0 - dist / _MAX_ANCHOR_CENTER_DISTANCE_PT) * 0.30)
        item = {
            "candidate_id": str(anchor.get("candidate_id") or ""),
            "anchor_type": str(anchor.get("anchor_type") or ""),
            "bbox": _round_bbox(bbox),
            "iou": round(iou, 4),
            "center_distance_pt": round(dist, 3),
            "score": round(score, 4),
        }
        if best is None or (item["score"], item["iou"]) > (best["score"], best["iou"]):
            best = item
    return best


def _geometry_bbox(
    candidate_binding: dict[str, Any] | None,
    anchor_binding: dict[str, Any] | None,
) -> dict[str, float] | None:
    bbox = None
    if candidate_binding:
        bbox = _bbox_from_any(candidate_binding.get("bbox"))
    if anchor_binding:
        anchor_bbox = _bbox_from_any(anchor_binding.get("bbox"))
        bbox = _union_bbox(bbox, anchor_bbox) if bbox and anchor_bbox else bbox or anchor_bbox
    return bbox


def _no_ocr_source(phrase: dict[str, Any]) -> bool:
    for segment in phrase.get("source_segments") or []:
        kind = str((segment or {}).get("kind") or "")
        if kind not in {"vector_digit_match", "vector_glyph_token"}:
            return False
    return True


def _bbox_from_any(item: Any) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    if "bbox" in item and isinstance(item.get("bbox"), dict):
        return _bbox_from_any(item["bbox"])
    if {"x", "y", "w", "h"} <= set(item):
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
            w = float(item.get("w"))
            h = float(item.get("h"))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x, y, w, h)):
            return None
        if w <= 0.0 or h <= 0.0:
            return None
        return {"x": x, "y": y, "w": w, "h": h}
    return None


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1 = a["x"] + a["w"]
    ay1 = a["y"] + a["h"]
    bx1 = b["x"] + b["w"]
    by1 = b["y"] + b["h"]
    ix0 = max(a["x"], b["x"])
    iy0 = max(a["y"], b["y"])
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0.0 else 0.0


def _center_distance(a: dict[str, float], b: dict[str, float]) -> float:
    ax = a["x"] + a["w"] / 2.0
    ay = a["y"] + a["h"] / 2.0
    bx = b["x"] + b["w"] / 2.0
    by = b["y"] + b["h"] / 2.0
    return math.hypot(ax - bx, ay - by)


def _union_bbox(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    x0 = min(a["x"], b["x"])
    y0 = min(a["y"], b["y"])
    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
    y1 = max(a["y"] + a["h"], b["y"] + b["h"])
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _quad_from_bbox(bbox: dict[str, float] | None) -> list[list[float]] | None:
    if bbox is None:
        return None
    x0 = float(bbox["x"])
    y0 = float(bbox["y"])
    x1 = x0 + float(bbox["w"])
    y1 = y0 + float(bbox["h"])
    return [
        [round(x0, 3), round(y0, 3)],
        [round(x1, 3), round(y0, 3)],
        [round(x1, 3), round(y1, 3)],
        [round(x0, 3), round(y1, 3)],
    ]


def _round_bbox(bbox: dict[str, float] | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {key: round(float(bbox[key]), 3) for key in ("x", "y", "w", "h")}


def _float_or(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
