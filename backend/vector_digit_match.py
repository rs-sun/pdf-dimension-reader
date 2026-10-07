"""Dump-only vector digit match rows.

This module is the stable production-facing shell for R2a digit recognition.
Prototype matching can later fill the same row schema, but the current version
only separates exact native PDF digit glyphs from unresolved digit slots.
"""

from __future__ import annotations

import math
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vector_digit_matches_v1"
STATS_SCHEMA_VERSION = "vector_digit_match_stats_v1"
CONSUMER_ALLOWED = False
DESCRIPTOR_SCHEMA_VERSION = "vector_digit_slot_descriptor_v2"
PROTOTYPE_LIBRARY_SCHEMA_VERSION = "vector_digit_prototype_library_v1"
GRID_SIZE = 4
DEFAULT_PROTOTYPE_THRESHOLD = 0.20
DEFAULT_PROTOTYPE_MARGIN = 0.03
PLUS_MINUS_ANCHORED_SLOT_SOURCE = "vector_text_component_plus_minus_anchored_slot"
STRUCTURAL_DIGIT_DECISION = "accepted_structural"
PLUS_MINUS_STRUCTURAL_TOP1_LABELS = frozenset({"1", "2"})
PLUS_MINUS_STRUCTURAL_MAX_DISTANCE = 0.32
PLUS_MINUS_STRUCTURAL_MIN_MARGIN = DEFAULT_PROTOTYPE_MARGIN
PRIMITIVE_INDEX_CELL_PT = 24.0
DESCRIPTOR_FEATURE_FIELDS = (
    "bbox_coverage_ratio",
    "covered_width_ratio",
    "covered_height_ratio",
    "stroke_width_norm",
    "slot_aspect_log",
    "primitive_count_norm",
    "point_count_norm",
)
STROKE_WIDTH_FEATURE_INDEX = (
    GRID_SIZE * GRID_SIZE + DESCRIPTOR_FEATURE_FIELDS.index("stroke_width_norm")
)
ABLATE_STROKE_WIDTH_ENV = "VECTOR_DIGIT_DESCRIPTOR_ABLATE_STROKE_WIDTH"
STROKE_WIDTH_MODE_ENV = "VECTOR_DIGIT_DESCRIPTOR_STROKE_WIDTH_MODE"
STROKE_WIDTH_MODE_RAW = "raw"
STROKE_WIDTH_MODE_SIZE_RELATIVE = "size_relative"
_PRIVATE_STROKE_WIDTH_MODE_KEY = "_stroke_width_feature_mode"


def build_vector_digit_matches_v1(
    *,
    vector_glyph_tokens: list[dict[str, Any]] | None,
    page: Any | None = None,
    page_index: int = 0,
    prototype_library: dict[str, Any] | None = None,
    prototype_threshold: float = DEFAULT_PROTOTYPE_THRESHOLD,
    prototype_margin: float = DEFAULT_PROTOTYPE_MARGIN,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    normalized_library = normalize_vector_digit_prototype_library(prototype_library)
    primitive_index = _page_primitive_index(page)
    for token in vector_glyph_tokens or []:
        glyph_type = str(token.get("glyph_type") or "")
        if glyph_type == "digit":
            row = _row_from_native_digit(token, page_index=page_index, order=len(rows))
        elif glyph_type == "digit_unknown":
            row = _row_from_digit_slot(
                token,
                page=page,
                primitive_index=primitive_index,
                page_index=page_index,
                order=len(rows),
                prototype_library=normalized_library,
                prototype_threshold=prototype_threshold,
                prototype_margin=prototype_margin,
            )
        else:
            continue
        if row is not None:
            rows.append(row)

    decision_counts = Counter(str(row.get("decision") or "unknown") for row in rows)
    kind_counts = Counter(str(row.get("match_kind") or "unknown") for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_digit_matches_v1": rows,
        "vector_digit_match_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": int(page_index),
            "row_count": len(rows),
            "counts_by_decision": dict(sorted(decision_counts.items())),
            "counts_by_match_kind": dict(sorted(kind_counts.items())),
            "accepted_exact_count": decision_counts.get("accepted_exact", 0),
            "accepted_prototype_count": decision_counts.get("accepted_prototype", 0),
            "prototype_not_run_count": decision_counts.get("prototype_not_run", 0),
            "unknown_count": decision_counts.get("unknown", 0),
            "low_margin_count": decision_counts.get("low_margin", 0),
            "prototype_library": normalized_library["debug"],
            "consumer_allowed_count": sum(
                1 for row in rows if row.get("consumer_allowed")
            ),
        },
    }


def load_vector_digit_prototype_library(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return normalize_vector_digit_prototype_library(payload)


def normalize_vector_digit_prototype_library(
    payload: dict[str, Any] | list[Any] | None,
) -> dict[str, Any]:
    if payload is None:
        return _empty_library("not_configured")
    if isinstance(payload, dict) and payload.get("normalized") is True:
        return payload
    rows: Any
    if isinstance(payload, dict):
        rows = payload.get("prototypes", payload.get("rows", []))
    else:
        rows = payload
    if not isinstance(rows, list):
        return _empty_library("invalid_payload")

    labels: dict[str, list[dict[str, Any]]] = {}
    skipped: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            skipped["non_object"] += 1
            continue
        label = str(
            row.get("label")
            or row.get("digit")
            or row.get("text")
            or ""
        )
        if len(label) != 1 or not label.isdigit():
            skipped["invalid_label"] += 1
            continue
        descriptor = row.get("descriptor") or row.get("geometry_descriptor") or row
        feature = _descriptor_feature(descriptor)
        if feature is None:
            skipped["invalid_descriptor"] += 1
            continue
        labels.setdefault(label, []).append({
            "label": label,
            "prototype_id": str(row.get("prototype_id") or row.get("id") or f"proto_{idx:06d}"),
            "feature": feature,
        })

    sample_count = sum(len(samples) for samples in labels.values())
    status = "loaded" if sample_count else ("invalid" if skipped else "empty")
    return {
        "normalized": True,
        "schema_version": PROTOTYPE_LIBRARY_SCHEMA_VERSION,
        "labels": labels,
        "debug": {
            "status": status,
            "label_count": len(labels),
            "sample_count": sample_count,
            "samples_per_label": {
                label: len(samples)
                for label, samples in sorted(labels.items())
            },
            "skipped": dict(sorted(skipped.items())),
        },
    }


def _empty_library(status: str) -> dict[str, Any]:
    return {
        "normalized": True,
        "schema_version": PROTOTYPE_LIBRARY_SCHEMA_VERSION,
        "labels": {},
        "debug": {
            "status": status,
            "label_count": 0,
            "sample_count": 0,
            "samples_per_label": {},
            "skipped": {},
        },
    }


def _row_from_native_digit(
    token: dict[str, Any],
    *,
    page_index: int,
    order: int,
) -> dict[str, Any] | None:
    text = str(token.get("text") or "")
    if len(text) != 1 or not text.isdigit():
        return None
    return _make_row(
        token,
        page_index=page_index,
        order=order,
        match_kind="native_pdf_digit",
        decision="accepted_exact",
        predicted_text=text,
        confidence=_float_or(token.get("confidence"), 1.0),
        strict_checks={
            "source": "vector_glyph_digit",
            "native_pdf_digit": True,
            "prototype_match_status": "not_required_native_digit",
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    )


def _row_from_digit_slot(
    token: dict[str, Any],
    *,
    page: Any | None,
    primitive_index: dict[str, Any] | None,
    page_index: int,
    order: int,
    prototype_library: dict[str, Any],
    prototype_threshold: float,
    prototype_margin: float,
) -> dict[str, Any] | None:
    checks = token.get("strict_checks") if isinstance(token.get("strict_checks"), dict) else {}
    descriptor = _slot_descriptor_for_token(page, token, primitive_index=primitive_index)
    prototype_match = _classify_with_prototypes(
        descriptor,
        prototype_library=prototype_library,
        threshold=prototype_threshold,
        margin=prototype_margin,
    )
    structural_rescue = _plus_minus_anchored_structural_rescue(
        token,
        descriptor,
        prototype_match,
    )
    row_decision = prototype_match["decision"]
    predicted_text = prototype_match["predicted_text"]
    confidence = prototype_match["confidence"]
    strict_extra: dict[str, Any] = {}
    if structural_rescue is not None:
        row_decision = structural_rescue["decision"]
        predicted_text = structural_rescue["predicted_text"]
        confidence = structural_rescue["confidence"]
        strict_extra = structural_rescue["strict_checks"]
    strict_checks = {
        "source": "vector_glyph_digit_unknown",
        "slot_source": checks.get("source"),
        "subtype": checks.get("subtype"),
        "orientation": checks.get("orientation"),
        "slot_role": checks.get("slot_role"),
        "prototype_match_status": prototype_match["status"],
        "prototype_threshold": prototype_threshold,
        "prototype_margin": prototype_margin,
        "slot_geometry_status": descriptor["status"],
        "slot_vector_primitive_count": descriptor["primitive_count"],
        "slot_descriptor_schema": descriptor["schema_version"],
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    strict_checks.update(strict_extra)
    return _make_row(
        token,
        page_index=page_index,
        order=order,
        match_kind="prototype_slot",
        decision=row_decision,
        predicted_text=predicted_text,
        confidence=confidence,
        strict_checks=strict_checks,
        geometry_descriptor=_public_descriptor(descriptor),
        prototype_match=prototype_match,
    )


def _plus_minus_anchored_structural_rescue(
    token: dict[str, Any],
    descriptor: dict[str, Any],
    prototype_match: dict[str, Any],
) -> dict[str, Any] | None:
    if not _plus_minus_anchored_structural_context_allowed(token):
        return None
    if str(prototype_match.get("decision") or "") == "accepted_prototype":
        return None

    prototype_label = _plus_minus_anchored_prototype_rescue_label(
        prototype_match,
    )
    if prototype_label:
        distance = _float_or(prototype_match.get("top1_distance"), 0.0)
        confidence = 1.0 - distance / max(PLUS_MINUS_STRUCTURAL_MAX_DISTANCE, 1e-9)
        return _structural_rescue_payload(
            label=prototype_label,
            confidence=confidence,
            reason="plus_minus_anchor_high_margin_top1",
            prototype_match=prototype_match,
        )

    if (
        str(prototype_match.get("decision") or "") == "prototype_not_run"
        and _plus_minus_anchored_single_vertical_component(token, descriptor)
    ):
        return _structural_rescue_payload(
            label="1",
            confidence=0.35,
            reason="plus_minus_anchor_single_vertical_component",
            prototype_match=prototype_match,
        )
    return None


def _plus_minus_anchored_structural_context_allowed(token: dict[str, Any]) -> bool:
    if bool(token.get("consumer_allowed")):
        return False
    checks = token.get("strict_checks") if isinstance(token.get("strict_checks"), dict) else {}
    if bool(checks.get("consumer_allowed")):
        return False
    if str(checks.get("source") or "") != PLUS_MINUS_ANCHORED_SLOT_SOURCE:
        return False
    if checks.get("dimension_anchor_bound") is not True:
        return False
    if str(checks.get("dimension_axis_source") or "") != "plus_minus":
        return False
    if str(checks.get("orientation") or "H").upper() != "H":
        return False
    return True


def _plus_minus_anchored_prototype_rescue_label(
    prototype_match: dict[str, Any],
) -> str:
    label = str(prototype_match.get("top1_label") or "")
    if label not in PLUS_MINUS_STRUCTURAL_TOP1_LABELS:
        return ""
    try:
        distance = float(prototype_match.get("top1_distance"))
        margin = float(prototype_match.get("margin"))
    except (TypeError, ValueError):
        return ""
    if distance > PLUS_MINUS_STRUCTURAL_MAX_DISTANCE:
        return ""
    if margin < PLUS_MINUS_STRUCTURAL_MIN_MARGIN:
        return ""
    return label


def _plus_minus_anchored_single_vertical_component(
    token: dict[str, Any],
    descriptor: dict[str, Any],
) -> bool:
    if descriptor.get("status") != "descriptor_ready":
        return False
    checks = token.get("strict_checks") if isinstance(token.get("strict_checks"), dict) else {}
    component_bbox = checks.get("component_bbox")
    slot_bbox = checks.get("slot_bbox") or token.get("bbox")
    if not isinstance(component_bbox, dict) or not isinstance(slot_bbox, dict):
        return False
    try:
        component_w = float(component_bbox["w"])
        component_h = float(component_bbox["h"])
        slot_w = float(slot_bbox["w"])
        slot_h = float(slot_bbox["h"])
    except (KeyError, TypeError, ValueError):
        return False
    if component_w <= 0.0 or component_h <= 0.0 or slot_w <= 0.0 or slot_h <= 0.0:
        return False
    source_segment_count = _int_or(
        checks.get("source_segment_count"),
        len(token.get("source_segments") or []),
    )
    primitive_count = _int_or(
        checks.get("primitive_count"),
        descriptor.get("primitive_count"),
    )
    if source_segment_count > 3 or primitive_count > 3:
        return False
    if component_w / max(component_h, 1e-9) > 0.35:
        return False
    if component_w / max(slot_w, 1e-9) > 0.45:
        return False
    if component_h / max(slot_h, 1e-9) < 0.65:
        return False
    return True


def _structural_rescue_payload(
    *,
    label: str,
    confidence: float,
    reason: str,
    prototype_match: dict[str, Any],
) -> dict[str, Any]:
    return {
        "decision": STRUCTURAL_DIGIT_DECISION,
        "predicted_text": str(label),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "strict_checks": {
            "structural_rescue_status": STRUCTURAL_DIGIT_DECISION,
            "structural_rescue_source": PLUS_MINUS_ANCHORED_SLOT_SOURCE,
            "structural_rescue_reason": reason,
            "structural_rescue_allowed_labels": sorted(
                PLUS_MINUS_STRUCTURAL_TOP1_LABELS
            ),
            "structural_rescue_preserved_prototype_status": str(
                prototype_match.get("status") or ""
            ),
            "structural_rescue_preserved_prototype_top1_label": (
                prototype_match.get("top1_label")
            ),
            "structural_rescue_preserved_prototype_top1_distance": (
                prototype_match.get("top1_distance")
            ),
            "structural_rescue_consumer_allowed": CONSUMER_ALLOWED,
        },
    }


def _make_row(
    token: dict[str, Any],
    *,
    page_index: int,
    order: int,
    match_kind: str,
    decision: str,
    predicted_text: str | None,
    confidence: float,
    strict_checks: dict[str, Any],
    geometry_descriptor: dict[str, Any] | None = None,
    prototype_match: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    bbox = token.get("bbox")
    if not isinstance(bbox, dict):
        return None
    source_token_id = str(token.get("token_id") or f"glyph_{order:06d}")
    token_page_index = _int_or(token.get("page_index"), page_index)
    return {
        "schema_version": SCHEMA_VERSION,
        "match_id": f"p{token_page_index + 1:03d}_digit_match_{int(order):06d}",
        "page_index": token_page_index,
        "source_token_id": source_token_id,
        "source_glyph_type": str(token.get("glyph_type") or ""),
        "match_kind": match_kind,
        "decision": decision,
        "predicted_text": predicted_text,
        "bbox": dict(bbox),
        "oriented_quad": list(token.get("oriented_quad") or []),
        "confidence": round(float(confidence), 4),
        "strict_checks": strict_checks,
        "geometry_descriptor": geometry_descriptor,
        "prototype_match": prototype_match,
        "source_segments": [{
            "kind": "vector_glyph_token",
            "token_id": source_token_id,
        }],
        "source_glyph_segments": list(token.get("source_segments") or []),
        "source_glyph_checks": dict(token.get("strict_checks") or {}),
        "drop_reason": None,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _float_or(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _slot_descriptor_for_token(
    page: Any | None,
    token: dict[str, Any],
    *,
    primitive_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bbox = token.get("bbox")
    if page is None or not isinstance(bbox, dict):
        return _empty_descriptor("not_available")
    slot_bbox = _bbox_from_any(bbox)
    if slot_bbox is None:
        return _empty_descriptor("invalid_slot_bbox")
    primitives = _slot_primitives(primitive_index, slot_bbox, token)
    if not primitives:
        return _empty_descriptor("no_vector_geometry")

    grid_counts = [0.0 for _ in range(GRID_SIZE * GRID_SIZE)]
    stroke_widths: list[float] = []
    point_count = 0
    covered_bbox: dict[str, float] | None = None
    for primitive in primitives:
        points = primitive["points"]
        if primitive.get("stroke_width") is not None:
            stroke_widths.append(float(primitive["stroke_width"]))
        covered_bbox = _union_bbox(covered_bbox, primitive["bbox"])
        for point in _sample_points(points):
            nx = _clamp01((point[0] - slot_bbox["x"]) / max(slot_bbox["w"], 1e-9))
            ny = _clamp01((point[1] - slot_bbox["y"]) / max(slot_bbox["h"], 1e-9))
            col = min(GRID_SIZE - 1, int(nx * GRID_SIZE))
            row = min(GRID_SIZE - 1, int(ny * GRID_SIZE))
            grid_counts[row * GRID_SIZE + col] += 1.0
            point_count += 1
    total = sum(grid_counts)
    grid = [round(value / total, 6) for value in grid_counts] if total else grid_counts
    coverage = _coverage_ratio(covered_bbox, slot_bbox) if covered_bbox else 0.0
    covered_width_ratio = (
        _axis_coverage_ratio(covered_bbox, slot_bbox, axis="w")
        if covered_bbox else 0.0
    )
    covered_height_ratio = (
        _axis_coverage_ratio(covered_bbox, slot_bbox, axis="h")
        if covered_bbox else 0.0
    )
    raw_stroke_width = _median(stroke_widths) if stroke_widths else 0.0
    stroke_width_norm = _stroke_width_norm_for_slot(raw_stroke_width, slot_bbox)
    slot_aspect_log = _slot_aspect_log(slot_bbox)
    descriptor = {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "status": "descriptor_ready" if point_count >= 2 else "insufficient_points",
        "primitive_count": len(primitives),
        "point_count": point_count,
        "grid_size": GRID_SIZE,
        "occupancy_grid": grid,
        "bbox_coverage_ratio": round(coverage, 6),
        "covered_width_ratio": round(covered_width_ratio, 6),
        "covered_height_ratio": round(covered_height_ratio, 6),
        "stroke_width_norm": round(stroke_width_norm, 6),
        "slot_aspect_log": round(slot_aspect_log, 6),
        "primitive_count_norm": round(min(len(primitives), 32) / 32.0, 6),
        "point_count_norm": round(min(point_count, 96) / 96.0, 6),
    }
    mode = _stroke_width_mode()
    if mode != STROKE_WIDTH_MODE_RAW:
        descriptor[_PRIVATE_STROKE_WIDTH_MODE_KEY] = mode
    return descriptor


def _classify_with_prototypes(
    descriptor: dict[str, Any],
    *,
    prototype_library: dict[str, Any],
    threshold: float,
    margin: float,
) -> dict[str, Any]:
    labels = prototype_library.get("labels") or {}
    if not labels:
        return _prototype_match_payload(
            decision="prototype_not_run",
            status="not_run",
            threshold=threshold,
            margin=margin,
        )
    if descriptor.get("status") != "descriptor_ready":
        return _prototype_match_payload(
            decision="unknown",
            status=str(descriptor.get("status") or "descriptor_not_ready"),
            threshold=threshold,
            margin=margin,
        )
    feature = _descriptor_feature(descriptor)
    if feature is None:
        return _prototype_match_payload(
            decision="unknown",
            status="invalid_descriptor",
            threshold=threshold,
            margin=margin,
        )

    ranked: list[dict[str, Any]] = []
    for label, samples in labels.items():
        distances = [
            _feature_distance(feature, sample["feature"])
            for sample in samples
        ]
        if not distances:
            continue
        ranked.append({
            "label": label,
            "distance": min(distances),
            "sample_count": len(samples),
        })
    ranked.sort(key=lambda row: (row["distance"], row["label"]))
    if not ranked:
        return _prototype_match_payload(
            decision="unknown",
            status="no_valid_prototypes",
            threshold=threshold,
            margin=margin,
        )

    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None
    top2_distance = float(top2["distance"]) if top2 else None
    distance = float(top1["distance"])
    actual_margin = (
        top2_distance - distance
        if top2_distance is not None else None
    )
    if distance > threshold:
        decision = "unknown"
        status = "distance_above_threshold"
        predicted_text = None
    elif actual_margin is not None and actual_margin < margin:
        decision = "low_margin"
        status = "low_margin"
        predicted_text = None
    else:
        decision = "accepted_prototype"
        status = "accepted_prototype"
        predicted_text = str(top1["label"])
    confidence = (
        max(0.0, min(1.0, 1.0 - distance / max(threshold, 1e-9)))
        if decision == "accepted_prototype" else 0.0
    )
    return _prototype_match_payload(
        decision=decision,
        status=status,
        threshold=threshold,
        margin=margin,
        predicted_text=predicted_text,
        confidence=confidence,
        top1_label=str(top1["label"]),
        top1_distance=distance,
        top2_label=str(top2["label"]) if top2 else None,
        top2_distance=top2_distance,
        actual_margin=actual_margin,
        ranked=ranked[:3],
    )


def _prototype_match_payload(
    *,
    decision: str,
    status: str,
    threshold: float,
    margin: float,
    predicted_text: str | None = None,
    confidence: float = 0.0,
    top1_label: str | None = None,
    top1_distance: float | None = None,
    top2_label: str | None = None,
    top2_distance: float | None = None,
    actual_margin: float | None = None,
    ranked: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "status": status,
        "predicted_text": predicted_text,
        "confidence": round(float(confidence), 4),
        "threshold": float(threshold),
        "margin_threshold": float(margin),
        "top1_label": top1_label,
        "top1_distance": _round_optional(top1_distance),
        "top2_label": top2_label,
        "top2_distance": _round_optional(top2_distance),
        "margin": _round_optional(actual_margin),
        "ranked": [
            {
                "label": str(row["label"]),
                "distance": round(float(row["distance"]), 6),
                "sample_count": int(row.get("sample_count") or 0),
            }
            for row in (ranked or [])
        ],
    }


def _descriptor_feature(descriptor: Any) -> list[float] | None:
    if not isinstance(descriptor, dict):
        return None
    if descriptor.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION:
        return None
    grid = descriptor.get("occupancy_grid")
    if not isinstance(grid, list) or len(grid) != GRID_SIZE * GRID_SIZE:
        return None
    try:
        values = [float(value) for value in grid]
        for field in DESCRIPTOR_FEATURE_FIELDS:
            if field not in descriptor or descriptor[field] is None:
                return None
            value = float(descriptor[field])
            if field == "stroke_width_norm":
                value = _stroke_width_feature_value(descriptor, value)
            values.append(value)
    except (TypeError, ValueError):
        return None
    return values


def _feature_distance(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return float("inf")
    ablate_stroke_width = _ablate_stroke_width_distance()
    return math.sqrt(sum(
        0.0
        if ablate_stroke_width and idx == STROKE_WIDTH_FEATURE_INDEX
        else (left[idx] - right[idx]) ** 2
        for idx in range(size)
    ))


def _ablate_stroke_width_distance() -> bool:
    return (
        _stroke_width_mode() == STROKE_WIDTH_MODE_RAW
        and os.environ.get(ABLATE_STROKE_WIDTH_ENV) == "1"
    )


def _stroke_width_mode() -> str:
    mode = os.environ.get(STROKE_WIDTH_MODE_ENV, STROKE_WIDTH_MODE_RAW)
    mode = str(mode or "").strip().lower()
    if mode == STROKE_WIDTH_MODE_SIZE_RELATIVE:
        return STROKE_WIDTH_MODE_SIZE_RELATIVE
    return STROKE_WIDTH_MODE_RAW


def _stroke_width_norm_for_slot(
    raw_stroke_width: float,
    slot_bbox: dict[str, float],
) -> float:
    if _stroke_width_mode() == STROKE_WIDTH_MODE_SIZE_RELATIVE:
        return float(raw_stroke_width) / max(float(slot_bbox["h"]), 1e-6)
    return float(raw_stroke_width) / max(
        min(float(slot_bbox["w"]), float(slot_bbox["h"])),
        1e-9,
    )


def _stroke_width_feature_value(
    descriptor: dict[str, Any],
    value: float,
) -> float:
    if _stroke_width_mode() != STROKE_WIDTH_MODE_SIZE_RELATIVE:
        return value
    if descriptor.get(_PRIVATE_STROKE_WIDTH_MODE_KEY) == STROKE_WIDTH_MODE_SIZE_RELATIVE:
        return value
    try:
        aspect_log = float(descriptor.get("slot_aspect_log"))
    except (TypeError, ValueError):
        return value
    width_to_height = math.exp(aspect_log * 4.0)
    return value * min(width_to_height, 1.0)


def _public_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in descriptor.items()
        if not str(key).startswith("_")
    }


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _empty_descriptor(status: str) -> dict[str, Any]:
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "status": status,
        "primitive_count": 0,
        "point_count": 0,
        "grid_size": GRID_SIZE,
        "occupancy_grid": [0.0 for _ in range(GRID_SIZE * GRID_SIZE)],
        "bbox_coverage_ratio": 0.0,
        "covered_width_ratio": 0.0,
        "covered_height_ratio": 0.0,
        "stroke_width_norm": 0.0,
        "slot_aspect_log": 0.0,
        "primitive_count_norm": 0.0,
        "point_count_norm": 0.0,
    }


def _page_primitive_index(page: Any | None) -> dict[str, Any] | None:
    if page is None:
        return None
    primitives: list[dict[str, Any]] = []
    by_segment: dict[tuple[str, int], dict[str, Any]] = {}
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for kind, items, builder in (
        ("line", list(getattr(page, "lines", []) or []), _primitive_from_line),
        ("curve", list(getattr(page, "curves", []) or []), _primitive_from_curve),
        ("rect", list(getattr(page, "rects", []) or []), _primitive_from_rect),
    ):
        for index, raw in enumerate(items):
            primitive = builder(raw)
            if not primitive:
                continue
            primitive["kind"] = kind
            primitive["index"] = int(index)
            primitives.append(primitive)
            by_segment[(kind, int(index))] = primitive
            cell = _bucket_cell_for_bbox(primitive["bbox"])
            buckets.setdefault(cell, []).append(primitive)
    return {
        "primitives": primitives,
        "by_segment": by_segment,
        "buckets": buckets,
        "cell_size": PRIMITIVE_INDEX_CELL_PT,
    }


def _slot_primitives(
    primitive_index: dict[str, Any] | None,
    slot_bbox: dict[str, float],
    token: dict[str, Any],
) -> list[dict[str, Any]]:
    if primitive_index is None:
        return []
    source_primitives = _source_segment_primitives(primitive_index, token)
    if source_primitives:
        return [
            primitive for primitive in source_primitives
            if _primitive_belongs_to_slot(primitive, slot_bbox)
        ] or source_primitives
    return [
        primitive for primitive in _query_primitive_index(primitive_index, slot_bbox)
        if _primitive_belongs_to_slot(primitive, slot_bbox)
    ]


def _source_segment_primitives(
    primitive_index: dict[str, Any],
    token: dict[str, Any],
) -> list[dict[str, Any]]:
    by_segment = primitive_index.get("by_segment") or {}
    primitives: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for segment in token.get("source_segments") or []:
        if not isinstance(segment, dict):
            continue
        kind = str(segment.get("kind") or "")
        if kind not in {"line", "curve", "rect"}:
            continue
        try:
            index = int(segment.get("index"))
        except (TypeError, ValueError):
            continue
        key = (kind, index)
        if key in seen:
            continue
        primitive = by_segment.get(key)
        if primitive is not None:
            primitives.append(primitive)
            seen.add(key)
    return primitives


def _query_primitive_index(
    primitive_index: dict[str, Any],
    slot_bbox: dict[str, float],
) -> list[dict[str, Any]]:
    buckets = primitive_index.get("buckets") or {}
    if not buckets:
        return list(primitive_index.get("primitives") or [])
    cell_size = float(primitive_index.get("cell_size") or PRIMITIVE_INDEX_CELL_PT)
    margin = max(float(slot_bbox["w"]), float(slot_bbox["h"]), cell_size)
    x0 = float(slot_bbox["x"]) - margin
    y0 = float(slot_bbox["y"]) - margin
    x1 = float(slot_bbox["x"]) + float(slot_bbox["w"]) + margin
    y1 = float(slot_bbox["y"]) + float(slot_bbox["h"]) + margin
    ix0 = math.floor(x0 / cell_size)
    iy0 = math.floor(y0 / cell_size)
    ix1 = math.floor(x1 / cell_size)
    iy1 = math.floor(y1 / cell_size)
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            for primitive in buckets.get((ix, iy), []):
                key = (
                    str(primitive.get("kind") or ""),
                    int(primitive.get("index") or -1),
                )
                if key in seen:
                    continue
                seen.add(key)
                result.append(primitive)
    return result


def _bucket_cell_for_bbox(bbox: dict[str, float]) -> tuple[int, int]:
    cx = float(bbox["x"]) + float(bbox["w"]) / 2.0
    cy = float(bbox["y"]) + float(bbox["h"]) / 2.0
    return (
        math.floor(cx / PRIMITIVE_INDEX_CELL_PT),
        math.floor(cy / PRIMITIVE_INDEX_CELL_PT),
    )


def _primitive_from_line(raw: dict[str, Any]) -> dict[str, Any] | None:
    try:
        points = [
            (float(raw.get("x0")), float(raw.get("top", raw.get("y0")))),
            (float(raw.get("x1")), float(raw.get("bottom", raw.get("y1")))),
        ]
    except (TypeError, ValueError):
        return None
    return _primitive_from_points(points, raw)


def _primitive_from_curve(raw: dict[str, Any]) -> dict[str, Any] | None:
    raw_points = raw.get("pts") or raw.get("points") or []
    points: list[tuple[float, float]] = []
    for point in raw_points:
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    if len(points) < 2:
        bbox = _bbox_from_any(raw)
        if bbox is None:
            return None
        points = _bbox_corners(bbox)
    return _primitive_from_points(points, raw)


def _primitive_from_rect(raw: dict[str, Any]) -> dict[str, Any] | None:
    bbox = _bbox_from_any(raw)
    if bbox is None:
        return None
    return _primitive_from_points(_bbox_corners(bbox), raw)


def _primitive_from_points(
    points: list[tuple[float, float]],
    raw: dict[str, Any],
) -> dict[str, Any] | None:
    if len(points) < 2:
        return None
    bbox = _bbox_from_points(points)
    stroke_width = raw.get("linewidth", raw.get("stroke_width", raw.get("width")))
    try:
        stroke = float(stroke_width) if stroke_width is not None else None
    except (TypeError, ValueError):
        stroke = None
    return {"points": points, "bbox": bbox, "stroke_width": stroke}


def _primitive_belongs_to_slot(
    primitive: dict[str, Any],
    slot_bbox: dict[str, float],
) -> bool:
    primitive_bbox = primitive["bbox"]
    slot_area = slot_bbox["w"] * slot_bbox["h"]
    primitive_area = primitive_bbox["w"] * primitive_bbox["h"]
    if primitive_area > max(slot_area * 4.0, 1.0):
        return False
    if primitive_bbox["w"] > slot_bbox["w"] * 3.0:
        return False
    if primitive_bbox["h"] > slot_bbox["h"] * 3.0:
        return False
    intersection = _intersection_area(primitive_bbox, slot_bbox)
    if intersection <= 0.0:
        return False
    if _center_inside(primitive_bbox, slot_bbox):
        return True
    return intersection / max(primitive_area, 1e-9) >= 0.5


def _sample_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for point in points:
        samples.append(point)
    for left, right in zip(points, points[1:]):
        samples.append(((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0))
    return samples


def _bbox_from_any(raw: dict[str, Any]) -> dict[str, float] | None:
    try:
        if all(key in raw for key in ("x", "y", "w", "h")):
            x = float(raw["x"])
            y = float(raw["y"])
            w = float(raw["w"])
            h = float(raw["h"])
            return {"x": x, "y": y, "w": w, "h": h} if w > 0 and h > 0 else None
        if all(key in raw for key in ("x0", "x1")):
            x0 = float(raw["x0"])
            x1 = float(raw["x1"])
            y0 = float(raw.get("top", raw.get("y0")))
            y1 = float(raw.get("bottom", raw.get("y1")))
            return _bbox_from_xyxy(x0, y0, x1, y1)
    except (TypeError, ValueError):
        return None
    return None


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> dict[str, float] | None:
    x = min(x0, x1)
    y = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _bbox_from_points(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return {"x": x0, "y": y0, "w": max(x1 - x0, 1e-9), "h": max(y1 - y0, 1e-9)}


def _bbox_corners(bbox: dict[str, float]) -> list[tuple[float, float]]:
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    ax1 = a["x"] + a["w"]
    ay1 = a["y"] + a["h"]
    bx1 = b["x"] + b["w"]
    by1 = b["y"] + b["h"]
    w = max(0.0, min(ax1, bx1) - max(a["x"], b["x"]))
    h = max(0.0, min(ay1, by1) - max(a["y"], b["y"]))
    return w * h


def _center_inside(a: dict[str, float], b: dict[str, float]) -> bool:
    cx = a["x"] + a["w"] / 2.0
    cy = a["y"] + a["h"] / 2.0
    return b["x"] <= cx <= b["x"] + b["w"] and b["y"] <= cy <= b["y"] + b["h"]


def _union_bbox(
    a: dict[str, float] | None,
    b: dict[str, float],
) -> dict[str, float]:
    if a is None:
        return dict(b)
    x0 = min(a["x"], b["x"])
    y0 = min(a["y"], b["y"])
    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
    y1 = max(a["y"] + a["h"], b["y"] + b["h"])
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _coverage_ratio(
    covered_bbox: dict[str, float],
    slot_bbox: dict[str, float],
) -> float:
    return _intersection_area(covered_bbox, slot_bbox) / max(
        slot_bbox["w"] * slot_bbox["h"],
        1e-9,
    )


def _axis_coverage_ratio(
    covered_bbox: dict[str, float] | None,
    slot_bbox: dict[str, float],
    *,
    axis: str,
) -> float:
    if covered_bbox is None:
        return 0.0
    key = "w" if axis == "w" else "h"
    return _clamp01(
        float(covered_bbox.get(key) or 0.0) / max(float(slot_bbox[key]), 1e-9)
    )


def _slot_aspect_log(slot_bbox: dict[str, float]) -> float:
    width = max(float(slot_bbox["w"]), 1e-9)
    height = max(float(slot_bbox["h"]), 1e-9)
    return max(-1.0, min(1.0, math.log(width / height) / 4.0))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))
