"""Deterministic fixed-font vector digit template matching.

R2m uses this module as a dump-only classifier for clean vector digit slots.
It consumes pre-normalized contour point templates and candidate contour points;
it does not run OCR and does not make any result consumer-eligible.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


SCHEMA_VERSION = "vector_digit_fixed_template_match_v1"
STATS_SCHEMA_VERSION = "vector_digit_fixed_template_match_stats_v1"
TEMPLATE_LIBRARY_SCHEMA_VERSION = "vector_digit_fixed_template_library_v1"
GLYPH_TEMPLATE_LIBRARY_SCHEMA_VERSION = "vector_glyph_fixed_template_library_v2"
CONSUMER_ALLOWED = False
DEFAULT_DISTANCE_THRESHOLD = 0.008
DEFAULT_MARGIN_THRESHOLD = 0.010
CLASSIFICATION_CACHE_MAX_ENTRIES = 4096
CLASSIFICATION_CACHE_MAX_CANDIDATE_BYTES = 65536
DENSE_CHAMFER_MAX_PAIR_COUNT = 600_000
V2_ALLOWED_LABELS = frozenset(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    ".+-:=±Ø°"
)


def load_fixed_digit_template_library(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return normalize_fixed_digit_template_library(json.load(f))


def load_fixed_digit_template_library_with_extensions(
    path: str | Path,
    extension_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    promote_extension_base_templates: bool = False,
) -> dict[str, Any]:
    """Load one fixed-template library plus deterministic trusted extensions.

    Pitch-slot digit matching deliberately uses ``base_templates_only=True``.
    R31 therefore promotes the reviewed R30 digit templates only in this
    runtime composite; the source fixtures keep their original semantics and
    other readers that share the r2x library are unaffected.
    """
    base_path = Path(path)
    with base_path.open("r", encoding="utf-8") as f:
        base_payload = json.load(f)
    if not isinstance(base_payload, dict):
        return normalize_fixed_digit_template_library(base_payload)

    base_templates = base_payload.get("templates")
    if not isinstance(base_templates, list):
        return normalize_fixed_digit_template_library(base_payload)

    merged_templates = list(base_templates)
    seen_template_ids = {
        str(row.get("template_id") or "")
        for row in merged_templates
        if isinstance(row, dict) and str(row.get("template_id") or "")
    }
    extension_debug: list[dict[str, Any]] = []
    for raw_extension_path in extension_paths:
        extension_path = Path(raw_extension_path)
        with extension_path.open("r", encoding="utf-8") as f:
            extension_payload = json.load(f)
        if not isinstance(extension_payload, dict):
            raise ValueError(f"invalid fixed-template extension payload: {extension_path}")
        if bool(extension_payload.get("consumer_allowed")):
            raise ValueError(f"consumer-enabled template extension rejected: {extension_path}")
        extension_templates = extension_payload.get("templates")
        if not isinstance(extension_templates, list):
            raise ValueError(f"template extension has no templates list: {extension_path}")

        added_count = 0
        for row in extension_templates:
            if not isinstance(row, dict):
                continue
            template_id = str(row.get("template_id") or "")
            if template_id and template_id in seen_template_ids:
                raise ValueError(f"duplicate template_id {template_id!r} in {extension_path}")
            copied = dict(row)
            source = dict(row.get("source") or {})
            source["extension_library_path"] = str(extension_path)
            if promote_extension_base_templates:
                source["base_template"] = True
                source["base_template_promotion"] = "trusted_runtime_extension"
            source["consumer_allowed"] = False
            copied["source"] = source
            merged_templates.append(copied)
            if template_id:
                seen_template_ids.add(template_id)
            added_count += 1
        extension_debug.append({
            "path": str(extension_path),
            "schema_version": str(extension_payload.get("schema_version") or ""),
            "template_count": added_count,
            "promoted_to_base_templates": bool(promote_extension_base_templates),
            "consumer_allowed": CONSUMER_ALLOWED,
        })

    merged_payload = dict(base_payload)
    merged_payload["templates"] = merged_templates
    merged_payload["consumer_allowed"] = CONSUMER_ALLOWED
    library = normalize_fixed_digit_template_library(merged_payload)
    library["debug"].update({
        "base_library_path": str(base_path),
        "base_library_template_count": len(base_templates),
        "extension_libraries": extension_debug,
        "base_template_count": len(library.get("base_templates") or []),
        "composite_runtime_only": True,
        "consumer_allowed": CONSUMER_ALLOWED,
    })
    return library


def normalize_fixed_digit_template_library(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("templates") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return _empty_library("invalid_payload")

    schema_version = str(payload.get("schema_version") or TEMPLATE_LIBRARY_SCHEMA_VERSION)
    v2_labels = schema_version == GLYPH_TEMPLATE_LIBRARY_SCHEMA_VERSION
    labels: dict[str, dict[str, Any]] = {}
    templates: list[dict[str, Any]] = []
    templates_by_label: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    base_templates: list[dict[str, Any]] = []
    base_templates_by_label: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            skipped["non_object"] += 1
            continue
        label = str(row.get("label") or "")
        if not _valid_template_label(label, v2_labels=v2_labels):
            skipped["invalid_label"] += 1
            continue
        if not v2_labels and label in labels:
            skipped["duplicate_label"] += 1
            continue
        points = normalized_points(row.get("points"))
        if points is None:
            skipped["invalid_points"] += 1
            continue
        points.setflags(write=False)
        source = dict(row.get("source") or {})
        template = {
            "label": label,
            "template_id": str(row.get("template_id") or f"template_{idx:03d}_{label}"),
            "source": source,
            "points": points,
            "tree": cKDTree(points),
            "point_count": int(points.shape[0]),
        }
        templates.append(template)
        templates_by_label[label].append(template)
        if bool(source.get("base_template")):
            base_templates.append(template)
            base_templates_by_label[label].append(template)
        labels.setdefault(label, template)

    status = "loaded" if templates else ("invalid" if skipped else "empty")
    return {
        "normalized": True,
        "schema_version": schema_version,
        "labels": labels,
        "templates": templates,
        "templates_by_label": dict(templates_by_label),
        "base_templates": base_templates,
        "base_templates_by_label": dict(base_templates_by_label),
        "_classification_cache": OrderedDict(),
        "_classification_cache_lock": threading.RLock(),
        "debug": {
            "status": status,
            "label_count": len(labels),
            "template_count": len(templates),
            "base_template_count": len(base_templates),
            "labels": sorted(labels),
            "samples_per_label": dict(sorted(Counter(row["label"] for row in templates).items())),
            "skipped": dict(sorted(skipped.items())),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    }


def _valid_template_label(label: str, *, v2_labels: bool) -> bool:
    if len(label) != 1:
        return False
    if v2_labels:
        return label in V2_ALLOWED_LABELS
    return label.isdigit()


def classify_fixed_digit_points(
    points: Any,
    *,
    template_library: dict[str, Any],
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    margin: float = DEFAULT_MARGIN_THRESHOLD,
    allowed_labels: set[str] | frozenset[str] | None = None,
    base_templates_only: bool = False,
) -> dict[str, Any]:
    candidate = normalized_points(points)
    if candidate is None:
        return _classification_payload(
            decision="unknown",
            status="invalid_candidate_points",
            threshold=threshold,
            margin=margin,
        )

    templates = _iter_library_templates(
        template_library,
        allowed_labels=allowed_labels,
        base_templates_only=base_templates_only,
    )
    if not templates:
        return _classification_payload(
            decision="template_not_run",
            status="not_run",
            threshold=threshold,
            margin=margin,
        )

    cache_key = _classification_cache_key(
        candidate=candidate,
        templates=templates,
        threshold=threshold,
        margin=margin,
        allowed_labels=allowed_labels,
        base_templates_only=base_templates_only,
    )
    cached = _classification_cache_get(template_library, cache_key)
    if cached is not None:
        return cached

    candidate_tree = cKDTree(candidate)
    ranked = [
        {
            "label": str(template.get("label") or ""),
            "template_id": str(template.get("template_id") or ""),
            "distance": _symmetric_chamfer_to_template(candidate, candidate_tree, template),
        }
        for template in templates
        if isinstance(template, dict) and isinstance(template.get("points"), np.ndarray)
    ]
    ranked.sort(key=lambda row: (row["distance"], row["label"]))
    if not ranked:
        return _classification_payload(
            decision="unknown",
            status="no_valid_templates",
            threshold=threshold,
            margin=margin,
        )

    top1 = ranked[0]
    top2 = next(
        (row for row in ranked[1:] if str(row.get("label") or "") != str(top1.get("label") or "")),
        None,
    )
    nearest_same_label = next(
        (row for row in ranked[1:] if str(row.get("label") or "") == str(top1.get("label") or "")),
        None,
    )
    distance = float(top1["distance"])
    top2_distance = float(top2["distance"]) if top2 else None
    actual_margin = top2_distance - distance if top2_distance is not None else None
    if distance > threshold:
        decision = "unknown"
        status = "distance_above_threshold"
        predicted_text = None
    elif actual_margin is not None and actual_margin < margin:
        decision = "low_margin"
        status = "low_margin"
        predicted_text = None
    else:
        decision = "accepted_template"
        status = "accepted_template"
        predicted_text = str(top1["label"])
    confidence = (
        max(0.0, min(1.0, 1.0 - distance / max(threshold, 1e-9)))
        if decision == "accepted_template" else 0.0
    )
    payload = _classification_payload(
        decision=decision,
        status=status,
        threshold=threshold,
        margin=margin,
        predicted_text=predicted_text,
        confidence=confidence,
        top1_label=str(top1["label"]),
        top1_distance=distance,
        top1_template_id=str(top1.get("template_id") or ""),
        top2_label=str(top2["label"]) if top2 else None,
        top2_distance=top2_distance,
        nearest_same_label_distance=(
            float(nearest_same_label["distance"]) if nearest_same_label else None
        ),
        actual_margin=actual_margin,
        ranked=ranked[:3],
    )
    _classification_cache_put(template_library, cache_key, payload)
    return payload


def _classification_cache_key(
    *,
    candidate: np.ndarray,
    templates: list[dict[str, Any]],
    threshold: float,
    margin: float,
    allowed_labels: set[str] | frozenset[str] | None,
    base_templates_only: bool,
) -> tuple[Any, ...] | None:
    if not math.isfinite(float(threshold)) or not math.isfinite(float(margin)):
        return None
    contiguous = np.ascontiguousarray(candidate)
    if contiguous.nbytes > CLASSIFICATION_CACHE_MAX_CANDIDATE_BYTES:
        return None
    # Chamfer distance treats the candidate as an unordered point cloud.  Slot
    # construction can present the exact same cloud in a different primitive
    # order, so key the cache by a canonical row order instead of paying for an
    # equivalent template sweep again.
    canonical_candidate = np.ascontiguousarray(
        contiguous[np.lexsort((contiguous[:, 1], contiguous[:, 0]))]
    )
    allowed_key = (
        None
        if allowed_labels is None
        else tuple(sorted(str(label) for label in allowed_labels))
    )
    template_signature = tuple(
        (
            str(template.get("label") or ""),
            str(template.get("template_id") or ""),
            id(template.get("points")),
            tuple(getattr(template.get("points"), "shape", ())),
            str(getattr(template.get("points"), "dtype", "")),
        )
        for template in templates
        if isinstance(template, dict)
    )
    return (
        canonical_candidate.shape,
        str(canonical_candidate.dtype),
        canonical_candidate.tobytes(),
        float(threshold),
        float(margin),
        allowed_key,
        bool(base_templates_only),
        template_signature,
    )


def _classification_cache_get(
    template_library: dict[str, Any],
    key: tuple[Any, ...] | None,
) -> dict[str, Any] | None:
    if key is None or not isinstance(template_library, dict):
        return None
    cache = template_library.get("_classification_cache")
    lock = template_library.get("_classification_cache_lock")
    if not isinstance(cache, OrderedDict) or not hasattr(lock, "__enter__"):
        return None
    with lock:
        payload = cache.get(key)
        if payload is None:
            return None
        cache.move_to_end(key)
        return _copy_classification_payload(payload)


def _classification_cache_put(
    template_library: dict[str, Any],
    key: tuple[Any, ...] | None,
    payload: dict[str, Any],
) -> None:
    if key is None or not isinstance(template_library, dict):
        return
    cache = template_library.get("_classification_cache")
    lock = template_library.get("_classification_cache_lock")
    if not isinstance(cache, OrderedDict) or not hasattr(lock, "__enter__"):
        return
    with lock:
        cache[key] = _copy_classification_payload(payload)
        cache.move_to_end(key)
        while len(cache) > CLASSIFICATION_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)


def _copy_classification_payload(payload: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    copied["ranked"] = [
        dict(row)
        for row in payload.get("ranked", [])
        if isinstance(row, dict)
    ]
    return copied


def _iter_library_templates(
    template_library: dict[str, Any],
    *,
    allowed_labels: set[str] | frozenset[str] | None = None,
    base_templates_only: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(template_library, dict):
        return []
    allowed = {str(label) for label in allowed_labels} if allowed_labels is not None else None
    if base_templates_only and isinstance(template_library.get("base_templates"), list):
        base_templates = template_library.get("base_templates") or []
        if base_templates:
            if allowed is None:
                return [row for row in base_templates if isinstance(row, dict)]
            base_by_label = template_library.get("base_templates_by_label")
            if isinstance(base_by_label, dict):
                return [
                    row
                    for label in sorted(allowed)
                    for row in base_by_label.get(label, [])
                    if isinstance(row, dict)
                ]
            return [
                row for row in base_templates
                if isinstance(row, dict) and str(row.get("label") or "") in allowed
            ]
    if allowed is not None and isinstance(template_library.get("templates_by_label"), dict):
        by_label = template_library.get("templates_by_label") or {}
        return [
            row
            for label in sorted(allowed)
            for row in by_label.get(label, [])
            if isinstance(row, dict)
        ]
    templates = template_library.get("templates")
    if isinstance(templates, list):
        has_base_templates = any(
            bool((row.get("source") or {}).get("base_template"))
            for row in templates
            if isinstance(row, dict)
        )
        return [
            row for row in templates
            if isinstance(row, dict)
            and (allowed is None or str(row.get("label") or "") in allowed)
            and (
                not base_templates_only
                or not has_base_templates
                or bool((row.get("source") or {}).get("base_template"))
            )
        ]
    labels = template_library.get("labels")
    if isinstance(labels, dict):
        return [
            row for row in labels.values()
            if isinstance(row, dict)
            and (allowed is None or str(row.get("label") or "") in allowed)
        ]
    return []


def classify_fixed_digit_with_reject_filter(
    points: Any,
    *,
    template_library: dict[str, Any],
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    margin: float = DEFAULT_MARGIN_THRESHOLD,
    allowed_labels: set[str] | frozenset[str] | None = None,
    base_templates_only: bool = False,
) -> dict[str, Any]:
    """Classify a clean slot as a digit or reject it as non-digit.

    R2m line B deliberately uses the conservative pure-reject route for
    digit/letter prefiltering: if every digit template is farther than the
    fixed digit threshold, the slot is marked non_digit instead of being
    allowed to masquerade as the nearest digit.
    """
    match = classify_fixed_digit_points(
        points,
        template_library=template_library,
        threshold=threshold,
        margin=margin,
        allowed_labels=allowed_labels,
        base_templates_only=base_templates_only,
    )
    if (
        match.get("decision") == "unknown"
        and match.get("status") == "distance_above_threshold"
    ):
        match = dict(match)
        match["decision"] = "non_digit"
        match["status"] = "non_digit_distance_reject"
        match["predicted_text"] = None
        match["confidence"] = 0.0
    match["non_digit_filter"] = {
        "filter_kind": "digit_distance_reject",
        "threshold": float(threshold),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return match


def build_fixed_digit_template_match_dump(
    *,
    slots: list[dict[str, Any]],
    template_library: dict[str, Any],
    page_index: int = 0,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    margin: float = DEFAULT_MARGIN_THRESHOLD,
    reject_non_digit: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for order, slot in enumerate(slots):
        classifier = (
            classify_fixed_digit_with_reject_filter
            if reject_non_digit else classify_fixed_digit_points
        )
        match = classifier(
            slot.get("points"),
            template_library=template_library,
            threshold=threshold,
            margin=margin,
        )
        token_page_index = _int_or(slot.get("page_index"), page_index)
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "match_id": str(slot.get("slot_id") or f"p{token_page_index + 1:03d}_fixed_template_{order:06d}"),
            "page_index": token_page_index,
            "source_slot_id": str(slot.get("slot_id") or ""),
            "match_kind": (
                "fixed_font_exact_template_with_non_digit_reject"
                if reject_non_digit else "fixed_font_exact_template"
            ),
            "decision": match["decision"],
            "predicted_text": match["predicted_text"],
            "bbox": slot.get("bbox"),
            "confidence": match["confidence"],
            "fixed_template_match": match,
            "strict_checks": {
                "source": str(slot.get("source") or "clean_vector_slot"),
                "template_library_schema": TEMPLATE_LIBRARY_SCHEMA_VERSION,
                "candidate_point_count": len(slot.get("points") or []),
                "reject_non_digit": bool(reject_non_digit),
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            "true_label": slot.get("true_label"),
            "is_template_seed": bool(slot.get("is_template_seed")),
            "drop_reason": None,
            "consumer_allowed": CONSUMER_ALLOWED,
        })

    decision_counts = Counter(str(row.get("decision") or "missing") for row in rows)
    correct_count = sum(
        1 for row in rows
        if row.get("true_label") not in (None, "")
        and row.get("predicted_text") == row.get("true_label")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_digit_fixed_template_matches_v1": rows,
        "vector_digit_fixed_template_match_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": int(page_index),
            "row_count": len(rows),
            "counts_by_decision": dict(sorted(decision_counts.items())),
            "accepted_template_count": decision_counts.get("accepted_template", 0),
            "correct_labeled_count": correct_count,
            "consumer_allowed_count": sum(1 for row in rows if row.get("consumer_allowed")),
            "template_library": template_library.get("debug", {}),
        },
    }


def normalized_points(raw_points: Any) -> np.ndarray | None:
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        return None
    try:
        arr = np.asarray([[float(x), float(y)] for x, y in raw_points], dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return None
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    span = max_xy - min_xy
    scale = max(float(span[0]), float(span[1]), 1e-9)
    normalized = (arr - min_xy) / scale
    if not np.isfinite(normalized).all():
        return None
    return normalized


def symmetric_chamfer(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] == 0 or right.shape[0] == 0:
        return float("inf")
    left_tree = cKDTree(left)
    right_tree = cKDTree(right)
    left_to_right = left_tree.query(right, k=1)[0]
    right_to_left = right_tree.query(left, k=1)[0]
    return float((np.mean(left_to_right) + np.mean(right_to_left)) / 2.0)


def _symmetric_chamfer_to_template(
    candidate: np.ndarray,
    candidate_tree: cKDTree,
    template: dict[str, Any],
) -> float:
    points = template.get("points")
    if not isinstance(points, np.ndarray) or candidate.shape[0] == 0 or points.shape[0] == 0:
        return float("inf")
    if candidate.shape[0] * points.shape[0] <= DENSE_CHAMFER_MAX_PAIR_COUNT:
        distances = cdist(candidate, points)
        return float((
            np.mean(np.min(distances, axis=0))
            + np.mean(np.min(distances, axis=1))
        ) / 2.0)
    template_tree = template.get("tree")
    if not isinstance(template_tree, cKDTree):
        template_tree = cKDTree(points)
        template["tree"] = template_tree
    candidate_to_template = template_tree.query(candidate, k=1)[0]
    template_to_candidate = candidate_tree.query(points, k=1)[0]
    return float((np.mean(candidate_to_template) + np.mean(template_to_candidate)) / 2.0)


def _classification_payload(
    *,
    decision: str,
    status: str,
    threshold: float,
    margin: float,
    predicted_text: str | None = None,
    confidence: float = 0.0,
    top1_label: str | None = None,
    top1_distance: float | None = None,
    top1_template_id: str | None = None,
    top2_label: str | None = None,
    top2_distance: float | None = None,
    nearest_same_label_distance: float | None = None,
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
        "top1_template_id": top1_template_id,
        "top2_label": top2_label,
        "top2_distance": _round_optional(top2_distance),
        "nearest_same_label_distance": _round_optional(nearest_same_label_distance),
        "margin": _round_optional(actual_margin),
        "ranked": [
            {
                "label": str(row.get("label") or ""),
                "template_id": str(row.get("template_id") or ""),
                "distance": round(float(row.get("distance") or 0.0), 6),
            }
            for row in (ranked or [])
        ],
    }


def _empty_library(status: str) -> dict[str, Any]:
    return {
        "normalized": True,
        "schema_version": TEMPLATE_LIBRARY_SCHEMA_VERSION,
        "labels": {},
        "debug": {
            "status": status,
            "label_count": 0,
            "template_count": 0,
            "labels": [],
            "samples_per_label": {},
            "skipped": {},
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    }


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None and math.isfinite(float(value)) else None


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
