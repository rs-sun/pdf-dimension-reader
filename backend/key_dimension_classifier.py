"""R7 KEY dimension shadow classifier.

This module classifies already-assembled dimensions with capsule/runway
geometry. It is deliberately shadow-only: it must not emit production
``is_key`` or mutate the input dimensions.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


SCHEMA_VERSION = "r7_key_shadow_v1"
CONSUMER_ALLOWED = False
_EPS = 1e-6


def classify_key_dimensions_shadow(
    *,
    dimensions: list[dict[str, Any]],
    capsule_candidates: list[dict[str, Any]],
    gold_key_values: set[str] | list[str] | tuple[str, ...] | None = None,
    consumer_allowed: bool = False,
) -> dict[str, Any]:
    """Classify assembled dimensions as KEY in a non-consuming shadow payload."""
    if consumer_allowed:
        raise ValueError("R7 key shadow consumer_allowed is blocked until the 95% gate")

    rows: list[dict[str, Any]] = []
    capsules_with_dimensions: set[int] = set()
    legacy_is_key_count = 0

    for dim in dimensions or []:
        row = deepcopy(dim)
        legacy_is_key = bool(
            row.pop("is_key", False) or row.pop("in_capsule_blue", False)
        )
        if legacy_is_key:
            legacy_is_key_count += 1
            row["legacy_is_key"] = True
        row["consumer_allowed"] = CONSUMER_ALLOWED

        bbox = _bbox_tuple(row.get("bbox"))
        evidence = None
        if bbox is not None:
            for index, capsule in enumerate(capsule_candidates or []):
                if _bbox_contained_by_capsule(bbox, capsule):
                    capsules_with_dimensions.add(index)
                    evidence = {
                        "capsule_id": str(capsule.get("id") or f"capsule_{index}"),
                        "geometry": (
                            "rotated_quad" if capsule.get("quad") else "axis_aligned_bbox"
                        ),
                        "containment": "full_bbox",
                    }
                    break

        if evidence:
            row["is_key_shadow"] = True
            row["key_evidence"] = evidence
        else:
            row["is_key_shadow"] = False
        rows.append(row)

    gold_rows = _gold_summary(gold_key_values, rows)
    marked_key_count = sum(1 for row in rows if row.get("is_key_shadow"))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "consumer_allowed_count": 0,
        "dimension_count": len(rows),
        "capsule_count": len(capsule_candidates or []),
        "empty_capsule_count": max(
            0,
            len(capsule_candidates or []) - len(capsules_with_dimensions),
        ),
        "legacy_is_key_count": legacy_is_key_count,
        "marked_key_count": marked_key_count,
        **gold_rows,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "dimensions": rows,
        "summary": summary,
    }


def _gold_summary(
    gold_key_values: set[str] | list[str] | tuple[str, ...] | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_values = list(gold_key_values or [])
    gold_by_key: dict[str, str] = {}
    for value in gold_values:
        key = _value_key(value)
        if key is not None and key not in gold_by_key:
            gold_by_key[key] = str(value)

    marked_keys = {
        key
        for row in rows
        if row.get("is_key_shadow")
        for key in [_value_key(row.get("nominal") or row.get("text"))]
        if key is not None
    }
    missed = [
        original
        for key, original in gold_by_key.items()
        if key not in marked_keys
    ]
    return {
        "gold_key_count": len(gold_by_key),
        "matched_gold_key_count": len(gold_by_key) - len(missed),
        "missed_gold_key_values": missed,
    }


def _bbox_tuple(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, dict):
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            w = float(raw["w"])
            h = float(raw["h"])
        except (KeyError, TypeError, ValueError):
            return None
        return x, y, x + w, y + h
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            x0, y0, x1, y1 = (float(v) for v in raw)
        except (TypeError, ValueError):
            return None
        return x0, y0, x1, y1
    return None


def _bbox_contained_by_capsule(
    bbox: tuple[float, float, float, float],
    capsule: dict[str, Any],
) -> bool:
    corners = _bbox_corners(bbox)
    quad = capsule.get("quad")
    if isinstance(quad, list) and len(quad) >= 3:
        polygon = _points_from_quad(quad)
        if len(polygon) >= 3:
            return all(_point_in_polygon_or_edge(point, polygon) for point in corners)
    cap_bbox = _capsule_bbox(capsule)
    if cap_bbox is None:
        return False
    cx0, cy0, cx1, cy1 = cap_bbox
    x0, y0, x1, y1 = bbox
    return (
        cx0 - _EPS <= x0
        and cy0 - _EPS <= y0
        and cx1 + _EPS >= x1
        and cy1 + _EPS >= y1
    )


def _bbox_corners(bbox: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    x0, y0, x1, y1 = bbox
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _points_from_quad(quad: list[Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in quad:
        try:
            if isinstance(point, dict):
                points.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        except (KeyError, TypeError, ValueError):
            return []
    return points


def _capsule_bbox(capsule: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        x = float(capsule["x"])
        y = float(capsule["y"])
        w = float(capsule["w"])
        h = float(capsule["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return x, y, x + w, y + h


def _point_in_polygon_or_edge(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    if _point_on_polygon_edge(point, polygon):
        return True

    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        j = i
    return inside


def _point_on_polygon_edge(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    for i, start in enumerate(polygon):
        end = polygon[(i + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
    return False


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > _EPS:
        return False
    return (
        min(ax, bx) - _EPS <= px <= max(ax, bx) + _EPS
        and min(ay, by) - _EPS <= py <= max(ay, by) + _EPS
    )


def _value_key(value: Any) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    return str(round(float(match.group(0)), 6))
