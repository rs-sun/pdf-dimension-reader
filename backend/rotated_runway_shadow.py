"""R7 rotated runway shadow candidates.

This module proposes capsule-like evidence for already-assembled dimensions
using non-axis decimal-point angle seeds. It is shadow-only and must never mint
production dimensions or production is_key fields.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "r7_rotated_runway_shadow_v1"
CONSUMER_ALLOWED = False


def build_rotated_runway_shadow_v1(
    *,
    dimensions: list[dict[str, Any]],
    decimal_point_rows: list[dict[str, Any]],
    consumer_allowed: bool = False,
) -> dict[str, Any]:
    if consumer_allowed:
        raise ValueError("R7 rotated runway shadow consumer_allowed is blocked until the 95% gate")

    candidates: list[dict[str, Any]] = []
    input_consumer_allowed_decimal_rows = 0
    skipped_axis_dimension = 0
    skipped_axis_seed = 0
    skipped_forbidden_seed = 0

    usable_dots = []
    for row in decimal_point_rows or []:
        if row.get("consumer_allowed"):
            input_consumer_allowed_decimal_rows += 1
            skipped_forbidden_seed += 1
            continue
        center = _point(row.get("center"))
        angle = _seed_angle(row)
        if center is None or angle is None:
            continue
        if _axis_delta(angle) <= 8.0:
            skipped_axis_seed += 1
            continue
        usable_dots.append((row, center, angle))

    used_dimension_ids: set[str] = set()
    for dim in dimensions or []:
        bbox = _bbox_tuple(dim.get("bbox"))
        if bbox is None:
            continue
        dim_id = _dimension_id(dim)
        if dim_id in used_dimension_ids:
            continue
        dots_inside = [
            (dot, center, angle)
            for dot, center, angle in usable_dots
            if _point_inside_bbox(center, bbox)
        ]
        if not dots_inside:
            continue

        dim_angle = _dimension_angle(dim, bbox)
        selected: tuple[dict[str, Any], float] | None = None
        if _axis_delta(dim_angle) > 8.0:
            for dot, _center, angle in dots_inside:
                if _angle_delta(angle, dim_angle) <= 15.0:
                    selected = (dot, angle)
                    break
        else:
            selected = _paired_seed_angle(dots_inside)
            if selected is None:
                skipped_axis_dimension += 1
                continue

        if selected is not None:
            dot, angle = selected
            if _angle_delta(angle, dim_angle) > 15.0:
                if _axis_delta(dim_angle) > 8.0:
                    continue
            candidates.append(_candidate_from_dimension(dim, bbox, dot, angle))
            used_dimension_ids.add(dim_id)

    stats = {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "consumer_allowed_count": 0,
        "dimension_count": len(dimensions or []),
        "decimal_point_count": len(decimal_point_rows or []),
        "input_consumer_allowed_decimal_rows": input_consumer_allowed_decimal_rows,
        "candidate_count": len(candidates),
        "skipped_axis_seed": skipped_axis_seed,
        "skipped_axis_dimension": skipped_axis_dimension,
        "skipped_forbidden_seed": skipped_forbidden_seed,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "r7_rotated_runway_shadow_v1": candidates,
        "r7_rotated_runway_shadow_stats_v1": stats,
    }


def _candidate_from_dimension(
    dim: dict[str, Any],
    bbox: tuple[float, float, float, float],
    dot: dict[str, Any],
    angle: float,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"r7_rotated_runway_{len(str(_dimension_id(dim))) :02d}_{_dimension_id(dim)}",
        "dimension_id": _dimension_id(dim),
        "dimension_text": dim.get("text") or dim.get("nominal"),
        "x": float(x0),
        "y": float(y0),
        "w": float(width),
        "h": float(height),
        "bbox": {"x": float(x0), "y": float(y0), "w": float(width), "h": float(height)},
        "quad": [
            [float(x0), float(y0)],
            [float(x1), float(y0)],
            [float(x1), float(y1)],
            [float(x0), float(y1)],
        ],
        "angle_deg": round(float(angle), 3),
        "decimal_point_id": dot.get("id"),
        "source": "decimal_angle_seeded_rotated_runway_shadow",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _paired_seed_angle(
    dots_inside: list[tuple[dict[str, Any], tuple[float, float], float]],
) -> tuple[dict[str, Any], float] | None:
    for idx, (dot, _center, angle) in enumerate(dots_inside):
        aligned = [angle]
        for _other_dot, _other_center, other_angle in dots_inside[idx + 1:]:
            if _angle_delta(angle, other_angle) <= 5.0:
                aligned.append(other_angle)
        if len(aligned) >= 2:
            return dot, angle
    return None


def _dimension_id(dim: dict[str, Any]) -> str:
    return str(dim.get("dimension_id") or dim.get("id") or dim.get("dim_id") or "")


def _bbox_tuple(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, dict):
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            w = float(raw["w"])
            h = float(raw["h"])
        except (KeyError, TypeError, ValueError):
            return None
        if w <= 0.0 or h <= 0.0:
            return None
        return x, y, x + w, y + h
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            x0, y0, x1, y1 = (float(v) for v in raw)
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    return None


def _point(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, dict):
        try:
            return float(raw["x"]), float(raw["y"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None
    return None


def _point_inside_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    x, y = point
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def _seed_angle(row: dict[str, Any]) -> float | None:
    try:
        return _angle_mod_180(float((row.get("short_axis") or {}).get("angle_deg")))
    except (TypeError, ValueError):
        return None


def _dimension_angle(
    dim: dict[str, Any],
    bbox: tuple[float, float, float, float],
) -> float:
    for key in ("orientation", "angle_deg", "angle", "rotation_deg", "rotation"):
        if dim.get(key) is None:
            continue
        try:
            return _angle_mod_180(float(dim.get(key)))
        except (TypeError, ValueError):
            continue
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if height > width * 1.4:
        return 90.0
    return 0.0


def _angle_mod_180(angle: float) -> float:
    value = float(angle) % 180.0
    if abs(value - 180.0) < 1e-6:
        return 0.0
    return value


def _angle_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def _axis_delta(angle: float) -> float:
    return min(_angle_delta(angle, 0.0), _angle_delta(angle, 90.0))
