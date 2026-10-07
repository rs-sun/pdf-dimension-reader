"""R1.2 oriented-quad OCR crop telemetry.

The current phase is dump-only: it compares current AABB crop plans with a
proposed oriented crop derived from candidate ``oriented_quad`` metadata. It
does not alter the crop plans consumed by OCR.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


SCHEMA_VERSION = "ocr_crop_oriented_quad_v1"
MIN_QUAD_AREA_PT2 = 4.0
MAX_QUAD_TO_AABB_AREA_RATIO = 2.5
DEFAULT_LONG_PAD_PT = 6.0
DEFAULT_SHORT_PAD_PT = 4.0


def build_ocr_crop_oriented_quad_v1(
    *,
    crop_plans: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    ocr_pass: str = "candidate_guided",
) -> dict[str, Any]:
    candidates_by_id = {
        candidate.get("candidate_id"): candidate
        for candidate in candidates or []
        if candidate.get("candidate_id")
    }
    rows: list[dict[str, Any]] = []
    for plan in crop_plans or []:
        candidate_id = plan.get("candidate_id")
        candidate = candidates_by_id.get(candidate_id, {})
        source_bbox = _bbox_from_any(candidate.get("bbox"))
        current_crop = _bbox_from_any(plan.get("crop_bbox"))
        source_quad = _normalize_quad(candidate.get("oriented_quad"))
        selected_quad, fallback_reason = _select_quad(
            source_quad,
            source_bbox=source_bbox,
            current_crop=current_crop,
            page_width=page_width,
            page_height=page_height,
        )
        fallback_used = fallback_reason is not None
        if selected_quad is None:
            selected_quad = _quad_from_bbox(current_crop)
        selected_crop_bbox = _bbox_from_quad(selected_quad) or current_crop
        selected_area = _quad_area(selected_quad)
        current_area = _bbox_area(current_crop)
        row = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "crop_id": plan.get("crop_id"),
            "source_bbox": _round_bbox(source_bbox),
            "source_oriented_quad": source_quad,
            "selected_crop_bbox": _round_bbox(selected_crop_bbox),
            "selected_crop_quad": selected_quad,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason or "",
            "area_pt2": _round(selected_area),
            "current_aabb_crop_bbox": _round_bbox(current_crop),
            "current_aabb_area_pt2": _round(current_area),
            "area_delta_vs_aabb_pt2": _round(selected_area - current_area),
            "area_ratio_vs_aabb": _round(
                selected_area / current_area if current_area > 0.0 else 0.0
            ),
            "ocr_pass": ocr_pass,
            "dedup_trace_id": plan.get("crop_id"),
        }
        rows.append(row)
    fallback_counts = Counter(
        row["fallback_reason"] or "oriented_quad"
        for row in rows
    )
    area_values = [float(row["area_pt2"] or 0.0) for row in rows]
    aabb_values = [float(row["current_aabb_area_pt2"] or 0.0) for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "ocr_crop_oriented_quad_v1": rows,
        "ocr_crop_oriented_quad_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "plan_count": len(rows),
            "oriented_quad_count": sum(1 for row in rows if not row["fallback_used"]),
            "fallback_count": sum(1 for row in rows if row["fallback_used"]),
            "fallback_reason_counts": dict(sorted(fallback_counts.items())),
            "mean_area_pt2": _mean(area_values),
            "mean_aabb_area_pt2": _mean(aabb_values),
            "mean_area_ratio_vs_aabb": _round(
                _mean(area_values) / _mean(aabb_values)
                if _mean(aabb_values) > 0.0 else 0.0
            ),
            "schema_field_coverage": _schema_field_coverage(rows),
        },
    }


def _select_quad(
    source_quad: dict[str, Any] | None,
    *,
    source_bbox: dict[str, float] | None,
    current_crop: dict[str, float] | None,
    page_width: float,
    page_height: float,
) -> tuple[list[list[float]] | None, str | None]:
    if not source_quad:
        return None, "missing_oriented_quad"
    points = source_quad.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return None, "invalid_oriented_quad"
    area = _quad_area(points)
    if area < MIN_QUAD_AREA_PT2:
        return None, "quad_area_too_small"
    current_area = _bbox_area(current_crop)
    if current_area > 0.0 and area / current_area > MAX_QUAD_TO_AABB_AREA_RATIO:
        return None, "quad_area_too_large"

    selected = _inflate_quad(
        points,
        long_pad=_crop_pad(current_crop, source_bbox, axis="long"),
        short_pad=_crop_pad(current_crop, source_bbox, axis="short"),
    )
    clipped = _clip_quad(selected, page_width=page_width, page_height=page_height)
    if _quad_area(clipped) < MIN_QUAD_AREA_PT2:
        return None, "selected_quad_area_too_small"
    return clipped, None


def _inflate_quad(
    points: list[list[float]],
    *,
    long_pad: float,
    short_pad: float,
) -> list[list[float]]:
    p0, p1, p2, p3 = [tuple(float(v) for v in point[:2]) for point in points]
    ux, uy, long_len = _unit_vector(p0, p1)
    vx, vy, short_len = _unit_vector(p1, p2)
    if long_len <= 0.0 or short_len <= 0.0:
        return _round_quad(points)
    cx = sum(point[0] for point in (p0, p1, p2, p3)) / 4.0
    cy = sum(point[1] for point in (p0, p1, p2, p3)) / 4.0
    half_long = long_len / 2.0 + max(0.0, float(long_pad))
    half_short = short_len / 2.0 + max(0.0, float(short_pad))
    return _round_quad([
        [cx - ux * half_long - vx * half_short, cy - uy * half_long - vy * half_short],
        [cx + ux * half_long - vx * half_short, cy + uy * half_long - vy * half_short],
        [cx + ux * half_long + vx * half_short, cy + uy * half_long + vy * half_short],
        [cx - ux * half_long + vx * half_short, cy - uy * half_long + vy * half_short],
    ])


def _crop_pad(
    current_crop: dict[str, float] | None,
    source_bbox: dict[str, float] | None,
    *,
    axis: str,
) -> float:
    if not current_crop or not source_bbox:
        return DEFAULT_LONG_PAD_PT if axis == "long" else DEFAULT_SHORT_PAD_PT
    try:
        cw = float(current_crop.get("w") or 0.0)
        ch = float(current_crop.get("h") or 0.0)
        sw = float(source_bbox.get("w") or 0.0)
        sh = float(source_bbox.get("h") or 0.0)
    except (TypeError, ValueError):
        return DEFAULT_LONG_PAD_PT if axis == "long" else DEFAULT_SHORT_PAD_PT
    if axis == "long":
        return max(DEFAULT_LONG_PAD_PT, (max(cw, ch) - max(sw, sh)) / 2.0)
    return max(DEFAULT_SHORT_PAD_PT, (min(cw, ch) - min(sw, sh)) / 2.0)


def _normalize_quad(quad: Any) -> dict[str, Any] | None:
    if not isinstance(quad, dict):
        return None
    points = quad.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return None
    normalized_points: list[list[float]] = []
    try:
        for point in points:
            normalized_points.append([round(float(point[0]), 3), round(float(point[1]), 3)])
    except (TypeError, ValueError, IndexError):
        return None
    out = dict(quad)
    out["points"] = normalized_points
    out.setdefault("coord_space", "page_pdf")
    return out


def _bbox_from_any(item: Any) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    nested = item.get("bbox")
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
            x0 = float(item.get("x0") or 0.0)
            y0 = float(item.get("y0") or 0.0)
            x1 = float(item.get("x1") or 0.0)
            y1 = float(item.get("y1") or 0.0)
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


def _bbox_area(bbox: dict[str, float] | None) -> float:
    if not bbox:
        return 0.0
    try:
        return max(0.0, float(bbox.get("w") or 0.0)) * max(
            0.0,
            float(bbox.get("h") or 0.0),
        )
    except (TypeError, ValueError):
        return 0.0


def _quad_area(points: Any) -> float:
    if not isinstance(points, list) or len(points) < 3:
        return 0.0
    try:
        coords = [(float(point[0]), float(point[1])) for point in points]
    except (TypeError, ValueError, IndexError):
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _bbox_from_quad(points: list[list[float]] | None) -> dict[str, float] | None:
    if not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    return _bbox_from_xyxy(min(xs), min(ys), max(xs), max(ys))


def _quad_from_bbox(bbox: dict[str, float] | None) -> list[list[float]]:
    if not bbox:
        return []
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    return _round_quad([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ])


def _clip_quad(
    points: list[list[float]],
    *,
    page_width: float,
    page_height: float,
) -> list[list[float]]:
    out: list[list[float]] = []
    for x, y in points:
        if page_width > 0.0:
            x = min(max(0.0, float(x)), float(page_width))
        if page_height > 0.0:
            y = min(max(0.0, float(y)), float(page_height))
        out.append([x, y])
    return _round_quad(out)


def _unit_vector(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return 0.0, 0.0, 0.0
    return dx / length, dy / length, length


def _round(value: float) -> float:
    return round(float(value), 6)


def _round_bbox(bbox: dict[str, float] | None) -> dict[str, float] | None:
    if not bbox:
        return None
    return {
        "x": round(float(bbox["x"]), 3),
        "y": round(float(bbox["y"]), 3),
        "w": round(float(bbox["w"]), 3),
        "h": round(float(bbox["h"]), 3),
    }


def _round_quad(points: list[list[float]]) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / len(values))


def _schema_field_coverage(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    required = {
        "schema_version",
        "candidate_id",
        "source_bbox",
        "source_oriented_quad",
        "selected_crop_bbox",
        "selected_crop_quad",
        "fallback_used",
        "fallback_reason",
        "area_pt2",
        "ocr_pass",
        "dedup_trace_id",
    }
    complete = 0
    for row in rows:
        if all(key in row for key in required):
            complete += 1
    return _round(complete / len(rows))
