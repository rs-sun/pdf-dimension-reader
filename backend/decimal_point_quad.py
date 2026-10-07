"""Strict quadrilateral decimal-point detector for R2c.

The detector is dump-only by default. When explicitly enabled for replacement,
pipeline wiring can convert the rows back into the existing dot-seed-only
region shape without changing candidate_builder.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from vector_draw_items import extract_items_from_page


SCHEMA_VERSION = "vector_decimal_point_quad_v1"
CONSUMER_ALLOWED = True


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def angle_delta(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def polygon_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for idx, point in enumerate(points):
        other = points[(idx + 1) % len(points)]
        total += point[0] * other[1] - other[0] * point[1]
    return abs(total) / 2.0


def normalize_quad_points(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    if len(points) >= 5 and distance(points[0], points[-1]) <= 1e-6:
        points = points[:-1]
    if len(points) != 4:
        return None
    return [(float(x), float(y)) for x, y in points]


def side_metrics(points: list[tuple[float, float]]) -> dict[str, Any] | None:
    sides = []
    angles = []
    for idx, point in enumerate(points):
        other = points[(idx + 1) % 4]
        sides.append(distance(point, other))
        angles.append(angle_deg(point, other))
    if min(sides) <= 0:
        return None
    short = min(sides)
    long = max(sides)
    short_idx = min(range(4), key=lambda idx: sides[idx])
    return {
        "sides": sides,
        "angles": angles,
        "short": short,
        "long": long,
        "ratio": long / short,
        "short_angle_deg": angles[short_idx],
        "opposite_len_err": max(
            abs(sides[0] - sides[2]) / max(sides[0], sides[2], 1e-6),
            abs(sides[1] - sides[3]) / max(sides[1], sides[3], 1e-6),
        ),
        "opposite_angle_err": max(
            angle_delta(angles[0], angles[2]),
            angle_delta(angles[1], angles[3]),
        ),
        "perpendicular_err": abs(angle_delta(angles[0], angles[1]) - 90.0),
    }


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_dict(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "x": round(bbox[0], 3),
        "y": round(bbox[1], 3),
        "w": round(bbox[2] - bbox[0], 3),
        "h": round(bbox[3] - bbox[1], 3),
    }


def classify_decimal_rect(item: Any) -> dict[str, Any] | None:
    if item.op != "qu":
        return None
    points = normalize_quad_points(list(item.points))
    if not points:
        return None
    metrics = side_metrics(points)
    if not metrics:
        return None
    bbox = bbox_from_points(points)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    bbox_area = width * height
    poly_area = polygon_area(points)
    short = float(metrics["short"])
    long = float(metrics["long"])
    ratio = float(metrics["ratio"])

    if not (0.45 <= short <= 4.8 and 0.75 <= long <= 7.2):
        return None
    if not (1.18 <= ratio <= 1.85):
        return None
    if not (0.6 <= poly_area <= 26.0 and bbox_area <= 42.0):
        return None
    if max(width, height) > 8.5:
        return None
    if float(metrics["opposite_len_err"]) > 0.22:
        return None
    if float(metrics["opposite_angle_err"]) > 6.0:
        return None
    if float(metrics["perpendicular_err"]) > 8.0:
        return None

    confidence = "high" if 1.30 <= ratio <= 1.62 and poly_area >= 1.2 else "medium"
    cx = sum(point[0] for point in points) / 4.0
    cy = sum(point[1] for point in points) / 4.0
    axis_len = max(long * 2.4, 8.0)
    theta = math.radians(float(metrics["short_angle_deg"]))
    ux, uy = math.cos(theta), math.sin(theta)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "",
        "bbox": bbox_dict(bbox),
        "quad": [[round(x, 3), round(y, 3)] for x, y in points],
        "center": {"x": round(cx, 3), "y": round(cy, 3)},
        "short_axis": {
            "angle_deg": round(float(metrics["short_angle_deg"]), 2),
            "x1": round(cx - ux * axis_len / 2.0, 3),
            "y1": round(cy - uy * axis_len / 2.0, 3),
            "x2": round(cx + ux * axis_len / 2.0, 3),
            "y2": round(cy + uy * axis_len / 2.0, 3),
        },
        "confidence": confidence,
        "source": "qu_rect_ratio",
        "consumer_allowed": CONSUMER_ALLOWED,
        "detail": {
            "short": round(short, 4),
            "long": round(long, 4),
            "ratio": round(ratio, 4),
            "poly_area": round(poly_area, 4),
            "bbox_area": round(bbox_area, 4),
            "bbox_w": round(width, 4),
            "bbox_h": round(height, 4),
            "opposite_len_err": round(float(metrics["opposite_len_err"]), 4),
            "opposite_angle_err": round(float(metrics["opposite_angle_err"]), 4),
            "perpendicular_err": round(float(metrics["perpendicular_err"]), 4),
            "drawing_order": int(item.drawing_order),
            "item_index": int(item.item_index),
            "draw_type": str(item.draw_type),
            "stroke_width": None if item.stroke_width is None else round(float(item.stroke_width), 4),
        },
    }


def dedupe(rows: list[dict[str, Any]], *, prefix: str = "dot") -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r["bbox"]["y"], r["bbox"]["x"], r["confidence"] != "high")):
        rc = row["center"]
        if any(math.hypot(rc["x"] - existing["center"]["x"], rc["y"] - existing["center"]["y"]) <= 1.2 for existing in kept):
            continue
        row = dict(row)
        row["id"] = f"{prefix}_{len(kept):04d}"
        kept.append(row)
    return kept


def detect_decimal_point_quads(
    page: Any,
    *,
    page_index: int = 0,
    pdf_stem: str = "page",
    item_primitives: Any = None,
) -> dict[str, Any]:
    items = (
        item_primitives
        if item_primitives is not None
        else extract_items_from_page(
            page,
            pdf_stem=pdf_stem,
            page_num=page_index + 1,
        )
    )
    rows = dedupe([row for item in items if (row := classify_decimal_rect(item))])
    counts = Counter(row["confidence"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_decimal_point_quad_v1": rows,
        "vector_decimal_point_quad_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "page_index": int(page_index),
            "count": len(rows),
            "counts_by_confidence": dict(sorted(counts.items())),
            "consumer_allowed_count": sum(1 for row in rows if row.get("consumer_allowed")),
        },
    }


def _dot_seed_region_from_quad(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["bbox"]
    cx = float(raw["x"]) + float(raw["w"]) / 2.0
    cy = float(raw["y"]) + float(raw["h"]) / 2.0
    width = max(float(raw["w"]) + 4.0, 10.0)
    height = max(float(raw["h"]) + 4.0, 10.0)
    return {
        "x": cx - width / 2.0,
        "y": cy - height / 2.0,
        "w": width,
        "h": height,
        "curve_count": 1,
        "avg_curve_size": 0,
        "confidence": "low",
        "source": "numeric_symbol_seed",
        "subtype": "decimal_point_candidate",
        "raw_symbol_bbox": dict(raw),
        "role": "dot_seed_only",
        "dot_seed_only": True,
        "excluded_from_dbscan_merge": True,
        "numeric_symbol_seed_count": 1,
        "numeric_symbol_seed_subtypes": ["decimal_point_candidate"],
        "attached_numeric_symbol_seeds": [],
        "r2c_source": SCHEMA_VERSION,
        "r2c_quad_id": row.get("id"),
    }


def replace_dot_seed_only_regions(
    text_regions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    DEDUP_DIST = 1.2

    def is_dot_seed(region: dict[str, Any]) -> bool:
        return (
            str(region.get("source") or "") == "numeric_symbol_seed"
            and str(region.get("subtype") or "") == "decimal_point_candidate"
            and (
                str(region.get("role") or "") == "dot_seed_only"
                or bool(region.get("dot_seed_only"))
            )
        )

    def is_legacy_dot_seed(region: dict[str, Any]) -> bool:
        return is_dot_seed(region) and "r2c_source" not in region

    def bbox(region: dict[str, Any]) -> tuple[float, float, float, float] | None:
        try:
            return (
                float(region.get("x")),
                float(region.get("y")),
                float(region.get("w")),
                float(region.get("h")),
            )
        except (TypeError, ValueError):
            return None

    def contains_point(region: dict[str, Any], cx: float, cy: float) -> bool:
        box = bbox(region)
        if box is None:
            return False
        rx, ry, rw, rh = box
        return rx <= cx <= rx + rw and ry <= cy <= ry + rh

    # Despite the legacy switch name, this path now augments: keep old dot
    # bridge seeds, and only add R2c seeds where legacy did not already cover.
    retained = [dict(region) for region in text_regions]
    retained_text_regions = [region for region in retained if not is_dot_seed(region)]
    legacy_centers = []
    for region in retained:
        if not is_legacy_dot_seed(region):
            continue
        box = bbox(region)
        if box is None:
            continue
        rx, ry, rw, rh = box
        legacy_centers.append((rx + rw / 2.0, ry + rh / 2.0))

    supplemental_rows = []
    for row in rows:
        center = row["center"]
        cx = float(center["x"])
        cy = float(center["y"])
        if any(contains_point(region, cx, cy) for region in retained_text_regions):
            continue
        if any(math.hypot(cx - lx, cy - ly) <= DEDUP_DIST for lx, ly in legacy_centers):
            continue
        supplemental_rows.append(row)
    retained.extend(_dot_seed_region_from_quad(row) for row in supplemental_rows)
    retained.sort(key=lambda r: (round(float(r.get("y") or 0.0) / 5.0) * 5.0, float(r.get("x") or 0.0)))
    return retained


def apply_decimal_point_quad_pipeline(
    *,
    page: Any,
    page_index: int,
    text_regions: list[dict[str, Any]],
    pdf_stem: str,
    dump_enabled: bool,
    replace_legacy: bool,
    item_primitives: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not dump_enabled and not replace_legacy:
        return text_regions, {}
    detector_kwargs = {
        "page_index": page_index,
        "pdf_stem": pdf_stem,
    }
    if item_primitives is not None:
        detector_kwargs["item_primitives"] = item_primitives
    dump = detect_decimal_point_quads(page, **detector_kwargs)
    rows = dump["vector_decimal_point_quad_v1"]
    fields = {
        "vector_decimal_point_quad_v1": rows,
        "vector_decimal_point_quad_stats_v1": dump["vector_decimal_point_quad_stats_v1"],
    } if dump_enabled else {}
    if replace_legacy:
        return replace_dot_seed_only_regions(text_regions, rows), fields
    return text_regions, fields
