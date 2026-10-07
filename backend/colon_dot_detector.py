"""R22 colon-only large dot detector.

The rows from this module are not decimal anchors. They are recall candidates
for colon pairing only, because ratio marks often draw colon dots as square
line boxes instead of the elongated ``qu`` rectangles accepted by the decimal
point detector.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from decimal_point_quad import (
    angle_delta,
    bbox_dict,
    bbox_from_points,
    normalize_quad_points,
    polygon_area,
    side_metrics,
)
from degree_polyline_strict import (
    Segment,
    connected_components,
    endpoint_nodes,
    extract_line_segments,
)
from vector_draw_items import extract_items_from_page


SCHEMA_VERSION = "r22_large_dot_marks_v1"
CONSUMER_ALLOWED = False


def detect_large_dot_marks(
    page: Any,
    *,
    page_index: int = 0,
    pdf_stem: str = "page",
    fail_on_error: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        items = extract_items_from_page(page, pdf_stem=pdf_stem, page_num=int(page_index) + 1)
    except Exception:
        if fail_on_error:
            raise
        items = []
    for item in items:
        row = classify_large_dot_quad(item)
        if row is not None:
            rows.append(row)
    try:
        rows.extend(detect_line_box_dot_marks(extract_line_segments(page)))
    except Exception:
        if fail_on_error:
            raise
        pass

    deduped = _dedupe(rows)
    for index, row in enumerate(deduped):
        row["id"] = f"p{int(page_index) + 1:03d}_large_dot_{index:06d}"
        row["schema_version"] = SCHEMA_VERSION
        row["page_index"] = int(page_index)
        row["consumer_allowed"] = CONSUMER_ALLOWED
    counts = Counter(str(row.get("source") or "unknown") for row in deduped)
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "r22_large_dot_marks_v1": deduped,
        "r22_large_dot_marks_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "page_index": int(page_index),
            "count": len(deduped),
            "counts_by_source": dict(sorted(counts.items())),
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def classify_large_dot_quad(item: Any) -> dict[str, Any] | None:
    if getattr(item, "op", None) != "qu":
        return None
    points = normalize_quad_points(list(getattr(item, "points", []) or []))
    if not points:
        return None
    metrics = side_metrics(points)
    if not metrics:
        return None
    bbox = bbox_from_points(points)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    short = float(metrics["short"])
    long = float(metrics["long"])
    ratio = float(metrics["ratio"])
    poly_area = polygon_area(points)
    bbox_area = width * height

    if not (1.0 <= short <= 4.8 and 1.0 <= long <= 7.2):
        return None
    if not (1.0 <= ratio <= 1.95):
        return None
    if not (1.0 <= poly_area <= 30.0 and bbox_area <= 42.0):
        return None
    if max(width, height) > 8.5:
        return None
    if float(metrics["opposite_len_err"]) > 0.24:
        return None
    if float(metrics["opposite_angle_err"]) > 6.0:
        return None
    if float(metrics["perpendicular_err"]) > 8.0:
        return None
    cx = sum(point[0] for point in points) / 4.0
    cy = sum(point[1] for point in points) / 4.0
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "",
        "bbox": bbox_dict(bbox),
        "quad": [[round(x, 3), round(y, 3)] for x, y in points],
        "center": {"x": round(cx, 3), "y": round(cy, 3)},
        "source": "large_colon_dot_quad",
        "anchor_eligible": False,
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
            "drawing_order": int(getattr(item, "drawing_order", -1)),
            "item_index": int(getattr(item, "item_index", -1)),
            "draw_type": str(getattr(item, "draw_type", "")),
            "stroke_width": (
                None
                if getattr(item, "stroke_width", None) is None
                else round(float(getattr(item, "stroke_width")), 4)
            ),
        },
    }


def detect_line_box_dot_marks(segments: list[Segment]) -> list[dict[str, Any]]:
    if not segments:
        return []
    nodes, segment_nodes = endpoint_nodes(segments)
    rows: list[dict[str, Any]] = []
    for component in connected_components(nodes, segment_nodes):
        segment_ids = sorted(int(segment_id) for segment_id in component.get("segments", set()))
        node_ids = sorted(int(node_id) for node_id in component.get("nodes", set()))
        if len(segment_ids) != 4 or len(node_ids) != 4:
            continue
        component_segments = [segments[segment_id] for segment_id in segment_ids]
        points = [nodes[node_id] for node_id in node_ids]
        bbox = bbox_from_points(points)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if min(width, height) < 0.8 or max(width, height) > 7.2:
            continue
        ratio = max(width, height) / max(min(width, height), 1e-6)
        if ratio > 2.05:
            continue
        if width * height > 42.0:
            continue
        lengths = sorted(float(segment.length) for segment in component_segments)
        if lengths[0] < 0.8 or lengths[-1] > 7.2:
            continue
        if _line_box_angle_error(component_segments) > 8.0:
            continue
        stroke_widths = [
            float(segment.stroke_width)
            for segment in component_segments
            if segment.stroke_width is not None
        ]
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "id": "",
            "bbox": bbox_dict(bbox),
            "center": {"x": round(cx, 3), "y": round(cy, 3)},
            "segment_ids": segment_ids,
            "source": "large_colon_dot_line_box",
            "anchor_eligible": False,
            "consumer_allowed": CONSUMER_ALLOWED,
            "detail": {
                "bbox_w": round(width, 4),
                "bbox_h": round(height, 4),
                "ratio": round(ratio, 4),
                "segment_lengths": [round(length, 4) for length in lengths],
                "stroke_width": round(sum(stroke_widths) / len(stroke_widths), 4) if stroke_widths else None,
            },
        })
    return rows


def _line_box_angle_error(segments: list[Segment]) -> float:
    angles = [float(segment.angle_deg) % 180.0 for segment in segments]
    axis_errors = [min(angle_delta(angle, 0.0), angle_delta(angle, 90.0)) for angle in angles]
    horizontal_count = sum(1 for angle in angles if angle_delta(angle, 0.0) <= 8.0)
    vertical_count = sum(1 for angle in angles if angle_delta(angle, 90.0) <= 8.0)
    if horizontal_count != 2 or vertical_count != 2:
        return 180.0
    return max(axis_errors)


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (
        float((r.get("center") or {}).get("y", 0.0)),
        float((r.get("center") or {}).get("x", 0.0)),
        str(r.get("source") or ""),
    )):
        center = row.get("center") or {}
        try:
            cx = float(center["x"])
            cy = float(center["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(_distance((cx, cy), _center(existing)) <= 0.75 for existing in kept):
            continue
        kept.append(dict(row))
    return kept


def _center(row: dict[str, Any]) -> tuple[float, float]:
    center = row.get("center") or {}
    return float(center.get("x", 0.0)), float(center.get("y", 0.0))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
