"""R29 dump-only text punctuation candidate vector marks.

R29 revoked semantic CJK punctuation classification. This module now keeps
only internal geometry candidates for layout-row diagnostics; decimal points
remain owned by decimal_point_quad.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from degree_polyline_strict import Segment, connected_components, endpoint_nodes, extract_line_segments


SCHEMA_VERSION = "r23_text_punct_marks_v1"
CONSUMER_ALLOWED = False
TEXT_PUNCT_ISOLATION_RADIUS_GLYPH_H = 0.6
TEXT_PUNCT_SINGLE_NEARBY_PRIMITIVE_RADIUS_PT = 10.0


def detect_text_punct_marks_geometry_v1(
    page: Any | None = None,
    *,
    segments: list[Segment] | None = None,
    glyph_h: float | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    if segments is None:
        segments = extract_line_segments(page) if page is not None else []
    primitive_marks, indexed_segments = _primitive_marks(segments)
    primitive_marks = _annotate_nearby_primitive_density(primitive_marks)
    glyph_h = _usable_glyph_h(glyph_h, primitive_marks)
    rows, isolation_rejected_count = _apply_isolation_gate(
        primitive_marks,
        indexed_segments,
        glyph_h=glyph_h,
    )
    rows = _dedupe(rows)
    for index, row in enumerate(rows):
        row["schema_version"] = SCHEMA_VERSION
        row["mark_id"] = f"p{int(page_index) + 1:03d}_text_punct_{index:06d}"
        row["page_index"] = int(page_index)
        row["is_text_punct"] = True
        row["consumer_allowed"] = CONSUMER_ALLOWED
    counts = Counter(str(row.get("kind") or "unknown") for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "r23_text_punct_marks_v1": rows,
        "r23_text_punct_marks_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "page_index": int(page_index),
            "raw_count": len(rows),
            "counts_by_kind": dict(sorted(counts.items())),
            "isolation_radius_glyph_h": TEXT_PUNCT_ISOLATION_RADIUS_GLYPH_H,
            "isolation_rejected_count": isolation_rejected_count,
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def finalize_text_punct_marks_v1(
    candidate_marks: list[dict[str, Any]],
    layout: dict[str, Any],
) -> list[dict[str, Any]]:
    tags = layout.get("anchor_region_tags") or {}
    out = []
    for row in candidate_marks or []:
        mark = dict(row)
        tag = tags.get(str(mark.get("mark_id") or mark.get("id") or ""))
        region = str((tag or {}).get("region") or "drawing")
        in_text_line = bool((tag or {}).get("text_line_id"))
        mark["region"] = region
        if in_text_line:
            mark["text_line_id"] = str((tag or {}).get("text_line_id") or "")
            mark["text_line_y"] = (tag or {}).get("text_line_y")
        mark["is_text_punct"] = True
        mark["consumer_allowed"] = CONSUMER_ALLOWED
        out.append(mark)
    return out


def _primitive_marks(
    segments: list[Segment],
) -> tuple[list[dict[str, Any]], list[Segment]]:
    if not segments:
        return [], []
    segments = [
        Segment(
            segment_id=index,
            p0=segment.p0,
            p1=segment.p1,
            length=segment.length,
            angle_deg=segment.angle_deg,
            drawing_order=segment.drawing_order,
            item_index=segment.item_index,
            stroke_width=segment.stroke_width,
        )
        for index, segment in enumerate(segments)
    ]
    nodes, segment_nodes = endpoint_nodes(segments)
    rows = []
    for component in connected_components(nodes, segment_nodes):
        segment_ids = sorted(int(segment_id) for segment_id in component.get("segments", set()))
        if not segment_ids:
            continue
        component_segments = [segments[segment_id] for segment_id in segment_ids]
        points = [nodes[node_id] for node_id in component.get("nodes", set())]
        bbox = _bbox_from_points(points)
        width = bbox["w"]
        height = bbox["h"]
        side = max(width, height)
        aspect = side / max(min(width, height), 1e-6)
        if len(segment_ids) == 4 and 1.0 <= side <= 3.5 and aspect <= 1.8:
            rows.append(_mark("punct_candidate", bbox, {
                "shape": "small_solid",
                "segment_ids": segment_ids,
                "source": "small_polyline_box",
            }))
        elif 5 <= len(segment_ids) <= 12 and 3.5 <= side <= 7.8 and aspect <= 1.6:
            rows.append(_mark("punct_candidate", bbox, {
                "shape": "ring",
                "segment_ids": segment_ids,
                "source": "small_closed_ring",
            }))
    return rows, segments


def _mark(kind: str, bbox: dict[str, float], detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "bbox": _round_bbox(bbox),
        "center": _center(bbox),
        "detail": detail,
        "source": "text_punct_marks",
        "is_text_punct": True,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    def priority(item: dict[str, Any]) -> tuple[int, float, float]:
        return (0, float(item["bbox"]["y"]), float(item["bbox"]["x"]))

    for row in sorted(rows, key=priority):
        bbox = row["bbox"]
        if any(_bbox_iou(bbox, existing["bbox"]) > 0.75 and row["kind"] == existing["kind"] for existing in kept):
            continue
        kept.append(row)
    return kept


def _annotate_nearby_primitive_density(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    centers = [_center(row["bbox"]) for row in rows]
    out = []
    for index, row in enumerate(rows):
        center = centers[index]
        nearby_count = sum(
            1
            for other_index, other_center in enumerate(centers)
            if other_index != index
            and _distance((center["x"], center["y"]), (other_center["x"], other_center["y"]))
            <= TEXT_PUNCT_SINGLE_NEARBY_PRIMITIVE_RADIUS_PT
        )
        detail = dict(row.get("detail") or {})
        detail["nearby_primitive_radius_pt"] = TEXT_PUNCT_SINGLE_NEARBY_PRIMITIVE_RADIUS_PT
        detail["nearby_primitive_count_10pt"] = nearby_count
        item = dict(row)
        item["detail"] = detail
        out.append(item)
    return out


def _apply_isolation_gate(
    rows: list[dict[str, Any]],
    segments: list[Segment],
    *,
    glyph_h: float,
) -> tuple[list[dict[str, Any]], int]:
    if not rows or not segments:
        return rows, 0
    radius = max(float(glyph_h) * TEXT_PUNCT_ISOLATION_RADIUS_GLYPH_H, 1.0)
    grid = _SegmentBBoxGrid(segments, cell_size=radius)
    kept: list[dict[str, Any]] = []
    rejected = 0
    for row in rows:
        bbox = row.get("bbox") or {}
        center = _center(bbox)
        own_segment_ids = {
            int(segment_id)
            for segment_id in (row.get("detail") or {}).get("segment_ids") or []
        }
        search_bbox = {
            "x": center["x"] - radius,
            "y": center["y"] - radius,
            "w": radius * 2.0,
            "h": radius * 2.0,
        }
        nearby_segments = grid.query(search_bbox)
        interfering = False
        for segment in nearby_segments:
            if int(segment.segment_id) in own_segment_ids:
                continue
            if _distance_point_to_segment((center["x"], center["y"]), segment) <= radius:
                interfering = True
                break
        if interfering:
            rejected += 1
            continue
        detail = dict(row.get("detail") or {})
        detail["isolation_gate"] = {
            "radius_glyph_h": TEXT_PUNCT_ISOLATION_RADIUS_GLYPH_H,
            "radius_pt": round(radius, 3),
        }
        out = dict(row)
        out["detail"] = detail
        kept.append(out)
    return kept, rejected


class _SegmentBBoxGrid:
    def __init__(self, segments: list[Segment], *, cell_size: float) -> None:
        self.cell_size = max(float(cell_size), 1.0)
        self.segments = {int(segment.segment_id): segment for segment in segments}
        self.cells: dict[tuple[int, int], list[int]] = {}
        for segment in segments:
            bbox = _segment_bbox(segment)
            gx0, gy0 = self._cell(bbox["x"], bbox["y"])
            gx1, gy1 = self._cell(bbox["x"] + bbox["w"], bbox["y"] + bbox["h"])
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    self.cells.setdefault((gx, gy), []).append(int(segment.segment_id))

    def query(self, bbox: dict[str, float]) -> list[Segment]:
        gx0, gy0 = self._cell(float(bbox["x"]), float(bbox["y"]))
        gx1, gy1 = self._cell(float(bbox["x"]) + float(bbox["w"]), float(bbox["y"]) + float(bbox["h"]))
        ids: set[int] = set()
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                ids.update(self.cells.get((gx, gy), []))
        return [self.segments[segment_id] for segment_id in ids if segment_id in self.segments]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(float(x) / self.cell_size)), int(math.floor(float(y) / self.cell_size))


def _segment_bbox(segment: Segment) -> dict[str, float]:
    x0 = min(float(segment.p0[0]), float(segment.p1[0]))
    y0 = min(float(segment.p0[1]), float(segment.p1[1]))
    x1 = max(float(segment.p0[0]), float(segment.p1[0]))
    y1 = max(float(segment.p0[1]), float(segment.p1[1]))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _distance_point_to_segment(point: tuple[float, float], segment: Segment) -> float:
    px, py = point
    x0, y0 = segment.p0
    x1, y1 = segment.p1
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return _distance(point, (float(x0), float(y0)))
    t = ((float(px) - float(x0)) * dx + (float(py) - float(y0)) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    nearest = (float(x0) + t * dx, float(y0) + t * dy)
    return _distance(point, nearest)


def _bbox_from_points(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def _usable_glyph_h(glyph_h: float | None, primitive_marks: list[dict[str, Any]]) -> float:
    try:
        value = float(glyph_h)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    sides = [
        max(float((row.get("bbox") or {}).get("w") or 0.0), float((row.get("bbox") or {}).get("h") or 0.0))
        for row in primitive_marks
    ]
    sides = [side for side in sides if side > 0]
    if not sides:
        return 8.0
    sides.sort()
    return max(sides[len(sides) // 2] * 1.5, 4.0)


def _center(bbox: dict[str, float]) -> dict[str, float]:
    return {"x": round(float(bbox["x"]) + float(bbox["w"]) / 2.0, 3), "y": round(float(bbox["y"]) + float(bbox["h"]) / 2.0, 3)}


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {key: round(float(bbox[key]), 3) for key in ("x", "y", "w", "h")}


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    x0 = max(a["x"], b["x"])
    y0 = max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(a["w"] * a["h"], 1e-6)
    area_b = max(b["w"] * b["h"], 1e-6)
    return inter / max(area_a + area_b - inter, 1e-6)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
