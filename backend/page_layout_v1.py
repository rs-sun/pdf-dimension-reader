"""R23 dump-only page layout layer.

The layer is deliberately independent from the legacy layout modules. It
extracts structural regions so later R21 shadow layers can tag, not erase,
anchors that live in title/table/note context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "r23_layout_v1"
CONSUMER_ALLOWED = False
DRAWING_ANCHOR_RATIO_HEALTH_MIN = 0.5
NOTES_MAX_PAGE_AREA_RATIO = 0.15
NOTES_MIN_ROWS = 3
NOTES_MIN_MARKS_PER_ROW = 3
GRID_MAX_DRAWING_AREA_RATIO = 0.20
VECTOR_TEXT_NOTES_MIN_ROWS = 12
VECTOR_TEXT_NOTES_MIN_MEDIAN_WIDTH = 120.0
VECTOR_TEXT_NOTES_MAX_X_LEFT_SPREAD = 565.0
VECTOR_TEXT_NOTES_MAX_ROBUST_X_LEFT_SPREAD = 400.0
VECTOR_TEXT_NOTES_MAX_PAGE_AREA_RATIO = 0.24
VECTOR_TEXT_NOTES_MAX_GROUP_AREA_RATIO = 0.24
VECTOR_TEXT_NOTES_MAX_REGIONS = 16
VECTOR_TEXT_NOTES_MERGE_Y_GAP = 72.0
VECTOR_TEXT_NOTES_LARGE_PAGE_MIN_WIDTH = 1200.0
VECTOR_TEXT_NOTES_LARGE_PAGE_MIN_HEIGHT = 900.0
VECTOR_TEXT_NOTES_MIN_SUBSTANTIAL_ROWS = 40
VECTOR_TEXT_NOTES_PAGE_GATE_MIN_ROWS = 24
NOTES_COLUMN_LEFT_CLAMP_QUANTILE = 0.15
NOTES_COLUMN_LEFT_CLAMP_SLACK_PT = 30.0
NOTES_ROW_CONTEXT_LONG_LENGTH_PT = 36.0
NOTES_ROW_CONTEXT_PROTRUSION_PT = 18.0
NOTES_ROW_CONTEXT_PAD_PT = 1.5
NOTES_ROW_CONTEXT_Y_BAND_PT = 64.0
NOTES_ROW_CONTEXT_EDGE_GRAZE_PT = 20.0
VECTOR_TEXT_NOTES_TIER2_MIN_ROWS = 6
VECTOR_TEXT_NOTES_TIER2_MIN_MEDIAN_WIDTH = 100.0
VECTOR_TEXT_NOTES_TIER2_MAX_X_LEFT_SPREAD = 120.0
VECTOR_TEXT_NOTES_X_LEFT_SPLIT_GAP_PT = 150.0
NOTES_ROW_EXTEND_GRAZE_PT = 28.0
NOTES_ROW_EXTEND_MAX_Y_GAP_PT = 30.0
VECTOR_TEXT_NOTES_TIER2_MAX_LONG_LINES = 2
VECTOR_TEXT_NOTES_FRAGMENT_MIN_ROWS = 4
VECTOR_TEXT_NOTES_OVERLAP_MERGE_MIN_RATIO = 0.30
VECTOR_TEXT_NOTES_OVERLAP_MERGE_MAX_PAGE_AREA_RATIO = 0.40
VECTOR_TEXT_NOTES_DOMINANT_LOWER_RIGHT_MIN_X_RATIO = 0.76
VECTOR_TEXT_NOTES_DOMINANT_LOWER_RIGHT_MIN_Y_RATIO = 0.55
VECTOR_TEXT_NOTES_REFERENCE_PANEL_MIN_TOP_Y_RATIO = 0.38
VECTOR_TEXT_NOTES_REFERENCE_PANEL_MAX_TOP_Y_RATIO = 0.55
VECTOR_TEXT_NOTES_REFERENCE_PANEL_MIN_BOTTOM_Y_RATIO = 0.82
VECTOR_TEXT_NOTES_REFERENCE_PANEL_MIN_HEIGHT_RATIO = 0.34
VECTOR_TEXT_NOTES_ANCHOR_TAG_SPLIT_MIN_GAP = 28.0
VECTOR_TEXT_NOTES_ANCHOR_TAG_MIN_TRAILING_ROWS = 30
VECTOR_TEXT_NOTES_DENSE_SHORT_HORIZONTAL_PER_ROW = 5.0
VECTOR_TEXT_NOTES_DENSE_SHORT_VERTICAL_PER_ROW = 4.0
VECTOR_TEXT_NOTES_GRID_MIN_HORIZONTAL_LINES = 4
VECTOR_TEXT_NOTES_GRID_MIN_VERTICAL_LINES = 3
VECTOR_TEXT_NOTES_GRID_MIN_SHORT_HORIZONTAL_LINES = 25
VECTOR_TEXT_NOTES_GRID_MIN_SHORT_VERTICAL_LINES = 25
VECTOR_TEXT_ROW_Y_TOLERANCE = 4.5
VECTOR_TEXT_ROW_X_GAP = 40.0
NOTES_BLOCKER_HORIZONTAL_MIN_LENGTH_PT = 25.0
NOTES_BLOCKER_VERTICAL_MIN_LENGTH_PT = 12.0
NOTES_BLOCKER_MAX_PAGE_AREA_RATIO = 0.12
NOTES_BLOCKER_MIN_GRID_CELLS = 6
NOTES_BLOCKER_STACK_MIN_ROWS = 4
NOTES_BLOCKER_STACK_EDGE_TOLERANCE_PT = 8.0
NOTES_BLOCKER_STACK_MIN_HEIGHT_PT = 40.0
NOTES_BLOCKER_STACK_MIN_LINE_LENGTH_PT = 80.0
NOTES_BLOCKER_MAX_DENSE_GRID_AXES = 35
TITLE_STRUCTURAL_SEARCH_WIDTH_PT = 760.0
TITLE_STRUCTURAL_SEARCH_HEIGHT_PT = 260.0
TITLE_STRUCTURAL_ROW_Y_TOLERANCE = 1.8
TITLE_STRUCTURAL_MIN_LINE_LENGTH_PT = 20.0
TITLE_STRUCTURAL_MAX_CHAIN_GAP_PT = 12.0
TITLE_STRUCTURAL_MAX_ROW_RIGHT_GAP_PT = 72.0
TITLE_STRUCTURAL_MIN_ROW_SPAN_PT = 480.0
TITLE_STRUCTURAL_MAX_ROW_SPAN_PT = 650.0
TITLE_STRUCTURAL_MIN_ROW_TOTAL_LENGTH_PT = 300.0
TITLE_STRUCTURAL_MIN_ROWS = 4
TITLE_STRUCTURAL_MAX_LEFT_SPREAD_PT = 36.0
TITLE_STRUCTURAL_MIN_WIDTH_PT = 480.0
TITLE_STRUCTURAL_MAX_WIDTH_PT = 650.0
TITLE_STRUCTURAL_MIN_HEIGHT_PT = 140.0
TITLE_STRUCTURAL_MAX_HEIGHT_PT = 220.0


@dataclass(frozen=True)
class _Line:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def horizontal(self) -> bool:
        return abs(self.y1 - self.y0) <= 1.5 and abs(self.x1 - self.x0) > 1.0

    @property
    def vertical(self) -> bool:
        return abs(self.x1 - self.x0) <= 1.5 and abs(self.y1 - self.y0) > 1.0

    @property
    def length(self) -> float:
        return ((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2) ** 0.5

    @property
    def x_left(self) -> float:
        return min(self.x0, self.x1)

    @property
    def x_right(self) -> float:
        return max(self.x0, self.x1)

    @property
    def y_top(self) -> float:
        return min(self.y0, self.y1)

    @property
    def y_bottom(self) -> float:
        return max(self.y0, self.y1)

    @property
    def x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def y(self) -> float:
        return (self.y0 + self.y1) / 2.0


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def build_page_layout_v1(
    *,
    page_rect: Any,
    line_segments: list[Any],
    reference_anchors: list[dict[str, Any]] | None = None,
    colon_marks: list[dict[str, Any]] | None = None,
    text_punct_marks: list[dict[str, Any]] | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    width, height = _page_size(page_rect)
    lines = [_line_from_segment(segment) for segment in line_segments]
    lines = [line for line in lines if line is not None]

    frame_border = _detect_frame_border(lines, width=width, height=height)
    drawing_area = (
        dict(frame_border["bbox"])
        if frame_border.get("border_found")
        else {"x": 0.0, "y": 0.0, "w": round(width, 3), "h": round(height, 3)}
    )
    title_grid_blocks = _detect_right_bottom_structural_title_grid(lines, drawing_area=drawing_area)
    grids = [
        *title_grid_blocks,
        *[
            block for block in _detect_grid_blocks(lines, drawing_area=drawing_area)
            if not any(_bbox_iou(block.get("bbox"), title.get("bbox")) > 0.60 for title in title_grid_blocks)
        ],
    ]
    title_blocks = [block for block in grids if block["region_kind"] == "title_block"]
    grid_tables = [block for block in grids if block["region_kind"] == "table"]
    notes_regions = _detect_notes_regions(
        reference_anchors or [],
        text_punct_marks or [],
        page_area={"x": 0.0, "y": 0.0, "w": width, "h": height},
        line_segments=lines,
        excluded_regions=title_blocks,
    )
    tables = _dedupe_layout_blocks(grid_tables)
    anchor_region_tags = _tag_regions(
        reference_anchors=reference_anchors or [],
        colon_marks=colon_marks or [],
        text_punct_marks=text_punct_marks or [],
        drawing_area=drawing_area,
        title_blocks=title_blocks,
        tables=tables,
        notes_regions=notes_regions,
    )
    anchor_region_stats = _anchor_region_health(
        reference_anchors=reference_anchors or [],
        anchor_region_tags=anchor_region_tags,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "healthy": anchor_region_stats["healthy"],
        "drawing_anchor_ratio": anchor_region_stats["drawing_anchor_ratio"],
        "frame_border": frame_border,
        "drawing_area": drawing_area,
        "title_blocks": title_blocks,
        "tables": tables,
        "notes_regions": notes_regions,
        "anchor_region_tags": anchor_region_tags,
        "stats": {
            "line_segment_count": len(lines),
            "title_block_count": len(title_blocks),
            "table_count": len(tables),
            "notes_region_count": len(notes_regions),
            "tagged_anchor_count": len(anchor_region_tags),
            "drawing_anchor_count": anchor_region_stats["drawing_anchor_count"],
            "reference_anchor_count": anchor_region_stats["reference_anchor_count"],
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def apply_layout_tags_to_l1(
    l1_calibration: dict[str, Any],
    layout: dict[str, Any],
    text_punct_marks: list[dict[str, Any]],
) -> dict[str, Any]:
    tags = layout.get("anchor_region_tags") or {}
    layout_healthy = bool(layout.get("healthy", True))
    out = dict(l1_calibration)
    anchors = []
    for anchor in l1_calibration.get("reference_anchors") or []:
        row = dict(anchor)
        tag = tags.get(str(row.get("anchor_id") or ""))
        region = str((tag or {}).get("region") or "drawing")
        row["region"] = region
        row["layout_healthy"] = layout_healthy
        if not layout_healthy:
            row["layout_unhealthy"] = True
        if layout_healthy and row.get("kind") == "degree":
            if region == "notes":
                row["demoted_to_text_punct"] = True
        anchors.append(row)
    out["reference_anchors"] = anchors
    out["text_punct_marks"] = list(text_punct_marks)
    out["layout_healthy"] = layout_healthy
    out["drawing_anchor_ratio"] = layout.get("drawing_anchor_ratio")
    stats = dict(out.get("stats") or {})
    stats["text_punct_mark_count"] = len(text_punct_marks)
    stats["degree_demoted_to_text_punct_count"] = sum(
        1 for row in anchors if row.get("demoted_to_text_punct")
    )
    stats["layout_healthy"] = layout_healthy
    stats["drawing_anchor_ratio"] = layout.get("drawing_anchor_ratio")
    out["stats"] = stats
    return out


def _detect_frame_border(lines: list[_Line], *, width: float, height: float) -> dict[str, Any]:
    h_long = [line for line in lines if line.horizontal and line.length >= width * 0.60]
    v_long = [line for line in lines if line.vertical and line.length >= height * 0.60]
    candidates = []
    for top in h_long:
        for bottom in h_long:
            if bottom.y <= top.y + 20:
                continue
            for left in v_long:
                for right in v_long:
                    if right.x <= left.x + 20:
                        continue
                    if not (
                        top.x_left <= left.x + 5 <= top.x_right
                        and top.x_left <= right.x - 5 <= top.x_right
                        and bottom.x_left <= left.x + 5 <= bottom.x_right
                        and bottom.x_left <= right.x - 5 <= bottom.x_right
                        and left.y_top <= top.y + 5 <= left.y_bottom
                        and left.y_top <= bottom.y - 5 <= left.y_bottom
                        and right.y_top <= top.y + 5 <= right.y_bottom
                        and right.y_top <= bottom.y - 5 <= right.y_bottom
                    ):
                        continue
                    bbox = {"x": left.x, "y": top.y, "w": right.x - left.x, "h": bottom.y - top.y}
                    area = bbox["w"] * bbox["h"]
                    if area >= width * height * 0.25:
                        candidates.append(_round_bbox(bbox))
    if not candidates:
        return {
            "border_found": False,
            "bbox": {"x": 0.0, "y": 0.0, "w": round(width, 3), "h": round(height, 3)},
            "consumer_allowed": CONSUMER_ALLOWED,
        }
    inner = min(candidates, key=lambda bbox: bbox["w"] * bbox["h"])
    return {"border_found": True, "bbox": inner, "consumer_allowed": CONSUMER_ALLOWED}


def _detect_grid_blocks(lines: list[_Line], *, drawing_area: dict[str, float]) -> list[dict[str, Any]]:
    h_lines = [
        line for line in lines
        if line.horizontal and _center_in_bbox((line.x, line.y), drawing_area)
        and line.length <= drawing_area["w"] * 0.55
        and line.length >= 8.0
    ]
    v_lines = [
        line for line in lines
        if line.vertical and _center_in_bbox((line.x, line.y), drawing_area)
        and line.length <= drawing_area["h"] * 0.55
        and line.length >= 8.0
    ]
    if len(h_lines) * len(v_lines) > 2_000_000:
        h_lines = [line for line in h_lines if line.length >= 16.0]
        v_lines = [line for line in v_lines if line.length >= 16.0]
    if len(h_lines) * len(v_lines) > 2_000_000:
        return []
    blocks = []
    used_pairs: set[tuple[float, float, float, float]] = set()
    for h in h_lines:
        xs = sorted({round(v.x, 1) for v in v_lines if _intersects_hv(h, v)})
        ys = sorted({round(other.y, 1) for other in h_lines if _h_overlap(h, other) >= 20})
        if len(xs) < 3 or len(ys) < 3:
            continue
        bbox = {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}
        key = (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
        if key in used_pairs:
            continue
        if not _grid_is_regular(
            xs=xs,
            ys=ys,
            bbox=bbox,
            drawing_area=drawing_area,
            h_lines=h_lines,
            v_lines=v_lines,
        ):
            continue
        used_pairs.add(key)
        cells = [
            _round_bbox({"x": xs[ix], "y": ys[iy], "w": xs[ix + 1] - xs[ix], "h": ys[iy + 1] - ys[iy]})
            for iy in range(len(ys) - 1)
            for ix in range(len(xs) - 1)
        ]
        kind = _grid_kind(bbox, drawing_area)
        blocks.append({
            "bbox": _round_bbox(bbox),
            "region_kind": kind,
            "row_count": len(ys) - 1,
            "col_count": len(xs) - 1,
            "cells": cells,
            "source": "orthogonal_grid",
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    deduped = []
    for block in sorted(blocks, key=lambda row: -(row["bbox"]["w"] * row["bbox"]["h"])):
        if any(_bbox_iou(block["bbox"], existing["bbox"]) > 0.80 for existing in deduped):
            continue
        deduped.append(block)
    return deduped


def _detect_right_bottom_structural_title_grid(
    lines: list[_Line],
    *,
    drawing_area: dict[str, float],
) -> list[dict[str, Any]]:
    right = float(drawing_area["x"]) + float(drawing_area["w"])
    bottom = float(drawing_area["y"]) + float(drawing_area["h"])
    search_zone = {
        "x": right - TITLE_STRUCTURAL_SEARCH_WIDTH_PT,
        "y": bottom - TITLE_STRUCTURAL_SEARCH_HEIGHT_PT,
        "w": TITLE_STRUCTURAL_SEARCH_WIDTH_PT + 10.0,
        "h": TITLE_STRUCTURAL_SEARCH_HEIGHT_PT + 10.0,
    }
    h_lines = [
        line for line in lines
        if line.horizontal
        and line.length >= TITLE_STRUCTURAL_MIN_LINE_LENGTH_PT
        and search_zone["y"] - 4.0 <= line.y <= search_zone["y"] + search_zone["h"] + 4.0
        and _h_overlap_line_bbox(line, search_zone) >= 5.0
    ]
    row_clusters = _cluster_horizontal_title_rows(h_lines)
    candidate_rows = []
    for cluster in row_clusters:
        chain = _right_anchored_horizontal_chain(cluster["lines"], frame_right=right)
        if chain is None:
            continue
        span = chain["x_right"] - chain["x_left"]
        right_gap = right - chain["x_right"]
        if not (-8.0 <= right_gap <= TITLE_STRUCTURAL_MAX_ROW_RIGHT_GAP_PT):
            continue
        if not (TITLE_STRUCTURAL_MIN_ROW_SPAN_PT <= span <= TITLE_STRUCTURAL_MAX_ROW_SPAN_PT):
            continue
        if chain["total_length"] < TITLE_STRUCTURAL_MIN_ROW_TOTAL_LENGTH_PT:
            continue
        candidate_rows.append({
            "y": cluster["y"],
            "x_left": chain["x_left"],
            "x_right": chain["x_right"],
            "span": span,
            "line_count": chain["line_count"],
            "total_length": chain["total_length"],
            "right_gap": right_gap,
        })
    if len(candidate_rows) < TITLE_STRUCTURAL_MIN_ROWS:
        return []

    left_seed = _median([row["x_left"] for row in candidate_rows])
    inlier_rows = [
        row for row in candidate_rows
        if abs(row["x_left"] - left_seed) <= TITLE_STRUCTURAL_MAX_LEFT_SPREAD_PT
    ]
    if len(inlier_rows) < TITLE_STRUCTURAL_MIN_ROWS:
        return []

    left = _median([row["x_left"] for row in inlier_rows])
    top = min(row["y"] for row in inlier_rows)
    bbox = {"x": left, "y": top, "w": right - left, "h": bottom - top}
    if not _looks_like_structural_title_bbox(bbox):
        return []

    rows_for_gate = sorted(inlier_rows, key=lambda row: row["y"])
    return [{
        "bbox": _round_bbox(bbox),
        "region_kind": "title_block",
        "row_count": max(len(rows_for_gate) - 1, 1),
        "col_count": 1,
        "cells": [],
        "source": "right_bottom_structural_title_grid",
        "gates": {
            "search_zone": _round_bbox(search_zone),
            "physical_width_range_pt": [
                TITLE_STRUCTURAL_MIN_WIDTH_PT,
                TITLE_STRUCTURAL_MAX_WIDTH_PT,
            ],
            "physical_height_range_pt": [
                TITLE_STRUCTURAL_MIN_HEIGHT_PT,
                TITLE_STRUCTURAL_MAX_HEIGHT_PT,
            ],
            "min_rows": TITLE_STRUCTURAL_MIN_ROWS,
            "row_count": len(rows_for_gate),
            "row_y_values": [round(float(row["y"]), 3) for row in rows_for_gate],
            "row_spans": [round(float(row["span"]), 3) for row in rows_for_gate],
            "row_right_gaps": [round(float(row["right_gap"]), 3) for row in rows_for_gate],
            "left_seed": round(float(left_seed), 3),
            "left_spread": round(max(row["x_left"] for row in rows_for_gate) - min(row["x_left"] for row in rows_for_gate), 3),
            "physical_size_prior": "constant_title_block_pt",
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }]


def _cluster_horizontal_title_rows(lines: list[_Line]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: item.y):
        if clusters and abs(float(clusters[-1]["y"]) - line.y) <= TITLE_STRUCTURAL_ROW_Y_TOLERANCE:
            cluster_lines = clusters[-1]["lines"]
            cluster_lines.append(line)
            clusters[-1]["y"] = sum(item.y for item in cluster_lines) / len(cluster_lines)
        else:
            clusters.append({"y": line.y, "lines": [line]})
    return clusters


def _right_anchored_horizontal_chain(
    lines: list[_Line],
    *,
    frame_right: float,
) -> dict[str, float] | None:
    intervals = sorted(
        [(line.x_left, line.x_right, line.length) for line in lines],
        key=lambda row: row[1],
        reverse=True,
    )
    seed = [
        interval for interval in intervals
        if interval[1] >= frame_right - TITLE_STRUCTURAL_MAX_ROW_RIGHT_GAP_PT
    ]
    if not seed:
        return None
    used = set(seed)
    x_left = min(interval[0] for interval in seed)
    x_right = max(interval[1] for interval in seed)
    total_length = sum(interval[2] for interval in seed)
    line_count = len(seed)
    changed = True
    while changed:
        changed = False
        for interval in intervals:
            if interval in used:
                continue
            if interval[1] >= x_left - TITLE_STRUCTURAL_MAX_CHAIN_GAP_PT and interval[0] < x_left:
                used.add(interval)
                x_left = min(x_left, interval[0])
                x_right = max(x_right, interval[1])
                total_length += interval[2]
                line_count += 1
                changed = True
    return {
        "x_left": x_left,
        "x_right": x_right,
        "total_length": total_length,
        "line_count": line_count,
    }


def _looks_like_structural_title_bbox(bbox: dict[str, float]) -> bool:
    return (
        TITLE_STRUCTURAL_MIN_WIDTH_PT <= float(bbox["w"]) <= TITLE_STRUCTURAL_MAX_WIDTH_PT
        and TITLE_STRUCTURAL_MIN_HEIGHT_PT <= float(bbox["h"]) <= TITLE_STRUCTURAL_MAX_HEIGHT_PT
    )


def _detect_notes_regions(
    reference_anchors: list[dict[str, Any]],
    text_punct_marks: list[dict[str, Any]],
    *,
    page_area: dict[str, float],
    line_segments: list[_Line],
    excluded_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # R29 U2: the vector-text detector is the authoritative notes path (it
    # carries the R29 truth-fenced gates and anchor-tagging policies). The
    # legacy aligned-punctuation lattice below is only a fallback for pages
    # without vector text rows; letting it preempt would strip those policies.
    vector_text_regions = _detect_vector_text_notes_regions(
        line_segments=line_segments,
        page_area=page_area,
        excluded_regions=excluded_regions or [],
    )
    if vector_text_regions:
        return vector_text_regions
    marks = []
    for row in [*reference_anchors, *text_punct_marks]:
        bbox = _xywh(row)
        center = _center(row, bbox)
        if bbox is None or center is None:
            continue
        kind = str(row.get("kind") or "")
        if kind not in {"dot", "punct_candidate"}:
            continue
        marks.append({"id": _row_id(row), "bbox": bbox, "center": center, "kind": kind})
    if len(marks) < NOTES_MIN_ROWS * NOTES_MIN_MARKS_PER_ROW:
        return []
    # U2 (R29): row-lattice tolerances adapt to the marks inside the row/region
    # being formed; a page-wide glyph_h estimate would couple the Chinese notes
    # lattice to unrelated drawing-area mark sizes.
    rows: list[list[dict[str, Any]]] = []
    for mark in sorted(marks, key=lambda item: item["center"]["y"]):
        if rows and abs(rows[-1][0]["center"]["y"] - mark["center"]["y"]) <= max(
            _estimate_mark_glyph_h(rows[-1]) * 0.45, 3.5
        ):
            rows[-1].append(mark)
        else:
            rows.append([mark])
    rows = [row for row in rows if len(row) >= NOTES_MIN_MARKS_PER_ROW]
    if len(rows) < NOTES_MIN_ROWS:
        return []
    rows = _rows_with_text_line_pitch(rows)
    if len(rows) < NOTES_MIN_ROWS:
        return []
    aligned_rows: list[list[dict[str, Any]]] = []
    for row in rows:
        first_x = min(mark["center"]["x"] for mark in row)
        group = [
            other for other in rows
            if abs(min(mark["center"]["x"] for mark in other) - first_x) <= 12.0
        ]
        if len(group) > len(aligned_rows):
            aligned_rows = group
    if len(aligned_rows) < NOTES_MIN_ROWS:
        return []
    rows = aligned_rows
    all_marks = [mark for row in rows for mark in row]
    glyph_h = _estimate_mark_glyph_h(all_marks)
    bbox = _union_bboxes([mark["bbox"] for mark in all_marks], pad=4.0)
    page_area_value = max(float(page_area["w"]) * float(page_area["h"]), 1e-6)
    if float(bbox["w"]) * float(bbox["h"]) > page_area_value * NOTES_MAX_PAGE_AREA_RATIO:
        return []
    if _long_line_density_in_bbox(line_segments, bbox, glyph_h=glyph_h) > max(2, len(all_marks) // 3):
        return []
    return [{
        "region_id": "notes_000000",
        "bbox": _round_bbox(bbox),
        "line_y_tolerance": round(max(glyph_h * 0.45, 3.5), 3),
        "line_baselines": [
            {
                "y": round(sum(mark["center"]["y"] for mark in row) / len(row), 3),
                "mark_ids": [mark["id"] for mark in row],
            }
            for row in rows
        ],
        "source": "aligned_punctuation_array",
        "gates": {
            "min_rows": NOTES_MIN_ROWS,
            "min_marks_per_row": NOTES_MIN_MARKS_PER_ROW,
            "max_page_area_ratio": NOTES_MAX_PAGE_AREA_RATIO,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }]


def _detect_vector_text_notes_regions(
    *,
    line_segments: list[_Line],
    page_area: dict[str, float],
    excluded_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows_before_clean = _vector_text_row_descriptors(line_segments)
    rows = _context_clean_vector_text_rows(rows_before_clean, line_segments)
    groups = [
        *_aligned_vector_text_row_groups(rows),
        *_loose_vector_text_column_groups(rows, page_area=page_area),
    ]
    blockers = _notes_blocker_regions(
        line_segments=line_segments,
        groups=groups,
        page_area=page_area,
        excluded_regions=excluded_regions or [],
    )
    candidates: list[dict[str, Any]] = []
    fragments: list[list[dict[str, Any]]] = []
    page_area_value = max(float(page_area["w"]) * float(page_area["h"]), 1e-6)
    for group_index, group in enumerate(groups):
        if len(group) < VECTOR_TEXT_NOTES_FRAGMENT_MIN_ROWS:
            continue
        for blocker_segment in _split_vector_text_rows_by_blockers(group, blockers):
            for segment in _split_rows_by_x_left_gap(blocker_segment):
                candidate = _vector_text_note_candidate(
                    segment,
                    line_segments=line_segments,
                    page_area=page_area,
                    page_area_value=page_area_value,
                    blockers=blockers,
                )
                if candidate is None:
                    if len(segment) >= VECTOR_TEXT_NOTES_FRAGMENT_MIN_ROWS:
                        fragments.append(list(segment))
                    continue
                candidate["source_group_index"] = group_index
                candidates.append(candidate)
    merged = _merge_vector_text_note_candidates(
        candidates,
        page_area=page_area,
        page_area_value=page_area_value,
        blockers=blockers,
    )
    merged = _attach_note_fragments(
        merged,
        fragments,
        blockers=blockers,
        page_area=page_area,
        page_area_value=page_area_value,
    )
    merged = _filter_vector_text_note_candidates_for_page(merged, page_area=page_area)
    merged = _extend_regions_with_edge_grazed_rows(
        merged,
        all_rows=rows_before_clean,
        line_segments=line_segments,
        blockers=blockers,
        page_area_value=page_area_value,
    )
    merged = _deoverlap_note_candidates(merged)
    regions = []
    for index, candidate in enumerate(merged[:VECTOR_TEXT_NOTES_MAX_REGIONS]):
        rows_for_region = sorted(candidate["rows"], key=lambda item: item["y"])
        region = {
            "region_id": f"notes_{index:06d}",
            "bbox": _round_bbox(candidate["bbox"]),
            "line_y_tolerance": round(VECTOR_TEXT_ROW_Y_TOLERANCE, 3),
            "line_baselines": [
                {
                    "y": round(float(row["y"]), 3),
                    "mark_ids": [f"vector_text_row_{row_index:03d}"],
                }
                for row_index, row in enumerate(rows_for_region)
            ],
            "source": "aligned_vector_text_rows",
            "gates": {
                "min_rows": VECTOR_TEXT_NOTES_MIN_ROWS,
                "min_median_width": VECTOR_TEXT_NOTES_MIN_MEDIAN_WIDTH,
                "max_x_left_spread": VECTOR_TEXT_NOTES_MAX_X_LEFT_SPREAD,
                "max_robust_x_left_spread": VECTOR_TEXT_NOTES_MAX_ROBUST_X_LEFT_SPREAD,
                "max_page_area_ratio": VECTOR_TEXT_NOTES_MAX_PAGE_AREA_RATIO,
                "merged_group_count": candidate["merged_group_count"],
                "long_line_count": candidate["long_line_count"],
                "grid_line_count": candidate["grid_line_count"],
                "blocker_count": len(blockers),
                "source_group_indices": candidate["source_group_indices"],
            },
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        region.update(_vector_text_note_anchor_tagging_policy(candidate, page_area=page_area))
        regions.append(region)
    return regions


def _context_clean_vector_text_rows(
    rows: list[dict[str, Any]],
    line_segments: list[_Line],
) -> list[dict[str, Any]]:
    """Drop pseudo text rows that live inside drawing linework.

    A paragraph row only contains short glyph strokes plus fully contained
    decorations (value boxes, underlines). Dimension text and view hatching
    always sit next to long segments that protrude beyond the row bbox, so a
    single protruding long segment marks the row as drawing context.
    """
    longs = [
        line for line in line_segments
        if line.length > NOTES_ROW_CONTEXT_LONG_LENGTH_PT
    ]
    if not longs or not rows:
        return rows
    band = NOTES_ROW_CONTEXT_Y_BAND_PT
    band_index: dict[int, list[_Line]] = {}
    for line in longs:
        y0 = min(line.y0, line.y1)
        y1 = max(line.y0, line.y1)
        for key in range(int(y0 // band), int(y1 // band) + 1):
            band_index.setdefault(key, []).append(line)
    out: list[dict[str, Any]] = []
    for row in rows:
        bbox = row["bbox"]
        x0 = float(bbox["x"]) - NOTES_ROW_CONTEXT_PAD_PT
        x1 = float(bbox["x"]) + float(bbox["w"]) + NOTES_ROW_CONTEXT_PAD_PT
        y0 = float(bbox["y"]) - NOTES_ROW_CONTEXT_PAD_PT
        y1 = float(bbox["y"]) + float(bbox["h"]) + NOTES_ROW_CONTEXT_PAD_PT
        dirty = False
        seen: set[int] = set()
        for key in range(int(y0 // band), int(y1 // band) + 1):
            for line in band_index.get(key, ()):
                marker = id(line)
                if marker in seen:
                    continue
                seen.add(marker)
                lx0 = min(line.x0, line.x1)
                lx1 = max(line.x0, line.x1)
                ly0 = min(line.y0, line.y1)
                ly1 = max(line.y0, line.y1)
                if lx0 >= x1 or lx1 <= x0 or ly0 >= y1 or ly1 <= y0:
                    continue
                if (
                    lx0 < x0 - NOTES_ROW_CONTEXT_PROTRUSION_PT
                    or lx1 > x1 + NOTES_ROW_CONTEXT_PROTRUSION_PT
                    or ly0 < y0 - NOTES_ROW_CONTEXT_PROTRUSION_PT
                    or ly1 > y1 + NOTES_ROW_CONTEXT_PROTRUSION_PT
                ):
                    if line.vertical:
                        # A vertical callout/border grazing the row edge is not
                        # drawing context; only a central crossing poisons.
                        depth = min(line.x - x0, x1 - line.x)
                        if depth <= NOTES_ROW_CONTEXT_EDGE_GRAZE_PT:
                            continue
                    dirty = True
                    break
            if dirty:
                break
        if not dirty:
            out.append(row)
    return out


def _notes_blocker_regions(
    *,
    line_segments: list[_Line],
    groups: list[list[dict[str, Any]]],
    page_area: dict[str, float],
    excluded_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for row in excluded_regions:
        bbox = _xywh(row)
        if bbox is None:
            continue
        blockers.append({
            "bbox": bbox,
            "source": str(row.get("source") or row.get("region_kind") or "excluded_region"),
        })

    for bbox in _grid_cell_blocker_bboxes(line_segments, page_area=page_area):
        blockers.append({"bbox": bbox, "source": "grid_cell_blocker"})

    for bbox in _bordered_stack_blocker_bboxes(line_segments, page_area=page_area):
        blockers.append({"bbox": bbox, "source": "bordered_stack_blocker"})

    for group in groups:
        if len(group) < 4:
            continue
        bbox = _union_bboxes([row["bbox"] for row in group], pad=4.0)
        if bbox["w"] > 700.0:
            continue
        grid = _table_grid_line_count_in_bbox(line_segments, bbox)
        if (
            (
                int(grid.get("horizontal") or 0) >= VECTOR_TEXT_NOTES_GRID_MIN_HORIZONTAL_LINES
                and int(grid.get("vertical") or 0) >= VECTOR_TEXT_NOTES_GRID_MIN_VERTICAL_LINES
            )
            or (
                int(grid.get("horizontal") or 0) >= 8
                and int(grid.get("vertical") or 0) >= 1
            )
        ):
            blockers.append({"bbox": bbox, "source": "vector_text_table_row_blocker"})

    return _dedupe_note_blockers(blockers)


def _grid_cell_blocker_bboxes(
    lines: list[_Line],
    *,
    page_area: dict[str, float],
) -> list[dict[str, float]]:
    h_lines = [
        line for line in lines
        if line.horizontal
        and NOTES_BLOCKER_HORIZONTAL_MIN_LENGTH_PT <= line.length <= page_area["w"] * 0.75
    ]
    v_lines = [
        line for line in lines
        if line.vertical
        and NOTES_BLOCKER_VERTICAL_MIN_LENGTH_PT <= line.length <= page_area["h"] * 0.75
    ]
    if not h_lines or not v_lines:
        return []

    dsu = _DisjointSet(len(h_lines) + len(v_lines))
    for h_index, h_line in enumerate(h_lines):
        for v_index, v_line in enumerate(v_lines):
            if _intersects_hv(h_line, v_line):
                dsu.union(h_index, len(h_lines) + v_index)

    components: dict[int, list[_Line]] = {}
    for index, line in enumerate([*h_lines, *v_lines]):
        components.setdefault(dsu.find(index), []).append(line)

    out: list[dict[str, float]] = []
    page_area_value = max(float(page_area["w"]) * float(page_area["h"]), 1e-6)
    for component in components.values():
        component_h = [line for line in component if line.horizontal]
        component_v = [line for line in component if line.vertical]
        if len(component_h) < 2 or len(component_v) < 2:
            continue
        xs = _cluster_axis_values([line.x for line in component_v], tolerance=2.0)
        ys = _cluster_axis_values([line.y for line in component_h], tolerance=2.0)
        if len(xs) < 2 or len(ys) < 2:
            continue
        cells: list[tuple[float, float, float, float]] = []
        for x_index in range(len(xs) - 1):
            x0 = xs[x_index]
            x1 = xs[x_index + 1]
            if x1 - x0 < 8.0:
                continue
            for y_index in range(len(ys) - 1):
                y0 = ys[y_index]
                y1 = ys[y_index + 1]
                if y1 - y0 < 8.0:
                    continue
                if (
                    _has_horizontal_span(component_h, y=y0, x0=x0, x1=x1)
                    and _has_horizontal_span(component_h, y=y1, x0=x0, x1=x1)
                    and _has_vertical_span(component_v, x=x0, y0=y0, y1=y1)
                    and _has_vertical_span(component_v, x=x1, y0=y0, y1=y1)
                ):
                    cells.append((x0, y0, x1, y1))
        if len(cells) < NOTES_BLOCKER_MIN_GRID_CELLS:
            continue
        if (
            len(xs) > NOTES_BLOCKER_MAX_DENSE_GRID_AXES
            and len(ys) > NOTES_BLOCKER_MAX_DENSE_GRID_AXES
        ):
            continue
        bbox = {
            "x": min(cell[0] for cell in cells),
            "y": min(cell[1] for cell in cells),
            "w": max(cell[2] for cell in cells) - min(cell[0] for cell in cells),
            "h": max(cell[3] for cell in cells) - min(cell[1] for cell in cells),
        }
        if bbox["w"] < 60.0 or bbox["h"] < 25.0:
            continue
        if bbox["w"] * bbox["h"] > page_area_value * NOTES_BLOCKER_MAX_PAGE_AREA_RATIO:
            continue
        out.append(bbox)
    return out


def _bordered_stack_blocker_bboxes(
    lines: list[_Line],
    *,
    page_area: dict[str, float],
) -> list[dict[str, float]]:
    tol = NOTES_BLOCKER_STACK_EDGE_TOLERANCE_PT
    h_lines = [
        line for line in lines
        if line.horizontal
        and NOTES_BLOCKER_STACK_MIN_LINE_LENGTH_PT <= line.length <= page_area["w"] * 0.35
    ]
    groups: list[dict[str, Any]] = []
    for line in sorted(h_lines, key=lambda item: (item.x_right, item.x_left)):
        target = None
        for group in groups:
            if (
                abs(line.x_right - group["x_right"]) <= tol
                and abs(line.x_left - group["x_left"]) <= tol * 3.0
            ):
                target = group
                break
        if target is None:
            groups.append({"x_left": line.x_left, "x_right": line.x_right, "ys": [line.y]})
        else:
            target["x_left"] = min(target["x_left"], line.x_left)
            target["ys"].append(line.y)
    out: list[dict[str, float]] = []
    for group in groups:
        ys = sorted(set(round(y, 1) for y in group["ys"]))
        if len(ys) < NOTES_BLOCKER_STACK_MIN_ROWS:
            continue
        height = ys[-1] - ys[0]
        if height < NOTES_BLOCKER_STACK_MIN_HEIGHT_PT:
            continue
        x0 = group["x_left"]
        x1 = group["x_right"]
        y_top = ys[0]
        y_bottom = ys[-1]
        intervals_by_x: dict[int, list[tuple[float, float]]] = {}
        for line in lines:
            if not line.vertical:
                continue
            if not (x0 - 4.0 <= line.x <= x1 + 4.0):
                continue
            seg_top = max(min(line.y0, line.y1), y_top)
            seg_bottom = min(max(line.y0, line.y1), y_bottom)
            if seg_bottom <= seg_top:
                continue
            intervals_by_x.setdefault(int(round(line.x / 3.0)), []).append((seg_top, seg_bottom))
        border_count = 0
        for intervals in intervals_by_x.values():
            covered = 0.0
            last_end: float | None = None
            for start, end in sorted(intervals):
                if last_end is None or start > last_end:
                    covered += end - start
                    last_end = end
                elif end > last_end:
                    covered += end - last_end
                    last_end = end
            if covered >= height * 0.7:
                border_count += 1
        if border_count < 2:
            continue
        out.append({"x": x0, "y": ys[0], "w": x1 - x0, "h": height})
    return out


def _dedupe_note_blockers(blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for blocker in sorted(
        blockers,
        key=lambda item: -float((item.get("bbox") or {}).get("w", 0.0)) * float((item.get("bbox") or {}).get("h", 0.0)),
    ):
        bbox = blocker.get("bbox")
        if not isinstance(bbox, dict):
            continue
        if any(_bbox_iou(bbox, existing.get("bbox")) > 0.85 for existing in out):
            continue
        out.append(blocker)
    return out


def _split_vector_text_rows_by_blockers(
    rows: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    segments = [list(rows)]
    for blocker in sorted(
        blockers,
        key=lambda item: -float((item.get("bbox") or {}).get("w", 0.0)) * float((item.get("bbox") or {}).get("h", 0.0)),
    ):
        blocker_bbox = blocker.get("bbox")
        if not isinstance(blocker_bbox, dict):
            continue
        next_segments: list[list[dict[str, Any]]] = []
        for segment in segments:
            if not segment:
                continue
            if (
                str(blocker.get("source") or "") == "grid_cell_blocker"
                and _median([float(row["width"]) for row in segment]) > 480.0
            ):
                next_segments.append(segment)
                continue
            segment_bbox = _union_bboxes([row["bbox"] for row in segment], pad=4.0)
            if _bbox_intersection_area(segment_bbox, blocker_bbox) <= 0.0:
                next_segments.append(segment)
                continue

            clean = [
                row for row in segment
                if _bbox_intersection_area(row["bbox"], blocker_bbox) <= 1.0
            ]
            if not clean:
                continue
            if len(clean) < len(segment):
                next_segments.extend(_contiguous_row_runs(clean))
                continue

            by0 = float(blocker_bbox["y"])
            by1 = float(blocker_bbox["y"]) + float(blocker_bbox["h"])
            bx0 = float(blocker_bbox["x"])
            bx1 = float(blocker_bbox["x"]) + float(blocker_bbox["w"])
            top: list[dict[str, Any]] = []
            bottom: list[dict[str, Any]] = []
            left: list[dict[str, Any]] = []
            right: list[dict[str, Any]] = []
            other: list[dict[str, Any]] = []
            for row in clean:
                bbox = row["bbox"]
                row_y0 = float(bbox["y"])
                row_y1 = float(bbox["y"]) + float(bbox["h"])
                row_x0 = float(bbox["x"])
                row_x1 = float(bbox["x"]) + float(bbox["w"])
                if row_y1 <= by0 + 1.0:
                    top.append(row)
                elif row_y0 >= by1 - 1.0:
                    bottom.append(row)
                elif row_x1 <= bx0 + 1.0:
                    left.append(row)
                elif row_x0 >= bx1 - 1.0:
                    right.append(row)
                else:
                    other.append(row)
            for part in (top, left, right, bottom, other):
                next_segments.extend(_contiguous_row_runs(part))
        segments = next_segments
    return segments


def _contiguous_row_runs(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    runs: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: item["y"]):
        if runs and float(row["y"]) - float(runs[-1][-1]["y"]) <= 42.0:
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def _vector_text_note_candidate(
    rows: list[dict[str, Any]],
    *,
    line_segments: list[_Line],
    page_area: dict[str, float],
    page_area_value: float,
    blockers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    rows, left_clamp = _clamp_left_outlier_rows(rows)
    if len(rows) < VECTOR_TEXT_NOTES_TIER2_MIN_ROWS:
        return None
    tier2 = len(rows) < VECTOR_TEXT_NOTES_MIN_ROWS
    bbox = _clamped_rows_union_bbox(rows, clamp=left_clamp, pad=4.0)
    if any(_bbox_intersection_area(bbox, blocker.get("bbox")) > 0.0 for blocker in blockers):
        return None
    if bbox["w"] * bbox["h"] > page_area_value * VECTOR_TEXT_NOTES_MAX_GROUP_AREA_RATIO:
        return None

    x_lefts = [float(row["x_left"]) for row in rows]
    x_left_spread = max(x_lefts) - min(x_lefts)
    robust_x_left_spread = _quantile(x_lefts, 0.90) - _quantile(x_lefts, 0.10)
    if x_left_spread > VECTOR_TEXT_NOTES_MAX_X_LEFT_SPREAD:
        return None
    if robust_x_left_spread > VECTOR_TEXT_NOTES_MAX_ROBUST_X_LEFT_SPREAD:
        return None

    widths = [float(row["width"]) for row in rows]
    median_width = _median(widths)
    min_median_width = (
        VECTOR_TEXT_NOTES_TIER2_MIN_MEDIAN_WIDTH if tier2 else VECTOR_TEXT_NOTES_MIN_MEDIAN_WIDTH
    )
    if median_width < min_median_width or median_width > page_area["w"] * 0.55:
        return None

    long_line_count = _long_line_density_in_bbox(line_segments, bbox, glyph_h=8.0)
    grid_line_count = _table_grid_line_count_in_bbox(line_segments, bbox)
    if _is_bottom_table_candidate(
        bbox,
        row_count=len(rows),
        long_line_count=long_line_count,
        page_area=page_area,
    ):
        return None
    if _is_top_right_tolerance_table_candidate(
        bbox,
        grid_line_count,
        page_area=page_area,
    ):
        return None
    if _is_vector_text_table_noise(
        grid_line_count,
        row_count=len(rows),
        median_width=median_width,
    ):
        return None
    if tier2:
        # Small standalone blocks (GB/T strips, deviation explanations) pass
        # only through much stricter gates than full note columns.
        if x_left_spread > VECTOR_TEXT_NOTES_TIER2_MAX_X_LEFT_SPREAD:
            return None
        if median_width < VECTOR_TEXT_NOTES_TIER2_MIN_MEDIAN_WIDTH:
            return None
        if int(grid_line_count.get("horizontal") or 0) > 0 or int(grid_line_count.get("vertical") or 0) > 0:
            return None
        if int(long_line_count) > VECTOR_TEXT_NOTES_TIER2_MAX_LONG_LINES:
            return None

    return {
        "rows": list(rows),
        "bbox": bbox,
        "median_width": median_width,
        "x_left_spread": x_left_spread,
        "robust_x_left_spread": robust_x_left_spread,
        "long_line_count": long_line_count,
        "grid_line_count": grid_line_count,
    }


def _is_bottom_table_candidate(
    bbox: dict[str, float],
    *,
    row_count: int,
    long_line_count: int,
    page_area: dict[str, float],
) -> bool:
    return (
        float(bbox["y"]) >= float(page_area["h"]) * 0.75
        and int(long_line_count) >= max(int(row_count), 12)
    )


def _is_top_right_tolerance_table_candidate(
    bbox: dict[str, float],
    grid_line_count: dict[str, int],
    *,
    page_area: dict[str, float],
) -> bool:
    right = float(bbox["x"]) + float(bbox["w"])
    if float(bbox["y"]) > float(page_area["h"]) * 0.20:
        return False
    if right < float(page_area["w"]) * 0.85:
        return False
    if float(bbox["w"]) > 700.0:
        return False
    return (
        int(grid_line_count.get("vertical") or 0) >= 2
        and int(grid_line_count.get("short_horizontal") or 0) >= 20
        and int(grid_line_count.get("short_vertical") or 0) >= 20
    )


def _is_vector_text_table_noise(
    grid_line_count: dict[str, int],
    *,
    row_count: int,
    median_width: float,
) -> bool:
    horizontal = int(grid_line_count.get("horizontal") or 0)
    vertical = int(grid_line_count.get("vertical") or 0)
    short_horizontal = int(grid_line_count.get("short_horizontal") or 0)
    short_vertical = int(grid_line_count.get("short_vertical") or 0)
    if (
        horizontal >= VECTOR_TEXT_NOTES_GRID_MIN_HORIZONTAL_LINES
        and vertical >= VECTOR_TEXT_NOTES_GRID_MIN_VERTICAL_LINES
    ):
        return True
    if horizontal >= 8 and vertical >= 1:
        return True
    per_row_h = short_horizontal / max(int(row_count), 1)
    per_row_v = short_vertical / max(int(row_count), 1)
    dense_short_lines = (
        per_row_h >= VECTOR_TEXT_NOTES_DENSE_SHORT_HORIZONTAL_PER_ROW
        and per_row_v >= VECTOR_TEXT_NOTES_DENSE_SHORT_VERTICAL_PER_ROW
    )
    if not dense_short_lines:
        return False
    return median_width < 280.0 or (per_row_h >= 10.0 and median_width < 350.0)


def _merge_vector_text_note_candidates(
    candidates: list[dict[str, Any]],
    *,
    page_area: dict[str, float],
    page_area_value: float,
    blockers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (item["bbox"]["x"], item["bbox"]["y"])):
        target = None
        for existing in merged:
            if _vector_text_note_candidates_should_merge(
                existing,
                candidate,
                page_area=page_area,
                page_area_value=page_area_value,
                blockers=blockers,
            ):
                target = existing
                break
        if target is None:
            merged.append({
                "rows": list(candidate["rows"]),
                "bbox": dict(candidate["bbox"]),
                "long_line_count": int(candidate["long_line_count"]),
                "grid_line_count": dict(candidate["grid_line_count"]),
                "merged_group_count": 1,
                "source_group_indices": [int(candidate["source_group_index"])],
            })
            continue
        target["rows"].extend(candidate["rows"])
        target["bbox"] = _union_bboxes([target["bbox"], candidate["bbox"]], pad=0.0)
        target["long_line_count"] = int(target["long_line_count"]) + int(candidate["long_line_count"])
        target["grid_line_count"] = {
            "horizontal": int(target["grid_line_count"]["horizontal"]) + int(candidate["grid_line_count"]["horizontal"]),
            "vertical": int(target["grid_line_count"]["vertical"]) + int(candidate["grid_line_count"]["vertical"]),
            "short_horizontal": (
                int(target["grid_line_count"].get("short_horizontal") or 0)
                + int(candidate["grid_line_count"].get("short_horizontal") or 0)
            ),
            "short_vertical": (
                int(target["grid_line_count"].get("short_vertical") or 0)
                + int(candidate["grid_line_count"].get("short_vertical") or 0)
            ),
        }
        target["merged_group_count"] = int(target["merged_group_count"]) + 1
        target["source_group_indices"].append(int(candidate["source_group_index"]))

    out = []
    for candidate in merged:
        rows, left_clamp = _clamp_left_outlier_rows(_dedupe_vector_text_rows(candidate["rows"]))
        candidate["rows"] = rows
        if len(candidate["rows"]) < VECTOR_TEXT_NOTES_TIER2_MIN_ROWS:
            continue
        candidate["bbox"] = _clamped_rows_union_bbox(rows, clamp=left_clamp, pad=4.0)
        bbox = candidate["bbox"]
        max_area_ratio = (
            VECTOR_TEXT_NOTES_OVERLAP_MERGE_MAX_PAGE_AREA_RATIO
            if int(candidate["merged_group_count"]) > 1
            else VECTOR_TEXT_NOTES_MAX_PAGE_AREA_RATIO
        )
        if bbox["w"] * bbox["h"] > page_area_value * max_area_ratio:
            continue
        if any(_bbox_intersection_area(bbox, blocker.get("bbox")) > 0.0 for blocker in blockers):
            continue
        candidate["source_group_indices"] = sorted(set(candidate["source_group_indices"]))
        out.append(candidate)
    out.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"], -len(item["rows"])))
    return out


def _split_rows_by_x_left_gap(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if len(rows) < 2 * VECTOR_TEXT_NOTES_FRAGMENT_MIN_ROWS:
        return [rows]
    ordered = sorted(rows, key=lambda item: float(item["x_left"]))
    best_index = None
    best_gap = VECTOR_TEXT_NOTES_X_LEFT_SPLIT_GAP_PT
    for index in range(len(ordered) - 1):
        left_count = index + 1
        right_count = len(ordered) - left_count
        # One side may be arbitrarily small: a couple of stray rows from the
        # neighboring block must not keep the main block from splitting off.
        if max(left_count, right_count) < VECTOR_TEXT_NOTES_FRAGMENT_MIN_ROWS:
            continue
        gap = float(ordered[index + 1]["x_left"]) - float(ordered[index]["x_left"])
        if gap > best_gap:
            best_gap = gap
            best_index = index
    if best_index is None:
        return [rows]
    left = ordered[: best_index + 1]
    right = ordered[best_index + 1:]
    return [*_split_rows_by_x_left_gap(left), *_split_rows_by_x_left_gap(right)]


def _row_context_poisons(
    row: dict[str, Any],
    long_lines: list[_Line],
    *,
    graze: float,
) -> bool:
    bbox = row["bbox"]
    x0 = float(bbox["x"]) - NOTES_ROW_CONTEXT_PAD_PT
    x1 = float(bbox["x"]) + float(bbox["w"]) + NOTES_ROW_CONTEXT_PAD_PT
    y0 = float(bbox["y"]) - NOTES_ROW_CONTEXT_PAD_PT
    y1 = float(bbox["y"]) + float(bbox["h"]) + NOTES_ROW_CONTEXT_PAD_PT
    for line in long_lines:
        lx0 = min(line.x0, line.x1)
        lx1 = max(line.x0, line.x1)
        ly0 = min(line.y0, line.y1)
        ly1 = max(line.y0, line.y1)
        if lx0 >= x1 or lx1 <= x0 or ly0 >= y1 or ly1 <= y0:
            continue
        if (
            lx0 < x0 - NOTES_ROW_CONTEXT_PROTRUSION_PT
            or lx1 > x1 + NOTES_ROW_CONTEXT_PROTRUSION_PT
            or ly0 < y0 - NOTES_ROW_CONTEXT_PROTRUSION_PT
            or ly1 > y1 + NOTES_ROW_CONTEXT_PROTRUSION_PT
        ):
            if line.vertical and min(line.x - x0, x1 - line.x) <= graze:
                continue
            if line.horizontal:
                overlap = min(lx1, x1) - max(lx0, x0)
                if overlap <= graze:
                    continue
            return True
    return False


def _extend_regions_with_edge_grazed_rows(
    candidates: list[dict[str, Any]],
    *,
    all_rows: list[dict[str, Any]],
    line_segments: list[_Line],
    blockers: list[dict[str, Any]],
    page_area_value: float,
) -> list[dict[str, Any]]:
    if not candidates or not all_rows:
        return candidates
    long_lines = [
        line for line in line_segments
        if line.length > NOTES_ROW_CONTEXT_LONG_LENGTH_PT
    ]
    for candidate in candidates:
        member_keys = {
            (round(float(r["bbox"]["x"]), 2), round(float(r["y"]), 3))
            for r in candidate["rows"]
        }
        changed = True
        while changed:
            changed = False
            bbox = candidate["bbox"]
            by0 = float(bbox["y"])
            by1 = float(bbox["y"]) + float(bbox["h"])
            for row in all_rows:
                key = (round(float(row["bbox"]["x"]), 2), round(float(row["y"]), 3))
                if key in member_keys:
                    continue
                row_bbox = row["bbox"]
                row_y = float(row["y"])
                if not (by0 - NOTES_ROW_EXTEND_MAX_Y_GAP_PT <= row_y <= by1 + NOTES_ROW_EXTEND_MAX_Y_GAP_PT):
                    continue
                overlap = max(
                    0.0,
                    min(float(row_bbox["x"]) + float(row_bbox["w"]), float(bbox["x"]) + float(bbox["w"]))
                    - max(float(row_bbox["x"]), float(bbox["x"])),
                )
                if overlap < float(row_bbox["w"]) * 0.6:
                    continue
                candidate_left = _median([float(r["x_left"]) for r in candidate["rows"]])
                if abs(float(row["x_left"]) - candidate_left) > 120.0:
                    continue
                if _row_context_poisons(row, long_lines, graze=NOTES_ROW_EXTEND_GRAZE_PT):
                    continue
                rows, left_clamp = _clamp_left_outlier_rows(
                    _dedupe_vector_text_rows([*candidate["rows"], row])
                )
                new_bbox = _clamped_rows_union_bbox(rows, clamp=left_clamp, pad=4.0)
                if any(
                    _bbox_intersection_area(new_bbox, blocker.get("bbox")) > 0.0
                    for blocker in blockers
                ):
                    continue
                if (
                    new_bbox["w"] * new_bbox["h"]
                    > page_area_value * VECTOR_TEXT_NOTES_OVERLAP_MERGE_MAX_PAGE_AREA_RATIO
                ):
                    continue
                candidate["rows"] = rows
                candidate["bbox"] = new_bbox
                member_keys.add(key)
                changed = True
    return candidates


def _deoverlap_note_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove residual overlaps between final regions.

    When two accepted regions still intersect (merge was blocked, e.g. by a
    blocker inside the union rectangle), the smaller region drops the rows
    whose centers fall inside the larger region and shrinks; it disappears if
    too few rows remain.
    """
    ordered = sorted(
        candidates,
        key=lambda item: -float(item["bbox"]["w"]) * float(item["bbox"]["h"]),
    )
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        shrunk = candidate
        for bigger in kept:
            if _bbox_intersection_area(shrunk["bbox"], bigger["bbox"]) <= 1.0:
                continue
            rows = [
                row for row in shrunk["rows"]
                if not _center_in_bbox(
                    (
                        float(row["bbox"]["x"]) + float(row["bbox"]["w"]) / 2.0,
                        float(row["bbox"]["y"]) + float(row["bbox"]["h"]) / 2.0,
                    ),
                    bigger["bbox"],
                )
            ]
            if len(rows) < VECTOR_TEXT_NOTES_TIER2_MIN_ROWS:
                shrunk = None
                break
            clamped, left_clamp = _clamp_left_outlier_rows(rows)
            if len(clamped) < VECTOR_TEXT_NOTES_TIER2_MIN_ROWS:
                shrunk = None
                break
            shrunk["rows"] = clamped
            shrunk["bbox"] = _clamped_rows_union_bbox(clamped, clamp=left_clamp, pad=4.0)
            if _bbox_intersection_area(shrunk["bbox"], bigger["bbox"]) > 0.0:
                # Residual sliver from row bboxes poking past the border:
                # clip geometrically along y against the bigger region.
                bbox = shrunk["bbox"]
                big = bigger["bbox"]
                big_y1 = float(big["y"]) + float(big["h"])
                y0 = float(bbox["y"])
                y1 = y0 + float(bbox["h"])
                if y0 + (y1 - y0) / 2.0 >= float(big["y"]) + float(big["h"]) / 2.0:
                    y0 = max(y0, big_y1 + 0.5)
                else:
                    y1 = min(y1, float(big["y"]) - 0.5)
                if y1 - y0 < 8.0:
                    shrunk = None
                    break
                shrunk["bbox"] = {"x": bbox["x"], "y": y0, "w": bbox["w"], "h": y1 - y0}
        if shrunk is not None:
            kept.append(shrunk)
    kept.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    return kept


def _attach_note_fragments(
    candidates: list[dict[str, Any]],
    fragments: list[list[dict[str, Any]]],
    *,
    blockers: list[dict[str, Any]],
    page_area: dict[str, float],
    page_area_value: float,
) -> list[dict[str, Any]]:
    if not candidates or not fragments:
        return candidates
    for fragment in fragments:
        frag_bbox = _union_bboxes([row["bbox"] for row in fragment], pad=4.0)
        target = None
        for candidate in candidates:
            bbox = candidate["bbox"]
            gap = max(
                frag_bbox["y"] - (bbox["y"] + bbox["h"]),
                bbox["y"] - (frag_bbox["y"] + frag_bbox["h"]),
                0.0,
            )
            if gap > VECTOR_TEXT_NOTES_MERGE_Y_GAP:
                continue
            overlap = max(
                0.0,
                min(frag_bbox["x"] + frag_bbox["w"], bbox["x"] + bbox["w"])
                - max(frag_bbox["x"], bbox["x"]),
            )
            if overlap < frag_bbox["w"] * 0.6:
                continue
            frag_left = _median([float(r["x_left"]) for r in fragment])
            candidate_left = _median([float(r["x_left"]) for r in candidate["rows"]])
            if abs(frag_left - candidate_left) > 120.0:
                continue
            target = candidate
            break
        if target is None:
            continue
        rows, left_clamp = _clamp_left_outlier_rows(
            _dedupe_vector_text_rows([*target["rows"], *fragment])
        )
        new_bbox = _clamped_rows_union_bbox(rows, clamp=left_clamp, pad=4.0)
        if any(_bbox_intersection_area(new_bbox, blocker.get("bbox")) > 0.0 for blocker in blockers):
            continue
        if (
            new_bbox["w"] * new_bbox["h"]
            > page_area_value * VECTOR_TEXT_NOTES_OVERLAP_MERGE_MAX_PAGE_AREA_RATIO
        ):
            continue
        target["rows"] = rows
        target["bbox"] = new_bbox
    return candidates


def _clamp_left_outlier_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float | None]:
    """Column left edge from the row x_left consensus.

    View-side text (dimension labels, balloons, FCF boxes) that slipped into a
    column group sits left of the real note text. The consensus left edge is
    the low quantile of member x_left values; rows fully left of it are
    dropped, rows bridging across it keep only their right part in the bbox
    (see _clamped_rows_union_bbox). The low quantile keeps bimodal merges
    (Chinese + English twin columns) intact, while a handful of leaked view
    rows cannot move it.
    """
    if not rows:
        return rows, None
    x_lefts = [float(row["x_left"]) for row in rows]
    clamp = _quantile(x_lefts, NOTES_COLUMN_LEFT_CLAMP_QUANTILE) - NOTES_COLUMN_LEFT_CLAMP_SLACK_PT
    kept = [row for row in rows if float(row["x_right"]) >= clamp]
    return kept, clamp


def _clamped_rows_union_bbox(
    rows: list[dict[str, Any]],
    *,
    clamp: float | None,
    pad: float,
) -> dict[str, float]:
    boxes = []
    for row in rows:
        bbox = row["bbox"]
        x0 = float(bbox["x"])
        x1 = x0 + float(bbox["w"])
        if clamp is not None and x0 < clamp:
            boxes.append({"x": clamp, "y": bbox["y"], "w": max(x1 - clamp, 0.0), "h": bbox["h"]})
        else:
            boxes.append(bbox)
    return _union_bboxes(boxes, pad=pad)


def _dedupe_vector_text_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for row in sorted(rows, key=lambda item: (item["y"], item["x_left"], item["width"])):
        bbox = row["bbox"]
        key = (
            round(float(bbox["x"]), 2),
            round(float(bbox["y"]), 2),
            round(float(bbox["w"]), 2),
            round(float(bbox["h"]), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _vector_text_note_anchor_tagging_policy(
    candidate: dict[str, Any],
    *,
    page_area: dict[str, float],
) -> dict[str, Any]:
    grid = candidate.get("grid_line_count") or {}
    row_count = max(len(candidate.get("rows") or []), 1)
    dense_linework = (
        int(grid.get("vertical") or 0) >= 20
        and int(grid.get("short_horizontal") or 0) / row_count >= 6.0
        and int(grid.get("short_vertical") or 0) / row_count >= 3.0
    )
    if dense_linework:
        return {
            "anchor_tagging": "disabled",
            "anchor_tagging_reason": "dense_note_linework",
        }

    if _is_lower_right_reference_panel(candidate, page_area=page_area):
        # Reference-definition values remain in the candidate pool; the assembly
        # layer decides whether they are dimensions. Keep the notes region for
        # diagnostics without tagging its anchors as notes.
        return {
            "anchor_tagging": "disabled",
            "anchor_tagging_reason": "lower_right_reference_panel",
        }

    anchor_bbox = _dominant_lower_right_anchor_tagging_bbox(candidate, page_area=page_area)
    if anchor_bbox is not None:
        return {
            "anchor_tagging": "limited",
            "anchor_tagging_bbox": _round_bbox(anchor_bbox),
        }
    if (
        _is_dominant_lower_right_note_panel(candidate, page_area=page_area)
        and len(candidate.get("rows") or []) < VECTOR_TEXT_NOTES_MIN_SUBSTANTIAL_ROWS
    ):
        # Small lower-right reference-definition panels enter notes for
        # extraction, but anchor isolation stays off; reference values stay in
        # the candidate pool and are filtered later by assembly.
        return {
            "anchor_tagging": "disabled",
            "anchor_tagging_reason": "lower_right_reference_panel",
        }
    return {"anchor_tagging": "enabled"}


def _is_lower_right_reference_panel(
    candidate: dict[str, Any],
    *,
    page_area: dict[str, float],
) -> bool:
    if len(candidate.get("rows") or []) < VECTOR_TEXT_NOTES_PAGE_GATE_MIN_ROWS:
        return False
    bbox = candidate.get("bbox") or {}
    try:
        page_x = float(page_area["x"])
        page_y = float(page_area["y"])
        page_w = max(float(page_area["w"]), 1e-6)
        page_h = max(float(page_area["h"]), 1e-6)
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return False
    center_x_ratio = (x + w / 2.0 - page_x) / page_w
    top_y_ratio = (y - page_y) / page_h
    bottom_y_ratio = (y + h - page_y) / page_h
    height_ratio = h / page_h
    return (
        center_x_ratio >= VECTOR_TEXT_NOTES_DOMINANT_LOWER_RIGHT_MIN_X_RATIO
        and VECTOR_TEXT_NOTES_REFERENCE_PANEL_MIN_TOP_Y_RATIO
        <= top_y_ratio
        < VECTOR_TEXT_NOTES_REFERENCE_PANEL_MAX_TOP_Y_RATIO
        and bottom_y_ratio >= VECTOR_TEXT_NOTES_REFERENCE_PANEL_MIN_BOTTOM_Y_RATIO
        and height_ratio >= VECTOR_TEXT_NOTES_REFERENCE_PANEL_MIN_HEIGHT_RATIO
    )


def _dominant_lower_right_anchor_tagging_bbox(
    candidate: dict[str, Any],
    *,
    page_area: dict[str, float],
) -> dict[str, float] | None:
    if not _is_dominant_lower_right_note_panel(candidate, page_area=page_area):
        return None
    rows = sorted(candidate.get("rows") or [], key=lambda item: item["y"])
    if len(rows) < VECTOR_TEXT_NOTES_MIN_SUBSTANTIAL_ROWS:
        return None

    best_gap = 0.0
    best_index = None
    for index in range(len(rows) - 1):
        top_count = index + 1
        trailing_count = len(rows) - top_count
        if trailing_count < VECTOR_TEXT_NOTES_ANCHOR_TAG_MIN_TRAILING_ROWS:
            continue
        gap = float(rows[index + 1]["y"]) - float(rows[index]["y"])
        if gap > best_gap:
            best_gap = gap
            best_index = index
    if best_index is None or best_gap < VECTOR_TEXT_NOTES_ANCHOR_TAG_SPLIT_MIN_GAP:
        return None

    trailing_rows = rows[best_index + 1:]
    return _union_bboxes([row["bbox"] for row in trailing_rows], pad=4.0)


def _filter_vector_text_note_candidates_for_page(
    candidates: list[dict[str, Any]],
    *,
    page_area: dict[str, float],
) -> list[dict[str, Any]]:
    if (
        float(page_area["w"]) < VECTOR_TEXT_NOTES_LARGE_PAGE_MIN_WIDTH
        or float(page_area["h"]) < VECTOR_TEXT_NOTES_LARGE_PAGE_MIN_HEIGHT
    ):
        return candidates
    if not candidates:
        return []

    if max(len(candidate["rows"]) for candidate in candidates) < VECTOR_TEXT_NOTES_PAGE_GATE_MIN_ROWS:
        return []

    dominant_lower_right = [
        candidate for candidate in candidates
        if _is_dominant_lower_right_note_panel(candidate, page_area=page_area)
    ]
    if dominant_lower_right:
        return [
            candidate for candidate in candidates
            if candidate in dominant_lower_right
            or len(candidate["rows"]) >= VECTOR_TEXT_NOTES_MIN_ROWS
        ]
    return candidates


def _is_dominant_lower_right_note_panel(
    candidate: dict[str, Any],
    *,
    page_area: dict[str, float],
    min_rows: int = VECTOR_TEXT_NOTES_PAGE_GATE_MIN_ROWS,
) -> bool:
    if len(candidate["rows"]) < min_rows:
        return False
    bbox = candidate["bbox"]
    page_x = float(page_area["x"])
    page_y = float(page_area["y"])
    page_w = max(float(page_area["w"]), 1e-6)
    page_h = max(float(page_area["h"]), 1e-6)
    center_x_ratio = (float(bbox["x"]) + float(bbox["w"]) / 2.0 - page_x) / page_w
    top_y_ratio = (float(bbox["y"]) - page_y) / page_h
    return (
        center_x_ratio >= VECTOR_TEXT_NOTES_DOMINANT_LOWER_RIGHT_MIN_X_RATIO
        and top_y_ratio >= VECTOR_TEXT_NOTES_DOMINANT_LOWER_RIGHT_MIN_Y_RATIO
    )


def _vector_text_note_candidates_should_merge(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    *,
    page_area: dict[str, float],
    page_area_value: float,
    blockers: list[dict[str, Any]],
) -> bool:
    a = existing["bbox"]
    b = candidate["bbox"]
    intersection = _bbox_intersection_area(a, b)
    min_area = max(min(a["w"] * a["h"], b["w"] * b["h"]), 1e-6)
    if intersection >= min_area * VECTOR_TEXT_NOTES_OVERLAP_MERGE_MIN_RATIO:
        # Two accepted note candidates covering the same text (e.g. the twin
        # Chinese/English columns on note sheets) must not stay as
        # overlapping regions; merge them under the relaxed area cap.
        union = _union_bboxes([a, b], pad=0.0)
        if (
            union["w"] * union["h"]
            <= page_area_value * VECTOR_TEXT_NOTES_OVERLAP_MERGE_MAX_PAGE_AREA_RATIO
            and not any(
                _bbox_intersection_area(union, blocker.get("bbox")) > 0.0
                for blocker in blockers
            )
        ):
            return True
    horizontal_overlap = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    min_width = max(min(a["w"], b["w"]), 1e-6)
    same_column = horizontal_overlap / min_width >= 0.45
    if not same_column:
        a_cx = a["x"] + a["w"] / 2.0
        b_cx = b["x"] + b["w"] / 2.0
        same_column = abs(a_cx - b_cx) <= page_area["w"] * 0.08
    if not same_column:
        return False
    gap = max(b["y"] - (a["y"] + a["h"]), a["y"] - (b["y"] + b["h"]), 0.0)
    if gap > VECTOR_TEXT_NOTES_MERGE_Y_GAP:
        return False
    union = _union_bboxes([a, b], pad=0.0)
    if union["w"] * union["h"] > page_area_value * VECTOR_TEXT_NOTES_MAX_PAGE_AREA_RATIO:
        return False
    if union["w"] > page_area["w"] * 0.68:
        return False
    return not any(_bbox_intersection_area(union, blocker.get("bbox")) > 0.0 for blocker in blockers)


def _table_grid_line_count_in_bbox(
    lines: list[_Line],
    bbox: dict[str, float],
    *,
    precomputed: dict[str, int] | None = None,
) -> dict[str, int]:
    if precomputed is not None:
        return {
            "horizontal": int(precomputed.get("horizontal") or 0),
            "vertical": int(precomputed.get("vertical") or 0),
            "short_horizontal": int(precomputed.get("short_horizontal") or 0),
            "short_vertical": int(precomputed.get("short_vertical") or 0),
        }
    horizontal = 0
    vertical = 0
    short_horizontal = 0
    short_vertical = 0
    min_h_overlap = max(float(bbox["w"]) * 0.35, 90.0)
    min_v_overlap = max(float(bbox["h"]) * 0.35, 60.0)
    for line in lines:
        if line.horizontal and bbox["y"] - 2.0 <= line.y <= bbox["y"] + bbox["h"] + 2.0:
            overlap = _h_overlap_line_bbox(line, bbox)
            if overlap >= min_h_overlap:
                horizontal += 1
            elif overlap >= 10.0:
                short_horizontal += 1
        elif line.vertical and bbox["x"] - 2.0 <= line.x <= bbox["x"] + bbox["w"] + 2.0:
            overlap = _v_overlap_line_bbox(line, bbox)
            if overlap >= min_v_overlap:
                vertical += 1
            elif overlap >= 10.0:
                short_vertical += 1
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "short_horizontal": short_horizontal,
        "short_vertical": short_vertical,
    }


def _is_table_grid_like(grid_line_count: dict[str, int]) -> bool:
    return (
        (
            int(grid_line_count.get("horizontal") or 0) >= VECTOR_TEXT_NOTES_GRID_MIN_HORIZONTAL_LINES
            and int(grid_line_count.get("vertical") or 0) >= VECTOR_TEXT_NOTES_GRID_MIN_VERTICAL_LINES
        )
        or (
            int(grid_line_count.get("horizontal") or 0) >= 8
            and int(grid_line_count.get("vertical") or 0) >= 1
        )
        or (
            int(grid_line_count.get("short_horizontal") or 0) >= VECTOR_TEXT_NOTES_GRID_MIN_SHORT_HORIZONTAL_LINES
            and int(grid_line_count.get("short_vertical") or 0) >= VECTOR_TEXT_NOTES_GRID_MIN_SHORT_VERTICAL_LINES
        )
    )


def _vector_text_row_descriptors(lines: list[_Line]) -> list[dict[str, Any]]:
    small = [line for line in lines if 0.4 <= line.length <= 18.0]
    y_rows: list[list[_Line]] = []
    y_values: list[float] = []
    for line in sorted(small, key=lambda item: item.y):
        if y_rows and abs(y_values[-1] - line.y) <= VECTOR_TEXT_ROW_Y_TOLERANCE:
            y_rows[-1].append(line)
            y_values[-1] = sum(item.y for item in y_rows[-1]) / len(y_rows[-1])
        else:
            y_rows.append([line])
            y_values.append(line.y)
    descriptors = []
    for y, row_lines in zip(y_values, y_rows, strict=True):
        clusters: list[list[_Line]] = []
        for line in sorted(row_lines, key=lambda item: item.x):
            if clusters and line.x - clusters[-1][-1].x <= VECTOR_TEXT_ROW_X_GAP:
                clusters[-1].append(line)
            else:
                clusters.append([line])
        for cluster in clusters:
            if len(cluster) < 10:
                continue
            bbox = _line_union_bbox(cluster)
            width = bbox["w"]
            if width < 60.0:
                continue
            descriptors.append({
                "y": y,
                "bbox": bbox,
                "x_left": bbox["x"],
                "x_right": bbox["x"] + bbox["w"],
                "width": width,
                "segment_count": len(cluster),
            })
    return descriptors


def _aligned_vector_text_row_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: (item["y"], item["x_left"])):
        best_index = None
        best_score = None
        for index, group in enumerate(groups):
            last = group[-1]
            y_gap = row["y"] - last["y"]
            if y_gap < 1.0 or y_gap > 36.0:
                continue
            x_lefts = [item["x_left"] for item in group]
            x_delta = abs(row["x_left"] - _median(x_lefts))
            if x_delta > VECTOR_TEXT_NOTES_MAX_X_LEFT_SPREAD:
                continue
            score = y_gap + x_delta * 0.1
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        if best_index is None:
            groups.append([row])
        else:
            groups[best_index].append(row)
    return groups


def _loose_vector_text_column_groups(
    rows: list[dict[str, Any]],
    *,
    page_area: dict[str, float],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(rows, key=lambda item: (item["y"], item["x_left"])):
        best_index = None
        best_score = None
        for index, group in enumerate(groups):
            last = group[-1]
            y_gap = float(row["y"]) - float(last["y"])
            if y_gap < 1.0 or y_gap > 36.0:
                continue
            group_bbox = _union_bboxes([item["bbox"] for item in group], pad=0.0)
            row_bbox = row["bbox"]
            union = _union_bboxes([group_bbox, row_bbox], pad=0.0)
            if union["w"] > page_area["w"] * 0.62:
                continue
            overlap = max(
                0.0,
                min(group_bbox["x"] + group_bbox["w"], row_bbox["x"] + row_bbox["w"])
                - max(group_bbox["x"], row_bbox["x"]),
            )
            min_width = max(min(group_bbox["w"], row_bbox["w"]), 1e-6)
            center_delta = abs(
                (group_bbox["x"] + group_bbox["w"] / 2.0)
                - (row_bbox["x"] + row_bbox["w"] / 2.0)
            )
            same_column = overlap / min_width >= 0.12 or center_delta <= page_area["w"] * 0.18
            if not same_column:
                continue
            score = y_gap + center_delta * 0.01
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        if best_index is None:
            groups.append([row])
        else:
            groups[best_index].append(row)
    return [group for group in groups if len(group) >= VECTOR_TEXT_NOTES_FRAGMENT_MIN_ROWS]


def _dedupe_layout_blocks(blocks: list[dict[str, Any]], *, iou_threshold: float = 0.70) -> list[dict[str, Any]]:
    deduped = []
    for block in sorted(blocks, key=lambda row: -(row["bbox"]["w"] * row["bbox"]["h"])):
        if any(_bbox_iou(block.get("bbox"), existing.get("bbox")) > iou_threshold for existing in deduped):
            continue
        deduped.append(block)
    deduped.sort(key=lambda row: (row["bbox"]["y"], row["bbox"]["x"]))
    return deduped


def _anchor_region_health(
    *,
    reference_anchors: list[dict[str, Any]],
    anchor_region_tags: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    anchors = [
        anchor for anchor in reference_anchors
        if str(anchor.get("kind") or "") in {"dot", "degree", "diameter"}
    ]
    total = len(anchors)
    drawing = 0
    for anchor in anchors:
        tag = anchor_region_tags.get(_row_id(anchor)) or {}
        if str(tag.get("region") or "drawing") == "drawing":
            drawing += 1
    ratio = 1.0 if total == 0 else drawing / total
    return {
        "reference_anchor_count": total,
        "drawing_anchor_count": drawing,
        "drawing_anchor_ratio": round(float(ratio), 4),
        "healthy": bool(ratio >= DRAWING_ANCHOR_RATIO_HEALTH_MIN),
    }


def _tag_regions(
    *,
    reference_anchors: list[dict[str, Any]],
    colon_marks: list[dict[str, Any]],
    text_punct_marks: list[dict[str, Any]],
    drawing_area: dict[str, float],
    title_blocks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    notes_regions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tags = {}
    for row in [*reference_anchors, *colon_marks, *text_punct_marks]:
        bbox = _xywh(row)
        center = _center(row, bbox)
        if center is None:
            continue
        region = "drawing"
        region_id = ""
        for note in notes_regions:
            if str(note.get("anchor_tagging") or "enabled") == "disabled":
                continue
            note_bbox = note.get("anchor_tagging_bbox")
            if not isinstance(note_bbox, dict):
                note_bbox = note["bbox"]
            if _center_in_bbox((center["x"], center["y"]), note_bbox):
                region, region_id = "notes", note["region_id"]
                break
        if region == "drawing":
            for title in title_blocks:
                if _center_in_bbox((center["x"], center["y"]), title["bbox"]):
                    region, region_id = "title_block", ""
                    break
        if region == "drawing":
            for table in tables:
                if _center_in_bbox((center["x"], center["y"]), table["bbox"]):
                    region, region_id = "table", ""
                    break
        if region == "drawing" and not _center_in_bbox((center["x"], center["y"]), drawing_area):
            region = "border"
        tag = {
            "region": region,
            "region_id": region_id,
            "center": {"x": round(center["x"], 3), "y": round(center["y"], 3)},
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        if region == "notes":
            line_tag = _note_line_for_center(notes_regions, region_id=region_id, center=center)
            if line_tag is not None:
                tag.update(line_tag)
        tags[_row_id(row)] = tag
    return tags


def _note_line_for_center(
    notes_regions: list[dict[str, Any]],
    *,
    region_id: str,
    center: dict[str, float],
) -> dict[str, Any] | None:
    for note in notes_regions:
        if str(note.get("region_id") or "") != str(region_id):
            continue
        tolerance = float(note.get("line_y_tolerance") or 3.5)
        for index, line in enumerate(note.get("line_baselines") or []):
            try:
                baseline_y = float(line.get("y"))
            except (TypeError, ValueError):
                continue
            if abs(float(center["y"]) - baseline_y) <= tolerance:
                return {
                    "text_line_id": f"{region_id}_line_{index:03d}",
                    "text_line_y": round(baseline_y, 3),
                }
    return None


def _grid_is_regular(
    *,
    xs: list[float],
    ys: list[float],
    bbox: dict[str, float],
    drawing_area: dict[str, float],
    h_lines: list[_Line],
    v_lines: list[_Line],
) -> bool:
    area = max(float(bbox["w"]) * float(bbox["h"]), 0.0)
    drawing_area_value = max(float(drawing_area["w"]) * float(drawing_area["h"]), 1e-6)
    if area > drawing_area_value * GRID_MAX_DRAWING_AREA_RATIO:
        return False
    widths = [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]
    heights = [ys[index + 1] - ys[index] for index in range(len(ys) - 1)]
    if not widths or not heights or min(widths + heights) <= 0:
        return False
    if not _diffs_are_table_like(widths) or not _diffs_are_table_like(heights):
        return False
    expected = len(xs) * len(ys)
    if expected <= 0:
        return False
    hits = 0
    for x in xs:
        for y in ys:
            if _has_horizontal_at(h_lines, x=x, y=y) and _has_vertical_at(v_lines, x=x, y=y):
                hits += 1
    return hits / expected >= 0.70


def _diffs_are_table_like(values: list[float]) -> bool:
    ordered = sorted(float(value) for value in values if value > 0)
    if not ordered:
        return False
    median = ordered[len(ordered) // 2]
    return min(ordered) >= median * 0.20 and max(ordered) <= median * 4.0


def _has_horizontal_at(lines: list[_Line], *, x: float, y: float) -> bool:
    return any(line.horizontal and abs(line.y - y) <= 2.0 and line.x_left - 2.0 <= x <= line.x_right + 2.0 for line in lines)


def _has_vertical_at(lines: list[_Line], *, x: float, y: float) -> bool:
    return any(line.vertical and abs(line.x - x) <= 2.0 and line.y_top - 2.0 <= y <= line.y_bottom + 2.0 for line in lines)


def _has_horizontal_span(lines: list[_Line], *, y: float, x0: float, x1: float) -> bool:
    required = max(float(x1) - float(x0) - 3.0, 0.0)
    return any(
        line.horizontal
        and abs(line.y - y) <= 2.0
        and max(0.0, min(line.x_right, x1) - max(line.x_left, x0)) >= required
        for line in lines
    )


def _has_vertical_span(lines: list[_Line], *, x: float, y0: float, y1: float) -> bool:
    required = max(float(y1) - float(y0) - 3.0, 0.0)
    return any(
        line.vertical
        and abs(line.x - x) <= 2.0
        and max(0.0, min(line.y_bottom, y1) - max(line.y_top, y0)) >= required
        for line in lines
    )


def _estimate_mark_glyph_h(marks: list[dict[str, Any]]) -> float:
    sides = [
        max(float(mark["bbox"]["w"]), float(mark["bbox"]["h"]))
        for mark in marks
        if mark.get("bbox")
    ]
    if not sides:
        return 8.0
    sides.sort()
    return max(sides[len(sides) // 2] * 3.0, 8.0)


def _rows_with_text_line_pitch(rows: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    if len(rows) < NOTES_MIN_ROWS:
        return rows
    out = [rows[0]]
    for row in rows[1:]:
        gap = min(mark["center"]["y"] for mark in row) - min(mark["center"]["y"] for mark in out[-1])
        glyph_h = _estimate_mark_glyph_h(out[-1])
        if glyph_h * 0.75 <= gap <= glyph_h * 2.5:
            out.append(row)
    return out


def _long_line_density_in_bbox(lines: list[_Line], bbox: dict[str, float], *, glyph_h: float) -> int:
    count = 0
    for line in lines:
        if line.length < glyph_h * 6.0:
            continue
        if _center_in_bbox((line.x, line.y), bbox):
            count += 1
    return count


def _line_union_bbox(lines: list[_Line]) -> dict[str, float]:
    return {
        "x": min(line.x_left for line in lines),
        "y": min(line.y_top for line in lines),
        "w": max(line.x_right for line in lines) - min(line.x_left for line in lines),
        "h": max(line.y_bottom for line in lines) - min(line.y_top for line in lines),
    }


def _median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * max(0.0, min(float(fraction), 1.0)))
    return ordered[int(index)]


def _cluster_axis_values(values: list[float], *, tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(float(item) for item in values):
        if clusters and abs(clusters[-1][-1] - value) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _grid_kind(bbox: dict[str, float], drawing_area: dict[str, float]) -> str:
    cx = bbox["x"] + bbox["w"] / 2.0
    cy = bbox["y"] + bbox["h"] / 2.0
    rightish = cx >= drawing_area["x"] + drawing_area["w"] * 0.58
    bottomish = cy >= drawing_area["y"] + drawing_area["h"] * 0.58
    return "title_block" if rightish and bottomish else "table"


def _line_from_segment(segment: Any) -> _Line | None:
    try:
        if hasattr(segment, "p0") and hasattr(segment, "p1"):
            return _Line(float(segment.p0[0]), float(segment.p0[1]), float(segment.p1[0]), float(segment.p1[1]))
        if isinstance(segment, dict):
            return _Line(float(segment["x0"]), float(segment["y0"]), float(segment["x1"]), float(segment["y1"]))
        if isinstance(segment, (tuple, list)) and len(segment) >= 4:
            return _Line(float(segment[0]), float(segment[1]), float(segment[2]), float(segment[3]))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _page_size(page_rect: Any) -> tuple[float, float]:
    if hasattr(page_rect, "width") and hasattr(page_rect, "height"):
        return float(page_rect.width), float(page_rect.height)
    if isinstance(page_rect, dict):
        return float(page_rect.get("w") or page_rect.get("width")), float(page_rect.get("h") or page_rect.get("height"))
    if isinstance(page_rect, (tuple, list)) and len(page_rect) >= 2:
        return float(page_rect[0]), float(page_rect[1])
    raise ValueError("page_rect must expose width/height")


def _intersects_hv(h: _Line, v: _Line) -> bool:
    return h.x_left - 2 <= v.x <= h.x_right + 2 and v.y_top - 2 <= h.y <= v.y_bottom + 2


def _h_overlap(a: _Line, b: _Line) -> float:
    return max(0.0, min(a.x_right, b.x_right) - max(a.x_left, b.x_left))


def _h_overlap_line_bbox(line: _Line, bbox: dict[str, float]) -> float:
    return max(0.0, min(line.x_right, bbox["x"] + bbox["w"]) - max(line.x_left, bbox["x"]))


def _v_overlap_line_bbox(line: _Line, bbox: dict[str, float]) -> float:
    return max(0.0, min(line.y_bottom, bbox["y"] + bbox["h"]) - max(line.y_top, bbox["y"]))


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("anchor_id") or row.get("mark_id") or row.get("colon_id") or row.get("id") or row.get("source_id") or "")


def _xywh(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("bbox") if isinstance(value.get("bbox"), dict) else value
    try:
        if all(key in raw for key in ("x", "y", "w", "h")):
            return {key: float(raw[key]) for key in ("x", "y", "w", "h")}
        if all(key in raw for key in ("x0", "y0", "x1", "y1")):
            x0 = float(raw["x0"])
            y0 = float(raw["y0"])
            x1 = float(raw["x1"])
            y1 = float(raw["y1"])
            return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    except (TypeError, ValueError):
        return None
    return None


def _center(row: dict[str, Any], bbox: dict[str, float] | None) -> dict[str, float] | None:
    center = row.get("center") if isinstance(row, dict) else None
    if isinstance(center, dict) and "x" in center and "y" in center:
        return {"x": float(center["x"]), "y": float(center["y"])}
    if bbox is None:
        return None
    return {"x": bbox["x"] + bbox["w"] / 2.0, "y": bbox["y"] + bbox["h"] / 2.0}


def _center_in_bbox(point: tuple[float, float], bbox: dict[str, float]) -> bool:
    return bbox["x"] <= point[0] <= bbox["x"] + bbox["w"] and bbox["y"] <= point[1] <= bbox["y"] + bbox["h"]


def _union_bboxes(bboxes: list[dict[str, float]], *, pad: float = 0.0) -> dict[str, float]:
    x0 = min(bbox["x"] for bbox in bboxes) - pad
    y0 = min(bbox["y"] for bbox in bboxes) - pad
    x1 = max(bbox["x"] + bbox["w"] for bbox in bboxes) + pad
    y1 = max(bbox["y"] + bbox["h"] for bbox in bboxes) + pad
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {key: round(float(bbox[key]), 3) for key in ("x", "y", "w", "h")}


def _bbox_iou(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    if not a or not b:
        return 0.0
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


def _bbox_intersection_area(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    if not a or not b:
        return 0.0
    try:
        x0 = max(float(a["x"]), float(b["x"]))
        y0 = max(float(a["y"]), float(b["y"]))
        x1 = min(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"]))
        y1 = min(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)
