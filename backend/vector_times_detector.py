"""Dump-only vector times/quantity separator detector.

The detector looks for an X-shaped glyph made from two crossing, near-equal
line strokes, then keeps only candidates that have a decimal-point anchor on
the text axis and local count-side ink on the opposite side.  It is diagnostic
only: rows are not consumed by OCR, assembler, or final dimensions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from decimal_point_quad import detect_decimal_point_quads
from vector_draw_items import ItemPrimitive, extract_items_from_page, item_length
from vector_glyph_contact_geometry import build_contact_graph
from vector_times_glyph import match_vector_times_glyph


SCHEMA_VERSION = "vector_times_detector_v1"
RAW_SCHEMA_VERSION = "vector_times_raw_cross_candidates_v1"
STATS_SCHEMA_VERSION = "vector_times_detector_stats_v1"
CONSUMER_ALLOWED = False

RAW_MIN_LINE_LENGTH_PT = 2.5
RAW_MAX_LINE_LENGTH_PT = 25.0
RAW_MAX_LINE_BBOX_SIDE_PT = 25.0
RAW_MAX_LEN_RATIO = 1.35
STRICT_MAX_LEN_RATIO = 1.25
RAW_MIN_ANGLE_DELTA_DEG = 35.0
RAW_MAX_ANGLE_DELTA_DEG = 90.000001
RAW_MIN_GLYPH_SIDE_PT = 2.5
RAW_MAX_GLYPH_SIDE_PT = 18.0
RAW_MAX_GLYPH_ASPECT = 2.2
PAIR_QUERY_CELL_PT = 16.0

LOCAL_INTRUDER_PAD_FACTOR = 0.0
LOCAL_INTRUDER_MAX_FACTOR = 2.2
LOCAL_INTRUDER_MIN_LENGTH_FACTOR = 0.12

DOT_MIN_MAIN_FACTOR = 3.0
DOT_MAX_MAIN_FACTOR = 7.0
DOT_MAX_CROSS_FACTOR = 0.90
DOT_CROSS_ABS_TOL_PT = 0.25
DOT_MAX_LONG_FACTOR = 0.55

COUNT_MIN_MAIN_FACTOR = 0.15
COUNT_MAX_MAIN_FACTOR = 3.2
COUNT_MAX_CROSS_FACTOR = 1.15
COUNT_MAX_SIDE_FACTOR = 2.2
COUNT_MIN_SIDE_FACTOR = 0.12
COUNT_LONGISH_LENGTH_FACTOR = 0.65
COUNT_MAX_LOCAL_PRIMITIVE_COUNT = 32
COUNT_MAX_COMPONENT_PRIMITIVE_COUNT = 24
COUNT_FRAGMENTED_MIN_PRIMITIVE_COUNT = 2
COUNT_FRAGMENTED_MIN_TOTAL_LENGTH_FACTOR = 1.2
COUNT_COMPONENT_MIN_MAIN_SPAN_FACTOR = 0.08
COUNT_COMPONENT_MAX_MAIN_SPAN_FACTOR = 1.25
COUNT_COMPONENT_MIN_CROSS_SPAN_FACTOR = 0.55
COUNT_COMPONENT_MAX_CROSS_SPAN_FACTOR = 1.25
COUNT_MAX_CHARACTER_LIKE_COMPONENTS = 3

def detect_vector_times_candidates(
    page: Any,
    *,
    page_index: int = 0,
    pdf_stem: str = "page",
    decimal_point_rows: list[dict[str, Any]] | None = None,
    item_primitives: list[ItemPrimitive] | tuple[ItemPrimitive, ...] | None = None,
) -> dict[str, Any]:
    """Detect vector X/× separator candidates on a page."""
    items = (
        list(item_primitives)
        if item_primitives is not None
        else extract_items_from_page(
            page,
            pdf_stem=pdf_stem,
            page_num=page_index + 1,
        )
    )
    if decimal_point_rows is None:
        decimal_dump = detect_decimal_point_quads(
            page,
            page_index=page_index,
            pdf_stem=pdf_stem,
        )
        decimal_point_rows = decimal_dump.get("vector_decimal_point_quad_v1", [])
    return build_vector_times_detector_dump(
        items=items,
        decimal_point_rows=decimal_point_rows,
        page_index=page_index,
    )


def build_vector_times_detector_dump(
    *,
    items: list[Any],
    decimal_point_rows: list[dict[str, Any]],
    page_index: int = 0,
) -> dict[str, Any]:
    raw_rows = _raw_cross_candidates(items=items, page_index=page_index)
    strict_rows: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    for raw in raw_rows:
        strict, reasons = _strict_candidate_from_raw(
            raw,
            items=items,
            decimal_point_rows=decimal_point_rows,
            page_index=page_index,
            row_index=len(strict_rows),
        )
        if strict is not None:
            strict_rows.append(strict)
        else:
            drop_reasons.update(reasons)
    stats = _stats(
        raw_rows=raw_rows,
        strict_rows=strict_rows,
        drop_reasons=drop_reasons,
        page_index=page_index,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "consumer_allowed": CONSUMER_ALLOWED,
        "page_index": int(page_index),
        "ocr_results_count": 0,
        "ocr_called": False,
        "vector_times_raw_cross_candidates_v1": raw_rows,
        "vector_times_candidates_v1": strict_rows,
        "vector_times_detector_stats_v1": stats,
        "definition": {
            "raw": [
                "two straight line strokes",
                "segments intersect",
                "near-equal stroke length",
                "X glyph matches one of the observed pure-vector dictionary signatures",
                "line angle, bbox aspect, and opposite-corner coverage agree",
                "intersection is central on both strokes and near the glyph bbox center",
                "same drawing and adjacent source items when source identity is available",
                "compact glyph bbox",
            ],
            "strict": [
                "strict equal-length X glyph",
                "no local intruder primitive inside the X glyph bbox",
                "decimal-point anchor on horizontal or vertical text axis",
                "decimal main-axis distance scaled by X glyph height",
                "decimal cross-axis offset scaled by X glyph height",
                "opposite-side local count ink on the same axis",
            ],
            "consumer_policy": "dump_only_false_for_all_rows",
        },
    }


def _raw_cross_candidates(*, items: list[Any], page_index: int) -> list[dict[str, Any]]:
    lines = _candidate_lines(items)
    buckets = _line_buckets(lines)
    rows: list[dict[str, Any]] = []
    for left_pos, left in enumerate(lines):
        nearby = _nearby_line_positions(left["bbox"], buckets)
        for right_pos in sorted(nearby):
            if right_pos <= left_pos:
                continue
            row = _raw_pair_row(
                left,
                lines[right_pos],
                page_index=page_index,
                row_index=len(rows),
            )
            if row is not None:
                rows.append(row)
    return _dedupe_raw_rows(rows)


def _candidate_lines(items: list[Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        if str(getattr(item, "op", "")) != "l":
            continue
        bbox = _bbox_tuple(getattr(item, "bbox", None))
        if bbox is None:
            continue
        length = item_length(item)
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)
        if not (
            RAW_MIN_LINE_LENGTH_PT <= length <= RAW_MAX_LINE_LENGTH_PT
            and max(width, height) <= RAW_MAX_LINE_BBOX_SIDE_PT
            and min(width, height) >= 0.05
        ):
            continue
        lines.append({
            "item_index": item_index,
            "item": item,
            "bbox": bbox,
            "length": length,
            "angle_deg": _line_angle(item),
        })
    return lines


def _line_buckets(lines: list[dict[str, Any]]) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for position, line in enumerate(lines):
        bbox = _expand_bbox(line["bbox"], 2.0)
        for cell in _cells_for_bbox(bbox):
            buckets.setdefault(cell, []).append(position)
    return buckets


def _nearby_line_positions(
    bbox: tuple[float, float, float, float],
    buckets: dict[tuple[int, int], list[int]],
) -> set[int]:
    out: set[int] = set()
    for cell in _cells_for_bbox(_expand_bbox(bbox, 2.0)):
        out.update(buckets.get(cell, []))
    return out


def _cells_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = bbox
    cells: list[tuple[int, int]] = []
    for ix in range(int(x0 // PAIR_QUERY_CELL_PT), int(x1 // PAIR_QUERY_CELL_PT) + 1):
        for iy in range(int(y0 // PAIR_QUERY_CELL_PT), int(y1 // PAIR_QUERY_CELL_PT) + 1):
            cells.append((ix, iy))
    return cells


def _raw_pair_row(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    page_index: int,
    row_index: int,
) -> dict[str, Any] | None:
    if not _intersects(_expand_bbox(left["bbox"], 1.2), right["bbox"]):
        return None
    len_ratio = max(left["length"], right["length"]) / max(
        min(left["length"], right["length"]),
        1e-9,
    )
    if len_ratio > RAW_MAX_LEN_RATIO:
        return None
    angle_delta = _angle_delta(left["angle_deg"], right["angle_deg"])
    if not (RAW_MIN_ANGLE_DELTA_DEG <= angle_delta <= RAW_MAX_ANGLE_DELTA_DEG):
        return None
    bbox = _union_bbox(left["bbox"], right["bbox"])
    width = _bbox_width(bbox)
    height = _bbox_height(bbox)
    if not (
        RAW_MIN_GLYPH_SIDE_PT <= width <= RAW_MAX_GLYPH_SIDE_PT
        and RAW_MIN_GLYPH_SIDE_PT <= height <= RAW_MAX_GLYPH_SIDE_PT
    ):
        return None
    if max(width, height) / max(min(width, height), 1e-9) > RAW_MAX_GLYPH_ASPECT:
        return None
    if not _segments_intersect(
        _endpoints(left["item"])[0],
        _endpoints(left["item"])[1],
        _endpoints(right["item"])[0],
        _endpoints(right["item"])[1],
    ):
        return None
    glyph_dictionary = match_vector_times_glyph(
        [left["item"], right["item"]]
    )
    if glyph_dictionary.get("matched") is not True:
        return None
    center = _bbox_center(bbox)
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "candidate_id": f"p{int(page_index) + 1:03d}_times_raw_{row_index:06d}",
        "page_index": int(page_index),
        "bbox": _xywh(bbox),
        "bbox_xyxy": _bbox_list(bbox),
        "center": {"x": round(center[0], 6), "y": round(center[1], 6)},
        "glyph_height_pt": round(max(width, height), 6),
        "glyph_width_pt": round(width, 6),
        "glyph_bbox_height_pt": round(height, 6),
        "line_lengths_pt": [
            round(float(left["length"]), 6),
            round(float(right["length"]), 6),
        ],
        "line_angles_deg": [
            round(float(left["angle_deg"]), 6),
            round(float(right["angle_deg"]), 6),
        ],
        "len_ratio": round(len_ratio, 6),
        "angle_delta_deg": round(angle_delta, 6),
        "glyph_dictionary": glyph_dictionary,
        "source_item_indices": [
            int(left["item_index"]),
            int(right["item_index"]),
        ],
        "source_segments": [
            _source_segment(left["item_index"], left["item"]),
            _source_segment(right["item_index"], right["item"]),
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _dedupe_raw_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["len_ratio"]),
            -float(row["glyph_height_pt"]),
            float(row["center"]["y"]),
            float(row["center"]["x"]),
        ),
    )
    kept: list[dict[str, Any]] = []
    for row in ordered:
        center = (float(row["center"]["x"]), float(row["center"]["y"]))
        height = float(row["glyph_height_pt"])
        if any(
            _distance(center, (float(existing["center"]["x"]), float(existing["center"]["y"])))
            <= 0.35 * max(height, float(existing["glyph_height_pt"]))
            for existing in kept
        ):
            continue
        row = dict(row)
        row["candidate_id"] = f"p{int(row['page_index']) + 1:03d}_times_raw_{len(kept):06d}"
        kept.append(row)
    return kept


def _strict_candidate_from_raw(
    raw: dict[str, Any],
    *,
    items: list[Any],
    decimal_point_rows: list[dict[str, Any]],
    page_index: int,
    row_index: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if float(raw["len_ratio"]) > STRICT_MAX_LEN_RATIO:
        reasons.append("len_ratio_not_strict_equal")
    intruders = _local_intruders(raw, items)
    if intruders:
        reasons.append("local_x_glyph_not_isolated")
    anchor_options = _anchor_options(raw, decimal_point_rows, items=items)
    if not anchor_options:
        reasons.append("missing_axis_aligned_decimal_anchor")
    if reasons:
        return None, reasons
    anchor = sorted(
        anchor_options,
        key=lambda item: (
            -int((item.get("count_side") or {}).get("longish_count") or 0),
            float(item["score"]),
        ),
    )[0]
    row = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": f"p{int(page_index) + 1:03d}_times_{row_index:06d}",
        "page_index": int(page_index),
        "source_raw_candidate_id": raw["candidate_id"],
        "glyph_type": "times_quantity_separator",
        "text": "×",
        "bbox": raw["bbox"],
        "bbox_xyxy": raw["bbox_xyxy"],
        "center": raw["center"],
        "glyph_height_pt": raw["glyph_height_pt"],
        "raw_geometry": {
            "line_lengths_pt": raw["line_lengths_pt"],
            "line_angles_deg": raw["line_angles_deg"],
            "len_ratio": raw["len_ratio"],
            "angle_delta_deg": raw["angle_delta_deg"],
        },
        "anchor_gate": {
            "passed": True,
            "anchor": anchor,
            "local_intruder_count": len(intruders),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "source_item_indices": raw["source_item_indices"],
        "source_segments": raw["source_segments"],
        "recognition": {
            "ocr_called": False,
            "predicted_text": None,
            "note": "glyph detector only; full dimension text is not recognized here",
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return row, []


def _local_intruders(raw: dict[str, Any], items: list[Any]) -> list[dict[str, Any]]:
    bbox = _bbox_tuple(raw.get("bbox_xyxy"))
    if bbox is None:
        return []
    glyph_height = float(raw["glyph_height_pt"])
    region = _expand_bbox(bbox, LOCAL_INTRUDER_PAD_FACTOR * glyph_height)
    source_indices = {int(value) for value in raw.get("source_item_indices") or []}
    intruders: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if idx in source_indices:
            continue
        if str(getattr(item, "op", "")) not in {"l", "qu", "c"}:
            continue
        item_bbox = _bbox_tuple(getattr(item, "bbox", None))
        if item_bbox is None or not _intersects(region, item_bbox):
            continue
        side = max(_bbox_width(item_bbox), _bbox_height(item_bbox))
        if side > LOCAL_INTRUDER_MAX_FACTOR * glyph_height:
            continue
        if item_length(item) < LOCAL_INTRUDER_MIN_LENGTH_FACTOR * glyph_height:
            continue
        intruders.append(_source_segment(idx, item))
    return intruders


def _anchor_options(
    raw: dict[str, Any],
    decimal_point_rows: list[dict[str, Any]],
    *,
    items: list[Any],
) -> list[dict[str, Any]]:
    cx = float(raw["center"]["x"])
    cy = float(raw["center"]["y"])
    glyph_height = float(raw["glyph_height_pt"])
    options: list[dict[str, Any]] = []
    for dot in decimal_point_rows:
        dot_center = _point_from_center(dot.get("center"))
        dot_bbox = _bbox_tuple(dot.get("bbox"))
        if dot_center is None or dot_bbox is None:
            continue
        dot_long = max(_bbox_width(dot_bbox), _bbox_height(dot_bbox))
        if dot_long > DOT_MAX_LONG_FACTOR * glyph_height:
            continue
        for axis in ("horizontal", "vertical"):
            if axis == "horizontal":
                main = dot_center[0] - cx
                cross = dot_center[1] - cy
            else:
                main = dot_center[1] - cy
                cross = dot_center[0] - cx
            abs_main = abs(main)
            abs_cross = abs(cross)
            if not (
                DOT_MIN_MAIN_FACTOR * glyph_height
                <= abs_main
                <= DOT_MAX_MAIN_FACTOR * glyph_height
            ):
                continue
            if abs_cross > DOT_MAX_CROSS_FACTOR * glyph_height + DOT_CROSS_ABS_TOL_PT:
                continue
            count_side = _count_side_evidence(raw, axis=axis, dot_main=main, items=items)
            if not count_side["character_like_ink"]:
                continue
            options.append({
                "decimal_point_id": str(dot.get("id") or ""),
                "axis": axis,
                "dot_center": {
                    "x": round(dot_center[0], 6),
                    "y": round(dot_center[1], 6),
                },
                "main_delta_pt": round(main, 6),
                "cross_delta_pt": round(cross, 6),
                "main_delta_glyph_heights": round(abs_main / max(glyph_height, 1e-9), 6),
                "cross_delta_glyph_heights": round(abs_cross / max(glyph_height, 1e-9), 6),
                "dot_long_glyph_heights": round(dot_long / max(glyph_height, 1e-9), 6),
                "count_side": count_side,
                "score": round(abs_main / max(glyph_height, 1e-9) + abs_cross / max(glyph_height, 1e-9), 6),
                "scale_basis": "x_glyph_height_pt",
                "consumer_allowed": CONSUMER_ALLOWED,
            })
    return options


def _count_side_evidence(
    raw: dict[str, Any],
    *,
    axis: str,
    dot_main: float,
    items: list[Any],
) -> dict[str, Any]:
    cx = float(raw["center"]["x"])
    cy = float(raw["center"]["y"])
    glyph_height = float(raw["glyph_height_pt"])
    dot_sign = 1.0 if dot_main >= 0.0 else -1.0
    source_indices = {int(value) for value in raw.get("source_item_indices") or []}
    primitive_count = 0
    longish_count = 0
    examples: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if idx in source_indices:
            continue
        if str(getattr(item, "op", "")) not in {"l", "qu", "c"}:
            continue
        item_bbox = _bbox_tuple_allow_degenerate(getattr(item, "bbox", None))
        if item_bbox is None:
            continue
        item_center = _bbox_center(item_bbox)
        if axis == "horizontal":
            main = item_center[0] - cx
            cross = item_center[1] - cy
        else:
            main = item_center[1] - cy
            cross = item_center[0] - cx
        opposite_main = main * dot_sign
        if not (
            -COUNT_MAX_MAIN_FACTOR * glyph_height
            <= opposite_main
            <= -COUNT_MIN_MAIN_FACTOR * glyph_height
        ):
            continue
        if abs(cross) > COUNT_MAX_CROSS_FACTOR * glyph_height:
            continue
        side = max(_bbox_width(item_bbox), _bbox_height(item_bbox))
        if not (
            COUNT_MIN_SIDE_FACTOR * glyph_height
            <= side
            <= COUNT_MAX_SIDE_FACTOR * glyph_height
        ):
            continue
        length = item_length(item)
        primitive_count += 1
        if length >= COUNT_LONGISH_LENGTH_FACTOR * glyph_height:
            longish_count += 1
        selected.append({
            "item_index": int(idx),
            "item": item,
            "bbox": item_bbox,
            "length_pt": float(length),
        })
        if len(examples) < 6:
            examples.append({
                "item_index": int(idx),
                "bbox": _xywh(item_bbox),
                "main_delta_pt": round(main, 6),
                "cross_delta_pt": round(cross, 6),
                "length_pt": round(length, 6),
            })
    component_result = _count_side_component_evidence(
        selected,
        axis=axis,
        glyph_height=glyph_height,
    )
    return {
        "primitive_count": primitive_count,
        "longish_count": longish_count,
        "connected_component_count": component_result["connected_component_count"],
        "character_like_component_count": component_result[
            "character_like_component_count"
        ],
        "character_like_ink": component_result["character_like_ink"],
        "fragmented_glyph_evidence": component_result[
            "fragmented_glyph_evidence"
        ],
        "best_component": component_result["best_component"],
        "contact_graph_reason": component_result["contact_graph_reason"],
        "examples": examples,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _count_side_component_evidence(
    selected: list[dict[str, Any]],
    *,
    axis: str,
    glyph_height: float,
) -> dict[str, Any]:
    empty = {
        "connected_component_count": 0,
        "character_like_component_count": 0,
        "character_like_ink": False,
        "fragmented_glyph_evidence": False,
        "best_component": {},
        "contact_graph_reason": "count_side_local_primitive_capacity_exceeded",
    }
    if not selected or len(selected) > COUNT_MAX_LOCAL_PRIMITIVE_COUNT:
        return empty
    primitives: list[tuple[tuple[float, float], ...]] = []
    usable_rows: list[dict[str, Any]] = []
    for row in selected:
        points = _item_contact_points(row.get("item"))
        if points is None:
            continue
        primitives.append(points)
        usable_rows.append(row)
    if not primitives:
        return {
            **empty,
            "contact_graph_reason": "count_side_missing_primitive_points",
        }
    graph = build_contact_graph(tuple(primitives))
    if graph.reason is not None:
        return {
            **empty,
            "contact_graph_reason": str(graph.reason),
        }
    components = [
        _count_side_component_row(
            [usable_rows[index] for index in component],
            axis=axis,
            glyph_height=glyph_height,
        )
        for component in graph.components
    ]
    character_like = [
        component for component in components
        if component["character_like_ink"]
    ]
    character_like_ink = bool(
        character_like
        and len(character_like) <= COUNT_MAX_CHARACTER_LIKE_COMPONENTS
    )
    best = max(
        components,
        key=lambda component: (
            bool(component["character_like_ink"]),
            bool(component["fragmented_glyph_evidence"]),
            float(component["total_length_pt"]),
            int(component["primitive_count"]),
        ),
    )
    return {
        "connected_component_count": len(components),
        "character_like_component_count": len(character_like),
        "character_like_ink": character_like_ink,
        "fragmented_glyph_evidence": bool(
            character_like_ink
            and any(row["fragmented_glyph_evidence"] for row in character_like)
        ),
        "best_component": best,
        "contact_graph_reason": "exact_contact_graph_ready",
    }


def _count_side_component_row(
    rows: list[dict[str, Any]],
    *,
    axis: str,
    glyph_height: float,
) -> dict[str, Any]:
    x0 = min(float(row["bbox"][0]) for row in rows)
    y0 = min(float(row["bbox"][1]) for row in rows)
    x1 = max(float(row["bbox"][2]) for row in rows)
    y1 = max(float(row["bbox"][3]) for row in rows)
    if axis == "horizontal":
        main_span = x1 - x0
        cross_span = y1 - y0
    else:
        main_span = y1 - y0
        cross_span = x1 - x0
    total_length = sum(float(row["length_pt"]) for row in rows)
    longish_count = sum(
        1
        for row in rows
        if float(row["length_pt"])
        >= COUNT_LONGISH_LENGTH_FACTOR * glyph_height
    )
    span_ok = bool(
        COUNT_COMPONENT_MIN_MAIN_SPAN_FACTOR * glyph_height
        <= main_span
        <= COUNT_COMPONENT_MAX_MAIN_SPAN_FACTOR * glyph_height
        and COUNT_COMPONENT_MIN_CROSS_SPAN_FACTOR * glyph_height
        <= cross_span
        <= COUNT_COMPONENT_MAX_CROSS_SPAN_FACTOR * glyph_height
    )
    component_size_ok = len(rows) <= COUNT_MAX_COMPONENT_PRIMITIVE_COUNT
    fragmented = bool(
        component_size_ok
        and span_ok
        and len(rows) >= COUNT_FRAGMENTED_MIN_PRIMITIVE_COUNT
        and total_length
        >= COUNT_FRAGMENTED_MIN_TOTAL_LENGTH_FACTOR * glyph_height
    )
    character_like = bool(
        component_size_ok
        and span_ok
        and (longish_count >= 1 or fragmented)
    )
    return {
        "primitive_count": len(rows),
        "source_item_indices": sorted(int(row["item_index"]) for row in rows),
        "bbox": _xywh((x0, y0, x1, y1)),
        "main_span_pt": round(main_span, 6),
        "cross_span_pt": round(cross_span, 6),
        "total_length_pt": round(total_length, 6),
        "longish_count": longish_count,
        "span_ok": span_ok,
        "fragmented_glyph_evidence": fragmented,
        "character_like_ink": character_like,
    }


def _item_contact_points(item: Any) -> tuple[tuple[float, float], ...] | None:
    raw_points = getattr(item, "points", None)
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) < 2:
        return None
    try:
        points = tuple((float(point[0]), float(point[1])) for point in raw_points)
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        return None
    return points


def _stats(
    *,
    raw_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    drop_reasons: Counter[str],
    page_index: int,
) -> dict[str, Any]:
    raw_count = len(raw_rows)
    strict_count = len(strict_rows)
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "page_index": int(page_index),
        "raw_candidate_count": raw_count,
        "strict_candidate_count": strict_count,
        "raw_to_strict_reduction_factor": (
            round(raw_count / strict_count, 6)
            if strict_count else None
        ),
        "strict_recall_policy": "requires external gold matching; detector emits glyphs only",
        "ocr_results_count": 0,
        "ocr_called": False,
        "consumer_allowed_count": sum(
            1 for row in [*raw_rows, *strict_rows]
            if bool(row.get("consumer_allowed"))
        ),
        "drop_reason_counts": dict(sorted(drop_reasons.items())),
        "strict_axis_counts": dict(sorted(Counter(
            str(((row.get("anchor_gate") or {}).get("anchor") or {}).get("axis") or "unknown")
            for row in strict_rows
        ).items())),
    }


def _source_segment(index: int, item: Any) -> dict[str, Any]:
    bbox = _bbox_tuple(getattr(item, "bbox", None))
    return {
        "kind": "drawing_item",
        "item_index": int(index),
        "drawing_order": int(getattr(item, "drawing_order", -1)),
        "drawing_item_index": int(getattr(item, "item_index", -1)),
        "op": str(getattr(item, "op", "")),
        "bbox": _xywh(bbox) if bbox else None,
    }


def _line_angle(item: Any) -> float:
    p0, p1 = _endpoints(item)
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0


def _endpoints(item: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    points = list(getattr(item, "points", []) or [])
    return (
        (float(points[0][0]), float(points[0][1])),
        (float(points[-1][0]), float(points[-1][1])),
    )


def _segments_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> bool:
    o1 = _orient(a0, a1, b0)
    o2 = _orient(a0, a1, b1)
    o3 = _orient(b0, b1, a0)
    o4 = _orient(b0, b1, a1)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return (
        abs(o1) <= 1e-6 and _point_on_segment(a0, a1, b0)
        or abs(o2) <= 1e-6 and _point_on_segment(a0, a1, b1)
        or abs(o3) <= 1e-6 and _point_on_segment(b0, b1, a0)
        or abs(o4) <= 1e-6 and _point_on_segment(b0, b1, a1)
    )


def _orient(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    return (
        min(a[0], b[0]) - 1e-6 <= c[0] <= max(a[0], b[0]) + 1e-6
        and min(a[1], b[1]) - 1e-6 <= c[1] <= max(a[1], b[1]) + 1e-6
    )


def _angle_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        if {"x", "y", "w", "h"}.issubset(value):
            x = float(value["x"])
            y = float(value["y"])
            return x, y, x + float(value["w"]), y + float(value["h"])
        if {"x0", "y0", "x1", "y1"}.issubset(value):
            return (
                float(value["x0"]),
                float(value["y0"]),
                float(value["x1"]),
                float(value["y1"]),
            )
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x0, y0, x1, y1 = [float(item) for item in value]
        if x1 > x0 and y1 > y0:
            return x0, y0, x1, y1
    return None


def _bbox_tuple_allow_degenerate(
    value: Any,
) -> tuple[float, float, float, float] | None:
    """Return stroke bounds while preserving horizontal/vertical elements."""
    if isinstance(value, dict):
        if {"x", "y", "w", "h"}.issubset(value):
            x = float(value["x"])
            y = float(value["y"])
            x1 = x + float(value["w"])
            y1 = y + float(value["h"])
        elif {"x0", "y0", "x1", "y1"}.issubset(value):
            x = float(value["x0"])
            y = float(value["y0"])
            x1 = float(value["x1"])
            y1 = float(value["y1"])
        else:
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        x, y, x1, y1 = [float(item) for item in value]
    else:
        return None
    if not all(math.isfinite(item) for item in (x, y, x1, y1)):
        return None
    if x1 < x or y1 < y or (x1 == x and y1 == y):
        return None
    return x, y, x1, y1


def _point_from_center(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict) and {"x", "y"}.issubset(value):
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _union_bbox(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    pad: float,
) -> tuple[float, float, float, float]:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _xywh(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "x": round(bbox[0], 6),
        "y": round(bbox[1], 6),
        "w": round(bbox[2] - bbox[0], 6),
        "h": round(bbox[3] - bbox[1], 6),
    }


def _bbox_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [round(float(value), 6) for value in bbox]
