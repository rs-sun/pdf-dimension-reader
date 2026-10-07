"""Strict pure-vector dictionary for multiplication / quantity ``X`` glyphs.

The dictionary deliberately describes topology and normalized geometry rather
than rendered pixels.  A glyph is only a *glyph candidate* here; phrase-level
code must still prove that it sits inside a valid engineering-dimension
grammar before emitting it.
"""

from __future__ import annotations

import math
import re
from typing import Any


SCHEMA_VERSION = "vector_times_glyph_dictionary_v1"
CONSUMER_ALLOWED = False

TIMES_GLYPH_SIGNATURES: tuple[dict[str, Any], ...] = (
    {
        "signature_id": "nx_iso_square_cross",
        "aspect_min": 1.0,
        "aspect_max": 1.35,
        "angle_delta_min_deg": 73.0,
        "angle_delta_max_deg": 90.000001,
    },
    {
        "signature_id": "nx_iso_narrow_cross",
        "aspect_min": 1.35,
        "aspect_max": 2.20,
        "angle_delta_min_deg": 45.0,
        "angle_delta_max_deg": 73.0,
    },
)

MIN_LINE_LENGTH_PT = 2.5
MAX_LINE_LENGTH_PT = 25.0
MAX_LINE_LENGTH_RATIO = 1.12
MIN_GLYPH_SIDE_PT = 2.5
MAX_GLYPH_SIDE_PT = 18.0
MIN_DIAGONAL_AXIS_COVERAGE = 0.92
MAX_BBOX_ANGLE_ERROR_DEG = 3.5
MIN_INTERSECTION_FRACTION = 0.35
MAX_INTERSECTION_FRACTION = 0.65
MAX_INTERSECTION_CENTER_OFFSET = 0.06
MAX_STROKE_WIDTH_RATIO = 1.25
MAX_STRAIGHTNESS_ERROR_FACTOR = 0.01

_ITEM_ID_RE = re.compile(r"_d(?P<drawing>\d+)_i(?P<item>\d+)_")


def match_vector_times_glyph(items: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    """Match exactly two vector strokes against the observed X dictionary.

    The return value is JSON-safe and contains every hard measurement.  It is
    intentionally independent from decimal points, KEY frames, OCR, and final
    dimension semantics.
    """
    rows = list(items)
    reasons: list[str] = []
    if len(rows) != 2:
        return _result(
            matched=False,
            reasons=["requires_exactly_two_primitives"],
            measurements={"primitive_count": len(rows)},
            signature_id=None,
            hard_gates={"exactly_two_primitives": False},
        )

    ops = [_op(row) for row in rows]
    points = [_points(row) for row in rows]
    exactly_two_lines = ops == ["l", "l"] and all(len(value) >= 2 for value in points)
    if not exactly_two_lines:
        reasons.append("requires_two_line_primitives")
    if any(len(value) < 2 for value in points):
        return _result(
            matched=False,
            reasons=reasons or ["missing_line_endpoints"],
            measurements={"primitive_count": 2, "ops": ops},
            signature_id=None,
            hard_gates={
                "exactly_two_primitives": True,
                "two_line_primitives": False,
            },
        )

    endpoints = [(value[0], value[-1]) for value in points]
    lengths = [_polyline_length(value) for value in points]
    straightness = [
        _straightness_error(value, start, end)
        for value, (start, end) in zip(points, endpoints)
    ]
    bbox = _bbox_from_points([point for value in points for point in value])
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    min_side = min(width, height)
    max_side = max(width, height)
    aspect = max_side / max(min_side, 1e-12)
    length_ratio = max(lengths) / max(min(lengths), 1e-12)
    angles = [_line_angle(start, end) for start, end in endpoints]
    angle_delta = _angle_delta(angles[0], angles[1])
    expected_angle_delta = math.degrees(2.0 * math.atan(1.0 / max(aspect, 1e-12)))
    bbox_angle_error = abs(angle_delta - expected_angle_delta)
    coverages = [
        {
            "x": abs(end[0] - start[0]) / max(width, 1e-12),
            "y": abs(end[1] - start[1]) / max(height, 1e-12),
        }
        for start, end in endpoints
    ]
    intersection = _segment_intersection_parameters(*endpoints[0], *endpoints[1])
    if intersection is None:
        intersection_point = None
        fraction_a = None
        fraction_b = None
        center_offset = None
    else:
        intersection_point, fraction_a, fraction_b = intersection
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        center_offset = math.hypot(
            intersection_point[0] - center[0],
            intersection_point[1] - center[1],
        ) / max(max_side, 1e-12)

    source_positions = [_source_position(row) for row in rows]
    source_group_known = all(value is not None for value in source_positions)
    source_group_adjacent = (
        not source_group_known
        or (
            source_positions[0][0] == source_positions[1][0]
            and abs(source_positions[0][1] - source_positions[1][1]) == 1
        )
    )
    stroke_widths = [_stroke_width(row) for row in rows]
    known_stroke_widths = [value for value in stroke_widths if value is not None and value > 0.0]
    stroke_width_ratio = (
        max(known_stroke_widths) / max(min(known_stroke_widths), 1e-12)
        if len(known_stroke_widths) == 2
        else None
    )
    signature = _dictionary_signature(aspect=aspect, angle_delta=angle_delta)

    hard_gates = {
        "exactly_two_primitives": True,
        "two_line_primitives": exactly_two_lines,
        "straight_line_strokes": all(
            error <= MAX_STRAIGHTNESS_ERROR_FACTOR * max(length, 1e-12)
            for error, length in zip(straightness, lengths)
        ),
        "line_length_range": all(
            MIN_LINE_LENGTH_PT <= length <= MAX_LINE_LENGTH_PT
            for length in lengths
        ),
        "near_equal_line_lengths": length_ratio <= MAX_LINE_LENGTH_RATIO,
        "glyph_side_range": (
            MIN_GLYPH_SIDE_PT <= min_side
            and max_side <= MAX_GLYPH_SIDE_PT
        ),
        "dictionary_signature": signature is not None,
        "bbox_angle_consistent": bbox_angle_error <= MAX_BBOX_ANGLE_ERROR_DEG,
        "both_lines_span_opposite_corners": all(
            coverage["x"] >= MIN_DIAGONAL_AXIS_COVERAGE
            and coverage["y"] >= MIN_DIAGONAL_AXIS_COVERAGE
            for coverage in coverages
        ),
        "segments_intersect": intersection is not None,
        "intersection_central_on_both_lines": (
            fraction_a is not None
            and fraction_b is not None
            and MIN_INTERSECTION_FRACTION <= fraction_a <= MAX_INTERSECTION_FRACTION
            and MIN_INTERSECTION_FRACTION <= fraction_b <= MAX_INTERSECTION_FRACTION
        ),
        "intersection_near_glyph_center": (
            center_offset is not None
            and center_offset <= MAX_INTERSECTION_CENTER_OFFSET
        ),
        "source_items_same_drawing_and_adjacent_when_known": source_group_adjacent,
        "stroke_widths_compatible_when_known": (
            stroke_width_ratio is None
            or stroke_width_ratio <= MAX_STROKE_WIDTH_RATIO
        ),
    }
    reason_by_gate = {
        "two_line_primitives": "requires_two_line_primitives",
        "straight_line_strokes": "line_primitive_not_straight",
        "line_length_range": "line_length_out_of_range",
        "near_equal_line_lengths": "line_lengths_not_equal_enough",
        "glyph_side_range": "glyph_bbox_out_of_range",
        "dictionary_signature": "glyph_style_not_in_times_dictionary",
        "bbox_angle_consistent": "line_angles_not_bbox_diagonals",
        "both_lines_span_opposite_corners": "lines_do_not_span_opposite_corners",
        "segments_intersect": "segments_do_not_intersect",
        "intersection_central_on_both_lines": "intersection_not_central_on_lines",
        "intersection_near_glyph_center": "intersection_not_near_glyph_center",
        "source_items_same_drawing_and_adjacent_when_known": "source_items_not_adjacent_in_one_drawing",
        "stroke_widths_compatible_when_known": "stroke_widths_not_compatible",
    }
    reasons.extend(
        reason_by_gate[gate]
        for gate, passed in hard_gates.items()
        if not passed and gate in reason_by_gate
    )
    measurements = {
        "primitive_count": 2,
        "ops": ops,
        "bbox_xyxy": [round(value, 6) for value in bbox],
        "width_pt": round(width, 6),
        "height_pt": round(height, 6),
        "aspect": round(aspect, 6),
        "line_lengths_pt": [round(value, 6) for value in lengths],
        "line_length_ratio": round(length_ratio, 6),
        "line_angles_deg": [round(value, 6) for value in angles],
        "angle_delta_deg": round(angle_delta, 6),
        "bbox_expected_angle_delta_deg": round(expected_angle_delta, 6),
        "bbox_angle_error_deg": round(bbox_angle_error, 6),
        "line_axis_coverages": [
            {key: round(value, 6) for key, value in coverage.items()}
            for coverage in coverages
        ],
        "straightness_error_pt": [round(value, 6) for value in straightness],
        "intersection": (
            {
                "x": round(intersection_point[0], 6),
                "y": round(intersection_point[1], 6),
                "line_a_fraction": round(float(fraction_a), 6),
                "line_b_fraction": round(float(fraction_b), 6),
                "center_offset_glyph_sides": round(float(center_offset), 6),
            }
            if intersection_point is not None
            and fraction_a is not None
            and fraction_b is not None
            and center_offset is not None
            else None
        ),
        "source_group_known": source_group_known,
        "source_positions": [
            {"drawing_order": int(value[0]), "item_index": int(value[1])}
            if value is not None else None
            for value in source_positions
        ],
        "stroke_widths_pt": [
            round(value, 6) if value is not None else None
            for value in stroke_widths
        ],
        "stroke_width_ratio": (
            round(stroke_width_ratio, 6)
            if stroke_width_ratio is not None else None
        ),
    }
    return _result(
        matched=all(hard_gates.values()),
        reasons=reasons,
        measurements=measurements,
        signature_id=(
            str(signature["signature_id"])
            if signature is not None else None
        ),
        hard_gates=hard_gates,
    )


def _result(
    *,
    matched: bool,
    reasons: list[str],
    measurements: dict[str, Any],
    signature_id: str | None,
    hard_gates: dict[str, bool],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matched": bool(matched),
        "label": "X" if matched else None,
        "canonical_symbol": "×" if matched else None,
        "signature_id": signature_id,
        "reject_reasons": sorted(set(str(reason) for reason in reasons if reason)),
        "hard_gates": hard_gates,
        "measurements": measurements,
        "semantic_context_required": True,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _dictionary_signature(*, aspect: float, angle_delta: float) -> dict[str, Any] | None:
    for signature in TIMES_GLYPH_SIGNATURES:
        if (
            float(signature["aspect_min"]) <= aspect <= float(signature["aspect_max"])
            and float(signature["angle_delta_min_deg"])
            <= angle_delta
            <= float(signature["angle_delta_max_deg"])
        ):
            return signature
    return None


def _op(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("op") or "")
    return str(getattr(item, "op", "") or "")


def _points(item: Any) -> list[tuple[float, float]]:
    raw = item.get("points") if isinstance(item, dict) else getattr(item, "points", None)
    out: list[tuple[float, float]] = []
    for point in list(raw or []):
        try:
            out.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _source_object(item: Any) -> Any:
    source = getattr(item, "item", None)
    return source if source is not None else item


def _source_position(item: Any) -> tuple[int, int] | None:
    source = _source_object(item)
    if isinstance(source, dict):
        drawing = source.get("drawing_order")
        item_index = source.get("item_index")
        if drawing is not None and item_index is not None:
            try:
                return int(drawing), int(item_index)
            except (TypeError, ValueError):
                pass
        match = _ITEM_ID_RE.search(str(source.get("item_id") or ""))
        if match is not None:
            return int(match.group("drawing")), int(match.group("item"))
        return None
    try:
        return int(getattr(source, "drawing_order")), int(getattr(source, "item_index"))
    except (AttributeError, TypeError, ValueError):
        return None


def _stroke_width(item: Any) -> float | None:
    source = _source_object(item)
    raw = source.get("stroke_width") if isinstance(source, dict) else getattr(source, "stroke_width", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:])
    )


def _straightness_error(
    points: list[tuple[float, float]],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        return float("inf")
    return max(
        abs(dx * (start[1] - point[1]) - (start[0] - point[0]) * dy) / length
        for point in points
    )


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _line_angle(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0


def _angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _segment_intersection_parameters(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> tuple[tuple[float, float], float, float] | None:
    ar = (a1[0] - a0[0], a1[1] - a0[1])
    br = (b1[0] - b0[0], b1[1] - b0[1])
    denominator = ar[0] * br[1] - ar[1] * br[0]
    if abs(denominator) <= 1e-12:
        return None
    delta = (b0[0] - a0[0], b0[1] - a0[1])
    fraction_a = (delta[0] * br[1] - delta[1] * br[0]) / denominator
    fraction_b = (delta[0] * ar[1] - delta[1] * ar[0]) / denominator
    if not (
        -1e-9 <= fraction_a <= 1.0 + 1e-9
        and -1e-9 <= fraction_b <= 1.0 + 1e-9
    ):
        return None
    point = (
        a0[0] + fraction_a * ar[0],
        a0[1] + fraction_a * ar[1],
    )
    return point, fraction_a, fraction_b
