"""Decimal-anchored pure-vector profile for blockfont tolerance signs.

The matcher consumes only primitives already admitted to a local dimension
phrase.  It does not discover phrases, query a page, or treat ``+`` / ``-``
as anchors.  Geometry is measured in the trusted phrase coordinate system and
normalized by the adjacent tolerance decimal-point height.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


SCHEMA_VERSION = "vector_tolerance_sign_glyph_profile_v2"
PROFILE_ID = "blockfont_decimal_anchored_tolerance_sign_v2"
CONSUMER_ALLOWED = False

SCOPE_CONTRACT = {
    "input_scope": "existing_phrase_local_primitives_only",
    "page_scan_allowed": False,
    "page_query_allowed": False,
    "sign_discovery_anchor_allowed": False,
    "decimal_anchor_required": True,
}

PROFILE = {
    "schema_version": SCHEMA_VERSION,
    "profile_id": PROFILE_ID,
    "font_family": "blockfont",
    "font_authority": {
        "source_format": "siemens_nx_fnx",
        "runtime_fnx_file_reads": False,
        "compiled_glyph_evidence": {
            "+": {
                "source_code": 43,
                "operations": "MLML",
                "stroke_count": 2,
                "source_line_lengths_over_dot_h": [3.0, 3.0],
                "pdf_line_lengths_over_dot_h": [2.1, 3.0],
            },
            "-": {
                "source_code": 45,
                "operations": "ML",
                "stroke_count": 1,
                "source_line_lengths_over_dot_h": [2.0],
                "pdf_line_lengths_over_dot_h": [1.4],
            },
            "±": {
                "operations": "MLMLML",
                "stroke_count": 3,
                "segment_count": 3,
                "source_line_lengths_over_dot_h": [3.0, 3.0, 3.0],
                "source_parallel_separation_over_dot_h": 2.05263,
                "exported_pdf_line_lengths_over_dot_h": [
                    1.756617,
                    1.756617,
                    2.513233,
                ],
            },
        },
    },
    "representation": "nx_pdf_polyline",
    "normalization": {
        "scale_authority": "adjacent_tolerance_decimal_bbox_height",
        "axis_authority": "trusted_phrase_axis",
        "candidate_self_scaling_allowed": False,
        "free_angle_search_allowed": False,
    },
    "decimal": {
        "primitive_count": 1,
        "allowed_ops": ["qu"],
        "width_h": [0.58, 0.82],
        "closed_path_required": True,
    },
    "shared_layout": {
        "center_offset_u_h": [-7.75, -7.25],
        "center_offset_v_h": [-2.25, -1.75],
        "axis_angle_error_max_deg": 3.0,
        "source_drawing_match_when_known": True,
        "stroke_width_ratio_max_when_known": 1.25,
    },
    "glyphs": {
        "+": {
            "primitive_count": 2,
            "main_axis_line_length_h": [2.0, 2.2],
            "cross_axis_line_length_h": [2.85, 3.15],
            "intersection_fraction": [0.42, 0.58],
            "source_items_adjacent_when_known": True,
        },
        "-": {
            "primitive_count": 1,
            "main_axis_line_length_h": [1.3, 1.5],
            "source_items_adjacent_when_known": True,
        },
        "±": {
            "primitive_count": 3,
            "semantic_anchor": "proper_crossing",
            "main_axis_line_length_h": [1.65, 1.88],
            "cross_axis_line_length_h": [2.38, 2.65],
            "main_to_cross_length_ratio": [0.66, 0.73],
            "parallel_line_length_ratio": [0.95, 1.05],
            "parallel_line_separation_h": [2.38, 2.65],
            "intersection_fraction": [0.42, 0.58],
            "center_offset_u_h": [-9.5, -9.0],
            "center_offset_v_h": [-2.25, -1.75],
            "source_items_adjacent_when_known": True,
        },
    },
    "scope_contract": SCOPE_CONTRACT,
}

PROFILE_SHA256 = hashlib.sha256(
    json.dumps(
        PROFILE,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_ITEM_ID_RE = re.compile(r"_d(?P<drawing>\d+)_i(?P<item>\d+)_")


def match_decimal_anchored_tolerance_sign(
    sign_items: list[Any] | tuple[Any, ...],
    decimal_candidates: list[list[Any] | tuple[Any, ...]],
    *,
    expected_label: str,
) -> dict[str, Any]:
    """Match one local ``+``, ``-`` or ``±`` against the blockfont profile.

    ``sign_items`` and every decimal candidate must already be projected into
    phrase ``(main, cross)`` coordinates.  The function has no page handle and
    therefore cannot accidentally expand into a page scan.
    """
    label = str(expected_label or "")
    rows = list(sign_items)
    reasons: list[str] = []
    glyph_profile = PROFILE["glyphs"].get(label)
    if glyph_profile is None:
        return _result(
            matched=False,
            label=None,
            reasons=["unsupported_sign_label"],
            hard_gates={"supported_sign_label": False},
            measurements={"expected_label": label},
        )

    line_rows = [_line_measurement(item) for item in rows]
    expected_count = int(glyph_profile["primitive_count"])
    exact_count = len(rows) == expected_count
    all_line_primitives = all(
        measurement is not None and measurement["op"] == "l"
        for measurement in line_rows
    )
    straight_lines = all(
        measurement is not None
        and measurement["straightness_error_pt"]
        <= max(measurement["length_pt"], 1e-12) * 0.01
        for measurement in line_rows
    )
    axis_rows = [
        _axis_for_line(measurement)
        if measurement is not None else None
        for measurement in line_rows
    ]
    expected_axes = {
        "+": ["main", "cross"],
        "-": ["main"],
        "±": ["main", "main", "cross"],
    }[label]
    axes_aligned = sorted(
        axis for axis in axis_rows if axis is not None
    ) == sorted(expected_axes)

    source_positions = [_source_position(item) for item in rows]
    source_adjacent = _source_items_adjacent_when_known(source_positions)
    sign_stroke_widths = [_stroke_width(item) for item in rows]
    sign_stroke_compatible = _stroke_widths_compatible(sign_stroke_widths)

    topology = _match_sign_topology(
        label,
        line_rows,
        axis_rows,
        glyph_profile,
    )

    base_hard_gates = {
        "supported_sign_label": True,
        "exact_profile_primitive_count": exact_count,
        "line_primitives_only": all_line_primitives,
        "straight_line_strokes": straight_lines,
        "strokes_aligned_to_phrase_axes": axes_aligned,
        "font_topology_match": bool(topology["matched"]),
        "source_items_adjacent_when_known": source_adjacent,
        "sign_stroke_widths_compatible_when_known": sign_stroke_compatible,
    }
    if not exact_count:
        reasons.append("wrong_profile_primitive_count")
    if not all_line_primitives:
        reasons.append("requires_line_primitives")
    if not straight_lines:
        reasons.append("line_primitive_not_straight")
    if not axes_aligned:
        reasons.append("strokes_not_aligned_to_phrase_axes")
    reasons.extend(topology["reject_reasons"])
    if not source_adjacent:
        reasons.append("source_items_not_adjacent_in_one_drawing")
    if not sign_stroke_compatible:
        reasons.append("sign_stroke_widths_not_compatible")

    sign_points = [
        point
        for measurement in line_rows
        if measurement is not None
        for point in measurement["points"]
    ]
    sign_bbox = _bbox_from_points(sign_points) if sign_points else None
    decimal_trials: list[dict[str, Any]] = []
    if all(base_hard_gates.values()) and sign_bbox is not None:
        for candidate in decimal_candidates:
            trial = _match_decimal_candidate(
                rows,
                line_rows,
                axis_rows,
                sign_bbox=sign_bbox,
                semantic_anchor=topology.get("semantic_anchor"),
                decimal_items=list(candidate),
                label=label,
            )
            decimal_trials.append(trial)

    matching_trials = [
        trial for trial in decimal_trials
        if trial["matched"] is True
    ]
    unique_decimal = len(matching_trials) == 1
    if len(matching_trials) > 1:
        reasons.append("ambiguous_multiple_decimal_anchors")
    elif not matching_trials:
        detailed_reasons = {
            reason
            for trial in decimal_trials
            for reason in trial.get("reject_reasons") or []
        }
        reasons.extend(sorted(detailed_reasons))
        reasons.append("no_decimal_anchor_at_profile_offset")

    selected = matching_trials[0] if unique_decimal else None
    measurements = {
        "expected_label": label,
        "sign_primitive_count": len(rows),
        "sign_primitive_ids": [_primitive_id(item) for item in rows],
        "sign_source_positions": [
            _json_source_position(value) for value in source_positions
        ],
        "sign_bbox_main_cross": (
            [round(value, 6) for value in sign_bbox]
            if sign_bbox is not None else None
        ),
        "line_angles_deg": [
            round(float(measurement["angle_deg"]), 6)
            if measurement is not None else None
            for measurement in line_rows
        ],
        "line_axes": axis_rows,
        "font_topology": topology["measurements"],
        "matching_decimal_count": len(matching_trials),
        "decimal_candidate_count": len(decimal_candidates),
    }
    if selected is not None:
        measurements.update(selected["measurements"])
    elif decimal_trials:
        measurements["decimal_trials"] = [
            trial["measurements"] for trial in decimal_trials
        ]

    hard_gates = {
        **base_hard_gates,
        "unique_decimal_anchor": unique_decimal,
        "decimal_profile_geometry": bool(
            selected and selected["hard_gates"]["decimal_profile_geometry"]
        ),
        "line_lengths_in_profile": bool(
            selected and selected["hard_gates"]["line_lengths_in_profile"]
        ),
        "decimal_relative_slot_in_profile": bool(
            selected
            and selected["hard_gates"]["decimal_relative_slot_in_profile"]
        ),
        "sign_and_decimal_same_drawing_when_known": bool(
            selected
            and selected["hard_gates"][
                "sign_and_decimal_same_drawing_when_known"
            ]
        ),
        "sign_and_decimal_stroke_widths_compatible_when_known": bool(
            selected
            and selected["hard_gates"][
                "sign_and_decimal_stroke_widths_compatible_when_known"
            ]
        ),
    }
    matched = all(hard_gates.values())
    return _result(
        matched=matched,
        label=label if matched else None,
        reasons=reasons,
        hard_gates=hard_gates,
        measurements=measurements,
    )


def _match_sign_topology(
    label: str,
    line_rows: list[dict[str, Any] | None],
    axis_rows: list[str | None],
    glyph_profile: dict[str, Any],
) -> dict[str, Any]:
    measurements: dict[str, Any] = {
        "semantic_anchor_kind": str(
            glyph_profile.get("semantic_anchor") or "bbox_center"
        ),
        "proper_crossing_count": 0,
        "central_crossing_count": 0,
    }
    if label == "-":
        return {
            "matched": len(line_rows) == 1 and line_rows[0] is not None,
            "semantic_anchor": None,
            "reject_reasons": [],
            "measurements": measurements,
        }
    if any(row is None for row in line_rows):
        return {
            "matched": False,
            "semantic_anchor": None,
            "reject_reasons": ["sign_topology_unavailable"],
            "measurements": measurements,
        }

    main_rows = [
        row
        for axis, row in zip(axis_rows, line_rows)
        if axis == "main" and row is not None
    ]
    cross_rows = [
        row
        for axis, row in zip(axis_rows, line_rows)
        if axis == "cross" and row is not None
    ]
    intersections: list[tuple[tuple[float, float], float, float]] = []
    for main_row in main_rows:
        for cross_row in cross_rows:
            intersection = _segment_intersection_parameters(
                main_row["start"],
                main_row["end"],
                cross_row["start"],
                cross_row["end"],
            )
            if intersection is not None:
                intersections.append(intersection)
    low, high = glyph_profile["intersection_fraction"]
    central = [
        row
        for row in intersections
        if float(low) <= row[1] <= float(high)
        and float(low) <= row[2] <= float(high)
    ]
    measurements.update({
        "proper_crossing_count": len(intersections),
        "central_crossing_count": len(central),
        "crossing_fractions": [
            [round(row[1], 6), round(row[2], 6)]
            for row in intersections
        ],
    })
    if label == "+":
        matched = (
            len(main_rows) == 1
            and len(cross_rows) == 1
            and len(intersections) == 1
            and len(central) == 1
        )
        return {
            "matched": matched,
            "semantic_anchor": central[0][0] if matched else None,
            "reject_reasons": (
                [] if matched else ["plus_strokes_do_not_intersect_centrally"]
            ),
            "measurements": measurements,
        }

    matched = (
        label == "±"
        and len(main_rows) == 2
        and len(cross_rows) == 1
        and len(intersections) == 1
        and len(central) == 1
    )
    return {
        "matched": matched,
        "semantic_anchor": central[0][0] if matched else None,
        "reject_reasons": (
            [] if matched else ["plus_minus_requires_one_central_crossing"]
        ),
        "measurements": measurements,
    }


def _match_decimal_candidate(
    sign_items: list[Any],
    line_rows: list[dict[str, Any] | None],
    axis_rows: list[str | None],
    *,
    sign_bbox: tuple[float, float, float, float],
    semantic_anchor: tuple[float, float] | None,
    decimal_items: list[Any],
    label: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    decimal_profile = PROFILE["decimal"]
    decimal_points = [
        point for item in decimal_items for point in _points(item)
    ]
    decimal_bbox = (
        _bbox_from_points(decimal_points) if decimal_points else None
    )
    decimal_height = (
        decimal_bbox[3] - decimal_bbox[1]
        if decimal_bbox is not None else 0.0
    )
    decimal_width = (
        decimal_bbox[2] - decimal_bbox[0]
        if decimal_bbox is not None else 0.0
    )
    dot_ops = [_op(item) for item in decimal_items]
    closed = bool(
        len(decimal_items) == 1
        and len(_points(decimal_items[0])) >= 4
        and _distance(
            _points(decimal_items[0])[0],
            _points(decimal_items[0])[-1],
        ) <= 1e-6
    )
    dot_width_h = (
        decimal_width / decimal_height
        if decimal_height > 1e-12 else float("inf")
    )
    decimal_geometry = (
        len(decimal_items) == int(decimal_profile["primitive_count"])
        and dot_ops == list(decimal_profile["allowed_ops"])
        and decimal_height > 0.0
        and float(decimal_profile["width_h"][0])
        <= dot_width_h
        <= float(decimal_profile["width_h"][1])
        and closed
    )

    ordered_lines = sorted(
        (
            (axis, measurement)
            for axis, measurement in zip(axis_rows, line_rows)
            if axis is not None and measurement is not None
        ),
        key=lambda row: 0 if row[0] == "main" else 1,
    )
    line_lengths_h = [
        float(measurement["length_pt"]) / max(decimal_height, 1e-12)
        for _axis, measurement in ordered_lines
    ]
    glyph_profile = PROFILE["glyphs"][label]
    if label == "+" and len(line_lengths_h) == 2:
        line_lengths_in_profile = (
            _in_range(
                line_lengths_h[0],
                glyph_profile["main_axis_line_length_h"],
            )
            and _in_range(
                line_lengths_h[1],
                glyph_profile["cross_axis_line_length_h"],
            )
        )
    elif label == "-" and len(line_lengths_h) == 1:
        line_lengths_in_profile = _in_range(
            line_lengths_h[0],
            glyph_profile["main_axis_line_length_h"],
        )
    elif label == "±" and len(line_lengths_h) == 3:
        main_lengths = line_lengths_h[:2]
        cross_length = line_lengths_h[2]
        short_to_long = [
            value / max(cross_length, 1e-12)
            for value in main_lengths
        ]
        parallel_ratio = min(main_lengths) / max(max(main_lengths), 1e-12)
        main_measurements = [
            measurement
            for axis, measurement in ordered_lines
            if axis == "main"
        ]
        parallel_separation_h = (
            abs(
                _line_center(main_measurements[0])[1]
                - _line_center(main_measurements[1])[1]
            ) / max(decimal_height, 1e-12)
            if len(main_measurements) == 2 else float("inf")
        )
        line_lengths_in_profile = (
            all(
                _in_range(value, glyph_profile["main_axis_line_length_h"])
                for value in main_lengths
            )
            and _in_range(
                cross_length,
                glyph_profile["cross_axis_line_length_h"],
            )
            and all(
                _in_range(value, glyph_profile["main_to_cross_length_ratio"])
                for value in short_to_long
            )
            and _in_range(
                parallel_ratio,
                glyph_profile["parallel_line_length_ratio"],
            )
            and _in_range(
                parallel_separation_h,
                glyph_profile["parallel_line_separation_h"],
            )
        )
    else:
        line_lengths_in_profile = False

    sign_center = semantic_anchor or (
        (sign_bbox[0] + sign_bbox[2]) / 2.0,
        (sign_bbox[1] + sign_bbox[3]) / 2.0,
    )
    decimal_center = (
        (
            (decimal_bbox[0] + decimal_bbox[2]) / 2.0
            if decimal_bbox is not None else float("nan")
        ),
        (
            (decimal_bbox[1] + decimal_bbox[3]) / 2.0
            if decimal_bbox is not None else float("nan")
        ),
    )
    offset_u_h = (
        (sign_center[0] - decimal_center[0]) / decimal_height
        if decimal_height > 1e-12 else float("nan")
    )
    offset_v_h = (
        (sign_center[1] - decimal_center[1]) / decimal_height
        if decimal_height > 1e-12 else float("nan")
    )
    shared_layout = PROFILE["shared_layout"]
    offset_u_range = glyph_profile.get(
        "center_offset_u_h",
        shared_layout["center_offset_u_h"],
    )
    offset_v_range = glyph_profile.get(
        "center_offset_v_h",
        shared_layout["center_offset_v_h"],
    )
    relative_slot = (
        _in_range(offset_u_h, offset_u_range)
        and _in_range(offset_v_h, offset_v_range)
    )

    sign_positions = [_source_position(item) for item in sign_items]
    decimal_positions = [_source_position(item) for item in decimal_items]
    same_drawing = _same_drawing_when_known(
        [*sign_positions, *decimal_positions]
    )
    all_stroke_widths = [
        *[_stroke_width(item) for item in sign_items],
        *[_stroke_width(item) for item in decimal_items],
    ]
    stroke_compatible = _stroke_widths_compatible(all_stroke_widths)
    hard_gates = {
        "decimal_profile_geometry": decimal_geometry,
        "line_lengths_in_profile": line_lengths_in_profile,
        "decimal_relative_slot_in_profile": relative_slot,
        "sign_and_decimal_same_drawing_when_known": same_drawing,
        "sign_and_decimal_stroke_widths_compatible_when_known": (
            stroke_compatible
        ),
    }
    if not decimal_geometry:
        reasons.append("invalid_decimal_anchor_geometry")
    if not line_lengths_in_profile:
        reasons.append("line_length_outside_profile")
    if not relative_slot:
        reasons.append("decimal_relative_slot_outside_profile")
    if not same_drawing:
        reasons.append("sign_and_decimal_not_from_same_drawing")
    if not stroke_compatible:
        reasons.append("sign_and_decimal_stroke_widths_not_compatible")

    measurements = {
        "decimal_primitive_ids": [
            _primitive_id(item) for item in decimal_items
        ],
        "decimal_source_positions": [
            _json_source_position(value) for value in decimal_positions
        ],
        "decimal_bbox_main_cross": (
            [round(value, 6) for value in decimal_bbox]
            if decimal_bbox is not None else None
        ),
        "decimal_height_pt": round(decimal_height, 6),
        "decimal_width_h": _round_finite(dot_width_h),
        "line_lengths_h": [round(value, 6) for value in line_lengths_h],
        "semantic_anchor": (
            [round(value, 6) for value in sign_center]
            if semantic_anchor is not None else None
        ),
        "offset_u_h": _round_finite(offset_u_h),
        "offset_v_h": _round_finite(offset_v_h),
    }
    return {
        "matched": all(hard_gates.values()),
        "reject_reasons": sorted(set(reasons)),
        "hard_gates": hard_gates,
        "measurements": measurements,
    }


def _line_center(line: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(line["start"][0]) + float(line["end"][0])) / 2.0,
        (float(line["start"][1]) + float(line["end"][1])) / 2.0,
    )


def _result(
    *,
    matched: bool,
    label: str | None,
    reasons: list[str],
    hard_gates: dict[str, bool],
    measurements: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "profile_sha256": PROFILE_SHA256,
        "matched": bool(matched),
        "label": label,
        "reject_reasons": sorted(set(
            str(reason) for reason in reasons if reason
        )),
        "hard_gates": hard_gates,
        "measurements": measurements,
        "font_authority": {
            "font_family": PROFILE["font_family"],
            "runtime_fnx_file_reads": False,
            "compiled_glyph_evidence": dict(
                PROFILE["font_authority"]["compiled_glyph_evidence"].get(
                    str(measurements.get("expected_label") or ""),
                    {},
                )
            ),
        },
        "uniform_pitch_short_circuit_allowed": bool(
            str(measurements.get("expected_label") or "") != "±"
        ),
        "scope_contract": dict(SCOPE_CONTRACT),
        "semantic_context_required": True,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _line_measurement(item: Any) -> dict[str, Any] | None:
    points = _points(item)
    if len(points) < 2:
        return None
    start, end = points[0], points[-1]
    length = _polyline_length(points)
    return {
        "op": _op(item),
        "points": points,
        "start": start,
        "end": end,
        "length_pt": length,
        "angle_deg": _line_angle(start, end),
        "straightness_error_pt": _straightness_error(points, start, end),
    }


def _axis_for_line(measurement: dict[str, Any]) -> str | None:
    angle = float(measurement["angle_deg"])
    main_error = min(angle, abs(180.0 - angle))
    cross_error = abs(angle - 90.0)
    maximum = float(PROFILE["shared_layout"]["axis_angle_error_max_deg"])
    if main_error <= maximum:
        return "main"
    if cross_error <= maximum:
        return "cross"
    return None


def _points(item: Any) -> list[tuple[float, float]]:
    raw = item.get("points") if isinstance(item, dict) else getattr(
        item, "points", None
    )
    out: list[tuple[float, float]] = []
    for point in list(raw or []):
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def _op(item: Any) -> str:
    return str(
        item.get("op") if isinstance(item, dict) else getattr(item, "op", "")
    )


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
        return int(getattr(source, "drawing_order")), int(
            getattr(source, "item_index")
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _primitive_id(item: Any) -> str | None:
    source = _source_object(item)
    if isinstance(source, dict):
        value = source.get("item_id")
        return str(value) if value else None
    value = getattr(source, "item_id", None)
    return str(value) if value else None


def _stroke_width(item: Any) -> float | None:
    source = _source_object(item)
    raw = source.get("stroke_width") if isinstance(source, dict) else getattr(
        source, "stroke_width", None
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _source_items_adjacent_when_known(
    positions: list[tuple[int, int] | None],
) -> bool:
    known = [position for position in positions if position is not None]
    if len(known) != len(positions):
        return True
    if not known:
        return True
    drawings = {position[0] for position in known}
    indices = sorted(position[1] for position in known)
    return (
        len(drawings) == 1
        and indices == list(range(indices[0], indices[-1] + 1))
    )


def _same_drawing_when_known(
    positions: list[tuple[int, int] | None],
) -> bool:
    known = [position for position in positions if position is not None]
    return not known or len({position[0] for position in known}) == 1


def _stroke_widths_compatible(widths: list[float | None]) -> bool:
    known = [value for value in widths if value is not None]
    if len(known) < 2:
        return True
    ratio = max(known) / max(min(known), 1e-12)
    return ratio <= float(
        PROFILE["shared_layout"]["stroke_width_ratio_max_when_known"]
    )


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(_distance(left, right) for left, right in zip(points, points[1:]))


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
        abs(dx * (start[1] - point[1]) - (start[0] - point[0]) * dy)
        / length
        for point in points
    )


def _line_angle(
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    return math.degrees(
        math.atan2(end[1] - start[1], end[0] - start[0])
    ) % 180.0


def _bbox_from_points(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


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
    fraction_a = (
        delta[0] * br[1] - delta[1] * br[0]
    ) / denominator
    fraction_b = (
        delta[0] * ar[1] - delta[1] * ar[0]
    ) / denominator
    if not (
        -1e-9 <= fraction_a <= 1.0 + 1e-9
        and -1e-9 <= fraction_b <= 1.0 + 1e-9
    ):
        return None
    return (
        (
            a0[0] + fraction_a * ar[0],
            a0[1] + fraction_a * ar[1],
        ),
        fraction_a,
        fraction_b,
    )


def _distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _in_range(value: float, bounds: list[float]) -> bool:
    return (
        math.isfinite(float(value))
        and float(bounds[0]) <= float(value) <= float(bounds[1])
    )


def _round_finite(value: float) -> float | None:
    return round(float(value), 6) if math.isfinite(float(value)) else None


def _json_source_position(
    value: tuple[int, int] | None,
) -> dict[str, int] | None:
    if value is None:
        return None
    return {"drawing_order": int(value[0]), "item_index": int(value[1])}
