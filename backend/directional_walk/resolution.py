"""Pure ordering and fragment resolution for R92 walk successes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
import math
import re
from typing import Any

from dimension_syntax import SYNTAX_OK, classify_dimension_syntax

from .glyph_key import project_point
from .phrase_output import stable_digest


_PRIMITIVE_ID = re.compile(r"^p\d+_d(?P<drawing>\d+)_i(?P<item>\d+)_")


def _character_positions_valid(text: str) -> bool:
    for prefix in ("R", "Ø"):
        if prefix in text and not (text.count(prefix) == 1 and text.startswith(prefix)):
            return False
    if any(
        char == "°" and (index == 0 or not text[index - 1].isdigit())
        for index, char in enumerate(text)
    ):
        return False
    return True


def _draw_order_direction(cells: Sequence[Mapping[str, Any]]) -> str | None:
    per_cell: list[dict[int, list[int]]] = []
    drawing_cell_counts: Counter[int] = Counter()
    for cell in cells:
        drawings: dict[int, list[int]] = {}
        for primitive_id in cell.get("primitive_ids") or ():
            match = _PRIMITIVE_ID.match(str(primitive_id))
            if match is None:
                continue
            drawing = int(match.group("drawing"))
            drawings.setdefault(drawing, []).append(int(match.group("item")))
        per_cell.append(drawings)
        drawing_cell_counts.update(drawings.keys())
    if not drawing_cell_counts:
        return None
    maximum = max(drawing_cell_counts.values())
    main_drawings = [
        drawing
        for drawing, count in drawing_cell_counts.items()
        if count == maximum
    ]
    if len(main_drawings) != 1:
        return None
    main = main_drawings[0]
    intervals = [
        (min(drawings[main]), max(drawings[main]))
        for drawings in per_cell
        if main in drawings
    ]
    if len(intervals) < 2:
        return None
    ascending = all(
        left[1] < right[0]
        for left, right in zip(intervals, intervals[1:])
    )
    descending = all(
        left[0] > right[1]
        for left, right in zip(intervals, intervals[1:])
    )
    if ascending == descending:
        return None
    return "forward" if ascending else "reverse"


def resolve_draw_order_direction_v1(
    cells: Sequence[Mapping[str, Any]],
) -> str | None:
    """Expose the strict primitive draw-order decision without other signals."""

    return _draw_order_direction(cells)


def _undirected_angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _undirected_mean(left: float, right: float) -> float:
    x = sum(
        math.cos(math.radians(angle * 2.0))
        for angle in (float(left), float(right))
    )
    y = sum(
        math.sin(math.radians(angle * 2.0))
        for angle in (float(left), float(right))
    )
    mean = (math.degrees(math.atan2(y, x)) / 2.0) % 180.0
    if math.cos(math.radians(mean - float(left))) < 0.0:
        mean = (mean + 180.0) % 360.0
    return mean


def _cells_uv_bbox(
    cells: Sequence[Mapping[str, Any]],
    *,
    angle_deg: float,
) -> tuple[float, float, float, float]:
    projected = []
    for cell in cells:
        raw_points = cell.get("points")
        if isinstance(raw_points, Sequence) and raw_points:
            points = [
                (float(point[0]), float(point[1]))
                for point in raw_points
            ]
        else:
            bbox = cell["bbox"]
            x0, y0 = float(bbox["x"]), float(bbox["y"])
            x1 = x0 + float(bbox["w"])
            y1 = y0 + float(bbox["h"])
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        projected.extend(
            project_point(point, angle_deg)
            for point in points
        )
    return (
        min(point[0] for point in projected),
        min(point[1] for point in projected),
        max(point[0] for point in projected),
        max(point[1] for point in projected),
    )


def _cell_u_center(cell: Mapping[str, Any], *, angle_deg: float) -> float:
    u0, _v0, u1, _v1 = _cells_uv_bbox([cell], angle_deg=angle_deg)
    return (u0 + u1) / 2.0


def _source_anchors(success: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    anchors = success.get("_source_anchors")
    if isinstance(anchors, Sequence):
        return list(anchors)
    return [success["anchor"]]


def _atomic_successes(success: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    members = success.get("_merge_members")
    if not isinstance(members, Sequence):
        return [success]
    return [
        atomic
        for member in members
        for atomic in _atomic_successes(member)
    ]


def _termination_for_direction(
    termination: Mapping[str, Any],
    *,
    direction: str,
) -> dict[str, Any]:
    result = dict(termination)
    result["direction"] = direction
    if "semantic_digest" in result:
        payload = {
            key: value
            for key, value in result.items()
            if key != "semantic_digest"
        }
        result["semantic_digest"] = stable_digest(payload)
    return result


def _axis_collinearity_evidence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    left_anchor = left["anchor"]
    right_anchor = right["anchor"]
    left_angle = float(left_anchor["axis_angle_deg"])
    right_angle = float(right_anchor["axis_angle_deg"])
    source_anchors = [*_source_anchors(left), *_source_anchors(right)]
    reference_angle = float(source_anchors[0]["axis_angle_deg"])
    measurements: list[dict[str, Any]] = []
    for anchor in source_anchors:
        angle = float(anchor["axis_angle_deg"])
        aligned_angle = reference_angle + (
            (angle - reference_angle + 90.0) % 180.0 - 90.0
        )
        resolution = anchor.get("axis_resolution_evidence")
        try:
            half_width = float(resolution["axis_precision_half_width_deg"])
        except (KeyError, TypeError, ValueError):
            half_width = math.nan
        measurements.append({
            "anchor_id": str(anchor["anchor_id"]),
            "axis_angle_deg": angle,
            "aligned_axis_angle_deg": aligned_angle,
            "axis_precision_half_width_deg": (
                half_width if math.isfinite(half_width) else None
            ),
        })
    valid_precision = all(
        measurement["axis_precision_half_width_deg"] is not None
        and measurement["axis_precision_half_width_deg"] >= 0.0
        for measurement in measurements
    )
    lower = max(
        measurement["aligned_axis_angle_deg"]
        - measurement["axis_precision_half_width_deg"]
        for measurement in measurements
    ) if valid_precision else math.nan
    upper = min(
        measurement["aligned_axis_angle_deg"]
        + measurement["axis_precision_half_width_deg"]
        for measurement in measurements
    ) if valid_precision else math.nan
    common_axis = (
        ((lower + upper) / 2.0) % 360.0
        if valid_precision and lower <= upper
        else _undirected_mean(left_angle, right_angle)
    )
    aligned_angles = [
        measurement["aligned_axis_angle_deg"]
        for measurement in measurements
    ]
    angle_delta = max(aligned_angles) - min(aligned_angles)
    allowed_delta = (
        sum(
            measurement["axis_precision_half_width_deg"]
            for measurement in measurements
        )
        if len(measurements) == 2 and valid_precision
        else None
    )
    angle_intervals_overlap = bool(
        valid_precision and lower <= upper
    )
    if not angle_intervals_overlap:
        return {
            "passed": False,
            "left_axis_angle_deg": left_angle,
            "right_axis_angle_deg": right_angle,
            "angle_delta_deg": angle_delta,
            "allowed_angle_delta_deg": allowed_delta,
            "common_axis_angle_deg": common_axis,
            "source_axis_measurements": measurements,
            "common_precision_interval_deg": None,
            "left_uv_bbox": None,
            "right_uv_bbox": None,
            "cross_axis_overlap": False,
        }, common_axis
    left_uv = _cells_uv_bbox(left["cells"], angle_deg=common_axis)
    right_uv = _cells_uv_bbox(right["cells"], angle_deg=common_axis)
    source_uv_bboxes = [
        list(_cells_uv_bbox(success["cells"], angle_deg=common_axis))
        for success in (*_atomic_successes(left), *_atomic_successes(right))
    ]
    cross_axis_overlap = max(
        bbox[1] for bbox in source_uv_bboxes
    ) <= min(
        bbox[3] for bbox in source_uv_bboxes
    )
    return {
        "passed": cross_axis_overlap,
        "left_axis_angle_deg": left_angle,
        "right_axis_angle_deg": right_angle,
        "angle_delta_deg": angle_delta,
        "allowed_angle_delta_deg": allowed_delta,
        "common_axis_angle_deg": common_axis,
        "source_axis_measurements": measurements,
        "common_precision_interval_deg": [lower, upper],
        "left_uv_bbox": list(left_uv),
        "right_uv_bbox": list(right_uv),
        "source_uv_bboxes": source_uv_bboxes,
        "cross_axis_overlap": cross_axis_overlap,
    }, common_axis


def _draw_order_non_interleaving_evidence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    drawing_cell_counts: Counter[int] = Counter()
    items_by_side: list[dict[int, list[int]]] = []
    for success in (left, right):
        side_items: dict[int, list[int]] = {}
        for cell in success["cells"]:
            drawings_in_cell: set[int] = set()
            for primitive_id in cell.get("primitive_ids") or ():
                match = _PRIMITIVE_ID.match(str(primitive_id))
                if match is None:
                    continue
                drawing = int(match.group("drawing"))
                side_items.setdefault(drawing, []).append(
                    int(match.group("item"))
                )
                drawings_in_cell.add(drawing)
            drawing_cell_counts.update(drawings_in_cell)
        items_by_side.append(side_items)
    if not drawing_cell_counts:
        return {
            "passed": False,
            "main_drawing_order": None,
            "left_item_index_interval": None,
            "right_item_index_interval": None,
            "relationship": "main_drawing_unavailable",
        }
    maximum = max(drawing_cell_counts.values())
    main_drawings = [
        drawing
        for drawing, count in drawing_cell_counts.items()
        if count == maximum
    ]
    if len(main_drawings) != 1:
        return {
            "passed": False,
            "main_drawing_order": None,
            "left_item_index_interval": None,
            "right_item_index_interval": None,
            "relationship": "main_drawing_tied",
        }
    main = main_drawings[0]
    if any(main not in side for side in items_by_side):
        return {
            "passed": False,
            "main_drawing_order": main,
            "left_item_index_interval": None,
            "right_item_index_interval": None,
            "relationship": "main_drawing_missing_from_fragment",
        }
    left_items = items_by_side[0][main]
    right_items = items_by_side[1][main]
    left_interval = [min(left_items), max(left_items)]
    right_interval = [min(right_items), max(right_items)]
    relationship = "interleaved"
    if left_interval[1] <= right_interval[0]:
        relationship = "left_before_right"
    elif right_interval[1] <= left_interval[0]:
        relationship = "right_before_left"
    return {
        "passed": relationship != "interleaved",
        "main_drawing_order": main,
        "left_item_index_interval": left_interval,
        "right_item_index_interval": right_interval,
        "relationship": relationship,
    }


def resolve_cell_direction_v1(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one ordered cell sequence, reversing only on positive evidence."""

    ordered = [dict(cell) for cell in cells]
    text = "".join(str(cell.get("char") or "") for cell in ordered)
    reversed_text = text[::-1]
    position_signals = {value for value in ("R", "Ø", "°") if value in text}
    forward_valid = _character_positions_valid(text)
    reverse_valid = _character_positions_valid(reversed_text)
    character_decision = None
    if position_signals and forward_valid != reverse_valid:
        character_decision = "forward" if forward_valid else "reverse"
    draw_order_decision = _draw_order_direction(ordered)
    forward_syntax = classify_dimension_syntax(text)
    reverse_syntax = classify_dimension_syntax(reversed_text)
    syntax_decision = None
    if (forward_syntax == SYNTAX_OK) != (reverse_syntax == SYNTAX_OK):
        syntax_decision = (
            "forward" if forward_syntax == SYNTAX_OK else "reverse"
        )
    selected = character_decision or draw_order_decision or syntax_decision
    should_reverse = selected == "reverse"
    criterion_decisions = {
        "character_position": character_decision,
        "draw_order": draw_order_decision,
        "syntax": syntax_decision,
    }
    decided = {value for value in criterion_decisions.values() if value is not None}
    return {
        "cells": list(reversed(ordered)) if should_reverse else ordered,
        "output_alternative_count": 1,
        "direction_source": (
            "character_position"
            if character_decision is not None
            else "draw_order"
            if draw_order_decision is not None
            else "syntax"
            if syntax_decision is not None
            else "kept_default"
        ),
        "reversed": should_reverse,
        "criterion_conflict": len(decided) > 1,
        "criterion_decisions": criterion_decisions,
        "position_signals": sorted(position_signals),
        "syntax_labels": {
            "forward": forward_syntax,
            "reverse": reverse_syntax,
        },
    }


def resolve_internal_cell_overlap_v1(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deduplicate exact cell claims without weakening phrase ownership."""

    copied = [dict(cell) for cell in cells]
    primitive_sets = [
        {str(value) for value in cell["primitive_ids"]}
        for cell in copied
    ]
    conflicts: list[dict[str, Any]] = []
    for index, left_set in enumerate(primitive_sets):
        for right_index in range(index + 1, len(primitive_sets)):
            right_set = primitive_sets[right_index]
            overlap = left_set.intersection(right_set)
            if overlap and left_set != right_set:
                conflicts.append({
                    "reason": "overlapping_nonidentical_cell_primitive_sets",
                    "primitive_ids": sorted(overlap),
                })
            elif (
                left_set == right_set
                and copied[index].get("char")
                != copied[right_index].get("char")
            ):
                conflicts.append({
                    "reason": "same_primitive_set_character_conflict",
                    "primitive_ids": sorted(left_set),
                })
    if conflicts:
        return {
            "compatible": False,
            "cells": copied,
            "deduplicated_cell_count": 0,
            "conflicts": conflicts,
        }
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for cell in copied:
        key = tuple(sorted(str(value) for value in cell["primitive_ids"]))
        unique.setdefault(key, cell)
    return {
        "compatible": True,
        "cells": list(unique.values()),
        "deduplicated_cell_count": len(copied) - len(unique),
        "conflicts": [],
    }


def _copy_success(success: Mapping[str, Any]) -> dict[str, Any]:
    copied = {
        **dict(success),
        "anchor": dict(success["anchor"]),
        "cells": [dict(cell) for cell in success["cells"]],
        "landing_group_rejections": [
            dict(row) for row in success.get("landing_group_rejections", ())
        ],
    }
    source_anchors = success.get("_source_anchors")
    copied["_source_anchors"] = [
        dict(anchor)
        for anchor in (
            source_anchors
            if isinstance(source_anchors, Sequence)
            else (success["anchor"],)
        )
    ]
    return copied


def _success_identity(success: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(
        str(anchor["anchor_id"])
        for anchor in success.get("_source_anchors", (success["anchor"],))
    ))


def _try_merge_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    axis_collinearity, common_axis = _axis_collinearity_evidence(left, right)
    if not axis_collinearity["passed"]:
        return None, None
    left_uv = tuple(axis_collinearity["left_uv_bbox"])
    right_uv = tuple(axis_collinearity["right_uv_bbox"])
    axis_interval_connection = {
        "passed": max(left_uv[0], right_uv[0]) <= min(
            left_uv[2], right_uv[2]
        ),
        "left_u_interval": [left_uv[0], left_uv[2]],
        "right_u_interval": [right_uv[0], right_uv[2]],
    }
    if not axis_interval_connection["passed"]:
        return None, None
    draw_order = _draw_order_non_interleaving_evidence(left, right)
    if not draw_order["passed"]:
        return None, None

    all_cells = [*left["cells"], *right["cells"]]
    cell_union = resolve_internal_cell_overlap_v1(all_cells)
    if not cell_union["compatible"]:
        return None, None
    merged_cells = sorted(
        cell_union["cells"],
        key=lambda cell: (
            _cell_u_center(cell, angle_deg=common_axis),
            _cells_uv_bbox([cell], angle_deg=common_axis)[1],
            tuple(cell["primitive_ids"]),
        ),
    )
    source_direction_resolutions = [
        resolve_cell_direction_v1(row["cells"])
        for row in (left, right)
    ]
    before = [
        "".join(str(cell["char"]) for cell in resolution["cells"])
        for resolution in source_direction_resolutions
    ]
    before_syntax = [classify_dimension_syntax(text) for text in before]
    merged_direction_resolution = resolve_cell_direction_v1(merged_cells)
    merged_text = "".join(
        str(cell["char"])
        for cell in merged_direction_resolution["cells"]
    )
    merged_syntax = classify_dimension_syntax(merged_text)
    syntax_non_degradation = {
        "passed": not (
            SYNTAX_OK in before_syntax and merged_syntax != SYNTAX_OK
        ),
        "before": before_syntax,
        "after": merged_syntax,
    }
    if not syntax_non_degradation["passed"]:
        return None, None

    representative = left
    source_anchors_by_id = {
        str(anchor["anchor_id"]): dict(anchor)
        for row in (left, right)
        for anchor in row.get("_source_anchors", (row["anchor"],))
    }
    source_anchors = [
        source_anchors_by_id[anchor_id]
        for anchor_id in sorted(source_anchors_by_id)
    ]
    merged_anchor = dict(representative["anchor"])
    merged_anchor["merged_anchor_ids"] = sorted(source_anchors_by_id)
    merged_anchor["source_candidate_ids"] = sorted({
        str(value)
        for anchor in source_anchors
        for value in anchor.get("source_candidate_ids", ())
    })
    merged_anchor["source_region_ids"] = sorted({
        str(value)
        for anchor in source_anchors
        for value in anchor.get("source_region_ids", ())
    })
    atomic_extents = [
        (
            _cells_uv_bbox(success["cells"], angle_deg=common_axis),
            success,
        )
        for success in (*_atomic_successes(left), *_atomic_successes(right))
    ]
    lower = min(
        atomic_extents,
        key=lambda row: (row[0][0], _success_identity(row[1])),
    )[1]
    upper = max(
        atomic_extents,
        key=lambda row: (row[0][2], _success_identity(row[1])),
    )[1]

    def source_termination(
        success: Mapping[str, Any],
        *,
        side: str,
    ) -> dict[str, Any]:
        aligned = math.cos(math.radians(
            float(success["anchor"]["axis_angle_deg"]) - common_axis
        )) >= 0.0
        source_direction = (
            side if aligned else "positive" if side == "negative" else "negative"
        )
        return _termination_for_direction(
            success[f"{source_direction}_termination"],
            direction=side,
        )

    negative_termination = source_termination(lower, side="negative")
    positive_termination = source_termination(upper, side="positive")
    claimed_primitive_ids = {
        str(value)
        for cell in merged_cells
        for value in cell["primitive_ids"]
    }
    outer_terminations = (
        negative_termination,
        positive_termination,
    )
    if any(
        claimed_primitive_ids.intersection(
            str(value) for value in termination.get("primitive_ids", ())
        )
        for termination in outer_terminations
    ):
        return None, None
    merged = {
        **dict(representative),
        "anchor": merged_anchor,
        "cells": merged_cells,
        "negative_termination": negative_termination,
        "positive_termination": positive_termination,
        "landing_group_rejections": [
            *left.get("landing_group_rejections", ()),
            *right.get("landing_group_rejections", ()),
        ],
        "_source_anchors": source_anchors,
        "_merge_members": [_copy_success(left), _copy_success(right)],
        "_input_order": min(
            int(left.get("_input_order", 0)),
            int(right.get("_input_order", 0)),
        ),
    }
    audit = {
        "source_anchor_ids": sorted(source_anchors_by_id),
        "input_fragment_anchor_ids": [
            list(_success_identity(left)),
            list(_success_identity(right)),
        ],
        "source_anchor_evidence": [
            dict(anchor["source_anchor_evidence"])
            for anchor in source_anchors
            if "source_anchor_evidence" in anchor
        ],
        "source_texts": before,
        "merged_text": merged_text,
        "source_direction_resolutions": [
            {
                "direction_source": resolution["direction_source"],
                "reversed": resolution["reversed"],
                "criterion_decisions": dict(
                    resolution["criterion_decisions"]
                ),
            }
            for resolution in source_direction_resolutions
        ],
        "merged_direction_resolution": {
            "direction_source": merged_direction_resolution[
                "direction_source"
            ],
            "reversed": merged_direction_resolution["reversed"],
            "criterion_decisions": dict(
                merged_direction_resolution["criterion_decisions"]
            ),
        },
        "axis_collinearity": axis_collinearity,
        "axis_interval_connection": axis_interval_connection,
        "draw_order_non_interleaving": draw_order,
        "syntax_non_degradation": syntax_non_degradation,
        "cell_union": {
            "passed": True,
            "input_cell_count": len(all_cells),
            "output_cell_count": len(merged_cells),
            "deduplicated_exact_cell_count": cell_union[
                "deduplicated_cell_count"
            ],
        },
        "termination_selection": {
            "passed": True,
            "negative_source_anchor_ids": list(_success_identity(lower)),
            "positive_source_anchor_ids": list(_success_identity(upper)),
            "negative_semantic_digest": negative_termination.get(
                "semantic_digest"
            ),
            "positive_semantic_digest": positive_termination.get(
                "semantic_digest"
            ),
        },
    }
    return merged, audit


def resolve_fragment_merges_v1(
    successes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Coalesce fragment successes without consulting gold or downstream state."""

    rows = sorted(
        (_copy_success(success) for success in successes),
        key=_success_identity,
    )
    merges: list[dict[str, Any]] = []
    while True:
        accepted = None
        for left_index, left in enumerate(rows):
            for right_index in range(left_index + 1, len(rows)):
                merged, audit = _try_merge_pair(left, rows[right_index])
                if merged is not None and audit is not None:
                    accepted = (left_index, right_index, merged, audit)
                    break
            if accepted is not None:
                break
        if accepted is None:
            break
        left_index, right_index, merged, audit = accepted
        rows = [
            row
            for index, row in enumerate(rows)
            if index not in {left_index, right_index}
        ]
        rows.append(merged)
        rows.sort(key=_success_identity)
        merges.append(audit)
    return {
        "successes": rows,
        "diagnostics": {
            "schema_version": "r92_fragment_merge_diagnostics_v1",
            "input_success_count": len(successes),
            "output_success_count": len(rows),
            "actual_merge_count": len(merges),
            "unexplained_success_loss_count": (
                len(successes) - len(rows) - len(merges)
            ),
            "merges": merges,
            "consumer_allowed": False,
        },
    }


__all__ = [
    "resolve_cell_direction_v1",
    "resolve_fragment_merges_v1",
    "resolve_internal_cell_overlap_v1",
]
