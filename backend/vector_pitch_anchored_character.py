"""Fail-closed pre-core exclusion for proven paired vector enclosures."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any

from vector_pitch_anchored_lane import normalize_anchored_local_lane
from vector_pitch_window_authority import (
    normalize_pitch_window_authority,
)


SCHEMA_VERSION = "vector_pitch_anchored_character_partition_v1"
PROOF_SCHEMA_VERSION = "vector_pitch_paired_enclosure_proof_v1"
EXCLUSION_SCHEMA_VERSION = "vector_pitch_primitive_exclusion_v1"
CONSUMER_ALLOWED = False
COORDINATE_JOIN_TOLERANCE_PT = 0.002
MIN_CHAIN_SEGMENTS = 4
MIN_PAIR_SIMILARITY = 0.90
MIN_CROSS_IOU = 0.90
MAX_CROSS_CENTER_DELTA_RATIO = 0.05
MAX_MIRROR_RMS_DEPTH_RATIO = 0.10
MAX_MIRROR_ERROR_DEPTH_RATIO = 0.20
MIN_DEPTH_TO_CROSS_SPAN = 0.15
MAX_DEPTH_TO_CROSS_SPAN = 0.45
MIN_PATH_TO_CROSS_SPAN = 1.05
MAX_PATH_TO_CROSS_SPAN = 1.60
MIN_ENCLOSURE_TO_BODY_CROSS_RATIO = 1.50
MAX_ENCLOSURE_TO_BODY_CROSS_RATIO = 2.50
MAX_BODY_CENTER_OFFSET_RATIO = 0.05
_PRIMITIVE_ID_RE = re.compile(
    r"^p[0-9]+_d(?P<drawing>[0-9]+)_i(?P<item>[0-9]+)_l$"
)


def prepare_anchored_character_partition(
    items: Any,
    *,
    window_authority: Any,
    anchored_lane_plan: Any,
    protected_primitive_ids: Any = None,
    boundary_cut_primitive_ids: Any = None,
) -> dict[str, Any]:
    """Exclude ink only when one complete paired-enclosure proof is unique."""
    authority = normalize_pitch_window_authority(window_authority)
    lane = normalize_anchored_local_lane(anchored_lane_plan)
    if authority is None or lane is None or not isinstance(items, list):
        return _failure(
            reason="anchored_character_input_invalid",
            authority=authority,
            lane=lane,
        )
    phrase_id = authority["phrase_id"]
    if (
        lane["phrase_id"] != phrase_id
        or lane["partition_digest"]
        != anchored_lane_plan.get("partition_digest")
        or authority["axis_angle_deg"] != lane["axis_angle_deg"]
    ):
        return _failure(
            reason="anchored_character_authority_mismatch",
            authority=authority,
            lane=lane,
        )
    rows = [_project_item(item, authority=authority) for item in items]
    if any(row is None for row in rows):
        return _failure(
            reason="anchored_character_input_invalid",
            authority=authority,
            lane=lane,
        )
    projected = [row for row in rows if row is not None]
    item_ids = sorted(row["primitive_id"] for row in projected)
    if (
        len(item_ids) != len(set(item_ids))
        or item_ids != lane["admitted_primitive_ids"]
    ):
        return _failure(
            reason="anchored_character_lane_partition_mismatch",
            authority=authority,
            lane=lane,
            input_ids=item_ids,
        )
    protected_ids = _canonical_optional_ids(
        protected_primitive_ids,
        default=authority["group_member_source_primitive_ids"],
    )
    cut_ids = _canonical_optional_ids(
        boundary_cut_primitive_ids,
        default=[],
    )
    if (
        protected_ids is None
        or cut_ids is None
        or not set(protected_ids).issubset(item_ids)
    ):
        return _failure(
            reason="anchored_character_input_invalid",
            authority=authority,
            lane=lane,
            input_ids=item_ids,
        )
    chains = _candidate_open_bow_chains(projected)
    if (
        authority.get("schema_version")
        == "vector_pitch_window_authority_v2"
        and chains
    ):
        return _failure(
            reason="anchored_character_group_ownership_unresolved",
            authority=authority,
            lane=lane,
            input_ids=item_ids,
            candidate_count=len(chains),
            protected_ids=protected_ids,
            cut_ids=cut_ids,
        )
    pairs = [
        proof
        for left_index, left in enumerate(chains)
        for right in chains[left_index + 1:]
        if (
            proof := _paired_enclosure_proof(
                left,
                right,
                projected=projected,
                authority=authority,
                lane=lane,
                protected_ids=protected_ids,
                cut_ids=cut_ids,
            )
        )
        is not None
    ]
    if len(pairs) > 1:
        return _failure(
            reason="anchored_character_enclosure_ambiguous",
            authority=authority,
            lane=lane,
            input_ids=item_ids,
            candidate_count=len(pairs),
            protected_ids=protected_ids,
            cut_ids=cut_ids,
        )
    proofs = pairs
    excluded_ids = (
        sorted([
            *proofs[0]["main_low_primitive_ids"],
            *proofs[0]["main_high_primitive_ids"],
        ])
        if proofs
        else []
    )
    admitted_ids = sorted(set(item_ids) - set(excluded_ids))
    exclusions = (
        [_exclusion_from_proof(
            proofs[0],
            authority=authority,
        )]
        if proofs
        else []
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "reason": None,
        "phrase_id": phrase_id,
        "authority_digest": authority["authority_digest"],
        "anchored_lane_partition_digest": lane["partition_digest"],
        "axis_angle_deg": authority["axis_angle_deg"],
        "input_primitive_ids": item_ids,
        "input_primitive_count": len(item_ids),
        "admitted_primitive_ids": admitted_ids,
        "admitted_primitive_count": len(admitted_ids),
        "excluded_primitive_ids": excluded_ids,
        "excluded_primitive_count": len(excluded_ids),
        "protected_primitive_ids": protected_ids,
        "boundary_cut_primitive_ids": cut_ids,
        "paired_enclosure_candidate_count": len(proofs),
        "paired_enclosure_proofs": proofs,
        "exclusions": exclusions,
        "rule_contract": _rule_contract(),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "partition_digest": _digest(payload),
    }


def normalize_anchored_character_partition(
    value: Any,
    *,
    window_authority: Any = None,
    anchored_lane_plan: Any = None,
) -> dict[str, Any] | None:
    """Validate exact ownership, proof arithmetic, and every digest binding."""
    authority = (
        normalize_pitch_window_authority(window_authority)
        if window_authority is not None
        else None
    )
    lane = (
        normalize_anchored_local_lane(anchored_lane_plan)
        if anchored_lane_plan is not None
        else None
    )
    if (
        (window_authority is not None and authority is None)
        or (anchored_lane_plan is not None and lane is None)
    ):
        return None
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "status",
        "reason",
        "phrase_id",
        "authority_digest",
        "anchored_lane_partition_digest",
        "axis_angle_deg",
        "input_primitive_ids",
        "input_primitive_count",
        "admitted_primitive_ids",
        "admitted_primitive_count",
        "excluded_primitive_ids",
        "excluded_primitive_count",
        "protected_primitive_ids",
        "boundary_cut_primitive_ids",
        "paired_enclosure_candidate_count",
        "paired_enclosure_proofs",
        "exclusions",
        "rule_contract",
        "consumer_allowed",
        "partition_digest",
    }:
        return None
    input_ids = _canonical_ids(value.get("input_primitive_ids"))
    admitted_ids = _canonical_ids(value.get("admitted_primitive_ids"))
    excluded_ids = _canonical_ids(value.get("excluded_primitive_ids"))
    protected_ids = _canonical_ids(value.get("protected_primitive_ids"))
    cut_ids = _canonical_ids(value.get("boundary_cut_primitive_ids"))
    axis_angle = _canonical_axis_angle(value.get("axis_angle_deg"))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == "ready"
        and value.get("reason") is None
        and type(value.get("phrase_id")) is str
        and bool(value["phrase_id"])
        and _valid_sha256(value.get("authority_digest"))
        and _valid_sha256(
            value.get("anchored_lane_partition_digest")
        )
        and axis_angle is not None
        and input_ids is not None
        and input_ids
        and admitted_ids is not None
        and excluded_ids is not None
        and protected_ids is not None
        and cut_ids is not None
        and set(admitted_ids).isdisjoint(excluded_ids)
        and set(admitted_ids).union(excluded_ids) == set(input_ids)
        and set(protected_ids).issubset(admitted_ids)
        and not set(cut_ids).intersection(excluded_ids)
        and value.get("input_primitive_count") == len(input_ids)
        and value.get("admitted_primitive_count") == len(admitted_ids)
        and value.get("excluded_primitive_count") == len(excluded_ids)
        and value.get("rule_contract") == _rule_contract()
        and value.get("consumer_allowed") is False
    ):
        return None
    raw_proofs = value.get("paired_enclosure_proofs")
    if not isinstance(raw_proofs, list) or len(raw_proofs) > 1:
        return None
    proofs = [
        _normalize_proof(
            proof,
            phrase_id=value["phrase_id"],
            authority_digest=value["authority_digest"],
            anchored_lane_partition_digest=value[
                "anchored_lane_partition_digest"
            ],
            expected_axis_angle=axis_angle,
            authority=authority,
        )
        for proof in raw_proofs
    ]
    if any(proof is None for proof in proofs):
        return None
    canonical_proofs = [
        proof for proof in proofs if proof is not None
    ]
    if (
        value.get("paired_enclosure_candidate_count")
        != len(canonical_proofs)
    ):
        return None
    exclusions = _normalize_exclusions(
        value.get("exclusions"),
        authority_digest=value["authority_digest"],
        axis_angle=axis_angle,
        proofs=canonical_proofs,
        authority=authority,
    )
    if exclusions is None:
        return None
    proof_excluded_ids = sorted({
        primitive_id
        for proof in canonical_proofs
        for primitive_id in [
            *proof["main_low_primitive_ids"],
            *proof["main_high_primitive_ids"],
        ]
    })
    exclusion_ids = sorted({
        primitive_id
        for exclusion in exclusions
        for primitive_id in exclusion["primitive_ids"]
    })
    if (
        proof_excluded_ids != excluded_ids
        or exclusion_ids != excluded_ids
        or bool(canonical_proofs) != bool(exclusions)
        or any(
            proof["interior_primitive_ids"] != admitted_ids
            for proof in canonical_proofs
        )
    ):
        return None
    if authority is not None and not (
        value["phrase_id"] == authority["phrase_id"]
        and value["authority_digest"] == authority["authority_digest"]
        and axis_angle == authority["axis_angle_deg"]
        and protected_ids
        == authority["group_member_source_primitive_ids"]
        and all(
            proof["source_primitive_id"]
            == authority["source_primitive_id"]
            for proof in canonical_proofs
        )
    ):
        return None
    if lane is not None and not (
        value["phrase_id"] == lane["phrase_id"]
        and value["anchored_lane_partition_digest"]
        == lane["partition_digest"]
        and axis_angle == lane["axis_angle_deg"]
        and input_ids == lane["admitted_primitive_ids"]
    ):
        return None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "reason": None,
        "phrase_id": value["phrase_id"],
        "authority_digest": value["authority_digest"],
        "anchored_lane_partition_digest": value[
            "anchored_lane_partition_digest"
        ],
        "axis_angle_deg": axis_angle,
        "input_primitive_ids": input_ids,
        "input_primitive_count": len(input_ids),
        "admitted_primitive_ids": admitted_ids,
        "admitted_primitive_count": len(admitted_ids),
        "excluded_primitive_ids": excluded_ids,
        "excluded_primitive_count": len(excluded_ids),
        "protected_primitive_ids": protected_ids,
        "boundary_cut_primitive_ids": cut_ids,
        "paired_enclosure_candidate_count": len(canonical_proofs),
        "paired_enclosure_proofs": canonical_proofs,
        "exclusions": exclusions,
        "rule_contract": _rule_contract(),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if value.get("partition_digest") != _digest(payload):
        return None
    canonical = {
        **payload,
        "partition_digest": value["partition_digest"],
    }
    return canonical if canonical == value else None


def _paired_enclosure_proof(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    projected: list[dict[str, Any]],
    authority: dict[str, Any],
    lane: dict[str, Any],
    protected_ids: list[str],
    cut_ids: list[str],
) -> dict[str, Any] | None:
    low, high = sorted(
        (first, second),
        key=lambda chain: chain["main_center"],
    )
    source_main = float(authority["dot_main_center"])
    pair_ids = {
        *low["primitive_ids"],
        *high["primitive_ids"],
    }
    if (
        low["main_max"] >= source_main
        or high["main_min"] <= source_main
        or low["outward_sign"] != -1
        or high["outward_sign"] != 1
        or low["segment_count"] != high["segment_count"]
        or low["drawing_order"] != high["drawing_order"]
        or abs(low["stroke_width"] - high["stroke_width"]) > 1e-6
        or pair_ids.intersection(protected_ids)
        or pair_ids.intersection(cut_ids)
        or not _adjacent_item_runs(low, high)
    ):
        return None
    low_vertices = low["vertices"]
    high_vertices = high["vertices"]
    cross_errors = [
        abs(left[1] - right[1])
        for left, right in zip(low_vertices, high_vertices, strict=True)
    ]
    cross_span_low = low["cross_span"]
    cross_span_high = high["cross_span"]
    cross_min = max(low["cross_min"], high["cross_min"])
    cross_max = min(low["cross_max"], high["cross_max"])
    cross_intersection = max(0.0, cross_max - cross_min)
    cross_union = max(
        low["cross_max"],
        high["cross_max"],
    ) - min(low["cross_min"], high["cross_min"])
    cross_iou = cross_intersection / max(cross_union, 1e-12)
    cross_center_delta_ratio = abs(
        low["cross_center"] - high["cross_center"]
    ) / max(low["cross_span"], high["cross_span"], 1e-12)
    midpoint_mains = [
        (left[0] + right[0]) / 2.0
        for left, right in zip(low_vertices, high_vertices, strict=True)
    ]
    mirror_center = sum(midpoint_mains) / len(midpoint_mains)
    mirror_errors = [
        abs(value - mirror_center) for value in midpoint_mains
    ]
    max_mirror_error = max(mirror_errors)
    mirror_rms = math.sqrt(
        sum(error * error for error in mirror_errors)
        / len(mirror_errors)
    )
    mean_depth = (low["bow_depth"] + high["bow_depth"]) / 2.0
    depth_similarity = _similarity(
        low["bow_depth"],
        high["bow_depth"],
    )
    path_similarity = _similarity(
        low["path_length"],
        high["path_length"],
    )
    span_similarity = _similarity(
        cross_span_low,
        cross_span_high,
    )
    low_lengths = _segment_lengths(low_vertices)
    high_lengths = _segment_lengths(high_vertices)
    segment_similarity = min(
        _similarity(left, right)
        for left, right in zip(
            low_lengths,
            high_lengths,
            strict=True,
        )
    )
    interior = [
        row for row in projected
        if row["primitive_id"] not in pair_ids
    ]
    if not interior:
        return None
    interior_points = [
        point for row in interior for point in row["projected_points"]
    ]
    interior_main_min = min(point[0] for point in interior_points)
    interior_main_max = max(point[0] for point in interior_points)
    interior_cross_min = min(point[1] for point in interior_points)
    interior_cross_max = max(point[1] for point in interior_points)
    low_inner_gap = interior_main_min - low["main_max"]
    high_inner_gap = high["main_min"] - interior_main_max
    cross_margin_low = interior_cross_min - cross_min
    cross_margin_high = cross_max - interior_cross_max
    body_cross_span = interior_cross_max - interior_cross_min
    enclosure_cross_span = (cross_span_low + cross_span_high) / 2.0
    enclosure_to_body_ratio = (
        enclosure_cross_span / max(body_cross_span, 1e-12)
    )
    body_cross_center = (
        interior_cross_min + interior_cross_max
    ) / 2.0
    enclosure_cross_center = (
        low["cross_center"] + high["cross_center"]
    ) / 2.0
    body_center_offset_ratio = abs(
        body_cross_center - enclosure_cross_center
    ) / max(enclosure_cross_span, 1e-12)
    if not (
        max(cross_errors) <= COORDINATE_JOIN_TOLERANCE_PT
        and min(
            span_similarity,
            depth_similarity,
            path_similarity,
            segment_similarity,
        )
        >= MIN_PAIR_SIMILARITY
        and cross_iou >= MIN_CROSS_IOU
        and cross_center_delta_ratio
        <= MAX_CROSS_CENTER_DELTA_RATIO
        and mirror_rms / max(mean_depth, 1e-12)
        <= MAX_MIRROR_RMS_DEPTH_RATIO
        and max_mirror_error / max(mean_depth, 1e-12)
        <= MAX_MIRROR_ERROR_DEPTH_RATIO
        and all(
            MIN_DEPTH_TO_CROSS_SPAN
            <= chain["bow_depth"] / chain["cross_span"]
            <= MAX_DEPTH_TO_CROSS_SPAN
            for chain in (low, high)
        )
        and all(
            MIN_PATH_TO_CROSS_SPAN
            <= chain["path_length"] / chain["cross_span"]
            <= MAX_PATH_TO_CROSS_SPAN
            for chain in (low, high)
        )
        and low_inner_gap > COORDINATE_JOIN_TOLERANCE_PT
        and high_inner_gap > COORDINATE_JOIN_TOLERANCE_PT
        and cross_margin_low >= -COORDINATE_JOIN_TOLERANCE_PT
        and cross_margin_high >= -COORDINATE_JOIN_TOLERANCE_PT
        and MIN_ENCLOSURE_TO_BODY_CROSS_RATIO
        <= enclosure_to_body_ratio
        <= MAX_ENCLOSURE_TO_BODY_CROSS_RATIO
        and body_center_offset_ratio <= MAX_BODY_CENTER_OFFSET_RATIO
        and authority["source_primitive_id"]
        in {row["primitive_id"] for row in interior}
    ):
        return None
    proof_payload = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "phrase_id": authority["phrase_id"],
        "authority_digest": authority["authority_digest"],
        "anchored_lane_partition_digest": lane["partition_digest"],
        "axis_angle_deg": authority["axis_angle_deg"],
        "source_primitive_id": authority["source_primitive_id"],
        "main_low_primitive_ids": sorted(low["primitive_ids"]),
        "main_high_primitive_ids": sorted(high["primitive_ids"]),
        "interior_primitive_ids": sorted(
            row["primitive_id"] for row in interior
        ),
        "main_low_vertices": _rounded_vertices(low_vertices),
        "main_high_vertices": _rounded_vertices(high_vertices),
        "segment_count": low["segment_count"],
        "mirror_center_main": round(mirror_center, 6),
        "max_connection_error_pt": round(max(
            low["max_connection_error"],
            high["max_connection_error"],
        ), 6),
        "max_cross_alignment_error_pt": round(max(cross_errors), 6),
        "max_mirror_error_pt": round(max_mirror_error, 6),
        "mirror_rms_error_pt": round(mirror_rms, 6),
        "main_low_bow_depth_pt": round(low["bow_depth"], 6),
        "main_high_bow_depth_pt": round(high["bow_depth"], 6),
        "main_low_path_length_pt": round(low["path_length"], 6),
        "main_high_path_length_pt": round(high["path_length"], 6),
        "cross_iou": round(cross_iou, 6),
        "span_similarity": round(span_similarity, 6),
        "depth_similarity": round(depth_similarity, 6),
        "path_similarity": round(path_similarity, 6),
        "segment_similarity": round(segment_similarity, 6),
        "cross_center_delta_ratio": round(
            cross_center_delta_ratio,
            6,
        ),
        "mirror_rms_depth_ratio": round(
            mirror_rms / mean_depth,
            6,
        ),
        "mirror_max_depth_ratio": round(
            max_mirror_error / mean_depth,
            6,
        ),
        "main_low_inner_gap_pt": round(low_inner_gap, 6),
        "main_high_inner_gap_pt": round(high_inner_gap, 6),
        "cross_margin_low_pt": round(cross_margin_low, 6),
        "cross_margin_high_pt": round(cross_margin_high, 6),
        "interior_main_min": round(interior_main_min, 6),
        "interior_main_max": round(interior_main_max, 6),
        "interior_cross_min": round(interior_cross_min, 6),
        "interior_cross_max": round(interior_cross_max, 6),
        "enclosure_to_body_cross_ratio": round(
            enclosure_to_body_ratio,
            6,
        ),
        "body_center_offset_ratio": round(
            body_center_offset_ratio,
            6,
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **proof_payload,
        "proof_digest": _digest(proof_payload),
    }


def _candidate_open_bow_chains(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lines = [
        row for row in rows
        if (
            row["op"] == "l"
            and len(row["projected_points"]) == 2
            and row["draw_type"] == "s"
            and row["drawing_order"] is not None
            and row["item_index"] is not None
            and row["stroke_width"] is not None
        )
    ]
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left_index, left in enumerate(lines):
        for right_index in range(left_index + 1, len(lines)):
            right = lines[right_index]
            if (
                left["drawing_order"] == right["drawing_order"]
                and abs(
                    left["stroke_width"] - right["stroke_width"]
                )
                <= 1e-6
                and _segments_touch(
                    left["projected_points"],
                    right["projected_points"],
                )
            ):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    components: list[list[int]] = []
    remaining = set(range(len(lines)))
    while remaining:
        start = min(remaining)
        stack = [start]
        component: list[int] = []
        remaining.remove(start)
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in adjacency[index]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)
    chains = [
        chain
        for component in components
        if (
            chain := _ordered_open_bow_chain(
                [lines[index] for index in component]
            )
        )
        is not None
    ]
    return sorted(chains, key=lambda chain: (
        chain["main_center"],
        chain["primitive_ids"],
    ))


def _ordered_open_bow_chain(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(rows) < MIN_CHAIN_SEGMENTS:
        return None
    row_by_id = {
        row["primitive_id"]: row for row in rows
    }
    if len(row_by_id) != len(rows):
        return None
    neighbor_ids: dict[str, list[str]] = {}
    for row in rows:
        neighbor_ids[row["primitive_id"]] = sorted(
            other["primitive_id"]
            for other in rows
            if (
                other is not row
                and _segments_touch(
                    row["projected_points"],
                    other["projected_points"],
                )
            )
        )
    endpoints = sorted(
        primitive_id
        for primitive_id, neighbors in neighbor_ids.items()
        if len(neighbors) == 1
    )
    if (
        len(endpoints) != 2
        or any(
            len(neighbors) not in {1, 2}
            for neighbors in neighbor_ids.values()
        )
    ):
        return None
    ordered_ids: list[str] = []
    prior: str | None = None
    current = endpoints[0]
    while True:
        ordered_ids.append(current)
        next_ids = [
            value for value in neighbor_ids[current]
            if value != prior
        ]
        if not next_ids:
            break
        if len(next_ids) != 1 or next_ids[0] in ordered_ids:
            return None
        prior, current = current, next_ids[0]
    if len(ordered_ids) != len(rows):
        return None
    vertices: list[tuple[float, float]] = []
    max_connection_error = 0.0
    for index, primitive_id in enumerate(ordered_ids):
        first, second = row_by_id[primitive_id]["projected_points"]
        if index == 0:
            next_points = row_by_id[ordered_ids[1]][
                "projected_points"
            ]
            if _point_to_segment_endpoint_distance(
                first,
                next_points,
            ) <= _point_to_segment_endpoint_distance(
                second,
                next_points,
            ):
                first, second = second, first
            vertices.extend([first, second])
            continue
        left_distance = _distance(vertices[-1], first)
        right_distance = _distance(vertices[-1], second)
        if min(left_distance, right_distance) > (
            COORDINATE_JOIN_TOLERANCE_PT
        ):
            return None
        if right_distance < left_distance:
            first, second = second, first
            left_distance = right_distance
        max_connection_error = max(
            max_connection_error,
            left_distance,
        )
        vertices.append(second)
    if _distance(vertices[0], vertices[-1]) <= (
        COORDINATE_JOIN_TOLERANCE_PT
    ):
        return None
    if vertices[0][1] < vertices[-1][1]:
        vertices = list(reversed(vertices))
        ordered_ids = list(reversed(ordered_ids))
    cross_steps = [
        vertices[index + 1][1] - vertices[index][1]
        for index in range(len(vertices) - 1)
    ]
    if any(step >= -COORDINATE_JOIN_TOLERANCE_PT for step in cross_steps):
        return None
    cross_span = vertices[0][1] - vertices[-1][1]
    if cross_span <= 0.0:
        return None
    start_main, start_cross = vertices[0]
    end_main, end_cross = vertices[-1]
    deviations = []
    for main, cross in vertices[1:-1]:
        fraction = (
            (cross - start_cross) / (end_cross - start_cross)
        )
        chord_main = start_main + fraction * (
            end_main - start_main
        )
        deviations.append(main - chord_main)
    if not deviations:
        return None
    peak = max(deviations, key=abs)
    outward_sign = 1 if peak > 0.0 else -1
    if (
        abs(peak) <= COORDINATE_JOIN_TOLERANCE_PT * 5.0
        or any(
            outward_sign * deviation
            < -COORDINATE_JOIN_TOLERANCE_PT
            for deviation in deviations
        )
        or abs(start_main - end_main)
        > max(
            COORDINATE_JOIN_TOLERANCE_PT,
            abs(peak) * 0.10,
        )
    ):
        return None
    item_indices = sorted(row["item_index"] for row in rows)
    if item_indices != list(range(
        item_indices[0],
        item_indices[0] + len(item_indices),
    )):
        return None
    all_points = [
        point for row in rows for point in row["raw_points"]
    ]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    main_values = [point[0] for point in vertices]
    cross_values = [point[1] for point in vertices]
    return {
        "primitive_ids": ordered_ids,
        "drawing_order": rows[0]["drawing_order"],
        "item_index_min": min(item_indices),
        "item_index_max": max(item_indices),
        "stroke_width": rows[0]["stroke_width"],
        "vertices": vertices,
        "segment_count": len(rows),
        "max_connection_error": max_connection_error,
        "outward_sign": outward_sign,
        "bow_depth": abs(peak),
        "path_length": sum(_segment_lengths(vertices)),
        "main_min": min(main_values),
        "main_max": max(main_values),
        "main_center": (
            min(main_values) + max(main_values)
        ) / 2.0,
        "cross_min": min(cross_values),
        "cross_max": max(cross_values),
        "cross_center": (
            min(cross_values) + max(cross_values)
        ) / 2.0,
        "cross_span": max(cross_values) - min(cross_values),
        "bbox": [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ],
    }


def _project_item(
    item: Any,
    *,
    authority: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    primitive_id = item.get("item_id")
    op = item.get("op")
    raw_points = item.get("points")
    if (
        type(primitive_id) is not str
        or not primitive_id
        or type(op) is not str
        or not isinstance(raw_points, list)
        or len(raw_points) < 2
    ):
        return None
    points: list[tuple[float, float]] = []
    for raw_point in raw_points:
        if (
            not isinstance(raw_point, (list, tuple))
            or len(raw_point) < 2
        ):
            return None
        try:
            point = (float(raw_point[0]), float(raw_point[1]))
        except (TypeError, ValueError, OverflowError):
            return None
        if any(not math.isfinite(value) for value in point):
            return None
        points.append(point)
    main_unit = authority["main_unit"]
    cross_unit = authority["cross_unit"]
    projected_points = [
        (
            x * float(main_unit["x"]) + y * float(main_unit["y"]),
            x * float(cross_unit["x"]) + y * float(cross_unit["y"]),
        )
        for x, y in points
    ]
    match = _PRIMITIVE_ID_RE.fullmatch(primitive_id)
    stroke_width = item.get("stroke_width")
    try:
        normalized_stroke = float(stroke_width)
    except (TypeError, ValueError, OverflowError):
        normalized_stroke = None
    if (
        normalized_stroke is not None
        and not math.isfinite(normalized_stroke)
    ):
        normalized_stroke = None
    return {
        "primitive_id": primitive_id,
        "op": op,
        "raw_points": points,
        "projected_points": projected_points,
        "draw_type": str(item.get("draw_type") or ""),
        "stroke_width": normalized_stroke,
        "drawing_order": (
            int(match.group("drawing")) if match is not None else None
        ),
        "item_index": (
            int(match.group("item")) if match is not None else None
        ),
    }


def _exclusion_from_proof(
    proof: dict[str, Any],
    *,
    authority: dict[str, Any],
) -> dict[str, Any]:
    vertices = [
        *proof["main_low_vertices"],
        *proof["main_high_vertices"],
    ]
    main_values = [point[0] for point in vertices]
    cross_values = [point[1] for point in vertices]
    main_min = min(main_values)
    main_max = max(main_values)
    cross_min = min(cross_values)
    cross_max = max(cross_values)
    raw_points = [
        (
            main * float(authority["main_unit"]["x"])
            + cross * float(authority["cross_unit"]["x"]),
            main * float(authority["main_unit"]["y"])
            + cross * float(authority["cross_unit"]["y"]),
        )
        for main, cross in vertices
    ]
    xs = [point[0] for point in raw_points]
    ys = [point[1] for point in raw_points]
    main_span = main_max - main_min
    cross_span = cross_max - cross_min
    char_width = float(authority["char_width_pt"])
    char_height = float(authority["char_height_pt"])
    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "primitive_ids": sorted([
            *proof["main_low_primitive_ids"],
            *proof["main_high_primitive_ids"],
        ]),
        "stage": "anchored_character_filter",
        "reason": "paired_curved_enclosure",
        "bbox": [
            round(min(xs), 6),
            round(min(ys), 6),
            round(max(xs), 6),
            round(max(ys), 6),
        ],
        "canonical_bbox": {
            "main_min": round(main_min, 6),
            "cross_min": round(cross_min, 6),
            "main_max": round(main_max, 6),
            "cross_max": round(cross_max, 6),
        },
        "measurements": {
            "main_span_pt": round(main_span, 6),
            "cross_span_pt": round(cross_span, 6),
            "char_width_pt": round(char_width, 6),
            "char_height_pt": round(char_height, 6),
            "main_to_char_width_ratio": round(
                main_span / char_width,
                6,
            ),
            "cross_to_char_height_ratio": round(
                cross_span / char_height,
                6,
            ),
        },
        "authority_digest": authority["authority_digest"],
        "window_source_digests": list(
            authority["group_member_source_digests"]
        ),
        "axis_angle_deg": authority["axis_angle_deg"],
        "relation": "paired_enclosure",
        "group_digest": proof["proof_digest"],
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _normalize_proof(
    value: Any,
    *,
    phrase_id: str,
    authority_digest: str,
    anchored_lane_partition_digest: str,
    expected_axis_angle: float,
    authority: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "phrase_id",
        "authority_digest",
        "anchored_lane_partition_digest",
        "axis_angle_deg",
        "source_primitive_id",
        "main_low_primitive_ids",
        "main_high_primitive_ids",
        "interior_primitive_ids",
        "main_low_vertices",
        "main_high_vertices",
        "segment_count",
        "mirror_center_main",
        "max_connection_error_pt",
        "max_cross_alignment_error_pt",
        "max_mirror_error_pt",
        "mirror_rms_error_pt",
        "main_low_bow_depth_pt",
        "main_high_bow_depth_pt",
        "main_low_path_length_pt",
        "main_high_path_length_pt",
        "cross_iou",
        "span_similarity",
        "depth_similarity",
        "path_similarity",
        "segment_similarity",
        "cross_center_delta_ratio",
        "mirror_rms_depth_ratio",
        "mirror_max_depth_ratio",
        "main_low_inner_gap_pt",
        "main_high_inner_gap_pt",
        "cross_margin_low_pt",
        "cross_margin_high_pt",
        "interior_main_min",
        "interior_main_max",
        "interior_cross_min",
        "interior_cross_max",
        "enclosure_to_body_cross_ratio",
        "body_center_offset_ratio",
        "consumer_allowed",
        "proof_digest",
    }:
        return None
    low_ids = _canonical_ids(value.get("main_low_primitive_ids"))
    high_ids = _canonical_ids(value.get("main_high_primitive_ids"))
    interior_ids = _canonical_ids(value.get("interior_primitive_ids"))
    low_vertices = _canonical_vertices(value.get("main_low_vertices"))
    high_vertices = _canonical_vertices(value.get("main_high_vertices"))
    proof_axis_angle = _canonical_axis_angle(
        value.get("axis_angle_deg")
    )
    numeric_keys = [
        key for key in value
        if key.endswith("_pt")
        or key in {
            "mirror_center_main",
            "cross_iou",
            "span_similarity",
            "depth_similarity",
            "path_similarity",
            "segment_similarity",
            "cross_center_delta_ratio",
            "mirror_rms_depth_ratio",
            "mirror_max_depth_ratio",
            "interior_main_min",
            "interior_main_max",
            "interior_cross_min",
            "interior_cross_max",
            "enclosure_to_body_cross_ratio",
            "body_center_offset_ratio",
        }
    ]
    if not (
        value.get("schema_version") == PROOF_SCHEMA_VERSION
        and value.get("phrase_id") == phrase_id
        and value.get("authority_digest") == authority_digest
        and value.get("anchored_lane_partition_digest")
        == anchored_lane_partition_digest
        and proof_axis_angle == expected_axis_angle
        and type(value.get("source_primitive_id")) is str
        and bool(value["source_primitive_id"])
        and low_ids is not None
        and high_ids is not None
        and interior_ids is not None
        and low_vertices is not None
        and high_vertices is not None
        and len(low_vertices) == len(high_vertices)
        and type(value.get("segment_count")) is int
        and value["segment_count"] >= MIN_CHAIN_SEGMENTS
        and len(low_vertices) == value["segment_count"] + 1
        and len(low_ids) == value["segment_count"]
        and len(high_ids) == value["segment_count"]
        and any(
            primitive_id in interior_ids
            for primitive_id in [value["source_primitive_id"]]
        )
        and set(low_ids).isdisjoint(high_ids)
        and set(low_ids + high_ids).isdisjoint(interior_ids)
        and all(
            type(value.get(key)) in {int, float}
            and math.isfinite(float(value[key]))
            for key in numeric_keys
        )
        and value.get("consumer_allowed") is False
        and _valid_sha256(value.get("proof_digest"))
    ):
        return None
    replayed = _replay_paired_enclosure_metrics(
        low_vertices,
        high_vertices,
        interior_main_min=float(value["interior_main_min"]),
        interior_main_max=float(value["interior_main_max"]),
        interior_cross_min=float(value["interior_cross_min"]),
        interior_cross_max=float(value["interior_cross_max"]),
    )
    if (
        replayed is None
        or float(value["max_connection_error_pt"])
        < 0.0
        or float(value["max_connection_error_pt"])
        > COORDINATE_JOIN_TOLERANCE_PT
        or any(
            abs(float(value[key]) - replayed[key]) > 0.00002
            for key in replayed
            if key in value
        )
        or not _replayed_proof_thresholds_pass(
            replayed,
            max_connection_error_pt=float(
                value["max_connection_error_pt"]
            ),
        )
        or (
            authority is not None
            and not (
                value["source_primitive_id"]
                == authority["source_primitive_id"]
                and replayed["main_low_max"]
                < float(authority["dot_main_center"])
                < replayed["main_high_min"]
            )
        )
    ):
        return None
    canonical = {
        key: value[key]
        for key in value
        if key != "proof_digest"
    }
    canonical.update({
        "axis_angle_deg": proof_axis_angle,
        "main_low_primitive_ids": low_ids,
        "main_high_primitive_ids": high_ids,
        "interior_primitive_ids": interior_ids,
        "main_low_vertices": low_vertices,
        "main_high_vertices": high_vertices,
    })
    for key in numeric_keys:
        canonical[key] = round(float(value[key]), 6)
    if value["proof_digest"] != _digest(canonical):
        return None
    return {
        **canonical,
        "proof_digest": value["proof_digest"],
    }


def _replay_paired_enclosure_metrics(
    low_vertices: list[list[float]],
    high_vertices: list[list[float]],
    *,
    interior_main_min: float,
    interior_main_max: float,
    interior_cross_min: float,
    interior_cross_max: float,
) -> dict[str, float] | None:
    if not (
        interior_main_min <= interior_main_max
        and interior_cross_min < interior_cross_max
    ):
        return None
    low = _replay_open_bow(low_vertices, outward_sign=-1)
    high = _replay_open_bow(high_vertices, outward_sign=1)
    if (
        low is None
        or high is None
        or low["main_max"] >= high["main_min"]
        or len(low_vertices) != len(high_vertices)
    ):
        return None
    cross_errors = [
        abs(left[1] - right[1])
        for left, right in zip(
            low_vertices,
            high_vertices,
            strict=True,
        )
    ]
    cross_min = max(low["cross_min"], high["cross_min"])
    cross_max = min(low["cross_max"], high["cross_max"])
    cross_intersection = max(0.0, cross_max - cross_min)
    cross_union = (
        max(low["cross_max"], high["cross_max"])
        - min(low["cross_min"], high["cross_min"])
    )
    cross_iou = cross_intersection / max(cross_union, 1e-12)
    cross_center_delta_ratio = abs(
        low["cross_center"] - high["cross_center"]
    ) / max(low["cross_span"], high["cross_span"], 1e-12)
    midpoint_mains = [
        (left[0] + right[0]) / 2.0
        for left, right in zip(
            low_vertices,
            high_vertices,
            strict=True,
        )
    ]
    mirror_center = sum(midpoint_mains) / len(midpoint_mains)
    mirror_errors = [
        abs(value - mirror_center) for value in midpoint_mains
    ]
    max_mirror_error = max(mirror_errors)
    mirror_rms = math.sqrt(
        sum(error * error for error in mirror_errors)
        / len(mirror_errors)
    )
    mean_depth = (low["bow_depth"] + high["bow_depth"]) / 2.0
    low_lengths = _segment_lengths(low_vertices)
    high_lengths = _segment_lengths(high_vertices)
    low_inner_gap = interior_main_min - low["main_max"]
    high_inner_gap = high["main_min"] - interior_main_max
    cross_margin_low = interior_cross_min - cross_min
    cross_margin_high = cross_max - interior_cross_max
    body_cross_span = interior_cross_max - interior_cross_min
    enclosure_cross_span = (
        low["cross_span"] + high["cross_span"]
    ) / 2.0
    body_cross_center = (
        interior_cross_min + interior_cross_max
    ) / 2.0
    enclosure_cross_center = (
        low["cross_center"] + high["cross_center"]
    ) / 2.0
    values = {
        "mirror_center_main": mirror_center,
        "max_cross_alignment_error_pt": max(cross_errors),
        "max_mirror_error_pt": max_mirror_error,
        "mirror_rms_error_pt": mirror_rms,
        "main_low_bow_depth_pt": low["bow_depth"],
        "main_high_bow_depth_pt": high["bow_depth"],
        "main_low_path_length_pt": low["path_length"],
        "main_high_path_length_pt": high["path_length"],
        "cross_iou": cross_iou,
        "span_similarity": _similarity(
            low["cross_span"],
            high["cross_span"],
        ),
        "depth_similarity": _similarity(
            low["bow_depth"],
            high["bow_depth"],
        ),
        "path_similarity": _similarity(
            low["path_length"],
            high["path_length"],
        ),
        "segment_similarity": min(
            _similarity(left, right)
            for left, right in zip(
                low_lengths,
                high_lengths,
                strict=True,
            )
        ),
        "cross_center_delta_ratio": cross_center_delta_ratio,
        "mirror_rms_depth_ratio": (
            mirror_rms / max(mean_depth, 1e-12)
        ),
        "mirror_max_depth_ratio": (
            max_mirror_error / max(mean_depth, 1e-12)
        ),
        "main_low_inner_gap_pt": low_inner_gap,
        "main_high_inner_gap_pt": high_inner_gap,
        "cross_margin_low_pt": cross_margin_low,
        "cross_margin_high_pt": cross_margin_high,
        "interior_main_min": interior_main_min,
        "interior_main_max": interior_main_max,
        "interior_cross_min": interior_cross_min,
        "interior_cross_max": interior_cross_max,
        "enclosure_to_body_cross_ratio": (
            enclosure_cross_span / max(body_cross_span, 1e-12)
        ),
        "body_center_offset_ratio": abs(
            body_cross_center - enclosure_cross_center
        ) / max(enclosure_cross_span, 1e-12),
        "main_low_depth_per_cross": (
            low["bow_depth"] / low["cross_span"]
        ),
        "main_high_depth_per_cross": (
            high["bow_depth"] / high["cross_span"]
        ),
        "main_low_path_per_cross": (
            low["path_length"] / low["cross_span"]
        ),
        "main_high_path_per_cross": (
            high["path_length"] / high["cross_span"]
        ),
        "main_low_max": low["main_max"],
        "main_high_min": high["main_min"],
    }
    return {
        key: round(float(number), 6)
        for key, number in values.items()
    }


def _replay_open_bow(
    vertices: list[list[float]],
    *,
    outward_sign: int,
) -> dict[str, float] | None:
    cross_steps = [
        vertices[index + 1][1] - vertices[index][1]
        for index in range(len(vertices) - 1)
    ]
    if any(
        step >= -COORDINATE_JOIN_TOLERANCE_PT
        for step in cross_steps
    ):
        return None
    start_main, start_cross = vertices[0]
    end_main, end_cross = vertices[-1]
    cross_span = start_cross - end_cross
    if cross_span <= 0.0:
        return None
    deviations = []
    for main, cross in vertices[1:-1]:
        fraction = (
            (cross - start_cross) / (end_cross - start_cross)
        )
        chord_main = start_main + fraction * (
            end_main - start_main
        )
        deviations.append(main - chord_main)
    if not deviations:
        return None
    peak = max(deviations, key=abs)
    if (
        outward_sign * peak <= 0.0
        or abs(peak) <= COORDINATE_JOIN_TOLERANCE_PT * 5.0
        or any(
            outward_sign * deviation
            < -COORDINATE_JOIN_TOLERANCE_PT
            for deviation in deviations
        )
        or abs(start_main - end_main)
        > max(
            COORDINATE_JOIN_TOLERANCE_PT,
            abs(peak) * 0.10,
        )
    ):
        return None
    main_values = [point[0] for point in vertices]
    cross_values = [point[1] for point in vertices]
    return {
        "bow_depth": abs(peak),
        "path_length": sum(_segment_lengths(vertices)),
        "main_min": min(main_values),
        "main_max": max(main_values),
        "cross_min": min(cross_values),
        "cross_max": max(cross_values),
        "cross_center": (
            min(cross_values) + max(cross_values)
        ) / 2.0,
        "cross_span": max(cross_values) - min(cross_values),
    }


def _replayed_proof_thresholds_pass(
    values: dict[str, float],
    *,
    max_connection_error_pt: float,
) -> bool:
    return bool(
        max_connection_error_pt <= COORDINATE_JOIN_TOLERANCE_PT
        and values["max_cross_alignment_error_pt"]
        <= COORDINATE_JOIN_TOLERANCE_PT
        and min(
            values["span_similarity"],
            values["depth_similarity"],
            values["path_similarity"],
            values["segment_similarity"],
        )
        >= MIN_PAIR_SIMILARITY
        and values["cross_iou"] >= MIN_CROSS_IOU
        and values["cross_center_delta_ratio"]
        <= MAX_CROSS_CENTER_DELTA_RATIO
        and values["mirror_rms_depth_ratio"]
        <= MAX_MIRROR_RMS_DEPTH_RATIO
        and values["mirror_max_depth_ratio"]
        <= MAX_MIRROR_ERROR_DEPTH_RATIO
        and all(
            MIN_DEPTH_TO_CROSS_SPAN
            <= values[key]
            <= MAX_DEPTH_TO_CROSS_SPAN
            for key in (
                "main_low_depth_per_cross",
                "main_high_depth_per_cross",
            )
        )
        and all(
            MIN_PATH_TO_CROSS_SPAN
            <= values[key]
            <= MAX_PATH_TO_CROSS_SPAN
            for key in (
                "main_low_path_per_cross",
                "main_high_path_per_cross",
            )
        )
        and values["main_low_inner_gap_pt"]
        > COORDINATE_JOIN_TOLERANCE_PT
        and values["main_high_inner_gap_pt"]
        > COORDINATE_JOIN_TOLERANCE_PT
        and values["cross_margin_low_pt"]
        >= -COORDINATE_JOIN_TOLERANCE_PT
        and values["cross_margin_high_pt"]
        >= -COORDINATE_JOIN_TOLERANCE_PT
        and MIN_ENCLOSURE_TO_BODY_CROSS_RATIO
        <= values["enclosure_to_body_cross_ratio"]
        <= MAX_ENCLOSURE_TO_BODY_CROSS_RATIO
        and values["body_center_offset_ratio"]
        <= MAX_BODY_CENTER_OFFSET_RATIO
    )


def _normalize_exclusions(
    value: Any,
    *,
    authority_digest: str,
    axis_angle: float,
    proofs: list[dict[str, Any]],
    authority: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or len(value) != len(proofs):
        return None
    proof_by_digest = {
        proof["proof_digest"]: proof for proof in proofs
    }
    rows: list[dict[str, Any]] = []
    expected_keys = {
        "schema_version",
        "primitive_ids",
        "stage",
        "reason",
        "bbox",
        "canonical_bbox",
        "measurements",
        "authority_digest",
        "window_source_digests",
        "axis_angle_deg",
        "relation",
        "group_digest",
        "consumer_allowed",
    }
    for raw in value:
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_keys
            or raw.get("schema_version") != EXCLUSION_SCHEMA_VERSION
            or raw.get("stage") != "anchored_character_filter"
            or raw.get("reason") != "paired_curved_enclosure"
            or raw.get("relation") != "paired_enclosure"
            or raw.get("authority_digest") != authority_digest
            or raw.get("axis_angle_deg") != axis_angle
            or raw.get("consumer_allowed") is not False
            or raw.get("group_digest") not in proof_by_digest
        ):
            return None
        proof = proof_by_digest[raw["group_digest"]]
        ids = _canonical_ids(raw.get("primitive_ids"))
        expected_ids = sorted([
            *proof["main_low_primitive_ids"],
            *proof["main_high_primitive_ids"],
        ])
        bbox = _canonical_bbox(raw.get("bbox"))
        canonical_bbox = _canonical_main_cross_bbox(
            raw.get("canonical_bbox")
        )
        measurements = _canonical_exclusion_measurements(
            raw.get("measurements")
        )
        source_digests = _canonical_ids(
            raw.get("window_source_digests")
        )
        if (
            ids != expected_ids
            or bbox is None
            or canonical_bbox is None
            or measurements is None
            or source_digests is None
            or not source_digests
            or any(
                not _valid_sha256(digest)
                for digest in source_digests
            )
            or (
                authority is not None
                and raw != _exclusion_from_proof(
                    proof,
                    authority=authority,
                )
            )
        ):
            return None
        rows.append({
            **raw,
            "primitive_ids": ids,
            "bbox": bbox,
            "canonical_bbox": canonical_bbox,
            "measurements": measurements,
            "window_source_digests": source_digests,
            "axis_angle_deg": axis_angle,
        })
    rows.sort(key=lambda row: row["primitive_ids"])
    return rows if rows == value else None


def _canonical_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(
        type(number) not in {int, float}
        or not math.isfinite(float(number))
        for number in value
    ):
        return None
    canonical = [round(float(number), 6) for number in value]
    if canonical[2] < canonical[0] or canonical[3] < canonical[1]:
        return None
    return canonical


def _canonical_main_cross_bbox(
    value: Any,
) -> dict[str, float] | None:
    keys = {
        "main_min",
        "cross_min",
        "main_max",
        "cross_max",
    }
    if not isinstance(value, dict) or set(value) != keys:
        return None
    if any(
        type(value[key]) not in {int, float}
        or not math.isfinite(float(value[key]))
        for key in keys
    ):
        return None
    canonical = {
        key: round(float(value[key]), 6) for key in keys
    }
    if (
        canonical["main_max"] < canonical["main_min"]
        or canonical["cross_max"] < canonical["cross_min"]
    ):
        return None
    return canonical


def _canonical_exclusion_measurements(
    value: Any,
) -> dict[str, float] | None:
    keys = {
        "main_span_pt",
        "cross_span_pt",
        "char_width_pt",
        "char_height_pt",
        "main_to_char_width_ratio",
        "cross_to_char_height_ratio",
    }
    if not isinstance(value, dict) or set(value) != keys:
        return None
    if any(
        type(value[key]) not in {int, float}
        or not math.isfinite(float(value[key]))
        for key in keys
    ):
        return None
    canonical = {
        key: round(float(value[key]), 6) for key in keys
    }
    if (
        canonical["main_span_pt"] < 0.0
        or canonical["cross_span_pt"] < 0.0
        or canonical["char_width_pt"] <= 0.0
        or canonical["char_height_pt"] <= 0.0
        or canonical["main_to_char_width_ratio"] != round(
            canonical["main_span_pt"]
            / canonical["char_width_pt"],
            6,
        )
        or canonical["cross_to_char_height_ratio"] != round(
            canonical["cross_span_pt"]
            / canonical["char_height_pt"],
            6,
        )
    ):
        return None
    return canonical


def _failure(
    *,
    reason: str,
    authority: dict[str, Any] | None,
    lane: dict[str, Any] | None,
    input_ids: list[str] | None = None,
    candidate_count: int = 0,
    protected_ids: list[str] | None = None,
    cut_ids: list[str] | None = None,
) -> dict[str, Any]:
    ids = sorted(input_ids or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail_closed",
        "reason": reason,
        "phrase_id": (
            str(authority.get("phrase_id") or "")
            if isinstance(authority, dict)
            else ""
        ),
        "authority_digest": (
            authority.get("authority_digest")
            if isinstance(authority, dict)
            else None
        ),
        "anchored_lane_partition_digest": (
            lane.get("partition_digest")
            if isinstance(lane, dict)
            else None
        ),
        "input_primitive_ids": ids,
        "input_primitive_count": len(ids),
        "admitted_primitive_ids": [],
        "admitted_primitive_count": 0,
        "excluded_primitive_ids": [],
        "excluded_primitive_count": 0,
        "protected_primitive_ids": sorted(protected_ids or []),
        "boundary_cut_primitive_ids": sorted(cut_ids or []),
        "paired_enclosure_candidate_count": int(candidate_count),
        "paired_enclosure_proofs": [],
        "exclusions": [],
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _rule_contract() -> dict[str, Any]:
    return {
        "coordinate_join_tolerance_pt": (
            COORDINATE_JOIN_TOLERANCE_PT
        ),
        "minimum_chain_segments": MIN_CHAIN_SEGMENTS,
        "minimum_pair_similarity": MIN_PAIR_SIMILARITY,
        "minimum_cross_iou": MIN_CROSS_IOU,
        "maximum_cross_center_delta_ratio": (
            MAX_CROSS_CENTER_DELTA_RATIO
        ),
        "maximum_mirror_rms_depth_ratio": (
            MAX_MIRROR_RMS_DEPTH_RATIO
        ),
        "maximum_mirror_error_depth_ratio": (
            MAX_MIRROR_ERROR_DEPTH_RATIO
        ),
        "minimum_depth_per_cross_span": (
            MIN_DEPTH_TO_CROSS_SPAN
        ),
        "maximum_depth_per_cross_span": (
            MAX_DEPTH_TO_CROSS_SPAN
        ),
        "minimum_path_per_cross_span": MIN_PATH_TO_CROSS_SPAN,
        "maximum_path_per_cross_span": MAX_PATH_TO_CROSS_SPAN,
        "minimum_enclosure_to_body_cross_ratio": (
            MIN_ENCLOSURE_TO_BODY_CROSS_RATIO
        ),
        "maximum_enclosure_to_body_cross_ratio": (
            MAX_ENCLOSURE_TO_BODY_CROSS_RATIO
        ),
        "maximum_body_center_offset_ratio": (
            MAX_BODY_CENTER_OFFSET_RATIO
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _segments_touch(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> bool:
    return any(
        _distance(left_point, right_point)
        <= COORDINATE_JOIN_TOLERANCE_PT
        for left_point in left
        for right_point in right
    )


def _point_to_segment_endpoint_distance(
    point: tuple[float, float],
    segment: list[tuple[float, float]],
) -> float:
    return min(_distance(point, endpoint) for endpoint in segment)


def _distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _segment_lengths(
    vertices: list[tuple[float, float]]
    | list[list[float]],
) -> list[float]:
    return [
        _distance(tuple(left), tuple(right))
        for left, right in zip(vertices, vertices[1:])
    ]


def _similarity(left: float, right: float) -> float:
    return min(left, right) / max(left, right, 1e-12)


def _adjacent_item_runs(
    low: dict[str, Any],
    high: dict[str, Any],
) -> bool:
    return bool(
        low["item_index_max"] + 1 == high["item_index_min"]
        or high["item_index_max"] + 1 == low["item_index_min"]
    )


def _rounded_vertices(
    value: list[tuple[float, float]],
) -> list[list[float]]:
    return [
        [round(main, 6), round(cross, 6)]
        for main, cross in value
    ]


def _canonical_vertices(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    rows: list[list[float]] = []
    for point in value:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                type(number) not in {int, float}
                or not math.isfinite(float(number))
                for number in point
            )
        ):
            return None
        rows.append([
            round(float(point[0]), 6),
            round(float(point[1]), 6),
        ])
    return rows


def _canonical_optional_ids(
    value: Any,
    *,
    default: list[str],
) -> list[str] | None:
    return _canonical_ids(default if value is None else value)


def _canonical_ids(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
    ):
        return None
    canonical = sorted(set(value))
    return canonical if canonical == value else None


def _canonical_axis_angle(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number % 180.0, 6)


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
