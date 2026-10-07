"""Exact primitive partitions for one anchored pitch phrase."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any


SCHEMA_VERSION = "vector_pitch_primitive_partition_v1"
DIGEST_SCHEMA_VERSION = "vector_pitch_primitive_partition_digest_v1"
SCHEMA_VERSION_V2 = "vector_pitch_primitive_partition_v2"
DIGEST_SCHEMA_VERSION_V2 = "vector_pitch_primitive_partition_digest_v2"
PRODUCER = "vector_pitch_phrase_reader_v1"
EXCLUSION_SCHEMA_VERSION = "vector_pitch_primitive_exclusion_v1"
CONSUMER_ALLOWED = False
EXCLUSION_REASON_CONTRACTS = {
    "outside_anchored_local_lane": (
        "anchored_lane",
        "outside_selected_cross_component",
    ),
    "cross_span_exceeds_anchored_char_height": (
        "anchored_character_filter",
        "exceeds_cross_envelope",
    ),
    "main_span_exceeds_anchored_char_width": (
        "anchored_character_filter",
        "exceeds_main_envelope",
    ),
    "long_thin_construction_ink": (
        "anchored_character_filter",
        "long_thin_axis_aligned",
    ),
    "paired_curved_enclosure": (
        "anchored_character_filter",
        "paired_enclosure",
    ),
    "outside_contiguous_numeric_lattice": (
        "anchored_character_filter",
        "outside_numeric_lattice",
    ),
    "associated_capsule_boundary": (
        "anchored_capsule_filter",
        "associated_capsule_outer_contour",
    ),
}


def build_pitch_primitive_partition(
    *,
    phrase_id: str,
    input_primitive_ids: Any,
    admitted_primitive_ids: Any,
    spans: Any,
    exclusions: Any,
    drops: Any,
    window_authority_digest: str,
    anchored_lane_partition_digest: str,
) -> dict[str, Any] | None:
    input_ids = _canonical_id_list(input_primitive_ids)
    admitted_ids = _canonical_id_list(admitted_primitive_ids)
    span_assignments = _assignment_rows(spans, kind="span")
    drop_assignments = _assignment_rows(drops, kind="drop")
    canonical_exclusions = _canonical_exclusions(exclusions)
    if (
        type(phrase_id) is not str
        or not phrase_id
        or input_ids is None
        or not input_ids
        or admitted_ids is None
        or span_assignments is None
        or drop_assignments is None
        or canonical_exclusions is None
        or not _valid_sha256(window_authority_digest)
        or not _valid_sha256(anchored_lane_partition_digest)
    ):
        return None
    span_ids = [
        primitive_id
        for row in span_assignments
        for primitive_id in row["primitive_ids"]
    ]
    drop_ids = [
        primitive_id
        for row in drop_assignments
        for primitive_id in row["primitive_ids"]
    ]
    exclusion_ids = [
        primitive_id
        for row in canonical_exclusions
        for primitive_id in row["primitive_ids"]
    ]
    assigned_ids = [*span_ids, *exclusion_ids, *drop_ids]
    if (
        any(count != 1 for count in Counter(assigned_ids).values())
        or set(assigned_ids) != set(input_ids)
        or set(span_ids + drop_ids) != set(admitted_ids)
        or set(exclusion_ids) != set(input_ids) - set(admitted_ids)
        or any(
            row["authority_digest"] != window_authority_digest
            for row in canonical_exclusions
        )
    ):
        return None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "phrase_id": phrase_id,
        "input_primitive_ids": input_ids,
        "admitted_primitive_ids": admitted_ids,
        "span_assignments": span_assignments,
        "exclusions": canonical_exclusions,
        "drop_assignments": drop_assignments,
        "input_primitive_count": len(input_ids),
        "admitted_primitive_count": len(admitted_ids),
        "span_primitive_count": len(span_ids),
        "exclusion_primitive_count": len(exclusion_ids),
        "drop_primitive_count": len(drop_ids),
        "window_authority_digest": window_authority_digest,
        "anchored_lane_partition_digest": (
            anchored_lane_partition_digest
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "partition_digest_schema_version": DIGEST_SCHEMA_VERSION,
        "partition_digest": _digest({
            "schema_version": DIGEST_SCHEMA_VERSION,
            "partition": payload,
        }),
    }


def normalize_pitch_primitive_partition(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "producer",
        "phrase_id",
        "input_primitive_ids",
        "admitted_primitive_ids",
        "span_assignments",
        "exclusions",
        "drop_assignments",
        "input_primitive_count",
        "admitted_primitive_count",
        "span_primitive_count",
        "exclusion_primitive_count",
        "drop_primitive_count",
        "window_authority_digest",
        "anchored_lane_partition_digest",
        "consumer_allowed",
        "partition_digest_schema_version",
        "partition_digest",
    }:
        return None
    rebuilt = build_pitch_primitive_partition(
        phrase_id=value.get("phrase_id"),
        input_primitive_ids=value.get("input_primitive_ids"),
        admitted_primitive_ids=value.get("admitted_primitive_ids"),
        spans=value.get("span_assignments"),
        exclusions=value.get("exclusions"),
        drops=value.get("drop_assignments"),
        window_authority_digest=value.get("window_authority_digest"),
        anchored_lane_partition_digest=value.get(
            "anchored_lane_partition_digest"
        ),
    )
    if rebuilt is None or rebuilt != value:
        return None
    return rebuilt


def build_pitch_primitive_partition_v2(
    *,
    phrase_id: str,
    input_primitive_ids: Any,
    admitted_primitive_ids: Any,
    spans: Any,
    semantic_occupation: Any,
    exclusions: Any,
    drops: Any,
    window_authority_digest: str,
    anchored_lane_partition_digest: str,
) -> dict[str, Any] | None:
    """Build the top-level four-state primitive partition.

    Child lattice ownership remains on the v1 span/drop contract.  This v2
    partition is the first layer allowed to account for strict semantic ink
    without disguising that ink as a recognized span, a drop, or an
    exclusion.
    """
    binding = _canonical_semantic_occupation_binding(semantic_occupation)
    input_ids = _canonical_id_list(input_primitive_ids)
    admitted_ids = _canonical_id_list(admitted_primitive_ids)
    span_assignments = _assignment_rows(spans, kind="span")
    drop_assignments = _assignment_rows(drops, kind="drop")
    canonical_exclusions = _canonical_exclusions(exclusions)
    if (
        type(phrase_id) is not str
        or not phrase_id
        or input_ids is None
        or not input_ids
        or admitted_ids is None
        or span_assignments is None
        or drop_assignments is None
        or canonical_exclusions is None
        or binding is None
        or binding.get("phrase_id") != phrase_id
        or binding.get("window_authority_digest")
        != window_authority_digest
        or not _valid_sha256(window_authority_digest)
        or not _valid_sha256(anchored_lane_partition_digest)
    ):
        return None
    occupied_ids = _canonical_id_list(
        binding.get("occupied_primitive_ids")
    )
    if (
        occupied_ids is None
        or len(occupied_ids) != 3
        or binding.get("occupied_primitive_count") != len(occupied_ids)
        or not _valid_sha256(binding.get("occupation_digest"))
    ):
        return None
    span_ids = [
        primitive_id
        for row in span_assignments
        for primitive_id in row["primitive_ids"]
    ]
    drop_ids = [
        primitive_id
        for row in drop_assignments
        for primitive_id in row["primitive_ids"]
    ]
    exclusion_ids = [
        primitive_id
        for row in canonical_exclusions
        for primitive_id in row["primitive_ids"]
    ]
    assigned_ids = [
        *span_ids,
        *occupied_ids,
        *exclusion_ids,
        *drop_ids,
    ]
    if (
        any(count != 1 for count in Counter(assigned_ids).values())
        or set(assigned_ids) != set(input_ids)
        or set([*span_ids, *occupied_ids, *drop_ids])
        != set(admitted_ids)
        or set(exclusion_ids) != set(input_ids) - set(admitted_ids)
        or any(
            row["authority_digest"] != window_authority_digest
            for row in canonical_exclusions
        )
    ):
        return None
    payload = {
        "schema_version": SCHEMA_VERSION_V2,
        "producer": PRODUCER,
        "phrase_id": phrase_id,
        "input_primitive_ids": input_ids,
        "admitted_primitive_ids": admitted_ids,
        "span_assignments": span_assignments,
        "semantic_occupation": binding,
        "occupied_primitive_ids": occupied_ids,
        "exclusions": canonical_exclusions,
        "drop_assignments": drop_assignments,
        "input_primitive_count": len(input_ids),
        "admitted_primitive_count": len(admitted_ids),
        "span_primitive_count": len(span_ids),
        "semantic_occupation_primitive_count": len(occupied_ids),
        "exclusion_primitive_count": len(exclusion_ids),
        "drop_primitive_count": len(drop_ids),
        "semantic_occupation_digest": binding["occupation_digest"],
        "window_authority_digest": window_authority_digest,
        "anchored_lane_partition_digest": (
            anchored_lane_partition_digest
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "partition_digest_schema_version": DIGEST_SCHEMA_VERSION_V2,
        "partition_digest": _digest({
            "schema_version": DIGEST_SCHEMA_VERSION_V2,
            "partition": payload,
        }),
    }


def normalize_pitch_primitive_partition_v2(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "producer",
        "phrase_id",
        "input_primitive_ids",
        "admitted_primitive_ids",
        "span_assignments",
        "semantic_occupation",
        "occupied_primitive_ids",
        "exclusions",
        "drop_assignments",
        "input_primitive_count",
        "admitted_primitive_count",
        "span_primitive_count",
        "semantic_occupation_primitive_count",
        "exclusion_primitive_count",
        "drop_primitive_count",
        "semantic_occupation_digest",
        "window_authority_digest",
        "anchored_lane_partition_digest",
        "consumer_allowed",
        "partition_digest_schema_version",
        "partition_digest",
    }:
        return None
    rebuilt = build_pitch_primitive_partition_v2(
        phrase_id=value.get("phrase_id"),
        input_primitive_ids=value.get("input_primitive_ids"),
        admitted_primitive_ids=value.get("admitted_primitive_ids"),
        spans=value.get("span_assignments"),
        semantic_occupation=value.get("semantic_occupation"),
        exclusions=value.get("exclusions"),
        drops=value.get("drop_assignments"),
        window_authority_digest=value.get("window_authority_digest"),
        anchored_lane_partition_digest=value.get(
            "anchored_lane_partition_digest"
        ),
    )
    if rebuilt is None or rebuilt != value:
        return None
    return rebuilt


def _assignment_rows(value: Any, *, kind: str) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(value):
        if not isinstance(raw_row, dict):
            return None
        summary_index_key = f"{kind}_index"
        summary_keys = (
            {
                summary_index_key,
                "primitive_ids",
                "char",
                "decision",
            }
            if kind == "span"
            else {
                summary_index_key,
                "primitive_ids",
                "reason",
            }
        )
        if set(raw_row) == summary_keys:
            if (
                type(raw_row.get(summary_index_key)) is not int
                or raw_row[summary_index_key] != index
            ):
                return None
            primitive_ids = _canonical_id_list(
                raw_row.get("primitive_ids")
            )
            if (
                primitive_ids is None
                or (not primitive_ids and kind != "span")
            ):
                return None
            summary = {
                summary_index_key: index,
                "primitive_ids": primitive_ids,
            }
            if kind == "span":
                char = raw_row.get("char")
                decision = raw_row.get("decision")
                if (
                    type(char) is not str
                    or len(char) != 1
                    or type(decision) is not str
                    or not decision
                ):
                    return None
                summary.update({
                    "char": char,
                    "decision": decision,
                })
            else:
                reason = raw_row.get("reason")
                if type(reason) is not str or not reason:
                    return None
                summary["reason"] = reason
            rows.append(summary)
            continue
        primitive_ids = _canonical_id_list(
            raw_row.get("primitive_ids")
        )
        if primitive_ids is None:
            return None
        declared_count = raw_row.get(
            "slot_item_count" if kind == "span" else "item_count"
        )
        if type(declared_count) is not int or declared_count != len(
            primitive_ids
        ):
            return None
        if not primitive_ids and not (
            kind == "span"
            and raw_row.get("decision")
            == "stacked_tolerance_row_separator"
        ):
            return None
        summary = {
            f"{kind}_index": index,
            "primitive_ids": primitive_ids,
        }
        if kind == "span":
            char = raw_row.get("char")
            decision = raw_row.get("decision")
            if (
                type(char) is not str
                or len(char) != 1
                or type(decision) is not str
                or not decision
            ):
                return None
            summary.update({
                "char": char,
                "decision": decision,
            })
        else:
            reason = raw_row.get("reason")
            if type(reason) is not str or not reason:
                return None
            summary["reason"] = reason
        rows.append(summary)
    return rows


def _canonical_exclusions(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    rows: list[dict[str, Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, dict) or set(raw_row) != {
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
        }:
            return None
        primitive_ids = _canonical_id_list(
            raw_row.get("primitive_ids")
        )
        stage = raw_row.get("stage")
        reason = raw_row.get("reason")
        bbox = _canonical_bbox(raw_row.get("bbox"))
        canonical_bbox = _canonical_axis_bbox(
            raw_row.get("canonical_bbox")
        )
        measurements = _canonical_measurements(
            raw_row.get("measurements")
        )
        authority_digest = raw_row.get("authority_digest")
        source_digests = _canonical_id_list(
            raw_row.get("window_source_digests")
        )
        axis_angle = _canonical_axis_angle(
            raw_row.get("axis_angle_deg")
        )
        relation = raw_row.get("relation")
        group_digest = raw_row.get("group_digest")
        expected_contract = EXCLUSION_REASON_CONTRACTS.get(reason)
        if (
            raw_row.get("schema_version") != EXCLUSION_SCHEMA_VERSION
            or primitive_ids is None
            or not primitive_ids
            or expected_contract is None
            or (stage, relation) != expected_contract
            or type(reason) is not str
            or not reason
            or bbox is None
            or canonical_bbox is None
            or measurements is None
            or not _valid_sha256(authority_digest)
            or source_digests is None
            or not source_digests
            or any(not _valid_sha256(digest) for digest in source_digests)
            or axis_angle is None
            or not _valid_exclusion_group_digest(
                reason=reason,
                group_digest=group_digest,
            )
            or raw_row.get("consumer_allowed") is not False
        ):
            return None
        rows.append({
            "schema_version": EXCLUSION_SCHEMA_VERSION,
            "primitive_ids": primitive_ids,
            "stage": stage,
            "reason": reason,
            "bbox": bbox,
            "canonical_bbox": canonical_bbox,
            "measurements": measurements,
            "authority_digest": authority_digest,
            "window_source_digests": source_digests,
            "axis_angle_deg": axis_angle,
            "relation": relation,
            "group_digest": group_digest,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    rows.sort(key=lambda row: (
        row["primitive_ids"],
        row["reason"],
        row["stage"],
    ))
    flattened_ids = [
        primitive_id
        for row in rows
        for primitive_id in row["primitive_ids"]
    ]
    if len(set(flattened_ids)) != len(flattened_ids):
        return None
    return rows


def _valid_exclusion_group_digest(
    *,
    reason: Any,
    group_digest: Any,
) -> bool:
    if reason in {
        "paired_curved_enclosure",
        "associated_capsule_boundary",
    }:
        return _valid_sha256(group_digest)
    if reason == "outside_anchored_local_lane":
        return group_digest is None or _valid_sha256(group_digest)
    return group_digest is None


def _canonical_semantic_occupation_binding(
    value: Any,
) -> dict[str, Any] | None:
    try:
        from vector_pitch_semantic_occupation import (
            semantic_occupation_binding,
        )
    except ImportError:
        return None
    binding = semantic_occupation_binding(value)
    return binding if isinstance(binding, dict) else None


def _canonical_id_list(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
    ):
        return None
    canonical = sorted(set(value))
    return canonical if value == canonical else None


def _canonical_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(type(number) not in {int, float} for number in value):
        return None
    numbers = [float(number) for number in value]
    if (
        any(not math.isfinite(number) for number in numbers)
        or numbers[2] < numbers[0]
        or numbers[3] < numbers[1]
    ):
        return None
    return [round(number, 6) for number in numbers]


def _canonical_axis_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != {
        "main_min",
        "cross_min",
        "main_max",
        "cross_max",
    }:
        return None
    numbers = {
        key: value.get(key)
        for key in (
            "main_min",
            "cross_min",
            "main_max",
            "cross_max",
        )
    }
    if (
        any(type(number) not in {int, float} for number in numbers.values())
        or any(
            not math.isfinite(float(number))
            for number in numbers.values()
        )
        or float(numbers["main_max"]) < float(numbers["main_min"])
        or float(numbers["cross_max"]) < float(numbers["cross_min"])
    ):
        return None
    return {
        key: round(float(number), 6)
        for key, number in numbers.items()
    }


def _canonical_measurements(value: Any) -> dict[str, float] | None:
    keys = (
        "main_span_pt",
        "cross_span_pt",
        "char_width_pt",
        "char_height_pt",
        "main_to_char_width_ratio",
        "cross_to_char_height_ratio",
    )
    if not isinstance(value, dict) or set(value) != set(keys):
        return None
    numbers = {key: value.get(key) for key in keys}
    if (
        any(type(number) not in {int, float} for number in numbers.values())
        or any(
            not math.isfinite(float(number))
            for number in numbers.values()
        )
        or any(float(numbers[key]) < 0.0 for key in (
            "main_span_pt",
            "cross_span_pt",
            "main_to_char_width_ratio",
            "cross_to_char_height_ratio",
        ))
        or float(numbers["char_width_pt"]) <= 0.0
        or float(numbers["char_height_pt"]) <= 0.0
    ):
        return None
    canonical = {
        key: round(float(number), 6)
        for key, number in numbers.items()
    }
    if (
        canonical["main_to_char_width_ratio"]
        != round(
            canonical["main_span_pt"] / canonical["char_width_pt"],
            6,
        )
        or canonical["cross_to_char_height_ratio"]
        != round(
            canonical["cross_span_pt"] / canonical["char_height_pt"],
            6,
        )
    ):
        return None
    return canonical


def _canonical_axis_angle(value: Any) -> float | None:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return None
    normalized = float(value) % 180.0
    if normalized < 0.0:
        normalized += 180.0
    if abs(normalized - 180.0) <= 1e-9:
        normalized = 0.0
    return round(normalized, 6)


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
