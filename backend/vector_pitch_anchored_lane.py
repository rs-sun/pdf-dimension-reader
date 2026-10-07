"""Pure exact-ID local-lane partition for the strict vector pitch reader."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from vector_pitch_window_authority import (
    directed_pitch_axis_angle,
    normalize_pitch_window_authority,
    normalize_pitch_window_source,
)


SCHEMA_VERSION = "vector_pitch_anchored_lane_v1"
GROUPED_SCHEMA_VERSION = "vector_pitch_anchored_lane_v2"
GROUP_LANE_SCHEMA_VERSION = "vector_pitch_anchored_lane_group_v1"
GROUPED_LANE_EXCLUSION_SCHEMA_VERSION = (
    "vector_pitch_anchored_lane_exclusion_v2"
)
LANE_EXCLUSION_SCHEMA_VERSION = "vector_pitch_lane_exclusion_v1"
EXCLUSION_SCHEMA_VERSION = "vector_pitch_primitive_exclusion_v1"
CONSUMER_ALLOWED = False
LANE_INTERVAL_PAD_CHAR_HEIGHT = 0.5
MAX_RECOGNITION_LANE_PRIMITIVES = 180
TRUSTED_AXIS_SOURCES = frozenset({
    "decimal_point_quad",
    "decimal_point_quad.short_axis",
    "oriented_quad:dot_axis_angle",
})


def prepare_anchored_local_lane(
    items: Any,
    *,
    phrase_id: str,
    pitch_window_sources: Any,
    axis_angle_deg: float,
    axis_angle_source: str,
    window_authority: Any = None,
) -> dict[str, Any]:
    """Partition immutable phrase input before the recognition complexity cap."""
    normalized_axis = _canonical_axis_angle(axis_angle_deg)
    directed_axis = directed_pitch_axis_angle(axis_angle_deg)
    if not (
        isinstance(items, list)
        and items
        and type(phrase_id) is str
        and phrase_id
        and normalized_axis is not None
        and directed_axis is not None
        and type(axis_angle_source) is str
        and axis_angle_source in TRUSTED_AXIS_SOURCES
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="anchored_lane_input_incomplete",
        )
    if not isinstance(pitch_window_sources, list) or not pitch_window_sources:
        return _failure(
            phrase_id=phrase_id,
            reason="window_authority_missing",
        )
    normalized_authority = (
        normalize_pitch_window_authority(window_authority)
        if window_authority is not None
        else None
    )
    if window_authority is not None and normalized_authority is None:
        return _failure(
            phrase_id=phrase_id,
            reason="window_authority_invalid",
        )
    if (
        normalized_authority is not None
        and normalized_authority.get("schema_version")
        == "vector_pitch_window_authority_v2"
    ):
        return _prepare_grouped_anchored_local_lane(
            items,
            phrase_id=phrase_id,
            pitch_window_sources=pitch_window_sources,
            axis_angle_deg=axis_angle_deg,
            axis_angle_source=axis_angle_source,
            window_authority=normalized_authority,
        )
    sources_by_digest: dict[str, dict[str, Any]] = {}
    for raw_source in pitch_window_sources:
        source = normalize_pitch_window_source(
            raw_source,
            verify_accounting_digests=False,
        )
        if source is None:
            return _failure(
                phrase_id=phrase_id,
                reason="window_authority_invalid",
            )
        if _axis_delta(source["axis_angle_deg"], normalized_axis) > 0.001:
            return _failure(
                phrase_id=phrase_id,
                reason="window_axis_authority_mismatch",
            )
        prior = sources_by_digest.get(source["source_digest"])
        if prior is not None and prior != source:
            return _failure(
                phrase_id=phrase_id,
                reason="window_authority_invalid",
            )
        sources_by_digest[source["source_digest"]] = source
    sources = [
        sources_by_digest[digest]
        for digest in sorted(sources_by_digest)
    ]
    representative = sources[0]
    if any(
        any(
            source[field] != representative[field]
            for field in ("char_height_pt", "char_width_pt", "pitch_pt")
        )
        for source in sources[1:]
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="window_authority_ambiguous",
        )
    source_cross_centers = [
        _project_point(
            float(source["dot_center"]["x"]),
            float(source["dot_center"]["y"]),
            angle_deg=directed_axis,
        )[1]
        for source in sources
    ]
    if (
        max(source_cross_centers) - min(source_cross_centers)
        > representative["char_height_pt"]
        * LANE_INTERVAL_PAD_CHAR_HEIGHT
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="window_authority_ambiguous",
        )
    projected_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in items:
        row = _project_item(
            item,
            angle_deg=directed_axis,
        )
        if row is None or row["primitive_id"] in seen_ids:
            return _failure(
                phrase_id=phrase_id,
                reason="anchored_lane_input_incomplete",
            )
        seen_ids.add(row["primitive_id"])
        projected_rows.append(row)
    input_primitive_ids = sorted(seen_ids)
    source_primitive_ids = sorted({
        source["source_primitive_id"] for source in sources
    })
    if any(
        primitive_id not in seen_ids
        for primitive_id in source_primitive_ids
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="anchored_lane_input_incomplete",
            input_primitive_ids=input_primitive_ids,
        )
    pad = (
        representative["char_height_pt"]
        * LANE_INTERVAL_PAD_CHAR_HEIGHT
    )
    components = _interval_components(projected_rows, pad=pad)
    matching_components = [
        component
        for component in components
        if set(source_primitive_ids).issubset({
            row["primitive_id"] for row in component
        })
    ]
    if len(matching_components) != 1:
        return _failure(
            phrase_id=phrase_id,
            reason="anchored_lane_ambiguous",
            input_primitive_ids=input_primitive_ids,
            candidate_lane_count=len(components),
        )
    admitted_rows = matching_components[0]
    admitted_primitive_ids = sorted(
        row["primitive_id"] for row in admitted_rows
    )
    admitted_set = set(admitted_primitive_ids)
    excluded_rows = sorted(
        (
            row for row in projected_rows
            if row["primitive_id"] not in admitted_set
        ),
        key=lambda row: row["primitive_id"],
    )
    source_digests = sorted(sources_by_digest)
    exclusions = [
        {
            "schema_version": LANE_EXCLUSION_SCHEMA_VERSION,
            "primitive_id": row["primitive_id"],
            "stage": "anchored_lane",
            "reason": "outside_anchored_local_lane",
            "bbox": row["bbox"],
            "canonical_bbox": {
                "main_min": round(row["main_min"], 6),
                "cross_min": round(row["cross_min"], 6),
                "main_max": round(row["main_max"], 6),
                "cross_max": round(row["cross_max"], 6),
            },
            "window_source_digests": source_digests,
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        for row in excluded_rows
    ]
    selected_cross_min = min(
        row["cross_min"] for row in admitted_rows
    )
    selected_cross_max = max(
        row["cross_max"] for row in admitted_rows
    )
    component_summaries = [
        {
            "primitive_count": len(component),
            "primitive_digest": _digest(sorted(
                row["primitive_id"] for row in component
            )),
            "cross_min": round(
                min(row["cross_min"] for row in component),
                6,
            ),
            "cross_max": round(
                max(row["cross_max"] for row in component),
                6,
            ),
            "contains_source_primitive": bool(
                set(source_primitive_ids) & {
                    row["primitive_id"] for row in component
                }
            ),
        }
        for component in components
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "reason": None,
        "phrase_id": phrase_id,
        "axis_angle_deg": normalized_axis,
        "axis_angle_source": axis_angle_source,
        "char_height_pt": representative["char_height_pt"],
        "char_width_pt": representative["char_width_pt"],
        "interval_pad_char_height": LANE_INTERVAL_PAD_CHAR_HEIGHT,
        "input_primitive_ids": input_primitive_ids,
        "input_primitive_count": len(input_primitive_ids),
        "admitted_primitive_ids": admitted_primitive_ids,
        "admitted_primitive_count": len(admitted_primitive_ids),
        "excluded_primitive_count": len(exclusions),
        "exclusions": exclusions,
        "source_primitive_ids": source_primitive_ids,
        "window_source_digests": source_digests,
        "candidate_lane_count": len(components),
        "candidate_lanes": component_summaries,
        "selected_cross_interval": [
            round(selected_cross_min, 6),
            round(selected_cross_max, 6),
        ],
        "selection_method": "unique_source_containing_cross_component",
        "max_recognition_lane_primitives": (
            MAX_RECOGNITION_LANE_PRIMITIVES
        ),
        "within_180_primitive_budget": (
            len(admitted_primitive_ids)
            <= MAX_RECOGNITION_LANE_PRIMITIVES
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "partition_digest": _digest(payload),
    }


def _prepare_grouped_anchored_local_lane(
    items: list[dict[str, Any]],
    *,
    phrase_id: str,
    pitch_window_sources: list[dict[str, Any]],
    axis_angle_deg: float,
    axis_angle_source: str,
    window_authority: dict[str, Any],
) -> dict[str, Any]:
    sources_by_digest: dict[str, dict[str, Any]] = {}
    for raw_source in pitch_window_sources:
        source = normalize_pitch_window_source(
            raw_source,
            verify_accounting_digests=False,
        )
        if source is None:
            return _failure(
                phrase_id=phrase_id,
                reason="window_authority_invalid",
            )
        prior = sources_by_digest.get(source["source_digest"])
        if prior is not None and prior != source:
            return _failure(
                phrase_id=phrase_id,
                reason="window_authority_invalid",
            )
        sources_by_digest[source["source_digest"]] = source
    groups = []
    for authority_group in window_authority["groups"]:
        group_authority = authority_group["window_authority"]
        size_class = authority_group["size_class"]
        group_sources = sorted(
            (
                source
                for source in sources_by_digest.values()
                if source.get("derivation", {}).get("cluster_role")
                == size_class
            ),
            key=lambda source: source["source_digest"],
        )
        if not group_sources:
            return _failure(
                phrase_id=phrase_id,
                reason="window_authority_missing",
            )
        group_sources = [
            source
            for source in group_sources
            if (
                source["page_index"] == group_authority["page_index"]
                and _axis_delta(
                    source["axis_angle_deg"],
                    group_authority["axis_angle_deg"],
                )
                <= 0.001
            )
        ]
        if not group_sources:
            return _failure(
                phrase_id=phrase_id,
                reason="window_axis_authority_mismatch",
            )
        lane_plan = prepare_anchored_local_lane(
            items,
            phrase_id=phrase_id,
            pitch_window_sources=group_sources,
            axis_angle_deg=axis_angle_deg,
            axis_angle_source=axis_angle_source,
        )
        if lane_plan.get("status") != "ready":
            return _failure(
                phrase_id=phrase_id,
                reason=str(
                    lane_plan.get("reason")
                    or "anchored_lane_ambiguous"
                ),
                input_primitive_ids=lane_plan.get("input_primitive_ids"),
                candidate_lane_count=int(
                    lane_plan.get("candidate_lane_count") or 0
                ),
            )
        group = {
            "schema_version": GROUP_LANE_SCHEMA_VERSION,
            "size_class": authority_group["size_class"],
            "group_digest": authority_group["group_digest"],
            "lane_plan": lane_plan,
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        group["group_lane_digest"] = _digest(group)
        groups.append(group)

    input_ids = groups[0]["lane_plan"]["input_primitive_ids"]
    if any(group["lane_plan"]["input_primitive_ids"] != input_ids for group in groups):
        return _failure(
            phrase_id=phrase_id,
            reason="anchored_lane_input_incomplete",
        )
    admitted_ids = sorted(set().union(*(
        set(group["lane_plan"]["admitted_primitive_ids"])
        for group in groups
    )))
    admitted_set = set(admitted_ids)
    excluded_ids = sorted(set(input_ids) - admitted_set)
    directed_axis = directed_pitch_axis_angle(axis_angle_deg)
    if directed_axis is None:
        return _failure(
            phrase_id=phrase_id,
            reason="anchored_lane_input_incomplete",
        )
    projected_by_id = {}
    for item in items:
        projected = _project_item(item, angle_deg=directed_axis)
        if projected is None or projected["primitive_id"] in projected_by_id:
            return _failure(
                phrase_id=phrase_id,
                reason="anchored_lane_input_incomplete",
            )
        projected_by_id[projected["primitive_id"]] = projected
    if sorted(projected_by_id) != input_ids:
        return _failure(
            phrase_id=phrase_id,
            reason="anchored_lane_input_incomplete",
        )
    primary_group = next(
        group for group in groups if group["size_class"] == "primary"
    )
    exclusions = []
    for primitive_id in excluded_ids:
        row = projected_by_id[primitive_id]
        exclusion = {
            "schema_version": GROUPED_LANE_EXCLUSION_SCHEMA_VERSION,
            "primitive_id": primitive_id,
            "stage": "anchored_lane",
            "reason": "outside_anchored_local_lane",
            "bbox": row["bbox"],
            "canonical_bbox": {
                "main_min": round(row["main_min"], 6),
                "cross_min": round(row["cross_min"], 6),
                "main_max": round(row["main_max"], 6),
                "cross_max": round(row["cross_max"], 6),
            },
            "window_source_digests": list(
                window_authority["group_member_source_digests"]
            ),
            "reference_size_class": "primary",
            "reference_group_digest": primary_group["group_digest"],
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        exclusions.append(exclusion)
    payload = {
        "schema_version": GROUPED_SCHEMA_VERSION,
        "status": "ready",
        "reason": None,
        "phrase_id": phrase_id,
        "axis_angle_deg": round(float(axis_angle_deg) % 180.0, 6),
        "axis_angle_source": axis_angle_source,
        "window_authority_digest": window_authority["authority_digest"],
        "cross_group_relation_digest": window_authority[
            "cross_group_relation"
        ]["relation_digest"],
        "groups": groups,
        "input_primitive_ids": list(input_ids),
        "input_primitive_count": len(input_ids),
        "admitted_primitive_ids": admitted_ids,
        "admitted_primitive_count": len(admitted_ids),
        "excluded_primitive_count": len(exclusions),
        "exclusions": exclusions,
        "source_primitive_ids": list(
            window_authority["group_member_source_primitive_ids"]
        ),
        "window_source_digests": list(
            window_authority["group_member_source_digests"]
        ),
        "candidate_lane_count": sum(
            int(group["lane_plan"]["candidate_lane_count"])
            for group in groups
        ),
        "selection_method": "union_of_strict_size_class_cross_components",
        "max_recognition_lane_primitives": MAX_RECOGNITION_LANE_PRIMITIVES,
        "within_180_primitive_budget": (
            len(admitted_ids) <= MAX_RECOGNITION_LANE_PRIMITIVES
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "partition_digest": _digest(payload),
    }


def normalize_anchored_local_lane(
    value: Any,
) -> dict[str, Any] | None:
    if isinstance(value, dict) and value.get("schema_version") == GROUPED_SCHEMA_VERSION:
        return _normalize_grouped_anchored_local_lane(value)
    return _normalize_anchored_local_lane_v1(value)


def _normalize_anchored_local_lane_v1(
    value: Any,
) -> dict[str, Any] | None:
    """Validate the complete ready-lane payload and its exact-ID partition."""
    expected_keys = {
        "schema_version",
        "status",
        "reason",
        "phrase_id",
        "axis_angle_deg",
        "axis_angle_source",
        "char_height_pt",
        "char_width_pt",
        "interval_pad_char_height",
        "input_primitive_ids",
        "input_primitive_count",
        "admitted_primitive_ids",
        "admitted_primitive_count",
        "excluded_primitive_count",
        "exclusions",
        "source_primitive_ids",
        "window_source_digests",
        "candidate_lane_count",
        "candidate_lanes",
        "selected_cross_interval",
        "selection_method",
        "max_recognition_lane_primitives",
        "within_180_primitive_budget",
        "partition_digest",
        "consumer_allowed",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    phrase_id = value.get("phrase_id")
    axis_angle = _canonical_axis_angle(value.get("axis_angle_deg"))
    char_height = _strict_positive_number(value.get("char_height_pt"))
    char_width = _strict_positive_number(value.get("char_width_pt"))
    input_ids = _canonical_ids(value.get("input_primitive_ids"))
    admitted_ids = _canonical_ids(value.get("admitted_primitive_ids"))
    source_ids = _canonical_ids(value.get("source_primitive_ids"))
    source_digests = _canonical_ids(value.get("window_source_digests"))
    selected_interval = _canonical_interval(
        value.get("selected_cross_interval")
    )
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == "ready"
        and value.get("reason") is None
        and type(phrase_id) is str
        and phrase_id
        and axis_angle is not None
        and value.get("axis_angle_source") in TRUSTED_AXIS_SOURCES
        and char_height is not None
        and char_width is not None
        and value.get("interval_pad_char_height")
        == LANE_INTERVAL_PAD_CHAR_HEIGHT
        and input_ids is not None
        and input_ids
        and admitted_ids is not None
        and admitted_ids
        and source_ids is not None
        and source_ids
        and source_digests is not None
        and source_digests
        and all(_valid_sha256(digest) for digest in source_digests)
        and selected_interval is not None
        and value.get("selection_method")
        == "unique_source_containing_cross_component"
        and value.get("max_recognition_lane_primitives")
        == MAX_RECOGNITION_LANE_PRIMITIVES
        and value.get("within_180_primitive_budget")
        is (
            len(admitted_ids)
            <= MAX_RECOGNITION_LANE_PRIMITIVES
        )
        and value.get("consumer_allowed") is False
    ):
        return None
    exclusions = _canonical_lane_exclusions(
        value.get("exclusions"),
        source_digests=source_digests,
    )
    candidate_lanes = _canonical_candidate_lanes(
        value.get("candidate_lanes")
    )
    if exclusions is None or candidate_lanes is None:
        return None
    excluded_ids = [
        row["primitive_id"] for row in exclusions
    ]
    if (
        value.get("input_primitive_count") != len(input_ids)
        or value.get("admitted_primitive_count") != len(admitted_ids)
        or value.get("excluded_primitive_count") != len(excluded_ids)
        or value.get("candidate_lane_count") != len(candidate_lanes)
        or sum(
            row["primitive_count"] for row in candidate_lanes
        ) != len(input_ids)
        or sum(
            row["contains_source_primitive"]
            for row in candidate_lanes
        ) != 1
        or set(admitted_ids).intersection(excluded_ids)
        or set(admitted_ids).union(excluded_ids) != set(input_ids)
        or not set(source_ids).issubset(admitted_ids)
    ):
        return None
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "reason": None,
        "phrase_id": phrase_id,
        "axis_angle_deg": axis_angle,
        "axis_angle_source": value["axis_angle_source"],
        "char_height_pt": round(char_height, 6),
        "char_width_pt": round(char_width, 6),
        "interval_pad_char_height": LANE_INTERVAL_PAD_CHAR_HEIGHT,
        "input_primitive_ids": input_ids,
        "input_primitive_count": len(input_ids),
        "admitted_primitive_ids": admitted_ids,
        "admitted_primitive_count": len(admitted_ids),
        "excluded_primitive_count": len(excluded_ids),
        "exclusions": exclusions,
        "source_primitive_ids": source_ids,
        "window_source_digests": source_digests,
        "candidate_lane_count": len(candidate_lanes),
        "candidate_lanes": candidate_lanes,
        "selected_cross_interval": selected_interval,
        "selection_method": "unique_source_containing_cross_component",
        "max_recognition_lane_primitives": (
            MAX_RECOGNITION_LANE_PRIMITIVES
        ),
        "within_180_primitive_budget": (
            len(admitted_ids)
            <= MAX_RECOGNITION_LANE_PRIMITIVES
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    partition_digest = value.get("partition_digest")
    if (
        not _valid_sha256(partition_digest)
        or partition_digest != _digest(canonical)
    ):
        return None
    canonical["partition_digest"] = partition_digest
    return canonical if canonical == value else None


def _normalize_grouped_anchored_local_lane(
    value: Any,
) -> dict[str, Any] | None:
    expected_keys = {
        "schema_version",
        "status",
        "reason",
        "phrase_id",
        "axis_angle_deg",
        "axis_angle_source",
        "window_authority_digest",
        "cross_group_relation_digest",
        "groups",
        "input_primitive_ids",
        "input_primitive_count",
        "admitted_primitive_ids",
        "admitted_primitive_count",
        "excluded_primitive_count",
        "exclusions",
        "source_primitive_ids",
        "window_source_digests",
        "candidate_lane_count",
        "selection_method",
        "max_recognition_lane_primitives",
        "within_180_primitive_budget",
        "consumer_allowed",
        "partition_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    phrase_id = value.get("phrase_id")
    axis_angle = _canonical_axis_angle(value.get("axis_angle_deg"))
    input_ids = _canonical_ids(value.get("input_primitive_ids"))
    admitted_ids = _canonical_ids(value.get("admitted_primitive_ids"))
    source_ids = _canonical_ids(value.get("source_primitive_ids"))
    source_digests = _canonical_ids(value.get("window_source_digests"))
    raw_groups = value.get("groups")
    groups = []
    if isinstance(raw_groups, list):
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict) or set(raw_group) != {
                "schema_version",
                "size_class",
                "group_digest",
                "lane_plan",
                "consumer_allowed",
                "group_lane_digest",
            }:
                return None
            lane_plan = _normalize_anchored_local_lane_v1(
                raw_group.get("lane_plan")
            )
            canonical_group = {
                "schema_version": GROUP_LANE_SCHEMA_VERSION,
                "size_class": raw_group.get("size_class"),
                "group_digest": raw_group.get("group_digest"),
                "lane_plan": lane_plan,
                "consumer_allowed": CONSUMER_ALLOWED,
                "group_lane_digest": raw_group.get("group_lane_digest"),
            }
            if not (
                raw_group.get("schema_version") == GROUP_LANE_SCHEMA_VERSION
                and raw_group.get("size_class") in {"primary", "tolerance"}
                and _valid_sha256(raw_group.get("group_digest"))
                and lane_plan is not None
                and raw_group.get("consumer_allowed") is False
                and _valid_sha256(raw_group.get("group_lane_digest"))
                and raw_group["group_lane_digest"] == _digest({
                    key: item
                    for key, item in canonical_group.items()
                    if key != "group_lane_digest"
                })
                and canonical_group == raw_group
            ):
                return None
            groups.append(canonical_group)
    if not (
        value.get("schema_version") == GROUPED_SCHEMA_VERSION
        and value.get("status") == "ready"
        and value.get("reason") is None
        and type(phrase_id) is str
        and bool(phrase_id)
        and axis_angle is not None
        and value.get("axis_angle_source") in TRUSTED_AXIS_SOURCES
        and _valid_sha256(value.get("window_authority_digest"))
        and _valid_sha256(value.get("cross_group_relation_digest"))
        and [group["size_class"] for group in groups]
        == ["primary", "tolerance"]
        and input_ids is not None
        and bool(input_ids)
        and admitted_ids is not None
        and bool(admitted_ids)
        and source_ids is not None
        and bool(source_ids)
        and source_digests is not None
        and bool(source_digests)
        and all(_valid_sha256(digest) for digest in source_digests)
        and type(value.get("candidate_lane_count")) is int
        and value["candidate_lane_count"] >= 1
        and value.get("selection_method")
        == "union_of_strict_size_class_cross_components"
        and value.get("max_recognition_lane_primitives")
        == MAX_RECOGNITION_LANE_PRIMITIVES
        and value.get("consumer_allowed") is False
        and _valid_sha256(value.get("partition_digest"))
    ):
        return None
    if any(
        group["lane_plan"]["phrase_id"] != phrase_id
        or _axis_delta(group["lane_plan"]["axis_angle_deg"], axis_angle) > 0.001
        or group["lane_plan"]["input_primitive_ids"] != input_ids
        for group in groups
    ):
        return None
    expected_admitted = sorted(set().union(*(
        set(group["lane_plan"]["admitted_primitive_ids"])
        for group in groups
    )))
    expected_source_ids = sorted(set().union(*(
        set(group["lane_plan"]["source_primitive_ids"])
        for group in groups
    )))
    expected_source_digests = sorted(set().union(*(
        set(group["lane_plan"]["window_source_digests"])
        for group in groups
    )))
    excluded_ids = sorted(set(input_ids) - set(admitted_ids))
    raw_exclusions = value.get("exclusions")
    exclusions = []
    primary_group_digest = groups[0]["group_digest"]
    if not isinstance(raw_exclusions, list):
        return None
    for raw in raw_exclusions:
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "primitive_id",
            "stage",
            "reason",
            "bbox",
            "canonical_bbox",
            "window_source_digests",
            "reference_size_class",
            "reference_group_digest",
            "consumer_allowed",
        }:
            return None
        primitive_id = raw.get("primitive_id")
        bbox = _canonical_bbox(raw.get("bbox"), points=[])
        canonical_bbox = _canonical_axis_bbox(raw.get("canonical_bbox"))
        exclusion_source_digests = _canonical_ids(
            raw.get("window_source_digests")
        )
        if not (
            raw.get("schema_version") == GROUPED_LANE_EXCLUSION_SCHEMA_VERSION
            and type(primitive_id) is str
            and bool(primitive_id)
            and raw.get("stage") == "anchored_lane"
            and raw.get("reason") == "outside_anchored_local_lane"
            and bbox is not None
            and canonical_bbox is not None
            and exclusion_source_digests == source_digests
            and raw.get("reference_size_class") == "primary"
            and raw.get("reference_group_digest") == primary_group_digest
            and raw.get("consumer_allowed") is False
        ):
            return None
        exclusions.append({
            "schema_version": GROUPED_LANE_EXCLUSION_SCHEMA_VERSION,
            "primitive_id": primitive_id,
            "stage": "anchored_lane",
            "reason": "outside_anchored_local_lane",
            "bbox": bbox,
            "canonical_bbox": canonical_bbox,
            "window_source_digests": exclusion_source_digests,
            "reference_size_class": "primary",
            "reference_group_digest": primary_group_digest,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    exclusions.sort(key=lambda row: row["primitive_id"])
    if (
        admitted_ids != expected_admitted
        or source_ids != expected_source_ids
        or source_digests != expected_source_digests
        or [row["primitive_id"] for row in exclusions] != excluded_ids
        or value.get("input_primitive_count") != len(input_ids)
        or value.get("admitted_primitive_count") != len(admitted_ids)
        or value.get("excluded_primitive_count") != len(exclusions)
        or value.get("candidate_lane_count") != sum(
            group["lane_plan"]["candidate_lane_count"] for group in groups
        )
        or value.get("within_180_primitive_budget")
        is not (len(admitted_ids) <= MAX_RECOGNITION_LANE_PRIMITIVES)
    ):
        return None
    canonical = {
        "schema_version": GROUPED_SCHEMA_VERSION,
        "status": "ready",
        "reason": None,
        "phrase_id": phrase_id,
        "axis_angle_deg": axis_angle,
        "axis_angle_source": value["axis_angle_source"],
        "window_authority_digest": value["window_authority_digest"],
        "cross_group_relation_digest": value[
            "cross_group_relation_digest"
        ],
        "groups": groups,
        "input_primitive_ids": input_ids,
        "input_primitive_count": len(input_ids),
        "admitted_primitive_ids": admitted_ids,
        "admitted_primitive_count": len(admitted_ids),
        "excluded_primitive_count": len(exclusions),
        "exclusions": exclusions,
        "source_primitive_ids": source_ids,
        "window_source_digests": source_digests,
        "candidate_lane_count": value["candidate_lane_count"],
        "selection_method": "union_of_strict_size_class_cross_components",
        "max_recognition_lane_primitives": MAX_RECOGNITION_LANE_PRIMITIVES,
        "within_180_primitive_budget": value[
            "within_180_primitive_budget"
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if value["partition_digest"] != _digest(canonical):
        return None
    canonical["partition_digest"] = value["partition_digest"]
    return canonical if canonical == value else None


def bind_anchored_lane_exclusions(
    lane_plan: Any,
    *,
    window_authority: Any,
) -> list[dict[str, Any]] | None:
    """Bind provisional outside-lane geometry to the final phrase authority."""
    lane = normalize_anchored_local_lane(lane_plan)
    authority = normalize_pitch_window_authority(window_authority)
    if lane is None or authority is None:
        return None
    if (
        lane.get("schema_version") == GROUPED_SCHEMA_VERSION
        or authority.get("schema_version")
        == "vector_pitch_window_authority_v2"
    ):
        if not (
            lane.get("schema_version") == GROUPED_SCHEMA_VERSION
            and authority.get("schema_version")
            == "vector_pitch_window_authority_v2"
            and lane["phrase_id"] == authority["phrase_id"]
            and _axis_delta(
                lane["axis_angle_deg"],
                authority["axis_angle_deg"],
            )
            <= 0.001
            and lane["window_authority_digest"]
            == authority["authority_digest"]
            and lane["cross_group_relation_digest"]
            == authority["cross_group_relation"]["relation_digest"]
            and lane["window_source_digests"]
            == authority["group_member_source_digests"]
            and lane["source_primitive_ids"]
            == authority["group_member_source_primitive_ids"]
        ):
            return None
        authority_groups = {
            group["size_class"]: group for group in authority["groups"]
        }
        for lane_group in lane["groups"]:
            authority_group = authority_groups.get(lane_group["size_class"])
            group_authority = (
                authority_group.get("window_authority")
                if isinstance(authority_group, dict)
                else None
            )
            if not (
                isinstance(group_authority, dict)
                and lane_group["group_digest"]
                == authority_group["group_digest"]
                and lane_group["lane_plan"]["char_height_pt"]
                == group_authority["char_height_pt"]
                and lane_group["lane_plan"]["char_width_pt"]
                == group_authority["char_width_pt"]
                and lane_group["lane_plan"]["window_source_digests"]
                == group_authority["group_member_source_digests"]
                and lane_group["lane_plan"]["source_primitive_ids"]
                == group_authority["group_member_source_primitive_ids"]
            ):
                return None
        rows = []
        for exclusion in lane["exclusions"]:
            reference_group = authority_groups.get(
                exclusion["reference_size_class"]
            )
            if not (
                isinstance(reference_group, dict)
                and exclusion["reference_group_digest"]
                == reference_group["group_digest"]
            ):
                return None
            group_authority = reference_group["window_authority"]
            char_width = float(group_authority["char_width_pt"])
            char_height = float(group_authority["char_height_pt"])
            canonical_bbox = exclusion["canonical_bbox"]
            main_span = (
                canonical_bbox["main_max"]
                - canonical_bbox["main_min"]
            )
            cross_span = (
                canonical_bbox["cross_max"]
                - canonical_bbox["cross_min"]
            )
            rows.append({
                "schema_version": EXCLUSION_SCHEMA_VERSION,
                "primitive_ids": [exclusion["primitive_id"]],
                "stage": "anchored_lane",
                "reason": "outside_anchored_local_lane",
                "bbox": list(exclusion["bbox"]),
                "canonical_bbox": dict(canonical_bbox),
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
                    lane["window_source_digests"]
                ),
                "axis_angle_deg": authority["axis_angle_deg"],
                "relation": "outside_selected_cross_component",
                "group_digest": reference_group["group_digest"],
                "consumer_allowed": CONSUMER_ALLOWED,
            })
        return rows
    if (
        lane["phrase_id"] != authority["phrase_id"]
        or _axis_delta(
            lane["axis_angle_deg"],
            authority["axis_angle_deg"],
        )
        > 0.001
        or lane["char_height_pt"] != authority["char_height_pt"]
        or lane["char_width_pt"] != authority["char_width_pt"]
        or lane["window_source_digests"]
        != authority["group_member_source_digests"]
        or lane["source_primitive_ids"]
        != authority["group_member_source_primitive_ids"]
    ):
        return None
    char_width = float(authority["char_width_pt"])
    char_height = float(authority["char_height_pt"])
    rows: list[dict[str, Any]] = []
    for exclusion in lane["exclusions"]:
        canonical_bbox = exclusion["canonical_bbox"]
        main_span = (
            canonical_bbox["main_max"]
            - canonical_bbox["main_min"]
        )
        cross_span = (
            canonical_bbox["cross_max"]
            - canonical_bbox["cross_min"]
        )
        rows.append({
            "schema_version": EXCLUSION_SCHEMA_VERSION,
            "primitive_ids": [exclusion["primitive_id"]],
            "stage": "anchored_lane",
            "reason": "outside_anchored_local_lane",
            "bbox": list(exclusion["bbox"]),
            "canonical_bbox": dict(canonical_bbox),
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
                lane["window_source_digests"]
            ),
            "axis_angle_deg": authority["axis_angle_deg"],
            "relation": "outside_selected_cross_component",
            "group_digest": None,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return rows


def _failure(
    *,
    phrase_id: Any,
    reason: str,
    input_primitive_ids: list[str] | None = None,
    candidate_lane_count: int = 0,
) -> dict[str, Any]:
    resolved_ids = list(input_primitive_ids or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail_closed",
        "reason": reason,
        "phrase_id": str(phrase_id or ""),
        "input_primitive_ids": resolved_ids,
        "input_primitive_count": len(resolved_ids),
        "admitted_primitive_ids": [],
        "admitted_primitive_count": 0,
        "excluded_primitive_count": 0,
        "exclusions": [],
        "candidate_lane_count": int(candidate_lane_count),
        "within_180_primitive_budget": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _project_item(
    item: Any,
    *,
    angle_deg: float,
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    primitive_id = item.get("item_id")
    if type(primitive_id) is not str or not primitive_id:
        return None
    raw_points = item.get("points")
    points: list[tuple[float, float]] = []
    if isinstance(raw_points, list):
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return None
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            points.append((x, y))
    if not points:
        return None
    projected = [
        _project_point(x, y, angle_deg=angle_deg)
        for x, y in points
    ]
    main_values = [point[0] for point in projected]
    cross_values = [point[1] for point in projected]
    bbox = _canonical_bbox(item.get("bbox"), points=points)
    if bbox is None:
        return None
    return {
        "primitive_id": primitive_id,
        "bbox": bbox,
        "main_min": min(main_values),
        "main_max": max(main_values),
        "cross_min": min(cross_values),
        "cross_max": max(cross_values),
    }


def _interval_components(
    rows: list[dict[str, Any]],
    *,
    pad: float,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["cross_min"] - pad,
            row["cross_max"] + pad,
            row["primitive_id"],
        ),
    )
    components: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_max = -math.inf
    for row in ordered:
        expanded_min = row["cross_min"] - pad
        expanded_max = row["cross_max"] + pad
        if current and expanded_min > current_max + 1e-9:
            components.append(current)
            current = []
            current_max = -math.inf
        current.append(row)
        current_max = max(current_max, expanded_max)
    if current:
        components.append(current)
    return components


def _project_point(
    x: float,
    y: float,
    *,
    angle_deg: float,
) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    main_x = math.cos(radians)
    main_y = math.sin(radians)
    return (
        x * main_x + y * main_y,
        x * -main_y + y * main_x,
    )


def _canonical_bbox(
    value: Any,
    *,
    points: list[tuple[float, float]],
) -> list[float] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(type(number) in {int, float} for number in value)
    ):
        numbers = [float(number) for number in value]
        if all(math.isfinite(number) for number in numbers):
            return [round(number, 6) for number in numbers]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [
        round(min(xs), 6),
        round(min(ys), 6),
        round(max(xs), 6),
        round(max(ys), 6),
    ]


def _canonical_axis_angle(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number % 180.0, 6)


def _canonical_ids(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        return None
    canonical = sorted(value)
    return canonical if canonical == value else None


def _canonical_interval(value: Any) -> list[float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(number) not in {int, float} for number in value)
    ):
        return None
    numbers = [float(number) for number in value]
    if (
        any(not math.isfinite(number) for number in numbers)
        or numbers[1] < numbers[0]
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
        key: (
            float(value[key])
            if type(value.get(key)) in {int, float}
            and math.isfinite(float(value[key]))
            else None
        )
        for key in value
    }
    if (
        any(number is None for number in numbers.values())
        or float(numbers["main_max"]) < float(numbers["main_min"])
        or float(numbers["cross_max"]) < float(numbers["cross_min"])
    ):
        return None
    return {
        key: round(float(numbers[key]), 6)
        for key in (
            "main_min",
            "cross_min",
            "main_max",
            "cross_max",
        )
    }


def _canonical_lane_exclusions(
    value: Any,
    *,
    source_digests: list[str],
) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    rows: list[dict[str, Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, dict) or set(raw_row) != {
            "schema_version",
            "primitive_id",
            "stage",
            "reason",
            "bbox",
            "canonical_bbox",
            "window_source_digests",
            "consumer_allowed",
        }:
            return None
        primitive_id = raw_row.get("primitive_id")
        bbox = _canonical_raw_bbox(raw_row.get("bbox"))
        canonical_bbox = _canonical_projected_bbox(
            raw_row.get("canonical_bbox")
        )
        if not (
            raw_row.get("schema_version")
            == LANE_EXCLUSION_SCHEMA_VERSION
            and type(primitive_id) is str
            and primitive_id
            and raw_row.get("stage") == "anchored_lane"
            and raw_row.get("reason")
            == "outside_anchored_local_lane"
            and bbox is not None
            and canonical_bbox is not None
            and raw_row.get("window_source_digests")
            == source_digests
            and raw_row.get("consumer_allowed") is False
        ):
            return None
        rows.append({
            "schema_version": LANE_EXCLUSION_SCHEMA_VERSION,
            "primitive_id": primitive_id,
            "stage": "anchored_lane",
            "reason": "outside_anchored_local_lane",
            "bbox": bbox,
            "canonical_bbox": canonical_bbox,
            "window_source_digests": source_digests,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    rows.sort(key=lambda row: row["primitive_id"])
    if len({row["primitive_id"] for row in rows}) != len(rows):
        return None
    return rows


def _canonical_candidate_lanes(
    value: Any,
) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    rows: list[dict[str, Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, dict) or set(raw_row) != {
            "primitive_count",
            "primitive_digest",
            "cross_min",
            "cross_max",
            "contains_source_primitive",
        }:
            return None
        primitive_count = raw_row.get("primitive_count")
        interval = _canonical_interval([
            raw_row.get("cross_min"),
            raw_row.get("cross_max"),
        ])
        if not (
            type(primitive_count) is int
            and primitive_count > 0
            and _valid_sha256(raw_row.get("primitive_digest"))
            and interval is not None
            and type(raw_row.get("contains_source_primitive")) is bool
        ):
            return None
        rows.append({
            "primitive_count": primitive_count,
            "primitive_digest": raw_row["primitive_digest"],
            "cross_min": interval[0],
            "cross_max": interval[1],
            "contains_source_primitive": raw_row[
                "contains_source_primitive"
            ],
        })
    return rows


def _canonical_raw_bbox(value: Any) -> list[float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(number) not in {int, float} for number in value)
    ):
        return None
    numbers = [float(number) for number in value]
    if (
        any(not math.isfinite(number) for number in numbers)
        or numbers[2] < numbers[0]
        or numbers[3] < numbers[1]
    ):
        return None
    return [round(number, 6) for number in numbers]


def _canonical_projected_bbox(
    value: Any,
) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != {
        "main_min",
        "cross_min",
        "main_max",
        "cross_max",
    }:
        return None
    raw = [
        value.get("main_min"),
        value.get("cross_min"),
        value.get("main_max"),
        value.get("cross_max"),
    ]
    bbox = _canonical_raw_bbox(raw)
    if bbox is None:
        return None
    return {
        "main_min": bbox[0],
        "cross_min": bbox[1],
        "main_max": bbox[2],
        "cross_max": bbox[3],
    }


def _strict_positive_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _axis_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
