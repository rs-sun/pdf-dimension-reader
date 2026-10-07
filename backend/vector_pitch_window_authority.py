"""Audited decimal-derived geometry for the strict vector pitch reader.

The existing ``vector_decimal_anchor_authority_v1`` remains semantic-only.
This module builds a separate geometry authority from the raw decimal quad and
its direct char-height corridor join.  It is internal evidence and is never a
UI consumer contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from decimal_anchor_authority import (
    normalize_decimal_anchor_authority,
)
from decimal_char_height_calibration import DEFAULT_K1, DEFAULT_K2


SOURCE_SCHEMA_VERSION = "vector_pitch_window_source_v1"
AUTHORITY_SCHEMA_VERSION = "vector_pitch_window_authority_v1"
AUTHORITY_BUNDLE_SCHEMA_VERSION = "vector_pitch_window_authority_v2"
GROUP_AUTHORITY_SCHEMA_VERSION = "vector_pitch_window_group_authority_v1"
CROSS_GROUP_RELATION_SCHEMA_VERSION = (
    "vector_pitch_window_cross_group_relation_v1"
)
AUTHORITY_PRODUCER = "phrase_bound_decimal_window_resolver"
GROUP_AUTHORITY_PRODUCER = "size_class_grouped_decimal_window_resolver"
CROSS_GROUP_RELATION_PRODUCER = "text_phrase_source_spatial_relation"
SOURCE_PRODUCER = "decimal_quad_direct_corridor_window_source"
DECIMAL_QUAD_SCHEMA_VERSION = "vector_decimal_point_quad_v1"
DECIMAL_QUAD_SOURCE = "qu_rect_ratio"
DECIMAL_CORRIDOR_SCHEMA_VERSION = "vector_decimal_corridor_v1"
DECIMAL_CORRIDOR_SOURCE = "decimal_char_height_back_inference"
AXIS_SOURCE = "decimal_point_quad.short_axis"
CONSUMER_ALLOWED = False
GEOMETRY_RELATIVE_TOLERANCE = 0.01
AXIS_GEOMETRY_TOLERANCE_DEG = 0.1
PERPENDICULAR_ERROR_LIMIT_DEG = 8.0
MULTI_ANCHOR_CROSS_TOLERANCE_HEIGHT = 0.5
MULTI_ANCHOR_PHASE_TOLERANCE_PITCH = 0.1
CROSS_GROUP_CROSS_TOLERANCE_HEIGHT = 0.5
DECIMAL_POINT_COORDINATE_QUANTIZATION_PT = 0.001
GROUPED_SIZE_CLASSES = ("primary", "tolerance")


def directed_pitch_axis_angle(value: Any) -> float | None:
    """Canonicalize a trusted mod-180 line to one reading direction."""
    axis_angle = _canonical_axis_angle(value)
    if axis_angle is None:
        return None
    return round(
        axis_angle - 180.0
        if axis_angle >= 45.0
        else axis_angle,
        6,
    )


def _normalize_decimal_anchor_for_window(
    value: Any,
    *,
    verify_accounting_digest: bool,
) -> dict[str, Any] | None:
    if verify_accounting_digest or not isinstance(value, dict):
        return normalize_decimal_anchor_authority(value)
    rebound = dict(value)
    rebound["authority_digest"] = _digest({
        key: item
        for key, item in rebound.items()
        if key != "authority_digest"
    })
    return normalize_decimal_anchor_authority(rebound)


def resolve_pitch_window_authority(
    source_candidate_evidence: Any,
    *,
    page_index: int,
    phrase_id: str,
    phrase_axis_angle_deg: float,
    phrase_axis_angle_source: str,
    phrase_primitive_ids: list[str],
    page_context_manifest: dict[str, Any],
    phrase_items: list[dict[str, Any]],
    verify_accounting_digests: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve phrase-bound geometry authority, or one explicit fail reason."""
    if not isinstance(source_candidate_evidence, list):
        return None, "window_authority_missing"
    source_rows = [
        row
        for row in source_candidate_evidence
        if isinstance(row, dict) and "pitch_window_source" in row
    ]
    if not source_rows:
        return None, "window_authority_missing"
    if not (
        type(page_index) is int
        and page_index >= 0
        and type(phrase_id) is str
        and bool(phrase_id)
        and type(phrase_axis_angle_source) is str
        and phrase_axis_angle_source in {
            "decimal_point_quad",
            "decimal_point_quad.short_axis",
            "oriented_quad:dot_axis_angle",
        }
        and (phrase_axis := _canonical_axis_angle(
            phrase_axis_angle_deg
        ))
        is not None
        and _valid_primitive_ids(phrase_primitive_ids)
        and _valid_page_context_manifest(
            page_context_manifest,
            page_index=page_index,
            minimum_primitive_count=len(phrase_primitive_ids),
        )
        and isinstance(phrase_items, list)
        and len(phrase_items) == len(phrase_primitive_ids)
    ):
        return None, "window_authority_invalid"
    by_source_digest: dict[str, dict[str, Any]] = {}
    candidate_ids_by_source_digest: dict[str, set[str]] = {}
    size_class_by_source_digest: dict[str, str] = {}
    rows_by_source_digest: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        source = normalize_pitch_window_source(
            row.get("pitch_window_source"),
            verify_accounting_digests=verify_accounting_digests,
        )
        semantic_authority = _normalize_decimal_anchor_for_window(
            row.get("decimal_anchor_authority"),
            verify_accounting_digest=verify_accounting_digests,
        )
        if (
            source is not None
            and semantic_authority is not None
            and not verify_accounting_digests
        ):
            source = {
                **source,
                "source_decimal_authority_digest": semantic_authority[
                    "authority_digest"
                ],
            }
            source["source_digest"] = _digest({
                key: item
                for key, item in source.items()
                if key != "source_digest"
            })
        candidate_id = row.get("candidate_id")
        if (
            source is None
            or semantic_authority is None
            or row.get("decimal_anchor_authority_source")
            != "dot_reference"
            or type(candidate_id) is not str
            or not candidate_id
            or type(row.get("page_index")) is not int
            or row["page_index"] != page_index
            or source["page_index"] != page_index
            or source["source_decimal_authority_digest"]
            != semantic_authority["authority_digest"]
            or source["decimal_quad_id"]
            != semantic_authority["decimal_quad_id"]
            or source["source_primitive_id"]
            != semantic_authority["source_primitive_id"]
            or semantic_authority.get("size_class")
            not in GROUPED_SIZE_CLASSES
            or source.get("derivation", {}).get("cluster_role")
            != semantic_authority.get("size_class")
        ):
            return None, "window_authority_invalid"
        source_digest = source["source_digest"]
        prior = by_source_digest.get(source_digest)
        if prior is not None and prior != source:
            return None, "window_authority_invalid"
        by_source_digest[source_digest] = source
        prior_size_class = size_class_by_source_digest.get(source_digest)
        size_class = str(semantic_authority["size_class"])
        if prior_size_class is not None and prior_size_class != size_class:
            return None, "window_authority_invalid"
        size_class_by_source_digest[source_digest] = size_class
        rows_by_source_digest.setdefault(source_digest, []).append(row)
        candidate_ids_by_source_digest.setdefault(
            source_digest,
            set(),
        ).add(candidate_id)
    ordered_source_pairs = sorted(by_source_digest.items())
    source_digest, source = min(
        ordered_source_pairs,
        key=lambda pair: (
            _project_center(
                pair[1]["dot_center"],
                angle_deg=phrase_axis,
            )[0],
            pair[0],
        ),
    )
    if any(
        _axis_delta(member["axis_angle_deg"], phrase_axis) > 0.001
        for _digest_value, member in ordered_source_pairs
    ):
        return None, "window_axis_authority_mismatch"
    size_classes = sorted(set(size_class_by_source_digest.values()))
    if len(size_classes) > 1:
        if size_classes != sorted(GROUPED_SIZE_CLASSES):
            return None, "window_authority_ambiguous"
        return _resolve_grouped_pitch_window_authority(
            source_candidate_evidence=source_rows,
            ordered_source_pairs=ordered_source_pairs,
            size_class_by_source_digest=size_class_by_source_digest,
            rows_by_source_digest=rows_by_source_digest,
            page_index=page_index,
            phrase_id=phrase_id,
            phrase_axis_angle_deg=phrase_axis,
            phrase_axis_angle_source=phrase_axis_angle_source,
            phrase_primitive_ids=phrase_primitive_ids,
            page_context_manifest=page_context_manifest,
            phrase_items=phrase_items,
            verify_accounting_digests=verify_accounting_digests,
        )
    if any(
        any(
            member[field] != source[field]
            for field in (
                "char_height_pt",
                "char_width_pt",
                "pitch_pt",
            )
        )
        for _digest_value, member in ordered_source_pairs
    ):
        return None, "window_authority_ambiguous"
    representative_main, representative_cross = _project_center(
        source["dot_center"],
        angle_deg=phrase_axis,
    )
    for other_digest, member in ordered_source_pairs:
        if other_digest == source_digest:
            continue
        member_main, member_cross = _project_center(
            member["dot_center"],
            angle_deg=phrase_axis,
        )
        if abs(member_cross - representative_cross) > (
            source["char_height_pt"]
            * MULTI_ANCHOR_CROSS_TOLERANCE_HEIGHT
        ):
            return None, "window_authority_ambiguous"
        phase_steps = (
            (member_main - representative_main) / source["pitch_pt"]
        )
        if abs(phase_steps - round(phase_steps)) > (
            MULTI_ANCHOR_PHASE_TOLERANCE_PITCH
        ):
            return None, "window_phase_ambiguous"
    primitive_id_set = set(phrase_primitive_ids)
    if (
        any(
            member["source_primitive_id"] not in primitive_id_set
            for _digest_value, member in ordered_source_pairs
        )
        or len(primitive_id_set) != len(phrase_primitive_ids)
    ):
        return None, "window_authority_invalid"
    items_by_id: dict[str, dict[str, Any]] = {}
    for item in phrase_items:
        if not isinstance(item, dict):
            return None, "window_authority_invalid"
        item_id = item.get("item_id")
        if (
            type(item_id) is not str
            or item_id not in primitive_id_set
            or item_id in items_by_id
        ):
            return None, "window_authority_invalid"
        items_by_id[item_id] = item
    if set(items_by_id) != primitive_id_set:
        return None, "window_authority_invalid"
    if any(
        (
            items_by_id[member["source_primitive_id"]].get("op") != "qu"
            or not _item_points_match_quad(
                items_by_id[member["source_primitive_id"]].get("points"),
                member["raw_dot_quad"],
            )
        )
        for _digest_value, member in ordered_source_pairs
    ):
        return None, "window_authority_invalid"
    from vector_pitch_text_transform import build_phrase_primitive_manifest

    phrase_manifest = build_phrase_primitive_manifest(
        phrase_id=phrase_id,
        page_context_manifest=page_context_manifest,
        items=phrase_items,
        include_geometry=True,
    )
    if phrase_manifest is None:
        return None, "window_authority_invalid"
    directed_axis = directed_pitch_axis_angle(
        source["axis_angle_deg"]
    )
    if directed_axis is None:
        return None, "window_authority_invalid"
    radians = math.radians(directed_axis)
    main_x = round(math.cos(radians), 9)
    main_y = round(math.sin(radians), 9)
    cross_x = round(-main_y, 9)
    cross_y = round(main_x, 9)
    dot_x = float(source["dot_center"]["x"])
    dot_y = float(source["dot_center"]["y"])
    authority = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "producer": AUTHORITY_PRODUCER,
        "page_index": page_index,
        "phrase_id": phrase_id,
        "decimal_quad_id": source["decimal_quad_id"],
        "source_primitive_id": source["source_primitive_id"],
        "source_candidate_ids": sorted(set().union(*(
            candidate_ids_by_source_digest[digest_value]
            for digest_value, _member in ordered_source_pairs
        ))),
        "source_decimal_authority_digest": source[
            "source_decimal_authority_digest"
        ],
        "raw_dot_quad": source["raw_dot_quad"],
        "dot_center": source["dot_center"],
        "dot_short_edge_pt": source["dot_short_edge_pt"],
        "dot_long_edge_pt": source["dot_long_edge_pt"],
        "char_height_pt": source["char_height_pt"],
        "char_width_pt": source["char_width_pt"],
        "pitch_pt": source["pitch_pt"],
        "axis_angle_deg": source["axis_angle_deg"],
        "axis_angle_source": source["axis_angle_source"],
        "main_unit": {"x": main_x, "y": main_y},
        "cross_unit": {"x": cross_x, "y": cross_y},
        "dot_main_center": round(dot_x * main_x + dot_y * main_y, 6),
        "dot_cross_center": round(
            dot_x * cross_x + dot_y * cross_y,
            6,
        ),
        "integer_slot_quads": source["integer_slot_quads"],
        "fraction_slot_quads": source["fraction_slot_quads"],
        "derivation": source["derivation"],
        "source_geometry_digest": source["source_geometry_digest"],
        "group_member_source_digests": sorted(by_source_digest),
        "group_member_decimal_quad_ids": sorted(
            member["decimal_quad_id"]
            for _digest_value, member in ordered_source_pairs
        ),
        "group_member_source_primitive_ids": sorted(
            member["source_primitive_id"]
            for _digest_value, member in ordered_source_pairs
        ),
        "page_context_primitive_sha256": str(
            page_context_manifest["primitive_sha256"]
        ),
        "phrase_primitive_manifest_digest": phrase_manifest[
            "manifest_digest"
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    authority["authority_digest"] = _digest(authority)
    normalized = normalize_pitch_window_authority(authority)
    if normalized is None:
        return None, "window_authority_invalid"
    return normalized, None


def _resolve_grouped_pitch_window_authority(
    *,
    source_candidate_evidence: list[dict[str, Any]],
    ordered_source_pairs: list[tuple[str, dict[str, Any]]],
    size_class_by_source_digest: dict[str, str],
    rows_by_source_digest: dict[str, list[dict[str, Any]]],
    page_index: int,
    phrase_id: str,
    phrase_axis_angle_deg: float,
    phrase_axis_angle_source: str,
    phrase_primitive_ids: list[str],
    page_context_manifest: dict[str, Any],
    phrase_items: list[dict[str, Any]],
    verify_accounting_digests: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve one strict authority per signed size class plus one relation."""
    group_rows: dict[str, list[dict[str, Any]]] = {
        size_class: [] for size_class in GROUPED_SIZE_CLASSES
    }
    group_source_digests: dict[str, list[str]] = {
        size_class: [] for size_class in GROUPED_SIZE_CLASSES
    }
    for source_digest, _source in ordered_source_pairs:
        size_class = size_class_by_source_digest[source_digest]
        group_source_digests[size_class].append(source_digest)
        group_rows[size_class].extend(rows_by_source_digest[source_digest])

    groups = []
    for size_class in GROUPED_SIZE_CLASSES:
        window_authority, reason = resolve_pitch_window_authority(
            group_rows[size_class],
            page_index=page_index,
            phrase_id=phrase_id,
            phrase_axis_angle_deg=phrase_axis_angle_deg,
            phrase_axis_angle_source=phrase_axis_angle_source,
            phrase_primitive_ids=phrase_primitive_ids,
            page_context_manifest=page_context_manifest,
            phrase_items=phrase_items,
            verify_accounting_digests=verify_accounting_digests,
        )
        if window_authority is None:
            return None, reason or "window_authority_ambiguous"
        if window_authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
            return None, "window_authority_invalid"
        group = {
            "schema_version": GROUP_AUTHORITY_SCHEMA_VERSION,
            "producer": GROUP_AUTHORITY_PRODUCER,
            "size_class": size_class,
            "window_authority": window_authority,
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        group["group_digest"] = _digest(group)
        groups.append(group)

    carrier_payloads_by_source: dict[
        str,
        dict[tuple[Any, ...], dict[str, Any]],
    ] = {}
    for source_digest, source in ordered_source_pairs:
        carriers: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows_by_source_digest[source_digest]:
            carrier = _canonical_text_phrase_carrier(
                row,
                page_index=page_index,
                phrase_axis_angle_deg=phrase_axis_angle_deg,
                raw_dot_quad=source["raw_dot_quad"],
            )
            if carrier is None:
                continue
            key = (
                carrier["candidate_id"],
                tuple(carrier["source_region_ids"]),
                carrier["corridor_bbox"]["x"],
                carrier["corridor_bbox"]["y"],
                carrier["corridor_bbox"]["w"],
                carrier["corridor_bbox"]["h"],
                carrier["axis_angle_deg"],
            )
            prior = carriers.get(key)
            if prior is not None and prior != carrier:
                return None, "window_authority_ambiguous"
            carriers[key] = carrier
        if not carriers:
            return None, "window_authority_ambiguous"
        carrier_payloads_by_source[source_digest] = carriers

    common_by_group: dict[str, set[tuple[Any, ...]]] = {}
    for size_class in GROUPED_SIZE_CLASSES:
        member_sets = [
            set(carrier_payloads_by_source[source_digest])
            for source_digest in group_source_digests[size_class]
        ]
        if not member_sets:
            return None, "window_authority_ambiguous"
        common = set.intersection(*member_sets)
        if not common:
            return None, "window_authority_ambiguous"
        common_by_group[size_class] = common
    shared_carriers = set.intersection(*common_by_group.values())
    if len(shared_carriers) != 1:
        return None, "window_authority_ambiguous"
    carrier_key = next(iter(shared_carriers))
    carrier = carrier_payloads_by_source[
        group_source_digests["primary"][0]
    ][carrier_key]

    groups_by_class = {group["size_class"]: group for group in groups}
    primary_authority = groups_by_class["primary"]["window_authority"]
    tolerance_authority = groups_by_class["tolerance"]["window_authority"]
    primary_main = float(primary_authority["dot_main_center"])
    tolerance_main = float(tolerance_authority["dot_main_center"])
    if not primary_main < tolerance_main:
        return None, "window_authority_ambiguous"
    cross_offset = abs(
        float(primary_authority["dot_cross_center"])
        - float(tolerance_authority["dot_cross_center"])
    )
    cross_limit = max(
        float(primary_authority["char_height_pt"]),
        float(tolerance_authority["char_height_pt"]),
    ) * CROSS_GROUP_CROSS_TOLERANCE_HEIGHT
    if cross_offset > cross_limit + 1e-9:
        return None, "window_authority_ambiguous"

    relation = {
        "schema_version": CROSS_GROUP_RELATION_SCHEMA_VERSION,
        "producer": CROSS_GROUP_RELATION_PRODUCER,
        "primary_group_digest": groups_by_class["primary"]["group_digest"],
        "tolerance_group_digest": groups_by_class["tolerance"]["group_digest"],
        "carrier_candidate_id": carrier["candidate_id"],
        "carrier_source_region_ids": carrier["source_region_ids"],
        "carrier_corridor_bbox": carrier["corridor_bbox"],
        "carrier_axis_angle_deg": carrier["axis_angle_deg"],
        "primary_dot_main_center": round(primary_main, 6),
        "tolerance_dot_main_center": round(tolerance_main, 6),
        "primary_dot_cross_center": round(
            float(primary_authority["dot_cross_center"]),
            6,
        ),
        "tolerance_dot_cross_center": round(
            float(tolerance_authority["dot_cross_center"]),
            6,
        ),
        "cross_offset_pt": round(cross_offset, 6),
        "cross_limit_pt": round(cross_limit, 6),
        "phrase_primitive_manifest_digest": primary_authority[
            "phrase_primitive_manifest_digest"
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    relation["relation_digest"] = _digest(relation)

    group_authorities = [
        group["window_authority"] for group in groups
    ]
    authority = {
        "schema_version": AUTHORITY_BUNDLE_SCHEMA_VERSION,
        "producer": GROUP_AUTHORITY_PRODUCER,
        "page_index": page_index,
        "phrase_id": phrase_id,
        "axis_angle_deg": round(phrase_axis_angle_deg, 6),
        "axis_angle_source": phrase_axis_angle_source,
        "main_unit": primary_authority["main_unit"],
        "cross_unit": primary_authority["cross_unit"],
        "group_order": list(GROUPED_SIZE_CLASSES),
        "groups": groups,
        "cross_group_relation": relation,
        "source_candidate_ids": sorted(set().union(*(
            set(member["source_candidate_ids"])
            for member in group_authorities
        ))),
        "group_member_source_digests": sorted(set().union(*(
            set(member["group_member_source_digests"])
            for member in group_authorities
        ))),
        "group_member_decimal_quad_ids": sorted(set().union(*(
            set(member["group_member_decimal_quad_ids"])
            for member in group_authorities
        ))),
        "group_member_source_primitive_ids": sorted(set().union(*(
            set(member["group_member_source_primitive_ids"])
            for member in group_authorities
        ))),
        "page_context_primitive_sha256": primary_authority[
            "page_context_primitive_sha256"
        ],
        "phrase_primitive_manifest_digest": primary_authority[
            "phrase_primitive_manifest_digest"
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    authority["authority_digest"] = _digest(authority)
    normalized = normalize_pitch_window_authority(authority)
    if normalized is None:
        return None, "window_authority_invalid"
    return normalized, None


def normalize_pitch_window_authority(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") == AUTHORITY_BUNDLE_SCHEMA_VERSION:
        return _normalize_grouped_pitch_window_authority(value)
    return _normalize_pitch_window_authority_v1(value)


def _normalize_pitch_window_authority_v1(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    expected_keys = {
        "schema_version",
        "producer",
        "page_index",
        "phrase_id",
        "decimal_quad_id",
        "source_primitive_id",
        "source_candidate_ids",
        "source_decimal_authority_digest",
        "raw_dot_quad",
        "dot_center",
        "dot_short_edge_pt",
        "dot_long_edge_pt",
        "char_height_pt",
        "char_width_pt",
        "pitch_pt",
        "axis_angle_deg",
        "axis_angle_source",
        "main_unit",
        "cross_unit",
        "dot_main_center",
        "dot_cross_center",
        "integer_slot_quads",
        "fraction_slot_quads",
        "derivation",
        "source_geometry_digest",
        "group_member_source_digests",
        "group_member_decimal_quad_ids",
        "group_member_source_primitive_ids",
        "page_context_primitive_sha256",
        "phrase_primitive_manifest_digest",
        "consumer_allowed",
        "authority_digest",
    }
    if set(value) != expected_keys:
        return None
    if not (
        value.get("schema_version") == AUTHORITY_SCHEMA_VERSION
        and value.get("producer") == AUTHORITY_PRODUCER
        and type(value.get("page_index")) is int
        and value["page_index"] >= 0
        and type(value.get("phrase_id")) is str
        and bool(value["phrase_id"])
        and type(value.get("decimal_quad_id")) is str
        and bool(value["decimal_quad_id"])
        and type(value.get("source_primitive_id")) is str
        and bool(value["source_primitive_id"])
        and value.get("axis_angle_source") == AXIS_SOURCE
        and value.get("consumer_allowed") is False
        and _valid_sha256(value.get("source_decimal_authority_digest"))
        and _valid_sha256(value.get("source_geometry_digest"))
        and _valid_sha256(value.get("page_context_primitive_sha256"))
        and _valid_sha256(value.get("phrase_primitive_manifest_digest"))
        and _valid_sha256(value.get("authority_digest"))
    ):
        return None
    source_candidate_ids = _canonical_string_list(
        value.get("source_candidate_ids")
    )
    member_source_digests = _canonical_string_list(
        value.get("group_member_source_digests")
    )
    member_quad_ids = _canonical_string_list(
        value.get("group_member_decimal_quad_ids")
    )
    member_primitive_ids = _canonical_string_list(
        value.get("group_member_source_primitive_ids")
    )
    raw_quad = _canonical_quad(value.get("raw_dot_quad"))
    raw_axis = _quad_short_axis_angle(raw_quad)
    edge_lengths = _quad_edge_lengths(raw_quad)
    axis_angle = _canonical_axis_angle(value.get("axis_angle_deg"))
    dot_center = value.get("dot_center")
    integer_slot_quads = _canonical_quad_list(
        value.get("integer_slot_quads")
    )
    fraction_slot_quads = _canonical_quad_list(
        value.get("fraction_slot_quads")
    )
    derivation = _canonical_derivation(value.get("derivation"))
    main_unit = _canonical_unit(value.get("main_unit"))
    cross_unit = _canonical_unit(value.get("cross_unit"))
    dot_main_center = _strict_number(value.get("dot_main_center"))
    dot_cross_center = _strict_number(value.get("dot_cross_center"))
    numeric = {
        key: _strict_positive_number(value.get(key))
        for key in (
            "dot_short_edge_pt",
            "dot_long_edge_pt",
            "char_height_pt",
            "char_width_pt",
            "pitch_pt",
        )
    }
    if (
        source_candidate_ids is None
        or not source_candidate_ids
        or member_source_digests is None
        or not member_source_digests
        or any(not _valid_sha256(digest) for digest in member_source_digests)
        or member_quad_ids is None
        or member_primitive_ids is None
        or len(member_source_digests) != len(member_quad_ids)
        or len(member_source_digests) != len(member_primitive_ids)
        or raw_quad is None
        or raw_axis is None
        or edge_lengths is None
        or axis_angle is None
        or not isinstance(dot_center, dict)
        or set(dot_center) != {"x", "y"}
        or _strict_number(dot_center.get("x")) is None
        or _strict_number(dot_center.get("y")) is None
        or integer_slot_quads is None
        or fraction_slot_quads is None
        or derivation is None
        or main_unit is None
        or cross_unit is None
        or dot_main_center is None
        or dot_cross_center is None
        or any(number is None for number in numeric.values())
    ):
        return None
    canonical = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "producer": AUTHORITY_PRODUCER,
        "page_index": value["page_index"],
        "phrase_id": value["phrase_id"],
        "decimal_quad_id": value["decimal_quad_id"],
        "source_primitive_id": value["source_primitive_id"],
        "source_candidate_ids": source_candidate_ids,
        "source_decimal_authority_digest": value[
            "source_decimal_authority_digest"
        ],
        "raw_dot_quad": raw_quad,
        "dot_center": {
            "x": round(float(dot_center["x"]), 6),
            "y": round(float(dot_center["y"]), 6),
        },
        "dot_short_edge_pt": round(float(numeric["dot_short_edge_pt"]), 6),
        "dot_long_edge_pt": round(float(numeric["dot_long_edge_pt"]), 6),
        "char_height_pt": round(float(numeric["char_height_pt"]), 6),
        "char_width_pt": round(float(numeric["char_width_pt"]), 6),
        "pitch_pt": round(float(numeric["pitch_pt"]), 6),
        "axis_angle_deg": round(axis_angle, 6),
        "axis_angle_source": AXIS_SOURCE,
        "main_unit": main_unit,
        "cross_unit": cross_unit,
        "dot_main_center": round(dot_main_center, 6),
        "dot_cross_center": round(dot_cross_center, 6),
        "integer_slot_quads": integer_slot_quads,
        "fraction_slot_quads": fraction_slot_quads,
        "derivation": derivation,
        "source_geometry_digest": value["source_geometry_digest"],
        "group_member_source_digests": member_source_digests,
        "group_member_decimal_quad_ids": member_quad_ids,
        "group_member_source_primitive_ids": member_primitive_ids,
        "page_context_primitive_sha256": value[
            "page_context_primitive_sha256"
        ],
        "phrase_primitive_manifest_digest": value[
            "phrase_primitive_manifest_digest"
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
        "authority_digest": value["authority_digest"],
    }
    directed_axis = directed_pitch_axis_angle(
        canonical["axis_angle_deg"]
    )
    if directed_axis is None:
        return None
    radians = math.radians(directed_axis)
    expected_main = {
        "x": round(math.cos(radians), 9),
        "y": round(math.sin(radians), 9),
    }
    expected_cross = {
        "x": round(-expected_main["y"], 9),
        "y": round(expected_main["x"], 9),
    }
    dot_x = canonical["dot_center"]["x"]
    dot_y = canonical["dot_center"]["y"]
    computed_short, computed_long = edge_lengths
    if (
        canonical["decimal_quad_id"] not in member_quad_ids
        or canonical["source_primitive_id"] not in member_primitive_ids
        or _axis_delta(canonical["axis_angle_deg"], raw_axis)
        > AXIS_GEOMETRY_TOLERANCE_DEG
        or _relative_error(
            canonical["dot_short_edge_pt"],
            computed_short,
        )
        > GEOMETRY_RELATIVE_TOLERANCE
        or _relative_error(
            canonical["dot_long_edge_pt"],
            computed_long,
        )
        > GEOMETRY_RELATIVE_TOLERANCE
        or canonical["char_height_pt"]
        != round(
            derivation["cluster_long_edge_median_pt"]
            * derivation["char_height_factor"],
            6,
        )
        or canonical["char_width_pt"]
        != round(
            canonical["char_height_pt"]
            * derivation["char_width_factor"],
            6,
        )
        or canonical["pitch_pt"] != canonical["char_width_pt"]
        or canonical["main_unit"] != expected_main
        or canonical["cross_unit"] != expected_cross
        or canonical["dot_main_center"]
        != round(
            dot_x * expected_main["x"] + dot_y * expected_main["y"],
            6,
        )
        or canonical["dot_cross_center"]
        != round(
            dot_x * expected_cross["x"] + dot_y * expected_cross["y"],
            6,
        )
        or canonical["authority_digest"] != _digest({
            key: item
            for key, item in canonical.items()
            if key != "authority_digest"
        })
    ):
        return None
    return canonical


def _canonical_text_phrase_carrier(
    row: Any,
    *,
    page_index: int,
    phrase_axis_angle_deg: float,
    raw_dot_quad: Any,
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    candidate_id = row.get("candidate_id")
    source_region_ids = _canonical_string_list(row.get("source_region_ids"))
    corridor_bbox = _canonical_bbox_dict(row.get("corridor_bbox"))
    carrier_axis = _canonical_axis_angle(row.get("axis_angle_deg"))
    quad = _canonical_quad(raw_dot_quad)
    if not (
        row.get("schema_version")
        == "vector_phrase_source_candidate_evidence_v1"
        and row.get("source_schema_version")
        == "anchor_corridor_candidates_v1"
        and row.get("anchor_type") == "text_phrase_source"
        and row.get("reference_anchor_type") == "dot"
        and type(row.get("page_index")) is int
        and row["page_index"] == page_index
        and type(candidate_id) is str
        and bool(candidate_id)
        and source_region_ids is not None
        and bool(source_region_ids)
        and corridor_bbox is not None
        and carrier_axis is not None
        and _axis_delta(carrier_axis, phrase_axis_angle_deg) <= 0.001
        and row.get("axis_angle_source") in {
            "decimal_point_quad",
            "decimal_point_quad.short_axis",
            "oriented_quad:dot_axis_angle",
        }
        and row.get("axis_trusted") is True
        and row.get("axis_join_status") == "matched"
        and row.get("consumer_allowed") is False
        and quad is not None
        and _quad_inside_bbox(quad, corridor_bbox)
    ):
        return None
    return {
        "candidate_id": candidate_id,
        "source_region_ids": source_region_ids,
        "corridor_bbox": corridor_bbox,
        "axis_angle_deg": round(carrier_axis, 6),
    }


def _normalize_pitch_window_group_authority(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "producer",
        "size_class",
        "window_authority",
        "consumer_allowed",
        "group_digest",
    }:
        return None
    window_authority = _normalize_pitch_window_authority_v1(
        value.get("window_authority")
    )
    size_class = value.get("size_class")
    if not (
        value.get("schema_version") == GROUP_AUTHORITY_SCHEMA_VERSION
        and value.get("producer") == GROUP_AUTHORITY_PRODUCER
        and size_class in GROUPED_SIZE_CLASSES
        and value.get("consumer_allowed") is False
        and _valid_sha256(value.get("group_digest"))
        and window_authority is not None
        and window_authority["derivation"]["cluster_role"] == size_class
    ):
        return None
    canonical = {
        "schema_version": GROUP_AUTHORITY_SCHEMA_VERSION,
        "producer": GROUP_AUTHORITY_PRODUCER,
        "size_class": size_class,
        "window_authority": window_authority,
        "consumer_allowed": CONSUMER_ALLOWED,
        "group_digest": value["group_digest"],
    }
    if canonical["group_digest"] != _digest({
        key: item
        for key, item in canonical.items()
        if key != "group_digest"
    }):
        return None
    return canonical


def _normalize_grouped_pitch_window_authority(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "producer",
        "page_index",
        "phrase_id",
        "axis_angle_deg",
        "axis_angle_source",
        "main_unit",
        "cross_unit",
        "group_order",
        "groups",
        "cross_group_relation",
        "source_candidate_ids",
        "group_member_source_digests",
        "group_member_decimal_quad_ids",
        "group_member_source_primitive_ids",
        "page_context_primitive_sha256",
        "phrase_primitive_manifest_digest",
        "consumer_allowed",
        "authority_digest",
    }:
        return None
    axis_angle = _canonical_axis_angle(value.get("axis_angle_deg"))
    main_unit = _canonical_unit(value.get("main_unit"))
    cross_unit = _canonical_unit(value.get("cross_unit"))
    source_candidate_ids = _canonical_string_list(
        value.get("source_candidate_ids")
    )
    member_source_digests = _canonical_string_list(
        value.get("group_member_source_digests")
    )
    member_quad_ids = _canonical_string_list(
        value.get("group_member_decimal_quad_ids")
    )
    member_primitive_ids = _canonical_string_list(
        value.get("group_member_source_primitive_ids")
    )
    raw_groups = value.get("groups")
    groups = (
        [_normalize_pitch_window_group_authority(group) for group in raw_groups]
        if isinstance(raw_groups, list)
        else []
    )
    if not (
        value.get("schema_version") == AUTHORITY_BUNDLE_SCHEMA_VERSION
        and value.get("producer") == GROUP_AUTHORITY_PRODUCER
        and type(value.get("page_index")) is int
        and value["page_index"] >= 0
        and type(value.get("phrase_id")) is str
        and bool(value["phrase_id"])
        and axis_angle is not None
        and value.get("axis_angle_source") in {
            "decimal_point_quad",
            "decimal_point_quad.short_axis",
            "oriented_quad:dot_axis_angle",
        }
        and main_unit is not None
        and cross_unit is not None
        and value.get("group_order") == list(GROUPED_SIZE_CLASSES)
        and len(groups) == len(GROUPED_SIZE_CLASSES)
        and all(group is not None for group in groups)
        and [group["size_class"] for group in groups]
        == list(GROUPED_SIZE_CLASSES)
        and source_candidate_ids is not None
        and bool(source_candidate_ids)
        and member_source_digests is not None
        and bool(member_source_digests)
        and all(_valid_sha256(item) for item in member_source_digests)
        and member_quad_ids is not None
        and bool(member_quad_ids)
        and member_primitive_ids is not None
        and bool(member_primitive_ids)
        and _valid_sha256(value.get("page_context_primitive_sha256"))
        and _valid_sha256(value.get("phrase_primitive_manifest_digest"))
        and value.get("consumer_allowed") is False
        and _valid_sha256(value.get("authority_digest"))
    ):
        return None
    normalized_groups = [group for group in groups if group is not None]
    group_authorities = [
        group["window_authority"] for group in normalized_groups
    ]
    primary = group_authorities[0]
    if any(
        authority["page_index"] != value["page_index"]
        or authority["phrase_id"] != value["phrase_id"]
        or _axis_delta(authority["axis_angle_deg"], axis_angle) > 0.001
        or authority["main_unit"] != main_unit
        or authority["cross_unit"] != cross_unit
        or authority["page_context_primitive_sha256"]
        != value["page_context_primitive_sha256"]
        or authority["phrase_primitive_manifest_digest"]
        != value["phrase_primitive_manifest_digest"]
        for authority in group_authorities
    ):
        return None
    expected_candidates = sorted(set().union(*(
        set(authority["source_candidate_ids"])
        for authority in group_authorities
    )))
    expected_source_digests = sorted(set().union(*(
        set(authority["group_member_source_digests"])
        for authority in group_authorities
    )))
    expected_quad_ids = sorted(set().union(*(
        set(authority["group_member_decimal_quad_ids"])
        for authority in group_authorities
    )))
    expected_primitive_ids = sorted(set().union(*(
        set(authority["group_member_source_primitive_ids"])
        for authority in group_authorities
    )))
    if (
        source_candidate_ids != expected_candidates
        or member_source_digests != expected_source_digests
        or member_quad_ids != expected_quad_ids
        or member_primitive_ids != expected_primitive_ids
        or main_unit != primary["main_unit"]
        or cross_unit != primary["cross_unit"]
    ):
        return None
    relation = _normalize_cross_group_relation(
        value.get("cross_group_relation"),
        groups=normalized_groups,
        phrase_primitive_manifest_digest=value[
            "phrase_primitive_manifest_digest"
        ],
        axis_angle_deg=axis_angle,
    )
    if relation is None:
        return None
    canonical = {
        "schema_version": AUTHORITY_BUNDLE_SCHEMA_VERSION,
        "producer": GROUP_AUTHORITY_PRODUCER,
        "page_index": value["page_index"],
        "phrase_id": value["phrase_id"],
        "axis_angle_deg": round(axis_angle, 6),
        "axis_angle_source": value["axis_angle_source"],
        "main_unit": main_unit,
        "cross_unit": cross_unit,
        "group_order": list(GROUPED_SIZE_CLASSES),
        "groups": normalized_groups,
        "cross_group_relation": relation,
        "source_candidate_ids": source_candidate_ids,
        "group_member_source_digests": member_source_digests,
        "group_member_decimal_quad_ids": member_quad_ids,
        "group_member_source_primitive_ids": member_primitive_ids,
        "page_context_primitive_sha256": value[
            "page_context_primitive_sha256"
        ],
        "phrase_primitive_manifest_digest": value[
            "phrase_primitive_manifest_digest"
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
        "authority_digest": value["authority_digest"],
    }
    if canonical["authority_digest"] != _digest({
        key: item
        for key, item in canonical.items()
        if key != "authority_digest"
    }):
        return None
    return canonical


def _normalize_cross_group_relation(
    value: Any,
    *,
    groups: list[dict[str, Any]],
    phrase_primitive_manifest_digest: str,
    axis_angle_deg: float,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "producer",
        "primary_group_digest",
        "tolerance_group_digest",
        "carrier_candidate_id",
        "carrier_source_region_ids",
        "carrier_corridor_bbox",
        "carrier_axis_angle_deg",
        "primary_dot_main_center",
        "tolerance_dot_main_center",
        "primary_dot_cross_center",
        "tolerance_dot_cross_center",
        "cross_offset_pt",
        "cross_limit_pt",
        "phrase_primitive_manifest_digest",
        "consumer_allowed",
        "relation_digest",
    }:
        return None
    groups_by_class = {group["size_class"]: group for group in groups}
    if set(groups_by_class) != set(GROUPED_SIZE_CLASSES):
        return None
    primary = groups_by_class["primary"]["window_authority"]
    tolerance = groups_by_class["tolerance"]["window_authority"]
    common_candidate_ids = (
        set(primary["source_candidate_ids"])
        & set(tolerance["source_candidate_ids"])
    )
    carrier_candidate_id = value.get("carrier_candidate_id")
    source_region_ids = _canonical_string_list(
        value.get("carrier_source_region_ids")
    )
    corridor_bbox = _canonical_bbox_dict(value.get("carrier_corridor_bbox"))
    carrier_axis = _canonical_axis_angle(value.get("carrier_axis_angle_deg"))
    numeric = {
        key: _strict_number(value.get(key))
        for key in (
            "primary_dot_main_center",
            "tolerance_dot_main_center",
            "primary_dot_cross_center",
            "tolerance_dot_cross_center",
            "cross_offset_pt",
            "cross_limit_pt",
        )
    }
    if not (
        value.get("schema_version") == CROSS_GROUP_RELATION_SCHEMA_VERSION
        and value.get("producer") == CROSS_GROUP_RELATION_PRODUCER
        and value.get("primary_group_digest")
        == groups_by_class["primary"]["group_digest"]
        and value.get("tolerance_group_digest")
        == groups_by_class["tolerance"]["group_digest"]
        and type(carrier_candidate_id) is str
        and bool(carrier_candidate_id)
        and carrier_candidate_id in common_candidate_ids
        and source_region_ids is not None
        and bool(source_region_ids)
        and corridor_bbox is not None
        and carrier_axis is not None
        and _axis_delta(carrier_axis, axis_angle_deg) <= 0.001
        and all(number is not None for number in numeric.values())
        and value.get("phrase_primitive_manifest_digest")
        == phrase_primitive_manifest_digest
        and value.get("consumer_allowed") is False
        and _valid_sha256(value.get("relation_digest"))
    ):
        return None
    primary_main = float(primary["dot_main_center"])
    tolerance_main = float(tolerance["dot_main_center"])
    primary_cross = float(primary["dot_cross_center"])
    tolerance_cross = float(tolerance["dot_cross_center"])
    cross_offset = abs(primary_cross - tolerance_cross)
    cross_limit = max(
        float(primary["char_height_pt"]),
        float(tolerance["char_height_pt"]),
    ) * CROSS_GROUP_CROSS_TOLERANCE_HEIGHT
    if not (
        primary_main < tolerance_main
        and round(float(numeric["primary_dot_main_center"]), 6)
        == round(primary_main, 6)
        and round(float(numeric["tolerance_dot_main_center"]), 6)
        == round(tolerance_main, 6)
        and round(float(numeric["primary_dot_cross_center"]), 6)
        == round(primary_cross, 6)
        and round(float(numeric["tolerance_dot_cross_center"]), 6)
        == round(tolerance_cross, 6)
        and round(float(numeric["cross_offset_pt"]), 6)
        == round(cross_offset, 6)
        and round(float(numeric["cross_limit_pt"]), 6)
        == round(cross_limit, 6)
        and cross_offset <= cross_limit + 1e-9
        and _quad_inside_bbox(primary["raw_dot_quad"], corridor_bbox)
        and _quad_inside_bbox(tolerance["raw_dot_quad"], corridor_bbox)
    ):
        return None
    canonical = {
        "schema_version": CROSS_GROUP_RELATION_SCHEMA_VERSION,
        "producer": CROSS_GROUP_RELATION_PRODUCER,
        "primary_group_digest": value["primary_group_digest"],
        "tolerance_group_digest": value["tolerance_group_digest"],
        "carrier_candidate_id": carrier_candidate_id,
        "carrier_source_region_ids": source_region_ids,
        "carrier_corridor_bbox": corridor_bbox,
        "carrier_axis_angle_deg": round(carrier_axis, 6),
        "primary_dot_main_center": round(primary_main, 6),
        "tolerance_dot_main_center": round(tolerance_main, 6),
        "primary_dot_cross_center": round(primary_cross, 6),
        "tolerance_dot_cross_center": round(tolerance_cross, 6),
        "cross_offset_pt": round(cross_offset, 6),
        "cross_limit_pt": round(cross_limit, 6),
        "phrase_primitive_manifest_digest": phrase_primitive_manifest_digest,
        "consumer_allowed": CONSUMER_ALLOWED,
        "relation_digest": value["relation_digest"],
    }
    if canonical["relation_digest"] != _digest({
        key: item
        for key, item in canonical.items()
        if key != "relation_digest"
    }):
        return None
    return canonical


def _canonical_bbox_dict(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != {"x", "y", "w", "h"}:
        return None
    numeric = {key: _strict_number(value.get(key)) for key in value}
    if (
        any(number is None for number in numeric.values())
        or float(numeric["w"]) <= 0.0
        or float(numeric["h"]) <= 0.0
    ):
        return None
    return {
        key: round(float(numeric[key]), 6)
        for key in ("x", "y", "w", "h")
    }


def _quad_inside_bbox(
    quad: Any,
    bbox: dict[str, float],
) -> bool:
    canonical_quad = _canonical_quad(quad)
    if canonical_quad is None:
        return False
    min_x = bbox["x"] - DECIMAL_POINT_COORDINATE_QUANTIZATION_PT
    min_y = bbox["y"] - DECIMAL_POINT_COORDINATE_QUANTIZATION_PT
    max_x = bbox["x"] + bbox["w"] + DECIMAL_POINT_COORDINATE_QUANTIZATION_PT
    max_y = bbox["y"] + bbox["h"] + DECIMAL_POINT_COORDINATE_QUANTIZATION_PT
    return all(
        min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
        for point in canonical_quad
    )


def build_pitch_window_source(
    decimal_quad: dict[str, Any],
    decimal_corridor: dict[str, Any],
    decimal_anchor_authority: dict[str, Any],
    *,
    page_index: int,
    repeated_size_cohort_proof: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build one context-free window source from an exact direct dot join."""
    if type(page_index) is not int or page_index < 0:
        return None
    if not (
        isinstance(decimal_quad, dict)
        and decimal_quad.get("schema_version") == DECIMAL_QUAD_SCHEMA_VERSION
        and decimal_quad.get("source") == DECIMAL_QUAD_SOURCE
        and decimal_quad.get("consumer_allowed") is False
        and isinstance(decimal_corridor, dict)
        and decimal_corridor.get("schema_version")
        == DECIMAL_CORRIDOR_SCHEMA_VERSION
        and decimal_corridor.get("source") == DECIMAL_CORRIDOR_SOURCE
        and decimal_corridor.get("consumer_allowed") is False
        and type(decimal_corridor.get("page_index")) is int
        and decimal_corridor.get("page_index") == page_index
    ):
        return None
    quad_page = decimal_quad.get("page_index")
    if (
        "page_index" in decimal_quad
        and (type(quad_page) is not int or quad_page != page_index)
    ):
        return None
    quad_id = decimal_quad.get("id")
    if (
        type(quad_id) is not str
        or not quad_id
        or decimal_corridor.get("decimal_quad_id") != quad_id
    ):
        return None
    quad_ordinal_match = re.fullmatch(r"dot_([0-9]+)", quad_id)
    corridor_quad_index = decimal_corridor.get("decimal_quad_index")
    if (
        quad_ordinal_match is None
        or quad_id != f"dot_{int(quad_ordinal_match.group(1)):04d}"
        or type(corridor_quad_index) is not int
        or corridor_quad_index != int(quad_ordinal_match.group(1))
    ):
        return None
    detail = decimal_quad.get("detail")
    short_axis = decimal_quad.get("short_axis")
    if not isinstance(detail, dict) or not isinstance(short_axis, dict):
        return None
    if not (
        type(detail.get("drawing_order")) is int
        and detail["drawing_order"] >= 0
        and type(detail.get("item_index")) is int
        and detail["item_index"] >= 0
    ):
        return None
    raw_quad = _canonical_quad(decimal_quad.get("quad"))
    edge_lengths = _quad_edge_lengths(raw_quad)
    raw_quad_axis = _quad_short_axis_angle(raw_quad)
    if (
        raw_quad is None
        or edge_lengths is None
        or raw_quad_axis is None
    ):
        return None
    dot_short_edge, dot_long_edge = edge_lengths
    declared_short = _strict_positive_number(detail.get("short"))
    declared_long = _strict_positive_number(detail.get("long"))
    axis_angle = _canonical_axis_angle(short_axis.get("angle_deg"))
    corridor_axis = _canonical_axis_angle(
        decimal_corridor.get("baseline_angle_deg")
    )
    if (
        declared_short is None
        or declared_long is None
        or axis_angle is None
        or corridor_axis is None
        or _relative_error(declared_short, dot_short_edge)
        > GEOMETRY_RELATIVE_TOLERANCE
        or _relative_error(declared_long, dot_long_edge)
        > GEOMETRY_RELATIVE_TOLERANCE
        or _axis_delta(axis_angle, raw_quad_axis)
        > AXIS_GEOMETRY_TOLERANCE_DEG
        or _axis_delta(axis_angle, corridor_axis) > 0.001
    ):
        return None
    cluster_membership = decimal_corridor.get("cluster_membership")
    cluster_id = decimal_corridor.get("char_height_cluster_id")
    size_class = decimal_corridor.get("size_class")
    if (
        cluster_membership != "direct_member_quad_id"
        or type(cluster_id) is not str
        or not cluster_id
        or type(size_class) is not str
        or not size_class
    ):
        return None
    cluster_long_edge = _strict_positive_number(
        decimal_corridor.get("cluster_long_edge_median")
    )
    char_height = _strict_positive_number(
        decimal_corridor.get("char_height")
    )
    char_width = _strict_positive_number(
        decimal_corridor.get("char_width")
    )
    if (
        cluster_long_edge is None
        or char_height is None
        or char_width is None
        or round(cluster_long_edge * DEFAULT_K1, 6)
        != round(char_height, 6)
        or round(char_height * DEFAULT_K2, 6) != round(char_width, 6)
    ):
        return None
    computed_center = _quad_center(raw_quad)
    declared_center = decimal_quad.get("center")
    if not (
        isinstance(declared_center, dict)
        and set(declared_center) == {"x", "y"}
        and (declared_center_x := _strict_number(
            declared_center.get("x")
        ))
        is not None
        and (declared_center_y := _strict_number(
            declared_center.get("y")
        ))
        is not None
        and abs(declared_center_x - computed_center[0])
        <= DECIMAL_POINT_COORDINATE_QUANTIZATION_PT + 1e-9
        and abs(declared_center_y - computed_center[1])
        <= DECIMAL_POINT_COORDINATE_QUANTIZATION_PT + 1e-9
    ):
        return None
    center = (declared_center_x, declared_center_y)
    integer_slot_quads = _corridor_slot_quads(
        decimal_corridor,
        rows_key="integer_slots",
        dot_center=center,
        axis_angle_deg=axis_angle,
        char_width=char_width,
        char_height=char_height,
        direction=-1.0,
    )
    fraction_slot_quads = _corridor_slot_quads(
        decimal_corridor,
        rows_key="fraction_slots",
        dot_center=center,
        axis_angle_deg=axis_angle,
        char_width=char_width,
        char_height=char_height,
        direction=1.0,
    )
    if integer_slot_quads is None or fraction_slot_quads is None:
        return None
    directed_axis = directed_pitch_axis_angle(axis_angle)
    if directed_axis is None:
        return None
    if directed_axis != axis_angle:
        integer_slot_quads, fraction_slot_quads = (
            fraction_slot_quads,
            integer_slot_quads,
        )
    semantic_authority = normalize_decimal_anchor_authority(
        decimal_anchor_authority
    )
    # A v2 authority has already validated and bound its cohort proof.  The
    # optional sibling copy is producer accounting, not additional geometry.
    expected_primitive_id = (
        f"p{page_index + 1:03d}_"
        f"d{detail['drawing_order']:06d}_"
        f"i{detail['item_index']:04d}_qu"
    )
    if (
        semantic_authority is None
        or semantic_authority["page_index"] != page_index
        or semantic_authority["decimal_quad_id"] != quad_id
        or semantic_authority["source_primitive_id"] != expected_primitive_id
        or semantic_authority["drawing_order"] != detail["drawing_order"]
        or semantic_authority["item_index"] != detail["item_index"]
        or semantic_authority["size_class"] != size_class
        or semantic_authority["char_height_cluster_id"] != cluster_id
        or semantic_authority["cluster_membership"]
        != cluster_membership
        or semantic_authority["dot_long_edge_pt"]
        != round(declared_long, 6)
        or semantic_authority["cluster_long_edge_median_pt"]
        != round(cluster_long_edge, 6)
        or _axis_delta(
            semantic_authority["baseline_angle_deg"],
            axis_angle,
        )
        > 0.001
    ):
        return None
    source_geometry = {
        "page_index": page_index,
        "decimal_quad_id": quad_id,
        "source_primitive_id": expected_primitive_id,
        "raw_dot_quad": raw_quad,
        "dot_center": {
            "x": round(center[0], 6),
            "y": round(center[1], 6),
        },
        "axis_angle_deg": axis_angle,
    }
    source = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "producer": SOURCE_PRODUCER,
        "page_index": page_index,
        "decimal_quad_id": quad_id,
        "source_primitive_id": expected_primitive_id,
        "drawing_order": detail["drawing_order"],
        "item_index": detail["item_index"],
        "source_decimal_authority_digest": semantic_authority[
            "authority_digest"
        ],
        "raw_dot_quad": raw_quad,
        "dot_center": {
            "x": round(center[0], 6),
            "y": round(center[1], 6),
        },
        "dot_short_edge_pt": round(dot_short_edge, 6),
        "dot_long_edge_pt": round(dot_long_edge, 6),
        "char_height_pt": round(char_height, 6),
        "char_width_pt": round(char_width, 6),
        "pitch_pt": round(char_width, 6),
        "axis_angle_deg": round(axis_angle, 6),
        "axis_angle_source": AXIS_SOURCE,
        "integer_slot_quads": integer_slot_quads,
        "fraction_slot_quads": fraction_slot_quads,
        "derivation": {
            "quad_source": DECIMAL_QUAD_SOURCE,
            "corridor_source": DECIMAL_CORRIDOR_SOURCE,
            "cluster_id": cluster_id,
            "cluster_role": size_class,
            "cluster_membership": cluster_membership,
            "cluster_long_edge_median_pt": round(cluster_long_edge, 6),
            "char_height_factor": float(DEFAULT_K1),
            "char_width_factor": float(DEFAULT_K2),
        },
        "source_geometry_digest": _digest(source_geometry),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    source["source_digest"] = _digest(source)
    return normalize_pitch_window_source(source)


def normalize_pitch_window_source(
    value: Any,
    *,
    verify_accounting_digests: bool = True,
) -> dict[str, Any] | None:
    """Canonicalize one source, optionally treating its digests as advisory."""
    if not isinstance(value, dict):
        return None
    expected_keys = {
        "schema_version",
        "producer",
        "page_index",
        "decimal_quad_id",
        "source_primitive_id",
        "drawing_order",
        "item_index",
        "source_decimal_authority_digest",
        "raw_dot_quad",
        "dot_center",
        "dot_short_edge_pt",
        "dot_long_edge_pt",
        "char_height_pt",
        "char_width_pt",
        "pitch_pt",
        "axis_angle_deg",
        "axis_angle_source",
        "integer_slot_quads",
        "fraction_slot_quads",
        "derivation",
        "source_geometry_digest",
        "consumer_allowed",
        "source_digest",
    }
    accounting_keys = {
        "source_decimal_authority_digest",
        "source_geometry_digest",
        "source_digest",
    }
    if (
        verify_accounting_digests
        and set(value) != expected_keys
    ) or (
        not verify_accounting_digests
        and not (
            expected_keys - accounting_keys
        ).issubset(value)
    ) or set(value) - expected_keys:
        return None
    if not (
        value.get("schema_version") == SOURCE_SCHEMA_VERSION
        and value.get("producer") == SOURCE_PRODUCER
        and type(value.get("page_index")) is int
        and value["page_index"] >= 0
        and type(value.get("decimal_quad_id")) is str
        and bool(value["decimal_quad_id"])
        and type(value.get("source_primitive_id")) is str
        and type(value.get("drawing_order")) is int
        and value["drawing_order"] >= 0
        and type(value.get("item_index")) is int
        and value["item_index"] >= 0
        and (
            not verify_accounting_digests
            or type(value.get("source_decimal_authority_digest")) is str
        )
        and (
            not verify_accounting_digests
            or type(value.get("source_geometry_digest")) is str
        )
        and (
            not verify_accounting_digests
            or type(value.get("source_digest")) is str
        )
        and value.get("axis_angle_source") == AXIS_SOURCE
        and value.get("consumer_allowed") is False
    ):
        return None
    raw_quad = _canonical_quad(value.get("raw_dot_quad"))
    edge_lengths = _quad_edge_lengths(raw_quad)
    raw_quad_axis = _quad_short_axis_angle(raw_quad)
    dot_center = value.get("dot_center")
    axis_angle = _canonical_axis_angle(value.get("axis_angle_deg"))
    numeric = {
        key: _strict_positive_number(value.get(key))
        for key in (
            "dot_short_edge_pt",
            "dot_long_edge_pt",
            "char_height_pt",
            "char_width_pt",
            "pitch_pt",
        )
    }
    if (
        raw_quad is None
        or edge_lengths is None
        or raw_quad_axis is None
        or axis_angle is None
        or not isinstance(dot_center, dict)
        or set(dot_center) != {"x", "y"}
        or _strict_number(dot_center.get("x")) is None
        or _strict_number(dot_center.get("y")) is None
        or any(number is None for number in numeric.values())
    ):
        return None
    integer_slot_quads = _canonical_quad_list(
        value.get("integer_slot_quads")
    )
    fraction_slot_quads = _canonical_quad_list(
        value.get("fraction_slot_quads")
    )
    derivation = _canonical_derivation(value.get("derivation"))
    if (
        integer_slot_quads is None
        or fraction_slot_quads is None
        or derivation is None
    ):
        return None
    canonical = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "producer": SOURCE_PRODUCER,
        "page_index": value["page_index"],
        "decimal_quad_id": value["decimal_quad_id"],
        "source_primitive_id": value["source_primitive_id"],
        "drawing_order": value["drawing_order"],
        "item_index": value["item_index"],
        "source_decimal_authority_digest": str(
            value.get("source_decimal_authority_digest") or ""
        ),
        "raw_dot_quad": raw_quad,
        "dot_center": {
            "x": round(float(dot_center["x"]), 6),
            "y": round(float(dot_center["y"]), 6),
        },
        "dot_short_edge_pt": round(float(numeric["dot_short_edge_pt"]), 6),
        "dot_long_edge_pt": round(float(numeric["dot_long_edge_pt"]), 6),
        "char_height_pt": round(float(numeric["char_height_pt"]), 6),
        "char_width_pt": round(float(numeric["char_width_pt"]), 6),
        "pitch_pt": round(float(numeric["pitch_pt"]), 6),
        "axis_angle_deg": round(axis_angle, 6),
        "axis_angle_source": AXIS_SOURCE,
        "integer_slot_quads": integer_slot_quads,
        "fraction_slot_quads": fraction_slot_quads,
        "derivation": derivation,
        "source_geometry_digest": str(
            value.get("source_geometry_digest") or ""
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
        "source_digest": str(value.get("source_digest") or ""),
    }
    expected_primitive_id = (
        f"p{canonical['page_index'] + 1:03d}_"
        f"d{canonical['drawing_order']:06d}_"
        f"i{canonical['item_index']:04d}_qu"
    )
    computed_short, computed_long = edge_lengths
    computed_center = _quad_center(raw_quad)
    source_geometry = {
        "page_index": canonical["page_index"],
        "decimal_quad_id": canonical["decimal_quad_id"],
        "source_primitive_id": canonical["source_primitive_id"],
        "raw_dot_quad": raw_quad,
        "dot_center": canonical["dot_center"],
        "axis_angle_deg": canonical["axis_angle_deg"],
    }
    computed_geometry_digest = _digest(source_geometry)
    if not verify_accounting_digests:
        canonical["source_geometry_digest"] = computed_geometry_digest
        canonical["source_digest"] = _digest({
            key: item
            for key, item in canonical.items()
            if key != "source_digest"
        })
    without_digest = {
        key: item
        for key, item in canonical.items()
        if key != "source_digest"
    }
    if (
        canonical["source_primitive_id"] != expected_primitive_id
        or _relative_error(
            canonical["dot_short_edge_pt"],
            computed_short,
        )
        > GEOMETRY_RELATIVE_TOLERANCE
        or _relative_error(
            canonical["dot_long_edge_pt"],
            computed_long,
        )
        > GEOMETRY_RELATIVE_TOLERANCE
        or _axis_delta(canonical["axis_angle_deg"], raw_quad_axis)
        > AXIS_GEOMETRY_TOLERANCE_DEG
        or abs(canonical["dot_center"]["x"] - computed_center[0])
        > DECIMAL_POINT_COORDINATE_QUANTIZATION_PT + 1e-9
        or abs(canonical["dot_center"]["y"] - computed_center[1])
        > DECIMAL_POINT_COORDINATE_QUANTIZATION_PT + 1e-9
        or canonical["char_height_pt"]
        != round(
            derivation["cluster_long_edge_median_pt"]
            * derivation["char_height_factor"],
            6,
        )
        or canonical["char_width_pt"]
        != round(
            canonical["char_height_pt"]
            * derivation["char_width_factor"],
            6,
        )
        or canonical["pitch_pt"] != canonical["char_width_pt"]
        or (
            verify_accounting_digests
            and canonical["source_geometry_digest"]
            != computed_geometry_digest
        )
        or (
            verify_accounting_digests
            and canonical["source_digest"] != _digest(without_digest)
        )
    ):
        return None
    return canonical


def _corridor_slot_quads(
    corridor: dict[str, Any],
    *,
    rows_key: str,
    dot_center: tuple[float, float],
    axis_angle_deg: float,
    char_width: float,
    char_height: float,
    direction: float,
) -> list[list[list[float]]] | None:
    rows = corridor.get(rows_key)
    if not isinstance(rows, list):
        return None
    radians = math.radians(axis_angle_deg)
    main = (math.cos(radians), math.sin(radians))
    quads: list[list[list[float]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        expected_center = (
            dot_center[0]
            + direction * char_width * (index + 1) * main[0],
            dot_center[1]
            + direction * char_width * (index + 1) * main[1],
        )
        expected_quad = _oriented_rect(
            expected_center,
            width=char_width,
            height=char_height,
            axis_angle_deg=axis_angle_deg,
        )
        quads.append(expected_quad)
    return quads


def _oriented_rect(
    center: tuple[float, float],
    *,
    width: float,
    height: float,
    axis_angle_deg: float,
) -> list[list[float]]:
    radians = math.radians(axis_angle_deg)
    ux, uy = math.cos(radians), math.sin(radians)
    vx, vy = -uy, ux
    half_width = width / 2.0
    half_height = height / 2.0
    cx, cy = center
    return [
        [
            round(cx - ux * half_width - vx * half_height, 6),
            round(cy - uy * half_width - vy * half_height, 6),
        ],
        [
            round(cx + ux * half_width - vx * half_height, 6),
            round(cy + uy * half_width - vy * half_height, 6),
        ],
        [
            round(cx + ux * half_width + vx * half_height, 6),
            round(cy + uy * half_width + vy * half_height, 6),
        ],
        [
            round(cx - ux * half_width + vx * half_height, 6),
            round(cy - uy * half_width + vy * half_height, 6),
        ],
    ]


def _canonical_quad_list(value: Any) -> list[list[list[float]]] | None:
    if not isinstance(value, list):
        return None
    quads: list[list[list[float]]] = []
    for raw_quad in value:
        quad = _canonical_quad(raw_quad)
        if quad is None:
            return None
        quads.append(quad)
    return quads


def _item_points_match_quad(
    value: Any,
    expected_quad: list[list[float]],
) -> bool:
    if not isinstance(value, list):
        return False
    raw_points = list(value)
    if (
        len(raw_points) == 5
        and isinstance(raw_points[0], (list, tuple))
        and isinstance(raw_points[-1], (list, tuple))
        and len(raw_points[0]) == 2
        and len(raw_points[-1]) == 2
        and all(
            _strict_number(raw_points[0][coordinate]) is not None
            and _strict_number(raw_points[-1][coordinate]) is not None
            and abs(
                float(raw_points[0][coordinate])
                - float(raw_points[-1][coordinate])
            )
            <= 1e-6
            for coordinate in (0, 1)
        )
    ):
        raw_points = raw_points[:-1]
    actual_quad = _canonical_quad(raw_points)
    if actual_quad is None:
        return False
    unmatched = [tuple(point) for point in expected_quad]
    for actual in actual_quad:
        candidates = [
            (
                max(
                    abs(actual[coordinate] - expected[coordinate])
                    for coordinate in (0, 1)
                ),
                index,
            )
            for index, expected in enumerate(unmatched)
        ]
        if not candidates:
            return False
        distance, index = min(candidates)
        if distance > DECIMAL_POINT_COORDINATE_QUANTIZATION_PT + 1e-9:
            return False
        unmatched.pop(index)
    return not unmatched


def _valid_primitive_ids(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(type(item) is str and item for item in value)
        and value == sorted(value)
        and len(value) == len(set(value))
    )


def _valid_page_context_manifest(
    value: Any,
    *,
    page_index: int,
    minimum_primitive_count: int,
) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version")
        == "vector_page_context_manifest_v1"
        and type(value.get("page_num")) is int
        and value["page_num"] == page_index + 1
        and type(value.get("primitive_count")) is int
        and value["primitive_count"] >= minimum_primitive_count
        and _valid_sha256(value.get("primitive_sha256"))
        and value.get("consumer_allowed") is False
    )


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _canonical_string_list(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
    ):
        return None
    canonical = sorted(set(value))
    return canonical if value == canonical else None


def _canonical_unit(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        return None
    x = _strict_number(value.get("x"))
    y = _strict_number(value.get("y"))
    if x is None or y is None:
        return None
    return {"x": round(x, 9), "y": round(y, 9)}


def _canonical_quad(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x = _strict_number(point[0])
        y = _strict_number(point[1])
        if x is None or y is None:
            return None
        points.append([round(x, 6), round(y, 6)])
    return points


def _quad_edge_lengths(
    quad: list[list[float]] | None,
) -> tuple[float, float] | None:
    if quad is None:
        return None
    lengths = [
        math.hypot(
            quad[(index + 1) % 4][0] - quad[index][0],
            quad[(index + 1) % 4][1] - quad[index][1],
        )
        for index in range(4)
    ]
    if any(not math.isfinite(length) or length <= 0.0 for length in lengths):
        return None
    opposite_error = max(
        _relative_error(lengths[0], lengths[2]),
        _relative_error(lengths[1], lengths[3]),
    )
    if opposite_error > GEOMETRY_RELATIVE_TOLERANCE:
        return None
    return min(lengths), max(lengths)


def _quad_short_axis_angle(
    quad: list[list[float]] | None,
) -> float | None:
    if quad is None:
        return None
    vectors = [
        (
            quad[(index + 1) % 4][0] - quad[index][0],
            quad[(index + 1) % 4][1] - quad[index][1],
        )
        for index in range(4)
    ]
    lengths = [math.hypot(*vector) for vector in vectors]
    if any(length <= 0.0 for length in lengths):
        return None
    perpendicular_errors = []
    for index in range(4):
        left = vectors[index]
        right = vectors[(index + 1) % 4]
        cosine = max(
            -1.0,
            min(
                1.0,
                (left[0] * right[0] + left[1] * right[1])
                / (lengths[index] * lengths[(index + 1) % 4]),
            ),
        )
        angle = math.degrees(math.acos(cosine))
        perpendicular_errors.append(abs(90.0 - angle))
    if max(perpendicular_errors) > PERPENDICULAR_ERROR_LIMIT_DEG:
        return None
    short_index = min(range(4), key=lambda index: lengths[index])
    short_vector = vectors[short_index]
    return round(
        math.degrees(math.atan2(short_vector[1], short_vector[0]))
        % 180.0,
        6,
    )


def _quad_center(quad: list[list[float]]) -> tuple[float, float]:
    return (
        round(sum(point[0] for point in quad) / 4.0, 6),
        round(sum(point[1] for point in quad) / 4.0, 6),
    )


def _project_center(
    center: dict[str, Any],
    *,
    angle_deg: float,
) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    main_x = math.cos(radians)
    main_y = math.sin(radians)
    cross_x = -main_y
    cross_y = main_x
    x = float(center["x"])
    y = float(center["y"])
    return (
        x * main_x + y * main_y,
        x * cross_x + y * cross_y,
    )


def _canonical_axis_angle(value: Any) -> float | None:
    number = _strict_number(value)
    if number is None:
        return None
    return round(number % 180.0, 6)


def _axis_delta(left: float, right: float) -> float:
    return abs((left - right + 90.0) % 180.0 - 90.0)


def _relative_error(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


def _strict_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _strict_positive_number(value: Any) -> float | None:
    number = _strict_number(value)
    return number if number is not None and number > 0.0 else None


def _canonical_derivation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "quad_source",
        "corridor_source",
        "cluster_id",
        "cluster_role",
        "cluster_membership",
        "cluster_long_edge_median_pt",
        "char_height_factor",
        "char_width_factor",
    }:
        return None
    cluster_long_edge = _strict_positive_number(
        value.get("cluster_long_edge_median_pt")
    )
    char_height_factor = _strict_positive_number(
        value.get("char_height_factor")
    )
    char_width_factor = _strict_positive_number(
        value.get("char_width_factor")
    )
    if (
        value.get("quad_source") != DECIMAL_QUAD_SOURCE
        or value.get("corridor_source") != DECIMAL_CORRIDOR_SOURCE
        or type(value.get("cluster_id")) is not str
        or not value["cluster_id"]
        or type(value.get("cluster_role")) is not str
        or not value["cluster_role"]
        or value.get("cluster_membership") != "direct_member_quad_id"
        or cluster_long_edge is None
        or char_height_factor != float(DEFAULT_K1)
        or char_width_factor != float(DEFAULT_K2)
    ):
        return None
    return {
        "quad_source": DECIMAL_QUAD_SOURCE,
        "corridor_source": DECIMAL_CORRIDOR_SOURCE,
        "cluster_id": value["cluster_id"],
        "cluster_role": value["cluster_role"],
        "cluster_membership": "direct_member_quad_id",
        "cluster_long_edge_median_pt": round(cluster_long_edge, 6),
        "char_height_factor": float(DEFAULT_K1),
        "char_width_factor": float(DEFAULT_K2),
    }


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
