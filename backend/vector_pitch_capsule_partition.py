"""Pure fail-closed capsule-boundary partition for one pitch phrase.

The adapter consumes already-proven authorities.  It does not discover
capsules and does not recognize text.  Its only permitted new exclusion is the
normalized topology's exact ``phrase_boundary_primitive_ids`` intersection.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from vector_capsule_topology import (
    normalize_vector_capsule_topology_v1,
)
from vector_pitch_anchored_character import (
    normalize_anchored_character_partition,
)
from vector_pitch_anchored_lane import normalize_anchored_local_lane
from vector_pitch_text_transform import build_phrase_primitive_manifest
from vector_pitch_window_authority import (
    normalize_pitch_window_authority,
)


SCHEMA_VERSION = "vector_pitch_capsule_partition_v1"
PRODUCER = "vector_pitch_capsule_partition_adapter_v1"
PARTITION_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_capsule_partition_digest_v1"
)
PHRASE_AUTHORITY_SCHEMA_VERSION = (
    "vector_pitch_capsule_phrase_authority_v1"
)
PHRASE_AUTHORITY_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_capsule_phrase_authority_digest_v1"
)
ITEM_GEOMETRY_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_capsule_item_geometry_digest_v1"
)
EXCLUSION_SCHEMA_VERSION = "vector_pitch_primitive_exclusion_v1"
RULE_SCHEMA_VERSION = "vector_pitch_capsule_partition_rule_v1"
CONSUMER_ALLOWED = False

EXCLUSION_STAGE = "anchored_capsule_filter"
EXCLUSION_REASON = "associated_capsule_boundary"
EXCLUSION_RELATION = "associated_capsule_outer_contour"
TRUSTED_AXIS_SOURCES = frozenset({
    "decimal_point_quad",
    "decimal_point_quad.short_axis",
    "oriented_quad:dot_axis_angle",
})


def build_capsule_phrase_authority_v1(
    *,
    phrase_id: str,
    physical_core_digest: str,
    page_context_primitive_sha256: str,
    phrase_primitive_manifest_digest: str,
    axis_angle_deg: float,
    axis_angle_source: str,
    topology_digest: str,
) -> dict[str, Any] | None:
    """Build the narrow phrase binding required by this adapter."""

    axis = _canonical_axis_angle(axis_angle_deg)
    if not (
        type(phrase_id) is str
        and phrase_id
        and _valid_sha256(physical_core_digest)
        and _valid_sha256(page_context_primitive_sha256)
        and _valid_sha256(phrase_primitive_manifest_digest)
        and axis is not None
        and type(axis_angle_source) is str
        and axis_angle_source in TRUSTED_AXIS_SOURCES
        and _valid_sha256(topology_digest)
    ):
        return None
    payload = {
        "schema_version": PHRASE_AUTHORITY_SCHEMA_VERSION,
        "phrase_id": phrase_id,
        "physical_core_digest": physical_core_digest,
        "page_context_primitive_sha256": (
            page_context_primitive_sha256
        ),
        "phrase_primitive_manifest_digest": (
            phrase_primitive_manifest_digest
        ),
        "axis_angle_deg": axis,
        "axis_angle_source": axis_angle_source,
        "topology_digest": topology_digest,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "authority_digest_schema_version": (
            PHRASE_AUTHORITY_DIGEST_SCHEMA_VERSION
        ),
        "authority_digest": _digest({
            "schema_version": (
                PHRASE_AUTHORITY_DIGEST_SCHEMA_VERSION
            ),
            "authority": payload,
        }),
    }


def normalize_capsule_phrase_authority_v1(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "phrase_id",
        "physical_core_digest",
        "page_context_primitive_sha256",
        "phrase_primitive_manifest_digest",
        "axis_angle_deg",
        "axis_angle_source",
        "topology_digest",
        "consumer_allowed",
        "authority_digest_schema_version",
        "authority_digest",
    }:
        return None
    rebuilt = build_capsule_phrase_authority_v1(
        phrase_id=value.get("phrase_id"),
        physical_core_digest=value.get("physical_core_digest"),
        page_context_primitive_sha256=value.get(
            "page_context_primitive_sha256"
        ),
        phrase_primitive_manifest_digest=value.get(
            "phrase_primitive_manifest_digest"
        ),
        axis_angle_deg=value.get("axis_angle_deg"),
        axis_angle_source=value.get("axis_angle_source"),
        topology_digest=value.get("topology_digest"),
    )
    return rebuilt if rebuilt == value else None


def prepare_vector_pitch_capsule_partition(
    *,
    full_phrase_items: Any,
    admitted_phrase_items: Any,
    phrase_authority: Any,
    window_authority: Any,
    anchored_lane_plan: Any,
    anchored_character_plan: Any,
    capsule_topology: Any,
    boundary_cut_primitive_ids: Any,
) -> dict[str, Any]:
    """Exclude only the topology-proven phrase-boundary intersection."""

    phrase_binding = normalize_capsule_phrase_authority_v1(
        phrase_authority
    )
    window = normalize_pitch_window_authority(window_authority)
    lane = normalize_anchored_local_lane(anchored_lane_plan)
    character = normalize_anchored_character_partition(
        anchored_character_plan,
        window_authority=window_authority,
        anchored_lane_plan=anchored_lane_plan,
    )
    full_rows = _normalize_items(full_phrase_items)
    upstream_rows = _normalize_items(admitted_phrase_items)
    phrase_id = (
        phrase_binding["phrase_id"]
        if phrase_binding is not None
        else None
    )
    if (
        phrase_binding is not None
        and window is not None
        and full_rows is not None
        and window.get("schema_version")
        == "vector_pitch_window_authority_v2"
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_group_ownership_unresolved",
            input_ids=sorted(full_rows),
        )
    if (
        phrase_binding is None
        or window is None
        or lane is None
        or character is None
        or full_rows is None
        or not full_rows
        or upstream_rows is None
        or not upstream_rows
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_partition_input_invalid",
        )

    full_ids = sorted(full_rows)
    upstream_ids = sorted(upstream_rows)
    page_manifest = {
        "schema_version": "vector_page_context_manifest_v1",
        "page_num": int(capsule_topology.get("page_num", 0))
        if isinstance(capsule_topology, dict)
        else 0,
        "primitive_count": len(full_ids),
        "primitive_sha256": phrase_binding[
            "page_context_primitive_sha256"
        ],
        "consumer_allowed": False,
    }
    topology = normalize_vector_capsule_topology_v1(
        capsule_topology,
        expected_page_context_manifest=page_manifest,
        expected_axis_angle_deg=phrase_binding["axis_angle_deg"],
        expected_phrase_primitive_ids=full_ids,
    )
    phrase_manifest = build_phrase_primitive_manifest(
        phrase_id=phrase_id,
        page_context_manifest=page_manifest,
        items=full_phrase_items,
        include_geometry=True,
    )
    if topology is None or phrase_manifest is None:
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_topology_authority_invalid",
            input_ids=full_ids,
        )
    cut_ids = _canonical_ids(boundary_cut_primitive_ids)
    topology_boundary_ids = list(topology["boundary_primitive_ids"])
    topology_phrase_boundary_ids = list(
        topology["phrase_boundary_primitive_ids"]
    )
    topology_non_boundary_ids = sorted(
        row["primitive_id"]
        for row in topology["non_boundary_component_primitives"]
    )
    if (
        cut_ids is None
        or not cut_ids
        or not set(cut_ids).issubset(topology_boundary_ids)
        or set(cut_ids).intersection(topology_non_boundary_ids)
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_boundary_cut_invalid",
            input_ids=full_ids,
        )

    if not _authorities_match(
        phrase_binding=phrase_binding,
        window=window,
        lane=lane,
        character=character,
        topology=topology,
        phrase_manifest=phrase_manifest,
        full_ids=full_ids,
        upstream_ids=upstream_ids,
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_partition_authority_mismatch",
            input_ids=full_ids,
        )
    if any(
        upstream_rows[primitive_id] != full_rows.get(primitive_id)
        for primitive_id in upstream_ids
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_partition_item_geometry_mismatch",
            input_ids=full_ids,
        )

    excluded_ids = list(topology["phrase_boundary_primitive_ids"])
    protected_ids = set(character["protected_primitive_ids"])
    if (
        not excluded_ids
        or not set(excluded_ids).issubset(upstream_ids)
        or set(excluded_ids) == set(topology_boundary_ids)
        or set(excluded_ids).intersection(protected_ids)
        or set(excluded_ids) != (
            set(topology["boundary_primitive_ids"])
            & set(full_ids)
        )
        or set(excluded_ids).intersection(
            row["primitive_id"]
            for row in topology[
                "non_boundary_component_primitives"
            ]
        )
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_boundary_partition_invalid",
            input_ids=full_ids,
        )

    upstream_excluded_ids = sorted(set(full_ids) - set(upstream_ids))
    admitted_ids = sorted(set(upstream_ids) - set(excluded_ids))
    if not _exact_partition(
        input_ids=full_ids,
        upstream_admitted_ids=upstream_ids,
        upstream_excluded_ids=upstream_excluded_ids,
        excluded_ids=excluded_ids,
        admitted_ids=admitted_ids,
    ):
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_boundary_partition_invalid",
            input_ids=full_ids,
        )

    exclusion = _build_exclusion(
        primitive_ids=excluded_ids,
        rows=full_rows,
        authority=window,
        topology_digest=topology["topology_digest"],
    )
    if exclusion is None:
        return _failure(
            phrase_id=phrase_id,
            reason="capsule_partition_item_geometry_mismatch",
            input_ids=full_ids,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "status": "ready",
        "reason": None,
        "phrase_id": phrase_id,
        "physical_core_digest": phrase_binding[
            "physical_core_digest"
        ],
        "phrase_authority_digest": phrase_binding["authority_digest"],
        "topology_digest_schema_version": topology[
            "topology_digest_schema_version"
        ],
        "topology_digest": topology["topology_digest"],
        "capsule_id": topology["capsule_id"],
        "topology_boundary_primitive_ids": topology_boundary_ids,
        "topology_phrase_boundary_primitive_ids": (
            topology_phrase_boundary_ids
        ),
        "topology_non_boundary_component_primitive_ids": (
            topology_non_boundary_ids
        ),
        "boundary_cut_primitive_ids": cut_ids,
        "page_context_primitive_sha256": phrase_binding[
            "page_context_primitive_sha256"
        ],
        "phrase_primitive_manifest_digest": phrase_binding[
            "phrase_primitive_manifest_digest"
        ],
        "window_authority_digest": window["authority_digest"],
        "window_source_digests": list(
            window["group_member_source_digests"]
        ),
        "anchored_lane_partition_digest": lane["partition_digest"],
        "anchored_character_partition_digest": character[
            "partition_digest"
        ],
        "axis_angle_deg": phrase_binding["axis_angle_deg"],
        "axis_angle_source": phrase_binding["axis_angle_source"],
        "input_primitive_ids": full_ids,
        "input_primitive_count": len(full_ids),
        "input_item_geometry_digest": _item_geometry_digest(
            full_rows,
            full_ids,
        ),
        "upstream_admitted_primitive_ids": upstream_ids,
        "upstream_admitted_primitive_count": len(upstream_ids),
        "upstream_admitted_item_geometry_digest": (
            _item_geometry_digest(upstream_rows, upstream_ids)
        ),
        "upstream_excluded_primitive_ids": upstream_excluded_ids,
        "upstream_excluded_primitive_count": len(
            upstream_excluded_ids
        ),
        "excluded_primitive_ids": excluded_ids,
        "excluded_primitive_count": len(excluded_ids),
        "excluded_item_geometry_digest": _item_geometry_digest(
            full_rows,
            excluded_ids,
        ),
        "admitted_primitive_ids": admitted_ids,
        "admitted_primitive_count": len(admitted_ids),
        "admitted_item_geometry_digest": _item_geometry_digest(
            full_rows,
            admitted_ids,
        ),
        "exclusions": [exclusion],
        "rule_contract": _rule_contract(),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "partition_digest_schema_version": (
            PARTITION_DIGEST_SCHEMA_VERSION
        ),
        "partition_digest": _digest({
            "schema_version": PARTITION_DIGEST_SCHEMA_VERSION,
            "partition": payload,
        }),
    }


def vector_pitch_capsule_partition_identity_reasons(
    value: Any,
) -> list[str]:
    """Replay the self-contained partition identity and set arithmetic."""

    expected_keys = {
        "schema_version",
        "producer",
        "status",
        "reason",
        "phrase_id",
        "physical_core_digest",
        "phrase_authority_digest",
        "topology_digest_schema_version",
        "topology_digest",
        "capsule_id",
        "topology_boundary_primitive_ids",
        "topology_phrase_boundary_primitive_ids",
        "topology_non_boundary_component_primitive_ids",
        "boundary_cut_primitive_ids",
        "page_context_primitive_sha256",
        "phrase_primitive_manifest_digest",
        "window_authority_digest",
        "window_source_digests",
        "anchored_lane_partition_digest",
        "anchored_character_partition_digest",
        "axis_angle_deg",
        "axis_angle_source",
        "input_primitive_ids",
        "input_primitive_count",
        "input_item_geometry_digest",
        "upstream_admitted_primitive_ids",
        "upstream_admitted_primitive_count",
        "upstream_admitted_item_geometry_digest",
        "upstream_excluded_primitive_ids",
        "upstream_excluded_primitive_count",
        "excluded_primitive_ids",
        "excluded_primitive_count",
        "excluded_item_geometry_digest",
        "admitted_primitive_ids",
        "admitted_primitive_count",
        "admitted_item_geometry_digest",
        "exclusions",
        "rule_contract",
        "consumer_allowed",
        "partition_digest_schema_version",
        "partition_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return ["capsule_partition_schema_invalid"]
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("producer") == PRODUCER
        and value.get("status") == "ready"
        and value.get("reason") is None
        and type(value.get("phrase_id")) is str
        and value["phrase_id"]
        and value.get("consumer_allowed") is False
        and value.get("partition_digest_schema_version")
        == PARTITION_DIGEST_SCHEMA_VERSION
    ):
        return ["capsule_partition_identity_invalid"]

    digest = value.get("partition_digest")
    payload = {
        key: field
        for key, field in value.items()
        if key not in {
            "partition_digest_schema_version",
            "partition_digest",
        }
    }
    if not _valid_sha256(digest):
        return ["capsule_partition_digest_invalid"]
    if digest != _digest({
        "schema_version": PARTITION_DIGEST_SCHEMA_VERSION,
        "partition": payload,
    }):
        return ["capsule_partition_digest_mismatch"]

    digest_fields = (
        "physical_core_digest",
        "phrase_authority_digest",
        "topology_digest",
        "page_context_primitive_sha256",
        "phrase_primitive_manifest_digest",
        "window_authority_digest",
        "anchored_lane_partition_digest",
        "anchored_character_partition_digest",
        "input_item_geometry_digest",
        "upstream_admitted_item_geometry_digest",
        "excluded_item_geometry_digest",
        "admitted_item_geometry_digest",
    )
    if any(not _valid_sha256(value.get(key)) for key in digest_fields):
        return ["capsule_partition_binding_digest_invalid"]
    topology_boundary_ids = _canonical_ids(
        value.get("topology_boundary_primitive_ids")
    )
    topology_phrase_boundary_ids = _canonical_ids(
        value.get("topology_phrase_boundary_primitive_ids")
    )
    topology_non_boundary_ids = _canonical_ids(
        value.get("topology_non_boundary_component_primitive_ids")
    )
    boundary_cut_ids = _canonical_ids(
        value.get("boundary_cut_primitive_ids")
    )
    input_ids = _canonical_ids(value.get("input_primitive_ids"))
    upstream_ids = _canonical_ids(
        value.get("upstream_admitted_primitive_ids")
    )
    upstream_excluded_ids = _canonical_ids(
        value.get("upstream_excluded_primitive_ids")
    )
    excluded_ids = _canonical_ids(
        value.get("excluded_primitive_ids")
    )
    admitted_ids = _canonical_ids(
        value.get("admitted_primitive_ids")
    )
    if (
        input_ids is None
        or not input_ids
        or upstream_ids is None
        or not upstream_ids
        or upstream_excluded_ids is None
        or excluded_ids is None
        or not excluded_ids
        or admitted_ids is None
        or topology_boundary_ids is None
        or not topology_boundary_ids
        or topology_phrase_boundary_ids is None
        or not topology_phrase_boundary_ids
        or topology_non_boundary_ids is None
        or boundary_cut_ids is None
        or not boundary_cut_ids
        or not _exact_partition(
            input_ids=input_ids,
            upstream_admitted_ids=upstream_ids,
            upstream_excluded_ids=upstream_excluded_ids,
            excluded_ids=excluded_ids,
            admitted_ids=admitted_ids,
        )
    ):
        return ["capsule_partition_exact_partition_invalid"]
    if (
        excluded_ids != topology_phrase_boundary_ids
        or not set(topology_phrase_boundary_ids).issubset(
            topology_boundary_ids
        )
        or set(topology_phrase_boundary_ids) == set(
            topology_boundary_ids
        )
        or set(topology_non_boundary_ids).intersection(
            topology_boundary_ids
        )
        or set(topology_non_boundary_ids).intersection(excluded_ids)
        or not set(boundary_cut_ids).issubset(topology_boundary_ids)
        or set(boundary_cut_ids).intersection(
            topology_non_boundary_ids
        )
    ):
        return ["capsule_partition_topology_boundary_mismatch"]
    count_pairs = (
        ("input_primitive_count", input_ids),
        ("upstream_admitted_primitive_count", upstream_ids),
        ("upstream_excluded_primitive_count", upstream_excluded_ids),
        ("excluded_primitive_count", excluded_ids),
        ("admitted_primitive_count", admitted_ids),
    )
    if any(value.get(key) != len(ids) for key, ids in count_pairs):
        return ["capsule_partition_count_invalid"]

    exclusions = value.get("exclusions")
    if (
        not isinstance(exclusions, list)
        or len(exclusions) != 1
        or not _valid_exclusion(
            exclusions[0],
            expected_ids=excluded_ids,
            topology_digest=value["topology_digest"],
            window_authority_digest=value[
                "window_authority_digest"
            ],
            window_source_digests=value["window_source_digests"],
            axis_angle_deg=value["axis_angle_deg"],
        )
    ):
        return ["capsule_partition_exclusion_invalid"]
    if value.get("rule_contract") != _rule_contract():
        return ["capsule_partition_rule_contract_invalid"]
    return []


def normalize_vector_pitch_capsule_partition(
    value: Any,
    *,
    full_phrase_items: Any,
    admitted_phrase_items: Any,
    phrase_authority: Any,
    window_authority: Any,
    anchored_lane_plan: Any,
    anchored_character_plan: Any,
    capsule_topology: Any,
    boundary_cut_primitive_ids: Any,
) -> dict[str, Any] | None:
    """Replay both the payload and every external authority binding."""

    if vector_pitch_capsule_partition_identity_reasons(value):
        return None
    rebuilt = prepare_vector_pitch_capsule_partition(
        full_phrase_items=full_phrase_items,
        admitted_phrase_items=admitted_phrase_items,
        phrase_authority=phrase_authority,
        window_authority=window_authority,
        anchored_lane_plan=anchored_lane_plan,
        anchored_character_plan=anchored_character_plan,
        capsule_topology=capsule_topology,
        boundary_cut_primitive_ids=boundary_cut_primitive_ids,
    )
    return rebuilt if rebuilt == value else None


def _authorities_match(
    *,
    phrase_binding: dict[str, Any],
    window: dict[str, Any],
    lane: dict[str, Any],
    character: dict[str, Any],
    topology: dict[str, Any],
    phrase_manifest: dict[str, Any],
    full_ids: list[str],
    upstream_ids: list[str],
) -> bool:
    phrase_id = phrase_binding["phrase_id"]
    axis = phrase_binding["axis_angle_deg"]
    return bool(
        phrase_id == window["phrase_id"]
        == lane["phrase_id"]
        == character["phrase_id"]
        and axis == window["axis_angle_deg"]
        == lane["axis_angle_deg"]
        == character["axis_angle_deg"]
        == topology["axis_angle_deg"]
        and phrase_binding["axis_angle_source"]
        == window["axis_angle_source"]
        and phrase_binding["topology_digest"]
        == topology["topology_digest"]
        and phrase_binding["page_context_primitive_sha256"]
        == window["page_context_primitive_sha256"]
        == topology["page_context_primitive_sha256"]
        and phrase_binding["phrase_primitive_manifest_digest"]
        == window["phrase_primitive_manifest_digest"]
        == phrase_manifest["manifest_digest"]
        and lane["input_primitive_ids"] == full_ids
        and character["input_primitive_ids"]
        == lane["admitted_primitive_ids"]
        and character["admitted_primitive_ids"] == upstream_ids
        and lane["partition_digest"]
        == character["anchored_lane_partition_digest"]
        and lane["window_source_digests"]
        == window["group_member_source_digests"]
    )


def _normalize_items(value: Any) -> dict[str, dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    rows: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, dict):
            return None
        primitive_id = raw.get("item_id")
        op = raw.get("op")
        points = _canonical_points(raw.get("points"))
        bbox = _canonical_bbox(raw.get("bbox"))
        draw_type = raw.get("draw_type")
        stroke_width = raw.get("stroke_width")
        if (
            type(primitive_id) is not str
            or not primitive_id
            or primitive_id in rows
            or type(op) is not str
            or not op
            or not primitive_id.endswith(f"_{op}")
            or points is None
            or bbox is None
            or type(draw_type) is not str
            or (
                stroke_width is not None
                and (
                    type(stroke_width) not in {int, float}
                    or not math.isfinite(float(stroke_width))
                )
            )
        ):
            return None
        point_bbox = [
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        ]
        if any(
            abs(point_bbox[index] - bbox[index]) > 0.000001
            for index in range(4)
        ):
            return None
        rows[primitive_id] = {
            "primitive_id": primitive_id,
            "op": op,
            "points": points,
            "bbox": bbox,
            "draw_type": draw_type,
            "stroke_width": (
                round(float(stroke_width), 6)
                if stroke_width is not None
                else None
            ),
        }
    return rows


def _build_exclusion(
    *,
    primitive_ids: list[str],
    rows: dict[str, dict[str, Any]],
    authority: dict[str, Any],
    topology_digest: str,
) -> dict[str, Any] | None:
    selected = [rows.get(primitive_id) for primitive_id in primitive_ids]
    if any(row is None for row in selected):
        return None
    present = [row for row in selected if row is not None]
    points = [
        point
        for row in present
        for point in row["points"]
    ]
    if not points:
        return None
    main = (
        float(authority["main_unit"]["x"]),
        float(authority["main_unit"]["y"]),
    )
    cross = (
        float(authority["cross_unit"]["x"]),
        float(authority["cross_unit"]["y"]),
    )
    main_values = [_project(point, main) for point in points]
    cross_values = [_project(point, cross) for point in points]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    main_span = max(main_values) - min(main_values)
    cross_span = max(cross_values) - min(cross_values)
    char_width = float(authority["char_width_pt"])
    char_height = float(authority["char_height_pt"])
    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "primitive_ids": list(primitive_ids),
        "stage": EXCLUSION_STAGE,
        "reason": EXCLUSION_REASON,
        "bbox": [
            round(min(xs), 6),
            round(min(ys), 6),
            round(max(xs), 6),
            round(max(ys), 6),
        ],
        "canonical_bbox": {
            "main_min": round(min(main_values), 6),
            "cross_min": round(min(cross_values), 6),
            "main_max": round(max(main_values), 6),
            "cross_max": round(max(cross_values), 6),
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
        "relation": EXCLUSION_RELATION,
        "group_digest": topology_digest,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _valid_exclusion(
    value: Any,
    *,
    expected_ids: list[str],
    topology_digest: str,
    window_authority_digest: str,
    window_source_digests: Any,
    axis_angle_deg: Any,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
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
        return False
    return bool(
        value.get("schema_version") == EXCLUSION_SCHEMA_VERSION
        and value.get("primitive_ids") == expected_ids
        and value.get("stage") == EXCLUSION_STAGE
        and value.get("reason") == EXCLUSION_REASON
        and _canonical_bbox(value.get("bbox")) is not None
        and _canonical_axis_bbox(value.get("canonical_bbox")) is not None
        and _canonical_measurements(value.get("measurements")) is not None
        and value.get("authority_digest")
        == window_authority_digest
        and value.get("window_source_digests")
        == window_source_digests
        and value.get("axis_angle_deg")
        == _canonical_axis_angle(axis_angle_deg)
        and value.get("relation") == EXCLUSION_RELATION
        and value.get("group_digest") == topology_digest
        and value.get("consumer_allowed") is False
    )


def _rule_contract() -> dict[str, Any]:
    return {
        "schema_version": RULE_SCHEMA_VERSION,
        "stage": EXCLUSION_STAGE,
        "reason": EXCLUSION_REASON,
        "relation": EXCLUSION_RELATION,
        "source_set": "normalized_topology.phrase_boundary_primitive_ids",
        "boundary_cut_set": "normalized_topology.boundary_primitive_ids",
        "nonempty_boundary_cut_required": True,
        "full_contour_exclusion_allowed": False,
        "non_boundary_component_exclusion_allowed": False,
        "exact_partition_required": True,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _exact_partition(
    *,
    input_ids: list[str],
    upstream_admitted_ids: list[str],
    upstream_excluded_ids: list[str],
    excluded_ids: list[str],
    admitted_ids: list[str],
) -> bool:
    partitions = (
        set(upstream_excluded_ids),
        set(excluded_ids),
        set(admitted_ids),
    )
    return bool(
        set(upstream_admitted_ids)
        == set(excluded_ids).union(admitted_ids)
        and set(input_ids)
        == set(upstream_excluded_ids).union(
            excluded_ids,
            admitted_ids,
        )
        and all(
            partitions[left].isdisjoint(partitions[right])
            for left in range(len(partitions))
            for right in range(left + 1, len(partitions))
        )
    )


def _item_geometry_digest(
    rows: dict[str, dict[str, Any]],
    primitive_ids: Iterable[str],
) -> str:
    return _digest({
        "schema_version": ITEM_GEOMETRY_DIGEST_SCHEMA_VERSION,
        "items": [rows[primitive_id] for primitive_id in primitive_ids],
    })


def _canonical_ids(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        return None
    canonical = sorted(value)
    return canonical if value == canonical else None


def _canonical_points(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    points: list[list[float]] = []
    for raw in value:
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or any(type(number) not in {int, float} for number in raw)
            or any(not math.isfinite(float(number)) for number in raw)
        ):
            return None
        points.append([
            round(float(raw[0]), 6),
            round(float(raw[1]), 6),
        ])
    return points


def _canonical_bbox(value: Any) -> list[float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(number) not in {int, float} for number in value)
    ):
        return None
    numbers = [round(float(number), 6) for number in value]
    if (
        any(not math.isfinite(number) for number in numbers)
        or numbers[2] < numbers[0]
        or numbers[3] < numbers[1]
    ):
        return None
    return numbers


def _canonical_axis_bbox(value: Any) -> dict[str, float] | None:
    keys = {"main_min", "cross_min", "main_max", "cross_max"}
    if not isinstance(value, dict) or set(value) != keys:
        return None
    numbers = {
        key: round(float(value[key]), 6)
        for key in keys
        if type(value[key]) in {int, float}
        and math.isfinite(float(value[key]))
    }
    if (
        len(numbers) != 4
        or numbers["main_max"] < numbers["main_min"]
        or numbers["cross_max"] < numbers["cross_min"]
    ):
        return None
    return numbers


def _canonical_measurements(value: Any) -> dict[str, float] | None:
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
    numbers = {key: round(float(value[key]), 6) for key in keys}
    if (
        numbers["main_span_pt"] < 0.0
        or numbers["cross_span_pt"] < 0.0
        or numbers["char_width_pt"] <= 0.0
        or numbers["char_height_pt"] <= 0.0
        or numbers["main_to_char_width_ratio"]
        != round(
            numbers["main_span_pt"] / numbers["char_width_pt"],
            6,
        )
        or numbers["cross_to_char_height_ratio"]
        != round(
            numbers["cross_span_pt"] / numbers["char_height_pt"],
            6,
        )
    ):
        return None
    return numbers


def _canonical_axis_angle(value: Any) -> float | None:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return None
    normalized = float(value) % 180.0
    return round(normalized, 6)


def _project(
    point: list[float],
    axis: tuple[float, float],
) -> float:
    return point[0] * axis[0] + point[1] * axis[1]


def _failure(
    *,
    phrase_id: Any,
    reason: str,
    input_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "status": "blocked",
        "reason": reason,
        "phrase_id": (
            phrase_id if type(phrase_id) is str else None
        ),
        "input_primitive_ids": list(input_ids or []),
        "admitted_primitive_ids": [],
        "excluded_primitive_ids": [],
        "exclusions": [],
        "consumer_allowed": CONSUMER_ALLOWED,
    }


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
