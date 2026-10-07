"""Exact semantic-ink occupation for grouped pitch phrase reads.

The ledger does not recognize a symbol.  It consumes one already canonical
``plus_minus`` authority, removes only its exact primitives from the numeric
groups, and binds the result to the grouped window and phrase manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from vector_anchor_semantic_bridge import (
    normalize_semantic_anchor_authority,
)
from vector_pitch_text_transform import valid_phrase_primitive_manifest
from vector_pitch_window_authority import normalize_pitch_window_authority


SCHEMA_VERSION = "vector_pitch_semantic_occupation_v1"
ENTRY_SCHEMA_VERSION = "vector_pitch_semantic_occupation_entry_v1"
PRODUCER = "vector_pitch_semantic_occupation"
OCCUPATION_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_semantic_occupation_digest_v1"
)
OCCUPIED_PRIMITIVE_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_semantic_occupation_primitive_set_digest_v1"
)
BINDING_SCHEMA_VERSION = "vector_pitch_semantic_occupation_binding_v1"
BINDING_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_semantic_occupation_binding_digest_v1"
)
CONSUMER_ALLOWED = False


def build_vector_pitch_semantic_occupation(
    *,
    phrase_id: str,
    page_index: int,
    items: Any,
    primitive_manifest: Any,
    window_authority: Any,
    semantic_authorities: Any,
) -> dict[str, Any] | None:
    """Build one exact ``±`` occupation and its numeric-side partition."""
    if not isinstance(semantic_authorities, list) or len(
        semantic_authorities
    ) != 1:
        return None
    authority = normalize_semantic_anchor_authority(
        semantic_authorities[0]
    )
    if authority is None:
        return None
    payload = _build_payload(
        phrase_id=phrase_id,
        page_index=page_index,
        items=items,
        primitive_manifest=primitive_manifest,
        window_authority=window_authority,
        authority=authority,
    )
    if payload is None:
        return None
    return {
        **payload,
        "occupation_digest": _digest({
            "schema_version": OCCUPATION_DIGEST_SCHEMA_VERSION,
            "occupation": payload,
        }),
    }


def normalize_vector_pitch_semantic_occupation(
    value: Any,
    *,
    phrase_id: str,
    page_index: int,
    items: Any,
    primitive_manifest: Any,
    window_authority: Any,
) -> dict[str, Any] | None:
    """Replay a ledger against its live items, manifest, and grouped authority."""
    if not isinstance(value, dict):
        return None
    authority = _authority_from_entry(
        value.get("entries"),
        page_index=page_index,
    )
    if authority is None:
        return None
    payload = _build_payload(
        phrase_id=phrase_id,
        page_index=page_index,
        items=items,
        primitive_manifest=primitive_manifest,
        window_authority=window_authority,
        authority=authority,
    )
    if payload is None:
        return None
    canonical = {
        **payload,
        "occupation_digest": _digest({
            "schema_version": OCCUPATION_DIGEST_SCHEMA_VERSION,
            "occupation": payload,
        }),
    }
    return canonical if value == canonical else None


def semantic_occupation_binding(value: Any) -> dict[str, Any] | None:
    """Build or idempotently normalize the compact downstream binding."""
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") == BINDING_SCHEMA_VERSION:
        expected_keys = {
            "schema_version",
            "producer",
            "page_index",
            "phrase_id",
            "window_authority_digest",
            "cross_group_relation_digest",
            "phrase_primitive_manifest_digest",
            "occupation_digest",
            "occupied_primitive_ids",
            "occupied_primitive_count",
            "occupied_primitive_digest",
            "consumer_allowed",
            "binding_digest_schema_version",
            "binding_digest",
        }
        if set(value) != expected_keys:
            return None
        payload = {
            key: value[key]
            for key in expected_keys
            if key not in {
                "binding_digest_schema_version",
                "binding_digest",
            }
        }
        if not _valid_binding_payload(payload):
            return None
        canonical = {
            **payload,
            "binding_digest_schema_version": BINDING_DIGEST_SCHEMA_VERSION,
            "binding_digest": _digest({
                "schema_version": BINDING_DIGEST_SCHEMA_VERSION,
                "binding": payload,
            }),
        }
        return canonical if value == canonical else None

    occupation = _normalize_embedded_occupation(value)
    if occupation is None:
        return None
    payload = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "producer": PRODUCER,
        "page_index": occupation["page_index"],
        "phrase_id": occupation["phrase_id"],
        "window_authority_digest": occupation[
            "window_authority_digest"
        ],
        "cross_group_relation_digest": occupation[
            "cross_group_relation_digest"
        ],
        "phrase_primitive_manifest_digest": occupation[
            "phrase_primitive_manifest_digest"
        ],
        "occupation_digest": occupation["occupation_digest"],
        "occupied_primitive_ids": list(
            occupation["occupied_primitive_ids"]
        ),
        "occupied_primitive_count": occupation[
            "occupied_primitive_count"
        ],
        "occupied_primitive_digest": occupation[
            "occupied_primitive_digest"
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "binding_digest_schema_version": BINDING_DIGEST_SCHEMA_VERSION,
        "binding_digest": _digest({
            "schema_version": BINDING_DIGEST_SCHEMA_VERSION,
            "binding": payload,
        }),
    }


def _build_payload(
    *,
    phrase_id: Any,
    page_index: Any,
    items: Any,
    primitive_manifest: Any,
    window_authority: Any,
    authority: dict[str, Any],
) -> dict[str, Any] | None:
    window = normalize_pitch_window_authority(window_authority)
    if not (
        type(phrase_id) is str
        and phrase_id
        and type(page_index) is int
        and page_index >= 0
        and isinstance(items, list)
        and items
        and isinstance(primitive_manifest, dict)
        and primitive_manifest.get("schema_version")
        == "vector_pitch_phrase_primitive_manifest_v2"
        and valid_phrase_primitive_manifest(
            primitive_manifest,
            phrase_id=phrase_id,
        )
        and window is not None
        and window.get("schema_version")
        == "vector_pitch_window_authority_v2"
        and window.get("page_index") == page_index
        and window.get("phrase_id") == phrase_id
        and window.get("phrase_primitive_manifest_digest")
        == primitive_manifest.get("manifest_digest")
        and authority.get("page_index") == page_index
        and authority.get("glyph_type") == "plus_minus"
        and authority.get("output_text") == "±"
        and authority.get("detector_source")
        == "decimal_corridor_plus_minus_strict"
    ):
        return None
    relation = window.get("cross_group_relation")
    groups = window.get("groups")
    main = window.get("main_unit")
    if not (
        isinstance(relation, dict)
        and isinstance(groups, list)
        and len(groups) == 2
        and [group.get("size_class") for group in groups]
        == ["primary", "tolerance"]
        and isinstance(main, dict)
    ):
        return None
    main_unit = _canonical_unit(main)
    if main_unit is None:
        return None

    matching_carrier_groups = [
        group
        for group in groups
        if authority["pitch_window_source_digest"]
        in set(
            group["window_authority"][
                "group_member_source_digests"
            ]
        )
        and authority["decimal_quad_id"]
        in set(
            group["window_authority"][
                "group_member_decimal_quad_ids"
            ]
        )
    ]
    if len(matching_carrier_groups) != 1:
        return None

    item_rows = _canonical_items(items)
    if item_rows is None or not _manifest_matches_items(
        primitive_manifest,
        item_rows=item_rows,
    ):
        return None
    items_by_id = {row["primitive_id"]: row for row in item_rows}
    authority_ids = list(authority["source_primitive_ids"])
    if (
        len(authority_ids) != 3
        or any(primitive_id not in items_by_id for primitive_id in authority_ids)
        or [items_by_id[primitive_id]["op"] for primitive_id in authority_ids]
        != authority["source_primitive_ops"]
        or _authority_manifest_bbox(
            authority_ids,
            items_by_id=items_by_id,
        )
        != authority["anchor_bbox"]
    ):
        return None

    symbol_interval = _union_main_interval(
        authority_ids,
        items_by_id=items_by_id,
        main_unit=main_unit,
    )
    if symbol_interval is None:
        return None
    occupied_ids = sorted(authority_ids)
    occupied_set = set(occupied_ids)
    numeric_ids = sorted(set(items_by_id) - occupied_set)
    if not numeric_ids:
        return None

    assignments_by_class: dict[str, list[dict[str, Any]]] = {
        "primary": [],
        "tolerance": [],
    }
    for primitive_id in numeric_ids:
        interval = _item_main_interval(
            items_by_id[primitive_id],
            main_unit=main_unit,
        )
        if interval is None:
            return None
        if interval[1] < symbol_interval[0]:
            size_class = "primary"
        elif interval[0] > symbol_interval[1]:
            size_class = "tolerance"
        else:
            return None
        assignments_by_class[size_class].append({
            "primitive_id": primitive_id,
            "main_min": interval[0],
            "main_max": interval[1],
            "consumer_allowed": CONSUMER_ALLOWED,
        })

    occupation_groups: list[dict[str, Any]] = []
    for group in groups:
        size_class = str(group["size_class"])
        child = group["window_authority"]
        assignments = assignments_by_class[size_class]
        primitive_ids = [row["primitive_id"] for row in assignments]
        source_ids = sorted(child["group_member_source_primitive_ids"])
        if not (
            assignments
            and set(source_ids).issubset(primitive_ids)
            and child["source_primitive_id"] in primitive_ids
        ):
            return None
        occupation_groups.append({
            "size_class": size_class,
            "group_digest": group["group_digest"],
            "window_authority_digest": child["authority_digest"],
            "source_primitive_ids": source_ids,
            "primitive_ids": primitive_ids,
            "primitive_main_intervals": assignments,
            "consumer_allowed": CONSUMER_ALLOWED,
        })

    entry_payload = {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "glyph_id": authority["glyph_id"],
        "glyph_type": authority["glyph_type"],
        "output_text": authority["output_text"],
        "detector_source": authority["detector_source"],
        "authority_digest_schema_version": authority[
            "authority_digest_schema_version"
        ],
        "authority_digest": authority["authority_digest"],
        "topology_proof_schema_version": authority[
            "topology_proof_schema_version"
        ],
        "topology_proof_digest": authority["topology_proof_digest"],
        "decimal_quad_id": authority["decimal_quad_id"],
        "decimal_authority_digest": authority[
            "decimal_authority_digest"
        ],
        "pitch_window_source_digest": authority[
            "pitch_window_source_digest"
        ],
        "source_primitive_ids": authority_ids,
        "source_primitive_ops": list(authority["source_primitive_ops"]),
        "source_primitive_roles": list(
            authority["source_primitive_roles"]
        ),
        "anchor_bbox": copy.deepcopy(authority["anchor_bbox"]),
        "absolute_main_interval": symbol_interval,
        "left_group": _entry_group_binding(occupation_groups[0]),
        "right_group": _entry_group_binding(occupation_groups[1]),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    entry = {
        **entry_payload,
        "entry_digest": _digest(entry_payload),
    }
    occupied_digest = _digest({
        "schema_version": OCCUPIED_PRIMITIVE_DIGEST_SCHEMA_VERSION,
        "primitive_ids": occupied_ids,
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "page_index": page_index,
        "phrase_id": phrase_id,
        "window_authority_digest": window["authority_digest"],
        "cross_group_relation_digest": relation["relation_digest"],
        "phrase_primitive_manifest_digest": primitive_manifest[
            "manifest_digest"
        ],
        "axis_angle_deg": window["axis_angle_deg"],
        "main_unit": copy.deepcopy(window["main_unit"]),
        "entries": [entry],
        "occupied_primitive_ids": occupied_ids,
        "occupied_primitive_count": len(occupied_ids),
        "occupied_primitive_digest": occupied_digest,
        "numeric_primitive_ids": numeric_ids,
        "groups": occupation_groups,
        "primitive_ids": sorted(items_by_id),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _normalize_embedded_occupation(
    value: Any,
) -> dict[str, Any] | None:
    expected_keys = {
        "schema_version",
        "producer",
        "page_index",
        "phrase_id",
        "window_authority_digest",
        "cross_group_relation_digest",
        "phrase_primitive_manifest_digest",
        "axis_angle_deg",
        "main_unit",
        "entries",
        "occupied_primitive_ids",
        "occupied_primitive_count",
        "occupied_primitive_digest",
        "numeric_primitive_ids",
        "groups",
        "primitive_ids",
        "consumer_allowed",
        "occupation_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    payload = {
        key: value[key]
        for key in expected_keys
        if key != "occupation_digest"
    }
    occupied_ids = _canonical_id_list(value.get("occupied_primitive_ids"))
    numeric_ids = _canonical_id_list(value.get("numeric_primitive_ids"))
    primitive_ids = _canonical_id_list(value.get("primitive_ids"))
    entries = value.get("entries")
    groups = value.get("groups")
    entry = (
        entries[0]
        if isinstance(entries, list)
        and len(entries) == 1
        and isinstance(entries[0], dict)
        else None
    )
    entry_ids = _ordered_unique_id_list(
        entry.get("source_primitive_ids")
        if isinstance(entry, dict)
        else None
    )
    entry_authority = (
        _authority_from_entry(
            entries,
            page_index=value["page_index"],
        )
        if type(value.get("page_index")) is int
        else None
    )
    group_primitive_ids = (
        [
            _canonical_id_list(group.get("primitive_ids"))
            if isinstance(group, dict)
            else None
            for group in groups
        ]
        if isinstance(groups, list)
        else []
    )
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("producer") == PRODUCER
        and type(value.get("page_index")) is int
        and value["page_index"] >= 0
        and type(value.get("phrase_id")) is str
        and bool(value["phrase_id"])
        and all(_valid_sha256(value.get(key)) for key in (
            "window_authority_digest",
            "cross_group_relation_digest",
            "phrase_primitive_manifest_digest",
            "occupation_digest",
            "occupied_primitive_digest",
        ))
        and occupied_ids is not None
        and len(occupied_ids) == 3
        and numeric_ids is not None
        and bool(numeric_ids)
        and primitive_ids is not None
        and not (set(occupied_ids) & set(numeric_ids))
        and sorted([*occupied_ids, *numeric_ids]) == primitive_ids
        and value.get("occupied_primitive_count") == len(occupied_ids)
        and value.get("occupied_primitive_digest") == _digest({
            "schema_version": OCCUPIED_PRIMITIVE_DIGEST_SCHEMA_VERSION,
            "primitive_ids": occupied_ids,
        })
        and isinstance(entries, list)
        and len(entries) == 1
        and entry is not None
        and entry_ids is not None
        and entry_authority is not None
        and sorted(entry_ids) == occupied_ids
        and isinstance(groups, list)
        and len(groups) == 2
        and all(isinstance(group, dict) for group in groups)
        and [group.get("size_class") for group in groups]
        == ["primary", "tolerance"]
        and all(ids is not None for ids in group_primitive_ids)
        and sorted(
            primitive_id
            for ids in group_primitive_ids
            if ids is not None
            for primitive_id in ids
        ) == numeric_ids
        and value.get("consumer_allowed") is False
        and value.get("occupation_digest") == _digest({
            "schema_version": OCCUPATION_DIGEST_SCHEMA_VERSION,
            "occupation": payload,
        })
    ):
        return None
    if not (
        entry is not None
        and entry.get("schema_version") == ENTRY_SCHEMA_VERSION
        and entry.get("glyph_type") == "plus_minus"
        and entry.get("output_text") == "±"
        and entry.get("source_primitive_ops") == ["l", "l", "l"]
        and entry.get("source_primitive_roles")
        == ["upper_bar", "lower_bar", "stem"]
        and entry.get("consumer_allowed") is False
        and entry.get("entry_digest") == _digest({
            key: item
            for key, item in entry.items()
            if key != "entry_digest"
        })
    ):
        return None
    return copy.deepcopy(value)


def _valid_binding_payload(value: Any) -> bool:
    occupied_ids = _canonical_id_list(
        value.get("occupied_primitive_ids")
        if isinstance(value, dict)
        else None
    )
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == BINDING_SCHEMA_VERSION
        and value.get("producer") == PRODUCER
        and type(value.get("page_index")) is int
        and value["page_index"] >= 0
        and type(value.get("phrase_id")) is str
        and bool(value["phrase_id"])
        and all(_valid_sha256(value.get(key)) for key in (
            "window_authority_digest",
            "cross_group_relation_digest",
            "phrase_primitive_manifest_digest",
            "occupation_digest",
            "occupied_primitive_digest",
        ))
        and occupied_ids is not None
        and len(occupied_ids) == 3
        and value.get("occupied_primitive_count") == len(occupied_ids)
        and value.get("occupied_primitive_digest") == _digest({
            "schema_version": OCCUPIED_PRIMITIVE_DIGEST_SCHEMA_VERSION,
            "primitive_ids": occupied_ids,
        })
        and value.get("consumer_allowed") is False
    )


def _authority_from_entry(
    entries: Any,
    *,
    page_index: int,
) -> dict[str, Any] | None:
    if not isinstance(entries, list) or len(entries) != 1:
        return None
    entry = entries[0]
    if not isinstance(entry, dict):
        return None
    authority = {
        "schema_version": "vector_semantic_anchor_authority_v1",
        "producer": "vector_anchor_semantic_bridge_v1",
        "page_index": page_index,
        **{
            key: entry.get(key)
            for key in (
                "glyph_id",
                "glyph_type",
                "output_text",
                "anchor_bbox",
                "detector_source",
                "source_primitive_ids",
                "source_primitive_ops",
                "source_primitive_roles",
            )
        },
        "consumer_allowed": CONSUMER_ALLOWED,
        **{
            key: entry.get(key)
            for key in (
                "topology_proof_schema_version",
                "topology_proof_digest",
                "decimal_quad_id",
                "decimal_authority_digest",
                "pitch_window_source_digest",
                "authority_digest_schema_version",
                "authority_digest",
            )
        },
    }
    return normalize_semantic_anchor_authority(authority)


def _canonical_items(items: list[Any]) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        primitive_id = item.get("item_id")
        op = item.get("op")
        bbox = _canonical_bbox(item.get("bbox"))
        points = _canonical_points(item.get("points"))
        if (
            type(primitive_id) is not str
            or not primitive_id
            or primitive_id in seen
            or type(op) is not str
            or not op
            or not primitive_id.endswith(f"_{op}")
            or bbox is None
            or points is None
            or _points_bbox(points) != bbox
        ):
            return None
        seen.add(primitive_id)
        rows.append({
            "primitive_id": primitive_id,
            "op": op,
            "bbox": bbox,
            "points": points,
        })
    return rows


def _manifest_matches_items(
    manifest: dict[str, Any],
    *,
    item_rows: list[dict[str, Any]],
) -> bool:
    return bool(
        manifest.get("entries")
        == [
            {"primitive_id": row["primitive_id"], "op": row["op"]}
            for row in item_rows
        ]
        and manifest.get("geometry_entries")
        == [
            {
                "primitive_id": row["primitive_id"],
                "bbox": row["bbox"],
            }
            for row in item_rows
        ]
    )


def _authority_manifest_bbox(
    primitive_ids: list[str],
    *,
    items_by_id: dict[str, dict[str, Any]],
) -> dict[str, float]:
    bboxes = [items_by_id[primitive_id]["bbox"] for primitive_id in primitive_ids]
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)
    return {
        "x": round(x0, 3),
        "y": round(y0, 3),
        "w": round(x1 - x0, 3),
        "h": round(y1 - y0, 3),
    }


def _union_main_interval(
    primitive_ids: list[str],
    *,
    items_by_id: dict[str, dict[str, Any]],
    main_unit: tuple[float, float],
) -> list[float] | None:
    intervals = [
        _item_main_interval(
            items_by_id[primitive_id],
            main_unit=main_unit,
        )
        for primitive_id in primitive_ids
    ]
    if any(interval is None for interval in intervals):
        return None
    return [
        round(min(interval[0] for interval in intervals if interval), 6),
        round(max(interval[1] for interval in intervals if interval), 6),
    ]


def _item_main_interval(
    item: dict[str, Any],
    *,
    main_unit: tuple[float, float],
) -> list[float] | None:
    points = item.get("points")
    if not isinstance(points, list) or not points:
        return None
    projected = [
        point[0] * main_unit[0] + point[1] * main_unit[1]
        for point in points
    ]
    if not projected or any(not math.isfinite(value) for value in projected):
        return None
    return [round(min(projected), 6), round(max(projected), 6)]


def _entry_group_binding(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "size_class": group["size_class"],
        "group_digest": group["group_digest"],
        "window_authority_digest": group["window_authority_digest"],
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _canonical_unit(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        return None
    if any(type(value.get(key)) not in {int, float} for key in ("x", "y")):
        return None
    unit = (float(value["x"]), float(value["y"]))
    if (
        any(not math.isfinite(number) for number in unit)
        or abs(math.hypot(*unit) - 1.0) > 1e-6
    ):
        return None
    return unit


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


def _canonical_points(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or not value:
        return None
    points: list[list[float]] = []
    for raw_point in value:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            return None
        if any(type(number) not in {int, float} for number in raw_point):
            return None
        point = [float(raw_point[0]), float(raw_point[1])]
        if any(not math.isfinite(number) for number in point):
            return None
        points.append(point)
    return points


def _points_bbox(points: list[list[float]]) -> list[float]:
    return [
        round(min(point[0] for point in points), 6),
        round(min(point[1] for point in points), 6),
        round(max(point[0] for point in points), 6),
        round(max(point[1] for point in points), 6),
    ]


def _canonical_id_list(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
    ):
        return None
    canonical = sorted(set(value))
    return canonical if value == canonical else None


def _ordered_unique_id_list(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        return None
    return list(value)


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
