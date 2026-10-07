"""Fail-closed vector authority for degree, diameter, and plus/minus anchors.

The bridge never recognizes a character from proximity alone. It binds one
already-detected strict vector glyph to its detector-selected page primitives.
Downstream text transforms may consume only those primitives and must replay
the authority digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from decimal_anchor_authority import (
    is_authoritative_decimal_anchor,
    normalize_decimal_anchor_authority,
)
from degree_polyline_strict import Segment
from plus_minus_strict import (
    ROLE_HULL_MARGIN_SCALE_FACTOR,
    prove_plus_minus_topology,
)
from vector_page_context import VectorPageContext
from vector_pitch_window_authority import normalize_pitch_window_source


SCHEMA_VERSION = "vector_semantic_anchor_authority_v1"
DIGEST_SCHEMA_VERSION = "vector_semantic_anchor_authority_digest_v1"
PRODUCER = "vector_anchor_semantic_bridge_v1"
CONSUMER_ALLOWED = False
COORDINATE_JOIN_EPSILON_PT = 0.001
PLUS_MINUS_AUTHORITY_FIELDS = (
    "topology_proof_schema_version",
    "topology_proof_digest",
    "decimal_quad_id",
    "decimal_authority_digest",
    "pitch_window_source_digest",
)

_GLYPH_CONTRACTS = {
    "degree": {
        "output_text": "°",
        "token_text": "°",
        "detector_source": "degree_polyline_strict",
    },
    "diameter": {
        "output_text": "Ø",
        "token_text": "⌀",
        "detector_source": "extract_diameter_glyphs",
    },
    "plus_minus": {
        "output_text": "±",
        "token_text": "±",
        "detector_source": "decimal_corridor_plus_minus_strict",
    },
}


def build_semantic_anchor_authority(
    *,
    token: Any,
    page_context: VectorPageContext,
    page_index: int,
) -> dict[str, Any] | None:
    """Bind one strict glyph token to an exact, unambiguous primitive set."""
    if not isinstance(page_context, VectorPageContext):
        return None
    try:
        page_context.assert_integrity()
    except ValueError:
        return None
    return _build_semantic_anchor_authority_verified(
        token=token,
        page_context=page_context,
        page_index=page_index,
        context_by_key=_context_primitive_index(page_context),
    )


def build_decimal_corridor_plus_minus_semantic_authority(
    *,
    token: Any,
    page_context: VectorPageContext,
    page_index: int,
    decimal_anchor_authority: Any,
    pitch_window_source: Any,
) -> dict[str, Any] | None:
    """Replay one corridor-scoped ``±`` against one trusted dot carrier."""
    if not isinstance(page_context, VectorPageContext):
        return None
    try:
        page_context.assert_integrity()
    except ValueError:
        return None
    context_by_key = _context_primitive_index(page_context)
    carrier = _trusted_plus_minus_carrier(
        decimal_anchor_authority=decimal_anchor_authority,
        pitch_window_source=pitch_window_source,
        page_context=page_context,
        page_index=page_index,
        context_by_key=context_by_key,
    )
    if carrier is None:
        return None
    return _build_semantic_anchor_authority_verified(
        token=token,
        page_context=page_context,
        page_index=page_index,
        context_by_key=context_by_key,
        trusted_plus_minus_carrier=carrier,
    )


def build_semantic_anchor_authorities(
    *,
    tokens: Any,
    page_context: VectorPageContext,
    page_index: int,
    plus_minus_decimal_carriers_by_token_index: Any = None,
) -> dict[int, dict[str, Any]]:
    """Batch-build authorities after one context integrity verification."""
    if (
        not isinstance(tokens, list)
        or not isinstance(page_context, VectorPageContext)
        or type(page_index) is not int
    ):
        return {}
    try:
        page_context.assert_integrity()
    except ValueError:
        return {}
    context_by_key = _context_primitive_index(page_context)
    carrier_inputs = (
        plus_minus_decimal_carriers_by_token_index
        if isinstance(
            plus_minus_decimal_carriers_by_token_index,
            dict,
        )
        else {}
    )
    authorities: dict[int, dict[str, Any]] = {}
    for token_index, token in enumerate(tokens):
        if (
            not isinstance(token, dict)
            or token.get("glyph_type") not in _GLYPH_CONTRACTS
        ):
            continue
        trusted_carrier = None
        if token.get("glyph_type") == "plus_minus":
            carrier_input = carrier_inputs.get(token_index)
            if (
                isinstance(carrier_input, (list, tuple))
                and len(carrier_input) == 2
            ):
                trusted_carrier = _trusted_plus_minus_carrier(
                    decimal_anchor_authority=carrier_input[0],
                    pitch_window_source=carrier_input[1],
                    page_context=page_context,
                    page_index=page_index,
                    context_by_key=context_by_key,
                )
        authority = _build_semantic_anchor_authority_verified(
            token=token,
            page_context=page_context,
            page_index=page_index,
            context_by_key=context_by_key,
            trusted_plus_minus_carrier=trusted_carrier,
        )
        if authority is not None:
            authorities[token_index] = authority
    return authorities


def _build_semantic_anchor_authority_verified(
    *,
    token: Any,
    page_context: VectorPageContext,
    page_index: int,
    context_by_key: dict[
        tuple[int, int, str],
        tuple[Any, ...],
    ],
    trusted_plus_minus_carrier: tuple[
        dict[str, Any],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any] | None:
    if (
        not isinstance(token, dict)
        or not isinstance(page_context, VectorPageContext)
        or type(page_index) is not int
        or token.get("schema_version") != "vector_glyph_v1"
        or token.get("page_index") != page_index
        or token.get("consumer_allowed") is not False
    ):
        return None
    glyph_type = str(token.get("glyph_type") or "")
    contract = _GLYPH_CONTRACTS.get(glyph_type)
    checks = token.get("strict_checks")
    glyph_id = token.get("token_id")
    bbox = _normalize_bbox(token.get("bbox"))
    if (
        contract is None
        or type(glyph_id) is not str
        or not glyph_id
        or not isinstance(checks, dict)
        or checks.get("consumer_allowed") is not False
        or str(checks.get("source") or "")
        != contract["detector_source"]
        or token.get("text") != contract["token_text"]
        or bbox is None
        or page_context.page_num != page_index + 1
        or (
            glyph_type == "diameter"
            and checks.get("circle_slash_topology") is not True
        )
    ):
        return None
    plus_minus_replay: dict[str, Any] | None = None
    if glyph_type == "plus_minus":
        resolved_plus_minus = _plus_minus_source_primitives(
            token,
            page_context=page_context,
            context_by_key=context_by_key,
            trusted_carrier=trusted_plus_minus_carrier,
        )
        if resolved_plus_minus is None:
            return None
        (
            selected,
            source_primitive_roles,
            plus_minus_replay,
        ) = resolved_plus_minus
    else:
        resolved = _direct_source_primitives(
            token,
            glyph_type=glyph_type,
            context_by_key=context_by_key,
        )
        if resolved is None:
            return None
        selected, source_primitive_roles = resolved
    if selected is None or not selected:
        return None

    detector_selected = (
        [
            item
            for item, role in zip(
                selected,
                source_primitive_roles,
                strict=True,
            )
            if role != "slash_line"
        ]
        if glyph_type == "diameter"
        else selected
    )
    detector_union = _primitive_union_bbox(detector_selected)
    if detector_union is None or detector_union != bbox:
        return None
    authority_bbox = _primitive_union_bbox(selected)
    if authority_bbox is None:
        return None
    if (
        glyph_type == "diameter"
        and not _diameter_source_topology_ok(
            selected,
            roles=source_primitive_roles,
            detector_bbox=bbox,
        )
    ):
        return None
    if (
        glyph_type == "degree"
        and not _degree_source_topology_ok(
            selected,
            checks=checks,
        )
    ):
        return None
    if glyph_type == "plus_minus" and authority_bbox != bbox:
        return None

    source_primitive_ids = [
        page_context.item_id(item) for item in selected
    ]
    if len(source_primitive_ids) != len(set(source_primitive_ids)):
        return None
    source_primitive_ops = [str(item.op) for item in selected]
    if (
        any(not value for value in source_primitive_ops)
        or len(source_primitive_roles) != len(source_primitive_ids)
    ):
        return None

    payload = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "page_index": page_index,
        "glyph_id": glyph_id,
        "glyph_type": glyph_type,
        "output_text": str(contract["output_text"]),
        "anchor_bbox": authority_bbox,
        "detector_source": str(contract["detector_source"]),
        "source_primitive_ids": source_primitive_ids,
        "source_primitive_ops": source_primitive_ops,
        "source_primitive_roles": source_primitive_roles,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if glyph_type == "plus_minus":
        if (
            plus_minus_replay is None
            or trusted_plus_minus_carrier is None
        ):
            return None
        decimal_authority, pitch_window = trusted_plus_minus_carrier
        payload.update({
            "topology_proof_schema_version": str(
                plus_minus_replay["schema_version"]
            ),
            "topology_proof_digest": str(
                plus_minus_replay["topology_digest"]
            ),
            "decimal_quad_id": str(
                decimal_authority["decimal_quad_id"]
            ),
            "decimal_authority_digest": str(
                decimal_authority["authority_digest"]
            ),
            "pitch_window_source_digest": str(
                pitch_window["source_digest"]
            ),
        })
    return {
        **payload,
        "authority_digest_schema_version": DIGEST_SCHEMA_VERSION,
        "authority_digest": _digest({
            "schema_version": DIGEST_SCHEMA_VERSION,
            "authority": payload,
        }),
    }


def normalize_semantic_anchor_authority(
    value: Any,
) -> dict[str, Any] | None:
    """Return a canonical authority only when every field and digest replay."""
    base_payload_keys = (
        "schema_version",
        "producer",
        "page_index",
        "glyph_id",
        "glyph_type",
        "output_text",
        "anchor_bbox",
        "detector_source",
        "source_primitive_ids",
        "source_primitive_ops",
        "source_primitive_roles",
        "consumer_allowed",
    )
    if type(value) is not dict or type(value.get("glyph_type")) is not str:
        return None
    payload_keys = (
        *base_payload_keys,
        *(
            PLUS_MINUS_AUTHORITY_FIELDS
            if value["glyph_type"] == "plus_minus"
            else ()
        ),
    )
    if (
        set(value) != {
            *payload_keys,
            "authority_digest_schema_version",
            "authority_digest",
        }
        or any(
            type(value.get(key)) is not str
            for key in (
                "schema_version",
                "producer",
                "glyph_id",
                "glyph_type",
                "output_text",
                "detector_source",
                "authority_digest_schema_version",
                "authority_digest",
            )
        )
        or type(value.get("page_index")) is not int
        or type(value.get("anchor_bbox")) is not dict
        or type(value.get("source_primitive_ids")) is not list
        or type(value.get("source_primitive_ops")) is not list
        or type(value.get("source_primitive_roles")) is not list
        or value.get("consumer_allowed") is not False
        or (
            value["glyph_type"] == "plus_minus"
            and any(
                type(value.get(key)) is not str or not value[key]
                for key in PLUS_MINUS_AUTHORITY_FIELDS
            )
        )
    ):
        return None
    if any(
        type(item) is not str or not item
        for key in (
            "source_primitive_ids",
            "source_primitive_ops",
            "source_primitive_roles",
        )
        for item in value[key]
    ):
        return None
    normalized_bbox = _normalize_bbox(value["anchor_bbox"])
    if normalized_bbox is None or value["anchor_bbox"] != normalized_bbox:
        return None
    payload = {
        key: value[key]
        for key in payload_keys
    }
    payload["anchor_bbox"] = normalized_bbox
    contract = _GLYPH_CONTRACTS.get(payload["glyph_type"])
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["producer"] != PRODUCER
        or payload["page_index"] < 0
        or not payload["glyph_id"]
        or contract is None
        or payload["output_text"] != contract["output_text"]
        or payload["detector_source"] != contract["detector_source"]
        or payload["anchor_bbox"] is None
        or not payload["source_primitive_ids"]
        or len(payload["source_primitive_ids"])
        != len(set(payload["source_primitive_ids"]))
        or len(payload["source_primitive_ids"])
        != len(payload["source_primitive_ops"])
        or len(payload["source_primitive_ids"])
        != len(payload["source_primitive_roles"])
        or not _valid_authority_topology(payload)
        or (
            payload["glyph_type"] == "plus_minus"
            and (
                payload["topology_proof_schema_version"]
                != "plus_minus_topology_proof_v1"
                or not _valid_sha256(payload["topology_proof_digest"])
                or not _valid_sha256(payload["decimal_authority_digest"])
                or not _valid_sha256(
                    payload["pitch_window_source_digest"]
                )
                or not payload["decimal_quad_id"]
            )
        )
        or value.get("authority_digest_schema_version")
        != DIGEST_SCHEMA_VERSION
        or not _valid_sha256(value.get("authority_digest"))
    ):
        return None
    expected_digest = _digest({
        "schema_version": DIGEST_SCHEMA_VERSION,
        "authority": payload,
    })
    if value.get("authority_digest") != expected_digest:
        return None
    return {
        **payload,
        "authority_digest_schema_version": DIGEST_SCHEMA_VERSION,
        "authority_digest": expected_digest,
    }


def canonical_semantic_anchor_authorities(
    rows: Iterable[Any],
) -> list[dict[str, Any]]:
    """Canonicalize authorities; conflicting glyph/primitive claims vanish."""
    normalized = [
        authority
        for row in rows
        if (authority := normalize_semantic_anchor_authority(row)) is not None
    ]
    glyph_digests: dict[str, set[str]] = {}
    primitive_digests: dict[str, set[str]] = {}
    for authority in normalized:
        digest = str(authority["authority_digest"])
        glyph_digests.setdefault(str(authority["glyph_id"]), set()).add(digest)
        for primitive_id in authority["source_primitive_ids"]:
            primitive_digests.setdefault(str(primitive_id), set()).add(digest)
    conflicting = {
        digest
        for digests in [*glyph_digests.values(), *primitive_digests.values()]
        if len(digests) > 1
        for digest in digests
    }
    by_digest = {
        str(authority["authority_digest"]): authority
        for authority in normalized
        if authority["authority_digest"] not in conflicting
    }
    return [
        by_digest[digest] for digest in sorted(by_digest)
    ]


def semantic_anchor_authorities_from_source_evidence(
    rows: Any,
) -> list[dict[str, Any]]:
    """Collect authorities only from one audited trusted-dot carrier each."""
    if not isinstance(rows, list):
        return []
    nested: list[Any] = []
    for row in rows:
        if not _valid_semantic_source_carrier(row):
            continue
        values = row.get("semantic_anchor_authorities")
        nested.extend(values)
    canonical = canonical_semantic_anchor_authorities(nested)
    digest_counts: dict[str, int] = {}
    for value in nested:
        authority = normalize_semantic_anchor_authority(value)
        if authority is None:
            return []
        digest = str(authority["authority_digest"])
        digest_counts[digest] = digest_counts.get(digest, 0) + 1
    if (
        len(canonical) != len(nested)
        or any(count != 1 for count in digest_counts.values())
    ):
        return []
    return canonical


def _valid_semantic_source_carrier(value: Any) -> bool:
    if (
        type(value) is not dict
        or value.get("schema_version")
        != "vector_phrase_source_candidate_evidence_v1"
        or type(value.get("source_schema_version")) is not str
        or not value.get("source_schema_version")
        or type(value.get("candidate_id")) is not str
        or not value.get("candidate_id")
        or type(value.get("anchor_id")) is not str
        or not value.get("anchor_id")
        or type(value.get("page_index")) is not int
        or value.get("anchor_type") != "dot_strict"
        or value.get("reference_anchor_type") != "dot"
        or type(value.get("axis_angle_deg")) not in {int, float}
        or not math.isfinite(float(value.get("axis_angle_deg")))
        or value.get("axis_angle_source")
        not in {
            "decimal_point_quad",
            "decimal_point_quad.short_axis",
            "oriented_quad:dot_axis_angle",
        }
        or value.get("axis_trusted") is not True
        or value.get("axis_join_status") != "matched"
        or value.get("consumer_allowed") is not False
        or value.get("decimal_anchor_authority_source")
        != "dot_reference"
        or not isinstance(value.get("semantic_anchor_authorities"), list)
        or not value["semantic_anchor_authorities"]
    ):
        return False
    decimal_authority = normalize_decimal_anchor_authority(
        value.get("decimal_anchor_authority")
    )
    if (
        decimal_authority is None
        or not is_authoritative_decimal_anchor(decimal_authority)
        or decimal_authority["page_index"] != value["page_index"]
        or _axis_delta_180(
            float(decimal_authority["baseline_angle_deg"]),
            float(value["axis_angle_deg"]),
        )
        > 0.001
    ):
        return False
    semantic_authorities = canonical_semantic_anchor_authorities(
        value["semantic_anchor_authorities"]
    )
    return bool(
        semantic_authorities == value["semantic_anchor_authorities"]
        and all(
            authority["page_index"] == value["page_index"]
            and decimal_authority["source_primitive_id"]
            not in authority["source_primitive_ids"]
            for authority in semantic_authorities
        )
    )


def _axis_delta_180(left: float, right: float) -> float:
    delta = abs((left - right) % 180.0)
    return min(delta, 180.0 - delta)


def _direct_source_primitives(
    token: dict[str, Any],
    *,
    glyph_type: str,
    context_by_key: dict[
        tuple[int, int, str],
        tuple[Any, ...],
    ],
) -> tuple[list[Any], list[str]] | None:
    kind = (
        "degree_polyline_strict"
        if glyph_type == "degree"
        else "diameter_glyph"
    )
    if glyph_type == "degree":
        expected_roles = ["ring_edge"] * 8
        index_key = "segment_index"
        expected_ops = ["l"] * 8
        allowed_keys = {
            "kind",
            "index",
            "ring_id",
            "size_class",
            "segment_index",
            "drawing_order",
            "item_index",
            "op",
        }
    else:
        expected_roles = ["circle_curve"] * 4 + ["slash_line"]
        index_key = "primitive_index"
        expected_ops = ["c"] * 4 + ["l"]
        allowed_keys = {
            "kind",
            "index",
            "primitive_index",
            "role",
            "drawing_order",
            "item_index",
            "op",
        }
    source_segments = token.get("source_segments")
    if (
        type(source_segments) is not list
        or len(source_segments) != len(expected_roles)
        or any(
            type(row) is not dict
            or row.get("kind") != kind
            or not set(row).issubset(allowed_keys)
            for row in source_segments
        )
    ):
        return None
    rows = source_segments
    if (
        [row.get(index_key) for row in rows]
        != list(range(len(expected_roles)))
        or [row.get("op") for row in rows] != expected_ops
        or [
            (
                "ring_edge"
                if glyph_type == "degree"
                else row.get("role")
            )
            for row in rows
        ]
        != expected_roles
        or any(
            type(row.get("drawing_order")) is not int
            or row["drawing_order"] < 0
            or type(row.get("item_index")) is not int
            or row["item_index"] < 0
            for row in rows
        )
    ):
        return None
    keys = [
        (
            int(row["drawing_order"]),
            int(row["item_index"]),
            str(row["op"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        return None
    if any(len(context_by_key.get(key) or []) != 1 for key in keys):
        return None
    return (
        [context_by_key[key][0] for key in keys],
        expected_roles,
    )


def _context_primitive_index(
    page_context: VectorPageContext,
) -> dict[tuple[int, int, str], tuple[Any, ...]]:
    grouped: dict[tuple[int, int, str], list[Any]] = {}
    for item in page_context.primitives:
        grouped.setdefault(
            (
                int(item.drawing_order),
                int(item.item_index),
                str(item.op),
            ),
            [],
        ).append(item)
    return {
        key: tuple(values)
        for key, values in grouped.items()
    }


def _degree_source_topology_ok(
    items: list[Any],
    *,
    checks: dict[str, Any],
) -> bool:
    required_checks = (
        "four_two_two",
        "C1_period",
        "C2_no_cardinal_adjacency",
        "C3_cardinal_spacing",
        "C4_sl_alternation",
        "C5_turn_symmetry",
    )
    return bool(
        len(items) == 8
        and all(str(item.op) == "l" for item in items)
        and checks.get("n_segments") == 8
        and all(checks.get(key) is True for key in required_checks)
    )


def _diameter_source_topology_ok(
    items: list[Any],
    *,
    roles: list[str],
    detector_bbox: dict[str, float],
) -> bool:
    if (
        len(items) != 5
        or roles != ["circle_curve"] * 4 + ["slash_line"]
        or [str(item.op) for item in items] != ["c"] * 4 + ["l"]
    ):
        return False
    slash = items[-1]
    if len(slash.points) != 2:
        return False
    x0 = float(detector_bbox["x"])
    y0 = float(detector_bbox["y"])
    width = float(detector_bbox["w"])
    height = float(detector_bbox["h"])
    center_x = x0 + width / 2.0
    center_y = y0 + height / 2.0
    start, end = slash.points
    dx = abs(float(end[0]) - float(start[0]))
    dy = abs(float(end[1]) - float(start[1]))
    length = math.hypot(dx, dy)
    midpoint_distance = math.hypot(
        (float(start[0]) + float(end[0])) / 2.0 - center_x,
        (float(start[1]) + float(end[1])) / 2.0 - center_y,
    )
    return bool(
        dx >= 1.0
        and dy >= 1.0
        and midpoint_distance < 3.0
        and length >= max(max(width, height) * 0.6, 3.0)
    )


def _primitive_union_bbox(items: list[Any]) -> dict[str, float] | None:
    if not items:
        return None
    return _normalize_bbox({
        "x": min(float(item.bbox[0]) for item in items),
        "y": min(float(item.bbox[1]) for item in items),
        "w": (
            max(float(item.bbox[2]) for item in items)
            - min(float(item.bbox[0]) for item in items)
        ),
        "h": (
            max(float(item.bbox[3]) for item in items)
            - min(float(item.bbox[1]) for item in items)
        ),
    })


def _plus_minus_source_primitives(
    token: dict[str, Any],
    *,
    page_context: VectorPageContext,
    context_by_key: dict[
        tuple[int, int, str],
        tuple[Any, ...],
    ],
    trusted_carrier: tuple[
        dict[str, Any],
        dict[str, Any],
    ]
    | None,
) -> tuple[list[Any], list[str], dict[str, Any]] | None:
    if trusted_carrier is None:
        return None
    decimal_authority, pitch_window = trusted_carrier
    checks = token.get("strict_checks")
    source_segments = token.get("source_segments")
    topology_proof = token.get("topology_proof")
    expected_roles = ["upper_bar", "lower_bar", "stem"]
    expected_check_keys = {
        "source",
        "subtype",
        "decimal_quad_id",
        "decimal_authority_digest",
        "pitch_window_source_digest",
        "topology_schema_version",
        "topology_digest",
        "scale_baseline",
        "owner_distance_pt",
        "owner_distance_margin_pt",
        "consumer_allowed",
    }
    if (
        type(checks) is not dict
        or set(checks) != expected_check_keys
        or checks.get("subtype") != "decimal_corridor_plus_minus"
        or checks.get("consumer_allowed") is not False
        or checks.get("decimal_quad_id")
        != decimal_authority["decimal_quad_id"]
        or checks.get("decimal_authority_digest")
        != decimal_authority["authority_digest"]
        or checks.get("pitch_window_source_digest")
        != pitch_window["source_digest"]
        or type(source_segments) is not list
        or len(source_segments) != 3
        or any(type(row) is not dict for row in source_segments)
        or type(topology_proof) is not dict
    ):
        return None
    allowed_segment_keys = {
        "kind",
        "segment_index",
        "role",
        "drawing_order",
        "item_index",
        "op",
        "primitive_id",
    }
    if (
        [row.get("segment_index") for row in source_segments]
        != [0, 1, 2]
        or [row.get("role") for row in source_segments]
        != expected_roles
        or any(set(row) != allowed_segment_keys for row in source_segments)
        or any(
            row.get("kind") != "decimal_corridor_plus_minus_line"
            or row.get("op") != "l"
            or type(row.get("drawing_order")) is not int
            or row["drawing_order"] < 0
            or type(row.get("item_index")) is not int
            or row["item_index"] < 0
            or type(row.get("primitive_id")) is not str
            or not row["primitive_id"]
            for row in source_segments
        )
    ):
        return None
    keys = [
        (
            int(row["drawing_order"]),
            int(row["item_index"]),
            "l",
        )
        for row in source_segments
    ]
    if (
        len(keys) != len(set(keys))
        or any(len(context_by_key.get(key) or []) != 1 for key in keys)
    ):
        return None
    selected = [context_by_key[key][0] for key in keys]
    if any(
        page_context.item_id(item) != row["primitive_id"]
        for item, row in zip(selected, source_segments, strict=True)
    ):
        return None

    try:
        axis_angle_deg = float(pitch_window["axis_angle_deg"])
        scale_baseline = max(
            2.5,
            float(pitch_window["char_height_pt"]),
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        type(token.get("axis_angle_deg")) not in {int, float}
        or type(checks.get("scale_baseline")) not in {int, float}
        or _axis_delta_180(
            float(token["axis_angle_deg"]),
            axis_angle_deg,
        )
        > 0.001
        or abs(float(checks["scale_baseline"]) - scale_baseline) > 1e-6
    ):
        return None
    selected_bbox = _primitive_union_bbox(selected)
    if selected_bbox is None:
        return None
    stroke_widths = [
        float(item.stroke_width)
        for item in selected
        if (
            item.stroke_width is not None
            and math.isfinite(float(item.stroke_width))
            and float(item.stroke_width) > 0.0
        )
    ]
    isolation_margin = max(
        1.0,
        scale_baseline * ROLE_HULL_MARGIN_SCALE_FACTOR * 2.0,
        max(stroke_widths, default=0.25) * 4.0,
    )
    all_line_items = [
        item
        for item in page_context.query_bbox((
            float(selected_bbox["x"]) - isolation_margin,
            float(selected_bbox["y"]) - isolation_margin,
            float(selected_bbox["x"] + selected_bbox["w"])
            + isolation_margin,
            float(selected_bbox["y"] + selected_bbox["h"])
            + isolation_margin,
        ))
        if str(item.op) == "l" and len(item.points) == 2
    ]
    all_segments: list[Segment] = []
    segment_by_item_identity: dict[int, Segment] = {}
    primitive_ids_by_segment_id: dict[int, str] = {}
    for segment_id, item in enumerate(all_line_items):
        try:
            p0 = (float(item.points[0][0]), float(item.points[0][1]))
            p1 = (float(item.points[1][0]), float(item.points[1][1]))
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        segment = Segment(
            segment_id=segment_id,
            p0=p0,
            p1=p1,
            length=math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
            angle_deg=math.degrees(
                math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            )
            % 180.0,
            drawing_order=int(item.drawing_order),
            item_index=int(item.item_index),
            stroke_width=(
                None
                if item.stroke_width is None
                else float(item.stroke_width)
            ),
        )
        all_segments.append(segment)
        segment_by_item_identity[id(item)] = segment
        primitive_ids_by_segment_id[segment_id] = page_context.item_id(item)
    role_segments = [
        segment_by_item_identity.get(id(item)) for item in selected
    ]
    if any(segment is None for segment in role_segments):
        return None
    replay = prove_plus_minus_topology(
        role_segments,
        axis_angle_deg=axis_angle_deg,
        scale_baseline=scale_baseline,
        neighbor_segments=all_segments,
        primitive_ids_by_segment_id=primitive_ids_by_segment_id,
    )
    if replay is None:
        return None
    serialized_replay = {
        key: replay[key]
        for key in (
            "schema_version",
            "axis_angle_deg",
            "role_bindings",
            "geometry",
            "topology_digest",
            "consumer_allowed",
        )
    }
    if (
        topology_proof != serialized_replay
        or checks.get("topology_schema_version")
        != replay["schema_version"]
        or checks.get("topology_digest") != replay["topology_digest"]
        or [
            {
                "role": row["role"],
                "primitive_id": row["primitive_id"],
                "drawing_order": row["drawing_order"],
                "item_index": row["item_index"],
            }
            for row in source_segments
        ]
        != replay["role_bindings"]
    ):
        return None
    return selected, expected_roles, replay


def _trusted_plus_minus_carrier(
    *,
    decimal_anchor_authority: Any,
    pitch_window_source: Any,
    page_context: VectorPageContext,
    page_index: int,
    context_by_key: dict[
        tuple[int, int, str],
        tuple[Any, ...],
    ],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    authority = normalize_decimal_anchor_authority(
        decimal_anchor_authority
    )
    window = normalize_pitch_window_source(pitch_window_source)
    if (
        authority is None
        or window is None
        or not is_authoritative_decimal_anchor(authority)
        or authority["page_index"] != page_index
        or window["page_index"] != page_index
        or authority["decimal_quad_id"] != window["decimal_quad_id"]
        or authority["source_primitive_id"]
        != window["source_primitive_id"]
        or authority["authority_digest"]
        != window["source_decimal_authority_digest"]
        or _axis_delta_180(
            float(authority["baseline_angle_deg"]),
            float(window["axis_angle_deg"]),
        )
        > 0.001
    ):
        return None
    key = (
        int(authority["drawing_order"]),
        int(authority["item_index"]),
        "qu",
    )
    matches = context_by_key.get(key) or ()
    if len(matches) != 1:
        return None
    item = matches[0]
    if (
        page_context.item_id(item) != authority["source_primitive_id"]
        or not _item_points_match_quad(
            item.points,
            window["raw_dot_quad"],
        )
    ):
        return None
    return authority, window


def _item_points_match_quad(
    points: Any,
    quad: Any,
) -> bool:
    if (
        not isinstance(points, (list, tuple))
        or len(points) not in {4, 5}
        or not isinstance(quad, list)
        or len(quad) != 4
    ):
        return False
    try:
        if not all(
            abs(float(point[axis]) - float(quad_point[axis]))
            <= COORDINATE_JOIN_EPSILON_PT
            for point, quad_point in zip(points[:4], quad, strict=True)
            for axis in (0, 1)
        ):
            return False
        return bool(
            len(points) == 4
            or all(
                abs(float(points[0][axis]) - float(points[4][axis]))
                <= COORDINATE_JOIN_EPSILON_PT
                for axis in (0, 1)
            )
        )
    except (TypeError, ValueError, IndexError, OverflowError):
        return False


def _normalize_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != {"x", "y", "w", "h"}:
        return None
    if any(
        type(value.get(key)) not in {int, float}
        for key in ("x", "y", "w", "h")
    ):
        return None
    result = {
        key: round(float(value[key]), 3)
        for key in ("x", "y", "w", "h")
    }
    if (
        not all(math.isfinite(number) for number in result.values())
        or result["w"] <= 0.0
        or result["h"] <= 0.0
    ):
        return None
    return result


def _valid_authority_topology(payload: dict[str, Any]) -> bool:
    glyph_type = payload["glyph_type"]
    ops = payload["source_primitive_ops"]
    roles = payload["source_primitive_roles"]
    if glyph_type == "degree":
        return bool(ops == ["l"] * 8 and roles == ["ring_edge"] * 8)
    if glyph_type == "diameter":
        return bool(
            ops == ["c"] * 4 + ["l"]
            and roles == ["circle_curve"] * 4 + ["slash_line"]
        )
    return bool(
        glyph_type == "plus_minus"
        and ops == ["l"] * 3
        and roles == ["upper_bar", "lower_bar", "stem"]
    )


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
