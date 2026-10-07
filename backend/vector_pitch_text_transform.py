"""Auditable R32 text transforms for the formal base pitch adapter.

The adapter derives a narrowly-scoped radius/tolerance repair from the raw
grammar chain, span roles, and authoritative primitive morphology.  It does
not trust a pre-repaired base-reader string.  Every derived output token is
partitioned back to exact vector primitive identities; unsupported repairs
return no ledger and therefore remain fail-closed at the coverage gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from r2p_dimension_format_validator import validate_dimension_text_format
from vector_anchor_semantic_bridge import (
    normalize_semantic_anchor_authority,
    semantic_anchor_authorities_from_source_evidence,
)


CONSUMER_ALLOWED = False
SCHEMA_VERSION = "vector_pitch_text_transform_v1"
BASE_ADAPTER_PRODUCER = "vector_pitch_phrase_reader_v1"
BASE_READER_DEPENDENCY_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_reader_dependency_digest_v1"
)
BASE_ADAPTER_IDENTITY_SCHEMA_VERSION = (
    "vector_pitch_base_core_adapter_identity_v1"
)
BASE_ADAPTER_IDENTITY_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_base_core_adapter_identity_digest_v1"
)
PRIMITIVE_MANIFEST_SCHEMA_VERSION = (
    "vector_pitch_phrase_primitive_manifest_v1"
)
PRIMITIVE_GEOMETRY_MANIFEST_SCHEMA_VERSION = (
    "vector_pitch_phrase_primitive_manifest_v2"
)
PRIMITIVE_MANIFEST_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_phrase_primitive_manifest_digest_v1"
)
TRANSFORM_BINDING_SCHEMA_VERSION = (
    "vector_pitch_text_transform_binding_v1"
)
TRANSFORM_BINDING_DIGEST_SCHEMA_VERSION = (
    "vector_pitch_text_transform_binding_digest_v1"
)
_RADIUS_REPAIR_PATTERN = re.compile(
    r"^±(?P<nominal>\d+\.\d+)±±(?P<tolerance_digit>\d)$"
)


def build_base_pitch_transform_ledger(
    *,
    phrase_id: str,
    final_text: str | None = None,
    items: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    debug: dict[str, Any],
    base_reader_dependency: dict[str, Any] | None = None,
    base_adapter: dict[str, Any] | None = None,
    source_candidate_evidence: list[dict[str, Any]] | None = None,
    candidate_axis_angle_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Build one exact primitive-partitioned repair or return no ledger."""
    raw_text = "".join(
        str(row.get("char") or "")
        for row in spans
        if isinstance(row, dict)
    )
    binding = build_base_pitch_transform_binding(
        phrase_id=phrase_id,
        base_reader_dependency=base_reader_dependency,
        base_adapter=base_adapter,
    )
    semantic_ledger = _build_anchor_semantic_transform_ledger(
        phrase_id=phrase_id,
        final_text=final_text,
        raw_text=raw_text,
        items=items,
        spans=spans,
        drops=drops,
        debug=debug,
        binding=binding,
        source_candidate_evidence=source_candidate_evidence,
        candidate_axis_angle_deg=candidate_axis_angle_deg,
        base_adapter=base_adapter,
    )
    if semantic_ledger:
        return semantic_ledger

    pattern = _derive_radius_pattern(raw_text)
    if (
        type(phrase_id) is not str
        or not phrase_id
        or pattern is None
        or (
            final_text is not None
            and final_text != pattern["final_text"]
        )
        or binding is None
        or drops != []
        or debug.get("consumer_allowed") is not False
        or debug.get("transform_ledger") not in (None, [])
    ):
        return []
    if not _valid_radius_grammar_debug(
        debug,
        raw_text=raw_text,
        derived_final_text=str(pattern["final_text"]),
        span_count=len(spans),
    ):
        return []
    if len(spans) != len(raw_text):
        return []
    prefix_index = int(pattern["prefix_index"])
    separator_index = int(pattern["separator_index"])
    tolerance_source_index = int(pattern["tolerance_source_index"])
    tolerance_digit_index = int(pattern["tolerance_digit_index"])
    item_ops: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            return []
        item_id = item.get("item_id")
        op = item.get("op")
        if (
            type(item_id) is not str
            or not item_id
            or item_id in item_ops
            or op not in {"l", "qu"}
        ):
            return []
        item_ops[item_id] = str(op)

    adapter_manifest = (
        base_adapter.get("primitive_manifest")
        if isinstance(base_adapter, dict)
        else None
    )
    if not isinstance(adapter_manifest, dict) or adapter_manifest.get(
        "entries"
    ) != [
        {"primitive_id": item_id, "op": op}
        for item_id, op in item_ops.items()
    ]:
        return []

    span_groups: list[list[str]] = []
    assigned_ids: list[str] = []
    for row in spans:
        if (
            not isinstance(row, dict)
            or row.get("consumer_allowed") is not False
            or not isinstance(row.get("primitive_ids"), list)
            or not row["primitive_ids"]
            or row.get("slot_item_count") != len(row["primitive_ids"])
            or any(
                type(item_id) is not str or item_id not in item_ops
                for item_id in row["primitive_ids"]
            )
        ):
            return []
        group = list(row["primitive_ids"])
        span_groups.append(group)
        assigned_ids.extend(group)
    if (
        len(assigned_ids) != len(set(assigned_ids))
        or set(assigned_ids) != set(item_ops)
    ):
        return []

    if not _valid_radius_span_roles(
        pattern=pattern,
        spans=spans,
        span_groups=span_groups,
        item_ops=item_ops,
    ):
        return []

    prefix_ids = span_groups[prefix_index]
    tolerance_source_ids = span_groups[tolerance_source_index]
    tolerance_digit_ids = span_groups[tolerance_digit_index]
    zero_ids = [
        item_id
        for item_id in tolerance_source_ids
        if item_ops[item_id] == "l"
    ]
    dot_ids = [
        item_id
        for item_id in tolerance_source_ids
        if item_ops[item_id] == "qu"
    ]
    if (
        not prefix_ids
        or any(item_ops[item_id] != "l" for item_id in prefix_ids)
        or not zero_ids
        or len(dot_ids) != 1
        or not tolerance_digit_ids
        or any(
            item_ops[item_id] != "l"
            for item_id in tolerance_digit_ids
        )
    ):
        return []

    changed_evidence_ids = [
        *prefix_ids,
        *tolerance_source_ids,
        *tolerance_digit_ids,
    ]
    rewrite_spans = [
        {
            "schema_version": "vector_phrase_span_rewrite_v1",
            "source_start": prefix_index,
            "source_end": prefix_index + 1,
            "before_text": raw_text[
                prefix_index:prefix_index + 1
            ],
            "after_text": "R",
            "reason": "radius_prefix_from_two_slot_plus_minus",
            "evidence_primitive_ids": prefix_ids,
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        {
            "schema_version": "vector_phrase_span_rewrite_v1",
            "source_start": tolerance_source_index,
            "source_end": tolerance_digit_index + 1,
            "before_text": raw_text[
                tolerance_source_index:tolerance_digit_index + 1
            ],
            "after_text": f"0.{pattern['tolerance_digit']}",
            "reason": (
                "tolerance_zero_dot_from_plus_minus_digit_pair"
            ),
            "evidence_primitive_ids": [
                *tolerance_source_ids,
                *tolerance_digit_ids,
            ],
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    ]
    output_tokens = [
        _output_token(
            output_index=int(pattern["output_prefix_index"]),
            char="R",
            source_span_index=prefix_index,
            primitive_ids=prefix_ids,
            item_ops=item_ops,
            reason="radius_prefix_from_two_slot_plus_minus",
        ),
        _output_token(
            output_index=int(pattern["output_zero_index"]),
            char="0",
            source_span_index=tolerance_source_index,
            primitive_ids=zero_ids,
            item_ops=item_ops,
            reason="tolerance_zero_from_line_partition",
        ),
        _output_token(
            output_index=int(pattern["output_dot_index"]),
            char=".",
            source_span_index=tolerance_source_index,
            primitive_ids=dot_ids,
            item_ops=item_ops,
            reason="tolerance_dot_from_quad_partition",
        ),
        _output_token(
            output_index=int(pattern["output_digit_index"]),
            char=str(pattern["tolerance_digit"]),
            source_span_index=tolerance_digit_index,
            primitive_ids=tolerance_digit_ids,
            item_ops=item_ops,
            reason="tolerance_digit_preserved",
        ),
    ]
    return [{
        "schema_version": "vector_phrase_text_transform_v1",
        "producer": SCHEMA_VERSION,
        "phrase_id": phrase_id,
        "binding": binding,
        "before_text": raw_text,
        "after_text": str(pattern["final_text"]),
        "reason": "r_context_radius_tolerance_repair",
        "evidence_primitive_ids": changed_evidence_ids,
        "rewrite_spans": rewrite_spans,
        "output_tokens": output_tokens,
        "accepted": True,
        "consumer_allowed": CONSUMER_ALLOWED,
    }]


def _build_anchor_semantic_transform_ledger(
    *,
    phrase_id: str,
    final_text: str | None,
    raw_text: str,
    items: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    debug: dict[str, Any],
    binding: dict[str, Any] | None,
    source_candidate_evidence: list[dict[str, Any]] | None,
    candidate_axis_angle_deg: float | None,
    base_adapter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Rewrite only exact whole span/drop ownership from strict authorities."""
    authorities = semantic_anchor_authorities_from_source_evidence(
        source_candidate_evidence
    )
    if (
        type(phrase_id) is not str
        or not phrase_id
        or not authorities
        or binding is None
        or not spans
        or len(spans) != len(raw_text)
        or debug.get("consumer_allowed") is not False
        or debug.get("transform_ledger") not in (None, [])
        or type(candidate_axis_angle_deg) not in {int, float}
        or not math.isfinite(float(candidate_axis_angle_deg))
    ):
        return []
    normalized_authorities = [
        authority
        for value in authorities
        if (
            authority := normalize_semantic_anchor_authority(value)
        ) is not None
    ]
    if len(normalized_authorities) != len(authorities):
        return []

    item_rows = _semantic_item_rows(items)
    if item_rows is None:
        return []
    item_ops = {
        primitive_id: row["op"]
        for primitive_id, row in item_rows.items()
    }
    adapter_manifest = (
        base_adapter.get("primitive_manifest")
        if isinstance(base_adapter, dict)
        else None
    )
    if (
        not isinstance(adapter_manifest, dict)
        or adapter_manifest.get("schema_version")
        != PRIMITIVE_GEOMETRY_MANIFEST_SCHEMA_VERSION
        or adapter_manifest.get("entries")
        != [
            {"primitive_id": primitive_id, "op": row["op"]}
            for primitive_id, row in item_rows.items()
        ]
        or adapter_manifest.get("geometry_entries")
        != [
            {
                "primitive_id": primitive_id,
                "bbox": [
                    round(float(value), 6)
                    for value in row["bbox"]
                ],
            }
            for primitive_id, row in item_rows.items()
        ]
    ):
        return []

    occupation_present = bool(
        isinstance(base_adapter, dict)
        and "semantic_occupation" in base_adapter
    )
    semantic_occupation = _normalized_adapter_semantic_occupation(
        phrase_id=phrase_id,
        items=items,
        primitive_manifest=adapter_manifest,
        base_adapter=base_adapter,
    )
    if occupation_present and semantic_occupation is None:
        return []

    span_groups = _semantic_assignment_groups(
        spans,
        count_key="slot_item_count",
        item_rows=item_rows,
        require_char=True,
    )
    drop_groups = _semantic_assignment_groups(
        drops,
        count_key="item_count",
        item_rows=item_rows,
        require_char=False,
    )
    if span_groups is None or drop_groups is None:
        return []
    assigned_ids = [
        primitive_id
        for group in [*span_groups, *drop_groups]
        for primitive_id in group
    ]
    occupied_ids = (
        list(semantic_occupation["occupied_primitive_ids"])
        if semantic_occupation is not None
        else []
    )
    assigned_ids.extend(occupied_ids)
    if (
        len(assigned_ids) != len(set(assigned_ids))
        or set(assigned_ids) != set(item_rows)
    ):
        return []

    axis_angle = float(candidate_axis_angle_deg) % 180.0
    span_intervals = [
        _semantic_group_axis_interval(
            group,
            item_rows=item_rows,
            axis_angle_deg=axis_angle,
        )
        for group in span_groups
    ]
    if any(interval is None for interval in span_intervals):
        return []
    resolved_intervals = [
        (float(interval[0]), float(interval[1]))
        for interval in span_intervals
        if interval is not None
    ]
    intervals_are_forward_disjoint = all(
        resolved_intervals[index][1]
        < resolved_intervals[index + 1][0]
        for index in range(len(resolved_intervals) - 1)
    )
    exact_reserved_degree_only = all(
        _exact_reserved_degree_drop_indices(
            authority=authority,
            drops=drops,
        )
        is not None
        for authority in normalized_authorities
    )
    if (
        not intervals_are_forward_disjoint
        and not (
            exact_reserved_degree_only
            and _strict_axis_center_direction(resolved_intervals) is not None
        )
    ):
        return []

    edits: list[dict[str, Any]] = []
    for authority in normalized_authorities:
        authority_ids = list(authority["source_primitive_ids"])
        authority_id_set = set(authority_ids)
        if (
            any(
                primitive_id not in item_rows
                for primitive_id in authority_ids
            )
            or [
                item_ops[primitive_id]
                for primitive_id in authority_ids
            ] != authority["source_primitive_ops"]
            or _semantic_group_bbox(
                authority_ids,
                item_rows=item_rows,
            ) != authority["anchor_bbox"]
        ):
            return []
        anchor_interval = _bbox_axis_interval(
            authority["anchor_bbox"],
            axis_angle_deg=axis_angle,
        )
        if anchor_interval is None:
            return []
        span_indices = [
            index
            for index, group in enumerate(span_groups)
            if set(group) & authority_id_set
        ]
        drop_indices = [
            index
            for index, group in enumerate(drop_groups)
            if set(group) & authority_id_set
        ]
        if span_indices and drop_indices:
            return []

        source_start: int
        source_end: int
        binding_role: str
        reason: str
        source_span_indices: list[int]
        source_drop_indices: list[int]
        source_semantic_occupation_indices: list[int] = []
        if span_indices:
            if (
                len(span_indices) != 1
                or set(span_groups[span_indices[0]])
                != authority_id_set
            ):
                return []
            owner_index = span_indices[0]
            replacement = _semantic_replacement_position(
                authority=authority,
                raw_text=raw_text,
                spans=spans,
                span_intervals=resolved_intervals,
                anchor_interval=anchor_interval,
                owner_index=owner_index,
            )
            if replacement is None:
                return []
            binding_role, reason = replacement
            if spans[owner_index]["char"] == authority["output_text"]:
                continue
            source_start = owner_index
            source_end = owner_index + 1
            source_span_indices = [owner_index]
            source_drop_indices = []
        elif drop_indices:
            if (
                any(
                    not set(drop_groups[index]).issubset(
                        authority_id_set
                    )
                    for index in drop_indices
                )
                or {
                    primitive_id
                    for index in drop_indices
                    for primitive_id in drop_groups[index]
                }
                != authority_id_set
            ):
                return []
            insertion = _semantic_insertion_position(
                authority=authority,
                raw_text=raw_text,
                spans=spans,
                drops=drops,
                span_intervals=resolved_intervals,
                anchor_interval=anchor_interval,
                consumed_drop_indices=drop_indices,
            )
            if insertion is None:
                return []
            source_start, binding_role, reason = insertion
            source_end = source_start
            source_span_indices = []
            source_drop_indices = drop_indices
        elif (
            semantic_occupation is not None
            and authority.get("glyph_type") == "plus_minus"
            and authority.get("authority_digest")
            == semantic_occupation["entries"][0].get(
                "authority_digest"
            )
        ):
            # Occupation intervals originate from live primitive points.  The
            # authority bbox is quantized and remains only a placement/audit
            # interval, so replay the exact point source before comparing.
            occupation_authority_interval = (
                _semantic_group_point_axis_interval(
                    authority_ids,
                    item_rows=item_rows,
                    main_unit=semantic_occupation.get("main_unit"),
                )
            )
            if occupation_authority_interval is None:
                return []
            insertion = _semantic_occupation_insertion_position(
                authority=authority,
                occupation=semantic_occupation,
                raw_text=raw_text,
                spans=spans,
                span_groups=span_groups,
                span_intervals=resolved_intervals,
                anchor_interval=anchor_interval,
                occupation_authority_interval=(
                    occupation_authority_interval
                ),
            )
            if insertion is None:
                return []
            source_start, binding_role, reason = insertion
            source_end = source_start
            source_span_indices = []
            source_drop_indices = []
            source_semantic_occupation_indices = [0]
        else:
            return []
        edits.append({
            "authority": authority,
            "authority_ids": authority_ids,
            "anchor_interval": anchor_interval,
            "source_start": source_start,
            "source_end": source_end,
            "binding_role": binding_role,
            "reason": reason,
            "source_span_indices": source_span_indices,
            "source_drop_indices": source_drop_indices,
            "source_semantic_occupation_indices": (
                source_semantic_occupation_indices
            ),
        })

    if not edits:
        return []
    edits.sort(key=lambda row: (
        int(row["source_start"]),
        int(row["source_end"]),
        str(row["authority"]["authority_digest"]),
    ))
    if (
        len({int(row["source_start"]) for row in edits})
        != len(edits)
        or any(
            int(edits[index - 1]["source_end"])
            > int(edits[index]["source_start"])
            for index in range(1, len(edits))
        )
    ):
        return []

    cursor = 0
    output_parts: list[str] = []
    anchor_bindings: list[dict[str, Any]] = []
    rewrite_spans: list[dict[str, Any]] = []
    output_tokens: list[dict[str, Any]] = []
    evidence_primitive_ids: list[str] = []
    consumed_drop_indices: list[int] = []
    consumed_semantic_occupation_indices: list[int] = []
    for edit in edits:
        source_start = int(edit["source_start"])
        source_end = int(edit["source_end"])
        authority = edit["authority"]
        authority_ids = list(edit["authority_ids"])
        output_parts.append(raw_text[cursor:source_start])
        output_index = sum(len(part) for part in output_parts)
        output_parts.append(str(authority["output_text"]))
        cursor = source_end
        evidence_primitive_ids.extend(authority_ids)
        consumed_drop_indices.extend(edit["source_drop_indices"])
        consumed_semantic_occupation_indices.extend(
            edit["source_semantic_occupation_indices"]
        )
        anchor_binding = {
            "schema_version": "vector_anchor_semantic_binding_v1",
            "glyph_id": str(authority["glyph_id"]),
            "glyph_type": str(authority["glyph_type"]),
            "detector_source": str(authority["detector_source"]),
            "authority_digest": str(authority["authority_digest"]),
            "anchor_bbox": dict(authority["anchor_bbox"]),
            "axis_angle_deg": round(axis_angle, 6),
            "anchor_axis_interval": [
                round(float(edit["anchor_interval"][0]), 6),
                round(float(edit["anchor_interval"][1]), 6),
            ],
            "span_axis_intervals": [
                [round(start, 6), round(end, 6)]
                for start, end in resolved_intervals
            ],
            "source_start": source_start,
            "source_end": source_end,
            "binding_role": str(edit["binding_role"]),
            "source_span_indices": list(
                edit["source_span_indices"]
            ),
            "source_drop_indices": list(
                edit["source_drop_indices"]
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        if edit["source_semantic_occupation_indices"]:
            anchor_binding.update({
                "source_semantic_occupation_indices": list(
                    edit["source_semantic_occupation_indices"]
                ),
                "semantic_occupation_digest": str(
                    semantic_occupation["occupation_digest"]
                ),
            })
        anchor_bindings.append(anchor_binding)
        rewrite_spans.append({
            "schema_version": "vector_phrase_span_rewrite_v1",
            "source_start": source_start,
            "source_end": source_end,
            "before_text": raw_text[source_start:source_end],
            "after_text": str(authority["output_text"]),
            "reason": str(edit["reason"]),
            "evidence_primitive_ids": authority_ids,
            "authority_digest": str(authority["authority_digest"]),
            "consumer_allowed": CONSUMER_ALLOWED,
        })
        output_token = {
            "schema_version": "vector_phrase_output_token_evidence_v1",
            "output_index": output_index,
            "char": str(authority["output_text"]),
            "source_span_indices": list(
                edit["source_span_indices"]
            ),
            "primitive_ids": authority_ids,
            "primitive_ops": [
                item_ops[primitive_id]
                for primitive_id in authority_ids
            ],
            "source_drop_indices": list(
                edit["source_drop_indices"]
            ),
            "glyph_id": str(authority["glyph_id"]),
            "detector_source": str(authority["detector_source"]),
            "authority_digest": str(authority["authority_digest"]),
            "reason": str(edit["reason"]),
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        if edit["source_semantic_occupation_indices"]:
            output_token.update({
                "source_semantic_occupation_indices": list(
                    edit["source_semantic_occupation_indices"]
                ),
                "semantic_occupation_digest": str(
                    semantic_occupation["occupation_digest"]
                ),
            })
        output_tokens.append(output_token)
    output_parts.append(raw_text[cursor:])
    output_text = "".join(output_parts)
    consumed_drop_indices = sorted(consumed_drop_indices)
    consumed_semantic_occupation_indices = sorted(
        consumed_semantic_occupation_indices
    )
    if (
        len(evidence_primitive_ids)
        != len(set(evidence_primitive_ids))
        or consumed_drop_indices
        != sorted(set(consumed_drop_indices))
        or consumed_semantic_occupation_indices
        != sorted(set(consumed_semantic_occupation_indices))
        or (
            semantic_occupation is not None
            and consumed_semantic_occupation_indices != [0]
        )
    ):
        return []
    formal = validate_dimension_text_format(output_text)
    if (
        formal.get("valid") is not True
        or formal.get("text") != output_text
        or (final_text is not None and final_text != output_text)
    ):
        return []
    step = {
        "schema_version": "vector_phrase_text_transform_v1",
        "producer": SCHEMA_VERSION,
        "phrase_id": phrase_id,
        "binding": binding,
        "anchor_bindings": anchor_bindings,
        "before_text": raw_text,
        "after_text": output_text,
        "reason": "anchor_semantic_bridge",
        "evidence_primitive_ids": evidence_primitive_ids,
        "consumed_drop_indices": consumed_drop_indices,
        "rewrite_spans": rewrite_spans,
        "output_tokens": output_tokens,
        "accepted": True,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if semantic_occupation is not None:
        step.update({
            "consumed_semantic_occupation_indices": (
                consumed_semantic_occupation_indices
            ),
            "semantic_occupation_digest": str(
                semantic_occupation["occupation_digest"]
            ),
        })
    return [step]


def _semantic_item_rows(
    items: Any,
) -> dict[str, dict[str, Any]] | None:
    if not isinstance(items, list) or not items:
        return None
    rows: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            return None
        primitive_id = item.get("item_id")
        op = item.get("op")
        bbox = _normalize_item_bbox(item.get("bbox"))
        if (
            type(primitive_id) is not str
            or not primitive_id
            or primitive_id in rows
            or type(op) is not str
            or not op
            or bbox is None
        ):
            return None
        rows[primitive_id] = {"op": op, "bbox": bbox}
        points = _normalize_item_points(item.get("points"))
        if points is not None:
            rows[primitive_id]["points"] = points
    return rows


def _semantic_assignment_groups(
    rows: Any,
    *,
    count_key: str,
    item_rows: dict[str, dict[str, Any]],
    require_char: bool,
) -> list[list[str]] | None:
    if not isinstance(rows, list):
        return None
    groups: list[list[str]] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("consumer_allowed") is not False
            or type(row.get("slot_index")) is not int
            or (require_char and type(row.get("char")) is not str)
            or not isinstance(row.get("primitive_ids"), list)
            or not row["primitive_ids"]
            or row.get(count_key) != len(row["primitive_ids"])
        ):
            return None
        group = list(row["primitive_ids"])
        if (
            len(group) != len(set(group))
            or any(
                type(primitive_id) is not str
                or primitive_id not in item_rows
                for primitive_id in group
            )
        ):
            return None
        groups.append(group)
    return groups


def _semantic_replacement_position(
    *,
    authority: dict[str, Any],
    raw_text: str,
    spans: list[dict[str, Any]],
    span_intervals: list[tuple[float, float]],
    anchor_interval: tuple[float, float],
    owner_index: int,
) -> tuple[str, str] | None:
    slot_indices = [int(row["slot_index"]) for row in spans]
    if (
        owner_index < 0
        or owner_index >= len(spans)
        or slot_indices != sorted(slot_indices)
        or len(slot_indices) != len(set(slot_indices))
        or any(
            not math.isclose(
                float(anchor_interval[bound]),
                float(span_intervals[owner_index][bound]),
                abs_tol=1e-6,
            )
            for bound in (0, 1)
        )
    ):
        return None
    glyph_type = str(authority["glyph_type"])
    if glyph_type == "diameter":
        if owner_index != 0:
            return None
        return "prefix", "diameter_prefix_from_vector_anchor_span"
    if glyph_type == "degree":
        if owner_index != len(spans) - 1:
            return None
        return "suffix", "degree_suffix_from_vector_anchor_span"
    if (
        glyph_type != "plus_minus"
        or owner_index <= 0
        or owner_index >= len(spans) - 1
        or not re.fullmatch(
            r"\d+(?:\.\d+)?",
            raw_text[:owner_index],
        )
        or not re.fullmatch(
            r"\d+(?:\.\d+)?",
            raw_text[owner_index + 1:],
        )
    ):
        return None
    return "separator", "plus_minus_separator_from_vector_anchor_span"


def _normalized_adapter_semantic_occupation(
    *,
    phrase_id: str,
    items: list[dict[str, Any]],
    primitive_manifest: dict[str, Any],
    base_adapter: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Replay an optional occupation against the live adapter context."""
    if not isinstance(base_adapter, dict):
        return None
    value = base_adapter.get("semantic_occupation")
    if value is None:
        return None
    page_index = value.get("page_index") if isinstance(value, dict) else None
    if type(page_index) is not int:
        return None
    try:
        from vector_pitch_semantic_occupation import (
            normalize_vector_pitch_semantic_occupation,
        )
    except ImportError:
        return None
    return normalize_vector_pitch_semantic_occupation(
        value,
        phrase_id=phrase_id,
        page_index=page_index,
        items=items,
        primitive_manifest=primitive_manifest,
        window_authority=base_adapter.get("window_authority"),
    )


def _semantic_occupation_insertion_position(
    *,
    authority: dict[str, Any],
    occupation: dict[str, Any],
    raw_text: str,
    spans: list[dict[str, Any]],
    span_groups: list[list[str]],
    span_intervals: list[tuple[float, float]],
    anchor_interval: tuple[float, float],
    occupation_authority_interval: tuple[float, float],
) -> tuple[int, str, str] | None:
    """Resolve the sole separator boundary from exact occupation groups."""
    entries = occupation.get("entries")
    groups = occupation.get("groups")
    if not (
        authority.get("glyph_type") == "plus_minus"
        and authority.get("output_text") == "±"
        and isinstance(entries, list)
        and len(entries) == 1
        and isinstance(entries[0], dict)
        and isinstance(groups, list)
        and len(groups) == 2
        and len(spans) == len(raw_text) == len(span_groups)
        and len(span_intervals) == len(spans)
    ):
        return None
    entry = entries[0]
    authority_keys = (
        "glyph_id",
        "glyph_type",
        "output_text",
        "anchor_bbox",
        "detector_source",
        "source_primitive_ids",
        "source_primitive_ops",
        "source_primitive_roles",
        "topology_proof_schema_version",
        "topology_proof_digest",
        "decimal_quad_id",
        "decimal_authority_digest",
        "pitch_window_source_digest",
        "authority_digest_schema_version",
        "authority_digest",
    )
    entry_interval = entry.get("absolute_main_interval")
    if (
        any(entry.get(key) != authority.get(key) for key in authority_keys)
        or not isinstance(entry_interval, list)
        or len(entry_interval) != 2
        or any(
            type(value) not in {int, float}
            for value in entry_interval
        )
        or any(
            not math.isclose(
                float(entry_interval[index]),
                float(occupation_authority_interval[index]),
                abs_tol=1e-6,
            )
            for index in (0, 1)
        )
    ):
        return None

    expected_classes = ["primary", "tolerance"]
    if [group.get("size_class") for group in groups] != expected_classes:
        return None
    group_by_class = {
        str(group["size_class"]): group for group in groups
    }
    span_classes: list[str] = []
    assigned_by_class: dict[str, list[str]] = {
        "primary": [],
        "tolerance": [],
    }
    for span, primitive_ids, interval in zip(
        spans,
        span_groups,
        span_intervals,
    ):
        size_class = span.get("window_size_class")
        if size_class not in group_by_class:
            return None
        group = group_by_class[str(size_class)]
        if (
            span.get("window_group_digest") != group.get("group_digest")
            or not set(primitive_ids).issubset(
                set(group.get("primitive_ids") or [])
            )
        ):
            return None
        if size_class == "primary":
            if float(interval[1]) >= float(anchor_interval[0]):
                return None
        elif float(interval[0]) <= float(anchor_interval[1]):
            return None
        span_classes.append(str(size_class))
        assigned_by_class[str(size_class)].extend(primitive_ids)

    if any(
        sorted(assigned_by_class[size_class])
        != sorted(group_by_class[size_class].get("primitive_ids") or [])
        for size_class in expected_classes
    ):
        return None
    boundaries = [
        index
        for index in range(1, len(span_classes))
        if (
            span_classes[:index] == ["primary"] * index
            and span_classes[index:]
            == ["tolerance"] * (len(span_classes) - index)
        )
    ]
    if len(boundaries) != 1:
        return None
    boundary = boundaries[0]
    candidate = f"{raw_text[:boundary]}±{raw_text[boundary:]}"
    formal = validate_dimension_text_format(candidate)
    if formal.get("valid") is not True or formal.get("text") != candidate:
        return None
    return (
        boundary,
        "separator",
        "plus_minus_separator_from_semantic_occupation",
    )


def _semantic_insertion_position(
    *,
    authority: dict[str, Any],
    raw_text: str,
    spans: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    span_intervals: list[tuple[float, float]],
    anchor_interval: tuple[float, float],
    consumed_drop_indices: list[int],
) -> tuple[int, str, str] | None:
    slot_indices = [int(row["slot_index"]) for row in spans]
    drop_slots = [
        int(drops[index]["slot_index"])
        for index in consumed_drop_indices
    ]
    if (
        slot_indices != sorted(slot_indices)
        or len(slot_indices) != len(set(slot_indices))
        or len(drop_slots) != len(set(drop_slots))
    ):
        return None
    glyph_type = str(authority["glyph_type"])
    if glyph_type == "diameter":
        if (
            anchor_interval[1] >= span_intervals[0][0]
            or max(drop_slots) >= slot_indices[0]
        ):
            return None
        return 0, "prefix", "diameter_prefix_from_vector_anchor"
    if glyph_type == "degree":
        exact_reserved_indices = _exact_reserved_degree_drop_indices(
            authority=authority,
            drops=drops,
        )
        axis_direction = _strict_axis_center_direction(span_intervals)
        anchor_center = (
            float(anchor_interval[0]) + float(anchor_interval[1])
        ) / 2.0
        last_span_center = (
            float(span_intervals[-1][0])
            + float(span_intervals[-1][1])
        ) / 2.0
        if exact_reserved_indices is not None:
            if (
                sorted(consumed_drop_indices) != exact_reserved_indices
                or axis_direction is None
                or min(drop_slots) <= slot_indices[-1]
                or not _reserved_degree_suffix_is_local_numeric_run(
                    raw_text=raw_text,
                    span_intervals=span_intervals,
                    anchor_center=anchor_center,
                    axis_direction=axis_direction,
                )
            ):
                return None
            return (
                len(raw_text),
                "suffix",
                "degree_suffix_from_reserved_vector_anchor",
            )
        if (
            anchor_interval[0] <= span_intervals[-1][1]
            or min(drop_slots) <= slot_indices[-1]
        ):
            return None
        return (
            len(raw_text),
            "suffix",
            "degree_suffix_from_vector_anchor",
        )
    boundaries = [
        index
        for index in range(1, len(span_intervals))
        if (
            span_intervals[index - 1][1] < anchor_interval[0]
            and anchor_interval[1] < span_intervals[index][0]
            and slot_indices[index - 1] < min(drop_slots)
            and max(drop_slots) < slot_indices[index]
        )
    ]
    if len(boundaries) != 1:
        return None
    index = boundaries[0]
    if not (
        re.fullmatch(r"\d+(?:\.\d+)?", raw_text[:index])
        and re.fullmatch(r"\d+(?:\.\d+)?", raw_text[index:])
    ):
        return None
    return index, "separator", "plus_minus_separator_from_vector_anchor"


def _exact_reserved_degree_drop_indices(
    *,
    authority: dict[str, Any],
    drops: list[dict[str, Any]],
) -> list[int] | None:
    if (
        authority.get("glyph_type") != "degree"
        or authority.get("output_text") != "°"
        or type(authority.get("glyph_id")) is not str
        or type(authority.get("authority_digest")) is not str
        or not isinstance(authority.get("source_primitive_ids"), list)
        or not authority["source_primitive_ids"]
    ):
        return None
    source_ids = set(authority["source_primitive_ids"])
    matched: list[int] = []
    matched_ids: set[str] = set()
    for index, row in enumerate(drops):
        if (
            not isinstance(row, dict)
            or row.get("decision") != "semantic_anchor_reserved"
            or row.get("reason") != "semantic_anchor_reserved"
            or row.get("glyph_type") != "degree"
            or row.get("glyph_id") != authority["glyph_id"]
            or row.get("semantic_authority_digest")
            != authority["authority_digest"]
            or row.get("consumer_allowed") is not False
            or row.get("slot_count") != 1
            or not isinstance(row.get("primitive_ids"), list)
            or row.get("item_count") != len(row["primitive_ids"])
        ):
            continue
        row_ids = set(row["primitive_ids"])
        if (
            len(row_ids) != len(row["primitive_ids"])
            or not row_ids
            or not row_ids.issubset(source_ids)
            or matched_ids.intersection(row_ids)
        ):
            return None
        matched.append(index)
        matched_ids.update(row_ids)
    return matched if matched and matched_ids == source_ids else None


def _reserved_degree_suffix_is_local_numeric_run(
    *,
    raw_text: str,
    span_intervals: list[tuple[float, float]],
    anchor_center: float,
    axis_direction: float,
) -> bool:
    """Bind a reserved degree ring to one local numeric run.

    The bound is derived from the run itself: the terminal ring must be closer
    than the largest observed character-center step or character axis span.
    This rejects a remote degree authority without introducing a drawing- or
    font-specific distance threshold.
    """
    if (
        not re.fullmatch(r"\d+(?:\.\d+)?", raw_text)
        or len(span_intervals) != len(raw_text)
        or axis_direction not in {-1.0, 1.0}
    ):
        return False
    centers = [
        (float(start) + float(end)) / 2.0
        for start, end in span_intervals
    ]
    local_steps = [
        (centers[index + 1] - centers[index]) * axis_direction
        for index in range(len(centers) - 1)
    ]
    if any(step <= 0.0 for step in local_steps):
        return False
    local_axis_spans = [
        abs(float(end) - float(start))
        for start, end in span_intervals
    ]
    local_reach = max([*local_steps, *local_axis_spans])
    terminal_gap = (
        float(anchor_center) - centers[-1]
    ) * axis_direction
    return 0.0 < terminal_gap < local_reach


def _strict_axis_center_direction(
    intervals: list[tuple[float, float]],
) -> float | None:
    centers = [
        (float(interval[0]) + float(interval[1])) / 2.0
        for interval in intervals
    ]
    if len(centers) < 2:
        return 1.0
    if all(
        centers[index] < centers[index + 1]
        for index in range(len(centers) - 1)
    ):
        return 1.0
    if all(
        centers[index] > centers[index + 1]
        for index in range(len(centers) - 1)
    ):
        return -1.0
    return None


def _semantic_group_axis_interval(
    primitive_ids: list[str],
    *,
    item_rows: dict[str, dict[str, Any]],
    axis_angle_deg: float,
) -> tuple[float, float] | None:
    bbox = _semantic_group_bbox(primitive_ids, item_rows=item_rows)
    return (
        _bbox_axis_interval(bbox, axis_angle_deg=axis_angle_deg)
        if bbox is not None
        else None
    )


def _semantic_group_point_axis_interval(
    primitive_ids: list[str],
    *,
    item_rows: dict[str, dict[str, Any]],
    main_unit: Any,
) -> tuple[float, float] | None:
    """Project live primitive points with the occupation's canonical axis."""
    if (
        not primitive_ids
        or not isinstance(main_unit, dict)
        or set(main_unit) != {"x", "y"}
        or any(
            type(main_unit.get(key)) not in {int, float}
            for key in ("x", "y")
        )
    ):
        return None
    unit = (float(main_unit["x"]), float(main_unit["y"]))
    if (
        any(not math.isfinite(value) for value in unit)
        or abs(math.hypot(*unit) - 1.0) > 1e-6
    ):
        return None
    projected: list[float] = []
    for primitive_id in primitive_ids:
        row = item_rows.get(primitive_id)
        points = row.get("points") if isinstance(row, dict) else None
        if not isinstance(points, list) or not points:
            return None
        projected.extend(
            point[0] * unit[0] + point[1] * unit[1]
            for point in points
        )
    if not projected or any(
        not math.isfinite(value) for value in projected
    ):
        return None
    return round(min(projected), 6), round(max(projected), 6)


def _semantic_group_bbox(
    primitive_ids: list[str],
    *,
    item_rows: dict[str, dict[str, Any]],
) -> dict[str, float] | None:
    if not primitive_ids:
        return None
    bboxes = [item_rows[primitive_id]["bbox"] for primitive_id in primitive_ids]
    x0 = min(float(bbox[0]) for bbox in bboxes)
    y0 = min(float(bbox[1]) for bbox in bboxes)
    x1 = max(float(bbox[2]) for bbox in bboxes)
    y1 = max(float(bbox[3]) for bbox in bboxes)
    return {
        "x": round(x0, 3),
        "y": round(y0, 3),
        "w": round(x1 - x0, 3),
        "h": round(y1 - y0, 3),
    }


def _bbox_axis_interval(
    bbox: Any,
    *,
    axis_angle_deg: float,
) -> tuple[float, float] | None:
    if isinstance(bbox, dict):
        try:
            x0 = float(bbox["x"])
            y0 = float(bbox["y"])
            x1 = x0 + float(bbox["w"])
            y1 = y0 + float(bbox["h"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
    else:
        normalized = _normalize_item_bbox(bbox)
        if normalized is None:
            return None
        x0, y0, x1, y1 = normalized
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    radians = math.radians(axis_angle_deg)
    axis_x = math.cos(radians)
    axis_y = math.sin(radians)
    values = [
        x * axis_x + y * axis_y
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ]
    return min(values), max(values)


def _normalize_item_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not all(math.isfinite(item) for item in (x0, y0, x1, y1))
        or x1 < x0
        or y1 < y0
        or (x1 == x0 and y1 == y0)
    ):
        return None
    return (x0, y0, x1, y1)


def _normalize_item_points(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, list) or not value:
        return None
    points: list[tuple[float, float]] = []
    for raw_point in value:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            return None
        if any(type(number) not in {int, float} for number in raw_point):
            return None
        point = (float(raw_point[0]), float(raw_point[1]))
        if any(not math.isfinite(number) for number in point):
            return None
        points.append(point)
    return points


def build_base_pitch_transform_repairs(
    ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project an authoritative transform ledger into its public audit rows."""
    return [
        {
            "schema_version": "vector_pitch_text_repair_v1",
            "producer": str(step["producer"]),
            "phrase_id": str(step["phrase_id"]),
            "reason": str(step["reason"]),
            "before_text": str(step["before_text"]),
            "after_text": str(step["after_text"]),
            "transform_binding_digest": str(
                step["binding"]["binding_digest"]
            ),
            "accepted": step.get("accepted") is True,
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        for step in ledger
    ]


def base_reader_dependency_digest(value: Any) -> str | None:
    canonical = _canonical_base_reader_dependency(value)
    if canonical is None:
        return None
    return _digest({
        "schema_version": BASE_READER_DEPENDENCY_DIGEST_SCHEMA_VERSION,
        "dependency": canonical,
    })


def build_base_pitch_adapter_identity(
    *,
    phrase_id: str,
    base_reader_dependency: Any,
    primitive_ownership: Any,
    primitive_manifest: Any,
) -> dict[str, Any] | None:
    dependency_digest = base_reader_dependency_digest(
        base_reader_dependency
    )
    ownership_schema = (
        primitive_ownership.get("schema_version")
        if isinstance(primitive_ownership, dict)
        else None
    )
    ownership_assignment_schema = (
        primitive_ownership.get("assignment_digest_schema_version")
        if isinstance(primitive_ownership, dict)
        else None
    )
    valid_ownership_schema = (
        (
            ownership_schema == "vector_pitch_primitive_ownership_v1"
            and ownership_assignment_schema
            == "vector_pitch_primitive_assignment_digest_v1"
        )
        or (
            ownership_schema == "vector_pitch_primitive_ownership_v2"
            and ownership_assignment_schema
            == "vector_pitch_primitive_assignment_digest_v2"
        )
    )
    if (
        type(phrase_id) is not str
        or not phrase_id
        or dependency_digest is None
        or not isinstance(primitive_ownership, dict)
        or not valid_ownership_schema
        or primitive_ownership.get("phrase_id") != phrase_id
        or primitive_ownership.get("status") != "applied"
        or not _valid_sha256(
            primitive_ownership.get("assignment_digest")
        )
        or primitive_ownership.get("consumer_allowed") is not False
        or not valid_phrase_primitive_manifest(
            primitive_manifest,
            phrase_id=phrase_id,
        )
    ):
        return None
    payload = {
        "schema_version": BASE_ADAPTER_IDENTITY_SCHEMA_VERSION,
        "producer": BASE_ADAPTER_PRODUCER,
        "phrase_id": phrase_id,
        "core_schema_version": "r2p_pitch_slot_segmenter_v1",
        "base_reader_dependency_digest_schema_version": (
            BASE_READER_DEPENDENCY_DIGEST_SCHEMA_VERSION
        ),
        "base_reader_dependency_digest": dependency_digest,
        "ownership_assignment_digest_schema_version": (
            str(ownership_assignment_schema)
        ),
        "ownership_assignment_digest": str(
            primitive_ownership["assignment_digest"]
        ),
        "primitive_manifest_digest_schema_version": (
            PRIMITIVE_MANIFEST_DIGEST_SCHEMA_VERSION
        ),
        "primitive_manifest_digest": str(
            primitive_manifest["manifest_digest"]
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        **payload,
        "identity_digest_schema_version": (
            BASE_ADAPTER_IDENTITY_DIGEST_SCHEMA_VERSION
        ),
        "identity_digest": _digest({
            "schema_version": (
                BASE_ADAPTER_IDENTITY_DIGEST_SCHEMA_VERSION
            ),
            "identity": payload,
        }),
    }


def build_base_pitch_transform_binding(
    *,
    phrase_id: str,
    base_reader_dependency: Any,
    base_adapter: Any,
) -> dict[str, Any] | None:
    if not isinstance(base_adapter, dict):
        return None
    ownership = base_adapter.get("primitive_ownership")
    primitive_manifest = base_adapter.get("primitive_manifest")
    expected_identity = build_base_pitch_adapter_identity(
        phrase_id=phrase_id,
        base_reader_dependency=base_reader_dependency,
        primitive_ownership=ownership,
        primitive_manifest=primitive_manifest,
    )
    identity = base_adapter.get("identity")
    if (
        expected_identity is None
        or identity != expected_identity
        or base_adapter.get("schema_version")
        != "vector_pitch_base_core_adapter_v1"
        or base_adapter.get("producer") != BASE_ADAPTER_PRODUCER
        or base_adapter.get("base_pitch_slot_core_connected") is not True
        or base_adapter.get("core_schema_version")
        != "r2p_pitch_slot_segmenter_v1"
        or base_adapter.get("phrase_id") != phrase_id
        or base_adapter.get("shared_context_bbox_query_available") is not True
        or base_adapter.get("complete_repair_retry_released") is not False
        or base_adapter.get("recognition_quality_claim_allowed") is not False
        or base_adapter.get("consumer_allowed") is not False
    ):
        return None
    semantic_occupation = base_adapter.get("semantic_occupation")
    semantic_binding: dict[str, Any] | None = None
    if semantic_occupation is not None:
        try:
            from vector_pitch_semantic_occupation import (
                semantic_occupation_binding,
            )
        except ImportError:
            return None
        semantic_binding = semantic_occupation_binding(
            semantic_occupation
        )
        if (
            semantic_binding is None
            or semantic_binding.get("phrase_id") != phrase_id
            or semantic_binding.get(
                "phrase_primitive_manifest_digest"
            )
            != primitive_manifest.get("manifest_digest")
        ):
            return None
    payload = {
        "schema_version": TRANSFORM_BINDING_SCHEMA_VERSION,
        "producer": SCHEMA_VERSION,
        "phrase_id": phrase_id,
        "base_reader_dependency_digest_schema_version": (
            BASE_READER_DEPENDENCY_DIGEST_SCHEMA_VERSION
        ),
        "base_reader_dependency_digest": str(
            identity["base_reader_dependency_digest"]
        ),
        "base_adapter_identity_digest_schema_version": (
            BASE_ADAPTER_IDENTITY_DIGEST_SCHEMA_VERSION
        ),
        "base_adapter_identity_digest": str(
            identity["identity_digest"]
        ),
        "ownership_assignment_digest_schema_version": str(
            identity["ownership_assignment_digest_schema_version"]
        ),
        "ownership_assignment_digest": str(
            identity["ownership_assignment_digest"]
        ),
        "primitive_manifest_digest_schema_version": (
            PRIMITIVE_MANIFEST_DIGEST_SCHEMA_VERSION
        ),
        "primitive_manifest_digest": str(
            identity["primitive_manifest_digest"]
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if semantic_binding is not None:
        payload.update({
            "semantic_occupation_binding_digest_schema_version": str(
                semantic_binding["binding_digest_schema_version"]
            ),
            "semantic_occupation_binding_digest": str(
                semantic_binding["binding_digest"]
            ),
            "semantic_occupation_digest": str(
                semantic_binding["occupation_digest"]
            ),
        })
    return {
        **payload,
        "binding_digest_schema_version": (
            TRANSFORM_BINDING_DIGEST_SCHEMA_VERSION
        ),
        "binding_digest": _digest({
            "schema_version": TRANSFORM_BINDING_DIGEST_SCHEMA_VERSION,
            "binding": payload,
        }),
    }


def build_phrase_primitive_manifest(
    *,
    phrase_id: str,
    page_context_manifest: Any,
    items: Any,
    include_geometry: bool = False,
) -> dict[str, Any] | None:
    if (
        type(phrase_id) is not str
        or not phrase_id
        or not isinstance(page_context_manifest, dict)
        or page_context_manifest.get("schema_version")
        != "vector_page_context_manifest_v1"
        or not _valid_sha256(
            page_context_manifest.get("primitive_sha256")
        )
        or type(page_context_manifest.get("primitive_count")) is not int
        or page_context_manifest.get("primitive_count") < 0
        or page_context_manifest.get("consumer_allowed") is not False
        or not isinstance(items, list)
        or not items
        or type(include_geometry) is not bool
    ):
        return None
    entries: list[dict[str, str]] = []
    geometry_entries: list[dict[str, Any]] = []
    geometry_complete = True
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        primitive_id = item.get("item_id")
        op = item.get("op")
        if (
            type(primitive_id) is not str
            or not primitive_id
            or primitive_id in seen_ids
            or type(op) is not str
            or not op
            or not primitive_id.endswith(f"_{op}")
        ):
            return None
        seen_ids.add(primitive_id)
        entries.append({"primitive_id": primitive_id, "op": op})
        raw_bbox = item.get("bbox")
        if (
            include_geometry
            and (
                not isinstance(raw_bbox, (list, tuple))
                or len(raw_bbox) != 4
                or any(
                    type(value) not in {int, float}
                    for value in raw_bbox
                )
            )
        ):
            return None
        bbox = _normalize_item_bbox(raw_bbox)
        if bbox is None:
            geometry_complete = False
        else:
            geometry_entries.append({
                "primitive_id": primitive_id,
                "bbox": [round(value, 6) for value in bbox],
            })
    if len(entries) > int(page_context_manifest["primitive_count"]):
        return None
    if (
        include_geometry
        and (
            not geometry_complete
            or len(geometry_entries) != len(entries)
        )
    ):
        return None
    payload = {
        "schema_version": (
            PRIMITIVE_GEOMETRY_MANIFEST_SCHEMA_VERSION
            if include_geometry
            else PRIMITIVE_MANIFEST_SCHEMA_VERSION
        ),
        "producer": BASE_ADAPTER_PRODUCER,
        "phrase_id": phrase_id,
        "page_context_primitive_sha256": str(
            page_context_manifest["primitive_sha256"]
        ),
        "primitive_count": len(entries),
        "entries": entries,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if include_geometry:
        payload["geometry_entries"] = geometry_entries
    return {
        **payload,
        "manifest_digest_schema_version": (
            PRIMITIVE_MANIFEST_DIGEST_SCHEMA_VERSION
        ),
        "manifest_digest": _digest({
            "schema_version": PRIMITIVE_MANIFEST_DIGEST_SCHEMA_VERSION,
            "manifest": payload,
        }),
    }


def valid_phrase_primitive_manifest(
    value: Any,
    *,
    phrase_id: str,
    page_context_manifest: Any | None = None,
) -> bool:
    if not isinstance(value, dict):
        return False
    schema_version = value.get("schema_version")
    if schema_version not in {
        PRIMITIVE_MANIFEST_SCHEMA_VERSION,
        PRIMITIVE_GEOMETRY_MANIFEST_SCHEMA_VERSION,
    }:
        return False
    include_geometry = (
        schema_version == PRIMITIVE_GEOMETRY_MANIFEST_SCHEMA_VERSION
    )
    entries = value.get("entries")
    if not isinstance(entries, list):
        return False
    geometry_entries = value.get("geometry_entries")
    geometry_by_id: dict[str, list[float]] | None = None
    if include_geometry:
        if not isinstance(geometry_entries, list):
            return False
        geometry_by_id = {}
        for row in geometry_entries:
            if not isinstance(row, dict) or set(row) != {
                "primitive_id",
                "bbox",
            }:
                return False
            primitive_id = row.get("primitive_id")
            raw_bbox = row.get("bbox")
            if (
                not isinstance(raw_bbox, list)
                or len(raw_bbox) != 4
                or any(
                    type(value) not in {int, float}
                    for value in raw_bbox
                )
            ):
                return False
            bbox = _normalize_item_bbox(raw_bbox)
            if (
                type(primitive_id) is not str
                or not primitive_id
                or primitive_id in geometry_by_id
                or bbox is None
            ):
                return False
            geometry_by_id[primitive_id] = [
                round(number, 6) for number in bbox
            ]
        if list(geometry_by_id) != [
            row.get("primitive_id")
            for row in entries
            if isinstance(row, dict)
        ]:
            return False
    elif geometry_entries is not None:
        return False
    reconstructed = build_phrase_primitive_manifest(
        phrase_id=phrase_id,
        page_context_manifest=(
            page_context_manifest
            if isinstance(page_context_manifest, dict)
            else {
                "schema_version": "vector_page_context_manifest_v1",
                "primitive_sha256": value.get(
                    "page_context_primitive_sha256"
                ),
                "primitive_count": len(entries),
                "consumer_allowed": False,
            }
        ),
        items=[
            {
                "item_id": row.get("primitive_id"),
                "op": row.get("op"),
                **(
                    {
                        "bbox": geometry_by_id.get(
                            str(row.get("primitive_id"))
                        )
                    }
                    if geometry_by_id is not None
                    else {}
                ),
            }
            if isinstance(row, dict)
            else row
            for row in entries
        ],
        include_geometry=include_geometry,
    )
    return bool(
        reconstructed is not None
        and value == reconstructed
        and (
            not isinstance(page_context_manifest, dict)
            or value.get("page_context_primitive_sha256")
            == page_context_manifest.get("primitive_sha256")
        )
    )


def _canonical_base_reader_dependency(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    canonical = {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "module": value.get("module"),
        "callable": value.get("callable"),
        "core_schema_version": value.get("core_schema_version"),
        "complete_repair_retry_released": value.get(
            "complete_repair_retry_released"
        ),
        "recognition_quality_claim_allowed": value.get(
            "recognition_quality_claim_allowed"
        ),
        "reason": value.get("reason"),
        "consumer_allowed": value.get("consumer_allowed"),
    }
    if canonical != {
        "schema_version": "vector_pitch_reader_dependency_v1",
        "status": "base_pitch_slot_core_connected",
        "module": "r2p_pitch_slot_segmenter",
        "callable": "segment_items_with_pitch_slots",
        "core_schema_version": "r2p_pitch_slot_segmenter_v1",
        "complete_repair_retry_released": False,
        "recognition_quality_claim_allowed": False,
        "reason": "r31_complete_repair_retry_reader_not_released_to_core",
        "consumer_allowed": False,
    }:
        return None
    return canonical


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


def _output_token(
    *,
    output_index: int,
    char: str,
    source_span_index: int,
    primitive_ids: list[str],
    item_ops: dict[str, str],
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "vector_phrase_output_token_evidence_v1",
        "output_index": output_index,
        "char": char,
        "source_span_indices": [source_span_index],
        "primitive_ids": list(primitive_ids),
        "primitive_ops": [item_ops[item_id] for item_id in primitive_ids],
        "reason": reason,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _derive_radius_pattern(
    raw_text: str,
) -> dict[str, Any] | None:
    match = _RADIUS_REPAIR_PATTERN.fullmatch(raw_text)
    if match is None:
        return None
    nominal = str(match.group("nominal"))
    tolerance_digit = str(match.group("tolerance_digit"))
    expected_final = f"R{nominal}±0.{tolerance_digit}"
    nominal_start = int(match.start("nominal"))
    nominal_end = int(match.end("nominal"))
    nominal_dot_index = raw_text.index(
        ".", nominal_start, nominal_end
    )
    tolerance_source_index = int(match.start("tolerance_digit")) - 1
    tolerance_digit_index = int(match.start("tolerance_digit"))
    output_zero_index = len(f"R{nominal}±")
    return {
        "final_text": expected_final,
        "prefix_index": 0,
        "nominal_digit_indices": [
            index
            for index in range(nominal_start, nominal_end)
            if index != nominal_dot_index
        ],
        "nominal_dot_index": nominal_dot_index,
        "separator_index": nominal_end,
        "tolerance_source_index": tolerance_source_index,
        "tolerance_digit_index": tolerance_digit_index,
        "output_prefix_index": 0,
        "output_zero_index": output_zero_index,
        "output_dot_index": output_zero_index + 1,
        "output_digit_index": output_zero_index + 2,
        "tolerance_digit": tolerance_digit,
    }


def _valid_radius_grammar_debug(
    debug: dict[str, Any],
    *,
    raw_text: str,
    derived_final_text: str,
    span_count: int,
) -> bool:
    reading = debug.get("reading_defect_grammar")
    tolerance = debug.get("tolerance_grammar")
    letter = debug.get("letter_glyph_grammar")
    letter_unchanged = bool(
        isinstance(letter, dict)
        and letter.get("text") == raw_text
        and letter.get("changed") is False
        and letter.get("reasons") == []
    )
    legacy_letter_repair = bool(
        isinstance(letter, dict)
        and letter.get("text") == derived_final_text
        and letter.get("changed") is True
        and letter.get("reasons") == [
            "r_context_radius_tolerance_repair"
        ]
    )
    return bool(
        debug.get("raw_predicted_text") == raw_text
        and isinstance(reading, dict)
        and reading.get("schema_version")
        == "r2p_reading_defect_grammar_v1"
        and reading.get("raw_text") == raw_text
        and reading.get("text") == raw_text
        and reading.get("changed") is False
        and reading.get("reasons") == []
        and isinstance(reading.get("evidence"), dict)
        and reading["evidence"].get("span_count") == span_count
        and reading["evidence"].get("drop_count") == 0
        and reading["evidence"].get("orientation") in {"H", "V"}
        and reading.get("consumer_allowed") is False
        and isinstance(tolerance, dict)
        and tolerance.get("schema_version") == "r2p_tolerance_grammar_v1"
        and tolerance.get("raw_text") == raw_text
        and tolerance.get("text") == raw_text
        and tolerance.get("changed") is False
        and tolerance.get("reasons") == []
        and isinstance(tolerance.get("evidence"), dict)
        and tolerance["evidence"].get("span_count") == span_count
        and tolerance["evidence"].get("drop_count") == 0
        and tolerance.get("consumer_allowed") is False
        and isinstance(letter, dict)
        and letter.get("schema_version") == "r2p_letter_glyph_grammar_v1"
        and letter.get("raw_text") == raw_text
        and (letter_unchanged or legacy_letter_repair)
        and isinstance(letter.get("evidence"), dict)
        and letter["evidence"].get("span_count") == span_count
        and letter["evidence"].get("drop_count") == 0
        and letter.get("consumer_allowed") is False
    )


def _valid_radius_span_roles(
    *,
    pattern: dict[str, Any],
    spans: list[dict[str, Any]],
    span_groups: list[list[str]],
    item_ops: dict[str, str],
) -> bool:
    prefix_index = int(pattern["prefix_index"])
    nominal_digit_indices = [
        int(value) for value in pattern["nominal_digit_indices"]
    ]
    nominal_dot_index = int(pattern["nominal_dot_index"])
    separator_index = int(pattern["separator_index"])
    tolerance_source_index = int(pattern["tolerance_source_index"])
    tolerance_digit_index = int(pattern["tolerance_digit_index"])

    if any(
        type(row.get("slot_count")) is not int
        or row["slot_count"] != (2 if index == prefix_index else 1)
        for index, row in enumerate(spans)
    ):
        return False
    if (
        spans[prefix_index].get("decision") != "plus_minus_pitch_slot"
        or any(
            spans[index].get("decision") != "accepted_template"
            for index in nominal_digit_indices
        )
        or spans[nominal_dot_index].get("decision") != "dot_pitch_slot"
        or spans[separator_index].get("decision")
        != "plus_minus_pitch_slot"
        or spans[tolerance_source_index].get("decision")
        != "plus_minus_pitch_slot"
        or spans[tolerance_digit_index].get("decision")
        != "accepted_template"
    ):
        return False

    def ops(index: int) -> list[str]:
        return [item_ops[item_id] for item_id in span_groups[index]]

    prefix_ops = ops(prefix_index)
    nominal_digit_ops = [
        op
        for index in nominal_digit_indices
        for op in ops(index)
    ]
    nominal_dot_ops = ops(nominal_dot_index)
    separator_ops = ops(separator_index)
    tolerance_source_ops = ops(tolerance_source_index)
    tolerance_digit_ops = ops(tolerance_digit_index)
    return bool(
        prefix_ops
        and all(op == "l" for op in prefix_ops)
        and nominal_digit_ops
        and all(op == "l" for op in nominal_digit_ops)
        and nominal_dot_ops == ["qu"]
        and separator_ops
        and all(op == "l" for op in separator_ops)
        and tolerance_source_ops.count("qu") == 1
        and tolerance_source_ops.count("l") >= 1
        and len(tolerance_source_ops)
        == tolerance_source_ops.count("l") + 1
        and tolerance_digit_ops
        and all(op == "l" for op in tolerance_digit_ops)
    )
