"""Route audited vector phrases without granting L3 or consumer access."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from directional_walk.phrase_output import (
    TERMINATION_REASONS as DIRECTIONAL_WALK_TERMINATION_REASONS,
    validate_directional_walk_dump_v1,
)
from vector_phrase_chain_validation import (
    request_local_artifact_safety_reasons,
)


DUMP_SCHEMA_VERSION = "vector_phrase_hypothesis_dump_v1"
ROW_SCHEMA_VERSION = "vector_phrase_hypothesis_v1"
GEOMETRY_GATE_SCHEMA_VERSION = "vector_phrase_geometry_gate_v1"
CONSUMER_ALLOWED = False


def build_vector_phrase_hypothesis_dump(
    *,
    phrase_dump: dict[str, Any],
    pitch_read_dump: dict[str, Any],
    quality_gate_dump: dict[str, Any],
) -> dict[str, Any]:
    """Route post-gate phrases to ordinary, GD&T, or quarantine.

    The authoritative phrase, typed read, and current quality rows determine
    routing.  No serialized source is rebuilt or replayed byte-for-byte as an
    admission proof.  Spatial deduplication and L3 assembly remain separate
    later stages.
    """
    trace_id = str(
        (phrase_dump.get("trace_id") or "")
        if isinstance(phrase_dump, dict)
        else ""
    )
    page_index = (
        phrase_dump.get("page_index")
        if isinstance(phrase_dump, dict)
        else -1
    )
    input_reasons: list[str] = []
    for label, artifact in (
        ("phrase", phrase_dump),
        ("pitch", pitch_read_dump),
        ("quality", quality_gate_dump),
    ):
        if _artifact_safety_reasons(
            artifact,
            path=label,
        ):
            input_reasons.append(f"{label}_artifact_safety_invalid")
    if not (
        isinstance(quality_gate_dump, dict)
        and quality_gate_dump.get("schema_version")
        == "vector_phrase_quality_gate_dump_v1"
        and quality_gate_dump.get("trace_id") == trace_id
        and quality_gate_dump.get("page_index") == page_index
        and quality_gate_dump.get("status") == "ok"
        and quality_gate_dump.get("error") is None
        and isinstance(quality_gate_dump.get("rows"), list)
        and quality_gate_dump.get("ocr_called") is False
        and quality_gate_dump.get("consumer_allowed") is False
    ):
        input_reasons.append("quality_gate_envelope_invalid")
    if not input_reasons:
        input_reasons.extend(_router_input_reasons(
            phrase_dump=phrase_dump,
            pitch_read_dump=pitch_read_dump,
            quality_gate_dump=quality_gate_dump,
        ))
    if input_reasons:
        return _failed_dump(
            trace_id=trace_id,
            page_index=page_index,
            reasons=input_reasons,
        )

    reads_by_phrase_id = _rows_grouped_by_phrase_id(
        pitch_read_dump["reads"]
    )
    gates_by_phrase_id = _rows_grouped_by_phrase_id(
        quality_gate_dump["rows"]
    )
    phrases_by_phrase_id: dict[str, list[dict[str, Any]]] = {}
    manifest = phrase_dump["page_context_manifest"]
    page_width = float(manifest["page_width"])
    page_height = float(manifest["page_height"])
    for phrase in sorted(
        (
            row
            for row in phrase_dump["phrases"]
            if isinstance(row, dict)
        ),
        key=lambda row: str(row.get("phrase_id") or ""),
    ):
        phrase_id = str(phrase.get("phrase_id") or "")
        if _phrase_router_reasons(
            phrase,
            trace_id=trace_id,
            page_index=page_index,
            page_width=page_width,
            page_height=page_height,
        ):
            continue
        phrases_by_phrase_id.setdefault(phrase_id, []).append(phrase)

    hypotheses: list[dict[str, Any]] = []
    for phrase_id in sorted(phrases_by_phrase_id):
        phrases = phrases_by_phrase_id[phrase_id]
        matching_reads = [
            read
            for read in reads_by_phrase_id.get(phrase_id) or []
            if _read_matches_phrase_scope(
                read,
                trace_id=trace_id,
                page_index=page_index,
                phrase_id=phrase_id,
            )
        ]
        matching_gates = gates_by_phrase_id.get(phrase_id) or []
        for ordinal, read in enumerate(matching_reads):
            phrase = phrases[min(ordinal, len(phrases) - 1)]
            gate = (
                matching_gates[ordinal]
                if ordinal < len(matching_gates)
                else {}
            )
            hypotheses.append(_hypothesis_row(
                phrase=phrase,
                read=read,
                gate=gate,
            ))
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": int(page_index),
        "page_context_manifest": copy.deepcopy(
            phrase_dump["page_context_manifest"]
        ),
        "status": "ok",
        "error": None,
        "hypotheses": hypotheses,
        "duplicate_ledger": [],
        "dedup_status": "pending_spatial_value_dedup",
        "semantic_digest": _stable_digest(hypotheses),
        "quality_semantic_digest": str(
            _stable_digest(quality_gate_dump["rows"])
        ),
        "stats": {
            "input_phrase_count": len(hypotheses),
            "three_gates_passed_count": sum(
                row["three_gates_passed"] is True for row in hypotheses
            ),
            "geometry_pass_count": sum(
                row["geometry_gate"]["passed"] is True
                for row in hypotheses
            ),
            "ordinary_candidate_count": sum(
                row["ordinary_candidate"] is True for row in hypotheses
            ),
            "gdt_candidate_count": sum(
                row["gdt_candidate"] is True for row in hypotheses
            ),
            "key_candidate_count": sum(
                row["key_candidate"] is True for row in hypotheses
            ),
            "pre_route_rejected_count": sum(
                row["route"] == "rejected_pre_route"
                for row in hypotheses
            ),
            "geometry_quarantine_count": sum(
                row["route"] == "geometry_quarantine"
                for row in hypotheses
            ),
            "feeds_display_l3_count": 0,
            "consumer_allowed_count": 0,
        },
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _artifact_safety_reasons(
    value: Any,
    *,
    path: str,
) -> list[str]:
    reasons = request_local_artifact_safety_reasons(value, path=path)
    if (
        path == "phrase"
        and isinstance(value, dict)
        and value.get("schema_version") == "r92_directional_walk_dump_v1"
    ):
        reasons = [
            reason
            for reason in reasons
            if not (
                reason.startswith(
                    "non_json_value:phrase.phrases["
                )
                and ".walk.cells[" in reason
                and reason.endswith(
                    ".step_evidence.next_shape_keys:tuple"
                )
            )
        ]
    return reasons


def _router_input_reasons(
    *,
    phrase_dump: dict[str, Any],
    pitch_read_dump: dict[str, Any],
    quality_gate_dump: dict[str, Any],
) -> list[str]:
    """Validate only shared envelopes; row failures never erase their peers."""
    reasons: list[str] = []
    trace_id = str(phrase_dump.get("trace_id") or "")
    page_index = phrase_dump.get("page_index")
    try:
        validate_directional_walk_dump_v1(phrase_dump)
    except (TypeError, ValueError):
        reasons.append("phrase_envelope_invalid")
    if not (
        phrase_dump.get("schema_version")
        == "r92_directional_walk_dump_v1"
        and phrase_dump.get("status") == "ok"
        and phrase_dump.get("error") is None
        and isinstance(phrase_dump.get("phrases"), list)
        and phrase_dump.get("consumer_allowed") is False
    ):
        reasons.append("phrase_envelope_invalid")
    if not (
        pitch_read_dump.get("schema_version")
        == "vector_pitch_phrase_read_dump_v1"
        and pitch_read_dump.get("status") == "ok"
        and pitch_read_dump.get("error") is None
        and isinstance(pitch_read_dump.get("reads"), list)
        and pitch_read_dump.get("ocr_called") is False
        and pitch_read_dump.get("consumer_allowed") is False
    ):
        reasons.append("pitch_envelope_invalid")
    else:
        phrase_by_id = {
            phrase.get("phrase_id"): phrase
            for phrase in phrase_dump.get("phrases") or []
            if isinstance(phrase, dict)
        }
        if any(
            not _native_projection_matches(
                read,
                phrase_by_id.get(read.get("phrase_id")),
            )
            for read in pitch_read_dump["reads"]
        ):
            reasons.append("pitch_projection_invalid")
    if not (
        type(trace_id) is str
        and bool(trace_id)
        and type(page_index) is int
        and page_index >= 0
        and pitch_read_dump.get("trace_id") == trace_id
        and pitch_read_dump.get("page_index") == page_index
        and quality_gate_dump.get("trace_id") == trace_id
        and quality_gate_dump.get("page_index") == page_index
    ):
        reasons.append("router_trace_or_page_invalid")
    manifest = phrase_dump.get("page_context_manifest")
    if not (
        isinstance(manifest, dict)
        and manifest.get("schema_version")
        == "vector_page_context_manifest_v1"
        and type(page_index) is int
        and manifest.get("page_num") == page_index + 1
        and _finite_number(manifest.get("page_width"))
        and float(manifest["page_width"]) > 0.0
        and _finite_number(manifest.get("page_height"))
        and float(manifest["page_height"]) > 0.0
        and manifest.get("consumer_allowed") is False
    ):
        reasons.append("router_page_geometry_invalid")
    return sorted(set(reasons))


def _native_projection_matches(read: Any, phrase: Any) -> bool:
    if not isinstance(read, dict) or not isinstance(phrase, dict):
        return False
    return bool(
        read.get("read_mode") == "upstream_directional_walk_projection"
        and all(
            read.get(read_key) == phrase.get(phrase_key)
            for read_key, phrase_key in (
                ("trace_id", "trace_id"),
                ("page_index", "page_index"),
                ("phrase_id", "phrase_id"),
                ("physical_core_digest", "physical_core_digest"),
                ("text", "text"),
                ("raw_text", "text"),
                ("bbox", "bbox"),
                ("oriented_quad", "oriented_quad"),
                ("primitive_ids", "primitive_run_ids"),
                ("source_anchor_evidence", "source_anchor_evidence"),
                ("axis_angle_deg", "axis_angle_deg"),
                ("axis_angle_source", "axis_angle_source"),
                ("axis_trusted", "axis_trusted"),
            )
        )
    )


def _rows_grouped_by_phrase_id(
    rows: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        phrase_id = row.get("phrase_id")
        if type(phrase_id) is str and phrase_id:
            grouped.setdefault(phrase_id, []).append(row)
    return grouped


def _read_matches_phrase_scope(
    read: Any,
    *,
    trace_id: str,
    page_index: int,
    phrase_id: str,
) -> bool:
    """Bind text to its real request/page/phrase scope, not copied geometry."""
    return bool(
        isinstance(read, dict)
        and read.get("trace_id") == trace_id
        and read.get("page_index") == page_index
        and read.get("phrase_id") == phrase_id
        and read.get("consumer_allowed") is False
        and read.get("ocr_called") is False
    )


def _phrase_router_reasons(
    phrase: dict[str, Any],
    *,
    trace_id: str,
    page_index: int,
    page_width: float,
    page_height: float,
) -> list[str]:
    """Return real row-safety failures, excluding replayed accounting."""
    phrase_id = str(phrase.get("phrase_id") or "")
    reasons: list[str] = []
    if not (
        phrase.get("schema_version") == "r92_directional_walk_phrase_v1"
        and phrase.get("trace_id") == trace_id
        and phrase.get("page_index") == page_index
        and phrase.get("consumer_allowed") is False
        and phrase_id
        and _valid_sha256(phrase.get("physical_core_digest"))
    ):
        reasons.append("phrase_identity_invalid")
    if not (
        phrase.get("axis_trusted") is True
        and _finite_number(phrase.get("axis_angle_deg"))
        and type(phrase.get("axis_angle_source")) is str
        and bool(phrase.get("axis_angle_source"))
        and phrase.get("axis_directionality")
        in {"directed", "undirected_fallback"}
    ):
        reasons.append("phrase_axis_contract_invalid")
    if not _geometry_valid(
        phrase.get("bbox"),
        phrase.get("oriented_quad"),
        page_width=page_width,
        page_height=page_height,
    ):
        reasons.append("phrase_geometry_invalid")
    if not _source_anchor_evidence_valid(
        phrase,
        page_index=page_index,
        page_width=page_width,
        page_height=page_height,
    ):
        reasons.append("source_anchor_evidence_invalid")
    if not _walk_terminations_valid(phrase):
        reasons.append("walk_termination_evidence_invalid")
    if _gdt_evidence_reasons(phrase):
        reasons.append("gdt_route_evidence_invalid")
    return sorted(set(reasons))


def _geometry_valid(
    bbox: Any,
    quad: Any,
    *,
    page_width: float,
    page_height: float,
) -> bool:
    if not _bbox_in_page(
        bbox,
        page_width=page_width,
        page_height=page_height,
    ) or not (
        isinstance(quad, list)
        and len(quad) == 4
        and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(_finite_number(value) for value in point)
            for point in quad
        )
    ):
        return False
    area = abs(sum(
        float(quad[index][0]) * float(quad[(index + 1) % 4][1])
        - float(quad[(index + 1) % 4][0]) * float(quad[index][1])
        for index in range(4)
    )) / 2.0
    return area > 0.0


def _bbox_in_page(
    value: Any,
    *,
    page_width: float,
    page_height: float,
) -> bool:
    if not isinstance(value, dict) or set(value) != {"x", "y", "w", "h"} or not all(
        _finite_number(value.get(key)) for key in ("x", "y", "w", "h")
    ):
        return False
    x = float(value["x"])
    y = float(value["y"])
    width = float(value["w"])
    height = float(value["h"])
    return bool(
        page_width > 0.0
        and page_height > 0.0
        and x >= -1e-6
        and y >= -1e-6
        and width > 0.0
        and height > 0.0
        and x + width <= page_width + 1e-6
        and y + height <= page_height + 1e-6
    )


def _source_anchor_evidence_valid(
    phrase: dict[str, Any],
    *,
    page_index: int,
    page_width: float,
    page_height: float,
) -> bool:
    rows = phrase.get("source_anchor_evidence")
    anchor_ids = phrase.get("anchor_ids")
    primitive_ids = phrase.get("primitive_run_ids")
    if not (
        isinstance(rows, list)
        and bool(rows)
        and isinstance(anchor_ids, list)
        and isinstance(primitive_ids, list)
    ):
        return False
    owned_primitives = set(primitive_ids)
    for row in rows:
        source_primitives = (
            row.get("source_primitive_ids")
            if isinstance(row, dict)
            else None
        )
        source_regions = (
            row.get("source_region_ids")
            if isinstance(row, dict)
            else None
        )
        if not (
            isinstance(row, dict)
            and row.get("schema_version")
            == "r92_source_anchor_evidence_v1"
            and isinstance(row.get("source_schema_version"), str)
            and bool(row.get("source_schema_version"))
            and isinstance(row.get("source_id"), str)
            and bool(row.get("source_id"))
            and row.get("anchor_id") in anchor_ids
            and row.get("page_index") == page_index
            and isinstance(row.get("anchor_type"), str)
            and bool(row.get("anchor_type"))
            and row.get("axis_angle_deg") == phrase.get("axis_angle_deg")
            and row.get("axis_angle_source")
            == phrase.get("axis_angle_source")
            and row.get("axis_directionality")
            == phrase.get("axis_directionality")
            and row.get("axis_trusted") is True
            and isinstance(source_primitives, list)
            and bool(source_primitives)
            and source_primitives == sorted(set(source_primitives))
            and set(source_primitives).issubset(owned_primitives)
            and isinstance(source_regions, list)
            and source_regions == sorted(set(source_regions))
            and _bbox_in_page(
                row.get("anchor_bbox"),
                page_width=page_width,
                page_height=page_height,
            )
            and row.get("consumer_allowed") is False
        ):
            return False
    return True


def _walk_terminations_valid(phrase: dict[str, Any]) -> bool:
    walk = phrase.get("walk")
    terminations = (
        walk.get("terminations")
        if isinstance(walk, dict)
        else None
    )
    if not (
        isinstance(walk, dict)
        and walk.get("schema_version")
        == "r92_directional_walk_evidence_v1"
        and isinstance(walk.get("cells"), list)
        and bool(walk.get("cells"))
        and isinstance(terminations, dict)
        and set(terminations) == {"negative", "positive"}
        and walk.get("consumer_allowed") is False
    ):
        return False
    return all(
        _walk_termination_valid(
            terminations[direction],
            direction=direction,
        )
        for direction in ("negative", "positive")
    )


def _walk_termination_valid(value: Any, *, direction: str) -> bool:
    if not isinstance(value, dict):
        return False
    semantic_digest = value.get("semantic_digest")
    payload = {
        key: item
        for key, item in value.items()
        if key != "semantic_digest"
    }
    return bool(
        value.get("schema_version") == "r92_walk_termination_v1"
        and value.get("direction") == direction
        and value.get("reason") in DIRECTIONAL_WALK_TERMINATION_REASONS
        and type(value.get("attempted_step_index")) is int
        and value["attempted_step_index"] >= 0
        and value.get("consumer_allowed") is False
        and _valid_sha256(semantic_digest)
        and semantic_digest == _stable_digest(payload)
    )


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hypothesis_row(
    *,
    phrase: dict[str, Any],
    read: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    # Page/termination/GD&T safety is owned by the phrase geometry.  A gate row is
    # useful diagnostics, but replaying its copied boolean must not erase a
    # visible read.
    boundary_passed = _basic_phrase_boundary_passed(phrase)
    visible_read = bool(
        isinstance(read.get("text"), str)
        and read["text"]
        and isinstance(read.get("spans"), list)
        and any(
            isinstance(span, dict)
            and isinstance(span.get("char"), str)
            and bool(span["char"])
            for span in read["spans"]
        )
    )
    three_gates_passed = bool(visible_read and boundary_passed)
    geometry_gate = _geometry_gate(
        phrase=phrase,
        read=read,
        three_gates_passed=three_gates_passed,
    )
    gdt_evidence = _gdt_evidence(phrase)
    source_anchor_types = sorted({
        str(row.get("anchor_type") or "")
        for row in phrase.get("source_anchor_evidence") or []
        if isinstance(row, dict) and str(row.get("anchor_type") or "")
    })
    key_candidate = "capsule" in source_anchor_types
    if not three_gates_passed:
        route = "rejected_pre_route"
    elif geometry_gate["passed"] is not True:
        route = "geometry_quarantine"
    elif gdt_evidence["routing_frame_ids"]:
        route = "gdt"
    else:
        route = "ordinary"
    ordinary_candidate = route == "ordinary"
    gdt_candidate = route == "gdt"
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "trace_id": str(phrase.get("trace_id") or ""),
        "page_index": int(phrase.get("page_index")),
        "phrase_id": str(phrase.get("phrase_id") or ""),
        "physical_core_digest": str(
            phrase.get("physical_core_digest") or ""
        ),
        "axis_angle_deg": float(phrase.get("axis_angle_deg")),
        "axis_angle_source": str(phrase.get("axis_angle_source") or ""),
        "axis_trusted": phrase.get("axis_trusted") is True,
        "bbox": copy.deepcopy(dict(phrase.get("bbox") or {})),
        "oriented_quad": copy.deepcopy(
            list(phrase.get("oriented_quad") or [])
        ),
        "text": str(read.get("text") or ""),
        "raw_text": str(read.get("raw_text") or ""),
        "source_anchor_types": source_anchor_types,
        "key_candidate": key_candidate,
        "three_gates_passed": three_gates_passed,
        "quality_gate_digest": str(
            gate.get("gate_digest")
            or _stable_digest({
                "phrase_id": phrase.get("phrase_id"),
                "visible_read": visible_read,
                "boundary_passed": boundary_passed,
            })
        ),
        "first_failed_gate": (
            None
            if three_gates_passed
            else "boundary"
            if not boundary_passed
            else "coverage"
        ),
        "geometry_gate": geometry_gate,
        "gdt_route_evidence": gdt_evidence,
        "route": route,
        "ordinary_candidate": ordinary_candidate,
        "gdt_candidate": gdt_candidate,
        "eligible_for_spatial_dedup": bool(
            ordinary_candidate or gdt_candidate
        ),
        "dedup_status": "pending_spatial_value_dedup",
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _basic_phrase_boundary_passed(phrase: dict[str, Any]) -> bool:
    overlap = phrase.get("gdt_overlap")
    return bool(
        _walk_terminations_valid(phrase)
        and isinstance(overlap, dict)
        and not list(overlap.get("boundary_frame_ids") or [])
        and overlap.get("crosses_frame_boundary") is False
        and overlap.get("consumer_allowed") is False
    )


def _geometry_gate(
    *,
    phrase: dict[str, Any],
    read: dict[str, Any],
    three_gates_passed: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not three_gates_passed:
        reasons.append("pre_route_gate_failed")
    if not _walk_terminations_valid(phrase):
        reasons.append("walk_termination_evidence_invalid")
    if _gdt_evidence_reasons(phrase):
        reasons.append("gdt_route_evidence_invalid")
    canonical_reasons = sorted(set(reasons))
    return {
        "schema_version": GEOMETRY_GATE_SCHEMA_VERSION,
        "status": "pass" if not canonical_reasons else "fail_closed",
        "passed": not canonical_reasons,
        "reasons": canonical_reasons,
        "evidence": {
            "bbox_equal": read.get("bbox") == phrase.get("bbox"),
            "oriented_quad_equal": (
                read.get("oriented_quad") == phrase.get("oriented_quad")
            ),
            "termination_evidence_valid": _walk_terminations_valid(phrase),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _gdt_evidence_reasons(phrase: dict[str, Any]) -> list[str]:
    value = phrase.get("gdt_overlap")
    if not isinstance(value, dict):
        return ["gdt_overlap_missing"]
    reasons: list[str] = []
    for key in ("frame_ids", "inside_frame_ids", "boundary_frame_ids"):
        raw = value.get(key)
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item for item in raw or []
        ):
            reasons.append(f"{key}_invalid")
    if value.get("consumer_allowed") is not False:
        reasons.append("gdt_consumer_flag_invalid")
    return reasons


def _gdt_evidence(phrase: dict[str, Any]) -> dict[str, Any]:
    value = phrase.get("gdt_overlap")
    if not isinstance(value, dict):
        value = {}
    frame_ids = sorted({
        str(item)
        for item in value.get("frame_ids") or []
        if str(item)
    })
    ownership_frame_ids = sorted({
        str(item)
        for item in phrase.get("gdt_ownership_frame_ids") or []
        if str(item)
    })
    inside_frame_ids = sorted({
        str(item)
        for item in value.get("inside_frame_ids") or []
        if str(item)
    })
    boundary_frame_ids = sorted({
        str(item)
        for item in value.get("boundary_frame_ids") or []
        if str(item)
    })
    routing_frame_ids = sorted(
        set(frame_ids)
        | set(inside_frame_ids)
        | set(boundary_frame_ids)
        | set(ownership_frame_ids)
    )
    return {
        "frame_ids": frame_ids,
        "ownership_frame_ids": ownership_frame_ids,
        "routing_frame_ids": routing_frame_ids,
        "inside_frame_ids": inside_frame_ids,
        "boundary_frame_ids": boundary_frame_ids,
        "crosses_frame_boundary": bool(boundary_frame_ids),
        "ordinary_eligible": not routing_frame_ids,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _failed_dump(
    *,
    trace_id: str,
    page_index: Any,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": str(trace_id),
        "page_index": page_index if type(page_index) is int else -1,
        "status": "fail_closed",
        "error": {
            "schema_version": "vector_phrase_hypothesis_error_v1",
            "code": "invalid_phrase_hypothesis_input",
            "reasons": sorted(set(reasons)),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "hypotheses": [],
        "duplicate_ledger": [],
        "dedup_status": "not_run",
        "semantic_digest": _stable_digest([]),
        "stats": {
            "input_phrase_count": 0,
            "three_gates_passed_count": 0,
            "geometry_pass_count": 0,
            "ordinary_candidate_count": 0,
            "gdt_candidate_count": 0,
            "key_candidate_count": 0,
            "pre_route_rejected_count": 0,
            "geometry_quarantine_count": 0,
            "feeds_display_l3_count": 0,
            "consumer_allowed_count": 0,
        },
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
