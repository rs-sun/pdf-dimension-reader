"""Independent dump-only quality gates for one vector pitch phrase read."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from decimal_anchor_authority import (
    is_authoritative_decimal_anchor,
    normalize_decimal_anchor_authority,
)
from directional_walk.phrase_output import (
    TERMINATION_REASONS as DIRECTIONAL_WALK_TERMINATION_REASONS,
    validate_directional_walk_dump_v1,
)
from dimension_syntax import classify_dimension_syntax
from r2p_dimension_format_validator import validate_dimension_text_format
from vector_anchor_semantic_bridge import (
    canonical_semantic_anchor_authorities,
    normalize_semantic_anchor_authority,
    semantic_anchor_authorities_from_source_evidence,
)
from vector_phrase_chain_validation import (
    request_local_artifact_safety_reasons,
)
from vector_pitch_anchored_lane import (
    bind_anchored_lane_exclusions,
    normalize_anchored_local_lane,
)
from vector_pitch_anchored_character import (
    normalize_anchored_character_partition,
)
from vector_capsule_topology import (
    vector_capsule_topology_identity_reasons,
)
from vector_pitch_capsule_partition import (
    normalize_capsule_phrase_authority_v1,
    vector_pitch_capsule_partition_identity_reasons,
)
from vector_pitch_primitive_partition import (
    build_pitch_primitive_partition,
    build_pitch_primitive_partition_v2,
    normalize_pitch_primitive_partition,
    normalize_pitch_primitive_partition_v2,
)
from vector_pitch_semantic_occupation import (
    normalize_vector_pitch_semantic_occupation,
    semantic_occupation_binding,
)
from vector_pitch_window_authority import (
    normalize_pitch_window_authority,
    normalize_pitch_window_source,
)
from vector_pitch_text_transform import (
    BASE_ADAPTER_PRODUCER,
    build_base_pitch_adapter_identity,
    build_base_pitch_transform_ledger,
    build_base_pitch_transform_repairs,
    valid_phrase_primitive_manifest,
)


SCHEMA_VERSION = "vector_phrase_gate_decision_v1"
QUALITY_DUMP_SCHEMA_VERSION = "vector_phrase_quality_gate_dump_v1"
COVERAGE_SCHEMA_VERSION = "vector_phrase_coverage_gate_v1"
FORMAT_SCHEMA_VERSION = "vector_phrase_format_gate_v1"
BOUNDARY_SCHEMA_VERSION = "vector_phrase_boundary_gate_v1"
PHRASE_COMPLETE_PRODUCER = "vector_phrase_gate_v1"
CONSUMER_ALLOWED = False
GROUPED_WINDOW_ALGORITHM = "grouped_window_authority_pitch_slots"


def build_vector_phrase_gate_dump(
    *,
    phrase_dump: dict[str, Any],
    pitch_read_dump: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every typed read against one authoritative phrase boundary."""
    try:
        reasons = _quality_dump_input_reasons(
            phrase_dump,
            pitch_read_dump,
        )
    except (
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        reasons = ["quality_gate_input_validation_exception"]
    trace_id = str(phrase_dump.get("trace_id") or "") if isinstance(phrase_dump, dict) else ""
    page_index = phrase_dump.get("page_index") if isinstance(phrase_dump, dict) else None
    if reasons:
        return _failed_quality_dump(
            trace_id=trace_id,
            page_index=page_index,
            reasons=reasons,
        )

    reads_by_phrase_id: dict[str, list[dict[str, Any]]] = {}
    for read in pitch_read_dump["reads"]:
        if not isinstance(read, dict):
            continue
        read_phrase_id = read.get("phrase_id")
        if type(read_phrase_id) is str and read_phrase_id:
            reads_by_phrase_id.setdefault(read_phrase_id, []).append(read)
    phrases_by_phrase_id: dict[str, list[dict[str, Any]]] = {}
    for phrase in phrase_dump["phrases"]:
        if not isinstance(phrase, dict):
            continue
        phrase_id = phrase.get("phrase_id")
        if (
            type(phrase_id) is str
            and phrase_id
            and not _phrase_row_contract_reasons(
                phrase,
                trace_id=trace_id,
                page_index=page_index,
            )
        ):
            phrases_by_phrase_id.setdefault(phrase_id, []).append(phrase)
    rows: list[dict[str, Any]] = []
    for phrase_id in sorted(phrases_by_phrase_id):
        phrases = phrases_by_phrase_id[phrase_id]
        matching_reads = reads_by_phrase_id.get(phrase_id) or []
        for ordinal, source_read in enumerate(matching_reads):
            phrase = phrases[min(ordinal, len(phrases) - 1)]
            read = copy.deepcopy(source_read)
            # Boundary belongs to the authoritative phrase geometry.  The
            # copy on a reader row is diagnostic, never an equality
            # admission.  Extra reads remain visible and are left for the
            # one final position deduplication stage.
            read["boundary"] = _authoritative_boundary_payload(phrase)
            rows.append(evaluate_vector_phrase_gates(
                read,
                reader_dependency=pitch_read_dump.get(
                    "reader_dependency"
                ),
                page_context_manifest=phrase_dump.get(
                    "page_context_manifest"
                ),
                expected_trace_id=trace_id,
                expected_page_index=page_index,
                expected_phrase_id=phrase_id,
                authoritative_phrase=phrase,
            ))
    combination_counts = Counter(
        "C{coverage}_F{format}_B{boundary}".format(
            coverage=int(row["coverage_gate"]["passed"] is True),
            format=int(row["format_gate"]["passed"] is True),
            boundary=int(row["boundary_gate"]["passed"] is True),
        )
        for row in rows
    )
    first_failed_counts = Counter(
        str(row["first_failed_gate"])
        for row in rows
        if row["first_failed_gate"] is not None
    )
    return {
        "schema_version": QUALITY_DUMP_SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": int(page_index),
        "page_context_manifest": copy.deepcopy(
            phrase_dump["page_context_manifest"]
        ),
        "status": "ok",
        "error": None,
        "rows": rows,
        "duplicate_ledger": [],
        "stats": {
            "input_phrase_count": len(phrase_dump["phrases"]),
            "input_read_count": len(pitch_read_dump["reads"]),
            "output_row_count": len(rows),
            "coverage_pass_count": sum(
                row["coverage_gate"]["passed"] is True for row in rows
            ),
            "format_pass_count": sum(
                row["format_gate"]["passed"] is True for row in rows
            ),
            "boundary_pass_count": sum(
                row["boundary_gate"]["passed"] is True for row in rows
            ),
            "phrase_complete_count": sum(
                row["phrase_complete"] is True for row in rows
            ),
            "gate_combination_counts": dict(sorted(combination_counts.items())),
            "first_failed_gate_counts": dict(sorted(first_failed_counts.items())),
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest(rows),
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _quality_dump_input_reasons(
    phrase_dump: Any,
    pitch_read_dump: Any,
) -> list[str]:
    if not isinstance(phrase_dump, dict) or not isinstance(pitch_read_dump, dict):
        return ["quality_gate_input_not_object"]
    reasons: list[str] = []
    try:
        validate_directional_walk_dump_v1(phrase_dump)
    except (TypeError, ValueError):
        reasons.append("phrase_dump_contract_invalid")
    if (
        phrase_dump.get("schema_version") != "r92_directional_walk_dump_v1"
        or phrase_dump.get("status") != "ok"
        or phrase_dump.get("error") is not None
        or not isinstance(phrase_dump.get("phrases"), list)
    ):
        reasons.append("phrase_dump_contract_invalid")
    if (
        pitch_read_dump.get("schema_version")
        != "vector_pitch_phrase_read_dump_v1"
        or pitch_read_dump.get("status") != "ok"
        or pitch_read_dump.get("error") is not None
        or not isinstance(pitch_read_dump.get("reads"), list)
        or pitch_read_dump.get("feeds_display_l3") is not False
        or pitch_read_dump.get("ocr_called") is not False
    ):
        reasons.append("pitch_read_dump_contract_invalid")
    trace_value = phrase_dump.get("trace_id")
    trace_id = trace_value if isinstance(trace_value, str) else ""
    page_index = phrase_dump.get("page_index")
    if (
        not trace_id
        or type(trace_value) is not str
        or pitch_read_dump.get("trace_id") != trace_id
        or type(page_index) is not int
        or page_index < 0
        or pitch_read_dump.get("page_index") != page_index
    ):
        reasons.append("quality_gate_trace_or_page_mismatch")
    phrase_manifest = phrase_dump.get("page_context_manifest")
    pitch_manifest = pitch_read_dump.get("page_context_manifest")
    if not (
        _valid_page_context_geometry_manifest(
            phrase_manifest,
            page_index=page_index,
        )
        and _valid_page_context_geometry_manifest(
            pitch_manifest,
            page_index=page_index,
        )
    ):
        reasons.append("page_context_manifest_invalid")
    elif any(
        phrase_manifest.get(key) != pitch_manifest.get(key)
        for key in (
            "schema_version",
            "pdf_stem",
            "page_num",
            "page_width",
            "page_height",
            "consumer_allowed",
        )
    ):
        reasons.append("page_context_manifest_mismatch")
    if (
        _recursive_invalid_consumer_count(phrase_dump)
        or _recursive_invalid_consumer_count(pitch_read_dump)
    ):
        reasons.append("quality_gate_consumer_flag_invalid")
    if (
        _recursive_true_key_count(phrase_dump, "ocr_called")
        or _recursive_true_key_count(pitch_read_dump, "ocr_called")
    ):
        reasons.append("quality_gate_ocr_flag_invalid")
    if (
        _recursive_true_key_count(phrase_dump, "feeds_display_l3")
        or _recursive_true_key_count(pitch_read_dump, "feeds_display_l3")
    ):
        reasons.append("quality_gate_l3_flag_invalid")
    for label, artifact in (
        ("phrase", phrase_dump),
        ("pitch", pitch_read_dump),
    ):
        if _artifact_safety_reasons(
            artifact,
            path=label,
        ):
            reasons.append(f"{label}_artifact_safety_invalid")

    return sorted(set(reasons))


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


def _failed_quality_dump(
    *,
    trace_id: str,
    page_index: Any,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": QUALITY_DUMP_SCHEMA_VERSION,
        "trace_id": str(trace_id),
        "page_index": page_index if type(page_index) is int else -1,
        "status": "fail_closed",
        "error": {
            "schema_version": "vector_phrase_quality_gate_error_v1",
            "code": "invalid_quality_gate_input",
            "reasons": sorted(set(reasons)),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "rows": [],
        "duplicate_ledger": [],
        "stats": {
            "input_phrase_count": 0,
            "input_read_count": 0,
            "output_row_count": 0,
            "coverage_pass_count": 0,
            "format_pass_count": 0,
            "boundary_pass_count": 0,
            "phrase_complete_count": 0,
            "gate_combination_counts": {},
            "first_failed_gate_counts": {},
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest([]),
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _phrase_row_contract_reasons(
    phrase: Any,
    *,
    trace_id: str,
    page_index: Any,
) -> list[str]:
    if not isinstance(phrase, dict):
        return ["phrase_row_not_object"]
    reasons: list[str] = []
    if phrase.get("schema_version") != "r92_directional_walk_phrase_v1":
        reasons.append("phrase_row_schema_invalid")
    if (
        type(phrase.get("trace_id")) is not str
        or not phrase["trace_id"]
        or phrase["trace_id"] != trace_id
    ):
        reasons.append("phrase_row_trace_invalid")
    if type(phrase.get("phrase_id")) is not str or not phrase["phrase_id"]:
        reasons.append("phrase_row_phrase_id_invalid")
    if (
        type(phrase.get("page_index")) is not int
        or phrase["page_index"] < 0
        or phrase["page_index"] != page_index
    ):
        reasons.append("phrase_row_page_index_invalid")
    return reasons


def _authoritative_boundary_payload(phrase: dict[str, Any]) -> dict[str, Any]:
    walk = phrase.get("walk")
    terminations = (
        walk.get("terminations")
        if isinstance(walk, dict)
        else None
    )
    return {
        "schema_version": "r92_directional_walk_boundary_v1",
        "boundary_producer": "r92_directional_walk_phrase_v1",
        "negative_termination": copy.deepcopy(
            terminations.get("negative")
            if isinstance(terminations, dict)
            else None
        ),
        "positive_termination": copy.deepcopy(
            terminations.get("positive")
            if isinstance(terminations, dict)
            else None
        ),
        "gdt_overlap": copy.deepcopy(phrase.get("gdt_overlap")),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _valid_page_context_geometry_manifest(
    value: Any,
    *,
    page_index: Any,
) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version")
        == "vector_page_context_manifest_v1"
        and isinstance(value.get("pdf_stem"), str)
        and bool(value.get("pdf_stem"))
        and type(page_index) is int
        and type(value.get("page_num")) is int
        and value.get("page_num") == page_index + 1
        and _positive_number(value.get("page_width"))
        and _positive_number(value.get("page_height"))
        and value.get("consumer_allowed") is False
    )


def _valid_page_context_manifest(value: Any, *, page_index: Any) -> bool:
    expected_keys = {
        "schema_version",
        "pdf_stem",
        "page_num",
        "page_width",
        "page_height",
        "primitive_count",
        "primitive_sha256",
        "consumer_allowed",
    }
    return bool(
        isinstance(value, dict)
        and set(value) == expected_keys
        and value.get("schema_version") == "vector_page_context_manifest_v1"
        and isinstance(value.get("pdf_stem"), str)
        and bool(value.get("pdf_stem"))
        and type(page_index) is int
        and type(value.get("page_num")) is int
        and value.get("page_num") == page_index + 1
        and _positive_number(value.get("page_width"))
        and _positive_number(value.get("page_height"))
        and _strict_nonnegative_int(value.get("primitive_count")) is not None
        and _valid_sha256(value.get("primitive_sha256"))
        and value.get("consumer_allowed") is False
    )


def _valid_base_reader_dependency(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version")
        == "vector_pitch_reader_dependency_v1"
        and value.get("status") == "base_pitch_slot_core_connected"
        and value.get("module") == "r2p_pitch_slot_segmenter"
        and value.get("callable") == "segment_items_with_pitch_slots"
        and value.get("core_schema_version")
        == "r2p_pitch_slot_segmenter_v1"
        and value.get("complete_repair_retry_released") is False
        and value.get("recognition_quality_claim_allowed") is False
        and value.get("consumer_allowed") is False
    )


def _valid_base_adapter_identity(
    read: dict[str, Any],
    *,
    phrase_id: str,
) -> bool:
    debug = read.get("reader_debug")
    adapter = (
        debug.get("r32_base_core_adapter")
        if isinstance(debug, dict)
        else None
    )
    return bool(
        isinstance(adapter, dict)
        and adapter.get("schema_version")
        == "vector_pitch_base_core_adapter_v1"
        and adapter.get("base_pitch_slot_core_connected") is True
        and adapter.get("core_schema_version")
        == "r2p_pitch_slot_segmenter_v1"
        and adapter.get("phrase_id") == phrase_id
        and adapter.get("shared_context_bbox_query_available") is True
        and adapter.get("complete_repair_retry_released") is False
        and adapter.get("recognition_quality_claim_allowed") is False
        and adapter.get("consumer_allowed") is False
    )


def evaluate_vector_phrase_gates(
    read: dict[str, Any],
    *,
    reader_dependency: Any = None,
    page_context_manifest: Any = None,
    expected_trace_id: str | None = None,
    expected_page_index: int | None = None,
    expected_phrase_id: str | None = None,
    authoritative_phrase: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate coverage, format, and boundary without allowing one to imply another."""
    trace_id = str(read.get("trace_id") or "") if isinstance(read, dict) else ""
    phrase_id = str(read.get("phrase_id") or "") if isinstance(read, dict) else ""
    page_index = read.get("page_index") if isinstance(read, dict) else None
    input_contract_reasons = _read_contract_reasons(
        read,
        expected_trace_id=expected_trace_id,
        expected_page_index=expected_page_index,
        expected_phrase_id=expected_phrase_id,
        authoritative_phrase=authoritative_phrase,
    )
    artifact_safety_reasons = _artifact_safety_reasons(
        read,
        path="read",
    )
    if artifact_safety_reasons:
        input_contract_reasons.append("input_artifact_safety_invalid")
    elif isinstance(read, dict) and _recursive_invalid_consumer_count(read):
        input_contract_reasons.append("input_consumer_flag_invalid")
    input_contract_reasons = sorted(set(input_contract_reasons))
    if input_contract_reasons:
        coverage = _failed_gate(
            COVERAGE_SCHEMA_VERSION,
            "pitch_read_contract_invalid",
        )
        format_gate = _failed_gate(
            FORMAT_SCHEMA_VERSION,
            "pitch_read_contract_invalid",
        )
        boundary = _failed_gate(
            BOUNDARY_SCHEMA_VERSION,
            "pitch_read_contract_invalid",
        )
    else:
        coverage = _coverage_gate(
            read,
            reader_dependency=reader_dependency,
            page_context_manifest=page_context_manifest,
        )
        format_gate = _format_gate(read)
        boundary = _boundary_gate(read)

    visible_read = bool(
        isinstance(read, dict)
        and isinstance(read.get("text"), str)
        and read["text"]
        and isinstance(read.get("spans"), list)
        and any(
            isinstance(row, dict) and str(row.get("char") or "")
            for row in read["spans"]
        )
    )
    phrase_complete = bool(
        not input_contract_reasons
        and visible_read
        and boundary["passed"] is True
    )
    first_failed_gate = next(
        (
            name
            for name, gate in (
                ("boundary", boundary),
            )
            if gate["passed"] is not True
        ),
        None if visible_read else "coverage",
    )
    semantic_gates = {
        "coverage_gate": coverage,
        "format_gate": format_gate,
        "boundary_gate": boundary,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": (
            expected_trace_id
            if type(expected_trace_id) is str and expected_trace_id
            else trace_id
        ),
        "phrase_id": (
            expected_phrase_id
            if type(expected_phrase_id) is str and expected_phrase_id
            else phrase_id
        ),
        "physical_core_digest": str(
            read.get("physical_core_digest") or ""
        ) if isinstance(read, dict) else "",
        "page_index": (
            expected_page_index
            if type(expected_page_index) is int
            else page_index if type(page_index) is int else -1
        ),
        "status": "ok" if not input_contract_reasons else "fail_closed",
        "input_contract_reasons": input_contract_reasons,
        "final_text": str(read.get("text") or "") if isinstance(read, dict) else "",
        "raw_text": str(read.get("raw_text") or "") if isinstance(read, dict) else "",
        **semantic_gates,
        "gate_digest": _digest(semantic_gates),
        "phrase_complete": phrase_complete,
        "phrase_complete_producer": PHRASE_COMPLETE_PRODUCER,
        "first_failed_gate": first_failed_gate,
        "three_gates_passed": phrase_complete,
        "eligible_for_post_gate_routing": phrase_complete,
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _read_contract_reasons(
    read: Any,
    *,
    expected_trace_id: str | None = None,
    expected_page_index: int | None = None,
    expected_phrase_id: str | None = None,
    authoritative_phrase: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(read, dict):
        return ["pitch_read_not_object"]
    reasons: list[str] = []
    if read.get("schema_version") != "vector_pitch_phrase_read_v1":
        reasons.append("pitch_read_schema_invalid")
    if type(read.get("trace_id")) is not str or not read["trace_id"]:
        reasons.append("pitch_read_trace_missing")
    if type(read.get("phrase_id")) is not str or not read["phrase_id"]:
        reasons.append("pitch_read_phrase_id_missing")
    if type(read.get("page_index")) is not int or read["page_index"] < 0:
        reasons.append("pitch_read_page_index_invalid")
    if (
        type(expected_trace_id) is str
        and expected_trace_id
        and read.get("trace_id") != expected_trace_id
    ):
        reasons.append("pitch_read_trace_mismatch")
    if (
        type(expected_page_index) is int
        and read.get("page_index") != expected_page_index
    ):
        reasons.append("pitch_read_page_mismatch")
    if (
        type(expected_phrase_id) is str
        and expected_phrase_id
        and read.get("phrase_id") != expected_phrase_id
    ):
        reasons.append("pitch_read_phrase_mismatch")
    digest = read.get("physical_core_digest")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    ):
        reasons.append("pitch_read_physical_core_digest_invalid")
    if type(read.get("raw_text")) is not str:
        reasons.append("pitch_read_raw_text_invalid")
    if type(read.get("text")) is not str:
        reasons.append("pitch_read_text_invalid")
    if authoritative_phrase is not None:
        reasons.extend(_native_projection_reasons(
            read,
            authoritative_phrase,
        ))
    if not isinstance(read.get("reader_debug"), dict):
        reasons.append("pitch_read_reader_debug_invalid")
    spans = read.get("spans")
    if (
        not isinstance(spans, list)
        or any(
            not isinstance(row, dict)
            or type(row.get("char")) is not str
            or len(row["char"]) != 1
            for row in spans
        )
    ):
        reasons.append("pitch_read_spans_invalid")
    if read.get("ocr_called") is not False:
        reasons.append("pitch_read_ocr_flag_invalid")
    if read.get("feeds_display_l3") is not False:
        reasons.append("pitch_read_l3_flag_invalid")
    if read.get("consumer_allowed") is not False:
        reasons.append("pitch_read_consumer_flag_invalid")
    return reasons


def _native_projection_reasons(
    read: dict[str, Any],
    phrase: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if phrase.get("schema_version") != "r92_directional_walk_phrase_v1":
        return ["native_phrase_projection_invalid"]
    if read.get("read_mode") != "upstream_directional_walk_projection":
        reasons.append("native_projection_read_mode_invalid")
    for read_key, phrase_key in (
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
    ):
        if read.get(read_key) != phrase.get(phrase_key):
            reasons.append(f"native_projection_{read_key}_mismatch")
    return reasons


def _coverage_gate(
    read: dict[str, Any],
    *,
    reader_dependency: Any = None,
    page_context_manifest: Any = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if read.get("schema_version") != "vector_pitch_phrase_read_v1":
        reasons.append("pitch_read_schema_invalid")

    primitive_ids = read.get("primitive_ids")
    if not isinstance(primitive_ids, list) or not primitive_ids or any(
        not isinstance(value, str) or not value for value in primitive_ids
    ) or len(set(primitive_ids or [])) != len(primitive_ids or []):
        reasons.append("primitive_identity_invalid")
        primitive_count = 0
    else:
        primitive_count = len(primitive_ids)
    manifest_primitive_count = (
        page_context_manifest.get("primitive_count")
        if isinstance(page_context_manifest, dict)
        else None
    )
    if (
        type(manifest_primitive_count) is int
        and primitive_count > manifest_primitive_count
    ):
        reasons.append(
            "page_context_manifest_primitive_count_insufficient"
        )
    reader_call_count = _strict_nonnegative_int(
        read.get("reader_call_count")
    )
    if reader_call_count not in {0, 1}:
        reasons.append("pitch_read_reader_call_count_invalid")

    spans = read.get("spans")
    exclusions = read.get("exclusions")
    drops = read.get("drops")
    debug = read.get("reader_debug")
    partition_evidence = _primitive_partition_evidence(
        read,
        page_context_manifest=page_context_manifest,
    )
    primitive_partition_required = partition_evidence["required"]
    primitive_partition_valid = partition_evidence["valid"]
    if (
        primitive_partition_required
        and not primitive_partition_valid
    ):
        reasons.append("primitive_partition_invalid")
    span_assignment = _assignment_primitive_ids(
        spans,
        count_key="slot_item_count",
        require_char=True,
        allow_derived_spans=True,
        debug=debug,
    )
    drop_assignment = _assignment_primitive_ids(
        drops,
        count_key="item_count",
        require_char=False,
    )
    used_item_count = _strict_nonnegative_int(
        debug.get("used_item_count") if isinstance(debug, dict) else None
    )
    filtered_item_count = _strict_nonnegative_int(
        debug.get("filtered_item_count") if isinstance(debug, dict) else None
    )
    if span_assignment is None:
        reasons.append("span_primitive_identity_invalid")
        span_primitive_ids = []
        derived_span_count = 0
    else:
        span_primitive_ids, derived_span_count = span_assignment
    if drop_assignment is None:
        reasons.append("drop_primitive_identity_invalid")
        drop_primitive_ids = []
    else:
        drop_primitive_ids, _ = drop_assignment
    exclusion_primitive_ids = (
        list(partition_evidence["exclusion_primitive_ids"])
        if (
            primitive_partition_required
            and primitive_partition_valid
        )
        else []
    )
    semantic_occupation_primitive_ids = (
        list(partition_evidence["semantic_occupation_primitive_ids"])
        if (
            primitive_partition_required
            and primitive_partition_valid
        )
        else []
    )
    span_item_count = len(span_primitive_ids)
    exclusion_item_count = len(exclusion_primitive_ids)
    semantic_occupation_item_count = len(
        semantic_occupation_primitive_ids
    )
    drop_item_count = len(drop_primitive_ids)
    ownership_required = bool(
        isinstance(debug, dict)
        and isinstance(debug.get("r32_base_core_adapter"), dict)
    )
    ownership_valid = (
        _valid_primitive_ownership_contract(read)
        if ownership_required
        else True
    )
    if not ownership_valid:
        reasons.append("primitive_ownership_contract_invalid")
    decimal_anchor_binding = _decimal_anchor_span_ownership_evidence(read)
    if (
        decimal_anchor_binding["required"] is True
        and decimal_anchor_binding["passed"] is not True
    ):
        reasons.append("decimal_anchor_span_ownership_invalid")
    assigned_primitive_ids = (
        span_primitive_ids
        + semantic_occupation_primitive_ids
        + exclusion_primitive_ids
        + drop_primitive_ids
    )
    assignment_counts = Counter(assigned_primitive_ids)
    duplicate_primitive_ids = sorted(
        primitive_id
        for primitive_id, count in assignment_counts.items()
        if count > 1
    )
    expected_primitive_ids = set(primitive_ids or [])
    assigned_primitive_id_set = set(assigned_primitive_ids)
    foreign_primitive_ids = sorted(
        assigned_primitive_id_set - expected_primitive_ids
    )
    unaccounted_primitive_ids = sorted(
        expected_primitive_ids - assigned_primitive_id_set
    )
    if duplicate_primitive_ids:
        reasons.append("duplicate_primitive_assignment")
    if foreign_primitive_ids:
        reasons.append("foreign_primitive")
    if unaccounted_primitive_ids:
        reasons.append("unaccounted_primitive")
    if used_item_count is None or filtered_item_count is None:
        reasons.append("coverage_accounting_missing")
    else:
        if (
            partition_evidence["anchored"] is True
            and filtered_item_count != 0
        ):
            reasons.append("primitive_partition_invalid")
        elif (
            partition_evidence["anchored"] is not True
            and filtered_item_count > 0
        ):
            reasons.append("filtered_items_without_exclusion_ledger")
        if (
            primitive_partition_required
            and primitive_partition_valid
        ):
            if (
                used_item_count + filtered_item_count
                != partition_evidence["admitted_primitive_count"]
            ):
                reasons.append("primitive_accounting_mismatch")
        elif used_item_count + filtered_item_count != primitive_count:
            reasons.append("primitive_accounting_mismatch")
        if (
            span_item_count
            + semantic_occupation_item_count
            + drop_item_count
            != used_item_count
        ):
            reasons.append("slot_accounting_mismatch")

    raw_text = str(read.get("raw_text") or "")
    final_text = str(read.get("text") or "")
    span_text = "".join(
        str(row.get("char") or "")
        for row in spans or []
        if isinstance(row, dict)
    )
    if not span_text:
        reasons.append("recognized_span_missing")
    if raw_text != span_text:
        reasons.append("raw_text_span_mismatch")
    transform_reasons, transform_evidence = _transform_ledger_evidence(
        debug=debug,
        raw_text=raw_text,
        final_text=final_text,
        expected_primitive_ids=expected_primitive_ids,
        span_rows=spans if isinstance(spans, list) else [],
        span_primitive_groups=[
            list(row.get("primitive_ids") or [])
            for row in spans or []
            if isinstance(row, dict)
        ],
        reader_dependency=reader_dependency,
        page_context_manifest=page_context_manifest,
        read=read,
    )
    reasons.extend(transform_reasons)
    semantic_anchor_binding = _semantic_anchor_ownership_evidence(
        read,
        transform_evidence=transform_evidence,
        page_context_manifest=page_context_manifest,
    )
    if (
        semantic_anchor_binding["required"] is True
        and semantic_anchor_binding["passed"] is not True
    ):
        reasons.append("semantic_anchor_ownership_invalid")
    drop_count = len(drops) if isinstance(drops, list) else 0
    consumed_transform_drop_indices = transform_evidence.get(
        "consumed_transform_drop_indices"
    )
    valid_consumed_drop_indices = bool(
        not transform_reasons
        and transform_evidence.get("transform_authority_valid") is True
        and isinstance(consumed_transform_drop_indices, list)
        and all(
            type(index) is int and 0 <= index < drop_count
            for index in consumed_transform_drop_indices
        )
        and consumed_transform_drop_indices
        == sorted(set(consumed_transform_drop_indices))
    )
    consumed_drop_index_set = (
        set(consumed_transform_drop_indices)
        if valid_consumed_drop_indices
        else set()
    )
    unresolved_drop_indices = [
        index
        for index in range(drop_count)
        if index not in consumed_drop_index_set
    ]
    unresolved_drop_count = len(unresolved_drop_indices)
    if unresolved_drop_count:
        reasons.append("unresolved_slot_drops")

    return _gate(
        schema_version=COVERAGE_SCHEMA_VERSION,
        reasons=reasons,
        evidence={
            "primitive_count": primitive_count,
            "expected_primitive_ids": sorted(expected_primitive_ids),
            "span_primitive_ids": sorted(span_primitive_ids),
            "exclusion_primitive_ids": sorted(
                exclusion_primitive_ids
            ),
            "semantic_occupation_primitive_ids": sorted(
                semantic_occupation_primitive_ids
            ),
            "drop_primitive_ids": sorted(drop_primitive_ids),
            "duplicate_primitive_ids": duplicate_primitive_ids,
            "foreign_primitive_ids": foreign_primitive_ids,
            "unaccounted_primitive_ids": unaccounted_primitive_ids,
            "used_item_count": used_item_count,
            "filtered_item_count": filtered_item_count,
            "span_item_count": span_item_count,
            "exclusion_item_count": exclusion_item_count,
            "semantic_occupation_item_count": (
                semantic_occupation_item_count
            ),
            "drop_item_count": drop_item_count,
            "span_count": len(spans) if isinstance(spans, list) else 0,
            "derived_span_count": derived_span_count,
            "primitive_ownership_required": ownership_required,
            "primitive_ownership_valid": ownership_valid,
            "primitive_partition_required": (
                primitive_partition_required
            ),
            "primitive_partition_valid": primitive_partition_valid,
            "anchored_partition": partition_evidence["anchored"],
            "primitive_partition_digest": partition_evidence[
                "partition_digest"
            ],
            "semantic_occupation_digest": partition_evidence[
                "semantic_occupation_digest"
            ],
            **(
                {"decimal_anchor_binding": decimal_anchor_binding}
                if decimal_anchor_binding["required"] is True
                else {}
            ),
            **(
                {"semantic_anchor_binding": semantic_anchor_binding}
                if semantic_anchor_binding["required"] is True
                else {}
            ),
            "unresolved_drop_count": unresolved_drop_count,
            "unresolved_drop_indices": unresolved_drop_indices,
            "raw_text": raw_text,
            "span_text": span_text,
            "final_text": final_text,
            **transform_evidence,
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    )


def _format_gate(read: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not str(read.get("text") or ""):
        reasons.append("upstream_read_failed")
    formal = validate_dimension_text_format(str(read.get("text") or ""))
    return _gate(
        schema_version=FORMAT_SCHEMA_VERSION,
        reasons=reasons,
        evidence={
            "validator": formal,
            "syntax": classify_dimension_syntax(
                str(read.get("text") or ""),
                drops=read.get("drops"),
                format_result=formal,
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    )


def _decimal_anchor_span_ownership_evidence(
    read: dict[str, Any],
) -> dict[str, Any]:
    reader_debug = read.get("reader_debug")
    adapter = (
        reader_debug.get("r32_base_core_adapter")
        if isinstance(reader_debug, dict)
        else None
    )
    window_authority = (
        normalize_pitch_window_authority(
            adapter.get("window_authority")
        )
        if isinstance(adapter, dict)
        else None
    )
    grouped_window_digests = (
        {
            group["size_class"]: group["group_digest"]
            for group in window_authority["groups"]
        }
        if isinstance(window_authority, dict)
        and window_authority.get("schema_version")
        == "vector_pitch_window_authority_v2"
        else None
    )
    authorities_by_primitive: dict[str, dict[str, dict[str, Any]]] = {}
    invalid_authority_count = 0
    for source in read.get("source_candidate_evidence") or []:
        if not isinstance(source, dict) or "decimal_anchor_authority" not in source:
            continue
        if source.get("decimal_anchor_authority_source") != "dot_reference":
            invalid_authority_count += 1
            continue
        authority = normalize_decimal_anchor_authority(
            source.get("decimal_anchor_authority")
        )
        if authority is None or not is_authoritative_decimal_anchor(authority):
            invalid_authority_count += 1
            continue
        primitive_authorities = authorities_by_primitive.setdefault(
            authority["source_primitive_id"],
            {},
        )
        primitive_authorities[authority["authority_digest"]] = authority
    authority_count = sum(
        len(rows) for rows in authorities_by_primitive.values()
    )
    conflicting_authority_primitive_ids = sorted(
        primitive_id
        for primitive_id, rows in authorities_by_primitive.items()
        if len(rows) > 1
    )
    authority_rows = {
        primitive_id: next(iter(rows.values()))
        for primitive_id, rows in authorities_by_primitive.items()
        if len(rows) == 1
    }
    required = bool(authority_count or invalid_authority_count)
    if not required:
        return {
            "schema_version": "vector_decimal_anchor_span_binding_v1",
            "required": False,
            "passed": True,
            "authority_count": 0,
            "invalid_authority_count": 0,
            "conflicting_authority_primitive_ids": [],
            "bindings": [],
            "consumer_allowed": CONSUMER_ALLOWED,
        }

    text = "".join(
        str(span.get("char") or "")
        for span in read.get("spans") or []
        if isinstance(span, dict)
    )
    decimal_roles = _decimal_text_roles(text)
    semantic_occupation_mode = bool(
        isinstance(adapter, dict)
        and isinstance(adapter.get("semantic_occupation"), dict)
        and isinstance(reader_debug, dict)
        and reader_debug.get("group_primitive_partition", {}).get(
            "schema_version"
        )
        == "vector_pitch_group_primitive_partition_v2"
    )
    bindings: list[dict[str, Any]] = []
    all_passed = bool(
        invalid_authority_count == 0
        and not conflicting_authority_primitive_ids
    )
    cursor = 0
    span_text_ranges: list[tuple[int, int]] = []
    for span in read.get("spans") or []:
        char = str(span.get("char") or "") if isinstance(span, dict) else ""
        span_text_ranges.append((cursor, cursor + len(char)))
        cursor += len(char)
    for primitive_id, authority in sorted(authority_rows.items()):
        owners = [
            index
            for index, span in enumerate(read.get("spans") or [])
            if isinstance(span, dict)
            and primitive_id in (span.get("primitive_ids") or [])
        ]
        drop_owners = [
            index
            for index, drop in enumerate(read.get("drops") or [])
            if isinstance(drop, dict)
            and primitive_id in (drop.get("primitive_ids") or [])
        ]
        owner = (
            (read.get("spans") or [])[owners[0]]
            if len(owners) == 1
            else {}
        )
        text_index = (
            span_text_ranges[owners[0]][0]
            if len(owners) == 1
            else None
        )
        observed_role = (
            owner.get("window_size_class")
            if semantic_occupation_mode and isinstance(owner, dict)
            else (
                decimal_roles.get(text_index)
                if text_index is not None
                else None
            )
        )
        expected_group_digest = (
            grouped_window_digests.get(authority["size_class"])
            if grouped_window_digests is not None
            else None
        )
        grouped_binding_passed = bool(
            grouped_window_digests is None
            or (
                expected_group_digest is not None
                and owner.get("window_size_class")
                == authority["size_class"]
                and owner.get("window_group_digest")
                == expected_group_digest
            )
        )
        binding_passed = bool(
            len(owners) == 1
            and not drop_owners
            and isinstance(owner, dict)
            and owner.get("char") == "."
            and owner.get("decision") in {
                "authority_source_decimal_quad",
                "dot_pitch_slot",
                "dot_multiscale_layout",
            }
            and observed_role == authority["size_class"]
            and grouped_binding_passed
        )
        all_passed = all_passed and binding_passed
        bindings.append({
            "decimal_quad_id": authority["decimal_quad_id"],
            "primitive_id": primitive_id,
            "expected_size_class": authority["size_class"],
            "observed_text_role": observed_role,
            "owner_span_indices": owners,
            "drop_indices": drop_owners,
            "owner_char": owner.get("char") if isinstance(owner, dict) else None,
            "owner_decision": (
                owner.get("decision") if isinstance(owner, dict) else None
            ),
            **(
                {
                    "observed_window_size_class": owner.get(
                        "window_size_class"
                    ),
                    "expected_window_group_digest": (
                        expected_group_digest
                    ),
                    "observed_window_group_digest": owner.get(
                        "window_group_digest"
                    ),
                    "window_group_binding_passed": (
                        grouped_binding_passed
                    ),
                }
                if grouped_window_digests is not None
                else {}
            ),
            "passed": binding_passed,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return {
        "schema_version": "vector_decimal_anchor_span_binding_v1",
        "required": True,
        "passed": all_passed,
        "authority_count": authority_count,
        "invalid_authority_count": invalid_authority_count,
        "conflicting_authority_primitive_ids": (
            conflicting_authority_primitive_ids
        ),
        "raw_span_text": text,
        "bindings": bindings,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _semantic_anchor_ownership_evidence(
    read: dict[str, Any],
    *,
    transform_evidence: dict[str, Any],
    page_context_manifest: Any,
) -> dict[str, Any]:
    raw_authorities: list[Any] = []
    carrier_count = 0
    for source in read.get("source_candidate_evidence") or []:
        if not isinstance(source, dict) or (
            "semantic_anchor_authorities" not in source
        ):
            continue
        carrier_count += 1
        nested = source.get("semantic_anchor_authorities")
        if isinstance(nested, list):
            raw_authorities.extend(nested)
        else:
            raw_authorities.append(nested)
    normalized = [
        authority
        for raw in raw_authorities
        if (
            authority := normalize_semantic_anchor_authority(raw)
        )
        is not None
    ]
    invalid_authority_count = len(raw_authorities) - len(normalized)
    unique_normalized = {
        str(authority["authority_digest"]): authority
        for authority in normalized
    }
    duplicate_authority_count = (
        len(normalized) - len(unique_normalized)
    )
    authorities = canonical_semantic_anchor_authorities(
        unique_normalized.values()
    )
    conflicting_authority_count = (
        len(unique_normalized) - len(authorities)
    )
    trusted_authorities = (
        semantic_anchor_authorities_from_source_evidence(
            list(read.get("source_candidate_evidence") or [])
        )
    )
    trusted_digests = {
        str(authority["authority_digest"])
        for authority in trusted_authorities
    }
    untrusted_authority_count = sum(
        str(authority["authority_digest"]) not in trusted_digests
        for authority in normalized
    )
    authorities = trusted_authorities
    required = bool(carrier_count)
    if not required:
        return {
            "schema_version": "vector_semantic_anchor_ownership_v1",
            "required": False,
            "passed": True,
            "authority_count": 0,
            "invalid_authority_count": 0,
            "conflicting_authority_count": 0,
            "duplicate_authority_count": 0,
            "untrusted_authority_count": 0,
            "bindings": [],
            "consumer_allowed": CONSUMER_ALLOWED,
        }

    spans = read.get("spans") if isinstance(read.get("spans"), list) else []
    drops = read.get("drops") if isinstance(read.get("drops"), list) else []
    ledger = (
        read.get("reader_debug", {}).get("transform_ledger")
        if isinstance(read.get("reader_debug"), dict)
        else None
    )
    output_tokens = [
        token
        for step in ledger or []
        if isinstance(step, dict)
        for token in step.get("output_tokens") or []
        if isinstance(token, dict)
    ]
    consumed_drop_indices = transform_evidence.get(
        "consumed_transform_drop_indices"
    )
    consumed_semantic_occupation_indices = transform_evidence.get(
        "consumed_semantic_occupation_indices"
    )
    transform_valid = bool(
        transform_evidence.get("transform_authority_valid") is True
        and isinstance(consumed_drop_indices, list)
        and isinstance(consumed_semantic_occupation_indices, list)
    )
    adapter = (
        read.get("reader_debug", {}).get("r32_base_core_adapter")
        if isinstance(read.get("reader_debug"), dict)
        else None
    )
    semantic_occupation = (
        _replay_semantic_occupation(
            adapter.get("semantic_occupation"),
            phrase_id=str(read.get("phrase_id") or ""),
            page_index=read.get("page_index"),
            primitive_manifest=adapter.get("primitive_manifest"),
            window_authority=adapter.get("window_authority"),
        )
        if isinstance(adapter, dict)
        and adapter.get("semantic_occupation") is not None
        else None
    )
    expected_primitive_ids = set(read.get("primitive_ids") or [])
    bindings: list[dict[str, Any]] = []
    authorized_transform_drop_indices: set[int] = set()
    authorized_semantic_occupation_indices: set[int] = set()
    passed = bool(
        invalid_authority_count == 0
        and conflicting_authority_count == 0
        and duplicate_authority_count == 0
        and untrusted_authority_count == 0
        and authorities
    )
    for authority in authorities:
        authority_ids = set(authority["source_primitive_ids"])
        span_indices = [
            index
            for index, row in enumerate(spans)
            if isinstance(row, dict)
            and authority_ids.intersection(row.get("primitive_ids") or [])
        ]
        drop_indices = [
            index
            for index, row in enumerate(drops)
            if isinstance(row, dict)
            and authority_ids.intersection(row.get("primitive_ids") or [])
        ]
        direct_owner = (
            spans[span_indices[0]]
            if len(span_indices) == 1
            else None
        )
        manifest_geometry_passed = (
            _semantic_authority_manifest_geometry_valid(
                authority,
                read=read,
                page_context_manifest=page_context_manifest,
            )
        )
        direct_passed = bool(
            authority_ids.issubset(expected_primitive_ids)
            and manifest_geometry_passed
            and len(span_indices) == 1
            and not drop_indices
            and isinstance(direct_owner, dict)
            and set(direct_owner.get("primitive_ids") or [])
            == authority_ids
            and direct_owner.get("char") == authority["output_text"]
            and _semantic_direct_span_position_valid(
                authority,
                owner_index=span_indices[0],
                spans=spans,
                read=read,
            )
        )
        matching_drop_tokens = [
            token
            for token in output_tokens
            if token.get("authority_digest")
            == authority["authority_digest"]
            and token.get("glyph_id") == authority["glyph_id"]
            and token.get("detector_source")
            == authority["detector_source"]
            and token.get("char") == authority["output_text"]
            and set(token.get("primitive_ids") or []) == authority_ids
            and token.get("source_drop_indices") == drop_indices
            and token.get("source_span_indices") == []
        ]
        transformed_drop_passed = bool(
            authority_ids.issubset(expected_primitive_ids)
            and manifest_geometry_passed
            and not span_indices
            and drop_indices
            and transform_valid
            and set(drop_indices).issubset(
                set(consumed_drop_indices)
            )
            and len(matching_drop_tokens) == 1
            and {
                primitive_id
                for index in drop_indices
                for primitive_id in (
                    drops[index].get("primitive_ids") or []
                )
            }
            == authority_ids
            and all(
                set(drops[index].get("primitive_ids") or [])
                .issubset(authority_ids)
                for index in drop_indices
            )
        )
        if transformed_drop_passed:
            authorized_transform_drop_indices.update(drop_indices)
        matching_span_tokens = [
            token
            for token in output_tokens
            if token.get("authority_digest")
            == authority["authority_digest"]
            and token.get("glyph_id") == authority["glyph_id"]
            and token.get("detector_source")
            == authority["detector_source"]
            and token.get("char") == authority["output_text"]
            and set(token.get("primitive_ids") or []) == authority_ids
            and token.get("source_span_indices") == span_indices
            and token.get("source_drop_indices") == []
        ]
        transformed_span_passed = bool(
            authority_ids.issubset(expected_primitive_ids)
            and manifest_geometry_passed
            and len(span_indices) == 1
            and not drop_indices
            and transform_valid
            and len(matching_span_tokens) == 1
            and isinstance(direct_owner, dict)
            and set(direct_owner.get("primitive_ids") or [])
            == authority_ids
            and _semantic_direct_span_position_valid(
                authority,
                owner_index=span_indices[0],
                spans=spans,
                read=read,
            )
        )
        occupation_entries = (
            semantic_occupation.get("entries")
            if isinstance(semantic_occupation, dict)
            else None
        )
        occupation_entry = (
            occupation_entries[0]
            if isinstance(occupation_entries, list)
            and len(occupation_entries) == 1
            and isinstance(occupation_entries[0], dict)
            else None
        )
        matching_occupation_tokens = [
            token
            for token in output_tokens
            if token.get("authority_digest")
            == authority["authority_digest"]
            and token.get("glyph_id") == authority["glyph_id"]
            and token.get("detector_source")
            == authority["detector_source"]
            and token.get("char") == authority["output_text"]
            and set(token.get("primitive_ids") or []) == authority_ids
            and token.get("source_span_indices") == []
            and token.get("source_drop_indices") == []
            and token.get("source_semantic_occupation_indices") == [0]
            and token.get("semantic_occupation_digest")
            == (
                semantic_occupation.get("occupation_digest")
                if isinstance(semantic_occupation, dict)
                else None
            )
            and token.get("reason")
            == "plus_minus_separator_from_semantic_occupation"
        ]
        transformed_occupation_passed = bool(
            authority.get("glyph_type") == "plus_minus"
            and authority.get("output_text") == "±"
            and authority_ids.issubset(expected_primitive_ids)
            and manifest_geometry_passed
            and not span_indices
            and not drop_indices
            and transform_valid
            and consumed_semantic_occupation_indices == [0]
            and isinstance(occupation_entry, dict)
            and occupation_entry.get("authority_digest")
            == authority.get("authority_digest")
            and set(occupation_entry.get("source_primitive_ids") or [])
            == authority_ids
            and set(
                semantic_occupation.get("occupied_primitive_ids") or []
            )
            == authority_ids
            and len(matching_occupation_tokens) == 1
        )
        if transformed_occupation_passed:
            authorized_semantic_occupation_indices.add(0)
        binding_passed = bool(
            direct_passed
            or transformed_drop_passed
            or transformed_span_passed
            or transformed_occupation_passed
        )
        passed = passed and binding_passed
        bindings.append({
            "glyph_id": str(authority["glyph_id"]),
            "glyph_type": str(authority["glyph_type"]),
            "detector_source": str(authority["detector_source"]),
            "authority_digest": str(authority["authority_digest"]),
            "source_primitive_ids": list(
                authority["source_primitive_ids"]
            ),
            "owner_span_indices": span_indices,
            "drop_indices": drop_indices,
            "ownership_mode": (
                "direct_symbol_span"
                if direct_passed
                else (
                    "authoritative_transform_drop"
                    if transformed_drop_passed
                    else (
                        "authoritative_transform_span"
                        if transformed_span_passed
                        else (
                            "authoritative_semantic_occupation"
                            if transformed_occupation_passed
                            else "unresolved"
                        )
                    )
                )
            ),
            "passed": binding_passed,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    if (
        transform_valid
        and authorized_transform_drop_indices
        != set(consumed_drop_indices)
    ):
        passed = False
    if (
        transform_valid
        and authorized_semantic_occupation_indices
        != set(consumed_semantic_occupation_indices)
    ):
        passed = False
    return {
        "schema_version": "vector_semantic_anchor_ownership_v1",
        "required": True,
        "passed": passed,
        "authority_count": len(authorities),
        "invalid_authority_count": invalid_authority_count,
        "conflicting_authority_count": conflicting_authority_count,
        "duplicate_authority_count": duplicate_authority_count,
        "untrusted_authority_count": untrusted_authority_count,
        "bindings": bindings,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _semantic_direct_span_position_valid(
    authority: dict[str, Any],
    *,
    owner_index: int,
    spans: list[dict[str, Any]],
    read: dict[str, Any],
) -> bool:
    if (
        owner_index < 0
        or owner_index >= len(spans)
        or any(
            not isinstance(row, dict)
            or type(row.get("slot_index")) is not int
            or type(row.get("char")) is not str
            for row in spans
        )
    ):
        return False
    slot_indices = [int(row["slot_index"]) for row in spans]
    if (
        slot_indices != sorted(slot_indices)
        or len(slot_indices) != len(set(slot_indices))
        or not _semantic_direct_span_axis_geometry_valid(
            authority,
            owner_index=owner_index,
            spans=spans,
            read=read,
        )
    ):
        return False
    glyph_type = str(authority.get("glyph_type") or "")
    if glyph_type == "diameter":
        return owner_index == 0
    if glyph_type == "degree":
        return owner_index == len(spans) - 1
    if glyph_type != "plus_minus":
        return False
    raw_text = "".join(str(row["char"]) for row in spans)
    return bool(
        owner_index > 0
        and owner_index < len(spans) - 1
        and re.fullmatch(
            r"\d+(?:\.\d+)?",
            raw_text[:owner_index],
        )
        and re.fullmatch(
            r"\d+(?:\.\d+)?",
            raw_text[owner_index + 1:],
        )
    )


def _semantic_direct_span_axis_geometry_valid(
    authority: dict[str, Any],
    *,
    owner_index: int,
    spans: list[dict[str, Any]],
    read: dict[str, Any],
) -> bool:
    manifest = read.get("primitive_manifest")
    angle = _finite_number(read.get("axis_angle_deg"))
    carrier_angle = _semantic_authority_carrier_axis_angle(
        authority,
        read=read,
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "vector_pitch_phrase_primitive_manifest_v2"
        or read.get("axis_trusted") is not True
        or angle is None
        or carrier_angle is None
        or not _semantic_axis_angles_match(
            float(angle),
            float(carrier_angle),
        )
    ):
        return False
    geometry_rows = manifest.get("geometry_entries")
    if not isinstance(geometry_rows, list):
        return False
    bbox_by_id = {
        row.get("primitive_id"): row.get("bbox")
        for row in geometry_rows
        if isinstance(row, dict)
        and type(row.get("primitive_id")) is str
    }
    if len(bbox_by_id) != len(geometry_rows):
        return False
    intervals: list[tuple[float, float]] = []
    assigned_ids: set[str] = set()
    for span in spans:
        primitive_ids = span.get("primitive_ids")
        if (
            not isinstance(primitive_ids, list)
            or not primitive_ids
            or span.get("slot_item_count") != len(primitive_ids)
            or len(primitive_ids) != len(set(primitive_ids))
            or any(
                type(primitive_id) is not str
                or primitive_id not in bbox_by_id
                or primitive_id in assigned_ids
                for primitive_id in primitive_ids
            )
        ):
            return False
        assigned_ids.update(primitive_ids)
        bboxes = [bbox_by_id[primitive_id] for primitive_id in primitive_ids]
        if any(
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(value) not in {int, float} for value in bbox)
            for bbox in bboxes
        ):
            return False
        x0 = min(float(bbox[0]) for bbox in bboxes)
        y0 = min(float(bbox[1]) for bbox in bboxes)
        x1 = max(float(bbox[2]) for bbox in bboxes)
        y1 = max(float(bbox[3]) for bbox in bboxes)
        union = {
            "x": round(x0, 3),
            "y": round(y0, 3),
            "w": round(x1 - x0, 3),
            "h": round(y1 - y0, 3),
        }
        interval = _semantic_bbox_axis_interval(
            union,
            axis_angle_deg=float(angle) % 180.0,
        )
        if interval is None:
            return False
        intervals.append(interval)
    if (
        owner_index < 0
        or owner_index >= len(intervals)
        or any(
            intervals[index][1] >= intervals[index + 1][0]
            for index in range(len(intervals) - 1)
        )
    ):
        return False
    authority_interval = _semantic_bbox_axis_interval(
        authority.get("anchor_bbox"),
        axis_angle_deg=float(angle) % 180.0,
    )
    return bool(
        authority_interval is not None
        and all(
            math.isclose(
                float(authority_interval[bound]),
                float(intervals[owner_index][bound]),
                abs_tol=1e-6,
            )
            for bound in (0, 1)
        )
    )


def _semantic_authority_carrier_axis_angle(
    authority: dict[str, Any],
    *,
    read: dict[str, Any],
) -> float | None:
    authority_digest = str(authority.get("authority_digest") or "")
    matches: list[float] = []
    for source in read.get("source_candidate_evidence") or []:
        if not isinstance(source, dict):
            continue
        carried = semantic_anchor_authorities_from_source_evidence(
            [source]
        )
        if not any(
            str(row.get("authority_digest") or "")
            == authority_digest
            for row in carried
        ):
            continue
        angle = _finite_number(source.get("axis_angle_deg"))
        if angle is None:
            return None
        matches.append(float(angle))
    return matches[0] if len(matches) == 1 else None


def _semantic_axis_angles_match(
    left: float,
    right: float,
) -> bool:
    left_normalized = float(left) % 180.0
    right_normalized = float(right) % 180.0
    delta = abs(left_normalized - right_normalized)
    return min(delta, 180.0 - delta) <= 0.001


def _anchored_lattice_origin_matches(
    debug: dict[str, Any],
    authority: dict[str, Any],
) -> bool:
    selected_origin = _finite_number(debug.get("selected_origin"))
    selected_pitch = _finite_number(debug.get("selected_pitch"))
    dot_main_center = _finite_number(
        authority.get("dot_main_center")
    )
    authority_pitch = _finite_number(authority.get("pitch_pt"))
    if (
        selected_origin is None
        or selected_pitch is None
        or dot_main_center is None
        or authority_pitch is None
        or selected_pitch <= 0.0
        or authority_pitch <= 0.0
        or abs(selected_pitch - authority_pitch) > 1e-6
    ):
        return False
    boundary_phase = dot_main_center - authority_pitch / 2.0
    phase_steps = (
        selected_origin - boundary_phase
    ) / authority_pitch
    return abs(phase_steps - round(phase_steps)) <= 1e-6


def _grouped_anchored_lattices_match(
    debug: Any,
    authority: Any,
    *,
    spans: Any,
    admitted_primitive_ids: Any,
    primitive_manifest: Any,
    semantic_occupation: Any = None,
) -> bool:
    if not (
        isinstance(debug, dict)
        and isinstance(authority, dict)
        and authority.get("schema_version")
        == "vector_pitch_window_authority_v2"
        and debug.get("algorithm") == GROUPED_WINDOW_ALGORITHM
        and debug.get("selected_pitch_source")
        == "window_authority_group_pitches"
        and debug.get("pitch_origin_mode")
        == "grouped_decimal_lattices"
        and "selected_pitch" not in debug
        and "selected_origin" not in debug
        and debug.get("window_authority_digest")
        == authority.get("authority_digest")
        and debug.get("cross_group_relation_digest")
        == authority.get("cross_group_relation", {}).get(
            "relation_digest"
        )
        and isinstance(spans, list)
        and spans
        and isinstance(admitted_primitive_ids, list)
        and admitted_primitive_ids == sorted(admitted_primitive_ids)
    ):
        return False
    grouped_partition = _replay_grouped_pitch_primitive_partition(
        debug.get("group_primitive_partition"),
        authority=authority,
        admitted_primitive_ids=admitted_primitive_ids,
        primitive_manifest=primitive_manifest,
        spans=spans,
        semantic_occupation=semantic_occupation,
    )
    if grouped_partition is None:
        return False
    groups = authority.get("groups")
    attempts = debug.get("pitch_group_attempts")
    if not (
        isinstance(groups, list)
        and isinstance(attempts, list)
        and len(groups) == len(attempts) == 2
        and [group.get("size_class") for group in groups]
        == ["primary", "tolerance"]
    ):
        return False
    groups_by_class = {
        group["size_class"]: group for group in groups
    }
    attempt_keys = {
        "size_class",
        "group_digest",
        "window_authority_digest",
        "source_primitive_id",
        "selected_pitch",
        "selected_origin",
        "selected_text",
        "selected_primitive_ids",
        "outcome",
        "consumer_allowed",
    }
    attempts_by_class: dict[str, dict[str, Any]] = {}
    for index, (attempt, group) in enumerate(
        zip(attempts, groups, strict=True)
    ):
        size_class = group["size_class"]
        child = group.get("window_authority")
        selected_ids = attempt.get("selected_primitive_ids")
        expected_outcome = (
            "selected_primary_prefix"
            if index == 0
            else "selected_tolerance_suffix"
        )
        if not (
            isinstance(attempt, dict)
            and set(attempt) == attempt_keys
            and isinstance(child, dict)
            and attempt.get("size_class") == size_class
            and attempt.get("group_digest") == group.get("group_digest")
            and attempt.get("window_authority_digest")
            == child.get("authority_digest")
            and attempt.get("source_primitive_id")
            == child.get("source_primitive_id")
            and attempt.get("outcome") == expected_outcome
            and attempt.get("consumer_allowed") is False
            and isinstance(attempt.get("selected_text"), str)
            and attempt["selected_text"]
            and isinstance(selected_ids, list)
            and selected_ids
            and selected_ids == sorted(selected_ids)
            and len(selected_ids) == len(set(selected_ids))
            and child.get("source_primitive_id") in selected_ids
            and _anchored_lattice_origin_matches(attempt, child)
        ):
            return False
        attempts_by_class[size_class] = attempt
    if set(attempts_by_class) != {"primary", "tolerance"}:
        return False

    span_classes: list[str] = []
    span_ids_by_class: dict[str, list[str]] = {
        "primary": [],
        "tolerance": [],
    }
    span_text_by_class: dict[str, list[str]] = {
        "primary": [],
        "tolerance": [],
    }
    all_span_ids: list[str] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            return False
        size_class = span.get("window_size_class")
        group = groups_by_class.get(size_class)
        primitive_ids = span.get("primitive_ids")
        if not (
            isinstance(group, dict)
            and span.get("window_group_digest")
            == group.get("group_digest")
            and span.get("window_group_authority_digest")
            == group.get("window_authority", {}).get(
                "authority_digest"
            )
            and span.get("slot_index") == index
            and span.get("slot_count") == 1
            and type(span.get("group_slot_index")) is int
            and span["group_slot_index"] >= 0
            and type(span.get("group_slot_count")) is int
            and span["group_slot_count"] >= 1
            and type(span.get("char")) is str
            and len(span["char"]) == 1
            and isinstance(primitive_ids, list)
            and primitive_ids
            and primitive_ids == sorted(primitive_ids)
            and len(primitive_ids) == len(set(primitive_ids))
            and span.get("consumer_allowed") is False
        ):
            return False
        span_classes.append(size_class)
        span_ids_by_class[size_class].extend(primitive_ids)
        span_text_by_class[size_class].append(span["char"])
        all_span_ids.extend(primitive_ids)
    if (
        "primary" not in span_classes
        or "tolerance" not in span_classes
        or span_classes != sorted(
            span_classes,
            key=lambda value: (
                0 if value == "primary" else 1
            ),
        )
        or len(all_span_ids) != len(set(all_span_ids))
        or sorted(all_span_ids) != (
            grouped_partition.get(
                "numeric_primitive_ids",
                admitted_primitive_ids,
            )
        )
    ):
        return False
    for size_class in ("primary", "tolerance"):
        if (
            sorted(span_ids_by_class[size_class])
            != attempts_by_class[size_class][
                "selected_primitive_ids"
            ]
            or "".join(span_text_by_class[size_class])
            != attempts_by_class[size_class]["selected_text"]
        ):
            return False
    raw_text = "".join(str(span["char"]) for span in spans)
    occupation_mode = (
        grouped_partition.get("schema_version")
        == "vector_pitch_group_primitive_partition_v2"
    )
    primary_text = attempts_by_class["primary"]["selected_text"]
    tolerance_text = attempts_by_class["tolerance"]["selected_text"]
    if occupation_mode:
        preview_text = f"{primary_text}±{tolerance_text}"
        return bool(
            isinstance(semantic_occupation, dict)
            and debug.get("semantic_occupation") == semantic_occupation
            and debug.get("semantic_preview_text") == preview_text
            and raw_text == debug.get("raw_predicted_text")
            and validate_dimension_text_format(preview_text).get("valid")
            and "±" not in raw_text
            and "±" not in primary_text
            and "±" not in tolerance_text
        )
    return bool(
        semantic_occupation is None
        and raw_text == debug.get("raw_predicted_text")
        and re.fullmatch(r".+±", primary_text) is not None
        and re.fullmatch(r"\d+\.\d+", tolerance_text) is not None
    )


def _replay_grouped_pitch_primitive_partition(
    value: Any,
    *,
    authority: dict[str, Any],
    admitted_primitive_ids: list[str],
    primitive_manifest: Any,
    spans: list[dict[str, Any]],
    semantic_occupation: Any = None,
) -> dict[str, Any] | None:
    """Validate gate-owned evidence before replaying grouped assignment."""
    if (
        isinstance(value, dict)
        and value.get("schema_version")
        == "vector_pitch_group_primitive_partition_v2"
    ):
        return _replay_grouped_pitch_primitive_partition_v2(
            value,
            authority=authority,
            admitted_primitive_ids=admitted_primitive_ids,
            primitive_manifest=primitive_manifest,
            spans=spans,
            semantic_occupation=semantic_occupation,
        )
    if semantic_occupation is not None:
        return None
    top_level_keys = {
        "schema_version",
        "method",
        "phrase_id",
        "window_authority_digest",
        "cross_group_relation_digest",
        "axis_angle_deg",
        "main_unit",
        "boundary_main",
        "primitive_ids",
        "groups",
        "consumer_allowed",
        "partition_digest",
    }
    relation = authority.get("cross_group_relation")
    authority_groups = authority.get("groups")
    if not (
        isinstance(value, dict)
        and set(value) == top_level_keys
        and value.get("schema_version")
        == "vector_pitch_group_primitive_partition_v1"
        and value.get("method")
        == "absolute_directed_main_disjoint_window_midpoint"
        and value.get("phrase_id") == authority.get("phrase_id")
        and value.get("window_authority_digest")
        == authority.get("authority_digest")
        and isinstance(relation, dict)
        and value.get("cross_group_relation_digest")
        == relation.get("relation_digest")
        and value.get("axis_angle_deg")
        == authority.get("axis_angle_deg")
        and value.get("main_unit") == authority.get("main_unit")
        and isinstance(admitted_primitive_ids, list)
        and admitted_primitive_ids == sorted(admitted_primitive_ids)
        and len(admitted_primitive_ids)
        == len(set(admitted_primitive_ids))
        and value.get("primitive_ids") == admitted_primitive_ids
        and value.get("consumer_allowed") is False
        and isinstance(authority_groups, list)
        and [group.get("size_class") for group in authority_groups]
        == ["primary", "tolerance"]
        and isinstance(spans, list)
        and spans
    ):
        return None
    partition_digest = value.get("partition_digest")
    try:
        expected_digest = _digest({
            key: item
            for key, item in value.items()
            if key != "partition_digest"
        })
    except (OverflowError, TypeError, ValueError):
        return None
    if not (
        _valid_sha256(partition_digest)
        and partition_digest == expected_digest
    ):
        return None
    if not (
        isinstance(primitive_manifest, dict)
        and primitive_manifest.get("schema_version")
        == "vector_pitch_phrase_primitive_manifest_v2"
        and isinstance(primitive_manifest.get("geometry_entries"), list)
        and isinstance(authority.get("main_unit"), dict)
    ):
        return None
    main_x = _finite_number(authority["main_unit"].get("x"))
    main_y = _finite_number(authority["main_unit"].get("y"))
    if (
        main_x is None
        or main_y is None
        or abs(math.hypot(main_x, main_y) - 1.0) > 1e-6
    ):
        return None
    geometry_by_id: dict[str, Any] = {}
    for row in primitive_manifest["geometry_entries"]:
        primitive_id = row.get("primitive_id") if isinstance(row, dict) else None
        if (
            type(primitive_id) is not str
            or primitive_id in geometry_by_id
        ):
            return None
        geometry_by_id[primitive_id] = row.get("bbox")
    raw_groups = value.get("groups")
    if not (
        isinstance(raw_groups, list)
        and len(raw_groups) == len(authority_groups) == 2
    ):
        return None
    boundary_main = _finite_number(value.get("boundary_main"))
    if boundary_main is None:
        return None
    group_keys = {
        "size_class",
        "group_digest",
        "window_authority_digest",
        "window_main_interval",
        "source_primitive_ids",
        "primitive_ids",
        "primitive_main_intervals",
        "consumer_allowed",
    }
    assignment_keys = {
        "primitive_id",
        "main_min",
        "main_max",
        "consumer_allowed",
    }
    observed_ids: list[str] = []
    intervals_by_id: dict[str, list[float]] = {}
    partition_ids_by_class: dict[str, list[str]] = {}
    expected_window_intervals: dict[str, list[float]] = {}
    for raw_group, authority_group in zip(
        raw_groups,
        authority_groups,
        strict=True,
    ):
        child = authority_group.get("window_authority")
        size_class = authority_group.get("size_class")
        window_interval = _grouped_authority_window_main_interval(
            child,
            main_unit=(main_x, main_y),
        )
        assignments = raw_group.get(
            "primitive_main_intervals"
        ) if isinstance(raw_group, dict) else None
        primitive_ids = raw_group.get("primitive_ids") if isinstance(
            raw_group, dict
        ) else None
        source_ids = (
            sorted(child.get("group_member_source_primitive_ids") or [])
            if isinstance(child, dict)
            else None
        )
        if not (
            isinstance(raw_group, dict)
            and set(raw_group) == group_keys
            and isinstance(child, dict)
            and size_class in {"primary", "tolerance"}
            and raw_group.get("size_class") == size_class
            and raw_group.get("group_digest")
            == authority_group.get("group_digest")
            and raw_group.get("window_authority_digest")
            == child.get("authority_digest")
            and window_interval is not None
            and raw_group.get("window_main_interval") == window_interval
            and isinstance(source_ids, list)
            and source_ids
            and raw_group.get("source_primitive_ids") == source_ids
            and isinstance(primitive_ids, list)
            and primitive_ids
            and primitive_ids == sorted(primitive_ids)
            and len(primitive_ids) == len(set(primitive_ids))
            and set(source_ids).issubset(primitive_ids)
            and isinstance(assignments, list)
            and len(assignments) == len(primitive_ids)
            and raw_group.get("consumer_allowed") is False
        ):
            return None
        expected_window_intervals[str(size_class)] = window_interval
        partition_ids_by_class[str(size_class)] = primitive_ids
        group_observed_ids: list[str] = []
        for assignment in assignments:
            primitive_id = (
                assignment.get("primitive_id")
                if isinstance(assignment, dict)
                else None
            )
            interval = _grouped_manifest_main_interval(
                geometry_by_id.get(primitive_id),
                main_unit=(main_x, main_y),
            )
            if not (
                isinstance(assignment, dict)
                and set(assignment) == assignment_keys
                and type(primitive_id) is str
                and interval is not None
                and assignment.get("main_min") == interval[0]
                and assignment.get("main_max") == interval[1]
                and assignment.get("consumer_allowed") is False
                and (
                    interval[1] < boundary_main
                    if size_class == "primary"
                    else interval[0] > boundary_main
                )
            ):
                return None
            group_observed_ids.append(primitive_id)
            observed_ids.append(primitive_id)
            intervals_by_id[primitive_id] = interval
        if group_observed_ids != primitive_ids:
            return None
    if (
        sorted(observed_ids) != admitted_primitive_ids
        or len(observed_ids) != len(set(observed_ids))
    ):
        return None
    primary_window = expected_window_intervals.get("primary")
    tolerance_window = expected_window_intervals.get("tolerance")
    if not (
        isinstance(primary_window, list)
        and isinstance(tolerance_window, list)
        and primary_window[1] < tolerance_window[0]
        and boundary_main
        == round((primary_window[1] + tolerance_window[0]) / 2.0, 6)
    ):
        return None

    span_ids_by_class: dict[str, list[str]] = {
        "primary": [],
        "tolerance": [],
    }
    span_centers: list[float] = []
    for span in spans:
        primitive_ids = span.get("primitive_ids") if isinstance(
            span, dict
        ) else None
        size_class = span.get("window_size_class") if isinstance(
            span, dict
        ) else None
        if not (
            isinstance(span, dict)
            and size_class in partition_ids_by_class
            and isinstance(primitive_ids, list)
            and primitive_ids
            and primitive_ids == sorted(primitive_ids)
            and len(primitive_ids) == len(set(primitive_ids))
            and set(primitive_ids).issubset(
                partition_ids_by_class[str(size_class)]
            )
        ):
            return None
        intervals = [intervals_by_id.get(item_id) for item_id in primitive_ids]
        if any(interval is None for interval in intervals):
            return None
        absolute_min = round(min(
            interval[0] for interval in intervals if interval is not None
        ), 6)
        absolute_max = round(max(
            interval[1] for interval in intervals if interval is not None
        ), 6)
        if not (
            span.get("absolute_main_min") == absolute_min
            and span.get("absolute_main_max") == absolute_max
        ):
            return None
        span_ids_by_class[str(size_class)].extend(primitive_ids)
        span_centers.append((absolute_min + absolute_max) / 2.0)
    if (
        any(
            right - left <= 1e-9
            for left, right in zip(span_centers, span_centers[1:])
        )
        or any(
            sorted(span_ids_by_class[size_class])
            != partition_ids_by_class[size_class]
            for size_class in ("primary", "tolerance")
        )
    ):
        return None
    return copy.deepcopy(value)


def _replay_grouped_pitch_primitive_partition_v2(
    value: Any,
    *,
    authority: dict[str, Any],
    admitted_primitive_ids: list[str],
    primitive_manifest: Any,
    spans: list[dict[str, Any]],
    semantic_occupation: Any,
) -> dict[str, Any] | None:
    """Replay the grouped numeric split from the exact occupied interval."""
    top_level_keys = {
        "schema_version",
        "method",
        "phrase_id",
        "window_authority_digest",
        "cross_group_relation_digest",
        "semantic_occupation",
        "semantic_occupation_digest",
        "axis_angle_deg",
        "main_unit",
        "primitive_ids",
        "numeric_primitive_ids",
        "occupied_primitive_ids",
        "groups",
        "consumer_allowed",
        "partition_digest",
    }
    binding = semantic_occupation_binding(semantic_occupation)
    if not (
        isinstance(value, dict)
        and set(value) == top_level_keys
        and value.get("schema_version")
        == "vector_pitch_group_primitive_partition_v2"
        and value.get("method")
        == "exact_plus_minus_semantic_occupation_separator"
        and isinstance(semantic_occupation, dict)
        and semantic_occupation.get("schema_version")
        == "vector_pitch_semantic_occupation_v1"
        and binding is not None
        and value.get("semantic_occupation") == binding
        and value.get("semantic_occupation_digest")
        == semantic_occupation.get("occupation_digest")
        and value.get("phrase_id") == authority.get("phrase_id")
        and value.get("window_authority_digest")
        == authority.get("authority_digest")
        and value.get("cross_group_relation_digest")
        == authority.get("cross_group_relation", {}).get(
            "relation_digest"
        )
        and value.get("axis_angle_deg") == authority.get("axis_angle_deg")
        and value.get("main_unit") == authority.get("main_unit")
        and isinstance(admitted_primitive_ids, list)
        and admitted_primitive_ids == sorted(set(admitted_primitive_ids))
        and value.get("primitive_ids") == admitted_primitive_ids
        and value.get("primitive_ids")
        == semantic_occupation.get("primitive_ids")
        and value.get("numeric_primitive_ids")
        == semantic_occupation.get("numeric_primitive_ids")
        and value.get("occupied_primitive_ids")
        == semantic_occupation.get("occupied_primitive_ids")
        and value.get("groups") == semantic_occupation.get("groups")
        and value.get("consumer_allowed") is False
        and isinstance(primitive_manifest, dict)
        and primitive_manifest.get("manifest_digest")
        == semantic_occupation.get("phrase_primitive_manifest_digest")
        and isinstance(spans, list)
        and spans
    ):
        return None
    try:
        expected_digest = _digest({
            key: item
            for key, item in value.items()
            if key != "partition_digest"
        })
    except (OverflowError, TypeError, ValueError):
        return None
    if not (
        _valid_sha256(value.get("partition_digest"))
        and value.get("partition_digest") == expected_digest
    ):
        return None

    intervals_by_id = {
        assignment["primitive_id"]: assignment
        for group in semantic_occupation["groups"]
        for assignment in group["primitive_main_intervals"]
    }
    group_ids_by_class = {
        group["size_class"]: set(group["primitive_ids"])
        for group in semantic_occupation["groups"]
    }
    observed_ids: list[str] = []
    observed_centers: list[float] = []
    observed_classes: list[str] = []
    for span in spans:
        primitive_ids = (
            span.get("primitive_ids")
            if isinstance(span, dict)
            else None
        )
        size_class = (
            span.get("window_size_class")
            if isinstance(span, dict)
            else None
        )
        if not (
            isinstance(span, dict)
            and size_class in {"primary", "tolerance"}
            and isinstance(primitive_ids, list)
            and primitive_ids == sorted(set(primitive_ids))
            and primitive_ids
            and set(primitive_ids).issubset(
                group_ids_by_class[str(size_class)]
            )
        ):
            return None
        intervals = [
            intervals_by_id.get(item_id)
            for item_id in primitive_ids
        ]
        if any(not isinstance(interval, dict) for interval in intervals):
            return None
        absolute_min = round(min(
            float(interval["main_min"])
            for interval in intervals
            if isinstance(interval, dict)
        ), 6)
        absolute_max = round(max(
            float(interval["main_max"])
            for interval in intervals
            if isinstance(interval, dict)
        ), 6)
        if not (
            span.get("absolute_main_min") == absolute_min
            and span.get("absolute_main_max") == absolute_max
        ):
            return None
        observed_ids.extend(primitive_ids)
        observed_centers.append((absolute_min + absolute_max) / 2.0)
        observed_classes.append(str(size_class))
    if not (
        sorted(observed_ids) == value["numeric_primitive_ids"]
        and len(observed_ids) == len(set(observed_ids))
        and observed_classes == sorted(
            observed_classes,
            key=lambda size_class: 0 if size_class == "primary" else 1,
        )
        and any(size_class == "primary" for size_class in observed_classes)
        and any(size_class == "tolerance" for size_class in observed_classes)
        and all(
            right - left > 1e-9
            for left, right in zip(observed_centers, observed_centers[1:])
        )
    ):
        return None
    return copy.deepcopy(value)


def _grouped_manifest_main_interval(
    bbox: Any,
    *,
    main_unit: tuple[float, float],
) -> list[float] | None:
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(type(value) in {int, float} for value in bbox)
    ):
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if (
        not all(math.isfinite(value) for value in (x0, y0, x1, y1))
        or x1 < x0
        or y1 < y0
    ):
        return None
    main_x, main_y = main_unit
    projections = [
        x * main_x + y * main_y
        for x, y in (
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0, y1),
        )
    ]
    return [round(min(projections), 6), round(max(projections), 6)]


def _grouped_authority_window_main_interval(
    child: Any,
    *,
    main_unit: tuple[float, float],
) -> list[float] | None:
    if not isinstance(child, dict):
        return None
    quads = [
        *(child.get("integer_slot_quads") or []),
        child.get("raw_dot_quad"),
        *(child.get("fraction_slot_quads") or []),
    ]
    projected: list[float] = []
    for quad in quads:
        if not isinstance(quad, list) or len(quad) != 4:
            return None
        for point in quad:
            if not isinstance(point, list) or len(point) != 2:
                return None
            x = _finite_number(point[0])
            y = _finite_number(point[1])
            if x is None or y is None:
                return None
            projected.append(x * main_unit[0] + y * main_unit[1])
    if not projected:
        return None
    return [round(min(projected), 6), round(max(projected), 6)]


def _replay_semantic_occupation(
    value: Any,
    *,
    phrase_id: str,
    page_index: Any,
    primitive_manifest: Any,
    window_authority: Any,
) -> dict[str, Any] | None:
    """Replay a full occupation against gate-owned signed context."""
    items = _semantic_occupation_manifest_items(primitive_manifest)
    if items is None or type(page_index) is not int:
        return None
    replayed = normalize_vector_pitch_semantic_occupation(
        value,
        phrase_id=phrase_id,
        page_index=page_index,
        items=items,
        primitive_manifest=primitive_manifest,
        window_authority=window_authority,
    )
    return copy.deepcopy(replayed) if replayed is not None else None


def _semantic_occupation_manifest_items(
    primitive_manifest: Any,
) -> list[dict[str, Any]] | None:
    if not (
        isinstance(primitive_manifest, dict)
        and primitive_manifest.get("schema_version")
        == "vector_pitch_phrase_primitive_manifest_v2"
        and isinstance(primitive_manifest.get("entries"), list)
        and isinstance(primitive_manifest.get("geometry_entries"), list)
    ):
        return None
    geometry_by_id: dict[str, list[float]] = {}
    for row in primitive_manifest["geometry_entries"]:
        primitive_id = (
            row.get("primitive_id")
            if isinstance(row, dict)
            else None
        )
        bbox = row.get("bbox") if isinstance(row, dict) else None
        if not (
            type(primitive_id) is str
            and primitive_id not in geometry_by_id
            and isinstance(bbox, list)
            and len(bbox) == 4
            and all(type(value) in {int, float} for value in bbox)
        ):
            return None
        numbers = [float(value) for value in bbox]
        if (
            any(not math.isfinite(value) for value in numbers)
            or numbers[2] < numbers[0]
            or numbers[3] < numbers[1]
        ):
            return None
        geometry_by_id[primitive_id] = numbers
    items: list[dict[str, Any]] = []
    for row in primitive_manifest["entries"]:
        primitive_id = (
            row.get("primitive_id")
            if isinstance(row, dict)
            else None
        )
        op = row.get("op") if isinstance(row, dict) else None
        bbox = geometry_by_id.get(str(primitive_id))
        if (
            type(primitive_id) is not str
            or type(op) is not str
            or not op
            or bbox is None
        ):
            return None
        x0, y0, x1, y1 = bbox
        items.append({
            "item_id": primitive_id,
            "op": op,
            "bbox": list(bbox),
            "points": [[x0, y0], [x1, y1]],
        })
    return items


def _semantic_bbox_axis_interval(
    bbox: Any,
    *,
    axis_angle_deg: float,
) -> tuple[float, float] | None:
    if isinstance(bbox, dict):
        if set(bbox) != {"x", "y", "w", "h"}:
            return None
        values = [bbox[key] for key in ("x", "y", "w", "h")]
        if any(type(value) not in {int, float} for value in values):
            return None
        x0, y0, width, height = (float(value) for value in values)
        x1 = x0 + width
        y1 = y0 + height
    elif isinstance(bbox, list) and len(bbox) == 4 and all(
        type(value) in {int, float} for value in bbox
    ):
        x0, y0, x1, y1 = (float(value) for value in bbox)
    else:
        return None
    if (
        not all(
            math.isfinite(value)
            for value in (x0, y0, x1, y1, axis_angle_deg)
        )
        or x1 < x0
        or y1 < y0
    ):
        return None
    radians = math.radians(axis_angle_deg)
    axis_x = math.cos(radians)
    axis_y = math.sin(radians)
    projections = [
        x * axis_x + y * axis_y
        for x, y in (
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0, y1),
        )
    ]
    return min(projections), max(projections)


def _semantic_authority_manifest_geometry_valid(
    authority: dict[str, Any],
    *,
    read: dict[str, Any],
    page_context_manifest: Any,
) -> bool:
    manifest = read.get("primitive_manifest")
    debug = read.get("reader_debug")
    adapter = (
        debug.get("r32_base_core_adapter")
        if isinstance(debug, dict)
        else None
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "vector_pitch_phrase_primitive_manifest_v2"
        or not isinstance(page_context_manifest, dict)
        or not isinstance(adapter, dict)
        or adapter.get("primitive_manifest") != manifest
        or not valid_phrase_primitive_manifest(
            manifest,
            phrase_id=str(read.get("phrase_id") or ""),
            page_context_manifest=page_context_manifest,
        )
        or [
            row.get("primitive_id")
            for row in manifest.get("entries") or []
            if isinstance(row, dict)
        ]
        != list(read.get("primitive_ids") or [])
    ):
        return False
    geometry_rows = manifest.get("geometry_entries")
    entry_rows = manifest.get("entries")
    if (
        not isinstance(geometry_rows, list)
        or not isinstance(entry_rows, list)
    ):
        return False
    bbox_by_id = {
        row.get("primitive_id"): row.get("bbox")
        for row in geometry_rows
        if isinstance(row, dict)
    }
    op_by_id = {
        row.get("primitive_id"): row.get("op")
        for row in entry_rows
        if isinstance(row, dict)
    }
    primitive_ids = list(authority["source_primitive_ids"])
    if (
        any(
            primitive_id not in bbox_by_id
            or primitive_id not in op_by_id
            for primitive_id in primitive_ids
        )
        or [
            op_by_id[primitive_id]
            for primitive_id in primitive_ids
        ] != authority["source_primitive_ops"]
    ):
        return False
    bboxes = [
        bbox_by_id[primitive_id]
        for primitive_id in primitive_ids
    ]
    if any(
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(value) not in {int, float} for value in bbox)
        for bbox in bboxes
    ):
        return False
    x0 = min(float(bbox[0]) for bbox in bboxes)
    y0 = min(float(bbox[1]) for bbox in bboxes)
    x1 = max(float(bbox[2]) for bbox in bboxes)
    y1 = max(float(bbox[3]) for bbox in bboxes)
    union = {
        "x": round(x0, 3),
        "y": round(y0, 3),
        "w": round(x1 - x0, 3),
        "h": round(y1 - y0, 3),
    }
    return union == authority["anchor_bbox"]


def _decimal_text_roles(text: str) -> dict[int, str]:
    ordinary = re.fullmatch(
        r"(?:(?:[0-9]{1,3})[*xX×])?(?:Ra|R|Ø)?"
        r"(?P<main>[+\-]?\d{1,3}(?:\.\d{1,3})?)(?:°)?"
        r"(?:±(?P<symmetric>\d(?:\.\d{1,3})?)"
        r"|(?P<deviations>(?:[+\-]\d(?:\.\d{1,3})?)"
        r"(?:/?[+\-]\d(?:\.\d{1,3})?)?))?"
        r"(?:MAX|MIN)?",
        text,
    )
    if ordinary is not None:
        roles: dict[int, str] = {}
        for group_name, role in (
            ("main", "primary"),
            ("symmetric", "tolerance"),
            ("deviations", "tolerance"),
        ):
            value = ordinary.group(group_name)
            if value is None:
                continue
            start = ordinary.start(group_name)
            roles.update({
                start + offset: role
                for offset, char in enumerate(value)
                if char == "."
            })
        return roles
    chamfer = re.fullmatch(
        r"(?P<size>[+\-]?\d{1,3}(?:\.\d{1,3})?)"
        r"[*xX×](?P<angle>\d{1,3}(?:\.\d{1,3})?)°",
        text,
    )
    if chamfer is None:
        return {}
    roles = {}
    for group_name in ("size", "angle"):
        value = chamfer.group(group_name)
        start = chamfer.start(group_name)
        roles.update({
            start + offset: "primary"
            for offset, char in enumerate(value)
            if char == "."
        })
    return roles


def _boundary_gate(read: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    boundary = read.get("boundary")
    if not isinstance(boundary, dict):
        boundary = {}
        reasons.append("phrase_boundary_evidence_missing")
    producer = str(boundary.get("boundary_producer") or "")
    if not (
        boundary.get("schema_version")
        == "r92_directional_walk_boundary_v1"
        and producer == "r92_directional_walk_phrase_v1"
        and boundary.get("consumer_allowed") is False
    ):
        reasons.append("phrase_boundary_producer_invalid")
    negative = boundary.get("negative_termination")
    positive = boundary.get("positive_termination")
    if not _valid_walk_termination(negative, direction="negative"):
        reasons.append("negative_walk_termination_invalid")
    if not _valid_walk_termination(positive, direction="positive"):
        reasons.append("positive_walk_termination_invalid")
    gdt_overlap = boundary.get("gdt_overlap")
    if not _valid_gdt_overlap(gdt_overlap):
        reasons.append("gdt_boundary_evidence_invalid")

    return _gate(
        schema_version=BOUNDARY_SCHEMA_VERSION,
        reasons=reasons,
        evidence={
            "boundary_producer": producer or None,
            "negative_termination_reason": (
                negative.get("reason")
                if isinstance(negative, dict)
                else None
            ),
            "negative_termination_digest": (
                negative.get("semantic_digest")
                if isinstance(negative, dict)
                else None
            ),
            "positive_termination_reason": (
                positive.get("reason")
                if isinstance(positive, dict)
                else None
            ),
            "positive_termination_digest": (
                positive.get("semantic_digest")
                if isinstance(positive, dict)
                else None
            ),
            "gdt_crosses_frame_boundary": (
                gdt_overlap.get("crosses_frame_boundary")
                if isinstance(gdt_overlap, dict)
                else None
            ),
            "gdt_ordinary_eligible": (
                gdt_overlap.get("ordinary_eligible")
                if isinstance(gdt_overlap, dict)
                else None
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    )


def _valid_walk_termination(value: Any, *, direction: str) -> bool:
    if not isinstance(value, dict):
        return False
    semantic_digest = value.get("semantic_digest")
    payload = {
        key: item
        for key, item in value.items()
        if key != "semantic_digest"
    }
    primitive_ids = value.get("primitive_ids")
    candidate_ids = value.get("candidate_dictionary_entry_ids")
    return bool(
        value.get("schema_version") == "r92_walk_termination_v1"
        and value.get("direction") == direction
        and value.get("reason") in DIRECTIONAL_WALK_TERMINATION_REASONS
        and type(value.get("attempted_step_index")) is int
        and value["attempted_step_index"] >= 0
        and isinstance(primitive_ids, list)
        and primitive_ids == sorted(set(primitive_ids))
        and all(isinstance(item, str) and item for item in primitive_ids)
        and isinstance(candidate_ids, list)
        and candidate_ids == sorted(set(candidate_ids))
        and all(isinstance(item, str) and item for item in candidate_ids)
        and value.get("consumer_allowed") is False
        and _valid_sha256(semantic_digest)
        and semantic_digest == _digest(payload)
    )


def _valid_gdt_overlap(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    crosses = value.get("crosses_frame_boundary")
    ordinary_eligible = value.get("ordinary_eligible")
    if (
        crosses is not False
        or type(ordinary_eligible) is not bool
        or value.get("consumer_allowed") is not False
    ):
        return False
    partition_keys = {
        "frame_ids",
        "inside_frame_ids",
        "boundary_frame_ids",
    }
    if not partition_keys.intersection(value):
        return True
    if not partition_keys.issubset(value):
        return False
    frame_ids = value.get("frame_ids")
    inside_ids = value.get("inside_frame_ids")
    boundary_ids = value.get("boundary_frame_ids")
    if not all(
        _valid_unique_id_list(ids)
        for ids in (frame_ids, inside_ids, boundary_ids)
    ):
        return False
    frame_set = set(frame_ids)
    inside_set = set(inside_ids)
    boundary_set = set(boundary_ids)
    return bool(
        inside_set.isdisjoint(boundary_set)
        and frame_set == inside_set | boundary_set
        and crosses is bool(boundary_ids)
        and ordinary_eligible is (not bool(frame_ids))
    )


def _assignment_primitive_ids(
    rows: Any,
    *,
    count_key: str,
    require_char: bool,
    allow_derived_spans: bool = False,
    debug: Any = None,
) -> tuple[list[str], int] | None:
    if not isinstance(rows, list):
        return None
    assigned: list[str] = []
    derived_span_count = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("consumer_allowed") is not False:
            return None
        if require_char and (
            type(row.get("char")) is not str
            or len(row["char"]) != 1
        ):
            return None
        count = _strict_nonnegative_int(row.get(count_key))
        primitive_ids = row.get("primitive_ids")
        if row.get("decision") == (
            "stacked_tolerance_row_separator"
        ) and not (
            row.get("char") == "/"
            and count == 0
            and primitive_ids == []
        ):
            return None
        if count == 0:
            if not (
                allow_derived_spans
                and _valid_stacked_tolerance_derived_span(
                    row=row,
                    row_index=row_index,
                    rows=rows,
                    debug=debug,
                )
            ):
                return None
            derived_span_count += 1
            continue
        if (
            count is None
            or not isinstance(primitive_ids, list)
            or len(primitive_ids) != count
            or any(
                not isinstance(value, str) or not value
                for value in primitive_ids
            )
        ):
            return None
        assigned.extend(primitive_ids)
    if allow_derived_spans:
        if derived_span_count > 1:
            return None
        ownership = (
            debug.get("r32_base_core_adapter", {}).get(
                "primitive_ownership"
            )
            if isinstance(debug, dict)
            and isinstance(debug.get("r32_base_core_adapter"), dict)
            else None
        )
        if derived_span_count:
            if (
                not isinstance(ownership, dict)
                or ownership.get("schema_version")
                != "vector_pitch_primitive_ownership_v1"
                or ownership.get("status") != "applied"
                or ownership.get("synthetic_span_count")
                != derived_span_count
                or ownership.get("consumer_allowed") is not False
            ):
                return None
        elif isinstance(ownership, dict) and ownership.get(
            "synthetic_span_count", 0
        ) != 0:
            return None
    return assigned, derived_span_count


def _primitive_partition_evidence(
    read: Any,
    *,
    page_context_manifest: Any = None,
) -> dict[str, Any]:
    empty = {
        "required": False,
        "valid": True,
        "anchored": False,
        "partition_digest": None,
        "admitted_primitive_count": 0,
        "admitted_primitive_ids": [],
        "exclusion_primitive_ids": [],
        "semantic_occupation_primitive_ids": [],
        "semantic_occupation_digest": None,
    }
    if not isinstance(read, dict):
        return {**empty, "valid": False}
    debug = read.get("reader_debug")
    adapter = (
        debug.get("r32_base_core_adapter")
        if isinstance(debug, dict)
        else None
    )
    if not isinstance(adapter, dict):
        exclusions = read.get("exclusions")
        if isinstance(exclusions, list) and exclusions:
            return {
                **empty,
                "required": True,
                "valid": False,
            }
        return empty
    # R92 projections have no legacy reader adapter.  Its presence is an
    # invalid cutover input, not an invitation to replay quarantined proofs.
    return {
        **empty,
        "required": True,
        "valid": False,
    }

    raw_partition = adapter.get("primitive_partition")
    raw_authority = adapter.get("window_authority")
    raw_lane = adapter.get("anchored_lane_plan")
    raw_character = adapter.get("anchored_character_plan")
    raw_capsule = adapter.get("anchored_capsule_plan")
    anchored_markers = (
        raw_authority is not None,
        raw_lane is not None,
        raw_character is not None,
        raw_capsule is not None,
        debug.get("window_authority_digest") is not None,
        debug.get("selected_pitch_source")
        == "window_authority_pitch",
        debug.get("pitch_origin_mode") == "decimal_lattice",
        debug.get("selected_pitch_source")
        == "window_authority_group_pitches",
        debug.get("pitch_origin_mode") == "grouped_decimal_lattices",
    )
    anchored = any(anchored_markers)
    exclusions = read.get("exclusions")
    nonempty_exclusions = bool(
        isinstance(exclusions, list) and exclusions
    )
    required = bool(
        raw_partition is not None
        or anchored
        or nonempty_exclusions
    )
    if not required:
        return empty
    partition_v2 = normalize_pitch_primitive_partition_v2(raw_partition)
    partition = (
        partition_v2
        if partition_v2 is not None
        else normalize_pitch_primitive_partition(raw_partition)
    )
    primitive_ids = read.get("primitive_ids")
    spans = read.get("spans")
    drops = read.get("drops")
    if not (
        partition is not None
        and isinstance(primitive_ids, list)
        and isinstance(spans, list)
        and isinstance(drops, list)
        and isinstance(exclusions, list)
    ):
        return {
            **empty,
            "required": True,
            "valid": False,
            "anchored": anchored,
        }
    raw_semantic_occupation = adapter.get("semantic_occupation")
    if partition_v2 is not None:
        semantic_occupation = _replay_semantic_occupation(
            raw_semantic_occupation,
            phrase_id=str(read.get("phrase_id") or ""),
            page_index=read.get("page_index"),
            primitive_manifest=adapter.get("primitive_manifest"),
            window_authority=raw_authority,
        )
        rebuilt = (
            build_pitch_primitive_partition_v2(
                phrase_id=str(read.get("phrase_id") or ""),
                input_primitive_ids=sorted(primitive_ids),
                admitted_primitive_ids=list(
                    partition["admitted_primitive_ids"]
                ),
                spans=spans,
                semantic_occupation=semantic_occupation,
                exclusions=exclusions,
                drops=drops,
                window_authority_digest=partition[
                    "window_authority_digest"
                ],
                anchored_lane_partition_digest=partition[
                    "anchored_lane_partition_digest"
                ],
            )
            if semantic_occupation is not None
            else None
        )
    else:
        semantic_occupation = None
        rebuilt = build_pitch_primitive_partition(
            phrase_id=str(read.get("phrase_id") or ""),
            input_primitive_ids=sorted(primitive_ids),
            admitted_primitive_ids=list(
                partition["admitted_primitive_ids"]
            ),
            spans=spans,
            exclusions=exclusions,
            drops=drops,
            window_authority_digest=partition[
                "window_authority_digest"
            ],
            anchored_lane_partition_digest=partition[
                "anchored_lane_partition_digest"
            ],
        )
    valid = bool(
        rebuilt == partition
        and partition["phrase_id"] == read.get("phrase_id")
        and partition["input_primitive_ids"]
        == sorted(primitive_ids)
        and partition["exclusions"] == exclusions
        and (
            partition_v2 is None
            or (
                semantic_occupation is not None
                and partition["semantic_occupation"]
                == semantic_occupation_binding(semantic_occupation)
                and partition["semantic_occupation_digest"]
                == semantic_occupation["occupation_digest"]
            )
        )
    )
    if valid and anchored:
        authority = normalize_pitch_window_authority(raw_authority)
        lane = normalize_anchored_local_lane(raw_lane)
        character = normalize_anchored_character_partition(
            raw_character,
            window_authority=authority,
            anchored_lane_plan=lane,
        )
        primitive_manifest = adapter.get("primitive_manifest")
        bound_lane_exclusions = (
            bind_anchored_lane_exclusions(
                lane,
                window_authority=authority,
            )
            if lane is not None and authority is not None
            else None
        )
        lane_exclusions = [
            row
            for row in exclusions
            if isinstance(row, dict)
            and row.get("stage") == "anchored_lane"
        ]
        character_exclusions = [
            row
            for row in exclusions
            if isinstance(row, dict)
            and row.get("stage") == "anchored_character_filter"
        ]
        capsule_exclusions = [
            row
            for row in exclusions
            if isinstance(row, dict)
            and row.get("stage") == "anchored_capsule_filter"
        ]
        character_exclusion_ids = {
            primitive_id
            for row in character_exclusions
            for primitive_id in row.get("primitive_ids") or []
            if isinstance(primitive_id, str)
        }
        boundary_cut_ids = _read_boundary_cut_primitive_ids(read)
        capsule_binding_valid, capsule = (
            _capsule_partition_binding(
                read=read,
                adapter=adapter,
                partition=partition,
                authority=authority,
                lane=lane,
                character=character,
                primitive_manifest=primitive_manifest,
                boundary_cut_ids=boundary_cut_ids,
                lane_exclusions=lane_exclusions,
                character_exclusions=character_exclusions,
                capsule_exclusions=capsule_exclusions,
            )
        )
        page_manifest_valid = (
            page_context_manifest is None
            or _valid_page_context_manifest(
                page_context_manifest,
                page_index=read.get("page_index"),
            )
        )
        lattice_binding_valid = bool(
            authority is not None
            and (
                _grouped_anchored_lattices_match(
                    debug,
                    authority,
                    spans=spans,
                    admitted_primitive_ids=partition[
                        "admitted_primitive_ids"
                    ],
                    primitive_manifest=primitive_manifest,
                    semantic_occupation=semantic_occupation,
                )
                if authority.get("schema_version")
                == "vector_pitch_window_authority_v2"
                else (
                    debug.get("selected_pitch_source")
                    == "window_authority_pitch"
                    and debug.get("pitch_origin_mode")
                    == "decimal_lattice"
                    and _anchored_lattice_origin_matches(
                        debug,
                        authority,
                    )
                )
            )
        )
        grouped_relation_binding_valid = bool(
            authority is not None
            and authority.get("schema_version")
            != "vector_pitch_window_authority_v2"
        )
        valid = bool(
            authority is not None
            and lane is not None
            and character is not None
            and bound_lane_exclusions is not None
            and lane_exclusions == bound_lane_exclusions
            and character_exclusions == character["exclusions"]
            and (
                len(lane_exclusions)
                + len(character_exclusions)
                + len(capsule_exclusions)
            )
            == len(exclusions)
            and capsule_binding_valid
            and authority["phrase_id"] == read.get("phrase_id")
            and authority["page_index"] == read.get("page_index")
            and lane["phrase_id"] == read.get("phrase_id")
            and lane["input_primitive_ids"]
            == partition["input_primitive_ids"]
            and character["phrase_id"] == read.get("phrase_id")
            and character["authority_digest"]
            == authority["authority_digest"]
            and character["anchored_lane_partition_digest"]
            == lane["partition_digest"]
            and character["axis_angle_deg"]
            == authority["axis_angle_deg"]
            and character["input_primitive_ids"]
            == lane["admitted_primitive_ids"]
            and character["admitted_primitive_ids"]
            == (
                capsule["upstream_admitted_primitive_ids"]
                if capsule is not None
                else partition["admitted_primitive_ids"]
            )
            and character["excluded_primitive_ids"]
            == sorted(character_exclusion_ids)
            and character["protected_primitive_ids"]
            == authority["group_member_source_primitive_ids"]
            and boundary_cut_ids is not None
            and character["boundary_cut_primitive_ids"]
            == boundary_cut_ids
            and partition["window_authority_digest"]
            == authority["authority_digest"]
            and partition["anchored_lane_partition_digest"]
            == lane["partition_digest"]
            and debug.get("window_authority_digest")
            == authority["authority_digest"]
            and debug.get("filtered_item_count") == 0
            and lattice_binding_valid
            and grouped_relation_binding_valid
            and (
                baseline_angle := _finite_number(
                    debug.get("baseline_angle_deg")
                )
            )
            is not None
            and _semantic_axis_angles_match(
                baseline_angle,
                authority["axis_angle_deg"],
            )
            and debug.get("baseline_angle_source")
            == "candidate_axis_angle_deg"
            and valid_phrase_primitive_manifest(
                primitive_manifest,
                phrase_id=str(read.get("phrase_id") or ""),
                page_context_manifest=page_context_manifest,
            )
            and primitive_manifest.get("manifest_digest")
            == authority["phrase_primitive_manifest_digest"]
            and (
                page_context_manifest is None
                or authority["page_context_primitive_sha256"]
                == page_context_manifest.get("primitive_sha256")
            )
            and page_manifest_valid
            and not (
                set(authority["group_member_source_primitive_ids"])
                & {
                    primitive_id
                    for row in exclusions
                    for primitive_id in row.get("primitive_ids") or []
                }
            )
        )
    exclusion_ids = [
        primitive_id
        for row in partition["exclusions"]
        for primitive_id in row["primitive_ids"]
    ]
    return {
        "required": True,
        "valid": valid,
        "anchored": anchored,
        "partition_digest": partition["partition_digest"],
        "admitted_primitive_count": partition[
            "admitted_primitive_count"
        ],
        "admitted_primitive_ids": list(
            partition["admitted_primitive_ids"]
        ),
        "exclusion_primitive_ids": exclusion_ids if valid else [],
        "semantic_occupation_primitive_ids": (
            list(partition.get("occupied_primitive_ids") or [])
            if valid and partition_v2 is not None
            else []
        ),
        "semantic_occupation_digest": (
            partition.get("semantic_occupation_digest")
            if valid and partition_v2 is not None
            else None
        ),
    }


def _capsule_partition_binding(
    *,
    read: dict[str, Any],
    adapter: dict[str, Any],
    partition: dict[str, Any],
    authority: dict[str, Any] | None,
    lane: dict[str, Any] | None,
    character: dict[str, Any] | None,
    primitive_manifest: Any,
    boundary_cut_ids: list[str] | None,
    lane_exclusions: list[dict[str, Any]],
    character_exclusions: list[dict[str, Any]],
    capsule_exclusions: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None]:
    raw_capsule = adapter.get("anchored_capsule_plan")
    raw_phrase_authority = adapter.get(
        "capsule_phrase_authority"
    )
    raw_topology = adapter.get("capsule_topology")
    present = (
        raw_capsule is not None,
        raw_phrase_authority is not None,
        raw_topology is not None,
    )
    if not any(present):
        return not capsule_exclusions, None
    if (
        not all(present)
        or not isinstance(raw_capsule, dict)
        or vector_pitch_capsule_partition_identity_reasons(
            raw_capsule
        )
        or authority is None
        or lane is None
        or character is None
        or not isinstance(primitive_manifest, dict)
        or boundary_cut_ids is None
        or not isinstance(raw_topology, dict)
        or vector_capsule_topology_identity_reasons(raw_topology)
        or raw_topology.get("status") != "accepted"
        or raw_topology.get("reason")
        != "unique_closed_outer_contour"
    ):
        return False, None
    phrase_authority = normalize_capsule_phrase_authority_v1(
        raw_phrase_authority
    )
    if phrase_authority is None:
        return False, None

    primitive_ids = list(read.get("primitive_ids") or [])
    topology_boundary_ids = set(
        raw_topology["boundary_primitive_ids"]
    )
    topology_phrase_boundary_ids = list(
        raw_topology["phrase_boundary_primitive_ids"]
    )
    capsule_exclusion_ids = sorted({
        primitive_id
        for row in capsule_exclusions
        for primitive_id in row.get("primitive_ids") or []
    })
    upstream_exclusion_ids = sorted({
        primitive_id
        for row in [*lane_exclusions, *character_exclusions]
        for primitive_id in row.get("primitive_ids") or []
    })
    non_boundary_ids = {
        row["primitive_id"]
        for row in raw_topology[
            "non_boundary_component_primitives"
        ]
    }
    interior_ids = set(
        raw_topology["interior_phrase_primitive_ids"]
    )
    valid = bool(
        raw_capsule["phrase_id"] == read.get("phrase_id")
        and raw_capsule["physical_core_digest"]
        == read.get("physical_core_digest")
        and raw_capsule["phrase_authority_digest"]
        == phrase_authority["authority_digest"]
        and raw_capsule["topology_digest"]
        == raw_topology["topology_digest"]
        == phrase_authority["topology_digest"]
        and raw_capsule["capsule_id"] == raw_topology["capsule_id"]
        and raw_capsule["page_context_primitive_sha256"]
        == raw_topology["page_context_primitive_sha256"]
        == phrase_authority["page_context_primitive_sha256"]
        and raw_capsule["phrase_primitive_manifest_digest"]
        == phrase_authority["phrase_primitive_manifest_digest"]
        == primitive_manifest.get("manifest_digest")
        and raw_capsule["window_authority_digest"]
        == authority["authority_digest"]
        and raw_capsule["anchored_lane_partition_digest"]
        == lane["partition_digest"]
        and raw_capsule["anchored_character_partition_digest"]
        == character["partition_digest"]
        and raw_capsule["axis_angle_deg"]
        == phrase_authority["axis_angle_deg"]
        == raw_topology["axis_angle_deg"]
        == authority["axis_angle_deg"]
        and raw_capsule["axis_angle_source"]
        == phrase_authority["axis_angle_source"]
        == authority["axis_angle_source"]
        and raw_capsule["input_primitive_ids"]
        == partition["input_primitive_ids"]
        == sorted(primitive_ids)
        and raw_capsule["upstream_admitted_primitive_ids"]
        == character["admitted_primitive_ids"]
        and raw_capsule["upstream_excluded_primitive_ids"]
        == upstream_exclusion_ids
        and raw_capsule["admitted_primitive_ids"]
        == partition["admitted_primitive_ids"]
        and raw_capsule["excluded_primitive_ids"]
        == topology_phrase_boundary_ids
        == capsule_exclusion_ids
        and raw_capsule["excluded_primitive_ids"]
        == sorted(topology_boundary_ids.intersection(primitive_ids))
        and raw_capsule["boundary_cut_primitive_ids"]
        == boundary_cut_ids
        and bool(boundary_cut_ids)
        and set(boundary_cut_ids).issubset(topology_boundary_ids)
        and raw_capsule["exclusions"] == capsule_exclusions
        and len(capsule_exclusions) == 1
        and not set(raw_capsule["excluded_primitive_ids"]).intersection(
            non_boundary_ids
        )
        and interior_ids.issubset(primitive_ids)
        and not interior_ids.intersection(topology_boundary_ids)
        and phrase_authority["phrase_id"] == read.get("phrase_id")
        and phrase_authority["physical_core_digest"]
        == read.get("physical_core_digest")
        and type(read.get("page_index")) is int
        and raw_topology["page_num"] == read["page_index"] + 1
    )
    return valid, raw_capsule if valid else None


def _read_boundary_cut_primitive_ids(
    read: dict[str, Any],
) -> list[str] | None:
    boundary = read.get("boundary")
    if not isinstance(boundary, dict):
        return None
    primitive_ids: list[str] = []
    for side in ("left", "right"):
        side_boundary = boundary.get(side)
        if not isinstance(side_boundary, dict):
            return None
        cut_ids = side_boundary.get("cut_primitive_ids")
        if (
            not isinstance(cut_ids, list)
            or any(
                type(primitive_id) is not str or not primitive_id
                for primitive_id in cut_ids
            )
        ):
            return None
        primitive_ids.extend(cut_ids)
    canonical = sorted(set(primitive_ids))
    return canonical


def _valid_primitive_ownership_contract(read: Any) -> bool:
    """Reject the quarantined R2P ownership proof if it reappears.

    Native R92 rows have no base-core adapter, so this function is called only
    for a legacy payload.  Failing closed here avoids importing the moved R2P
    implementation or pretending its assignment proof still exists.
    """
    return False


def _valid_stacked_tolerance_derived_span(
    *,
    row: dict[str, Any],
    row_index: int,
    rows: list[Any],
    debug: Any,
) -> bool:
    if row_index == 0 or row_index + 1 >= len(rows):
        return False
    left = rows[row_index - 1]
    right = rows[row_index + 1]
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_ids = left.get("primitive_ids")
    right_ids = right.get("primitive_ids")
    if (
        not isinstance(left_ids, list)
        or not left_ids
        or not isinstance(right_ids, list)
        or not right_ids
    ):
        return False
    evidence = row.get("derived_evidence")
    if not isinstance(evidence, dict):
        return False
    if not isinstance(debug, dict):
        return False
    layout = debug.get("multiscale_tolerance_layout")
    adapter = debug.get("r32_base_core_adapter")
    span_text = "".join(
        str(span.get("char") or "")
        for span in rows
        if isinstance(span, dict)
    )
    stacked_match = re.fullmatch(
        r"(?P<main>(?:\d+\*)?(?:Ø)?\d+(?:\.\d+)?)"
        r"(?P<upper>[+±]\d+(?:\.\d+)?)/"
        r"(?P<lower>[+\-]\d+(?:\.\d+)?)",
        span_text,
    )
    if stacked_match is None:
        return False
    main_text = str(stacked_match.group("main"))
    upper_text = str(stacked_match.group("upper"))
    lower_text = str(stacked_match.group("lower"))
    expected_separator_index = len(main_text) + len(upper_text)
    hard_gates = (
        layout.get("hard_gates")
        if isinstance(layout, dict)
        else None
    )
    return bool(
        row.get("char") == "/"
        and row.get("decision") == "stacked_tolerance_row_separator"
        and row.get("core_slot_group_count") == 0
        and row.get("primitive_ids") == []
        and debug.get("algorithm")
        == "multiscale_pitch_slots_with_tolerance_layout"
        and isinstance(layout, dict)
        and layout.get("schema_version")
        == "r2p_multiscale_tolerance_layout_v1"
        and layout.get("layout_kind") == "stacked_tolerance_rows"
        and layout.get("assembled_text") == span_text
        and layout.get("row_texts")
        == [upper_text, main_text, lower_text]
        and layout.get("used_row_indices") == [0, 1, 2]
        and layout.get("ignored_layout_rows") == []
        and type(layout.get("token_count")) is int
        and layout.get("token_count") == len(rows) - 1
        and type(layout.get("group_count")) is int
        and layout.get("group_count") > 0
        and isinstance(hard_gates, dict)
        and hard_gates.get("orientation") in {"H", "V"}
        and hard_gates.get("row_count") == 3
        and hard_gates.get("signed_tolerance_row_count") == 2
        and hard_gates.get("assembled_tolerance_text") is True
        and hard_gates.get("consumer_allowed") is False
        and layout.get("consumer_allowed") is False
        and isinstance(adapter, dict)
        and adapter.get("schema_version")
        == "vector_pitch_base_core_adapter_v1"
        and adapter.get("consumer_allowed") is False
        and evidence.get("schema_version")
        == "vector_pitch_derived_span_evidence_v1"
        and evidence.get("producer")
        == "vector_pitch_primitive_ownership_v1"
        and evidence.get("kind")
        == "stacked_tolerance_row_separator"
        and evidence.get("source_algorithm")
        == "multiscale_pitch_slots_with_tolerance_layout"
        and evidence.get("source_layout_kind")
        == "stacked_tolerance_rows"
        and evidence.get("left_span_index") == row_index - 1
        and evidence.get("right_span_index") == row_index + 1
        and row_index == expected_separator_index
        and evidence.get("left_primitive_ids") == left_ids
        and evidence.get("right_primitive_ids") == right_ids
        and evidence.get("consumer_allowed") is False
    )


def _gate_transform_manifest_items(
    primitive_manifest: Any,
    *,
    semantic_occupation: Any = None,
) -> list[dict[str, Any]] | None:
    if not (
        isinstance(primitive_manifest, dict)
        and isinstance(primitive_manifest.get("entries"), list)
    ):
        return None
    if semantic_occupation is not None:
        return _semantic_occupation_manifest_items(primitive_manifest)
    geometry_by_id = {
        str(row["primitive_id"]): list(row["bbox"])
        for row in primitive_manifest.get("geometry_entries") or []
        if isinstance(row, dict)
        and type(row.get("primitive_id")) is str
        and isinstance(row.get("bbox"), list)
    }
    items = [
        {
            "item_id": str(row["primitive_id"]),
            "op": str(row["op"]),
            **(
                {"bbox": list(geometry_by_id[str(row["primitive_id"])])}
                if str(row["primitive_id"]) in geometry_by_id
                else {}
            ),
        }
        for row in primitive_manifest["entries"]
        if isinstance(row, dict)
        and type(row.get("primitive_id")) is str
        and type(row.get("op")) is str
    ]
    return items if len(items) == len(primitive_manifest["entries"]) else None


def _authoritative_transform_ledger_evidence(
    *,
    debug: Any,
    raw_text: str,
    final_text: str,
    expected_primitive_ids: set[str],
    span_rows: list[Any],
    reader_dependency: Any,
    page_context_manifest: Any,
    read: Any,
) -> tuple[list[str], dict[str, Any]]:
    ledger = debug.get("transform_ledger") if isinstance(debug, dict) else None
    empty_evidence = {
        "transform_step_count": 0,
        "transform_replay_text": raw_text,
        "transform_ledger_digest": _digest([]),
        "transform_authority_valid": False,
        "consumed_transform_drop_indices": [],
        "consumed_semantic_occupation_indices": [],
        "semantic_occupation_digest": None,
    }
    no_transform = raw_text == final_text and ledger in (None, [])
    semantic_carrier_present = bool(
        isinstance(read, dict)
        and any(
            isinstance(source, dict)
            and "semantic_anchor_authorities" in source
            for source in read.get("source_candidate_evidence") or []
        )
    )
    if no_transform:
        if (
            not isinstance(debug, dict)
            or debug.get("repairs") not in (None, [])
            or not isinstance(read, dict)
            or read.get("repairs") not in (None, [])
        ):
            return ["transform_repair_ledger_mismatch"], empty_evidence
        if not semantic_carrier_present:
            return [], {
                **empty_evidence,
                "transform_authority_valid": True,
            }
    if not no_transform and (
        not isinstance(ledger, list) or not ledger
    ):
        reasons = ["transform_ledger_missing"]
        if (
            not isinstance(debug, dict)
            or debug.get("repairs") not in (None, [])
            or not isinstance(read, dict)
            or read.get("repairs") not in (None, [])
        ):
            reasons.append("transform_repair_ledger_mismatch")
        return sorted(set(reasons)), empty_evidence

    phrase_id = str(read.get("phrase_id") or "") if isinstance(read, dict) else ""
    adapter = (
        debug.get("r32_base_core_adapter")
        if isinstance(debug, dict)
        else None
    )
    ownership = (
        adapter.get("primitive_ownership")
        if isinstance(adapter, dict)
        else None
    )
    row_manifest = read.get("primitive_manifest") if isinstance(read, dict) else None
    adapter_manifest = (
        adapter.get("primitive_manifest")
        if isinstance(adapter, dict)
        else None
    )
    provenance_valid = bool(
        _valid_base_reader_dependency(reader_dependency)
        and isinstance(adapter, dict)
        and adapter.get("producer") == BASE_ADAPTER_PRODUCER
        and _valid_base_adapter_identity(read, phrase_id=phrase_id)
        and _valid_primitive_ownership_contract(read)
    )
    if not provenance_valid:
        return ["transform_reader_provenance_invalid"], empty_evidence
    manifest_valid = bool(
        adapter_manifest == row_manifest
        and valid_phrase_primitive_manifest(
            row_manifest,
            phrase_id=phrase_id,
            page_context_manifest=page_context_manifest,
        )
        and isinstance(row_manifest, dict)
        and [
            row.get("primitive_id")
            for row in row_manifest.get("entries") or []
            if isinstance(row, dict)
        ]
        == list(read.get("primitive_ids") or [])
        and {
            str(row.get("primitive_id"))
            for row in row_manifest.get("entries") or []
            if isinstance(row, dict)
        }
        == expected_primitive_ids
    )
    if not manifest_valid:
        return ["transform_primitive_manifest_invalid"], empty_evidence
    expected_identity = build_base_pitch_adapter_identity(
        phrase_id=phrase_id,
        base_reader_dependency=reader_dependency,
        primitive_ownership=ownership,
        primitive_manifest=row_manifest,
    )
    if expected_identity is None or adapter.get("identity") != expected_identity:
        return ["transform_reader_provenance_invalid"], empty_evidence
    if no_transform:
        return [], {
            **empty_evidence,
            "transform_authority_valid": True,
            "primitive_manifest_digest": row_manifest.get(
                "manifest_digest"
            ),
        }

    replay_debug = copy.deepcopy(debug)
    replay_debug.pop("transform_ledger", None)
    replay_debug.pop("repairs", None)
    semantic_occupation = (
        _replay_semantic_occupation(
            adapter.get("semantic_occupation"),
            phrase_id=phrase_id,
            page_index=read.get("page_index"),
            primitive_manifest=row_manifest,
            window_authority=adapter.get("window_authority"),
        )
        if adapter.get("semantic_occupation") is not None
        else None
    )
    manifest_items = _gate_transform_manifest_items(
        row_manifest,
        semantic_occupation=semantic_occupation,
    )
    if manifest_items is None:
        return ["transform_primitive_manifest_invalid"], empty_evidence
    expected_ledger = build_base_pitch_transform_ledger(
        phrase_id=phrase_id,
        final_text=final_text,
        items=manifest_items,
        spans=copy.deepcopy(span_rows),
        drops=copy.deepcopy(list(read.get("drops") or [])),
        debug=replay_debug,
        base_reader_dependency=copy.deepcopy(reader_dependency),
        base_adapter=copy.deepcopy(adapter),
        source_candidate_evidence=copy.deepcopy(
            list(read.get("source_candidate_evidence") or [])
        ),
        candidate_axis_angle_deg=read.get("axis_angle_deg"),
    )
    reasons: list[str] = []
    if not expected_ledger:
        reasons.append("transform_not_authorized")
    elif ledger != expected_ledger:
        reasons.append("transform_ledger_not_authoritative")
    expected_repairs = build_base_pitch_transform_repairs(expected_ledger)
    if (
        debug.get("repairs") != expected_repairs
        or not isinstance(read, dict)
        or read.get("repairs") != expected_repairs
    ):
        reasons.append("transform_repair_ledger_mismatch")
    replay_text = (
        str(expected_ledger[-1]["after_text"])
        if expected_ledger
        else raw_text
    )
    if replay_text != final_text:
        reasons.append("transform_replay_mismatch")
    consumed_drop_indices = (
        list(expected_ledger[-1].get("consumed_drop_indices") or [])
        if expected_ledger and not reasons
        else []
    )
    consumed_semantic_occupation_indices = (
        list(
            expected_ledger[-1].get(
                "consumed_semantic_occupation_indices"
            )
            or []
        )
        if expected_ledger and not reasons
        else []
    )
    semantic_occupation_digest = (
        expected_ledger[-1].get("semantic_occupation_digest")
        if expected_ledger and not reasons
        else None
    )
    return sorted(set(reasons)), {
        "transform_step_count": len(ledger),
        "transform_replay_text": replay_text,
        "transform_ledger_digest": _digest(expected_ledger),
        "transform_authority_valid": not reasons,
        "consumed_transform_drop_indices": consumed_drop_indices,
        "consumed_semantic_occupation_indices": (
            consumed_semantic_occupation_indices
        ),
        "semantic_occupation_digest": semantic_occupation_digest,
        "primitive_manifest_digest": row_manifest.get("manifest_digest"),
    }


def _transform_ledger_evidence(
    *,
    debug: Any,
    raw_text: str,
    final_text: str,
    expected_primitive_ids: set[str],
    span_rows: list[Any],
    span_primitive_groups: list[list[str]],
    reader_dependency: Any = None,
    page_context_manifest: Any = None,
    read: Any = None,
) -> tuple[list[str], dict[str, Any]]:
    return _authoritative_transform_ledger_evidence(
        debug=debug,
        raw_text=raw_text,
        final_text=final_text,
        expected_primitive_ids=expected_primitive_ids,
        span_rows=span_rows,
        reader_dependency=reader_dependency,
        page_context_manifest=page_context_manifest,
        read=read,
    )


def _gate(
    *,
    schema_version: str,
    reasons: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    canonical_reasons = sorted(set(str(reason) for reason in reasons if reason))
    passed = not canonical_reasons
    return {
        "schema_version": schema_version,
        "status": "pass" if passed else "fail_closed",
        "passed": passed,
        "reasons": canonical_reasons,
        "evidence": evidence,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _failed_gate(schema_version: str, reason: str) -> dict[str, Any]:
    return _gate(
        schema_version=schema_version,
        reasons=[reason],
        evidence={"consumer_allowed": CONSUMER_ALLOWED},
    )


def _strict_nonnegative_int(value: Any) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _nonnegative_number(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and number >= 0.0


def _positive_number(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and number > 0.0


def _finite_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_id_list(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and all(isinstance(item, str) and bool(item) for item in value)
    )


def _valid_unique_id_list(value: Any) -> bool:
    return bool(
        _valid_id_list(value)
        and len(value) == len(set(value))
    )


def _recursive_invalid_consumer_count(value: Any) -> int:
    if isinstance(value, dict):
        return int(
            "consumer_allowed" in value
            and value["consumer_allowed"] is not False
        ) + sum(
            _recursive_invalid_consumer_count(child)
            for child in value.values()
        )
    if isinstance(value, list):
        return sum(_recursive_invalid_consumer_count(child) for child in value)
    return 0


def _recursive_true_key_count(value: Any, key: str) -> int:
    if isinstance(value, dict):
        return int(value.get(key) is True) + sum(
            _recursive_true_key_count(child, key)
            for child in value.values()
        )
    if isinstance(value, list):
        return sum(_recursive_true_key_count(child, key) for child in value)
    return 0


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
