"""Same-trace, dump-only L4 recovery routing for strict vector phrases."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from vector_phrase_chain_validation import (
    request_local_artifact_safety_reasons,
)


DUMP_SCHEMA_VERSION = "vector_phrase_l4_shadow_dump_v1"
UNRESOLVED_SCHEMA_VERSION = "vector_phrase_l4_unresolved_v1"
CONSUMER_ALLOWED = False
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
MAX_JSON_NESTING_DEPTH = 128


def build_vector_phrase_l4_shadow_dump(
    *,
    phrase_dump: dict[str, Any],
    pitch_read_dump: dict[str, Any],
    quality_gate_dump: dict[str, Any],
    hypothesis_dump: dict[str, Any],
    spatial_dedup_dump: dict[str, Any],
    l3_shadow_dump: dict[str, Any],
) -> dict[str, Any]:
    """Route L3 suspects without rerunning a reader or scanning the page."""
    trace_id = str(
        l3_shadow_dump.get("trace_id") or ""
        if isinstance(l3_shadow_dump, dict)
        else ""
    )
    page_index = (
        l3_shadow_dump.get("page_index")
        if isinstance(l3_shadow_dump, dict)
        else -1
    )
    reasons = _l4_input_structure_reasons(l3_shadow_dump)
    if _l4_artifact_safety_reasons(l3_shadow_dump):
        reasons.append("l3_artifact_safety_invalid")
    if reasons:
        return _failed_dump(
            trace_id=trace_id,
            page_index=page_index,
            reasons=reasons,
        )
    source_l3 = l3_shadow_dump

    unresolved = sorted(
        (
            _unresolved_row(
                row,
                source_l3_semantic_digest=str(
                    source_l3["semantic_digest"]
                ),
            )
            for row in source_l3["L3_suspects"]
        ),
        key=lambda row: str(row["phrase_id"]),
    )
    stats = {
        "input_l3_dimension_count": len(source_l3["L3_dimensions"]),
        "input_l3_suspect_count": len(source_l3["L3_suspects"]),
        "input_l3_gdt_routed_count": len(source_l3["L3_gdt_routed"]),
        "recovery_attempt_count": 0,
        "recovered_count": 0,
        "unresolved_count": len(unresolved),
        "reader_call_count_delta": 0,
        "bbox_query_count_delta": 0,
        "template_classification_count_delta": 0,
        "feeds_display_l4_count": 0,
        "consumer_allowed_count": 0,
    }
    trace_relation = "same_trace_l3_l4_shadow"
    recovery_policy = {
        "schema_version": "vector_phrase_l4_recovery_policy_v1",
        "mode": "strict_existing_evidence_only",
        "approved_recovery_producer": None,
        "full_page_search_allowed": False,
        "reader_recall_allowed": False,
        "free_angle_search_allowed": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    semantic_payload = {
        "trace_relation": trace_relation,
        "source_l3_semantic_digest": str(source_l3["semantic_digest"]),
        "reader_provenance_status": str(
            source_l3["reader_provenance_status"]
        ),
        "recognition_quality_claim_allowed": False,
        "recovery_policy": recovery_policy,
        "L4_recovered": [],
        "L4_unresolved_suspects": unresolved,
        "stats": stats,
        "feeds_display_l4": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": str(source_l3["trace_id"]),
        "page_index": int(source_l3["page_index"]),
        "page_context_manifest": copy.deepcopy(
            source_l3["page_context_manifest"]
        ),
        "status": "ok",
        "error": None,
        "trace_relation": trace_relation,
        "source_l3_semantic_digest": str(source_l3["semantic_digest"]),
        "reader_provenance_status": str(
            source_l3["reader_provenance_status"]
        ),
        "recognition_quality_claim_allowed": False,
        "recovery_policy": recovery_policy,
        "L4_recovered": [],
        "L4_unresolved_suspects": unresolved,
        "stats": stats,
        "semantic_digest": _digest(semantic_payload),
        "feeds_display_l4": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _unresolved_row(
    source: dict[str, Any],
    *,
    source_l3_semantic_digest: str,
) -> dict[str, Any]:
    primary_reason = str(source.get("primary_reason") or "")
    first_drop_stage = str(source.get("first_drop_stage") or "") or None
    if primary_reason in {"coverage", "format", "boundary"}:
        recovery_class = "strict_quality_gate_failure"
        recovery_status = "not_recoverable_strict_gate_failed"
    elif primary_reason in {
        "spatial_value_conflict",
        "same_value_bridge_ambiguity",
    }:
        recovery_class = "spatial_ambiguity"
        recovery_status = "not_recoverable_spatial_ambiguity"
    elif primary_reason == "semantic_assembly_unsupported":
        recovery_class = "semantic_existing_evidence_candidate"
        recovery_status = "not_attempted_missing_approved_recovery_producer"
    else:
        recovery_class = "geometry_or_upstream_rejection"
        recovery_status = "not_recoverable_geometry_or_upstream"
    return {
        "schema_version": UNRESOLVED_SCHEMA_VERSION,
        "trace_id": str(source["trace_id"]),
        "page_index": int(source["page_index"]),
        "phrase_id": str(source["phrase_id"]),
        "physical_core_digest": str(source["physical_core_digest"]),
        "source_l3_semantic_digest": source_l3_semantic_digest,
        "source_text": str(source.get("source_text") or ""),
        "canonical_text": str(source.get("canonical_text") or ""),
        "bbox": copy.deepcopy(source.get("bbox")),
        "key_candidate": source.get("key_candidate") is True,
        "source_primary_reason": primary_reason,
        "source_first_drop_stage": first_drop_stage,
        "quality_gate_digest": str(
            source.get("quality_gate_digest") or ""
        ),
        "quality_gate_reference": copy.deepcopy(
            source.get("quality_gate_reference")
        ),
        "geometry_gate": copy.deepcopy(source.get("geometry_gate")),
        "source_provenance": copy.deepcopy(
            source.get("source_provenance")
        ),
        "dedup_evidence": copy.deepcopy(source.get("dedup_evidence")),
        "ambiguity_evidence": copy.deepcopy(
            source.get("ambiguity_evidence")
        ),
        "recovery_class": recovery_class,
        "recovery_status": recovery_status,
        "reader_call_count_delta": 0,
        "bbox_query_count_delta": 0,
        "template_classification_count_delta": 0,
        "feeds_display_l4": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _l4_input_structure_reasons(value: Any) -> list[str]:
    """Validate the carried L3 rows without reconstructing their producers."""
    if not isinstance(value, dict):
        return ["l3_structure_invalid"]
    trace_id = value.get("trace_id")
    page_index = value.get("page_index")
    reasons: list[str] = []
    if not (
        value.get("schema_version") == "vector_phrase_l3_shadow_dump_v1"
        and value.get("status") == "ok"
        and value.get("error") is None
        and type(trace_id) is str
        and bool(trace_id)
        and type(page_index) is int
        and page_index >= 0
        and isinstance(value.get("page_context_manifest"), dict)
        and isinstance(value.get("L3_dimensions"), list)
        and isinstance(value.get("L3_suspects"), list)
        and isinstance(value.get("L3_gdt_routed"), list)
        and isinstance(value.get("L3_key_index"), list)
        and type(value.get("reader_provenance_status")) is str
        and bool(value["reader_provenance_status"])
        and type(value.get("semantic_digest")) is str
        and bool(value["semantic_digest"])
        and value.get("recognition_quality_claim_allowed") is False
        and value.get("feeds_display_l3") is False
        and value.get("ocr_called") is False
        and value.get("consumer_allowed") is False
    ):
        reasons.append("l3_structure_invalid")
        return reasons
    for collection in (
        "L3_dimensions",
        "L3_suspects",
        "L3_gdt_routed",
        "L3_key_index",
    ):
        if any(
            not isinstance(row, dict)
            or row.get("trace_id") != trace_id
            or row.get("page_index") != page_index
            for row in value[collection]
        ):
            reasons.append("l3_row_trace_invalid")
    if any(
        not _l3_suspect_shape(row, trace_id, page_index)
        for row in value["L3_suspects"]
    ):
        reasons.append("l3_suspect_structure_invalid")
    if any(
        not _l3_dimension_shape(row, trace_id, page_index)
        for row in value["L3_dimensions"]
    ):
        reasons.append("l3_dimension_structure_invalid")
    return sorted(set(reasons))


def _l4_artifact_safety_reasons(value: Any) -> list[str]:
    """Allow the intentional L3 eligibility marker; retain all safety bans."""
    return [
        reason
        for reason in request_local_artifact_safety_reasons(
            value,
            path="l3",
        )
        if not (
            reason.startswith("dump_only_flag_nonfalse:")
            and reason.endswith(".l3_shadow_eligible")
        )
    ]


def _l3_suspect_shape(
    row: Any,
    trace_id: str,
    page_index: int,
) -> bool:
    return bool(
        isinstance(row, dict)
        and row.get("trace_id") == trace_id
        and row.get("page_index") == page_index
        and type(row.get("phrase_id")) is str
        and bool(row["phrase_id"])
        and type(row.get("physical_core_digest")) is str
        and bool(row["physical_core_digest"])
        and type(row.get("source_text")) is str
        and type(row.get("canonical_text")) is str
        and isinstance(row.get("bbox"), dict)
        and type(row.get("key_candidate")) is bool
        and type(row.get("primary_reason")) is str
        and bool(row["primary_reason"])
        and isinstance(row.get("quality_gate_reference"), dict)
        and isinstance(row.get("geometry_gate"), dict)
        and isinstance(row.get("source_provenance"), dict)
        and isinstance(row.get("dedup_evidence"), dict)
        and row.get("feeds_display_l3") is False
        and row.get("ocr_called") is False
        and row.get("consumer_allowed") is False
    )


def _l3_dimension_shape(
    row: Any,
    trace_id: str,
    page_index: int,
) -> bool:
    return bool(
        isinstance(row, dict)
        and row.get("trace_id") == trace_id
        and row.get("page_index") == page_index
        and type(row.get("phrase_id")) is str
        and bool(row["phrase_id"])
        and type(row.get("physical_core_digest")) is str
        and bool(row["physical_core_digest"])
        and type(row.get("source_text")) is str
        and type(row.get("canonical_text")) is str
        and isinstance(row.get("dimension_semantics"), dict)
        and isinstance(row.get("bbox"), dict)
        and isinstance(row.get("oriented_quad"), list)
        and type(row.get("axis_angle_deg")) in {int, float}
        and type(row.get("axis_angle_source")) is str
        and type(row.get("key_candidate")) is bool
        and type(row.get("three_gates_passed")) is bool
        and isinstance(row.get("quality_gate_reference"), dict)
        and isinstance(row.get("geometry_gate"), dict)
        and isinstance(row.get("gdt_route_evidence"), dict)
        and isinstance(row.get("source_provenance"), dict)
        and isinstance(row.get("dedup_evidence"), dict)
        and row.get("feeds_display_l3") is False
        and row.get("ocr_called") is False
        and row.get("consumer_allowed") is False
    )


def _failed_dump(
    *,
    trace_id: str,
    page_index: Any,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": _safe_trace_id(trace_id),
        "page_index": _safe_page_index(page_index),
        "status": "fail_closed",
        "error": {
            "schema_version": "vector_phrase_l4_shadow_error_v1",
            "code": "invalid_l4_shadow_input",
            "reasons": sorted(set(reasons)),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "recognition_quality_claim_allowed": False,
        "L4_recovered": [],
        "L4_unresolved_suspects": [],
        "stats": {
            "input_l3_dimension_count": 0,
            "input_l3_suspect_count": 0,
            "input_l3_gdt_routed_count": 0,
            "recovery_attempt_count": 0,
            "recovered_count": 0,
            "unresolved_count": 0,
            "reader_call_count_delta": 0,
            "bbox_query_count_delta": 0,
            "template_classification_count_delta": 0,
            "feeds_display_l4_count": 0,
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest([]),
        "feeds_display_l4": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _strict_json_tree(
    value: Any,
    *,
    ancestor_ids: set[int],
    depth: int,
) -> bool:
    if depth > MAX_JSON_NESTING_DEPTH:
        return False
    if type(value) is dict:
        container_id = id(value)
        if container_id in ancestor_ids:
            return False
        next_ancestors = ancestor_ids | {container_id}
        return all(
            type(key) is str
            and _utf8_encodable(key)
            and _strict_json_tree(
                child,
                ancestor_ids=next_ancestors,
                depth=depth + 1,
            )
            for key, child in value.items()
        )
    if type(value) is list:
        container_id = id(value)
        if container_id in ancestor_ids:
            return False
        next_ancestors = ancestor_ids | {container_id}
        return all(
            _strict_json_tree(
                child,
                ancestor_ids=next_ancestors,
                depth=depth + 1,
            )
            for child in value
        )
    if type(value) is str:
        return _utf8_encodable(value)
    if value is None or type(value) is bool:
        return True
    if type(value) is int:
        return abs(value) <= MAX_SAFE_JSON_INTEGER
    if type(value) is float:
        return math.isfinite(value)
    return False


def _utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _safe_trace_id(value: Any) -> str:
    return value if type(value) is str and _utf8_encodable(value) else ""


def _safe_page_index(value: Any) -> int:
    if (
        type(value) is int
        and 0 <= value <= MAX_SAFE_JSON_INTEGER
    ):
        return value
    return -1
