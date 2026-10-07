"""Same-trace L5 review routing for the strict pure-vector shadow chain."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from vector_phrase_chain_validation import (
    request_local_artifact_safety_reasons,
)
from vector_phrase_l4_shadow import (
    _l4_artifact_safety_reasons,
    _l4_input_structure_reasons,
    _safe_page_index,
    _safe_trace_id,
    _strict_json_tree,
)


DUMP_SCHEMA_VERSION = "vector_phrase_l5_shadow_dump_v1"
REVIEW_SCHEMA_VERSION = "vector_phrase_l5_review_required_v1"
CONSUMER_ALLOWED = False
RELEASE_BLOCK_REASONS = (
    "consumer_approval_missing",
    "dimension_geometry_owner_missing",
    "instance_quality_gate_not_passed",
    "reader_provenance_unverified",
    "real_l4_recovery_producer_missing",
)

_DUMP_KEYS = frozenset({
    "schema_version",
    "trace_id",
    "page_index",
    "page_context_manifest",
    "status",
    "error",
    "trace_relation",
    "source_l3_semantic_digest",
    "source_l4_semantic_digest",
    "reader_provenance_status",
    "geometry_ownership_status",
    "recognition_quality_claim_allowed",
    "release_gate",
    "L5_high_conf",
    "L5_release_ready",
    "L5_review_required",
    "L5_gdt_shadow",
    "L5_key_index",
    "stats",
    "semantic_digest",
    "release_allowed",
    "feeds_display_l5",
    "ocr_called",
    "consumer_allowed",
})
_RELEASE_GATE_KEYS = frozenset({
    "schema_version",
    "passed",
    "reasons",
    "consumer_allowed",
})
_DIMENSION_REVIEW_KEYS = frozenset({
    "schema_version",
    "trace_id",
    "page_index",
    "review_id",
    "phrase_id",
    "physical_core_digest",
    "source_collection",
    "source_l3_semantic_digest",
    "source_text",
    "canonical_text",
    "dimension_semantics",
    "bbox",
    "oriented_quad",
    "axis_angle_deg",
    "axis_angle_source",
    "key_candidate",
    "quality_gate_reference",
    "geometry_gate",
    "gdt_route_evidence",
    "source_provenance",
    "dedup_evidence",
    "release_block_reasons",
    "release_allowed",
    "feeds_display_l5",
    "ocr_called",
    "consumer_allowed",
})
_UNRESOLVED_REVIEW_KEYS = frozenset({
    "schema_version",
    "trace_id",
    "page_index",
    "review_id",
    "phrase_id",
    "physical_core_digest",
    "source_collection",
    "source_l3_semantic_digest",
    "source_text",
    "canonical_text",
    "dimension_semantics",
    "bbox",
    "key_candidate",
    "source_primary_reason",
    "source_first_drop_stage",
    "recovery_status",
    "quality_gate_reference",
    "geometry_gate",
    "source_provenance",
    "dedup_evidence",
    "ambiguity_evidence",
    "release_block_reasons",
    "release_allowed",
    "feeds_display_l5",
    "ocr_called",
    "consumer_allowed",
})
_RECOVERED_REVIEW_KEYS = frozenset({
    "schema_version",
    "trace_id",
    "page_index",
    "review_id",
    "phrase_id",
    "physical_core_digest",
    "source_collection",
    "source_l3_semantic_digest",
    "source_text",
    "canonical_text",
    "dimension_semantics",
    "bbox",
    "key_candidate",
    "release_block_reasons",
    "release_allowed",
    "feeds_display_l5",
    "ocr_called",
    "consumer_allowed",
})
_GDT_ROW_KEYS = frozenset({
    "schema_version",
    "trace_id",
    "page_index",
    "phrase_id",
    "physical_core_digest",
    "source_text",
    "canonical_text",
    "bbox",
    "oriented_quad",
    "axis_angle_deg",
    "axis_angle_source",
    "cluster_phrase_ids",
    "routing_frame_ids",
    "key_candidate",
    "quality_gate_reference",
    "geometry_gate",
    "source_provenance",
    "dedup_evidence",
    "strict_gdt_l3_eligible",
    "primary_reason",
    "feeds_display_l3",
    "ocr_called",
    "consumer_allowed",
})
_KEY_ROW_KEYS = frozenset({
    "schema_version",
    "trace_id",
    "page_index",
    "key_id",
    "phrase_id",
    "physical_core_digest",
    "cluster_phrase_ids",
    "target_collection",
    "primary_reason",
    "feeds_display_l3",
    "ocr_called",
    "consumer_allowed",
})


def validate_vector_phrase_l5_shadow_dump(
    value: Any,
    *,
    phrase_dump: dict[str, Any] | None = None,
    pitch_read_dump: dict[str, Any] | None = None,
    quality_gate_dump: dict[str, Any] | None = None,
    hypothesis_dump: dict[str, Any] | None = None,
    spatial_dedup_dump: dict[str, Any] | None = None,
    l3_shadow_dump: dict[str, Any] | None = None,
    l4_shadow_dump: dict[str, Any] | None = None,
) -> list[str]:
    """Validate terminal structure without replaying recognition stages."""
    reasons = _validate_l5_structure(value)
    source_chain = (
        phrase_dump,
        pitch_read_dump,
        quality_gate_dump,
        hypothesis_dump,
        spatial_dedup_dump,
        l3_shadow_dump,
        l4_shadow_dump,
    )
    provided = tuple(source is not None for source in source_chain)
    if any(provided) and not all(provided):
        reasons.append("l5_authoritative_source_chain_incomplete")
    elif all(provided):
        if not (
            isinstance(value, dict)
            and isinstance(l3_shadow_dump, dict)
            and isinstance(l4_shadow_dump, dict)
            and value.get("trace_id") == l3_shadow_dump.get("trace_id")
            == l4_shadow_dump.get("trace_id")
            and value.get("page_index") == l3_shadow_dump.get("page_index")
            == l4_shadow_dump.get("page_index")
        ):
            reasons.append("l5_source_trace_invalid")
    return sorted(set(reasons))


def _validate_l5_structure(value: Any) -> list[str]:
    """Validate a self-contained terminal shadow envelope."""
    if not _strict_json_tree(value, ancestor_ids=set(), depth=0):
        return ["l5_artifact_not_strict_json"]
    if not isinstance(value, dict):
        return ["l5_artifact_not_object"]
    reasons: list[str] = []
    if set(value) != _DUMP_KEYS:
        reasons.append("l5_envelope_schema_invalid")
    if not (
        value.get("schema_version") == DUMP_SCHEMA_VERSION
        and value.get("status") == "ok"
        and value.get("error") is None
        and type(value.get("trace_id")) is str
        and bool(value.get("trace_id"))
        and type(value.get("page_index")) is int
        and value["page_index"] >= 0
        and isinstance(value.get("page_context_manifest"), dict)
        and value.get("trace_relation")
        == "same_trace_l3_l4_l5_shadow"
    ):
        reasons.append("l5_envelope_invalid")
    if not (
        value.get("reader_provenance_status")
        == "contract_only_unreleased_or_unverified_reader"
        and value.get("geometry_ownership_status")
        == "gdt_only_dimension_feature_owner_missing"
        and value.get("recognition_quality_claim_allowed") is False
        and value.get("release_allowed") is False
        and value.get("feeds_display_l5") is False
        and value.get("ocr_called") is False
        and value.get("consumer_allowed") is False
    ):
        reasons.append("l5_release_safety_invalid")
    release_gate = value.get("release_gate")
    if not (
        isinstance(release_gate, dict)
        and set(release_gate) == _RELEASE_GATE_KEYS
        and release_gate.get("schema_version")
        == "vector_phrase_l5_release_gate_v1"
        and release_gate.get("passed") is False
        and release_gate.get("reasons") == list(RELEASE_BLOCK_REASONS)
        and release_gate.get("consumer_allowed") is False
    ):
        reasons.append("l5_release_gate_invalid")
    high_conf = value.get("L5_high_conf")
    release_ready = value.get("L5_release_ready")
    review = value.get("L5_review_required")
    gdt = value.get("L5_gdt_shadow")
    key_index = value.get("L5_key_index")
    if not (
        high_conf == []
        and release_ready == []
        and isinstance(review, list)
        and isinstance(gdt, list)
        and isinstance(key_index, list)
    ):
        reasons.append("l5_collections_invalid")
        review = review if isinstance(review, list) else []
        gdt = gdt if isinstance(gdt, list) else []
        key_index = key_index if isinstance(key_index, list) else []
    trace_id = value.get("trace_id")
    page_index = value.get("page_index")
    source_collections: list[Any] = []
    review_phrase_ids: list[Any] = []
    review_physical_ids: list[Any] = []
    review_ids: list[Any] = []
    for row in review:
        if not isinstance(row, dict):
            reasons.append("l5_review_row_invalid")
            continue
        source_collection = row.get("source_collection")
        source_collections.append(source_collection)
        review_phrase_ids.append(row.get("phrase_id"))
        review_physical_ids.append(row.get("physical_core_digest"))
        review_ids.append(row.get("review_id"))
        row_valid = _valid_review_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
        )
        if not row_valid:
            reasons.append("l5_review_row_invalid")
            continue
    if len(source_collections) != len(review) or any(
        collection not in (
            "L3_dimensions",
            "L4_recovered",
            "L4_unresolved_suspects",
        )
        for collection in source_collections
    ):
        reasons.append("l5_review_route_invalid")
    if not _all_nonempty_strings(review_phrase_ids):
        reasons.append("l5_review_phrase_identity_invalid")
    if not _all_valid_sha256(review_physical_ids):
        reasons.append("l5_review_physical_identity_invalid")
    if not _all_nonempty_strings(review_ids):
        reasons.append("l5_review_id_invalid")

    gdt_phrase_ids: list[Any] = []
    gdt_physical_ids: list[Any] = []
    for row in gdt:
        if not isinstance(row, dict):
            reasons.append("l5_gdt_row_invalid")
            continue
        gdt_phrase_ids.append(row.get("phrase_id"))
        gdt_physical_ids.append(row.get("physical_core_digest"))
        row_valid = _valid_gdt_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
        )
        if not row_valid:
            reasons.append("l5_gdt_row_invalid")
            continue
    if not _all_nonempty_strings(gdt_phrase_ids):
        reasons.append("l5_gdt_phrase_identity_invalid")
    if not _all_valid_sha256(gdt_physical_ids):
        reasons.append("l5_gdt_physical_identity_invalid")
    key_ids: list[Any] = []
    key_phrase_ids: list[Any] = []
    key_physical_ids: list[Any] = []
    for row in key_index:
        if not isinstance(row, dict):
            reasons.append("l5_key_row_invalid")
            continue
        key_ids.append(row.get("key_id"))
        key_phrase_ids.append(row.get("phrase_id"))
        key_physical_ids.append(row.get("physical_core_digest"))
        row_valid = _valid_key_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
        )
        if not row_valid:
            reasons.append("l5_key_row_invalid")
            continue
    if not _all_nonempty_strings(key_ids):
        reasons.append("l5_key_id_invalid")
    if not _all_nonempty_strings(key_phrase_ids):
        reasons.append("l5_key_phrase_identity_invalid")
    if not _all_valid_sha256(key_physical_ids):
        reasons.append("l5_key_physical_identity_invalid")
    stats_keys = {
        "input_l3_dimension_count",
        "input_l4_recovered_count",
        "input_l4_unresolved_count",
        "input_l3_gdt_routed_count",
        "input_l3_key_index_count",
        "high_conf_count",
        "release_ready_count",
        "review_required_count",
        "gdt_shadow_count",
        "key_index_count",
        "release_allowed_count",
        "feeds_display_l5_count",
        "consumer_allowed_count",
    }
    stats = value.get("stats")
    if not (
        isinstance(stats, dict)
        and set(stats) == stats_keys
        and all(
            type(stats.get(key)) is int
            and stats[key] >= 0
            for key in stats_keys
        )
    ):
        reasons.append("l5_stats_invalid")
    if _recursive_release_pollution(value):
        reasons.append("l5_recursive_release_safety_invalid")
    if not (
        type(value.get("source_l3_semantic_digest")) is str
        and bool(value["source_l3_semantic_digest"])
        and type(value.get("source_l4_semantic_digest")) is str
        and bool(value["source_l4_semantic_digest"])
        and _valid_sha256(value.get("semantic_digest"))
    ):
        reasons.append("l5_digest_contract_invalid")
    return sorted(set(reasons))


def build_vector_phrase_l5_shadow_dump(
    *,
    phrase_dump: dict[str, Any],
    pitch_read_dump: dict[str, Any],
    quality_gate_dump: dict[str, Any],
    hypothesis_dump: dict[str, Any],
    spatial_dedup_dump: dict[str, Any],
    l3_shadow_dump: dict[str, Any],
    l4_shadow_dump: dict[str, Any],
) -> dict[str, Any]:
    """Route every unreleased row to review; never infer confidence release."""
    trace_id = str(
        l4_shadow_dump.get("trace_id") or ""
        if isinstance(l4_shadow_dump, dict)
        else ""
    )
    page_index = (
        l4_shadow_dump.get("page_index")
        if isinstance(l4_shadow_dump, dict)
        else -1
    )
    reasons = _l5_source_structure_reasons(
        l3_shadow_dump=l3_shadow_dump,
        l4_shadow_dump=l4_shadow_dump,
    )
    if reasons:
        return _failed_dump(
            trace_id=trace_id,
            page_index=page_index,
            reasons=reasons,
        )
    source_l3 = l3_shadow_dump
    source_l4 = l4_shadow_dump

    release_gate = {
        "schema_version": "vector_phrase_l5_release_gate_v1",
        "passed": False,
        "reasons": list(RELEASE_BLOCK_REASONS),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    review_rows = [
        *(
            _dimension_review_row(
                row,
                source_l3_semantic_digest=str(
                    source_l3["semantic_digest"]
                ),
            )
            for row in source_l3["L3_dimensions"]
        ),
        *(
            _unresolved_review_row(row)
            for row in source_l4["L4_unresolved_suspects"]
        ),
        *(
            _recovered_review_row(row)
            for row in source_l4["L4_recovered"]
        ),
    ]
    review_rows = sorted(
        review_rows,
        key=lambda row: (
            str(row["phrase_id"]),
            str(row["source_collection"]),
        ),
    )

    gdt_shadow = copy.deepcopy(source_l3["L3_gdt_routed"])
    key_index = copy.deepcopy(source_l3["L3_key_index"])
    stats = {
        "input_l3_dimension_count": len(source_l3["L3_dimensions"]),
        "input_l4_recovered_count": len(source_l4["L4_recovered"]),
        "input_l4_unresolved_count": len(
            source_l4["L4_unresolved_suspects"]
        ),
        "input_l3_gdt_routed_count": len(source_l3["L3_gdt_routed"]),
        "input_l3_key_index_count": len(source_l3["L3_key_index"]),
        "high_conf_count": 0,
        "release_ready_count": 0,
        "review_required_count": len(review_rows),
        "gdt_shadow_count": len(gdt_shadow),
        "key_index_count": len(key_index),
        "release_allowed_count": 0,
        "feeds_display_l5_count": 0,
        "consumer_allowed_count": 0,
    }
    trace_relation = "same_trace_l3_l4_l5_shadow"
    geometry_ownership_status = (
        "gdt_only_dimension_feature_owner_missing"
    )
    semantic_payload = {
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
        "source_l4_semantic_digest": str(source_l4["semantic_digest"]),
        "reader_provenance_status": str(
            source_l3["reader_provenance_status"]
        ),
        "geometry_ownership_status": geometry_ownership_status,
        "recognition_quality_claim_allowed": False,
        "release_gate": release_gate,
        "L5_high_conf": [],
        "L5_release_ready": [],
        "L5_review_required": review_rows,
        "L5_gdt_shadow": gdt_shadow,
        "L5_key_index": key_index,
        "stats": stats,
        "release_allowed": False,
        "feeds_display_l5": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    result = {
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
        "source_l4_semantic_digest": str(source_l4["semantic_digest"]),
        "reader_provenance_status": str(
            source_l3["reader_provenance_status"]
        ),
        "geometry_ownership_status": geometry_ownership_status,
        "recognition_quality_claim_allowed": False,
        "release_gate": release_gate,
        "L5_high_conf": [],
        "L5_release_ready": [],
        "L5_review_required": review_rows,
        "L5_gdt_shadow": gdt_shadow,
        "L5_key_index": key_index,
        "stats": stats,
        "semantic_digest": _digest(semantic_payload),
        "release_allowed": False,
        "feeds_display_l5": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    validation_reasons = _validate_l5_structure(result)
    if validation_reasons:
        return _failed_dump(
            trace_id=result["trace_id"],
            page_index=result["page_index"],
            reasons=[
                f"l5_output_validation:{reason}"
                for reason in validation_reasons
            ],
        )
    return result


def _dimension_review_row(
    source: dict[str, Any],
    *,
    source_l3_semantic_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "trace_id": str(source["trace_id"]),
        "page_index": int(source["page_index"]),
        "review_id": f"review_{str(source['physical_core_digest'])[:16]}",
        "phrase_id": str(source["phrase_id"]),
        "physical_core_digest": str(source["physical_core_digest"]),
        "source_collection": "L3_dimensions",
        "source_l3_semantic_digest": source_l3_semantic_digest,
        "source_text": str(source["source_text"]),
        "canonical_text": str(source["canonical_text"]),
        "dimension_semantics": copy.deepcopy(source["dimension_semantics"]),
        "bbox": copy.deepcopy(source["bbox"]),
        "oriented_quad": copy.deepcopy(source["oriented_quad"]),
        "axis_angle_deg": float(source["axis_angle_deg"]),
        "axis_angle_source": str(source["axis_angle_source"]),
        "key_candidate": source.get("key_candidate") is True,
        "quality_gate_reference": copy.deepcopy(
            source["quality_gate_reference"]
        ),
        "geometry_gate": copy.deepcopy(source["geometry_gate"]),
        "gdt_route_evidence": copy.deepcopy(source["gdt_route_evidence"]),
        "source_provenance": copy.deepcopy(source["source_provenance"]),
        "dedup_evidence": copy.deepcopy(source["dedup_evidence"]),
        "release_block_reasons": list(RELEASE_BLOCK_REASONS),
        "release_allowed": False,
        "feeds_display_l5": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _unresolved_review_row(source: dict[str, Any]) -> dict[str, Any]:
    reasons = sorted({
        *RELEASE_BLOCK_REASONS,
        f"l4_unresolved:{str(source['recovery_status'])}",
    })
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "trace_id": str(source["trace_id"]),
        "page_index": int(source["page_index"]),
        "review_id": f"review_{str(source['physical_core_digest'])[:16]}",
        "phrase_id": str(source["phrase_id"]),
        "physical_core_digest": str(source["physical_core_digest"]),
        "source_collection": "L4_unresolved_suspects",
        "source_l3_semantic_digest": str(
            source["source_l3_semantic_digest"]
        ),
        "source_text": str(source["source_text"]),
        "canonical_text": str(source["canonical_text"]),
        "dimension_semantics": None,
        "bbox": copy.deepcopy(source["bbox"]),
        "key_candidate": source.get("key_candidate") is True,
        "source_primary_reason": str(source["source_primary_reason"]),
        "source_first_drop_stage": source["source_first_drop_stage"],
        "recovery_status": str(source["recovery_status"]),
        "quality_gate_reference": copy.deepcopy(
            source["quality_gate_reference"]
        ),
        "geometry_gate": copy.deepcopy(source["geometry_gate"]),
        "source_provenance": copy.deepcopy(source["source_provenance"]),
        "dedup_evidence": copy.deepcopy(source["dedup_evidence"]),
        "ambiguity_evidence": copy.deepcopy(
            source["ambiguity_evidence"]
        ),
        "release_block_reasons": reasons,
        "release_allowed": False,
        "feeds_display_l5": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _recovered_review_row(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "trace_id": str(source["trace_id"]),
        "page_index": int(source["page_index"]),
        "review_id": f"review_{str(source['physical_core_digest'])[:16]}",
        "phrase_id": str(source["phrase_id"]),
        "physical_core_digest": str(source["physical_core_digest"]),
        "source_collection": "L4_recovered",
        "source_l3_semantic_digest": str(
            source["source_l3_semantic_digest"]
        ),
        "source_text": str(source.get("source_text") or ""),
        "canonical_text": str(source.get("canonical_text") or ""),
        "dimension_semantics": copy.deepcopy(
            source.get("dimension_semantics")
        ),
        "bbox": copy.deepcopy(source.get("bbox")),
        "key_candidate": source.get("key_candidate") is True,
        "release_block_reasons": list(RELEASE_BLOCK_REASONS),
        "release_allowed": False,
        "feeds_display_l5": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _l5_source_structure_reasons(
    *,
    l3_shadow_dump: Any,
    l4_shadow_dump: Any,
) -> list[str]:
    """Admit carried rows by shape and safety, never by replayed accounting."""
    reasons = _l4_input_structure_reasons(l3_shadow_dump)
    if _l4_artifact_safety_reasons(l3_shadow_dump):
        reasons.append("l3_artifact_safety_invalid")
    if not isinstance(l4_shadow_dump, dict):
        reasons.append("l4_structure_invalid")
        return sorted(set(reasons))
    trace_id = l3_shadow_dump.get("trace_id")
    page_index = l3_shadow_dump.get("page_index")
    if not (
        l4_shadow_dump.get("schema_version")
        == "vector_phrase_l4_shadow_dump_v1"
        and l4_shadow_dump.get("status") == "ok"
        and l4_shadow_dump.get("error") is None
        and l4_shadow_dump.get("trace_id") == trace_id
        and l4_shadow_dump.get("page_index") == page_index
        and isinstance(l4_shadow_dump.get("page_context_manifest"), dict)
        and isinstance(l4_shadow_dump.get("L4_recovered"), list)
        and isinstance(
            l4_shadow_dump.get("L4_unresolved_suspects"),
            list,
        )
        and type(l4_shadow_dump.get("semantic_digest")) is str
        and bool(l4_shadow_dump["semantic_digest"])
        and l4_shadow_dump.get("recognition_quality_claim_allowed") is False
        and l4_shadow_dump.get("feeds_display_l4") is False
        and l4_shadow_dump.get("ocr_called") is False
        and l4_shadow_dump.get("consumer_allowed") is False
    ):
        reasons.append("l4_structure_invalid")
    if request_local_artifact_safety_reasons(
        l4_shadow_dump,
        path="l4",
    ):
        reasons.append("l4_artifact_safety_invalid")
    if isinstance(l3_shadow_dump, dict) and any(
        not _l3_dimension_source_shape(row, trace_id, page_index)
        for row in l3_shadow_dump.get("L3_dimensions") or []
    ):
        reasons.append("l3_dimension_structure_invalid")
    if any(
        not _l4_unresolved_source_shape(row, trace_id, page_index)
        for row in l4_shadow_dump.get("L4_unresolved_suspects") or []
    ):
        reasons.append("l4_unresolved_structure_invalid")
    for collection in ("L4_recovered", "L4_unresolved_suspects"):
        if any(
            not isinstance(row, dict)
            or row.get("trace_id") != trace_id
            or row.get("page_index") != page_index
            for row in l4_shadow_dump.get(collection) or []
        ):
            reasons.append("l4_row_trace_invalid")
    return sorted(set(reasons))


def _l3_dimension_source_shape(
    row: Any,
    trace_id: Any,
    page_index: Any,
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
        and isinstance(row.get("quality_gate_reference"), dict)
        and isinstance(row.get("geometry_gate"), dict)
        and isinstance(row.get("gdt_route_evidence"), dict)
        and isinstance(row.get("source_provenance"), dict)
        and isinstance(row.get("dedup_evidence"), dict)
        and row.get("feeds_display_l3") is False
        and row.get("ocr_called") is False
        and row.get("consumer_allowed") is False
    )


def _l4_unresolved_source_shape(
    row: Any,
    trace_id: Any,
    page_index: Any,
) -> bool:
    return bool(
        isinstance(row, dict)
        and row.get("trace_id") == trace_id
        and row.get("page_index") == page_index
        and type(row.get("phrase_id")) is str
        and bool(row["phrase_id"])
        and type(row.get("physical_core_digest")) is str
        and bool(row["physical_core_digest"])
        and type(row.get("source_l3_semantic_digest")) is str
        and type(row.get("source_text")) is str
        and type(row.get("canonical_text")) is str
        and isinstance(row.get("bbox"), dict)
        and type(row.get("key_candidate")) is bool
        and type(row.get("source_primary_reason")) is str
        and type(row.get("recovery_status")) is str
        and isinstance(row.get("quality_gate_reference"), dict)
        and isinstance(row.get("geometry_gate"), dict)
        and isinstance(row.get("source_provenance"), dict)
        and isinstance(row.get("dedup_evidence"), dict)
        and row.get("feeds_display_l4") is False
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
            "schema_version": "vector_phrase_l5_shadow_error_v1",
            "code": "invalid_l5_shadow_input",
            "reasons": sorted(set(reasons)),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "recognition_quality_claim_allowed": False,
        "release_gate": {
            "schema_version": "vector_phrase_l5_release_gate_v1",
            "passed": False,
            "reasons": sorted(set(reasons)),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "L5_high_conf": [],
        "L5_release_ready": [],
        "L5_review_required": [],
        "L5_gdt_shadow": [],
        "L5_key_index": [],
        "stats": {
            "input_l3_dimension_count": 0,
            "input_l4_recovered_count": 0,
            "input_l4_unresolved_count": 0,
            "input_l3_gdt_routed_count": 0,
            "input_l3_key_index_count": 0,
            "high_conf_count": 0,
            "release_ready_count": 0,
            "review_required_count": 0,
            "gdt_shadow_count": 0,
            "key_index_count": 0,
            "release_allowed_count": 0,
            "feeds_display_l5_count": 0,
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest([]),
        "release_allowed": False,
        "feeds_display_l5": False,
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


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _all_nonempty_strings(values: list[Any]) -> bool:
    return all(type(value) is str and bool(value) for value in values)


def _all_valid_sha256(values: list[Any]) -> bool:
    return all(_valid_sha256(value) for value in values)


def _false_terminal_flags(row: dict[str, Any], *, display_key: str) -> bool:
    return bool(
        row.get("release_allowed") is False
        and row.get(display_key) is False
        and row.get("ocr_called") is False
        and row.get("consumer_allowed") is False
    )


def _valid_review_row(
    row: dict[str, Any],
    *,
    trace_id: Any,
    page_index: Any,
) -> bool:
    source_collection = row.get("source_collection")
    if type(source_collection) is not str:
        return False
    if source_collection == "L4_recovered":
        return bool(
            set(row) == _RECOVERED_REVIEW_KEYS
            and row.get("schema_version") == REVIEW_SCHEMA_VERSION
            and row.get("trace_id") == trace_id
            and row.get("page_index") == page_index
            and type(row.get("phrase_id")) is str
            and bool(row["phrase_id"])
            and _valid_sha256(row.get("physical_core_digest"))
            and type(row.get("review_id")) is str
            and bool(row["review_id"])
            and type(row.get("source_l3_semantic_digest")) is str
            and bool(row["source_l3_semantic_digest"])
            and type(row.get("source_text")) is str
            and type(row.get("canonical_text")) is str
            and type(row.get("dimension_semantics")) is dict
            and type(row.get("bbox")) is dict
            and type(row.get("key_candidate")) is bool
            and row.get("release_block_reasons")
            == list(RELEASE_BLOCK_REASONS)
            and _false_terminal_flags(
                row,
                display_key="feeds_display_l5",
            )
        )
    expected_keys = {
        "L3_dimensions": _DIMENSION_REVIEW_KEYS,
        "L4_unresolved_suspects": _UNRESOLVED_REVIEW_KEYS,
    }.get(source_collection)
    physical_digest = row.get("physical_core_digest")
    if not (
        expected_keys is not None
        and set(row) == expected_keys
        and row.get("schema_version") == REVIEW_SCHEMA_VERSION
        and type(row.get("trace_id")) is str
        and row.get("trace_id") == trace_id
        and type(row.get("page_index")) is int
        and row.get("page_index") == page_index
        and type(row.get("phrase_id")) is str
        and bool(row.get("phrase_id"))
        and _valid_sha256(physical_digest)
        and type(row.get("review_id")) is str
        and bool(row["review_id"])
        and type(row.get("source_l3_semantic_digest")) is str
        and bool(row["source_l3_semantic_digest"])
        and type(row.get("source_text")) is str
        and type(row.get("canonical_text")) is str
        and type(row.get("bbox")) is dict
        and type(row.get("key_candidate")) is bool
        and type(row.get("quality_gate_reference")) is dict
        and type(row.get("geometry_gate")) is dict
        and type(row.get("source_provenance")) is dict
        and type(row.get("dedup_evidence")) is dict
        and _false_terminal_flags(row, display_key="feeds_display_l5")
    ):
        return False
    if source_collection == "L3_dimensions":
        return bool(
            type(row.get("dimension_semantics")) is dict
            and type(row.get("oriented_quad")) is list
            and type(row.get("axis_angle_deg")) is float
            and type(row.get("axis_angle_source")) is str
            and bool(row.get("axis_angle_source"))
            and type(row.get("gdt_route_evidence")) is dict
            and row.get("release_block_reasons")
            == list(RELEASE_BLOCK_REASONS)
        )
    recovery_status = row.get("recovery_status")
    expected_reasons = sorted({
        *RELEASE_BLOCK_REASONS,
        f"l4_unresolved:{str(recovery_status)}",
    })
    return bool(
        row.get("dimension_semantics") is None
        and type(row.get("source_primary_reason")) is str
        and bool(row.get("source_primary_reason"))
        and (
            row.get("source_first_drop_stage") is None
            or type(row.get("source_first_drop_stage")) is str
        )
        and type(recovery_status) is str
        and bool(recovery_status)
        and (
            row.get("ambiguity_evidence") is None
            or type(row.get("ambiguity_evidence")) in {dict, list}
        )
        and row.get("release_block_reasons") == expected_reasons
    )


def _valid_gdt_row(
    row: dict[str, Any],
    *,
    trace_id: Any,
    page_index: Any,
) -> bool:
    cluster_phrase_ids = row.get("cluster_phrase_ids")
    routing_frame_ids = row.get("routing_frame_ids")
    return bool(
        set(row) == _GDT_ROW_KEYS
        and row.get("schema_version")
        == "vector_phrase_l3_gdt_shadow_v1"
        and type(row.get("trace_id")) is str
        and row.get("trace_id") == trace_id
        and type(row.get("page_index")) is int
        and row.get("page_index") == page_index
        and type(row.get("phrase_id")) is str
        and bool(row.get("phrase_id"))
        and _valid_sha256(row.get("physical_core_digest"))
        and type(row.get("source_text")) is str
        and type(row.get("canonical_text")) is str
        and type(row.get("bbox")) is dict
        and type(row.get("oriented_quad")) is list
        and type(row.get("axis_angle_deg")) is float
        and type(row.get("axis_angle_source")) is str
        and bool(row.get("axis_angle_source"))
        and type(cluster_phrase_ids) is list
        and bool(cluster_phrase_ids)
        and all(
            type(value) is str and bool(value)
            for value in cluster_phrase_ids
        )
        and row.get("phrase_id") in cluster_phrase_ids
        and type(routing_frame_ids) is list
        and bool(routing_frame_ids)
        and all(
            type(value) is str and bool(value)
            for value in routing_frame_ids
        )
        and type(row.get("key_candidate")) is bool
        and type(row.get("quality_gate_reference")) is dict
        and type(row.get("geometry_gate")) is dict
        and type(row.get("source_provenance")) is dict
        and type(row.get("dedup_evidence")) is dict
        and row.get("strict_gdt_l3_eligible") is False
        and row.get("primary_reason")
        == "gdt_dedup_disabled_missing_compartment_ownership"
        and row.get("feeds_display_l3") is False
        and row.get("ocr_called") is False
        and row.get("consumer_allowed") is False
    )


def _valid_key_row(
    row: dict[str, Any],
    *,
    trace_id: Any,
    page_index: Any,
) -> bool:
    physical_digest = row.get("physical_core_digest")
    cluster_phrase_ids = row.get("cluster_phrase_ids")
    return bool(
        set(row) == _KEY_ROW_KEYS
        and row.get("schema_version") == "vector_phrase_l3_key_index_v1"
        and type(row.get("trace_id")) is str
        and row.get("trace_id") == trace_id
        and type(row.get("page_index")) is int
        and row.get("page_index") == page_index
        and type(row.get("phrase_id")) is str
        and bool(row.get("phrase_id"))
        and _valid_sha256(physical_digest)
        and type(row.get("key_id")) is str
        and bool(row["key_id"])
        and type(cluster_phrase_ids) is list
        and bool(cluster_phrase_ids)
        and all(
            type(value) is str and bool(value)
            for value in cluster_phrase_ids
        )
        and row.get("phrase_id") in cluster_phrase_ids
        and row.get("target_collection")
        in {"L3_dimensions", "L3_suspects", "L3_gdt_routed"}
        and (
            row.get("primary_reason") is None
            or type(row.get("primary_reason")) is str
        )
        and row.get("feeds_display_l3") is False
        and row.get("ocr_called") is False
        and row.get("consumer_allowed") is False
    )


def _recursive_release_pollution(
    value: Any,
    *,
    release_context: bool = False,
) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            compact_key = re.sub(r"[^a-z0-9]+", "", str(key).lower())
            child_release_context = bool(
                release_context
                or compact_key in {"releasegate", "l5releaseready"}
            )
            if compact_key in {
                "l5highconf",
                "l5releaseready",
            } and child != []:
                return True
            if "ocr" in compact_key and not _zero_or_empty_ocr_value(child):
                return True
            if compact_key in {
                "consumerallowed",
                "consumerapproved",
                "releaseallowed",
                "releaseready",
                "highconfidence",
                "highconf",
                "finalapproved",
                "feedsdisplayl3",
                "feedsdisplayl4",
                "feedsdisplayl5",
                "ocrcalled",
                "recognitionqualityclaimallowed",
            } and child is not False:
                return True
            if compact_key in {
                "consumerallowedcount",
                "consumerapprovedcount",
                "releaseallowedcount",
                "releasereadycount",
                "l5releasereadycount",
                "highconfidencecount",
                "highconfcount",
                "l5highconfcount",
                "finalapprovedcount",
                "feedsdisplayl3count",
                "feedsdisplayl4count",
                "feedsdisplayl5count",
                "ocrinitcount",
                "ocrcallcount",
                "ocrretrycount",
                "ocrartifactcount",
                "recognitionqualityclaimallowedcount",
            } and not (type(child) is int and child == 0):
                return True
            if (
                release_context
                and compact_key == "passed"
                and child is not False
            ):
                return True
            if _recursive_release_pollution(
                child,
                release_context=child_release_context,
            ):
                return True
        return False
    if isinstance(value, list):
        return any(
            _recursive_release_pollution(
                child,
                release_context=release_context,
            )
            for child in value
        )
    return False


def _zero_or_empty_ocr_value(value: Any) -> bool:
    return bool(
        value is False
        or value is None
        or (type(value) is int and value == 0)
        or value == ""
        or value == []
        or value == {}
    )
