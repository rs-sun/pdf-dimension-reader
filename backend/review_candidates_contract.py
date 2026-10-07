"""The one public R33-M1 strict-vector review candidate contract.

The producer passes the dict returned by ``build_r33_m1_phrase_chain``
directly.  This module projects its authoritative linear L5 rows and merges
already fail-closed strict-vector GD&T items; it does not define an
intermediate candidate envelope, a legacy alias, or a fallback adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

from dimension_syntax import (
    DIMENSION_SYNTAX_LABELS,
    SYNTAX_INVALID_DIMENSION,
    SYNTAX_OK,
    classify_dimension_syntax,
)
from vector_phrase_l3_shadow import _dimension_semantics


ROOT_KEY = "review_candidates_v1"
SCHEMA_VERSION = ROOT_KEY
AUDIT_SCHEMA_VERSION = "review_candidates_audit_v1"
COORDINATE_SPACE = "pdf_page_points_top_left"
SOURCE_STAGE = "L5_review_required"

ENVELOPE_KEYS = frozenset({
    "schema_version", "trace_id", "page_index", "coordinate_space",
    "items", "audit",
})
LINEAR_ITEM_KEYS = frozenset({
    "candidate_id", "bbox", "type", "nominal", "symmetric_tolerance",
    "upper_tolerance", "lower_tolerance", "prefix", "unit",
    "review_status", "source_stage", "trace",
})
OPTIONAL_ITEM_KEYS = frozenset({"syntax"})
LINEAR_OPTIONAL_ITEM_KEYS = OPTIONAL_ITEM_KEYS | frozenset({"quantity"})
GDT_ITEM_KEYS = frozenset({
    "candidate_id", "bbox", "type", "symbol_name", "symbol_unicode",
    "tolerance_value", "datum_1", "datum_2", "datum_3",
    "review_status", "source_stage", "trace",
})
BBOX_KEYS = frozenset({"x", "y", "w", "h"})
LINEAR_ITEM_TRACE_KEYS = frozenset({
    "l5_review_id", "phrase_id", "physical_core_digest",
    "source_l3_semantic_digest", "source_text", "canonical_text",
    "candidate_digest",
})
GDT_ITEM_TRACE_KEYS = frozenset({
    "frame_digest", "decimal_quad_id", "symbol_method",
    "symbol_confidence", "symbol_compartment_index",
    "tolerance_compartment_index", "reader_phrase_id",
    "template_content_sha256", "candidate_digest",
})
# Compatibility aliases for existing linear-only importers.
ITEM_KEYS = LINEAR_ITEM_KEYS
ITEM_TRACE_KEYS = LINEAR_ITEM_TRACE_KEYS
AUDIT_KEYS = frozenset({
    "schema_version", "status", "unavailable_reason", "chain", "counts",
    "template_identity", "runtime", "authority", "semantic_digest",
})
CHAIN_KEYS = frozenset({
    "schema_version", "trace_id", "page_index", "status",
    "semantic_projection_sha256", "stage_status", "stage_digests",
    "chain_counts",
})
STAGE_STATUS_KEYS = frozenset({
    "phrase_region", "pitch_reader", "quality_gate", "hypothesis",
    "spatial_dedup", "l3", "l4", "l5",
})
STAGE_DIGEST_KEYS = frozenset({
    "pitch_reader", "quality_gate", "hypothesis", "spatial_dedup",
    "l3", "l4", "l5",
})
CHAIN_COUNT_KEYS = frozenset({
    "page_context_build_count", "page_context_extraction_call_count",
    "phrase_region_count", "reader_call_count", "reader_success_count",
    "quality_pass_count", "ordinary_hypothesis_count",
    "dedup_winner_count", "l3_dimension_count", "l4_recovered_count",
    "l5_review_required_count",
})
COUNTS_KEYS = frozenset({
    "source_review_count", "candidate_count", "dropped_count",
    "drop_reasons",
})
DROP_REASON_KEYS = frozenset({"reason", "count"})
TEMPLATE_IDENTITY_KEYS = frozenset({
    "schema_version", "template_version", "content_sha256",
})
RUNTIME_KEYS = frozenset({
    "ocr_family_call_count", "paddleocr_call_count",
    "rapidocr_call_count", "yolo_char_call_count", "yolo_b_call_count",
})
AUTHORITY_KEYS = frozenset({
    "review_only", "formal_project_data_allowed", "consumer_allowed",
    "release_allowed",
})

CHAIN_AUDIT_FIELD = "r33_m1_phrase_chain_audit_v1"
CHAIN_DUMP_FIELDS = {
    "pitch_reader": "vector_pitch_phrase_read_dump_v1",
    "quality_gate": "vector_phrase_quality_gate_dump_v1",
    "hypothesis": "vector_phrase_hypothesis_dump_v1",
    "spatial_dedup": "vector_phrase_spatial_dedup_dump_v1",
    "l3": "vector_phrase_l3_shadow_dump_v1",
    "l4": "vector_phrase_l4_shadow_dump_v1",
    "l5": "vector_phrase_l5_shadow_dump_v1",
}
CHAIN_FIELD_KEYS = frozenset({
    *CHAIN_DUMP_FIELDS.values(),
    CHAIN_AUDIT_FIELD,
})
_CHAIN_AUDIT_KEYS = frozenset({
    "schema_version", "trace_id", "page_index", "status", "reason",
    "details", "template_identity", "stage_status", "counts",
    "semantic_projection_sha256", "ocr_family_call_count",
    "paddleocr_call_count", "rapidocr_call_count", "yolo_char_call_count",
    "yolo_b_call_count",
    "dimensions_promoted_count", "release_allowed", "consumer_allowed",
})
_DROP_REASON_ORDER = (
    "source_not_l3_dimension", "source_trace_invalid", "semantics_invalid",
    "kind_not_linear", "quantity_not_one", "prefix_not_null",
    "unit_not_null", "qualifier_not_null",
    "tolerance_kind_not_supported", "decimal_text_invalid", "bbox_invalid",
    "text_missing", "strict_boundary_or_geometry_rejected",
    "final_position_duplicate",
)
_DECIMAL_TEXT = re.compile(r"^[+\-]?\d{1,3}(?:\.\d{1,3})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
GDT_CORE_SYMBOLS = {
    "position": "⊕",
    "perpendicularity": "⊥",
    "parallelism": "∥",
    "coaxiality": "◎",
    "roundness": "○",
    "cylindricity": "⌭",
    "straightness": "—",
    "angularity": "∠",
    "surface_profile": "⌓",
    "line_profile": "⌒",
    "flatness": "▱",
    "symmetry": "≡",
}
FINAL_DEDUP_IOU_THRESHOLD = 0.30
FINAL_DEDUP_CONTAINMENT_THRESHOLD = 0.80
_SYNTAX_RANK = {
    "ok": 0,
    "incomplete_separator": 1,
    "unread_glyph_present": 2,
    "invalid_dimension_syntax": 3,
}


def build_review_candidates_v1(
    *,
    chain_fields: dict[str, Any] | None,
    expected_trace_id: str,
    expected_page_index: int,
    gdt_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project linear L5 rows and merge validated strict-vector GD&T items."""
    _require_identity(expected_trace_id, expected_page_index)
    if gdt_items is not None and not isinstance(gdt_items, list):
        raise ValueError("gdt_items must be a list or None")
    normalized_gdt_items: list[dict[str, Any]] = []
    for source_item in gdt_items or []:
        item = copy.deepcopy(source_item)
        if not isinstance(item, dict) or item.get("type") != "gdt":
            raise ValueError("gdt_items may contain only type=gdt items")
        item_reasons = _validate_item(item, expected_page_index)
        if item_reasons:
            raise ValueError(
                "invalid strict GD&T review candidate: "
                + ",".join(sorted(set(item_reasons)))
            )
        normalized_gdt_items.append(item)
    normalized = _normalize_chain_fields(
        chain_fields,
        trace_id=expected_trace_id,
        page_index=expected_page_index,
    )
    if normalized is None:
        return _unavailable(
            trace_id=expected_trace_id,
            page_index=expected_page_index,
            reason=_unavailable_reason(chain_fields),
        )
    audit_source, dumps, stage_digests = normalized
    rows = dumps["l5"].get("L5_review_required")
    if not isinstance(rows, list):
        return _unavailable(
            trace_id=expected_trace_id,
            page_index=expected_page_index,
            reason="l5_review_collection_invalid",
        )

    items: list[dict[str, Any]] = []
    dropped = {reason: 0 for reason in _DROP_REASON_ORDER}
    read_by_phrase_id = {
        str(row.get("phrase_id") or ""): row
        for row in dumps["pitch_reader"].get("reads") or []
        if isinstance(row, dict)
    }
    for row in rows:
        item, reason = _project_row(
            row,
            trace_id=expected_trace_id,
            page_index=expected_page_index,
            pitch_read=read_by_phrase_id.get(
                str(row.get("phrase_id") or "")
                if isinstance(row, dict)
                else ""
            ),
        )
        if item is None:
            resolved_reason = reason or "semantics_invalid"
            dropped.setdefault(resolved_reason, 0)
            dropped[resolved_reason] += 1
        else:
            items.append(item)
    items.extend(normalized_gdt_items)
    items, final_duplicate_count = _final_position_dedup(items)
    dropped["final_position_duplicate"] += final_duplicate_count
    items.sort(key=lambda item: item["candidate_id"])
    ids = [item["candidate_id"] for item in items]
    if len(ids) != len(set(ids)):
        return _unavailable(
            trace_id=expected_trace_id,
            page_index=expected_page_index,
            reason="duplicate_candidate_id",
        )
    counts = {
        "source_review_count": len(rows) + len(normalized_gdt_items),
        "candidate_count": len(items),
        "dropped_count": sum(dropped.values()),
        "drop_reasons": [
            {"reason": reason, "count": dropped[reason]}
            for reason in _DROP_REASON_ORDER if dropped[reason]
        ],
    }
    result = _envelope(
        trace_id=expected_trace_id,
        page_index=expected_page_index,
        items=items,
        status="ok",
        unavailable_reason=None,
        chain=_chain_projection(audit_source, stage_digests),
        counts=counts,
        template_identity=copy.deepcopy(audit_source["template_identity"]),
        runtime={key: audit_source[key] for key in RUNTIME_KEYS},
    )
    reasons = validate_review_candidates_v1(result)
    if reasons:
        raise ValueError(f"review_candidates_v1 builder bug: {reasons}")
    return result


def build_gdt_review_candidate_item(
    *,
    page_index: int,
    bbox: Any,
    symbol_name: str,
    symbol_unicode: str,
    tolerance_value: str,
    frame_digest: str,
    decimal_quad_id: str,
    symbol_method: str,
    symbol_confidence: float,
    symbol_compartment_index: int,
    tolerance_compartment_index: int,
    reader_phrase_id: str,
    template_content_sha256: str,
    syntax: str | None = None,
) -> dict[str, Any]:
    """Build one complete, datum-blank strict-vector GD&T review item."""
    if not _valid_page(page_index):
        raise ValueError("page_index must be non-negative")
    if not _valid_bbox(bbox):
        raise ValueError("bbox must be a finite positive rectangle")
    if not _valid_gdt_symbol_pair(symbol_name, symbol_unicode):
        raise ValueError("symbol_name and symbol_unicode must be a core GD&T pair")
    if syntax is not None and syntax not in DIMENSION_SYNTAX_LABELS:
        raise ValueError("syntax must be a stable dimension syntax label")
    if not _valid_gdt_tolerance_or_correction(tolerance_value, syntax):
        raise ValueError(
            "tolerance_value must be an unsigned decimal or a labelled "
            "review-only correction string"
        )

    trace_without_digest = {
        "frame_digest": frame_digest,
        "decimal_quad_id": decimal_quad_id,
        "symbol_method": symbol_method,
        "symbol_confidence": (
            float(symbol_confidence)
            if _finite_number(symbol_confidence)
            else symbol_confidence
        ),
        "symbol_compartment_index": symbol_compartment_index,
        "tolerance_compartment_index": tolerance_compartment_index,
        "reader_phrase_id": reader_phrase_id,
        "template_content_sha256": template_content_sha256,
    }
    trace_reasons = _validate_gdt_trace_fields(trace_without_digest)
    if trace_reasons:
        raise ValueError(
            "invalid strict GD&T trace: "
            + ",".join(sorted(set(trace_reasons)))
        )

    normalized_bbox = {
        key: float(bbox[key])
        for key in ("x", "y", "w", "h")
    }
    digest = _gdt_candidate_digest(
        page_index=page_index,
        bbox=normalized_bbox,
        symbol_name=symbol_name,
        symbol_unicode=symbol_unicode,
        tolerance_value=tolerance_value,
        trace=trace_without_digest,
        syntax=syntax,
    )
    item = {
        "candidate_id": f"rcv1_{digest[:24]}",
        "bbox": normalized_bbox,
        "type": "gdt",
        "symbol_name": symbol_name,
        "symbol_unicode": symbol_unicode,
        "tolerance_value": tolerance_value,
        "datum_1": None,
        "datum_2": None,
        "datum_3": None,
        "review_status": "pending",
        "source_stage": SOURCE_STAGE,
        "trace": {
            **trace_without_digest,
            "candidate_digest": digest,
        },
    }
    if syntax is not None:
        item["syntax"] = syntax
    reasons = _validate_item(item, page_index)
    if reasons:
        raise ValueError(
            "GD&T review candidate builder bug: "
            + ",".join(sorted(set(reasons)))
        )
    return item


def is_valid_unsigned_decimal_text(value: Any) -> bool:
    """Return whether ``value`` is the unsigned subset of the v1 decimal grammar."""
    return _valid_unsigned_decimal(value)


def validate_review_candidates_v1(value: Any) -> list[str]:
    """Validate the exact recursive schema; aliases and extra keys fail."""
    if not isinstance(value, dict):
        return ["envelope_not_object"]
    reasons: list[str] = []
    if set(value) != ENVELOPE_KEYS:
        reasons.append("envelope_keys_invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        reasons.append("schema_version_invalid")
    trace_id, page_index = value.get("trace_id"), value.get("page_index")
    if not _valid_text(trace_id):
        reasons.append("trace_id_invalid")
    if not _valid_page(page_index):
        reasons.append("page_index_invalid")
    if value.get("coordinate_space") != COORDINATE_SPACE:
        reasons.append("coordinate_space_invalid")

    items = value.get("items")
    if not isinstance(items, list):
        reasons.append("items_invalid")
        items = []
    else:
        for item in items:
            reasons.extend(_validate_item(item, page_index))
        ids = [
            item.get("candidate_id")
            for item in items
            if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
        ]
        if len(ids) != len(items) or ids != sorted(ids) or len(ids) != len(set(ids)):
            reasons.append("candidate_order_or_uniqueness_invalid")
    reasons.extend(_validate_audit(
        value.get("audit"), trace_id=trace_id, page_index=page_index,
        items=items,
    ))
    return sorted(set(reasons))


def canonical_review_candidates_v1_sample() -> dict[str, Any]:
    """Construct a one-item JSON contract for producer and consumer tests."""
    page_index = 0
    physical = "1" * 64
    bbox = {"x": 12.5, "y": 20.0, "w": 18.0, "h": 8.0}
    digest = _candidate_digest(page_index, physical, bbox, "234.567", "0.089")
    item = {
        "candidate_id": f"rcv1_{digest[:24]}",
        "bbox": bbox,
        "type": "linear",
        "nominal": "234.567",
        "symmetric_tolerance": "0.089",
        "upper_tolerance": None,
        "lower_tolerance": None,
        "prefix": None,
        "unit": None,
        "review_status": "pending",
        "source_stage": SOURCE_STAGE,
        "trace": {
            "l5_review_id": "review_1111111111111111",
            "phrase_id": "phrase_001",
            "physical_core_digest": physical,
            "source_l3_semantic_digest": "2" * 64,
            "source_text": "234.567±0.089",
            "canonical_text": "234.567±0.089",
            "candidate_digest": digest,
        },
    }
    trace_id = "trace_review_candidates_sample"
    stage_status = {key: "ok" for key in STAGE_STATUS_KEYS}
    stage_digests = {
        key: str(index + 3) * 64
        for index, key in enumerate(sorted(STAGE_DIGEST_KEYS))
    }
    chain_counts = {key: 1 for key in CHAIN_COUNT_KEYS}
    chain_counts["l4_recovered_count"] = 0
    chain = {
        "schema_version": "r33_m1_phrase_chain_audit_v1",
        "trace_id": trace_id,
        "page_index": page_index,
        "status": "ok",
        "semantic_projection_sha256": "a" * 64,
        "stage_status": stage_status,
        "stage_digests": stage_digests,
        "chain_counts": chain_counts,
    }
    return _envelope(
        trace_id=trace_id,
        page_index=page_index,
        items=[item],
        status="ok",
        unavailable_reason=None,
        chain=chain,
        counts={
            "source_review_count": 1, "candidate_count": 1,
            "dropped_count": 0, "drop_reasons": [],
        },
        template_identity={
            "schema_version": "r33_m1_template_identity_v1",
            "template_version": "controlled_template_v1",
            "content_sha256": "b" * 64,
        },
        runtime={key: 0 for key in RUNTIME_KEYS},
    )


def _normalize_chain_fields(
    value: Any,
    *,
    trace_id: str,
    page_index: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]] | None:
    if not isinstance(value, dict) or set(value) != CHAIN_FIELD_KEYS:
        return None
    audit = value.get("r33_m1_phrase_chain_audit_v1")
    if not _valid_chain_audit(audit, trace_id=trace_id, page_index=page_index):
        return None
    dumps = {
        name: value.get(field)
        for name, field in CHAIN_DUMP_FIELDS.items()
    }
    for dump in dumps.values():
        if not (
            isinstance(dump, dict)
            and dump.get("trace_id") == trace_id
            and dump.get("page_index") == page_index
            and dump.get("status") == "ok"
            and dump.get("ocr_called") is False
            and dump.get("consumer_allowed") is False
        ):
            return None
    l5 = dumps["l5"]
    if not (
        l5.get("release_allowed") is False
        and l5.get("feeds_display_l5") is False
    ):
        return None
    digests = {
        stage: _safe_dump_digest(dump)
        for stage, dump in dumps.items()
    }
    if any(digest is None for digest in digests.values()):
        return None
    canonical_digests = {
        stage: str(digest)
        for stage, digest in digests.items()
    }
    counts = _canonical_chain_counts(dumps)
    stage_status = {key: "ok" for key in STAGE_STATUS_KEYS}
    projection = {
        "template_content_sha256": audit["template_identity"]["content_sha256"],
        "stage_status": stage_status,
        "counts": counts,
        "pitch_reads": canonical_digests["pitch_reader"],
        "quality": canonical_digests["quality_gate"],
        "hypothesis": canonical_digests["hypothesis"],
        "dedup": canonical_digests["spatial_dedup"],
        "l3": canonical_digests["l3"],
        "l4": canonical_digests["l4"],
        "l5": canonical_digests["l5"],
    }
    canonical_audit = copy.deepcopy(audit)
    canonical_audit["stage_status"] = stage_status
    canonical_audit["counts"] = counts
    canonical_audit["semantic_projection_sha256"] = _digest(projection)
    return canonical_audit, dumps, canonical_digests


def _valid_chain_audit(value: Any, *, trace_id: str, page_index: int) -> bool:
    diagnostic_keys = {
        "details",
        "stage_status",
        "counts",
        "semantic_projection_sha256",
    }
    required_keys = _CHAIN_AUDIT_KEYS - diagnostic_keys
    if not (
        isinstance(value, dict)
        and required_keys.issubset(value)
        and set(value).issubset(_CHAIN_AUDIT_KEYS)
        and value.get("schema_version") == "r33_m1_phrase_chain_audit_v1"
        and value.get("trace_id") == trace_id
        and value.get("page_index") == page_index
        and value.get("status") == "ok"
        and value.get("reason") is None
        and value.get("ocr_family_call_count") == 0
        and value.get("paddleocr_call_count") == 0
        and value.get("rapidocr_call_count") == 0
        and value.get("yolo_char_call_count") == 0
        and value.get("yolo_b_call_count") == 0
        and value.get("dimensions_promoted_count") == 0
        and value.get("release_allowed") is False
        and value.get("consumer_allowed") is False
    ):
        return False
    identity = value.get("template_identity")
    return bool(
        isinstance(identity, dict)
        and set(identity) == TEMPLATE_IDENTITY_KEYS
        and identity.get("schema_version") == "r33_m1_template_identity_v1"
        and _valid_text(identity.get("template_version"))
        and _valid_sha(identity.get("content_sha256"))
    )


def _project_row(
    row: Any,
    *,
    trace_id: str,
    page_index: int,
    pitch_read: Any = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(row, dict) or row.get("source_collection") not in {
        "L3_dimensions",
        "L4_unresolved_suspects",
        "L4_recovered",
    }:
        return None, "source_not_l3_dimension"
    if row.get("trace_id") != trace_id or row.get("page_index") != page_index:
        return None, "source_trace_invalid"
    if (
        row.get("source_collection") == "L4_unresolved_suspects"
        and row.get("source_primary_reason") in {"boundary", "geometry"}
    ):
        # Accounting and syntax failures may stay review-visible.  A failed
        # page/view/GD&T boundary is real evidence safety, so its text must
        # never become a confirmable public candidate.
        return None, "strict_boundary_or_geometry_rejected"
    text = str(
        row.get("canonical_text")
        or row.get("source_text")
        or (
            pitch_read.get("text")
            if isinstance(pitch_read, dict)
            else ""
        )
        or (
            pitch_read.get("raw_text")
            if isinstance(pitch_read, dict)
            else ""
        )
        or ""
    )
    if not text:
        return None, "text_missing"
    syntax = classify_dimension_syntax(
        text,
        drops=(
            pitch_read.get("drops")
            if isinstance(pitch_read, dict)
            else None
        ),
    )
    semantics = row.get("dimension_semantics")
    if not _valid_semantics(semantics):
        semantics = _dimension_semantics(text)
    if not _valid_semantics(semantics):
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=(
                syntax
                if syntax != SYNTAX_OK
                else SYNTAX_INVALID_DIMENSION
            ),
        )
    if semantics["qualifier"] is not None:
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=SYNTAX_INVALID_DIMENSION,
        )
    quantity = semantics["quantity"]
    if not 1 <= quantity <= 999:
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=SYNTAX_INVALID_DIMENSION,
        )
    kind = semantics["kind"]
    if kind not in {"linear", "diameter", "radius", "angle"}:
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=(
                syntax
                if syntax != SYNTAX_OK
                else SYNTAX_INVALID_DIMENSION
            ),
        )
    prefix = semantics["prefix"]
    expected_prefix = {
        "linear": None,
        "diameter": "Ø",
        "radius": "R",
        "angle": None,
    }[kind]
    if prefix != expected_prefix:
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=SYNTAX_INVALID_DIMENSION,
        )
    unit = semantics["unit"]
    expected_unit = "degree" if kind == "angle" else None
    if unit != expected_unit:
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=SYNTAX_INVALID_DIMENSION,
        )
    tolerance = semantics["tolerance"]
    if tolerance["kind"] not in {
        "none", "symmetric", "bilateral",
        "unilateral_upper", "unilateral_lower",
    }:
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=SYNTAX_INVALID_DIMENSION,
        )
    nominal = semantics["main_value_text"]
    if not _valid_decimal(nominal):
        return _project_correction_row(
            row,
            trace_id=trace_id,
            page_index=page_index,
            text=text,
            syntax=SYNTAX_INVALID_DIMENSION,
        )
    symmetric: str | None = None
    upper_tolerance: str | None = None
    lower_tolerance: str | None = None
    if tolerance["kind"] == "none":
        if any(tolerance[key] is not None for key in ("upper_text", "lower_text", "unit")):
            return _project_correction_row(
                row,
                trace_id=trace_id,
                page_index=page_index,
                text=text,
                syntax=SYNTAX_INVALID_DIMENSION,
            )
    elif tolerance["kind"] == "symmetric":
        upper, lower = tolerance["upper_text"], tolerance["lower_text"]
        if not (
            isinstance(upper, str) and upper.startswith("+")
            and lower == f"-{upper[1:]}" and _valid_unsigned_decimal(upper[1:])
            and tolerance["unit"] == unit
        ):
            return _project_correction_row(
                row,
                trace_id=trace_id,
                page_index=page_index,
                text=text,
                syntax=SYNTAX_INVALID_DIMENSION,
            )
        symmetric = upper[1:]
    elif tolerance["kind"] == "bilateral":
        upper, lower = tolerance["upper_text"], tolerance["lower_text"]
        if not (
            isinstance(upper, str) and upper.startswith("+")
            and _valid_unsigned_decimal(upper[1:])
            and isinstance(lower, str) and lower.startswith("-")
            and _valid_unsigned_decimal(lower[1:])
            and tolerance["unit"] == unit
        ):
            return _project_correction_row(
                row,
                trace_id=trace_id,
                page_index=page_index,
                text=text,
                syntax=SYNTAX_INVALID_DIMENSION,
            )
        upper_tolerance = upper
        lower_tolerance = lower
    elif tolerance["kind"] == "unilateral_upper":
        upper, lower = tolerance["upper_text"], tolerance["lower_text"]
        if not (
            isinstance(upper, str) and upper.startswith("+")
            and _valid_unsigned_decimal(upper[1:])
            and lower == "0"
            and tolerance["unit"] == unit
        ):
            return _project_correction_row(
                row,
                trace_id=trace_id,
                page_index=page_index,
                text=text,
                syntax=SYNTAX_INVALID_DIMENSION,
            )
        upper_tolerance = upper
        lower_tolerance = lower
    else:
        upper, lower = tolerance["upper_text"], tolerance["lower_text"]
        if not (
            upper == "0"
            and isinstance(lower, str) and lower.startswith("-")
            and _valid_unsigned_decimal(lower[1:])
            and tolerance["unit"] == unit
        ):
            return _project_correction_row(
                row,
                trace_id=trace_id,
                page_index=page_index,
                text=text,
                syntax=SYNTAX_INVALID_DIMENSION,
            )
        upper_tolerance = upper
        lower_tolerance = lower
    bbox = row.get("bbox")
    if not _valid_bbox(bbox):
        return None, "bbox_invalid"
    physical = row.get("physical_core_digest")
    l3_digest = row.get("source_l3_semantic_digest")
    text_keys = ("review_id", "phrase_id", "source_text", "canonical_text")
    if not (
        _valid_sha(physical) and _valid_text(l3_digest)
        and all(_valid_text(row.get(key)) for key in text_keys)
    ):
        return None, "source_trace_invalid"
    normalized_bbox = {key: float(bbox[key]) for key in ("x", "y", "w", "h")}
    digest = _candidate_digest(
        page_index,
        physical,
        normalized_bbox,
        nominal,
        symmetric,
        upper_tolerance=upper_tolerance,
        lower_tolerance=lower_tolerance,
        prefix=prefix,
        unit=unit,
        quantity=quantity,
        syntax=syntax,
    )
    return {
        "candidate_id": f"rcv1_{digest[:24]}",
        "bbox": normalized_bbox,
        "type": "linear",
        "nominal": nominal,
        "symmetric_tolerance": symmetric,
        "upper_tolerance": upper_tolerance,
        "lower_tolerance": lower_tolerance,
        "prefix": prefix,
        "unit": unit,
        **({"quantity": quantity} if quantity != 1 else {}),
        "review_status": "pending",
        "source_stage": SOURCE_STAGE,
        "syntax": syntax,
        "trace": {
            "l5_review_id": row["review_id"],
            "phrase_id": row["phrase_id"],
            "physical_core_digest": physical,
            "source_l3_semantic_digest": l3_digest,
            "source_text": row["source_text"],
            "canonical_text": text,
            "candidate_digest": digest,
        },
    }, None


def _project_correction_row(
    row: dict[str, Any],
    *,
    trace_id: str,
    page_index: int,
    text: str,
    syntax: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Expose an invalid string as review-only data without inventing a value."""
    if row.get("trace_id") != trace_id or row.get("page_index") != page_index:
        return None, "source_trace_invalid"
    bbox = row.get("bbox")
    if not _valid_bbox(bbox):
        return None, "bbox_invalid"
    physical = row.get("physical_core_digest")
    l3_digest = row.get("source_l3_semantic_digest")
    review_id = row.get("review_id")
    phrase_id = row.get("phrase_id")
    if not (
        _valid_sha(physical)
        and _valid_text(l3_digest)
        and _valid_text(review_id)
        and _valid_text(phrase_id)
        and _valid_text(text)
        and syntax in DIMENSION_SYNTAX_LABELS - {SYNTAX_OK}
    ):
        return None, "source_trace_invalid"
    normalized_bbox = {
        key: float(bbox[key])
        for key in ("x", "y", "w", "h")
    }
    digest = _candidate_digest(
        page_index,
        physical,
        normalized_bbox,
        None,
        None,
        syntax=syntax,
    )
    return {
        "candidate_id": f"rcv1_{digest[:24]}",
        "bbox": normalized_bbox,
        "type": "linear",
        "nominal": None,
        "symmetric_tolerance": None,
        "upper_tolerance": None,
        "lower_tolerance": None,
        "prefix": None,
        "unit": None,
        "review_status": "pending",
        "source_stage": SOURCE_STAGE,
        "syntax": syntax,
        "trace": {
            "l5_review_id": review_id,
            "phrase_id": phrase_id,
            "physical_core_digest": physical,
            "source_l3_semantic_digest": l3_digest,
            "source_text": str(row.get("source_text") or text),
            "canonical_text": text,
            "candidate_digest": digest,
        },
    }, None


def _final_position_dedup(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Greedily retain one deterministic winner per same-type position."""
    ranked = sorted(items, key=_final_position_rank)
    kept: list[dict[str, Any]] = []
    suppressed = 0
    for candidate in ranked:
        if any(
            prior.get("type") == candidate.get("type")
            and _position_duplicate(
                prior.get("bbox"),
                candidate.get("bbox"),
            )
            for prior in kept
        ):
            suppressed += 1
            continue
        kept.append(candidate)
    return kept, suppressed


def _final_position_rank(item: dict[str, Any]) -> tuple[Any, ...]:
    bbox = item.get("bbox") if isinstance(item, dict) else {}
    area = (
        float(bbox.get("w") or 0.0) * float(bbox.get("h") or 0.0)
        if isinstance(bbox, dict)
        else 0.0
    )
    syntax_rank = _SYNTAX_RANK.get(str(item.get("syntax") or "ok"), 4)
    trace = item.get("trace") if isinstance(item.get("trace"), dict) else {}
    text_length = len(str(trace.get("canonical_text") or ""))
    if item.get("type") == "gdt":
        confidence = (
            float(trace.get("symbol_confidence"))
            if _finite_number(trace.get("symbol_confidence"))
            else 0.0
        )
        return (
            "gdt",
            syntax_rank,
            -area,
            -confidence,
            str(item.get("candidate_id") or ""),
        )
    return (
        "linear",
        syntax_rank,
        -text_length,
        area,
        str(item.get("candidate_id") or ""),
    )


def _position_duplicate(left: Any, right: Any) -> bool:
    if not (_valid_bbox(left) and _valid_bbox(right)):
        return False
    lx0, ly0 = float(left["x"]), float(left["y"])
    rx0, ry0 = float(right["x"]), float(right["y"])
    lx1, ly1 = lx0 + float(left["w"]), ly0 + float(left["h"])
    rx1, ry1 = rx0 + float(right["w"]), ry0 + float(right["h"])
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0,
        min(ly1, ry1) - max(ly0, ry0),
    )
    if intersection <= 0.0:
        return False
    left_area = float(left["w"]) * float(left["h"])
    right_area = float(right["w"]) * float(right["h"])
    union = left_area + right_area - intersection
    return bool(
        intersection / union >= FINAL_DEDUP_IOU_THRESHOLD
        or intersection / min(left_area, right_area)
        >= FINAL_DEDUP_CONTAINMENT_THRESHOLD
    )


def _valid_semantics(value: Any) -> bool:
    keys = {
        "schema_version", "kind", "quantity", "prefix", "main_value_text",
        "unit", "qualifier", "tolerance", "consumer_allowed",
    }
    tolerance = value.get("tolerance") if isinstance(value, dict) else None
    return bool(
        isinstance(value, dict) and set(value) == keys
        and value.get("schema_version") == "vector_dimension_semantics_v1"
        and type(value.get("quantity")) is int
        and isinstance(tolerance, dict)
        and set(tolerance) == {"kind", "upper_text", "lower_text", "unit"}
        and value.get("consumer_allowed") is False
    )


def _validate_item(value: Any, page_index: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["item_not_object"]
    item_type = value.get("type")
    if item_type == "linear":
        return _validate_linear_item(value, page_index)
    if item_type == "gdt":
        return _validate_gdt_item(value, page_index)
    return ["item_keys_invalid", "item_type_invalid"]


def _valid_item_keys(
    value: dict[str, Any],
    required: frozenset[str],
    optional: frozenset[str] = OPTIONAL_ITEM_KEYS,
) -> bool:
    keys = set(value)
    return bool(
        required.issubset(keys)
        and keys - required <= optional
    )


def _validate_linear_item(value: dict[str, Any], page_index: Any) -> list[str]:
    reasons: list[str] = []
    if not _valid_item_keys(
        value,
        LINEAR_ITEM_KEYS,
        LINEAR_OPTIONAL_ITEM_KEYS,
    ):
        reasons.append("item_keys_invalid")
    if value.get("type") != "linear":
        reasons.append("item_type_invalid")
    if not _valid_bbox(value.get("bbox")):
        reasons.append("item_bbox_invalid")
    syntax = value.get("syntax")
    if syntax is not None and syntax not in DIMENSION_SYNTAX_LABELS:
        reasons.append("item_syntax_invalid")
    correction_candidate = bool(
        value.get("nominal") is None
        and syntax in DIMENSION_SYNTAX_LABELS - {SYNTAX_OK}
    )
    if not (_valid_decimal(value.get("nominal")) or correction_candidate):
        reasons.append("item_nominal_invalid")
    symmetric = value.get("symmetric_tolerance")
    upper = value.get("upper_tolerance")
    lower = value.get("lower_tolerance")
    if not (
        _valid_linear_tolerances(symmetric, upper, lower)
        and (
            not correction_candidate
            or all(value is None for value in (symmetric, upper, lower))
        )
    ):
        reasons.append("item_tolerance_invalid")
    prefix = value.get("prefix")
    if prefix not in {None, "Ø", "R"}:
        reasons.append("item_prefix_invalid")
    unit = value.get("unit")
    if unit not in {None, "degree"}:
        reasons.append("item_unit_invalid")
    if not (
        (prefix in {None, "Ø", "R"} and unit is None)
        or (prefix is None and unit == "degree")
    ):
        reasons.append("item_prefix_unit_invalid")
    has_quantity = "quantity" in value
    quantity = value.get("quantity", 1)
    quantity_valid = bool(
        type(quantity) is int
        and (
            (not has_quantity and quantity == 1)
            or (has_quantity and 2 <= quantity <= 999)
        )
    )
    if not quantity_valid:
        reasons.append("item_quantity_invalid")
    if value.get("review_status") != "pending":
        reasons.append("item_review_status_invalid")
    if value.get("source_stage") != SOURCE_STAGE:
        reasons.append("item_source_stage_invalid")
    trace = value.get("trace")
    if not isinstance(trace, dict) or set(trace) != LINEAR_ITEM_TRACE_KEYS:
        reasons.append("item_trace_keys_invalid")
        return reasons
    if not (
        _valid_sha(trace.get("physical_core_digest"))
        and _valid_text(trace.get("source_l3_semantic_digest"))
        and all(_valid_text(trace.get(key)) for key in (
            "l5_review_id", "phrase_id", "source_text", "canonical_text"
        ))
    ):
        reasons.append("item_trace_invalid")
    if (
        _valid_page(page_index)
        and _valid_bbox(value.get("bbox"))
        and (_valid_decimal(value.get("nominal")) or correction_candidate)
        and _valid_linear_tolerances(symmetric, upper, lower)
        and prefix in {None, "Ø", "R"}
        and (
            (prefix in {None, "Ø", "R"} and unit is None)
            or (prefix is None and unit == "degree")
        )
        and quantity_valid
        and _valid_sha(trace.get("physical_core_digest"))
    ):
        digest = _candidate_digest(
            page_index, trace.get("physical_core_digest"), value["bbox"],
            value.get("nominal"), symmetric,
            upper_tolerance=upper,
            lower_tolerance=lower,
            prefix=prefix,
            unit=unit,
            quantity=quantity,
            syntax=syntax,
        )
        if trace.get("candidate_digest") != digest:
            reasons.append("item_candidate_digest_invalid")
        if value.get("candidate_id") != f"rcv1_{digest[:24]}":
            reasons.append("item_candidate_id_invalid")
    return reasons


def _validate_gdt_item(value: dict[str, Any], page_index: Any) -> list[str]:
    reasons: list[str] = []
    if not _valid_item_keys(value, GDT_ITEM_KEYS):
        reasons.append("item_keys_invalid")
    if value.get("type") != "gdt":
        reasons.append("item_type_invalid")
    if not _valid_bbox(value.get("bbox")):
        reasons.append("item_bbox_invalid")
    symbol_name = value.get("symbol_name")
    symbol_unicode = value.get("symbol_unicode")
    if not _valid_gdt_symbol_pair(symbol_name, symbol_unicode):
        reasons.append("item_gdt_symbol_invalid")
    syntax = value.get("syntax")
    if syntax is not None and syntax not in DIMENSION_SYNTAX_LABELS:
        reasons.append("item_syntax_invalid")
    if not _valid_gdt_tolerance_or_correction(
        value.get("tolerance_value"),
        syntax,
    ):
        reasons.append("item_gdt_tolerance_invalid")
    if any(value.get(key) is not None for key in (
        "datum_1", "datum_2", "datum_3",
    )):
        reasons.append("item_gdt_datum_not_null")
    if value.get("review_status") != "pending":
        reasons.append("item_review_status_invalid")
    if value.get("source_stage") != SOURCE_STAGE:
        reasons.append("item_source_stage_invalid")

    trace = value.get("trace")
    if not isinstance(trace, dict) or set(trace) != GDT_ITEM_TRACE_KEYS:
        reasons.append("item_trace_keys_invalid")
        return reasons
    reasons.extend(_validate_gdt_trace_fields(trace))
    if not _valid_sha(trace.get("candidate_digest")):
        reasons.append("item_candidate_digest_invalid")

    if (
        _valid_page(page_index)
        and _valid_bbox(value.get("bbox"))
        and _valid_gdt_symbol_pair(symbol_name, symbol_unicode)
        and _valid_gdt_tolerance_or_correction(
            value.get("tolerance_value"),
            syntax,
        )
        and all(value.get(key) is None for key in (
            "datum_1", "datum_2", "datum_3",
        ))
        and not _validate_gdt_trace_fields(trace)
    ):
        normalized_bbox = {
            key: float(value["bbox"][key])
            for key in ("x", "y", "w", "h")
        }
        digest = _gdt_candidate_digest(
            page_index=page_index,
            bbox=normalized_bbox,
            symbol_name=symbol_name,
            symbol_unicode=symbol_unicode,
            tolerance_value=value.get("tolerance_value"),
            trace=trace,
            syntax=syntax,
        )
        if trace.get("candidate_digest") != digest:
            reasons.append("item_candidate_digest_invalid")
        if value.get("candidate_id") != f"rcv1_{digest[:24]}":
            reasons.append("item_candidate_id_invalid")
    return reasons


def _valid_gdt_tolerance_or_correction(
    value: Any,
    syntax: Any,
) -> bool:
    return bool(
        _valid_unsigned_decimal(value)
        or (
            _valid_text(value)
            and syntax in DIMENSION_SYNTAX_LABELS - {SYNTAX_OK}
        )
    )


def _validate_gdt_trace_fields(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["item_gdt_trace_invalid"]
    reasons: list[str] = []
    if not _valid_sha(value.get("frame_digest")):
        reasons.append("item_gdt_frame_digest_invalid")
    if not _valid_text(value.get("decimal_quad_id")):
        reasons.append("item_gdt_decimal_quad_id_invalid")
    if value.get("symbol_method") != "vector_inversion":
        reasons.append("item_gdt_symbol_method_invalid")
    confidence = value.get("symbol_confidence")
    if not (
        _finite_number(confidence)
        and 0.60 <= float(confidence) <= 1.0
    ):
        reasons.append("item_gdt_symbol_confidence_invalid")
    symbol_index = value.get("symbol_compartment_index")
    if type(symbol_index) is not int or symbol_index != 0:
        reasons.append("item_gdt_symbol_compartment_invalid")
    tolerance_index = value.get("tolerance_compartment_index")
    if type(tolerance_index) is not int or tolerance_index < 1:
        reasons.append("item_gdt_tolerance_compartment_invalid")
    if not _valid_text(value.get("reader_phrase_id")):
        reasons.append("item_gdt_reader_phrase_id_invalid")
    if not _valid_sha(value.get("template_content_sha256")):
        reasons.append("item_gdt_template_sha_invalid")
    return reasons


def _validate_audit(
    value: Any, *, trace_id: Any, page_index: Any, items: list[Any],
) -> list[str]:
    if not isinstance(value, dict):
        return ["audit_not_object"]
    reasons: list[str] = []
    if set(value) != AUDIT_KEYS:
        reasons.append("audit_keys_invalid")
    if value.get("schema_version") != AUDIT_SCHEMA_VERSION:
        reasons.append("audit_schema_invalid")
    status, unavailable_reason = value.get("status"), value.get("unavailable_reason")
    if status not in {"ok", "unavailable"}:
        reasons.append("audit_status_invalid")
    if (status == "ok" and unavailable_reason is not None) or (
        status == "unavailable" and not _valid_text(unavailable_reason)
    ):
        reasons.append("audit_reason_invalid")
    chain = value.get("chain")
    if not isinstance(chain, dict) or set(chain) != CHAIN_KEYS:
        reasons.append("audit_chain_keys_invalid")
    else:
        if chain.get("trace_id") != trace_id or chain.get("page_index") != page_index:
            reasons.append("audit_chain_identity_invalid")
        if chain.get("schema_version") != "r33_m1_phrase_chain_audit_v1":
            reasons.append("audit_chain_schema_invalid")
        stage_status, stage_digests = chain.get("stage_status"), chain.get("stage_digests")
        chain_counts = chain.get("chain_counts")
        if not isinstance(stage_status, dict) or set(stage_status) != STAGE_STATUS_KEYS:
            reasons.append("audit_stage_status_keys_invalid")
        if not isinstance(stage_digests, dict) or set(stage_digests) != STAGE_DIGEST_KEYS:
            reasons.append("audit_stage_digest_keys_invalid")
        if not isinstance(chain_counts, dict) or set(chain_counts) != CHAIN_COUNT_KEYS:
            reasons.append("audit_chain_count_keys_invalid")
        if status == "ok":
            if chain.get("status") != "ok" or not _valid_sha(chain.get("semantic_projection_sha256")):
                reasons.append("audit_chain_status_invalid")
            if isinstance(stage_status, dict) and any(value != "ok" for value in stage_status.values()):
                reasons.append("audit_stage_status_invalid")
            if isinstance(stage_digests, dict) and any(not _valid_sha(value) for value in stage_digests.values()):
                reasons.append("audit_stage_digest_invalid")
            if isinstance(chain_counts, dict) and any(not _valid_count(value) for value in chain_counts.values()):
                reasons.append("audit_chain_count_invalid")
        elif not (
            chain.get("status") == "unavailable"
            and chain.get("semantic_projection_sha256") is None
            and isinstance(stage_digests, dict)
            and all(value is None for value in stage_digests.values())
            and isinstance(chain_counts, dict)
            and all(value == 0 for value in chain_counts.values())
        ):
            reasons.append("unavailable_chain_invalid")
    counts = value.get("counts")
    if not isinstance(counts, dict) or set(counts) != COUNTS_KEYS:
        reasons.append("audit_counts_keys_invalid")
    else:
        drops = counts.get("drop_reasons")
        if not all(_valid_count(counts.get(key)) for key in (
            "source_review_count", "candidate_count", "dropped_count"
        )) or not isinstance(drops, list):
            reasons.append("audit_counts_invalid")
        else:
            names = []
            for drop in drops:
                if not isinstance(drop, dict) or set(drop) != DROP_REASON_KEYS:
                    reasons.append("audit_drop_keys_invalid")
                    continue
                names.append(drop.get("reason"))
                if drop.get("reason") not in _DROP_REASON_ORDER or not (
                    _valid_count(drop.get("count")) and drop["count"] > 0
                ):
                    reasons.append("audit_drop_invalid")
            if names != [reason for reason in _DROP_REASON_ORDER if reason in names]:
                reasons.append("audit_drop_order_invalid")
            drop_sum = sum(
                drop.get("count", 0)
                for drop in drops
                if isinstance(drop, dict) and _valid_count(drop.get("count"))
            )
            if drop_sum != counts.get("dropped_count"):
                reasons.append("audit_drop_count_mismatch")
        if counts.get("candidate_count") != len(items):
            reasons.append("audit_candidate_count_mismatch")
        if status == "ok" and counts.get("source_review_count") != counts.get("candidate_count", 0) + counts.get("dropped_count", 0):
            reasons.append("audit_source_count_mismatch")
        if status == "unavailable" and any(counts.get(key) != 0 for key in (
            "source_review_count", "candidate_count", "dropped_count"
        )):
            reasons.append("unavailable_counts_invalid")
    identity = value.get("template_identity")
    if not isinstance(identity, dict) or set(identity) != TEMPLATE_IDENTITY_KEYS:
        reasons.append("audit_template_keys_invalid")
    elif status == "ok" and not (
        identity.get("schema_version") == "r33_m1_template_identity_v1"
        and _valid_text(identity.get("template_version"))
        and _valid_sha(identity.get("content_sha256"))
    ):
        reasons.append("audit_template_invalid")
    elif status == "unavailable" and not (
        identity.get("schema_version") == "r33_m1_template_identity_v1"
        and identity.get("template_version") is None
        and identity.get("content_sha256") is None
    ):
        reasons.append("unavailable_template_invalid")
    if (
        status == "ok"
        and isinstance(identity, dict)
        and _valid_sha(identity.get("content_sha256"))
        and any(
            isinstance(item, dict)
            and item.get("type") == "gdt"
            and (
                not isinstance(item.get("trace"), dict)
                or item["trace"].get("template_content_sha256")
                != identity["content_sha256"]
            )
            for item in items
        )
    ):
        reasons.append("item_gdt_template_identity_mismatch")
    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS or any(
        runtime.get(key) != 0 for key in RUNTIME_KEYS
    ):
        reasons.append("audit_runtime_invalid")
    authority = value.get("authority")
    if not (
        isinstance(authority, dict)
        and set(authority) == AUTHORITY_KEYS
        and authority == _authority()
    ):
        reasons.append("audit_authority_invalid")
    if status == "unavailable" and items:
        reasons.append("unavailable_items_invalid")
    try:
        expected_semantic_digest = _semantic_digest(page_index, items)
    except (TypeError, ValueError, UnicodeError):
        reasons.append("audit_semantic_payload_invalid")
    else:
        if value.get("semantic_digest") != expected_semantic_digest:
            reasons.append("audit_semantic_digest_invalid")
    return reasons


def _chain_projection(audit: dict[str, Any], digests: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": audit["schema_version"],
        "trace_id": audit["trace_id"],
        "page_index": audit["page_index"],
        "status": audit["status"],
        "semantic_projection_sha256": audit["semantic_projection_sha256"],
        "stage_status": copy.deepcopy(audit["stage_status"]),
        "stage_digests": copy.deepcopy(digests),
        "chain_counts": copy.deepcopy(audit["counts"]),
    }


def _envelope(
    *, trace_id: str, page_index: int, items: list[dict[str, Any]],
    status: str, unavailable_reason: str | None, chain: dict[str, Any],
    counts: dict[str, Any], template_identity: dict[str, Any],
    runtime: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": page_index,
        "coordinate_space": COORDINATE_SPACE,
        "items": items,
        "audit": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": status,
            "unavailable_reason": unavailable_reason,
            "chain": chain,
            "counts": counts,
            "template_identity": template_identity,
            "runtime": copy.deepcopy(runtime),
            "authority": _authority(),
            "semantic_digest": _semantic_digest(page_index, items),
        },
    }


def _unavailable(*, trace_id: str, page_index: int, reason: str) -> dict[str, Any]:
    result = _envelope(
        trace_id=trace_id,
        page_index=page_index,
        items=[],
        status="unavailable",
        unavailable_reason=reason,
        chain={
            "schema_version": "r33_m1_phrase_chain_audit_v1",
            "trace_id": trace_id,
            "page_index": page_index,
            "status": "unavailable",
            "semantic_projection_sha256": None,
            "stage_status": {key: "not_run" for key in STAGE_STATUS_KEYS},
            "stage_digests": {key: None for key in STAGE_DIGEST_KEYS},
            "chain_counts": {key: 0 for key in CHAIN_COUNT_KEYS},
        },
        counts={
            "source_review_count": 0, "candidate_count": 0,
            "dropped_count": 0, "drop_reasons": [],
        },
        template_identity={
            "schema_version": "r33_m1_template_identity_v1",
            "template_version": None,
            "content_sha256": None,
        },
        runtime={key: 0 for key in RUNTIME_KEYS},
    )
    reasons = validate_review_candidates_v1(result)
    if reasons:
        raise ValueError(f"unavailable review schema bug: {reasons}")
    return result


def _unavailable_reason(value: Any) -> str:
    if isinstance(value, dict):
        audit = value.get("r33_m1_phrase_chain_audit_v1")
        if isinstance(audit, dict) and _valid_text(audit.get("reason")):
            return audit["reason"]
    return "phrase_chain_unavailable"


def _authority() -> dict[str, bool]:
    return {
        "review_only": True,
        "formal_project_data_allowed": False,
        "consumer_allowed": False,
        "release_allowed": False,
    }


def _candidate_digest(
    page: Any,
    physical: Any,
    bbox: Any,
    nominal: Any,
    tolerance: Any,
    *,
    upper_tolerance: Any = None,
    lower_tolerance: Any = None,
    prefix: Any = None,
    unit: Any = None,
    quantity: Any = 1,
    syntax: Any = None,
) -> str:
    """Keep legacy linear ids stable while binding every widened field."""
    payload = {
        "schema_version": SCHEMA_VERSION, "page_index": page,
        "physical_core_digest": physical, "bbox": bbox, "nominal": nominal,
        "symmetric_tolerance": tolerance,
    }
    if any(
        value is not None
        for value in (upper_tolerance, lower_tolerance, prefix, unit)
    ):
        payload.update({
            "type": "linear",
            "upper_tolerance": upper_tolerance,
            "lower_tolerance": lower_tolerance,
            "prefix": prefix,
            "unit": unit,
        })
    if quantity != 1:
        payload["quantity"] = quantity
    if syntax is not None:
        payload["syntax"] = syntax
    return _digest(payload)


def _gdt_candidate_digest(
    *,
    page_index: Any,
    bbox: Any,
    symbol_name: Any,
    symbol_unicode: Any,
    tolerance_value: Any,
    trace: dict[str, Any],
    syntax: Any = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "page_index": page_index,
        "type": "gdt",
        "bbox": bbox,
        "symbol_name": symbol_name,
        "symbol_unicode": symbol_unicode,
        "tolerance_value": tolerance_value,
        "datum_1": None,
        "datum_2": None,
        "datum_3": None,
        "trace": {
            "frame_digest": trace.get("frame_digest"),
            "decimal_quad_id": trace.get("decimal_quad_id"),
            "symbol_method": trace.get("symbol_method"),
            "symbol_confidence": (
                float(trace["symbol_confidence"])
                if _finite_number(trace.get("symbol_confidence"))
                else trace.get("symbol_confidence")
            ),
            "symbol_compartment_index": trace.get(
                "symbol_compartment_index"
            ),
            "tolerance_compartment_index": trace.get(
                "tolerance_compartment_index"
            ),
            "reader_phrase_id": trace.get("reader_phrase_id"),
            "template_content_sha256": trace.get(
                "template_content_sha256"
            ),
        },
    }
    if syntax is not None:
        payload["syntax"] = syntax
    return _digest(payload)


def _semantic_digest(page: Any, items: list[Any]) -> str:
    semantic = []
    for item in items:
        trace = item.get("trace", {}) if isinstance(item, dict) else {}
        if isinstance(item, dict) and item.get("type") == "gdt":
            gdt_item = {
                "candidate_id": item.get("candidate_id"),
                "bbox": item.get("bbox"),
                "type": "gdt",
                "symbol_name": item.get("symbol_name"),
                "symbol_unicode": item.get("symbol_unicode"),
                "tolerance_value": item.get("tolerance_value"),
                "datum_1": item.get("datum_1"),
                "datum_2": item.get("datum_2"),
                "datum_3": item.get("datum_3"),
                "review_status": item.get("review_status"),
                "source_stage": item.get("source_stage"),
                "trace": {
                    key: trace.get(key)
                    for key in sorted(GDT_ITEM_TRACE_KEYS)
                },
            }
            if "syntax" in item:
                gdt_item["syntax"] = item.get("syntax")
            semantic.append(gdt_item)
        else:
            linear_item = {
                "candidate_id": (
                    item.get("candidate_id")
                    if isinstance(item, dict)
                    else None
                ),
                "bbox": item.get("bbox") if isinstance(item, dict) else None,
                "nominal": (
                    item.get("nominal")
                    if isinstance(item, dict)
                    else None
                ),
                "symmetric_tolerance": (
                    item.get("symmetric_tolerance")
                    if isinstance(item, dict)
                    else None
                ),
                "review_status": (
                    item.get("review_status")
                    if isinstance(item, dict)
                    else None
                ),
                "source_stage": (
                    item.get("source_stage")
                    if isinstance(item, dict)
                    else None
                ),
                "physical_core_digest": trace.get("physical_core_digest"),
            }
            if isinstance(item, dict) and any(
                item.get(key) is not None
                for key in (
                    "upper_tolerance", "lower_tolerance", "prefix", "unit",
                )
            ):
                linear_item.update({
                    "type": "linear",
                    "upper_tolerance": item.get("upper_tolerance"),
                    "lower_tolerance": item.get("lower_tolerance"),
                    "prefix": item.get("prefix"),
                    "unit": item.get("unit"),
                })
            if isinstance(item, dict) and "syntax" in item:
                linear_item["syntax"] = item.get("syntax")
            if isinstance(item, dict) and "quantity" in item:
                linear_item["quantity"] = item.get("quantity")
            # The all-null widened shape keeps the exact pre-R34 byte
            # projection; extended rows bind every new field.
            semantic.append(linear_item)
    return _digest({
        "schema_version": SCHEMA_VERSION, "page_index": page,
        "coordinate_space": COORDINATE_SPACE, "items": semantic,
    })


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _safe_dump_digest(value: Any) -> str | None:
    """Bind diagnostics to the actual dump without trusting its own seal."""
    try:
        return _digest(value)
    except (
        TypeError,
        ValueError,
        UnicodeError,
        RecursionError,
        OverflowError,
    ):
        return None


def _canonical_chain_counts(
    dumps: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Derive non-authoritative public diagnostics from concrete collections."""
    reads = _dump_rows(dumps["pitch_reader"], "reads")
    quality_rows = _dump_rows(dumps["quality_gate"], "rows")
    hypotheses = _dump_rows(dumps["hypothesis"], "hypotheses")
    return {
        "page_context_build_count": 1,
        "page_context_extraction_call_count": _stat(
            dumps["pitch_reader"],
            "page_context_extraction_call_count",
        ),
        "phrase_region_count": len(reads),
        "reader_call_count": sum(
            row.get("reader_call_count")
            if isinstance(row, dict)
            and _valid_count(row.get("reader_call_count"))
            else 0
            for row in reads
        ),
        "reader_success_count": sum(
            isinstance(row, dict) and row.get("status") == "read"
            for row in reads
        ),
        "quality_pass_count": sum(
            isinstance(row, dict) and row.get("phrase_complete") is True
            for row in quality_rows
        ),
        "ordinary_hypothesis_count": sum(
            isinstance(row, dict)
            and row.get("ordinary_candidate") is True
            for row in hypotheses
        ),
        "dedup_winner_count": len(_dump_rows(
            dumps["spatial_dedup"],
            "winner_hypotheses",
        )),
        "l3_dimension_count": len(_dump_rows(
            dumps["l3"],
            "L3_dimensions",
        )),
        "l4_recovered_count": len(_dump_rows(
            dumps["l4"],
            "L4_recovered",
        )),
        "l5_review_required_count": len(_dump_rows(
            dumps["l5"],
            "L5_review_required",
        )),
    }


def _dump_rows(dump: dict[str, Any], key: str) -> list[Any]:
    value = dump.get(key)
    return value if isinstance(value, list) else []


def _stat(dump: dict[str, Any], key: str) -> int:
    value = (dump.get("stats") or {}).get(key)
    return value if _valid_count(value) else 0


def _valid_bbox(value: Any) -> bool:
    return bool(
        isinstance(value, dict) and set(value) == BBOX_KEYS
        and all(type(value[key]) in {int, float} and math.isfinite(float(value[key])) for key in BBOX_KEYS)
        and value["w"] > 0 and value["h"] > 0
    )


def _finite_number(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _valid_gdt_symbol_pair(name: Any, unicode_value: Any) -> bool:
    return bool(
        isinstance(name, str)
        and isinstance(unicode_value, str)
        and GDT_CORE_SYMBOLS.get(name) == unicode_value
    )


def _valid_decimal(value: Any) -> bool:
    return isinstance(value, str) and _DECIMAL_TEXT.fullmatch(value) is not None


def _valid_unsigned_decimal(value: Any) -> bool:
    return _valid_decimal(value) and not value.startswith(("+", "-"))


def _valid_linear_tolerances(
    symmetric: Any,
    upper: Any,
    lower: Any,
) -> bool:
    if symmetric is not None:
        return bool(
            _valid_unsigned_decimal(symmetric)
            and upper is None
            and lower is None
        )
    if upper is None or lower is None:
        return upper is None and lower is None
    valid_upper = bool(
        upper == "0"
        or (
            isinstance(upper, str)
            and upper.startswith("+")
            and _valid_unsigned_decimal(upper[1:])
        )
    )
    valid_lower = bool(
        lower == "0"
        or (
            isinstance(lower, str)
            and lower.startswith("-")
            and _valid_unsigned_decimal(lower[1:])
        )
    )
    return bool(
        valid_upper
        and valid_lower
        and not (upper == "0" and lower == "0")
    )


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _valid_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value


def _valid_page(value: Any) -> bool:
    return type(value) is int and value >= 0


def _valid_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _require_identity(trace_id: Any, page_index: Any) -> None:
    if not _valid_text(trace_id):
        raise ValueError("expected_trace_id must be non-empty")
    if not _valid_page(page_index):
        raise ValueError("expected_page_index must be non-negative")
