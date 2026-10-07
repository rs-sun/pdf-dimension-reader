"""Physical sidecar boundary for strict success diagnostics.

Only explicitly classified top-level keys move out of the guarded product
mapping.  The complement is the product by construction, so new keys fail
closed into the existing fresh success-payload guard.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from vector_artifact_safety import vector_artifact_safety_reasons


EIGHT_LINE_HEX_CROSS_PIPELINE_DIAGNOSTIC_KEYS_V1 = frozenset({
    "vector_eight_line_hex_cross_shadow_v1",
    "vector_eight_line_hex_cross_shadow_stats_v1",
})

BOUNDARY_WITNESS_LINEAGE_PIPELINE_DIAGNOSTIC_KEYS_V1 = frozenset({
    "vector_phrase_boundary_witness_lineage_v1",
})


R21_PIPELINE_DIAGNOSTIC_KEYS_V1 = frozenset({
    "r21_l1_calibration_v1",
    "r21_l2_5_gdt_v1",
    "r21_l2_screened_v1",
    "r21_l3_dimensions_v1",
    "r21_l3_suspects_v1",
    "r21_l4_recovered_v1",
    "r21_l5_final_shadow_v1",
    "r21_layered_vector_dimension_stats_v1",
    "r22_pm_strict_stats_v1",
    "r22_pm_strict_v1",
    "r23_gdt_frames_fallback_v1",
    "r23_layout_v1",
    "r23_text_punct_marks_stats_v1",
    "r23_text_punct_marks_v1",
    "r24_l0_baseline_v1",
})


STRICT_PIPELINE_DIAGNOSTIC_KEYS_V1 = frozenset({
    "anchor_corridor_candidates_v1",
    "anchor_corridor_stats_v1",
    "anchor_phrase_candidates_v1",
    "anchor_phrase_stats_v1",
    "annotation_lw",
    "candidate_debug_stats",
    "candidate_drop_ledger",
    "capsule_candidates",
    "connectivity_diag",
    "detected_dim_lines",
    "detected_leader_lines",
    "diameter_glyphs",
    "dimension_candidates",
    "frame_borders",
    "gdt_frames",
    "gdt_frames_all",
    "gdt_frames_from_lines",
    "gdt_symbol_results",
    "lines_result",
    "r21_l1_calibration_v1",
    "r21_l2_5_gdt_v1",
    "r21_l2_screened_v1",
    "r21_l3_dimensions_v1",
    "r21_l3_suspects_v1",
    "r21_l4_recovered_v1",
    "r21_l5_final_shadow_v1",
    "r21_diagnostic_status_v1",
    "r21_layered_vector_dimension_stats_v1",
    "r22_pm_strict_stats_v1",
    "r22_pm_strict_v1",
    "r23_gdt_frames_fallback_v1",
    "r23_layout_v1",
    "r23_text_punct_marks_stats_v1",
    "r23_text_punct_marks_v1",
    "r24_l0_baseline_v1",
    "reconstructed_rects",
    "rects_result",
    "strict_gdt_candidate_audit_v1",
    "text_cluster_extra_points_diag",
    "text_regions",
    "vector_gdt_results",
    "yolo_gdt_results",
}) | (
    EIGHT_LINE_HEX_CROSS_PIPELINE_DIAGNOSTIC_KEYS_V1
    | BOUNDARY_WITNESS_LINEAGE_PIPELINE_DIAGNOSTIC_KEYS_V1
)

STRICT_RESPONSE_DIAGNOSTIC_KEYS_V1 = frozenset({
    "anchor_corridor_candidates_v1",
    "anchor_corridor_stats_v1",
    "anchor_phrase_candidates_v1",
    "anchor_phrase_stats_v1",
    "assembler_debug",
    "candidate_debug_stats",
    "candidate_drop_ledger",
    "dimension_candidates",
    "gdt_frames",
    "page_height",
    "page_width",
    "pdf_type",
    "post_filter_debug_stats",
    "post_filter_drop_ledger",
    "settings_warnings",
    "strict_gdt_candidate_audit_v1",
}) | R21_PIPELINE_DIAGNOSTIC_KEYS_V1 | frozenset({
    "r21_diagnostic_status_v1",
}) | EIGHT_LINE_HEX_CROSS_PIPELINE_DIAGNOSTIC_KEYS_V1


_R21_ROOT_SCHEMAS_V1 = {
    "r21_l1_calibration_v1": "r21_l1_calibration_v1",
    "r21_l2_screened_v1": "r21_l2_screened_v1",
    "r21_l5_final_shadow_v1": "r21_l5_final_shadow_v1",
    "r21_layered_vector_dimension_stats_v1": (
        "r21_layered_vector_dimension_stats_v1"
    ),
    "r22_pm_strict_stats_v1": "r22_pm_strict_v1",
    "r23_layout_v1": "r23_layout_v1",
    "r23_text_punct_marks_stats_v1": "r23_text_punct_marks_v1",
    "r24_l0_baseline_v1": "r24_l0_baseline_v1",
}
_R21_ROOT_ALLOWED_KEYS_V1 = {
    "r24_l0_baseline_v1": frozenset({
        "schema_version",
        "page_index",
        "scale_baseline",
        "font_profile",
        "reference_anchors",
        "core_anchors",
        "quarantined_anchors",
        "layout_healthy",
        "drawing_anchor_ratio",
        "stats",
        "consumer_allowed",
    }),
    "r21_l1_calibration_v1": frozenset({
        "schema_version",
        "page_index",
        "scale_baseline",
        "reference_anchors",
        "colon_marks",
        "colon_filtered",
        "text_punct_marks",
        "layout_healthy",
        "drawing_anchor_ratio",
        "stats",
        "consumer_allowed",
    }),
    "r21_l2_screened_v1": frozenset({
        "schema_version",
        "page_index",
        "glyph_specs",
        "local_glyphs_near_refs",
        "gdt_frames",
        "gdt_fallback_frames",
        "key_capsules",
        "key_capsules_seam",
        "stats",
        "consumer_allowed",
    }),
    "r21_l5_final_shadow_v1": frozenset({
        "schema_version",
        "page_index",
        "high_conf",
        "low_conf",
        "stats",
        "consumer_allowed",
    }),
    "r21_layered_vector_dimension_stats_v1": frozenset({
        "schema_version",
        "page_index",
        "l1_anchor_count",
        "l1_diameter_anchor_count",
        "r24_l0_core_anchor_count",
        "r24_l0_quarantined_anchor_count",
        "r23_text_punct_mark_count",
        "r23_notes_region_count",
        "r23_layout_healthy",
        "r23_drawing_anchor_ratio",
        "l2_local_glyph_count",
        "l2_strict_pitch_dump_status",
        "l2_strict_pitch_input_read_count",
        "l2_strict_pitch_evidence_count",
        "l2_strict_pitch_local_evidence_count",
        "l2_strict_pitch_rejected_read_count",
        "l2_gdt_frame_count",
        "l2_gdt_fallback_frame_count",
        "l2_key_capsule_count",
        "r22_pm_strict_raw_count",
        "l2_pm_strict_count",
        "l3_dimension_count",
        "l3_suspect_count",
        "l4_recovered_count",
        "l5_high_conf_count",
        "l5_low_conf_count",
        "consumer_allowed",
        "consumer_allowed_count",
    }),
    "r22_pm_strict_stats_v1": frozenset({
        "schema_version",
        "page_index",
        "count",
        "counts_by_symbol",
        "consumer_allowed_count",
    }),
    "r23_layout_v1": frozenset({
        "schema_version",
        "page_index",
        "frame_border",
        "drawing_area",
        "title_blocks",
        "tables",
        "notes_regions",
        "anchor_region_tags",
        "healthy",
        "drawing_anchor_ratio",
        "stats",
        "consumer_allowed",
    }),
    "r23_text_punct_marks_stats_v1": frozenset({
        "schema_version",
        "page_index",
        "raw_count",
        "counts_by_kind",
        "counts_by_geometry_candidate",
        "isolation_radius_glyph_h",
        "isolation_rejected_count",
        "consumer_allowed_count",
    }),
}
_R21_PAGE_BOUND_ROOTS_V1 = frozenset({
    "r21_l1_calibration_v1",
    "r21_l2_screened_v1",
    "r21_l5_final_shadow_v1",
    "r21_layered_vector_dimension_stats_v1",
    "r22_pm_strict_stats_v1",
    "r23_layout_v1",
    "r23_text_punct_marks_stats_v1",
    "r24_l0_baseline_v1",
})
_R21_LIST_ROOTS_V1 = R21_PIPELINE_DIAGNOSTIC_KEYS_V1 - frozenset(
    _R21_ROOT_SCHEMAS_V1
)
_R21_LIST_ROW_SCHEMAS_V1 = {
    "r21_l2_5_gdt_v1": "r21_l2_5_gdt_v1",
    "r21_l3_dimensions_v1": "r21_l3_assembly_v1",
    "r21_l3_suspects_v1": "r21_l3_assembly_v1",
    "r21_l4_recovered_v1": "r21_l4_recovery_v1",
    "r22_pm_strict_v1": "r22_pm_strict_v1",
    "r23_gdt_frames_fallback_v1": "r21_l2_screened_v1",
    "r23_text_punct_marks_v1": "r23_text_punct_marks_v1",
}
_R21_ROOT_REQUIRED_CONTAINERS_V1 = {
    "r24_l0_baseline_v1": {
        "scale_baseline": dict,
        "font_profile": dict,
        "reference_anchors": list,
        "core_anchors": list,
        "quarantined_anchors": list,
        "stats": dict,
        "consumer_allowed": bool,
    },
    "r21_l1_calibration_v1": {
        "scale_baseline": dict,
        "reference_anchors": list,
        "colon_marks": list,
        "colon_filtered": list,
        "text_punct_marks": list,
        "stats": dict,
        "consumer_allowed": bool,
    },
    "r23_layout_v1": {
        "drawing_area": (dict, type(None)),
        "title_blocks": list,
        "tables": list,
        "notes_regions": list,
        "anchor_region_tags": dict,
        "stats": dict,
        "consumer_allowed": bool,
    },
    "r21_l2_screened_v1": {
        "glyph_specs": dict,
        "local_glyphs_near_refs": list,
        "gdt_frames": list,
        "gdt_fallback_frames": list,
        "key_capsules": list,
        "key_capsules_seam": dict,
        "stats": dict,
        "consumer_allowed": bool,
    },
    "r21_l5_final_shadow_v1": {
        "high_conf": list,
        "low_conf": list,
        "stats": dict,
        "consumer_allowed": bool,
    },
    "r21_layered_vector_dimension_stats_v1": {
        "consumer_allowed": bool,
        "consumer_allowed_count": int,
    },
    "r22_pm_strict_stats_v1": {
        "count": int,
        "counts_by_symbol": dict,
        "consumer_allowed_count": int,
    },
    "r23_text_punct_marks_stats_v1": {
        "raw_count": int,
        "counts_by_kind": dict,
        "consumer_allowed_count": int,
    },
}
_R21_LIST_ROW_REQUIRED_CONTAINERS_V1 = {
    "r21_l2_5_gdt_v1": {
        "frame_id": str,
        "frame": dict,
        "symbols": list,
        "datums": list,
        "numeric_glyphs": list,
    },
    "r21_l3_dimensions_v1": {
        "dimension_id": str,
        "anchor_id": str,
        "value_text": str,
        "normalized_value_text": str,
        "oriented_bbox": dict,
        "flags": list,
        "source_glyph_ids": list,
    },
    "r21_l3_suspects_v1": {
        "dimension_id": str,
        "anchor_id": str,
        "value_text": str,
        "normalized_value_text": str,
        "oriented_bbox": dict,
        "flags": list,
        "source_glyph_ids": list,
    },
    "r21_l4_recovered_v1": {
        "dimension_id": str,
        "recovered_from_suspect_id": str,
        "recovery_kind": str,
        "bbox": dict,
        "flags": list,
    },
    "r22_pm_strict_v1": {
        "id": str,
        "anchor_id": str,
        "symbol": str,
        "bbox": dict,
    },
    "r23_gdt_frames_fallback_v1": {
        "frame_id": str,
        "source_id": str,
        "bbox": dict,
        "compartments": list,
        "fallback_reason": str,
    },
    "r23_text_punct_marks_v1": {
        "mark_id": str,
        "kind": str,
        "bbox": dict,
    },
}
_R21_LIST_ROW_REQUIRED_NONEMPTY_STRINGS_V1 = {
    "r21_l2_5_gdt_v1": ("frame_id",),
    "r21_l3_dimensions_v1": ("dimension_id", "anchor_id", "value_text"),
    "r21_l3_suspects_v1": ("dimension_id", "anchor_id", "value_text"),
    "r21_l4_recovered_v1": (
        "dimension_id",
        "recovered_from_suspect_id",
        "recovery_kind",
    ),
    "r22_pm_strict_v1": ("id", "anchor_id", "symbol"),
    "r23_gdt_frames_fallback_v1": (
        "frame_id",
        "source_id",
        "fallback_reason",
    ),
    "r23_text_punct_marks_v1": ("mark_id", "kind"),
}
_R21_LIST_ROW_REQUIRED_NONEMPTY_MAPPINGS_V1 = {
    "r21_l2_5_gdt_v1": ("frame",),
    "r21_l3_dimensions_v1": ("oriented_bbox",),
    "r21_l3_suspects_v1": ("oriented_bbox",),
    "r21_l4_recovered_v1": ("bbox",),
    "r22_pm_strict_v1": ("bbox",),
    "r23_gdt_frames_fallback_v1": ("bbox",),
    "r23_text_punct_marks_v1": ("bbox",),
}
_R21_LIST_ROW_REQUIRED_NONEMPTY_STRING_LISTS_V1 = {
    "r21_l3_dimensions_v1": ("source_glyph_ids",),
    "r21_l3_suspects_v1": ("source_glyph_ids",),
}
_R21_REQUIRED_ZERO_COUNT_PATHS_V1 = {
    "r24_l0_baseline_v1": (("stats", "consumer_allowed_count"),),
    "r21_l1_calibration_v1": (("stats", "consumer_allowed_count"),),
    "r21_l2_screened_v1": (("stats", "consumer_allowed_count"),),
    "r21_layered_vector_dimension_stats_v1": (
        ("consumer_allowed_count",),
    ),
    "r22_pm_strict_stats_v1": (("consumer_allowed_count",),),
    "r23_layout_v1": (("stats", "consumer_allowed_count"),),
    "r23_text_punct_marks_stats_v1": (
        ("consumer_allowed_count",),
    ),
}
_R21_NESTED_ROW_RULES_V1 = (
    {
        "root": "r24_l0_baseline_v1",
        "field": "reference_anchors",
        "schema": "r21_l1_calibration_v1",
        "strings": ("anchor_id", "kind", "source", "source_id"),
        "mappings": ("bbox", "center"),
    },
    {
        "root": "r24_l0_baseline_v1",
        "field": "core_anchors",
        "schema": "r21_l1_calibration_v1",
        "strings": ("anchor_id", "kind", "source", "source_id"),
        "mappings": ("bbox", "center"),
    },
    {
        "root": "r24_l0_baseline_v1",
        "field": "quarantined_anchors",
        "schema": "r21_l1_calibration_v1",
        "strings": ("anchor_id", "kind", "source", "source_id"),
        "mappings": ("bbox", "center"),
    },
    {
        "root": "r21_l1_calibration_v1",
        "field": "reference_anchors",
        "schema": "r21_l1_calibration_v1",
        "strings": ("anchor_id", "kind", "source", "source_id"),
        "mappings": ("bbox", "center"),
    },
    {
        "root": "r21_l1_calibration_v1",
        "field": "colon_marks",
        "schema": "r21_l1_calibration_v1",
        "strings": ("colon_id", "pair_id", "kind", "source"),
        "mappings": ("bbox", "center"),
        "string_lists": ("point_source_ids",),
    },
    {
        "root": "r21_l1_calibration_v1",
        "field": "colon_filtered",
        "schema": None,
        "strings": ("dot_id", "reason"),
        "mappings": ("bbox", "center"),
    },
    {
        "root": "r21_l1_calibration_v1",
        "field": "text_punct_marks",
        "schema": "r23_text_punct_marks_v1",
        "strings": ("mark_id", "kind"),
        "mappings": ("bbox",),
        "true_flags": ("is_text_punct",),
    },
    {
        "root": "r21_l2_screened_v1",
        "field": "local_glyphs_near_refs",
        "schema": "r21_l2_screened_v1",
        "strings": (
            "source_id",
            "source",
            "text",
            "anchor_id",
            "anchor_kind",
        ),
        "mappings": ("bbox", "center"),
    },
    {
        "root": "r21_l2_screened_v1",
        "field": "gdt_frames",
        "schema": "r21_l2_screened_v1",
        "strings": ("frame_id", "source"),
        "mappings": ("bbox",),
    },
    {
        "root": "r21_l2_screened_v1",
        "field": "gdt_fallback_frames",
        "schema": "r21_l2_screened_v1",
        "strings": ("frame_id", "source", "fallback_reason"),
        "mappings": ("bbox",),
    },
    {
        "root": "r23_layout_v1",
        "field": "title_blocks",
        "schema": None,
        "strings": ("region_kind", "source"),
        "mappings": ("bbox",),
    },
    {
        "root": "r23_layout_v1",
        "field": "tables",
        "schema": None,
        "strings": ("region_kind", "source"),
        "mappings": ("bbox",),
    },
    {
        "root": "r23_layout_v1",
        "field": "notes_regions",
        "schema": None,
        "strings": ("region_id", "source"),
        "mappings": ("bbox",),
    },
)
_R21_EXPECTED_INPUT_STAT_PATHS_V1 = {
    "decimal_point_row_count": (
        "r21_l1_calibration_v1",
        "stats",
        "decimal_input_count",
    ),
    "degree_row_count": (
        "r21_l1_calibration_v1",
        "stats",
        "degree_input_count",
    ),
    "diameter_glyph_row_count": (
        "r21_l1_calibration_v1",
        "stats",
        "diameter_input_count",
    ),
    "vector_glyph_token_count": (
        "r21_l2_screened_v1",
        "stats",
        "vector_glyph_input_count",
    ),
    "digit_match_row_count": (
        "r21_l2_screened_v1",
        "stats",
        "digit_match_input_count",
    ),
    "gdt_line_frame_count": (
        "r21_l2_screened_v1",
        "stats",
        "gdt_line_frame_input_count",
    ),
    "capsule_row_count": (
        "r21_l2_screened_v1",
        "stats",
        "key_capsule_input_count",
    ),
    "strict_pitch_read_count": (
        "r21_l2_screened_v1",
        "stats",
        "strict_pitch_input_read_count",
    ),
}
_R21_FORBIDDEN_PRODUCT_ARTIFACT_KEYS_V1 = frozenset({
    "dimensions",
    "formalstamps",
    "reviewcandidatesv1",
    "reviewenvelope",
    "reviewedvector",
    "stamps",
})


def validate_r21_pipeline_diagnostics_v1(
    fields: Mapping[str, Any],
    *,
    page_index: int,
    expected_input_counts: Mapping[str, int] | None = None,
) -> str | None:
    """Return a non-sensitive status when an R21 sidecar is not trustworthy."""
    if not isinstance(fields, Mapping):
        return "unsafe_fields"
    observed_keys = set(fields)
    if observed_keys - R21_PIPELINE_DIAGNOSTIC_KEYS_V1:
        return "unexpected_fields"
    if R21_PIPELINE_DIAGNOSTIC_KEYS_V1 - observed_keys:
        return "incomplete_fields"
    if (
        expected_input_counts is not None
        and not _r21_input_census_matches_v1(
            fields,
            expected_input_counts,
        )
    ):
        return "unsafe_fields"
    try:
        json.dumps(fields, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return "unsafe_fields"

    for key, expected_schema in _R21_ROOT_SCHEMAS_V1.items():
        value = fields.get(key)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != expected_schema
            or set(value) - _R21_ROOT_ALLOWED_KEYS_V1[key]
        ):
            return "unsafe_fields"
        if key in _R21_PAGE_BOUND_ROOTS_V1 and (
            type(value.get("page_index")) is not int
            or value.get("page_index") != int(page_index)
        ):
            return "unsafe_fields"
        if not _r21_has_required_containers_v1(
            value,
            _R21_ROOT_REQUIRED_CONTAINERS_V1.get(key, {}),
        ):
            return "unsafe_fields"
        if not _r21_has_required_zero_counts_v1(
            value,
            _R21_REQUIRED_ZERO_COUNT_PATHS_V1.get(key, ()),
        ):
            return "unsafe_fields"
    for key in _R21_LIST_ROOTS_V1:
        rows = fields.get(key)
        if not isinstance(rows, list):
            return "unsafe_fields"
        expected_schema = _R21_LIST_ROW_SCHEMAS_V1[key]
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != expected_schema
                or type(row.get("page_index")) is not int
                or row.get("page_index") != int(page_index)
                or row.get("consumer_allowed") is not False
                or not _r21_has_required_containers_v1(
                    row,
                    _R21_LIST_ROW_REQUIRED_CONTAINERS_V1[key],
                )
                or not _r21_has_meaningful_row_identity_v1(row, key=key)
                or (
                    key == "r23_text_punct_marks_v1"
                    and row.get("is_text_punct") is not True
                )
            ):
                return "unsafe_fields"
    if not _r21_nested_evidence_is_safe_v1(fields):
        return "unsafe_fields"
    if not _r21_mirrored_evidence_is_coherent_v1(fields):
        return "unsafe_fields"
    if not _r21_cross_layer_references_are_coherent_v1(fields):
        return "unsafe_fields"
    if vector_artifact_safety_reasons(fields, path="r21_diagnostics"):
        return "unsafe_fields"
    if _r21_recursive_authority_is_unsafe_v1(
        fields,
        page_index=int(page_index),
    ):
        return "unsafe_fields"
    if not _r21_counts_are_coherent_v1(fields):
        return "unsafe_fields"
    if not _r21_l5_partition_is_coherent_v1(fields):
        return "unsafe_fields"
    return None


def _r21_input_census_matches_v1(
    fields: Mapping[str, Any],
    expected: Mapping[str, int],
) -> bool:
    if (
        not isinstance(expected, Mapping)
        or set(expected) != set(_R21_EXPECTED_INPUT_STAT_PATHS_V1)
    ):
        return False
    for name, path in _R21_EXPECTED_INPUT_STAT_PATHS_V1.items():
        expected_count = expected.get(name)
        if type(expected_count) is not int or expected_count < 0:
            return False
        current: Any = fields
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                return False
            current = current[key]
        if type(current) is not int or current != expected_count:
            return False
    return True


def _r21_has_required_containers_v1(
    value: Mapping[str, Any],
    required: Mapping[str, type | tuple[type, ...]],
) -> bool:
    return all(
        key in value and isinstance(value[key], expected_type)
        for key, expected_type in required.items()
    )


def _r21_has_required_zero_counts_v1(
    value: Mapping[str, Any],
    paths: tuple[tuple[str, ...], ...],
) -> bool:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                return False
            current = current[key]
        if type(current) is not int or current != 0:
            return False
    return True


def _r21_has_meaningful_row_identity_v1(
    row: Mapping[str, Any],
    *,
    key: str,
) -> bool:
    if any(
        not isinstance(row.get(field), str)
        or not row[field].strip()
        for field in _R21_LIST_ROW_REQUIRED_NONEMPTY_STRINGS_V1[key]
    ):
        return False
    if any(
        not isinstance(row.get(field), Mapping)
        or not row[field]
        or not _r21_list_row_mapping_geometry_is_valid_v1(
            key,
            field,
            row[field],
            row=row,
        )
        for field in _R21_LIST_ROW_REQUIRED_NONEMPTY_MAPPINGS_V1[key]
    ):
        return False
    return all(
        isinstance(row.get(field), list)
        and bool(row[field])
        and all(
            isinstance(value, str) and bool(value.strip())
            for value in row[field]
        )
        for field in _R21_LIST_ROW_REQUIRED_NONEMPTY_STRING_LISTS_V1.get(
            key,
            (),
        )
    )


def _r21_list_row_mapping_geometry_is_valid_v1(
    key: str,
    field: str,
    value: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> bool:
    if (
        key == "r22_pm_strict_v1"
        and field == "bbox"
        and row.get("symbol") == "minus"
    ):
        return _r21_bbox_is_valid_v1(
            value,
            allow_single_axis_degenerate=True,
        )
    return _r21_mapping_geometry_is_valid_v1(field, value)


def _r21_nested_evidence_is_safe_v1(
    fields: Mapping[str, Any],
) -> bool:
    for rule in _R21_NESTED_ROW_RULES_V1:
        rows = fields[rule["root"]][rule["field"]]
        if not isinstance(rows, list):
            return False
        if any(
            not _r21_nested_row_is_meaningful_v1(row, rule=rule)
            for row in rows
        ):
            return False

    l2 = fields["r21_l2_screened_v1"]
    seam = l2.get("key_capsules_seam")
    if (
        l2.get("key_capsules") != []
        or not isinstance(seam, Mapping)
        or seam.get("status")
        != "removed_pending_candidate_dimension_to_outer_frame_inversion"
        or _r21_nonnegative_int_v1(
            seam.get("raw_capsule_input_count")
        )
        is None
        or seam.get("consumer_allowed") is not False
    ):
        return False

    layout = fields["r23_layout_v1"]
    drawing_area = layout.get("drawing_area")
    if drawing_area is not None and (
        not isinstance(drawing_area, Mapping)
        or not drawing_area
        or not _r21_bbox_is_valid_v1(drawing_area)
    ):
        return False
    tags = layout.get("anchor_region_tags")
    if not isinstance(tags, Mapping):
        return False
    for anchor_id, row in tags.items():
        if (
            not isinstance(anchor_id, str)
            or not anchor_id.strip()
            or not _r21_nested_row_is_meaningful_v1(
                row,
                rule={
                    "schema": None,
                    "strings": ("region",),
                    "mappings": ("center",),
                },
            )
        ):
            return False
    return True


def _r21_nested_row_is_meaningful_v1(
    row: Any,
    *,
    rule: Mapping[str, Any],
) -> bool:
    if not isinstance(row, Mapping) or row.get("consumer_allowed") is not False:
        return False
    expected_schema = rule.get("schema")
    if (
        expected_schema is not None
        and row.get("schema_version") != expected_schema
    ):
        return False
    if any(
        not isinstance(row.get(field), str)
        or not row[field].strip()
        for field in rule.get("strings", ())
    ):
        return False
    if any(
        not isinstance(row.get(field), Mapping)
        or not row[field]
        or not _r21_mapping_geometry_is_valid_v1(
            field,
            row[field],
        )
        for field in rule.get("mappings", ())
    ):
        return False
    if any(
        not isinstance(row.get(field), list)
        or not row[field]
        or any(
            not isinstance(value, str) or not value.strip()
            for value in row[field]
        )
        for field in rule.get("string_lists", ())
    ):
        return False
    return all(
        row.get(field) is True
        for field in rule.get("true_flags", ())
    )


def _r21_mapping_geometry_is_valid_v1(
    field: str,
    value: Mapping[str, Any],
) -> bool:
    if field == "bbox":
        return _r21_bbox_is_valid_v1(value)
    if field == "center":
        return _r21_point_is_valid_v1(value)
    if field in {"frame", "oriented_bbox"}:
        return (
            isinstance(value.get("bbox"), Mapping)
            and _r21_bbox_is_valid_v1(value["bbox"])
        )
    return True


def _r21_bbox_is_valid_v1(
    value: Mapping[str, Any],
    *,
    allow_single_axis_degenerate: bool = False,
) -> bool:
    raw = [value.get(key) for key in ("x", "y", "w", "h")]
    if any(type(number) not in {int, float} for number in raw):
        return False
    try:
        x, y, w, h = (float(number) for number in raw)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        all(math.isfinite(number) for number in (x, y, w, h))
        and (
            (w >= 0.0 and h >= 0.0 and (w > 0.0 or h > 0.0))
            if allow_single_axis_degenerate
            else (w > 0.0 and h > 0.0)
        )
    )


def _r21_point_is_valid_v1(value: Mapping[str, Any]) -> bool:
    raw = [value.get(key) for key in ("x", "y")]
    if any(type(number) not in {int, float} for number in raw):
        return False
    try:
        x, y = (float(number) for number in raw)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(x) and math.isfinite(y)


def _r21_mirrored_evidence_is_coherent_v1(
    fields: Mapping[str, Any],
) -> bool:
    mirrors = (
        (
            fields["r21_l1_calibration_v1"]["text_punct_marks"],
            fields["r23_text_punct_marks_v1"],
        ),
        (
            fields["r21_l2_screened_v1"]["gdt_fallback_frames"],
            fields["r23_gdt_frames_fallback_v1"],
        ),
    )

    def canonical(rows: list[Any]) -> list[str]:
        return sorted(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for row in rows
        )

    return all(canonical(left) == canonical(right) for left, right in mirrors)


def _r21_cross_layer_references_are_coherent_v1(
    fields: Mapping[str, Any],
) -> bool:
    l0 = fields["r24_l0_baseline_v1"]
    l1 = fields["r21_l1_calibration_v1"]
    l2 = fields["r21_l2_screened_v1"]
    l2_5 = fields["r21_l2_5_gdt_v1"]
    l3 = fields["r21_l3_dimensions_v1"]
    suspects = fields["r21_l3_suspects_v1"]
    l4 = fields["r21_l4_recovered_v1"]
    layout = fields["r23_layout_v1"]

    if _r21_canonical_rows_v1(l0["reference_anchors"]) != (
        _r21_canonical_rows_v1(l1["reference_anchors"])
    ):
        return False
    anchor_ids = [row["anchor_id"] for row in l1["reference_anchors"]]
    if len(set(anchor_ids)) != len(anchor_ids):
        return False
    anchor_id_set = set(anchor_ids)
    anchor_by_id = {
        row["anchor_id"]: row for row in l1["reference_anchors"]
    }
    if any(
        row["anchor_id"] not in anchor_id_set
        for row in l2["local_glyphs_near_refs"]
    ):
        return False
    subset_id_sets: list[set[str]] = []
    for subset_key in ("core_anchors", "quarantined_anchors"):
        subset_ids = [row["anchor_id"] for row in l0[subset_key]]
        if (
            len(set(subset_ids)) != len(subset_ids)
            or not set(subset_ids) <= anchor_id_set
            or any(
                _r21_canonical_row_v1(row)
                != _r21_canonical_row_v1(
                    anchor_by_id[row["anchor_id"]]
                )
                for row in l0[subset_key]
            )
        ):
            return False
        subset_id_sets.append(set(subset_ids))
    if subset_id_sets[0] & subset_id_sets[1]:
        return False
    baseline_anchor_ids = {
        row["anchor_id"]
        for row in l1["reference_anchors"]
        if row["kind"] in {"dot", "degree"}
    }
    if subset_id_sets[0] | subset_id_sets[1] != baseline_anchor_ids:
        return False
    known_layout_tag_ids = {
        row[identity_key]
        for rows, identity_key in (
            (l1["reference_anchors"], "anchor_id"),
            (l1["colon_marks"], "colon_id"),
            (l1["text_punct_marks"], "mark_id"),
        )
        for row in rows
    }
    if not set(layout["anchor_region_tags"]) <= known_layout_tag_ids:
        return False
    if any(
        anchor.get("region")
        != layout["anchor_region_tags"][anchor["anchor_id"]].get("region")
        for anchor in l1["reference_anchors"]
        if anchor["anchor_id"] in layout["anchor_region_tags"]
    ):
        return False
    pm_rows = fields["r22_pm_strict_v1"]
    pm_ids = [row["id"] for row in pm_rows]
    if (
        len(set(pm_ids)) != len(pm_ids)
        or any(
            row["anchor_id"] not in anchor_id_set
            or anchor_by_id[row["anchor_id"]].get("kind") != "dot"
            for row in pm_rows
        )
    ):
        return False

    glyph_ids = [row["source_id"] for row in l2["local_glyphs_near_refs"]]
    if len(set(glyph_ids)) != len(glyph_ids):
        return False
    glyph_id_set = set(glyph_ids)
    dimension_ids = [row["dimension_id"] for row in l3]
    if len(set(dimension_ids)) != len(dimension_ids):
        return False
    dimension_id_set = set(dimension_ids)
    for row in [*l3, *suspects]:
        if (
            row["anchor_id"] not in anchor_id_set
            or not set(row["source_glyph_ids"]) <= glyph_id_set
        ):
            return False
    suspect_ids = [row["dimension_id"] for row in suspects]
    if (
        len(set(suspect_ids)) != len(suspect_ids)
        or not set(suspect_ids) <= dimension_id_set
    ):
        return False
    l3_by_id = {row["dimension_id"]: row for row in l3}
    if any(
        _r21_canonical_row_v1(row)
        != _r21_canonical_row_v1(l3_by_id[row["dimension_id"]])
        for row in suspects
    ):
        return False

    gdt_by_id = {
        row["frame_id"]: row for row in l2["gdt_frames"]
    }
    if len(gdt_by_id) != len(l2["gdt_frames"]):
        return False
    if any(
        row["frame_id"] not in gdt_by_id
        or _r21_canonical_row_v1(row["frame"])
        != _r21_canonical_row_v1(gdt_by_id[row["frame_id"]])
        for row in l2_5
    ):
        return False

    l4_ids = [row["dimension_id"] for row in l4]
    if (
        len(set(l4_ids)) != len(l4_ids)
        or any(
            row["recovered_from_suspect_id"] not in set(suspect_ids)
            for row in l4
        )
    ):
        return False
    return True


def _r21_canonical_rows_v1(rows: list[Any]) -> list[str]:
    return sorted(_r21_canonical_row_v1(row) for row in rows)


def _r21_canonical_row_v1(row: Any) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _r21_nonnegative_int_v1(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _r21_counts_are_coherent_v1(fields: Mapping[str, Any]) -> bool:
    l0 = fields["r24_l0_baseline_v1"]
    l1 = fields["r21_l1_calibration_v1"]
    l2 = fields["r21_l2_screened_v1"]
    l5 = fields["r21_l5_final_shadow_v1"]
    global_stats = fields["r21_layered_vector_dimension_stats_v1"]
    pm_stats = fields["r22_pm_strict_stats_v1"]
    layout = fields["r23_layout_v1"]
    punct_stats = fields["r23_text_punct_marks_stats_v1"]
    l0_stats = l0["stats"]
    l1_stats = l1["stats"]
    l2_stats = l2["stats"]
    l5_stats = l5["stats"]
    layout_stats = layout["stats"]
    required_counts = {
        "l0_core": (
            l0_stats.get("core_anchor_count"),
            len(l0["core_anchors"]),
        ),
        "l0_quarantined": (
            l0_stats.get("quarantined_anchor_count"),
            len(l0["quarantined_anchors"]),
        ),
        "l1_colon_marks": (
            l1_stats.get("colon_mark_count"),
            len(l1["colon_marks"]),
        ),
        "l1_colon_filtered": (
            l1_stats.get("colon_dropped_count"),
            sum(
                isinstance(row, Mapping) and row.get("keep") is not True
                for row in l1["colon_filtered"]
            ),
        ),
        "l1_text_punct": (
            l1_stats.get("text_punct_mark_count"),
            len(l1["text_punct_marks"]),
        ),
        "l2_local": (
            l2_stats.get("local_glyph_count"),
            len(l2["local_glyphs_near_refs"]),
        ),
        "l2_gdt": (
            l2_stats.get("gdt_frame_count"),
            len(l2["gdt_frames"]),
        ),
        "l2_gdt_fallback": (
            l2_stats.get("gdt_fallback_frame_count"),
            len(l2["gdt_fallback_frames"]),
        ),
        "l2_key_capsules": (
            l2_stats.get("key_capsule_count"),
            len(l2["key_capsules"]),
        ),
        "pm": (
            pm_stats.get("count"),
            len(fields["r22_pm_strict_v1"]),
        ),
        "punct": (
            punct_stats.get("raw_count"),
            len(fields["r23_text_punct_marks_v1"]),
        ),
        "layout_title_blocks": (
            layout_stats.get("title_block_count"),
            len(layout["title_blocks"]),
        ),
        "layout_tables": (
            layout_stats.get("table_count"),
            len(layout["tables"]),
        ),
        "layout_notes": (
            layout_stats.get("notes_region_count"),
            len(layout["notes_regions"]),
        ),
        "layout_tags": (
            layout_stats.get("tagged_anchor_count"),
            len(layout["anchor_region_tags"]),
        ),
        "l5_input_l3": (
            l5_stats.get("input_l3_count"),
            len(fields["r21_l3_dimensions_v1"]),
        ),
        "l5_input_l4": (
            l5_stats.get("input_l4_count"),
            len(fields["r21_l4_recovered_v1"]),
        ),
        "l5_high": (
            l5_stats.get("high_conf_count"),
            len(l5["high_conf"]),
        ),
        "l5_low": (
            l5_stats.get("low_conf_count"),
            len(l5["low_conf"]),
        ),
        "global_l1": (
            global_stats.get("l1_anchor_count"),
            len(l1["reference_anchors"]),
        ),
        "global_l1_diameter": (
            global_stats.get("l1_diameter_anchor_count"),
            sum(
                isinstance(row, Mapping)
                and row.get("kind") == "diameter"
                for row in l1["reference_anchors"]
            ),
        ),
        "global_l0_core": (
            global_stats.get("r24_l0_core_anchor_count"),
            len(l0["core_anchors"]),
        ),
        "global_l0_quarantined": (
            global_stats.get("r24_l0_quarantined_anchor_count"),
            len(l0["quarantined_anchors"]),
        ),
        "global_text_punct": (
            global_stats.get("r23_text_punct_mark_count"),
            len(fields["r23_text_punct_marks_v1"]),
        ),
        "global_notes": (
            global_stats.get("r23_notes_region_count"),
            len(layout["notes_regions"]),
        ),
        "global_l2_local": (
            global_stats.get("l2_local_glyph_count"),
            len(l2["local_glyphs_near_refs"]),
        ),
        "global_l2_gdt": (
            global_stats.get("l2_gdt_frame_count"),
            len(l2["gdt_frames"]),
        ),
        "global_l2_gdt_fallback": (
            global_stats.get("l2_gdt_fallback_frame_count"),
            len(l2["gdt_fallback_frames"]),
        ),
        "global_l2_key": (
            global_stats.get("l2_key_capsule_count"),
            len(l2["key_capsules"]),
        ),
        "global_l2_pm": (
            global_stats.get("l2_pm_strict_count"),
            sum(
                isinstance(row, Mapping)
                and row.get("source") == "pm_strict"
                for row in l2["local_glyphs_near_refs"]
            ),
        ),
        "global_pm_raw": (
            global_stats.get("r22_pm_strict_raw_count"),
            len(fields["r22_pm_strict_v1"]),
        ),
        "global_l3": (
            global_stats.get("l3_dimension_count"),
            len(fields["r21_l3_dimensions_v1"]),
        ),
        "global_l3_suspects": (
            global_stats.get("l3_suspect_count"),
            len(fields["r21_l3_suspects_v1"]),
        ),
        "global_l4": (
            global_stats.get("l4_recovered_count"),
            len(fields["r21_l4_recovered_v1"]),
        ),
        "global_l5_high": (
            global_stats.get("l5_high_conf_count"),
            len(l5["high_conf"]),
        ),
        "global_l5_low": (
            global_stats.get("l5_low_conf_count"),
            len(l5["low_conf"]),
        ),
    }
    if not all(
        _r21_nonnegative_int_v1(observed) == expected
        for observed, expected in required_counts.values()
    ):
        return False

    strict_status = l2_stats.get("strict_pitch_dump_status")
    if (
        strict_status not in {"not_provided", "ok", "fail_closed"}
        or global_stats.get("l2_strict_pitch_dump_status") != strict_status
    ):
        return False
    strict_count_fields = (
        ("strict_pitch_input_read_count", "l2_strict_pitch_input_read_count"),
        ("strict_pitch_evidence_count", "l2_strict_pitch_evidence_count"),
        (
            "strict_pitch_local_evidence_count",
            "l2_strict_pitch_local_evidence_count",
        ),
        (
            "strict_pitch_rejected_read_count",
            "l2_strict_pitch_rejected_read_count",
        ),
    )
    strict_counts: dict[str, int] = {}
    for l2_key, global_key in strict_count_fields:
        l2_count = _r21_nonnegative_int_v1(l2_stats.get(l2_key))
        global_count = _r21_nonnegative_int_v1(
            global_stats.get(global_key)
        )
        if l2_count is None or l2_count != global_count:
            return False
        strict_counts[l2_key] = l2_count
    if (
        strict_counts["strict_pitch_input_read_count"]
        != strict_counts["strict_pitch_evidence_count"]
        + strict_counts["strict_pitch_rejected_read_count"]
        or strict_counts["strict_pitch_local_evidence_count"]
        > strict_counts["strict_pitch_evidence_count"]
        or strict_counts["strict_pitch_local_evidence_count"]
        != sum(
            isinstance(row, Mapping)
            and row.get("source") == "r33_m1_pitch_reader"
            for row in l2["local_glyphs_near_refs"]
        )
        or (
            strict_status == "not_provided"
            and any(strict_counts.values())
        )
        or (
            strict_status == "fail_closed"
            and (
                strict_counts["strict_pitch_evidence_count"] != 0
                or strict_counts["strict_pitch_local_evidence_count"] != 0
            )
        )
    ):
        return False
    rejection_counts = l2_stats.get("strict_pitch_rejection_counts")
    if (
        not isinstance(rejection_counts, Mapping)
        or any(
            not isinstance(reason, str)
            or not reason.strip()
            or type(count) is not int
            or count <= 0
            for reason, count in rejection_counts.items()
        )
        or (strict_status == "not_provided" and bool(rejection_counts))
    ):
        return False

    pm_counts = Counter(
        row["symbol"] for row in fields["r22_pm_strict_v1"]
    )
    punct_counts = Counter(
        row["kind"] for row in fields["r23_text_punct_marks_v1"]
    )
    return (
        dict(sorted(pm_counts.items())) == pm_stats.get("counts_by_symbol")
        and dict(sorted(punct_counts.items()))
        == punct_stats.get("counts_by_kind")
    )


def _r21_l5_partition_is_coherent_v1(fields: Mapping[str, Any]) -> bool:
    l5 = fields["r21_l5_final_shadow_v1"]
    source_rows = [
        *fields["r21_l3_dimensions_v1"],
        *fields["r21_l4_recovered_v1"],
    ]
    routed_rows = [*l5["high_conf"], *l5["low_conf"]]
    if not all(isinstance(row, Mapping) for row in routed_rows):
        return False

    def canonical(row: Mapping[str, Any]) -> str:
        return json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    return sorted(map(canonical, source_rows)) == sorted(
        map(canonical, routed_rows)
    )


def _r21_recursive_authority_is_unsafe_v1(
    value: Any,
    *,
    page_index: int,
) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                _r21_compact_identifier_v1(key)
                in _R21_FORBIDDEN_PRODUCT_ARTIFACT_KEYS_V1
            ):
                return True
            if key == "consumer_allowed" and child is not False:
                return True
            if key == "consumer_allowed_count" and (
                type(child) is not int or child != 0
            ):
                return True
            if key == "ocr_called" and child is not False:
                return True
            if key == "page_index" and (
                type(child) is not int or child != page_index
            ):
                return True
            if "datum" in key.lower() and child not in (
                None,
                "",
                [],
            ):
                return True
            if _r21_recursive_authority_is_unsafe_v1(
                child,
                page_index=page_index,
            ):
                return True
        return False
    if isinstance(value, list):
        return any(
            _r21_recursive_authority_is_unsafe_v1(
                child,
                page_index=page_index,
            )
            for child in value
        )
    return False


def _r21_compact_identifier_v1(value: str) -> str:
    return "".join(
        character
        for character in value.lower()
        if (
            "a" <= character <= "z"
            or "0" <= character <= "9"
        )
    )


@dataclass
class StrictDiagnosticCaptureV1:
    """Request/task-local observation target for non-HTTP diagnostics."""

    pipeline: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    producer_pipeline: dict[str, Any] = field(default_factory=dict)


_ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1: ContextVar[
    StrictDiagnosticCaptureV1 | None
] = ContextVar(
    "active_strict_diagnostic_capture_v1",
    default=None,
)


def begin_strict_diagnostic_request_v1() -> None:
    """Clear an active capture before one ``/analyze`` request starts."""
    capture = _ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1.get()
    if capture is not None:
        capture.pipeline.clear()
        capture.response.clear()
        capture.producer_pipeline.clear()


def strict_diagnostic_capture_active_v1() -> bool:
    """Return whether the current request/task explicitly captures sidecars."""
    return _ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1.get() is not None


@contextmanager
def capture_strict_diagnostics_v1() -> Iterator[StrictDiagnosticCaptureV1]:
    """Capture strict sidecars in the current request/task until context exit."""
    capture = StrictDiagnosticCaptureV1()
    token = _ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1.reset(token)


def publish_strict_pipeline_diagnostics_v1(
    diagnostics: Mapping[str, Any],
) -> None:
    """Publish pipeline diagnostics to an active local capture, if present."""
    capture = _ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1.get()
    if capture is not None:
        duplicate_keys = set(diagnostics).intersection(
            capture.producer_pipeline
        )
        if duplicate_keys:
            names = ",".join(sorted(duplicate_keys))
            raise ValueError(
                f"duplicate producer pipeline diagnostics: {names}"
            )
        capture.pipeline.clear()
        capture.pipeline.update(diagnostics)
        capture.pipeline.update(capture.producer_pipeline)


def publish_vector_phrase_boundary_witness_lineage_v1(
    diagnostic: Mapping[str, Any],
) -> None:
    """Publish producer-only boundary lineage into an active local sidecar."""
    capture = _ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1.get()
    if capture is None:
        return
    key = "vector_phrase_boundary_witness_lineage_v1"
    if key in capture.producer_pipeline:
        raise ValueError("boundary witness lineage already published")
    capture.producer_pipeline[key] = dict(diagnostic)


def publish_strict_response_diagnostics_v1(
    diagnostics: Mapping[str, Any],
) -> None:
    """Publish response diagnostics to an active local capture, if present."""
    capture = _ACTIVE_STRICT_DIAGNOSTIC_CAPTURE_V1.get()
    if capture is not None:
        capture.response.clear()
        capture.response.update(diagnostics)


def read_strict_diagnostic_field_v1(
    product: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None,
    key: str,
    default: Any = None,
) -> Any:
    """Read a diagnostic field without rebuilding the former flat envelope."""
    if diagnostics is not None and key in diagnostics:
        return diagnostics[key]
    return product.get(key, default)


def merge_strict_diagnostic_sidecars_v1(
    *sidecars: Mapping[str, Any],
) -> dict[str, Any]:
    """Shallowly join current-request sidecars with unique key ownership."""
    merged: dict[str, Any] = {}
    for sidecar in sidecars:
        duplicate_keys = set(merged).intersection(sidecar)
        if duplicate_keys:
            names = ",".join(sorted(duplicate_keys))
            raise ValueError(f"duplicate diagnostic producers: {names}")
        merged.update(sidecar)
    return merged


def split_strict_pipeline_product_v1(
    payload: dict[str, Any],
    *,
    diagnostics_out: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the guarded product complement and optionally expose diagnostics."""
    if diagnostics_out:
        raise ValueError("diagnostics_out must be empty")
    product = {
        key: value
        for key, value in payload.items()
        if key not in STRICT_PIPELINE_DIAGNOSTIC_KEYS_V1
    }
    if diagnostics_out is not None:
        diagnostics_out.update(
            (key, value)
            for key, value in payload.items()
            if key in STRICT_PIPELINE_DIAGNOSTIC_KEYS_V1
        )
    return product


def split_strict_response_product_v1(
    payload: dict[str, Any],
    *,
    diagnostics_out: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the HTTP product complement and optionally expose diagnostics."""
    if diagnostics_out:
        raise ValueError("diagnostics_out must be empty")
    product = {
        key: value
        for key, value in payload.items()
        if key not in STRICT_RESPONSE_DIAGNOSTIC_KEYS_V1
    }
    if diagnostics_out is not None:
        diagnostics_out.update(
            (key, value)
            for key, value in payload.items()
            if key in STRICT_RESPONSE_DIAGNOSTIC_KEYS_V1
        )
    return product
