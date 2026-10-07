"""R2a vector glyph dump layer.

See docs/restart_plans/R2a.md for the dump-only contract. Tokens emitted here
are diagnostic evidence only; ``consumer_allowed`` stays false until a later
flag-gated route explicitly proves final-dimension safety.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Any


SCHEMA_VERSION = "vector_glyph_v1"
CONSUMER_ALLOWED = False
DEGREE_RING_MIN_PT = 2.0
DEGREE_RING_MAX_PT = 8.0
DEGREE_RING_NEAR_TEXT_PT = 18.0
DEDUPE_IOU = 0.75
DIGIT_SLOT_CONFIDENCE = 0.35
DIGIT_COMPONENT_SLOT_CONFIDENCE = 0.32
DIGIT_SLOT_MIN_REGION_SHORT_PT = 3.0
DIGIT_SLOT_MAX_REGION_LONG_PT = 240.0
DIGIT_SLOT_MIN_SIZE_PT = 2.0
DIGIT_SLOT_MAX_SIZE_PT = 24.0
DIGIT_COMPONENT_SLOT_GAP_FACTOR = 0.28
DIGIT_COMPONENT_SLOT_MIN_PRIMITIVE_PT = 0.05
DIGIT_COMPONENT_SLOT_MAX_SEGMENTS = 32
DIGIT_COMPONENT_CLUSTER_SLOT_ENV = "VECTOR_DIGIT_COMPONENT_CLUSTER_SLOT_DUMP"
DIGIT_COMPONENT_CLUSTER_SLOT_DISTANCE_MAX_ENV = (
    "VECTOR_DIGIT_COMPONENT_CLUSTER_SLOT_DISTANCE_MAX"
)
DIGIT_COMPONENT_CLUSTER_SLOT_TOP_K_ENV = "VECTOR_DIGIT_COMPONENT_CLUSTER_SLOT_TOP_K"
DIGIT_COMPONENT_CLUSTER_SLOT_LIBRARY_ENV = (
    "VECTOR_DIGIT_COMPONENT_CLUSTER_SLOT_LIBRARY"
)
DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV = (
    "VECTOR_DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND"
)
DIGIT_COMPONENT_CLUSTER_SLOT_CAP = 2000
DIGIT_COMPONENT_DEGREE_ANCHORED_SLOT_ENV = (
    "VECTOR_DIGIT_COMPONENT_DEGREE_ANCHORED_SLOT_DUMP"
)
DIGIT_COMPONENT_PLUS_MINUS_ANCHORED_SLOT_ENV = (
    "VECTOR_DIGIT_COMPONENT_PLUS_MINUS_ANCHORED_SLOT_DUMP"
)
DIGIT_STRUCTURAL_FEATURES_ENV = "VECTOR_DIGIT_STRUCTURAL_FEATURES_DUMP"
DIGIT_STRUCTURAL_FEATURES_SOURCE = "vector_text_component_structural_features"
DIGIT_COMPONENT_DEGREE_ANCHORED_SLOT_CAP = 2000
DIGIT_COMPONENT_PLUS_MINUS_ANCHORED_SLOT_CAP = 200
DIGIT_COMPONENT_PRIMITIVE_INDEX_CELL_PT = 24.0
DIGIT_COMPONENT_MAX_MAIN_AXIS_FACTOR = 1.8
DIGIT_COMPONENT_MIN_REGION_OVERLAP = 0.60
DIMENSION_ANCHOR_CROSS_BAND_FACTOR = 0.50
DIMENSION_ANCHOR_MAX_MAIN_DELTA_FACTOR = 4.0
DECIMAL_ANCHOR_NATIVE_CHAR_RADIUS_PT = 1.5
DECIMAL_ANCHOR_SUPPORT_CROSS_FACTOR = 0.75
DECIMAL_ANCHOR_SUPPORT_MAX_MAIN_GAP_FACTOR = 0.85
DECIMAL_ANCHOR_MAX_DOT_SIZE_PT = 4.5
DECIMAL_ANCHOR_MAX_DOT_AREA_PT2 = 20.0
DECIMAL_ANCHOR_DENSE_DOT_MAIN_FACTOR = 2.5
DECIMAL_ANCHOR_DENSE_DOT_CROSS_FACTOR = 1.25
DECIMAL_AXIS_PAIR_MIN_DISTANCE_PT = 2.0
DECIMAL_AXIS_PAIR_MAX_GAP_FACTOR = 1.25
DECIMAL_AXIS_PAIR_MAX_CROSS_FACTOR = 0.50
DECIMAL_AXIS_COMPONENT_MAX_GAP_FACTOR = 1.25
DECIMAL_SHAPE_AXIS_ANGLE_TOLERANCE_DEG = 8.0
DECIMAL_SHAPE_AXIS_PERP_TOLERANCE_DEG = 8.0
DECIMAL_SHAPE_AXIS_SHORT_LONG_MIN_RATIO = 0.45
DECIMAL_SHAPE_AXIS_SHORT_LONG_MAX_RATIO = 0.90
DECIMAL_SHAPE_MATCH_CENTER_PT = 1.25
DECIMAL_SHAPE_CLUSTER_MAX_MAIN_DISTANCE_PT = 15.0
DECIMAL_SHAPE_CLUSTER_MAX_MAIN_SIZE_PT = 15.0
DECIMAL_SHAPE_CLUSTER_MAX_CROSS_SIZE_PT = 16.0
DECIMAL_SHAPE_CLUSTER_MAX_CROSS_GAP_PT = 6.0
DECIMAL_SHAPE_CLUSTER_GAP_PT = 1.25
DECIMAL_SHAPE_CLUSTER_PAD_PT = 0.35
DECIMAL_SHAPE_CLUSTER_MIN_MAIN_PT = 3.0
NATIVE_TEXT_OVERLAP_MIN_RATIO = 0.20
DEDUPE_INDEX_CELL_PT = 24.0

DIAMETER_CHARS = frozenset("⌀∅ØøΦφ")


def build_vector_glyph_v1(
    *,
    page: Any | None = None,
    page_index: int = 0,
    text_regions: list[dict[str, Any]] | None = None,
    diameter_glyphs: list[dict[str, Any]] | None = None,
    excluded_digit_slot_regions: list[dict[str, Any]] | None = None,
    include_digit_slots: bool = False,
) -> dict[str, Any]:
    """Build a unified glyph-token dump from existing vector evidence."""
    tokens: list[dict[str, Any]] = []

    for idx, glyph in enumerate(diameter_glyphs or []):
        bbox = _bbox_from_any(glyph)
        if not bbox:
            continue
        source_primitives = glyph.get("source_primitives")
        source_segments = (
            [
                {
                    "kind": "diameter_glyph",
                    "index": idx,
                    "primitive_index": primitive_index,
                    **dict(source_primitive),
                }
                for primitive_index, source_primitive in enumerate(
                    source_primitives
                )
                if isinstance(source_primitive, dict)
            ]
            if isinstance(source_primitives, list)
            and source_primitives
            else [{"kind": "diameter_glyph", "index": idx}]
        )
        tokens.append(_make_token(
            page_index=page_index,
            order=len(tokens),
            glyph_type="diameter",
            text="⌀",
            bbox=bbox,
            confidence=0.95,
            strict_checks={
                "source": "extract_diameter_glyphs",
                "circle_slash_topology": True,
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            source_segments=source_segments,
        ))

    tokens.extend(_tokens_from_numeric_symbol_seeds(
        text_regions or [],
        page_index=page_index,
        start_order=len(tokens),
    ))
    if include_digit_slots:
        tokens.extend(_digit_slot_tokens_from_numeric_symbol_context(
            page,
            text_regions or [],
            page_index=page_index,
            start_order=len(tokens),
            excluded_regions=excluded_digit_slot_regions or [],
        ))
        tokens.extend(_digit_slot_tokens_from_vector_text_components(
            page,
            text_regions or [],
            page_index=page_index,
            start_order=len(tokens),
            excluded_regions=excluded_digit_slot_regions or [],
        ))
    tokens.extend(_tokens_from_page_chars(
        page,
        page_index=page_index,
        start_order=len(tokens),
    ))

    deduped = _dedupe_tokens(tokens)
    counts = Counter(str(row.get("glyph_type") or "unknown") for row in deduped)
    low_conf_or_unknown = sum(
        1 for row in deduped
        if row.get("glyph_type") == "unknown"
        or float(row.get("confidence") or 0.0) < 0.80
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "tokens": deduped,
        "stats": {
            "schema_version": SCHEMA_VERSION,
            "page_index": int(page_index),
            "token_count": len(deduped),
            "counts_by_type": dict(sorted(counts.items())),
            "low_conf_or_unknown": low_conf_or_unknown,
            "fallback_ratio": (
                low_conf_or_unknown / len(deduped)
                if deduped else 0.0
            ),
            "consumer_allowed_count": sum(
                1 for row in deduped if row.get("consumer_allowed")
            ),
        },
    }


def _tokens_from_numeric_symbol_seeds(
    text_regions: list[dict[str, Any]],
    *,
    page_index: int,
    start_order: int,
) -> list[dict[str, Any]]:
    # DEPRECATED: dot tokens derived from numeric_symbol_seed are legacy R2a
    # diagnostic evidence. Current decimal-point truth is
    # decimal_point_quad.vector_decimal_point_quad_v1.
    tokens: list[dict[str, Any]] = []
    for region_idx, region in enumerate(text_regions):
        attached = list(region.get("attached_numeric_symbol_seeds") or [])
        if region.get("source") == "numeric_symbol_seed":
            attached.append({
                "subtype": region.get("subtype"),
                "x": region.get("x"),
                "y": region.get("y"),
                "w": region.get("w"),
                "h": region.get("h"),
                "raw_symbol_bbox": region.get("raw_symbol_bbox"),
                "raw_symbol_segments": region.get(
                    "raw_symbol_segments"
                ),
                "dot_seed_only": bool(region.get("dot_seed_only")),
            })
        for seed_idx, seed in enumerate(attached):
            subtype = str(seed.get("subtype") or "")
            glyph_type = _glyph_type_from_seed_subtype(subtype)
            if not glyph_type:
                continue
            bbox = (
                _numeric_symbol_seed_anchor_bbox(seed)
                if glyph_type in {"dot", "plus_minus"}
                else _bbox_from_any(seed)
            )
            if not bbox:
                continue
            is_orphan = bool(seed.get("dot_seed_only") or region.get("dot_seed_only"))
            tokens.append(_make_token(
                page_index=page_index,
                order=start_order + len(tokens),
                glyph_type=glyph_type,
                text="." if glyph_type == "dot" else "±",
                bbox=bbox,
                confidence=0.88 if is_orphan else 0.93,
                strict_checks={
                    "source": "numeric_symbol_seed",
                    "subtype": subtype,
                    "attached_to_text_region": not is_orphan,
                    "dot_seed_only": is_orphan,
                    "consumer_allowed": CONSUMER_ALLOWED,
                },
                source_segments=[
                    {
                        "kind": "numeric_symbol_seed",
                        "text_region_index": region_idx,
                        "seed_index": seed_idx,
                        "subtype": subtype,
                    },
                    *_numeric_symbol_seed_line_segments(seed),
                ],
            ))
    return tokens


def _digit_slot_tokens_from_numeric_symbol_context(
    page: Any | None,
    text_regions: list[dict[str, Any]],
    *,
    page_index: int,
    start_order: int,
    excluded_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    primitive_index = _vector_text_primitive_index(page) if page is not None else None
    for region_idx, region in enumerate(text_regions):
        region_bbox = _bbox_from_any(region)
        if not region_bbox or not _region_can_host_digit_slots(region_bbox):
            continue
        if _bbox_center_inside_any(region_bbox, excluded_regions or []):
            continue
        region_orientation = _region_orientation(region_bbox)
        local_primitives = (
            [
                primitive for primitive in _query_text_primitives(
                    primitive_index,
                    region_bbox,
                )
                if _primitive_can_belong_to_text_region(
                    primitive,
                    region_bbox,
                    orientation=region_orientation,
                )
            ]
            if primitive_index is not None else []
        )
        attached = list(region.get("attached_numeric_symbol_seeds") or [])
        for seed_idx, seed in enumerate(attached):
            subtype = str(seed.get("subtype") or "")
            if _glyph_type_from_seed_subtype(subtype) != "dot":
                continue
            if seed.get("dot_seed_only") or region.get("dot_seed_only"):
                continue
            seed_bbox = _numeric_symbol_seed_anchor_bbox(seed)
            if not seed_bbox:
                continue
            if not _decimal_anchor_dot_shape_ok(seed_bbox):
                continue
            shape_axis = _decimal_shape_axis_for_anchor(page, seed_bbox)
            if shape_axis is None:
                continue
            for role, slot_geometry in _decimal_shape_digit_slot_geometries(
                seed_bbox,
                region_bbox,
                shape_axis,
                primitives=local_primitives,
            ):
                slot_bbox = slot_geometry["bbox"]
                if not _bbox_size_ok(slot_bbox, min_size=DIGIT_SLOT_MIN_SIZE_PT):
                    continue
                source_segments = [
                    {
                        "kind": str(segment["kind"]),
                        "index": int(segment["index"]),
                    }
                    for segment in slot_geometry.get("source_segments", [])
                    if str(segment.get("kind") or "") in {"line", "curve", "rect"}
                ]
                if not source_segments:
                    continue
                tokens.append(_make_token(
                    page_index=page_index,
                    order=start_order + len(tokens),
                    glyph_type="digit_unknown",
                    text="?",
                    bbox=slot_bbox,
                    confidence=DIGIT_SLOT_CONFIDENCE,
                    strict_checks={
                        "source": "decimal_point_shape_digit_slot",
                        "subtype": subtype,
                        "orientation": slot_geometry["orientation"],
                        "slot_role": role,
                        "adjacent_to_numeric_symbol_seed": True,
                        "dimension_anchor_bound": True,
                        "dimension_anchor_count": 1,
                        "dimension_anchor_types": ["dot"],
                        "dimension_axis_source": "decimal_point_shape_anchor",
                        "dimension_axis_orientation": slot_geometry["orientation"],
                        "dimension_axis_vector": slot_geometry["axis_vector"],
                        "dimension_axis_angle_deg": slot_geometry["axis_angle_deg"],
                        "dimension_anchor_gate": "decimal_point_shape_short_edge",
                        "nearest_dimension_anchor_type": "dot",
                        "nearest_dimension_anchor_bbox": _round_bbox(seed_bbox),
                        "nearest_dimension_anchor_validation": {
                            "axis_source": "decimal_point_shape_short_edge",
                            "shape_curve_index": shape_axis.get("curve_index"),
                            "shape_edge_count": shape_axis.get("edge_count"),
                            "short_edge_length_pt": shape_axis.get(
                                "short_edge_length_pt"
                            ),
                            "long_edge_length_pt": shape_axis.get(
                                "long_edge_length_pt"
                            ),
                        },
                        "nearest_dimension_anchor_main_side": role,
                        "dimension_anchor_spanning_component": False,
                        "nearest_dimension_anchor_main_delta_pt": (
                            slot_geometry["anchor_main_delta_pt"]
                        ),
                        "nearest_dimension_anchor_signed_main_delta_pt": (
                            slot_geometry["signed_anchor_main_delta_pt"]
                        ),
                        "nearest_dimension_anchor_main_gap_pt": (
                            slot_geometry["anchor_main_gap_pt"]
                        ),
                        "nearest_dimension_anchor_main_overlap_pt": 0.0,
                        "nearest_dimension_anchor_cross_delta_pt": 0.0,
                        "nearest_dimension_anchor_cross_ratio": 0.0,
                        "nearest_dimension_anchor_main_ratio": (
                            slot_geometry["anchor_main_ratio"]
                        ),
                        "slot_cluster_primitive_count": (
                            slot_geometry["cluster_primitive_count"]
                        ),
                        "slot_cluster_main_gap_pt": (
                            slot_geometry["cluster_main_gap_pt"]
                        ),
                        "slot_bbox": _round_bbox(slot_bbox),
                        "slot_oriented_quad": slot_geometry["oriented_quad"],
                        "slot_expand_strategy": "decimal_point_shape_topology_cluster",
                        "slot_target_w": round(float(slot_bbox["w"]), 3),
                        "slot_target_h": round(float(slot_bbox["h"]), 3),
                        "slot_target_main": slot_geometry["slot_target_main"],
                        "slot_target_cross": slot_geometry["slot_target_cross"],
                        "prototype_match_status": "not_run",
                        "consumer_allowed": CONSUMER_ALLOWED,
                    },
                    source_segments=source_segments,
                    oriented_quad=slot_geometry["oriented_quad"],
                ))
    return tokens


def _digit_slot_tokens_from_vector_text_components(
    page: Any | None,
    text_regions: list[dict[str, Any]],
    *,
    page_index: int,
    start_order: int,
    excluded_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    if page is None:
        return tokens
    primitive_index = _vector_text_primitive_index(page)
    if not primitive_index["primitives"]:
        return tokens
    emit_structural_features = _env_flag_enabled(DIGIT_STRUCTURAL_FEATURES_ENV)
    emit_degree_anchored_slots = _env_flag_enabled(
        DIGIT_COMPONENT_DEGREE_ANCHORED_SLOT_ENV
    )
    emit_plus_minus_anchored_slots = _env_flag_enabled(
        DIGIT_COMPONENT_PLUS_MINUS_ANCHORED_SLOT_ENV
    )
    emit_cluster_slots = _env_flag_enabled(DIGIT_COMPONENT_CLUSTER_SLOT_ENV)
    cluster_filter_config = (
        _component_cluster_slot_filter_config()
        if emit_cluster_slots else _empty_component_cluster_slot_filter_config()
    )
    buffer_cluster_slots = _component_cluster_slot_filtering_enabled(
        cluster_filter_config
    )
    cluster_slot_emit_count = 0
    degree_slot_emit_count = 0
    plus_minus_slot_emit_count = 0
    cluster_candidates: list[dict[str, Any]] = []

    for region_idx, region in enumerate(text_regions):
        region_bbox = _bbox_from_any(region)
        if not region_bbox or not _region_can_host_digit_slots(region_bbox):
            continue
        if _bbox_center_inside_any(region_bbox, excluded_regions or []):
            continue
        orientation = _region_orientation(region_bbox)
        local = [
            primitive for primitive in _query_text_primitives(primitive_index, region_bbox)
            if _primitive_can_belong_to_text_region(
                primitive,
                region_bbox,
                orientation=orientation,
            )
        ]
        if not local:
            continue
        components = _component_slots_from_primitives(
            local,
            region_bbox,
            orientation=orientation,
        )
        if not components:
            continue
        if emit_structural_features:
            for component_idx, component in enumerate(components):
                token = _token_from_structural_feature_component(
                    component,
                    region_bbox,
                    orientation=orientation,
                    region_idx=region_idx,
                    component_idx=component_idx,
                    page_index=page_index,
                    order=start_order + len(tokens),
                )
                if token is not None:
                    tokens.append(token)
        anchor_context = _dimension_anchor_context_for_region(
            page,
            region,
            region_bbox,
            orientation=orientation,
            components=components,
        )
        if int(anchor_context["anchor_count"]) < 1:
            semantic_spec: dict[str, Any] | None = None
            if (
                emit_degree_anchored_slots
                and int(anchor_context.get("degree_anchor_count") or 0) > 0
            ):
                semantic_spec = {
                    "source": "vector_text_component_degree_anchored_slot",
                    "anchor_type": "degree",
                    "dimension_axis_source": "degree",
                    "slot_expand_strategy": "component_bbox_degree_anchor",
                    "cap": DIGIT_COMPONENT_DEGREE_ANCHORED_SLOT_CAP,
                }
            elif (
                emit_plus_minus_anchored_slots
                and int(anchor_context.get("plus_minus_anchor_count") or 0) > 0
            ):
                semantic_spec = {
                    "source": "vector_text_component_plus_minus_anchored_slot",
                    "anchor_type": "plus_minus",
                    "dimension_axis_source": "plus_minus",
                    "slot_expand_strategy": "component_bbox_plus_minus_anchor",
                    "cap": DIGIT_COMPONENT_PLUS_MINUS_ANCHORED_SLOT_CAP,
                }
            if semantic_spec is not None:
                semantic_context = _semantic_anchor_context(
                    anchor_context,
                    anchor_type=str(semantic_spec["anchor_type"]),
                )
                for component_idx, component in enumerate(components):
                    if semantic_spec["anchor_type"] == "degree":
                        if degree_slot_emit_count >= int(semantic_spec["cap"]):
                            break
                    elif plus_minus_slot_emit_count >= int(semantic_spec["cap"]):
                        break
                    token = _token_from_semantic_anchored_component(
                        page,
                        component,
                        region_bbox,
                        semantic_context,
                        source=str(semantic_spec["source"]),
                        dimension_axis_source=str(
                            semantic_spec["dimension_axis_source"]
                        ),
                        slot_expand_strategy=str(
                            semantic_spec["slot_expand_strategy"]
                        ),
                        orientation=orientation,
                        region_idx=region_idx,
                        component_idx=component_idx,
                        page_index=page_index,
                        order=start_order + len(tokens),
                    )
                    if token is None:
                        continue
                    tokens.append(token)
                    if semantic_spec["anchor_type"] == "degree":
                        degree_slot_emit_count += 1
                    else:
                        plus_minus_slot_emit_count += 1
                continue
            if emit_cluster_slots:
                for component_idx, component in enumerate(components):
                    if cluster_slot_emit_count >= DIGIT_COMPONENT_CLUSTER_SLOT_CAP:
                        break
                    candidate = _component_cluster_slot_candidate(
                        page,
                        component,
                        region_bbox,
                        anchor_context,
                        orientation=orientation,
                        region_idx=region_idx,
                        component_idx=component_idx,
                        candidate_idx=cluster_slot_emit_count,
                    )
                    if candidate is None:
                        continue
                    cluster_slot_emit_count += 1
                    if buffer_cluster_slots:
                        cluster_candidates.append(candidate)
                    else:
                        tokens.append(_token_from_component_cluster_candidate(
                            candidate,
                            page_index=page_index,
                            order=start_order + len(tokens),
                        ))
            continue
        for component_idx, component in enumerate(components):
            slot_bbox = component["bbox"]
            if not _bbox_size_ok(slot_bbox, min_size=DIGIT_SLOT_MIN_SIZE_PT):
                continue
            anchor_debug = _component_anchor_debug(
                component["component_bbox"],
                region_bbox,
                anchor_context,
                orientation=orientation,
            )
            if (
                not bool(anchor_debug["anchor_bound"])
                or bool(anchor_debug["component_spans_anchor"])
            ):
                continue
            native_overlap = _component_native_text_overlap(
                page,
                component["component_bbox"],
            )
            if native_overlap["has_non_digit"]:
                continue
            slot_geometry = _oriented_component_slot_geometry(
                component["component_bbox"],
                region_bbox,
                anchor_debug,
            )
            slot_bbox = slot_geometry["bbox"]
            slot_quad = slot_geometry["oriented_quad"]
            if not _bbox_size_ok(slot_bbox, min_size=DIGIT_SLOT_MIN_SIZE_PT):
                continue
            source_segments = [
                {
                    "kind": str(segment["kind"]),
                    "index": int(segment["index"]),
                }
                for segment in component["source_segments"][
                    :DIGIT_COMPONENT_SLOT_MAX_SEGMENTS
                ]
            ]
            tokens.append(_make_token(
                page_index=page_index,
                order=start_order + len(tokens),
                glyph_type="digit_unknown",
                text="?",
                bbox=slot_bbox,
                confidence=DIGIT_COMPONENT_SLOT_CONFIDENCE,
                strict_checks={
                    "source": "vector_text_component_slot",
                    "orientation": orientation,
                    "slot_role": "component",
                    "text_region_index": region_idx,
                    "component_index": component_idx,
                    "primitive_count": int(component["primitive_count"]),
                    "source_segment_count": len(component["source_segments"]),
                    "component_merge_strategy": "axis_projection_gap",
                    "component_bbox": _round_bbox(component["component_bbox"]),
                    "raw_component_bbox": _round_bbox(
                        component["raw_component_bbox"]
                    ),
                    "component_clipped_to_region": bool(
                        component.get("component_clipped_to_region")
                    ),
                    "text_region_bbox": _round_bbox(region_bbox),
                    "slot_bbox": _round_bbox(slot_bbox),
                    "slot_oriented_quad": slot_quad,
                    "slot_expand_strategy": slot_geometry["slot_expand_strategy"],
                    "slot_target_w": round(float(slot_bbox["w"]), 3),
                    "slot_target_h": round(float(slot_bbox["h"]), 3),
                    "slot_target_main": slot_geometry["slot_target_main"],
                    "slot_target_cross": slot_geometry["slot_target_cross"],
                    "slot_component_width_ratio": _safe_ratio(
                        float(component["component_bbox"]["w"]),
                        float(slot_bbox["w"]),
                    ),
                    "slot_component_height_ratio": _safe_ratio(
                        float(component["component_bbox"]["h"]),
                        float(slot_bbox["h"]),
                    ),
                    "dimension_anchor_bound": anchor_debug["anchor_bound"],
                    "dimension_anchor_count": anchor_context["anchor_count"],
                    "dimension_anchor_types": anchor_context["anchor_types"],
                    "dimension_axis_source": anchor_context["axis_source"],
                    "dimension_axis_orientation": orientation,
                    "dimension_axis_vector": anchor_debug["axis_vector"],
                    "dimension_axis_angle_deg": anchor_debug["axis_angle_deg"],
                    "dimension_anchor_gate": anchor_context[
                        "component_slot_gate"
                    ],
                    "ignored_dimension_anchor_counts": anchor_context[
                        "ignored_anchor_counts"
                    ],
                    "nearest_dimension_anchor_type": anchor_debug["nearest_anchor_type"],
                    "nearest_dimension_anchor_bbox": anchor_debug["nearest_anchor_bbox"],
                    "nearest_dimension_anchor_validation": anchor_debug[
                        "nearest_anchor_validation"
                    ],
                    "nearest_dimension_anchor_main_side": anchor_debug[
                        "nearest_anchor_main_side"
                    ],
                    "dimension_anchor_spanning_component": anchor_debug[
                        "component_spans_anchor"
                    ],
                    "nearest_dimension_anchor_main_delta_pt": anchor_debug["main_delta_pt"],
                    "nearest_dimension_anchor_signed_main_delta_pt": anchor_debug[
                        "signed_main_delta_pt"
                    ],
                    "nearest_dimension_anchor_main_gap_pt": anchor_debug[
                        "main_gap_pt"
                    ],
                    "nearest_dimension_anchor_main_overlap_pt": anchor_debug[
                        "main_overlap_pt"
                    ],
                    "nearest_dimension_anchor_cross_delta_pt": anchor_debug["cross_delta_pt"],
                    "nearest_dimension_anchor_cross_ratio": anchor_debug["cross_ratio"],
                    "nearest_dimension_anchor_main_ratio": anchor_debug["main_ratio"],
                    "native_digit_text_overlap": native_overlap["digit_text"],
                    "native_non_digit_text_overlap": native_overlap["non_digit_text"],
                    "prototype_match_status": "not_run",
                    "consumer_allowed": CONSUMER_ALLOWED,
                },
                source_segments=source_segments,
                oriented_quad=slot_quad,
            ))
    if cluster_candidates:
        for candidate in _filtered_component_cluster_slot_candidates(
            page,
            cluster_candidates,
            cluster_filter_config,
        ):
            tokens.append(_token_from_component_cluster_candidate(
                candidate,
                page_index=page_index,
                order=start_order + len(tokens),
            ))
    return tokens


def _component_cluster_slot_candidate(
    page: Any | None,
    component: dict[str, Any],
    region_bbox: dict[str, float],
    anchor_context: dict[str, Any],
    *,
    orientation: str,
    region_idx: int,
    component_idx: int,
    candidate_idx: int,
) -> dict[str, Any] | None:
    slot_bbox = component["bbox"]
    if not _bbox_size_ok(slot_bbox, min_size=DIGIT_SLOT_MIN_SIZE_PT):
        return None
    native_overlap = _component_native_text_overlap(
        page,
        component["component_bbox"],
    )
    if native_overlap["has_non_digit"]:
        return None
    slot_quad = _quad_from_bbox(slot_bbox)
    source_segments = [
        {
            "kind": str(segment["kind"]),
            "index": int(segment["index"]),
        }
        for segment in component["source_segments"][
            :DIGIT_COMPONENT_SLOT_MAX_SEGMENTS
        ]
    ]
    checks = {
        "source": "vector_text_component_cluster_slot",
        "orientation": orientation,
        "slot_role": "component",
        "text_region_index": region_idx,
        "component_index": component_idx,
        "primitive_count": int(component["primitive_count"]),
        "source_segment_count": len(component["source_segments"]),
        "component_merge_strategy": "axis_projection_gap",
        "component_bbox": _round_bbox(component["component_bbox"]),
        "raw_component_bbox": _round_bbox(component["raw_component_bbox"]),
        "component_clipped_to_region": bool(
            component.get("component_clipped_to_region")
        ),
        "text_region_bbox": _round_bbox(region_bbox),
        "slot_bbox": _round_bbox(slot_bbox),
        "slot_oriented_quad": slot_quad,
        "slot_expand_strategy": "component_bbox_no_anchor",
        "slot_target_w": round(float(slot_bbox["w"]), 3),
        "slot_target_h": round(float(slot_bbox["h"]), 3),
        "slot_target_main": round(
            float(slot_bbox["w"] if orientation == "H" else slot_bbox["h"]),
            3,
        ),
        "slot_target_cross": round(
            float(slot_bbox["h"] if orientation == "H" else slot_bbox["w"]),
            3,
        ),
        "slot_component_width_ratio": _safe_ratio(
            float(component["component_bbox"]["w"]),
            float(slot_bbox["w"]),
        ),
        "slot_component_height_ratio": _safe_ratio(
            float(component["component_bbox"]["h"]),
            float(slot_bbox["h"]),
        ),
        "dimension_anchor_bound": False,
        "dimension_anchor_count": anchor_context["anchor_count"],
        "dimension_anchor_types": anchor_context["anchor_types"],
        "dimension_axis_source": "none",
        "dimension_axis_orientation": orientation,
        "dimension_axis_vector": [],
        "dimension_axis_angle_deg": None,
        "dimension_anchor_gate": anchor_context["component_slot_gate"],
        "ignored_dimension_anchor_counts": anchor_context["ignored_anchor_counts"],
        "nearest_dimension_anchor_type": None,
        "nearest_dimension_anchor_bbox": None,
        "nearest_dimension_anchor_validation": None,
        "nearest_dimension_anchor_main_side": None,
        "dimension_anchor_spanning_component": False,
        "nearest_dimension_anchor_main_delta_pt": None,
        "nearest_dimension_anchor_signed_main_delta_pt": None,
        "nearest_dimension_anchor_main_gap_pt": None,
        "nearest_dimension_anchor_main_overlap_pt": None,
        "nearest_dimension_anchor_cross_delta_pt": None,
        "nearest_dimension_anchor_cross_ratio": None,
        "nearest_dimension_anchor_main_ratio": None,
        "native_digit_text_overlap": native_overlap["digit_text"],
        "native_non_digit_text_overlap": native_overlap["non_digit_text"],
        "prototype_match_status": "not_run",
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return {
        "candidate_idx": int(candidate_idx),
        "bbox": slot_bbox,
        "strict_checks": checks,
        "source_segments": source_segments,
        "oriented_quad": slot_quad,
    }


def _token_from_component_cluster_candidate(
    candidate: dict[str, Any],
    *,
    page_index: int,
    order: int,
) -> dict[str, Any]:
    return _make_token(
        page_index=page_index,
        order=order,
        glyph_type="digit_unknown",
        text="?",
        bbox=candidate["bbox"],
        confidence=DIGIT_COMPONENT_SLOT_CONFIDENCE,
        strict_checks=candidate["strict_checks"],
        source_segments=candidate["source_segments"],
        oriented_quad=candidate["oriented_quad"],
    )


def _token_from_structural_feature_component(
    component: dict[str, Any],
    region_bbox: dict[str, float],
    *,
    orientation: str,
    region_idx: int,
    component_idx: int,
    page_index: int,
    order: int,
) -> dict[str, Any] | None:
    bbox = component.get("component_bbox")
    if not isinstance(bbox, dict):
        return None
    if not _bbox_size_ok(bbox, min_size=DIGIT_COMPONENT_SLOT_MIN_PRIMITIVE_PT):
        return None
    source_segments = [
        {
            "kind": str(segment["kind"]),
            "index": int(segment["index"]),
        }
        for segment in component.get("source_segments", [])[
            :DIGIT_COMPONENT_SLOT_MAX_SEGMENTS
        ]
    ]
    structural_features = _structural_features_for_component(component)
    return _make_token(
        page_index=page_index,
        order=order,
        glyph_type="digit_unknown",
        text="?",
        bbox=bbox,
        confidence=DIGIT_COMPONENT_SLOT_CONFIDENCE,
        strict_checks={
            "source": DIGIT_STRUCTURAL_FEATURES_SOURCE,
            "orientation": orientation,
            "slot_role": "component",
            "text_region_index": region_idx,
            "component_index": component_idx,
            "primitive_count": int(component["primitive_count"]),
            "source_segment_count": len(component.get("source_segments") or []),
            "component_merge_strategy": "axis_projection_gap",
            "component_bbox": _round_bbox(component["component_bbox"]),
            "raw_component_bbox": _round_bbox(component["raw_component_bbox"]),
            "component_clipped_to_region": bool(
                component.get("component_clipped_to_region")
            ),
            "text_region_bbox": _round_bbox(region_bbox),
            "slot_bbox": _round_bbox(component["bbox"]),
            "slot_oriented_quad": _quad_from_bbox(component["bbox"]),
            "slot_expand_strategy": "component_bbox_structural_features",
            "prototype_match_status": "not_run_structural_features_dump",
            "structural_features": structural_features,
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        source_segments=source_segments,
        oriented_quad=_quad_from_bbox(bbox),
    )


def _structural_features_for_component(component: dict[str, Any]) -> dict[str, Any]:
    primitives = [
        primitive for primitive in component.get("primitives", [])
        if isinstance(primitive, dict)
    ]
    return structural_features_from_primitives(
        primitives,
        bbox=component["component_bbox"],
        source=DIGIT_STRUCTURAL_FEATURES_SOURCE,
    )


def structural_features_from_primitives(
    primitives: list[dict[str, Any]],
    *,
    bbox: dict[str, float] | None = None,
    source: str = DIGIT_STRUCTURAL_FEATURES_SOURCE,
    bbox_reference: dict[str, float] | None = None,
) -> dict[str, Any]:
    feature_bbox = (
        dict(bbox)
        if isinstance(bbox, dict)
        else _bbox_from_primitives(primitives)
    )
    if feature_bbox is None:
        feature_bbox = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    reported_bbox = (
        _bbox_relative_to(feature_bbox, bbox_reference)
        if isinstance(bbox_reference, dict) else feature_bbox
    )
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    primitive_line_count = 0
    primitive_curve_count = 0
    primitive_polyline_count = 0
    for primitive in primitives:
        kind = str(primitive.get("kind") or "")
        if kind == "line":
            primitive_line_count += 1
        bezier_count = _primitive_bezier_op_count(primitive)
        primitive_curve_count += bezier_count
        if kind in {"curve", "rect"} and bezier_count == 0:
            primitive_polyline_count += 1
        segments.extend(_primitive_flattened_segments(primitive))

    graph = _segment_graph(segments)
    closed_loop_count = _closed_loop_count_from_graph(graph)
    junction_count = sum(
        1 for neighbors in graph["adjacency"].values() if len(neighbors) >= 3
    )
    lengths = [_segment_length(segment) for segment in segments]
    nonzero_lengths = [length for length in lengths if length > 1e-9]
    max_length = max(nonzero_lengths) if nonzero_lengths else 0.0
    return {
        "schema_version": "vector_digit_structural_features_v1",
        "source": str(source),
        "bbox": _round_bbox(reported_bbox),
        "aspect": _round_float(
            float(reported_bbox["w"]) / max(float(reported_bbox["h"]), 1e-9)
        ),
        "primitive_line_count": primitive_line_count,
        "primitive_curve_count": primitive_curve_count,
        "primitive_polyline_count": primitive_polyline_count,
        "total_segment_count": len(segments),
        "vertex_count": len(graph["adjacency"]),
        "closed_loop_count": closed_loop_count,
        "junction_count": junction_count,
        "segment_angle_histogram": _segment_angle_histogram(segments),
        "segment_length_ratios": [
            _round_float(length / max_length)
            for length in sorted(nonzero_lengths, reverse=True)
        ] if max_length > 0.0 else [],
        "is_convex_hull_closed": (
            closed_loop_count == 1
            and _graph_component_count(graph) == 1
            and junction_count == 0
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _bbox_from_primitives(primitives: list[dict[str, Any]]) -> dict[str, float] | None:
    bbox: dict[str, float] | None = None
    for primitive in primitives:
        primitive_bbox = primitive.get("bbox")
        if isinstance(primitive_bbox, dict):
            bbox = _union_bbox(bbox, primitive_bbox) if bbox else dict(primitive_bbox)
    return bbox


def _bbox_relative_to(
    bbox: dict[str, float],
    reference: dict[str, float],
) -> dict[str, float]:
    return {
        "x": float(bbox["x"]) - float(reference["x"]),
        "y": float(bbox["y"]) - float(reference["y"]),
        "w": float(bbox["w"]),
        "h": float(bbox["h"]),
    }


def _primitive_bezier_op_count(primitive: dict[str, Any]) -> int:
    return sum(
        1
        for op, _points in _primitive_path(primitive)
        if op in {"c", "v", "y"}
    )


def _primitive_flattened_segments(
    primitive: dict[str, Any],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    path = _primitive_path(primitive)
    if path:
        return _flatten_path_segments(path)
    points = [
        point for point in primitive.get("points", [])
        if _point_from_any(point) is not None
    ]
    parsed = [_point_from_any(point) for point in points]
    clean = [point for point in parsed if point is not None]
    return _segments_from_points(clean)


def _primitive_path(
    primitive: dict[str, Any],
) -> list[tuple[str, list[tuple[float, float]]]]:
    raw_path = primitive.get("path")
    if not isinstance(raw_path, list):
        return []
    path: list[tuple[str, list[tuple[float, float]]]] = []
    for item in raw_path:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        op = str(item[0] or "").lower()
        points = [
            parsed for parsed in (_point_from_any(value) for value in item[1:])
            if parsed is not None
        ]
        path.append((op, points))
    return path


def _flatten_path_segments(
    path: list[tuple[str, list[tuple[float, float]]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    current: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None
    for op, points in path:
        if op == "m":
            current = points[-1] if points else None
            subpath_start = current
            continue
        if op == "h":
            if current is not None and subpath_start is not None:
                segments.append((current, subpath_start))
                current = subpath_start
            continue
        if not points:
            continue
        if current is None:
            current = points[-1]
            subpath_start = current if subpath_start is None else subpath_start
            continue
        if op == "l":
            end = points[-1]
            segments.append((current, end))
            current = end
            continue
        if op in {"c", "v", "y"}:
            curve_segments = _flatten_curve_op(current, points, steps=6)
            segments.extend(curve_segments)
            current = points[-1]
            continue
        for point in points:
            segments.append((current, point))
            current = point
    return [
        segment for segment in segments
        if _segment_length(segment) > 1e-9
    ]


def _flatten_curve_op(
    start: tuple[float, float],
    points: list[tuple[float, float]],
    *,
    steps: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    end = points[-1]
    if len(points) < 3:
        return [(start, end)]
    c1, c2 = points[0], points[1]
    samples = [start]
    for idx in range(1, max(1, steps) + 1):
        t = idx / max(1, steps)
        samples.append(_cubic_point(start, c1, c2, end, t))
    return _segments_from_points(samples)


def _cubic_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    return (
        u ** 3 * p0[0]
        + 3.0 * u ** 2 * t * p1[0]
        + 3.0 * u * t ** 2 * p2[0]
        + t ** 3 * p3[0],
        u ** 3 * p0[1]
        + 3.0 * u ** 2 * t * p1[1]
        + 3.0 * u * t ** 2 * p2[1]
        + t ** 3 * p3[1],
    )


def _segments_from_points(
    points: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        (left, right)
        for left, right in zip(points, points[1:])
        if _segment_length((left, right)) > 1e-9
    ]


def _point_from_any(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _segment_graph(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> dict[str, Any]:
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for start, end in segments:
        left = _quantized_point_key(start)
        right = _quantized_point_key(end)
        if left == right:
            continue
        edge = tuple(sorted((left, right)))
        edges.add(edge)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return {"adjacency": adjacency, "edges": edges}


def _closed_loop_count_from_graph(graph: dict[str, Any]) -> int:
    vertex_count = len(graph["adjacency"])
    edge_count = len(graph["edges"])
    if vertex_count == 0:
        return 0
    return max(0, edge_count - vertex_count + _graph_component_count(graph))


def _graph_component_count(graph: dict[str, Any]) -> int:
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = graph["adjacency"]
    unseen = set(adjacency)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return count


def _segment_angle_histogram(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[int]:
    histogram = [0 for _ in range(8)]
    for start, end in segments:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        bin_idx = min(7, int(angle // 22.5))
        histogram[bin_idx] += 1
    return histogram


def _segment_length(
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    start, end = segment
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _quantized_point_key(point: tuple[float, float]) -> tuple[int, int]:
    return (round(float(point[0]) * 1000), round(float(point[1]) * 1000))


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _semantic_anchor_context(
    anchor_context: dict[str, Any],
    *,
    anchor_type: str,
) -> dict[str, Any]:
    anchors = list(anchor_context.get(f"{anchor_type}_anchors") or [])
    return {
        **anchor_context,
        "anchors": anchors,
        "anchor_count": len(anchors),
        "anchor_types": [anchor_type] if anchors else [],
        "axis_source": anchor_type if anchors else "none",
        "component_slot_gate": (
            f"{anchor_type}_anchor"
            if anchors else f"no_valid_{anchor_type}_anchor"
        ),
    }


def _token_from_semantic_anchored_component(
    page: Any | None,
    component: dict[str, Any],
    region_bbox: dict[str, float],
    anchor_context: dict[str, Any],
    *,
    source: str,
    dimension_axis_source: str,
    slot_expand_strategy: str,
    orientation: str,
    region_idx: int,
    component_idx: int,
    page_index: int,
    order: int,
) -> dict[str, Any] | None:
    slot_bbox = component["bbox"]
    if not _bbox_size_ok(slot_bbox, min_size=DIGIT_SLOT_MIN_SIZE_PT):
        return None
    anchor_debug = _component_anchor_debug(
        component["component_bbox"],
        region_bbox,
        anchor_context,
        orientation=orientation,
    )
    if (
        not bool(anchor_debug["anchor_bound"])
        or bool(anchor_debug["component_spans_anchor"])
    ):
        return None
    native_overlap = _component_native_text_overlap(
        page,
        component["component_bbox"],
    )
    if native_overlap["has_non_digit"]:
        return None
    slot_geometry = _oriented_component_slot_geometry(
        component["component_bbox"],
        region_bbox,
        anchor_debug,
    )
    slot_bbox = slot_geometry["bbox"]
    slot_quad = slot_geometry["oriented_quad"]
    if not _bbox_size_ok(slot_bbox, min_size=DIGIT_SLOT_MIN_SIZE_PT):
        return None
    source_segments = [
        {
            "kind": str(segment["kind"]),
            "index": int(segment["index"]),
        }
        for segment in component["source_segments"][
            :DIGIT_COMPONENT_SLOT_MAX_SEGMENTS
        ]
    ]
    return _make_token(
        page_index=page_index,
        order=order,
        glyph_type="digit_unknown",
        text="?",
        bbox=slot_bbox,
        confidence=DIGIT_COMPONENT_SLOT_CONFIDENCE,
        strict_checks={
            "source": source,
            "orientation": orientation,
            "slot_role": "component",
            "text_region_index": region_idx,
            "component_index": component_idx,
            "primitive_count": int(component["primitive_count"]),
            "source_segment_count": len(component["source_segments"]),
            "component_merge_strategy": "axis_projection_gap",
            "component_bbox": _round_bbox(component["component_bbox"]),
            "raw_component_bbox": _round_bbox(component["raw_component_bbox"]),
            "component_clipped_to_region": bool(
                component.get("component_clipped_to_region")
            ),
            "text_region_bbox": _round_bbox(region_bbox),
            "slot_bbox": _round_bbox(slot_bbox),
            "slot_oriented_quad": slot_quad,
            "slot_expand_strategy": slot_expand_strategy,
            "slot_target_w": round(float(slot_bbox["w"]), 3),
            "slot_target_h": round(float(slot_bbox["h"]), 3),
            "slot_target_main": slot_geometry["slot_target_main"],
            "slot_target_cross": slot_geometry["slot_target_cross"],
            "slot_component_width_ratio": _safe_ratio(
                float(component["component_bbox"]["w"]),
                float(slot_bbox["w"]),
            ),
            "slot_component_height_ratio": _safe_ratio(
                float(component["component_bbox"]["h"]),
                float(slot_bbox["h"]),
            ),
            "dimension_anchor_bound": anchor_debug["anchor_bound"],
            "dimension_anchor_count": anchor_context["anchor_count"],
            "dimension_anchor_types": anchor_context["anchor_types"],
            "dimension_axis_source": dimension_axis_source,
            "dimension_axis_orientation": orientation,
            "dimension_axis_vector": anchor_debug["axis_vector"],
            "dimension_axis_angle_deg": anchor_debug["axis_angle_deg"],
            "dimension_anchor_gate": anchor_context["component_slot_gate"],
            "ignored_dimension_anchor_counts": anchor_context[
                "ignored_anchor_counts"
            ],
            "nearest_dimension_anchor_type": anchor_debug["nearest_anchor_type"],
            "nearest_dimension_anchor_bbox": anchor_debug["nearest_anchor_bbox"],
            "nearest_dimension_anchor_validation": anchor_debug[
                "nearest_anchor_validation"
            ],
            "nearest_dimension_anchor_main_side": anchor_debug[
                "nearest_anchor_main_side"
            ],
            "dimension_anchor_spanning_component": anchor_debug[
                "component_spans_anchor"
            ],
            "nearest_dimension_anchor_main_delta_pt": anchor_debug["main_delta_pt"],
            "nearest_dimension_anchor_signed_main_delta_pt": anchor_debug[
                "signed_main_delta_pt"
            ],
            "nearest_dimension_anchor_main_gap_pt": anchor_debug["main_gap_pt"],
            "nearest_dimension_anchor_main_overlap_pt": anchor_debug[
                "main_overlap_pt"
            ],
            "nearest_dimension_anchor_cross_delta_pt": anchor_debug["cross_delta_pt"],
            "nearest_dimension_anchor_cross_ratio": anchor_debug["cross_ratio"],
            "nearest_dimension_anchor_main_ratio": anchor_debug["main_ratio"],
            "native_digit_text_overlap": native_overlap["digit_text"],
            "native_non_digit_text_overlap": native_overlap["non_digit_text"],
            "prototype_match_status": "not_run",
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        source_segments=source_segments,
        oriented_quad=slot_quad,
    )


def _filtered_component_cluster_slot_candidates(
    page: Any | None,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    filtered = list(candidates)
    physical_band = config.get("physical_band")
    if physical_band:
        filtered = _component_cluster_slot_candidates_with_physical_band(
            page,
            filtered,
            physical_band,
        )
    if config.get("library_path"):
        filtered = _component_cluster_slot_candidates_with_preview(
            page,
            filtered,
            str(config["library_path"]),
        )
    distance_max = config.get("distance_max")
    if distance_max is not None:
        filtered = [
            candidate for candidate in filtered
            if (
                candidate.get("preview_top1_distance") is not None
                and float(candidate["preview_top1_distance"]) <= float(distance_max)
            )
        ]
    top_k = int(config.get("top_k") or 0)
    if top_k > 0:
        filtered = sorted(
            filtered,
            key=lambda candidate: (
                _cluster_slot_distance_sort_value(candidate),
                int(candidate["candidate_idx"]),
            ),
        )[:top_k]
    return filtered


def _component_cluster_slot_candidates_with_physical_band(
    page: Any | None,
    candidates: list[dict[str, Any]],
    band: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    from vector_digit_match import (  # Local import keeps the default path cheap.
        _page_primitive_index,
        _slot_descriptor_for_token,
    )

    primitive_index = _page_primitive_index(page)
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        token = {
            "glyph_type": "digit_unknown",
            "text": "?",
            "bbox": candidate["bbox"],
            "source_segments": candidate["source_segments"],
            "strict_checks": candidate["strict_checks"],
        }
        descriptor = _slot_descriptor_for_token(
            page,
            token,
            primitive_index=primitive_index,
        )
        metrics = _component_cluster_slot_physical_metrics(
            candidate,
            descriptor,
        )
        drop_reason = _component_cluster_slot_physical_drop_reason(metrics, band)
        checks = dict(candidate["strict_checks"])
        checks.update({
            "physical_band_source": band.get("source"),
            "physical_band_status": "pass" if drop_reason is None else "drop",
            "physical_band_drop_reason": drop_reason,
            "physical_width_pt": round(metrics["width"], 6),
            "physical_height_pt": round(metrics["height"], 6),
            "physical_aspect": round(metrics["aspect"], 6),
            "physical_segment_count": metrics["segment_count"],
            "physical_vertex_count": metrics["vertex_count"],
            "physical_primitive_count": metrics["primitive_count"],
        })
        enriched = {
            **candidate,
            "strict_checks": checks,
            "physical_metrics": metrics,
            "drop_reason": drop_reason,
        }
        if drop_reason is None:
            filtered.append(enriched)
    return filtered


def _component_cluster_slot_physical_metrics(
    candidate: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, float | int]:
    bbox = candidate["bbox"]
    checks = candidate.get("strict_checks") or {}
    width = float(bbox["w"])
    height = float(bbox["h"])
    return {
        "width": width,
        "height": height,
        "aspect": width / max(height, 1e-9),
        "segment_count": _int_or_none(checks.get("source_segment_count")) or 0,
        "vertex_count": _int_or_none(descriptor.get("point_count")) or 0,
        "primitive_count": _int_or_none(checks.get("primitive_count")) or 0,
    }


def _component_cluster_slot_physical_drop_reason(
    metrics: dict[str, float | int],
    band: dict[str, Any],
) -> str | None:
    checks = (
        ("width", "width_band", "drop_band_width"),
        ("height", "height_band", "drop_band_height"),
        ("aspect", "aspect_band", "drop_band_aspect"),
        ("segment_count", "segment_count_band", "drop_band_segment"),
        ("vertex_count", "vertex_count_band", "drop_band_vertex"),
    )
    for metric_name, band_name, drop_reason in checks:
        low, high = band[band_name]
        value = float(metrics[metric_name])
        if value < float(low) or value > float(high):
            return drop_reason
    primitive_band = band.get("primitive_count_band")
    if primitive_band is not None:
        low, high = primitive_band
        value = float(metrics["primitive_count"])
        if value < float(low) or value > float(high):
            return "drop_band_primitive"
    return None


def _component_cluster_slot_candidates_with_preview(
    page: Any | None,
    candidates: list[dict[str, Any]],
    library_path: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    from vector_digit_match import (  # Local import keeps the default path cheap.
        STROKE_WIDTH_MODE_ENV,
        STROKE_WIDTH_MODE_SIZE_RELATIVE,
        _classify_with_prototypes,
        _page_primitive_index,
        _slot_descriptor_for_token,
        load_vector_digit_prototype_library,
    )

    original_mode = os.environ.get(STROKE_WIDTH_MODE_ENV)
    os.environ[STROKE_WIDTH_MODE_ENV] = STROKE_WIDTH_MODE_SIZE_RELATIVE
    try:
        library = load_vector_digit_prototype_library(library_path)
        primitive_index = _page_primitive_index(page)
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            token = {
                "glyph_type": "digit_unknown",
                "text": "?",
                "bbox": candidate["bbox"],
                "source_segments": candidate["source_segments"],
                "strict_checks": candidate["strict_checks"],
            }
            descriptor = _slot_descriptor_for_token(
                page,
                token,
                primitive_index=primitive_index,
            )
            match = _classify_with_prototypes(
                descriptor,
                prototype_library=library,
                threshold=1e9,
                margin=0.0,
            )
            checks = dict(candidate["strict_checks"])
            checks["preview_descriptor_mode"] = STROKE_WIDTH_MODE_SIZE_RELATIVE
            checks["preview_match_status"] = match["status"]
            checks["preview_top1_distance"] = match["top1_distance"]
            checks["preview_top1_label"] = match["top1_label"]
            checks["preview_top2_distance"] = match["top2_distance"]
            checks["preview_top2_label"] = match["top2_label"]
            enriched.append({
                **candidate,
                "strict_checks": checks,
                "preview_top1_distance": match["top1_distance"],
                "preview_top1_label": match["top1_label"],
            })
        return enriched
    finally:
        if original_mode is None:
            os.environ.pop(STROKE_WIDTH_MODE_ENV, None)
        else:
            os.environ[STROKE_WIDTH_MODE_ENV] = original_mode


def _cluster_slot_distance_sort_value(candidate: dict[str, Any]) -> float:
    value = candidate.get("preview_top1_distance")
    if value is None:
        return float("inf")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return parsed if math.isfinite(parsed) else float("inf")


def _tokens_from_page_chars(
    page: Any | None,
    *,
    page_index: int,
    start_order: int,
) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for char_idx, char in enumerate(getattr(page, "chars", []) or []):
        text = str(char.get("text") or "")
        mapped = _glyph_type_from_text(text)
        if not mapped:
            continue
        bbox = _bbox_from_char(char)
        if not bbox:
            continue
        glyph_type, normalized_text = mapped
        tokens.append(_make_token(
            page_index=page_index,
            order=start_order + len(tokens),
            glyph_type=glyph_type,
            text=normalized_text,
            bbox=bbox,
            confidence=1.0,
            strict_checks={
                "source": "page.chars",
                "char_text": text,
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            source_segments=[{"kind": "pdf_char", "index": char_idx}],
        ))
    return tokens


def _make_token(
    *,
    page_index: int,
    order: int,
    glyph_type: str,
    text: str,
    bbox: dict[str, float],
    confidence: float,
    strict_checks: dict[str, Any],
    source_segments: list[dict[str, Any]],
    oriented_quad: list[list[float]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "token_id": f"p{int(page_index) + 1:03d}_glyph_{int(order):06d}",
        "page_index": int(page_index),
        "glyph_type": glyph_type,
        "text": text,
        "bbox": _round_bbox(bbox),
        "oriented_quad": oriented_quad or _quad_from_bbox(bbox),
        "confidence": round(float(confidence), 4),
        "strict_checks": dict(strict_checks),
        "source_segments": list(source_segments),
        "drop_reason": None,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _glyph_type_from_seed_subtype(subtype: str) -> str | None:
    if subtype == "decimal_point_candidate":
        return "dot"
    if subtype == "plus_minus_candidate":
        return "plus_minus"
    return None


def _glyph_type_from_text(text: str) -> tuple[str, str] | None:
    if len(text) == 1 and text.isdigit():
        return "digit", text
    if text == "±":
        return "plus_minus", "±"
    if text in DIAMETER_CHARS:
        return "diameter", "⌀"
    return None


def _bbox_from_char(char: dict[str, Any]) -> dict[str, float] | None:
    try:
        x0 = float(char.get("x0", char.get("x", 0.0)))
        y0 = float(char.get("top", char.get("y", 0.0)))
        x1 = float(char.get("x1", x0))
        y1 = float(char.get("bottom", char.get("y1", y0)))
    except (TypeError, ValueError):
        return None
    return _bbox_from_xyxy(x0, y0, x1, y1)


def _bbox_from_any(item: dict[str, Any]) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    nested = item.get("bbox")
    if isinstance(nested, dict):
        return _bbox_from_any(nested)
    try:
        if all(key in item for key in ("x", "y", "w", "h")):
            x = float(item.get("x") or 0.0)
            y = float(item.get("y") or 0.0)
            w = float(item.get("w") or 0.0)
            h = float(item.get("h") or 0.0)
            if w <= 0.0 or h <= 0.0:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
        if all(key in item for key in ("x0", "y0", "x1", "y1")):
            return _bbox_from_xyxy(
                float(item["x0"]),
                float(item["y0"]),
                float(item["x1"]),
                float(item["y1"]),
            )
    except (TypeError, ValueError):
        return None
    return None


def _numeric_symbol_seed_anchor_bbox(seed: dict[str, Any]) -> dict[str, float] | None:
    raw = seed.get("raw_symbol_bbox")
    if isinstance(raw, dict):
        bbox = _bbox_from_any(raw)
        if bbox:
            return bbox
    return _bbox_from_any(seed)


def _numeric_symbol_seed_line_segments(
    seed: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(
        seed.get("raw_symbol_segments") or []
    ):
        if not isinstance(segment, dict):
            return []
        try:
            bbox = [
                round(float(segment[key]), 3)
                for key in ("x0", "y0", "x1", "y1")
            ]
        except (KeyError, TypeError, ValueError):
            return []
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            return []
        rows.append({
            "kind": "numeric_symbol_seed_line",
            "segment_index": index,
            "bbox": bbox,
        })
    return rows


def _decimal_anchor_dot_shape_ok(anchor_bbox: dict[str, float]) -> bool:
    w = float(anchor_bbox["w"])
    h = float(anchor_bbox["h"])
    if w <= 0.0 or h <= 0.0:
        return False
    if w > DECIMAL_ANCHOR_MAX_DOT_SIZE_PT or h > DECIMAL_ANCHOR_MAX_DOT_SIZE_PT:
        return False
    if w * h > DECIMAL_ANCHOR_MAX_DOT_AREA_PT2:
        return False
    if max(w, h) / max(min(w, h), 1e-9) > 1.8:
        return False
    return True


def _curve_bbox(curve: dict[str, Any]) -> dict[str, float] | None:
    pts = curve.get("pts") or []
    if pts:
        try:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
        except (TypeError, ValueError, IndexError):
            return None
        return _bbox_from_xyxy(min(xs), min(ys), max(xs), max(ys))
    return _bbox_from_any(curve)


def _degree_ring_bbox_from_curve(curve: dict[str, Any]) -> dict[str, float] | None:
    bbox = _curve_bbox(curve)
    if not bbox:
        return None
    w = bbox["w"]
    h = bbox["h"]
    if not (DEGREE_RING_MIN_PT <= w <= DEGREE_RING_MAX_PT):
        return None
    if not (DEGREE_RING_MIN_PT <= h <= DEGREE_RING_MAX_PT):
        return None
    if max(w, h) / max(min(w, h), 1e-6) > 1.35:
        return None
    pts = curve.get("pts") or []
    if pts and len(pts) < 4:
        return None
    return bbox


def _dimension_anchor_context_for_region(
    page: Any | None,
    region: dict[str, Any],
    region_bbox: dict[str, float],
    *,
    orientation: str,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    degree_anchors: list[dict[str, Any]] = []
    plus_minus_anchors: list[dict[str, Any]] = []
    ignored: Counter[str] = Counter()
    attached = list(region.get("attached_numeric_symbol_seeds") or [])
    if region.get("source") == "numeric_symbol_seed":
        attached.append({
            "subtype": region.get("subtype"),
            "x": region.get("x"),
            "y": region.get("y"),
            "w": region.get("w"),
            "h": region.get("h"),
            "raw_symbol_bbox": region.get("raw_symbol_bbox"),
            "dot_seed_only": bool(region.get("dot_seed_only")),
        })
    dot_bboxes = [
        bbox for bbox in (
            _numeric_symbol_seed_anchor_bbox(seed)
            for seed in attached
            if _glyph_type_from_seed_subtype(str(seed.get("subtype") or "")) == "dot"
        )
        if bbox is not None
    ]
    for seed in attached:
        subtype = str(seed.get("subtype") or "")
        glyph_type = _glyph_type_from_seed_subtype(subtype)
        if glyph_type is None:
            continue
        if glyph_type != "dot":
            bbox = _bbox_from_any(seed)
            if glyph_type == "plus_minus" and bbox:
                plus_minus_anchors.append({
                    "type": glyph_type,
                    "source": subtype,
                    "bbox": bbox,
                    "validation": {},
                })
            ignored[f"{glyph_type}_diagnostic_only"] += 1
            continue
        if seed.get("dot_seed_only") or region.get("dot_seed_only"):
            ignored["dot_seed_only"] += 1
            continue
        bbox = _numeric_symbol_seed_anchor_bbox(seed)
        if not bbox:
            ignored["invalid_dot_bbox"] += 1
            continue
        if not _decimal_anchor_dot_shape_ok(bbox):
            ignored["non_decimal_dot_shape"] += 1
            continue
        validation = _decimal_anchor_validation_for_region(
            page,
            bbox,
            components,
            region_bbox,
            dot_bboxes=dot_bboxes,
            orientation=orientation,
        )
        if not validation["valid"]:
            if validation.get("colon_like_sibling"):
                ignored["colon_like_dot_anchor"] += 1
            elif validation.get("dense_dot_neighbors"):
                ignored["dense_dot_anchor"] += 1
            else:
                ignored["unvalidated_dot_anchor"] += 1
            continue
        anchors.append({
            "type": glyph_type,
            "source": subtype,
            "bbox": bbox,
            "validation": validation,
        })

    for curve_idx, curve in enumerate(getattr(page, "curves", []) or []):
        bbox = _degree_ring_bbox_from_curve(curve)
        if not bbox:
            continue
        distance = _bbox_distance(bbox, region_bbox)
        if distance > DEGREE_RING_NEAR_TEXT_PT:
            continue
        degree_anchors.append({
            "type": "degree",
            "source": "degree_curve",
            "bbox": bbox,
            "validation": {},
        })
        ignored["degree_diagnostic_only"] += 1

    anchor_types = sorted({str(anchor["type"]) for anchor in anchors})
    if "dot" in anchor_types:
        axis_source = "decimal_point_anchor"
    else:
        axis_source = "region_aspect_fallback"
    return {
        "anchors": anchors,
        "anchor_count": len(anchors),
        "anchor_types": anchor_types,
        "degree_anchors": degree_anchors,
        "degree_anchor_count": len(degree_anchors),
        "plus_minus_anchors": plus_minus_anchors,
        "plus_minus_anchor_count": len(plus_minus_anchors),
        "axis_source": axis_source,
        "axis_orientation": orientation,
        "component_slot_gate": (
            "validated_decimal_point_anchor"
            if anchors else "no_valid_decimal_point_anchor"
        ),
        "ignored_anchor_counts": dict(sorted(ignored.items())),
    }


def _decimal_anchor_validation_for_region(
    page: Any | None,
    anchor_bbox: dict[str, float],
    components: list[dict[str, Any]],
    region_bbox: dict[str, float],
    *,
    dot_bboxes: list[dict[str, float]],
    orientation: str,
) -> dict[str, Any]:
    native_char = _page_has_decimal_char_near(page, anchor_bbox)
    support = _decimal_anchor_component_support(
        anchor_bbox,
        components,
        region_bbox,
        orientation=orientation,
    )
    axis_support = _decimal_anchor_axis_support(
        anchor_bbox,
        components,
        region_bbox,
        orientation=orientation,
    )
    colon_like_sibling = _decimal_anchor_has_cross_axis_sibling(
        anchor_bbox,
        dot_bboxes,
        region_bbox,
        orientation=orientation,
        axis_vector=axis_support.get("axis_vector"),
    )
    dense_dot_neighbors = _decimal_anchor_has_dense_dot_neighbors(
        anchor_bbox,
        dot_bboxes,
        region_bbox,
        orientation=orientation,
    )
    one_sided = bool(support["before_count"]) or bool(support["after_count"])
    two_sided = bool(support["before_count"]) and bool(support["after_count"])
    return {
        "valid": bool(
            axis_support["axis_valid"]
            and not colon_like_sibling
            and not dense_dot_neighbors
        ),
        "colon_like_sibling": bool(colon_like_sibling),
        "dense_dot_neighbors": bool(dense_dot_neighbors),
        "native_decimal_char": bool(native_char),
        "one_sided_component_support": bool(one_sided),
        "two_sided_component_support": bool(two_sided),
        **axis_support,
        **support,
    }


def _decimal_anchor_has_cross_axis_sibling(
    anchor_bbox: dict[str, float],
    dot_bboxes: list[dict[str, float]],
    region_bbox: dict[str, float],
    *,
    orientation: str,
    axis_vector: Any = None,
) -> bool:
    anchor_center = _center(anchor_bbox)
    short_size = (
        float(region_bbox["h"])
        if orientation == "H"
        else float(region_bbox["w"])
    )
    main_tolerance = max(2.0, short_size * 0.35)
    max_cross_delta = max(3.0, short_size * 1.20)
    min_cross_delta = max(
        1.0,
        min(float(anchor_bbox["w"]), float(anchor_bbox["h"])) * 1.25,
    )
    parsed_axis = _axis_unit_from_any(axis_vector)
    for other in dot_bboxes:
        if other is anchor_bbox or other == anchor_bbox:
            continue
        other_center = _center(other)
        if parsed_axis is not None:
            ux, uy = parsed_axis
            main_delta = abs(_dot2(
                other_center[0] - anchor_center[0],
                other_center[1] - anchor_center[1],
                ux,
                uy,
            ))
            cross_delta = abs(_dot2(
                other_center[0] - anchor_center[0],
                other_center[1] - anchor_center[1],
                -uy,
                ux,
            ))
        elif orientation == "H":
            main_delta = abs(anchor_center[0] - other_center[0])
            cross_delta = abs(anchor_center[1] - other_center[1])
        else:
            main_delta = abs(anchor_center[1] - other_center[1])
            cross_delta = abs(anchor_center[0] - other_center[0])
        if (
            main_delta <= main_tolerance
            and min_cross_delta <= cross_delta <= max_cross_delta
        ):
            return True
    return False


def _decimal_anchor_has_dense_dot_neighbors(
    anchor_bbox: dict[str, float],
    dot_bboxes: list[dict[str, float]],
    region_bbox: dict[str, float],
    *,
    orientation: str,
) -> bool:
    anchor_center = _center(anchor_bbox)
    short_size = (
        float(region_bbox["h"])
        if orientation == "H"
        else float(region_bbox["w"])
    )
    main_radius = max(8.0, short_size * DECIMAL_ANCHOR_DENSE_DOT_MAIN_FACTOR)
    cross_radius = max(4.0, short_size * DECIMAL_ANCHOR_DENSE_DOT_CROSS_FACTOR)
    neighbor_count = 0
    for other in dot_bboxes:
        if other is anchor_bbox or other == anchor_bbox:
            continue
        other_center = _center(other)
        if orientation == "H":
            main_delta = abs(anchor_center[0] - other_center[0])
            cross_delta = abs(anchor_center[1] - other_center[1])
        else:
            main_delta = abs(anchor_center[1] - other_center[1])
            cross_delta = abs(anchor_center[0] - other_center[0])
        if main_delta <= main_radius and cross_delta <= cross_radius:
            neighbor_count += 1
            if neighbor_count >= 2:
                return True
    return False


def _decimal_anchor_axis_support(
    anchor_bbox: dict[str, float],
    components: list[dict[str, Any]],
    region_bbox: dict[str, float],
    *,
    orientation: str,
) -> dict[str, Any]:
    anchor_center = _center(anchor_bbox)
    short_size = min(float(region_bbox["w"]), float(region_bbox["h"]))
    max_gap = max(
        DIGIT_SLOT_MIN_SIZE_PT,
        short_size * DECIMAL_AXIS_PAIR_MAX_GAP_FACTOR,
    )
    max_cross = max(
        DIGIT_SLOT_MIN_SIZE_PT,
        short_size * DECIMAL_AXIS_PAIR_MAX_CROSS_FACTOR,
    )
    best: dict[str, Any] | None = None
    component_bboxes = [
        component.get("component_bbox")
        for component in components
        if isinstance(component.get("component_bbox"), dict)
    ]
    for ux, uy, axis_priority in ((1.0, 0.0, 0), (0.0, 1.0, 1)):
        counts: Counter[str] = Counter()
        nearest_gap: dict[str, float | None] = {"before": None, "after": None}
        max_seen_cross = 0.0
        gap_sum = 0.0
        for component_bbox in component_bboxes:
            side_and_gap = _projected_component_side_and_gap(
                component_bbox,
                anchor_bbox,
                ux,
                uy,
            )
            if side_and_gap is None:
                continue
            side, gap = side_and_gap
            if gap > max_gap:
                continue
            cross = abs(_projected_cross_delta(component_bbox, anchor_bbox, ux, uy))
            if cross > max_cross:
                continue
            counts[side] += 1
            max_seen_cross = max(max_seen_cross, cross)
            gap_sum += gap
            current = nearest_gap[side]
            if current is None or gap < current:
                nearest_gap[side] = gap
        if not (counts["before"] and counts["after"]):
            continue
        score = (
            max_seen_cross,
            (nearest_gap["before"] or 0.0) + (nearest_gap["after"] or 0.0),
            axis_priority,
            gap_sum,
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "axis_unit": (ux, uy),
                "counts": counts,
                "nearest_gap": nearest_gap,
                "max_seen_cross": max_seen_cross,
            }
    if best is None:
        return {
            "axis_valid": False,
            "axis_vector": None,
            "axis_angle_deg": None,
            "axis_before_count": 0,
            "axis_after_count": 0,
            "axis_nearest_before_gap_pt": None,
            "axis_nearest_after_gap_pt": None,
            "axis_max_cross_delta_pt": None,
            "axis_cross_scale_pt": round(float(short_size), 3),
        }
    ux, uy = best["axis_unit"]
    counts = best["counts"]
    nearest_gap = best["nearest_gap"]
    max_seen_cross = best["max_seen_cross"]
    return {
        "axis_valid": bool(counts["before"] and counts["after"]),
        "axis_vector": [round(float(ux), 6), round(float(uy), 6)],
        "axis_angle_deg": round(math.degrees(math.atan2(uy, ux)), 3),
        "axis_before_count": int(counts["before"]),
        "axis_after_count": int(counts["after"]),
        "axis_nearest_before_gap_pt": (
            None if nearest_gap["before"] is None
            else round(float(nearest_gap["before"]), 3)
        ),
        "axis_nearest_after_gap_pt": (
            None if nearest_gap["after"] is None
            else round(float(nearest_gap["after"]), 3)
        ),
        "axis_max_cross_delta_pt": round(float(max_seen_cross), 3),
        "axis_cross_scale_pt": round(float(short_size), 3),
    }


def _decimal_shape_axis_for_anchor(
    page: Any | None,
    anchor_bbox: dict[str, float],
) -> dict[str, Any] | None:
    if page is None:
        return None
    best: tuple[float, int, dict[str, Any]] | None = None
    anchor_center = _center(anchor_bbox)
    for curve_idx, curve in enumerate(getattr(page, "curves", []) or []):
        curve_bbox = _curve_bbox(curve)
        if not curve_bbox:
            continue
        curve_center = _center(curve_bbox)
        center_delta = math.hypot(
            curve_center[0] - anchor_center[0],
            curve_center[1] - anchor_center[1],
        )
        if (
            center_delta > DECIMAL_SHAPE_MATCH_CENTER_PT
            and _iou(anchor_bbox, curve_bbox) < 0.45
        ):
            continue
        axis = _decimal_shape_axis_from_curve(curve)
        if axis is None:
            continue
        score = (center_delta, -int(axis["edge_count"]), axis)
        if best is None or score[:2] < best[:2]:
            axis["curve_index"] = curve_idx
            best = score
    return None if best is None else best[2]


def _decimal_shape_axis_from_curve(curve: dict[str, Any]) -> dict[str, Any] | None:
    # decimal-point shape spec 必须读原始 raw pts 是否显式闭合；`_curve_points` 内部
    # 会对未闭合 path 自动补回起点（计算便利），但 spec 层不允许把开放折线/半框
    # 当作合法小数点形状。
    if not _curve_pts_is_explicitly_closed(curve):
        return None
    points = _curve_points(curve)
    if len(points) < 4:
        return None
    segments: list[dict[str, float]] = []
    for start, end in zip(points, points[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        length = math.hypot(dx, dy)
        if length <= 0.25:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        segments.append({
            "length": length,
            "angle": angle,
            "ux": dx / length,
            "uy": dy / length,
        })
    if len(segments) < 4:
        return None
    groups = _parallel_segment_groups(segments)
    paired = [group for group in groups if len(group["segments"]) >= 2]
    # 用户口述的真实小数点形状：两长两短两两垂直。这里强制 spec：
    # (1) 恰好 2 组成对的方向；多于或少于都不是合法小数点。
    # (2) 两组方向接近垂直（默认 ±8°）。
    # (3) 短/长 median 长度比落在 [0.45, 0.90]；接近 1.0 即正方形，没有方向，拒。
    if len(paired) != 2:
        return None
    perp_delta = abs(_angle_delta_180(
        float(paired[0]["angle"]),
        float(paired[1]["angle"]),
    ) - 90.0)
    if perp_delta > DECIMAL_SHAPE_AXIS_PERP_TOLERANCE_DEG:
        return None
    short_group = min(paired, key=lambda group: float(group["median_length"]))
    long_group = max(paired, key=lambda group: float(group["median_length"]))
    short_len = float(short_group["median_length"])
    long_len = float(long_group["median_length"])
    if long_len <= 0.0:
        return None
    short_long_ratio = short_len / long_len
    if (
        short_long_ratio < DECIMAL_SHAPE_AXIS_SHORT_LONG_MIN_RATIO
        or short_long_ratio > DECIMAL_SHAPE_AXIS_SHORT_LONG_MAX_RATIO
    ):
        return None
    ux = float(short_group["ux"])
    uy = float(short_group["uy"])
    ux, uy = _canonical_axis_unit(ux, uy)
    bbox = _curve_bbox(curve)
    if not bbox:
        return None
    main0, main1 = _bbox_projected_interval(bbox, ux, uy)
    cross0, cross1 = _bbox_projected_interval(bbox, -uy, ux)
    return {
        "axis_vector": [round(float(ux), 6), round(float(uy), 6)],
        "axis_angle_deg": round(math.degrees(math.atan2(uy, ux)) % 180.0, 3),
        "short_edge_length_pt": round(float(short_len), 3),
        "long_edge_length_pt": round(float(long_len), 3),
        "short_long_ratio": round(float(short_long_ratio), 6),
        "perp_delta_deg": round(float(perp_delta), 3),
        "edge_count": len(segments),
        "anchor_main_extent_pt": round(float(main1 - main0), 3),
        "anchor_cross_extent_pt": round(float(cross1 - cross0), 3),
    }


def _curve_pts_is_explicitly_closed(
    curve: dict[str, Any],
    *,
    tol: float = 1e-3,
) -> bool:
    """raw `pts` 是否首尾点显式相同（pdfplumber 把 PDF 'h' / 'b' 闭合 op 转成
    回起点 segment 时会保留首尾重复点）。开放折线、半圆、字母轮廓等不会自带这一
    显式重复点，不能被 decimal-shape gate 接受。
    """
    raw_pts = curve.get("pts") or []
    if len(raw_pts) < 4:
        return False
    try:
        first_x = float(raw_pts[0][0])
        first_y = float(raw_pts[0][1])
        last_x = float(raw_pts[-1][0])
        last_y = float(raw_pts[-1][1])
    except (TypeError, ValueError, IndexError):
        return False
    return abs(first_x - last_x) <= tol and abs(first_y - last_y) <= tol


def _curve_points(curve: dict[str, Any]) -> list[tuple[float, float]]:
    raw_points = curve.get("pts") or []
    points: list[tuple[float, float]] = []
    for point in raw_points:
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError, IndexError):
            continue
    if len(points) >= 2 and points[0] != points[-1]:
        points.append(points[0])
    return points


def _parallel_segment_groups(
    segments: list[dict[str, float]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for segment in segments:
        angle = float(segment["angle"])
        matched: dict[str, Any] | None = None
        for group in groups:
            if _angle_delta_180(angle, float(group["angle"])) <= (
                DECIMAL_SHAPE_AXIS_ANGLE_TOLERANCE_DEG
            ):
                matched = group
                break
        if matched is None:
            matched = {
                "angle": angle,
                "segments": [],
            }
            groups.append(matched)
        matched["segments"].append(segment)

    result: list[dict[str, Any]] = []
    for group in groups:
        members = list(group["segments"])
        total_length = sum(float(item["length"]) for item in members)
        if total_length <= 0.0:
            continue
        # 闭合多边形的对边方向相反；按带符号 ux/uy 直接求和会让 (+1,0) 与 (-1,0) 抵消
        # 为 0，导致 _axis_unit_from_any 返回 None，整组被丢弃。先把每条 segment
        # 翻转到组的参考方向（取第一条 segment 的 canonical 方向）再求和。
        ref_ux, ref_uy = _canonical_axis_unit(
            float(members[0]["ux"]), float(members[0]["uy"])
        )
        ux_sum = 0.0
        uy_sum = 0.0
        for item in members:
            seg_ux = float(item["ux"])
            seg_uy = float(item["uy"])
            if seg_ux * ref_ux + seg_uy * ref_uy < 0.0:
                seg_ux, seg_uy = -seg_ux, -seg_uy
            length = float(item["length"])
            ux_sum += seg_ux * length
            uy_sum += seg_uy * length
        unit = _axis_unit_from_any([ux_sum, uy_sum])
        if unit is None:
            continue
        lengths = sorted(float(item["length"]) for item in members)
        result.append({
            "angle": math.degrees(math.atan2(unit[1], unit[0])) % 180.0,
            "ux": unit[0],
            "uy": unit[1],
            "segments": members,
            "median_length": lengths[len(lengths) // 2],
        })
    return result


def _angle_delta_180(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _canonical_axis_unit(ux: float, uy: float) -> tuple[float, float]:
    unit = _axis_unit_from_any([ux, uy])
    if unit is None:
        return 1.0, 0.0
    ux, uy = unit
    if abs(ux) >= abs(uy):
        if ux < 0.0:
            ux, uy = -ux, -uy
    elif uy < 0.0:
        ux, uy = -ux, -uy
    return ux, uy


def _decimal_shape_digit_slot_geometries(
    anchor_bbox: dict[str, float],
    region_bbox: dict[str, float],
    shape_axis: dict[str, Any],
    *,
    primitives: list[dict[str, Any]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    axis_unit = _axis_unit_from_any(shape_axis.get("axis_vector"))
    if axis_unit is None:
        return []
    ux, uy = axis_unit
    vx, vy = -uy, ux
    if not primitives:
        return []
    orientation = "H" if abs(ux) >= abs(uy) else "V"
    rows: list[tuple[str, dict[str, Any]]] = []
    anchor_main0, anchor_main1 = _bbox_projected_interval(anchor_bbox, ux, uy)
    anchor_cross0, anchor_cross1 = _bbox_projected_interval(anchor_bbox, vx, vy)
    anchor_cross_center = (anchor_cross0 + anchor_cross1) / 2.0
    anchor_curve_index = _int_or_none(shape_axis.get("curve_index"))
    candidates = _decimal_shape_topology_candidates(
        primitives,
        ux,
        uy,
        vx,
        vy,
        anchor_bbox=anchor_bbox,
        anchor_main0=anchor_main0,
        anchor_main1=anchor_main1,
        anchor_cross_center=anchor_cross_center,
        anchor_curve_index=anchor_curve_index,
    )
    for role in ("before", "after"):
        cluster = _nearest_decimal_shape_topology_cluster(
            [item for item in candidates if item["role"] == role],
            role=role,
            anchor_main0=anchor_main0,
            anchor_main1=anchor_main1,
        )
        if cluster is None:
            continue
        geometry = _decimal_shape_geometry_from_topology_cluster(
            cluster,
            anchor_bbox,
            ux,
            uy,
            vx,
            vy,
            orientation=orientation,
            role=role,
            anchor_main0=anchor_main0,
            anchor_main1=anchor_main1,
        )
        if geometry is not None:
            rows.append((role, geometry))
    return rows


def _decimal_shape_topology_candidates(
    primitives: list[dict[str, Any]],
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    *,
    anchor_bbox: dict[str, float],
    anchor_main0: float,
    anchor_main1: float,
    anchor_cross_center: float,
    anchor_curve_index: int | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for primitive in primitives:
        kind = str(primitive.get("kind") or "")
        index = _int_or_none(primitive.get("index"))
        if kind == "curve" and anchor_curve_index is not None and index == anchor_curve_index:
            continue
        bbox = primitive.get("bbox")
        if not isinstance(bbox, dict):
            continue
        if _intersection_area(bbox, anchor_bbox) > 0.0:
            continue
        main0, main1, cross0, cross1 = _projected_bbox_extents(bbox, ux, uy, vx, vy)
        role: str
        main_gap: float
        if main1 <= anchor_main0:
            role = "before"
            main_gap = anchor_main0 - main1
        elif main0 >= anchor_main1:
            role = "after"
            main_gap = main0 - anchor_main1
        else:
            continue
        if main_gap > DECIMAL_SHAPE_CLUSTER_MAX_MAIN_DISTANCE_PT:
            continue
        main_size = main1 - main0
        cross_size = cross1 - cross0
        if main_size > DECIMAL_SHAPE_CLUSTER_MAX_MAIN_SIZE_PT:
            continue
        if cross_size > DECIMAL_SHAPE_CLUSTER_MAX_CROSS_SIZE_PT:
            continue
        cross_gap = _range_point_gap(anchor_cross_center, cross0, cross1)
        if cross_gap > DECIMAL_SHAPE_CLUSTER_MAX_CROSS_GAP_PT:
            continue
        candidates.append({
            "primitive": primitive,
            "kind": kind,
            "index": index,
            "role": role,
            "main_start": main0,
            "main_end": main1,
            "cross_start": cross0,
            "cross_end": cross1,
            "main_gap": main_gap,
        })
    return candidates


def _nearest_decimal_shape_topology_cluster(
    candidates: list[dict[str, Any]],
    *,
    role: str,
    anchor_main0: float,
    anchor_main1: float,
) -> list[dict[str, Any]] | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: (
        float(item["main_start"]),
        float(item["cross_start"]),
    ))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_main0 = current_main1 = current_cross0 = current_cross1 = 0.0
    for item in ordered:
        if not current:
            current = [item]
            current_main0 = float(item["main_start"])
            current_main1 = float(item["main_end"])
            current_cross0 = float(item["cross_start"])
            current_cross1 = float(item["cross_end"])
            continue
        gap = float(item["main_start"]) - current_main1
        cross_overlap = _range_overlap_ratio(
            current_cross0,
            current_cross1,
            float(item["cross_start"]),
            float(item["cross_end"]),
        )
        if gap <= DECIMAL_SHAPE_CLUSTER_GAP_PT and cross_overlap >= 0.05:
            current.append(item)
            current_main0 = min(current_main0, float(item["main_start"]))
            current_main1 = max(current_main1, float(item["main_end"]))
            current_cross0 = min(current_cross0, float(item["cross_start"]))
            current_cross1 = max(current_cross1, float(item["cross_end"]))
        else:
            groups.append(current)
            current = [item]
            current_main0 = float(item["main_start"])
            current_main1 = float(item["main_end"])
            current_cross0 = float(item["cross_start"])
            current_cross1 = float(item["cross_end"])
    if current:
        groups.append(current)

    best: tuple[float, int, list[dict[str, Any]]] | None = None
    for group in groups:
        group_main0 = min(float(item["main_start"]) for item in group)
        group_main1 = max(float(item["main_end"]) for item in group)
        group_cross0 = min(float(item["cross_start"]) for item in group)
        group_cross1 = max(float(item["cross_end"]) for item in group)
        if group_main1 - group_main0 > DECIMAL_SHAPE_CLUSTER_MAX_MAIN_SIZE_PT:
            continue
        if group_cross1 - group_cross0 > DECIMAL_SHAPE_CLUSTER_MAX_CROSS_SIZE_PT:
            continue
        gap = (
            anchor_main0 - group_main1
            if role == "before" else group_main0 - anchor_main1
        )
        if gap < 0.0 or gap > DECIMAL_SHAPE_CLUSTER_MAX_MAIN_DISTANCE_PT:
            continue
        score = (gap, -len(group), group)
        if best is None or score[:2] < best[:2]:
            best = score
    return None if best is None else best[2]


def _decimal_shape_geometry_from_topology_cluster(
    cluster: list[dict[str, Any]],
    anchor_bbox: dict[str, float],
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    *,
    orientation: str,
    role: str,
    anchor_main0: float,
    anchor_main1: float,
) -> dict[str, Any] | None:
    if not cluster:
        return None
    main0 = min(float(item["main_start"]) for item in cluster)
    main1 = max(float(item["main_end"]) for item in cluster)
    cross0 = min(float(item["cross_start"]) for item in cluster)
    cross1 = max(float(item["cross_end"]) for item in cluster)
    main_gap = (
        anchor_main0 - main1
        if role == "before" else main0 - anchor_main1
    )
    if main_gap < 0.0:
        return None
    main0 -= DECIMAL_SHAPE_CLUSTER_PAD_PT
    main1 += DECIMAL_SHAPE_CLUSTER_PAD_PT
    cross0 -= DECIMAL_SHAPE_CLUSTER_PAD_PT
    cross1 += DECIMAL_SHAPE_CLUSTER_PAD_PT
    min_main = DECIMAL_SHAPE_CLUSTER_MIN_MAIN_PT
    if main1 - main0 < min_main:
        center = (main0 + main1) / 2.0
        main0 = center - min_main / 2.0
        main1 = center + min_main / 2.0
    if role == "before" and main1 >= anchor_main0:
        shift = main1 - anchor_main0 + 0.05
        main0 -= shift
        main1 -= shift
    elif role == "after" and main0 <= anchor_main1:
        shift = anchor_main1 - main0 + 0.05
        main0 += shift
        main1 += shift
    points = _oriented_rect_points(main0, main1, cross0, cross1, ux, uy, vx, vy)
    bbox = _bbox_from_points(points)
    if not _bbox_size_ok(bbox, min_size=DIGIT_SLOT_MIN_SIZE_PT):
        return None
    anchor_center = _center(anchor_bbox)
    slot_center = _center(bbox)
    signed_main_delta = _dot2(
        slot_center[0] - anchor_center[0],
        slot_center[1] - anchor_center[1],
        ux,
        uy,
    )
    slot_cross = cross1 - cross0
    source_segments = [
        {"kind": item["kind"], "index": int(item["index"])}
        for item in cluster
        if item.get("index") is not None
    ]
    return {
        "bbox": bbox,
        "oriented_quad": [
            [round(float(x), 3), round(float(y), 3)]
            for x, y in points
        ],
        "orientation": orientation,
        "axis_vector": [round(float(ux), 6), round(float(uy), 6)],
        "axis_angle_deg": round(math.degrees(math.atan2(uy, ux)) % 180.0, 3),
        "slot_target_main": round(float(main1 - main0), 3),
        "slot_target_cross": round(float(slot_cross), 3),
        "anchor_main_delta_pt": round(float(abs(signed_main_delta)), 3),
        "signed_anchor_main_delta_pt": round(float(signed_main_delta), 3),
        "anchor_main_gap_pt": round(float(main_gap), 3),
        "anchor_main_ratio": round(
            float(abs(signed_main_delta)) / max(slot_cross, 1e-9),
            6,
        ),
        "cluster_primitive_count": len(source_segments),
        "cluster_main_gap_pt": round(float(main_gap), 3),
        "source_segments": source_segments,
    }


def _projected_bbox_extents(
    bbox: dict[str, float],
    ux: float,
    uy: float,
    vx: float,
    vy: float,
) -> tuple[float, float, float, float]:
    main_values = []
    cross_values = []
    for x, y in _bbox_corners(bbox)[:4]:
        main_values.append(_dot2(x, y, ux, uy))
        cross_values.append(_dot2(x, y, vx, vy))
    return min(main_values), max(main_values), min(cross_values), max(cross_values)


def _oriented_rect_points(
    main0: float,
    main1: float,
    cross0: float,
    cross1: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
) -> list[tuple[float, float]]:
    return [
        (ux * main0 + vx * cross0, uy * main0 + vy * cross0),
        (ux * main1 + vx * cross0, uy * main1 + vy * cross0),
        (ux * main1 + vx * cross1, uy * main1 + vy * cross1),
        (ux * main0 + vx * cross1, uy * main0 + vy * cross1),
    ]


def _range_point_gap(value: float, start: float, end: float) -> float:
    if value < start:
        return start - value
    if value > end:
        return value - end
    return 0.0


def _projected_axis_extent(
    bbox: dict[str, float],
    ux: float,
    uy: float,
) -> float:
    start, end = _bbox_projected_interval(bbox, ux, uy)
    return max(0.0, end - start)


def _page_has_decimal_char_near(
    page: Any | None,
    anchor_bbox: dict[str, float],
) -> bool:
    if page is None:
        return False
    anchor_center = _center(anchor_bbox)
    for char in getattr(page, "chars", []) or []:
        if str(char.get("text") or "") != ".":
            continue
        bbox = _bbox_from_char(char)
        if not bbox:
            continue
        if _bbox_distance(anchor_bbox, bbox) <= DECIMAL_ANCHOR_NATIVE_CHAR_RADIUS_PT:
            return True
        char_center = _center(bbox)
        if math.hypot(
            anchor_center[0] - char_center[0],
            anchor_center[1] - char_center[1],
        ) <= DECIMAL_ANCHOR_NATIVE_CHAR_RADIUS_PT:
            return True
    return False


def _component_native_text_overlap(
    page: Any | None,
    component_bbox: dict[str, float],
) -> dict[str, Any]:
    digit_text: set[str] = set()
    non_digit_text: set[str] = set()
    component_area = float(component_bbox["w"]) * float(component_bbox["h"])
    if page is None or component_area <= 0.0:
        return {
            "has_non_digit": False,
            "digit_text": [],
            "non_digit_text": [],
        }

    for char in getattr(page, "chars", []) or []:
        text = str(char.get("text") or "")
        if not text or text.isspace():
            continue
        bbox = _bbox_from_char(char)
        if not bbox:
            continue
        overlap_area = _intersection_area(component_bbox, bbox)
        if overlap_area <= 0.0:
            continue
        char_area = float(bbox["w"]) * float(bbox["h"])
        overlap_ratio = overlap_area / max(min(component_area, char_area), 1e-9)
        if overlap_ratio < NATIVE_TEXT_OVERLAP_MIN_RATIO:
            continue
        if text.isdigit():
            digit_text.add(text)
        else:
            non_digit_text.add(text)

    return {
        "has_non_digit": bool(non_digit_text),
        "digit_text": sorted(digit_text),
        "non_digit_text": sorted(non_digit_text),
    }


def _decimal_anchor_component_support(
    anchor_bbox: dict[str, float],
    components: list[dict[str, Any]],
    region_bbox: dict[str, float],
    *,
    orientation: str,
) -> dict[str, Any]:
    short_size = (
        float(region_bbox["h"])
        if orientation == "H"
        else float(region_bbox["w"])
    )
    max_cross_delta = max(
        DIGIT_SLOT_MIN_SIZE_PT,
        short_size * DECIMAL_ANCHOR_SUPPORT_CROSS_FACTOR,
    )
    max_main_gap = max(
        DIGIT_SLOT_MIN_SIZE_PT,
        short_size * DECIMAL_ANCHOR_SUPPORT_MAX_MAIN_GAP_FACTOR,
    )
    counts: Counter[str] = Counter()
    nearest_gap: dict[str, float | None] = {"before": None, "after": None}
    for component in components:
        component_bbox = component.get("component_bbox")
        if not isinstance(component_bbox, dict):
            continue
        side_and_gap = _decimal_anchor_component_side_and_gap(
            component_bbox,
            anchor_bbox,
            orientation=orientation,
        )
        if side_and_gap is None:
            continue
        side, main_gap = side_and_gap
        if main_gap > max_main_gap:
            continue
        cross_delta = _axis_cross_delta(
            component_bbox,
            anchor_bbox,
            orientation=orientation,
        )
        if cross_delta > max_cross_delta:
            continue
        counts[side] += 1
        current = nearest_gap[side]
        if current is None or main_gap < current:
            nearest_gap[side] = main_gap
    return {
        "before_count": int(counts["before"]),
        "after_count": int(counts["after"]),
        "nearest_before_gap_pt": (
            None if nearest_gap["before"] is None
            else round(float(nearest_gap["before"]), 3)
        ),
        "nearest_after_gap_pt": (
            None if nearest_gap["after"] is None
            else round(float(nearest_gap["after"]), 3)
        ),
    }


def _decimal_anchor_component_side_and_gap(
    component_bbox: dict[str, float],
    anchor_bbox: dict[str, float],
    *,
    orientation: str,
) -> tuple[str, float] | None:
    if _axis_overlap(component_bbox, anchor_bbox, orientation=orientation) > 0.0:
        return None
    if orientation == "H":
        component_start = float(component_bbox["x"])
        component_end = component_start + float(component_bbox["w"])
        anchor_start = float(anchor_bbox["x"])
        anchor_end = anchor_start + float(anchor_bbox["w"])
    else:
        component_start = float(component_bbox["y"])
        component_end = component_start + float(component_bbox["h"])
        anchor_start = float(anchor_bbox["y"])
        anchor_end = anchor_start + float(anchor_bbox["h"])
    if component_end <= anchor_start:
        return "before", anchor_start - component_end
    if component_start >= anchor_end:
        return "after", component_start - anchor_end
    return None


def _axis_cross_delta(
    a: dict[str, float],
    b: dict[str, float],
    *,
    orientation: str,
) -> float:
    ac = _center(a)
    bc = _center(b)
    if orientation == "H":
        return abs(ac[1] - bc[1])
    return abs(ac[0] - bc[0])


def _component_anchor_debug(
    component_bbox: dict[str, float],
    region_bbox: dict[str, float],
    anchor_context: dict[str, Any],
    *,
    orientation: str,
) -> dict[str, Any]:
    anchors = list(anchor_context.get("anchors") or [])
    if not anchors:
        return {
            "anchor_bound": False,
            "nearest_anchor_type": "",
            "nearest_anchor_main_side": "",
            "component_spans_anchor": False,
            "main_delta_pt": None,
            "signed_main_delta_pt": None,
            "main_gap_pt": None,
            "main_overlap_pt": None,
            "cross_delta_pt": None,
            "cross_ratio": None,
            "main_ratio": None,
            "nearest_anchor_bbox": None,
            "nearest_anchor_validation": None,
            "axis_vector": None,
            "axis_angle_deg": None,
            "axis_cross_scale_pt": None,
        }
    component_center = _center(component_bbox)
    short_size = (
        float(region_bbox["h"])
        if orientation == "H"
        else float(region_bbox["w"])
    )
    best: dict[str, Any] | None = None
    for anchor in anchors:
        anchor_bbox = anchor["bbox"]
        anchor_center = _center(anchor_bbox)
        validation = anchor.get("validation") or {}
        axis_unit = _axis_unit_from_any(validation.get("axis_vector"))
        if axis_unit is not None:
            ux, uy = axis_unit
            signed_main_delta = _dot2(
                component_center[0] - anchor_center[0],
                component_center[1] - anchor_center[1],
                ux,
                uy,
            )
            main_delta = abs(signed_main_delta)
            cross_delta = abs(_dot2(
                component_center[0] - anchor_center[0],
                component_center[1] - anchor_center[1],
                -uy,
                ux,
            ))
            main_overlap = _projected_axis_overlap(
                component_bbox,
                anchor_bbox,
                ux,
                uy,
            )
            side_and_gap = _projected_component_side_and_gap(
                component_bbox,
                anchor_bbox,
                ux,
                uy,
            )
            side = side_and_gap[0] if side_and_gap is not None else "overlap"
            main_gap = side_and_gap[1] if side_and_gap is not None else 0.0
            axis_vector = [round(float(ux), 6), round(float(uy), 6)]
            axis_angle_deg = round(math.degrees(math.atan2(uy, ux)), 3)
            axis_cross_scale_pt = _float_or_none(
                validation.get("axis_cross_scale_pt")
            ) or short_size
        elif orientation == "H":
            signed_main_delta = component_center[0] - anchor_center[0]
            main_delta = abs(signed_main_delta)
            cross_delta = abs(component_center[1] - anchor_center[1])
            main_overlap = _axis_overlap(component_bbox, anchor_bbox, orientation=orientation)
            side = _anchor_side(
                component_bbox,
                anchor_bbox,
                orientation=orientation,
            )
            side_and_gap = _projected_component_side_and_gap(
                component_bbox,
                anchor_bbox,
                1.0,
                0.0,
            )
            main_gap = side_and_gap[1] if side_and_gap is not None else 0.0
            axis_vector = [1.0, 0.0]
            axis_angle_deg = 0.0
            axis_cross_scale_pt = short_size
        else:
            signed_main_delta = component_center[1] - anchor_center[1]
            main_delta = abs(signed_main_delta)
            cross_delta = abs(component_center[0] - anchor_center[0])
            main_overlap = _axis_overlap(component_bbox, anchor_bbox, orientation=orientation)
            side = _anchor_side(
                component_bbox,
                anchor_bbox,
                orientation=orientation,
            )
            side_and_gap = _projected_component_side_and_gap(
                component_bbox,
                anchor_bbox,
                0.0,
                1.0,
            )
            main_gap = side_and_gap[1] if side_and_gap is not None else 0.0
            axis_vector = [0.0, 1.0]
            axis_angle_deg = 90.0
            axis_cross_scale_pt = short_size
        score = (cross_delta, main_delta)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "type": str(anchor.get("type") or ""),
                "main_delta": main_delta,
                "signed_main_delta": signed_main_delta,
                "main_gap": main_gap,
                "cross_delta": cross_delta,
                "main_overlap": main_overlap,
                "side": side,
                "bbox": anchor_bbox,
                "validation": validation,
                "axis_vector": axis_vector,
                "axis_angle_deg": axis_angle_deg,
                "axis_cross_scale_pt": axis_cross_scale_pt,
            }
    assert best is not None
    axis_cross_scale_pt = float(best["axis_cross_scale_pt"] or short_size)
    cross_ratio = best["cross_delta"] / max(axis_cross_scale_pt, 1e-9)
    main_ratio = best["main_delta"] / max(axis_cross_scale_pt, 1e-9)
    max_main_gap = axis_cross_scale_pt * DECIMAL_AXIS_COMPONENT_MAX_GAP_FACTOR
    return {
        "anchor_bound": (
            cross_ratio <= DIMENSION_ANCHOR_CROSS_BAND_FACTOR
            and main_ratio <= DIMENSION_ANCHOR_MAX_MAIN_DELTA_FACTOR
            and best["main_gap"] <= max_main_gap
        ),
        "nearest_anchor_type": best["type"],
        "nearest_anchor_main_side": best["side"],
        "component_spans_anchor": best["main_overlap"] > 0.0,
        "main_delta_pt": round(float(best["main_delta"]), 3),
        "signed_main_delta_pt": round(float(best["signed_main_delta"]), 3),
        "main_gap_pt": round(float(best["main_gap"]), 3),
        "main_overlap_pt": round(float(best["main_overlap"]), 3),
        "cross_delta_pt": round(float(best["cross_delta"]), 3),
        "cross_ratio": round(float(cross_ratio), 6),
        "main_ratio": round(float(main_ratio), 6),
        "nearest_anchor_bbox": _round_bbox(best["bbox"]),
        "nearest_anchor_validation": dict(best["validation"]),
        "axis_vector": list(best["axis_vector"]),
        "axis_angle_deg": best["axis_angle_deg"],
        "axis_cross_scale_pt": round(float(axis_cross_scale_pt), 3),
    }


def _vector_text_primitives(page: Any) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    order = 0
    for kind, items in (
        ("line", getattr(page, "lines", []) or []),
        ("curve", getattr(page, "curves", []) or []),
        ("rect", getattr(page, "rects", []) or []),
    ):
        for index, raw in enumerate(items):
            primitive = _primitive_from_vector_item(raw, kind=kind, index=index)
            if primitive is not None:
                primitive["order"] = order
                primitives.append(primitive)
                order += 1
    return primitives


def _vector_text_primitive_index(page: Any) -> dict[str, Any]:
    primitives = _vector_text_primitives(page)
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for primitive in primitives:
        cell = _cell_for_point(
            _center(primitive["bbox"]),
            cell_size=DIGIT_COMPONENT_PRIMITIVE_INDEX_CELL_PT,
        )
        buckets.setdefault(cell, []).append(primitive)
    return {
        "primitives": primitives,
        "buckets": buckets,
        "cell_size": DIGIT_COMPONENT_PRIMITIVE_INDEX_CELL_PT,
    }


def _query_text_primitives(
    primitive_index: dict[str, Any],
    region_bbox: dict[str, float],
) -> list[dict[str, Any]]:
    buckets = primitive_index.get("buckets") or {}
    if not buckets:
        return list(primitive_index.get("primitives") or [])
    cell_size = float(primitive_index.get("cell_size") or DIGIT_COMPONENT_PRIMITIVE_INDEX_CELL_PT)
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for cell in _cells_for_bbox(region_bbox, cell_size=cell_size):
        for primitive in buckets.get(cell, []):
            key = (str(primitive.get("kind") or ""), int(primitive.get("index") or -1))
            if key in seen:
                continue
            seen.add(key)
            result.append(primitive)
    result.sort(key=lambda item: int(item.get("order") or 0))
    return result


def _primitive_from_vector_item(
    raw: dict[str, Any],
    *,
    kind: str,
    index: int,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    points: list[tuple[float, float]] = []
    if kind == "line":
        try:
            points = [
                (float(raw.get("x0")), float(raw.get("top", raw.get("y0")))),
                (float(raw.get("x1")), float(raw.get("bottom", raw.get("y1")))),
            ]
        except (TypeError, ValueError):
            return None
    elif kind == "curve":
        for point in raw.get("pts") or raw.get("points") or []:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError, IndexError):
                continue
        if len(points) < 2:
            bbox = _bbox_from_any(raw)
            if bbox is None:
                return None
            points = _bbox_corners(bbox)
    else:
        bbox = _bbox_from_any(raw)
        if bbox is None:
            return None
        points = _bbox_corners(bbox)
    if len(points) < 2:
        return None
    bbox = _bbox_from_points(points)
    if kind == "line":
        bbox = _expand_bbox_by_linewidth(bbox, raw)
    if max(bbox["w"], bbox["h"]) < DIGIT_COMPONENT_SLOT_MIN_PRIMITIVE_PT:
        return None
    return {
        "kind": kind,
        "index": int(index),
        "bbox": bbox,
        "points": points,
        "path": list(raw.get("path") or []),
    }


def _primitive_can_belong_to_text_region(
    primitive: dict[str, Any],
    region_bbox: dict[str, float],
    *,
    orientation: str,
) -> bool:
    bbox = primitive["bbox"]
    if not _center_inside(bbox, region_bbox):
        return False
    region_short = min(float(region_bbox["w"]), float(region_bbox["h"]))
    region_long = max(float(region_bbox["w"]), float(region_bbox["h"]))
    primitive_long = max(float(bbox["w"]), float(bbox["h"]))
    if primitive_long > max(region_long, region_short * 2.5):
        return False
    if orientation == "H":
        if float(bbox["h"]) > float(region_bbox["h"]) * 1.35:
            return False
    else:
        if float(bbox["w"]) > float(region_bbox["w"]) * 1.35:
            return False
    return _intersection_area(bbox, region_bbox) > 0.0


def _component_slots_from_primitives(
    primitives: list[dict[str, Any]],
    region_bbox: dict[str, float],
    *,
    orientation: str,
) -> list[dict[str, Any]]:
    if orientation == "H":
        axis = "x"
        cross = "y"
        gap_limit = min(
            DIGIT_SLOT_MAX_SIZE_PT,
            max(0.75, float(region_bbox["h"]) * DIGIT_COMPONENT_SLOT_GAP_FACTOR),
        )
    else:
        axis = "y"
        cross = "x"
        gap_limit = min(
            DIGIT_SLOT_MAX_SIZE_PT,
            max(0.75, float(region_bbox["w"]) * DIGIT_COMPONENT_SLOT_GAP_FACTOR),
        )

    ordered = sorted(
        primitives,
        key=lambda item: (
            float(item["bbox"][axis]),
            float(item["bbox"][cross]),
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bbox: dict[str, float] | None = None
    for primitive in ordered:
        bbox = primitive["bbox"]
        if not current or current_bbox is None:
            current = [primitive]
            current_bbox = dict(bbox)
            continue
        if _same_component_projection(
            current_bbox,
            bbox,
            orientation=orientation,
            gap_limit=gap_limit,
        ):
            current.append(primitive)
            current_bbox = _union_bbox(current_bbox, bbox)
        else:
            groups.append(current)
            current = [primitive]
            current_bbox = dict(bbox)
    if current:
        groups.append(current)

    components: list[dict[str, Any]] = []
    for group in groups:
        component_bbox: dict[str, float] | None = None
        source_segments: list[dict[str, Any]] = []
        for primitive in group:
            component_bbox = (
                _union_bbox(component_bbox, primitive["bbox"])
                if component_bbox else dict(primitive["bbox"])
            )
            source_segments.append({
                "kind": primitive["kind"],
                "index": primitive["index"],
            })
        if component_bbox is None:
            continue
        if not _component_bbox_can_be_digit_slot(
            component_bbox,
            region_bbox,
            orientation=orientation,
        ):
            continue
        raw_component_bbox = dict(component_bbox)
        component_bbox = _clip_bbox_to_bbox(component_bbox, region_bbox)
        if not _bbox_size_ok(
            component_bbox,
            min_size=DIGIT_COMPONENT_SLOT_MIN_PRIMITIVE_PT,
        ):
            continue
        slot_bbox, expand_debug = _expanded_component_slot_bbox(
            component_bbox,
            region_bbox,
            orientation=orientation,
        )
        components.append({
            "bbox": slot_bbox,
            "component_bbox": component_bbox,
            "raw_component_bbox": raw_component_bbox,
            "component_clipped_to_region": _bbox_changed(
                raw_component_bbox,
                component_bbox,
            ),
            "primitive_count": len(group),
            "source_segments": source_segments,
            "primitives": list(group),
            **expand_debug,
        })
    return components


def _component_bbox_can_be_digit_slot(
    component_bbox: dict[str, float],
    region_bbox: dict[str, float],
    *,
    orientation: str,
) -> bool:
    if orientation == "H":
        main_size = float(component_bbox["w"])
        digit_scale = float(region_bbox["h"])
    else:
        main_size = float(component_bbox["h"])
        digit_scale = float(region_bbox["w"])
    max_main_size = min(
        DIGIT_SLOT_MAX_SIZE_PT,
        max(
            DIGIT_SLOT_MIN_SIZE_PT,
            digit_scale * DIGIT_COMPONENT_MAX_MAIN_AXIS_FACTOR,
        ),
    )
    if main_size > max_main_size:
        return False
    component_area = float(component_bbox["w"]) * float(component_bbox["h"])
    if component_area <= 0.0:
        return False
    region_overlap_ratio = (
        _intersection_area(component_bbox, region_bbox) / component_area
    )
    return region_overlap_ratio >= DIGIT_COMPONENT_MIN_REGION_OVERLAP


def _same_component_projection(
    left: dict[str, float],
    right: dict[str, float],
    *,
    orientation: str,
    gap_limit: float,
) -> bool:
    if orientation == "H":
        left_end = float(left["x"]) + float(left["w"])
        right_start = float(right["x"])
        gap = right_start - left_end
        left_start = float(left["x"])
        right_end = float(right["x"]) + float(right["w"])
        cross_overlap = _range_overlap_ratio(
            float(left["y"]),
            float(left["y"]) + float(left["h"]),
            float(right["y"]),
            float(right["y"]) + float(right["h"]),
        )
    else:
        left_end = float(left["y"]) + float(left["h"])
        right_start = float(right["y"])
        gap = right_start - left_end
        left_start = float(left["y"])
        right_end = float(right["y"]) + float(right["h"])
        cross_overlap = _range_overlap_ratio(
            float(left["x"]),
            float(left["x"]) + float(left["w"]),
            float(right["x"]),
            float(right["x"]) + float(right["w"]),
        )
    main_overlap = min(left_end, right_end) - max(left_start, right_start)
    if main_overlap > 0.0:
        return True
    return gap <= gap_limit and cross_overlap >= 0.15


def _expanded_component_slot_bbox(
    component_bbox: dict[str, float],
    region_bbox: dict[str, float],
    *,
    orientation: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    bbox = dict(component_bbox)
    if orientation == "H":
        target_w = _clamp(
            max(float(bbox["w"]), float(region_bbox["h"]) * 0.42),
            DIGIT_SLOT_MIN_SIZE_PT,
            min(DIGIT_SLOT_MAX_SIZE_PT, float(region_bbox["w"])),
        )
        target_h = _clamp(
            max(float(bbox["h"]), float(region_bbox["h"]) * 0.82),
            DIGIT_SLOT_MIN_SIZE_PT,
            min(DIGIT_SLOT_MAX_SIZE_PT, float(region_bbox["h"])),
        )
    else:
        target_w = _clamp(
            max(float(bbox["w"]), float(region_bbox["w"]) * 0.82),
            DIGIT_SLOT_MIN_SIZE_PT,
            min(DIGIT_SLOT_MAX_SIZE_PT, float(region_bbox["w"])),
        )
        target_h = _clamp(
            max(float(bbox["h"]), float(region_bbox["w"]) * 0.42),
            DIGIT_SLOT_MIN_SIZE_PT,
            min(DIGIT_SLOT_MAX_SIZE_PT, float(region_bbox["h"])),
        )
    cx, cy = _center(bbox)
    clipped = _clip_bbox_to_bbox(
        {
            "x": cx - target_w / 2.0,
            "y": cy - target_h / 2.0,
            "w": target_w,
            "h": target_h,
        },
        region_bbox,
    )
    return clipped, {
        "slot_expand_strategy": "component_bbox_plus_region_height",
        "slot_target_w": float(target_w),
        "slot_target_h": float(target_h),
    }


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> dict[str, float] | None:
    x_min = min(x0, x1)
    y_min = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x_min, "y": y_min, "w": w, "h": h}


def _bbox_from_points(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return {
        "x": x0,
        "y": y0,
        "w": max(x1 - x0, 1e-9),
        "h": max(y1 - y0, 1e-9),
    }


def _expand_bbox_by_linewidth(
    bbox: dict[str, float],
    raw: dict[str, Any],
) -> dict[str, float]:
    try:
        linewidth = float(raw.get("linewidth") or 0.0)
    except (TypeError, ValueError):
        return bbox
    if linewidth <= 0.0:
        return bbox
    pad = min(linewidth, DIGIT_SLOT_MAX_SIZE_PT) / 2.0
    return {
        "x": float(bbox["x"]) - pad,
        "y": float(bbox["y"]) - pad,
        "w": float(bbox["w"]) + linewidth,
        "h": float(bbox["h"]) + linewidth,
    }


def _bbox_corners(bbox: dict[str, float]) -> list[tuple[float, float]]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]


def _region_can_host_digit_slots(bbox: dict[str, float]) -> bool:
    try:
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return False
    if min(w, h) < DIGIT_SLOT_MIN_REGION_SHORT_PT:
        return False
    if max(w, h) > DIGIT_SLOT_MAX_REGION_LONG_PT:
        return False
    return True


def _region_orientation(bbox: dict[str, float]) -> str:
    return "H" if float(bbox["w"]) >= float(bbox["h"]) else "V"


def _adjacent_digit_slot_bboxes(
    region_bbox: dict[str, float],
    seed_bbox: dict[str, float],
    *,
    orientation: str,
) -> list[tuple[str, dict[str, float]]]:
    region_x = float(region_bbox["x"])
    region_y = float(region_bbox["y"])
    region_w = float(region_bbox["w"])
    region_h = float(region_bbox["h"])
    seed_x = float(seed_bbox["x"])
    seed_y = float(seed_bbox["y"])
    seed_w = float(seed_bbox["w"])
    seed_h = float(seed_bbox["h"])
    if orientation == "H":
        slot_w = _clamp(region_h * 0.75, DIGIT_SLOT_MIN_SIZE_PT, DIGIT_SLOT_MAX_SIZE_PT)
        return [
            ("before", _clip_bbox_to_bbox(
                {"x": seed_x - slot_w, "y": region_y, "w": slot_w, "h": region_h},
                region_bbox,
            )),
            ("after", _clip_bbox_to_bbox(
                {"x": seed_x + seed_w, "y": region_y, "w": slot_w, "h": region_h},
                region_bbox,
            )),
        ]
    slot_h = _clamp(region_w * 0.75, DIGIT_SLOT_MIN_SIZE_PT, DIGIT_SLOT_MAX_SIZE_PT)
    return [
        ("before", _clip_bbox_to_bbox(
            {"x": region_x, "y": seed_y - slot_h, "w": region_w, "h": slot_h},
            region_bbox,
        )),
        ("after", _clip_bbox_to_bbox(
            {"x": region_x, "y": seed_y + seed_h, "w": region_w, "h": slot_h},
            region_bbox,
        )),
    ]


def _clip_bbox_to_bbox(
    bbox: dict[str, float],
    bounds: dict[str, float],
) -> dict[str, float]:
    x0 = max(float(bbox["x"]), float(bounds["x"]))
    y0 = max(float(bbox["y"]), float(bounds["y"]))
    x1 = min(
        float(bbox["x"]) + float(bbox["w"]),
        float(bounds["x"]) + float(bounds["w"]),
    )
    y1 = min(
        float(bbox["y"]) + float(bbox["h"]),
        float(bounds["y"]) + float(bounds["h"]),
    )
    return {"x": x0, "y": y0, "w": max(0.0, x1 - x0), "h": max(0.0, y1 - y0)}


def _bbox_size_ok(bbox: dict[str, float], *, min_size: float) -> bool:
    try:
        return float(bbox["w"]) >= min_size and float(bbox["h"]) >= min_size
    except (KeyError, TypeError, ValueError):
        return False


def _bbox_changed(
    left: dict[str, float],
    right: dict[str, float],
    *,
    epsilon: float = 1e-6,
) -> bool:
    return any(
        abs(float(left[key]) - float(right[key])) > epsilon
        for key in ("x", "y", "w", "h")
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-9:
        return None
    return round(float(numerator) / float(denominator), 6)


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "0")
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _empty_component_cluster_slot_filter_config() -> dict[str, Any]:
    return {
        "distance_max": None,
        "top_k": 0,
        "library_path": "",
        "physical_band_path": "",
        "physical_band": None,
    }


def _component_cluster_slot_filter_config() -> dict[str, Any]:
    distance_max = _cluster_slot_distance_max_from_env()
    top_k = _cluster_slot_top_k_from_env()
    library_path = str(
        os.environ.get(DIGIT_COMPONENT_CLUSTER_SLOT_LIBRARY_ENV, "") or ""
    ).strip()
    physical_band_path, physical_band = _cluster_slot_physical_band_from_env()
    if distance_max is not None and not library_path:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_DISTANCE_MAX_ENV} requires "
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_LIBRARY_ENV}"
        )
    if top_k > 0 and not library_path:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_TOP_K_ENV} requires "
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_LIBRARY_ENV}"
        )
    return {
        "distance_max": distance_max,
        "top_k": top_k,
        "library_path": library_path,
        "physical_band_path": physical_band_path,
        "physical_band": physical_band,
    }


def _component_cluster_slot_filtering_enabled(config: dict[str, Any]) -> bool:
    return (
        bool(config.get("physical_band"))
        or bool(config.get("library_path"))
        or config.get("distance_max") is not None
        or int(config.get("top_k") or 0) > 0
    )


def _cluster_slot_distance_max_from_env() -> float | None:
    raw = str(
        os.environ.get(DIGIT_COMPONENT_CLUSTER_SLOT_DISTANCE_MAX_ENV, "none")
        or "none"
    ).strip().lower()
    if raw in {"", "none"}:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_DISTANCE_MAX_ENV} must be 'none' "
            "or a non-negative number"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_DISTANCE_MAX_ENV} must be 'none' "
            "or a non-negative number"
        )
    return value


def _cluster_slot_top_k_from_env() -> int:
    raw = str(os.environ.get(DIGIT_COMPONENT_CLUSTER_SLOT_TOP_K_ENV, "0") or "0")
    raw = raw.strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_TOP_K_ENV} must be a non-negative integer"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_TOP_K_ENV} must be a non-negative integer"
        )
    return value


def _cluster_slot_physical_band_from_env() -> tuple[str, dict[str, Any] | None]:
    raw = str(
        os.environ.get(DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV, "none")
        or "none"
    ).strip()
    if raw.lower() in {"", "none"}:
        return "", None
    if not os.path.exists(raw):
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} path does not exist: "
            f"{raw}"
        )
    try:
        with open(raw, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} could not be read: "
            f"{raw}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} must point to JSON"
        ) from exc
    return raw, _validated_cluster_slot_physical_band(payload)


def _validated_cluster_slot_physical_band(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} JSON must be an object"
        )
    required = (
        "width_band",
        "height_band",
        "aspect_band",
        "segment_count_band",
        "vertex_count_band",
        "sample_count",
        "source",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} missing keys: "
            f"{', '.join(missing)}"
        )
    validated = dict(payload)
    for key in (
        "width_band",
        "height_band",
        "aspect_band",
        "segment_count_band",
        "vertex_count_band",
        "primitive_count_band",
    ):
        if key not in payload:
            continue
        validated[key] = _validated_numeric_band(key, payload[key])
    try:
        sample_count = int(payload["sample_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} sample_count "
            "must be an integer"
        ) from exc
    if sample_count <= 0:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} sample_count "
            "must be positive"
        )
    validated["sample_count"] = sample_count
    return validated


def _validated_numeric_band(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} {name} "
            "must be [low, high]"
        )
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} {name} "
            "must contain numbers"
        ) from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(
            f"{DIGIT_COMPONENT_CLUSTER_SLOT_PHYSICAL_BAND_ENV} {name} "
            "must contain finite low <= high"
        )
    return (low, high)


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(float(bbox["x"]), 3),
        "y": round(float(bbox["y"]), 3),
        "w": round(float(bbox["w"]), 3),
        "h": round(float(bbox["h"]), 3),
    }


def _quad_from_bbox(bbox: dict[str, float]) -> list[list[float]]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    return [
        [round(x, 3), round(y, 3)],
        [round(x + w, 3), round(y, 3)],
        [round(x + w, 3), round(y + h, 3)],
        [round(x, 3), round(y + h, 3)],
    ]


def _center(bbox: dict[str, float]) -> tuple[float, float]:
    return (
        float(bbox["x"]) + float(bbox["w"]) / 2.0,
        float(bbox["y"]) + float(bbox["h"]) / 2.0,
    )


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _axis_unit_from_any(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        ux = float(value[0])
        uy = float(value[1])
    except (TypeError, ValueError):
        return None
    length = math.hypot(ux, uy)
    if length <= 1e-9 or not math.isfinite(length):
        return None
    return (ux / length, uy / length)


def _dot2(x: float, y: float, ux: float, uy: float) -> float:
    return float(x) * float(ux) + float(y) * float(uy)


def _bbox_projected_interval(
    bbox: dict[str, float],
    ux: float,
    uy: float,
) -> tuple[float, float]:
    projections = [
        _dot2(point[0], point[1], ux, uy)
        for point in _bbox_corners(bbox)[:4]
    ]
    return min(projections), max(projections)


def _projected_axis_overlap(
    a: dict[str, float],
    b: dict[str, float],
    ux: float,
    uy: float,
) -> float:
    a0, a1 = _bbox_projected_interval(a, ux, uy)
    b0, b1 = _bbox_projected_interval(b, ux, uy)
    return max(0.0, min(a1, b1) - max(a0, b0))


def _projected_main_gap(
    a: dict[str, float],
    b: dict[str, float],
    ux: float,
    uy: float,
) -> float | None:
    a0, a1 = _bbox_projected_interval(a, ux, uy)
    b0, b1 = _bbox_projected_interval(b, ux, uy)
    if a1 <= b0:
        return b0 - a1
    if b1 <= a0:
        return a0 - b1
    return None


def _projected_cross_delta(
    a: dict[str, float],
    b: dict[str, float],
    ux: float,
    uy: float,
) -> float:
    ac = _center(a)
    bc = _center(b)
    return _dot2(ac[0] - bc[0], ac[1] - bc[1], -uy, ux)


def _projected_component_side_and_gap(
    component_bbox: dict[str, float],
    anchor_bbox: dict[str, float],
    ux: float,
    uy: float,
) -> tuple[str, float] | None:
    component_start, component_end = _bbox_projected_interval(
        component_bbox,
        ux,
        uy,
    )
    anchor_start, anchor_end = _bbox_projected_interval(anchor_bbox, ux, uy)
    if component_end <= anchor_start:
        return "before", anchor_start - component_end
    if component_start >= anchor_end:
        return "after", component_start - anchor_end
    return None


def _oriented_component_slot_geometry(
    component_bbox: dict[str, float],
    region_bbox: dict[str, float],
    anchor_debug: dict[str, Any],
) -> dict[str, Any]:
    axis_unit = _axis_unit_from_any(anchor_debug.get("axis_vector"))
    if axis_unit is None:
        bbox = dict(component_bbox)
        return {
            "bbox": bbox,
            "oriented_quad": _quad_from_bbox(bbox),
            "slot_target_main": round(float(bbox["w"]), 3),
            "slot_target_cross": round(float(bbox["h"]), 3),
            "slot_expand_strategy": "component_bbox_axis_fallback",
        }

    ux, uy = axis_unit
    vx, vy = -uy, ux
    component_main_start, component_main_end = _bbox_projected_interval(
        component_bbox,
        ux,
        uy,
    )
    component_cross_start, component_cross_end = _bbox_projected_interval(
        component_bbox,
        vx,
        vy,
    )
    component_main = max(component_main_end - component_main_start, 1e-9)
    component_cross = max(component_cross_end - component_cross_start, 1e-9)
    scale = _float_or_none(anchor_debug.get("axis_cross_scale_pt"))
    if scale is None:
        scale = min(float(region_bbox["w"]), float(region_bbox["h"]))
    target_main = _clamp(
        max(component_main, scale * 0.42),
        DIGIT_SLOT_MIN_SIZE_PT,
        DIGIT_SLOT_MAX_SIZE_PT,
    )
    target_cross = _clamp(
        max(component_cross, scale * 0.82),
        DIGIT_SLOT_MIN_SIZE_PT,
        DIGIT_SLOT_MAX_SIZE_PT,
    )
    cx, cy = _center(component_bbox)
    half_main = target_main / 2.0
    half_cross = target_cross / 2.0
    points = [
        (
            cx - ux * half_main - vx * half_cross,
            cy - uy * half_main - vy * half_cross,
        ),
        (
            cx + ux * half_main - vx * half_cross,
            cy + uy * half_main - vy * half_cross,
        ),
        (
            cx + ux * half_main + vx * half_cross,
            cy + uy * half_main + vy * half_cross,
        ),
        (
            cx - ux * half_main + vx * half_cross,
            cy - uy * half_main + vy * half_cross,
        ),
    ]
    return {
        "bbox": _bbox_from_points(points),
        "oriented_quad": [
            [round(float(x), 3), round(float(y), 3)]
            for x, y in points
        ],
        "slot_target_main": round(float(target_main), 3),
        "slot_target_cross": round(float(target_cross), 3),
        "slot_expand_strategy": "component_bbox_plus_decimal_axis",
    }


def _bbox_distance(a: dict[str, float], b: dict[str, float]) -> float:
    ax0 = float(a["x"])
    ay0 = float(a["y"])
    ax1 = ax0 + float(a["w"])
    ay1 = ay0 + float(a["h"])
    bx0 = float(b["x"])
    by0 = float(b["y"])
    bx1 = bx0 + float(b["w"])
    by1 = by0 + float(b["h"])
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _axis_overlap(
    a: dict[str, float],
    b: dict[str, float],
    *,
    orientation: str,
) -> float:
    if orientation == "H":
        a0 = float(a["x"])
        a1 = a0 + float(a["w"])
        b0 = float(b["x"])
        b1 = b0 + float(b["w"])
    else:
        a0 = float(a["y"])
        a1 = a0 + float(a["h"])
        b0 = float(b["y"])
        b1 = b0 + float(b["h"])
    return max(0.0, min(a1, b1) - max(a0, b0))


def _anchor_side(
    component_bbox: dict[str, float],
    anchor_bbox: dict[str, float],
    *,
    orientation: str,
) -> str:
    if _axis_overlap(component_bbox, anchor_bbox, orientation=orientation) > 0.0:
        return "overlap"
    component_center = _center(component_bbox)
    anchor_center = _center(anchor_bbox)
    if orientation == "H":
        return "before" if component_center[0] < anchor_center[0] else "after"
    return "before" if component_center[1] < anchor_center[1] else "after"


def _center_inside(a: dict[str, float], b: dict[str, float]) -> bool:
    cx, cy = _center(a)
    return (
        float(b["x"]) <= cx <= float(b["x"]) + float(b["w"])
        and float(b["y"]) <= cy <= float(b["y"]) + float(b["h"])
    )


def _bbox_center_inside_any(
    bbox: dict[str, float],
    regions: list[dict[str, Any]],
) -> bool:
    cx, cy = _center(bbox)
    for raw in regions:
        region = _bbox_from_any(raw)
        if not region:
            continue
        if (
            float(region["x"]) <= cx <= float(region["x"]) + float(region["w"])
            and float(region["y"]) <= cy <= float(region["y"]) + float(region["h"])
        ):
            return True
    return False


def _union_bbox(
    left: dict[str, float] | None,
    right: dict[str, float],
) -> dict[str, float]:
    if left is None:
        return dict(right)
    x0 = min(float(left["x"]), float(right["x"]))
    y0 = min(float(left["y"]), float(right["y"]))
    x1 = max(float(left["x"]) + float(left["w"]), float(right["x"]) + float(right["w"]))
    y1 = max(float(left["y"]) + float(left["h"]), float(right["y"]) + float(right["h"]))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    ax1 = float(a["x"]) + float(a["w"])
    ay1 = float(a["y"]) + float(a["h"])
    bx1 = float(b["x"]) + float(b["w"])
    by1 = float(b["y"]) + float(b["h"])
    w = max(0.0, min(ax1, bx1) - max(float(a["x"]), float(b["x"])))
    h = max(0.0, min(ay1, by1) - max(float(a["y"]), float(b["y"])))
    return w * h


def _range_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return overlap / max(min(a1 - a0, b1 - b0), 1e-9)


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax0 = float(a["x"])
    ay0 = float(a["y"])
    ax1 = ax0 + float(a["w"])
    ay1 = ay0 + float(a["h"])
    bx0 = float(b["x"])
    by0 = float(b["y"])
    bx1 = bx0 + float(b["w"])
    by1 = by0 + float(b["h"])
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = float(a["w"]) * float(a["h"]) + float(b["w"]) * float(b["h"]) - inter
    return inter / union if union > 0.0 else 0.0


def _dedupe_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str, tuple[int, int]], list[dict[str, Any]]] = {}
    for token in tokens:
        duplicate = False
        glyph_type = str(token.get("glyph_type") or "")
        source_key = _dedupe_source_key(token)
        candidates: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for cell in _cells_for_bbox(token["bbox"], cell_size=DEDUPE_INDEX_CELL_PT):
            for existing in buckets.get((glyph_type, source_key, cell), []):
                existing_id = id(existing)
                if existing_id in seen_ids:
                    continue
                seen_ids.add(existing_id)
                candidates.append(existing)
        for existing in candidates:
            if _iou(token["bbox"], existing["bbox"]) >= DEDUPE_IOU:
                duplicate = True
                break
        if not duplicate:
            kept.append(token)
            for cell in _cells_for_bbox(token["bbox"], cell_size=DEDUPE_INDEX_CELL_PT):
                buckets.setdefault((glyph_type, source_key, cell), []).append(token)
    for idx, token in enumerate(kept):
        token["token_id"] = f"p{int(token['page_index']) + 1:03d}_glyph_{idx:06d}"
    return kept


def _dedupe_source_key(token: dict[str, Any]) -> str:
    checks = token.get("strict_checks")
    source = checks.get("source") if isinstance(checks, dict) else None
    if source == DIGIT_STRUCTURAL_FEATURES_SOURCE:
        return DIGIT_STRUCTURAL_FEATURES_SOURCE
    return ""


def _cell_for_point(point: tuple[float, float], *, cell_size: float) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / cell_size),
        math.floor(float(point[1]) / cell_size),
    )


def _cells_for_bbox(
    bbox: dict[str, float],
    *,
    cell_size: float,
) -> list[tuple[int, int]]:
    x0 = float(bbox["x"])
    y0 = float(bbox["y"])
    x1 = x0 + float(bbox["w"])
    y1 = y0 + float(bbox["h"])
    ix0 = math.floor(x0 / cell_size)
    iy0 = math.floor(y0 / cell_size)
    ix1 = math.floor(x1 / cell_size)
    iy1 = math.floor(y1 / cell_size)
    return [
        (ix, iy)
        for ix in range(ix0, ix1 + 1)
        for iy in range(iy0, iy1 + 1)
    ]
