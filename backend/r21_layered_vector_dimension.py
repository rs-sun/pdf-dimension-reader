"""R21 layered vector-dimension reader shadow pipeline.

The module is dump-only. Every public builder returns structures with
``consumer_allowed=False`` and none of these rows may be routed into final
``dimensions`` until a later graduation gate approves that consume path.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from statistics import median
from typing import Any

from decimal_colon_filter import filter_colon_decimal_pairs, xywh_from_any
from colon_dot_detector import detect_large_dot_marks
from degree_polyline_strict import (
    candidate_from_ring,
    detect_closed_rings,
    detect_degree_polylines,
    extract_line_segments,
)
from gdt_vector_inversion import GDTVectorInverter
from page_layout_v1 import apply_layout_tags_to_l1, build_page_layout_v1
from plus_minus_strict import detect_plus_minus_strict
from text_punct_marks import detect_text_punct_marks_geometry_v1, finalize_text_punct_marks_v1
from vector_artifact_safety import vector_artifact_safety_reasons


CONSUMER_ALLOWED = False
L0_SCHEMA_VERSION = "r24_l0_baseline_v1"
L1_SCHEMA_VERSION = "r21_l1_calibration_v1"
L2_SCHEMA_VERSION = "r21_l2_screened_v1"
L2_5_SCHEMA_VERSION = "r21_l2_5_gdt_v1"
L3_SCHEMA_VERSION = "r21_l3_assembly_v1"
L4_SCHEMA_VERSION = "r21_l4_recovery_v1"
L5_SCHEMA_VERSION = "r21_l5_final_shadow_v1"
STATS_SCHEMA_VERSION = "r21_layered_vector_dimension_stats_v1"


def build_l1_calibration(
    *,
    decimal_point_rows: list[dict[str, Any]] | None,
    degree_rows: list[dict[str, Any]] | None,
    colon_candidate_rows: list[dict[str, Any]] | None = None,
    diameter_glyph_rows: list[dict[str, Any]] | None = None,
    vector_glyph_rows: list[dict[str, Any]] | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    """L1 auxiliary marker/calibration layer.

    Input: raw decimal-point rows and strict degree rows from existing vector
    detectors. Output: legacy-compatible reference anchors plus auxiliary marks.
    R24 keeps scale/reference fields here for existing consumers, but L0 owns
    the baseline interpretation.
    """
    decimal_rows = [
        deepcopy(row) for row in (decimal_point_rows or [])
        if row.get("consumer_allowed") is not True
    ]
    degree_rows = [
        deepcopy(row) for row in (degree_rows or [])
        if row.get("consumer_allowed") is not True
    ]
    colon_candidate_rows = [
        deepcopy(row) for row in (colon_candidate_rows or [])
        if row.get("consumer_allowed") is not True
    ]
    diameter_rows = [
        deepcopy(row) for row in (diameter_glyph_rows or [])
        if row.get("consumer_allowed") is not True
    ]
    colon_filter = filter_colon_decimal_pairs(
        decimal_rows,
        extra_colon_dots=colon_candidate_rows,
    )
    anchors: list[dict[str, Any]] = []
    for dot in colon_filter["kept"]:
        anchor = _anchor_from_decimal(dot, page_index=page_index, order=len(anchors))
        if anchor is not None:
            anchors.append(anchor)
    for degree in degree_rows:
        anchor = _anchor_from_degree(degree, page_index=page_index, order=len(anchors))
        if anchor is not None:
            anchors.append(anchor)
    for index, glyph in enumerate(diameter_rows):
        anchor = _anchor_from_diameter(
            glyph,
            page_index=page_index,
            order=len(anchors),
            source_index=index,
            existing_anchors=anchors,
        )
        if anchor is not None:
            anchors.append(anchor)

    return {
        "schema_version": L1_SCHEMA_VERSION,
        "page_index": int(page_index),
        "scale_baseline": _scale_baseline(colon_filter["kept"], degree_rows),
        "reference_anchors": anchors,
        "colon_marks": [
            _colon_mark_from_pair(pair, page_index=page_index, order=index)
            for index, pair in enumerate(colon_filter.get("colon_pairs") or [])
        ],
        "colon_filtered": colon_filter["rows"],
        "stats": {
            "decimal_input_count": len(decimal_rows),
            "degree_input_count": len(degree_rows),
            "colon_candidate_input_count": len(colon_candidate_rows),
            "decimal_anchor_count": sum(1 for row in anchors if row["kind"] == "dot"),
            "degree_anchor_count": sum(1 for row in anchors if row["kind"] == "degree"),
            "diameter_input_count": len(diameter_rows),
            "diameter_anchor_count": sum(1 for row in anchors if row["kind"] == "diameter"),
            "colon_dropped_count": colon_filter["summary"]["dropped_colon_dot_count"],
            "colon_mark_count": len(colon_filter.get("colon_pairs") or []),
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def build_l0_baseline(
    *,
    l1_calibration: dict[str, Any],
    layout: dict[str, Any] | None = None,
    vector_glyph_rows: list[dict[str, Any]] | None = None,
    fitz_page: Any | None = None,
    page_index: int = 0,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """R24 pure baseline layer.

    L0 is dump-only. It grades the existing L1 anchor list in place so L0 and
    L1 cannot drift: dot/degree anchors become ``core`` only when the shared
    layout health flag allows region filtering and the anchor remains in the
    drawing region. Diameter stays in the legacy reference anchor pool as a
    marker, not a baseline anchor.
    """
    layout = layout or {}
    layout_healthy = bool(layout.get("healthy", True))
    anchors = l1_calibration.get("reference_anchors") or []
    core: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for anchor in anchors:
        kind = str(anchor.get("kind") or "")
        if kind == "diameter":
            anchor["baseline_grade"] = "mark"
            continue
        if kind not in {"dot", "degree"}:
            continue
        if not layout_healthy:
            anchor["baseline_grade"] = "core"
            anchor["layout_unhealthy"] = True
            core.append(anchor)
            continue
        if str(anchor.get("region") or "drawing") == "drawing" and anchor.get("demoted_to_text_punct") is not True:
            anchor["baseline_grade"] = "core"
            core.append(anchor)
        else:
            anchor["baseline_grade"] = "quarantined"
            anchor["quarantine_reason"] = _baseline_quarantine_reason(anchor)
            quarantined.append(anchor)

    # U2 (R29): the Chinese notes/title glyphs may come from an unrelated font,
    # so the drawing-scale estimate must never mix the two populations.
    text_region_bboxes = _layout_text_region_bboxes(layout) if layout_healthy else []
    scale_rows: list[dict[str, Any]] = []
    text_region_excluded = 0
    for row in vector_glyph_rows or []:
        bbox = _xywh(row)
        center = _center(row, bbox)
        if center is not None and _point_in_any_bbox(center, text_region_bboxes):
            text_region_excluded += 1
            continue
        scale_rows.append(row)
    scale = _l0_scale_baseline(
        legacy_scale=l1_calibration.get("scale_baseline") or {},
        vector_glyph_rows=scale_rows,
    )
    if not layout_healthy:
        scale["region_isolation"] = "page_wide_layout_unhealthy"
    elif scale.get("source") == "connected_glyph_group_median":
        scale["region_isolation"] = "drawing_only"
    else:
        scale["region_isolation"] = "page_wide_legacy_fallback"
    scale["scale_input_row_count"] = len(scale_rows)
    scale["text_region_excluded_row_count"] = text_region_excluded
    font_profile = _font_profile(
        fitz_page=fitz_page,
        vector_glyph_rows=vector_glyph_rows or [],
        scale=scale,
        fail_on_error=fail_on_error,
    )
    return {
        "schema_version": L0_SCHEMA_VERSION,
        "page_index": int(page_index),
        "scale_baseline": scale,
        "font_profile": font_profile,
        "reference_anchors": anchors,
        "core_anchors": core,
        "quarantined_anchors": quarantined,
        "layout_healthy": layout_healthy,
        "drawing_anchor_ratio": layout.get("drawing_anchor_ratio"),
        "stats": {
            "core_anchor_count": len(core),
            "quarantined_anchor_count": len(quarantined),
            "font_source": font_profile.get("font_source"),
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def detect_degree_polylines_with_recall(page: Any, *, page_index: int = 0) -> dict[str, Any]:
    """Return strict degree rows plus conservative small-symbol recall rows.

    The existing detector accepts small candidates only when a nearby normal
    degree symbol anchors scale. R21 needs degree anchors even on small-angle
    clusters, so this wrapper reuses the strict ring classifier and promotes
    otherwise-valid small candidates as ``small_recall`` rows.
    """
    base_dump = detect_degree_polylines(page, page_index=page_index)
    rows = [deepcopy(row) for row in base_dump.get("vector_degree_polyline_v1", [])]
    seen = [_center_tuple(row) for row in rows]

    segments = extract_line_segments(page)
    rings = detect_closed_rings(segments)
    recall_rows: list[dict[str, Any]] = []
    for ring in rings:
        candidate = candidate_from_ring(ring)
        if candidate is None or candidate.get("size_class") != "small_candidate":
            continue
        center = _center_tuple(candidate)
        if center is None:
            continue
        if any(existing is not None and _distance(center, existing) <= 0.75 for existing in seen):
            continue
        row = deepcopy(candidate)
        row["schema_version"] = "vector_degree_polyline_v1"
        row["id"] = f"degree_recall_{len(recall_rows):04d}"
        row["size_class"] = "small_recall"
        row["source"] = "degree_polyline_strict_recall"
        row["consumer_allowed"] = CONSUMER_ALLOWED
        recall_rows.append(row)
        seen.append(center)

    rows.extend(recall_rows)
    for index, row in enumerate(rows):
        row["id"] = str(row.get("id") or f"degree_{index:04d}")
        row["schema_version"] = "vector_degree_polyline_v1"
        row["consumer_allowed"] = CONSUMER_ALLOWED
    counts = Counter(str(row.get("size_class") or "unknown") for row in rows)
    return {
        "schema_version": "vector_degree_polyline_v1",
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_degree_polyline_v1": rows,
        "vector_degree_polyline_stats_v1": {
            "schema_version": "vector_degree_polyline_v1",
            "page_index": int(page_index),
            "count": len(rows),
            "counts_by_size_class": dict(sorted(counts.items())),
            "consumer_allowed_count": 0,
            "r21_recall_count": len(recall_rows),
        },
    }


def build_l2_screened(
    *,
    l1_calibration: dict[str, Any],
    vector_glyph_tokens: list[dict[str, Any]] | None,
    gdt_line_frames: list[dict[str, Any]] | None,
    gdt_decimal_seeded_shadow: dict[str, Any] | None,
    capsule_rows: list[dict[str, Any]] | None,
    digit_match_rows: list[dict[str, Any]] | None = None,
    strict_pitch_read_dump: dict[str, Any] | None = None,
    pm_strict_rows: list[dict[str, Any]] | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    """L2 Screening.

    Input: L1 calibration, optional strict R33-M1 phrase reads, and raw existing
    detector rows retained for comparison. Output: topology specs, local glyph
    evidence near reference anchors, canonical GD&T frames, and KEY capsule
    frames. L2 does no string assembly.
    """
    scale = l1_calibration.get("scale_baseline") or {}
    glyph_specs = _glyph_specs_from_scale(scale)
    glyph_h = float(glyph_specs["digit"]["height_range"][1]) / 1.5
    raw_anchors = list(l1_calibration.get("reference_anchors") or [])
    layout_healthy = bool(l1_calibration.get("layout_healthy", True))
    if layout_healthy:
        anchors = [
            anchor for anchor in raw_anchors
            if str(anchor.get("region") or "drawing") == "drawing"
            and anchor.get("demoted_to_text_punct") is not True
        ]
    else:
        anchors = list(raw_anchors)
    strict_pitch_rows, strict_pitch_stats = _strict_pitch_read_evidence_rows(
        strict_pitch_read_dump,
        page_index=page_index,
    )
    # R33-M1 phrase rows are whole-phrase evidence: the strict reader does not
    # publish per-character bboxes.  An explicitly supplied strict dump owns
    # this evidence seam completely.  Invalid strict input therefore stays
    # empty instead of falling back to the legacy character-level comparison.
    # Only the absent-dump case preserves the pre-existing vector+digit path.
    all_glyph_rows = (
        strict_pitch_rows
        if strict_pitch_read_dump is not None
        else list(vector_glyph_tokens or []) + list(digit_match_rows or [])
    )
    raw_key_capsule_count = len(capsule_rows or [])
    key_capsules: list[dict[str, Any]] = []
    local_glyphs = _local_glyphs_near_refs(
        anchors=anchors,
        rows=all_glyph_rows,
        glyph_h=glyph_h,
    )
    if strict_pitch_read_dump is None:
        local_glyphs = _replace_overlapping_pm_glyphs(
            local_glyphs=local_glyphs,
            pm_rows=pm_strict_rows or [],
            anchors=anchors,
        )
    confirmed_line_frames, fallback_line_frames = _partition_gdt_line_frames_by_seed(
        gdt_line_frames or [],
        anchors=anchors,
        glyph_h=glyph_h,
    )
    gdt_frames = _canonical_gdt_frames(
        confirmed_line_frames,
        (gdt_decimal_seeded_shadow or {}).get("r7_gdt_vertical_shadow_v1", []),
        page_index=page_index,
    )
    gdt_fallback_frames = _canonical_gdt_fallback_frames(
        fallback_line_frames,
        page_index=page_index,
    )
    return {
        "schema_version": L2_SCHEMA_VERSION,
        "page_index": int(page_index),
        "glyph_specs": glyph_specs,
        "local_glyphs_near_refs": local_glyphs,
        "gdt_frames": gdt_frames,
        "gdt_fallback_frames": gdt_fallback_frames,
        "key_capsules": key_capsules,
        "key_capsules_seam": {
            "status": "removed_pending_candidate_dimension_to_outer_frame_inversion",
            "raw_capsule_input_count": raw_key_capsule_count,
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "stats": {
            "reference_anchor_count": len(anchors),
            "reference_anchor_input_count": len(raw_anchors),
            "vector_glyph_input_count": len(vector_glyph_tokens or []),
            "digit_match_input_count": len(digit_match_rows or []),
            "gdt_line_frame_input_count": len(gdt_line_frames or []),
            "local_glyph_count": len(local_glyphs),
            "strict_pitch_dump_status": strict_pitch_stats["status"],
            "strict_pitch_input_read_count": strict_pitch_stats["input_read_count"],
            "strict_pitch_evidence_count": strict_pitch_stats["evidence_count"],
            "strict_pitch_local_evidence_count": sum(
                row.get("source") == "r33_m1_pitch_reader"
                for row in local_glyphs
            ),
            "strict_pitch_rejected_read_count": strict_pitch_stats["rejected_read_count"],
            "strict_pitch_rejection_counts": strict_pitch_stats["rejection_counts"],
            "gdt_frame_count": len(gdt_frames),
            "gdt_fallback_frame_count": len(gdt_fallback_frames),
            "key_capsule_count": len(key_capsules),
            "key_capsule_input_count": raw_key_capsule_count,
            "pm_strict_count": len([
                row for row in local_glyphs
                if row.get("source") == "pm_strict"
            ]),
            "layout_healthy": layout_healthy,
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def build_l2_5_gdt(
    *,
    gdt_frames: list[dict[str, Any]],
    fitz_page: Any,
    inverter: Any | None = None,
    allow_datum_inversion: bool = False,
    fail_on_error: bool = False,
    page_index: int = 0,
) -> list[dict[str, Any]]:
    """L2.5 GD&T in-frame screening.

    Input: L2 canonical GD&T frames. Output: per-frame vector symbols and,
    only for legacy diagnostic callers that explicitly retain it, datum
    topology. Strict diagnostics disable datum inversion entirely. This layer
    never emits numeric glyphs or dimension values.
    """
    inverter = inverter or GDTVectorInverter(
        fail_on_error=fail_on_error,
    )
    rows: list[dict[str, Any]] = []
    for frame in gdt_frames or []:
        bbox = frame.get("bbox") or {}
        compartments = frame.get("compartments") or []
        try:
            symbols = inverter.invert_in_frame(fitz_page, bbox, compartments)
        except Exception:
            if fail_on_error:
                raise
            symbols = []
        datums = []
        if allow_datum_inversion:
            try:
                datums = inverter.invert_datum(fitz_page, bbox)
            except Exception:
                if fail_on_error:
                    raise
                datums = []
        rows.append({
            "schema_version": L2_5_SCHEMA_VERSION,
            "page_index": int(page_index),
            "frame_id": str(frame.get("frame_id") or ""),
            "frame": deepcopy(frame),
            "symbols": _clean_symbol_rows(symbols),
            "datums": _clean_symbol_rows(datums),
            "numeric_glyphs": [],
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return rows


def build_l3_dimensions(
    *,
    l1_calibration: dict[str, Any],
    l2_screened: dict[str, Any],
    l2_5_gdt: list[dict[str, Any]],
    r27_digit_cluster_probe: dict[str, Any] | None = None,
    page_index: int = 0,
) -> dict[str, Any]:
    """L3 arbitrary-angle assembly and semantic validation.

    Input: L1 anchors, L2 non-frame glyph evidence and KEY capsules, and L2.5
    frame results for avoidance context. Output: assembled shadow dimensions
    plus suspect rows. L3 marks pollution and uncertainty instead of consuming.
    """
    del l2_5_gdt  # Reserved seam: frame avoidance/de-dup is not graduated.
    dims: list[dict[str, Any]] = []
    glyphs = list(l2_screened.get("local_glyphs_near_refs") or [])
    scale = l1_calibration.get("scale_baseline") or {}
    glyph_h = _float_or(scale.get("glyph_h"), 8.0)
    r27_context = _r27_digit_cluster_context(r27_digit_cluster_probe)
    for anchor in l1_calibration.get("reference_anchors") or []:
        run = _axis_run(anchor, glyphs, glyph_h=glyph_h)
        if not run:
            continue
        dim = _dimension_from_run(
            run,
            anchor=anchor,
            page_index=page_index,
            order=len(dims),
            r27_context=r27_context,
        )
        if dim is not None:
            dims.append(dim)

    suspects = [row for row in dims if _is_suspect(row)]
    return {
        "schema_version": L3_SCHEMA_VERSION,
        "page_index": int(page_index),
        "L3_dimensions": dims,
        "L3_suspects": suspects,
        "stats": {
            "dimension_count": len(dims),
            "suspect_count": len(suspects),
            "r27_digit_cluster_l3_enabled": bool(r27_context["enabled"]),
            "r27_digit_cluster_label_count": int(r27_context["label_count"]),
            "r27_digit_cluster_complete_anchor_count": int(r27_context["complete_anchor_count"]),
            "r27_digit_cluster_phrase_complete_anchor_count": int(
                r27_context["phrase_complete_anchor_count"]
            ),
            "r27_digit_cluster_fragment_anchor_count": int(r27_context["fragment_anchor_count"]),
            "r27_phrase_boundary_producer": r27_context["phrase_boundary_producer"],
            "r27_phrase_gate_status": r27_context["phrase_gate_status"],
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def build_l4_recovered(
    *,
    l3_suspects: list[dict[str, Any]] | None,
    l2_screened: dict[str, Any],
    diameter_glyphs: list[dict[str, Any]] | None,
    page_index: int = 0,
) -> dict[str, Any]:
    """L4 targeted recovery.

    Input: L3 suspects and L2 parameters. Output: recovery rows only for
    evidence overlapping suspect regions. L4 never consumes a full-page scan
    without a suspect target.
    """
    del l2_screened
    recovered: list[dict[str, Any]] = []
    suspect_rows = list(l3_suspects or [])
    diameter_rows = list(diameter_glyphs or [])
    for suspect in suspect_rows:
        suspect_bbox = _suspect_bbox(suspect)
        if suspect_bbox is None:
            continue
        for glyph in diameter_rows:
            glyph_bbox = _xywh(glyph)
            if glyph_bbox is None or _intersection_area(suspect_bbox, glyph_bbox) <= 0:
                continue
            recovered.append({
                "schema_version": L4_SCHEMA_VERSION,
                "page_index": int(page_index),
                "dimension_id": f"p{int(page_index) + 1:03d}_l4_recovered_{len(recovered):06d}",
                "recovered_from_suspect_id": str(suspect.get("dimension_id") or ""),
                "recovery_kind": "diameter_glyph_overlap",
                "bbox": _round_bbox(glyph_bbox),
                "pre_confidence": 0.70,
                "flags": ["l4_recovered"],
                "consumer_allowed": CONSUMER_ALLOWED,
            })
            break
    return {
        "schema_version": L4_SCHEMA_VERSION,
        "page_index": int(page_index),
        "L4_recovered": recovered,
        "stats": {
            "input_suspect_count": len(suspect_rows),
            "input_diameter_glyph_count": len(diameter_rows),
            "recovered_count": len(recovered),
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def build_l5_final_shadow(
    *,
    l3_dimensions: list[dict[str, Any]] | None,
    l4_recovered: list[dict[str, Any]] | None,
    high_conf_threshold: float = 0.90,
    page_index: int = 0,
) -> dict[str, Any]:
    """L5 confidence routing.

    Input: L3 dimensions and L4 recovered rows. Output: high/low confidence
    shadow queues only. Suspect flags always route to low confidence.
    """
    l3_rows = [deepcopy(row) for row in (l3_dimensions or [])]
    l4_rows = [deepcopy(row) for row in (l4_recovered or [])]
    high_conf: list[dict[str, Any]] = []
    low_conf: list[dict[str, Any]] = []
    for row in l3_rows + l4_rows:
        row["consumer_allowed"] = CONSUMER_ALLOWED
        flags = list(row.get("flags") or [])
        confidence = _float_or(row.get("pre_confidence"), 0.0)
        if flags or confidence < high_conf_threshold:
            low_conf.append(row)
        else:
            high_conf.append(row)
    return {
        "schema_version": L5_SCHEMA_VERSION,
        "page_index": int(page_index),
        "high_conf": high_conf,
        "low_conf": low_conf,
        "stats": {
            "input_l3_count": len(l3_rows),
            "input_l4_count": len(l4_rows),
            "high_conf_count": len(high_conf),
            "low_conf_count": len(low_conf),
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def build_r21_layered_vector_dimension_shadow(
    *,
    decimal_point_rows: list[dict[str, Any]] | None,
    degree_rows: list[dict[str, Any]] | None,
    vector_glyph_tokens: list[dict[str, Any]] | None,
    gdt_line_frames: list[dict[str, Any]] | None,
    gdt_decimal_seeded_shadow: dict[str, Any] | None,
    capsule_rows: list[dict[str, Any]] | None,
    diameter_glyphs: list[dict[str, Any]] | None,
    fitz_page: Any,
    digit_match_rows: list[dict[str, Any]] | None = None,
    strict_pitch_read_dump: dict[str, Any] | None = None,
    page_index: int = 0,
    consumer_allowed: bool = False,
    strict_diagnostic_mode: bool = False,
    timing_sink: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build all R21 layer dumps without touching final dimensions."""
    total_started = time.perf_counter()
    timings_ms: dict[str, float] = {}

    def finish_stage(name: str, started: float) -> None:
        timings_ms[name] = round((time.perf_counter() - started) * 1000.0, 3)

    if consumer_allowed:
        raise ValueError("R21 layered vector dimension is dump-only; consumer_allowed must be False")
    colon_candidate_rows: list[dict[str, Any]] = []
    stage_started = time.perf_counter()
    if fitz_page is not None:
        try:
            colon_candidate_rows = detect_large_dot_marks(
                fitz_page,
                page_index=page_index,
                fail_on_error=strict_diagnostic_mode,
            ).get("r22_large_dot_marks_v1", [])
        except Exception:
            if strict_diagnostic_mode:
                raise
            colon_candidate_rows = []
    finish_stage("l1.colon_detection", stage_started)
    stage_started = time.perf_counter()
    l1 = build_l1_calibration(
        decimal_point_rows=deepcopy(decimal_point_rows or []),
        degree_rows=deepcopy(degree_rows or []),
        colon_candidate_rows=deepcopy(colon_candidate_rows),
        diameter_glyph_rows=deepcopy(diameter_glyphs or []),
        vector_glyph_rows=deepcopy(vector_glyph_tokens or []),
        page_index=page_index,
    )
    finish_stage("l1.calibration", stage_started)
    l0: dict[str, Any] | None = None
    text_punct_candidate_dump: dict[str, Any] = {}
    text_punct_marks: list[dict[str, Any]] = []
    layout: dict[str, Any] = {
        "schema_version": "r23_layout_v1",
        "page_index": int(page_index),
        "frame_border": {"border_found": False, "bbox": None, "consumer_allowed": CONSUMER_ALLOWED},
        "drawing_area": None,
        "title_blocks": [],
        "tables": [],
        "notes_regions": [],
        "anchor_region_tags": {},
        "healthy": True,
        "drawing_anchor_ratio": 1.0,
        "stats": {
            "title_block_count": 0,
            "table_count": 0,
            "notes_region_count": 0,
            "tagged_anchor_count": 0,
            "consumer_allowed_count": 0,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    text_punct_started = time.perf_counter()
    layout_started: float | None = None
    if fitz_page is not None:
        try:
            scale = l1.get("scale_baseline") or {}
            text_punct_candidate_dump = detect_text_punct_marks_geometry_v1(
                fitz_page,
                glyph_h=_float_or(scale.get("glyph_h"), 8.0),
                page_index=page_index,
            )
            text_punct_candidates = text_punct_candidate_dump.get("r23_text_punct_marks_v1", [])
        except Exception:
            if strict_diagnostic_mode:
                raise
            text_punct_candidates = []
            text_punct_candidate_dump = {}
        finish_stage("l1_5.text_punct_candidates", text_punct_started)
        layout_started = time.perf_counter()
        try:
            layout = build_page_layout_v1(
                page_rect=fitz_page.rect,
                line_segments=extract_line_segments(fitz_page),
                reference_anchors=l1.get("reference_anchors") or [],
                colon_marks=l1.get("colon_marks") or [],
                text_punct_marks=text_punct_candidates,
                page_index=page_index,
            )
            text_punct_marks = finalize_text_punct_marks_v1(text_punct_candidates, layout)
            text_punct_candidate_dump["r23_text_punct_marks_stats_v1"] = _text_punct_stats(
                text_punct_marks,
                page_index=page_index,
                source_stats=text_punct_candidate_dump.get("r23_text_punct_marks_stats_v1") or {},
            )
            l1 = apply_layout_tags_to_l1(l1, layout, text_punct_marks)
        except Exception:
            if strict_diagnostic_mode:
                raise
            text_punct_marks = text_punct_candidates
        finish_stage("l1_5.layout", layout_started)
    else:
        finish_stage("l1_5.text_punct_candidates", text_punct_started)
        timings_ms["l1_5.layout"] = 0.0
    if "text_punct_marks" not in l1:
        l1 = apply_layout_tags_to_l1(l1, layout, text_punct_marks)
    stage_started = time.perf_counter()
    l0 = build_l0_baseline(
        l1_calibration=l1,
        layout=layout,
        vector_glyph_rows=vector_glyph_tokens or [],
        fitz_page=fitz_page,
        page_index=page_index,
        fail_on_error=strict_diagnostic_mode,
    )
    finish_stage("l0.baseline", stage_started)
    pm_strict_rows: list[dict[str, Any]] = []
    pm_strict_dump: dict[str, Any] = {}
    stage_started = time.perf_counter()
    if fitz_page is not None:
        try:
            pm_strict_dump = detect_plus_minus_strict(
                fitz_page,
                _anchors_for_pm_strict(l1),
                page_index=page_index,
            )
            pm_strict_rows = pm_strict_dump.get("r22_pm_strict_v1", [])
        except Exception:
            if strict_diagnostic_mode:
                raise
            pm_strict_rows = []
            pm_strict_dump = {}
    finish_stage("l2.plus_minus_strict", stage_started)
    stage_started = time.perf_counter()
    l2 = build_l2_screened(
        l1_calibration=l1,
        vector_glyph_tokens=deepcopy(vector_glyph_tokens or []),
        digit_match_rows=deepcopy(digit_match_rows or []),
        strict_pitch_read_dump=(
            deepcopy(strict_pitch_read_dump)
            if strict_pitch_read_dump is not None
            else None
        ),
        gdt_line_frames=deepcopy(gdt_line_frames or []),
        gdt_decimal_seeded_shadow=deepcopy(gdt_decimal_seeded_shadow or {}),
        capsule_rows=deepcopy(capsule_rows or []),
        pm_strict_rows=deepcopy(pm_strict_rows),
        page_index=page_index,
    )
    finish_stage("l2.screened", stage_started)
    stage_started = time.perf_counter()
    l2_5 = build_l2_5_gdt(
        gdt_frames=l2["gdt_frames"],
        fitz_page=fitz_page,
        allow_datum_inversion=not strict_diagnostic_mode,
        fail_on_error=strict_diagnostic_mode,
        page_index=page_index,
    )
    finish_stage("l2_5.gdt_inversion", stage_started)
    stage_started = time.perf_counter()
    l3 = build_l3_dimensions(
        l1_calibration=l1,
        l2_screened=l2,
        l2_5_gdt=l2_5,
        page_index=page_index,
    )
    finish_stage("l3.assembly", stage_started)
    stage_started = time.perf_counter()
    l4 = build_l4_recovered(
        l3_suspects=l3["L3_suspects"],
        l2_screened=l2,
        diameter_glyphs=deepcopy(diameter_glyphs or []),
        page_index=page_index,
    )
    finish_stage("l4.recovery", stage_started)
    stage_started = time.perf_counter()
    l5 = build_l5_final_shadow(
        l3_dimensions=l3["L3_dimensions"],
        l4_recovered=l4["L4_recovered"],
        page_index=page_index,
    )
    finish_stage("l5.routing", stage_started)
    timings_ms["total"] = round((time.perf_counter() - total_started) * 1000.0, 3)
    if timing_sink is not None:
        timing_sink.clear()
        timing_sink.update(timings_ms)
    return {
        "r24_l0_baseline_v1": l0,
        "r21_l1_calibration_v1": l1,
        "r23_layout_v1": layout,
        "r23_text_punct_marks_v1": text_punct_marks,
        "r23_text_punct_marks_stats_v1": deepcopy(text_punct_candidate_dump.get("r23_text_punct_marks_stats_v1") or {
            "schema_version": "r23_text_punct_marks_v1",
            "page_index": int(page_index),
            "raw_count": len(text_punct_marks),
            "counts_by_kind": dict(sorted(Counter(str(row.get("kind") or "unknown") for row in text_punct_marks).items())),
            "consumer_allowed_count": 0,
        }),
        "r21_l2_screened_v1": l2,
        "r21_l2_5_gdt_v1": l2_5,
        "r23_gdt_frames_fallback_v1": l2["gdt_fallback_frames"],
        "r21_l3_dimensions_v1": l3["L3_dimensions"],
        "r21_l3_suspects_v1": l3["L3_suspects"],
        "r21_l4_recovered_v1": l4["L4_recovered"],
        "r21_l5_final_shadow_v1": l5,
        "r22_pm_strict_v1": deepcopy(pm_strict_rows),
        "r22_pm_strict_stats_v1": deepcopy(pm_strict_dump.get("r22_pm_strict_stats_v1") or {
            "schema_version": "r22_pm_strict_v1",
            "page_index": int(page_index),
            "count": len(pm_strict_rows),
            "counts_by_symbol": dict(sorted(Counter(str(row.get("symbol") or "unknown") for row in pm_strict_rows).items())),
            "consumer_allowed_count": sum(1 for row in pm_strict_rows if row.get("consumer_allowed")),
        }),
        "r21_layered_vector_dimension_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": int(page_index),
            "l1_anchor_count": len(l1["reference_anchors"]),
            "l1_diameter_anchor_count": sum(1 for row in l1["reference_anchors"] if row.get("kind") == "diameter"),
            "r24_l0_core_anchor_count": len((l0 or {}).get("core_anchors") or []),
            "r24_l0_quarantined_anchor_count": len((l0 or {}).get("quarantined_anchors") or []),
            "r23_text_punct_mark_count": len(text_punct_marks),
            "r23_notes_region_count": len(layout.get("notes_regions") or []),
            "r23_layout_healthy": bool(layout.get("healthy", True)),
            "r23_drawing_anchor_ratio": layout.get("drawing_anchor_ratio"),
            "l2_local_glyph_count": len(l2["local_glyphs_near_refs"]),
            "l2_strict_pitch_dump_status": l2["stats"]["strict_pitch_dump_status"],
            "l2_strict_pitch_input_read_count": l2["stats"]["strict_pitch_input_read_count"],
            "l2_strict_pitch_evidence_count": l2["stats"]["strict_pitch_evidence_count"],
            "l2_strict_pitch_local_evidence_count": l2["stats"]["strict_pitch_local_evidence_count"],
            "l2_strict_pitch_rejected_read_count": l2["stats"]["strict_pitch_rejected_read_count"],
            "l2_gdt_frame_count": len(l2["gdt_frames"]),
            "l2_gdt_fallback_frame_count": len(l2["gdt_fallback_frames"]),
            "l2_key_capsule_count": len(l2["key_capsules"]),
            "r22_pm_strict_raw_count": len(pm_strict_rows),
            "l2_pm_strict_count": len([
                row for row in l2["local_glyphs_near_refs"]
                if row.get("source") == "pm_strict"
            ]),
            "l3_dimension_count": len(l3["L3_dimensions"]),
            "l3_suspect_count": len(l3["L3_suspects"]),
            "l4_recovered_count": len(l4["L4_recovered"]),
            "l5_high_conf_count": len(l5["high_conf"]),
            "l5_low_conf_count": len(l5["low_conf"]),
            "consumer_allowed": CONSUMER_ALLOWED,
            "consumer_allowed_count": 0,
        },
    }


def _anchor_from_decimal(row: dict[str, Any], *, page_index: int, order: int) -> dict[str, Any] | None:
    bbox = _xywh(row)
    center = _center(row, bbox)
    angle = _decimal_axis_angle(row)
    if bbox is None or center is None or angle is None:
        return None
    return {
        "schema_version": L1_SCHEMA_VERSION,
        "anchor_id": f"p{int(page_index) + 1:03d}_r21_anchor_{int(order):06d}",
        "source_id": str(row.get("id") or f"dot_{order:04d}"),
        "kind": "dot",
        "bbox": _round_bbox(bbox),
        "center": _round_point(center),
        "axis_angle_deg": round(float(angle) % 180.0, 6),
        "axis_angle_source": "decimal_point_quad.short_axis",
        "source": "decimal_point_quad",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _anchor_from_degree(row: dict[str, Any], *, page_index: int, order: int) -> dict[str, Any] | None:
    bbox = _xywh(row)
    center = _center(row, bbox)
    angle = _float_or_none(row.get("axis_angle_deg"))
    if bbox is None or center is None or angle is None:
        return None
    return {
        "schema_version": L1_SCHEMA_VERSION,
        "anchor_id": f"p{int(page_index) + 1:03d}_r21_anchor_{int(order):06d}",
        "source_id": str(row.get("id") or f"degree_{order:04d}"),
        "kind": "degree",
        "bbox": _round_bbox(bbox),
        "center": _round_point(center),
        "axis_angle_deg": round(float(angle) % 180.0, 6),
        "axis_angle_source": "degree_polyline_strict.axis_angle_deg",
        "source": str(row.get("source") or "degree_polyline_strict"),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _anchor_from_diameter(
    row: dict[str, Any],
    *,
    page_index: int,
    order: int,
    source_index: int,
    existing_anchors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bbox = _xywh(row)
    center = _center(row, bbox)
    if bbox is None or center is None:
        return None
    nearest_dot = _nearest_dot_anchor(center, existing_anchors)
    if nearest_dot is not None:
        angle = _float_or(nearest_dot.get("axis_angle_deg"), 0.0)
        axis_source = f"nearest_dot:{nearest_dot.get('anchor_id')}"
    else:
        angle = 0.0
        axis_source = "default"
    return {
        "schema_version": L1_SCHEMA_VERSION,
        "anchor_id": f"p{int(page_index) + 1:03d}_r21_anchor_{int(order):06d}",
        "source_id": str(row.get("id") or row.get("source_id") or f"diameter_glyph_{int(source_index):06d}"),
        "kind": "diameter",
        "bbox": _round_bbox(bbox),
        "center": _round_point(center),
        "axis_angle_deg": round(float(angle) % 180.0, 6),
        "axis_angle_source": axis_source,
        "source": "extract_diameter_glyphs",
        "source_index": int(source_index),
        "baseline_grade": "mark",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _nearest_dot_anchor(center: dict[str, float], anchors: list[dict[str, Any]]) -> dict[str, Any] | None:
    dots = [
        anchor for anchor in anchors
        if anchor.get("kind") == "dot" and isinstance(anchor.get("center"), dict)
    ]
    if not dots:
        return None
    point = (float(center["x"]), float(center["y"]))
    return min(
        dots,
        key=lambda anchor: _distance(
            point,
            (float(anchor["center"]["x"]), float(anchor["center"]["y"])),
        ),
    )


def _colon_mark_from_pair(pair: dict[str, Any], *, page_index: int, order: int) -> dict[str, Any]:
    points = [
        dict(point)
        for point in pair.get("points") or []
        if isinstance(point, dict)
    ]
    return {
        "schema_version": L1_SCHEMA_VERSION,
        "colon_id": f"p{int(page_index) + 1:03d}_r22_colon_{int(order):06d}",
        "kind": "colon",
        "pair_id": str(pair.get("pair_id") or f"colon_pair_{order:06d}"),
        "point_source_ids": [str(point.get("dot_id") or "") for point in points],
        "points": points,
        "bbox": _round_bbox(_xywh(pair.get("bbox") or pair)),
        "center": _round_point(pair.get("center") or {"x": 0.0, "y": 0.0}),
        "direction": str(pair.get("direction") or "vertical"),
        "dx": round(_float_or(pair.get("dx"), 0.0), 4),
        "dy": round(_float_or(pair.get("dy"), 0.0), 4),
        "dy_over_size": round(_float_or(pair.get("dy_over_size"), 0.0), 4),
        "size_ratio": round(_float_or(pair.get("size_ratio"), 0.0), 4),
        "size_to_page_median": (
            None
            if pair.get("size_to_page_median") is None
            else round(_float_or(pair.get("size_to_page_median"), 0.0), 4)
        ),
        "source": "decimal_colon_filter",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _anchors_for_pm_strict(l1_calibration: dict[str, Any]) -> list[dict[str, Any]]:
    scale = l1_calibration.get("scale_baseline") or {}
    glyph_h = _float_or(scale.get("glyph_h"), 8.0)
    stroke_w = _float_or_none(scale.get("stroke_w"))
    rows: list[dict[str, Any]] = []
    for anchor in l1_calibration.get("reference_anchors") or []:
        if anchor.get("kind") != "dot":
            continue
        row = deepcopy(anchor)
        row["scale_baseline"] = glyph_h
        if stroke_w is not None:
            row["stroke_w"] = stroke_w
        rows.append(row)
    return rows


def _text_punct_stats(
    rows: list[dict[str, Any]],
    *,
    page_index: int,
    source_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = Counter(str(row.get("kind") or "unknown") for row in rows)
    candidate_counts = Counter(
        str((row.get("detail") or {}).get("candidate_kind") or row.get("kind") or "unknown")
        for row in rows
    )
    source_stats = source_stats or {}
    return {
        "schema_version": "r23_text_punct_marks_v1",
        "page_index": int(page_index),
        "raw_count": len(rows),
        "counts_by_kind": dict(sorted(counts.items())),
        "counts_by_geometry_candidate": dict(sorted(candidate_counts.items())),
        "isolation_radius_glyph_h": source_stats.get("isolation_radius_glyph_h"),
        "isolation_rejected_count": source_stats.get("isolation_rejected_count"),
        "consumer_allowed_count": 0,
    }


def _baseline_quarantine_reason(anchor: dict[str, Any]) -> str:
    if anchor.get("demoted_to_text_punct") is True:
        return "demoted_to_text_punct"
    region = str(anchor.get("region") or "drawing")
    if region != "drawing":
        return f"region:{region}"
    return "unknown"


def _mean_anchor_center(anchors: list[dict[str, Any]]) -> dict[str, float]:
    xs = [float((anchor.get("center") or {}).get("x")) for anchor in anchors]
    ys = [float((anchor.get("center") or {}).get("y")) for anchor in anchors]
    return {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys)}


def _mean_axis_angle(anchors: list[dict[str, Any]]) -> float:
    sx = 0.0
    sy = 0.0
    for anchor in anchors:
        theta = math.radians(_float_or(anchor.get("axis_angle_deg"), 0.0) * 2.0)
        sx += math.cos(theta)
        sy += math.sin(theta)
    return (math.degrees(math.atan2(sy, sx)) / 2.0) % 180.0


def _axis_angle_delta(left: float, right: float) -> float:
    delta = abs((float(left) % 180.0) - (float(right) % 180.0))
    return min(delta, 180.0 - delta)


def _font_profile(
    *,
    fitz_page: Any | None,
    vector_glyph_rows: list[dict[str, Any]],
    scale: dict[str, Any],
    fail_on_error: bool = False,
) -> dict[str, Any]:
    text_profile = _text_layer_font_profile(
        fitz_page,
        fail_on_error=fail_on_error,
    )
    if text_profile is not None:
        return text_profile
    return _vector_font_fingerprint(
        fitz_page=fitz_page,
        vector_glyph_rows=vector_glyph_rows,
        scale=scale,
        fail_on_error=fail_on_error,
    )


def _text_layer_font_profile(
    fitz_page: Any | None,
    *,
    fail_on_error: bool = False,
) -> dict[str, Any] | None:
    if fitz_page is None:
        return None
    try:
        text_dict = fitz_page.get_text("dict")
    except Exception:
        if fail_on_error:
            raise
        return None
    spans: list[dict[str, Any]] = []
    for block in text_dict.get("blocks") or []:
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                if str(span.get("text") or "").strip():
                    spans.append(span)
    if not spans:
        return None
    font_counts = Counter(str(span.get("font") or "unknown") for span in spans)
    sizes = sorted(_float_or(span.get("size"), 0.0) for span in spans if _float_or(span.get("size"), 0.0) > 0)
    return {
        "font_source": "text_layer",
        "confidence": 1.0,
        "font_names": dict(sorted(font_counts.items())),
        "font_size_distribution": _distribution(sizes),
        "span_count": len(spans),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _vector_font_fingerprint(
    *,
    fitz_page: Any | None,
    vector_glyph_rows: list[dict[str, Any]],
    scale: dict[str, Any],
    fail_on_error: bool = False,
) -> dict[str, Any]:
    group_heights = _connected_glyph_group_heights(vector_glyph_rows)
    drawing_counts = _drawing_primitive_counts(
        fitz_page,
        fail_on_error=fail_on_error,
    )
    line_count = drawing_counts.get("line_count", 0)
    curve_count = drawing_counts.get("curve_count", 0)
    fill_count = drawing_counts.get("fill_count", 0)
    if fill_count and curve_count:
        drawing_method = "fill_outline_mixed"
    elif curve_count > line_count * 0.20:
        drawing_method = "stroke_curve_mixed"
    else:
        drawing_method = "stroke_polyline"
    return {
        "font_source": "vector_fingerprint",
        "confidence": 0.0,
        "drawing_method": drawing_method,
        "stroke_width_median": _float_or(scale.get("stroke_w"), 0.0),
        "connected_group_glyph_h_median": median(group_heights) if group_heights else _float_or(scale.get("glyph_h"), 8.0),
        "line_primitive_ratio": _ratio(line_count, line_count + curve_count),
        "curve_primitive_ratio": _ratio(curve_count, line_count + curve_count),
        "glyph_group_count": len(group_heights),
        "note": "vector fingerprint only; no font-name claim",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _drawing_primitive_counts(
    fitz_page: Any | None,
    *,
    fail_on_error: bool = False,
) -> dict[str, int]:
    counts = {"line_count": 0, "curve_count": 0, "fill_count": 0}
    if fitz_page is None:
        return counts
    try:
        drawings = fitz_page.get_drawings()
    except Exception:
        if fail_on_error:
            raise
        return counts
    for drawing in drawings or []:
        if drawing.get("fill") is not None:
            counts["fill_count"] += 1
        for item in drawing.get("items") or []:
            op = str(item[0] if item else "")
            if op in {"l", "re"}:
                counts["line_count"] += 1
            elif op == "c":
                counts["curve_count"] += 1
    return counts


def _connected_glyph_group_heights(rows: list[dict[str, Any]]) -> list[float]:
    glyphs = []
    for row in rows or []:
        bbox = _xywh(row)
        center = _center(row, bbox)
        if bbox is None or center is None:
            continue
        glyphs.append({"bbox": bbox, "center": center})
    if not glyphs:
        return []
    heights = [abs(float(item["bbox"]["h"])) for item in glyphs]
    heights = [height for height in heights if height > 0]
    base_h = median(heights) if heights else 8.0
    lines: list[list[dict[str, Any]]] = []
    for glyph in sorted(glyphs, key=lambda item: item["center"]["y"]):
        if lines and abs(lines[-1][0]["center"]["y"] - glyph["center"]["y"]) <= max(base_h * 0.55, 3.0):
            lines[-1].append(glyph)
        else:
            lines.append([glyph])
    group_heights = []
    for line in lines:
        y0 = min(item["bbox"]["y"] for item in line)
        y1 = max(item["bbox"]["y"] + item["bbox"]["h"] for item in line)
        group_heights.append(float(y1 - y0))
    return [height for height in group_heights if height > 0]


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(float(min(values)), 3),
        "median": round(float(median(values)), 3),
        "max": round(float(max(values)), 3),
    }


def _ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(part) / float(total), 4)


def _layout_text_region_bboxes(layout: dict[str, Any] | None) -> list[dict[str, float]]:
    bboxes: list[dict[str, float]] = []
    for key in ("notes_regions", "title_blocks"):
        for region in (layout or {}).get(key) or []:
            if not isinstance(region, dict):
                continue
            bbox = region.get("bbox")
            if not isinstance(bbox, dict):
                continue
            try:
                bboxes.append({k: float(bbox[k]) for k in ("x", "y", "w", "h")})
            except (KeyError, TypeError, ValueError):
                continue
    return bboxes


def _point_in_any_bbox(point: dict[str, float], bboxes: list[dict[str, float]]) -> bool:
    x = float(point["x"])
    y = float(point["y"])
    for bbox in bboxes:
        if bbox["x"] <= x <= bbox["x"] + bbox["w"] and bbox["y"] <= y <= bbox["y"] + bbox["h"]:
            return True
    return False


def _l0_scale_baseline(
    *,
    legacy_scale: dict[str, Any],
    vector_glyph_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    connected = _scale_baseline([], [], vector_glyph_rows)
    if connected.get("source") == "connected_glyph_group_median":
        stroke_w = _float_or_none(legacy_scale.get("stroke_w"))
        if stroke_w is not None:
            connected["stroke_w"] = round(stroke_w, 4)
        connected["legacy_consumer_scale_baseline"] = deepcopy(legacy_scale)
        return connected
    return deepcopy(legacy_scale)


def _scale_baseline(
    decimal_rows: list[dict[str, Any]],
    degree_rows: list[dict[str, Any]],
    vector_glyph_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    heights: list[float] = []
    widths: list[float] = []
    stroke_widths: list[float] = []
    connected_heights = _connected_glyph_group_heights(vector_glyph_rows or [])
    if connected_heights:
        heights.extend(connected_heights)
        for row in vector_glyph_rows or []:
            bbox = _xywh(row)
            if bbox is not None:
                widths.append(abs(float(bbox["w"])))
    for row in decimal_rows:
        bbox = _xywh(row)
        if bbox is not None and not connected_heights:
            widths.append(abs(float(bbox["w"])))
            heights.append(abs(float(bbox["h"])) * 3.0)
        detail = row.get("detail") or {}
        stroke = _float_or_none(detail.get("stroke_width"))
        if stroke is not None and stroke > 0:
            stroke_widths.append(stroke)
    for row in degree_rows:
        bbox = _xywh(row)
        if bbox is not None and not connected_heights:
            widths.append(abs(float(bbox["w"])))
            heights.append(abs(float(bbox["h"])))
    glyph_h = median(heights) if heights else 8.0
    glyph_w = median(widths) if widths else max(glyph_h * 0.55, 1.0)
    stroke_w = median(stroke_widths) if stroke_widths else max(glyph_h * 0.025, 0.1)
    return {
        "glyph_h": round(float(glyph_h), 3),
        "glyph_w": round(float(glyph_w), 3),
        "stroke_w": round(float(stroke_w), 4),
        "source": "connected_glyph_group_median" if connected_heights else "decimal_degree_median",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _glyph_specs_from_scale(scale: dict[str, Any]) -> dict[str, Any]:
    glyph_h = max(_float_or(scale.get("glyph_h"), 8.0), 1.0)
    stroke_w = max(_float_or(scale.get("stroke_w"), glyph_h * 0.025), 0.01)
    return {
        "digit": {
            "height_range": _range(glyph_h * 0.60, glyph_h * 1.50),
            "width_range": _range(glyph_h * 0.20, glyph_h * 1.10),
            "topology_signature": "vector_digit_slot",
        },
        "letter": {
            "height_range": _range(glyph_h * 0.55, glyph_h * 1.55),
            "width_range": _range(glyph_h * 0.20, glyph_h * 1.25),
            "topology_signature": "vector_letter_slot",
        },
        "symbol": {
            "height_range": _range(glyph_h * 0.50, glyph_h * 1.60),
            "width_range": _range(glyph_h * 0.15, glyph_h * 1.60),
            "topology_signature": "vector_symbol_slot",
        },
        "gdt_frame": {
            "height_range": _range(glyph_h, glyph_h * 4.0),
            "line_width_estimate": round(stroke_w, 4),
            "topology_signature": "parallel_line_compartment_frame",
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _strict_pitch_read_evidence_rows(
    dump: dict[str, Any] | None,
    *,
    page_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt trusted R33-M1 whole-phrase reads into dump-only L2 evidence.

    The adapter is deliberately read-only and row-fail-closed.  A malformed or
    unsafe envelope contributes no evidence.  Within a valid envelope, failed,
    malformed, duplicate, empty, or uncalled rows are counted but never enter
    L2 assembly.
    """
    stats: dict[str, Any] = {
        "status": "not_provided",
        "input_read_count": 0,
        "evidence_count": 0,
        "rejected_read_count": 0,
        "rejection_counts": {},
    }
    if dump is None:
        return [], stats

    reads = dump.get("reads") if isinstance(dump, dict) else None
    if isinstance(reads, list):
        stats["input_read_count"] = len(reads)
    envelope_reasons = _strict_pitch_dump_reasons(
        dump,
        page_index=page_index,
    )
    if envelope_reasons:
        stats["status"] = "fail_closed"
        stats["rejected_read_count"] = len(reads) if isinstance(reads, list) else 0
        stats["rejection_counts"] = {
            reason: 1 for reason in envelope_reasons
        }
        return [], stats

    trace_id = str(dump.get("trace_id") or "")
    phrase_id_counts = Counter(
        str(row.get("phrase_id") or "")
        for row in reads
        if isinstance(row, dict) and str(row.get("phrase_id") or "")
    )
    evidence: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row in reads:
        reasons = _strict_pitch_read_row_reasons(
            row,
            trace_id=trace_id,
            page_index=page_index,
            duplicate_phrase_id=bool(
                isinstance(row, dict)
                and phrase_id_counts[str(row.get("phrase_id") or "")] > 1
            ),
        )
        if reasons:
            rejection_counts.update(reasons)
            continue
        # R33-M1 exposes a phrase bbox but no authoritative character bbox.
        # Preserve that PDF-point/top-left rectangle without scaling or offset.
        evidence.append({
            "schema_version": "r21_strict_pitch_phrase_evidence_v1",
            "source_id": str(row["phrase_id"]),
            "phrase_id": str(row["phrase_id"]),
            "text": str(row["text"]),
            "bbox": deepcopy(row["bbox"]),
            "axis_angle_deg": float(row["axis_angle_deg"]),
            # Reader success is not a calibrated probability.  Do not invent
            # one merely to promote a dump-only R21 confidence route.
            "confidence": 0.0,
            "source": "r33_m1_pitch_reader",
            "reader_call_count": 1,
            "consumer_allowed": CONSUMER_ALLOWED,
        })

    stats["status"] = "ok"
    stats["evidence_count"] = len(evidence)
    stats["rejected_read_count"] = len(reads) - len(evidence)
    stats["rejection_counts"] = dict(sorted(rejection_counts.items()))
    return evidence, stats


def _strict_pitch_dump_reasons(
    dump: Any,
    *,
    page_index: int,
) -> list[str]:
    if not isinstance(dump, dict):
        return ["dump_not_object"]
    reasons: list[str] = []
    if dump.get("schema_version") != "vector_pitch_phrase_read_dump_v1":
        reasons.append("dump_schema_invalid")
    if not isinstance(dump.get("trace_id"), str) or not dump.get("trace_id"):
        reasons.append("dump_trace_id_invalid")
    if type(dump.get("page_index")) is not int or dump.get("page_index") != int(page_index):
        reasons.append("dump_page_index_mismatch")
    if dump.get("status") != "ok" or dump.get("error") is not None:
        reasons.append("dump_status_not_ok")
    if not isinstance(dump.get("reads"), list):
        reasons.append("dump_reads_not_list")
    if dump.get("feeds_display_l3") is not False:
        reasons.append("dump_feeds_display_l3_not_false")
    if dump.get("ocr_called") is not False:
        reasons.append("dump_ocr_called_not_false")
    if dump.get("consumer_allowed") is not False:
        reasons.append("dump_consumer_allowed_not_false")
    if vector_artifact_safety_reasons(dump, path="strict_pitch_read_dump"):
        reasons.append("dump_recursive_safety_invalid")
    return sorted(set(reasons))


def _strict_pitch_read_row_reasons(
    row: Any,
    *,
    trace_id: str,
    page_index: int,
    duplicate_phrase_id: bool,
) -> list[str]:
    if not isinstance(row, dict):
        return ["row_not_object"]
    reasons: list[str] = []
    phrase_id = row.get("phrase_id")
    if row.get("schema_version") != "vector_pitch_phrase_read_v1":
        reasons.append("row_schema_invalid")
    if row.get("trace_id") != trace_id:
        reasons.append("row_trace_id_mismatch")
    if type(row.get("page_index")) is not int or row.get("page_index") != int(page_index):
        reasons.append("row_page_index_mismatch")
    if not isinstance(phrase_id, str) or not phrase_id:
        reasons.append("row_phrase_id_invalid")
    elif duplicate_phrase_id:
        reasons.append("row_phrase_id_duplicate")
    if row.get("status") != "read" or row.get("fail_closed_reason") is not None:
        reasons.append("row_status_not_read")
    if (
        type(row.get("reader_call_count")) is not int
        or row.get("reader_call_count") != 1
    ):
        reasons.append("row_reader_not_called_once")
    if not isinstance(row.get("text"), str) or not row.get("text"):
        reasons.append("row_text_empty_or_invalid")
    if _strict_pitch_bbox(row.get("bbox")) is None:
        reasons.append("row_bbox_invalid")
    if (
        type(row.get("axis_angle_deg")) not in {int, float}
        or isinstance(row.get("axis_angle_deg"), bool)
        or not math.isfinite(float(row.get("axis_angle_deg")))
        or row.get("axis_trusted") is not True
    ):
        reasons.append("row_axis_invalid")
    if not isinstance(row.get("spans"), list):
        reasons.append("row_spans_invalid")
    if row.get("feeds_display_l3") is not False:
        reasons.append("row_feeds_display_l3_not_false")
    if row.get("ocr_called") is not False:
        reasons.append("row_ocr_called_not_false")
    if row.get("consumer_allowed") is not False:
        reasons.append("row_consumer_allowed_not_false")
    return sorted(set(reasons))


def _strict_pitch_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, float] = {}
    for key in ("x", "y", "w", "h"):
        raw = value.get(key)
        if type(raw) not in {int, float} or isinstance(raw, bool):
            return None
        number = float(raw)
        if not math.isfinite(number):
            return None
        parsed[key] = number
    if parsed["w"] <= 0.0 or parsed["h"] <= 0.0:
        return None
    return parsed


def _local_glyphs_near_refs(
    *,
    anchors: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    glyph_h: float,
) -> list[dict[str, Any]]:
    local: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        bbox = _xywh(row)
        center = _center(row, bbox)
        if bbox is None or center is None:
            continue
        best: tuple[float, dict[str, Any], float] | None = None
        for anchor in anchors:
            angle = _float_or_none(anchor.get("axis_angle_deg"))
            ac = anchor.get("center") or {}
            if angle is None or "x" not in ac or "y" not in ac:
                continue
            main, cross = _project(
                (float(center["x"]), float(center["y"])),
                (float(ac["x"]), float(ac["y"])),
                angle,
            )
            if abs(cross) > max(glyph_h * 1.5, 6.0):
                continue
            if abs(main) > max(glyph_h * 8.0, 64.0):
                continue
            score = abs(cross) + abs(main) * 0.05
            if best is None or score < best[0]:
                best = (score, anchor, main)
        if best is None:
            continue
        source_id = _source_id(row)
        if source_id in seen:
            continue
        seen.add(source_id)
        anchor = best[1]
        local.append({
            "schema_version": L2_SCHEMA_VERSION,
            "source_id": source_id,
            "anchor_id": anchor.get("anchor_id"),
            "anchor_kind": anchor.get("kind"),
            "text": _glyph_text(row),
            "bbox": _round_bbox(bbox),
            "center": _round_point(center),
            "axis_angle_deg": anchor.get("axis_angle_deg"),
            "main_offset": round(float(best[2]), 3),
            "confidence": _confidence(row),
            "source": _glyph_source(row),
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    local.sort(key=lambda row: (str(row.get("anchor_id") or ""), float(row.get("main_offset") or 0.0)))
    return local


def _replace_overlapping_pm_glyphs(
    *,
    local_glyphs: list[dict[str, Any]],
    pm_rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pm_tokens = _pm_tokens_near_refs(pm_rows=pm_rows, anchors=anchors)
    if not pm_tokens:
        return local_glyphs
    kept: list[dict[str, Any]] = []
    used_pm_source_ids: set[str] = set()
    for glyph in local_glyphs:
        glyph_bbox = _xywh(glyph)
        glyph_center = _center(glyph, glyph_bbox)
        overlapping_pm = [
            pm_token for pm_token in pm_tokens
            if glyph_bbox is not None
            and glyph_center is not None
            and _pm_overlaps_glyph(pm_token, glyph_bbox, glyph_center)
        ]
        if overlapping_pm:
            used_pm_source_ids.update(str(pm_token.get("source_id") or "") for pm_token in overlapping_pm)
            continue
        kept.append(glyph)
    kept.extend(pm_tokens)
    kept.sort(key=lambda row: (str(row.get("anchor_id") or ""), float(row.get("main_offset") or 0.0)))
    return kept


def _pm_tokens_near_refs(
    *,
    pm_rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchor_by_id = {str(anchor.get("anchor_id") or ""): anchor for anchor in anchors}
    tokens: list[dict[str, Any]] = []
    for row in pm_rows or []:
        if row.get("consumer_allowed") is True:
            continue
        if str(row.get("source") or "") != "pm_strict":
            continue
        bbox = _xywh(row)
        center = _center(row, bbox)
        anchor = anchor_by_id.get(str(row.get("anchor_id") or ""))
        if bbox is None or center is None or anchor is None:
            continue
        angle = _float_or_none(row.get("axis_angle_deg"))
        ac = anchor.get("center") or {}
        if angle is None or "x" not in ac or "y" not in ac:
            continue
        main, _cross = _project(
            (float(center["x"]), float(center["y"])),
            (float(ac["x"]), float(ac["y"])),
            angle,
        )
        tokens.append({
            "schema_version": L2_SCHEMA_VERSION,
            "source_id": str(row.get("id") or ""),
            "anchor_id": anchor.get("anchor_id"),
            "anchor_kind": anchor.get("kind"),
            "text": _pm_symbol_text(row.get("symbol")),
            "bbox": _round_bbox(bbox),
            "center": _round_point(center),
            "axis_angle_deg": round(float(angle) % 180.0, 6),
            "main_offset": round(float(main), 3),
            "confidence": _float_or(row.get("confidence"), 0.98),
            "source": "pm_strict",
            "segment_ids": list(row.get("segment_ids") or []),
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return tokens


def _pm_overlaps_glyph(
    pm_token: dict[str, Any],
    glyph_bbox: dict[str, float],
    glyph_center: dict[str, float],
) -> bool:
    pm_bbox = _xywh(pm_token)
    if pm_bbox is None:
        return False
    # R22 B replacement gate: dirty vector_glyph tokens are removed only when
    # their center falls inside the strict pm bbox, or when bbox IoU is clear.
    if _point_inside_bbox((float(glyph_center["x"]), float(glyph_center["y"])), _padded_bbox(pm_bbox, 0.75)):
        return True
    inter = _intersection_area(pm_bbox, glyph_bbox)
    if inter <= 0:
        return False
    pm_area = max(float(pm_bbox["w"]) * float(pm_bbox["h"]), 1e-6)
    glyph_area = max(float(glyph_bbox["w"]) * float(glyph_bbox["h"]), 1e-6)
    union = pm_area + glyph_area - inter
    return inter / max(union, 1e-6) >= 0.08


def _pm_symbol_text(symbol: Any) -> str:
    return {
        "plus_minus": "±",
        "plus": "+",
        "minus": "-",
    }.get(str(symbol or ""), str(symbol or ""))


def _padded_bbox(bbox: dict[str, float], pad: float) -> dict[str, float]:
    return {
        "x": float(bbox["x"]) - float(pad),
        "y": float(bbox["y"]) - float(pad),
        "w": float(bbox["w"]) + float(pad) * 2.0,
        "h": float(bbox["h"]) + float(pad) * 2.0,
    }


def _canonical_gdt_frames(
    line_frames: list[dict[str, Any]],
    decimal_seeded_frames: list[dict[str, Any]],
    *,
    page_index: int,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for row in line_frames:
        bbox = _xywh(row.get("bbox") or row)
        if bbox is None:
            continue
        frames.append({
            "schema_version": L2_SCHEMA_VERSION,
            "page_index": int(page_index),
            "frame_id": f"p{int(page_index) + 1:03d}_gdt_line_{len(frames):06d}",
            "source_id": str(row.get("id") or ""),
            "bbox": _round_bbox(bbox),
            "compartments": [_round_bbox(box) for box in _valid_bboxes(row.get("compartments") or [])],
            "confidence": _confidence(row),
            "source": "detect_gdt_frames_from_lines",
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    for index, row in enumerate(decimal_seeded_frames):
        bbox = _xywh(row.get("bbox") or row)
        if bbox is None:
            continue
        frames.append({
            "schema_version": L2_SCHEMA_VERSION,
            "page_index": int(page_index),
            "frame_id": f"p{int(page_index) + 1:03d}_gdt_decimal_seeded_{index:06d}",
            "source_id": str(row.get("id") or ""),
            "bbox": _round_bbox(bbox),
            "compartments": [_round_bbox(box) for box in _valid_bboxes(row.get("compartments") or [])],
            "confidence": _confidence(row),
            "source": str(row.get("source") or "gdt_line_decimal_seeded_vertical_shadow"),
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return _dedupe_gdt_frames_cross_source(frames)


def _canonical_gdt_fallback_frames(
    line_frames: list[dict[str, Any]],
    *,
    page_index: int,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for row in line_frames:
        bbox = _xywh(row.get("bbox") or row)
        if bbox is None:
            continue
        frames.append({
            "schema_version": L2_SCHEMA_VERSION,
            "page_index": int(page_index),
            "frame_id": f"p{int(page_index) + 1:03d}_gdt_fallback_{len(frames):06d}",
            "source_id": str(row.get("id") or ""),
            "bbox": _round_bbox(bbox),
            "compartments": [_round_bbox(box) for box in _valid_bboxes(row.get("compartments") or [])],
            "confidence": _confidence(row),
            "source": "detect_gdt_frames_from_lines",
            "fallback_reason": "no_l1_seed_confirmation",
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return frames


def _partition_gdt_line_frames_by_seed(
    line_frames: list[dict[str, Any]],
    *,
    anchors: list[dict[str, Any]],
    glyph_h: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    confirmed: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    seed_anchors = [
        anchor for anchor in anchors
        if anchor.get("kind") in {"dot", "degree", "diameter"} and isinstance(anchor.get("center"), dict)
    ]
    for frame in line_frames:
        if _gdt_frame_has_seed_confirmation(frame, seed_anchors, glyph_h=glyph_h):
            confirmed.append(frame)
        else:
            fallback.append(frame)
    return confirmed, fallback


def _gdt_frame_has_seed_confirmation(
    frame: dict[str, Any],
    anchors: list[dict[str, Any]],
    *,
    glyph_h: float,
) -> bool:
    bbox = _xywh(frame.get("bbox") or frame)
    if bbox is None:
        return False
    frame_h = max(float(bbox["h"]), _float_or(frame.get("frame_height"), 0.0), 1.0)
    for anchor in anchors:
        center = anchor.get("center") or {}
        try:
            point = (float(center["x"]), float(center["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if _point_inside_bbox(point, _padded_bbox(bbox, frame_h)):
            return True
        if _point_to_bbox_distance(point, bbox) <= max(4.0 * float(glyph_h), 1.0):
            return True
    return False


def _dedupe_gdt_frames_cross_source(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def priority(frame: dict[str, Any]) -> tuple[int, float, float]:
        source = str(frame.get("source") or "")
        source_rank = 0 if source != "detect_gdt_frames_from_lines" else 1
        comp_count = float(len(frame.get("compartments") or []))
        return source_rank, -_confidence(frame), -comp_count

    final: list[dict[str, Any]] = []
    for frame in sorted(frames, key=priority):
        bbox = _xywh(frame.get("bbox") or frame)
        if bbox is None:
            continue
        if any(
            _bbox_iou_xywh(bbox, _xywh(existing.get("bbox") or existing) or {}) > 0.35
            or _shared_horizontal_edge_xywh(bbox, _xywh(existing.get("bbox") or existing) or {})
            for existing in final
        ):
            continue
        final.append(frame)
    final.sort(key=lambda row: str(row.get("frame_id") or ""))
    return final


def _point_to_bbox_distance(point: tuple[float, float], bbox: dict[str, float]) -> float:
    x0 = float(bbox["x"])
    y0 = float(bbox["y"])
    x1 = x0 + float(bbox["w"])
    y1 = y0 + float(bbox["h"])
    dx = max(x0 - point[0], 0.0, point[0] - x1)
    dy = max(y0 - point[1], 0.0, point[1] - y1)
    return math.hypot(dx, dy)


def _bbox_iou_xywh(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = max(float(a["w"]) * float(a["h"]), 1e-6)
    area_b = max(float(b["w"]) * float(b["h"]), 1e-6)
    return inter / max(area_a + area_b - inter, 1e-6)


def _shared_horizontal_edge_xywh(a: dict[str, float], b: dict[str, float]) -> bool:
    if not a or not b:
        return False
    a_edges = [float(a["y"]), float(a["y"]) + float(a["h"])]
    b_edges = [float(b["y"]), float(b["y"]) + float(b["h"])]
    if not any(abs(a_edge - b_edge) <= 1.5 for a_edge in a_edges for b_edge in b_edges):
        return False
    ax0 = float(a["x"])
    ax1 = ax0 + float(a["w"])
    bx0 = float(b["x"])
    bx1 = bx0 + float(b["w"])
    overlap = min(ax1, bx1) - max(ax0, bx0)
    if overlap <= 0:
        return False
    return overlap / max(min(ax1 - ax0, bx1 - bx0), 1e-6) >= 0.60


def _glyphs_inside_key_capsules(
    *,
    capsules: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    existing_source_ids: set[str],
) -> list[dict[str, Any]]:
    local: list[dict[str, Any]] = []
    seen = set(existing_source_ids)
    for capsule in capsules:
        capsule_bbox = _xywh(capsule.get("bbox") or capsule)
        if capsule_bbox is None:
            continue
        for row in rows:
            source_id = _source_id(row)
            if source_id in seen:
                continue
            bbox = _xywh(row)
            center = _center(row, bbox)
            if bbox is None or center is None:
                continue
            if not _point_inside_bbox((float(center["x"]), float(center["y"])), capsule_bbox):
                continue
            seen.add(source_id)
            local.append({
                "schema_version": L2_SCHEMA_VERSION,
                "source_id": source_id,
                "anchor_id": capsule.get("capsule_id"),
                "anchor_kind": "key_capsule",
                "key_capsule_id": capsule.get("capsule_id"),
                "text": _glyph_text(row),
                "bbox": _round_bbox(bbox),
                "center": _round_point(center),
                "axis_angle_deg": _float_or(capsule.get("angle"), 0.0),
                "main_offset": round(float(center["x"]) - float(capsule_bbox["x"]), 3),
                "confidence": _confidence(row),
                "source": _glyph_source(row),
                "consumer_allowed": CONSUMER_ALLOWED,
            })
    local.sort(key=lambda row: (
        str(row.get("key_capsule_id") or ""),
        float((row.get("center") or {}).get("x", 0.0)),
        float((row.get("center") or {}).get("y", 0.0)),
    ))
    return local


def _canonical_key_capsules(rows: list[dict[str, Any]], *, page_index: int) -> list[dict[str, Any]]:
    capsules: list[dict[str, Any]] = []
    for row in rows:
        bbox = _xywh(row)
        if bbox is None:
            continue
        capsules.append({
            "capsule_id": f"p{int(page_index) + 1:03d}_key_capsule_{len(capsules):06d}",
            "bbox": _round_bbox(bbox),
            "source": "pdf_analyzer_capsule.detect_capsules",
            "key_role": "rounded_rectangle_key_dimension",
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return capsules


def _axis_run(anchor: dict[str, Any], glyphs: list[dict[str, Any]], *, glyph_h: float) -> list[dict[str, Any]]:
    angle = _float_or_none(anchor.get("axis_angle_deg"))
    ac = anchor.get("center") or {}
    if angle is None or "x" not in ac or "y" not in ac:
        return []
    anchor_id = str(anchor.get("anchor_id") or "")
    origin = (float(ac["x"]), float(ac["y"]))
    run: list[tuple[float, dict[str, Any]]] = []
    for glyph in glyphs:
        # A strict row already represents a complete phrase and L2 binds it to
        # its nearest reference anchor.  Reusing that whole phrase from a
        # neighbouring anchor would duplicate or concatenate diagnostic
        # dimensions even though the authoritative strict chain did neither.
        # Preserve the legacy character-row geometry path, but keep strict
        # phrase evidence on its explicit L2 anchor identity.
        if (
            glyph.get("source") == "r33_m1_pitch_reader"
            and str(glyph.get("anchor_id") or "") != anchor_id
        ):
            continue
        if _is_self_diameter_glyph(anchor, glyph):
            continue
        bbox = _xywh(glyph)
        center = _center(glyph, bbox)
        if bbox is None or center is None:
            continue
        main, cross = _project((float(center["x"]), float(center["y"])), origin, angle)
        if abs(cross) > max(glyph_h * 1.5, 6.0) or abs(main) > max(glyph_h * 8.0, 64.0):
            continue
        run.append((main, glyph))
    return [row for _, row in sorted(run, key=lambda item: item[0])]


def _is_self_diameter_glyph(anchor: dict[str, Any], glyph: dict[str, Any]) -> bool:
    if anchor.get("kind") != "diameter":
        return False
    if _glyph_text(glyph) not in {"⌀", "∅", "Ø", "ø", "Φ", "φ"}:
        return False
    anchor_bbox = _xywh(anchor)
    glyph_bbox = _xywh(glyph)
    if anchor_bbox is None or glyph_bbox is None:
        return False
    inter = _intersection_area(anchor_bbox, glyph_bbox)
    if inter <= 0:
        return False
    anchor_area = max(float(anchor_bbox["w"]) * float(anchor_bbox["h"]), 1e-6)
    glyph_area = max(float(glyph_bbox["w"]) * float(glyph_bbox["h"]), 1e-6)
    return inter / max(min(anchor_area, glyph_area), 1e-6) >= 0.60


def _capsule_run(capsule: dict[str, Any], glyphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    capsule_bbox = _xywh(capsule.get("bbox") or capsule)
    if capsule_bbox is None:
        return []
    run = []
    for glyph in glyphs:
        bbox = _xywh(glyph)
        center = _center(glyph, bbox)
        if bbox is None or center is None:
            continue
        if _point_inside_bbox((float(center["x"]), float(center["y"])), capsule_bbox):
            run.append(glyph)
    return sorted(run, key=lambda row: (float((_center(row, _xywh(row)) or {}).get("x", 0.0)), float((_center(row, _xywh(row)) or {}).get("y", 0.0))))


_R27_L3_ALLOWED_TEMPLATE_DECISIONS = frozenset({
    "accepted_template",
    "accepted_template_conditional",
})
_R27_STRICT_KNOWN_SYMBOL_DETECTORS = {
    "plus_minus_pool_excluded": "pm_strict",
    "degree_anchor_symbol_pool_excluded": "degree_polyline_strict_anchor",
    "diameter_anchor_symbol_pool_excluded": "diameter_glyph_detector",
}


def _dedupe_r27_label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        source_id = str(row.get("source_id") or "")
        label_source = str(row.get("label_source") or "")
        label = str(row.get("label") or "")
        key = (source_id, label_source, label)
        if not source_id or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _r27_digit_cluster_context(probe: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(probe, dict):
        return {
            "enabled": False,
            "labels_by_source_id": {},
            "labels_by_anchor": {},
            "reads_by_anchor": {},
            "label_count": 0,
            "complete_anchor_count": 0,
            "phrase_complete_anchor_count": 0,
            "fragment_anchor_count": 0,
            "phrase_boundary_producer": None,
            "phrase_gate_status": "not_applicable",
        }

    # The R27 sidecar is permanently a cluster/coverage diagnostic.  A real
    # R32 phrase-region producer must arrive through its own typed core seam;
    # declarations embedded in an R27 JSON are never authority to promote L3.
    phrase_boundary_producer = None
    phrase_gate_status = "fail_closed_missing_phrase_boundary_producer"

    clusters_by_id = {
        str(row.get("cluster_id") or ""): row
        for row in probe.get("clusters") or []
        if isinstance(row, dict) and str(row.get("cluster_id") or "")
    }
    labels_by_source_id: dict[str, dict[str, Any]] = {}
    labels_by_anchor: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in probe.get("candidates") or []:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "")
        label = str(row.get("propagated_label") or "")
        cluster_id = str(row.get("cluster_id") or "")
        cluster = clusters_by_id.get(cluster_id) or {}
        classifier = cluster.get("classifier") or {}
        decision = str(classifier.get("decision") or "")
        if not source_id or not label:
            continue
        if decision not in _R27_L3_ALLOWED_TEMPLATE_DECISIONS:
            continue
        label_row = {
            "source_id": source_id,
            "candidate_id": row.get("candidate_id"),
            "anchor_id": row.get("anchor_id"),
            "cluster_id": cluster_id,
            "label": label,
            "label_source": "template_cluster",
            "decision": decision,
            "status": classifier.get("status"),
            "top1_distance": classifier.get("top1_distance"),
            "actual_margin": classifier.get("actual_margin"),
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        labels_by_source_id[source_id] = label_row
        labels_by_anchor[str(row.get("anchor_id") or "")].append(label_row)

    for row in probe.get("known_symbols") or []:
        if not isinstance(row, dict):
            continue
        if not _r27_known_symbol_row_allowed(row):
            continue
        source_id = str(row.get("source_id") or "")
        anchor_id = str(row.get("anchor_id") or "")
        label = str(row.get("label") or row.get("propagated_label") or "")
        if not source_id or not anchor_id or not label:
            continue
        label_row = {
            "source_id": source_id,
            "known_symbol_id": row.get("known_symbol_id"),
            "anchor_id": anchor_id,
            "cluster_id": None,
            "label": label,
            "label_source": "known_symbol_detector",
            "decision": row.get("decision") or "known_symbol_detector",
            "status": row.get("status"),
            "detector_source": row.get("detector_source"),
            "pool_exclusion_reason": row.get("pool_exclusion_reason"),
            "label_source_reason": row.get("label_source_reason"),
            "top1_distance": None,
            "actual_margin": None,
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        labels_by_source_id[source_id] = label_row
        labels_by_anchor[anchor_id].append(label_row)

    reads_by_anchor = {
        str(row.get("anchor_id") or ""): _r27_apply_phrase_producer_gate(
            row,
            phrase_boundary_producer=phrase_boundary_producer,
            phrase_gate_status=phrase_gate_status,
        )
        for row in probe.get("reads_by_anchor") or []
        if isinstance(row, dict) and str(row.get("anchor_id") or "")
    }
    return {
        "enabled": True,
        "labels_by_source_id": labels_by_source_id,
        "labels_by_anchor": {
            anchor_id: _dedupe_r27_label_rows(rows)
            for anchor_id, rows in labels_by_anchor.items()
        },
        "reads_by_anchor": reads_by_anchor,
        "label_count": len(labels_by_source_id),
        "complete_anchor_count": sum(
            1 for row in reads_by_anchor.values()
            if row.get("has_numeric_value") is True
        ),
        "phrase_complete_anchor_count": sum(
            1 for row in reads_by_anchor.values()
            if row.get("phrase_complete") is True
        ),
        "fragment_anchor_count": sum(
            1 for row in reads_by_anchor.values()
            if row.get("coverage_status") == "fragment"
        ),
        "phrase_boundary_producer": phrase_boundary_producer,
        "phrase_gate_status": phrase_gate_status,
    }


def _r27_apply_phrase_producer_gate(
    read: dict[str, Any],
    *,
    phrase_boundary_producer: str | None,
    phrase_gate_status: str,
) -> dict[str, Any]:
    gated = dict(read)
    boundary_source = str(gated.get("phrase_boundary_source") or "")
    boundary_complete = bool(
        phrase_boundary_producer
        and phrase_gate_status == "operational"
        and gated.get("phrase_boundary_complete") is True
        and boundary_source == phrase_boundary_producer
    )
    if boundary_complete:
        gated["phrase_gate_status"] = phrase_gate_status
        return gated

    reasons = list(gated.get("phrase_reasons") or [])
    if "missing_phrase_boundary_evidence" not in reasons:
        reasons.append("missing_phrase_boundary_evidence")
    gated.update({
        "phrase_complete": False,
        "phrase_boundary_complete": False,
        "phrase_boundary_source": "missing",
        "phrase_gate_status": (
            "fail_closed_missing_phrase_boundary_producer"
            if not phrase_boundary_producer
            else phrase_gate_status
        ),
        "phrase_status": "incomplete",
        "phrase_reasons": reasons,
    })
    return gated


def _r27_known_symbol_row_allowed(row: dict[str, Any]) -> bool:
    reason = str(row.get("label_source_reason") or "")
    expected_detector = _R27_STRICT_KNOWN_SYMBOL_DETECTORS.get(reason)
    if not expected_detector:
        return False
    if str(row.get("pool_exclusion_reason") or reason) != reason:
        return False
    if str(row.get("detector_source") or "") != expected_detector:
        return False
    label = str(row.get("label") or row.get("propagated_label") or "")
    if reason == "plus_minus_pool_excluded":
        return label in {"+", "-", "±"}
    if reason == "degree_anchor_symbol_pool_excluded":
        return label == "°"
    if reason == "diameter_anchor_symbol_pool_excluded":
        return label in {"⌀", "∅", "Ø", "ø", "Φ", "φ"}
    return False


def _r27_l3_read_evidence(
    *,
    read: dict[str, Any] | None,
    label_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    read = read or {}
    return {
        "schema_version": "r27_digit_cluster_l3_read_evidence_v1",
        "anchor_id": read.get("anchor_id") or (
            label_rows[0].get("anchor_id") if label_rows else ""
        ),
        "coverage_status": read.get("coverage_status"),
        "coverage_complete": bool(read.get("coverage_complete")),
        "coverage_has_numeric_value": bool(read.get("coverage_has_numeric_value")),
        "raw_has_numeric_value": bool(read.get("raw_has_numeric_value")),
        "has_numeric_value": bool(read.get("has_numeric_value")),
        "phrase_complete": bool(read.get("phrase_complete")),
        "phrase_boundary_complete": bool(read.get("phrase_boundary_complete")),
        "phrase_boundary_source": read.get("phrase_boundary_source"),
        "phrase_gate_status": read.get("phrase_gate_status"),
        "phrase_status": read.get("phrase_status"),
        "phrase_reasons": list(read.get("phrase_reasons") or []),
        "phrase_validation": dict(read.get("phrase_validation") or {}),
        "unsupported_labeled_tokens": list(read.get("unsupported_labeled_tokens") or []),
        "probe_text": read.get("probe_text"),
        "dimension_label_text": read.get("dimension_label_text"),
        "normalized_probe_value": read.get("normalized_probe_value"),
        "anchor_window_glyph_count": read.get("anchor_window_glyph_count"),
        "clustered_glyph_count": read.get("clustered_glyph_count"),
        "known_symbol_labeled_count": read.get("known_symbol_labeled_count"),
        "accounted_glyph_count": read.get("accounted_glyph_count"),
        "labeled_glyph_count": read.get("labeled_glyph_count"),
        "fragment_reasons": list(read.get("fragment_reasons") or []),
        "accepted_label_count_in_l3_run": len(label_rows),
        "accepted_labels": [
            {
                "source_id": row.get("source_id"),
                "candidate_id": row.get("candidate_id"),
                "known_symbol_id": row.get("known_symbol_id"),
                "cluster_id": row.get("cluster_id"),
                "label": row.get("label"),
                "label_source": row.get("label_source"),
                "decision": row.get("decision"),
                "detector_source": row.get("detector_source"),
                "pool_exclusion_reason": row.get("pool_exclusion_reason"),
                "label_source_reason": row.get("label_source_reason"),
                "top1_distance": row.get("top1_distance"),
                "actual_margin": row.get("actual_margin"),
                "consumer_allowed": CONSUMER_ALLOWED,
            }
            for row in label_rows
        ],
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _r27_read_can_set_l3_value(
    *,
    read: dict[str, Any] | None,
    label_rows: list[dict[str, Any]],
    phrase_boundary_producer: str | None,
    phrase_gate_status: str,
) -> bool:
    if not isinstance(read, dict) or read.get("has_numeric_value") is not True:
        return False
    if not phrase_boundary_producer or phrase_gate_status != "operational":
        return False
    if read.get("coverage_complete") is not True:
        return False
    if (read.get("phrase_validation") or {}).get("valid") is not True:
        return False
    if read.get("phrase_boundary_complete") is not True:
        return False
    if str(read.get("phrase_boundary_source") or "") != phrase_boundary_producer:
        return False
    if read.get("phrase_complete") is not True:
        return False
    labeled_count = int(_float_or(read.get("labeled_glyph_count"), 0.0))
    if labeled_count <= 0:
        return False
    return len(label_rows) >= labeled_count


def _dimension_from_run(
    run: list[dict[str, Any]],
    *,
    anchor: dict[str, Any],
    page_index: int,
    order: int,
    r27_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    r27_read = None
    r27_label_rows: list[dict[str, Any]] = []
    if r27_context and r27_context.get("enabled"):
        anchor_id = str(anchor.get("anchor_id") or "")
        r27_read = (r27_context.get("reads_by_anchor") or {}).get(anchor_id)
        labels_by_source_id = r27_context.get("labels_by_source_id") or {}
        labels_by_anchor = r27_context.get("labels_by_anchor") or {}
        for row in run:
            label_row = labels_by_source_id.get(_source_id(row))
            if label_row is not None:
                r27_label_rows.append(label_row)
        r27_label_rows = _dedupe_r27_label_rows(
            r27_label_rows + list(labels_by_anchor.get(anchor_id) or [])
        )

    r27_can_set_value = _r27_read_can_set_l3_value(
        read=r27_read,
        label_rows=r27_label_rows,
        phrase_boundary_producer=(r27_context or {}).get("phrase_boundary_producer"),
        phrase_gate_status=str((r27_context or {}).get("phrase_gate_status") or ""),
    )
    if r27_can_set_value:
        text = str(r27_read.get("dimension_label_text") or "")
    else:
        text = "".join(_glyph_text(row) for row in run)
    if not text:
        return None
    angle = _float_or(anchor.get("axis_angle_deg"), 0.0)
    oriented = _oriented_bbox_for_rows(run, angle_deg=angle)
    normalized = _normalize_dimension_value(text)
    if r27_read and not r27_can_set_value:
        normalized = ""
    flags = _semantic_flags(text, normalized)
    if r27_read and r27_read.get("coverage_status") == "fragment":
        if "suspect_r27_digit_fragment" not in flags:
            flags.append("suspect_r27_digit_fragment")
    if r27_read and r27_read.get("phrase_complete") is not True:
        if "suspect_r27_phrase_incomplete" not in flags:
            flags.append("suspect_r27_phrase_incomplete")
    confidence = min((_confidence(row) for row in run), default=0.0)
    if flags:
        confidence = min(confidence, 0.83)
    out = {
        "schema_version": L3_SCHEMA_VERSION,
        "page_index": int(page_index),
        "dimension_id": f"p{int(page_index) + 1:03d}_l3_dim_{int(order):06d}",
        "value_text": text,
        "normalized_value_text": normalized,
        "oriented_bbox": oriented,
        "anchor_kind": anchor.get("kind"),
        "anchor_id": anchor.get("anchor_id"),
        "pre_confidence": round(float(confidence), 4),
        "flags": flags,
        "source_glyph_ids": [_source_id(row) for row in run],
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if r27_read is not None or r27_label_rows:
        out["r27_digit_cluster_read"] = _r27_l3_read_evidence(
            read=r27_read,
            label_rows=r27_label_rows,
        )
    return out


def _dimension_from_capsule_run(
    run: list[dict[str, Any]],
    *,
    capsule: dict[str, Any],
    page_index: int,
    order: int,
) -> dict[str, Any] | None:
    text = "".join(_glyph_text(row) for row in run)
    if not text:
        return None
    normalized = _normalize_dimension_value(text)
    flags = ["key_capsule"]
    flags.extend(_semantic_flags(text, normalized))
    return {
        "schema_version": L3_SCHEMA_VERSION,
        "page_index": int(page_index),
        "dimension_id": f"p{int(page_index) + 1:03d}_l3_dim_{int(order):06d}",
        "value_text": text,
        "normalized_value_text": normalized,
        "oriented_bbox": _oriented_bbox_for_rows(run, angle_deg=_float_or(capsule.get("angle"), 0.0)),
        "anchor_kind": "key_capsule",
        "key_capsule_id": capsule.get("capsule_id"),
        "pre_confidence": min((_confidence(row) for row in run), default=0.9),
        "flags": flags,
        "source_glyph_ids": [_source_id(row) for row in run],
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _oriented_bbox_for_rows(rows: list[dict[str, Any]], *, angle_deg: float) -> dict[str, Any]:
    points: list[tuple[float, float]] = []
    for row in rows:
        bbox = _xywh(row)
        if bbox is None:
            continue
        points.extend(_bbox_corners(bbox))
    if not points:
        return {"angle_deg": round(float(angle_deg), 3), "quad": [], "bbox": None}
    theta = math.radians(float(angle_deg))
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -uy, ux
    proj = [(x * ux + y * uy, x * vx + y * vy) for x, y in points]
    min_u = min(row[0] for row in proj)
    max_u = max(row[0] for row in proj)
    min_v = min(row[1] for row in proj)
    max_v = max(row[1] for row in proj)
    quad = [
        _from_axis(min_u, min_v, ux, uy, vx, vy),
        _from_axis(max_u, min_v, ux, uy, vx, vy),
        _from_axis(max_u, max_v, ux, uy, vx, vy),
        _from_axis(min_u, max_v, ux, uy, vx, vy),
    ]
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return {
        "angle_deg": round(float(angle_deg), 3),
        "quad": [[round(x, 3), round(y, 3)] for x, y in quad],
        "bbox": _round_bbox({
            "x": min(xs),
            "y": min(ys),
            "w": max(xs) - min(xs),
            "h": max(ys) - min(ys),
        }),
    }


def _semantic_flags(text: str, normalized: str) -> list[str]:
    flags: list[str] = []
    if not normalized:
        flags.append("suspect_no_numeric_value")
    if "+±-" in text or "±±" in text or re.search(r"[+\-±]{2,}", text):
        flags.append("suspect_symbol_pollution")
    if text.count(".") > 1:
        flags.append("suspect_multiple_decimal_points")
    return flags


def _normalize_dimension_value(text: str) -> str:
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return match.group(0) if match else ""


def _is_suspect(row: dict[str, Any]) -> bool:
    return any(str(flag).startswith("suspect") for flag in row.get("flags") or [])


def _clean_symbol_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        out.append(deepcopy(row))
    return out


def _suspect_bbox(suspect: dict[str, Any]) -> dict[str, float] | None:
    oriented = suspect.get("oriented_bbox")
    if isinstance(oriented, dict):
        bbox = _xywh(oriented.get("bbox") or {})
        if bbox is not None:
            return bbox
    return _xywh(suspect.get("bbox") or suspect)


def _decimal_axis_angle(row: dict[str, Any]) -> float | None:
    direct = _float_or_none(row.get("axis_angle_deg"))
    if direct is not None:
        return direct
    short_axis = row.get("short_axis")
    if isinstance(short_axis, dict):
        return _float_or_none(short_axis.get("angle_deg"))
    return None


def _xywh(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict) and "bbox" in value and not all(key in value for key in ("x", "y", "w", "h")):
        nested = xywh_from_any(value.get("bbox"))
        if nested is not None:
            return nested
    if isinstance(value, dict) and all(key in value for key in ("x0", "y0", "x1", "y1")):
        return xywh_from_any(value)
    if isinstance(value, dict) and all(key in value for key in ("x", "y", "w", "h")):
        return xywh_from_any(value)
    return xywh_from_any(value)


def _valid_bboxes(rows: list[Any]) -> list[dict[str, float]]:
    out = []
    for row in rows:
        bbox = _xywh(row)
        if bbox is not None:
            out.append(bbox)
    return out


def _round_bbox(bbox: dict[str, Any] | None) -> dict[str, float]:
    if bbox is None:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    return {key: round(float(bbox[key]), 3) for key in ("x", "y", "w", "h")}


def _bbox_from_points(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def _round_point(point: dict[str, Any] | tuple[float, float]) -> dict[str, float]:
    if isinstance(point, dict):
        return {"x": round(float(point["x"]), 3), "y": round(float(point["y"]), 3)}
    return {"x": round(float(point[0]), 3), "y": round(float(point[1]), 3)}


def _center(row: dict[str, Any], bbox: dict[str, float] | None) -> dict[str, float] | None:
    center = row.get("center") if isinstance(row, dict) else None
    if isinstance(center, dict) and "x" in center and "y" in center:
        try:
            return {"x": float(center["x"]), "y": float(center["y"])}
        except (TypeError, ValueError):
            return None
    if bbox is None:
        return None
    return {"x": float(bbox["x"]) + float(bbox["w"]) / 2.0, "y": float(bbox["y"]) + float(bbox["h"]) / 2.0}


def _center_tuple(row: dict[str, Any]) -> tuple[float, float] | None:
    bbox = _xywh(row)
    center = _center(row, bbox)
    if center is None:
        return None
    return float(center["x"]), float(center["y"])


def _bbox_corners(bbox: dict[str, float]) -> list[tuple[float, float]]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _from_axis(u: float, v: float, ux: float, uy: float, vx: float, vy: float) -> tuple[float, float]:
    return u * ux + v * vx, u * uy + v * vy


def _project(
    point: tuple[float, float],
    origin: tuple[float, float],
    angle_deg: float,
) -> tuple[float, float]:
    theta = math.radians(float(angle_deg))
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -uy, ux
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    return dx * ux + dy * uy, dx * vx + dy * vy


def _point_inside_bbox(point: tuple[float, float], bbox: dict[str, float]) -> bool:
    return (
        float(bbox["x"]) <= point[0] <= float(bbox["x"]) + float(bbox["w"])
        and float(bbox["y"]) <= point[1] <= float(bbox["y"]) + float(bbox["h"])
    )


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    x0 = max(float(a["x"]), float(b["x"]))
    y0 = max(float(a["y"]), float(b["y"]))
    x1 = min(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"]))
    y1 = min(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _glyph_text(row: dict[str, Any]) -> str:
    value = row.get("predicted_text")
    if value is None or value == "":
        value = row.get("text")
    return str(value or "")


def _glyph_source(row: dict[str, Any]) -> str:
    if row.get("match_id"):
        return "vector_digit_match"
    return str(row.get("source") or row.get("glyph_type") or "vector_glyph")


def _source_id(row: dict[str, Any]) -> str:
    if row.get("match_kind") or row.get("schema_version") == "vector_digit_matches_v1":
        return str(row.get("match_id") or row.get("source_token_id") or row.get("token_id") or row.get("id") or row.get("source_id") or "")
    return str(row.get("token_id") or row.get("match_id") or row.get("id") or row.get("source_id") or "")


def _confidence(row: dict[str, Any]) -> float:
    value = row.get("confidence")
    if isinstance(value, str):
        return {"high": 0.95, "medium": 0.75, "low": 0.45}.get(value.lower(), 0.0)
    return _float_or(value, 0.0)


def _range(low: float, high: float) -> list[float]:
    return [round(float(low), 3), round(float(high), 3)]


def _float_or(value: Any, fallback: float) -> float:
    parsed = _float_or_none(value)
    return float(fallback) if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
