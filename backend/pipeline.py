"""
pipeline.py — Type B 主管线抽取

集中协调页面分析阶段，为严格矢量路径及隔离的 legacy 兼容路径提供入口。

用法:
    from pipeline import run_type_b_pipeline

    result = run_type_b_pipeline(
        pdf_bytes=pdf_bytes,
        page=page,                   # pdfplumber page
        page_index=page_index,       # 0-based
        dpi=200,
        gdt_detector=_gdt_detector,  # 可选，None = 跳过 YOLO
        verbose=False,
    )

返回 rich dict：
    - 最终产物: dimensions, references, gdt_frames
    - 视图切分: view_boxes
    - 计时: timings
    - 中间产物（debug 脚本可取用）:
        lines_result, text_regions, rects_result, capsule_candidates,
        diameter_glyphs, gdt_frames_from_lines, gdt_frames_all,
        reconstructed_rects, detected_dim_lines, detected_leader_lines, ocr_results,
        ocr_breakdown, yolo_gdt_results, annotation_lw, frame_borders

严格遵守 CLAUDE.md 里的调用顺序约束：
    1. segment_views
    2. extract_diameter_glyphs
    3. 矢量几何 (lines / rects / capsules / GD&T line frames / arrows)
    4. build_augmented_gdt_frames  ← 统一 GD&T 框候选
    5. OCR passes (candidate-guided flag → directed → compartment → capsule → dimline → hires → strip → 90/270)
    6. dedup_ocr_engineering
    7. inject_diameter_prefix
    8. YOLO-B 分类
    9. assemble_type_b
"""

import math
import time
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from pdf_analyzer import (
    extract_lines_by_width,
    cluster_text_regions,
    extract_rects,
    detect_capsules,
    detect_gdt_frames_from_lines,
    get_annotation_and_contour_lines,
    reconstruct_rectangles_from_lines,
    detect_annotation_linewidth,
    filter_annotation_segments_by_connectivity,
    extract_diameter_glyphs,
)
from ocr_engine import (
    run_directed_ocr,
    run_empty_region_retry,
    run_strip_ocr,
    run_gdt_compartment_ocr,
    run_capsule_ocr,
    run_dimline_strip_ocr,
    run_angle_label_ocr,
    run_radius_label_ocr,
    run_rotated_ocr,
    render_page_to_image,
    inject_diameter_prefix,
    dedup_ocr_by_iou,
    dedup_ocr_engineering,
    transform_region_cw90,
    transform_region_cw270,
    transform_ocr_back_cw90,
    transform_ocr_back_cw270,
)
from arrow_detector import detect_arrows
from assembler import assemble_type_b, build_augmented_gdt_frames
from view_seg_router import get_last_view_seg_status, segment_views
from config import Config
from final_consume_approval import resolve_final_consume_approval
from candidate_builder import build_dimension_candidates, SCHEMA_VERSION as CANDIDATE_SCHEMA_VERSION
from ocr_crop_planner import build_candidate_crop_plans, run_candidate_guided_ocr
from ocr_region_priority import sort_text_regions_for_ocr
from vector_semantics import build_vector_semantics
from gdt_vector_inversion import GDTVectorInverter
from notes_extractor import extract_notes, extract_pdf_text_words
from evidence_shadow import build_evidence_shadow
from key_dimension_classifier import classify_key_dimensions_shadow
from rotated_runway_shadow import build_rotated_runway_shadow_v1
from gdt_vertical_shadow import build_gdt_decimal_seeded_frames_shadow_v1
from view_seg_status_dump import build_view_seg_status_v1
from vector_glyph import build_vector_glyph_v1
from vector_digit_match import (
    build_vector_digit_matches_v1,
    load_vector_digit_prototype_library,
)
from vector_numeric_phrase import build_vector_numeric_phrases_v1
from vector_dimension_hypothesis import build_vector_dimension_hypotheses_v1
from vector_final_consumer import (
    consume_approved_vector_dimension_hypotheses as consume_vector_dimension_hypotheses,
)
from recognition_settings import RecognitionSettings
from ocr_engine_utils import ocr_backend_override
from r2n_nominal_rescue_shadow import build_r2n_nominal_rescue_shadow_v1
from anchor_axis import build_vector_anchor_axis_v1
from ocr_crop_oriented_quad import build_ocr_crop_oriented_quad_v1
from ocr_pass_necessity import build_ocr_pass_necessity_v1
from assembler_polygon_compat import build_assembler_polygon_compat_v1
from decimal_point_quad import apply_decimal_point_quad_pipeline, detect_decimal_point_quads
from eight_line_hex_cross_shadow import (
    detect_eight_line_hex_cross_shadows_v1,
    normalize_eight_line_hex_cross_shadow_rows_v1,
    validate_eight_line_hex_cross_shadow_stats_v1,
)
from decimal_char_height_calibration import (
    build_decimal_char_height_v1,
    decimal_char_height_dump_enabled,
)
from decimal_corridor_reconstruct import build_decimal_corridors_v1
from degree_polyline_strict import build_degree_polyline_pipeline_fields, merge_degree_polyline_glyph_dump
from vector_times_detector import detect_vector_times_candidates
from vector_prelabel_fast import run_vector_prelabel_fast_profile
from vector_only_contract import (
    VectorOnlyStageUnavailable,
    zero_ocr_breakdown,
)
from yolo_view_async import (
    StrictYoloViewUnavailable,
    resolve_strict_yolo_view_source,
)
from vector_only_runtime_guard import (
    VectorOnlyRuntimeForbidden,
    guarded_consumer_entrypoint,
    guarded_ocr_entrypoint,
)
from vector_page_context import VectorPageContext
from directional_walk_pipeline import (
    build_shape_native_directional_walk_dump_v1,
)
from r21_layered_vector_dimension import (
    build_r21_layered_vector_dimension_shadow,
    detect_degree_polylines_with_recall,
)
from r33_m1_phrase_runtime import (
    R33M1PhraseRuntime,
    R33M1RuntimeUnavailable,
    build_r33_m1_phrase_chain,
    build_r33_m1_unavailable_audit,
    load_r33_m1_phrase_runtime,
)
from review_candidates_contract import (
    ROOT_KEY,
    build_review_candidates_v1,
    validate_review_candidates_v1,
)
from strict_gdt_candidates import (
    AUDIT_FIELD as STRICT_GDT_AUDIT_FIELD,
    build_strict_gdt_review_candidates,
    build_strict_gdt_symbol_evidence,
    empty_strict_gdt_candidate_result,
)
from strict_diagnostic_payload import (
    split_strict_pipeline_product_v1,
    strict_diagnostic_capture_active_v1,
    validate_r21_pipeline_diagnostics_v1,
)


run_directed_ocr = guarded_ocr_entrypoint("run_directed_ocr", run_directed_ocr)
run_empty_region_retry = guarded_ocr_entrypoint(
    "run_empty_region_retry",
    run_empty_region_retry,
)
run_strip_ocr = guarded_ocr_entrypoint("run_strip_ocr", run_strip_ocr)
run_gdt_compartment_ocr = guarded_ocr_entrypoint(
    "run_gdt_compartment_ocr",
    run_gdt_compartment_ocr,
)
run_capsule_ocr = guarded_ocr_entrypoint("run_capsule_ocr", run_capsule_ocr)
run_dimline_strip_ocr = guarded_ocr_entrypoint(
    "run_dimline_strip_ocr",
    run_dimline_strip_ocr,
)
run_angle_label_ocr = guarded_ocr_entrypoint(
    "run_angle_label_ocr",
    run_angle_label_ocr,
)
run_radius_label_ocr = guarded_ocr_entrypoint(
    "run_radius_label_ocr",
    run_radius_label_ocr,
)
run_rotated_ocr = guarded_ocr_entrypoint("run_rotated_ocr", run_rotated_ocr)
run_candidate_guided_ocr = guarded_ocr_entrypoint(
    "run_candidate_guided_ocr",
    run_candidate_guided_ocr,
)
consume_vector_dimension_hypotheses = guarded_consumer_entrypoint(
    "consume_vector_dimension_hypotheses",
    consume_vector_dimension_hypotheses,
)


VECTOR_DIGIT_SLOT_EXCLUSION_PAD_PT = 18.0


class _DeepcopySafePageProxy:
    """Proxy a fitz page while keeping debug-only builder kwargs deepcopy-safe."""

    def __init__(self, page):
        self._page = page

    def __getattr__(self, name):
        return getattr(self._page, name)

    def __deepcopy__(self, memo):
        return self


def _effective_ocr_backend(settings: "RecognitionSettings | None") -> str | None:
    if settings is None:
        return None
    return settings.effective_ocr_engine()


def _effective_gdt_backend(settings: "RecognitionSettings | None") -> str | None:
    if settings is None:
        return None
    return settings.gdt_backend


def _vector_digit_slot_exclusion_regions(rects_result: dict[str, Any]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for region in rects_result.get('frame_borders', []) or []:
        if str(region.get('source') or '') == 'title_block':
            regions.append(_pad_region(region, VECTOR_DIGIT_SLOT_EXCLUSION_PAD_PT))
    for key in ('table_regions', 'table_cells', 'gdt_frames', 'datum_candidates'):
        for region in rects_result.get(key, []) or []:
            regions.append(_pad_region(region, VECTOR_DIGIT_SLOT_EXCLUSION_PAD_PT))
    return regions


def _pad_region(region: dict[str, Any], pad: float) -> dict[str, Any]:
    try:
        x = float(region.get('x', region.get('x0')))
        y = float(region.get('y', region.get('top', region.get('y0'))))
        if 'w' in region:
            w = float(region.get('w'))
        else:
            w = float(region.get('x1')) - x
        if 'h' in region:
            h = float(region.get('h'))
        else:
            h = float(region.get('bottom', region.get('y1'))) - y
    except (TypeError, ValueError):
        return dict(region)
    amount = max(0.0, float(pad))
    return {
        'x': x - amount,
        'y': y - amount,
        'w': max(0.0, w + amount * 2.0),
        'h': max(0.0, h + amount * 2.0),
        'source': str(region.get('source') or ''),
    }


def _accept_gdt_vector_hit(frame: dict, hit: dict | None, min_confidence: float) -> bool:
    """Return True when a vector-inversion symbol is safe to publish.

    Rect-origin candidates come from generic rectangle reconstruction and are
    treated as needs-review by default. Only permitted position FCFs pass this
    policy; other hits are downgraded to unknown so unrelated contours cannot
    become confident GD&T symbols.
    """
    if not hit or hit.get('confidence', 0.0) < min_confidence:
        return False
    if frame.get('_origin') == 'rect' and hit.get('symbol_class') != 0:
        return False
    return True


def _bbox_overlap_ratio(a: dict, b: dict) -> float:
    """Return intersection area divided by area of bbox a."""
    try:
        ax0 = float(a.get('x', 0.0))
        ay0 = float(a.get('y', 0.0))
        ax1 = ax0 + float(a.get('w', 0.0))
        ay1 = ay0 + float(a.get('h', 0.0))
        bx0 = float(b.get('x', 0.0))
        by0 = float(b.get('y', 0.0))
        bx1 = bx0 + float(b.get('w', 0.0))
        by1 = by0 + float(b.get('h', 0.0))
    except (TypeError, ValueError):
        return 0.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    return ((ix1 - ix0) * (iy1 - iy0)) / area


def _bbox_center_inside(inner: dict, outer: dict) -> bool:
    try:
        cx = float(inner.get('x', 0.0)) + float(inner.get('w', 0.0)) / 2.0
        cy = float(inner.get('y', 0.0)) + float(inner.get('h', 0.0)) / 2.0
        ox = float(outer.get('x', 0.0))
        oy = float(outer.get('y', 0.0))
        ow = float(outer.get('w', 0.0))
        oh = float(outer.get('h', 0.0))
    except (TypeError, ValueError):
        return False
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def _filter_ocr_table_regions(
    table_regions: list,
    view_boxes: list,
    table_cells: list | None = None,
) -> tuple[list, dict]:
    """Avoid suppressing OCR inside detected drawing views misclassified as tables."""
    threshold = float(getattr(Config, 'OCR_TABLE_SKIP_VIEW_OVERLAP_RATIO', 0.50))
    min_cell_count = int(getattr(Config, 'OCR_TABLE_SKIP_MIN_CELL_COUNT', 4))
    kept = []
    dropped = 0
    protected_by_cells = 0
    max_overlap_seen = 0.0
    for region in table_regions or []:
        max_overlap = max(
            (_bbox_overlap_ratio(region, vb) for vb in (view_boxes or [])),
            default=0.0,
        )
        max_overlap_seen = max(max_overlap_seen, max_overlap)
        if max_overlap >= threshold:
            cell_count = sum(
                1 for cell in (table_cells or [])
                if _bbox_center_inside(cell, region)
            )
            if cell_count >= min_cell_count:
                protected_by_cells += 1
                kept.append(region)
                continue
            dropped += 1
            continue
        kept.append(region)
    return kept, {
        'raw': len(table_regions or []),
        'kept': len(kept),
        'dropped_view_overlap': dropped,
        'protected_by_cells': protected_by_cells,
        'view_overlap_threshold': threshold,
        'min_cell_count': min_cell_count,
        'max_view_overlap': round(max_overlap_seen, 3),
    }


def _tiny_curve_count(page) -> int:
    count = 0
    for curve in getattr(page, 'curves', []) or []:
        try:
            width = abs(float(curve.get('x1', 0.0)) - float(curve.get('x0', 0.0)))
            height = abs(
                float(curve.get('bottom', 0.0)) - float(curve.get('top', 0.0))
            )
        except (TypeError, ValueError):
            continue
        if width < 15.0 and height < 15.0:
            count += 1
    return count


def _select_text_cluster_extra_points(
    conn_segments: list[dict[str, Any]] | None,
    *,
    tiny_curve_count: int,
    max_absolute: int | None = None,
    max_tiny_ratio: float | None = None,
) -> tuple[list[tuple[float, float]] | None, dict[str, Any]]:
    """Select connectivity points for text DBSCAN without letting dense geometry dominate."""
    segment_count = len(conn_segments or [])
    absolute_limit = (
        Config.TEXT_CLUSTER_EXTRA_POINTS_MAX_ABSOLUTE
        if max_absolute is None
        else int(max_absolute)
    )
    ratio_limit = (
        Config.TEXT_CLUSTER_EXTRA_POINTS_MAX_TINY_RATIO
        if max_tiny_ratio is None
        else float(max_tiny_ratio)
    )

    skip_reasons: list[str] = []
    if absolute_limit > 0 and segment_count > absolute_limit:
        skip_reasons.append('absolute_limit')
    if tiny_curve_count > 0 and ratio_limit > 0:
        ratio_max = int(tiny_curve_count * ratio_limit)
        if segment_count > ratio_max:
            skip_reasons.append('tiny_curve_ratio')
    else:
        ratio_max = None

    diag = {
        'retained_segments': segment_count,
        'tiny_curve_count': int(tiny_curve_count),
        'max_absolute': int(absolute_limit),
        'max_tiny_ratio': float(ratio_limit),
        'ratio_max': ratio_max,
        'skipped': bool(skip_reasons),
        'skip_reasons': skip_reasons,
        'used_points': 0,
    }
    if not segment_count or skip_reasons:
        return None, diag

    points: list[tuple[float, float]] = []
    for segment in conn_segments or []:
        try:
            x0 = float(segment.get('x0', 0.0))
            y0 = float(segment.get('y0', 0.0))
            x1 = float(segment.get('x1', x0))
            y1 = float(segment.get('y1', y0))
        except (TypeError, ValueError):
            continue
        points.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    diag['used_points'] = len(points)
    return points or None, diag


def _line_length(ln: dict) -> float:
    try:
        return math.hypot(
            float(ln.get('x1', 0.0)) - float(ln.get('x0', 0.0)),
            float(ln.get('y1', 0.0)) - float(ln.get('y0', 0.0)),
        )
    except (TypeError, ValueError):
        return 0.0


def _candidate_vector_context_segments(
    lines: list[dict[str, Any]],
    *,
    min_length: float = 5.0,
    min_line_width: float = 0.01,
) -> list[dict[str, Any]]:
    """Return existing vector lines suitable for candidate context density.

    Keep this as a cheap projection of `extract_lines_by_width` output; it must
    not reopen or reparse the PDF drawing stream.
    """
    segments: list[dict[str, Any]] = []
    for ln in lines or []:
        try:
            x0 = float(ln.get('x0', 0.0))
            y0 = float(ln.get('y0', 0.0))
            x1 = float(ln.get('x1', 0.0))
            y1 = float(ln.get('y1', 0.0))
            line_width = float(ln.get('lineWidth', 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if line_width < min_line_width:
            continue
        length = math.hypot(x1 - x0, y1 - y0)
        if length < min_length:
            continue
        segments.append({
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'length': length,
            'lineWidth': line_width,
        })
    return segments


def _select_secondary_arrow_lines(
    all_lines: list,
    annotation_lw: float | None,
    thin_thick: float,
) -> tuple[list, dict]:
    """Pick the adjacent thin line-width band for single-arrow leader recovery."""
    if not all_lines or annotation_lw is None or annotation_lw <= 0:
        return [], {
            'enabled': False,
            'reason': 'missing_annotation_width',
            'selected_widths': [],
            'line_count': 0,
        }

    ann_tol = max(annotation_lw * 0.15, 0.01)
    upper = max(annotation_lw * 1.35, thin_thick * 1.25)
    width_totals = defaultdict(float)
    ann_total = 0.0
    for ln in all_lines:
        try:
            width = float(ln.get('lineWidth') or 0.0)
        except (TypeError, ValueError):
            continue
        if width <= 0:
            continue
        length = _line_length(ln)
        if length <= 0:
            continue
        if abs(width - annotation_lw) <= ann_tol:
            ann_total += length
            continue
        if width <= annotation_lw * 1.02 or width > upper:
            continue
        width_totals[round(width, 4)] += length

    if not width_totals:
        return [], {
            'enabled': False,
            'reason': 'no_adjacent_width_band',
            'selected_widths': [],
            'line_count': 0,
            'annotation_total_length': round(ann_total, 2),
        }

    selected_width, selected_total = max(width_totals.items(), key=lambda kv: kv[1])
    min_total = max(1000.0, ann_total * 0.15)
    if selected_total < min_total:
        return [], {
            'enabled': False,
            'reason': 'adjacent_width_band_too_small',
            'selected_widths': [],
            'line_count': 0,
            'annotation_total_length': round(ann_total, 2),
            'best_width': selected_width,
            'best_total_length': round(selected_total, 2),
            'min_total_length': round(min_total, 2),
        }

    selected_tol = max(selected_width * 0.03, 0.01)
    selected_lines = [
        ln for ln in all_lines
        if abs(float(ln.get('lineWidth') or 0.0) - selected_width) <= selected_tol
    ]
    return selected_lines, {
        'enabled': True,
        'selected_widths': [selected_width],
        'line_count': len(selected_lines),
        'annotation_total_length': round(ann_total, 2),
        'selected_total_length': round(selected_total, 2),
        'width_upper_bound': round(upper, 4),
    }


def _vector_semantic_debug_fields(semantic_graph: dict | None) -> dict:
    """Return debug-only vector semantic fields for pipeline dumps."""
    if not semantic_graph:
        return {}
    fields = {
        'vector_semantics': semantic_graph.get('summary', {}),
        'vector_semantic_region_prior_conflicts': (
            semantic_graph.get('region_prior_conflicts', [])
        ),
    }
    if Config.VECTOR_SEMANTICS_EXPORT_OBJECTS:
        fields['vector_semantic_objects'] = semantic_graph.get('objects', [])
    if Config.VECTOR_SEMANTICS_EXPORT_PRIMITIVES:
        fields['vector_semantic_primitives'] = semantic_graph.get('primitives', [])
        fields['vector_semantic_edges'] = semantic_graph.get('edges', [])
        fields['vector_semantic_components'] = semantic_graph.get('components', [])
        fields['vector_semantic_region_priors'] = semantic_graph.get('region_priors', [])
    return fields


def _extract_page_vector_geometry(
    *,
    _drawing_snapshot_kwargs,
    _vector_only,
    page,
    page_index,
    pdf_bytes,
    strict_shared_fitz_page,
    timings,
    verbose,
):
    # ── Step 1: 矢量几何提取 ────────────────────────────────────────
    t_vec = time.time()

    # vec_1: 触发 pdfplumber 惰性解析缓存
    t0 = time.time()
    _ = page.lines
    _ = page.curves
    _ = page.rects
    timings['vec_1_page_parse'] = round(time.time() - t0, 3)

    # vec_2: 线段分层
    t0 = time.time()
    lines_result = extract_lines_by_width(
        page,
        **_drawing_snapshot_kwargs('lines_by_width'),
    )
    l1_lines = lines_result.get('l1_lines', [])
    all_lines_raw = lines_result.get('all_lines', [])
    thresholds = lines_result.get('thresholds', {})
    thin_thick = thresholds.get('thin_thick', 0.5)
    annotation_lw = detect_annotation_linewidth(all_lines_raw, thin_thick)
    timings['vec_2_lines_by_width'] = round(time.time() - t0, 3)
    if verbose and annotation_lw is not None:
        print(f"[pipeline] annotation_lw = {annotation_lw}")

    # vec_2b: 连通分量预过滤 annotation 线段
    extra_points = None
    conn_diag = None
    text_cluster_extra_points_diag = None
    if annotation_lw is not None:
        t0 = time.time()
        fitz_doc_conn = None
        try:
            if strict_shared_fitz_page is not None:
                fitz_page_conn = strict_shared_fitz_page
            else:
                fitz_doc_conn = fitz.open(stream=pdf_bytes, filetype="pdf")
                fitz_page_conn = fitz_doc_conn[page_index]
            conn_segments, conn_diag = filter_annotation_segments_by_connectivity(
                fitz_page_conn,
                annotation_lw,
                **_drawing_snapshot_kwargs('connectivity_filter'),
            )
        finally:
            if fitz_doc_conn is not None:
                fitz_doc_conn.close()
        if conn_segments:
            tiny_curve_count = _tiny_curve_count(page)
            extra_points, text_cluster_extra_points_diag = _select_text_cluster_extra_points(
                conn_segments,
                tiny_curve_count=tiny_curve_count,
            )
            if verbose and text_cluster_extra_points_diag.get('skipped'):
                print(
                    "[pipeline WARN] skipped connectivity text-cluster points: "
                    f"retained={text_cluster_extra_points_diag['retained_segments']} "
                    f"tiny_curves={tiny_curve_count} "
                    f"reasons={text_cluster_extra_points_diag['skip_reasons']}"
                )
        else:
            text_cluster_extra_points_diag = {
                'retained_segments': 0,
                'tiny_curve_count': _tiny_curve_count(page),
                'max_absolute': int(Config.TEXT_CLUSTER_EXTRA_POINTS_MAX_ABSOLUTE),
                'max_tiny_ratio': float(Config.TEXT_CLUSTER_EXTRA_POINTS_MAX_TINY_RATIO),
                'ratio_max': None,
                'skipped': False,
                'skip_reasons': [],
                'used_points': 0,
            }
        if verbose:
            print(f"[pipeline connectivity] total={conn_diag['total_segs']} "
                  f"components={conn_diag['components']} "
                  f"threshold={conn_diag['threshold']} "
                  f"retained={conn_diag['retained']} "
                  f"discarded={conn_diag['discarded']}")
        timings['text_cluster_extra_points'] = text_cluster_extra_points_diag
        timings['connectivity_filter'] = round(time.time() - t0, 3)

    # vec_3: 文字区域聚类
    t0 = time.time()
    legacy_plus_minus_full_page_shadow = []
    text_regions = cluster_text_regions(
        page,
        annotation_width=annotation_lw,
        extra_points=extra_points,
        enable_plus_minus_symbol_seeds=not _vector_only,
        plus_minus_shadow_sink=(
            legacy_plus_minus_full_page_shadow if _vector_only else None
        ),
    )
    timings['vec_3_cluster_text'] = round(time.time() - t0, 3)
    if _vector_only:
        legacy_seed_eligible_count = sum(
            1
            for row in legacy_plus_minus_full_page_shadow
            if row.get('legacy_seed_eligible') is True
        )
        timings['legacy_plus_minus_full_page_shadow_v1'] = {
            'schema_version': 'legacy_plus_minus_full_page_shadow_v1',
            'detector': 'pdf_analyzer_cluster._looks_like_plus_minus',
            'scan_executed': True,
            'candidate_count': legacy_seed_eligible_count,
            'raw_candidate_count': len(
                legacy_plus_minus_full_page_shadow
            ),
            'legacy_seed_eligible_count': legacy_seed_eligible_count,
            'seed_count': 0,
            'region_expansion_count': 0,
            'consumer_allowed': False,
        }

    # vec_4: 矩形提取
    t0 = time.time()
    rects_result = extract_rects(page)
    gdt_frame_candidates = rects_result.get('gdt_frames', [])
    datum_candidates = rects_result.get('datum_candidates', [])
    frame_borders = rects_result.get('frame_borders', [])
    table_cells = rects_result.get('table_cells', [])
    timings['vec_4_extract_rects'] = round(time.time() - t0, 3)

    # vec_5: 跑道框检测
    t0 = time.time()
    capsule_candidates = detect_capsules(
        page,
        **_drawing_snapshot_kwargs('capsule_detect'),
    )
    timings['vec_5_detect_capsules'] = round(time.time() - t0, 3)

    timings['vector_extract'] = round(time.time() - t_vec, 3)
    return (
        lines_result,
        l1_lines,
        all_lines_raw,
        thresholds,
        thin_thick,
        annotation_lw,
        conn_diag,
        text_cluster_extra_points_diag,
        text_regions,
        rects_result,
        gdt_frame_candidates,
        datum_candidates,
        frame_borders,
        table_cells,
        capsule_candidates,
    )


def _resolve_view_boxes_and_table_regions(
    *,
    _vector_only,
    page,
    page_index,
    pdf_bytes,
    rects_result,
    settings,
    strict_yolo_source,
    strict_yolo_trace_id,
    timings,
    verbose,
):
    # ── Step 1a: 视图切分 / strict request join ───────────────────
    t0 = time.time()
    strict_yolo_view_status_v1 = None
    if _vector_only:
        (
            view_boxes,
            view_seg_status_v1,
            strict_yolo_view_status_v1,
        ) = resolve_strict_yolo_view_source(
            strict_yolo_source,
            trace_id=str(strict_yolo_trace_id or ""),
            page_index=page_index,
        )
        timings['view_segment_status'] = dict(view_seg_status_v1)
    else:
        try:
            # Legacy modes retain their configured router and fallback behavior.
            view_boxes = segment_views(
                page,
                pdf_bytes=pdf_bytes,
                page_index=page_index,
                backend=(settings.view_seg if settings is not None else None),
            )
        except Exception as e:
            if verbose:
                print(f"[pipeline view_seg] failed: {e}")
            view_boxes = []
            timings['view_segment_status'] = {
                'requested_backend': 'unknown',
                'effective_backend': 'failed',
                'source': 'pipeline',
                'fallback': False,
                'fallback_reason': str(e),
                'view_count': 0,
            }
    timings['view_segment'] = round(time.time() - t0, 3)
    if not _vector_only and 'view_segment_status' not in timings:
        timings['view_segment_status'] = get_last_view_seg_status()
    if not _vector_only:
        view_seg_status_v1 = (
            build_view_seg_status_v1(
                timings.get('view_segment_status'),
                page_index=page_index,
                view_boxes=view_boxes,
            )
            if Config.VIEW_SEG_STATUS_DUMP
            else None
        )

    ocr_table_regions, table_skip_filter = _filter_ocr_table_regions(
        rects_result.get('table_regions', []),
        view_boxes,
        rects_result.get('table_cells', []),
    )
    timings['ocr_table_region_filter'] = table_skip_filter
    return (
        view_boxes,
        view_seg_status_v1,
        strict_yolo_view_status_v1,
        ocr_table_regions,
    )


def _extract_diameter_glyphs(
    *,
    _drawing_snapshot_kwargs,
    _vector_only,
    page,
    page_index,
    strict_yolo_trace_id,
    timings,
    verbose,
):
    # ── Step 1a2: ⌀ 矢量符号定位 ──────────────────────────────────
    t0 = time.time()
    try:
        diameter_glyphs = extract_diameter_glyphs(
            page,
            **_drawing_snapshot_kwargs('diameter_glyph_extract'),
        )
    except Exception as e:
        if _vector_only:
            raise VectorOnlyStageUnavailable(
                reason="diameter_glyph_producer_failed",
                stage="diameter_glyphs",
                trace_id=str(strict_yolo_trace_id or ""),
                page_index=page_index,
            ) from e
        if verbose:
            print(f"[pipeline diam_glyph] failed: {e}")
        diameter_glyphs = []
    timings['diameter_glyphs'] = round(time.time() - t0, 3)
    return diameter_glyphs


# Source-safety span marker: vector_digit_slots_required = (
def _build_vector_glyph_chain(
    *,
    _drawing_snapshot_kwargs,
    _vector_only,
    diameter_glyphs,
    page,
    page_index,
    pdf_bytes,
    strict_shared_fitz_page,
    text_regions,
    timings,
    vector_digit_slots_required,
    vector_drawing_snapshot,
    vector_glyph_fields,
    vector_digit_match_fields,
    vector_numeric_phrase_fields,
    vector_glyph_tokens_for_anchor,
    vector_digit_match_rows_for_phrase,
    vector_numeric_phrase_rows_for_hypothesis,
    vector_numeric_phrase_stats_for_hypothesis,
):
    if (
        _vector_only
        or Config.VECTOR_GLYPH_LAYER_DUMP
        or Config.ANCHOR_CORRIDOR_CANDIDATES_V1_DUMP
        or Config.VECTOR_GLYPH_ANCHOR_AXIS_DUMP
        or Config.VECTOR_DIGIT_MATCH_DUMP
        or Config.VECTOR_NUMERIC_PHRASE_DUMP
        or Config.VECTOR_DIMENSION_HYPOTHESIS_DUMP
        or Config.VECTOR_GLYPH_CONSUME_FINAL
        or Config.R2N_NOMINAL_RESCUE_SHADOW_DUMP
    ):
        t0 = time.time()
        glyph_dump = build_vector_glyph_v1(
            page=page,
            page_index=page_index,
            text_regions=text_regions,
            diameter_glyphs=diameter_glyphs,
            include_digit_slots=vector_digit_slots_required,
        )
        degree_merge_kwargs = {}
        if vector_drawing_snapshot is not None:
            degree_merge_kwargs = {
                'page': strict_shared_fitz_page,
                **_drawing_snapshot_kwargs('degree_polyline_merge'),
            }
        glyph_dump = merge_degree_polyline_glyph_dump(
            glyph_dump,
            pdf_bytes=pdf_bytes,
            page_index=page_index,
            enabled=True,
            **degree_merge_kwargs,
        )
        degree_merge_kwargs.clear()
        vector_glyph_tokens_for_anchor = glyph_dump.get('tokens', [])
        if Config.VECTOR_GLYPH_LAYER_DUMP:
            vector_glyph_fields = {
                'vector_glyph_v1': vector_glyph_tokens_for_anchor,
                'vector_glyph_stats_v1': glyph_dump.get('stats', {}),
            }
            timings['vector_glyph_v1'] = round(time.time() - t0, 3)
        elif (
            _vector_only
            or Config.ANCHOR_CORRIDOR_CANDIDATES_V1_DUMP
            or Config.VECTOR_GLYPH_ANCHOR_AXIS_DUMP
        ):
            timings[
                'vector_glyph_for_anchor_corridor_v1'
                if (
                    _vector_only
                    or Config.ANCHOR_CORRIDOR_CANDIDATES_V1_DUMP
                )
                else 'vector_glyph_for_anchor_axis_v1'
            ] = round(
                time.time() - t0,
                3,
            )
        else:
            timings['vector_glyph_for_digit_match_v1'] = round(
                time.time() - t0,
                3,
            )
        if (
            Config.VECTOR_DIGIT_MATCH_DUMP
            or Config.VECTOR_NUMERIC_PHRASE_DUMP
            or Config.VECTOR_DIMENSION_HYPOTHESIS_DUMP
            or Config.VECTOR_GLYPH_CONSUME_FINAL
            or Config.R2N_NOMINAL_RESCUE_SHADOW_DUMP
        ):
            t_digit_match = time.time()
            prototype_library = (
                load_vector_digit_prototype_library(
                    Config.VECTOR_DIGIT_PROTOTYPE_LIBRARY,
                )
                if Config.VECTOR_DIGIT_PROTOTYPE_LIBRARY else None
            )
            digit_match_dump = build_vector_digit_matches_v1(
                vector_glyph_tokens=vector_glyph_tokens_for_anchor,
                page=page,
                page_index=page_index,
                prototype_library=prototype_library,
                prototype_threshold=Config.VECTOR_DIGIT_PROTOTYPE_THRESHOLD,
                prototype_margin=Config.VECTOR_DIGIT_PROTOTYPE_MARGIN,
            )
            vector_digit_match_rows_for_phrase = digit_match_dump.get(
                'vector_digit_matches_v1',
                [],
            )
            if Config.VECTOR_DIGIT_MATCH_DUMP:
                vector_digit_match_fields = {
                    'vector_digit_matches_v1': vector_digit_match_rows_for_phrase,
                    'vector_digit_match_stats_v1': digit_match_dump.get(
                        'vector_digit_match_stats_v1',
                        {},
                    ),
                }
            timings[
                'vector_digit_matches_v1'
                if Config.VECTOR_DIGIT_MATCH_DUMP
                else 'vector_digit_match_for_numeric_phrase_v1'
            ] = round(
                time.time() - t_digit_match,
                3,
            )
        if (
            Config.VECTOR_NUMERIC_PHRASE_DUMP
            or Config.VECTOR_DIMENSION_HYPOTHESIS_DUMP
            or Config.VECTOR_GLYPH_CONSUME_FINAL
            or Config.R2N_NOMINAL_RESCUE_SHADOW_DUMP
        ):
            t_numeric_phrase = time.time()
            numeric_phrase_dump = build_vector_numeric_phrases_v1(
                vector_digit_matches=vector_digit_match_rows_for_phrase,
                vector_glyph_tokens=vector_glyph_tokens_for_anchor,
                page_index=page_index,
            )
            vector_numeric_phrase_rows_for_hypothesis = numeric_phrase_dump.get(
                'vector_numeric_phrases_v1',
                [],
            )
            vector_numeric_phrase_stats_for_hypothesis = numeric_phrase_dump.get(
                'vector_numeric_phrase_stats_v1',
                {},
            )
            if (
                Config.VECTOR_NUMERIC_PHRASE_DUMP
                or Config.VECTOR_DIMENSION_HYPOTHESIS_DUMP
                or Config.VECTOR_GLYPH_CONSUME_FINAL
            ):
                vector_numeric_phrase_fields = {
                    'vector_numeric_phrases_v1': vector_numeric_phrase_rows_for_hypothesis,
                    'vector_numeric_phrase_stats_v1': vector_numeric_phrase_stats_for_hypothesis,
                }
            timings['vector_numeric_phrases_v1'] = round(
                time.time() - t_numeric_phrase,
                3,
            )
    return (
        vector_glyph_fields,
        vector_digit_match_fields,
        vector_numeric_phrase_fields,
        vector_glyph_tokens_for_anchor,
        vector_digit_match_rows_for_phrase,
        vector_numeric_phrase_rows_for_hypothesis,
        vector_numeric_phrase_stats_for_hypothesis,
    )


def _build_decimal_anchor_fields(
    *,
    _drawing_snapshot_kwargs,
    _vector_only,
    fitz_page,
    page_index,
    r2c_dump_fields,
    text_regions,
    vector_page_context,
):
    decimal_corridor_dump = Config.VECTOR_DECIMAL_CORRIDOR_DUMP
    decimal_char_height_dump = decimal_char_height_dump_enabled() or decimal_corridor_dump
    decimal_anchor_authority_required = bool(
        _vector_only or decimal_char_height_dump or decimal_corridor_dump
    )
    text_regions, decimal_quad_fields = apply_decimal_point_quad_pipeline(
        page=fitz_page,
        page_index=page_index,
        text_regions=text_regions,
        pdf_stem=f"p{page_index + 1:03d}",
        dump_enabled=(
            _vector_only
            or Config.VECTOR_DECIMAL_POINT_QUAD_DUMP
            or decimal_char_height_dump
            or Config.ANCHOR_CORRIDOR_PRECISION_FILTERS_ENABLED
            or Config.VECTOR_GLYPH_CONSUME_FINAL
            or Config.VECTOR_TIMES_DETECTOR_DUMP
            or Config.R7_ROTATED_RUNWAY_SHADOW_DUMP
        ),
        replace_legacy=Config.DECIMAL_POINT_QUAD_REPLACE_LEGACY,
        item_primitives=(
            vector_page_context.primitives
            if vector_page_context is not None
            else None
        ),
    )
    r2d_decimal_quad_rows = decimal_quad_fields.get('vector_decimal_point_quad_v1', [])
    r2c_dump_fields.update(decimal_quad_fields if Config.VECTOR_DECIMAL_POINT_QUAD_DUMP else {})
    decimal_char_height_analysis = (
        build_decimal_char_height_v1(
            r2d_decimal_quad_rows,
            page_index=page_index,
        )
        if decimal_anchor_authority_required
        else {
            'vector_decimal_char_height_v1': [],
            'vector_decimal_char_height_stats_v1': {},
        }
    )
    decimal_char_height_rows = decimal_char_height_analysis.get(
        'vector_decimal_char_height_v1',
        [],
    )
    decimal_corridor_analysis = (
        build_decimal_corridors_v1(
            r2d_decimal_quad_rows,
            decimal_char_height_rows,
            page_index=page_index,
        )
        if decimal_anchor_authority_required
        else {
            'vector_decimal_corridor_v1': [],
            'vector_decimal_repeated_size_cohort_v1': [],
            'vector_decimal_corridor_stats_v1': {},
        }
    )
    r2d_decimal_corridor_rows = decimal_corridor_analysis.get(
        'vector_decimal_corridor_v1',
        [],
    )
    r2d_decimal_repeated_size_cohort_rows = decimal_corridor_analysis.get(
        'vector_decimal_repeated_size_cohort_v1',
        [],
    )
    if decimal_char_height_dump:
        r2c_dump_fields.update({
            'vector_decimal_char_height_v1': decimal_char_height_rows,
            'vector_decimal_char_height_stats_v1': decimal_char_height_analysis.get(
                'vector_decimal_char_height_stats_v1',
                {},
            ),
        })
    if decimal_corridor_dump:
        r2c_dump_fields.update({
            'vector_decimal_corridor_v1': r2d_decimal_corridor_rows,
            'vector_decimal_corridor_stats_v1': decimal_corridor_analysis.get(
                'vector_decimal_corridor_stats_v1',
                {},
            ),
        })
    degree_dump_enabled = bool(
        _vector_only or Config.VECTOR_DEGREE_POLYLINE_DUMP
    )
    degree_fields = build_degree_polyline_pipeline_fields(
        page=fitz_page,
        page_index=page_index,
        dump_enabled=degree_dump_enabled,
        **(
            _drawing_snapshot_kwargs('degree_polyline_dump')
            if degree_dump_enabled
            else {}
        ),
    )
    r92_degree_rows = degree_fields.get(
        'vector_degree_polyline_v1',
        [],
    )
    if Config.VECTOR_DEGREE_POLYLINE_DUMP:
        r2c_dump_fields.update(degree_fields)
    return (
        text_regions,
        r2d_decimal_quad_rows,
        r92_degree_rows,
        decimal_char_height_rows,
        r2d_decimal_corridor_rows,
        r2d_decimal_repeated_size_cohort_rows,
    )


def _build_vector_times_fields(
    *,
    fitz_page,
    page_index,
    r2d_decimal_quad_rows,
    timings,
    vector_page_context,
    vector_times_detector_fields,
):
    if Config.VECTOR_TIMES_DETECTOR_DUMP:
        t_times = time.time()
        times_dump = detect_vector_times_candidates(
            fitz_page,
            page_index=page_index,
            pdf_stem=f"p{page_index + 1:03d}",
            decimal_point_rows=r2d_decimal_quad_rows,
            **(
                {'item_primitives': vector_page_context.primitives}
                if vector_page_context is not None
                else {}
            ),
        )
        vector_times_detector_fields = {
            'vector_times_raw_cross_candidates_v1': times_dump.get(
                'vector_times_raw_cross_candidates_v1', []
            ),
            'vector_times_candidates_v1': times_dump.get(
                'vector_times_candidates_v1', []
            ),
            'vector_times_detector_stats_v1': times_dump.get(
                'vector_times_detector_stats_v1', {}
            ),
        }
        timings['vector_times_detector_v1'] = round(time.time() - t_times, 3)
    return vector_times_detector_fields


def _build_context_segments_and_gdt_line_frames(
    *,
    _drawing_snapshot_kwargs,
    all_lines_raw,
    fitz_page,
    frame_borders,
    thresholds,
    timings,
):
    context_t0 = time.time()
    if Config.CANDIDATE_VECTOR_CONTEXT_DENSITY_ENABLED:
        zero_cutoff = float((thresholds or {}).get('zero_cutoff') or 0.01)
        candidate_vector_context_segments = _candidate_vector_context_segments(
            all_lines_raw,
            min_length=Config.CANDIDATE_VECTOR_CONTEXT_DENSITY_MIN_LENGTH,
            min_line_width=zero_cutoff,
        )
    else:
        candidate_vector_context_segments = []
    timings['candidate_vector_context_segments'] = len(candidate_vector_context_segments)
    timings['candidate_vector_context_source'] = 'extract_lines_by_width.all_lines'
    timings['candidate_vector_context'] = round(time.time() - context_t0, 3)
    t0 = time.time()
    pdf_text_words = (
        extract_pdf_text_words(fitz_page)
        if Config.CANDIDATE_PDF_TEXT_WORDS_ENABLED
        else []
    )
    gdt_frames_from_lines = detect_gdt_frames_from_lines(
        fitz_page,
        frame_borders=frame_borders,
        **_drawing_snapshot_kwargs('gdt_frame_geometry'),
    )
    timings['gdt_line_detect'] = round(time.time() - t0, 3)
    return (
        candidate_vector_context_segments,
        pdf_text_words,
        gdt_frames_from_lines,
    )


def _detect_legacy_arrow_lines(
    *,
    _vector_only,
    all_lines_raw,
    annotation_lw,
    text_regions,
    thin_thick,
    timings,
    verbose,
):
    # 箭头检测。strict vector-only 的 phrase authority 不依赖箭头，且后续
    # OCR/assembler 在 strict 早退之后不会执行，因此固定旁路旧全页扫描。
    t0 = time.time()
    detected_dim_lines = []
    detected_leader_lines = []
    leader_filter_stats = {
        'single_arrow_bound': 0,
        'reject_not_diagonal': 0,
        'reject_length': 0,
        'reject_alignment': 0,
        'reject_no_near_text': 0,
        'accept': 0,
    }
    if annotation_lw is not None and not _vector_only:
        def _point_near_text_region(px: float, py: float, margin: float) -> bool:
            for region in text_regions:
                try:
                    rx = float(region.get('x', 0.0))
                    ry = float(region.get('y', 0.0))
                    rw = float(region.get('w', 0.0))
                    rh = float(region.get('h', 0.0))
                except (TypeError, ValueError):
                    continue
                if rx - margin <= px <= rx + rw + margin and ry - margin <= py <= ry + rh + margin:
                    return True
            return False

        def _text_near_leader_corridor(ax: float, ay: float, tx: float, ty: float,
                                       margin: float) -> bool:
            dx = tx - ax
            dy = ty - ay
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                return False
            lower = min(25.0, seg_len * 0.15)
            upper = seg_len + max(20.0, margin * 0.35)
            perp_limit = max(18.0, min(32.0, margin * 0.45))
            for region in text_regions:
                try:
                    cx = float(region.get('x', 0.0)) + float(region.get('w', 0.0)) / 2.0
                    cy = float(region.get('y', 0.0)) + float(region.get('h', 0.0)) / 2.0
                except (TypeError, ValueError):
                    continue
                vx = cx - ax
                vy = cy - ay
                proj = (vx * dx + vy * dy) / seg_len
                if proj < lower or proj > upper:
                    continue
                perp = abs(vx * dy - vy * dx) / seg_len
                if perp <= perp_limit:
                    return True
            return False

        _tol = annotation_lw * 0.15
        anno_lines = [ln for ln in all_lines_raw
                      if (annotation_lw - _tol) <= ln['lineWidth'] <= (annotation_lw + _tol)]
        timings['arrow_detector_eps'] = Config.ARROW_DETECTOR_EPS
        timings['arrow_detector_min_samples'] = Config.ARROW_DETECTOR_MIN_SAMPLES
        seen_leader_keys = set()

        def _collect_single_arrow_leaders(arrow_result: dict, source_name: str):
            source_stats = leader_filter_stats.setdefault('sources', {}).setdefault(
                source_name,
                {
                    'single_arrow_bound': 0,
                    'reject_not_diagonal': 0,
                    'reject_length': 0,
                    'reject_alignment': 0,
                    'reject_no_near_text': 0,
                    'reject_duplicate': 0,
                    'accept': 0,
                },
            )

            def _bump(key: str):
                leader_filter_stats[key] = leader_filter_stats.get(key, 0) + 1
                source_stats[key] = source_stats.get(key, 0) + 1

            arrows_by_line = {}
            for arrow in arrow_result.get('arrows', []):
                li = arrow.get('bound_line_idx', -1)
                if li >= 0:
                    arrows_by_line.setdefault(li, []).append(arrow)
            for li, bound_arrows in arrows_by_line.items():
                # Single-arrow leader lines are not true paired dimension lines,
                # but they are valuable OCR targets for radius/chamfer callouts.
                # Keep this OCR-only and diagonal-only to avoid flooding the
                # assembler with frame/grid lines.
                if len(bound_arrows) != 1:
                    continue
                _bump('single_arrow_bound')
                long_lines = arrow_result.get('long_lines', [])
                if li >= len(long_lines):
                    continue
                ll = long_lines[li]
                if ll.get('orientation') != 'D':
                    _bump('reject_not_diagonal')
                    continue
                start = (float(ll['x0']), float(ll['y0']))
                end = (float(ll['x1']), float(ll['y1']))
                length = float(ll.get('length') or 0.0)
                if length < 25.0 or length > 320.0:
                    _bump('reject_length')
                    continue
                line_angle = math.degrees(
                    math.atan2(end[1] - start[1], end[0] - start[0])
                ) % 180.0
                if line_angle <= 8.0 or line_angle >= 172.0 or 82.0 <= line_angle <= 98.0:
                    _bump('reject_not_diagonal')
                    continue
                arrow = bound_arrows[0]
                ux = (end[0] - start[0]) / length
                uy = (end[1] - start[1]) / length
                try:
                    adx, ady = arrow.get('direction') or (0.0, 0.0)
                    arrow_alignment = abs(float(adx) * ux + float(ady) * uy)
                except (TypeError, ValueError):
                    continue
                if arrow_alignment < 0.85:
                    _bump('reject_alignment')
                    continue

                bind_ep = arrow.get('_bind_endpoint')
                if bind_ep == 'p0':
                    arrow_end = start
                    tail = end
                elif bind_ep == 'p1':
                    arrow_end = end
                    tail = start
                else:
                    tip = arrow.get('tip') or start
                    d0 = math.hypot(float(tip[0]) - start[0], float(tip[1]) - start[1])
                    d1 = math.hypot(float(tip[0]) - end[0], float(tip[1]) - end[1])
                    if d0 <= d1:
                        arrow_end = start
                        tail = end
                    else:
                        arrow_end = end
                        tail = start

                tail_margin = max(35.0, min(80.0, length * 0.45))
                tail_point_margin = max(15.0, min(35.0, length * 0.18))
                if not (
                    _point_near_text_region(tail[0], tail[1], tail_point_margin)
                    and _text_near_leader_corridor(
                        arrow_end[0], arrow_end[1], tail[0], tail[1], tail_margin)
                ):
                    _bump('reject_no_near_text')
                    continue

                leader_key = tuple(round(v, 1) for v in (
                    start[0], start[1], end[0], end[1],
                ))
                reverse_key = tuple(round(v, 1) for v in (
                    end[0], end[1], start[0], start[1],
                ))
                if leader_key in seen_leader_keys or reverse_key in seen_leader_keys:
                    _bump('reject_duplicate')
                    continue
                seen_leader_keys.add(leader_key)

                _bump('accept')
                detected_leader_lines.append({
                    'start': start,
                    'end': end,
                    'midpoint': ((start[0] + end[0]) / 2.0,
                                 (start[1] + end[1]) / 2.0),
                    'length': length,
                    'orientation': 'D',
                    'line_idx': ll.get('idx'),
                    'line_kind': 'single_arrow_leader',
                    'arrow': arrow,
                    'arrow_end': arrow_end,
                    'tail': tail,
                    'tail_text_margin': round(tail_margin, 2),
                    'arrow_alignment': round(arrow_alignment, 3),
                    'arrow_source': source_name,
                })

        # Production drawings often have converted arrow tips a few points off line endpoints.
        arrow_result = detect_arrows(
            anno_lines,
            arrow_eps=Config.ARROW_DETECTOR_EPS,
            arrow_min_samples=Config.ARROW_DETECTOR_MIN_SAMPLES,
            bind_dist_threshold=max(5.0, 3.0 * 1.7),
            verbose=verbose,
        )
        detected_dim_lines = arrow_result['dimension_lines']
        timings['arrow_detector_primary_line_count'] = len(anno_lines)
        timings['arrow_detector_primary_arrow_count'] = len(arrow_result.get('arrows', []))
        timings['arrow_detector_primary_dimline_count'] = len(detected_dim_lines)
        _collect_single_arrow_leaders(arrow_result, 'annotation_width')

        secondary_arrow_lines, secondary_arrow_diag = _select_secondary_arrow_lines(
            all_lines_raw, annotation_lw, thin_thick)
        timings['secondary_arrow_line_selection'] = secondary_arrow_diag
        if secondary_arrow_lines:
            secondary_detection_lines = anno_lines + secondary_arrow_lines
            secondary_arrow_result = detect_arrows(
                secondary_detection_lines,
                arrow_eps=Config.ARROW_DETECTOR_EPS,
                arrow_min_samples=Config.ARROW_DETECTOR_MIN_SAMPLES,
                bind_dist_threshold=max(5.0, 3.0 * 1.7),
                verbose=verbose,
            )
            timings['arrow_detector_secondary_line_count'] = len(secondary_detection_lines)
            timings['arrow_detector_secondary_arrow_count'] = len(
                secondary_arrow_result.get('arrows', []))
            timings['arrow_detector_secondary_dimline_count'] = len(
                secondary_arrow_result.get('dimension_lines', []))
            _collect_single_arrow_leaders(secondary_arrow_result, 'adjacent_thin_width')
    if _vector_only:
        timings['arrow_detect_status'] = 'bypassed_strict_vector_only'
        timings['arrow_detector_call_count'] = 0
    timings['arrow_detect'] = round(time.time() - t0, 3)
    timings['leader_filter_stats'] = leader_filter_stats
    return detected_dim_lines, detected_leader_lines


def _reconstruct_rects_and_gdt_frames(
    *,
    fitz_page,
    _drawing_snapshot_kwargs,
    timings,
    gdt_frames_from_lines,
    frame_borders,
    page_width,
    page_height,
    _vector_only,
    vector_drawing_snapshot,
    strict_yolo_trace_id,
    page_index,
    _close_pipeline_fitz_doc,
):
    # 线段矩形重建
    t0 = time.time()
    h_lines_ac, v_lines_ac = get_annotation_and_contour_lines(
        fitz_page,
        **_drawing_snapshot_kwargs('rectangle_reconstruct'),
    )
    reconstructed_rects = reconstruct_rectangles_from_lines(h_lines_ac, v_lines_ac)
    timings['rect_reconstruct'] = round(time.time() - t0, 3)

    # 统一 GD&T 候选框（line-detect + reconstructed 派生）
    gdt_frames_all = build_augmented_gdt_frames(
        gdt_frames_from_lines, reconstructed_rects, frame_borders=frame_borders,
        page_width=page_width, page_height=page_height)
    strict_gdt_symbol_evidence = None
    if _vector_only:
        t_strict_gdt_symbol = time.time()
        try:
            strict_gdt_symbol_evidence = (
                build_strict_gdt_symbol_evidence(
                    frames=gdt_frames_all,
                    fitz_page=fitz_page,
                    drawing_snapshot=vector_drawing_snapshot,
                    trace_id=str(strict_yolo_trace_id or ""),
                    page_index=page_index,
                )
            )
        except Exception as exc:
            _close_pipeline_fitz_doc()
            raise VectorOnlyStageUnavailable(
                reason="strict_gdt_symbol_evidence_failed",
                stage="strict_gdt_symbol_evidence",
                trace_id=str(strict_yolo_trace_id or ""),
                page_index=page_index,
            ) from exc
        timings["strict_gdt_symbol_evidence"] = round(
            time.time() - t_strict_gdt_symbol,
            3,
        )
    return reconstructed_rects, gdt_frames_all, strict_gdt_symbol_evidence


def _build_semantic_graph_audit(
    *,
    page,
    fitz_page,
    page_index,
    text_regions,
    frame_borders,
    table_cells,
    rects_result,
    view_boxes,
    l1_lines,
    timings,
    verbose,
    vector_drawing_snapshot,
    vector_drawing_snapshot_metrics,
):
    # Phase 0 Vector Semantic Graph is audit-only. It runs before OCR so the
    # debug dump can explain vector structure near later OCR/assembly misses,
    # but none of its output is routed into OCR targets, candidate generation,
    # assembler evidence, table skip, or final dimensions.
    semantic_graph = None
    if Config.VECTOR_SEMANTICS_ENABLED:
        t0 = time.time()
        try:
            semantic_graph = build_vector_semantics(
                page=page,
                fitz_page=fitz_page,
                page_index=page_index,
                doc_id='pipeline',
                text_regions=text_regions,
                frame_borders=frame_borders,
                table_cells=table_cells,
                table_regions=rects_result.get('table_regions', []),
                view_boxes=view_boxes,
                export_primitives=Config.VECTOR_SEMANTICS_EXPORT_PRIMITIVES,
                prefetched_fitz_lines=l1_lines,
                skip_pdfplumber_lines=True,
            )
            timings['vector_semantics_ms'] = (
                semantic_graph.get('summary', {})
                .get('timings', {})
                .get('vector_semantics_ms', round((time.time() - t0) * 1000))
            )
        except Exception as e:
            semantic_graph = None
            timings['vector_semantics_ms'] = round((time.time() - t0) * 1000)
            timings['vector_semantics_error'] = str(e)
            if verbose:
                print(f"[pipeline vector_semantics] failed: {e}")

    # No downstream phrase/candidate stage consumes raw PyMuPDF drawing
    # dictionaries.  Drop the page snapshot before the expensive reader so it
    # does not remain resident alongside the expanded VectorPageContext.
    if vector_drawing_snapshot is not None:
        vector_drawing_snapshot = None
        assert vector_drawing_snapshot_metrics is not None
        vector_drawing_snapshot_metrics[
            'released_before_phrase_reader'
        ] = True
        vector_drawing_snapshot_metrics['release_stage'] = (
            'after_vector_geometry'
        )
    return semantic_graph, vector_drawing_snapshot


def _run_legacy_ocr_tail(
    *,
    _build_candidate_package,
    _build_r2n_nominal_rescue_shadow_fields,
    _build_vector_anchor_axis_fields,
    _build_vector_dimension_hypothesis_fields,
    _close_pipeline_fitz_doc,
    _do_render,
    _mark_deadline_skipped,
    _mark_no_targets_skipped,
    _ocr_budget_expired,
    _render_log,
    annotation_lw,
    candidate_package,
    capsule_candidates,
    conn_diag,
    datum_candidates,
    detected_dim_lines,
    detected_leader_lines,
    diameter_glyphs,
    directed_deadline,
    dpi,
    fitz_page,
    frame_borders,
    gdt_detector,
    gdt_frame_candidates,
    gdt_frames_all,
    gdt_frames_from_lines,
    hires_retry_cap,
    l1_lines,
    lines_result,
    ocr_deadline,
    ocr_pass_counters,
    ocr_table_regions,
    page,
    page_height,
    page_index,
    page_width,
    pdf_bytes,
    pdf_text_words,
    r21_layered_vector_dimension_fields,
    r2c_dump_fields,
    r2d_decimal_quad_rows,
    r7_gdt_vertical_shadow_fields,
    r7_key_shadow_fields,
    r7_rotated_runway_candidates,
    r7_rotated_runway_shadow_fields,
    reconstructed_rects,
    rects_result,
    semantic_graph,
    settings,
    t_start,
    table_cells,
    text_cluster_extra_points_diag,
    text_regions,
    timings,
    vector_digit_match_fields,
    vector_digit_match_rows_for_phrase,
    vector_final_consume_fields,
    vector_glyph_fields,
    vector_glyph_tokens_for_anchor,
    vector_numeric_phrase_fields,
    vector_times_detector_fields,
    verbose,
    view_boxes,
    view_seg_status_v1,
):
    ocr_results = []
    candidate_guided_ocr_results = []
    candidate_guided_skip_regions = []
    ocr_crop_oriented_quad_fields = {}
    page_img_ocr = None

    _eff_ocr_backend = _effective_ocr_backend(settings)
    with ocr_backend_override(_eff_ocr_backend):
        # Step 2a: Candidate-guided OCR（默认开启；candidate → crop → OCR）
        # This runs before broad directed OCR so high-value candidate crops do not
        # have to survive on whatever deadline remains after the page-wide pass.
        t0 = time.time()
        if not Config.CANDIDATE_GUIDED_OCR_ENABLED:
            timings['candidate_guided_ocr_skipped_disabled'] = True
            ocr_pass_counters['candidate_guided']['skipped_disabled'] = 1
        else:
            candidate_guided_candidates = (
                (candidate_package or {}).get('dimension_candidates') or []
            )
            candidate_guided_plans = build_candidate_crop_plans(
                candidate_guided_candidates,
                page_width=page_width,
                page_height=page_height,
                low_priority_budget_min=Config.OCR_LOW_PRIORITY_BUDGET_MIN,
                low_priority_budget_ratio=Config.OCR_LOW_PRIORITY_BUDGET_RATIO,
                max_plan_count=Config.OCR_CANDIDATE_GUIDED_MAX_PLANS,
                diag=ocr_pass_counters['candidate_guided'],
            )
            timings['candidate_guided_ocr_target_count'] = len(candidate_guided_candidates)
            timings['candidate_guided_ocr_plan_count'] = len(candidate_guided_plans)
            timings['candidate_guided_low_priority_budget'] = (
                ocr_pass_counters['candidate_guided'].get('low_priority_budget', 0)
            )
            timings['candidate_guided_low_priority_planned'] = (
                ocr_pass_counters['candidate_guided'].get('low_priority_planned', 0)
            )
            if Config.OCR_CROP_USE_ORIENTED_QUAD:
                t_crop_quad = time.time()
                crop_quad_dump = build_ocr_crop_oriented_quad_v1(
                    crop_plans=candidate_guided_plans,
                    candidates=candidate_guided_candidates,
                    page_width=page_width,
                    page_height=page_height,
                    ocr_pass='candidate_guided',
                )
                ocr_crop_oriented_quad_fields = {
                    'ocr_crop_oriented_quad_v1': crop_quad_dump.get(
                        'ocr_crop_oriented_quad_v1', []
                    ),
                    'ocr_crop_oriented_quad_stats_v1': crop_quad_dump.get(
                        'ocr_crop_oriented_quad_stats_v1', {}
                    ),
                }
                timings['ocr_crop_oriented_quad_v1'] = round(time.time() - t_crop_quad, 3)
            if not candidate_guided_plans:
                _mark_no_targets_skipped(
                    'candidate_guided_ocr',
                    ocr_pass_counters['candidate_guided'],
                )
            elif not _ocr_budget_expired(
                'candidate_guided_ocr',
                ocr_pass_counters['candidate_guided'],
            ):
                candidate_guided_deadline = None
                candidate_guided_min_remaining = max(
                    0,
                    int(Config.OCR_CANDIDATE_GUIDED_MIN_REMAINING_SECONDS or 0),
                )
                if ocr_deadline is not None and candidate_guided_min_remaining > 0:
                    remaining = ocr_deadline - time.monotonic()
                    timings['candidate_guided_ocr_remaining_before_sec'] = round(remaining, 3)
                    if remaining <= candidate_guided_min_remaining:
                        timings['candidate_guided_ocr_skipped_reserve'] = True
                        ocr_pass_counters['candidate_guided']['skipped_reserve'] = 1
                    else:
                        candidate_guided_deadline = ocr_deadline - candidate_guided_min_remaining
                if not timings.get('candidate_guided_ocr_skipped_reserve'):
                    max_seconds = max(0, int(Config.OCR_CANDIDATE_GUIDED_MAX_SECONDS or 0))
                    if max_seconds > 0:
                        local_deadline = time.monotonic() + max_seconds
                        candidate_guided_deadline = (
                            min(candidate_guided_deadline, local_deadline)
                            if candidate_guided_deadline is not None else local_deadline
                        )
                    page_img_ocr = _do_render(dpi)
                    candidate_guided_ocr_results = run_candidate_guided_ocr(
                        page_img_ocr,
                        candidate_guided_plans,
                        candidate_guided_candidates,
                        dpi=dpi,
                        verbose=verbose,
                        deadline=candidate_guided_deadline,
                        deadline_counter_key='stopped_local_budget',
                        diag=ocr_pass_counters['candidate_guided'],
                    )
                    if candidate_guided_ocr_results:
                        ocr_results.extend(candidate_guided_ocr_results)
                        candidate_guided_skip_regions = [
                            result.get('candidate_crop_bbox')
                            for result in candidate_guided_ocr_results
                            if result.get('candidate_crop_bbox')
                        ]
        timings['candidate_guided_ocr'] = round(time.time() - t0, 3)

        # Step 2b: 尺寸线条带 OCR
        # Run vector/arrow-targeted dimension-line strips before broad directed OCR.
        # Dense drawings can spend the entire OCR budget in directed OCR; if that
        # happens, vertical/diagonal dimension-line targets never get a chance.
        t0 = time.time()
        dimline_ocr_results = []
        dimline_ocr_targets = detected_dim_lines + detected_leader_lines
        timings['dimline_strip_ocr_target_count'] = len(dimline_ocr_targets)
        timings['dimline_strip_ocr_paired_target_count'] = len(detected_dim_lines)
        timings['dimline_strip_ocr_leader_target_count'] = len(detected_leader_lines)
        if not Config.OCR_PASS_DIMLINE_ENABLED:
            timings['dimline_strip_ocr_skipped_disabled'] = True
            ocr_pass_counters['dimline']['skipped_disabled'] = 1
        elif not dimline_ocr_targets:
            _mark_no_targets_skipped('dimline_strip_ocr', ocr_pass_counters['dimline'])
        elif not _ocr_budget_expired('dimline_strip_ocr', ocr_pass_counters['dimline']):
            if page_img_ocr is None:
                page_img_ocr = _do_render(dpi)
            dimline_ocr_results = run_dimline_strip_ocr(
                page_img_ocr, dimline_ocr_targets, ocr_results, dpi=dpi,
                verbose=verbose,
                deadline=ocr_deadline,
                diag=ocr_pass_counters['dimline'])
        if dimline_ocr_results:
            ocr_results.extend(dimline_ocr_results)
        timings['dimline_strip_ocr'] = round(time.time() - t0, 3)

        # Step 2c: 角度弧标注 OCR
        # Inspect compact arc-angle labels through the isolated legacy OCR pass.
        t0 = time.time()
        angle_label_ocr_results = []
        if not Config.OCR_ANGLE_LABEL_ENABLED:
            timings['angle_label_ocr_skipped_disabled'] = True
            ocr_pass_counters['angle_label']['skipped_disabled'] = 1
        elif not _ocr_budget_expired('angle_label_ocr', ocr_pass_counters['angle_label']):
            angle_deadline = ocr_deadline
            angle_max_seconds = max(0, int(Config.OCR_ANGLE_LABEL_MAX_SECONDS or 0))
            if angle_max_seconds > 0:
                local_deadline = time.monotonic() + angle_max_seconds
                angle_deadline = (
                    min(angle_deadline, local_deadline)
                    if angle_deadline is not None else local_deadline
                )
            if page_img_ocr is None:
                page_img_ocr = _do_render(dpi)
            angle_label_ocr_results = run_angle_label_ocr(
                page_img_ocr,
                text_regions,
                ocr_results,
                dpi=dpi,
                verbose=verbose,
                deadline=angle_deadline,
                max_regions=Config.OCR_ANGLE_LABEL_MAX_REGIONS,
                diag=ocr_pass_counters['angle_label'],
            )
        if angle_label_ocr_results:
            ocr_results.extend(angle_label_ocr_results)
        timings['angle_label_ocr'] = round(time.time() - t0, 3)

        # Step 2e: 旋转半径标签 OCR
        # Bottom rib/circle radius labels are often small slanted vector text.
        # A bounded targeted pass recovers them before the broad DBSCAN OCR budget
        # is spent elsewhere.
        t0 = time.time()
        radius_label_ocr_results = []
        if not Config.OCR_RADIUS_LABEL_ENABLED:
            timings['radius_label_ocr_skipped_disabled'] = True
            ocr_pass_counters['radius_label']['skipped_disabled'] = 1
        elif not _ocr_budget_expired('radius_label_ocr', ocr_pass_counters['radius_label']):
            radius_deadline = ocr_deadline
            radius_max_seconds = max(0, int(Config.OCR_RADIUS_LABEL_MAX_SECONDS or 0))
            if radius_max_seconds > 0:
                local_deadline = time.monotonic() + radius_max_seconds
                radius_deadline = (
                    min(radius_deadline, local_deadline)
                    if radius_deadline is not None else local_deadline
                )
            if page_img_ocr is None:
                page_img_ocr = _do_render(dpi)
            radius_label_ocr_results = run_radius_label_ocr(
                page_img_ocr,
                text_regions,
                ocr_results,
                dpi=dpi,
                verbose=verbose,
                deadline=radius_deadline,
                max_regions=Config.OCR_RADIUS_LABEL_MAX_REGIONS,
                diag=ocr_pass_counters['radius_label'],
            )
        if radius_label_ocr_results:
            ocr_results.extend(radius_label_ocr_results)
        timings['radius_label_ocr'] = round(time.time() - t0, 3)

        # Step 2f: directed OCR（DBSCAN ROI）
        t0 = time.time()
        directed_ocr_results = []
        if not Config.OCR_PASS_DIRECTED_ENABLED:
            timings['ocr_skipped_disabled'] = True
            ocr_pass_counters['directed']['skipped_disabled'] = 1
        else:
            directed_text_regions = sort_text_regions_for_ocr(
                text_regions, page_width, page_height,
            )
            timings['ocr_directed_region_order'] = 'dimension_priority'
            directed_ocr_results = run_directed_ocr(
                pdf_bytes=pdf_bytes,
                page_index=page_index,
                text_regions=directed_text_regions,
                l1_lines=l1_lines,
                dpi=dpi,
                table_regions=ocr_table_regions,
                verbose=verbose,
                diag=ocr_pass_counters['directed'],
                deadline=directed_deadline,
                skip_regions=candidate_guided_skip_regions,
            )
        timings['ocr'] = round(time.time() - t0, 3)
        if directed_ocr_results:
            ocr_results.extend(directed_ocr_results)
        base_ocr_count = len(directed_ocr_results)

        # Step 2e: GD&T compartment 强制 OCR（用 gdt_frames_all）
        # These target-driven OCR passes run before broad retry scans so they are
        # not starved when directed OCR leaves only a small OCR budget remainder.
        t0 = time.time()
        gdt_comp_results = []
        timings['gdt_compartment_ocr_target_count'] = len(gdt_frames_all)
        if not Config.OCR_PASS_GDT_COMP_ENABLED:
            timings['gdt_compartment_ocr_skipped_disabled'] = True
            ocr_pass_counters['gdt_comp']['skipped_disabled'] = 1
        elif not gdt_frames_all:
            _mark_no_targets_skipped('gdt_compartment_ocr', ocr_pass_counters['gdt_comp'])
        elif not _ocr_budget_expired('gdt_compartment_ocr', ocr_pass_counters['gdt_comp']):
            gdt_comp_frames = [
                {'bbox': gf['bbox'], 'compartments': gf.get('compartments', [])}
                for gf in gdt_frames_all
            ]
            page_img_ocr = _do_render(dpi)
            gdt_comp_results = run_gdt_compartment_ocr(
                page_img_ocr, gdt_comp_frames, ocr_results, dpi=dpi,
                pdf_bytes=pdf_bytes, page_index=page_index,
                hires_dpi=Config.DPI_HIRES,
                verbose=verbose,
                deadline=ocr_deadline,
                diag=ocr_pass_counters['gdt_comp'])
            if gdt_comp_results:
                ocr_results.extend(gdt_comp_results)
        timings['gdt_compartment_ocr'] = round(time.time() - t0, 3)

        # Step 2f: Capsule 独立 OCR
        t0 = time.time()
        capsule_ocr_results = []
        timings['capsule_ocr_target_count'] = len(capsule_candidates)
        if not Config.OCR_PASS_CAPSULE_ENABLED:
            timings['capsule_ocr_skipped_disabled'] = True
            ocr_pass_counters['capsule']['skipped_disabled'] = 1
        elif not capsule_candidates:
            _mark_no_targets_skipped('capsule_ocr', ocr_pass_counters['capsule'])
        elif not _ocr_budget_expired('capsule_ocr', ocr_pass_counters['capsule']):
            if page_img_ocr is None:
                page_img_ocr = _do_render(dpi)
            capsule_ocr_results = run_capsule_ocr(
                page_img_ocr, capsule_candidates, ocr_results, dpi=dpi,
                verbose=verbose,
                deadline=ocr_deadline,
                diag=ocr_pass_counters['capsule'])
            if capsule_ocr_results:
                ocr_results.extend(capsule_ocr_results)
        timings['capsule_ocr'] = round(time.time() - t0, 3)

        # Step 2g: 条带补充扫描
        # Keep this before broad hires retry.  Once table-like false positives are
        # removed, directed OCR can consume much more of the page budget; strip OCR
        # is still geometry-guided and is the better use of the remaining time for
        # vertical/angled dimension text.
        t0 = time.time()
        strip_results = []
        if not Config.OCR_PASS_STRIP_ENABLED:
            timings['strip_ocr_skipped_disabled'] = True
            ocr_pass_counters['strip']['skipped_disabled'] = 1
        elif not _ocr_budget_expired('strip_ocr', ocr_pass_counters['strip']):
            strip_results = run_strip_ocr(
                page=page,
                l1_lines=l1_lines,
                existing_ocr_results=ocr_results,
                pdf_bytes=pdf_bytes,
                page_index=page_index,
                frame_borders=frame_borders,
                dpi=dpi,
                verbose=verbose,
                deadline=ocr_deadline,
                diag=ocr_pass_counters['strip'],
            )
        if strip_results:
            ocr_results.extend(strip_results)
        timings['strip_ocr'] = round(time.time() - t0, 3)

        # Step 2h: hires retry（对空 region 用 300DPI）
        t0 = time.time()
        hires_results = []
        if not Config.OCR_PASS_HIRES_RETRY_ENABLED:
            timings['hires_retry_skipped_disabled'] = True
            ocr_pass_counters['hires_retry']['skipped_disabled'] = 1
        elif not _ocr_budget_expired('hires_retry', ocr_pass_counters['hires_retry']):
            hires_results = run_empty_region_retry(
                pdf_bytes=pdf_bytes,
                page_index=page_index,
                text_regions=text_regions,
                existing_ocr_results=ocr_results,
                hires_dpi=Config.DPI_HIRES,
                verbose=verbose,
                diag=ocr_pass_counters['hires_retry'],
                max_retry=hires_retry_cap,
                deadline=ocr_deadline,
                page_width=page_width,
                page_height=page_height,
            )
        if hires_results:
            ocr_results.extend(hires_results)
        timings['hires_retry'] = round(time.time() - t0, 3)

        # Step 2i: GD&T 符号分类。hybrid 默认先用矢量反演；只有 vector 未命中
        # 的框才进入 YOLO fallback，避免 raster 模型覆盖高置信矢量结果。
        vector_gdt_results = None
        yolo_gdt_results = None
        gdt_symbol_results = None
        _req_gdt = _effective_gdt_backend(settings)
        _gdt_src = _req_gdt if _req_gdt in {'vector', 'yolo', 'hybrid'} else Config.GDT_BACKEND
        gdt_backend = _gdt_src if _gdt_src in {'vector', 'yolo', 'hybrid'} else 'hybrid'

        def _unknown_gdt_symbol(method='unknown', **extra):
            return {
                'symbol_class': -1,
                'symbol_name': 'unknown',
                'symbol_unicode': '?',
                'confidence': 0.0,
                'method': method,
                **extra,
            }

        def _vector_frame_symbols(hits):
            symbols = []
            for hit in hits or []:
                try:
                    cls = int(hit.get('symbol_class', -1))
                except (TypeError, ValueError):
                    cls = -1
                if cls < 0 or hit.get('confidence', 0.0) < Config.GDT_VECTOR_CONF_THRESHOLD:
                    continue
                symbols.append({
                    'symbol_class': cls,
                    'symbol_name': hit.get('symbol_name', ''),
                    'symbol_unicode': hit.get('symbol_unicode', ''),
                    'confidence': hit.get('confidence', 0.0),
                    'compartment_index': hit.get('compartment_index'),
                    'method': 'vector_inversion',
                })
            return symbols

        if gdt_frames_all and gdt_backend in {'vector', 'hybrid'}:
            t0 = time.time()
            inverter = GDTVectorInverter()
            vector_gdt_results = []
            for frame in gdt_frames_all:
                vector_has_image = False
                hits = []

                try:
                    hits = inverter.invert_in_frame(
                        fitz_page,
                        frame.get('bbox'),
                        frame.get('compartments'),
                    )
                    first = next(
                        (h for h in hits if h.get('compartment_index') in (0, None)),
                        None,
                    )
                except Exception as e:
                    if verbose:
                        print(f"[pipeline gdt_vector] failed: {e}")
                    first = None

                if _accept_gdt_vector_hit(frame, first, Config.GDT_VECTOR_CONF_THRESHOLD):
                    vector_gdt_results.append({
                        'symbol_class': first['symbol_class'],
                        'symbol_name': first['symbol_name'],
                        'symbol_unicode': first['symbol_unicode'],
                        'confidence': first.get('confidence', 0.0),
                        'method': 'vector_inversion',
                        'vector_has_image': vector_has_image,
                        'frame_symbols': _vector_frame_symbols(hits),
                    })
                else:
                    try:
                        regions = inverter._compartment_regions(
                            inverter._normalize_bbox(frame.get('bbox')),
                            frame.get('compartments'),
                        )
                        if regions:
                            region = inverter._inset_region(regions[0][1])
                            drawings = inverter._collect_region_drawings(fitz_page, region)
                            vector_has_image = bool(drawings.get('image_blocks'))
                    except Exception:
                        vector_has_image = False
                    vector_gdt_results.append(_unknown_gdt_symbol(
                        method='vector_inversion',
                        vector_has_image=vector_has_image,
                        frame_symbols=_vector_frame_symbols(hits),
                        needs_review=frame.get('_origin') == 'rect' or vector_has_image,
                    ))
            gdt_symbol_results = list(vector_gdt_results)
            timings['gdt_vector'] = round(time.time() - t0, 3)
            timings['gdt_vector_hits'] = sum(1 for r in vector_gdt_results if r.get('symbol_class', -1) >= 0)

        if gdt_backend == 'yolo' and gdt_frames_all:
            gdt_symbol_results = [
                _unknown_gdt_symbol(method='yolo')
                for _ in gdt_frames_all
            ]

        yolo_fallback_indices = []
        if gdt_detector and gdt_frames_all and gdt_backend in {'yolo', 'hybrid'}:
            if gdt_backend == 'yolo':
                yolo_fallback_indices = list(range(len(gdt_frames_all)))
            else:
                yolo_fallback_indices = [
                    idx for idx, result in enumerate(gdt_symbol_results or [])
                    if result.get('symbol_class', -1) < 0
                    and (
                        gdt_frames_all[idx].get('_origin', 'line') == 'line'
                        or result.get('vector_has_image')
                    )
                ]

        if gdt_detector and gdt_frames_all and yolo_fallback_indices:
            t0 = time.time()
            yolo_dpi = Config.DPI_YOLO
            page_img_yolo = _do_render(yolo_dpi)
            scale_yolo = yolo_dpi / 72.0
            pixel_gdt_frames = []
            for frame_index in yolo_fallback_indices:
                frame = gdt_frames_all[frame_index]
                fb = frame['bbox']
                px1 = int(fb['x'] * scale_yolo)
                py1 = int(fb['y'] * scale_yolo)
                px2 = int((fb['x'] + fb['w']) * scale_yolo)
                py2 = int((fb['y'] + fb['h']) * scale_yolo)
                pixel_gdt_frames.append({'bbox': [px1, py1, px2, py2]})
            yolo_subset_results = gdt_detector.detect(page_img_yolo, pixel_gdt_frames)
            yolo_gdt_results = [
                _unknown_gdt_symbol(method='yolo')
                for _ in gdt_frames_all
            ]
            if gdt_symbol_results is None:
                gdt_symbol_results = [
                    _unknown_gdt_symbol(method='yolo')
                    for _ in gdt_frames_all
            ]
            for frame_index, yolo_det in zip(yolo_fallback_indices, yolo_subset_results):
                previous_vector = (
                    gdt_symbol_results[frame_index]
                    if gdt_symbol_results and frame_index < len(gdt_symbol_results)
                    else {}
                )
                yolo_det = {**yolo_det, 'method': 'yolo'}
                if previous_vector.get('frame_symbols'):
                    yolo_det['frame_symbols'] = previous_vector.get('frame_symbols')
                yolo_gdt_results[frame_index] = yolo_det
                if yolo_det.get('symbol_class', -1) >= 0:
                    gdt_symbol_results[frame_index] = yolo_det
            timings['yolo_gdt'] = round(time.time() - t0, 3)
            timings['yolo_gdt_fallback_count'] = len(yolo_fallback_indices)
            if verbose:
                _hit = sum(1 for r in yolo_gdt_results if r.get('symbol_class', -1) >= 0)
                print(f"[pipeline yolo_gdt] frames={len(pixel_gdt_frames)} classified={_hit}")
        timings['gdt_backend'] = gdt_backend

        # Step 2j: Phase 2 — 90°/270° 多方向 OCR
        # 这是分钟级深扫兜底。默认关闭，避免普通 /analyze 和批量评估在
        # 数百个 text_regions 上串行 OCR 两个旋转方向；需要压榨召回时用
        # OCR_DEEP_SCAN=1 或 OCR_PHASE2_ENABLED=1 显式打开。
        t0 = time.time()
        phase2_budget_available = bool(Config.OCR_PHASE2_ENABLED)
        if phase2_budget_available and _ocr_budget_expired('phase2_ocr'):
            _mark_deadline_skipped(ocr_pass_counters['phase2_90'])
            _mark_deadline_skipped(ocr_pass_counters['phase2_270'])
            phase2_budget_available = False
        if not Config.OCR_PHASE2_ENABLED:
            timings['phase2_ocr_skipped_disabled'] = True
            ocr_pass_counters['phase2_90']['skipped_disabled'] = 1
            ocr_pass_counters['phase2_270']['skipped_disabled'] = 1
        elif not Config.OCR_PASS_PHASE2_90_ENABLED and not Config.OCR_PASS_PHASE2_270_ENABLED:
            timings['phase2_ocr_skipped_disabled'] = True
            ocr_pass_counters['phase2_90']['skipped_disabled'] = 1
            ocr_pass_counters['phase2_270']['skipped_disabled'] = 1
            phase2_budget_available = False
        if phase2_budget_available and page_img_ocr is None:
            page_img_ocr = _do_render(dpi)

        # 90° 副本（R14.2：单独打点，看 90/270 哪个是大头）
        t_90 = time.time()
        ocr_90 = []
        if phase2_budget_available and not Config.OCR_PASS_PHASE2_90_ENABLED:
            timings['phase2_ocr_90_skipped_disabled'] = True
            ocr_pass_counters['phase2_90']['skipped_disabled'] = 1
        elif phase2_budget_available and not _ocr_budget_expired(
            'phase2_ocr_90', ocr_pass_counters['phase2_90']
        ):
            phase2_regions = sort_text_regions_for_ocr(
                text_regions, page_width, page_height,
            )
            page_img_90 = page_img_ocr.transpose(Image.Transpose.ROTATE_270)
            regions_90 = [transform_region_cw90(r, page_width, page_height)
                          for r in phase2_regions]
            ocr_90 = run_rotated_ocr(
                page_img_90, regions_90, dpi=dpi,
                diag=ocr_pass_counters['phase2_90'],
                deadline=ocr_deadline,
                max_regions=Config.OCR_PHASE2_MAX_REGIONS,
            )
        ocr_90_back = [transform_ocr_back_cw90(r, page_width, page_height)
                       for r in ocr_90]
        timings['phase2_ocr_90'] = round(time.time() - t_90, 3)

        # 270° 副本
        t_270 = time.time()
        ocr_270 = []
        if phase2_budget_available and not Config.OCR_PASS_PHASE2_270_ENABLED:
            timings['phase2_ocr_270_skipped_disabled'] = True
            ocr_pass_counters['phase2_270']['skipped_disabled'] = 1
        elif phase2_budget_available and not _ocr_budget_expired(
            'phase2_ocr_270', ocr_pass_counters['phase2_270']
        ):
            phase2_regions = sort_text_regions_for_ocr(
                text_regions, page_width, page_height,
            )
            page_img_270 = page_img_ocr.transpose(Image.Transpose.ROTATE_90)
            regions_270 = [transform_region_cw270(r, page_width, page_height)
                           for r in phase2_regions]
            ocr_270 = run_rotated_ocr(
                page_img_270, regions_270, dpi=dpi,
                diag=ocr_pass_counters['phase2_270'],
                deadline=ocr_deadline,
                max_regions=Config.OCR_PHASE2_MAX_REGIONS,
            )
        ocr_270_back = [transform_ocr_back_cw270(r, page_width, page_height)
                        for r in ocr_270]
        timings['phase2_ocr_270'] = round(time.time() - t_270, 3)

        # dedup 单独打点（理论上 IoU dedup 跟 region 数 ~O(N²)，要小心其膨胀）
        t_dedup = time.time()
        ocr_results.extend(ocr_90_back)
        ocr_results.extend(ocr_270_back)
        n_total_raw = len(ocr_results)
        ocr_results = dedup_ocr_by_iou(ocr_results, iou_threshold=Config.OCR_DEDUP_IOU)
        timings['phase2_dedup'] = round(time.time() - t_dedup, 3)
        if verbose:
            print(f"[pipeline Phase2] 90°={len(ocr_90)} 270°={len(ocr_270)} "
                  f"raw={n_total_raw} dedup={len(ocr_results)}")
        timings['phase2_ocr'] = round(time.time() - t0, 3)

    # Step 2k: 多源工程去重
    n_before_eng = len(ocr_results)
    ocr_results = dedup_ocr_engineering(ocr_results, iou_threshold=Config.OCR_DEDUP_IOU)
    n_eng_dedup = n_before_eng - len(ocr_results)
    if verbose and n_eng_dedup:
        print(f"[pipeline eng_dedup] {n_before_eng} → {len(ocr_results)}")

    # Step 2l: ⌀ 前缀回填
    if diameter_glyphs:
        ocr_results = inject_diameter_prefix(ocr_results, diameter_glyphs, verbose=verbose)

    # ── Step 3: 语义组装 ─────────────────────────────────────────
    t0 = time.time()
    assembled = assemble_type_b(
        ocr_results=ocr_results,
        capsule_candidates=capsule_candidates,
        gdt_frame_candidates=gdt_frame_candidates,
        datum_candidates=datum_candidates,
        l1_lines=l1_lines,
        frame_borders=frame_borders,
        page_width=page_width,
        page_height=page_height,
        gdt_frames_from_lines=gdt_frames_all,
        page_fitz=fitz_page,
        yolo_gdt=gdt_symbol_results,
        table_cells=table_cells,
        reconstructed_rects=reconstructed_rects,
        detected_dim_lines=detected_dim_lines,
        pdf_bytes=pdf_bytes,
        page_index=page_index,
    )
    timings['assembler'] = round(time.time() - t0, 3)

    # ── Step 4: DimensionCandidate v0 debug layer ───────────────
    # Without CANDIDATE_GUIDED_OCR this remains diagnostic-only. When the
    # feature flag is on, the pre-OCR package is reused so candidate OCR
    # attempts stay attached to their originating candidates.
    if candidate_package is None:
        t0 = time.time()
        candidate_package = _build_candidate_package()
        timings['candidate_builder'] = round(time.time() - t0, 3)
        timings['candidate_builder_stage'] = 'post_assembler'
    else:
        timings['candidate_builder_reused_for_candidate_guided_ocr'] = True
    vector_anchor_axis_fields = _build_vector_anchor_axis_fields(candidate_package)
    vector_dimension_hypothesis_fields = _build_vector_dimension_hypothesis_fields(
        candidate_package,
    )
    r2n_nominal_rescue_shadow_fields = _build_r2n_nominal_rescue_shadow_fields(
        candidate_package,
    )
    final_consume_approval = resolve_final_consume_approval()
    if final_consume_approval.consumer_allowed:
        t0 = time.time()
        consume_result = consume_vector_dimension_hypotheses(
            approval=final_consume_approval,
            legacy_dimensions=assembled.get('dimensions', []),
            vector_dimension_hypotheses=vector_dimension_hypothesis_fields.get(
                'vector_dimension_hypotheses_v1',
                [],
            ),
            page_width=page_width,
            page_height=page_height,
        )
        assembled['dimensions'] = consume_result['dimensions']
        assembler_debug = assembled.setdefault('assembler_debug', {})
        assembler_debug['vector_final_consume_stats'] = consume_result['stats']
        assembler_debug['vector_final_consume_ledger'] = consume_result['ledger']
        vector_final_consume_fields = {
            'vector_final_dimensions_v1': consume_result['vector_dimensions'],
            'vector_final_consume_stats_v1': consume_result['stats'],
            'vector_final_consume_ledger_v1': consume_result['ledger'],
        }
        timings['vector_final_consume'] = round(time.time() - t0, 3)
    if Config.R7_GDT_VERTICAL_SHADOW_DUMP:
        t0 = time.time()
        r7_gdt_decimal_quad_rows = r2d_decimal_quad_rows
        if not r7_gdt_decimal_quad_rows:
            r7_gdt_decimal_quad_dump = detect_decimal_point_quads(
                fitz_page,
                page_index=page_index,
                pdf_stem=f"p{page_index + 1:03d}",
            )
            r7_gdt_decimal_quad_rows = r7_gdt_decimal_quad_dump.get(
                'vector_decimal_point_quad_v1',
                [],
            )
        gdt_vertical_shadow = build_gdt_decimal_seeded_frames_shadow_v1(
            page=fitz_page,
            decimal_point_rows=r7_gdt_decimal_quad_rows,
            current_frames=deepcopy(gdt_frames_all or []),
            consumer_allowed=False,
        )
        r7_gdt_vertical_shadow_fields = {
            'r7_gdt_vertical_shadow_v1': gdt_vertical_shadow.get(
                'r7_gdt_vertical_shadow_v1',
                [],
            ),
            'r7_gdt_vertical_shadow_stats_v1': gdt_vertical_shadow.get(
                'r7_gdt_vertical_shadow_stats_v1',
                {},
            ),
        }
        timings['r7_gdt_vertical_shadow_v1'] = round(time.time() - t0, 3)
    if Config.R7_ROTATED_RUNWAY_SHADOW_DUMP:
        t0 = time.time()
        rotated_runway_shadow = build_rotated_runway_shadow_v1(
            dimensions=assembled.get('dimensions', []),
            decimal_point_rows=r2d_decimal_quad_rows,
            consumer_allowed=False,
        )
        r7_rotated_runway_candidates = rotated_runway_shadow.get(
            'r7_rotated_runway_shadow_v1',
            [],
        )
        r7_rotated_runway_shadow_fields = {
            'r7_rotated_runway_shadow_v1': r7_rotated_runway_candidates,
            'r7_rotated_runway_shadow_stats_v1': rotated_runway_shadow.get(
                'r7_rotated_runway_shadow_stats_v1',
                {},
            ),
        }
        timings['r7_rotated_runway_shadow_v1'] = round(time.time() - t0, 3)
    if Config.R7_KEY_SHADOW_DUMP:
        t0 = time.time()
        key_shadow = classify_key_dimensions_shadow(
            dimensions=assembled.get('dimensions', []),
            capsule_candidates=list(capsule_candidates or []) + list(r7_rotated_runway_candidates or []),
            consumer_allowed=False,
        )
        r7_key_shadow_fields = {
            'r7_key_shadow_v1': key_shadow.get('dimensions', []),
            'r7_key_shadow_stats_v1': key_shadow.get('summary', {}),
        }
        timings['r7_key_shadow_v1'] = round(time.time() - t0, 3)
    if Config.R21_LAYERED_VECTOR_DIMENSION_DUMP:
        t0 = time.time()
        r21_decimal_rows = r2d_decimal_quad_rows
        if not r21_decimal_rows:
            r21_decimal_dump = detect_decimal_point_quads(
                fitz_page,
                page_index=page_index,
                pdf_stem=f"p{page_index + 1:03d}",
            )
            r21_decimal_rows = r21_decimal_dump.get(
                'vector_decimal_point_quad_v1',
                [],
            )
        r21_degree_dump = detect_degree_polylines_with_recall(
            fitz_page,
            page_index=page_index,
        )
        r21_degree_rows = r21_degree_dump.get('vector_degree_polyline_v1', [])
        r21_vector_glyph_tokens = vector_glyph_tokens_for_anchor
        if r21_vector_glyph_tokens is None:
            glyph_dump = build_vector_glyph_v1(
                page=page,
                page_index=page_index,
                text_regions=text_regions,
                diameter_glyphs=diameter_glyphs,
                include_digit_slots=True,
            )
            r21_vector_glyph_tokens = glyph_dump.get('tokens', [])
        r21_digit_match_rows = vector_digit_match_rows_for_phrase
        if not r21_digit_match_rows:
            prototype_library = (
                load_vector_digit_prototype_library(
                    Config.VECTOR_DIGIT_PROTOTYPE_LIBRARY,
                )
                if Config.VECTOR_DIGIT_PROTOTYPE_LIBRARY else None
            )
            digit_match_dump = build_vector_digit_matches_v1(
                vector_glyph_tokens=r21_vector_glyph_tokens,
                page=page,
                page_index=page_index,
                prototype_library=prototype_library,
                prototype_threshold=Config.VECTOR_DIGIT_PROTOTYPE_THRESHOLD,
                prototype_margin=Config.VECTOR_DIGIT_PROTOTYPE_MARGIN,
            )
            r21_digit_match_rows = digit_match_dump.get(
                'vector_digit_matches_v1',
                [],
            )
        r21_gdt_vertical_shadow = build_gdt_decimal_seeded_frames_shadow_v1(
            page=fitz_page,
            decimal_point_rows=r21_decimal_rows,
            current_frames=deepcopy(gdt_frames_all or []),
            consumer_allowed=False,
            allow_any_seed_angle=True,
        )
        r21_layered_vector_dimension_fields = (
            build_r21_layered_vector_dimension_shadow(
                decimal_point_rows=deepcopy(r21_decimal_rows or []),
                degree_rows=deepcopy(r21_degree_rows or []),
                vector_glyph_tokens=deepcopy(r21_vector_glyph_tokens or []),
                digit_match_rows=deepcopy(r21_digit_match_rows or []),
                gdt_line_frames=deepcopy(gdt_frames_from_lines or []),
                gdt_decimal_seeded_shadow=deepcopy(r21_gdt_vertical_shadow or {}),
                capsule_rows=deepcopy(capsule_candidates or []),
                diameter_glyphs=deepcopy(diameter_glyphs or []),
                fitz_page=_DeepcopySafePageProxy(fitz_page),
                page_index=page_index,
                consumer_allowed=False,
            )
        )
        timings['r21_layered_vector_dimension_v1'] = round(time.time() - t0, 3)
    assembler_polygon_compat_fields = {}
    if Config.ASSEMBLER_POLYGON_COMPAT:
        t0 = time.time()
        polygon_dump = build_assembler_polygon_compat_v1(
            dimensions=assembled.get('dimensions', []),
            dimension_candidates=candidate_package['dimension_candidates'],
            candidate_drop_ledger=candidate_package['candidate_drop_ledger'],
        )
        assembler_polygon_compat_fields = {
            'assembler_polygon_compat_v1': polygon_dump.get(
                'assembler_polygon_compat_v1',
                [],
            ),
            'assembler_polygon_compat_stats_v1': polygon_dump.get(
                'assembler_polygon_compat_stats_v1',
                {},
            ),
        }
        timings['assembler_polygon_compat_v1'] = round(time.time() - t0, 3)

    t0 = time.time()
    notes = []
    try:
        notes = extract_notes(
            fitz_page,
            pdf_words=pdf_text_words or None,
            ocr_results=ocr_results,
        )
    except Exception as e:
        timings['notes_extractor_error'] = str(e)
        if verbose:
            print(f"[pipeline notes_extractor] failed: {e}")
    timings['notes_extractor'] = round(time.time() - t0, 3)
    timings['notes_count'] = len(notes)

    _close_pipeline_fitz_doc()

    # R14.2 perf 诊断汇总：_do_render 调用统计
    timings['render_count']     = len(_render_log)
    timings['render_total_sec'] = round(sum(c['sec'] for c in _render_log), 3)
    timings['_render_calls']    = _render_log  # 完整列表 [{dpi, sec}, ...]

    directed_stopped_deadline = bool(
        (ocr_pass_counters.get('directed') or {}).get('stopped_deadline')
    )
    timings['ocr_directed_stopped_deadline'] = directed_stopped_deadline
    timings['ocr_directed_reserve_deadline_hit'] = bool(
        timings.get('ocr_directed_reserve_applied') and directed_stopped_deadline
    )
    global_deadline_counter_hit = any(
        counters.get('stopped_deadline')
        for pass_name, counters in ocr_pass_counters.items()
        if pass_name != 'directed' or not timings.get('ocr_directed_reserve_applied')
    )
    deadline_skip_hit = any(
        key.endswith('_skipped_deadline') and value
        for key, value in timings.items()
    )
    timings['ocr_global_deadline_hit'] = bool(global_deadline_counter_hit or deadline_skip_hit)
    timings['ocr_deadline_hit'] = bool(
        directed_stopped_deadline or global_deadline_counter_hit or deadline_skip_hit
    )
    ocr_pass_necessity_fields = {}
    if Config.OCR_PASS_NECESSITY_DUMP:
        t0 = time.time()
        necessity_dump = build_ocr_pass_necessity_v1(
            dimensions=assembled.get('dimensions', []),
            ocr_results=ocr_results,
            ocr_breakdown={
                'base': base_ocr_count,
                'hires': len(hires_results or []),
                'strip': len(strip_results or []),
                'gdt_comp': len(gdt_comp_results),
                'capsule': len(capsule_ocr_results),
                'dimline': len(dimline_ocr_results),
                'angle_label': len(angle_label_ocr_results),
                'radius_label': len(radius_label_ocr_results),
                'candidate_guided': len(candidate_guided_ocr_results),
                'phase2_90': len(ocr_90_back),
                'phase2_270': len(ocr_270_back),
                'total': len(ocr_results),
            },
            ocr_pass_counters=ocr_pass_counters,
            pass_results_by_name={
                'candidate_guided': candidate_guided_ocr_results,
                'dimline': dimline_ocr_results,
                'angle_label': angle_label_ocr_results,
                'radius_label': radius_label_ocr_results,
                'directed': directed_ocr_results,
                'gdt_comp': gdt_comp_results,
                'capsule': capsule_ocr_results,
                'strip': strip_results,
                'hires_retry': hires_results,
                'phase2_90': ocr_90_back,
                'phase2_270': ocr_270_back,
            },
            vector_glyph_tokens=vector_glyph_fields.get('vector_glyph_v1'),
        )
        ocr_pass_necessity_fields = {
            'ocr_pass_necessity_v1': necessity_dump.get('ocr_pass_necessity_v1', []),
            'ocr_pass_necessity_stats_v1': necessity_dump.get(
                'ocr_pass_necessity_stats_v1',
                {},
            ),
        }
        timings['ocr_pass_necessity_v1'] = round(time.time() - t0, 3)
    evidence_shadow_fields = {}
    if Config.EVIDENCE_SHADOW_DUMP:
        t0 = time.time()
        evidence_shadow_fields = build_evidence_shadow(
            dimensions=assembled.get('dimensions', []),
            dimension_candidates=candidate_package['dimension_candidates'],
            candidate_drop_ledger=candidate_package['candidate_drop_ledger'],
            ocr_results=ocr_results,
            diameter_glyphs=diameter_glyphs,
            capsule_candidates=capsule_candidates,
            gdt_frames_all=gdt_frames_all,
            detected_dim_lines=detected_dim_lines,
            detected_leader_lines=detected_leader_lines,
            assembler_debug=assembled.get('assembler_debug', {}),
            post_filter_drop_ledger=(
                assembled.get('assembler_debug', {}).get('post_filter_drop_ledger', [])
            ),
        )
        timings['evidence_shadow'] = round(time.time() - t0, 3)
    timings['total'] = round(time.time() - t_start, 3)

    return {
        **_vector_semantic_debug_fields(semantic_graph),
        **vector_glyph_fields,
        **vector_digit_match_fields,
        **vector_numeric_phrase_fields,
        **vector_dimension_hypothesis_fields,
        **vector_final_consume_fields,
        **r2n_nominal_rescue_shadow_fields,
        **r7_gdt_vertical_shadow_fields,
        **r7_rotated_runway_shadow_fields,
        **r7_key_shadow_fields,
        **r21_layered_vector_dimension_fields,
        **vector_anchor_axis_fields,
        **vector_times_detector_fields,
        **r2c_dump_fields,
        **ocr_crop_oriented_quad_fields,
        **ocr_pass_necessity_fields,
        **assembler_polygon_compat_fields,
        **evidence_shadow_fields,
        **({'view_seg_status_v1': view_seg_status_v1} if view_seg_status_v1 else {}),
        # ── 最终产物 ────────────────────────────────────────────
        'dimensions': assembled.get('dimensions', []),
        'references': assembled.get('references', []),
        'basic_dimensions': assembled.get('basic_dimensions', []),
        'gdt_frames': assembled.get('gdt_frames', []),
        'notes': notes,
        'dimension_candidates': candidate_package['dimension_candidates'],
        'candidate_drop_ledger': candidate_package['candidate_drop_ledger'],
        'candidate_debug_stats': candidate_package['candidate_debug_stats'],
        'assembler_debug': assembled.get('assembler_debug', {}),
        'post_filter_drop_ledger': assembled.get('assembler_debug', {}).get('post_filter_drop_ledger', []),
        'post_filter_debug_stats': assembled.get('assembler_debug', {}).get('post_filter_debug_stats', {}),
        'view_boxes': view_boxes,
        'timings': timings,

        # ── 中间产物（debug / test 用） ─────────────────────────
        'ocr_results': ocr_results,
        # Per-pass fields are raw pass additions; total is final merged OCR
        # boxes after downstream dedup/engineering cleanup.
        'ocr_breakdown': {
            'base': base_ocr_count,
            'hires': len(hires_results or []),
            'strip': len(strip_results or []),
            'gdt_comp': len(gdt_comp_results),
            'capsule': len(capsule_ocr_results),
            'dimline': len(dimline_ocr_results),
            'angle_label': len(angle_label_ocr_results),
            'radius_label': len(radius_label_ocr_results),
            'candidate_guided': len(candidate_guided_ocr_results),
            'phase2_90': len(ocr_90_back),
            'phase2_270': len(ocr_270_back),
            'total': len(ocr_results),
        },
        # 独立 OCR 列表（合并前的原始产出，供测试断言单个 pass 的效果）
        'hires_results': hires_results,
        'strip_results': strip_results,
        'gdt_comp_results': gdt_comp_results,
        'capsule_ocr_results': capsule_ocr_results,
        'dimline_ocr_results': dimline_ocr_results,
        'angle_label_ocr_results': angle_label_ocr_results,
        'radius_label_ocr_results': radius_label_ocr_results,
        'candidate_guided_ocr_results': candidate_guided_ocr_results,
        'n_eng_dedup': n_eng_dedup,
        'base_ocr_count': base_ocr_count,

        'lines_result': lines_result,
        'text_regions': text_regions,
        'rects_result': rects_result,
        'ocr_pass_counters': ocr_pass_counters,
        'capsule_candidates': capsule_candidates,
        'diameter_glyphs': diameter_glyphs,
        'gdt_frames_from_lines': gdt_frames_from_lines,
        'gdt_frames_all': gdt_frames_all,
        'reconstructed_rects': reconstructed_rects,
        'detected_dim_lines': detected_dim_lines,
        'detected_leader_lines': detected_leader_lines,
        'vector_gdt_results': vector_gdt_results,
        'yolo_gdt_results': yolo_gdt_results,
        'gdt_symbol_results': gdt_symbol_results,
        'annotation_lw': annotation_lw,
        'frame_borders': frame_borders,
        'connectivity_diag': conn_diag,
        'text_cluster_extra_points_diag': text_cluster_extra_points_diag,
    }


def _run_strict_candidate_tail(
    *,
    _build_candidate_package,
    _build_r2n_nominal_rescue_shadow_fields,
    _build_r33_m1_phrase_chain_fields,
    _build_vector_anchor_axis_fields,
    _build_vector_dimension_hypothesis_fields,
    _build_vector_phrase_region_fields,
    _close_pipeline_fitz_doc,
    _vector_only,
    annotation_lw,
    candidate_only,
    capsule_candidates,
    conn_diag,
    detected_dim_lines,
    detected_leader_lines,
    diameter_glyphs,
    fitz_page,
    frame_borders,
    gdt_frames_all,
    gdt_frames_from_lines,
    get_resolved_runtime,
    lines_result,
    page,
    page_index,
    r21_layered_vector_dimension_fields,
    r2c_dump_fields,
    r2d_decimal_quad_rows,
    reconstructed_rects,
    rects_result,
    semantic_graph,
    strict_diagnostics_out,
    strict_gdt_symbol_evidence,
    strict_yolo_trace_id,
    strict_yolo_view_status_v1,
    t_start,
    text_cluster_extra_points_diag,
    text_regions,
    timings,
    vector_digit_match_fields,
    vector_digit_match_rows_for_phrase,
    vector_glyph_fields,
    vector_glyph_tokens_for_anchor,
    vector_numeric_phrase_fields,
    vector_page_context,
    vector_times_detector_fields,
    view_boxes,
    view_seg_status_v1,
):
    t0 = time.time()
    candidate_package = _build_candidate_package()
    timings['candidate_builder'] = round(time.time() - t0, 3)
    try:
        vector_phrase_region_fields = _build_vector_phrase_region_fields()
        r33_m1_phrase_chain_fields = _build_r33_m1_phrase_chain_fields(
            vector_phrase_region_fields,
        )
        if _vector_only:
            if strict_gdt_symbol_evidence is None:
                raise VectorOnlyStageUnavailable(
                    reason="strict_gdt_symbol_evidence_missing",
                    stage="strict_gdt_candidates",
                    trace_id=str(strict_yolo_trace_id or ""),
                    page_index=page_index,
                )
            strict_gdt_started = time.time()
            chain_audit = r33_m1_phrase_chain_fields.get(
                "r33_m1_phrase_chain_audit_v1",
                {},
            )
            try:
                if (
                    chain_audit.get("status") == "ok"
                    and get_resolved_runtime() is not None
                ):
                    strict_gdt_result = (
                        build_strict_gdt_review_candidates(
                            symbol_evidence=strict_gdt_symbol_evidence,
                            decimal_point_rows=r2d_decimal_quad_rows,
                            page_context=vector_page_context,
                            runtime=get_resolved_runtime(),
                            trace_id=str(strict_yolo_trace_id or ""),
                            page_index=page_index,
                        )
                    )
                else:
                    strict_gdt_result = (
                        empty_strict_gdt_candidate_result(
                            symbol_evidence=strict_gdt_symbol_evidence,
                            trace_id=str(strict_yolo_trace_id or ""),
                            page_index=page_index,
                            reason=str(
                                chain_audit.get("reason")
                                or "r33_m1_runtime_unavailable"
                            ),
                        )
                    )
                review_candidates = build_review_candidates_v1(
                    chain_fields=r33_m1_phrase_chain_fields,
                    gdt_items=strict_gdt_result["items"],
                    expected_trace_id=str(
                        strict_yolo_trace_id or ""
                    ),
                    expected_page_index=page_index,
                )
            except VectorOnlyStageUnavailable:
                raise
            except Exception as exc:
                raise VectorOnlyStageUnavailable(
                    reason="strict_gdt_candidate_producer_failed",
                    stage="strict_gdt_candidates",
                    trace_id=str(strict_yolo_trace_id or ""),
                    page_index=page_index,
                ) from exc
            violations = validate_review_candidates_v1(
                review_candidates
            )
            if violations:
                raise VectorOnlyStageUnavailable(
                    reason="review_candidates_v1_invalid",
                    stage="strict_gdt_candidates",
                    trace_id=str(strict_yolo_trace_id or ""),
                    page_index=page_index,
                )
            r33_m1_phrase_chain_fields = {
                **r33_m1_phrase_chain_fields,
                STRICT_GDT_AUDIT_FIELD: strict_gdt_result["audit"],
                ROOT_KEY: review_candidates,
            }
            timings["strict_gdt_candidates"] = round(
                time.time() - strict_gdt_started,
                3,
            )
            if (
                Config.R21_LAYERED_VECTOR_DIMENSION_DUMP
                and strict_diagnostic_capture_active_v1()
            ):
                r21_started = time.time()
                try:
                    line_hex_cross_dump = (
                        detect_eight_line_hex_cross_shadows_v1(
                            vector_page_context.primitives,
                            page_index=page_index,
                        )
                    )
                    if line_hex_cross_dump.get("status") != "ok":
                        raise ValueError(
                            "eight-line hex-cross shadow failed closed"
                        )
                    line_hex_cross_rows = line_hex_cross_dump.get(
                        "vector_eight_line_hex_cross_shadow_v1"
                    )
                    line_hex_cross_stats = line_hex_cross_dump.get(
                        "vector_eight_line_hex_cross_shadow_stats_v1"
                    )
                    normalized_line_hex_cross_rows = (
                        normalize_eight_line_hex_cross_shadow_rows_v1(
                            line_hex_cross_rows,
                            page_index=page_index,
                            page_context=vector_page_context,
                        )
                    )
                    if (
                        normalized_line_hex_cross_rows is None
                        or not (
                            validate_eight_line_hex_cross_shadow_stats_v1(
                                line_hex_cross_stats,
                                page_index=page_index,
                                row_count=len(
                                    normalized_line_hex_cross_rows
                                ),
                                expected_status=(
                                    line_hex_cross_dump.get("status")
                                ),
                                item_primitives=(
                                    vector_page_context.primitives
                                ),
                            )
                        )
                    ):
                        raise ValueError(
                            "eight-line hex-cross shadow is unsafe"
                        )
                    line_hex_cross_rows = (
                        normalized_line_hex_cross_rows
                    )
                    line_hex_cross_fields = {
                        "vector_eight_line_hex_cross_shadow_v1": (
                            line_hex_cross_rows
                        ),
                        "vector_eight_line_hex_cross_shadow_stats_v1": (
                            line_hex_cross_stats
                        ),
                    }
                    r21_degree_dump = detect_degree_polylines_with_recall(
                        fitz_page,
                        page_index=page_index,
                    )
                    r21_degree_rows = r21_degree_dump.get(
                        "vector_degree_polyline_v1",
                        [],
                    )
                    r21_vector_glyph_tokens = vector_glyph_tokens_for_anchor
                    if r21_vector_glyph_tokens is None:
                        glyph_dump = build_vector_glyph_v1(
                            page=page,
                            page_index=page_index,
                            text_regions=text_regions,
                            diameter_glyphs=diameter_glyphs,
                            include_digit_slots=True,
                        )
                        r21_vector_glyph_tokens = glyph_dump.get(
                            "tokens",
                            [],
                        )
                    r21_gdt_vertical_shadow = (
                        build_gdt_decimal_seeded_frames_shadow_v1(
                            page=fitz_page,
                            decimal_point_rows=r2d_decimal_quad_rows,
                            current_frames=deepcopy(
                                gdt_frames_all or []
                            ),
                            consumer_allowed=False,
                            allow_any_seed_angle=True,
                        )
                    )
                    r21_decimal_inputs = deepcopy(
                        r2d_decimal_quad_rows or []
                    )
                    r21_degree_inputs = deepcopy(
                        r21_degree_rows or []
                    )
                    r21_vector_glyph_inputs = deepcopy(
                        r21_vector_glyph_tokens or []
                    )
                    r21_digit_match_inputs = deepcopy(
                        vector_digit_match_rows_for_phrase or []
                    )
                    r21_strict_pitch_input = deepcopy(
                        r33_m1_phrase_chain_fields.get(
                            "vector_pitch_phrase_read_dump_v1"
                        )
                    )
                    r21_gdt_line_inputs = deepcopy(
                        gdt_frames_from_lines or []
                    )
                    r21_capsule_inputs = deepcopy(
                        capsule_candidates or []
                    )
                    r21_diameter_inputs = deepcopy(
                        diameter_glyphs or []
                    )
                    r21_expected_input_counts = {
                        "decimal_point_row_count": sum(
                            isinstance(row, Mapping)
                            and row.get("consumer_allowed") is not True
                            for row in r21_decimal_inputs
                        ),
                        "degree_row_count": sum(
                            isinstance(row, Mapping)
                            and row.get("consumer_allowed") is not True
                            for row in r21_degree_inputs
                        ),
                        "diameter_glyph_row_count": sum(
                            isinstance(row, Mapping)
                            and row.get("consumer_allowed") is not True
                            for row in r21_diameter_inputs
                        ),
                        "vector_glyph_token_count": len(
                            r21_vector_glyph_inputs
                        ),
                        "digit_match_row_count": len(
                            r21_digit_match_inputs
                        ),
                        "gdt_line_frame_count": len(
                            r21_gdt_line_inputs
                        ),
                        "capsule_row_count": len(
                            r21_capsule_inputs
                        ),
                        "strict_pitch_read_count": len(
                            r21_strict_pitch_input.get(
                                "reads",
                                [],
                            )
                        )
                        if (
                            isinstance(
                                r21_strict_pitch_input,
                                Mapping,
                            )
                            and isinstance(
                                r21_strict_pitch_input.get(
                                    "reads",
                                    [],
                                ),
                                list,
                            )
                        )
                        else 0,
                    }
                    built_r21_fields = (
                        build_r21_layered_vector_dimension_shadow(
                            decimal_point_rows=r21_decimal_inputs,
                            degree_rows=r21_degree_inputs,
                            vector_glyph_tokens=(
                                r21_vector_glyph_inputs
                            ),
                            digit_match_rows=r21_digit_match_inputs,
                            strict_pitch_read_dump=(
                                r21_strict_pitch_input
                            ),
                            gdt_line_frames=r21_gdt_line_inputs,
                            gdt_decimal_seeded_shadow=deepcopy(
                                r21_gdt_vertical_shadow or {}
                            ),
                            capsule_rows=r21_capsule_inputs,
                            diameter_glyphs=r21_diameter_inputs,
                            fitz_page=_DeepcopySafePageProxy(fitz_page),
                            page_index=page_index,
                            consumer_allowed=False,
                            strict_diagnostic_mode=True,
                        )
                    )
                    r21_validation_status = (
                        validate_r21_pipeline_diagnostics_v1(
                            built_r21_fields,
                            page_index=page_index,
                            expected_input_counts=(
                                r21_expected_input_counts
                            ),
                        )
                    )
                    if r21_validation_status is not None:
                        r21_layered_vector_dimension_fields = {
                            "r21_diagnostic_status_v1": {
                                "schema_version": (
                                    "r21_diagnostic_status_v1"
                                ),
                                "page_index": int(page_index),
                                "status": r21_validation_status,
                                "consumer_allowed": False,
                            }
                        }
                    else:
                        r21_layered_vector_dimension_fields = {
                            **line_hex_cross_fields,
                            **built_r21_fields,
                            "r21_diagnostic_status_v1": {
                                "schema_version": (
                                    "r21_diagnostic_status_v1"
                                ),
                                "page_index": int(page_index),
                                "status": "ok",
                                "consumer_allowed": False,
                            },
                        }
                except VectorOnlyRuntimeForbidden:
                    raise
                except Exception:
                    r21_layered_vector_dimension_fields = {
                        "r21_diagnostic_status_v1": {
                            "schema_version": (
                                "r21_diagnostic_status_v1"
                            ),
                            "page_index": int(page_index),
                            "status": "builder_failed",
                            "consumer_allowed": False,
                        }
                    }
                finally:
                    timings[
                        "r21_layered_vector_dimension_v1"
                    ] = round(time.time() - r21_started, 3)
    except Exception:
        _close_pipeline_fitz_doc()
        raise
    vector_anchor_axis_fields = _build_vector_anchor_axis_fields(candidate_package)
    vector_dimension_hypothesis_fields = (
        _build_vector_dimension_hypothesis_fields(
            candidate_package,
        )
    )
    r2n_nominal_rescue_shadow_fields = (
        _build_r2n_nominal_rescue_shadow_fields(
            candidate_package,
        )
    )
    timings['candidate_only'] = True
    _vo_dimensions = []
    if _vector_only:
        timings["recognition_mode"] = "vector_only"
    timings['total'] = round(time.time() - t_start, 3)
    _close_pipeline_fitz_doc()
    candidate_payload = {
        **_vector_semantic_debug_fields(semantic_graph),
        **vector_glyph_fields,
        **vector_digit_match_fields,
        **vector_numeric_phrase_fields,
        **vector_dimension_hypothesis_fields,
        **r2n_nominal_rescue_shadow_fields,
        **vector_anchor_axis_fields,
        **vector_times_detector_fields,
        **r2c_dump_fields,
        **vector_phrase_region_fields,
        **r33_m1_phrase_chain_fields,
        **r21_layered_vector_dimension_fields,
        **({'view_seg_status_v1': view_seg_status_v1} if view_seg_status_v1 else {}),
        **(
            {'strict_yolo_view_status_v1': strict_yolo_view_status_v1}
            if strict_yolo_view_status_v1
            else {}
        ),
        'dimensions': _vo_dimensions,
        **(
            {
                'consumer_allowed': False,
                'display_allowed': False,
                'release_allowed': False,
            }
            if _vector_only
            else {}
        ),
        'references': [],
        'basic_dimensions': [],
        'gdt_frames': [],
        'notes': [],
        'dimension_candidates': candidate_package['dimension_candidates'],
        'candidate_drop_ledger': candidate_package['candidate_drop_ledger'],
        'candidate_debug_stats': candidate_package['candidate_debug_stats'],
        'view_boxes': view_boxes,
        'timings': timings,
        'ocr_results': [],
        'ocr_breakdown': zero_ocr_breakdown(),
        'ocr_pass_counters': {},
        'hires_results': [],
        'strip_results': [],
        'gdt_comp_results': [],
        'capsule_ocr_results': [],
        'dimline_ocr_results': [],
        'angle_label_ocr_results': [],
        'radius_label_ocr_results': [],
        'lines_result': lines_result,
        'text_regions': text_regions,
        'rects_result': rects_result,
        'capsule_candidates': capsule_candidates,
        'diameter_glyphs': diameter_glyphs,
        'gdt_frames_from_lines': gdt_frames_from_lines,
        'gdt_frames_all': gdt_frames_all,
        'reconstructed_rects': reconstructed_rects,
        'detected_dim_lines': detected_dim_lines,
        'detected_leader_lines': detected_leader_lines,
        'vector_gdt_results': None,
        'yolo_gdt_results': None,
        'gdt_symbol_results': None,
        'annotation_lw': annotation_lw,
        'frame_borders': frame_borders,
        'connectivity_diag': conn_diag,
        'text_cluster_extra_points_diag': text_cluster_extra_points_diag,
    }
    if candidate_only or _vector_only:
        return split_strict_pipeline_product_v1(
            candidate_payload,
            diagnostics_out=strict_diagnostics_out,
        )
    return candidate_payload


def run_type_b_pipeline(
    pdf_bytes: bytes,
    page,                        # pdfplumber.Page
    page_index: int,             # 0-based
    dpi: int = None,
    gdt_detector=None,           # 可选 YOLO-B 检测器实例（None 则跳过）
    verbose: bool = False,
    hires_retry_cap: int | None = None,   # None = Config.OCR_HIRES_RETRY_CAP
    candidate_only: bool = False,
    settings: "RecognitionSettings | None" = None,
    strict_yolo_source: Any = None,
    strict_yolo_trace_id: str | None = None,
    shared_fitz_doc: Any = None,
    r33_m1_phrase_runtime: R33M1PhraseRuntime | None = None,
    strict_diagnostics_out: dict[str, Any] | None = None,
) -> dict:
    """
    Type B PDF 的完整主管线（矢量 + OCR + 装配）。

    参数:
        pdf_bytes:    PDF 字节流
        page:         pdfplumber.Page 对象（调用方负责打开/关闭）
        page_index:   0-based 页码（给 fitz 和 ocr_engine 用）
        dpi:          渲染 DPI，默认 Config.DPI_DEFAULT
        gdt_detector: GDTDetector 实例或 None
        verbose:      打印诊断信息

    返回:
        详见模块 docstring 的 rich dict
    """
    dpi = dpi or Config.DPI_DEFAULT
    if hires_retry_cap is None:
        hires_retry_cap = Config.OCR_HIRES_RETRY_CAP
    t_start = time.time()
    ocr_max_seconds = Config.OCR_MAX_SECONDS
    ocr_deadline = None
    timings: dict = {}
    timings['ocr_max_seconds'] = ocr_max_seconds
    timings['ocr_directed_reserve_seconds'] = max(0, int(Config.OCR_DIRECTED_RESERVE_SECONDS or 0))
    timings['ocr_deep_scan'] = bool(Config.OCR_DEEP_SCAN)
    timings['phase2_enabled'] = bool(Config.OCR_PHASE2_ENABLED)
    timings['phase2_max_regions'] = Config.OCR_PHASE2_MAX_REGIONS
    _vector_only = settings is not None and settings.is_vector_only
    if not _vector_only and (
        strict_yolo_source is not None or strict_yolo_trace_id is not None
    ):
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_source_for_non_vector_mode",
            trace_id=str(strict_yolo_trace_id or ""),
            page_index=page_index,
        )
    if _vector_only and strict_yolo_source is None:
        resolve_strict_yolo_view_source(
            None,
            trace_id=str(strict_yolo_trace_id or ""),
            page_index=page_index,
        )

    if bool(getattr(Config, 'VECTOR_PRELABEL_FAST', False)):
        payload = run_vector_prelabel_fast_profile(
            pdf_bytes=pdf_bytes,
            page=page,
            page_index=page_index,
            verbose=verbose,
            strict_yolo_source=(strict_yolo_source if _vector_only else None),
            strict_yolo_trace_id=(strict_yolo_trace_id if _vector_only else None),
        )
        fast_timings = dict((payload.get('summary') or {}).get('timings') or {})
        fast_payload = {
            'dimensions': [],
            'references': [],
            'gdt_frames': [],
            'view_boxes': payload.get('view_boxes', []),
            'timings': fast_timings,
            'text_regions': [],
            'rects_result': {},
            'capsule_candidates': [],
            'diameter_glyphs': [],
            'gdt_frames_from_lines': [],
            'detected_dim_lines': [],
            'detected_leader_lines': [],
            'ocr_results': [],
            'ocr_breakdown': zero_ocr_breakdown(),
            'ocr_pass_counters': {},
            'annotation_lw': None,
            'frame_borders': [],
            'summary': payload.get('summary', {}),
            'vector_prelabel_v1': payload.get('vector_prelabel_v1', []),
            'vector_prelabel_summary_v1': payload.get('summary', {}),
            **(
                {'view_seg_status_v1': payload['view_seg_status_v1']}
                if 'view_seg_status_v1' in payload
                else {}
            ),
            **(
                {
                    'strict_yolo_view_status_v1': payload[
                        'strict_yolo_view_status_v1'
                    ]
                }
                if 'strict_yolo_view_status_v1' in payload
                else {}
            ),
        }
        if candidate_only or _vector_only:
            return split_strict_pipeline_product_v1(
                fast_payload,
                diagnostics_out=strict_diagnostics_out,
            )
        return fast_payload

    def _mark_deadline_skipped(diag: dict | None = None):
        if diag is not None:
            diag['skipped_deadline'] = diag.get('skipped_deadline', 0) + 1

    def _mark_no_targets_skipped(pass_name: str, diag: dict | None = None):
        timings[f'{pass_name}_skipped_no_targets'] = True
        if diag is not None:
            diag['skipped_no_targets'] = diag.get('skipped_no_targets', 0) + 1

    def _ocr_budget_expired(pass_name: str, diag: dict | None = None) -> bool:
        if ocr_deadline is None or time.monotonic() < ocr_deadline:
            return False
        timings[f'{pass_name}_skipped_deadline'] = True
        _mark_deadline_skipped(diag)
        return True

    # 记录 render_page_to_image 调用，并在 timings 中汇总次数、耗时和来源。
    # 这些诊断用于检查页面渲染是否被复用。
    _render_log: list = []
    def _do_render(render_dpi):
        _t = time.time()
        _img = render_page_to_image(pdf_bytes, page_index, render_dpi)
        _render_log.append({'dpi': render_dpi, 'sec': round(time.time() - _t, 3)})
        return _img

    page_width = float(page.width)
    page_height = float(page.height)

    # The strict request already owns one live PyMuPDF document for the YOLO
    # join.  Materialize its page drawings once and pass the same immutable
    # outer snapshot to every vector consumer.  Legacy/non-strict callers keep
    # their existing lazy extraction behavior.
    strict_shared_fitz_page = None
    vector_drawing_snapshot = None
    vector_drawing_snapshot_metrics = None
    if _vector_only and shared_fitz_doc is not None:
        snapshot_started = time.time()
        strict_shared_fitz_page = shared_fitz_doc[page_index]
        vector_drawing_snapshot = tuple(
            strict_shared_fitz_page.get_drawings()
        )
        vector_drawing_snapshot_metrics = {
            'schema_version': 'vector_drawing_snapshot_metrics_v1',
            'build_count': 1,
            'drawing_count': len(vector_drawing_snapshot),
            'reuse_consumer_count': 0,
            'consumers': [],
            'build_seconds': round(time.time() - snapshot_started, 6),
        }
        timings['vector_drawing_snapshot'] = (
            vector_drawing_snapshot_metrics
        )

    def _drawing_snapshot_kwargs(consumer: str) -> dict[str, Any]:
        if vector_drawing_snapshot is None:
            return {}
        assert vector_drawing_snapshot_metrics is not None
        vector_drawing_snapshot_metrics['reuse_consumer_count'] += 1
        vector_drawing_snapshot_metrics['consumers'].append(str(consumer))
        return {'drawing_snapshot': vector_drawing_snapshot}

    (
        lines_result,
        l1_lines,
        all_lines_raw,
        thresholds,
        thin_thick,
        annotation_lw,
        conn_diag,
        text_cluster_extra_points_diag,
        text_regions,
        rects_result,
        gdt_frame_candidates,
        datum_candidates,
        frame_borders,
        table_cells,
        capsule_candidates,
    ) = _extract_page_vector_geometry(
        _drawing_snapshot_kwargs=_drawing_snapshot_kwargs,
        _vector_only=_vector_only,
        page=page,
        page_index=page_index,
        pdf_bytes=pdf_bytes,
        strict_shared_fitz_page=strict_shared_fitz_page,
        timings=timings,
        verbose=verbose,
    )

    (
        view_boxes,
        view_seg_status_v1,
        strict_yolo_view_status_v1,
        ocr_table_regions,
    ) = _resolve_view_boxes_and_table_regions(
        _vector_only=_vector_only,
        page=page,
        page_index=page_index,
        pdf_bytes=pdf_bytes,
        rects_result=rects_result,
        settings=settings,
        strict_yolo_source=strict_yolo_source,
        strict_yolo_trace_id=strict_yolo_trace_id,
        timings=timings,
        verbose=verbose,
    )

    diameter_glyphs = _extract_diameter_glyphs(
        _drawing_snapshot_kwargs=_drawing_snapshot_kwargs,
        _vector_only=_vector_only,
        page=page,
        page_index=page_index,
        strict_yolo_trace_id=strict_yolo_trace_id,
        timings=timings,
        verbose=verbose,
    )
    vector_glyph_fields = {}
    vector_digit_match_fields = {}
    vector_numeric_phrase_fields = {}
    vector_dimension_hypothesis_fields = {}
    vector_final_consume_fields = {}
    r2n_nominal_rescue_shadow_fields = {}
    r7_key_shadow_fields = {}
    r7_rotated_runway_shadow_fields = {}
    r7_gdt_vertical_shadow_fields = {}
    r7_rotated_runway_candidates = []
    r21_layered_vector_dimension_fields = {}
    vector_anchor_axis_fields = {}
    vector_times_detector_fields = {}
    r2c_dump_fields = {}
    vector_glyph_tokens_for_anchor = None
    vector_digit_match_rows_for_phrase = []
    vector_numeric_phrase_rows_for_hypothesis = []
    vector_numeric_phrase_stats_for_hypothesis = {}
    vector_digit_slots_required = (
        Config.VECTOR_GLYPH_DIGIT_SLOTS_DUMP
        or Config.VECTOR_GLYPH_ANCHOR_AXIS_DUMP
        or Config.VECTOR_DIGIT_MATCH_DUMP
        or Config.VECTOR_NUMERIC_PHRASE_DUMP
        or Config.VECTOR_DIMENSION_HYPOTHESIS_DUMP
        or Config.VECTOR_GLYPH_CONSUME_FINAL
        or Config.R2N_NOMINAL_RESCUE_SHADOW_DUMP
    )
    (
        vector_glyph_fields,
        vector_digit_match_fields,
        vector_numeric_phrase_fields,
        vector_glyph_tokens_for_anchor,
        vector_digit_match_rows_for_phrase,
        vector_numeric_phrase_rows_for_hypothesis,
        vector_numeric_phrase_stats_for_hypothesis,
    ) = _build_vector_glyph_chain(
        _drawing_snapshot_kwargs=_drawing_snapshot_kwargs,
        _vector_only=_vector_only,
        diameter_glyphs=diameter_glyphs,
        page=page,
        page_index=page_index,
        pdf_bytes=pdf_bytes,
        strict_shared_fitz_page=strict_shared_fitz_page,
        text_regions=text_regions,
        timings=timings,
        vector_digit_slots_required=vector_digit_slots_required,
        vector_drawing_snapshot=vector_drawing_snapshot,
        vector_glyph_fields=vector_glyph_fields,
        vector_digit_match_fields=vector_digit_match_fields,
        vector_numeric_phrase_fields=vector_numeric_phrase_fields,
        vector_glyph_tokens_for_anchor=vector_glyph_tokens_for_anchor,
        vector_digit_match_rows_for_phrase=(
            vector_digit_match_rows_for_phrase
        ),
        vector_numeric_phrase_rows_for_hypothesis=(
            vector_numeric_phrase_rows_for_hypothesis
        ),
        vector_numeric_phrase_stats_for_hypothesis=(
            vector_numeric_phrase_stats_for_hypothesis
        ),
    )

    # ── Step 1b: GD&T 线段网格框 + 箭头 + 矩形重建 ─────────────────
    t0 = time.time()
    fitz_doc_owned = shared_fitz_doc is None
    fitz_doc = (
        fitz.open(stream=pdf_bytes, filetype="pdf")
        if fitz_doc_owned
        else shared_fitz_doc
    )
    fitz_page = (
        strict_shared_fitz_page
        if strict_shared_fitz_page is not None
        else fitz_doc[page_index]
    )
    vector_page_context: VectorPageContext | None = None
    if _vector_only:
        context_started = time.time()
        vector_page_context = VectorPageContext.from_page(
            fitz_page,
            pdf_stem=f"p{page_index + 1:03d}",
            page_num=page_index + 1,
            **_drawing_snapshot_kwargs('vector_page_context'),
        )
        timings['vector_page_context'] = round(
            time.time() - context_started,
            3,
        )

    def _close_pipeline_fitz_doc() -> None:
        if fitz_doc_owned:
            fitz_doc.close()

    (
        text_regions,
        r2d_decimal_quad_rows,
        r92_degree_rows,
        decimal_char_height_rows,
        r2d_decimal_corridor_rows,
        r2d_decimal_repeated_size_cohort_rows,
    ) = _build_decimal_anchor_fields(
        _drawing_snapshot_kwargs=_drawing_snapshot_kwargs,
        _vector_only=_vector_only,
        fitz_page=fitz_page,
        page_index=page_index,
        r2c_dump_fields=r2c_dump_fields,
        text_regions=text_regions,
        vector_page_context=vector_page_context,
    )
    vector_times_detector_fields = _build_vector_times_fields(
        fitz_page=fitz_page,
        page_index=page_index,
        r2d_decimal_quad_rows=r2d_decimal_quad_rows,
        timings=timings,
        vector_page_context=vector_page_context,
        vector_times_detector_fields=vector_times_detector_fields,
    )
    (
        candidate_vector_context_segments,
        pdf_text_words,
        gdt_frames_from_lines,
    ) = _build_context_segments_and_gdt_line_frames(
        _drawing_snapshot_kwargs=_drawing_snapshot_kwargs,
        all_lines_raw=all_lines_raw,
        fitz_page=fitz_page,
        frame_borders=frame_borders,
        thresholds=thresholds,
        timings=timings,
    )

    (
        detected_dim_lines,
        detected_leader_lines,
    ) = _detect_legacy_arrow_lines(
        _vector_only=_vector_only,
        all_lines_raw=all_lines_raw,
        annotation_lw=annotation_lw,
        text_regions=text_regions,
        thin_thick=thin_thick,
        timings=timings,
        verbose=verbose,
    )

    (
        reconstructed_rects,
        gdt_frames_all,
        strict_gdt_symbol_evidence,
    ) = _reconstruct_rects_and_gdt_frames(
        fitz_page=fitz_page,
        _drawing_snapshot_kwargs=_drawing_snapshot_kwargs,
        timings=timings,
        gdt_frames_from_lines=gdt_frames_from_lines,
        frame_borders=frame_borders,
        page_width=page_width,
        page_height=page_height,
        _vector_only=_vector_only,
        vector_drawing_snapshot=vector_drawing_snapshot,
        strict_yolo_trace_id=strict_yolo_trace_id,
        page_index=page_index,
        _close_pipeline_fitz_doc=_close_pipeline_fitz_doc,
    )

    (
        semantic_graph,
        vector_drawing_snapshot,
    ) = _build_semantic_graph_audit(
        page=page,
        fitz_page=fitz_page,
        page_index=page_index,
        text_regions=text_regions,
        frame_borders=frame_borders,
        table_cells=table_cells,
        rects_result=rects_result,
        view_boxes=view_boxes,
        l1_lines=l1_lines,
        timings=timings,
        verbose=verbose,
        vector_drawing_snapshot=vector_drawing_snapshot,
        vector_drawing_snapshot_metrics=vector_drawing_snapshot_metrics,
    )

    def _build_candidate_package() -> dict[str, Any]:
        if Config.CANDIDATE_EXPORT_ENABLED:
            try:
                return build_dimension_candidates(
                    page_index=page_index,
                    page_width=page_width,
                    page_height=page_height,
                    text_regions=text_regions,
                    pdf_text_words=pdf_text_words,
                    capsule_candidates=capsule_candidates,
                    gdt_frames_all=gdt_frames_all,
                    datum_candidates=datum_candidates,
                    detected_dim_lines=detected_dim_lines,
                    detected_leader_lines=detected_leader_lines,
                    diameter_glyphs=diameter_glyphs,
                    table_cells=table_cells,
                    table_regions=rects_result.get('table_regions', []),
                    ocr_table_regions=ocr_table_regions,
                    frame_borders=frame_borders,
                    vector_context_segments=candidate_vector_context_segments,
                    vector_context_density_threshold=(
                        Config.CANDIDATE_VECTOR_CONTEXT_DENSITY_WARNING_THRESHOLD
                        if Config.CANDIDATE_VECTOR_CONTEXT_DENSITY_ENABLED
                        else None
                    ),
                    vector_context_density_pad=Config.CANDIDATE_VECTOR_CONTEXT_DENSITY_PAD,
                    vector_context_density_inner_pad=(
                        Config.CANDIDATE_VECTOR_CONTEXT_DENSITY_INNER_PAD
                    ),
                    include_text_region_fallback=Config.CANDIDATE_TEXT_REGION_FALLBACK_ENABLED,
                )
            except Exception as e:
                if _vector_only:
                    raise VectorOnlyStageUnavailable(
                        reason="dimension_candidate_producer_failed",
                        stage="dimension_candidates",
                        trace_id=str(strict_yolo_trace_id or ""),
                        page_index=page_index,
                    ) from e
                if verbose:
                    print(f"[pipeline candidate_builder] failed: {e}")
                return {
                    'dimension_candidates': [],
                    'candidate_drop_ledger': [],
                    'candidate_debug_stats': {
                        'schema_version': CANDIDATE_SCHEMA_VERSION,
                        'enabled': False,
                        'page_index': page_index,
                        'active_count': 0,
                        'drop_count': 0,
                        'drop_reason_coverage': 1.0,
                        'error': str(e),
                    },
                }
        return {
            'dimension_candidates': [],
            'candidate_drop_ledger': [],
            'candidate_debug_stats': {
                'schema_version': CANDIDATE_SCHEMA_VERSION,
                'enabled': False,
                'page_index': page_index,
                'active_count': 0,
                'drop_count': 0,
                'drop_reason_coverage': 1.0,
            },
        }

    candidate_package: dict[str, Any] | None = None
    resolved_r33_m1_runtime: R33M1PhraseRuntime | None = (
        r33_m1_phrase_runtime
    )

    def _resolve_r33_m1_runtime() -> R33M1PhraseRuntime:
        nonlocal resolved_r33_m1_runtime
        if resolved_r33_m1_runtime is None:
            resolved_r33_m1_runtime = load_r33_m1_phrase_runtime(
                Config.R33_M1_TEMPLATE_LIBRARY_PATH,
            )
        return resolved_r33_m1_runtime

    # Strict recognition starts from the
    # shape detectors in ``_build_vector_phrase_region_fields`` below.
    def _build_vector_phrase_region_fields() -> dict[str, Any]:
        if not _vector_only:
            return {}
        if vector_page_context is None:
            raise RuntimeError("vector_page_context_missing")
        page_context = vector_page_context
        try:
            runtime = _resolve_r33_m1_runtime()
        except R33M1RuntimeUnavailable as exc:
            raise VectorOnlyStageUnavailable(
                reason=exc.reason,
                stage="r33_m1_phrase_runtime",
                trace_id=str(strict_yolo_trace_id or ""),
                page_index=page_index,
            ) from exc
        phrase_started = time.time()
        phrase_dump = build_shape_native_directional_walk_dump_v1(
            page_context=page_context,
            decimal_point_rows=r2d_decimal_quad_rows,
            degree_rows=r92_degree_rows,
            diameter_rows=diameter_glyphs,
            gdt_frames=gdt_frames_all,
            trace_id=str(strict_yolo_trace_id or ""),
            template_library=runtime.template_library,
        )
        timings['r92_directional_walk'] = round(
            time.time() - phrase_started,
            3,
        )
        context_metrics = page_context.runtime_metrics()
        timings['vector_page_context_runtime_metrics'] = context_metrics
        timings['r92_directional_walk_phrase_count'] = len(
            phrase_dump.get('phrases') or []
        )
        return {
            'r92_directional_walk_dump_v1': phrase_dump,
            'r32_phrase_runtime_audit_v1': {
                'schema_version': 'r32_phrase_runtime_audit_v1',
                'page_context_build_count': 1,
                'page_context_extraction_call_count': context_metrics.get(
                    'extraction_call_count',
                    0,
                ),
                'phrase_region_build_count': 1,
                'reader_call_count': 0,
                'consumer_allowed': False,
            },
        }

    def _build_r33_m1_phrase_chain_fields(
        phrase_fields: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal resolved_r33_m1_runtime
        if not _vector_only:
            return {}
        if vector_page_context is None:
            raise RuntimeError("vector_page_context_missing")
        phrase_dump = phrase_fields.get("r92_directional_walk_dump_v1")
        if not isinstance(phrase_dump, dict):
            fields = {
                "r33_m1_phrase_chain_audit_v1": (
                    build_r33_m1_unavailable_audit(
                        trace_id=str(strict_yolo_trace_id or ""),
                        page_index=page_index,
                        reason="directional_walk_dump_unavailable",
                        page_context=vector_page_context,
                    )
                ),
            }
        else:
            try:
                runtime = _resolve_r33_m1_runtime()
            except R33M1RuntimeUnavailable as exc:
                fields = {
                    "r33_m1_phrase_chain_audit_v1": (
                        build_r33_m1_unavailable_audit(
                            trace_id=str(phrase_dump.get("trace_id") or ""),
                            page_index=page_index,
                            reason=exc.reason,
                            details=exc.details,
                            page_context=vector_page_context,
                            phrase_dump=phrase_dump,
                        )
                    ),
                }
                runtime = None
            resolved_r33_m1_runtime = runtime
            if runtime is not None:
                chain_started = time.time()
                try:
                    chain_kwargs: dict[str, Any] = {
                        "phrase_dump": phrase_dump,
                        "page_context": vector_page_context,
                        "runtime": runtime,
                    }
                    if Config.R41_PRIMARY_PITCH_ATTEMPT_EXPERIMENT:
                        # R41 条件性实验：已入库，未采纳。这里只传页表
                        # primary char_width 的数值；不得构造 decimal quad
                        # 或 decimal_anchor_authority。
                        primary_rows = [
                            row
                            for row in decimal_char_height_rows
                            if (
                                isinstance(row, dict)
                                and row.get("role") == "primary"
                            )
                        ]
                        if len(primary_rows) == 1:
                            primary_pitch = primary_rows[0].get("char_width")
                            if (
                                isinstance(primary_pitch, (int, float))
                                and not isinstance(primary_pitch, bool)
                                and math.isfinite(float(primary_pitch))
                                and float(primary_pitch) > 0.0
                            ):
                                chain_kwargs[
                                    "experimental_primary_pitch_pt"
                                ] = float(primary_pitch)
                    fields = build_r33_m1_phrase_chain(
                        **chain_kwargs,
                    )
                except R33M1RuntimeUnavailable as exc:
                    fields = {
                        "r33_m1_phrase_chain_audit_v1": (
                            build_r33_m1_unavailable_audit(
                                trace_id=str(phrase_dump.get("trace_id") or ""),
                                page_index=page_index,
                                reason=exc.reason,
                                details=exc.details,
                                page_context=vector_page_context,
                                phrase_dump=phrase_dump,
                                status="fail_closed",
                            )
                        ),
                    }
                except (
                    KeyError,
                    OverflowError,
                    RecursionError,
                    RuntimeError,
                    TypeError,
                    UnicodeError,
                    ValueError,
                ) as exc:
                    fields = {
                        "r33_m1_phrase_chain_audit_v1": (
                            build_r33_m1_unavailable_audit(
                                trace_id=str(phrase_dump.get("trace_id") or ""),
                                page_index=page_index,
                                reason="phrase_chain_execution_failed",
                                details={"exception_type": type(exc).__name__},
                                page_context=vector_page_context,
                                phrase_dump=phrase_dump,
                                status="fail_closed",
                            )
                        ),
                    }
                timings["r33_m1_phrase_chain"] = round(
                    time.time() - chain_started,
                    3,
                )

        chain_audit = fields.get("r33_m1_phrase_chain_audit_v1") or {}
        chain_counts = chain_audit.get("counts") or {}
        phrase_runtime_audit = phrase_fields.get(
            "r32_phrase_runtime_audit_v1"
        )
        if isinstance(phrase_runtime_audit, dict):
            phrase_runtime_audit[
                "page_context_extraction_call_count"
            ] = int(
                chain_counts.get("page_context_extraction_call_count") or 0
            )
            phrase_runtime_audit["reader_call_count"] = int(
                chain_counts.get("reader_call_count") or 0
            )
        return fields

    def _build_vector_anchor_axis_fields(
        package: dict[str, Any],
    ) -> dict[str, Any]:
        if not Config.VECTOR_GLYPH_ANCHOR_AXIS_DUMP:
            return {}
        t_axis = time.time()
        dump = build_vector_anchor_axis_v1(
            case_id=f"p{int(page_index) + 1:03d}",
            page_index=page_index,
            dimension_candidates=package.get('dimension_candidates', []),
            vector_glyph_tokens=vector_glyph_tokens_for_anchor or [],
        )
        timings['vector_anchor_axis_v1'] = round(time.time() - t_axis, 3)
        return {
            'vector_anchor_axis_v1': dump.get('vector_anchor_axis_v1', []),
            'vector_anchor_axis_stats_v1': dump.get(
                'vector_anchor_axis_stats_v1',
                {},
            ),
        }

    def _build_vector_dimension_hypothesis_fields(
        package: dict[str, Any],
    ) -> dict[str, Any]:
        if not (
            Config.VECTOR_DIMENSION_HYPOTHESIS_DUMP
            or Config.VECTOR_GLYPH_CONSUME_FINAL
        ):
            return {}
        t_vector_hypothesis = time.time()
        dump = build_vector_dimension_hypotheses_v1(
            vector_numeric_phrases=vector_numeric_phrase_rows_for_hypothesis,
            dimension_candidates=package.get('dimension_candidates', []),
            page_index=page_index,
        )
        timings['vector_dimension_hypotheses_v1'] = round(
            time.time() - t_vector_hypothesis,
            3,
        )
        return {
            'vector_dimension_hypotheses_v1': dump.get(
                'vector_dimension_hypotheses_v1',
                [],
            ),
            'vector_dimension_hypothesis_stats_v1': dump.get(
                'vector_dimension_hypothesis_stats_v1',
                {},
            ),
        }

    def _build_r2n_nominal_rescue_shadow_fields(
        package: dict[str, Any],
    ) -> dict[str, Any]:
        if not Config.R2N_NOMINAL_RESCUE_SHADOW_DUMP:
            return {}
        t_r2n_nominal = time.time()
        dump = build_r2n_nominal_rescue_shadow_v1(
            vector_numeric_phrases=vector_numeric_phrase_rows_for_hypothesis,
            vector_numeric_phrase_stats=vector_numeric_phrase_stats_for_hypothesis,
            dimension_candidates=package.get('dimension_candidates', []),
            page_index=page_index,
        )
        timings['r2n_nominal_rescue_shadow_v1'] = round(
            time.time() - t_r2n_nominal,
            3,
        )
        return {
            'r2n_nominal_rescue_shadow_v1': dump.get(
                'r2n_nominal_rescue_shadow_v1',
                [],
            ),
            'r2n_nominal_rescue_shadow_stats_v1': dump.get(
                'r2n_nominal_rescue_shadow_stats_v1',
                {},
            ),
        }

    if candidate_only or _vector_only:
        return _run_strict_candidate_tail(
            _build_candidate_package=_build_candidate_package,
            _build_r2n_nominal_rescue_shadow_fields=_build_r2n_nominal_rescue_shadow_fields,
            _build_r33_m1_phrase_chain_fields=_build_r33_m1_phrase_chain_fields,
            _build_vector_anchor_axis_fields=_build_vector_anchor_axis_fields,
            _build_vector_dimension_hypothesis_fields=_build_vector_dimension_hypothesis_fields,
            _build_vector_phrase_region_fields=_build_vector_phrase_region_fields,
            _close_pipeline_fitz_doc=_close_pipeline_fitz_doc,
            _vector_only=_vector_only,
            annotation_lw=annotation_lw,
            candidate_only=candidate_only,
            capsule_candidates=capsule_candidates,
            conn_diag=conn_diag,
            detected_dim_lines=detected_dim_lines,
            detected_leader_lines=detected_leader_lines,
            diameter_glyphs=diameter_glyphs,
            fitz_page=fitz_page,
            frame_borders=frame_borders,
            gdt_frames_all=gdt_frames_all,
            gdt_frames_from_lines=gdt_frames_from_lines,
            get_resolved_runtime=lambda: resolved_r33_m1_runtime,
            lines_result=lines_result,
            page=page,
            page_index=page_index,
            r21_layered_vector_dimension_fields=r21_layered_vector_dimension_fields,
            r2c_dump_fields=r2c_dump_fields,
            r2d_decimal_quad_rows=r2d_decimal_quad_rows,
            reconstructed_rects=reconstructed_rects,
            rects_result=rects_result,
            semantic_graph=semantic_graph,
            strict_diagnostics_out=strict_diagnostics_out,
            strict_gdt_symbol_evidence=strict_gdt_symbol_evidence,
            strict_yolo_trace_id=strict_yolo_trace_id,
            strict_yolo_view_status_v1=strict_yolo_view_status_v1,
            t_start=t_start,
            text_cluster_extra_points_diag=text_cluster_extra_points_diag,
            text_regions=text_regions,
            timings=timings,
            vector_digit_match_fields=vector_digit_match_fields,
            vector_digit_match_rows_for_phrase=vector_digit_match_rows_for_phrase,
            vector_glyph_fields=vector_glyph_fields,
            vector_glyph_tokens_for_anchor=vector_glyph_tokens_for_anchor,
            vector_numeric_phrase_fields=vector_numeric_phrase_fields,
            vector_page_context=vector_page_context,
            vector_times_detector_fields=vector_times_detector_fields,
            view_boxes=view_boxes,
            view_seg_status_v1=view_seg_status_v1,
        )

    if Config.CANDIDATE_GUIDED_OCR_ENABLED:
        t0 = time.time()
        candidate_package = _build_candidate_package()
        timings['candidate_builder'] = round(time.time() - t0, 3)
        timings['candidate_builder_stage'] = 'pre_ocr'

    # ── Step 2: OCR passes ───────────────────────────────────────
    # codex 二审 P3 telemetry：每个 OCR pass 收集 raw / final / rejected counters
    ocr_pass_counters = {
        'directed': {},
        'hires_retry': {},
        'strip': {},
        'gdt_comp': {},
        'capsule': {},
        'dimline': {},
        'angle_label': {},
        'radius_label': {},
        'candidate_guided': {},
        'phase2_90': {},
        'phase2_270': {},
    }

    # The budget is for OCR passes, not the pre-OCR vector/detection setup.
    ocr_deadline = (
        time.monotonic() + ocr_max_seconds
        if ocr_max_seconds and ocr_max_seconds > 0 else None
    )
    timings['ocr_budget_started_after_sec'] = round(time.time() - t_start, 3)
    directed_deadline = ocr_deadline
    directed_reserve = timings['ocr_directed_reserve_seconds']
    if ocr_deadline is not None and directed_reserve > 0:
        directed_deadline = ocr_deadline - directed_reserve
        if directed_deadline <= time.monotonic():
            directed_deadline = ocr_deadline
            timings['ocr_directed_reserve_applied'] = False
            timings['ocr_directed_reserve_skipped_reason'] = 'reserve_exceeds_remaining_budget'
        else:
            timings['ocr_directed_reserve_applied'] = True
            timings['ocr_directed_deadline_offset_sec'] = (
                round(directed_deadline - time.monotonic(), 3)
            )
    else:
        timings['ocr_directed_reserve_applied'] = False

    return _run_legacy_ocr_tail(
        _build_candidate_package=_build_candidate_package,
        _build_r2n_nominal_rescue_shadow_fields=_build_r2n_nominal_rescue_shadow_fields,
        _build_vector_anchor_axis_fields=_build_vector_anchor_axis_fields,
        _build_vector_dimension_hypothesis_fields=_build_vector_dimension_hypothesis_fields,
        _close_pipeline_fitz_doc=_close_pipeline_fitz_doc,
        _do_render=_do_render,
        _mark_deadline_skipped=_mark_deadline_skipped,
        _mark_no_targets_skipped=_mark_no_targets_skipped,
        _ocr_budget_expired=_ocr_budget_expired,
        _render_log=_render_log,
        annotation_lw=annotation_lw,
        candidate_package=candidate_package,
        capsule_candidates=capsule_candidates,
        conn_diag=conn_diag,
        datum_candidates=datum_candidates,
        detected_dim_lines=detected_dim_lines,
        detected_leader_lines=detected_leader_lines,
        diameter_glyphs=diameter_glyphs,
        directed_deadline=directed_deadline,
        dpi=dpi,
        fitz_page=fitz_page,
        frame_borders=frame_borders,
        gdt_detector=gdt_detector,
        gdt_frame_candidates=gdt_frame_candidates,
        gdt_frames_all=gdt_frames_all,
        gdt_frames_from_lines=gdt_frames_from_lines,
        hires_retry_cap=hires_retry_cap,
        l1_lines=l1_lines,
        lines_result=lines_result,
        ocr_deadline=ocr_deadline,
        ocr_pass_counters=ocr_pass_counters,
        ocr_table_regions=ocr_table_regions,
        page=page,
        page_height=page_height,
        page_index=page_index,
        page_width=page_width,
        pdf_bytes=pdf_bytes,
        pdf_text_words=pdf_text_words,
        r21_layered_vector_dimension_fields=r21_layered_vector_dimension_fields,
        r2c_dump_fields=r2c_dump_fields,
        r2d_decimal_quad_rows=r2d_decimal_quad_rows,
        r7_gdt_vertical_shadow_fields=r7_gdt_vertical_shadow_fields,
        r7_key_shadow_fields=r7_key_shadow_fields,
        r7_rotated_runway_candidates=r7_rotated_runway_candidates,
        r7_rotated_runway_shadow_fields=r7_rotated_runway_shadow_fields,
        reconstructed_rects=reconstructed_rects,
        rects_result=rects_result,
        semantic_graph=semantic_graph,
        settings=settings,
        t_start=t_start,
        table_cells=table_cells,
        text_cluster_extra_points_diag=text_cluster_extra_points_diag,
        text_regions=text_regions,
        timings=timings,
        vector_digit_match_fields=vector_digit_match_fields,
        vector_digit_match_rows_for_phrase=vector_digit_match_rows_for_phrase,
        vector_final_consume_fields=vector_final_consume_fields,
        vector_glyph_fields=vector_glyph_fields,
        vector_glyph_tokens_for_anchor=vector_glyph_tokens_for_anchor,
        vector_numeric_phrase_fields=vector_numeric_phrase_fields,
        vector_times_detector_fields=vector_times_detector_fields,
        verbose=verbose,
        view_boxes=view_boxes,
        view_seg_status_v1=view_seg_status_v1,
    )
