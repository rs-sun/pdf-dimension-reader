"""Fast vector-only prelabel dump for R2q.

This module intentionally does not import or call OCR or text-region DBSCAN.
It emits structure-level rows for a prelabel/review workflow and leaves value
text unknown unless a future source is explicitly approved.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from typing import Any

import fitz

from arrow_detector import detect_arrows
from decimal_point_quad import detect_decimal_point_quads
from degree_polyline_strict import detect_degree_polylines
from pdf_analyzer import (
    detect_annotation_linewidth,
    detect_capsules,
    detect_gdt_frames_from_lines,
    extract_diameter_glyphs,
    extract_lines_by_width,
    extract_rects,
)
from vector_times_detector import detect_vector_times_candidates
from view_seg_router import segment_views
from yolo_view_async import resolve_strict_yolo_view_source


SCHEMA_VERSION = "vector_prelabel_v1"
SUMMARY_SCHEMA_VERSION = "vector_prelabel_summary_v1"
CONSUMER_ALLOWED = False
FAST_PROFILE_VEC3_STATUS = "skipped_fast_profile"
FAST_PROFILE_BUDGET_SKIP_STATUS = "skipped_fast_profile_budget"
UNKNOWN_VALUE_REASON = "round7_anchor_candidate_precision_too_low"
APPROVED_VALUE_SOURCES = frozenset({
    "round7_gold_box_clean_horizontal_decimal",
})


def build_vector_prelabel_v1(
    *,
    page_index: int,
    page_size_pt: tuple[float, float],
    decimal_point_rows: list[dict[str, Any]] | None = None,
    vector_glyph_tokens: list[dict[str, Any]] | None = None,
    diameter_glyphs: list[dict[str, Any]] | None = None,
    degree_rows: list[dict[str, Any]] | None = None,
    times_rows: list[dict[str, Any]] | None = None,
    gdt_frames: list[dict[str, Any]] | None = None,
    capsule_candidates: list[dict[str, Any]] | None = None,
    view_boxes: list[dict[str, Any]] | None = None,
    arrows: list[dict[str, Any]] | None = None,
    dimension_lines: list[dict[str, Any]] | None = None,
    timings: dict[str, Any] | None = None,
    view_seg_status: dict[str, Any] | None = None,
    strict_yolo_status: dict[str, Any] | None = None,
    view_segment_route: str = "product_router_yolo_a_allowed",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def append_row(
        *,
        kind: str,
        anchor_type: str | None,
        bbox: dict[str, Any] | None,
        symbol_prefix: str | None = None,
        source: str | None = None,
        source_id: str | None = None,
        frame_refs: list[str] | None = None,
        value_text: str | None = None,
        value_source: str | None = None,
        unknown_reason: str | None = None,
    ) -> dict[str, Any] | None:
        norm_bbox = _bbox_from_any(bbox)
        if norm_bbox is None:
            return None
        row = {
            "schema_version": SCHEMA_VERSION,
            "row_id": f"p{int(page_index) + 1:03d}_{kind}_{len(rows):06d}",
            "page": int(page_index) + 1,
            "kind": kind,
            "anchor_type": anchor_type,
            "bbox_pt": norm_bbox,
            "orientation": _orientation_from_bbox(norm_bbox),
            "symbol_prefix": symbol_prefix,
            "frame_refs": list(frame_refs or []),
            "value_text": value_text,
            "value_source": value_source,
            "unknown_reason": unknown_reason,
            "consumer_allowed": CONSUMER_ALLOWED,
            "source": source,
            "source_id": source_id,
        }
        rows.append(row)
        return row

    anchor_rows: list[dict[str, Any]] = []
    for row in decimal_point_rows or []:
        anchor = append_row(
            kind="anchor",
            anchor_type="decimal_point_quad",
            bbox=row.get("bbox") if isinstance(row, dict) else None,
            symbol_prefix=".",
            source="decimal_point_quad",
            source_id=_source_id(row),
        )
        if anchor:
            anchor_rows.append(anchor)

    for token in vector_glyph_tokens or []:
        glyph_type = str(token.get("glyph_type") or token.get("anchor_type") or "")
        if glyph_type != "plus_minus":
            continue
        anchor = append_row(
            kind="anchor",
            anchor_type="plus_minus",
            bbox=token,
            symbol_prefix="±",
            source="vector_glyph_token",
            source_id=_source_id(token),
        )
        if anchor:
            anchor_rows.append(anchor)

    for index, glyph in enumerate(diameter_glyphs or []):
        anchor = append_row(
            kind="anchor",
            anchor_type="diameter",
            bbox=glyph,
            symbol_prefix="⌀",
            source="extract_diameter_glyphs",
            source_id=_source_id(glyph, fallback=f"diameter_{index:04d}"),
        )
        if anchor:
            anchor_rows.append(anchor)

    for row in degree_rows or []:
        anchor = append_row(
            kind="anchor",
            anchor_type="degree_polyline_strict",
            bbox=row.get("bbox") if isinstance(row, dict) else None,
            symbol_prefix="°",
            source="degree_polyline_strict",
            source_id=_source_id(row),
        )
        if anchor:
            anchor_rows.append(anchor)

    for row in times_rows or []:
        anchor = append_row(
            kind="anchor",
            anchor_type="times",
            bbox=row.get("bbox") if isinstance(row, dict) else None,
            symbol_prefix="×",
            source="vector_times_detector",
            source_id=_source_id(row),
        )
        if anchor:
            anchor_rows.append(anchor)

    for row in gdt_frames or []:
        append_row(
            kind="frame",
            anchor_type="gdt_frame",
            bbox=row.get("bbox") if isinstance(row, dict) else None,
            source="detect_gdt_frames_from_lines_or_rects",
            source_id=_source_id(row),
        )

    for index, row in enumerate(capsule_candidates or []):
        append_row(
            kind="frame",
            anchor_type="capsule",
            bbox=row.get("bbox") if isinstance(row, dict) and isinstance(row.get("bbox"), dict) else row,
            source="detect_capsules",
            source_id=_source_id(row, fallback=f"capsule_{index:04d}"),
        )

    for index, row in enumerate(arrows or []):
        append_row(
            kind="frame",
            anchor_type="arrow",
            bbox=row.get("bbox") if isinstance(row, dict) else None,
            source="detect_arrows",
            source_id=_source_id(row, fallback=f"arrow_{index:04d}"),
        )

    for index, row in enumerate(dimension_lines or []):
        append_row(
            kind="frame",
            anchor_type="dimension_line",
            bbox=_dimension_line_bbox(row),
            source="detect_arrows",
            source_id=_source_id(row, fallback=f"dimension_line_{index:04d}"),
        )

    for index, row in enumerate(view_boxes or []):
        append_row(
            kind="view",
            anchor_type="view_box",
            bbox=row,
            source="view_seg_router.segment_views",
            source_id=_source_id(row, fallback=f"view_{index:04d}"),
        )

    for anchor in anchor_rows:
        append_row(
            kind="value",
            anchor_type=anchor["anchor_type"],
            bbox=anchor["bbox_pt"],
            symbol_prefix=anchor["symbol_prefix"],
            source="r2q_conservative_value_placeholder",
            frame_refs=[anchor["row_id"]],
            value_text=None,
            value_source=None,
            unknown_reason=UNKNOWN_VALUE_REASON,
        )

    normalized_timings = _normalized_timings(timings or {})
    counts_by_kind = Counter(str(row["kind"]) for row in rows)
    counts_by_anchor_type = Counter(str(row["anchor_type"]) for row in rows if row.get("anchor_type"))
    consumer_allowed_count = sum(1 for row in rows if row.get("consumer_allowed"))
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "prelabel_schema_version": SCHEMA_VERSION,
        "page": int(page_index) + 1,
        "page_size_pt": {
            "width": round(float(page_size_pt[0]), 6),
            "height": round(float(page_size_pt[1]), 6),
        },
        "row_count": len(rows),
        "counts_by_kind": dict(sorted(counts_by_kind.items())),
        "counts_by_anchor_type": dict(sorted(counts_by_anchor_type.items())),
        "value_text_count": sum(1 for row in rows if row.get("value_text") not in (None, "")),
        "ocr_results_count": 0,
        "consumer_allowed": CONSUMER_ALLOWED,
        "consumer_allowed_count": consumer_allowed_count,
        "timings": normalized_timings,
        "view_segment_route": str(view_segment_route),
        "value_policy": "default_null_until_round7_approved_subset",
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "vector_prelabel_v1": rows,
        "summary": summary,
        "vector_prelabel_summary_v1": summary,
        "page_index": int(page_index),
        "page": int(page_index) + 1,
        "view_boxes": list(view_boxes or []),
        "ocr_results_count": 0,
        "consumer_allowed": CONSUMER_ALLOWED,
        "consumer_allowed_count": consumer_allowed_count,
    }
    if view_seg_status is not None:
        payload["view_seg_status_v1"] = dict(view_seg_status)
    if strict_yolo_status is not None:
        payload["strict_yolo_view_status_v1"] = dict(strict_yolo_status)
    return payload


def run_vector_prelabel_fast_profile(
    *,
    pdf_bytes: bytes,
    page: Any,
    page_index: int,
    verbose: bool = False,
    strict_yolo_source: Any = None,
    strict_yolo_trace_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    timings: dict[str, Any] = {}

    t0 = time.perf_counter()
    _ = page.lines
    _ = page.curves
    _ = page.rects
    timings["vec_1_page_parse"] = _elapsed(t0)

    t0 = time.perf_counter()
    lines_result = extract_lines_by_width(page)
    all_lines_raw = lines_result.get("all_lines", []) or []
    thresholds = lines_result.get("thresholds", {}) or {}
    thin_thick = thresholds.get("thin_thick", 0.5)
    annotation_lw = detect_annotation_linewidth(all_lines_raw, thin_thick)
    timings["vec_2_lines_by_width"] = _elapsed(t0)
    timings["vec_3_cluster_text"] = FAST_PROFILE_VEC3_STATUS

    t0 = time.perf_counter()
    rects_result = extract_rects(page)
    timings["vec_4_extract_rects"] = _elapsed(t0)

    t0 = time.perf_counter()
    capsule_candidates = detect_capsules(page)
    timings["vec_5_detect_capsules"] = _elapsed(t0)

    t0 = time.perf_counter()
    view_seg_status = None
    strict_yolo_status = None
    view_segment_route = "product_router_yolo_a_allowed"
    if strict_yolo_source is not None or strict_yolo_trace_id is not None:
        view_boxes, view_seg_status, strict_yolo_status = (
            resolve_strict_yolo_view_source(
                strict_yolo_source,
                trace_id=str(strict_yolo_trace_id or ""),
                page_index=page_index,
            )
        )
        view_segment_route = "strict_yolo_no_fallback"
    else:
        try:
            view_boxes = segment_views(page, pdf_bytes=pdf_bytes, page_index=page_index)
        except Exception as exc:
            if verbose:
                print(f"[vector_prelabel_fast] segment_views failed: {exc}")
            view_boxes = []
            timings["view_segment_error"] = str(exc)
    timings["view_segment"] = _elapsed(t0)

    t0 = time.perf_counter()
    try:
        diameter_glyphs = extract_diameter_glyphs(page)
    except Exception as exc:
        if verbose:
            print(f"[vector_prelabel_fast] extract_diameter_glyphs failed: {exc}")
        diameter_glyphs = []
        timings["diameter_glyphs_error"] = str(exc)
    timings["diameter_glyphs"] = _elapsed(t0)

    t0 = time.perf_counter()
    arrows: list[dict[str, Any]] = []
    dimension_lines: list[dict[str, Any]] = []
    if annotation_lw is not None:
        tol = float(annotation_lw) * 0.15
        anno_lines = [
            line for line in all_lines_raw
            if (float(annotation_lw) - tol) <= float(line.get("lineWidth", 0.0)) <= (float(annotation_lw) + tol)
        ]
        arrow_result = detect_arrows(anno_lines)
        arrows = arrow_result.get("arrows", []) or []
        dimension_lines = arrow_result.get("dimension_lines", []) or []
    timings["arrow_detect"] = _elapsed(t0)

    fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        fitz_page = fitz_doc[page_index]

        t0 = time.perf_counter()
        decimal_dump = detect_decimal_point_quads(
            fitz_page,
            page_index=page_index,
            pdf_stem=f"p{page_index + 1:03d}",
        )
        decimal_rows = decimal_dump.get("vector_decimal_point_quad_v1", []) or []
        timings["vector_decimal_point_quad_v1"] = _elapsed(t0)

        t0 = time.perf_counter()
        degree_dump = detect_degree_polylines(fitz_page, page_index=page_index)
        degree_rows = degree_dump.get("vector_degree_polyline_v1", []) or []
        timings["vector_degree_polyline_v1"] = _elapsed(t0)

        times_rows = []
        if _env_bool("VECTOR_PRELABEL_FAST_TIMES", False):
            t0 = time.perf_counter()
            times_dump = detect_vector_times_candidates(
                fitz_page,
                page_index=page_index,
                pdf_stem=f"p{page_index + 1:03d}",
                decimal_point_rows=decimal_rows,
            )
            times_rows = times_dump.get("vector_times_candidates_v1", []) or []
            timings["vector_times_detector_v1"] = _elapsed(t0)
        else:
            timings["vector_times_detector_v1"] = FAST_PROFILE_BUDGET_SKIP_STATUS

        t0 = time.perf_counter()
        gdt_frames_from_lines = detect_gdt_frames_from_lines(
            fitz_page,
            frame_borders=rects_result.get("frame_borders", []),
        )
        timings["gdt_line_detect"] = _elapsed(t0)
    finally:
        fitz_doc.close()

    gdt_frames = list(rects_result.get("gdt_frames", []) or [])
    gdt_frames.extend(gdt_frames_from_lines or [])
    timings["total_sec"] = _elapsed(started)
    timings["total"] = timings["total_sec"]
    return build_vector_prelabel_v1(
        page_index=page_index,
        page_size_pt=(float(page.width), float(page.height)),
        decimal_point_rows=decimal_rows,
        vector_glyph_tokens=[],
        diameter_glyphs=diameter_glyphs,
        degree_rows=degree_rows,
        times_rows=times_rows,
        gdt_frames=gdt_frames,
        capsule_candidates=capsule_candidates,
        view_boxes=view_boxes,
        arrows=arrows,
        dimension_lines=dimension_lines,
        timings=timings,
        view_seg_status=view_seg_status,
        strict_yolo_status=strict_yolo_status,
        view_segment_route=view_segment_route,
    )


def stable_vector_prelabel_digest(payload: dict[str, Any], *, exclude_view_rows: bool = True) -> str:
    return _stable_digest(stable_vector_prelabel_field_dump(payload, exclude_view_rows=exclude_view_rows))


def stable_vector_prelabel_field_dump(
    payload: dict[str, Any],
    *,
    exclude_view_rows: bool = True,
) -> dict[str, Any]:
    rows = [
        _digest_row(row)
        for row in payload.get("vector_prelabel_v1", []) or []
        if not (exclude_view_rows and row.get("kind") == "view")
    ]
    rows.sort(key=lambda row: (
        str(row.get("kind")),
        str(row.get("anchor_type")),
        json.dumps(row.get("bbox_pt"), sort_keys=True, separators=(",", ":")),
        str(row.get("symbol_prefix")),
    ))
    return {
        "schema_version": payload.get("schema_version"),
        "rows": rows,
    }


def fast_prelabel_gate_summary(
    payload: dict[str, Any],
    *,
    repeat_digests: list[str] | None = None,
) -> dict[str, Any]:
    rows = payload.get("vector_prelabel_v1", []) or []
    summary = payload.get("summary", {}) or {}
    timings = summary.get("timings", {}) or {}
    if int(summary.get("ocr_results_count") or payload.get("ocr_results_count") or 0) != 0:
        return _gate("no_go_policy_violation", payload, repeat_digests)
    if int(summary.get("consumer_allowed_count") or payload.get("consumer_allowed_count") or 0) != 0:
        return _gate("no_go_policy_violation", payload, repeat_digests)
    if any(row.get("consumer_allowed") for row in rows):
        return _gate("no_go_policy_violation", payload, repeat_digests)
    if timings.get("vec_3_cluster_text") != FAST_PROFILE_VEC3_STATUS:
        return _gate("no_go_vec3_not_skipped", payload, repeat_digests)
    if _unapproved_value_rows(rows):
        return _gate("no_go_unapproved_value_text", payload, repeat_digests)
    if repeat_digests is not None and (len(repeat_digests) < 3 or len(set(repeat_digests)) != 1):
        return _gate("no_go_determinism_failed", payload, repeat_digests)
    return _gate("go_fast_prelabel_dump", payload, repeat_digests)


def _unapproved_value_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("kind") == "value"
        and row.get("value_text") not in (None, "")
        and row.get("value_source") not in APPROVED_VALUE_SOURCES
    ]


def _gate(
    status: str,
    payload: dict[str, Any],
    repeat_digests: list[str] | None,
) -> dict[str, Any]:
    rows = payload.get("vector_prelabel_v1", []) or []
    summary = payload.get("summary", {}) or {}
    return {
        "gate_status": status,
        "passed": status == "go_fast_prelabel_dump",
        "ocr_results_count": int(summary.get("ocr_results_count") or payload.get("ocr_results_count") or 0),
        "consumer_allowed_count": int(summary.get("consumer_allowed_count") or payload.get("consumer_allowed_count") or 0),
        "unapproved_value_text_count": len(_unapproved_value_rows(rows)),
        "repeat_count": len(repeat_digests or []),
        "deterministic": None if repeat_digests is None else len(set(repeat_digests)) == 1,
    }


def _normalized_timings(timings: dict[str, Any]) -> dict[str, Any]:
    out = dict(timings)
    out["vec_3_cluster_text"] = FAST_PROFILE_VEC3_STATUS
    total = out.get("total_sec", out.get("total", 0.0))
    try:
        total_sec = float(total)
    except (TypeError, ValueError):
        total_sec = 0.0
    out["total_sec"] = round(max(0.0, total_sec), 6)
    out["total"] = out["total_sec"]
    return out


def _bbox_from_any(item: Any) -> dict[str, float] | None:
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
            return {
                "x": round(x, 6),
                "y": round(y, 6),
                "w": round(w, 6),
                "h": round(h, 6),
            }
        if all(key in item for key in ("x0", "y0", "x1", "y1")):
            x0 = float(item["x0"])
            y0 = float(item["y0"])
            x1 = float(item["x1"])
            y1 = float(item["y1"])
            w = x1 - x0
            h = y1 - y0
            if w <= 0.0 or h <= 0.0:
                return None
            return {
                "x": round(x0, 6),
                "y": round(y0, 6),
                "w": round(w, 6),
                "h": round(h, 6),
            }
        if isinstance(item.get("bbox_pt"), dict):
            return _bbox_from_any(item["bbox_pt"])
    except (TypeError, ValueError):
        return None
    return None


def _dimension_line_bbox(row: dict[str, Any]) -> dict[str, float] | None:
    if not isinstance(row, dict):
        return None
    bbox = _bbox_from_any(row)
    if bbox:
        return bbox
    start = row.get("start")
    end = row.get("end")
    if not (
        isinstance(start, (list, tuple))
        and isinstance(end, (list, tuple))
        and len(start) >= 2
        and len(end) >= 2
    ):
        return None
    x0 = min(float(start[0]), float(end[0]))
    y0 = min(float(start[1]), float(end[1]))
    x1 = max(float(start[0]), float(end[0]))
    y1 = max(float(start[1]), float(end[1]))
    return {
        "x": round(x0, 6),
        "y": round(y0, 6),
        "w": round(max(0.001, x1 - x0), 6),
        "h": round(max(0.001, y1 - y0), 6),
    }


def _orientation_from_bbox(bbox: dict[str, float]) -> str:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if width <= 0.0 or height <= 0.0:
        return "unknown"
    ratio = width / height
    if ratio >= 1.8:
        return "H"
    if ratio <= 1 / 1.8:
        return "V"
    return "D"


def _source_id(row: Any, *, fallback: str | None = None) -> str | None:
    if not isinstance(row, dict):
        return fallback
    for key in ("id", "candidate_id", "row_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def _digest_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "kind",
        "anchor_type",
        "bbox_pt",
        "orientation",
        "symbol_prefix",
        "frame_refs",
        "value_text",
        "value_source",
        "unknown_reason",
        "consumer_allowed",
        "source",
        "source_id",
    }
    return {key: row.get(key) for key in sorted(keep)}


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
