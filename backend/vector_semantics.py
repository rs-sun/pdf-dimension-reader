"""Vector Semantic Graph Phase 0.

This module is intentionally audit-only. It extracts vector primitives, records
typed spatial relations, builds conservative physical components, and emits
proposal objects for review overlays. It must not create OCR targets, candidate
evidence, assembler scores, or final dimensions.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from statistics import median
from typing import Any

try:
    from config import Config
except Exception:  # pragma: no cover - import fallback for standalone tooling
    Config = None


SCHEMA_VERSION = "vector_semantics_phase0_v0"
EXTRACTOR_VERSION = "phase0.1"
STRONG_EDGE_TYPES = {"duplicate", "path_adjacency", "endpoint_endpoint"}
EDGE_TYPES = (
    "duplicate",
    "path_adjacency",
    "endpoint_endpoint",
    "endpoint_on_line",
    "intersection_cross",
    "collinear_overlap",
    "near_parallel_bundle",
    "region_neighbor",
)
PROPOSAL_TYPES = (
    "text_glyph_cluster",
    "hatching_bundle",
    "table_grid_candidate",
    "title_grid_candidate",
    "view_structure_candidate",
    "arrowhead_candidate",
    "dimension_line_candidate",
    "leader_candidate",
)


def build_vector_semantics(
    *,
    page,
    fitz_page=None,
    page_index: int = 0,
    doc_id: str | None = None,
    text_regions: list[dict[str, Any]] | None = None,
    frame_borders: list[dict[str, Any]] | None = None,
    table_cells: list[dict[str, Any]] | None = None,
    table_regions: list[dict[str, Any]] | None = None,
    view_boxes: list[dict[str, Any]] | None = None,
    export_primitives: bool | None = None,
    prefetched_fitz_lines: list[dict[str, Any]] | None = None,
    skip_pdfplumber_lines: bool | None = None,
) -> dict[str, Any]:
    """Build the Phase 0 audit graph for one page.

    The return value is pure debug data. Callers should never feed it back into
    OCR, table filtering, candidate construction, or assembly while Phase 0
    flags remain debug-only.
    """
    t_start = time.time()
    page_width = float(getattr(page, "width", 0.0) or 0.0)
    page_height = float(getattr(page, "height", 0.0) or 0.0)
    doc_id = doc_id or "document"
    if export_primitives is None:
        export_primitives = bool(
            getattr(Config, "VECTOR_SEMANTICS_EXPORT_PRIMITIVES", False)
            if Config is not None else False
        )

    timings: dict[str, float] = {}

    t0 = time.time()
    primitives = extract_primitives(
        page=page,
        fitz_page=fitz_page,
        page_index=page_index,
        doc_id=doc_id,
        text_regions=text_regions or [],
        prefetched_fitz_lines=prefetched_fitz_lines,
        skip_pdfplumber_lines=skip_pdfplumber_lines,
    )
    timings["primitive_extract_ms"] = _elapsed_ms(t0)

    t0 = time.time()
    _assign_duplicate_groups(primitives)
    edges, edge_counts = build_typed_edges(
        primitives=primitives,
        page_width=page_width,
        page_height=page_height,
        text_regions=text_regions or [],
        frame_borders=frame_borders or [],
        table_cells=table_cells or [],
        table_regions=table_regions or [],
    )
    timings["edge_build_ms"] = _elapsed_ms(t0)

    t0 = time.time()
    components, primitive_to_component = build_physical_components(
        primitives,
        edges,
        page_width=page_width,
        page_height=page_height,
    )
    for primitive in primitives:
        primitive["physical_component_id"] = primitive_to_component.get(primitive["id"])
    timings["component_build_ms"] = _elapsed_ms(t0)

    t0 = time.time()
    region_priors, region_prior_conflicts = build_region_priors(
        frame_borders=frame_borders or [],
        table_cells=table_cells or [],
        table_regions=table_regions or [],
        text_regions=text_regions or [],
        view_boxes=view_boxes or [],
    )
    _add_region_neighbor_edges(edges, edge_counts, primitives, region_priors)
    timings["region_prior_ms"] = _elapsed_ms(t0)

    t0 = time.time()
    objects = build_proposal_objects(
        primitives=primitives,
        components=components,
        primitive_to_component=primitive_to_component,
        text_regions=text_regions or [],
        frame_borders=frame_borders or [],
        table_cells=table_cells or [],
        table_regions=table_regions or [],
        region_prior_conflicts=region_prior_conflicts,
    )
    timings["proposal_build_ms"] = _elapsed_ms(t0)

    _finalize_edges(edges)
    edge_counts = Counter(edge["type"] for edge in edges)
    object_counts = Counter(obj["type"] for obj in objects)
    primitive_counts = Counter(primitive["source"] for primitive in primitives)
    largest_component = max(
        (component["primitive_count"] for component in components),
        default=0,
    )
    overlarge_count = sum(1 for component in components if component.get("overlarge_flag"))
    snap_values = [p.get("snap_tol", 0.0) for p in primitives if p.get("snap_tol")]

    vector_ms = _elapsed_ms(t_start)
    timings["vector_semantics_ms"] = vector_ms
    summary = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "enabled": True,
        "debug_only": bool(
            getattr(Config, "VECTOR_SEMANTICS_DEBUG_ONLY", True)
            if Config is not None else True
        ),
        "consumer_decisions": _consumer_no_ops(),
        "page_index": page_index,
        "page_size": {"w": page_width, "h": page_height},
        "primitive_count": len(primitives),
        "primitive_count_by_source": dict(sorted(primitive_counts.items())),
        "edge_count": len(edges),
        "edge_count_by_type": {edge_type: int(edge_counts.get(edge_type, 0)) for edge_type in EDGE_TYPES},
        "component_count": len(components),
        "largest_component_primitive_count": largest_component,
        "overlarge_component_count": overlarge_count,
        "object_count": len(objects),
        "object_count_by_type": {obj_type: int(object_counts.get(obj_type, 0)) for obj_type in PROPOSAL_TYPES},
        "region_prior_count": len(region_priors),
        "region_prior_conflict_count": len(region_prior_conflicts),
        "snap_tol_stats": _number_stats(snap_values),
        "timings": timings,
        "safety_flags": {
            "no_final_dimensions": True,
            "no_ocr_targets": True,
            "no_ocr_order_change": True,
            "no_candidate_change": True,
            "no_assembler_score_change": True,
            "no_table_skip_change": True,
            "view_boxes_truth_used": False,
            "vector_glyph_ocr_enabled": bool(
                getattr(Config, "VECTOR_GLYPH_OCR_ENABLED", False)
                if Config is not None else False
            ),
        },
    }

    result = {
        "summary": summary,
        "objects": objects,
        "region_priors": region_priors,
        "region_prior_conflicts": region_prior_conflicts,
        "edges": edges,
        "components": components,
    }
    if export_primitives:
        result["primitives"] = primitives
    else:
        result["primitives"] = []
        result["primitive_export_omitted"] = True
    return result


def extract_primitives(
    *,
    page,
    fitz_page=None,
    page_index: int,
    doc_id: str,
    text_regions: list[dict[str, Any]] | None = None,
    prefetched_fitz_lines: list[dict[str, Any]] | None = None,
    skip_pdfplumber_lines: bool | None = None,
) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    text_index = _BBoxCenterIndex(
        [
            {"id": f"text_region:{idx}", "bbox": bbox}
            for idx, region in enumerate(text_regions or [])
            for bbox in [_normalize_bbox(region)]
            if bbox
        ],
        cell_size=48.0,
    )
    if prefetched_fitz_lines is not None:
        primitives.extend(_extract_prefetched_fitz_lines(
            prefetched_fitz_lines,
            page_index,
            doc_id,
            text_index,
        ))
    elif fitz_page is not None:
        primitives.extend(_extract_fitz_primitives(fitz_page, page_index, doc_id, text_index))
    if skip_pdfplumber_lines is None:
        skip_pdfplumber_lines = prefetched_fitz_lines is not None
    primitives.extend(_extract_pdfplumber_primitives(
        page,
        page_index,
        doc_id,
        text_index,
        skip_lines=bool(skip_pdfplumber_lines),
    ))
    _finalize_primitives(primitives, page_index)
    return primitives


def _extract_prefetched_fitz_lines(
    lines: list[dict[str, Any]],
    page_index: int,
    doc_id: str,
    text_index,
) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    for idx, line in enumerate(lines or []):
        primitive = _make_line_primitive(
            doc_id=doc_id,
            page_index=page_index,
            source="fitz_prefetched",
            source_key=f"line:{idx}",
            kind="line",
            path_id=f"fitz_prefetched:{idx}",
            draw_id=idx,
            item_index=0,
            x0=_f(line.get("x0")),
            y0=_f(line.get("y0")),
            x1=_f(line.get("x1")),
            y1=_f(line.get("y1")),
            line_width=_f(line.get("lineWidth", line.get("linewidth")), 0.0),
            stroke_color=None,
            fill_color=None,
            dash_pattern=None,
            is_filled=False,
            is_closed_path=False,
            page_rotation=0,
        )
        if _keep_extracted_primitive(primitive, text_index):
            primitives.append(primitive)
    return primitives


def _extract_fitz_primitives(
    fitz_page,
    page_index: int,
    doc_id: str,
    text_index,
) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    try:
        drawings = fitz_page.get_drawings()
    except Exception:
        return primitives

    rotation = int(getattr(fitz_page, "rotation", 0) or 0)
    for draw_idx, drawing in enumerate(drawings or []):
        width = _f(drawing.get("width"), 0.0)
        path_id = f"fitz:{page_index}:{draw_idx}"
        stroke_color = _json_safe(drawing.get("color"))
        fill_color = _json_safe(drawing.get("fill"))
        dash_pattern = _json_safe(drawing.get("dashes"))
        is_filled = drawing.get("fill") is not None
        is_closed = bool(
            drawing.get("closePath")
            or drawing.get("close_path")
            or drawing.get("even_odd")
        )
        for item_idx, item in enumerate(drawing.get("items") or []):
            if not item:
                continue
            op = item[0]
            if op == "l" and len(item) >= 3:
                try:
                    p0, p1 = item[1], item[2]
                    x0, y0, x1, y1 = _f(p0.x), _f(p0.y), _f(p1.x), _f(p1.y)
                except Exception:
                    continue
                primitive = _make_line_primitive(
                    doc_id=doc_id,
                    page_index=page_index,
                    source="fitz",
                    source_key=f"d{draw_idx}:i{item_idx}",
                    kind="line",
                    path_id=path_id,
                    draw_id=draw_idx,
                    item_index=item_idx,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line_width=width,
                    stroke_color=stroke_color,
                    fill_color=fill_color,
                    dash_pattern=dash_pattern,
                    is_filled=is_filled,
                    is_closed_path=is_closed,
                    page_rotation=rotation,
                )
                if _keep_extracted_primitive(primitive, text_index):
                    primitives.append(primitive)
            elif op == "re" and len(item) >= 2:
                rect = item[1]
                try:
                    x0, y0, x1, y1 = _f(rect.x0), _f(rect.y0), _f(rect.x1), _f(rect.y1)
                except Exception:
                    continue
                for edge_idx, (a, b, c, d) in enumerate(_rect_edges(x0, y0, x1, y1)):
                    primitive = _make_line_primitive(
                        doc_id=doc_id,
                        page_index=page_index,
                        source="fitz",
                        source_key=f"d{draw_idx}:i{item_idx}:e{edge_idx}",
                        kind="rect_edge",
                        path_id=path_id,
                        draw_id=draw_idx,
                        item_index=(item_idx * 10) + edge_idx,
                        x0=a,
                        y0=b,
                        x1=c,
                        y1=d,
                        line_width=width,
                        stroke_color=stroke_color,
                        fill_color=fill_color,
                        dash_pattern=dash_pattern,
                        is_filled=is_filled,
                        is_closed_path=True,
                        page_rotation=rotation,
                    )
                    if _keep_extracted_primitive(primitive, text_index):
                        primitives.append(primitive)
            elif op == "c" and len(item) >= 5:
                pts = []
                ok = True
                for pt in item[1:5]:
                    try:
                        pts.append((_f(pt.x), _f(pt.y)))
                    except Exception:
                        ok = False
                        break
                if not ok:
                    continue
                primitive = _make_curve_primitive(
                    doc_id=doc_id,
                    page_index=page_index,
                    source="fitz",
                    source_key=f"d{draw_idx}:i{item_idx}",
                    path_id=path_id,
                    draw_id=draw_idx,
                    item_index=item_idx,
                    points=pts,
                    line_width=width,
                    stroke_color=stroke_color,
                    fill_color=fill_color,
                    dash_pattern=dash_pattern,
                    is_filled=is_filled,
                    is_closed_path=is_closed,
                    page_rotation=rotation,
                )
                if _keep_extracted_primitive(primitive, text_index):
                    primitives.append(primitive)
    return primitives


def _extract_pdfplumber_primitives(
    page,
    page_index: int,
    doc_id: str,
    text_index,
    *,
    skip_lines: bool = False,
) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    if not skip_lines:
        for idx, line in enumerate(getattr(page, "lines", []) or []):
            x0 = _f(line.get("x0"))
            y0 = _f(line.get("top", line.get("y0")))
            x1 = _f(line.get("x1"))
            y1 = _f(line.get("bottom", line.get("y1")))
            primitive = _make_line_primitive(
                doc_id=doc_id,
                page_index=page_index,
                source="pdfplumber_line",
                source_key=f"line:{idx}",
                kind="line",
                path_id=f"pdfplumber:line:{idx}",
                draw_id=idx,
                item_index=0,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line_width=_f(line.get("linewidth", line.get("lineWidth")), 0.0),
                stroke_color=_json_safe(line.get("stroking_color")),
                fill_color=None,
                dash_pattern=_json_safe(line.get("dash")),
                is_filled=False,
                is_closed_path=False,
                page_rotation=0,
            )
            if _keep_extracted_primitive(primitive, text_index):
                primitives.append(primitive)

    for idx, curve in enumerate(getattr(page, "curves", []) or []):
        pts = [(_f(x), _f(y)) for x, y in (curve.get("pts") or [])]
        if not pts:
            x0 = _f(curve.get("x0"))
            y0 = _f(curve.get("top", curve.get("y0")))
            x1 = _f(curve.get("x1"))
            y1 = _f(curve.get("bottom", curve.get("y1")))
            pts = [(x0, y0), (x1, y1)]
        primitive = _make_curve_primitive(
            doc_id=doc_id,
            page_index=page_index,
            source="pdfplumber_curve",
            source_key=f"curve:{idx}",
            path_id=f"pdfplumber:curve:{idx}",
            draw_id=idx,
            item_index=0,
            points=pts,
            line_width=_f(curve.get("linewidth", curve.get("lineWidth")), 0.0),
            stroke_color=_json_safe(curve.get("stroking_color")),
            fill_color=_json_safe(curve.get("non_stroking_color")),
            dash_pattern=_json_safe(curve.get("dash")),
            is_filled=bool(curve.get("fill")),
            is_closed_path=bool(curve.get("closepath")),
            page_rotation=0,
        )
        if _keep_extracted_primitive(primitive, text_index):
            primitives.append(primitive)

    for idx, rect in enumerate(getattr(page, "rects", []) or []):
        x0 = _f(rect.get("x0"))
        y0 = _f(rect.get("top", rect.get("y0")))
        x1 = _f(rect.get("x1"))
        y1 = _f(rect.get("bottom", rect.get("y1")))
        for edge_idx, (a, b, c, d) in enumerate(_rect_edges(x0, y0, x1, y1)):
            primitive = _make_line_primitive(
                doc_id=doc_id,
                page_index=page_index,
                source="pdfplumber_rect",
                source_key=f"rect:{idx}:e{edge_idx}",
                kind="rect_edge",
                path_id=f"pdfplumber:rect:{idx}",
                draw_id=idx,
                item_index=edge_idx,
                x0=a,
                y0=b,
                x1=c,
                y1=d,
                line_width=_f(rect.get("linewidth", rect.get("lineWidth")), 0.0),
                stroke_color=_json_safe(rect.get("stroking_color")),
                fill_color=_json_safe(rect.get("non_stroking_color")),
                dash_pattern=_json_safe(rect.get("dash")),
                is_filled=bool(rect.get("fill")),
                is_closed_path=True,
                page_rotation=0,
            )
            if _keep_extracted_primitive(primitive, text_index):
                primitives.append(primitive)
    return primitives


def _make_line_primitive(**kwargs) -> dict[str, Any]:
    x0, y0, x1, y1 = kwargs["x0"], kwargs["y0"], kwargs["x1"], kwargs["y1"]
    length = math.hypot(x1 - x0, y1 - y0)
    angle = _line_angle(x0, y0, x1, y1)
    bbox = _bbox_from_xyxy(x0, y0, x1, y1)
    primitive = {
        **kwargs,
        "bbox": bbox,
        "length": length,
        "angle": angle,
        "coord_space": "pdf_top_left",
        "extractor_version": EXTRACTOR_VERSION,
    }
    primitive["canonical_hash"] = _canonical_hash_for_primitive(primitive)
    primitive["snap_tol"] = _snap_tol(primitive)
    return primitive


def _make_curve_primitive(**kwargs) -> dict[str, Any]:
    points = kwargs.pop("points")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    length = _polyline_length(points)
    primitive = {
        **kwargs,
        "kind": "curve",
        "points": [[round(x, 3), round(y, 3)] for x, y in points],
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "bbox": _bbox_from_xyxy(x0, y0, x1, y1),
        "length": length,
        "angle": _line_angle(points[0][0], points[0][1], points[-1][0], points[-1][1]),
        "coord_space": "pdf_top_left",
        "extractor_version": EXTRACTOR_VERSION,
    }
    primitive["canonical_hash"] = _canonical_hash_for_primitive(primitive)
    primitive["snap_tol"] = _snap_tol(primitive)
    return primitive


def _keep_extracted_primitive(primitive: dict[str, Any], text_index) -> bool:
    if not text_index.contains_point(_bbox_center(primitive["bbox"])):
        return True
    if primitive.get("kind") == "curve":
        return False
    length = _f(primitive.get("length"), 0.0)
    width = _f(primitive.get("line_width"), 0.0)
    # Text regions are already represented by text_glyph_cluster proposals.
    # Keep long strokes that may be dimension/extension lines crossing text,
    # but drop small glyph strokes before graph construction.
    return length > 30.0 and width > 0.12


def _finalize_primitives(primitives: list[dict[str, Any]], page_index: int) -> None:
    for idx, primitive in enumerate(primitives):
        primitive["id"] = f"p{page_index:03d}_{idx:06d}_{primitive['canonical_hash'][:10]}"


def _assign_duplicate_groups(primitives: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for primitive in primitives:
        by_hash[primitive["canonical_hash"]].append(primitive)
    for hsh, group in by_hash.items():
        group_id = f"dup_{hsh[:12]}" if len(group) > 1 else None
        for primitive in group:
            primitive["dup_group_id"] = group_id


def build_typed_edges(
    *,
    primitives: list[dict[str, Any]],
    page_width: float,
    page_height: float,
    text_regions: list[dict[str, Any]],
    frame_borders: list[dict[str, Any]],
    table_cells: list[dict[str, Any]],
    table_regions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    counts: Counter = Counter()
    text_index = _BBoxCenterIndex(
        [
            {"id": f"text_region:{idx}", "bbox": bbox}
            for idx, region in enumerate(text_regions or [])
            for bbox in [_normalize_bbox(region)]
            if bbox
        ],
        cell_size=48.0,
    )

    def add_edge(edge_type: str, a: str, b: str, **attrs) -> None:
        if a == b:
            return
        left, right = sorted((a, b))
        key = (edge_type, left, right)
        if key in seen:
            return
        seen.add(key)
        edge = {
            "type": edge_type,
            "a": left,
            "b": right,
            "consumer_decision": "no-op",
        }
        edge.update(attrs)
        edges.append(edge)
        counts[edge_type] += 1

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for primitive in primitives:
        by_hash[primitive["canonical_hash"]].append(primitive)
    for group in by_hash.values():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=lambda p: p["id"])
        for a, b in zip(ordered, ordered[1:]):
            add_edge(
                "duplicate",
                a["id"],
                b["id"],
                canonical_hash=a["canonical_hash"],
                dup_group_id=a.get("dup_group_id"),
            )

    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for primitive in primitives:
        path_id = primitive.get("path_id")
        if path_id:
            by_path[path_id].append(primitive)
    for path_id, group in by_path.items():
        if len(group) <= 1:
            continue
        safe_group = [
            p for p in group
            if not _is_zero_width_glyph_stroke(p)
            and not _primitive_center_in_index(p, text_index)
        ]
        if len(safe_group) <= 1:
            continue
        ordered = sorted(safe_group, key=lambda p: (str(p.get("draw_id")), float(p.get("item_index", 0))))
        for a, b in zip(ordered, ordered[1:]):
            add_edge("path_adjacency", a["id"], b["id"], path_id=path_id)

    line_primitives = [
        p for p in primitives
        if _is_line_like(p)
        and not _is_zero_width_glyph_stroke(p)
        and not _primitive_center_in_index(p, text_index)
    ]
    _add_endpoint_edges(line_primitives, add_edge)
    _add_weak_line_edges(line_primitives, add_edge, page_width, page_height)

    return edges, counts


def _add_endpoint_edges(line_primitives: list[dict[str, Any]], add_edge) -> None:
    endpoints: list[tuple[float, float, str, str, float]] = []
    for primitive in line_primitives:
        endpoints.append((primitive["x0"], primitive["y0"], primitive["id"], "p0", primitive["snap_tol"]))
        endpoints.append((primitive["x1"], primitive["y1"], primitive["id"], "p1", primitive["snap_tol"]))
    if not endpoints:
        return
    cell_size = max(1.0, median([ep[4] for ep in endpoints]) * 2.0)
    buckets: dict[tuple[int, int], list[tuple[float, float, str, str, float]]] = defaultdict(list)
    for ep in endpoints:
        buckets[(int(math.floor(ep[0] / cell_size)), int(math.floor(ep[1] / cell_size)))].append(ep)

    for bx, by in list(buckets.keys()):
        bucket_eps = buckets[(bx, by)]
        nearby = []
        for nx in (bx - 1, bx, bx + 1):
            for ny in (by - 1, by, by + 1):
                nearby.extend(buckets.get((nx, ny), []))
        for ep_a in bucket_eps:
            for ep_b in nearby:
                if ep_a[2] >= ep_b[2]:
                    continue
                tol = max(ep_a[4], ep_b[4])
                distance = math.hypot(ep_a[0] - ep_b[0], ep_a[1] - ep_b[1])
                if distance <= tol:
                    add_edge(
                        "endpoint_endpoint",
                        ep_a[2],
                        ep_b[2],
                        distance=round(distance, 3),
                        tolerance=round(tol, 3),
                        endpoints=[ep_a[3], ep_b[3]],
                    )


def _add_weak_line_edges(
    line_primitives: list[dict[str, Any]],
    add_edge,
    page_width: float,
    page_height: float,
) -> None:
    pairs_seen: set[tuple[str, str]] = set()
    cell_size = 32.0
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for primitive in line_primitives:
        bbox = primitive["bbox"]
        x0 = int(math.floor(bbox["x"] / cell_size))
        x1 = int(math.floor((bbox["x"] + bbox["w"]) / cell_size))
        y0 = int(math.floor(bbox["y"] / cell_size))
        y1 = int(math.floor((bbox["y"] + bbox["h"]) / cell_size))
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                buckets[(gx, gy)].append(primitive)

    pair_budget = max(40000, min(120000, len(line_primitives) * 30))
    pair_count = 0
    for bucket in buckets.values():
        if len(bucket) > 80:
            bucket = sorted(bucket, key=lambda p: p.get("length", 0), reverse=True)[:80]
        for i, a in enumerate(bucket):
            for b in bucket[i + 1:]:
                left, right = sorted((a["id"], b["id"]))
                key = (left, right)
                if key in pairs_seen:
                    continue
                pairs_seen.add(key)
                pair_count += 1
                if pair_count > pair_budget:
                    return
                _classify_weak_line_pair(a, b, add_edge)


def _classify_weak_line_pair(a: dict[str, Any], b: dict[str, Any], add_edge) -> None:
    if _bbox_distance(a["bbox"], b["bbox"]) > 8.0:
        return
    for ex, ey, ename in ((a["x0"], a["y0"], "p0"), (a["x1"], a["y1"], "p1")):
        dist, t = _point_to_segment_distance(ex, ey, b["x0"], b["y0"], b["x1"], b["y1"])
        if 0.05 < t < 0.95 and dist <= max(0.5, a.get("snap_tol", 0.3), b.get("snap_tol", 0.3)):
            add_edge(
                "endpoint_on_line",
                a["id"],
                b["id"],
                endpoint=ename,
                distance=round(dist, 3),
            )
            break
    for ex, ey, ename in ((b["x0"], b["y0"], "p0"), (b["x1"], b["y1"], "p1")):
        dist, t = _point_to_segment_distance(ex, ey, a["x0"], a["y0"], a["x1"], a["y1"])
        if 0.05 < t < 0.95 and dist <= max(0.5, a.get("snap_tol", 0.3), b.get("snap_tol", 0.3)):
            add_edge(
                "endpoint_on_line",
                a["id"],
                b["id"],
                endpoint=ename,
                distance=round(dist, 3),
            )
            break

    if _segments_cross(a, b):
        add_edge("intersection_cross", a["id"], b["id"])

    angle_delta = _angle_delta(a.get("angle", 0.0), b.get("angle", 0.0))
    if angle_delta <= 3.0:
        line_dist = _line_to_line_distance(a, b)
        overlap = _projected_overlap(a, b)
        if line_dist <= 1.0 and overlap >= 3.0:
            add_edge(
                "collinear_overlap",
                a["id"],
                b["id"],
                distance=round(line_dist, 3),
                overlap=round(overlap, 3),
            )
        elif _midpoint_distance(a, b) <= 18.0 and min(a.get("length", 0), b.get("length", 0)) <= 90:
            add_edge(
                "near_parallel_bundle",
                a["id"],
                b["id"],
                distance=round(_midpoint_distance(a, b), 3),
                angle_delta=round(angle_delta, 3),
            )


def build_physical_components(
    primitives: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    dsu = _DSU([p["id"] for p in primitives])
    primitive_by_id = {p["id"]: p for p in primitives}
    for edge in edges:
        if edge.get("type") in STRONG_EDGE_TYPES:
            dsu.union(edge["a"], edge["b"])

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for primitive in primitives:
        groups[dsu.find(primitive["id"])].append(primitive)

    edge_counts_by_pair: dict[str, Counter] = defaultdict(Counter)
    root_to_component_id: dict[str, str] = {}
    ordered_roots = sorted(
        groups,
        key=lambda root: min(primitive_by_id[p["id"]].get("id", "") for p in groups[root]),
    )
    for idx, root in enumerate(ordered_roots):
        root_to_component_id[root] = f"cc_{idx:06d}"

    primitive_to_component: dict[str, str] = {}
    for root, component_id in root_to_component_id.items():
        for primitive in groups[root]:
            primitive_to_component[primitive["id"]] = component_id

    for edge in edges:
        ca = primitive_to_component.get(edge["a"])
        cb = primitive_to_component.get(edge["b"])
        if ca and cb and ca == cb:
            edge_counts_by_pair[ca][edge["type"]] += 1

    page_area = max(1.0, page_width * page_height)
    components: list[dict[str, Any]] = []
    for root in ordered_roots:
        group = groups[root]
        component_id = root_to_component_id[root]
        bbox = _union_bboxes([p["bbox"] for p in group])
        bbox_area = bbox["w"] * bbox["h"]
        overlarge = len(group) > 200 or bbox_area / page_area > 0.15
        components.append({
            "id": component_id,
            "primitive_ids": [p["id"] for p in group],
            "primitive_count": len(group),
            "bbox": bbox,
            "bbox_area_ratio": round(bbox_area / page_area, 6),
            "edge_count_by_type": dict(edge_counts_by_pair.get(component_id, {})),
            "physical_edge_types": sorted(STRONG_EDGE_TYPES),
            "overlarge_flag": bool(overlarge),
            "consumer_decision": "no-op",
        })
    return components, primitive_to_component


def build_region_priors(
    *,
    frame_borders: list[dict[str, Any]],
    table_cells: list[dict[str, Any]],
    table_regions: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    view_boxes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    priors: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    def add_prior(region_type: str, source: str, idx: int, raw: dict[str, Any], confidence: str) -> None:
        bbox = _normalize_bbox(raw)
        if not bbox:
            return
        priors.append({
            "id": f"{region_type}:{idx}",
            "type": region_type,
            "source": source,
            "bbox": bbox,
            "score": 1.0,
            "confidence": confidence,
            "reasons": [source],
            "conflicts": [],
            "consumer_policy": "audit_only_no_skip",
        })

    for idx, region in enumerate(frame_borders):
        source = str(region.get("source") or "frame_border")
        region_type = "title_block" if source == "title_block" else "frame_border"
        add_prior(region_type, source, idx, region, "medium")
    for idx, region in enumerate(table_cells):
        add_prior("table_cell", "rects_result.table_cells", idx, region, "medium")
    for idx, region in enumerate(table_regions):
        add_prior("table_region", "rects_result.table_regions_conflict_only", idx, region, "low")
    for idx, region in enumerate(text_regions):
        add_prior("text_region", "cluster_text_regions", idx, region, "medium")

    view_regions = [
        {
            "id": f"view_box:{idx}",
            "bbox": _normalize_bbox(region),
            "source": "view_boxes_conflict_only",
        }
        for idx, region in enumerate(view_boxes)
    ]
    view_regions = [r for r in view_regions if r["bbox"]]
    for prior in priors:
        if prior["type"] not in {"table_region", "table_cell", "title_block"}:
            continue
        for view in view_regions:
            overlap = _bbox_overlap_ratio(prior["bbox"], view["bbox"])
            if overlap <= 0.05:
                continue
            conflict = {
                "region_id": prior["id"],
                "region_type": prior["type"],
                "view_box_id": view["id"],
                "overlap_ratio": round(overlap, 4),
                "policy": "diagnostic_only_view_box_not_truth",
            }
            conflicts.append(conflict)
            prior["conflicts"].append(conflict)
    return priors, conflicts


def _add_region_neighbor_edges(
    edges: list[dict[str, Any]],
    edge_counts: Counter,
    primitives: list[dict[str, Any]],
    region_priors: list[dict[str, Any]],
) -> None:
    seen = {(edge["type"], edge["a"], edge["b"]) for edge in edges}

    def add(edge: dict[str, Any]) -> None:
        key = (edge["type"], edge["a"], edge["b"])
        if key in seen:
            return
        seen.add(key)
        edges.append(edge)
        edge_counts[edge["type"]] += 1

    useful_regions = [r for r in region_priors if r["type"] != "text_region"]
    text_index = _BBoxCenterIndex(
        [
            {"id": r["id"], "bbox": r["bbox"]}
            for r in region_priors
            if r["type"] == "text_region"
        ],
        cell_size=48.0,
    )
    for primitive in primitives:
        if _is_zero_width_glyph_stroke(primitive):
            continue
        if _primitive_center_in_index(primitive, text_index):
            continue
        center = _bbox_center(primitive["bbox"])
        for region in useful_regions:
            bbox = region["bbox"]
            if not _point_in_bbox(center, bbox) and _bbox_distance(primitive["bbox"], bbox) > 2.0:
                continue
            add({
                "type": "region_neighbor",
                "a": primitive["id"],
                "b": region["id"],
                "region_type": region["type"],
                "consumer_decision": "no-op",
            })


def build_proposal_objects(
    *,
    primitives: list[dict[str, Any]],
    components: list[dict[str, Any]],
    primitive_to_component: dict[str, str],
    text_regions: list[dict[str, Any]],
    frame_borders: list[dict[str, Any]],
    table_cells: list[dict[str, Any]],
    table_regions: list[dict[str, Any]],
    region_prior_conflicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    primitive_index = _BBoxCenterIndex(
        [{"id": p["id"], "bbox": p["bbox"]} for p in primitives],
        cell_size=16.0,
    )
    text_bboxes = [_normalize_bbox(r) for r in text_regions]
    text_bboxes = [b for b in text_bboxes if b]
    text_index = _BBoxCenterIndex(
        [{"id": f"text_region:{idx}", "bbox": bbox} for idx, bbox in enumerate(text_bboxes)],
        cell_size=48.0,
    )
    line_primitives = [
        p for p in primitives
        if _is_line_like(p)
        and not _is_zero_width_glyph_stroke(p)
        and not _primitive_center_in_index(p, text_index)
    ]
    conflict_by_region = defaultdict(list)
    for conflict in region_prior_conflicts:
        conflict_by_region[conflict.get("region_id")].append(conflict)

    def add_object(
        obj_type: str,
        bbox: dict[str, float],
        members: list[str] | None = None,
        score: float = 0.5,
        reasons: list[str] | None = None,
        related_region_ids: list[str] | None = None,
        related_text_region_ids: list[str] | None = None,
        conflicts: list[dict[str, Any]] | None = None,
    ) -> None:
        members = members or []
        related_components = sorted({
            primitive_to_component[mid]
            for mid in members
            if mid in primitive_to_component
        })
        objects.append({
            "id": f"so_{len(objects):06d}",
            "type": obj_type,
            "type_scores": {obj_type: round(float(score), 3)},
            "bbox": _round_bbox(bbox),
            "members": members,
            "related_component_ids": related_components,
            "related_region_ids": related_region_ids or [],
            "related_text_region_ids": related_text_region_ids or [],
            "score": round(float(score), 3),
            "reasons": reasons or [],
            "conflicts": conflicts or [],
            "consumer_decisions": _consumer_no_ops(),
            "safety_flags": _object_safety_flags(),
        })

    # Existing text region clusters are vector-derived audit proposals, not OCR
    # targets. Keeping them as objects gives the audit nearest-neighbor layer a
    # stable text-glyph proposal without changing OCR order.
    for idx, bbox in enumerate(text_bboxes):
        members = primitive_index.ids_in_bbox(bbox, limit=40)
        add_object(
            "text_glyph_cluster",
            bbox,
            members=members,
            score=0.55 + min(0.35, len(members) / 80.0),
            reasons=["cluster_text_regions_audit"],
            related_region_ids=[f"text_region:{idx}"],
            related_text_region_ids=[f"text_region:{idx}"],
        )

    for idx, region in enumerate(table_cells):
        bbox = _normalize_bbox(region)
        if bbox:
            add_object(
                "table_grid_candidate",
                bbox,
                members=primitive_index.ids_in_bbox(bbox, limit=80),
                score=0.5,
                reasons=["table_cell_prior_audit_only"],
                related_region_ids=[f"table_cell:{idx}"],
                conflicts=conflict_by_region.get(f"table_cell:{idx}", []),
            )
    for idx, region in enumerate(table_regions):
        bbox = _normalize_bbox(region)
        if bbox:
            add_object(
                "table_grid_candidate",
                bbox,
                members=primitive_index.ids_in_bbox(bbox, limit=120),
                score=0.45,
                reasons=["table_region_conflict_prior_audit_only"],
                related_region_ids=[f"table_region:{idx}"],
                conflicts=conflict_by_region.get(f"table_region:{idx}", []),
            )
    for idx, region in enumerate(frame_borders):
        bbox = _normalize_bbox(region)
        if not bbox:
            continue
        if str(region.get("source") or "") == "title_block":
            add_object(
                "title_grid_candidate",
                bbox,
                members=primitive_index.ids_in_bbox(bbox, limit=140),
                score=0.65,
                reasons=["title_block_prior_audit_only"],
                related_region_ids=[f"title_block:{idx}"],
                conflicts=conflict_by_region.get(f"title_block:{idx}", []),
            )
        else:
            add_object(
                "view_structure_candidate",
                bbox,
                members=primitive_index.ids_in_bbox(bbox, limit=160),
                score=0.45,
                reasons=["frame_border_prior_audit_only"],
                related_region_ids=[f"frame_border:{idx}"],
            )

    _add_hatching_objects(line_primitives, add_object)
    arrow_objects = _add_arrowhead_objects(line_primitives, add_object)
    _add_dimension_and_leader_objects(
        line_primitives,
        text_index,
        arrow_objects,
        add_object,
    )

    for component in components:
        if not component.get("overlarge_flag"):
            continue
        if component.get("primitive_count", 0) < 20:
            continue
        add_object(
            "view_structure_candidate",
            component["bbox"],
            members=component["primitive_ids"][:160],
            score=0.35,
            reasons=["overlarge_physical_component_audit_only"],
        )

    return objects


def _add_hatching_objects(line_primitives: list[dict[str, Any]], add_object) -> None:
    buckets: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for primitive in line_primitives:
        length = primitive.get("length", 0.0)
        if length < 4.0 or length > 100.0:
            continue
        angle_bucket = int(round((primitive.get("angle", 0.0) % 180) / 15.0))
        mx, my = _line_midpoint(primitive)
        buckets[(angle_bucket, int(mx // 120), int(my // 120))].append(primitive)
    for group in buckets.values():
        if len(group) < 6:
            continue
        bbox = _union_bboxes([p["bbox"] for p in group])
        add_object(
            "hatching_bundle",
            bbox,
            members=[p["id"] for p in group[:120]],
            score=min(0.85, 0.35 + len(group) / 40.0),
            reasons=["parallel_short_line_tile"],
        )


def _add_arrowhead_objects(line_primitives: list[dict[str, Any]], add_object) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for primitive in line_primitives:
        if primitive.get("length", 0.0) > 5.0:
            continue
        mx, my = _line_midpoint(primitive)
        buckets[(int(mx // 8), int(my // 8))].append(primitive)
    arrow_objects: list[dict[str, Any]] = []
    used: set[str] = set()
    for key, group in buckets.items():
        merged = list(group)
        for nx in (key[0] - 1, key[0], key[0] + 1):
            for ny in (key[1] - 1, key[1], key[1] + 1):
                if (nx, ny) != key:
                    merged.extend(buckets.get((nx, ny), []))
        merged = [p for p in merged if p["id"] not in used]
        if len(merged) < 3:
            continue
        bbox = _union_bboxes([p["bbox"] for p in merged])
        max_dim = max(bbox["w"], bbox["h"])
        min_dim = max(0.1, min(bbox["w"], bbox["h"]))
        if max_dim > 20.0 or max_dim / min_dim < 1.4:
            continue
        members = [p["id"] for p in merged[:60]]
        used.update(members)
        arrow_objects.append({"bbox": bbox, "members": members})
        add_object(
            "arrowhead_candidate",
            bbox,
            members=members,
            score=0.55,
            reasons=["short_line_cluster"],
        )
    return arrow_objects


def _add_dimension_and_leader_objects(
    line_primitives: list[dict[str, Any]],
    text_index,
    arrow_objects: list[dict[str, Any]],
    add_object,
) -> None:
    width_values = [p.get("line_width", 0.0) for p in line_primitives if p.get("line_width", 0.0) > 0]
    median_width = median(width_values) if width_values else 0.25
    max_dim_width = max(1.0, median_width * 3.0)
    for primitive in line_primitives:
        length = primitive.get("length", 0.0)
        if length < 15.0 or length > 600.0:
            continue
        if primitive.get("line_width", 0.0) > max_dim_width:
            continue
        orientation = _orientation(primitive)
        nearest_text_ids = _nearest_text_region_ids(primitive["bbox"], text_index, radius=45.0, limit=3)
        nearest_arrow = min(
            (_bbox_distance(primitive["bbox"], arrow["bbox"]) for arrow in arrow_objects),
            default=9999.0,
        )
        reasons = ["thin_vector_line"]
        score = 0.35
        if nearest_text_ids:
            reasons.append("near_text_glyph_cluster")
            score += 0.2
        if nearest_arrow <= 12.0:
            reasons.append("near_arrowhead_candidate")
            score += 0.2
        if orientation in {"H", "V"}:
            if not nearest_text_ids and nearest_arrow > 12.0 and length > 350.0:
                continue
            add_object(
                "dimension_line_candidate",
                _inflate_bbox(primitive["bbox"], 2.0),
                members=[primitive["id"]],
                score=min(0.9, score),
                reasons=reasons + [f"orientation_{orientation.lower()}"],
                related_text_region_ids=nearest_text_ids,
            )
        else:
            if not nearest_text_ids and nearest_arrow > 12.0:
                continue
            object_type = "leader_candidate"
            extra_reasons = ["orientation_diagonal"]
            if nearest_text_ids and nearest_arrow <= 12.0:
                object_type = "dimension_line_candidate"
                extra_reasons.append("diagonal_arrow_text_dimension_evidence")
            add_object(
                object_type,
                _inflate_bbox(primitive["bbox"], 2.0),
                members=[primitive["id"]],
                score=min(0.9, score + 0.05),
                reasons=reasons + extra_reasons,
                related_text_region_ids=nearest_text_ids,
            )


def _finalize_edges(edges: list[dict[str, Any]]) -> None:
    for idx, edge in enumerate(edges):
        edge["id"] = f"se_{idx:07d}"


class _BBoxCenterIndex:
    def __init__(self, rows: list[dict[str, Any]], cell_size: float = 32.0):
        self.cell_size = max(1.0, float(cell_size))
        self.rows = [row for row in rows if row.get("bbox")]
        self.buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            cx, cy = _bbox_center(row["bbox"])
            self.buckets[self._key(cx, cy)].append(row)

    def ids_in_bbox(self, bbox: dict[str, float], *, limit: int | None = None) -> list[str]:
        ids = []
        for row in self.rows_near_bbox(bbox, pad=0.0):
            if _point_in_bbox(_bbox_center(row["bbox"]), bbox):
                ids.append(row["id"])
                if limit is not None and len(ids) >= limit:
                    break
        return ids

    def rows_near_bbox(self, bbox: dict[str, float], *, pad: float) -> list[dict[str, Any]]:
        expanded = _inflate_bbox(bbox, pad)
        gx0 = int(math.floor(expanded["x"] / self.cell_size))
        gy0 = int(math.floor(expanded["y"] / self.cell_size))
        gx1 = int(math.floor((expanded["x"] + expanded["w"]) / self.cell_size))
        gy1 = int(math.floor((expanded["y"] + expanded["h"]) / self.cell_size))
        rows = []
        seen = set()
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                for row in self.buckets.get((gx, gy), []):
                    row_id = row["id"]
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    rows.append(row)
        return rows

    def contains_point(self, point: tuple[float, float]) -> bool:
        gx, gy = self._key(point[0], point[1])
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for row in self.buckets.get((nx, ny), []):
                    if _point_in_bbox(point, row["bbox"]):
                        return True
        return False

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size))


def _primitive_ids_in_bbox(
    primitives: list[dict[str, Any]],
    bbox: dict[str, float],
    *,
    limit: int,
) -> list[str]:
    members = []
    expanded = _inflate_bbox(bbox, 1.5)
    for primitive in primitives:
        if _point_in_bbox(_bbox_center(primitive["bbox"]), expanded):
            members.append(primitive["id"])
            if len(members) >= limit:
                break
    return members


def _nearest_text_region_ids(
    bbox: dict[str, float],
    text_index,
    *,
    radius: float,
    limit: int,
) -> list[str]:
    ranked = []
    for row in text_index.rows_near_bbox(bbox, pad=radius):
        text_bbox = row["bbox"]
        distance = _bbox_distance(bbox, text_bbox)
        if distance <= radius:
            ranked.append((distance, row["id"]))
    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked[:limit]]


def _consumer_no_ops() -> dict[str, str]:
    return {
        "table_skip": "no-op",
        "ocr_priority": "no-op",
        "strip_targets": "no-op",
        "assembler_evidence": "no-op",
        "final_dimension": "no-op",
    }


def _object_safety_flags() -> dict[str, bool]:
    return {
        "debug_only": True,
        "no_final_effect": True,
        "view_boxes_truth_used": False,
        "vector_glyph_ocr_enabled": bool(
            getattr(Config, "VECTOR_GLYPH_OCR_ENABLED", False)
            if Config is not None else False
        ),
    }


def _canonical_hash_for_primitive(primitive: dict[str, Any]) -> str:
    if _is_line_like(primitive):
        p0 = (_q(primitive["x0"]), _q(primitive["y0"]))
        p1 = (_q(primitive["x1"]), _q(primitive["y1"]))
        pts = sorted([p0, p1])
        payload = {"kind": "line", "points": pts}
    else:
        bbox = primitive["bbox"]
        payload = {
            "kind": primitive.get("kind", "primitive"),
            "bbox": [_q(bbox["x"]), _q(bbox["y"]), _q(bbox["w"]), _q(bbox["h"])],
        }
    return _stable_hash(payload)


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _number_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(float(v) for v in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(median(ordered), 4),
        "max": round(ordered[-1], 4),
    }


def _rect_edges(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float, float, float]]:
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return [
        (left, top, right, top),
        (right, top, right, bottom),
        (right, bottom, left, bottom),
        (left, bottom, left, top),
    ]


def _is_line_like(primitive: dict[str, Any]) -> bool:
    return primitive.get("kind") in {"line", "rect_edge"}


def _is_zero_width_glyph_stroke(primitive: dict[str, Any]) -> bool:
    return _is_line_like(primitive) and _f(primitive.get("line_width"), 0.0) <= 0.12


def _primitive_center_in_index(primitive: dict[str, Any], index) -> bool:
    return index.contains_point(_bbox_center(primitive["bbox"]))


def _line_angle(x0: float, y0: float, x1: float, y1: float) -> float:
    return (math.degrees(math.atan2(y1 - y0, x1 - x0)) + 180.0) % 180.0


def _angle_delta(a: float, b: float) -> float:
    delta = abs((a % 180.0) - (b % 180.0))
    return min(delta, 180.0 - delta)


def _orientation(primitive: dict[str, Any]) -> str:
    angle = primitive.get("angle", 0.0) % 180.0
    if angle <= 8.0 or angle >= 172.0:
        return "H"
    if 82.0 <= angle <= 98.0:
        return "V"
    return "D"


def _snap_tol(primitive: dict[str, Any]) -> float:
    width = _f(primitive.get("line_width"), 0.0)
    length = _f(primitive.get("length"), 0.0)
    if width <= 0.12 or length <= 3.0:
        return 0.2
    return max(0.25, min(2.0, width * 1.5))


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )


def _line_midpoint(primitive: dict[str, Any]) -> tuple[float, float]:
    return (
        (primitive["x0"] + primitive["x1"]) / 2.0,
        (primitive["y0"] + primitive["y1"]) / 2.0,
    )


def _midpoint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _line_midpoint(a)
    bx, by = _line_midpoint(b)
    return math.hypot(ax - bx, ay - by)


def _point_to_segment_distance(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float]:
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - x0, py - y0), 0.0
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
    qx, qy = x0 + t * dx, y0 + t * dy
    return math.hypot(px - qx, py - qy), t


def _segments_cross(a: dict[str, Any], b: dict[str, Any]) -> bool:
    p1 = (a["x0"], a["y0"])
    p2 = (a["x1"], a["y1"])
    p3 = (b["x0"], b["y0"])
    p4 = (b["x1"], b["y1"])
    if _bbox_distance(a["bbox"], b["bbox"]) > 0.5:
        return False
    denom = ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
    if abs(denom) < 1e-9:
        return False
    t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom
    u = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])) / denom
    return 0.05 < t < 0.95 and 0.05 < u < 0.95


def _line_to_line_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    d0, _ = _point_to_segment_distance(a["x0"], a["y0"], b["x0"], b["y0"], b["x1"], b["y1"])
    d1, _ = _point_to_segment_distance(a["x1"], a["y1"], b["x0"], b["y0"], b["x1"], b["y1"])
    d2, _ = _point_to_segment_distance(b["x0"], b["y0"], a["x0"], a["y0"], a["x1"], a["y1"])
    d3, _ = _point_to_segment_distance(b["x1"], b["y1"], a["x0"], a["y0"], a["x1"], a["y1"])
    return min(d0, d1, d2, d3)


def _projected_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax = a["x1"] - a["x0"]
    ay = a["y1"] - a["y0"]
    length = math.hypot(ax, ay)
    if length <= 1e-9:
        return 0.0
    ux, uy = ax / length, ay / length

    def project(px: float, py: float) -> float:
        return (px - a["x0"]) * ux + (py - a["y0"]) * uy

    a0, a1 = 0.0, length
    b0 = project(b["x0"], b["y0"])
    b1 = project(b["x1"], b["y1"])
    left = max(min(a0, a1), min(b0, b1))
    right = min(max(a0, a1), max(b0, b1))
    return max(0.0, right - left)


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> dict[str, float]:
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    return _round_bbox({"x": left, "y": top, "w": right - left, "h": bottom - top})


def _normalize_bbox(item: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    try:
        if {"x", "y", "w", "h"} <= set(item):
            return _round_bbox({
                "x": _f(item.get("x")),
                "y": _f(item.get("y")),
                "w": abs(_f(item.get("w"))),
                "h": abs(_f(item.get("h"))),
            })
        if {"x0", "y0", "x1", "y1"} <= set(item):
            return _bbox_from_xyxy(_f(item["x0"]), _f(item["y0"]), _f(item["x1"]), _f(item["y1"]))
        if {"x0", "top", "x1", "bottom"} <= set(item):
            return _bbox_from_xyxy(_f(item["x0"]), _f(item["top"]), _f(item["x1"]), _f(item["bottom"]))
        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            return _normalize_bbox(bbox)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return _bbox_from_xyxy(_f(bbox[0]), _f(bbox[1]), _f(bbox[2]), _f(bbox[3]))
    except Exception:
        return None
    return None


def _bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return bbox["x"] + bbox["w"] / 2.0, bbox["y"] + bbox["h"] / 2.0


def _point_in_bbox(point: tuple[float, float], bbox: dict[str, float]) -> bool:
    x, y = point
    return bbox["x"] <= x <= bbox["x"] + bbox["w"] and bbox["y"] <= y <= bbox["y"] + bbox["h"]


def _inflate_bbox(bbox: dict[str, float], pad: float) -> dict[str, float]:
    return _round_bbox({
        "x": bbox["x"] - pad,
        "y": bbox["y"] - pad,
        "w": bbox["w"] + pad * 2.0,
        "h": bbox["h"] + pad * 2.0,
    })


def _union_bboxes(bboxes: list[dict[str, float]]) -> dict[str, float]:
    if not bboxes:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    x0 = min(b["x"] for b in bboxes)
    y0 = min(b["y"] for b in bboxes)
    x1 = max(b["x"] + b["w"] for b in bboxes)
    y1 = max(b["y"] + b["h"] for b in bboxes)
    return _round_bbox({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})


def _bbox_distance(a: dict[str, float], b: dict[str, float]) -> float:
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = bx0 + b["w"], by0 + b["h"]
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _bbox_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    ix0 = max(a["x"], b["x"])
    iy0 = max(a["y"], b["y"])
    ix1 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy1 = min(a["y"] + a["h"], b["y"] + b["h"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / max(1e-6, a["w"] * a["h"])


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(float(bbox["x"]), 3),
        "y": round(float(bbox["y"]), 3),
        "w": round(float(bbox["w"]), 3),
        "h": round(float(bbox["h"]), 3),
    }


def _q(value: float, quantum: float = 0.1) -> float:
    return round(round(float(value) / quantum) * quantum, 3)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except Exception:
        return str(value)


def _elapsed_ms(t0: float) -> int:
    return int(round((time.time() - t0) * 1000))


class _DSU:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
