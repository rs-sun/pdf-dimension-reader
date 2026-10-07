"""DimensionCandidate v0 builder.

This module is intentionally small: it turns existing vector evidence into a
lightweight, auditable candidate layer without changing OCR or assembler
behavior.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any


SCHEMA_VERSION = "candidate_v0"
PIPELINE_VERSION = "candidate_v0.1.0"

REJECT_REASONS = {
    "capsule_inside_table_without_text_evidence",
    "capsule_geometry_noise",
    "capsule_without_text_evidence",
    "datum_feature_too_small",
    "duplicate_candidate",
    "dimension_text_fragment_too_small",
    "dimension_text_border_grid",
    "dimension_text_capsule_overmerge",
    "dimension_text_dense_vector_context_noise",
    "dimension_text_top_short_fragment",
    "dimension_text_unstable_fragment_group",
    "dimension_text_seeded_overmerge",
    "dimension_text_weak_non_seed_phrase",
    "dimension_text_review_size_floor",
    "dimension_diameter_review_size_floor",
    "gdt_fragment_too_narrow",
    "gdt_fragment_too_short",
    "gdt_without_text_evidence",
    "gdt_review_size_floor",
    "inside_table_region",
    "inside_title_block",
    "inside_title_block_with_dimension_evidence",
    "leader_without_text_evidence",
    "low_confidence_gdt_frame",
    "low_priority_gdt_fragment",
    "low_priority_noise",
    "no_object_evidence",
    "invalid_geometry",
    "outside_page",
    "merged_into_parent",
    "gold_scope_excluded",
    "dimline_without_text_evidence",
    "debug_only",
}

SOFT_QUALITY_REASONS = {
    "datum_feature_too_small",
    "dimension_table_text_weak_evidence",
    "dimension_text_border_grid",
    "dimension_text_capsule_overmerge",
    "dimension_text_seeded_overmerge",
    "dimension_text_weak_non_seed_phrase",
    "dimension_text_review_size_floor",
    "dimension_diameter_review_size_floor",
    "gdt_review_size_floor",
    "inside_title_block_with_dimension_evidence",
}

_SOURCE_LEVEL = "pipeline_seed"
_SPATIAL_CELL_SIZE = 64.0
_MAX_SPATIAL_CELLS_PER_BOX = 4096


def _spatial_cells_for_bbox(
    bbox: dict[str, float],
) -> tuple[tuple[int, int], ...] | None:
    try:
        x0 = float(bbox["x"])
        y0 = float(bbox["y"])
        x1 = x0 + float(bbox["w"])
        y1 = y0 + float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    if x1 < x0 or y1 < y0:
        return None
    gx0 = math.floor(x0 / _SPATIAL_CELL_SIZE)
    gy0 = math.floor(y0 / _SPATIAL_CELL_SIZE)
    gx1 = math.floor(x1 / _SPATIAL_CELL_SIZE)
    gy1 = math.floor(y1 / _SPATIAL_CELL_SIZE)
    cell_count = (gx1 - gx0 + 1) * (gy1 - gy0 + 1)
    if cell_count > _MAX_SPATIAL_CELLS_PER_BOX:
        return None
    return tuple(
        (gx, gy)
        for gx in range(gx0, gx1 + 1)
        for gy in range(gy0, gy1 + 1)
    )


class _PreparedVectorContextSegments:
    """Request-scoped immutable broad phase for vector density queries."""

    def __init__(self, segments: list[dict[str, Any]]) -> None:
        rows: list[
            tuple[
                int,
                tuple[float, float],
                tuple[float, float],
                float,
                dict[str, float],
            ]
        ] = []
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        overflow_indices: list[int] = []
        for ordinal, segment in enumerate(segments):
            points = _segment_points(segment)
            if points is None:
                continue
            p0, p1 = points
            full_length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if full_length <= 0.0:
                continue
            segment_bbox = _segment_bbox(p0, p1)
            row_index = len(rows)
            rows.append((ordinal, p0, p1, full_length, segment_bbox))
            occupied_cells = _spatial_cells_for_bbox(segment_bbox)
            if occupied_cells is None:
                overflow_indices.append(row_index)
                continue
            for cell in occupied_cells:
                cells[cell].append(row_index)
        self._rows = tuple(rows)
        self._cells = {
            cell: tuple(indices)
            for cell, indices in cells.items()
        }
        self._overflow_indices = tuple(overflow_indices)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def query(
        self,
        bbox: dict[str, float],
    ) -> tuple[
        tuple[
            int,
            tuple[float, float],
            tuple[float, float],
            float,
            dict[str, float],
        ],
        ...,
    ]:
        occupied_cells = _spatial_cells_for_bbox(bbox)
        if occupied_cells is None:
            return self._rows
        row_indices = set(self._overflow_indices)
        for cell in occupied_cells:
            row_indices.update(self._cells.get(cell, ()))
        return tuple(self._rows[index] for index in sorted(row_indices))


def normalize_bbox(item: Any, pad: float = 0.0) -> dict[str, float] | None:
    """Return a normalized ``{x,y,w,h}`` bbox or None.

    Supported inputs include existing repo bbox dicts, xyxy tuples, direct
    x/y/w/h dicts, start/end line dicts, and PyMuPDF Rect-like objects.
    """
    if item is None:
        return None

    if isinstance(item, dict) and "bbox" in item:
        bbox = normalize_bbox(item.get("bbox"), pad=0.0)
        return _pad_bbox(bbox, pad) if bbox else None

    if isinstance(item, dict):
        keys = set(item)
        if {"x", "y", "w", "h"} <= keys:
            try:
                x0 = float(item.get("x") or 0.0)
                y0 = float(item.get("y") or 0.0)
                x1 = x0 + float(item.get("w") or 0.0)
                y1 = y0 + float(item.get("h") or 0.0)
            except Exception:
                return None
            return _pad_bbox(_xyxy_to_bbox(x0, y0, x1, y1), pad)
        if {"x0", "y0", "x1", "y1"} <= keys:
            try:
                return _pad_bbox(_xyxy_to_bbox(
                    float(item.get("x0") or 0.0),
                    float(item.get("y0") or 0.0),
                    float(item.get("x1") or 0.0),
                    float(item.get("y1") or 0.0),
                ), pad)
            except Exception:
                return None
        if {"start", "end"} <= keys:
            try:
                sx, sy = item.get("start") or (0.0, 0.0)
                ex, ey = item.get("end") or (0.0, 0.0)
                line_pad = max(pad, 2.0)
                return _xyxy_to_bbox(
                    float(sx) - line_pad,
                    float(sy) - line_pad,
                    float(ex) + line_pad,
                    float(ey) + line_pad,
                )
            except Exception:
                return None
        if {"cx", "cy", "w", "h"} <= keys:
            try:
                cx = float(item.get("cx") or 0.0)
                cy = float(item.get("cy") or 0.0)
                w = float(item.get("w") or 0.0)
                h = float(item.get("h") or 0.0)
                return _pad_bbox(_xyxy_to_bbox(
                    cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
                ), pad)
            except Exception:
                return None

    if isinstance(item, (list, tuple)) and len(item) >= 4:
        try:
            return _pad_bbox(_xyxy_to_bbox(
                float(item[0] or 0.0),
                float(item[1] or 0.0),
                float(item[2] or 0.0),
                float(item[3] or 0.0),
            ), pad)
        except Exception:
            return None

    if all(hasattr(item, attr) for attr in ("x0", "y0", "x1", "y1")):
        try:
            return _pad_bbox(_xyxy_to_bbox(
                float(item.x0), float(item.y0), float(item.x1), float(item.y1)
            ), pad)
        except Exception:
            return None

    return None


def build_dimension_candidates(
    *,
    page_index: int,
    page_width: float,
    page_height: float,
    text_regions: list[dict[str, Any]] | None = None,
    pdf_text_words: list[dict[str, Any]] | None = None,
    capsule_candidates: list[dict[str, Any]] | None = None,
    gdt_frames_all: list[dict[str, Any]] | None = None,
    datum_candidates: list[dict[str, Any]] | None = None,
    detected_dim_lines: list[dict[str, Any]] | None = None,
    detected_leader_lines: list[dict[str, Any]] | None = None,
    diameter_glyphs: list[dict[str, Any]] | None = None,
    table_cells: list[dict[str, Any]] | None = None,
    table_regions: list[dict[str, Any]] | None = None,
    ocr_table_regions: list[dict[str, Any]] | None = None,
    frame_borders: list[dict[str, Any]] | None = None,
    vector_context_segments: list[dict[str, Any]] | None = None,
    vector_context_density_threshold: float | None = None,
    vector_context_density_pad: float = 45.0,
    vector_context_density_inner_pad: float = 3.0,
    include_vector_text_phrase_candidates: bool = True,
    include_text_region_fallback: bool = False,
    include_table_text_candidates: bool = True,
) -> dict[str, Any]:
    """Build active candidates, dropped candidates, and v0 debug stats."""
    builder = _CandidateBuilder(
        page_index=page_index,
        page_width=float(page_width or 0.0),
        page_height=float(page_height or 0.0),
        table_regions=list(table_cells or []) + list(table_regions or []),
        table_region_count=len(table_regions or []),
        table_cell_count=len(table_cells or []),
        ocr_table_region_count=len(ocr_table_regions or []),
        title_regions=[
            fb for fb in (frame_borders or [])
            if str(fb.get("source") or "") == "title_block"
        ],
        vector_context_segments=vector_context_segments or [],
        vector_context_density_threshold=vector_context_density_threshold,
        vector_context_density_pad=vector_context_density_pad,
        vector_context_density_inner_pad=vector_context_density_inner_pad,
    )

    raw_text_items = list(text_regions or [])
    dot_seed_items: list[dict[str, Any]] = []
    text_items: list[dict[str, Any]] = []
    # From here on, text_region source-node indices refer to this filtered list;
    # dot seed-only regions are carried separately and must not become candidates alone.
    for original_idx, region in enumerate(raw_text_items):
        if _is_dot_seed_region(region):
            seeded = dict(region)
            seeded["_text_region_index"] = original_idx
            dot_seed_items.append(seeded)
        else:
            text_items.append(region)
    pdf_word_items = list(pdf_text_words or [])
    used_text_region_indices: set[int] = set()

    for idx, frame in enumerate(gdt_frames_all or []):
        frame_bbox = normalize_bbox(frame)
        gdt_text_hits = _count_text_regions_inside(frame_bbox, text_items)
        builder.add_candidate(
            cand_type="gdt",
            subtype="feature_control_frame",
            bbox=frame_bbox,
            source_nodes=_gdt_source_nodes(frame, idx, text_region_hits=gdt_text_hits),
            priority_hint="high",
            oriented_quad=_gdt_oriented_quad(frame),
        )

    for idx, datum in enumerate(datum_candidates or []):
        builder.add_candidate(
            cand_type="datum",
            subtype="datum_feature",
            bbox=normalize_bbox(datum),
            source_nodes=[_source_node("datum_box", idx, "pdf_analyzer")],
            priority_hint="high",
        )

    for idx, capsule in enumerate(capsule_candidates or []):
        builder.add_candidate(
            cand_type="dimension",
            subtype="unknown",
            bbox=normalize_bbox(capsule),
            source_nodes=[_source_node("capsule", idx, "pdf_analyzer")],
            priority_hint="high",
            oriented_quad=_capsule_oriented_quad(capsule),
        )

    for group_idx, group in enumerate(_capsule_vector_text_phrase_groups(
        capsule_candidates or [],
        text_items,
    )):
        for region_idx in group["indices"]:
            used_text_region_indices.add(region_idx)
        group_node = _source_node("vector_text_group", group_idx, "pdf_analyzer")
        if group.get("orientation") in {"H", "V"}:
            group_node["orientation"] = group["orientation"]
        builder.add_candidate(
            cand_type="dimension",
            subtype="vector_text_phrase",
            bbox=group["bbox"],
            source_nodes=[
                _source_node("capsule", group["capsule_index"], "pdf_analyzer"),
                group_node,
                *[
                    _source_node("text_region", region_idx, "pdf_analyzer")
                    for region_idx in group["indices"][:6]
                ],
                *_numeric_seed_source_nodes(text_items, group["indices"]),
            ],
            priority_hint=group["priority_hint"],
            oriented_quad=_capsule_oriented_quad(
                (capsule_candidates or [])[group["capsule_index"]]
            ),
        )

    for idx, dimline in enumerate(detected_dim_lines or []):
        dim_bbox = _dimline_bbox(dimline)
        nearest_text = _nearest_text_region(dim_bbox, dimline, text_items)
        candidate_bbox = dim_bbox
        dimline_node = _source_node("dimension_line", idx, "arrow_detector")
        dimline_orientation = str(dimline.get("orientation") or "").upper()
        if dimline_orientation in {"H", "V", "D"}:
            dimline_node["orientation"] = dimline_orientation
            dimline_angle = _dimline_angle(dimline)
            if dimline_angle is not None:
                dimline_node["angle_deg"] = round(dimline_angle, 3)
        nodes = [dimline_node]
        subtype = "unknown"
        priority = "low"
        if nearest_text:
            nearest_bbox = normalize_bbox(nearest_text["region"])
            if nearest_bbox and builder.is_inside_table(nearest_bbox):
                nearest_text = None
            else:
                used_text_region_indices.add(nearest_text["index"])
                candidate_bbox = nearest_bbox
                nodes.append(_source_node(
                    "text_region", nearest_text["index"], "pdf_analyzer"
                ))
                subtype = "linear"
                priority = "high" if _dimline_has_arrowheads(dimline) else "normal"
        builder.add_candidate(
            cand_type="dimension",
            subtype=subtype,
            bbox=candidate_bbox,
            source_nodes=nodes,
            priority_hint=priority,
            oriented_quad=_dimline_oriented_quad(dimline, candidate_bbox),
        )

    for idx, leader in enumerate(detected_leader_lines or []):
        leader_bbox = _dimline_bbox(leader)
        nearest_text = _nearest_text_region(leader_bbox, leader, text_items)
        candidate_bbox = leader_bbox
        leader_node = _line_source_node("leader_line", idx, leader)
        nodes = [leader_node]
        if nearest_text:
            nearest_bbox = normalize_bbox(nearest_text["region"])
            if nearest_bbox and not builder.is_inside_table(nearest_bbox):
                used_text_region_indices.add(nearest_text["index"])
                candidate_bbox = nearest_bbox
                nodes.append(_source_node(
                    "text_region", nearest_text["index"], "pdf_analyzer"
                ))
        builder.add_candidate(
            cand_type="dimension",
            subtype="leader",
            bbox=candidate_bbox,
            source_nodes=nodes,
            priority_hint="low",
        )

    for idx, glyph in enumerate(diameter_glyphs or []):
        glyph_bbox = normalize_bbox(glyph, pad=2.0)
        nearest_text = _nearest_text_region(glyph_bbox, None, text_items, max_dist=70.0)
        candidate_bbox = glyph_bbox
        nodes = [_source_node("diameter_glyph", idx, "pdf_analyzer")]
        priority = "low"
        if nearest_text:
            used_text_region_indices.add(nearest_text["index"])
            candidate_bbox = normalize_bbox(nearest_text["region"])
            nodes.append(_source_node(
                "text_region", nearest_text["index"], "pdf_analyzer"
            ))
            priority = "high"
        builder.add_candidate(
            cand_type="dimension",
            subtype="diameter",
            bbox=candidate_bbox,
            source_nodes=nodes,
            priority_hint=priority,
        )

    for idx, word in enumerate(pdf_word_items):
        text = str(word.get("text") or "").strip()
        if not _is_dimension_like_pdf_text_word(text):
            continue
        builder.add_candidate(
            cand_type="dimension",
            subtype="pdf_text",
            bbox=normalize_bbox(word, pad=1.0),
            source_nodes=[_source_node("pdf_text_word", idx, "pdf_text")],
            priority_hint="normal",
            text=text,
        )

    if include_vector_text_phrase_candidates:
        vertical_phrase_groups = _rotated_vertical_text_phrase_groups(
            text_items,
            used_indices=used_text_region_indices,
            page_width=builder.page_width,
        )
        for group_idx, group in enumerate(vertical_phrase_groups):
            group_node = _source_node("vector_text_group", group_idx, "pdf_analyzer")
            group_node["orientation"] = "V"
            group_node["scan_pass"] = "rotated_90"
            accepted = builder.add_candidate(
                cand_type="dimension",
                subtype="vector_text_phrase",
                bbox=group["bbox"],
                source_nodes=[
                    group_node,
                    *[
                        _source_node("text_region", region_idx, "pdf_analyzer")
                        for region_idx in group["indices"][:6]
                    ],
                    *_numeric_seed_source_nodes(text_items, group["indices"]),
                ],
                priority_hint=group["priority_hint"],
                oriented_quad=group.get("oriented_quad"),
            )
            if accepted:
                for region_idx in group["indices"]:
                    used_text_region_indices.add(region_idx)

        phrase_groups = _vector_text_phrase_groups(
            text_items,
            used_indices=used_text_region_indices,
            dot_seed_regions=dot_seed_items,
        )
        for group_idx, group in enumerate(phrase_groups):
            for region_idx in group["indices"]:
                used_text_region_indices.add(region_idx)
            group_node = _source_node("vector_text_group", group_idx, "pdf_analyzer")
            if group.get("angle_deg") is not None:
                group_node["orientation"] = "D"
                group_node["angle_deg"] = group["angle_deg"]
            elif group.get("orientation") == "V":
                group_node["orientation"] = "V"
            builder.add_candidate(
                cand_type="dimension",
                subtype="vector_text_phrase",
                bbox=group["bbox"],
                source_nodes=[
                    group_node,
                    *[
                        _source_node("text_region", region_idx, "pdf_analyzer")
                        for region_idx in group["indices"][:6]
                    ],
                    *_numeric_seed_source_nodes(text_items, group["indices"]),
                    *[
                        _source_node("numeric_symbol_seed", region_idx, "pdf_analyzer")
                        for region_idx in group.get("dot_seed_indices", [])[:6]
                    ],
                ],
                priority_hint=group["priority_hint"],
                oriented_quad=group.get("oriented_quad"),
            )

    if include_table_text_candidates:
        for idx, region in enumerate(text_items):
            if idx in used_text_region_indices:
                continue
            bbox = normalize_bbox(region)
            if not bbox or not builder.is_inside_table(bbox):
                continue
            if not _is_plausible_table_text_region(bbox):
                continue
            builder.add_candidate(
                cand_type="dimension",
                subtype="table_text",
                bbox=bbox,
                source_nodes=[
                    _source_node("text_region", idx, "pdf_analyzer"),
                    _source_node("table_text_region", idx, "pdf_analyzer"),
                    *_numeric_seed_source_nodes(text_items, [idx]),
                ],
                priority_hint="normal" if region.get("confidence") == "high" else "low",
            )

    if include_text_region_fallback:
        for idx, region in enumerate(text_items):
            if idx in used_text_region_indices:
                continue
            builder.add_candidate(
                cand_type="dimension",
                subtype="text_region",
                bbox=normalize_bbox(region),
                source_nodes=[_source_node("text_region", idx, "pdf_analyzer")],
                priority_hint="normal" if region.get("confidence") == "high" else "low",
            )

    return builder.finish()


class _CandidateBuilder:
    def __init__(
        self,
        *,
        page_index: int,
        page_width: float,
        page_height: float,
        table_regions: list[dict[str, Any]],
        title_regions: list[dict[str, Any]],
        table_region_count: int = 0,
        table_cell_count: int = 0,
        ocr_table_region_count: int = 0,
        vector_context_segments: list[dict[str, Any]] | None = None,
        vector_context_density_threshold: float | None = None,
        vector_context_density_pad: float = 45.0,
        vector_context_density_inner_pad: float = 3.0,
    ) -> None:
        self.page_index = int(page_index)
        self.page_width = page_width
        self.page_height = page_height
        self.table_regions = [b for b in (normalize_bbox(r) for r in table_regions) if b]
        self.table_region_count = int(table_region_count)
        self.table_cell_count = int(table_cell_count)
        self.ocr_table_region_count = int(ocr_table_region_count)
        self.title_regions = [
            _expand_title_region(b, page_width=page_width, page_height=page_height)
            for b in (normalize_bbox(r) for r in title_regions)
            if b
        ]
        self.vector_context_segments = list(vector_context_segments or [])
        self.prepared_vector_context_segments = _PreparedVectorContextSegments(
            self.vector_context_segments
        )
        self.vector_context_density_threshold = (
            float(vector_context_density_threshold)
            if vector_context_density_threshold is not None
            else None
        )
        self.vector_context_density_pad = float(vector_context_density_pad or 0.0)
        self.vector_context_density_inner_pad = float(
            vector_context_density_inner_pad or 0.0
        )
        self._next_id = 1
        self.candidates: list[dict[str, Any]] = []
        self.drop_ledger: list[dict[str, Any]] = []

    def add_candidate(
        self,
        *,
        cand_type: str,
        subtype: str,
        bbox: dict[str, float] | None,
        source_nodes: list[dict[str, str]],
        priority_hint: str,
        text: str | None = None,
        oriented_quad: dict[str, Any] | None = None,
    ) -> bool:
        candidate_id = self._candidate_id()
        bbox = bbox or {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        inside_table = _center_inside_any(bbox, self.table_regions)
        inside_title = _center_inside_any(bbox, self.title_regions)
        vector_context = self._vector_context_metrics(
            bbox,
            cand_type=cand_type,
            subtype=subtype,
        )
        reject_reason = self._reject_reason(
            bbox,
            source_nodes,
            cand_type=cand_type,
            subtype=subtype,
            priority_hint=priority_hint,
            inside_table=inside_table,
            inside_title=inside_title,
        )
        quality_warnings: list[str] = []
        if reject_reason in SOFT_QUALITY_REASONS:
            quality_warnings.append(reject_reason)
            reject_reason = None
            priority_hint = "low"
        if not reject_reason and vector_context and vector_context.get("dense_vector_context"):
            if _is_dense_vector_context_noise(source_nodes):
                reject_reason = "dimension_text_dense_vector_context_noise"
            else:
                quality_warnings.append("dimension_text_dense_vector_context")
                priority_hint = "low"
        if inside_table and not reject_reason:
            priority_hint = "low"
        candidate = self._make_candidate(
            candidate_id=candidate_id,
            cand_type=cand_type,
            subtype=subtype,
            bbox=bbox,
            source_nodes=source_nodes,
            priority_hint=priority_hint,
            reject_reason=reject_reason,
            inside_table=inside_table,
            inside_title=inside_title,
            text=text,
            vector_context=vector_context,
            quality_warnings=quality_warnings,
            oriented_quad=oriented_quad,
        )

        if reject_reason:
            self.drop_ledger.append(candidate)
            return False

        duplicate_of = self._find_duplicate(candidate)
        if duplicate_of:
            candidate["reject_reason"] = "duplicate_candidate"
            candidate["provenance"]["duplicate_of"] = duplicate_of
            self.drop_ledger.append(candidate)
            return False

        self.candidates.append(candidate)
        return True

    def is_inside_table(self, bbox: dict[str, float]) -> bool:
        return _center_inside_any(bbox, self.table_regions)

    def finish(self) -> dict[str, Any]:
        self._drop_nested_duplicates()
        source_counts: Counter[str] = Counter()
        for candidate in self.candidates:
            for node in candidate.get("source_nodes") or []:
                source_counts[node.get("kind") or "unknown"] += 1
        reject_counts = Counter(
            c.get("reject_reason") or "missing_reason"
            for c in self.drop_ledger
        )
        type_counts = Counter(c.get("type") for c in self.candidates)
        subtype_counts = Counter(c.get("subtype") for c in self.candidates)
        inside_table_active_count = sum(
            1
            for c in self.candidates
            if (c.get("evidence_summary") or {}).get("inside_table")
        )
        drops_with_reason = sum(1 for c in self.drop_ledger if c.get("reject_reason"))
        drop_count = len(self.drop_ledger)
        coverage = 1.0 if drop_count == 0 else drops_with_reason / drop_count
        return {
            "dimension_candidates": self.candidates,
            "candidate_drop_ledger": self.drop_ledger,
            "candidate_debug_stats": {
                "schema_version": SCHEMA_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "enabled": True,
                "page_index": self.page_index,
                "active_count": len(self.candidates),
                "drop_count": drop_count,
                "drop_reason_coverage": round(coverage, 6),
                "source_counts": dict(sorted(source_counts.items())),
                "type_counts": dict(sorted(type_counts.items())),
                "subtype_counts": dict(sorted(subtype_counts.items())),
                "reject_reason_counts": dict(sorted(reject_counts.items())),
                "inside_table_active_count": inside_table_active_count,
                "table_region_count": self.table_region_count,
                "table_cell_count": self.table_cell_count,
                "ocr_table_region_count": self.ocr_table_region_count,
            },
        }

    def _candidate_id(self) -> str:
        candidate_id = f"p{self.page_index + 1:03d}_cand_{self._next_id:06d}"
        self._next_id += 1
        return candidate_id

    def _make_candidate(
        self,
        *,
        candidate_id: str,
        cand_type: str,
        subtype: str,
        bbox: dict[str, float],
        source_nodes: list[dict[str, str]],
        priority_hint: str,
        reject_reason: str | None,
        inside_table: bool,
        inside_title: bool,
        text: str | None,
        vector_context: dict[str, Any] | None = None,
        quality_warnings: list[str] | None = None,
        oriented_quad: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence = _evidence_summary(
            source_nodes,
            inside_table=inside_table,
            inside_title=inside_title,
        )
        if vector_context:
            evidence["has_vector_context"] = True
            evidence["vector_context_density_x1000"] = vector_context[
                "density_x1000"
            ]
            evidence["vector_context_segment_count"] = vector_context[
                "segment_count"
            ]
            evidence["dense_vector_context"] = bool(
                vector_context.get("dense_vector_context")
            )
        candidate = {
            "candidate_id": candidate_id,
            "schema_version": SCHEMA_VERSION,
            "page_index": self.page_index,
            "type": cand_type,
            "subtype": subtype,
            "bbox": _round_bbox(bbox),
            "source_nodes": source_nodes,
            "evidence_summary": evidence,
            "priority_hint": priority_hint,
            "reject_reason": reject_reason,
            "quality_warning_count": len(quality_warnings or []),
            "provenance": {
                "source_level": _SOURCE_LEVEL,
                "pipeline_version": PIPELINE_VERSION,
            },
        }
        if quality_warnings:
            candidate["provenance"]["quality_warnings"] = quality_warnings
        if vector_context:
            candidate["provenance"]["vector_context"] = vector_context
        if oriented_quad:
            candidate["oriented_quad"] = oriented_quad
        candidate["provenance"]["diagnostics"] = _candidate_diagnostics(
            cand_type=cand_type,
            subtype=subtype,
            bbox=bbox,
            source_nodes=source_nodes,
            oriented_quad=oriented_quad,
            vector_context=vector_context,
            inside_table=inside_table,
            inside_title=inside_title,
        )
        if text:
            candidate["text"] = text
        return candidate

    def _vector_context_metrics(
        self,
        bbox: dict[str, float],
        *,
        cand_type: str,
        subtype: str,
    ) -> dict[str, Any] | None:
        if (
            not self.vector_context_segments
            or cand_type != "dimension"
            or subtype != "vector_text_phrase"
        ):
            return None
        metrics = _vector_context_metrics_for_bbox(
            bbox,
            self.prepared_vector_context_segments,
            pad=self.vector_context_density_pad,
            inner_pad=self.vector_context_density_inner_pad,
        )
        if metrics is None:
            return None
        threshold = self.vector_context_density_threshold
        metrics["dense_vector_context"] = (
            threshold is not None
            and threshold > 0.0
            and metrics["density_x1000"] >= threshold
        )
        if threshold is not None:
            metrics["density_threshold_x1000"] = round(threshold, 3)
        return metrics

    def _reject_reason(
        self,
        bbox: dict[str, float] | None,
        source_nodes: list[dict[str, str]],
        *,
        cand_type: str,
        subtype: str,
        priority_hint: str,
        inside_table: bool,
        inside_title: bool,
    ) -> str | None:
        if not source_nodes:
            return "no_object_evidence"
        if not bbox or bbox.get("w", 0.0) <= 0.0 or bbox.get("h", 0.0) <= 0.0:
            return "invalid_geometry"
        if not _overlaps_page(bbox, self.page_width, self.page_height):
            return "outside_page"
        if inside_title and _has_title_block_rescue_evidence(
            cand_type,
            subtype,
            source_nodes,
        ):
            return "inside_title_block_with_dimension_evidence"
        if inside_title:
            return "inside_title_block"
        if cand_type == "gdt":
            if _has_source_kind(source_nodes, "low_confidence_gdt_frame"):
                return "low_confidence_gdt_frame"
            if _has_source_kind(source_nodes, "empty_gdt_text_evidence"):
                return "gdt_without_text_evidence"
        if (
            cand_type == "dimension"
            and subtype == "unknown"
            and len(source_nodes) == 1
            and _has_source_kind(source_nodes, "dimension_line")
        ):
            return "dimline_without_text_evidence"
        if (
            cand_type == "dimension"
            and subtype == "unknown"
            and len(source_nodes) == 1
            and _has_source_kind(source_nodes, "leader_line")
        ):
            return "leader_without_text_evidence"
        if (
            cand_type == "dimension"
            and subtype == "unknown"
            and _has_source_kind(source_nodes, "capsule")
            and len(source_nodes) == 1
            and not _is_topology_capsule_bbox(bbox)
        ):
            return "capsule_geometry_noise"
        if (
            cand_type == "dimension"
            and subtype == "unknown"
            and _has_source_kind(source_nodes, "capsule")
            and len(source_nodes) == 1
        ):
            if inside_table:
                return "capsule_inside_table_without_text_evidence"
            return "capsule_without_text_evidence"
        if (
            cand_type == "dimension"
            and subtype == "vector_text_phrase"
            and _is_border_grid_text(bbox, self.page_width, self.page_height)
        ):
            return "dimension_text_border_grid"
        if (
            cand_type == "dimension"
            and subtype == "vector_text_phrase"
            and _is_top_short_vector_text_fragment(bbox, self.page_width, self.page_height)
        ):
            return "dimension_text_top_short_fragment"
        if (
            cand_type == "dimension"
            and subtype == "vector_text_phrase"
            and _is_unstable_vector_text_fragment_group(
                source_nodes,
                bbox=bbox,
                page_width=self.page_width,
                page_height=self.page_height,
                inside_table=inside_table,
            )
        ):
            return "dimension_text_unstable_fragment_group"
        quality_reason = _quality_reject_reason(
            bbox,
            cand_type=cand_type,
            subtype=subtype,
            priority_hint=priority_hint,
            source_nodes=source_nodes,
        )
        if quality_reason:
            return quality_reason
        return None

    def _find_duplicate(self, candidate: dict[str, Any]) -> str | None:
        bbox = candidate["bbox"]
        for existing in self.candidates:
            if not _same_candidate_bucket(existing, candidate):
                continue
            if _bbox_iou(bbox, existing["bbox"]) >= 0.85:
                return str(existing.get("candidate_id"))
        return None

    def _drop_nested_duplicates(self) -> None:
        dropped: set[int] = set()
        for i, left in enumerate(self.candidates):
            if i in dropped:
                continue
            for j in range(i + 1, len(self.candidates)):
                if j in dropped:
                    continue
                right = self.candidates[j]
                if not _same_candidate_bucket(left, right):
                    continue
                left_area = _bbox_area(left["bbox"])
                right_area = _bbox_area(right["bbox"])
                if min(left_area, right_area) <= 0.0:
                    continue
                small, large = (i, j) if left_area <= right_area else (j, i)
                small_box = self.candidates[small]["bbox"]
                large_box = self.candidates[large]["bbox"]
                contained = _bbox_intersection_area(small_box, large_box) / _bbox_area(small_box)
                overlap = _bbox_iou(left["bbox"], right["bbox"])
                if contained >= 0.85 or (
                    left.get("type") == "gdt" and contained >= 0.70 and overlap >= 0.35
                ):
                    dropped.add(small)
                    self.candidates[small]["reject_reason"] = "duplicate_candidate"
                    self.candidates[small]["provenance"]["duplicate_of"] = self.candidates[large][
                        "candidate_id"
                    ]
        if not dropped:
            return
        self.drop_ledger.extend(self.candidates[idx] for idx in sorted(dropped))
        self.candidates = [
            candidate
            for idx, candidate in enumerate(self.candidates)
            if idx not in dropped
        ]


def _source_node(kind: str, index: int, source: str) -> dict[str, str]:
    return {
        "id": f"{kind}_{index:04d}",
        "kind": kind,
        "source": source,
    }


def _line_source_node(kind: str, index: int, line: dict[str, Any]) -> dict[str, Any]:
    node = _source_node(kind, index, "arrow_detector")
    orientation = str(line.get("orientation") or "").upper()
    if orientation in {"H", "V", "D"}:
        node["orientation"] = orientation
    angle = _dimline_angle(line)
    if angle is not None:
        node["angle_deg"] = round(angle, 3)
    line_kind = str(line.get("line_kind") or "")
    if line_kind:
        node["line_kind"] = line_kind
    return node


def _gdt_source_nodes(
    frame: dict[str, Any],
    index: int,
    *,
    text_region_hits: int = 0,
) -> list[dict[str, str]]:
    nodes = [_source_node("gdt_frame", index, "pdf_analyzer")]
    if _is_low_confidence_gdt_noise(frame):
        nodes.append(_source_node("low_confidence_gdt_frame", index, "pdf_analyzer"))
    if _is_empty_gdt_text_evidence_noise(frame, text_region_hits):
        nodes.append(_source_node("empty_gdt_text_evidence", index, "pdf_analyzer"))
    return nodes


def _is_low_confidence_gdt_noise(frame: dict[str, Any]) -> bool:
    """Flag review-learned GD&T false positives without dropping accepted shapes."""
    try:
        confidence = float(frame.get("confidence"))
    except Exception:
        return False
    if confidence > 0.5:
        return False
    bbox = normalize_bbox(frame)
    if not bbox:
        return False
    origin = str(frame.get("_origin") or "")
    compartment_count = len(frame.get("compartments") or [])
    height = float(bbox.get("h") or 0.0)
    return origin == "line" or compartment_count <= 2 or height >= 25.0


def _is_empty_gdt_text_evidence_noise(frame: dict[str, Any], text_region_hits: int) -> bool:
    """Drop review-learned empty GD&T-looking rectangles from geometry."""
    if text_region_hits > 0:
        return False
    try:
        confidence = float(frame.get("confidence"))
    except Exception:
        return False
    if confidence > 0.5:
        return False
    if str(frame.get("_origin") or "") != "rect":
        return False
    compartment_count = len(frame.get("compartments") or [])
    return compartment_count >= 3


def _has_source_kind(source_nodes: list[dict[str, str]], kind: str) -> bool:
    return any(node.get("kind") == kind for node in source_nodes)


def _has_title_block_rescue_evidence(
    cand_type: str,
    subtype: str,
    source_nodes: list[dict[str, str]] | None,
) -> bool:
    nodes = source_nodes or []
    if cand_type in {"gdt", "datum"}:
        return True
    if cand_type != "dimension":
        return False
    strong_kinds = {
        "dimension_line",
        "leader_line",
        "diameter_glyph",
        "numeric_symbol_seed",
        "pdf_text_word",
    }
    if any(node.get("kind") in strong_kinds for node in nodes):
        return True
    if _has_source_kind(nodes, "capsule") and _has_source_kind(nodes, "text_region"):
        return True
    if subtype in {"linear", "diameter", "pdf_text"}:
        return True
    return False


def _is_dot_seed_region(region: dict[str, Any]) -> bool:
    return (
        str(region.get("source") or "") == "numeric_symbol_seed"
        and str(region.get("subtype") or "") == "decimal_point_candidate"
        and (
            str(region.get("role") or "") == "dot_seed_only"
            or bool(region.get("dot_seed_only"))
        )
    )


def _has_dot_seed_source(source_nodes: list[dict[str, str]] | None) -> bool:
    return _has_source_kind(source_nodes or [], "numeric_symbol_seed")


def _has_text_region_source(source_nodes: list[dict[str, str]] | None) -> bool:
    return _has_source_kind(source_nodes or [], "text_region")


def _text_region_source_count(source_nodes: list[dict[str, str]] | None) -> int:
    return sum(1 for node in (source_nodes or []) if node.get("kind") == "text_region")


def _numeric_seed_source_nodes(
    text_regions: list[dict[str, Any]],
    indices: list[int],
) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    for region_idx in indices[:6]:
        if region_idx < 0 or region_idx >= len(text_regions):
            continue
        region = text_regions[region_idx]
        if int(region.get("numeric_symbol_seed_count") or 0) <= 0:
            continue
        subtypes = {
            str(subtype)
            for subtype in (region.get("numeric_symbol_seed_subtypes") or [])
        }
        if "decimal_point_candidate" not in subtypes:
            continue
        nodes.append(_source_node("numeric_symbol_seed", region_idx, "pdf_analyzer"))
    return nodes


def _has_scan_pass(source_nodes: list[dict[str, str]], scan_pass: str) -> bool:
    return any(node.get("scan_pass") == scan_pass for node in source_nodes)


def _has_directional_text_source(source_nodes: list[dict[str, str]] | None) -> bool:
    for node in source_nodes or []:
        if node.get("kind") != "vector_text_group":
            continue
        if node.get("scan_pass") == "rotated_90":
            return True
        if node.get("orientation") in {"D", "V"}:
            return True
        if node.get("angle_deg") is not None:
            return True
    return False


def _has_dimension_context_source(source_nodes: list[dict[str, str]] | None) -> bool:
    kinds = {node.get("kind") for node in (source_nodes or [])}
    return bool(kinds & {"dimension_line", "leader_line", "diameter_glyph", "pdf_text_word"})


def _is_dense_vector_context_noise(source_nodes: list[dict[str, str]] | None) -> bool:
    kinds = {node.get("kind") for node in (source_nodes or [])}
    positive_kinds = {
        "dimension_line",
        "leader_line",
        "capsule",
        "gdt_frame",
        "pdf_text_word",
        "numeric_symbol_seed",
        "diameter_glyph",
    }
    return not bool(kinds & positive_kinds)


def _evidence_summary(
    source_nodes: list[dict[str, str]],
    *,
    inside_table: bool,
    inside_title: bool,
) -> dict[str, Any]:
    kinds = {node.get("kind") for node in source_nodes}
    return {
        "has_text_region": "text_region" in kinds,
        "has_pdf_text_word": "pdf_text_word" in kinds,
        "has_dimline": "dimension_line" in kinds,
        "has_leader": "leader_line" in kinds,
        "has_arrow": "dimension_line" in kinds or "leader_line" in kinds,
        "has_gdt_frame": "gdt_frame" in kinds,
        "has_low_confidence_gdt": "low_confidence_gdt_frame" in kinds,
        "has_empty_gdt_text_evidence": "empty_gdt_text_evidence" in kinds,
        "has_datum_box": "datum_box" in kinds,
        "has_capsule": "capsule" in kinds,
        "has_numeric_symbol_seed": "numeric_symbol_seed" in kinds,
        "inside_table": bool(inside_table),
        "inside_title_block": bool(inside_title),
        "source_count": len(source_nodes),
    }


def _candidate_diagnostics(
    *,
    cand_type: str,
    subtype: str,
    bbox: dict[str, float],
    source_nodes: list[dict[str, str]],
    oriented_quad: dict[str, Any] | None,
    vector_context: dict[str, Any] | None,
    inside_table: bool,
    inside_title: bool,
) -> dict[str, Any]:
    kinds = {node.get("kind") for node in source_nodes}
    text_region_count = _text_region_source_count(source_nodes)
    has_dot_seed = _has_dot_seed_source(source_nodes)
    has_capsule = "capsule" in kinds
    has_gdt = "gdt_frame" in kinds
    has_dimline = "dimension_line" in kinds
    has_leader = "leader_line" in kinds
    has_pdf_text = "pdf_text_word" in kinds
    has_datum = "datum_box" in kinds
    has_vector_group = "vector_text_group" in kinds
    has_dimension_context = _has_dimension_context_source(source_nodes)
    anchor_kind = _candidate_anchor_kind(
        cand_type=cand_type,
        subtype=subtype,
        source_nodes=source_nodes,
    )
    diagnostics: dict[str, Any] = {
        "anchor_kind": anchor_kind,
        "text_region_count": text_region_count,
        "has_dot_seed": has_dot_seed,
        "has_capsule": has_capsule,
        "has_gdt_frame": has_gdt,
        "has_dimline": has_dimline,
        "has_leader": has_leader,
        "has_pdf_text_word": has_pdf_text,
        "has_datum_box": has_datum,
        "has_vector_text_group": has_vector_group,
        "has_axis_anchor": bool(has_capsule or has_gdt or has_dimline or has_leader),
        "has_dimension_context": has_dimension_context,
        "inside_table": bool(inside_table),
        "inside_title_block": bool(inside_title),
        "bbox_area": round(float(bbox.get("w") or 0.0) * float(bbox.get("h") or 0.0), 3),
    }
    if vector_context:
        diagnostics["dense_vector_context"] = bool(vector_context.get("dense_vector_context"))
        diagnostics["vector_context_density_x1000"] = vector_context.get("density_x1000")
        diagnostics["vector_context_segment_count"] = vector_context.get("segment_count")
    if oriented_quad:
        diagnostics["oriented_source"] = str(oriented_quad.get("source") or "")
        diagnostics["oriented_angle_deg"] = oriented_quad.get("angle_deg")
        diagnostics["oriented_long"] = oriented_quad.get("long_length")
        diagnostics["oriented_short"] = oriented_quad.get("short_length")
        try:
            long_length = float(oriented_quad.get("long_length") or 0.0)
            short_length = float(oriented_quad.get("short_length") or 0.0)
        except (TypeError, ValueError):
            long_length = 0.0
            short_length = 0.0
        if long_length > 0.0:
            diagnostics["oriented_short_long_ratio"] = round(short_length / long_length, 4)
    return diagnostics


def _candidate_anchor_kind(
    *,
    cand_type: str,
    subtype: str,
    source_nodes: list[dict[str, str]],
) -> str:
    kinds = {node.get("kind") for node in source_nodes}
    has_text = "text_region" in kinds
    if "gdt_frame" in kinds:
        return "gdt_frame_text" if has_text else "gdt_frame_only"
    if "datum_box" in kinds:
        return "datum_box"
    if "capsule" in kinds:
        return "capsule_text" if has_text else "capsule_only"
    if "dimension_line" in kinds:
        return "dimline_text" if has_text else "dimline_only"
    if "leader_line" in kinds:
        return "leader_text" if has_text else "leader_only"
    if "diameter_glyph" in kinds:
        return "diameter_glyph_text" if has_text else "diameter_glyph_only"
    if "pdf_text_word" in kinds:
        return "pdf_text_word"
    if "table_text_region" in kinds:
        return "table_text"
    if "vector_text_group" in kinds:
        return _free_vector_text_anchor_kind(source_nodes)
    if has_text:
        return "text_region"
    if cand_type:
        return f"{cand_type}_{subtype or 'unknown'}"
    return "unknown"


def _free_vector_text_anchor_kind(source_nodes: list[dict[str, str]]) -> str:
    group_nodes = [
        node for node in source_nodes
        if node.get("kind") == "vector_text_group"
    ]
    for node in group_nodes:
        if node.get("scan_pass") == "rotated_90":
            return "free_vertical_rotated90"
    for node in group_nodes:
        if node.get("orientation") == "V":
            return "free_vertical"
    for node in group_nodes:
        if node.get("orientation") == "D" or node.get("angle_deg") is not None:
            return "free_diagonal"
    return "free_horizontal"


def _quality_reject_reason(
    bbox: dict[str, float],
    *,
    cand_type: str,
    subtype: str,
    priority_hint: str,
    source_nodes: list[dict[str, str]] | None = None,
) -> str | None:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if cand_type == "gdt":
        if height < 9.0:
            return "gdt_fragment_too_short"
        if width < 35.0:
            return "gdt_fragment_too_narrow"
        if priority_hint == "low" and height < 12.0:
            return "low_priority_gdt_fragment"
        if width < 60.0 or height < 16.0:
            return "gdt_review_size_floor"
    if cand_type == "datum":
        if width < 12.0 or height < 12.0:
            return "datum_feature_too_small"
    if cand_type == "dimension" and subtype in {"linear", "diameter", "text_region", "table_text", "pdf_text", "vector_text_phrase"}:
        area = width * height
        if subtype == "vector_text_phrase":
            text_region_count = _text_region_source_count(source_nodes)
            has_dot_seed = _has_dot_seed_source(source_nodes)
            has_capsule = _has_source_kind(source_nodes or [], "capsule")
            has_dimension_context = _has_dimension_context_source(source_nodes)
            is_stacked_tolerance = _is_stacked_tolerance_phrase_bbox(bbox)
            is_compact_stacked_tolerance = _is_compact_stacked_tolerance_phrase_bbox(bbox)
            is_vertical_stacked_tolerance = _is_vertical_stacked_tolerance_phrase_bbox(bbox)
            if (
                has_capsule
                and not has_dot_seed
                and (max(width, height) < 24.0 or area < 220.0)
            ):
                return "dimension_text_review_size_floor"
            if (
                text_region_count >= 6
                and has_dot_seed
                and not has_capsule
            ):
                return "dimension_text_review_size_floor"
            if (
                text_region_count >= 5
                and _has_scan_pass(source_nodes or [], "rotated_90")
                and not _is_vertical_stacked_tolerance_phrase_bbox(bbox)
                and (
                    height > 90.0
                    or area > 5000.0
                    or not _has_dot_seed_source(source_nodes)
                )
            ):
                return "dimension_text_review_size_floor"
            if width < 6.0 or height < 5.0 or area < 60.0:
                return "dimension_text_fragment_too_small"
            if not has_dot_seed and not has_capsule and (max(width, height) < 16.0 or area < 130.0):
                return "dimension_text_review_size_floor"
            if (
                has_capsule
                and not has_dimension_context
                and not is_stacked_tolerance
                and not is_compact_stacked_tolerance
                and not is_vertical_stacked_tolerance
                and (
                    (text_region_count >= 2 and (height > 45.0 or area > 3500.0))
                    or (text_region_count >= 4 and width > 90.0)
                )
            ):
                return "dimension_text_capsule_overmerge"
            if (
                has_dot_seed
                and not has_capsule
                and not has_dimension_context
                and text_region_count >= 4
                and not is_stacked_tolerance
                and not is_compact_stacked_tolerance
                and not is_vertical_stacked_tolerance
                and (height > 55.0 or width > 95.0 or area > 2800.0)
            ):
                return "dimension_text_seeded_overmerge"
            if (
                text_region_count >= 3
                and not has_dot_seed
                and not has_capsule
                and not is_vertical_stacked_tolerance
                and (width > 90.0 or height > 32.0)
            ):
                return "dimension_text_weak_non_seed_phrase"
            if (
                text_region_count >= 1
                and not has_dot_seed
                and not has_capsule
                and not has_dimension_context
                and not is_stacked_tolerance
                and not is_compact_stacked_tolerance
                and not is_vertical_stacked_tolerance
            ):
                return "dimension_text_weak_non_seed_phrase"
            if (
                width > 160.0
                or (
                    height > 45.0
                    and not _is_large_shallow_diagonal_phrase_bbox(bbox)
                    and not _is_vertical_stacked_tolerance_phrase_bbox(bbox)
                    and not (
                        _has_scan_pass(source_nodes or [], "rotated_90")
                        and _is_rotated_vertical_phrase_bbox(bbox)
                    )
                    and not (
                        has_capsule
                        and _is_capsule_backed_vector_text_phrase_bbox(bbox)
                    )
                )
            ):
                return "dimension_text_review_size_floor"
            return None
        if subtype == "diameter":
            if height > 45.0 or area > 3200.0:
                return "dimension_diameter_review_size_floor"
        if width < 8.0 or height < 6.0 or area < 120.0:
            return "dimension_text_fragment_too_small"
        if subtype == "pdf_text":
            return None
        if subtype == "table_text":
            if max(width, height) < 24.0 or min(width, height) < 8.0 or area < 220.0:
                return "dimension_text_review_size_floor"
            return "dimension_table_text_weak_evidence"
        if subtype == "text_region":
            if max(width, height) < 55.0 or min(width, height) < 10.0 or area < 800.0:
                return "dimension_text_review_size_floor"
            return None
        if subtype == "linear" and priority_hint == "high":
            return None
        if width < 55.0 or area < 800.0:
            return "dimension_text_review_size_floor"
    return None


def _is_plausible_table_text_region(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    area = width * height
    if width < 18.0 or height < 7.0 or area < 160.0:
        return False
    if width > 360.0 or height > 36.0:
        return False
    return True


def _count_text_regions_inside(
    bbox: dict[str, float] | None,
    text_regions: list[dict[str, Any]],
) -> int:
    if not bbox:
        return 0
    x0 = float(bbox.get("x") or 0.0)
    y0 = float(bbox.get("y") or 0.0)
    x1 = x0 + float(bbox.get("w") or 0.0)
    y1 = y0 + float(bbox.get("h") or 0.0)
    count = 0
    for region in text_regions:
        region_bbox = normalize_bbox(region)
        if not region_bbox:
            continue
        cx, cy = _bbox_center(region_bbox)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            count += 1
    return count


def _is_border_grid_text(
    bbox: dict[str, float],
    page_width: float,
    page_height: float,
) -> bool:
    """Filter drawing-zone coordinates around the outer sheet border."""
    if page_width <= 0.0 or page_height <= 0.0:
        return False
    if page_width < 1000.0 or page_height < 700.0:
        return False
    x = float(bbox.get("x") or 0.0)
    y = float(bbox.get("y") or 0.0)
    w = float(bbox.get("w") or 0.0)
    h = float(bbox.get("h") or 0.0)
    cx = x + w / 2.0
    cy = y + h / 2.0
    top_strip = min(80.0, page_height * 0.06)
    side_strip = max(120.0, min(170.0, page_width * 0.065))
    side_top_band = min(260.0, page_height * 0.16)
    if y < top_strip:
        return True
    if cy < side_top_band and (cx < side_strip or cx > page_width - side_strip):
        return True
    return False


def _is_top_short_vector_text_fragment(
    bbox: dict[str, float],
    page_width: float,
    page_height: float,
) -> bool:
    """Drop short top-of-sheet vector fragments that are drawing indices."""
    if page_width <= 0.0 or page_height <= 0.0:
        return False
    if page_width < 1000.0 or page_height < 700.0:
        return False
    if _is_vertical_stacked_tolerance_phrase_bbox(bbox):
        return False
    y = float(bbox.get("y") or 0.0)
    width = float(bbox.get("w") or 0.0)
    top_band = min(200.0, page_height * 0.12)
    return y < top_band and width < 40.0


def _is_unstable_vector_text_fragment_group(
    source_nodes: list[dict[str, str]],
    *,
    bbox: dict[str, float],
    page_width: float,
    page_height: float,
    inside_table: bool = False,
) -> bool:
    """Filter vector phrase group sizes using conservative geometry gates."""
    if page_width < 1000.0 or page_height < 700.0:
        return False
    text_region_count = sum(
        1
        for node in source_nodes
        if node.get("kind") == "text_region"
    )
    if _has_dot_seed_source(source_nodes) and 1 <= text_region_count <= 4:
        return False
    if (
        _has_dot_seed_source(source_nodes)
        and text_region_count <= 8
        and _has_directional_text_source(source_nodes)
    ):
        return False
    if (
        _is_shallow_diagonal_phrase_bbox(bbox)
        or _is_large_shallow_diagonal_phrase_bbox(bbox)
        or _is_vertical_stacked_tolerance_phrase_bbox(bbox)
        or (
            _has_scan_pass(source_nodes, "rotated_90")
            and _is_rotated_vertical_phrase_bbox(bbox)
        )
    ):
        return False
    if (
        text_region_count == 2
        and inside_table
        and _is_compact_table_dimension_phrase_bbox(bbox)
    ):
        return False
    return text_region_count == 2 or text_region_count >= 6


def _is_compact_table_dimension_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if height <= 0.0:
        return False
    aspect = width / height
    return 45.0 <= width <= 75.0 and 10.0 <= height <= 22.0 and 2.4 <= aspect <= 5.8


def _is_stacked_tolerance_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if height <= 0.0:
        return False
    area = width * height
    aspect = width / height
    return (
        45.0 <= width <= 95.0
        and 22.0 <= height <= 40.0
        and 1000.0 <= area <= 3400.0
        and 1.15 <= aspect <= 3.0
    )


def _is_compact_stacked_tolerance_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if height <= 0.0:
        return False
    area = width * height
    aspect = width / height
    return (
        28.0 <= width <= 45.0
        and 24.0 <= height <= 40.0
        and 650.0 <= area <= 1800.0
        and 0.75 <= aspect <= 1.85
    )


def _is_vertical_stacked_tolerance_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if width <= 0.0:
        return False
    area = width * height
    aspect = height / width
    return (
        8.0 <= width <= 26.0
        and 30.0 <= height <= 90.0
        and 300.0 <= area <= 2200.0
        and 1.8 <= aspect <= 9.0
    )


def _is_shallow_diagonal_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    return 55.0 <= width <= 125.0 and 24.0 <= height <= 50.0


def _is_large_shallow_diagonal_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if height <= 0.0:
        return False
    area = width * height
    aspect = width / height
    return (
        70.0 <= width <= 145.0
        and 35.0 <= height <= 65.0
        and 2500.0 <= area <= 8000.0
        and 1.35 <= aspect <= 4.0
    )


def _vector_text_phrase_groups(
    text_regions: list[dict[str, Any]],
    *,
    used_indices: set[int],
    dot_seed_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, region in enumerate(text_regions):
        if idx in used_indices:
            continue
        bbox = normalize_bbox(region)
        if not bbox:
            continue
        width = float(bbox.get("w") or 0.0)
        height = float(bbox.get("h") or 0.0)
        area = width * height
        is_large_shallow_diagonal = _is_large_shallow_diagonal_phrase_bbox(bbox)
        is_vertical_stacked_tolerance = _is_vertical_stacked_tolerance_phrase_bbox(bbox)
        if width < 4.0 or height < 4.0 or area < 25.0:
            continue
        if (
            not is_large_shallow_diagonal
            and not is_vertical_stacked_tolerance
            and (width > 120.0 or height > 35.0)
        ):
            continue
        if is_large_shallow_diagonal and (width > 145.0 or height > 65.0):
            continue
        if _is_track_b_vertical_noise(region, bbox):
            continue
        items.append({
            "index": idx,
            "bbox": bbox,
            "confidence": str(region.get("confidence") or "high"),
            "source": str(region.get("source") or ""),
        })
    if not items:
        return []
    dot_items: list[dict[str, Any]] = []
    for seed_idx, seed in enumerate(dot_seed_regions or []):
        bbox = normalize_bbox(seed)
        if not bbox:
            continue
        dot_items.append({
            "index": int(seed.get("_text_region_index", seed_idx)),
            "bbox": bbox,
        })

    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj

    for i, left_item in enumerate(items):
        left = left_item["bbox"]
        left_center = _bbox_center(left)
        for j in range(i + 1, len(items)):
            right = items[j]["bbox"]
            right_center = _bbox_center(right)
            if abs(left_center[0] - right_center[0]) > 140.0:
                continue
            if abs(left_center[1] - right_center[1]) > 40.0:
                continue
            if _should_merge_text_regions(left, right) or _dot_seed_bridges_text_regions(
                left,
                right,
                dot_items,
            ):
                union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)

    out: list[dict[str, Any]] = []
    member_sets: list[list[dict[str, Any]]] = []
    for members in groups.values():
        member_sets.extend(_split_overmerged_vertical_columns(members))

    for members in member_sets:
        bbox = _union_bboxes([item["bbox"] for item in members])
        if not bbox:
            continue
        dot_hits = _dot_seeds_for_group_bbox(bbox, dot_items)
        if dot_hits:
            bbox = _union_bboxes([bbox, *[item["bbox"] for item in dot_hits]])
            if not bbox:
                continue
        width = float(bbox.get("w") or 0.0)
        height = float(bbox.get("h") or 0.0)
        area = width * height
        if len(members) == 1:
            source = str(members[0].get("source") or "")
            is_large_shallow_diagonal = _is_large_shallow_diagonal_phrase_bbox(bbox)
            is_stacked_tolerance = _is_stacked_tolerance_phrase_bbox(bbox)
            is_compact_stacked_tolerance = _is_compact_stacked_tolerance_phrase_bbox(bbox)
            is_vertical_stacked_tolerance = _is_vertical_stacked_tolerance_phrase_bbox(bbox)
            is_low_text_line = (
                source != "track_b"
                and width >= 10.0
                and 5.5 <= height < 8.0
                and area >= 60.0
            )
            if width < 6.0 or area < 60.0:
                continue
            if height < 8.0 and not is_low_text_line:
                continue
            if height >= 8.0 and area < 100.0:
                continue
            if (
                not is_large_shallow_diagonal
                and not is_stacked_tolerance
                and not is_compact_stacked_tolerance
                and not is_vertical_stacked_tolerance
                and (width > 140.0 or height > 28.0)
            ):
                continue
            if is_large_shallow_diagonal and (width > 145.0 or height > 65.0):
                continue
        else:
            is_vertical_stacked_tolerance = _is_vertical_stacked_tolerance_phrase_bbox(bbox)
            if width < 8.0 or height < 6.0 or area < 120.0:
                continue
            if not is_vertical_stacked_tolerance and (width > 140.0 or height > 50.0):
                continue
        priority = "normal" if any(item["confidence"] == "high" for item in members) else "low"
        angle_deg = _vector_text_group_angle(members)
        orientation = "V" if _is_vertical_stacked_tolerance_phrase_bbox(bbox) else "H"
        out.append({
            "bbox": bbox,
            "indices": sorted(item["index"] for item in members),
            "dot_seed_indices": sorted(item["index"] for item in dot_hits),
            "dot_seed_count": len(dot_hits),
            "priority_hint": priority,
            "angle_deg": angle_deg,
            "orientation": orientation,
            "oriented_quad": _text_group_oriented_quad(members, angle_deg),
        })
    out.sort(key=lambda item: (round(item["bbox"]["y"] / 5.0) * 5.0, item["bbox"]["x"]))
    return out


def _dot_seed_bridges_text_regions(
    left: dict[str, float],
    right: dict[str, float],
    dot_items: list[dict[str, Any]],
) -> bool:
    if not dot_items:
        return False
    left_center = _bbox_center(left)
    right_center = _bbox_center(right)
    if abs(left_center[0] - right_center[0]) > 60.0 and abs(left_center[1] - right_center[1]) > 32.0:
        return False
    union_bbox = _union_bboxes([left, right])
    padded = _pad_bbox(union_bbox, 8.0)
    if not padded:
        return False
    horizontal_bridge = abs(left_center[0] - right_center[0]) >= abs(left_center[1] - right_center[1])
    min_center_x = min(left_center[0], right_center[0]) - 4.0
    max_center_x = max(left_center[0], right_center[0]) + 4.0
    min_center_y = min(left_center[1], right_center[1]) - 4.0
    max_center_y = max(left_center[1], right_center[1]) + 4.0
    for dot in dot_items:
        dot_bbox = dot["bbox"]
        dcx, dcy = _bbox_center(dot_bbox)
        if horizontal_bridge and not (min_center_x <= dcx <= max_center_x):
            continue
        if not horizontal_bridge and not (min_center_y <= dcy <= max_center_y):
            continue
        if (
            padded["x"] <= dcx <= padded["x"] + padded["w"]
            and padded["y"] <= dcy <= padded["y"] + padded["h"]
        ):
            return True
    return False


def _dot_seeds_for_group_bbox(
    bbox: dict[str, float],
    dot_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not dot_items:
        return []
    pad = min(12.0, max(4.0, float(bbox.get("h") or 0.0) * 0.6))
    padded = _pad_bbox(bbox, pad)
    if not padded:
        return []
    hits: list[dict[str, Any]] = []
    for dot in dot_items:
        dcx, dcy = _bbox_center(dot["bbox"])
        if (
            padded["x"] <= dcx <= padded["x"] + padded["w"]
            and padded["y"] <= dcy <= padded["y"] + padded["h"]
        ):
            hits.append(dot)
    return hits


def _capsule_vector_text_phrase_groups(
    capsule_candidates: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose text inside rounded key-dimension capsules to topology matching."""
    groups: list[dict[str, Any]] = []
    text_items: list[dict[str, Any]] = []
    for idx, region in enumerate(text_regions):
        bbox = normalize_bbox(region)
        if not bbox:
            continue
        width = float(bbox.get("w") or 0.0)
        height = float(bbox.get("h") or 0.0)
        area = width * height
        if width < 3.0 or height < 3.0 or area < 18.0:
            continue
        text_items.append({
            "index": idx,
            "bbox": bbox,
            "confidence": str(region.get("confidence") or "high"),
        })

    for capsule_index, capsule in enumerate(capsule_candidates):
        capsule_bbox = normalize_bbox(capsule)
        if not capsule_bbox or not _is_topology_capsule_bbox(capsule_bbox):
            continue
        hits: list[dict[str, Any]] = []
        for item in text_items:
            bbox = item["bbox"]
            if _text_bbox_belongs_to_capsule(bbox, capsule_bbox):
                hits.append(item)
        if not hits:
            continue
        bbox = _union_bboxes([item["bbox"] for item in hits])
        if not bbox:
            continue
        if not _is_capsule_backed_vector_text_phrase_bbox(bbox):
            continue
        groups.append({
            "capsule_index": capsule_index,
            "bbox": bbox,
            "indices": sorted(item["index"] for item in hits),
            "priority_hint": "normal" if any(item["confidence"] == "high" for item in hits) else "low",
            "orientation": "V" if float(bbox.get("h") or 0.0) > float(bbox.get("w") or 0.0) * 1.15 else "H",
        })
    return groups


def _rotated_vertical_text_phrase_groups(
    text_regions: list[dict[str, Any]],
    *,
    used_indices: set[int],
    page_width: float,
) -> list[dict[str, Any]]:
    """Run a horizontal-style grouping pass after a 90-degree bbox transform."""
    items: list[dict[str, Any]] = []
    for idx, region in enumerate(text_regions):
        if idx in used_indices:
            continue
        bbox = normalize_bbox(region)
        if not bbox:
            continue
        width = float(bbox.get("w") or 0.0)
        height = float(bbox.get("h") or 0.0)
        area = width * height
        if width < 3.5 or height < 4.0 or area < 35.0:
            continue
        if width > 70.0 or height > 90.0 or area > 4500.0:
            continue
        rotated = {
            "x": bbox["y"],
            "y": float(page_width or 0.0) - (bbox["x"] + bbox["w"]),
            "w": bbox["h"],
            "h": bbox["w"],
        }
        items.append({
            "index": idx,
            "bbox": bbox,
            "rotated_bbox": rotated,
            "confidence": str(region.get("confidence") or "high"),
        })
    if not items:
        return []

    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[ri] = rj

    for i, left_item in enumerate(items):
        left = left_item["rotated_bbox"]
        left_center = _bbox_center(left)
        for j in range(i + 1, len(items)):
            right = items[j]["rotated_bbox"]
            right_center = _bbox_center(right)
            if abs(left_center[0] - right_center[0]) > 170.0:
                continue
            if abs(left_center[1] - right_center[1]) > 55.0:
                continue
            if _should_merge_rotated_scan_regions(left, right):
                union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)

    out: list[dict[str, Any]] = []
    for members in groups.values():
        for column_members in _split_overmerged_vertical_columns(members):
            if len(column_members) < 2:
                continue
            bbox = _union_bboxes([item["bbox"] for item in column_members])
            if not bbox or not _is_rotated_vertical_phrase_bbox(bbox):
                continue
            if len(column_members) < 3 and float(bbox.get("h") or 0.0) < 40.0:
                continue
            priority = "normal" if any(item["confidence"] == "high" for item in column_members) else "low"
            out.append({
                "bbox": bbox,
                "indices": sorted(item["index"] for item in column_members),
                "priority_hint": priority,
                "orientation": "V",
                "oriented_quad": _bbox_oriented_quad(
                    bbox,
                    90.0,
                    source="vector_text_group",
                ),
            })
    out.sort(key=lambda item: (round(item["bbox"]["x"] / 5.0) * 5.0, item["bbox"]["y"]))
    return out


def _should_merge_rotated_scan_regions(
    left: dict[str, float],
    right: dict[str, float],
) -> bool:
    left_center = _bbox_center(left)
    right_center = _bbox_center(right)
    horizontal_left, horizontal_right = (left, right) if left["x"] <= right["x"] else (right, left)
    horizontal_gap = horizontal_right["x"] - (horizontal_left["x"] + horizontal_left["w"])
    max_height = max(float(left.get("h") or 0.0), float(right.get("h") or 0.0))
    y_overlap = _axis_overlap(
        left["y"],
        left["y"] + left["h"],
        right["y"],
        right["y"] + right["h"],
    )
    if -6.0 <= horizontal_gap <= 28.0:
        if abs(left_center[1] - right_center[1]) <= max(9.0, max_height * 1.25):
            return True
        if y_overlap >= min(left["h"], right["h"]) * 0.25:
            return True
    if -8.0 <= horizontal_gap <= 42.0:
        center_dx = abs(right_center[0] - left_center[0])
        center_dy = abs(right_center[1] - left_center[1])
        if 8.0 <= center_dx <= 80.0 and 8.0 <= center_dy <= 34.0:
            slope = center_dy / max(center_dx, 1e-6)
            return 0.12 <= slope <= 1.25
    return False


def _is_rotated_vertical_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if width <= 0.0:
        return False
    area = width * height
    aspect = height / width
    return (
        7.0 <= width <= 95.0
        and 24.0 <= height <= 185.0
        and 220.0 <= area <= 12000.0
        and aspect >= 1.3
    )


def _is_topology_capsule_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    area = width * height
    if width < 18.0 or height < 8.0 or area < 220.0:
        return False
    if width > 220.0 or height > 180.0 or area > 18000.0:
        return False
    return True


def _text_bbox_belongs_to_capsule(
    text_bbox: dict[str, float],
    capsule_bbox: dict[str, float],
) -> bool:
    expanded = _pad_bbox(capsule_bbox, 3.0) or capsule_bbox
    cx, cy = _bbox_center(text_bbox)
    if (
        expanded["x"] <= cx <= expanded["x"] + expanded["w"]
        and expanded["y"] <= cy <= expanded["y"] + expanded["h"]
    ):
        return True
    overlap = _bbox_intersection_area(text_bbox, expanded)
    return overlap / max(_bbox_area(text_bbox), 1.0) >= 0.45


def _is_capsule_backed_vector_text_phrase_bbox(bbox: dict[str, float]) -> bool:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    area = width * height
    if width < 6.0 or height < 5.0 or area < 60.0:
        return False
    if width > 180.0 or height > 160.0 or area > 14000.0:
        return False
    return True


def _split_overmerged_vertical_columns(
    members: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split adjacent vertical text columns that transitive merging fused."""
    if len(members) < 6:
        return [members]
    bbox = _union_bboxes([item["bbox"] for item in members])
    if not bbox:
        return [members]
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if width > 90.0 or height < 55.0 or height / max(width, 1.0) < 1.2:
        return [members]

    sorted_members = sorted(members, key=lambda item: _bbox_center(item["bbox"])[0])
    widths = [
        float(item["bbox"].get("w") or 0.0)
        for item in sorted_members
        if float(item["bbox"].get("w") or 0.0) > 0.0
    ]
    median_width = sorted(widths)[len(widths) // 2] if widths else 10.0
    split_gap = max(14.0, median_width * 1.35)

    columns: list[list[dict[str, Any]]] = [[sorted_members[0]]]
    last_x = _bbox_center(sorted_members[0]["bbox"])[0]
    for item in sorted_members[1:]:
        cx = _bbox_center(item["bbox"])[0]
        if cx - last_x > split_gap:
            columns.append([])
        columns[-1].append(item)
        last_x = cx

    if len(columns) <= 1:
        return [members]

    valid_columns: list[list[dict[str, Any]]] = []
    for column in columns:
        if len(column) < 2:
            continue
        column_bbox = _union_bboxes([item["bbox"] for item in column])
        if column_bbox and _is_vertical_stacked_tolerance_phrase_bbox(column_bbox):
            valid_columns.append(column)

    if len(valid_columns) < 1:
        return [members]
    return valid_columns


def _should_merge_text_regions(left: dict[str, float], right: dict[str, float]) -> bool:
    if _is_tiny_edge_fragment_stuck_to_broad_text(left, right):
        return False

    left_center = _bbox_center(left)
    right_center = _bbox_center(right)
    max_height = max(float(left.get("h") or 0.0), float(right.get("h") or 0.0))
    max_width = max(float(left.get("w") or 0.0), float(right.get("w") or 0.0))

    horizontal_left, horizontal_right = (left, right) if left["x"] <= right["x"] else (right, left)
    horizontal_gap = horizontal_right["x"] - (horizontal_left["x"] + horizontal_left["w"])
    y_overlap = _axis_overlap(
        left["y"],
        left["y"] + left["h"],
        right["y"],
        right["y"] + right["h"],
    )
    if -4.0 <= horizontal_gap <= 10.0:
        if abs(left_center[1] - right_center[1]) <= max_height * 0.9:
            return True
        if y_overlap >= min(left["h"], right["h"]) * 0.45:
            return True
    if -4.0 <= horizontal_gap <= 30.0:
        center_dx = abs(right_center[0] - left_center[0])
        center_dy = abs(right_center[1] - left_center[1])
        if 12.0 <= center_dx <= 46.0 and 3.0 <= center_dy <= 24.0:
            slope = center_dy / max(center_dx, 1e-6)
            if 0.12 <= slope <= 0.75 and max_height <= 24.0 and max_width <= 36.0:
                return True

    vertical_top, vertical_bottom = (left, right) if left["y"] <= right["y"] else (right, left)
    vertical_gap = vertical_bottom["y"] - (vertical_top["y"] + vertical_top["h"])
    x_overlap = _axis_overlap(
        left["x"],
        left["x"] + left["w"],
        right["x"],
        right["x"] + right["w"],
    )
    if _is_wide_parallel_text_stack(left, right, vertical_gap, x_overlap):
        return False
    if -4.0 <= vertical_gap <= 6.0:
        if x_overlap >= min(left["w"], right["w"]) * 0.25:
            return True
        if abs(left_center[0] - right_center[0]) <= max_width * 0.9:
            return True
    if -4.0 <= vertical_gap <= 16.0:
        narrow_aligned = (
            max_width <= 20.0
            and min(left["w"], right["w"]) >= 4.0
            and abs(left_center[0] - right_center[0]) <= max_width * 0.45
        )
        if narrow_aligned:
            return True

    return False


def _is_tiny_edge_fragment_stuck_to_broad_text(
    left: dict[str, float],
    right: dict[str, float],
) -> bool:
    """Avoid merging a wide stacked label with an adjacent dimension-line stub."""
    left_area = float(left.get("w") or 0.0) * float(left.get("h") or 0.0)
    right_area = float(right.get("w") or 0.0) * float(right.get("h") or 0.0)
    broad, tiny = (left, right) if left_area >= right_area else (right, left)
    broad_w = float(broad.get("w") or 0.0)
    broad_h = float(broad.get("h") or 0.0)
    tiny_w = float(tiny.get("w") or 0.0)
    tiny_h = float(tiny.get("h") or 0.0)
    if not (broad_w >= 60.0 and broad_h >= 30.0 and tiny_w <= 14.0 and tiny_h <= 14.5):
        return False

    broad_center = _bbox_center(broad)
    tiny_center = _bbox_center(tiny)
    vertical_offset = abs(tiny_center[1] - broad_center[1])
    if vertical_offset < broad_h * 0.35:
        return False
    if vertical_offset > broad_h * 0.85:
        return False

    edge_band = max(12.0, broad_w * 0.18)
    near_left_edge = tiny_center[0] <= float(broad.get("x") or 0.0) + edge_band
    near_right_edge = tiny_center[0] >= float(broad.get("x") or 0.0) + broad_w - edge_band
    return near_left_edge or near_right_edge


def _is_wide_parallel_text_stack(
    left: dict[str, float],
    right: dict[str, float],
    vertical_gap: float,
    x_overlap: float,
) -> bool:
    """Avoid fusing separate note/roughness lines into one tall text box."""
    if not (-4.0 <= vertical_gap <= 8.0):
        return False
    min_width = min(float(left.get("w") or 0.0), float(right.get("w") or 0.0))
    max_width = max(float(left.get("w") or 0.0), float(right.get("w") or 0.0))
    if min_width < 24.0 or max_width < 60.0:
        return False
    if x_overlap < min_width * 0.5:
        return False
    left_center = _bbox_center(left)
    right_center = _bbox_center(right)
    return abs(left_center[0] - right_center[0]) <= max_width * 0.45


def _capsule_oriented_quad(capsule: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(capsule, dict):
        return None
    if "oriented_long" not in capsule or "oriented_short" not in capsule:
        return None
    bbox = normalize_bbox(capsule)
    if not bbox:
        return None
    try:
        long_length = float(capsule.get("oriented_long") or 0.0)
        short_length = float(capsule.get("oriented_short") or 0.0)
        angle_deg = float(capsule.get("angle") or 0.0)
    except Exception:
        return None
    if long_length <= 0.0 or short_length <= 0.0:
        return None
    cx, cy = _bbox_center(bbox)
    return _oriented_quad_from_center(
        cx,
        cy,
        long_length,
        short_length,
        angle_deg,
        source="capsule",
    )


def _gdt_oriented_quad(frame: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(frame, dict):
        return None
    quad = frame.get("oriented_quad")
    if isinstance(quad, dict):
        points = quad.get("points")
        if isinstance(points, list) and len(points) == 4:
            normalized = dict(quad)
            normalized.setdefault("coord_space", "page_pdf")
            normalized.setdefault("source", "gdt_frame")
            return normalized
    angle = frame.get("angle_deg", frame.get("angle"))
    if angle is None:
        return None
    try:
        angle_value = float(angle)
    except (TypeError, ValueError):
        return None
    if angle_value % 180.0 in (0.0, 90.0):
        return None
    return _bbox_oriented_quad(
        normalize_bbox(frame),
        angle_value,
        source="gdt_frame",
    )


def _dimline_oriented_quad(
    dimline: dict[str, Any],
    bbox: dict[str, float] | None,
) -> dict[str, Any] | None:
    angle = _dimline_angle(dimline)
    if angle is None:
        return None
    orientation = str(dimline.get("orientation") or "").upper()
    if orientation == "H" and (angle <= 10.0 or angle >= 170.0):
        return None
    return _bbox_oriented_quad(
        bbox,
        angle,
        source="dimension_line",
    )


def _text_group_oriented_quad(
    members: list[dict[str, Any]],
    angle_deg: float | None,
) -> dict[str, Any] | None:
    if angle_deg is None:
        return None
    try:
        angle = float(angle_deg)
    except Exception:
        return None
    bboxes = [
        normalize_bbox(item.get("bbox") if isinstance(item, dict) else item)
        for item in members
    ]
    return _oriented_quad_for_bboxes(
        [bbox for bbox in bboxes if bbox],
        angle,
        source="vector_text_group",
    )


def _bbox_oriented_quad(
    bbox: dict[str, float] | None,
    angle_deg: float | None,
    *,
    source: str,
) -> dict[str, Any] | None:
    if angle_deg is None:
        return None
    try:
        angle = float(angle_deg)
    except Exception:
        return None
    bbox = normalize_bbox(bbox)
    if not bbox:
        return None
    return _oriented_quad_for_bboxes([bbox], angle, source=source)


def _oriented_quad_for_bboxes(
    bboxes: list[dict[str, float]],
    angle: float,
    *,
    source: str,
) -> dict[str, Any] | None:
    rad = math.radians(angle)
    ux, uy = math.cos(rad), math.sin(rad)
    vx, vy = -uy, ux
    u_values: list[float] = []
    v_values: list[float] = []
    for bbox in bboxes:
        x0 = bbox["x"]
        y0 = bbox["y"]
        x1 = x0 + bbox["w"]
        y1 = y0 + bbox["h"]
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            u_values.append(x * ux + y * uy)
            v_values.append(x * vx + y * vy)
    if not u_values or not v_values:
        return None
    min_u, max_u = min(u_values), max(u_values)
    min_v, max_v = min(v_values), max(v_values)
    long_length = max_u - min_u
    short_length = max_v - min_v
    if long_length <= 0.0 or short_length <= 0.0:
        return None
    uc = (min_u + max_u) / 2.0
    vc = (min_v + max_v) / 2.0
    cx = uc * ux + vc * vx
    cy = uc * uy + vc * vy
    return _oriented_quad_from_center(
        cx,
        cy,
        long_length,
        short_length,
        angle,
        source=source,
    )


def _oriented_quad_from_center(
    cx: float,
    cy: float,
    long_length: float,
    short_length: float,
    angle_deg: float,
    *,
    source: str,
) -> dict[str, Any]:
    angle = angle_deg % 180.0
    rad = math.radians(angle)
    ux, uy = math.cos(rad), math.sin(rad)
    vx, vy = -uy, ux
    half_long = long_length / 2.0
    half_short = short_length / 2.0
    points = [
        (cx - ux * half_long - vx * half_short, cy - uy * half_long - vy * half_short),
        (cx + ux * half_long - vx * half_short, cy + uy * half_long - vy * half_short),
        (cx + ux * half_long + vx * half_short, cy + uy * half_long + vy * half_short),
        (cx - ux * half_long + vx * half_short, cy - uy * half_long + vy * half_short),
    ]
    return {
        "coord_space": "page_pdf",
        "points": [[round(x, 3), round(y, 3)] for x, y in points],
        "angle_deg": round(angle, 3),
        "long_length": round(float(long_length), 3),
        "short_length": round(float(short_length), 3),
        "source": source,
    }


def _vector_text_group_angle(members: list[dict[str, Any]]) -> float | None:
    if len(members) < 2:
        return None
    centers = sorted((_bbox_center(item["bbox"]) for item in members), key=lambda pt: pt[0])
    angles: list[float] = []
    for left, right in zip(centers, centers[1:]):
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        if abs(dx) < 8.0:
            continue
        slope = abs(dy / dx)
        if 0.12 <= slope <= 0.75:
            angles.append((math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0)
    if not angles:
        return None
    angles.sort()
    angle = angles[len(angles) // 2]
    if angle <= 10.0 or angle >= 170.0 or 80.0 <= angle <= 100.0:
        return None
    return round(angle, 3)


def _is_track_b_vertical_noise(region: dict[str, Any], bbox: dict[str, float]) -> bool:
    """Filter short-line TrackB fragments that are more like leaders than text."""
    if region.get("source") != "track_b":
        return False
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if width <= 0.0:
        return False
    aspect = height / width
    if height >= 18.0 and aspect >= 1.5 and width <= 20.0:
        return True
    return height >= 22.0 and aspect >= 1.0 and width <= 24.0


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _is_dimension_like_pdf_text_word(text: str) -> bool:
    if not text or len(text) > 28 or not any(ch.isdigit() for ch in text):
        return False
    upper = text.upper()
    if "%" in text or "GB/T" in upper:
        return False
    if re.search(r"[\u3000-\u303f\u3400-\u9fff]", text):
        return False
    if re.match(r"^[A-Z]{2,}\d", upper):
        return False
    if re.fullmatch(r"\d{4,}[-–—]\d{3,}", text):
        return False
    if "/" in text:
        compact = re.sub(r"\s+", "", text)
        if compact.count("/") >= 2:
            return False
        if re.search(r"[A-Za-z]", compact) and not re.match(r"^[RrMmØøΦ⌀]", compact):
            return False
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return "." in text or text.startswith(("+", "-")) or len(text) in {2, 3}
    if re.search(r"[±+\-/°ØøΦ⌀*Xx]", text):
        return True
    if re.match(r"^[RrMm]\d", text) and sum(ch.isalpha() for ch in text) <= 1:
        return True
    return False


def _nearest_text_region(
    bbox: dict[str, float] | None,
    dimline: dict[str, Any] | None,
    text_regions: list[dict[str, Any]],
    *,
    max_dist: float | None = None,
) -> dict[str, Any] | None:
    if not bbox:
        return None
    cx, cy = _bbox_center(bbox)
    if dimline and dimline.get("midpoint"):
        cx, cy = dimline["midpoint"]
    search_dist = max_dist
    if search_dist is None:
        length = float((dimline or {}).get("length") or max(bbox["w"], bbox["h"], 1.0))
        search_dist = max(45.0, min(130.0, length * 0.45))

    best: dict[str, Any] | None = None
    best_dist: float | None = None
    for idx, region in enumerate(text_regions):
        rb = normalize_bbox(region)
        if not rb:
            continue
        rcx, rcy = _bbox_center(rb)
        dist = ((rcx - cx) ** 2 + (rcy - cy) ** 2) ** 0.5
        if dist > search_dist:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = {"index": idx, "region": region, "distance": dist}
    return best


def _dimline_bbox(dimline: dict[str, Any]) -> dict[str, float] | None:
    parts = [normalize_bbox(dimline, pad=3.0)]
    for key in ("arrow_a", "arrow_b"):
        if isinstance(dimline.get(key), dict):
            parts.append(normalize_bbox(dimline[key], pad=1.0))
    return _union_bboxes([p for p in parts if p])


def _dimline_has_arrowheads(dimline: dict[str, Any]) -> bool:
    return any(isinstance(dimline.get(key), dict) for key in ("arrow_a", "arrow_b"))


def _dimline_angle(dimline: dict[str, Any]) -> float | None:
    start = dimline.get("start")
    end = dimline.get("end")
    if not start or not end or len(start) < 2 or len(end) < 2:
        return None
    try:
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (sx, sy, ex, ey)):
        return None
    if sx == ex and sy == ey:
        return None
    return (math.degrees(math.atan2(ey - sy, ex - sx)) + 180.0) % 180.0


def _union_bboxes(bboxes: list[dict[str, float]]) -> dict[str, float] | None:
    if not bboxes:
        return None
    x0 = min(b["x"] for b in bboxes)
    y0 = min(b["y"] for b in bboxes)
    x1 = max(b["x"] + b["w"] for b in bboxes)
    y1 = max(b["y"] + b["h"] for b in bboxes)
    return _xyxy_to_bbox(x0, y0, x1, y1)


def _xyxy_to_bbox(x0: float, y0: float, x1: float, y1: float) -> dict[str, float] | None:
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    lx = min(x0, x1)
    rx = max(x0, x1)
    ty = min(y0, y1)
    by = max(y0, y1)
    return {"x": lx, "y": ty, "w": rx - lx, "h": by - ty}


def _pad_bbox(bbox: dict[str, float] | None, pad: float) -> dict[str, float] | None:
    if not bbox or not pad:
        return bbox
    return {
        "x": bbox["x"] - pad,
        "y": bbox["y"] - pad,
        "w": bbox["w"] + 2 * pad,
        "h": bbox["h"] + 2 * pad,
    }


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {key: round(float(bbox[key]), 3) for key in ("x", "y", "w", "h")}


def _bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return bbox["x"] + bbox["w"] / 2.0, bbox["y"] + bbox["h"] / 2.0


def _point_inside_bbox(bbox: dict[str, float], x: float, y: float) -> bool:
    return bbox["x"] <= x <= bbox["x"] + bbox["w"] and bbox["y"] <= y <= bbox["y"] + bbox["h"]


def _bbox_intersects(a: dict[str, float], b: dict[str, float]) -> bool:
    return not (
        a["x"] + a["w"] < b["x"]
        or b["x"] + b["w"] < a["x"]
        or a["y"] + a["h"] < b["y"]
        or b["y"] + b["h"] < a["y"]
    )


def _vector_context_metrics_for_bbox(
    bbox: dict[str, float],
    segments: list[dict[str, Any]] | _PreparedVectorContextSegments,
    *,
    pad: float,
    inner_pad: float,
) -> dict[str, Any] | None:
    if not segments:
        return None
    outer = _pad_bbox(bbox, max(0.0, pad))
    inner = _pad_bbox(bbox, max(0.0, inner_pad))
    if not outer or not inner:
        return None
    context_area = max(1.0, _bbox_area(outer) - _bbox_area(inner))
    total_length = 0.0
    segment_count = 0
    long_segment_count = 0
    if isinstance(segments, _PreparedVectorContextSegments):
        prepared_rows = segments.query(outer)
    else:
        prepared_rows = []
        for ordinal, segment in enumerate(segments):
            points = _segment_points(segment)
            if points is None:
                continue
            p0, p1 = points
            full_length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if full_length <= 0.0:
                continue
            prepared_rows.append(
                (ordinal, p0, p1, full_length, _segment_bbox(p0, p1))
            )
    for _, p0, p1, full_length, segment_bbox in prepared_rows:
        if not _bbox_intersects(segment_bbox, outer):
            continue
        outer_length = _line_length_inside_bbox(p0, p1, outer)
        if outer_length <= 0.0:
            continue
        inner_length = (
            _line_length_inside_bbox(p0, p1, inner)
            if _bbox_intersects(segment_bbox, inner)
            else 0.0
        )
        ring_length = max(0.0, outer_length - inner_length)
        if ring_length <= 0.001:
            continue
        total_length += ring_length
        segment_count += 1
        if full_length >= 12.0:
            long_segment_count += 1
    density = total_length / context_area * 1000.0
    return {
        "length": round(total_length, 3),
        "area": round(context_area, 3),
        "density_x1000": round(density, 3),
        "segment_count": segment_count,
        "long_segment_count": long_segment_count,
    }


def _segment_points(
    segment: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    try:
        if {"x0", "y0", "x1", "y1"} <= set(segment):
            return (
                (float(segment["x0"]), float(segment["y0"])),
                (float(segment["x1"]), float(segment["y1"])),
            )
        if {"start", "end"} <= set(segment):
            sx, sy = segment.get("start") or (0.0, 0.0)
            ex, ey = segment.get("end") or (0.0, 0.0)
            return ((float(sx), float(sy)), (float(ex), float(ey)))
        bbox = normalize_bbox(segment.get("bbox") if isinstance(segment, dict) else None)
        if bbox:
            x0 = float(bbox["x"])
            y0 = float(bbox["y"])
            x1 = x0 + float(bbox["w"])
            y1 = y0 + float(bbox["h"])
            if bbox["w"] >= bbox["h"]:
                cy = y0 + bbox["h"] / 2.0
                return ((x0, cy), (x1, cy))
            cx = x0 + bbox["w"] / 2.0
            return ((cx, y0), (cx, y1))
    except (TypeError, ValueError):
        return None
    return None


def _segment_bbox(
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> dict[str, float]:
    return _xyxy_to_bbox(p0[0], p0[1], p1[0], p1[1])


def _line_length_inside_bbox(
    p0: tuple[float, float],
    p1: tuple[float, float],
    bbox: dict[str, float],
) -> float:
    clipped = _clip_segment_to_bbox(p0, p1, bbox)
    if clipped is None:
        return 0.0
    c0, c1 = clipped
    return math.hypot(c1[0] - c0[0], c1[1] - c0[1])


def _clip_segment_to_bbox(
    p0: tuple[float, float],
    p1: tuple[float, float],
    bbox: dict[str, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    rx0 = bbox["x"]
    ry0 = bbox["y"]
    rx1 = bbox["x"] + bbox["w"]
    ry1 = bbox["y"] + bbox["h"]
    u0 = 0.0
    u1 = 1.0
    for p, q in (
        (-dx, x0 - rx0),
        (dx, rx1 - x0),
        (-dy, y0 - ry0),
        (dy, ry1 - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        t = q / p
        if p < 0.0:
            if t > u1:
                return None
            if t > u0:
                u0 = t
        else:
            if t < u0:
                return None
            if t < u1:
                u1 = t
    return ((x0 + u0 * dx, y0 + u0 * dy), (x0 + u1 * dx, y0 + u1 * dy))


def _center_inside_any(
    bbox: dict[str, float],
    regions: list[dict[str, float]],
) -> bool:
    cx, cy = _bbox_center(bbox)
    return any(
        region["x"] <= cx <= region["x"] + region["w"]
        and region["y"] <= cy <= region["y"] + region["h"]
        for region in regions
    )


def _expand_title_region(
    bbox: dict[str, float],
    *,
    page_width: float,
    page_height: float,
) -> dict[str, float]:
    """Cover footer text attached to bottom-right title blocks."""
    if page_width <= 0.0 or page_height <= 0.0:
        return bbox
    x = float(bbox.get("x") or 0.0)
    y = float(bbox.get("y") or 0.0)
    w = float(bbox.get("w") or 0.0)
    h = float(bbox.get("h") or 0.0)
    cx = x + w / 2.0
    cy = y + h / 2.0
    if cx < page_width * 0.45 or cy < page_height * 0.65:
        return bbox
    right = max(x + w, page_width)
    bottom = max(y + h, page_height)
    return {
        "x": x,
        "y": y,
        "w": right - x,
        "h": bottom - y,
    }


def _overlaps_page(bbox: dict[str, float], page_width: float, page_height: float) -> bool:
    if page_width <= 0.0 or page_height <= 0.0:
        return True
    return not (
        bbox["x"] + bbox["w"] < 0.0
        or bbox["y"] + bbox["h"] < 0.0
        or bbox["x"] > page_width
        or bbox["y"] > page_height
    )


def _same_candidate_bucket(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("type") != right.get("type"):
        return False
    if left.get("subtype") == right.get("subtype"):
        return True
    if left.get("type") == "dimension":
        subtypes = {left.get("subtype"), right.get("subtype")}
        if "unknown" in subtypes:
            return subtypes == {"unknown"}
        text_subtypes = {"text_region", "table_text", "pdf_text", "vector_text_phrase"}
        return bool(subtypes & text_subtypes) and bool(
            subtypes & {"linear", "diameter", "pdf_text", "vector_text_phrase"}
        )
    return False


def _bbox_area(box: dict[str, float]) -> float:
    return max(0.0, float(box.get("w") or 0.0)) * max(0.0, float(box.get("h") or 0.0))


def _bbox_intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    ax1 = a["x"] + a["w"]
    ay1 = a["y"] + a["h"]
    bx1 = b["x"] + b["w"]
    by1 = b["y"] + b["h"]
    ix0 = max(a["x"], b["x"])
    iy0 = max(a["y"], b["y"])
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    inter = _bbox_intersection_area(a, b)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
