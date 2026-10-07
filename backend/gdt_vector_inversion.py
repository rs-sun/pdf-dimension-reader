"""
gdt_vector_inversion.py — PDF vector inversion for GD&T symbols.

NOTE:
This module was revised to use PyMuPDF ``fitz.Page.get_drawings()`` as the
single vector source and to follow a two-stage recognition flow:
topology prefilter first, then symbol-specific geometric inversion.  Dimension
prefix glyphs such as diameter / SR / S∅ / square are intentionally excluded
from the public dispatcher because those belong to dimension-prefix extraction
(`pdf_analyzer_extract.extract_diameter_glyphs` already handles ∅).

Current matcher inventory after the revision:
- Kept and routed: 12 core GD&T symbols, 7 circled modifiers, 2 datum symbols.
- Deprecated and not routed: diameter, spherical_radius, spherical_diameter,
  square.  Their helper matchers remain below for future prefix-module reuse.
- Rewritten data source: all active matchers now consume primitives derived
  from fitz drawing path items (`l`, `c`, `re`) plus optional native text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


SYMBOL_DEFS: list[tuple[str, str]] = [
    ("position", "⊕"),
    ("perpendicularity", "⊥"),
    ("parallelism", "∥"),
    ("coaxiality", "◎"),
    ("roundness", "○"),
    ("cylindricity", "⌭"),
    ("straightness", "—"),
    ("angularity", "∠"),
    ("surface_profile", "⌓"),
    ("line_profile", "⌒"),
    ("flatness", "▱"),
    ("symmetry", "≡"),
    ("mmc", "Ⓜ"),
    ("lmc", "Ⓛ"),
    ("projected", "Ⓟ"),
    ("free_state", "Ⓕ"),
    ("tangent", "Ⓣ"),
    ("rfs", "Ⓢ"),
    ("continuous", "Ⓒ"),
    ("datum_filled", "▲"),
    ("datum_hollow", "△"),
]

CLASS_NAMES = [name for name, _ in SYMBOL_DEFS]
CLASS_SYMBOLS = {name: glyph for name, glyph in SYMBOL_DEFS}
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DEPRECATED_PREFIX_SYMBOLS = {
    # DEPRECATED: prefix glyph routing is not part of the active GD&T inverter.
    # Diameter is handled by pdf_analyzer_extract.extract_diameter_glyphs; keep
    # these names only as historical helper inventory.
    "diameter": "∅",
    "spherical_radius": "SR",
    "spherical_diameter": "S∅",
    "square": "□",
}


@dataclass(frozen=True)
class InversionThresholds:
    angle_tol_deg: float = 10.0
    parallel_tol_deg: float = 8.0
    perpendicular_tol_deg: float = 12.0
    circle_aspect_tol: float = 0.22
    center_tol_ratio: float = 0.16
    endpoint_tol_ratio: float = 0.18
    min_symbol_size: float = 2.0
    min_path_segment_size: float = 0.6
    border_tol: float = 1.25
    border_length_ratio: float = 0.72
    tiny_curve_density_threshold: float = 0.003
    min_match_confidence: float = 0.60


class GDTVectorInverter:
    """Recognize GD&T symbols from fitz vector path topology."""

    def __init__(
        self,
        thresholds: InversionThresholds | None = None,
        *,
        drawing_snapshot=None,
        fail_on_error: bool = False,
    ):
        self.t = thresholds or InversionThresholds()
        self._fail_on_error = bool(fail_on_error)
        self._drawing_snapshot = (
            tuple(drawing_snapshot)
            if drawing_snapshot is not None
            else None
        )
        self._drawings_cache: dict[tuple[int, int], list] = {}
        self._raw_text_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._words_cache: dict[tuple[int, int], list] = {}

    def invert_in_frame(self, fitz_page, frame_bbox, frame_compartments=None) -> list[dict[str, Any]]:
        """Invert GD&T symbols inside a feature-control frame (classes 0-18)."""
        frame_region = self._normalize_bbox(frame_bbox)
        if not frame_region:
            return []

        regions = self._compartment_regions(frame_region, frame_compartments)
        hits: list[dict[str, Any]] = []
        for comp_index, region in regions:
            region = self._inset_region(region)
            drawings = self._collect_region_drawings(fitz_page, region)
            signature = self._compute_topology_signature(drawings)
            candidate_classes = [
                cls for cls in self._dispatch_by_topology(signature, drawings)
                if 0 <= cls <= 18
            ]
            if comp_index == 0:
                candidate_classes = [cls for cls in candidate_classes if 0 <= cls <= 11]
            elif comp_index is not None:
                candidate_classes = [cls for cls in candidate_classes if 12 <= cls <= 18]
            if not candidate_classes:
                continue

            best = self._best_candidate(drawings, candidate_classes)
            if not best or best["confidence"] < self.t.min_match_confidence:
                continue
            hits.append({
                **best,
                "compartment_index": comp_index,
                "topology_signature": signature,
                "candidate_classes": candidate_classes,
            })
        return hits

    def invert_datum(self, fitz_page, region_bbox) -> list[dict[str, Any]]:
        """Invert datum triangle symbols inside a candidate region (classes 19/20)."""
        region = self._normalize_bbox(region_bbox)
        if not region:
            return []
        drawings = self._collect_region_drawings(fitz_page, self._inset_region(region, frac=0.02))
        signature = self._compute_topology_signature(drawings)
        candidate_classes = [
            cls for cls in self._dispatch_by_topology(signature, drawings)
            if cls in {19, 20}
        ]
        hits = []
        for cls in candidate_classes:
            conf = self._match_by_class(cls, drawings)
            if conf >= self.t.min_match_confidence:
                hits.append(self._result(cls, conf))
        hits.sort(key=lambda r: -r["confidence"])
        return hits

    def invert(self, fitz_page, gdt_frame_pdf_bbox) -> dict[str, Any]:
        """Deprecated compatibility wrapper: return the first frame hit."""
        hits = self.invert_in_frame(fitz_page, gdt_frame_pdf_bbox)
        if hits:
            hit = dict(hits[0])
            hit.pop("compartment_index", None)
            hit.pop("topology_signature", None)
            hit.pop("candidate_classes", None)
            return hit
        return self._unknown()

    # ------------------------------------------------------------------
    # Stage 1: topology signature
    # ------------------------------------------------------------------

    def _compute_topology_signature(self, region_drawings) -> dict[str, Any]:
        lines = region_drawings["lines"]
        circles = region_drawings["circles"]
        arcs = region_drawings["arcs"]
        polygons = region_drawings["polygons"]
        region = region_drawings["region"]

        concentric = 0
        for i, outer in enumerate(circles):
            for inner in circles[i + 1:]:
                big, small = (outer, inner) if outer["r"] >= inner["r"] else (inner, outer)
                ratio = big["r"] / max(small["r"], 1e-6)
                center_dist = self._dist((big["cx"], big["cy"]), (small["cx"], small["cy"]))
                if 1.35 <= ratio <= 2.6 and center_dist <= max(small["r"] * 0.12, 0.8):
                    concentric += 1

        filled_triangles = sum(1 for p in polygons if self._is_triangle_polygon(p) and p.get("filled"))
        hollow_poly_triangles = sum(1 for p in polygons if self._is_triangle_polygon(p) and not p.get("filled"))
        hollow_line_triangles = 1 if self._has_triangle_from_lines(lines) else 0

        parallel_pairs = 0
        perpendicular_pairs = 0
        acute_pairs = 0
        for i, a in enumerate(lines):
            for b in lines[i + 1:]:
                if self._parallel(a, b):
                    parallel_pairs += 1
                if self._angle_delta(a["angle"], b["angle"]) >= 90 - self.t.perpendicular_tol_deg:
                    if self._segments_touch_or_intersect(a, b):
                        perpendicular_pairs += 1
                angle = self._angle_delta(a["angle"], b["angle"])
                if 30 <= angle <= 60 and self._segments_touch_or_intersect(a, b):
                    acute_pairs += 1

        # Count only open curve primitives.  Full-circle bezier quarters are
        # already promoted into `closed_circles` and must not make a plain
        # roundness symbol look like a circled text glyph.
        tiny_curves = sum(1 for a in arcs if max(a["w"], a["h"]) <= 8.0)
        area = max(region["w"] * region["h"], 1.0)
        return {
            "closed_circles": len(circles),
            "concentric_circle_pairs": concentric,
            "open_arcs": len([a for a in arcs if not a.get("closed")]),
            "line_segments": len(lines),
            "filled_triangles": filled_triangles,
            "hollow_triangles": max(hollow_poly_triangles, hollow_line_triangles),
            "parallel_pairs": parallel_pairs,
            "perpendicular_pairs": perpendicular_pairs,
            "acute_angle_pairs": acute_pairs,
            "tiny_curves_density": tiny_curves / area,
        }

    # ------------------------------------------------------------------
    # Stage 2: topology dispatcher
    # ------------------------------------------------------------------

    def _dispatch_by_topology(self, signature, region_drawings) -> list[int]:
        if signature["filled_triangles"] >= 1:
            return [19]
        if signature["hollow_triangles"] >= 1:
            return [20]
        if signature["concentric_circle_pairs"] >= 1:
            return [3]

        if signature["closed_circles"] == 1:
            if self._has_outer_parallel_tangents(region_drawings):
                return [5]
            if self._has_center_cross(region_drawings):
                return [0]
            if self._has_circled_letter_evidence(region_drawings, signature):
                return [12, 13, 14, 15, 16, 17, 18]
            return [4]

        if signature["closed_circles"] == 0:
            if self._has_polyline_position_signature(region_drawings):
                return [0]
            if not region_drawings.get("image_blocks") and self._has_parallelogram_component(region_drawings):
                return [10]
            if self._has_three_equal_parallel_lines(region_drawings):
                return [11]
            if self._match_perpendicularity(region_drawings) >= self.t.min_match_confidence:
                return [1]
            if self._match_parallelism(region_drawings) >= self.t.min_match_confidence:
                return [2]
            if not region_drawings.get("image_blocks") and self._looks_like_flatness_polyline(region_drawings["lines"]):
                return [10]
            if self._has_polyline_surface_profile(region_drawings):
                return [8]
            if self._has_polyline_line_profile(region_drawings):
                return [9]
            if signature["open_arcs"] >= 1:
                if self._has_arc_position_signature(region_drawings):
                    return [0]
                if self._has_arc_chord(region_drawings):
                    return [8]
                return [9]
            if signature["line_segments"] == 1:
                return [6]
            if signature["line_segments"] == 2:
                if signature["parallel_pairs"] >= 1:
                    return [2]
                if signature["perpendicular_pairs"] >= 1:
                    return [1]
                if signature["acute_angle_pairs"] >= 1:
                    return [7]
        return []

    # ------------------------------------------------------------------
    # Stage 3: geometric inversion matchers
    # ------------------------------------------------------------------

    def _best_candidate(self, drawings, candidate_classes: list[int]) -> dict[str, Any] | None:
        scored = [(cls, self._match_by_class(cls, drawings)) for cls in candidate_classes]
        scored = [(cls, conf) for cls, conf in scored if conf > 0]
        if not scored:
            return None
        cls, conf = max(scored, key=lambda item: item[1])
        return self._result(cls, conf)

    def _match_by_class(self, cls: int, drawings) -> float:
        matchers = {
            0: self._match_position,
            1: self._match_perpendicularity,
            2: self._match_parallelism,
            3: self._match_coaxiality,
            4: self._match_roundness,
            5: self._match_cylindricity,
            6: self._match_straightness,
            7: self._match_angularity,
            8: self._match_surface_profile,
            9: self._match_line_profile,
            10: self._match_flatness,
            11: self._match_symmetry,
            12: lambda d: self._match_circled_letter(d, "M"),
            13: lambda d: self._match_circled_letter(d, "L"),
            14: lambda d: self._match_circled_letter(d, "P"),
            15: lambda d: self._match_circled_letter(d, "F"),
            16: lambda d: self._match_circled_letter(d, "T"),
            17: lambda d: self._match_circled_letter(d, "S"),
            18: lambda d: self._match_circled_letter(d, "C"),
            19: lambda d: self._match_datum(d, filled=True),
            20: lambda d: self._match_datum(d, filled=False),
        }
        matcher = matchers.get(cls)
        return float(matcher(drawings)) if matcher else 0.0

    def _match_position(self, d) -> float:
        for c in d["circles"]:
            tol = max(c["r"] * self.t.center_tol_ratio, 1.0)
            hs = [ln for ln in d["lines"] if self._is_horizontal(ln) and self._line_through_point(ln, c["cx"], c["cy"], tol)]
            vs = [ln for ln in d["lines"] if self._is_vertical(ln) and self._line_through_point(ln, c["cx"], c["cy"], tol)]
            if any(h["length"] >= c["r"] * 1.2 for h in hs) and any(v["length"] >= c["r"] * 1.2 for v in vs):
                return 0.93
        if self._has_arc_position_signature(d):
            return 0.87
        if self._has_polyline_position_signature(d):
            return 0.86
        return 0.0

    def _match_perpendicularity(self, d) -> float:
        hs = [ln for ln in d["lines"] if self._is_horizontal(ln) and ln["length"] >= 3.0]
        vs = [ln for ln in d["lines"] if self._is_vertical(ln) and ln["length"] >= 3.0]
        for h in hs:
            hx, hy = self._midpoint(h)
            for v in vs:
                ratio = v["length"] / max(h["length"], 1e-6)
                if 0.55 <= ratio <= 1.55:
                    endpoint_dist = min(self._dist((hx, hy), p) for p in self._endpoints(v))
                    if endpoint_dist <= max(h["length"] * 0.16, 1.2):
                        return 0.90
        return 0.0

    def _match_parallelism(self, d) -> float:
        if self._has_three_equal_parallel_lines(d):
            return 0.0
        lines = d["lines"]
        for i, a in enumerate(lines):
            for b in lines[i + 1:]:
                if not self._parallel(a, b):
                    continue
                axis_aligned = (
                    self._is_horizontal(a) or self._is_vertical(a)
                    or self._is_horizontal(b) or self._is_vertical(b)
                )
                if axis_aligned and len(lines) != 2:
                    continue
                length_ratio = min(a["length"], b["length"]) / max(a["length"], b["length"])
                spacing_ratio = self._line_midpoint_distance_perp(a, b) / max((a["length"] + b["length"]) / 2, 1e-6)
                if length_ratio >= 0.72 and 0.18 <= spacing_ratio <= 0.8:
                    return min(0.94, 0.84 + 0.10 * length_ratio)
        return 0.0

    def _match_coaxiality(self, d) -> float:
        circles = sorted(d["circles"], key=lambda c: c["r"], reverse=True)
        for i, outer in enumerate(circles):
            for inner in circles[i + 1:]:
                ratio = outer["r"] / max(inner["r"], 1e-6)
                center_dist = self._dist((outer["cx"], outer["cy"]), (inner["cx"], inner["cy"]))
                if 1.35 <= ratio <= 2.6 and center_dist <= max(inner["r"] * 0.12, 0.8):
                    return 0.95
        return 0.0

    def _match_roundness(self, d) -> float:
        if len(d["circles"]) == 1 and not d["lines"] and not d["arcs"] and not d["polygons"]:
            return 0.88
        return 0.0

    def _match_cylindricity(self, d) -> float:
        for c in d["circles"]:
            verticals = [ln for ln in d["lines"] if self._is_vertical(ln)]
            for i, left in enumerate(verticals):
                for right in verticals[i + 1:]:
                    if left["x0"] > right["x0"]:
                        left, right = right, left
                    len_ok = 0.65 <= left["length"] / max(2 * c["r"], 1e-6) <= 1.45
                    len_ok = len_ok and 0.65 <= right["length"] / max(2 * c["r"], 1e-6) <= 1.45
                    y_ok = max(abs(self._midpoint(left)[1] - c["cy"]), abs(self._midpoint(right)[1] - c["cy"])) <= c["r"] * 0.4
                    tangent = abs(abs(left["x0"] - c["cx"]) - c["r"]) <= max(c["r"] * 0.45, 1.4)
                    tangent = tangent and abs(abs(right["x0"] - c["cx"]) - c["r"]) <= max(c["r"] * 0.45, 1.4)
                    if len_ok and y_ok and tangent:
                        return 0.89
        return 0.0

    def _match_straightness(self, d) -> float:
        return 0.86 if len(d["lines"]) == 1 and self._is_horizontal(d["lines"][0]) else 0.0

    def _match_angularity(self, d) -> float:
        for i, a in enumerate(d["lines"]):
            for b in d["lines"][i + 1:]:
                if self._segments_touch_or_intersect(a, b):
                    angle = self._angle_delta(a["angle"], b["angle"])
                    if 30 <= angle <= 60:
                        return 0.89
        return 0.0

    def _match_surface_profile(self, d) -> float:
        for arc in d["arcs"]:
            if arc["w"] < self.t.min_symbol_size or arc["h"] < 0.5:
                continue
            chord = self._find_chord_line(d["lines"], arc["start"], arc["end"])
            if chord and self._is_horizontal(chord):
                return 0.89
        if self._has_polyline_surface_profile(d):
            return 0.84
        return 0.0

    def _match_line_profile(self, d) -> float:
        for arc in d["arcs"]:
            if arc.get("closed") or arc["w"] < self.t.min_symbol_size or arc["h"] < 0.5:
                continue
            if not self._find_chord_line(d["lines"], arc["start"], arc["end"]):
                return 0.85
        if self._has_polyline_line_profile(d):
            return 0.82
        return 0.0

    def _match_flatness(self, d) -> float:
        if d.get("image_blocks"):
            return 0.0
        if self._has_parallelogram_component(d):
            return 0.88
        if self._looks_like_flatness_polyline(d["lines"]):
            return 0.78
        return 0.0

    def _match_symmetry(self, d) -> float:
        return 0.90 if self._has_three_equal_parallel_lines(d) else 0.0

    def _match_circled_letter(self, d, letter: str) -> float:
        circle = max(d["circles"], key=lambda c: c["r"], default=None)
        if not circle:
            return 0.0
        text = self._chars_text(d).upper()
        if letter in text or CLASS_SYMBOLS.get(self._modifier_name(letter), "") in text:
            return 0.94
        inner_lines = [ln for ln in d["lines"] if self._line_inside_bbox(ln, circle, margin=-circle["r"] * 0.08)]
        inner_arcs = [a for a in d["arcs"] if self._bbox_inside(a, circle, margin=circle["r"] * 0.20)]
        checks = {
            "M": self._letter_m(inner_lines),
            "L": self._letter_l(inner_lines),
            "P": self._letter_p(inner_lines, inner_arcs),
            "F": self._letter_f(inner_lines),
            "T": self._letter_t(inner_lines),
            "S": len(inner_arcs) >= 2,
            "C": len(inner_arcs) >= 1 and len(inner_lines) <= 1,
        }
        return 0.84 if checks.get(letter) else 0.0

    def _match_datum(self, d, filled: bool) -> float:
        for poly in d["polygons"]:
            if self._is_triangle_polygon(poly) and bool(poly.get("filled")) is filled:
                return 0.92
        if not filled and self._has_triangle_from_lines(d["lines"]):
            return 0.86
        return 0.0

    # ------------------------------------------------------------------
    # fitz drawing extraction
    # ------------------------------------------------------------------

    def _collect_region_drawings(self, fitz_page, region: dict[str, float]) -> dict[str, Any]:
        raw_drawings = []
        for draw_idx, drawing in enumerate(self._page_drawings(fitz_page)):
            filtered = []
            for item_idx, item in enumerate(drawing.get("items") or []):
                ib = self._item_bbox(item)
                if not ib or not self._bbox_overlaps(ib, region):
                    continue
                filtered.append((item_idx, item))
            if filtered:
                raw_drawings.append({
                    "draw_idx": draw_idx,
                    "items": [item for _, item in filtered],
                    "item_indices": [idx for idx, _ in filtered],
                    "fill": drawing.get("fill"),
                    "stroke": drawing.get("color"),
                    "width": drawing.get("width", 0) or 0,
                    "rect": drawing.get("rect"),
                    "closePath": bool(drawing.get("closePath")),
                })

        primitives = self._extract_primitives(raw_drawings, region)
        primitives["raw_drawings"] = raw_drawings
        primitives["chars"] = self._collect_native_chars(fitz_page, region)
        primitives["image_blocks"] = self._collect_image_blocks(fitz_page, region)
        primitives["region"] = region
        return primitives

    def _page_drawings(self, fitz_page) -> list:
        key = self._page_cache_key(fitz_page)
        if key not in self._drawings_cache:
            self._drawings_cache[key] = (
                list(self._drawing_snapshot)
                if self._drawing_snapshot is not None
                else fitz_page.get_drawings() or []
            )
        return self._drawings_cache[key]

    def _page_raw_text(self, fitz_page) -> dict[str, Any]:
        key = self._page_cache_key(fitz_page)
        if key not in self._raw_text_cache:
            self._raw_text_cache[key] = fitz_page.get_text("rawdict") or {}
        return self._raw_text_cache[key]

    def _page_words(self, fitz_page) -> list:
        key = self._page_cache_key(fitz_page)
        if key not in self._words_cache:
            self._words_cache[key] = fitz_page.get_text("words") or []
        return self._words_cache[key]

    def _page_cache_key(self, fitz_page) -> tuple[int, int]:
        parent = getattr(fitz_page, "parent", None)
        page_number = int(getattr(fitz_page, "number", -1))
        return (id(parent) if parent is not None else id(fitz_page), page_number)

    def _extract_primitives(self, raw_drawings, region) -> dict[str, list]:
        lines: list[dict[str, Any]] = []
        circles: list[dict[str, Any]] = []
        arcs: list[dict[str, Any]] = []
        polygons: list[dict[str, Any]] = []
        rects: list[dict[str, Any]] = []

        for raw in raw_drawings:
            draw_lines, draw_curves = [], []
            for item in raw["items"]:
                op = item[0]
                if op == "l":
                    line = self._line_from_points(self._as_point(item[1]), self._as_point(item[2]), raw)
                    if line and not self._is_region_border_line(line, region):
                        draw_lines.append(line)
                elif op == "re":
                    rb = self._rect_from_item(item)
                    if rb:
                        rects.append(rb)
                        for edge in self._rect_edges(rb, source="re", meta=raw):
                            if not self._is_region_border_line(edge, region):
                                draw_lines.append(edge)
                elif op == "qu":
                    for edge in self._quad_edges(item, raw):
                        if edge and not self._is_region_border_line(edge, region):
                            draw_lines.append(edge)
                elif op == "c":
                    curve = self._curve_from_item(item, raw)
                    if curve:
                        draw_curves.append(curve)

            circle = self._circle_from_curves(draw_curves, raw)
            if circle:
                circles.append(circle)
            else:
                arcs.extend(draw_curves)
            polygon = self._polygon_from_lines(draw_lines, raw)
            if polygon:
                polygons.append(polygon)
            lines.extend(draw_lines)

        return {
            "lines": self._dedup_lines(lines),
            "circles": self._dedup_circles(circles),
            "arcs": arcs,
            "polygons": polygons,
            "rects": rects,
        }

    def _collect_native_chars(self, fitz_page, region) -> list[dict[str, Any]]:
        chars = []
        try:
            raw = self._page_raw_text(fitz_page)
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        for ch in span.get("chars", []):
                            bbox = self._bbox_from_seq(ch.get("bbox"))
                            if bbox and self._bbox_center_inside(bbox, region, margin=1.0):
                                chars.append({**bbox, "text": ch.get("c", "")})
        except Exception:
            if self._fail_on_error:
                raise
            chars = []
        if chars:
            return chars
        try:
            for word in self._page_words(fitz_page):
                x0, y0, x1, y1, text, *_ = word
                bbox = self._xyxy_to_bbox(x0, y0, x1, y1)
                if self._bbox_center_inside(bbox, region, margin=1.0):
                    chars.append({**bbox, "text": str(text)})
        except Exception:
            if self._fail_on_error:
                raise
            return []
        return chars

    def _collect_image_blocks(self, fitz_page, region) -> list[dict[str, Any]]:
        blocks = []
        clip = (region["x"], region["y"], region["x"] + region["w"], region["y"] + region["h"])
        try:
            raw = fitz_page.get_text("dict", clip=clip)
            for block in raw.get("blocks", []):
                if block.get("type") != 1:
                    continue
                bbox = self._bbox_from_seq(block.get("bbox"))
                if bbox:
                    overlap_w = min(region["x"] + region["w"], bbox["x"] + bbox["w"]) - max(region["x"], bbox["x"])
                    overlap_h = min(region["y"] + region["h"], bbox["y"] + bbox["h"]) - max(region["y"], bbox["y"])
                    if overlap_w >= region["w"] * 0.5 or overlap_h >= region["h"] * 0.5:
                        blocks.append(bbox)
        except Exception:
            if self._fail_on_error:
                raise
            return []
        return blocks

    # ------------------------------------------------------------------
    # Deprecated prefix matchers: moved to dimension-prefix recognition.
    # ------------------------------------------------------------------

    def _match_diameter(self, d) -> float:
        """Deprecated: diameter prefix belongs to pdf_analyzer_extract."""
        for c in d["circles"]:
            tol = max(c["r"] * 0.22, 1.0)
            for ln in d["lines"]:
                if not self._is_horizontal(ln) and not self._is_vertical(ln):
                    if self._line_through_point(ln, c["cx"], c["cy"], tol):
                        return 0.90
        return 0.0

    def _match_spherical_radius(self, d) -> float:
        """Deprecated: SR is a dimension prefix, not a GD&T frame symbol."""
        text = self._chars_text(d).upper().replace(" ", "")
        return 0.92 if "SR" in text else 0.0

    def _match_spherical_diameter(self, d) -> float:
        """Deprecated: S∅ is a dimension prefix, not a GD&T frame symbol."""
        return 0.90 if ("S" in self._chars_text(d).upper() and self._match_diameter(d) >= 0.6) else 0.0

    def _match_square(self, d) -> float:
        """Deprecated: square prefix belongs to dimension-prefix recognition."""
        return 0.88 if self._looks_like_square(d) else 0.0

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _has_center_cross(self, d) -> bool:
        for c in d["circles"]:
            tol = max(c["r"] * self.t.center_tol_ratio, 1.0)
            has_h = any(self._is_horizontal(ln) and self._line_through_point(ln, c["cx"], c["cy"], tol) for ln in d["lines"])
            has_v = any(self._is_vertical(ln) and self._line_through_point(ln, c["cx"], c["cy"], tol) for ln in d["lines"])
            if has_h and has_v:
                return True
        return False

    def _has_outer_parallel_tangents(self, d) -> bool:
        verticals = [ln for ln in d["lines"] if self._is_vertical(ln)]
        non_verticals = [ln for ln in d["lines"] if not self._is_vertical(ln)]
        if len(verticals) != 2 or non_verticals:
            return False
        return self._match_cylindricity(d) >= self.t.min_match_confidence

    def _has_circled_letter_evidence(self, d, signature) -> bool:
        if self._chars_text(d).strip():
            return True
        inner_line_count = len(d["lines"])
        inner_arc_count = len(d["arcs"])
        return (
            signature["tiny_curves_density"] > self.t.tiny_curve_density_threshold
            or inner_line_count >= 2
            or inner_arc_count >= 1
        )

    def _has_arc_chord(self, d) -> bool:
        return any(self._find_chord_line(d["lines"], arc["start"], arc["end"]) for arc in d["arcs"])

    def _has_arc_position_signature(self, d) -> bool:
        if len(d["arcs"]) < 2:
            return False
        bbox = self._bbox_from_points([pt for arc in d["arcs"] for pt in (arc["start"], arc["end"])])
        cx, cy = bbox["x"] + bbox["w"] / 2, bbox["y"] + bbox["h"] / 2
        tol = max(max(bbox["w"], bbox["h"]) * 0.18, 1.2)
        has_h = any(self._is_horizontal(ln) and self._line_through_point(ln, cx, cy, tol) for ln in d["lines"])
        has_v = any(self._is_vertical(ln) and self._line_through_point(ln, cx, cy, tol) for ln in d["lines"])
        return has_h and has_v

    def _has_polyline_position_signature(self, d) -> bool:
        lines = d["lines"]
        hs = [ln for ln in lines if self._is_horizontal(ln) and ln["length"] >= 6.0]
        vs = [ln for ln in lines if self._is_vertical(ln) and ln["length"] >= 6.0]
        if not hs or not vs:
            return False
        short_lines = [ln for ln in lines if ln["length"] <= 5.0]
        if len(short_lines) < 8:
            return False
        for h in hs:
            for v in vs:
                cx, cy = self._midpoint(v)[0], self._midpoint(h)[1]
                tol = max(min(h["length"], v["length"]) * 0.12, 1.2)
                if not self._line_through_point(h, cx, cy, tol) or not self._line_through_point(v, cx, cy, tol):
                    continue
                local = [
                    ln for ln in short_lines
                    if abs(self._midpoint(ln)[0] - cx) <= max(h["length"] * 0.45, 4.0)
                    and abs(self._midpoint(ln)[1] - cy) <= max(v["length"] * 0.45, 4.0)
                ]
                if len(local) < 8:
                    continue
                bbox = self._bbox_from_points([pt for ln in local for pt in self._endpoints(ln)])
                if bbox["w"] < 4.0 or bbox["h"] < 4.0:
                    continue
                aspect = abs(bbox["w"] - bbox["h"]) / max(bbox["w"], bbox["h"], 1e-6)
                center_ok = abs((bbox["x"] + bbox["w"] / 2) - cx) <= max(bbox["w"] * 0.25, 1.5)
                center_ok = center_ok and abs((bbox["y"] + bbox["h"] / 2) - cy) <= max(bbox["h"] * 0.25, 1.5)
                if aspect <= 0.38 and center_ok:
                    return True
        return False

    def _has_polyline_surface_profile(self, d) -> bool:
        for comp in self._line_components(d["lines"]):
            if len(comp) < 5:
                continue
            bbox = self._bbox_from_points([pt for ln in comp for pt in self._endpoints(ln)])
            if bbox["w"] < 7.0 or bbox["h"] < 3.0:
                continue
            horizontals = [ln for ln in comp if self._is_horizontal(ln, tol=12)]
            if not horizontals:
                continue
            chord = max(horizontals, key=lambda ln: ln["length"])
            hy = self._midpoint(chord)[1]
            if chord["length"] < bbox["w"] * 0.72 or hy < bbox["y"] + bbox["h"] * 0.62:
                continue
            upper = [ln for ln in comp if ln is not chord and self._midpoint(ln)[1] <= hy + 0.8]
            if len(upper) >= 4:
                return True
        return False

    def _has_polyline_line_profile(self, d) -> bool:
        for comp in self._line_components(d["lines"]):
            if len(comp) < 5:
                continue
            bbox = self._bbox_from_points([pt for ln in comp for pt in self._endpoints(ln)])
            if bbox["w"] < 6.0 or bbox["h"] < 3.0:
                continue
            aspect = bbox["w"] / max(bbox["h"], 1e-6)
            if not 0.8 <= aspect <= 4.5:
                continue
            if self._line_graph_closed(comp) and abs(bbox["w"] - bbox["h"]) / max(bbox["w"], bbox["h"]) <= 0.45:
                continue
            long_h = [
                ln for ln in comp
                if self._is_horizontal(ln, tol=12)
                and ln["length"] >= bbox["w"] * 0.65
                and self._midpoint(ln)[1] >= bbox["y"] + bbox["h"] * 0.55
            ]
            if not long_h:
                return True
        return False

    def _has_three_equal_parallel_lines(self, d) -> bool:
        hs = sorted(
            [ln for ln in d["lines"] if self._is_horizontal(ln) and ln["length"] >= 3.0],
            key=lambda ln: self._midpoint(ln)[1],
        )
        if len(hs) < 3:
            return False
        for i in range(len(hs) - 2):
            trio = hs[i:i + 3]
            lengths = [ln["length"] for ln in trio]
            top_len, mid_len, bot_len = lengths
            outer_ratio = min(top_len, bot_len) / max(top_len, bot_len)
            if outer_ratio < 0.60:
                continue
            if mid_len < max(top_len, bot_len) * 0.90:
                continue
            mids = [self._midpoint(ln) for ln in trio]
            center_tol = max(mid_len * 0.18, 1.5)
            if max(abs(mx - mids[1][0]) for mx, _ in mids) > center_tol:
                continue
            gap1, gap2 = mids[1][1] - mids[0][1], mids[2][1] - mids[1][1]
            if abs(gap1 - gap2) <= max((gap1 + gap2) * 0.20, 0.9):
                return True
        return False

    def _has_parallelogram_component(self, d) -> bool:
        for comp in self._line_components(d["lines"]):
            if self._is_parallelogram(comp):
                return True
        return False

    def _is_parallelogram(self, lines) -> bool:
        if len(lines) != 4 or not self._line_graph_closed(lines):
            return False
        groups = self._parallel_groups(lines)
        if len(groups) != 2 or any(len(g) != 2 for g in groups):
            return False
        angles = [self._canonical_angle(g[0]) for g in groups]
        return abs(self._angle_delta(angles[0], angles[1]) - 90.0) > 6.0

    def _looks_like_square(self, d) -> bool:
        for rect in d["rects"]:
            if abs(rect["w"] - rect["h"]) / max(rect["w"], rect["h"], 1e-6) <= 0.18:
                return True
        lines = d["lines"]
        if len(lines) != 4 or not self._line_graph_closed(lines):
            return False
        lengths = [ln["length"] for ln in lines]
        return min(lengths) / max(lengths) >= 0.78

    # ------------------------------------------------------------------
    # Letter helpers
    # ------------------------------------------------------------------

    def _modifier_name(self, letter: str) -> str:
        return {
            "M": "mmc", "L": "lmc", "P": "projected", "F": "free_state",
            "T": "tangent", "S": "rfs", "C": "continuous",
        }.get(letter, "")

    def _letter_m(self, lines) -> bool:
        verticals = [ln for ln in lines if self._is_vertical(ln, tol=14)]
        diagonals = [ln for ln in lines if not self._is_vertical(ln, tol=14) and not self._is_horizontal(ln, tol=14)]
        return len(verticals) >= 2 and len(diagonals) >= 2

    def _letter_l(self, lines) -> bool:
        verticals = [ln for ln in lines if self._is_vertical(ln, tol=14)]
        horizontals = [ln for ln in lines if self._is_horizontal(ln, tol=14)]
        if len(horizontals) != 1:
            return False
        for v in verticals:
            vx = self._midpoint(v)[0]
            v_bottom = max(v["y0"], v["y1"])
            for h in horizontals:
                hy = self._midpoint(h)[1]
                near_bottom = abs(hy - v_bottom) <= max(v["length"] * 0.18, 1.2)
                touches_vertical = min(abs(vx - h["x0"]), abs(vx - h["x1"])) <= max(h["length"] * 0.18, 1.2)
                if near_bottom and touches_vertical:
                    return True
        return False

    def _letter_p(self, lines, arcs) -> bool:
        return bool([ln for ln in lines if self._is_vertical(ln, tol=14)] and arcs)

    def _letter_f(self, lines) -> bool:
        verticals = [ln for ln in lines if self._is_vertical(ln, tol=14)]
        horizontals = [ln for ln in lines if self._is_horizontal(ln, tol=14)]
        return bool(verticals and len(horizontals) >= 2)

    def _letter_t(self, lines) -> bool:
        verticals = [ln for ln in lines if self._is_vertical(ln, tol=14)]
        horizontals = [ln for ln in lines if self._is_horizontal(ln, tol=14)]
        return any(abs(self._midpoint(v)[0] - self._midpoint(h)[0]) <= max(h["length"] * 0.18, 1.0)
                   for v in verticals for h in horizontals)

    # ------------------------------------------------------------------
    # Primitive construction helpers
    # ------------------------------------------------------------------

    def _line_from_points(self, p0, p1, raw=None):
        if p0 is None or p1 is None:
            return None
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length < self.t.min_path_segment_size:
            return None
        return {
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "length": length,
            "angle": self._angle_deg(x0, y0, x1, y1),
            "width": float((raw or {}).get("width", 0) or 0),
            "source": "fitz",
        }

    def _curve_from_item(self, item, raw=None):
        pts = [self._as_point(p) for p in item[1:]]
        pts = [p for p in pts if p is not None]
        if len(pts) != 4:
            return None
        bbox = self._bbox_from_points(pts)
        return {
            **bbox,
            "start": pts[0],
            "end": pts[-1],
            "closed": self._points_close(pts[0], pts[-1], 0.8),
            "filled": bool((raw or {}).get("fill")),
            "points": pts,
        }

    def _circle_from_curves(self, curves, raw=None):
        if len(curves) < 3:
            return None
        points = [p for curve in curves for p in curve["points"]]
        bbox = self._bbox_from_points(points)
        if bbox["w"] < self.t.min_symbol_size or bbox["h"] < self.t.min_symbol_size:
            return None
        aspect = abs(bbox["w"] - bbox["h"]) / max(bbox["w"], bbox["h"], 1e-6)
        starts_ends_close = self._points_close(curves[0]["start"], curves[-1]["end"], 1.0)
        if aspect <= self.t.circle_aspect_tol and starts_ends_close:
            diameter = (bbox["w"] + bbox["h"]) / 2.0
            return {
                **bbox,
                "cx": bbox["x"] + bbox["w"] / 2,
                "cy": bbox["y"] + bbox["h"] / 2,
                "r": diameter / 2,
                "filled": bool((raw or {}).get("fill")),
            }
        return None

    def _polygon_from_lines(self, lines, raw=None):
        if len(lines) < 3:
            return None
        if not self._line_graph_closed(lines):
            return None
        points = self._distinct_points([pt for ln in lines for pt in self._endpoints(ln)])
        if len(points) < 3:
            return None
        return {
            "points": points,
            "bbox": self._bbox_from_points(points),
            "filled": bool((raw or {}).get("fill")),
        }

    def _rect_from_item(self, item):
        if len(item) < 2:
            return None
        rect = item[1]
        if hasattr(rect, "x0"):
            return self._xyxy_to_bbox(rect.x0, rect.y0, rect.x1, rect.y1)
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            return self._xyxy_to_bbox(*rect)
        return None

    def _quad_points(self, item):
        if len(item) < 2:
            return []
        quad = item[1]
        if all(hasattr(quad, attr) for attr in ("ul", "ur", "lr", "ll")):
            pts = [self._as_point(getattr(quad, attr)) for attr in ("ul", "ur", "lr", "ll")]
            return [pt for pt in pts if pt is not None]
        if isinstance(quad, (list, tuple)) and len(quad) == 4:
            pts = [self._as_point(pt) for pt in quad]
            return [pt for pt in pts if pt is not None]
        return []

    def _quad_edges(self, item, meta=None):
        pts = self._quad_points(item)
        if len(pts) != 4:
            return []
        return [self._line_from_points(pts[i], pts[(i + 1) % 4], meta) for i in range(4)]

    def _rect_edges(self, rb, source="re", meta=None):
        x, y, w, h = rb["x"], rb["y"], rb["w"], rb["h"]
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return [self._line_from_points(pts[i], pts[(i + 1) % 4], meta) for i in range(4)]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _normalize_bbox(self, bbox):
        if bbox is None:
            return None
        if isinstance(bbox, dict) and "bbox" in bbox:
            bbox = bbox["bbox"]
        if isinstance(bbox, dict) and {"x", "y", "w", "h"}.issubset(bbox):
            return {"x": float(bbox["x"]), "y": float(bbox["y"]), "w": float(bbox["w"]), "h": float(bbox["h"])}
        if isinstance(bbox, dict) and {"x0", "y0", "x1", "y1"}.issubset(bbox):
            return self._xyxy_to_bbox(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return self._xyxy_to_bbox(*bbox)
        return None

    def _compartment_regions(self, frame_region, compartments):
        if compartments:
            out = []
            for idx, comp in enumerate(compartments):
                region = self._normalize_bbox(comp)
                if region:
                    if idx == 0:
                        bleed = min(region["h"] * 0.65, 14.0)
                        region = {**region, "w": region["w"] + bleed}
                    out.append((idx, region))
            return out
        return [(None, frame_region)]

    def _inset_region(self, region, frac=0.08):
        pad_x = min(max(region["w"] * frac, 0.4), 2.0)
        pad_y = min(max(region["h"] * frac, 0.4), 2.0)
        return {
            "x": region["x"] + pad_x,
            "y": region["y"] + pad_y,
            "w": max(0.0, region["w"] - 2 * pad_x),
            "h": max(0.0, region["h"] - 2 * pad_y),
        }

    def _xyxy_to_bbox(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        return {"x": min(x0, x1), "y": min(y0, y1), "w": abs(x1 - x0), "h": abs(y1 - y0)}

    def _bbox_from_seq(self, seq):
        if isinstance(seq, (list, tuple)) and len(seq) == 4:
            return self._xyxy_to_bbox(*seq)
        return None

    def _item_bbox(self, item):
        if not item:
            return None
        op = item[0]
        if op in {"l", "c"}:
            points = [self._as_point(p) for p in item[1:]]
            points = [p for p in points if p is not None]
            return self._bbox_from_points(points) if points else None
        if op == "qu":
            points = self._quad_points(item)
            return self._bbox_from_points(points) if points else None
        if op == "re":
            return self._rect_from_item(item)
        return None

    def _as_point(self, raw):
        if raw is None:
            return None
        if hasattr(raw, "x") and hasattr(raw, "y"):
            return float(raw.x), float(raw.y)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
        return None

    def _bbox_from_points(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}

    def _bbox_overlaps(self, a, b):
        return not (a["x"] + a["w"] < b["x"] or b["x"] + b["w"] < a["x"]
                    or a["y"] + a["h"] < b["y"] or b["y"] + b["h"] < a["y"])

    def _bbox_center_inside(self, bbox, region, margin=0.0):
        cx, cy = bbox["x"] + bbox["w"] / 2, bbox["y"] + bbox["h"] / 2
        return region["x"] - margin <= cx <= region["x"] + region["w"] + margin and region["y"] - margin <= cy <= region["y"] + region["h"] + margin

    def _bbox_inside(self, inner, outer, margin=0.0):
        return (
            outer["x"] - margin <= inner["x"]
            and outer["y"] - margin <= inner["y"]
            and outer["x"] + outer["w"] + margin >= inner["x"] + inner["w"]
            and outer["y"] + outer["h"] + margin >= inner["y"] + inner["h"]
        )

    def _angle_deg(self, x0, y0, x1, y1):
        return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0

    def _canonical_angle(self, line):
        return line["angle"] % 180.0

    def _angle_delta(self, a, b):
        d = abs((a - b) % 180.0)
        return min(d, 180.0 - d)

    def _is_horizontal(self, line, tol=None):
        tol = self.t.angle_tol_deg if tol is None else tol
        return min(abs(line["angle"]), abs(180.0 - line["angle"])) <= tol

    def _is_vertical(self, line, tol=None):
        tol = self.t.angle_tol_deg if tol is None else tol
        return abs(line["angle"] - 90.0) <= tol

    def _parallel(self, a, b, tol=None):
        tol = self.t.parallel_tol_deg if tol is None else tol
        return self._angle_delta(a["angle"], b["angle"]) <= tol

    def _midpoint(self, line):
        return (line["x0"] + line["x1"]) / 2, (line["y0"] + line["y1"]) / 2

    def _endpoints(self, line):
        return (line["x0"], line["y0"]), (line["x1"], line["y1"])

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _points_close(self, a, b, tol):
        return self._dist(a, b) <= tol

    def _line_through_point(self, line, x, y, tol):
        ax, ay = line["x0"], line["y0"]
        bx, by = line["x1"], line["y1"]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 <= 1e-9:
            return False
        t = ((x - ax) * dx + (y - ay) * dy) / length2
        if not -0.08 <= t <= 1.08:
            return False
        return self._dist((x, y), (ax + t * dx, ay + t * dy)) <= tol

    def _line_midpoint_distance_perp(self, a, b):
        bx, by = self._midpoint(b)
        ax, ay = a["x0"], a["y0"]
        dx, dy = a["x1"] - ax, a["y1"] - ay
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return 0.0
        return abs((bx - ax) * dy - (by - ay) * dx) / length

    def _shared_endpoint(self, a, b, tol=None):
        tol = tol or max(min(a["length"], b["length"]) * self.t.endpoint_tol_ratio, 1.2)
        for pa in self._endpoints(a):
            for pb in self._endpoints(b):
                if self._dist(pa, pb) <= tol:
                    return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        return None

    def _segments_touch_or_intersect(self, a, b):
        if self._shared_endpoint(a, b) is not None:
            return True
        p, r = self._endpoints(a)[0], (a["x1"] - a["x0"], a["y1"] - a["y0"])
        q, s = self._endpoints(b)[0], (b["x1"] - b["x0"], b["y1"] - b["y0"])
        denom = r[0] * s[1] - r[1] * s[0]
        if abs(denom) < 1e-9:
            return False
        qp = (q[0] - p[0], q[1] - p[1])
        t = (qp[0] * s[1] - qp[1] * s[0]) / denom
        u = (qp[0] * r[1] - qp[1] * r[0]) / denom
        return -0.05 <= t <= 1.05 and -0.05 <= u <= 1.05

    def _find_chord_line(self, lines, p0, p1):
        for ln in lines:
            a, b = self._endpoints(ln)
            direct = self._dist(a, p0) + self._dist(b, p1)
            reverse = self._dist(a, p1) + self._dist(b, p0)
            if min(direct, reverse) <= max(ln["length"] * 0.18, 2.0):
                return ln
        return None

    def _parallel_groups(self, lines):
        groups = []
        for ln in lines:
            for group in groups:
                if self._parallel(group[0], ln):
                    group.append(ln)
                    break
            else:
                groups.append([ln])
        return groups

    def _line_graph_closed(self, lines):
        endpoints = [pt for ln in lines for pt in self._endpoints(ln)]
        used = [False] * len(endpoints)
        pairs = 0
        for i, pt in enumerate(endpoints):
            if used[i]:
                continue
            cluster = [j for j, other in enumerate(endpoints) if not used[j] and self._dist(pt, other) <= 1.4]
            if len(cluster) >= 2:
                for j in cluster:
                    used[j] = True
                pairs += 1
        return pairs >= len(lines)

    def _line_inside_bbox(self, line, bbox, margin=0.0):
        for x, y in self._endpoints(line):
            if not (bbox["x"] - margin <= x <= bbox["x"] + bbox["w"] + margin):
                return False
            if not (bbox["y"] - margin <= y <= bbox["y"] + bbox["h"] + margin):
                return False
        return True

    def _is_region_border_line(self, line, region):
        tol = self.t.border_tol
        if self._is_vertical(line) and line["length"] >= region["h"] * 1.08:
            return True
        if self._is_horizontal(line) and line["length"] >= region["w"] * 1.08:
            return True
        if self._is_horizontal(line) and line["length"] >= region["w"] * self.t.border_length_ratio:
            my = self._midpoint(line)[1]
            return abs(my - region["y"]) <= tol or abs(my - (region["y"] + region["h"])) <= tol
        if self._is_vertical(line) and line["length"] >= region["h"] * self.t.border_length_ratio:
            mx = self._midpoint(line)[0]
            return abs(mx - region["x"]) <= tol or abs(mx - (region["x"] + region["w"])) <= tol
        return False

    def _is_triangle_polygon(self, poly):
        return len(self._distinct_points(poly.get("points", []))) == 3

    def _has_triangle_from_lines(self, lines):
        if len(lines) != 3 or not self._line_graph_closed(lines):
            return False
        return len(self._distinct_points([pt for ln in lines for pt in self._endpoints(ln)])) == 3

    def _line_components(self, lines, tol=1.2):
        if not lines:
            return []
        parent = list(range(len(lines)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        endpoints = [self._endpoints(ln) for ln in lines]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if any(self._dist(a, b) <= tol for a in endpoints[i] for b in endpoints[j]):
                    union(i, j)
        groups = {}
        for i, ln in enumerate(lines):
            groups.setdefault(find(i), []).append(ln)
        return list(groups.values())

    def _looks_like_flatness_polyline(self, lines):
        return any(self._component_looks_like_flatness_polyline(comp) for comp in self._line_components(lines))

    def _component_looks_like_flatness_polyline(self, lines):
        if not (6 <= len(lines) <= 16):
            return False
        bbox = self._bbox_from_points([pt for ln in lines for pt in self._endpoints(ln)])
        if bbox["w"] < 4.5 or bbox["h"] < 4.0:
            return False
        aspect = bbox["w"] / max(bbox["h"], 1e-6)
        if not 0.30 <= aspect <= 3.0:
            return False
        horizontals = [ln for ln in lines if self._is_horizontal(ln, tol=12)]
        slanted = [
            ln for ln in lines
            if not self._is_horizontal(ln, tol=12) and not self._is_vertical(ln, tol=12)
        ]
        if len(horizontals) < 2 or len(slanted) < 4:
            return False
        long_horizontals = [ln for ln in horizontals if ln["length"] >= bbox["w"] * 0.24]
        if len(long_horizontals) < 2:
            return False
        top = min(self._midpoint(ln)[1] for ln in horizontals)
        bottom = max(self._midpoint(ln)[1] for ln in horizontals)
        if bottom - top < bbox["h"] * 0.55:
            return False
        parallel_pairs = sum(1 for i, a in enumerate(lines) for b in lines[i + 1:] if self._parallel(a, b, tol=14))
        return parallel_pairs >= 4

    def _distinct_points(self, points, tol=0.8):
        out = []
        for pt in points:
            if not any(self._dist(pt, existing) <= tol for existing in out):
                out.append(pt)
        return out

    def _dedup_lines(self, lines):
        kept = []
        seen = set()
        for ln in lines:
            key = tuple(round(v * 2) for v in (ln["x0"], ln["y0"], ln["x1"], ln["y1"]))
            rev = tuple(round(v * 2) for v in (ln["x1"], ln["y1"], ln["x0"], ln["y0"]))
            if key in seen or rev in seen:
                continue
            seen.add(key)
            kept.append(ln)
        return kept

    def _dedup_circles(self, circles):
        kept = []
        for c in circles:
            if any(self._dist((c["cx"], c["cy"]), (k["cx"], k["cy"])) < 0.5 and abs(c["r"] - k["r"]) < 0.5 for k in kept):
                continue
            kept.append(c)
        return kept

    def _chars_text(self, d):
        return "".join(ch.get("text", "") for ch in sorted(d.get("chars", []), key=lambda ch: (ch["x"], ch["y"])))

    def _result(self, cls: int, confidence: float):
        name = CLASS_NAMES[cls]
        return {
            "symbol_class": cls,
            "symbol_name": name,
            "symbol_unicode": CLASS_SYMBOLS[name],
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "method": "vector_inversion",
        }

    def _unknown(self):
        return {
            "symbol_class": -1,
            "symbol_name": "unknown",
            "symbol_unicode": "?",
            "confidence": 0.0,
            "method": "vector_inversion",
        }
