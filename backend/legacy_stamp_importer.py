"""Legacy stamped-PDF importer.

Extracts old manually numbered stamp annotations from a PDF without OCR. The
old files store the sequence labels as PDF text and the stamp/leader shapes as
PDF annotations, so PyMuPDF text and annotation geometry is enough.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import fitz


_LABEL_RE = re.compile(r"^\d+(?:-\d+)?$")
_MIN_LABEL_FONT_SIZE = 10.0
_MAX_LEGACY_PAGES = 200
_PDF_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
_PDF_TOKEN_RE = re.compile(
    r"/[A-Za-z0-9#._-]+"
    r"|[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
    r"|[A-Za-z][A-Za-z0-9]*\*?"
    r"|\*"
)
_PDF_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_IDENTITY_MATRIX = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_seq(cls, values) -> "_Rect":
        x0, y0, x1, y1 = [float(v) for v in values[:4]]
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def aspect_ratio(self) -> float:
        short = max(0.001, min(self.w, self.h))
        return max(self.w, self.h) / short

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    def contains_point(self, x: float, y: float, pad: float = 0.0) -> bool:
        return (
            self.x0 - pad <= x <= self.x1 + pad
            and self.y0 - pad <= y <= self.y1 + pad
        )

    def as_bbox(self) -> dict[str, float]:
        return {
            "x": round(self.x0, 2),
            "y": round(self.y0, 2),
            "w": round(self.w, 2),
            "h": round(self.h, 2),
        }


def _round_point(point: tuple[float, float]) -> dict[str, float]:
    return {"x": round(point[0], 2), "y": round(point[1], 2)}


def _matrix_multiply(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * l + e,
        b * k + d * l + f,
    )


def _matrix_apply(
    matrix: tuple[float, float, float, float, float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def _parse_pdf_float_array(raw: str | None, expected: int | None = None) -> list[float]:
    if not raw:
        return []
    values = [float(m.group(0)) for m in _PDF_FLOAT_RE.finditer(raw)]
    if expected is not None and len(values) < expected:
        return []
    return values[:expected] if expected is not None else values


def _xref_object(doc: fitz.Document, xref: int) -> str:
    try:
        return doc.xref_object(xref, compressed=False)
    except Exception:
        return ""


def _xref_stream_text(doc: fitz.Document, xref: int) -> str:
    try:
        stream = doc.xref_stream(xref) or b""
    except Exception:
        return ""
    return stream.decode("latin1", errors="ignore")


def _form_matrix(doc: fitz.Document, xref: int) -> tuple[float, float, float, float, float, float]:
    obj = _xref_object(doc, xref)
    match = re.search(r"/Matrix\s*\[([^\]]+)\]", obj, flags=re.S)
    values = _parse_pdf_float_array(match.group(1) if match else None, expected=6)
    return tuple(values) if values else _IDENTITY_MATRIX


def _form_bbox(doc: fitz.Document, xref: int) -> _Rect | None:
    obj = _xref_object(doc, xref)
    match = re.search(r"/BBox\s*\[([^\]]+)\]", obj, flags=re.S)
    values = _parse_pdf_float_array(match.group(1) if match else None, expected=4)
    return _Rect.from_seq(values) if values else None


def _normal_appearance_xref(doc: fitz.Document, annot_xref: int) -> int | None:
    obj = _xref_object(doc, annot_xref)
    match = re.search(r"/AP\s*<<.*?/N\s+(\d+)\s+0\s+R", obj, flags=re.S)
    return int(match.group(1)) if match else None


def _xobject_refs(doc: fitz.Document, form_xref: int) -> dict[str, int]:
    obj = _xref_object(doc, form_xref)
    match = re.search(r"/XObject\s*<<(.+?)>>", obj, flags=re.S)
    if not match:
        return {}
    return {
        name: int(xref)
        for name, xref in re.findall(r"/([A-Za-z0-9#._-]+)\s+(\d+)\s+0\s+R", match.group(1))
    }


def _is_form_xobject(doc: fitz.Document, xref: int) -> bool:
    obj = _xref_object(doc, xref)
    return "/Subtype /Form" in obj and "/Type /XObject" in obj


def _parse_content_path_points(
    doc: fitz.Document,
    stream: str,
    resources: dict[str, int],
    base_matrix: tuple[float, float, float, float, float, float],
    seen: set[int],
) -> list[tuple[float, float]]:
    tokens = _PDF_TOKEN_RE.findall(stream)
    stack: list[float | str] = []
    matrices = [base_matrix]
    points: list[tuple[float, float]] = []

    def add_point(x: float, y: float) -> None:
        points.append(_matrix_apply(matrices[-1], (x, y)))

    for token in tokens:
        if _PDF_NUMBER_RE.match(token):
            stack.append(float(token))
            continue
        if token.startswith("/"):
            stack.append(token)
            continue

        op = token
        if op == "q":
            matrices.append(matrices[-1])
        elif op == "Q":
            if len(matrices) > 1:
                matrices.pop()
        elif op == "cm" and len(stack) >= 6:
            vals = [float(v) for v in stack[-6:]]
            stack.clear()
            matrices[-1] = _matrix_multiply(matrices[-1], tuple(vals))
        elif op in {"m", "l"} and len(stack) >= 2:
            x, y = [float(v) for v in stack[-2:]]
            stack.clear()
            add_point(x, y)
        elif op == "c" and len(stack) >= 6:
            vals = [float(v) for v in stack[-6:]]
            stack.clear()
            for idx in range(0, 6, 2):
                add_point(vals[idx], vals[idx + 1])
        elif op in {"v", "y"} and len(stack) >= 4:
            vals = [float(v) for v in stack[-4:]]
            stack.clear()
            for idx in range(0, 4, 2):
                add_point(vals[idx], vals[idx + 1])
        elif op == "re" and len(stack) >= 4:
            x, y, w, h = [float(v) for v in stack[-4:]]
            stack.clear()
            for point in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
                add_point(*point)
        elif op == "Do":
            name = next((v[1:] for v in reversed(stack) if isinstance(v, str) and v.startswith("/")), "")
            stack.clear()
            child_xref = resources.get(name)
            if child_xref:
                child_matrix = _form_matrix(doc, child_xref)
                child_base = _matrix_multiply(matrices[-1], child_matrix)
                points.extend(_collect_form_path_points(doc, child_xref, child_base, seen))
        elif op in {
            "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n",
            "W", "W*", "h", "gs", "cs", "CS", "scn", "SCN", "w",
            "J", "j", "M", "d", "ri", "i", "g", "G", "rg", "RG",
            "k", "K",
        } or op == "*":
            stack.clear()
        else:
            stack.clear()
    return points


def _collect_form_path_points(
    doc: fitz.Document,
    form_xref: int,
    base_matrix: tuple[float, float, float, float, float, float] = _IDENTITY_MATRIX,
    seen: set[int] | None = None,
) -> list[tuple[float, float]]:
    if seen is None:
        seen = set()
    if form_xref in seen or not _is_form_xobject(doc, form_xref):
        return []
    seen.add(form_xref)
    try:
        return _parse_content_path_points(
            doc,
            _xref_stream_text(doc, form_xref),
            _xobject_refs(doc, form_xref),
            base_matrix,
            seen,
        )
    finally:
        seen.remove(form_xref)


def _appearance_bbox_extents(
    bbox: _Rect | None,
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    if bbox is None:
        return (0.0, 0.0, 0.0, 0.0)
    corners = [
        _matrix_apply(matrix, point)
        for point in (
            (bbox.x0, bbox.y0),
            (bbox.x0, bbox.y1),
            (bbox.x1, bbox.y0),
            (bbox.x1, bbox.y1),
        )
    ]
    min_x = min(x for x, _ in corners)
    min_y = min(y for _, y in corners)
    max_x = max(x for x, _ in corners)
    max_y = max(y for _, y in corners)
    return min_x, min_y, max_x - min_x, max_y - min_y


def _page_point_from_appearance(
    annot_rect: _Rect,
    appearance_point: tuple[float, float],
) -> tuple[float, float]:
    x, y = appearance_point
    return annot_rect.x0 + x, annot_rect.y1 - y


def _direction_for_vector(circle: tuple[float, float], tip: tuple[float, float]) -> tuple[str, float]:
    dx = tip[0] - circle[0]
    dy = tip[1] - circle[1]
    angle = math.degrees(math.atan2(dy, dx)) if abs(dx) > 0.001 or abs(dy) > 0.001 else 0.0
    if abs(dx) >= abs(dy):
        direction = "right" if dx >= 0 else "left"
    else:
        direction = "down" if dy >= 0 else "up"
    return direction, round(angle, 2)


def _point_from_axis(
    axis: tuple[float, float],
    perp: tuple[float, float],
    u: float,
    v: float,
) -> tuple[float, float]:
    return axis[0] * u + perp[0] * v, axis[1] * u + perp[1] * v


def _principal_stamp_geometry(points: list[tuple[float, float]]) -> dict[str, Any] | None:
    if len(points) < 12:
        return None

    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    var_x = sum((x - mean_x) ** 2 for x, _ in points) / len(points)
    var_y = sum((y - mean_y) ** 2 for _, y in points) / len(points)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in points) / len(points)
    angle = 0.5 * math.atan2(2.0 * cov_xy, var_x - var_y)
    axis = (math.cos(angle), math.sin(angle))
    perp = (-axis[1], axis[0])
    projected = [
        (x * axis[0] + y * axis[1], x * perp[0] + y * perp[1], (x, y))
        for x, y in points
    ]
    min_u = min(u for u, _, _ in projected)
    max_u = max(u for u, _, _ in projected)
    min_v = min(v for _, v, _ in projected)
    max_v = max(v for _, v, _ in projected)
    extent_u = max_u - min_u
    extent_v = max_v - min_v
    if extent_u <= 0.001 or extent_v <= 0.001 or extent_u < extent_v * 1.05:
        return None

    edge_band = max(2.0, extent_u * 0.06)
    low_edge = [(u, v, p) for u, v, p in projected if u <= min_u + edge_band]
    high_edge = [(u, v, p) for u, v, p in projected if u >= max_u - edge_band]
    if not low_edge or not high_edge:
        return None

    def v_span(items: list[tuple[float, float, tuple[float, float]]]) -> float:
        return max(v for _, v, _ in items) - min(v for _, v, _ in items)

    low_span = v_span(low_edge)
    high_span = v_span(high_edge)
    tip_is_low = low_span < high_span
    tip_item = min(projected, key=lambda item: item[0]) if tip_is_low else max(projected, key=lambda item: item[0])

    if tip_is_low:
        circle_side = [(u, v, p) for u, v, p in projected if u >= max_u - extent_u * 0.55]
        circle_edge_u = max(u for u, _, _ in circle_side) if circle_side else max_u
    else:
        circle_side = [(u, v, p) for u, v, p in projected if u <= min_u + extent_u * 0.55]
        circle_edge_u = min(u for u, _, _ in circle_side) if circle_side else min_u
    if len(circle_side) < 8:
        return None

    c_min_v = min(v for _, v, _ in circle_side)
    c_max_v = max(v for _, v, _ in circle_side)
    radius = (c_max_v - c_min_v) / 2.0
    if radius < 3.0:
        return None

    circle_v = (c_min_v + c_max_v) / 2.0
    circle_u = circle_edge_u - radius if tip_is_low else circle_edge_u + radius
    tip_u, tip_v, tip_point = tip_item
    center_to_tip = math.hypot(tip_u - circle_u, tip_v - circle_v)
    if center_to_tip < radius * 1.05 or center_to_tip > radius * 10.0:
        return None

    tip_span = low_span if tip_is_low else high_span
    if tip_span > radius * 1.35:
        return None

    return {
        "circle": _point_from_axis(axis, perp, circle_u, circle_v),
        "tip": tip_point,
        "radius": radius,
        "center_to_tip": center_to_tip,
    }


def _label_sort_key(label: str) -> tuple[Any, ...]:
    if label.isdigit():
        return (0, int(label), 0, label)
    match = re.match(r"^(\d+)-(\d+)$", label)
    if match:
        return (0, int(match.group(1)), int(match.group(2)), label)
    head = label.split("-", 1)[0]
    if head.isdigit():
        return (0, int(head), 1_000_000, label)
    return (1, 0, 0, label)


def _collect_label_spans(page: fitz.Page) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not _LABEL_RE.match(text):
                    continue
                size = float(span.get("size") or 0.0)
                if size < _MIN_LABEL_FONT_SIZE:
                    continue
                bbox = _Rect.from_seq(span.get("bbox") or [0, 0, 0, 0])
                if bbox.w <= 0 or bbox.h <= 0:
                    continue
                spans.append({
                    "label": text,
                    "font_size": size,
                    "bbox": bbox,
                })
    return spans


def _extract_appearance_geometry(
    doc: fitz.Document,
    annot: fitz.Annot,
    annot_rect: _Rect,
) -> dict[str, Any] | None:
    ap_xref = _normal_appearance_xref(doc, annot.xref)
    if ap_xref is None:
        return None

    points = _collect_form_path_points(doc, ap_xref)
    if len(points) < 3:
        return None

    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    if max_x <= min_x or max_y <= min_y:
        return None

    local_geometry = _principal_stamp_geometry(points)
    if local_geometry is None:
        return None
    tip_local = local_geometry["tip"]
    circle_local = local_geometry["circle"]

    ap_matrix = _form_matrix(doc, ap_xref)
    origin_x, origin_y, appearance_w, appearance_h = _appearance_bbox_extents(
        _form_bbox(doc, ap_xref),
        ap_matrix,
    )
    if appearance_w <= 0.001 or appearance_h <= 0.001:
        transformed_points = [_matrix_apply(ap_matrix, point) for point in points]
        origin_x = min(x for x, _ in transformed_points)
        origin_y = min(y for _, y in transformed_points)
        appearance_w = max(x for x, _ in transformed_points) - origin_x
        appearance_h = max(y for _, y in transformed_points) - origin_y
    scale_x = annot_rect.w / appearance_w if appearance_w > 0.001 else 1.0
    scale_y = annot_rect.h / appearance_h if appearance_h > 0.001 else 1.0

    def to_page(local_point: tuple[float, float]) -> tuple[float, float]:
        ax, ay = _matrix_apply(ap_matrix, local_point)
        return _page_point_from_appearance(
            annot_rect,
            ((ax - origin_x) * scale_x, (ay - origin_y) * scale_y),
        )

    circle = to_page(circle_local)
    tip = to_page(tip_local)
    direction, angle = _direction_for_vector(circle, tip)
    return {
        "circle_center": _round_point(circle),
        "arrow_tip": _round_point(tip),
        "routing_path": [_round_point(circle), _round_point(tip)],
        "arrow_geometry": "appearance_stream",
        "arrow_geometry_precise": True,
        "direction": direction,
        "direction_angle_degrees": angle,
    }


def _collect_annotation_entries(page: fitz.Page, doc: fitz.Document) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for annot in page.annots() or []:
        r = annot.rect
        rect = _Rect.from_seq((r.x0, r.y0, r.x1, r.y1))
        if rect.w <= 1 or rect.h <= 1:
            continue
        precise = None
        if annot.type[1] == "Stamp":
            precise = _extract_appearance_geometry(doc, annot, rect)
        entries.append({
            "rect": rect,
            "precise_geometry": precise,
        })
    return entries


def _choose_stamp_entry(label_bbox: _Rect, annot_entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    cx, cy = label_bbox.center
    min_outer_area = max(1200.0, label_bbox.area * 2.2)
    max_outer_area = max(5000.0, label_bbox.area * 50.0)
    candidates = [
        entry for entry in annot_entries
        for rect in [entry["rect"]]
        if (
            min_outer_area <= rect.area <= max_outer_area
            and rect.aspect_ratio <= 2.2
            and rect.contains_point(cx, cy, pad=2.0)
        )
    ]
    if not candidates:
        return None
    # Pick the tightest outer annotation, not the tiny text annotation itself.
    candidates.sort(key=lambda entry: (entry["rect"].area, entry["rect"].w + entry["rect"].h))
    return candidates[0]


def _infer_geometry(
    label_bbox: _Rect,
    stamp_rect: _Rect | None,
    precise_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_cx, label_cy = label_bbox.center
    if stamp_rect is None:
        pad = max(14.0, min(label_bbox.w, label_bbox.h))
        tip = (label_cx, label_cy)
        dim = _Rect(label_cx - pad / 2, label_cy - pad / 2,
                    label_cx + pad / 2, label_cy + pad / 2)
        return {
            "circle_center": _round_point((label_cx, label_cy)),
            "arrow_tip": _round_point(tip),
            "dim_bbox": dim.as_bbox(),
            "dim_bbox_placeholder": True,
            "geometry_source": "label_bbox_placeholder",
            "stamp_bbox": None,
            "direction": "unknown",
        }

    if precise_geometry:
        tip = precise_geometry["arrow_tip"]
        dim_size = max(16.0, min(28.0, max(stamp_rect.w, stamp_rect.h) * 0.22))
        dim = _Rect(tip["x"] - dim_size / 2, tip["y"] - dim_size / 2,
                    tip["x"] + dim_size / 2, tip["y"] + dim_size / 2)
        return {
            **precise_geometry,
            "dim_bbox": dim.as_bbox(),
            "dim_bbox_placeholder": True,
            "geometry_source": "stamp_appearance_stream",
            "stamp_bbox": stamp_rect.as_bbox(),
        }

    mid_x, mid_y = stamp_rect.center
    edge_dists = {
        "left": label_cx - stamp_rect.x0,
        "right": stamp_rect.x1 - label_cx,
        "up": label_cy - stamp_rect.y0,
        "down": stamp_rect.y1 - label_cy,
    }
    direction = max(edge_dists, key=edge_dists.get)

    if direction == "right":
        tip = (stamp_rect.x1, mid_y)
        radius = stamp_rect.h / 2.0
        circle = (stamp_rect.x0 + radius, mid_y)
    elif direction == "left":
        tip = (stamp_rect.x0, mid_y)
        radius = stamp_rect.h / 2.0
        circle = (stamp_rect.x1 - radius, mid_y)
    elif direction == "down":
        tip = (mid_x, stamp_rect.y1)
        radius = stamp_rect.w / 2.0
        circle = (mid_x, stamp_rect.y0 + radius)
    else:
        tip = (mid_x, stamp_rect.y0)
        radius = stamp_rect.w / 2.0
        circle = (mid_x, stamp_rect.y1 - radius)

    dim_size = max(16.0, min(28.0, max(stamp_rect.w, stamp_rect.h) * 0.22))
    dim = _Rect(tip[0] - dim_size / 2, tip[1] - dim_size / 2,
                tip[0] + dim_size / 2, tip[1] + dim_size / 2)
    return {
        "circle_center": _round_point(circle),
        "arrow_tip": _round_point(tip),
        "dim_bbox": dim.as_bbox(),
        "dim_bbox_placeholder": True,
        "geometry_source": "stamp_rect_inferred",
        "stamp_bbox": stamp_rect.as_bbox(),
        "arrow_geometry": "stamp_rect",
        "arrow_geometry_precise": False,
        "direction": direction,
    }


def extract_legacy_stamp_marks(pdf_bytes: bytes) -> dict[str, Any]:
    """Extract old sequence stamps from a stamped PDF.

    Returns page-local coordinates in PDF points. The frontend adds page offsets
    after rendering the clean base PDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count > _MAX_LEGACY_PAGES:
            raise ValueError(f"旧序号 PDF 页数过多 ({doc.page_count} > {_MAX_LEGACY_PAGES})")
        marks: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        warnings: list[str] = []

        for page_index in range(doc.page_count):
            page = doc[page_index]
            page_number = page_index + 1
            pages.append({
                "page": page_number,
                "width": round(float(page.rect.width), 2),
                "height": round(float(page.rect.height), 2),
            })
            annots = _collect_annotation_entries(page, doc)
            spans = _collect_label_spans(page)
            seen: set[tuple[str, int, int]] = set()

            for span in spans:
                label = span["label"]
                bbox: _Rect = span["bbox"]
                cx, cy = bbox.center
                dedupe_key = (label, round(cx * 2), round(cy * 2))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                stamp_entry = _choose_stamp_entry(bbox, annots)
                if stamp_entry is None:
                    warnings.append(f"page {page_number}: ignored label {label} without stamp geometry")
                    continue
                geom = _infer_geometry(
                    bbox,
                    stamp_entry["rect"],
                    stamp_entry.get("precise_geometry"),
                )
                marks.append({
                    "label": label,
                    "page": page_number,
                    "font_size": round(float(span["font_size"]), 2),
                    "label_bbox": bbox.as_bbox(),
                    **geom,
                })

            if annots and not spans:
                warnings.append(f"page {page_number}: found annotations but no sequence labels")

        marks.sort(key=lambda m: (m["page"], _label_sort_key(m["label"])))
        return {
            "status": "success",
            "page_count": doc.page_count,
            "pages": pages,
            "marks": marks,
            "warnings": warnings,
            "stats": {
                "mark_count": len(marks),
                "pages_with_marks": len({m["page"] for m in marks}),
            },
        }
    finally:
        doc.close()
