"""Core PyMuPDF drawing item primitives used by vector detectors.

This module intentionally does not import from ``backend/scripts``. Probe
scripts may re-export these helpers, but production/shadow detectors should use
this core copy directly.
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Any, Iterable, Sequence

import fitz


BEZIER_STEPS = 16

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


_ItemPrimitiveTuple = namedtuple(
    "_ItemPrimitiveTuple",
    (
        "pdf",
        "page",
        "drawing_order",
        "item_index",
        "op",
        "points",
        "bbox",
        "draw_type",
        "stroke_width",
    ),
)


class ItemPrimitive(_ItemPrimitiveTuple):
    """Deeply immutable drawing primitive.

    ``points`` and ``bbox`` are copied into tuples at construction so callers
    cannot retain a mutable list alias into a page snapshot.
    """

    __slots__ = ()

    pdf: str
    page: int
    drawing_order: int
    item_index: int
    op: str
    points: tuple[Point, ...]
    bbox: BBox
    draw_type: str
    stroke_width: float | None

    def __new__(
        cls,
        pdf: str,
        page: int,
        drawing_order: int,
        item_index: int,
        op: str,
        points: Iterable[Point],
        bbox: Iterable[float],
        draw_type: str,
        stroke_width: float | None,
    ) -> "ItemPrimitive":
        immutable_points = tuple(
            (float(point[0]), float(point[1]))
            for point in points
        )
        immutable_bbox = tuple(float(value) for value in bbox)
        immutable_stroke_width = (
            None
            if stroke_width is None
            else float(stroke_width)
        )
        return super().__new__(
            cls,
            str(pdf),
            int(page),
            int(drawing_order),
            int(item_index),
            str(op),
            immutable_points,
            immutable_bbox,
            str(draw_type),
            immutable_stroke_width,
        )

    @classmethod
    def _make(
        cls,
        iterable: Iterable[Any],
    ) -> "ItemPrimitive":
        values = tuple(iterable)
        if len(values) != len(cls._fields):
            raise TypeError(
                f"Expected {len(cls._fields)} arguments, got {len(values)}"
            )
        return cls(*values)

    def _replace(
        self,
        /,
        **changes: Any,
    ) -> "ItemPrimitive":
        unexpected = changes.keys() - self._fields
        if unexpected:
            raise TypeError(
                f"Got unexpected field names: {sorted(unexpected)!r}"
            )
        return type(self)(
            *(
                changes.get(field_name, getattr(self, field_name))
                for field_name in self._fields
            )
        )


def extract_items_from_page(page: fitz.Page, *, pdf_stem: str, page_num: int) -> list[ItemPrimitive]:
    """Expand PyMuPDF drawings to item-level primitives."""
    return extract_items_from_drawings(
        page.get_drawings(),
        pdf_stem=pdf_stem,
        page_num=page_num,
    )


def extract_items_from_drawings(
    drawings: Iterable[dict[str, Any]],
    *,
    pdf_stem: str,
    page_num: int,
) -> list[ItemPrimitive]:
    """Expand an existing immutable drawing snapshot to item primitives."""
    items: list[ItemPrimitive] = []
    for drawing_order, drawing in enumerate(drawings):
        width = drawing.get("width")
        stroke_width = float(width) if width is not None else None
        draw_type = str(drawing.get("type") or "")
        for item_index, raw_item in enumerate(drawing.get("items", [])):
            points, op, _is_closed = points_for_item(raw_item)
            if len(points) < 2 or not op:
                continue
            items.append(
                ItemPrimitive(
                    pdf=pdf_stem,
                    page=int(page_num),
                    drawing_order=int(drawing_order),
                    item_index=int(item_index),
                    op=op,
                    points=points,
                    bbox=bbox_from_points(points),
                    draw_type=draw_type,
                    stroke_width=stroke_width,
                )
            )
    return items


def item_length(item: ItemPrimitive) -> float:
    return sum(
        math.hypot(
            item.points[idx][0] - item.points[idx - 1][0],
            item.points[idx][1] - item.points[idx - 1][1],
        )
        for idx in range(1, len(item.points))
    )


def points_for_item(item: Any) -> tuple[list[Point], str, bool]:
    if not item:
        return [], "", False
    op = str(item[0])
    if op == "l" and len(item) >= 3:
        p0 = point_xy(item[1])
        p1 = point_xy(item[2])
        return ([p0, p1] if p0 is not None and p1 is not None else []), op, False
    if op == "c" and len(item) >= 5:
        cubic = [point_xy(item[i]) for i in range(1, 5)]
        if all(p is not None for p in cubic):
            p0, p1, p2, p3 = cubic
            return sample_cubic(p0, p1, p2, p3), op, False  # type: ignore[arg-type]
        return [], op, False
    if op == "re" and len(item) >= 2:
        return rect_points(item[1]), op, True
    if op == "qu" and len(item) >= 2:
        return quad_points(item[1]), op, True
    return [], op, False


def point_xy(raw: Any) -> Point | None:
    if hasattr(raw, "x") and hasattr(raw, "y"):
        return float(raw.x), float(raw.y)
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    return None


def rect_bbox(raw: Any) -> BBox | None:
    if hasattr(raw, "x0") and hasattr(raw, "y0") and hasattr(raw, "x1") and hasattr(raw, "y1"):
        return float(raw.x0), float(raw.y0), float(raw.x1), float(raw.y1)
    if isinstance(raw, dict):
        if {"x", "y", "w", "h"}.issubset(raw):
            x0 = float(raw["x"])
            y0 = float(raw["y"])
            return x0, y0, x0 + float(raw["w"]), y0 + float(raw["h"])
        if {"x0", "y0", "x1", "y1"}.issubset(raw):
            return float(raw["x0"]), float(raw["y0"]), float(raw["x1"]), float(raw["y1"])
    if isinstance(raw, (tuple, list)) and len(raw) >= 4:
        return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
    return None


def rect_points(raw_rect: Any) -> list[Point]:
    bbox = rect_bbox(raw_rect)
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def quad_points(raw: Any) -> list[Point]:
    points: list[Point] = []
    for attr in ("ul", "ur", "lr", "ll"):
        xy = point_xy(getattr(raw, attr, None))
        if xy is not None:
            points.append(xy)
    if len(points) == 4:
        return points + [points[0]]
    try:
        iterable = list(raw)
    except TypeError:
        return []
    points = []
    for item in iterable:
        xy = point_xy(item)
        if xy is not None:
            points.append(xy)
    return points + [points[0]] if len(points) >= 3 else points


def sample_cubic(p0: Point, p1: Point, p2: Point, p3: Point) -> list[Point]:
    sampled: list[Point] = []
    for step in range(BEZIER_STEPS + 1):
        t = step / BEZIER_STEPS
        mt = 1.0 - t
        sampled.append((
            mt * mt * mt * p0[0]
            + 3.0 * mt * mt * t * p1[0]
            + 3.0 * mt * t * t * p2[0]
            + t * t * t * p3[0],
            mt * mt * mt * p0[1]
            + 3.0 * mt * mt * t * p1[1]
            + 3.0 * mt * t * t * p2[1]
            + t * t * t * p3[1],
        ))
    return sampled


def bbox_from_points(points: Sequence[Point]) -> BBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)
