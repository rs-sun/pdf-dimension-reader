"""Shared, indexed vector primitive context for one PDF page.

The context owns one immutable primitive snapshot.  Query counters are runtime
diagnostics and are deliberately kept out of semantic phrase payloads.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from vector_draw_items import (
    BBox,
    ItemPrimitive,
    extract_items_from_drawings,
    extract_items_from_page,
)


DEFAULT_CELL_SIZE = 32.0
_PENDING_CONTEXT_MINT_TOKENS: set[object] = set()


def _stable_item_key(item: ItemPrimitive) -> tuple[int, int, str]:
    return int(item.drawing_order), int(item.item_index), str(item.op)


def _intersects(left: BBox, right: BBox) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


@dataclass(slots=True)
class _RuntimeMetricsState:
    query_count: int = 0
    candidate_scan_count: int = 0
    extraction_call_count: int = 0


@dataclass(frozen=True, slots=True)
class VectorPageContext:
    """Immutable page primitives plus a coarse bbox grid index."""

    page_width: float
    page_height: float
    primitives: tuple[ItemPrimitive, ...]
    pdf_stem: str = ""
    page_num: int = 0
    cell_size: float = DEFAULT_CELL_SIZE
    _grid: Mapping[tuple[int, int], tuple[int, ...]] = field(
        repr=False,
        default_factory=lambda: MappingProxyType({}),
    )
    _runtime_state: _RuntimeMetricsState = field(
        repr=False,
        compare=False,
        default_factory=_RuntimeMetricsState,
    )
    _context_metadata_valid: bool = field(repr=False, default=False)
    _primitive_metadata_valid: bool = field(repr=False, default=False)
    _primitive_sha256: str = field(repr=False, default="")
    _integrity_sha256: str = field(repr=False, default="")
    _mint_token: object = field(
        repr=False,
        compare=False,
        default_factory=object,
    )
    _primitives_identity: tuple[ItemPrimitive, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _grid_identity: Mapping[tuple[int, int], tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runtime_state_identity: _RuntimeMetricsState = field(
        init=False,
        repr=False,
        compare=False,
    )
    _snapshot_structure_stamp: tuple[Any, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.primitives) is not tuple
            or self._mint_token not in _PENDING_CONTEXT_MINT_TOKENS
            or any(type(item) is not ItemPrimitive for item in self.primitives)
            or type(self._grid) is not MappingProxyType
            or type(self._context_metadata_valid) is not bool
            or type(self._primitive_metadata_valid) is not bool
            or not _valid_sha256(self._primitive_sha256)
            or not _valid_sha256(self._integrity_sha256)
            or type(self._runtime_state.extraction_call_count) is not int
            or self._runtime_state.extraction_call_count < 0
        ):
            raise ValueError("vector page context integrity invalid")
        object.__setattr__(
            self,
            "_primitives_identity",
            self.primitives,
        )
        object.__setattr__(
            self,
            "_grid_identity",
            self._grid,
        )
        object.__setattr__(
            self,
            "_runtime_state_identity",
            self._runtime_state,
        )
        object.__setattr__(
            self,
            "_snapshot_structure_stamp",
            _snapshot_structure_stamp(self),
        )

    @classmethod
    def from_page(
        cls,
        page: Any,
        *,
        pdf_stem: str,
        page_num: int,
        cell_size: float = DEFAULT_CELL_SIZE,
        drawing_snapshot: Iterable[dict[str, Any]] | None = None,
    ) -> "VectorPageContext":
        primitives = (
            extract_items_from_page(
                page,
                pdf_stem=str(pdf_stem),
                page_num=int(page_num),
            )
            if drawing_snapshot is None
            else extract_items_from_drawings(
                drawing_snapshot,
                pdf_stem=str(pdf_stem),
                page_num=int(page_num),
            )
        )
        return cls._from_minted_primitives(
            primitives,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            pdf_stem=str(pdf_stem),
            page_num=int(page_num),
            cell_size=cell_size,
            extraction_call_count=1,
        )

    @classmethod
    def from_primitives(
        cls,
        primitives: Iterable[ItemPrimitive],
        *,
        page_width: float,
        page_height: float,
        pdf_stem: str | None = None,
        page_num: int | None = None,
        cell_size: float = DEFAULT_CELL_SIZE,
    ) -> "VectorPageContext":
        canonical_primitives = tuple(
            _canonical_primitive_copy(item)
            for item in primitives
        )
        return cls._from_minted_primitives(
            canonical_primitives,
            page_width=page_width,
            page_height=page_height,
            pdf_stem=pdf_stem,
            page_num=page_num,
            cell_size=cell_size,
            extraction_call_count=0,
        )

    @classmethod
    def _from_minted_primitives(
        cls,
        primitives: Iterable[ItemPrimitive],
        *,
        page_width: float,
        page_height: float,
        pdf_stem: str | None,
        page_num: int | None,
        cell_size: float,
        extraction_call_count: int,
    ) -> "VectorPageContext":
        ordered = tuple(sorted(primitives, key=_stable_item_key))
        resolved_pdf_stem = str(pdf_stem if pdf_stem is not None else (ordered[0].pdf if ordered else ""))
        resolved_page_num = int(page_num if page_num is not None else (ordered[0].page if ordered else 0))
        resolved_page_width = float(page_width)
        resolved_page_height = float(page_height)
        resolved_cell_size = max(1.0, float(cell_size))
        grid = _build_grid(
            ordered,
            cell_size=resolved_cell_size,
        )
        primitive_sha256 = _primitive_digest(ordered)
        context_metadata_valid = _valid_context_metadata(
            page_width=resolved_page_width,
            page_height=resolved_page_height,
            pdf_stem=resolved_pdf_stem,
            page_num=resolved_page_num,
            cell_size=resolved_cell_size,
        )
        primitive_metadata_valid = all(
            item.pdf == resolved_pdf_stem
            and type(item.page) is int
            and item.page == resolved_page_num
            for item in ordered
        )
        integrity_sha256 = _context_integrity_digest(
            page_width=resolved_page_width,
            page_height=resolved_page_height,
            cell_size=resolved_cell_size,
            pdf_stem=resolved_pdf_stem,
            page_num=resolved_page_num,
            grid=grid,
            primitive_sha256=primitive_sha256,
        )
        mint_token = object()
        _PENDING_CONTEXT_MINT_TOKENS.add(mint_token)
        try:
            return cls(
                page_width=resolved_page_width,
                page_height=resolved_page_height,
                primitives=ordered,
                pdf_stem=resolved_pdf_stem,
                page_num=resolved_page_num,
                cell_size=resolved_cell_size,
                _grid=grid,
                _runtime_state=_RuntimeMetricsState(
                    extraction_call_count=extraction_call_count,
                ),
                _context_metadata_valid=context_metadata_valid,
                _primitive_metadata_valid=primitive_metadata_valid,
                _primitive_sha256=primitive_sha256,
                _integrity_sha256=integrity_sha256,
                _mint_token=mint_token,
            )
        finally:
            _PENDING_CONTEXT_MINT_TOKENS.discard(mint_token)

    def query_bbox(self, bbox: BBox) -> tuple[ItemPrimitive, ...]:
        """Return primitives intersecting ``bbox`` in stable page order."""
        x0, y0, x1, y1 = (float(value) for value in bbox)
        query = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        indices: set[int] = set()
        for cell in self._cells_for_bbox(query):
            indices.update(self._grid.get(cell, ()))
        self._runtime_state.query_count += 1
        self._runtime_state.candidate_scan_count += len(indices)
        return tuple(
            self.primitives[index]
            for index in sorted(indices)
            if _intersects(self.primitives[index].bbox, query)
        )

    def item_id(self, item: ItemPrimitive) -> str:
        return (
            f"p{int(item.page):03d}_d{int(item.drawing_order):06d}_"
            f"i{int(item.item_index):04d}_{item.op}"
        )

    def runtime_metrics(self) -> dict[str, int | float]:
        return {
            "primitive_count": len(self.primitives),
            "extraction_call_count": self._runtime_state.extraction_call_count,
            "query_count": self._runtime_state.query_count,
            "candidate_scan_count": self._runtime_state.candidate_scan_count,
            "cell_count": len(self._grid),
            "cell_size": self.cell_size,
        }

    def manifest_v1(self) -> dict[str, int | float | str | bool]:
        return {
            "schema_version": "vector_page_context_manifest_v1",
            "pdf_stem": self.pdf_stem,
            "page_num": self.page_num,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "primitive_count": len(self.primitives),
            "primitive_sha256": self._primitive_sha256,
            "consumer_allowed": False,
        }

    def assert_integrity(self) -> None:
        """Fail closed if immutable snapshot containers were replaced."""
        try:
            invalid = (
                self.primitives is not self._primitives_identity
                or self._grid is not self._grid_identity
                or self._runtime_state is not self._runtime_state_identity
                or not self._context_metadata_valid
                or not self._primitive_metadata_valid
                or _snapshot_structure_stamp(self)
                != self._snapshot_structure_stamp
            )
        except (AttributeError, TypeError, ValueError):
            invalid = True
        if invalid:
            raise ValueError("vector page context integrity invalid")

    def _cells_for_bbox(self, bbox: BBox) -> tuple[tuple[int, int], ...]:
        return _cells_for_bbox(
            bbox,
            cell_size=self.cell_size,
        )


def _cells_for_bbox(
    bbox: BBox,
    *,
    cell_size: float,
) -> tuple[tuple[int, int], ...]:
    x0, y0, x1, y1 = bbox
    cx0 = math.floor(x0 / cell_size)
    cy0 = math.floor(y0 / cell_size)
    cx1 = math.floor(x1 / cell_size)
    cy1 = math.floor(y1 / cell_size)
    return tuple(
        (cx, cy)
        for cy in range(cy0, cy1 + 1)
        for cx in range(cx0, cx1 + 1)
    )


def _canonical_primitive_copy(item: ItemPrimitive) -> ItemPrimitive:
    if type(item) is not ItemPrimitive:
        raise ValueError("vector page context primitive type invalid")
    return ItemPrimitive(
        pdf=item.pdf,
        page=item.page,
        drawing_order=item.drawing_order,
        item_index=item.item_index,
        op=item.op,
        points=item.points,
        bbox=item.bbox,
        draw_type=item.draw_type,
        stroke_width=item.stroke_width,
    )


def _build_grid(
    primitives: tuple[ItemPrimitive, ...],
    *,
    cell_size: float,
) -> Mapping[tuple[int, int], tuple[int, ...]]:
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, primitive in enumerate(primitives):
        for cell in _cells_for_bbox(
            primitive.bbox,
            cell_size=cell_size,
        ):
            grid[cell].append(index)
    return MappingProxyType({
        cell: tuple(indices)
        for cell, indices in grid.items()
    })


def _primitive_digest(primitives: tuple[ItemPrimitive, ...]) -> str:
    digest = hashlib.sha256()
    for item in primitives:
        digest.update(
            (
                f"{item.pdf}\x1f{int(item.page)}\x1f{int(item.drawing_order)}\x1f"
                f"{int(item.item_index)}\x1f{item.op}\x1f{item.draw_type}\x1f"
                f"{item.stroke_width!r}\x1f"
            ).encode("utf-8")
        )
        for value in item.bbox:
            digest.update(float(value).hex().encode("ascii"))
            digest.update(b"\x1e")
        for x, y in item.points:
            digest.update(float(x).hex().encode("ascii"))
            digest.update(b",")
            digest.update(float(y).hex().encode("ascii"))
            digest.update(b"\x1d")
        digest.update(b"\n")
    return digest.hexdigest()


def _context_integrity_digest(
    *,
    page_width: float,
    page_height: float,
    cell_size: float,
    pdf_stem: str,
    page_num: int,
    grid: Mapping[tuple[int, int], tuple[int, ...]],
    primitive_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"vector_page_context_integrity_v1\n")
    for value in (
        page_width,
        page_height,
        cell_size,
    ):
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\x1f")
    digest.update(pdf_stem.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(str(int(page_num)).encode("ascii"))
    digest.update(b"\x1f")
    digest.update(primitive_sha256.encode("ascii"))
    digest.update(b"\n")
    for (cell_x, cell_y), indices in sorted(grid.items()):
        digest.update(f"{int(cell_x)},{int(cell_y)}:".encode("ascii"))
        for index in indices:
            digest.update(str(int(index)).encode("ascii"))
            digest.update(b",")
        digest.update(b"\n")
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_context_metadata(
    *,
    page_width: Any,
    page_height: Any,
    pdf_stem: Any,
    page_num: Any,
    cell_size: Any,
) -> bool:
    return (
        type(page_width) is float
        and math.isfinite(page_width)
        and page_width > 0.0
        and type(page_height) is float
        and math.isfinite(page_height)
        and page_height > 0.0
        and type(pdf_stem) is str
        and bool(pdf_stem)
        and type(page_num) is int
        and page_num >= 1
        and type(cell_size) is float
        and math.isfinite(cell_size)
        and cell_size > 0.0
    )


def _snapshot_structure_stamp(
    context: VectorPageContext,
) -> tuple[Any, ...]:
    return (
        type(context.page_width),
        context.page_width,
        type(context.page_height),
        context.page_height,
        type(context.pdf_stem),
        context.pdf_stem,
        type(context.page_num),
        context.page_num,
        type(context.cell_size),
        context.cell_size,
        type(context.primitives),
        len(context.primitives),
        type(context._grid),
        len(context._grid),
        type(context._context_metadata_valid),
        context._context_metadata_valid,
        type(context._primitive_metadata_valid),
        context._primitive_metadata_valid,
        type(context._primitive_sha256),
        context._primitive_sha256,
        type(context._integrity_sha256),
        context._integrity_sha256,
    )
