"""Pure bounded geometry kernel for vector primitive contact graphs.

The kernel knows only ordered primitive point sequences and integer primitive
indices.  It deliberately has no knowledge of PDFs, pages, ROIs, glyph IDs,
authority digests, recognition, or product consumers.  The page authority
envelope maps its immutable index result back to trusted primitive identities.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal


CONTACT_GEOMETRY_VERSION = "exact_contact_grid_v1"

Point = tuple[float, float]
PrimitivePoints = tuple[Point, ...]
ContactKind = Literal[
    "endpoint_endpoint",
    "endpoint_segment",
    "proper_intersection",
    "collinear_overlap",
]
_CONTACT_KINDS = frozenset({
    "endpoint_endpoint",
    "endpoint_segment",
    "proper_intersection",
    "collinear_overlap",
})


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


@dataclass(frozen=True, slots=True)
class ContactGeometryLimits:
    coordinate_epsilon: float = 1e-9
    grid_cell_size: float = 4.0
    max_points_per_primitive: int = 512
    max_total_segments: int = 65_536
    max_grid_references: int = 262_144
    max_cells_per_segment: int = 256
    max_segments_per_cell: int = 256
    max_segment_pair_checks: int = 1_000_000
    max_contact_edges: int = 16_384

    def __post_init__(self) -> None:
        if (
            not _finite_number(self.coordinate_epsilon)
            or self.coordinate_epsilon <= 0.0
            or not _finite_number(self.grid_cell_size)
            or self.grid_cell_size <= 0.0
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.max_points_per_primitive,
                    self.max_total_segments,
                    self.max_grid_references,
                    self.max_cells_per_segment,
                    self.max_segments_per_cell,
                    self.max_segment_pair_checks,
                    self.max_contact_edges,
                )
            )
        ):
            raise ValueError("contact geometry limits invalid")

    def contract_v1(self) -> dict[str, Any]:
        return {
            "schema_version": "contact_geometry_contract_v1",
            "contract_version": CONTACT_GEOMETRY_VERSION,
            "semantic_contacts": [
                "endpoint_endpoint",
                "endpoint_segment",
                "proper_intersection",
                "collinear_overlap",
            ],
            "coordinate_epsilon": self.coordinate_epsilon,
            "grid_cell_size": self.grid_cell_size,
            "candidate_adaptive_scaling": False,
            "bbox_proximity_merge": False,
            "pitch_or_phase_search": False,
            "capacities": {
                "max_points_per_primitive": self.max_points_per_primitive,
                "max_total_segments": self.max_total_segments,
                "max_grid_references": self.max_grid_references,
                "max_cells_per_segment": self.max_cells_per_segment,
                "max_segments_per_cell": self.max_segments_per_cell,
                "max_segment_pair_checks": self.max_segment_pair_checks,
                "max_contact_edges": self.max_contact_edges,
            },
        }


DEFAULT_CONTACT_GEOMETRY_LIMITS = ContactGeometryLimits()


@dataclass(frozen=True, slots=True)
class ContactGraphStats:
    primitive_count: int
    point_count: int
    segment_count: int
    grid_cell_count: int
    grid_reference_count: int
    segment_pair_check_count: int
    contact_edge_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "primitive_count": self.primitive_count,
            "point_count": self.point_count,
            "segment_count": self.segment_count,
            "grid_cell_count": self.grid_cell_count,
            "grid_reference_count": self.grid_reference_count,
            "segment_pair_check_count": self.segment_pair_check_count,
            "contact_edge_count": self.contact_edge_count,
        }


@dataclass(frozen=True, slots=True)
class ContactGraphResult:
    reason: str | None
    contact_edges: tuple[tuple[int, int, tuple[ContactKind, ...]], ...]
    components: tuple[tuple[int, ...], ...]
    stats: ContactGraphStats

    def __post_init__(self) -> None:
        pairs = tuple((row[0], row[1]) for row in self.contact_edges)
        flattened = tuple(
            sorted(index for component in self.components for index in component)
        )
        if (
            any(
                type(value) is not int or value < 0
                for value in self.stats.to_dict().values()
            )
            or self.stats.contact_edge_count != len(self.contact_edges)
            or (self.reason is None) != bool(self.components)
            or self.contact_edges != tuple(sorted(set(self.contact_edges)))
            or pairs != tuple(sorted(set(pairs)))
            or any(
                type(left) is not int
                or type(right) is not int
                or left < 0
                or left >= right
                or right >= self.stats.primitive_count
                or not kinds
                or kinds != tuple(sorted(set(kinds)))
                or any(kind not in _CONTACT_KINDS for kind in kinds)
                for left, right, kinds in self.contact_edges
            )
            or self.components != tuple(sorted(
                self.components,
                key=lambda row: row[0],
            ))
            or any(
                not component
                or component != tuple(sorted(set(component)))
                for component in self.components
            )
            or (
                self.reason is None
                and flattened != tuple(range(self.stats.primitive_count))
            )
            or (
                self.reason is not None
                and (self.contact_edges or self.components)
            )
        ):
            raise ValueError("contact graph result invalid")


def build_contact_graph(
    primitives: tuple[PrimitivePoints, ...],
    *,
    limits: ContactGeometryLimits = DEFAULT_CONTACT_GEOMETRY_LIMITS,
) -> ContactGraphResult:
    """Build a bounded exact-contact graph over ordered primitive points."""
    if type(primitives) is not tuple or not isinstance(
        limits,
        ContactGeometryLimits,
    ):
        raise ValueError("contact geometry input invalid")
    point_count = sum(
        len(row) if type(row) is tuple else 0
        for row in primitives
    )
    if not primitives:
        return _result(
            reason="contact_primitive_shape_or_point_capacity_exceeded",
            primitive_count=0,
            point_count=0,
            segment_count=0,
        )
    preflight_reason, observed_segments = _preflight(primitives, limits)
    if preflight_reason is not None:
        return _result(
            reason=preflight_reason,
            primitive_count=len(primitives),
            point_count=point_count,
            segment_count=observed_segments,
        )

    segment_rows: list[tuple[int, int, Point, Point]] = []
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    grid_reference_count = 0
    for primitive_index, points in enumerate(primitives):
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            if _point_distance(start, end) <= limits.coordinate_epsilon:
                continue
            row_index = len(segment_rows)
            segment_rows.append((primitive_index, segment_index, start, end))
            cells = _segment_cells(start, end, limits)
            if len(cells) > limits.max_cells_per_segment:
                return _result(
                    reason="contact_segment_grid_span_exceeded",
                    primitive_count=len(primitives),
                    point_count=point_count,
                    segment_count=len(segment_rows),
                    grid_cell_count=len(grid),
                    grid_reference_count=grid_reference_count,
                )
            grid_reference_count += len(cells)
            if grid_reference_count > limits.max_grid_references:
                return _result(
                    reason="contact_grid_reference_capacity_exceeded",
                    primitive_count=len(primitives),
                    point_count=point_count,
                    segment_count=len(segment_rows),
                    grid_cell_count=len(grid),
                    grid_reference_count=grid_reference_count,
                )
            for cell in cells:
                grid[cell].append(row_index)

    if any(
        len(rows) > limits.max_segments_per_cell
        for rows in grid.values()
    ):
        return _result(
            reason="contact_dense_grid_cell_exceeded",
            primitive_count=len(primitives),
            point_count=point_count,
            segment_count=len(segment_rows),
            grid_cell_count=len(grid),
            grid_reference_count=grid_reference_count,
        )

    pair_checks = 0
    seen_segment_pairs: set[tuple[int, int]] = set()
    kinds_by_pair: dict[tuple[int, int], set[ContactKind]] = defaultdict(set)
    for cell in sorted(grid):
        rows = sorted(set(grid[cell]))
        for left_offset, left_index in enumerate(rows):
            left_primitive, _left_segment, left_start, left_end = (
                segment_rows[left_index]
            )
            for right_index in rows[left_offset + 1:]:
                right_primitive, _right_segment, right_start, right_end = (
                    segment_rows[right_index]
                )
                pair_checks += 1
                if pair_checks > limits.max_segment_pair_checks:
                    return _result(
                        reason="contact_pair_work_capacity_exceeded",
                        primitive_count=len(primitives),
                        point_count=point_count,
                        segment_count=len(segment_rows),
                        grid_cell_count=len(grid),
                        grid_reference_count=grid_reference_count,
                        pair_checks=pair_checks,
                    )
                segment_pair = tuple(sorted((left_index, right_index)))
                if segment_pair in seen_segment_pairs:
                    continue
                seen_segment_pairs.add(segment_pair)
                if left_primitive == right_primitive:
                    continue
                kind = segment_contact_kind(
                    left_start,
                    left_end,
                    right_start,
                    right_end,
                    limits=limits,
                )
                if kind is None:
                    continue
                primitive_pair = tuple(sorted((left_primitive, right_primitive)))
                kinds_by_pair[primitive_pair].add(kind)
                if len(kinds_by_pair) > limits.max_contact_edges:
                    return _result(
                        reason="contact_edge_capacity_exceeded",
                        primitive_count=len(primitives),
                        point_count=point_count,
                        segment_count=len(segment_rows),
                        grid_cell_count=len(grid),
                        grid_reference_count=grid_reference_count,
                        pair_checks=pair_checks,
                    )

    edges = tuple(
        (left, right, tuple(sorted(kinds)))
        for (left, right), kinds in sorted(kinds_by_pair.items())
    )
    return _result(
        reason=None,
        primitive_count=len(primitives),
        point_count=point_count,
        segment_count=len(segment_rows),
        grid_cell_count=len(grid),
        grid_reference_count=grid_reference_count,
        pair_checks=pair_checks,
        edges=edges,
        components=_connected_components(len(primitives), edges),
    )


def segment_contact_kind(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    *,
    limits: ContactGeometryLimits = DEFAULT_CONTACT_GEOMETRY_LIMITS,
) -> ContactKind | None:
    epsilon = limits.coordinate_epsilon
    if not _bbox_intersects(_segment_bbox(a, b), _segment_bbox(c, d), epsilon):
        return None
    if _positive_collinear_overlap(a, b, c, d, epsilon):
        return "collinear_overlap"
    if any(
        _point_distance(left, right) <= epsilon
        for left, right in ((a, c), (a, d), (b, c), (b, d))
    ):
        return "endpoint_endpoint"
    if any((
        _point_segment_distance(a, c, d, epsilon) <= epsilon,
        _point_segment_distance(b, c, d, epsilon) <= epsilon,
        _point_segment_distance(c, a, b, epsilon) <= epsilon,
        _point_segment_distance(d, a, b, epsilon) <= epsilon,
    )):
        return "endpoint_segment"
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return "proper_intersection"
    return None


def _preflight(
    primitives: tuple[PrimitivePoints, ...],
    limits: ContactGeometryLimits,
) -> tuple[str | None, int]:
    segment_count = 0
    for points in primitives:
        if (
            type(points) is not tuple
            or not 2 <= len(points) <= limits.max_points_per_primitive
            or any(
                type(point) is not tuple
                or len(point) != 2
                or any(not _finite_number(value) for value in point)
                for point in points
            )
        ):
            return "contact_primitive_shape_or_point_capacity_exceeded", segment_count
        usable_segment_count = 0
        for start, end in zip(points, points[1:]):
            dx = float(end[0]) - float(start[0])
            dy = float(end[1]) - float(start[1])
            grid_span = max(abs(dx), abs(dy)) / limits.grid_cell_size
            if not all(math.isfinite(row) for row in (dx, dy, grid_span)):
                return "contact_coordinate_span_nonfinite", segment_count
            if grid_span > limits.max_cells_per_segment:
                return "contact_segment_grid_span_exceeded", segment_count
            segment_count += 1
            if segment_count > limits.max_total_segments:
                return "contact_segment_capacity_exceeded", segment_count
            if math.hypot(dx, dy) > limits.coordinate_epsilon:
                usable_segment_count += 1
        if usable_segment_count == 0:
            return "contact_degenerate_primitive", segment_count
    return None, segment_count


def _result(
    *,
    reason: str | None,
    primitive_count: int,
    point_count: int,
    segment_count: int,
    grid_cell_count: int = 0,
    grid_reference_count: int = 0,
    pair_checks: int = 0,
    edges: tuple[tuple[int, int, tuple[ContactKind, ...]], ...] = (),
    components: tuple[tuple[int, ...], ...] = (),
) -> ContactGraphResult:
    return ContactGraphResult(
        reason=reason,
        contact_edges=edges if reason is None else (),
        components=components if reason is None else (),
        stats=ContactGraphStats(
            primitive_count=primitive_count,
            point_count=point_count,
            segment_count=segment_count,
            grid_cell_count=grid_cell_count,
            grid_reference_count=grid_reference_count,
            segment_pair_check_count=pair_checks,
            contact_edge_count=len(edges) if reason is None else 0,
        ),
    )


def _connected_components(
    primitive_count: int,
    edges: tuple[tuple[int, int, tuple[ContactKind, ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    parents = list(range(primitive_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right, _kinds in edges:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(primitive_count):
        groups[find(index)].append(index)
    return tuple(
        tuple(indices)
        for _root, indices in sorted(groups.items())
    )


def _segment_cells(
    start: Point,
    end: Point,
    limits: ContactGeometryLimits,
) -> tuple[tuple[int, int], ...]:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    steps = max(1, int(math.ceil(
        max(abs(dx), abs(dy)) / limits.grid_cell_size
    )))
    cells: set[tuple[int, int]] = set()
    previous = start
    for step in range(1, steps + 1):
        ratio = step / steps
        current = (
            float(start[0]) + dx * ratio,
            float(start[1]) + dy * ratio,
        )
        min_x, max_x = sorted((float(previous[0]), float(current[0])))
        min_y, max_y = sorted((float(previous[1]), float(current[1])))
        min_x -= limits.coordinate_epsilon
        max_x += limits.coordinate_epsilon
        min_y -= limits.coordinate_epsilon
        max_y += limits.coordinate_epsilon
        for cell_x in range(
            math.floor(min_x / limits.grid_cell_size),
            math.floor(max_x / limits.grid_cell_size) + 1,
        ):
            for cell_y in range(
                math.floor(min_y / limits.grid_cell_size),
                math.floor(max_y / limits.grid_cell_size) + 1,
            ):
                cells.add((cell_x, cell_y))
                if len(cells) > limits.max_cells_per_segment:
                    return tuple(sorted(cells))
        previous = current
    return tuple(sorted(cells))


def _positive_collinear_overlap(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    epsilon: float,
) -> bool:
    ab_length = _point_distance(a, b)
    cd_length = _point_distance(c, d)
    if (
        ab_length <= epsilon
        or cd_length <= epsilon
        or not math.isfinite(ab_length)
        or not math.isfinite(cd_length)
    ):
        return False
    if cd_length > ab_length:
        a, b, c, d = c, d, a, b
        ab_length = cd_length
    if (
        _point_line_distance(c, a, b, epsilon) > epsilon
        or _point_line_distance(d, a, b, epsilon) > epsilon
    ):
        return False
    unit_x = (float(b[0]) - float(a[0])) / ab_length
    unit_y = (float(b[1]) - float(a[1])) / ab_length
    c_projection = (
        (float(c[0]) - float(a[0])) * unit_x
        + (float(c[1]) - float(a[1])) * unit_y
    )
    d_projection = (
        (float(d[0]) - float(a[0])) * unit_x
        + (float(d[1]) - float(a[1])) * unit_y
    )
    overlap = min(ab_length, max(c_projection, d_projection)) - max(
        0.0,
        min(c_projection, d_projection),
    )
    return overlap > epsilon


def _point_line_distance(
    point: Point,
    start: Point,
    end: Point,
    epsilon: float,
) -> float:
    length = _point_distance(start, end)
    if length <= epsilon or not math.isfinite(length):
        return _point_distance(point, start)
    return abs(_orientation(start, end, point)) / length


def _point_segment_distance(
    point: Point,
    start: Point,
    end: Point,
    epsilon: float,
) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= epsilon or not math.isfinite(length):
        return _point_distance(point, start)
    unit_x = dx / length
    unit_y = dy / length
    projection = (
        (float(point[0]) - float(start[0])) * unit_x
        + (float(point[1]) - float(start[1])) * unit_y
    )
    projection = min(length, max(0.0, projection))
    closest = (
        float(start[0]) + projection * unit_x,
        float(start[1]) + projection * unit_y,
    )
    return _point_distance(point, closest)


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (
        (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
        - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
    )


def _segment_bbox(start: Point, end: Point) -> tuple[float, float, float, float]:
    return (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    )


def _bbox_intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    epsilon: float,
) -> bool:
    return not (
        left[2] < right[0] - epsilon
        or right[2] < left[0] - epsilon
        or left[3] < right[1] - epsilon
        or right[3] < left[1] - epsilon
    )


def _point_distance(left: Point, right: Point) -> float:
    return math.hypot(
        float(left[0]) - float(right[0]),
        float(left[1]) - float(right[1]),
    )
