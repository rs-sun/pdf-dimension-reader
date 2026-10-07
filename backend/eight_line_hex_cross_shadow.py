"""Dump-only topology shadow for an eight-line hexagon/cross candidate.

This module deliberately does not mint decimal-anchor authority.  It reports
one exact local vector encoding family with complete primitive provenance so
that an independent holdout can decide whether a later semantic adapter is
safe.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable

from vector_draw_items import ItemPrimitive


SCHEMA_VERSION = "vector_eight_line_hex_cross_shadow_v1"
STATS_SCHEMA_VERSION = "vector_eight_line_hex_cross_shadow_stats_v1"
DIGEST_SCHEMA_VERSION = "vector_eight_line_hex_cross_digest_v1"
SOURCE = "eight_line_hex_cross_shadow"
ENDPOINT_TOLERANCE_PT = 1e-3
MIN_BBOX_SHORT_PT = 0.45
MAX_BBOX_SHORT_PT = 4.8
MIN_BBOX_LONG_PT = 0.75
MAX_BBOX_LONG_PT = 7.2
MIN_BBOX_RATIO = 1.18
MAX_BBOX_RATIO = 1.40
MIN_BBOX_FILL_RATIO = 0.65
MAX_BBOX_FILL_RATIO = 0.82
MAX_OPPOSITE_EDGE_RELATIVE_ERROR = 0.22
MAX_SHADOW_CANDIDATES = 5000
ROW_KEYS_V1 = frozenset({
    "schema_version",
    "candidate_id",
    "page_index",
    "bbox",
    "center",
    "boundary_polygon",
    "source",
    "source_primitive_ids",
    "source_primitives",
    "topology_metrics",
    "topology_digest_schema_version",
    "source_primitive_digest",
    "consumer_allowed",
    "release_allowed",
    "feeds_display_l3",
    "ocr_called",
})
PROVENANCE_KEYS_V1 = frozenset({
    "primitive_id",
    "drawing_order",
    "item_index",
    "op",
    "role",
    "points",
    "draw_type",
    "stroke_width",
})
STATS_KEYS_V1 = frozenset({
    "schema_version",
    "page_index",
    "status",
    "primitive_count",
    "drawing_group_count",
    "window_count",
    "topology_match_count",
    "count",
    "candidate_limit",
    "consumer_allowed_count",
    "release_allowed_count",
    "consumer_allowed",
})

Point = tuple[float, float]


def detect_eight_line_hex_cross_shadows_v1(
    item_primitives: Iterable[ItemPrimitive],
    *,
    page_index: int,
) -> dict[str, Any]:
    """Group/sort in O(N log N), then scan fixed eight-item windows."""
    page_num = int(page_index) + 1
    primitives = tuple(item_primitives)
    groups: dict[tuple[int, int], list[ItemPrimitive]] = defaultdict(list)
    for item in primitives:
        if type(item.page) is int and item.page == page_num:
            groups[(item.page, item.drawing_order)].append(item)

    ordered_groups = [
        (
            group_key,
            sorted(
                groups[group_key],
                key=lambda item: (item.item_index, item.op),
            ),
        )
        for group_key in sorted(groups)
    ]
    rows: list[dict[str, Any]] = []
    window_count = sum(
        max(0, len(group) - 7)
        for _group_key, group in ordered_groups
    )
    topology_match_count = 0
    status = "ok"
    for _group_key, group in ordered_groups:
        for start in range(max(0, len(group) - 7)):
            candidate = _classify_window(
                group[start:start + 8],
                page_index=page_index,
            )
            if candidate is None:
                continue
            topology_match_count += 1
            if len(rows) >= MAX_SHADOW_CANDIDATES:
                rows = []
                status = "candidate_limit_exceeded"
                break
            rows.append(candidate)
        if status != "ok":
            break

    if status == "ok":
        rows.sort(key=_source_order_key)
        for index, row in enumerate(rows):
            row["candidate_id"] = (
                f"p{page_num:03d}_hex_cross_{index:06d}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "status": status,
        "vector_eight_line_hex_cross_shadow_v1": rows,
        "vector_eight_line_hex_cross_shadow_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": int(page_index),
            "status": status,
            "primitive_count": len(primitives),
            "drawing_group_count": len(groups),
            "window_count": window_count,
            "topology_match_count": topology_match_count,
            "count": len(rows),
            "candidate_limit": MAX_SHADOW_CANDIDATES,
            "consumer_allowed_count": 0,
            "release_allowed_count": 0,
            "consumer_allowed": False,
        },
        "consumer_allowed": False,
        "release_allowed": False,
        "feeds_display_l3": False,
        "ocr_called": False,
    }


def normalize_eight_line_hex_cross_shadow_row_v1(
    value: Any,
    *,
    page_index: int,
    page_context: Any | None = None,
) -> dict[str, Any] | None:
    """Replay one shadow row and optionally bind it to an immutable page."""
    page_primitive_index = None
    if page_context is not None:
        page_primitive_index = _build_page_primitive_index(
            page_context,
            page_index=page_index,
        )
        if page_primitive_index is None:
            return None
    return _normalize_eight_line_hex_cross_shadow_row_v1(
        value,
        page_index=page_index,
        page_primitive_index=page_primitive_index,
    )


def normalize_eight_line_hex_cross_shadow_rows_v1(
    values: Any,
    *,
    page_index: int,
    page_context: Any | None = None,
) -> list[dict[str, Any]] | None:
    """Replay a batch against one integrity check and primitive-ID index."""
    if not isinstance(values, list):
        return None
    candidate_ids = [
        value.get("candidate_id") if isinstance(value, dict) else None
        for value in values
    ]
    if (
        any(type(candidate_id) is not str for candidate_id in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        return None
    page_primitive_index = None
    if page_context is not None:
        page_primitive_index = _build_page_primitive_index(
            page_context,
            page_index=page_index,
        )
        if page_primitive_index is None:
            return None
    normalized = []
    seen_provenance: set[tuple[tuple[str, ...], str]] = set()
    for value in values:
        row = _normalize_eight_line_hex_cross_shadow_row_v1(
            value,
            page_index=page_index,
            page_primitive_index=page_primitive_index,
        )
        if row is None:
            return None
        provenance = (
            tuple(row["source_primitive_ids"]),
            row["source_primitive_digest"],
        )
        if provenance in seen_provenance:
            return None
        seen_provenance.add(provenance)
        normalized.append(row)
    return normalized


def validate_eight_line_hex_cross_shadow_stats_v1(
    value: Any,
    *,
    page_index: int,
    row_count: int,
    expected_status: str,
    item_primitives: Iterable[ItemPrimitive],
) -> bool:
    """Validate exact stats against the same page primitive snapshot."""
    integer_keys = (
        "primitive_count",
        "drawing_group_count",
        "window_count",
        "topology_match_count",
        "count",
        "candidate_limit",
        "consumer_allowed_count",
        "release_allowed_count",
    )
    if (
        not isinstance(value, dict)
        or set(value) != STATS_KEYS_V1
        or value.get("schema_version") != STATS_SCHEMA_VERSION
        or type(value.get("page_index")) is not int
        or value.get("page_index") != int(page_index)
        or type(value.get("status")) is not str
        or value.get("status") not in {
            "ok",
            "candidate_limit_exceeded",
        }
        or type(expected_status) is not str
        or value.get("status") != expected_status
        or any(
            type(value.get(key)) is not int or value[key] < 0
            for key in integer_keys
        )
        or type(row_count) is not int
        or row_count < 0
        or value.get("count") != row_count
        or value.get("candidate_limit") != MAX_SHADOW_CANDIDATES
        or value.get("consumer_allowed_count") != 0
        or value.get("release_allowed_count") != 0
        or value.get("consumer_allowed") is not False
    ):
        return False

    try:
        primitives = tuple(item_primitives)
        page_num = int(page_index) + 1
        group_sizes: dict[tuple[int, int], int] = defaultdict(int)
        for item in primitives:
            if type(item.page) is int and item.page == page_num:
                group_sizes[(item.page, item.drawing_order)] += 1
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    expected_window_count = sum(
        max(0, group_size - 7)
        for group_size in group_sizes.values()
    )
    if (
        value["primitive_count"] != len(primitives)
        or value["drawing_group_count"] != len(group_sizes)
        or value["window_count"] != expected_window_count
        or value["topology_match_count"] > value["window_count"]
    ):
        return False
    if value["status"] == "ok":
        return bool(
            value["topology_match_count"] == value["count"]
            and value["count"] <= MAX_SHADOW_CANDIDATES
        )
    return bool(
        value["count"] == 0
        and value["topology_match_count"] == MAX_SHADOW_CANDIDATES + 1
    )


def _normalize_eight_line_hex_cross_shadow_row_v1(
    value: Any,
    *,
    page_index: int,
    page_primitive_index: dict[str, ItemPrimitive] | None,
) -> dict[str, Any] | None:
    if (
        not isinstance(value, dict)
        or set(value) != ROW_KEYS_V1
        or value.get("schema_version") != SCHEMA_VERSION
        or type(value.get("candidate_id")) is not str
        or not _candidate_id_matches_page(
            value.get("candidate_id"),
            page_index=page_index,
        )
        or type(value.get("page_index")) is not int
        or value.get("page_index") != int(page_index)
        or value.get("source") != SOURCE
        or value.get("topology_digest_schema_version")
        != DIGEST_SCHEMA_VERSION
        or value.get("consumer_allowed") is not False
        or value.get("release_allowed") is not False
        or value.get("feeds_display_l3") is not False
        or value.get("ocr_called") is not False
    ):
        return None
    provenance = value.get("source_primitives")
    source_ids = value.get("source_primitive_ids")
    if (
        not isinstance(provenance, list)
        or len(provenance) != 8
        or any(
            not isinstance(row, dict)
            or set(row) != PROVENANCE_KEYS_V1
            for row in provenance
        )
        or not isinstance(source_ids, list)
        or len(source_ids) != 8
        or len(set(source_ids)) != 8
        or source_ids != [
            row.get("primitive_id") for row in provenance
        ]
    ):
        return None
    if [row.get("role") for row in provenance] != (
        ["boundary"] * 6 + ["diagonal"] * 2
    ):
        return None
    try:
        replay_items = [
            _item_from_provenance(
                row,
                page_num=int(page_index) + 1,
            )
            for row in provenance
        ]
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    replay = _classify_window(
        replay_items,
        page_index=page_index,
        canonicalize_provenance=False,
    )
    if replay is None:
        return None
    try:
        for key in ROW_KEYS_V1 - {"candidate_id"}:
            if _canonical_json(value[key]) != _canonical_json(replay[key]):
                return None
    except (TypeError, ValueError, RecursionError):
        return None
    if (
        page_primitive_index is not None
        and not _matches_page_primitive_index(
            replay,
            page_primitive_index=page_primitive_index,
        )
    ):
        return None
    replay["candidate_id"] = value["candidate_id"]
    return replay


def _candidate_id_matches_page(
    value: str,
    *,
    page_index: int,
) -> bool:
    prefix = f"p{int(page_index) + 1:03d}_hex_cross_"
    suffix = value[len(prefix):] if value.startswith(prefix) else ""
    return bool(
        len(suffix) == 6
        and suffix.isascii()
        and suffix.isdigit()
    )


def _classify_window(
    items: list[ItemPrimitive],
    *,
    page_index: int,
    canonicalize_provenance: bool = True,
) -> dict[str, Any] | None:
    if len(items) != 8 or any(item.op != "l" for item in items):
        return None
    indices = [item.item_index for item in items]
    if indices != list(range(indices[0], indices[0] + 8)):
        return None
    if len({item.drawing_order for item in items}) != 1:
        return None
    if len({item.draw_type for item in items}) != 1:
        return None
    if not _same_stroke_width(items):
        return None
    if any(_line_points(item) is None for item in items):
        return None

    boundary = _boundary_cycle(items[:6])
    if boundary is None:
        return None
    vertices, area, bbox, edge_lengths = boundary
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    oriented_extents = _principal_axis_extents(vertices)
    if oriented_extents is None:
        return None
    short, long = sorted(oriented_extents)
    bbox_area = short * long
    if not (
        MIN_BBOX_SHORT_PT <= short <= MAX_BBOX_SHORT_PT
        and MIN_BBOX_LONG_PT <= long <= MAX_BBOX_LONG_PT
    ):
        return None
    bbox_ratio = long / short
    fill_ratio = area / bbox_area
    if not MIN_BBOX_RATIO <= bbox_ratio <= MAX_BBOX_RATIO:
        return None
    if not MIN_BBOX_FILL_RATIO <= fill_ratio <= MAX_BBOX_FILL_RATIO:
        return None
    opposite_errors = [
        abs(edge_lengths[index] - edge_lengths[index + 3])
        / max(edge_lengths[index], edge_lengths[index + 3])
        for index in range(3)
    ]
    max_opposite_error = max(opposite_errors)
    if max_opposite_error > MAX_OPPOSITE_EDGE_RELATIVE_ERROR:
        return None

    diagonal_pairs: list[tuple[int, int]] = []
    for item in items[6:]:
        points = _line_points(item)
        if points is None:
            return None
        endpoint_indices = [
            _unique_vertex_index(point, vertices)
            for point in points
        ]
        if any(index is None for index in endpoint_indices):
            return None
        first, second = endpoint_indices
        if first == second or _cycle_distance(first, second, 6) != 3:
            return None
        diagonal_pairs.append((int(first), int(second)))
    normalized_pairs = {
        frozenset(pair) for pair in diagonal_pairs
    }
    if (
        len(normalized_pairs) != 2
        or len(set(diagonal_pairs[0] + diagonal_pairs[1])) != 4
    ):
        return None
    first_segment = (
        vertices[diagonal_pairs[0][0]],
        vertices[diagonal_pairs[0][1]],
    )
    second_segment = (
        vertices[diagonal_pairs[1][0]],
        vertices[diagonal_pairs[1][1]],
    )
    if not _properly_intersects(first_segment, second_segment):
        return None

    source_primitives = [
        _primitive_provenance(item, role=(
            "boundary" if index < 6 else "diagonal"
        ))
        for index, item in enumerate(items)
    ]
    source_primitive_ids = [
        row["primitive_id"] for row in source_primitives
    ]
    metrics = {
        "oriented_bbox_short_pt": _round(short),
        "oriented_bbox_long_pt": _round(long),
        "oriented_bbox_ratio": _round(bbox_ratio),
        "polygon_area": _round(area),
        "oriented_bbox_fill_ratio": _round(fill_ratio),
        "max_opposite_edge_relative_error": _round(max_opposite_error),
        "proper_diagonal_intersection": True,
    }
    boundary_polygon = [
        [_round(point[0]), _round(point[1])]
        for point in vertices
    ]
    digest_payload = {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "page_index": int(page_index),
        "source": SOURCE,
        "source_primitives": source_primitives,
        "boundary_polygon": boundary_polygon,
        "topology_metrics": metrics,
    }
    topology_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    row = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": "",
        "page_index": int(page_index),
        "bbox": {
            "x": _round(bbox[0]),
            "y": _round(bbox[1]),
            "w": _round(width),
            "h": _round(height),
        },
        "center": {
            "x": _round((bbox[0] + bbox[2]) / 2.0),
            "y": _round((bbox[1] + bbox[3]) / 2.0),
        },
        "boundary_polygon": boundary_polygon,
        "source": SOURCE,
        "source_primitive_ids": source_primitive_ids,
        "source_primitives": source_primitives,
        "topology_metrics": metrics,
        "topology_digest_schema_version": DIGEST_SCHEMA_VERSION,
        "source_primitive_digest": topology_digest,
        "consumer_allowed": False,
        "release_allowed": False,
        "feeds_display_l3": False,
        "ocr_called": False,
    }
    if not canonicalize_provenance:
        return row
    try:
        canonical_items = [
            _item_from_provenance(
                provenance,
                page_num=int(page_index) + 1,
            )
            for provenance in source_primitives
        ]
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return _classify_window(
        canonical_items,
        page_index=page_index,
        canonicalize_provenance=False,
    )


def _boundary_cycle(
    items: list[ItemPrimitive],
) -> tuple[list[Point], float, tuple[float, float, float, float], list[float]] | None:
    endpoints = [
        point
        for item in items
        for point in (_line_points(item) or ())
    ]
    if len(endpoints) != 12:
        return None
    parent = list(range(len(endpoints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            if _distance(endpoints[left], endpoints[right]) <= ENDPOINT_TOLERANCE_PT:
                union(left, right)
    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(endpoints)):
        clusters[find(index)].append(index)
    members = sorted(clusters.values(), key=lambda group: min(group))
    if len(members) != 6 or any(len(group) != 2 for group in members):
        return None
    if any(
        _distance(endpoints[group[0]], endpoints[group[1]])
        > ENDPOINT_TOLERANCE_PT
        for group in members
    ):
        return None

    representatives = [
        (
            sum(endpoints[index][0] for index in group) / len(group),
            sum(endpoints[index][1] for index in group) / len(group),
        )
        for group in members
    ]
    endpoint_to_cluster = {
        endpoint_index: cluster_index
        for cluster_index, group in enumerate(members)
        for endpoint_index in group
    }
    edges = []
    adjacency: dict[int, set[int]] = defaultdict(set)
    for item_index in range(6):
        left = endpoint_to_cluster[item_index * 2]
        right = endpoint_to_cluster[item_index * 2 + 1]
        if left == right:
            return None
        edge = frozenset((left, right))
        if edge in edges:
            return None
        edges.append(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    if any(len(adjacency[index]) != 2 for index in range(6)):
        return None

    start = min(
        range(6),
        key=lambda index: representatives[index],
    )
    neighbor_candidates = sorted(
        adjacency[start],
        key=lambda index: representatives[index],
    )
    cycles = [
        _walk_cycle(
            start,
            first_neighbor,
            adjacency,
        )
        for first_neighbor in neighbor_candidates
    ]
    if any(cycle is None for cycle in cycles):
        return None
    cycle = min(
        cycles,
        key=lambda indices: tuple(
            representatives[index] for index in indices or []
        ),
    )
    if cycle is None or len(cycle) != 6:
        return None
    vertices = [representatives[index] for index in cycle]
    turns = [
        _cross(
            vertices[index],
            vertices[(index + 1) % 6],
            vertices[(index + 2) % 6],
        )
        for index in range(6)
    ]
    if any(abs(turn) <= 1e-12 for turn in turns):
        return None
    if not (all(turn > 0.0 for turn in turns) or all(turn < 0.0 for turn in turns)):
        return None
    area = _polygon_area(vertices)
    if not math.isfinite(area) or area <= 0.0:
        return None
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    edge_lengths = [
        _distance(vertices[index], vertices[(index + 1) % 6])
        for index in range(6)
    ]
    if any(length <= ENDPOINT_TOLERANCE_PT for length in edge_lengths):
        return None
    return vertices, area, bbox, edge_lengths


def _walk_cycle(
    start: int,
    first_neighbor: int,
    adjacency: dict[int, set[int]],
) -> list[int] | None:
    cycle = [start, first_neighbor]
    previous = start
    current = first_neighbor
    while len(cycle) < 6:
        next_candidates = adjacency[current] - {previous}
        if len(next_candidates) != 1:
            return None
        following = next(iter(next_candidates))
        if following in cycle:
            return None
        cycle.append(following)
        previous, current = current, following
    if start not in adjacency[current]:
        return None
    return cycle


def _unique_vertex_index(point: Point, vertices: list[Point]) -> int | None:
    matches = [
        index
        for index, vertex in enumerate(vertices)
        if _distance(point, vertex) <= ENDPOINT_TOLERANCE_PT
    ]
    return matches[0] if len(matches) == 1 else None


def _properly_intersects(
    first: tuple[Point, Point],
    second: tuple[Point, Point],
) -> bool:
    p0, p1 = first
    q0, q1 = second
    rx = p1[0] - p0[0]
    ry = p1[1] - p0[1]
    sx = q1[0] - q0[0]
    sy = q1[1] - q0[1]
    denominator = rx * sy - ry * sx
    if abs(denominator) <= 1e-12:
        return False
    qpx = q0[0] - p0[0]
    qpy = q0[1] - p0[1]
    first_parameter = (qpx * sy - qpy * sx) / denominator
    second_parameter = (qpx * ry - qpy * rx) / denominator
    epsilon = 1e-9
    return bool(
        epsilon < first_parameter < 1.0 - epsilon
        and epsilon < second_parameter < 1.0 - epsilon
    )


def _primitive_provenance(
    item: ItemPrimitive,
    *,
    role: str,
) -> dict[str, Any]:
    points = _line_points(item)
    if points is None:
        raise ValueError("line provenance requires finite endpoints")
    normalized_points = sorted(points)
    return {
        "primitive_id": _primitive_id(item),
        "drawing_order": int(item.drawing_order),
        "item_index": int(item.item_index),
        "op": "l",
        "role": role,
        "points": [
            [_round(point[0]), _round(point[1])]
            for point in normalized_points
        ],
        "draw_type": str(item.draw_type),
        "stroke_width": (
            None
            if item.stroke_width is None
            else _round(float(item.stroke_width))
        ),
    }


def _primitive_id(item: ItemPrimitive) -> str:
    return (
        f"p{int(item.page):03d}_d{int(item.drawing_order):06d}_"
        f"i{int(item.item_index):04d}_l"
    )


def _item_from_provenance(
    value: dict[str, Any],
    *,
    page_num: int,
) -> ItemPrimitive:
    if (
        value.get("op") != "l"
        or type(value.get("drawing_order")) is not int
        or type(value.get("item_index")) is not int
        or type(value.get("draw_type")) is not str
        or value.get("role") not in {"boundary", "diagonal"}
        or not isinstance(value.get("points"), list)
        or len(value["points"]) != 2
    ):
        raise ValueError("invalid provenance")
    points = [
        (float(point[0]), float(point[1]))
        for point in value["points"]
    ]
    stroke_width = value.get("stroke_width")
    if stroke_width is not None:
        stroke_width = float(stroke_width)
    item = ItemPrimitive(
        pdf="",
        page=page_num,
        drawing_order=value["drawing_order"],
        item_index=value["item_index"],
        op="l",
        points=points,
        bbox=(
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        ),
        draw_type=value["draw_type"],
        stroke_width=stroke_width,
    )
    if value.get("primitive_id") != _primitive_id(item):
        raise ValueError("primitive identity mismatch")
    return item


def _build_page_primitive_index(
    page_context: Any,
    *,
    page_index: int,
) -> dict[str, ItemPrimitive] | None:
    try:
        page_context.assert_integrity()
        if page_context.page_num != int(page_index) + 1:
            return None
        by_id: dict[str, ItemPrimitive] = {}
        for item in page_context.primitives:
            primitive_id = page_context.item_id(item)
            if type(primitive_id) is not str or primitive_id in by_id:
                return None
            by_id[primitive_id] = item
    except (AttributeError, TypeError, ValueError):
        return None
    return by_id


def _matches_page_primitive_index(
    value: dict[str, Any],
    *,
    page_primitive_index: dict[str, ItemPrimitive],
) -> bool:
    try:
        for index, provenance in enumerate(value["source_primitives"]):
            item = page_primitive_index.get(provenance["primitive_id"])
            if item is None:
                return False
            expected = _primitive_provenance(
                item,
                role="boundary" if index < 6 else "diagonal",
            )
            if _canonical_json(expected) != _canonical_json(provenance):
                return False
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _source_order_key(row: dict[str, Any]) -> tuple[int, int]:
    first = row["source_primitives"][0]
    return int(first["drawing_order"]), int(first["item_index"])


def _same_stroke_width(items: list[ItemPrimitive]) -> bool:
    values = [item.stroke_width for item in items]
    if all(value is None for value in values):
        return True
    if any(value is None for value in values):
        return False
    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return bool(
        all(math.isfinite(value) for value in numeric)
        and max(numeric) - min(numeric) <= 1e-9
    )


def _line_points(item: ItemPrimitive) -> tuple[Point, Point] | None:
    if len(item.points) != 2:
        return None
    try:
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in item.points
        )
    except (TypeError, ValueError, IndexError):
        return None
    if any(
        not math.isfinite(value)
        for point in points
        for value in point
    ):
        return None
    if _distance(points[0], points[1]) <= 0.0:
        return None
    return points


def _cycle_distance(left: int, right: int, size: int) -> int:
    delta = abs(left - right)
    return min(delta, size - delta)


def _distance(left: Point, right: Point) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _cross(origin: Point, left: Point, right: Point) -> float:
    return (
        (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
    )


def _polygon_area(points: list[Point]) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0


def _principal_axis_extents(points: list[Point]) -> tuple[float, float] | None:
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    covariance_xx = sum(
        (point[0] - center_x) ** 2 for point in points
    ) / len(points)
    covariance_yy = sum(
        (point[1] - center_y) ** 2 for point in points
    ) / len(points)
    covariance_xy = sum(
        (point[0] - center_x) * (point[1] - center_y)
        for point in points
    ) / len(points)
    if not all(math.isfinite(value) for value in (
        covariance_xx,
        covariance_yy,
        covariance_xy,
    )):
        return None
    angle = 0.5 * math.atan2(
        2.0 * covariance_xy,
        covariance_xx - covariance_yy,
    )
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    along = [
        (point[0] - center_x) * ux + (point[1] - center_y) * uy
        for point in points
    ]
    across = [
        (point[0] - center_x) * vx + (point[1] - center_y) * vy
        for point in points
    ]
    width = max(along) - min(along)
    height = max(across) - min(across)
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def _round(value: float) -> float:
    return round(float(value), 6)
