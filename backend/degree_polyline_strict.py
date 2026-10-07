"""Strict 8-edge polyline degree-symbol detector for R2c."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import fitz


SCHEMA_VERSION = "vector_degree_polyline_v1"
GLYPH_SCHEMA_VERSION = "vector_glyph_v1"
CONSUMER_ALLOWED = True

ENDPOINT_EPS = 0.1
CANDIDATE_MIN_MAX_DIM_NORMAL = 2.5
CANDIDATE_MIN_MAX_DIM_SMALL = 1.2
CANDIDATE_MAX_MAX_DIM = 7.0
CANDIDATE_MAX_ASPECT_DELTA = 0.35
CANDIDATE_MAX_SEGMENT_LENGTH_RATIO = 3.0
PARALLEL_ANGLE_TOL_DEG = 3.0
PERPENDICULAR_ANGLE_TOL_DEG = 3.0
DIAGONAL_LENGTH_REL_TOL = 0.08
STRICT_TURN_PAIR_TOL_DEG = 1.5
STRICT_TURN_SUM_MIN_DEG = 355.0
STRICT_TURN_SUM_MAX_DEG = 365.0
RATIO_LONG_SHORT_SEED_MIN = 1.20
SMALL_TOLERANCE_NEIGHBOR_PT = 30.0
SMALL_TOLERANCE_AXIS_DEG = 5.0


@dataclass(frozen=True)
class Segment:
    segment_id: int
    p0: tuple[float, float]
    p1: tuple[float, float]
    length: float
    angle_deg: float
    drawing_order: int
    item_index: int
    stroke_width: float | None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def segment_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def directed_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def angle_delta(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def mean_axis_angle(angles: list[float]) -> float:
    sx = sum(math.cos(math.radians(angle * 2.0)) for angle in angles)
    sy = sum(math.sin(math.radians(angle * 2.0)) for angle in angles)
    return (math.degrees(math.atan2(sy, sx)) / 2.0) % 180.0


def weighted_mean_axis_angle(angles: list[float], weights: list[float]) -> float:
    sx = sum(math.cos(math.radians(angle * 2.0)) * weight for angle, weight in zip(angles, weights, strict=True))
    sy = sum(math.sin(math.radians(angle * 2.0)) * weight for angle, weight in zip(angles, weights, strict=True))
    return (math.degrees(math.atan2(sy, sx)) / 2.0) % 180.0


def turn_values(angles360: list[float]) -> list[float]:
    return [
        ((angles360[(index + 1) % len(angles360)] - angles360[index] + 540.0) % 360.0) - 180.0
        for index in range(len(angles360))
    ]


def rotate_list(values: list[Any], start_index: int) -> list[Any]:
    return values[start_index:] + values[:start_index]


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_dict(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "x": round(bbox[0], 3),
        "y": round(bbox[1], 3),
        "w": round(bbox[2] - bbox[0], 3),
        "h": round(bbox[3] - bbox[1], 3),
    }


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def extract_line_segments(
    page: fitz.Page,
    *,
    drawing_snapshot=None,
) -> list[Segment]:
    segments: list[Segment] = []
    drawings = (
        drawing_snapshot
        if drawing_snapshot is not None
        else page.get_drawings()
    )
    for drawing_order, drawing in enumerate(drawings):
        stroke_width_raw = drawing.get("width")
        stroke_width = None if stroke_width_raw is None else float(stroke_width_raw)
        for item_index, item in enumerate(drawing.get("items", [])):
            if not item or item[0] != "l":
                continue
            start = (float(item[1].x), float(item[1].y))
            end = (float(item[2].x), float(item[2].y))
            length = distance(start, end)
            if length <= 0:
                continue
            segments.append(
                Segment(
                    segment_id=len(segments),
                    p0=start,
                    p1=end,
                    length=length,
                    angle_deg=segment_angle(start, end),
                    drawing_order=drawing_order,
                    item_index=item_index,
                    stroke_width=stroke_width,
                )
            )
    return segments


def cluster_endpoints_spatial_hash(
    endpoints: list[tuple[float, float]],
    *,
    eps: float,
) -> tuple[list[tuple[float, float]], list[int]]:
    uf = UnionFind(len(endpoints))
    grid_size = eps
    grid: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(endpoints):
        gx = int(math.floor(point[0] / grid_size))
        gy = int(math.floor(point[1] / grid_size))
        for nx in range(gx - 1, gx + 2):
            for ny in range(gy - 1, gy + 2):
                for other_index in grid.get((nx, ny), []):
                    if distance(point, endpoints[other_index]) <= eps:
                        uf.union(index, other_index)
        grid.setdefault((gx, gy), []).append(index)

    label_to_node: dict[int, int] = {}
    sums: list[list[float]] = []
    endpoint_node_ids: list[int] = []
    for point_index, point in enumerate(endpoints):
        label = uf.find(point_index)
        if label not in label_to_node:
            label_to_node[label] = len(sums)
            sums.append([0.0, 0.0, 0.0])
        node_id = label_to_node[label]
        sums[node_id][0] += point[0]
        sums[node_id][1] += point[1]
        sums[node_id][2] += 1.0
        endpoint_node_ids.append(node_id)
    nodes = [(entry[0] / entry[2], entry[1] / entry[2]) for entry in sums]
    return nodes, endpoint_node_ids


def endpoint_nodes(segments: list[Segment]) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    endpoints: list[tuple[float, float]] = []
    for segment in segments:
        endpoints.extend([segment.p0, segment.p1])
    nodes, endpoint_node_ids = cluster_endpoints_spatial_hash(endpoints, eps=ENDPOINT_EPS)
    segment_nodes = [
        (endpoint_node_ids[segment.segment_id * 2], endpoint_node_ids[segment.segment_id * 2 + 1])
        for segment in segments
    ]
    return nodes, segment_nodes


def connected_components(nodes: list[tuple[float, float]], segment_nodes: list[tuple[int, int]]) -> list[dict[str, Any]]:
    adjacency: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(nodes))}
    for segment_id, (a_node, b_node) in enumerate(segment_nodes):
        if a_node == b_node:
            continue
        adjacency[a_node].append((b_node, segment_id))
        adjacency[b_node].append((a_node, segment_id))

    seen_nodes: set[int] = set()
    components: list[dict[str, Any]] = []
    for start in range(len(nodes)):
        if start in seen_nodes or not adjacency[start]:
            continue
        queue: deque[int] = deque([start])
        seen_nodes.add(start)
        component_nodes: set[int] = set()
        component_segments: set[int] = set()
        while queue:
            node = queue.popleft()
            component_nodes.add(node)
            for other, segment_id in adjacency[node]:
                component_segments.add(segment_id)
                if other not in seen_nodes:
                    seen_nodes.add(other)
                    queue.append(other)
        components.append({"nodes": component_nodes, "segments": component_segments, "adjacency": adjacency})
    return components


def ring_order(
    component_nodes: set[int],
    component_segments: set[int],
    adjacency: dict[int, list[tuple[int, int]]],
    nodes: list[tuple[float, float]],
    segment_nodes: list[tuple[int, int]],
) -> tuple[list[int], list[int]] | None:
    if len(component_nodes) < 3 or len(component_segments) != len(component_nodes):
        return None
    for node in component_nodes:
        if len([edge for edge in adjacency[node] if edge[1] in component_segments]) != 2:
            return None

    edge_by_pair: dict[tuple[int, int], int] = {}
    for segment_id in component_segments:
        a_node, b_node = segment_nodes[segment_id]
        edge_by_pair[tuple(sorted((a_node, b_node)))] = segment_id

    start = min(component_nodes, key=lambda node: (nodes[node][0], -nodes[node][1]))
    first_neighbors = sorted(
        (edge for edge in adjacency[start] if edge[1] in component_segments),
        key=lambda edge: math.atan2(nodes[edge[0]][1] - nodes[start][1], nodes[edge[0]][0] - nodes[start][0]),
    )
    current = start
    next_node = first_neighbors[0][0]
    ordered_nodes = [start]

    for _ in range(len(component_nodes)):
        if next_node == start:
            break
        ordered_nodes.append(next_node)
        candidates = [
            other
            for other, sid in adjacency[next_node]
            if sid in component_segments and other != current
        ]
        if len(candidates) != 1:
            return None
        current, next_node = next_node, candidates[0]
    if next_node != start or len(ordered_nodes) != len(component_nodes):
        return None

    walk_angles = [
        directed_angle(nodes[ordered_nodes[index]], nodes[ordered_nodes[(index + 1) % len(ordered_nodes)]])
        for index in range(len(ordered_nodes))
    ]
    if sum(turn_values(walk_angles)) < 0:
        ordered_nodes = [ordered_nodes[0]] + list(reversed(ordered_nodes[1:]))
    left_bottom_index = min(range(len(ordered_nodes)), key=lambda index: (nodes[ordered_nodes[index]][0], -nodes[ordered_nodes[index]][1]))
    ordered_nodes = rotate_list(ordered_nodes, left_bottom_index)
    ordered_segments: list[int] = []
    for index, node in enumerate(ordered_nodes):
        other = ordered_nodes[(index + 1) % len(ordered_nodes)]
        segment_id = edge_by_pair.get(tuple(sorted((node, other))))
        if segment_id is None:
            return None
        ordered_segments.append(segment_id)
    return ordered_nodes, ordered_segments


def detect_closed_rings(segments: list[Segment]) -> list[dict[str, Any]]:
    nodes, segment_nodes = endpoint_nodes(segments)
    components = connected_components(nodes, segment_nodes)
    rings: list[dict[str, Any]] = []
    for component in components:
        ordered = ring_order(
            component["nodes"],
            component["segments"],
            component["adjacency"],
            nodes,
            segment_nodes,
        )
        if ordered is None:
            continue
        ordered_node_ids, ordered_segment_ids = ordered
        ordered_points = [nodes[node_id] for node_id in ordered_node_ids]
        ring_segments = [segments[segment_id] for segment_id in ordered_segment_ids]
        bbox = bbox_from_points(ordered_points)
        lengths = []
        angles360 = []
        angles180 = []
        ring_walk_entries = []
        for index, segment in enumerate(ring_segments):
            start = ordered_points[index]
            end = ordered_points[(index + 1) % len(ordered_points)]
            length = distance(start, end)
            angle360 = directed_angle(start, end)
            lengths.append(length)
            angles360.append(angle360)
            angles180.append(angle360 % 180.0)
            ring_walk_entries.append({
                "segment_index": segment.segment_id,
                "length": round(length, 5),
                "angle_deg": round(angle360, 4),
                "start_node": ordered_node_ids[index],
                "end_node": ordered_node_ids[(index + 1) % len(ordered_node_ids)],
            })
        rings.append({
            "ring_id": f"ring_{len(rings):05d}",
            "node_ids": ordered_node_ids,
            "segment_ids": ordered_segment_ids,
            "points": ordered_points,
            "ring_walk": ring_walk_entries,
            "bbox": bbox,
            "n_segments": len(ring_segments),
            "lengths": lengths,
            "angles": angles180,
            "angles360": angles360,
            "drawing_orders": sorted({segment.drawing_order for segment in ring_segments}),
            "item_indices": [segment.item_index for segment in ring_segments],
            "source_primitives": [
                {
                    "segment_index": index,
                    "drawing_order": int(segment.drawing_order),
                    "item_index": int(segment.item_index),
                    "op": "l",
                }
                for index, segment in enumerate(ring_segments)
            ],
        })
    return rings


def analyze_422(lengths: list[float], angles: list[float]) -> dict[str, Any]:
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    short_indices = order[0:2]
    long_indices = order[2:4]
    diagonal_indices = order[4:8]
    short_angles = [angles[index] for index in short_indices]
    long_angles = [angles[index] for index in long_indices]
    diagonal_lengths = [lengths[index] for index in diagonal_indices]
    short_axis = mean_axis_angle(short_angles)
    long_axis = mean_axis_angle(long_angles)
    checks = {
        "short_parallel": angle_delta(short_angles[0], short_angles[1]) <= PARALLEL_ANGLE_TOL_DEG,
        "long_parallel": angle_delta(long_angles[0], long_angles[1]) <= PARALLEL_ANGLE_TOL_DEG,
        "short_long_perpendicular": abs(angle_delta(short_axis, long_axis) - 90.0) <= PERPENDICULAR_ANGLE_TOL_DEG,
        "diagonals_equal": (max(diagonal_lengths) - min(diagonal_lengths)) / max(max(diagonal_lengths), 1e-9) <= DIAGONAL_LENGTH_REL_TOL,
    }
    return {
        "short_indices": short_indices,
        "long_indices": long_indices,
        "diagonal_indices": diagonal_indices,
        "checks": checks,
        "has_direction_axis": all(checks.values()),
        "axis_angle_deg": round(short_axis, 4) if all(checks.values()) else None,
    }


def classify_sld_sequence(lengths: list[float]) -> dict[str, Any]:
    sorted_lengths = sorted(lengths)
    short_threshold = (sorted_lengths[1] + sorted_lengths[2]) / 2.0
    diag_threshold = (sorted_lengths[3] + sorted_lengths[4]) / 2.0
    types = [
        "S" if length <= short_threshold else "L" if length <= diag_threshold else "D"
        for length in lengths
    ]
    counts = Counter(types)
    return {
        "types": types,
        "counts": dict(counts),
        "type_classification_failed": not (counts["S"] == 2 and counts["L"] == 2 and counts["D"] == 4),
    }


def sequence_matches_period(types: list[str]) -> bool:
    target = list("SDLDSDLD")
    return any(rotate_list(types, offset) == target for offset in range(len(types)))


def strict_sequence_diagnostics(
    lengths: list[float],
    angles180: list[float],
    angles360: list[float],
) -> dict[str, Any]:
    classified = classify_sld_sequence(lengths)
    if classified["type_classification_failed"]:
        return {
            **classified,
            "diagnostics": {
                "C1_period": False,
                "C2_no_cardinal_adjacency": False,
                "C3_cardinal_spacing": False,
                "C4_sl_alternation": False,
                "C5_turn_symmetry": False,
            },
            "has_direction_axis_strict_sequence_only": False,
            "axis_angle_deg_strict": None,
            "ratio_long_short": None,
            "ratio_diag_long": None,
        }

    types = classified["types"]
    forbidden_pairs = {("S", "S"), ("L", "L"), ("S", "L"), ("L", "S")}
    cardinal_indices = [index for index, item_type in enumerate(types) if item_type in {"S", "L"}]
    cardinal_diffs = [
        (cardinal_indices[(index + 1) % len(cardinal_indices)] - cardinal_indices[index]) % len(types)
        for index in range(len(cardinal_indices))
    ]
    cardinal_types = [types[index] for index in cardinal_indices]
    turns = turn_values(angles360)
    turn_sum = sum(turns)
    c5_pairs = [abs(turns[index] - turns[(index + 4) % len(turns)]) <= STRICT_TURN_PAIR_TOL_DEG for index in range(len(turns))]
    diagnostics = {
        "C1_period": sequence_matches_period(types),
        "C2_no_cardinal_adjacency": all((types[index], types[(index + 1) % len(types)]) not in forbidden_pairs for index in range(len(types))),
        "C3_cardinal_spacing": all(diff == 2 for diff in cardinal_diffs),
        "C4_sl_alternation": all(cardinal_types[index] != cardinal_types[(index + 1) % len(cardinal_types)] for index in range(len(cardinal_types))),
        "C5_turn_symmetry": all(c5_pairs) and STRICT_TURN_SUM_MIN_DEG <= turn_sum <= STRICT_TURN_SUM_MAX_DEG,
    }
    short_indices = [index for index, item_type in enumerate(types) if item_type == "S"]
    long_indices = [index for index, item_type in enumerate(types) if item_type == "L"]
    diagonal_indices = [index for index, item_type in enumerate(types) if item_type == "D"]
    avg_short = sum(lengths[index] for index in short_indices) / len(short_indices)
    avg_long = sum(lengths[index] for index in long_indices) / len(long_indices)
    avg_diag = sum(lengths[index] for index in diagonal_indices) / len(diagonal_indices)
    axis_strict = weighted_mean_axis_angle(
        [angles180[index] for index in short_indices],
        [lengths[index] for index in short_indices],
    )
    return {
        **classified,
        "diagnostics": diagnostics,
        "turn_sum": round(turn_sum, 4),
        "has_direction_axis_strict_sequence_only": all(diagnostics.values()),
        "axis_angle_deg_strict": round(axis_strict, 4) if all(diagnostics.values()) else None,
        "ratio_long_short": round(avg_long / max(avg_short, 1e-9), 6),
        "ratio_diag_long": round(avg_diag / max(avg_long, 1e-9), 6),
    }


def passes_required_strict_checks(four_two_two: dict[str, Any], strict_sequence: dict[str, Any]) -> bool:
    diagnostics = strict_sequence.get("diagnostics") or {}
    return (
        bool(four_two_two.get("has_direction_axis"))
        and all(bool(diagnostics.get(name)) for name in (
            "C1_period",
            "C2_no_cardinal_adjacency",
            "C3_cardinal_spacing",
            "C4_sl_alternation",
            "C5_turn_symmetry",
        ))
        and strict_sequence.get("ratio_long_short") is not None
        and float(strict_sequence["ratio_long_short"]) >= RATIO_LONG_SHORT_SEED_MIN
    )


def candidate_from_ring(ring: dict[str, Any]) -> dict[str, Any] | None:
    if int(ring["n_segments"]) != 8:
        return None
    bbox = ring["bbox"]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    max_dim = max(width, height)
    min_dim = min(width, height)
    if not (CANDIDATE_MIN_MAX_DIM_SMALL <= max_dim <= CANDIDATE_MAX_MAX_DIM):
        return None
    if abs(width - height) / max(max_dim, 1e-9) > CANDIDATE_MAX_ASPECT_DELTA:
        return None
    lengths = [float(value) for value in ring["lengths"]]
    if min(lengths) <= 0:
        return None
    if max(lengths) / min(lengths) > CANDIDATE_MAX_SEGMENT_LENGTH_RATIO:
        return None

    four_two_two = analyze_422(lengths, [float(value) for value in ring["angles"]])
    strict_sequence = strict_sequence_diagnostics(
        lengths,
        [float(value) for value in ring["angles"]],
        [float(value) for value in ring["angles360"]],
    )
    if not passes_required_strict_checks(four_two_two, strict_sequence):
        return None

    center = bbox_center(bbox)
    diagnostics = strict_sequence["diagnostics"]
    return {
        "ring_id": ring["ring_id"],
        "bbox": bbox_dict(bbox),
        "ring": [[round(point[0], 3), round(point[1], 3)] for point in ring["points"]],
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "axis_angle_deg": strict_sequence["axis_angle_deg_strict"],
        "source_primitives": [
            dict(row) for row in ring["source_primitives"]
        ],
        "size_class": "normal" if max_dim >= CANDIDATE_MIN_MAX_DIM_NORMAL else "small_candidate",
        "strict_checks": {
            "ring_id": ring["ring_id"],
            "n_segments": int(ring["n_segments"]),
            "max_dim": round(max_dim, 5),
            "min_dim": round(min_dim, 5),
            "four_two_two": bool(four_two_two["has_direction_axis"]),
            "four_two_two_checks": dict(four_two_two["checks"]),
            "C1_period": bool(diagnostics["C1_period"]),
            "C2_no_cardinal_adjacency": bool(diagnostics["C2_no_cardinal_adjacency"]),
            "C3_cardinal_spacing": bool(diagnostics["C3_cardinal_spacing"]),
            "C4_sl_alternation": bool(diagnostics["C4_sl_alternation"]),
            "C5_turn_symmetry": bool(diagnostics["C5_turn_symmetry"]),
            "ratio_long_short": strict_sequence["ratio_long_short"],
            "ratio_diag_long": strict_sequence["ratio_diag_long"],
        },
        "source": "degree_polyline_strict",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def accept_small_tolerance_candidates(
    normal_rows: list[dict[str, Any]],
    small_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for row in small_rows:
        center = row["center"]
        axis = float(row["axis_angle_deg"])
        for normal in normal_rows:
            ncenter = normal["center"]
            if distance((center["x"], center["y"]), (ncenter["x"], ncenter["y"])) > SMALL_TOLERANCE_NEIGHBOR_PT:
                continue
            if angle_delta(axis, float(normal["axis_angle_deg"])) > SMALL_TOLERANCE_AXIS_DEG:
                continue
            accepted_row = dict(row)
            accepted_row["size_class"] = "small_tolerance"
            accepted.append(accepted_row)
            break
    return accepted


def detect_degree_polylines(
    page: fitz.Page,
    *,
    page_index: int = 0,
    drawing_snapshot=None,
) -> dict[str, Any]:
    segments = extract_line_segments(
        page,
        drawing_snapshot=drawing_snapshot,
    )
    rings = detect_closed_rings(segments)
    candidates = [candidate for ring in rings if (candidate := candidate_from_ring(ring)) is not None]
    normal_rows = [row for row in candidates if row["size_class"] == "normal"]
    small_rows = [row for row in candidates if row["size_class"] == "small_candidate"]
    rows = normal_rows + accept_small_tolerance_candidates(normal_rows, small_rows)
    for idx, row in enumerate(rows):
        row["schema_version"] = SCHEMA_VERSION
        row["id"] = f"degree_{idx:04d}"
    counts = Counter(row["size_class"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_degree_polyline_v1": rows,
        "vector_degree_polyline_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "page_index": int(page_index),
            "count": len(rows),
            "counts_by_size_class": dict(sorted(counts.items())),
            "consumer_allowed_count": sum(1 for row in rows if row.get("consumer_allowed")),
        },
    }


def _bbox_quad(bbox: dict[str, float]) -> list[list[float]]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    return [[round(x, 3), round(y, 3)], [round(x + w, 3), round(y, 3)], [round(x + w, 3), round(y + h, 3)], [round(x, 3), round(y + h, 3)]]


def degree_rows_to_glyph_tokens(
    rows: list[dict[str, Any]],
    *,
    page_index: int,
    start_order: int,
) -> list[dict[str, Any]]:
    tokens = []
    for idx, row in enumerate(rows):
        tokens.append({
            "schema_version": GLYPH_SCHEMA_VERSION,
            "token_id": f"p{int(page_index) + 1:03d}_glyph_{int(start_order) + idx:06d}",
            "page_index": int(page_index),
            "glyph_type": "degree",
            "text": "°",
            "bbox": dict(row["bbox"]),
            "oriented_quad": _bbox_quad(row["bbox"]),
            **(
                {"axis_angle_deg": row["axis_angle_deg"]}
                if row.get("axis_angle_deg") is not None
                else {}
            ),
            "confidence": 0.96 if row["size_class"] == "normal" else 0.91,
            "strict_checks": {
                **dict(row["strict_checks"]),
                "source": "degree_polyline_strict",
                "size_class": row["size_class"],
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            "source_segments": (
                [
                    {
                        "kind": "degree_polyline_strict",
                        "index": idx,
                        "ring_id": row.get("ring_id"),
                        "size_class": row["size_class"],
                        **dict(source_primitive),
                    }
                    for source_primitive in row["source_primitives"]
                    if isinstance(source_primitive, dict)
                ]
                if isinstance(row.get("source_primitives"), list)
                and row["source_primitives"]
                else [{
                    "kind": "degree_polyline_strict",
                    "index": idx,
                    "ring_id": row.get("ring_id"),
                    "size_class": row["size_class"],
                }]
            ),
            "drop_reason": None,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return tokens


def _glyph_stats(page_index: int, tokens: list[dict[str, Any]], old_stats: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row.get("glyph_type") or "unknown") for row in tokens)
    low_conf_or_unknown = sum(
        1 for row in tokens
        if row.get("glyph_type") == "unknown" or float(row.get("confidence") or 0.0) < 0.80
    )
    stats = dict(old_stats)
    stats.update({
        "schema_version": GLYPH_SCHEMA_VERSION,
        "page_index": int(page_index),
        "token_count": len(tokens),
        "counts_by_type": dict(sorted(counts.items())),
        "low_conf_or_unknown": low_conf_or_unknown,
        "fallback_ratio": low_conf_or_unknown / len(tokens) if tokens else 0.0,
        "consumer_allowed_count": sum(1 for row in tokens if row.get("consumer_allowed")),
    })
    return stats


def _token_order(token: dict[str, Any]) -> int | None:
    token_id = str(token.get("token_id") or "")
    marker = "_glyph_"
    if marker not in token_id:
        return None
    try:
        return int(token_id.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _next_token_order(tokens: list[dict[str, Any]]) -> int:
    orders = [
        order
        for token in tokens
        if (order := _token_order(token)) is not None
    ]
    return (max(orders) + 1) if orders else len(tokens)


def merge_degree_polyline_glyph_dump(
    glyph_dump: dict[str, Any],
    *,
    pdf_bytes: bytes,
    page_index: int,
    enabled: bool,
    page: fitz.Page | None = None,
    drawing_snapshot=None,
) -> dict[str, Any]:
    if not enabled:
        return glyph_dump
    if page is not None:
        dump = detect_degree_polylines(
            page,
            page_index=page_index,
            drawing_snapshot=drawing_snapshot,
        )
    else:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            dump = detect_degree_polylines(
                doc[page_index],
                page_index=page_index,
            )
        finally:
            doc.close()
    tokens = [
        token for token in list(glyph_dump.get("tokens") or [])
        if token.get("glyph_type") != "degree"
    ]
    tokens.extend(degree_rows_to_glyph_tokens(
        dump["vector_degree_polyline_v1"],
        page_index=page_index,
        start_order=_next_token_order(tokens),
    ))
    out = dict(glyph_dump)
    out["tokens"] = tokens
    out["stats"] = _glyph_stats(page_index, tokens, dict(glyph_dump.get("stats") or {}))
    return out


def build_degree_polyline_pipeline_fields(
    *,
    page: fitz.Page,
    page_index: int,
    dump_enabled: bool,
    drawing_snapshot=None,
) -> dict[str, Any]:
    if not dump_enabled:
        return {}
    dump = detect_degree_polylines(
        page,
        page_index=page_index,
        drawing_snapshot=drawing_snapshot,
    )
    return {
        "vector_degree_polyline_v1": dump["vector_degree_polyline_v1"],
        "vector_degree_polyline_stats_v1": dump["vector_degree_polyline_stats_v1"],
    }
