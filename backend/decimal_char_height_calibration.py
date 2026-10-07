from __future__ import annotations

import math
import os
from statistics import median
from typing import Any

import numpy as np


SCHEMA_VERSION = "vector_decimal_char_height_v1"
STATS_SCHEMA_VERSION = "vector_decimal_char_height_stats_v1"
DEFAULT_K1 = 2.0
DEFAULT_K2 = 0.65
MIN_POINTS_FOR_MULTI_CLUSTER = 10
MAX_CLUSTERS = 3
MIN_FINAL_CLUSTER_RELATIVE_GAP = 0.20


def decimal_char_height_dump_enabled() -> bool:
    value = os.environ.get("VECTOR_DECIMAL_CHAR_HEIGHT_DUMP", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _axis_bucket(angle: float | None) -> str:
    if angle is None:
        return "oblique"
    normalized = float(angle) % 180.0
    if _angle_delta(normalized, 0.0) <= 15.0:
        return "horizontal"
    if _angle_delta(normalized, 90.0) <= 15.0:
        return "vertical"
    return "oblique"


def _quad_side_lengths(quad: Any) -> tuple[float, float] | None:
    if not isinstance(quad, list) or len(quad) != 4:
        return None
    try:
        points = [(float(p[0]), float(p[1])) for p in quad]
    except (TypeError, ValueError, IndexError):
        return None
    lengths = []
    for idx in range(4):
        x0, y0 = points[idx]
        x1, y1 = points[(idx + 1) % 4]
        lengths.append(math.hypot(x1 - x0, y1 - y0))
    return min(lengths), max(lengths)


def _edge_lengths(row: dict[str, Any]) -> tuple[float, float] | None:
    long_edge = row.get("long_edge")
    short_edge = row.get("short_edge")
    detail = row.get("detail")
    if isinstance(detail, dict):
        long_edge = long_edge if long_edge is not None else detail.get("long")
        short_edge = short_edge if short_edge is not None else detail.get("short")
    if long_edge is None or short_edge is None:
        computed = _quad_side_lengths(row.get("quad"))
        if computed is None:
            return None
        short_edge, long_edge = computed
    try:
        long_value = float(long_edge)
        short_value = float(short_edge)
    except (TypeError, ValueError):
        return None
    if long_value <= 0.0 or short_value <= 0.0:
        return None
    if short_value > long_value:
        short_value, long_value = long_value, short_value
    return short_value, long_value


def _axis_angle(row: dict[str, Any]) -> float | None:
    axis = row.get("short_axis")
    if not isinstance(axis, dict) or axis.get("angle_deg") is None:
        return None
    try:
        return float(axis["angle_deg"]) % 180.0
    except (TypeError, ValueError):
        return None


def _lloyd_1d(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 1:
        labels = np.zeros(len(values), dtype=int)
        centers = np.array([float(np.median(values))], dtype=float)
        return labels, centers
    unique_values = np.unique(values)
    if len(unique_values) >= k:
        percentiles = np.linspace(0.0, 100.0, k + 2)[1:-1]
        centers = np.percentile(unique_values, percentiles).astype(float)
    else:
        centers = np.linspace(float(np.min(values)), float(np.max(values)), k)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(50):
        distances = np.abs(values[:, None] - centers[None, :])
        next_labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for idx in range(k):
            members = values[next_labels == idx]
            if len(members):
                next_centers[idx] = float(np.median(members))
        if np.array_equal(next_labels, labels) and np.allclose(next_centers, centers):
            break
        labels = next_labels
        centers = next_centers
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    sorted_labels = np.array([remap[int(label)] for label in labels], dtype=int)
    sorted_centers = centers[order]
    return sorted_labels, sorted_centers


def _interval_l1_cost(
    sorted_values: np.ndarray,
    prefix_sums: np.ndarray,
    start: int,
    stop: int,
) -> float:
    median_index = (start + stop - 1) // 2
    pivot = float(sorted_values[median_index])
    left_cost = (
        pivot * (median_index - start)
        - float(prefix_sums[median_index] - prefix_sums[start])
    )
    right_cost = (
        float(prefix_sums[stop] - prefix_sums[median_index + 1])
        - pivot * (stop - median_index - 1)
    )
    return left_cost + right_cost


def _optimal_1d_k_medians(
    values: np.ndarray,
    k: int,
    *,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the canonical supported global L1 partition.

    Equal values are never split between clusters.  When multiple partitions
    have equal L1 cost, the lexicographically greatest split tuple keeps an
    equidistant boundary value with the lower center, matching ``np.argmin``.
    """
    n = len(values)
    if k <= 1:
        return _lloyd_1d(values, 1)
    if min_cluster_size < 1 or n < k * min_cluster_size:
        return None

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    prefix_sums = np.concatenate((np.array([0.0]), np.cumsum(sorted_values)))
    states: list[list[tuple[float, tuple[int, ...]] | None]] = [
        [None] * (n + 1)
        for _ in range(k + 1)
    ]
    states[0][0] = (0.0, ())
    for cluster_count in range(1, k + 1):
        minimum_stop = cluster_count * min_cluster_size
        for stop in range(minimum_stop, n + 1):
            best: tuple[float, tuple[int, ...]] | None = None
            minimum_start = (cluster_count - 1) * min_cluster_size
            maximum_start = stop - min_cluster_size
            for start in range(minimum_start, maximum_start + 1):
                previous = states[cluster_count - 1][start]
                if previous is None:
                    continue
                if (
                    start > 0
                    and float(sorted_values[start - 1])
                    == float(sorted_values[start])
                ):
                    continue
                cost = previous[0] + _interval_l1_cost(
                    sorted_values,
                    prefix_sums,
                    start,
                    stop,
                )
                splits = previous[1] + (
                    (start,) if cluster_count > 1 else ()
                )
                candidate = (cost, splits)
                if (
                    best is None
                    or cost < best[0] - 1e-12
                    or (
                        math.isclose(cost, best[0], rel_tol=1e-12, abs_tol=1e-12)
                        and splits > best[1]
                    )
                ):
                    best = candidate
            states[cluster_count][stop] = best

    final = states[k][n]
    if final is None:
        return None
    boundaries = (0, *final[1], n)
    sorted_labels = np.zeros(n, dtype=int)
    centers = []
    for label, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        sorted_labels[start:stop] = label
        centers.append(float(np.median(sorted_values[start:stop])))
    labels = np.empty(n, dtype=int)
    labels[order] = sorted_labels
    return labels, np.array(centers, dtype=float)


def _exact_assignment_is_supported(
    values: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    *,
    min_cluster_size: int,
) -> bool:
    retained_medians = []
    for label, center in enumerate(centers):
        center_value = float(center)
        if not math.isfinite(center_value) or center_value <= 0.0:
            return False
        members = values[labels == label]
        retained = members[np.abs(members - center_value) <= 0.5 * center_value]
        if len(retained) < min_cluster_size:
            return False
        retained_medians.append(float(np.median(retained)))
    for index, own_median in enumerate(retained_medians):
        nearest_gap = min(
            abs(other_median - own_median)
            for other_index, other_median in enumerate(retained_medians)
            if other_index != index
        )
        if nearest_gap / own_median < MIN_FINAL_CLUSTER_RELATIVE_GAP:
            return False
    return True


def _extreme_core_mask(values: np.ndarray) -> np.ndarray:
    mask = np.ones(len(values), dtype=bool)
    n = len(values)
    if n < MIN_POINTS_FOR_MULTI_CLUSTER:
        return mask
    min_cluster_size = max(3, math.ceil(n * 0.05))
    order = np.argsort(values)

    low_value = float(values[order[0]])
    low_tol = max(0.02, abs(low_value) * 0.02)
    low_group = [idx for idx in order if abs(float(values[idx]) - low_value) <= low_tol]
    if len(low_group) < min_cluster_size and len(low_group) < n:
        next_value = float(values[order[len(low_group)]])
        if next_value - low_value > 0.5 * next_value:
            mask[low_group] = False

    high_value = float(values[order[-1]])
    high_tol = max(0.02, abs(high_value) * 0.02)
    high_group = [idx for idx in order if abs(float(values[idx]) - high_value) <= high_tol]
    if len(high_group) < min_cluster_size and len(high_group) < n:
        prev_value = float(values[order[-len(high_group) - 1]])
        if high_value - prev_value > 0.5 * prev_value:
            mask[high_group] = False
    return mask


def _silhouette_1d(values: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(set(int(v) for v in labels))
    if len(unique_labels) <= 1 or len(unique_labels) >= len(values):
        return 0.0
    scores = []
    for idx, value in enumerate(values):
        same = values[labels == labels[idx]]
        if len(same) <= 1:
            scores.append(0.0)
            continue
        a = float(np.mean(np.abs(same[same != value] - value))) if np.any(same != value) else 0.0
        b_values = []
        for label in unique_labels:
            if label == int(labels[idx]):
                continue
            other = values[labels == label]
            if len(other):
                b_values.append(float(np.mean(np.abs(other - value))))
        b = min(b_values) if b_values else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def select_k_for_long_edges(long_edges: list[float]) -> int:
    values = np.array([float(v) for v in long_edges if float(v) > 0.0], dtype=float)
    values = values[_extreme_core_mask(values)]
    n = len(values)
    if n < MIN_POINTS_FOR_MULTI_CLUSTER:
        return 1
    median_value = float(np.median(values))
    if median_value <= 0.0:
        return 1
    if (float(np.max(values)) - float(np.min(values))) / median_value <= 0.08:
        return 1
    max_k = min(MAX_CLUSTERS, n)
    min_cluster_size = max(3, math.ceil(n * 0.05))
    best_k = 1
    best_score = -1.0
    for k in range(2, max_k + 1):
        labels, _centers = _lloyd_1d(values, k)
        counts = [int(np.sum(labels == idx)) for idx in range(k)]
        if min(counts) < min_cluster_size:
            continue
        score = _silhouette_1d(values, labels)
        if score > best_score + 0.02 or (abs(score - best_score) <= 0.02 and k < best_k):
            best_k = k
            best_score = score
    return best_k


def _angle_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    distribution = {"horizontal": 0, "vertical": 0, "oblique": 0}
    for row in rows:
        distribution[_axis_bucket(_axis_angle(row))] += 1
    return distribution


def _angle_p50(rows: list[dict[str, Any]]) -> float | None:
    angles = [_axis_angle(row) for row in rows]
    filtered = [angle for angle in angles if angle is not None]
    if not filtered:
        return None
    return _round(float(median(filtered)))


def _cluster_row(
    *,
    cluster_id: str,
    role: str,
    members: list[tuple[int, dict[str, Any], float, float]],
    k1: float,
    k2: float,
) -> dict[str, Any]:
    long_median = float(median(item[3] for item in members))
    short_median = float(median(item[2] for item in members))
    char_height = long_median * k1
    return {
        "schema_version": SCHEMA_VERSION,
        "cluster_id": cluster_id,
        "role": role,
        "count": len(members),
        "long_edge_median": _round(long_median),
        "short_edge_median": _round(short_median),
        "char_height": _round(char_height),
        "char_width": _round(char_height * k2),
        "short_axis_angle_p50": _angle_p50([item[1] for item in members]),
        "short_axis_angle_distribution": _angle_distribution([item[1] for item in members]),
        "member_quad_ids": [
            str(item[1].get("id") if item[1].get("id") is not None else item[0])
            for item in members
        ],
        "consumer_allowed": False,
    }


def build_decimal_char_height_v1(
    decimal_quad_rows: list[dict[str, Any]],
    *,
    page_index: int = 0,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
) -> dict[str, Any]:
    items: list[tuple[int, dict[str, Any], float, float]] = []
    for idx, row in enumerate(decimal_quad_rows):
        edge_lengths = _edge_lengths(row)
        if edge_lengths is None:
            continue
        short_edge, long_edge = edge_lengths
        items.append((idx, row, short_edge, long_edge))
    if not items:
        return {
            "schema_version": SCHEMA_VERSION,
            "consumer_allowed": False,
            "vector_decimal_char_height_v1": [],
            "vector_decimal_char_height_stats_v1": {
                "schema_version": STATS_SCHEMA_VERSION,
                "page_index": page_index,
                "total_quads": 0,
                "usable_quads": 0,
                "selected_k": 0,
                "total_clusters": 0,
                "primary_char_height": None,
                "tolerance_char_height": None,
                "outlier_count": 0,
            },
        }

    all_values = np.array([item[3] for item in items], dtype=float)
    core_mask = _extreme_core_mask(all_values)
    core_items = [item for item, keep in zip(items, core_mask, strict=True) if bool(keep)]
    outliers = [item for item, keep in zip(items, core_mask, strict=True) if not bool(keep)]
    long_edges = [item[3] for item in core_items]
    selected_k = select_k_for_long_edges(long_edges)
    values = np.array(long_edges, dtype=float)
    labels, centers = _lloyd_1d(values, selected_k)
    if selected_k > 1:
        min_cluster_size = max(3, math.ceil(len(values) * 0.05))
        exact_assignment = _optimal_1d_k_medians(
            values,
            selected_k,
            min_cluster_size=min_cluster_size,
        )
        if exact_assignment is not None and _exact_assignment_is_supported(
            values,
            exact_assignment[0],
            exact_assignment[1],
            min_cluster_size=min_cluster_size,
        ):
            labels, centers = exact_assignment
    cluster_members: dict[int, list[tuple[int, dict[str, Any], float, float]]] = {
        idx: [] for idx in range(selected_k)
    }
    for item, label in zip(core_items, labels, strict=True):
        center = float(centers[int(label)])
        if abs(item[3] - center) > 0.5 * center:
            outliers.append(item)
        else:
            cluster_members[int(label)].append(item)

    clusters = [members for members in cluster_members.values() if members]
    clusters.sort(key=lambda members: -median(item[3] for item in members))
    roles = ["primary", "tolerance", "aux"]
    rows = [
        _cluster_row(
            cluster_id=f"cluster_{idx:04d}",
            role=roles[idx] if idx < len(roles) else "aux",
            members=members,
            k1=k1,
            k2=k2,
        )
        for idx, members in enumerate(clusters)
    ]
    if outliers:
        rows.append(
            _cluster_row(
                cluster_id="outlier",
                role="outlier",
                members=outliers,
                k1=k1,
                k2=k2,
            )
        )

    primary = next((row for row in rows if row["role"] == "primary"), None)
    tolerance = next((row for row in rows if row["role"] == "tolerance"), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": False,
        "vector_decimal_char_height_v1": rows,
        "vector_decimal_char_height_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": page_index,
            "total_quads": len(decimal_quad_rows),
            "usable_quads": len(items),
            "selected_k": selected_k,
            "total_clusters": len(clusters),
            "primary_char_height": None if primary is None else primary["char_height"],
            "tolerance_char_height": None if tolerance is None else tolerance["char_height"],
            "outlier_count": len(outliers),
            "outlier_ratio": _round(len(outliers) / max(len(items), 1)),
        },
    }


def build_decimal_char_height_pipeline_fields(
    decimal_quad_rows: list[dict[str, Any]],
    *,
    page_index: int,
    dump_enabled: bool,
) -> dict[str, Any]:
    if not dump_enabled:
        return {}
    dump = build_decimal_char_height_v1(decimal_quad_rows, page_index=page_index)
    return {
        "vector_decimal_char_height_v1": dump["vector_decimal_char_height_v1"],
        "vector_decimal_char_height_stats_v1": dump["vector_decimal_char_height_stats_v1"],
    }
