from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from decimal_repeated_size_cohort import (
    build_decimal_repeated_size_cohort_proofs_v1,
)


SCHEMA_VERSION = "vector_decimal_corridor_v1"
STATS_SCHEMA_VERSION = "vector_decimal_corridor_stats_v1"
SOURCE = "decimal_char_height_back_inference"
CONSUMER_ALLOWED = False
CHAR_HEIGHT_CLUSTER_SCHEMA_VERSION = "vector_decimal_char_height_v1"
CHAR_HEIGHT_CLUSTER_ROLES = frozenset({
    "primary",
    "tolerance",
    "aux",
    "outlier",
})
DEFAULT_INTEGER_SLOT_LIMIT = 6
DEFAULT_FRACTION_SLOT_LIMIT = 4


def build_decimal_corridors_v1(
    decimal_quad_rows: list[dict[str, Any]],
    char_height_clusters: list[dict[str, Any]],
    *,
    page_index: int = 0,
    integer_slot_limit: int = DEFAULT_INTEGER_SLOT_LIMIT,
    fraction_slot_limit: int = DEFAULT_FRACTION_SLOT_LIMIT,
) -> dict[str, Any]:
    input_quad_ids = {
        str(row.get("id") or quad_index)
        for quad_index, row in enumerate(decimal_quad_rows)
        if isinstance(row, dict)
    }
    validated_clusters = _validated_char_height_clusters(
        char_height_clusters,
        allowed_member_ids=input_quad_ids,
    )
    if validated_clusters is None:
        validated_clusters = []
    cluster_index = _cluster_index(validated_clusters)
    page_clusters = [
        cluster
        for cluster in validated_clusters
        if str(cluster.get("role") or "") != "outlier"
    ]
    primary_cluster = next(
        (
            cluster
            for cluster in page_clusters
            if str(cluster.get("role") or "") == "primary"
        ),
        None,
    )
    primary_long_edge = _positive_float(
        (primary_cluster or {}).get("long_edge_median")
    )
    rows = []
    for quad_index, quad_row in enumerate(decimal_quad_rows):
        center = _center(quad_row)
        angle = _baseline_angle(quad_row)
        if center is None or angle is None:
            continue
        cluster = _cluster_for_quad(
            quad_row,
            quad_index=quad_index,
            clusters=validated_clusters,
            by_member_id=cluster_index,
        )
        if cluster is None:
            continue
        char_height = _positive_float(cluster.get("char_height"))
        char_width = _positive_float(cluster.get("char_width"))
        if char_height is None or char_width is None:
            continue
        size_class = _size_class(cluster)
        cluster_long_edge = _positive_float(cluster.get("long_edge_median"))
        cluster_member_count = int(cluster["count"])
        if (
            cluster_long_edge is None
            or primary_long_edge is None
            or cluster_member_count < 1
        ):
            continue
        quad_id = str(quad_row.get("id") or quad_index)
        direct_member_ids = {
            str(value) for value in cluster.get("member_quad_ids") or []
        }
        cluster_membership = (
            "direct_member_quad_id"
            if quad_id in direct_member_ids
            else "nearest_cluster_long_edge"
        )
        dot_long_edge = _long_edge(quad_row)
        if dot_long_edge is None:
            continue
        other_cluster_edges = [
            value
            for other in page_clusters
            if other is not cluster
            if (value := _positive_float(other.get("long_edge_median")))
            is not None
        ]
        nearest_cluster_relative_gap = (
            min(abs(value - cluster_long_edge) for value in other_cluster_edges)
            / cluster_long_edge
            if other_cluster_edges
            else 1.0
        )
        integer_slots = [
            _slot_row(
                center=_offset_center(center, angle, -char_width * (idx + 1)),
                width=char_width,
                height=char_height,
                angle_deg=angle,
                index=idx,
            )
            for idx in range(max(0, int(integer_slot_limit)))
        ]
        fraction_slots = [
            _slot_row(
                center=_offset_center(center, angle, char_width * (idx + 1)),
                width=char_width,
                height=char_height,
                angle_deg=angle,
                index=idx,
            )
            for idx in range(max(0, int(fraction_slot_limit)))
        ]
        all_points = [
            point
            for slot in [*integer_slots, *fraction_slots]
            for point in slot["quad"]
        ]
        all_points.extend(_quad_points(quad_row))
        corridor_quad = _oriented_enclosing_quad(all_points, angle)
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "page_index": int(page_index),
            "decimal_quad_id": quad_id,
            "decimal_quad_index": int(quad_index),
            "char_height_cluster_id": str(cluster.get("cluster_id") or ""),
            "size_class": size_class,
            "cluster_membership": cluster_membership,
            "cluster_member_count": cluster_member_count,
            "page_cluster_count": len(page_clusters),
            "cluster_long_edge_median": _round(cluster_long_edge),
            "long_edge_ratio_to_primary": _round(
                cluster_long_edge / primary_long_edge
            ),
            "within_cluster_relative_error": _round(
                abs(dot_long_edge - cluster_long_edge) / cluster_long_edge
            ),
            "nearest_cluster_relative_gap": _round(
                nearest_cluster_relative_gap
            ),
            "char_height": _round(char_height),
            "char_width": _round(char_width),
            "baseline_angle_deg": _round(angle),
            "integer_slots": integer_slots,
            "fraction_slots": fraction_slots,
            "n_integer_slots": len(integer_slots),
            "n_fraction_slots": len(fraction_slots),
            "corridor_bbox": _bbox_dict(corridor_quad),
            "corridor_quad": _round_points(corridor_quad),
            "source": SOURCE,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    counts = Counter(row["size_class"] for row in rows)
    repeated_size_cohort_proofs = (
        build_decimal_repeated_size_cohort_proofs_v1(
            decimal_quad_rows,
            validated_clusters,
            rows,
            page_index=page_index,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_decimal_corridor_v1": rows,
        "vector_decimal_repeated_size_cohort_v1": (
            repeated_size_cohort_proofs
        ),
        "vector_decimal_corridor_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": int(page_index),
            "total_corridors": len(rows),
            "counts_by_size_class": dict(sorted(counts.items())),
            "consumer_allowed_count": sum(
                1 for row in rows if bool(row.get("consumer_allowed"))
            ),
        },
    }


def build_decimal_corridor_pipeline_fields(
    decimal_quad_rows: list[dict[str, Any]],
    char_height_clusters: list[dict[str, Any]],
    *,
    page_index: int,
    dump_enabled: bool,
) -> dict[str, Any]:
    if not dump_enabled:
        return {}
    dump = build_decimal_corridors_v1(
        decimal_quad_rows,
        char_height_clusters,
        page_index=page_index,
    )
    return {
        "vector_decimal_corridor_v1": dump["vector_decimal_corridor_v1"],
        "vector_decimal_corridor_stats_v1": dump["vector_decimal_corridor_stats_v1"],
    }


def _cluster_index(clusters: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        for member_id in cluster.get("member_quad_ids") or []:
            out[str(member_id)] = cluster
    return out


def _validated_char_height_clusters(
    clusters: Any,
    *,
    allowed_member_ids: set[str],
) -> list[dict[str, Any]] | None:
    """Validate the complete R2d cluster set before consuming any member.

    ``vector_decimal_char_height_v1`` predates an explicit ``source`` field,
    so absence is the only valid source representation for that schema.  Any
    injected source value is contract drift and is rejected.  Returning
    ``None`` rejects the whole set: partially consuming a conflicting set
    would let malformed rows influence nearest-cluster fallback.
    """
    if not isinstance(clusters, list):
        return None
    validated: list[dict[str, Any]] = []
    cluster_ids: set[str] = set()
    member_owners: dict[str, str] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            return None
        if (
            cluster.get("schema_version")
            != CHAR_HEIGHT_CLUSTER_SCHEMA_VERSION
            or "source" in cluster
            or cluster.get("consumer_allowed") is not False
        ):
            return None
        cluster_id = cluster.get("cluster_id")
        role = cluster.get("role")
        count = cluster.get("count")
        member_ids = cluster.get("member_quad_ids")
        if (
            type(cluster_id) is not str
            or not cluster_id
            or cluster_id in cluster_ids
            or type(role) is not str
            or role not in CHAR_HEIGHT_CLUSTER_ROLES
            or type(count) is not int
            or count < 1
            or not isinstance(member_ids, list)
            or count != len(member_ids)
            or _strict_positive_number(cluster.get("char_height")) is None
            or _strict_positive_number(cluster.get("char_width")) is None
            or _strict_positive_number(cluster.get("long_edge_median")) is None
        ):
            return None
        if any(type(member_id) is not str or not member_id for member_id in member_ids):
            return None
        if len(set(member_ids)) != len(member_ids):
            return None
        if any(member_id not in allowed_member_ids for member_id in member_ids):
            return None
        for member_id in member_ids:
            if member_id in member_owners:
                return None
            member_owners[member_id] = cluster_id
        cluster_ids.add(cluster_id)
        validated.append(cluster)
    return sorted(validated, key=lambda cluster: str(cluster["cluster_id"]))


def _cluster_for_quad(
    quad_row: dict[str, Any],
    *,
    quad_index: int,
    clusters: list[dict[str, Any]],
    by_member_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in (quad_row.get("id"), quad_index, str(quad_index)):
        if str(key) in by_member_id:
            return by_member_id[str(key)]
    quad_long = _long_edge(quad_row)
    if quad_long is None:
        return None
    best_cluster = None
    best_distance = math.inf
    for cluster in clusters:
        cluster_long = _positive_float(cluster.get("long_edge_median"))
        if cluster_long is None:
            continue
        distance = abs(cluster_long - quad_long)
        if distance < best_distance:
            best_distance = distance
            best_cluster = cluster
    return best_cluster


def _slot_row(
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    angle_deg: float,
    index: int,
) -> dict[str, Any]:
    quad = _oriented_rect(center, width=width, height=height, angle_deg=angle_deg)
    return {
        "index": int(index),
        "center": {"x": _round(center[0]), "y": _round(center[1])},
        "bbox": _bbox_dict(quad),
        "quad": _round_points(quad),
    }


def _oriented_rect(
    center: tuple[float, float],
    *,
    width: float,
    height: float,
    angle_deg: float,
) -> list[tuple[float, float]]:
    ux, uy = _unit(angle_deg)
    vx, vy = -uy, ux
    half_w = width / 2.0
    half_h = height / 2.0
    cx, cy = center
    return [
        (cx - ux * half_w - vx * half_h, cy - uy * half_w - vy * half_h),
        (cx + ux * half_w - vx * half_h, cy + uy * half_w - vy * half_h),
        (cx + ux * half_w + vx * half_h, cy + uy * half_w + vy * half_h),
        (cx - ux * half_w + vx * half_h, cy - uy * half_w + vy * half_h),
    ]


def _oriented_enclosing_quad(
    points: list[list[float] | tuple[float, float]],
    angle_deg: float,
) -> list[tuple[float, float]]:
    if not points:
        return []
    ux, uy = _unit(angle_deg)
    vx, vy = -uy, ux
    origin = np.array([0.0, 0.0], dtype=float)
    coords = []
    for point in points:
        p = np.array([float(point[0]), float(point[1])], dtype=float)
        rel = p - origin
        coords.append((float(rel.dot(np.array([ux, uy]))), float(rel.dot(np.array([vx, vy])))))
    min_u = min(coord[0] for coord in coords)
    max_u = max(coord[0] for coord in coords)
    min_v = min(coord[1] for coord in coords)
    max_v = max(coord[1] for coord in coords)
    return [
        _from_basis(min_u, min_v, ux, uy, vx, vy),
        _from_basis(max_u, min_v, ux, uy, vx, vy),
        _from_basis(max_u, max_v, ux, uy, vx, vy),
        _from_basis(min_u, max_v, ux, uy, vx, vy),
    ]


def _from_basis(
    u: float,
    v: float,
    ux: float,
    uy: float,
    vx: float,
    vy: float,
) -> tuple[float, float]:
    return (ux * u + vx * v, uy * u + vy * v)


def _offset_center(
    center: tuple[float, float],
    angle_deg: float,
    distance: float,
) -> tuple[float, float]:
    ux, uy = _unit(angle_deg)
    return (center[0] + ux * distance, center[1] + uy * distance)


def _unit(angle_deg: float) -> tuple[float, float]:
    theta = math.radians(float(angle_deg))
    return math.cos(theta), math.sin(theta)


def _baseline_angle(row: dict[str, Any]) -> float | None:
    axis = row.get("short_axis")
    if not isinstance(axis, dict):
        return None
    angle = _positive_or_zero_float(axis.get("angle_deg"))
    if angle is None:
        return None
    return angle % 180.0


def _center(row: dict[str, Any]) -> tuple[float, float] | None:
    center = row.get("center")
    if isinstance(center, dict):
        try:
            return float(center["x"]), float(center["y"])
        except (KeyError, TypeError, ValueError):
            return None
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        try:
            x = float(bbox["x"])
            y = float(bbox["y"])
            w = float(bbox["w"])
            h = float(bbox["h"])
        except (KeyError, TypeError, ValueError):
            return None
        return x + w / 2.0, y + h / 2.0
    return None


def _quad_points(row: dict[str, Any]) -> list[list[float]]:
    quad = row.get("quad")
    if isinstance(quad, list) and len(quad) == 4:
        out = []
        for point in quad:
            try:
                out.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError, IndexError):
                return []
        return out
    bbox = row.get("bbox")
    if not isinstance(bbox, dict):
        return []
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return []
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _long_edge(row: dict[str, Any]) -> float | None:
    value = _positive_float(row.get("long_edge"))
    detail = row.get("detail")
    if value is None and isinstance(detail, dict):
        value = _positive_float(detail.get("long"))
    if value is not None:
        return value
    points = _quad_points(row)
    if len(points) != 4:
        return None
    lengths = []
    for idx in range(4):
        x0, y0 = points[idx]
        x1, y1 = points[(idx + 1) % 4]
        lengths.append(math.hypot(x1 - x0, y1 - y0))
    return max(lengths) if lengths else None


def _size_class(cluster: dict[str, Any]) -> str:
    value = str(cluster.get("size_class") or cluster.get("role") or "outlier")
    return value if value in {"primary", "tolerance", "aux", "outlier"} else "outlier"


def _positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def _strict_positive_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    out = float(value)
    return out if math.isfinite(out) and out > 0.0 else None


def _positive_or_zero_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0.0 else None


def _bbox_dict(points: list[tuple[float, float]] | list[list[float]]) -> dict[str, float]:
    if not points:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return {
        "x": _round(min(xs)),
        "y": _round(min(ys)),
        "w": _round(max(xs) - min(xs)),
        "h": _round(max(ys) - min(ys)),
    }


def _round_points(points: list[tuple[float, float]] | list[list[float]]) -> list[list[float]]:
    return [[_round(point[0]), _round(point[1])] for point in points]


def _round(value: float) -> float:
    return round(float(value), 6)
