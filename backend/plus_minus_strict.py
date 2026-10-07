"""Strict local-geometry +/- detector for R22 dump-only measurement."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any

import fitz

from degree_polyline_strict import Segment, UnionFind, extract_line_segments


SCHEMA_VERSION = "r22_pm_strict_v1"
CONSUMER_ALLOWED = False
TOPOLOGY_PROOF_SCHEMA_VERSION = "plus_minus_topology_proof_v1"

SOURCE = "pm_strict"
ANGLE_TOL_DEG = 8.0
PLUS_MIDPOINT_TOL_FACTOR = 0.18
PLUS_LEN_RATIO_MIN = 0.65
PLUS_LEN_RATIO_MAX = 1.55
STROKE_LEN_MIN_FACTOR = 0.25
STROKE_LEN_MAX_FACTOR = 1.20
PM_ALIGN_FACTOR = 0.18
PM_GAP_MIN_FACTOR = 0.05
PM_GAP_MAX_FACTOR = 0.35
NEAR_EXTRA_U_FACTOR = 1.10
NEAR_EXTRA_V_FACTOR = 1.40
ROLE_HULL_MARGIN_SCALE_FACTOR = 0.30


@dataclass(frozen=True)
class AnchorContext:
    index: int
    anchor_id: str
    center: tuple[float, float]
    axis_angle_deg: float
    scale_baseline: float
    stroke_w: float | None
    main_limit: float
    cross_limit: float


@dataclass(frozen=True)
class LocalSegment:
    segment: Segment
    p0: tuple[float, float]
    p1: tuple[float, float]
    mid: tuple[float, float]
    local_angle_deg: float
    is_axis: bool
    is_cross: bool


def detect_plus_minus_strict(
    page: fitz.Page,
    anchor_rows: list[dict[str, Any]],
    *,
    page_index: int = 0,
) -> dict[str, Any]:
    return detect_plus_minus_from_segments(
        extract_line_segments(page),
        anchor_rows,
        page_index=page_index,
    )


def prove_plus_minus_topology(
    segments: list[Segment],
    *,
    axis_angle_deg: float,
    scale_baseline: float,
    stroke_w: float | None = None,
    neighbor_segments: list[Segment] | None = None,
    primitive_ids_by_segment_id: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    """Prove one complete ``±`` from exactly three role-assigned lines.

    The proof is intentionally page/context free so discovery and semantic
    replay can call the same predicate.  Standalone ``+`` and ``-`` are not
    accepted by this interface.
    """
    if (
        not isinstance(segments, list)
        or len(segments) != 3
        or any(not isinstance(segment, Segment) for segment in segments)
        or len({segment.segment_id for segment in segments}) != 3
    ):
        return None
    try:
        axis = float(axis_angle_deg) % 180.0
        scale = float(scale_baseline)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(axis) or not math.isfinite(scale) or scale <= 0.0:
        return None
    minimum_length = max(1.0, scale * STROKE_LEN_MIN_FACTOR)
    maximum_length = max(2.5, scale * STROKE_LEN_MAX_FACTOR)
    if any(
        not math.isfinite(float(segment.length))
        or not (minimum_length <= float(segment.length) <= maximum_length)
        or not _stroke_width_matches(segment.stroke_width, stroke_w)
        for segment in segments
    ):
        return None

    bars = [
        segment
        for segment in segments
        if _angle_delta(segment.angle_deg, axis) <= ANGLE_TOL_DEG
    ]
    stems = [
        segment
        for segment in segments
        if _angle_delta(segment.angle_deg, (axis + 90.0) % 180.0)
        <= ANGLE_TOL_DEG
    ]
    if len(bars) != 2 or len(stems) != 1:
        return None
    stem = stems[0]
    midpoint_tolerance = max(0.8, scale * PLUS_MIDPOINT_TOL_FACTOR)
    intersecting: list[
        tuple[Segment, tuple[float, float], float, float]
    ] = []
    for bar in bars:
        intersection = _segment_intersection(
            bar.p0,
            bar.p1,
            stem.p0,
            stem.p1,
        )
        if intersection is None:
            continue
        point, bar_fraction, stem_fraction = intersection
        if not (
            0.20 <= bar_fraction <= 0.80
            and 0.20 <= stem_fraction <= 0.80
            and _distance(point, _segment_midpoint(bar))
            <= midpoint_tolerance
            and _distance(point, _segment_midpoint(stem))
            <= midpoint_tolerance
        ):
            continue
        ratio = bar.length / max(stem.length, 1e-9)
        if not (PLUS_LEN_RATIO_MIN <= ratio <= PLUS_LEN_RATIO_MAX):
            continue
        intersecting.append((bar, point, bar_fraction, stem_fraction))
    if len(intersecting) != 1:
        return None
    upper_bar, intersection, bar_fraction, stem_fraction = intersecting[0]
    lower_candidates = [bar for bar in bars if bar is not upper_bar]
    if len(lower_candidates) != 1:
        return None
    lower_bar = lower_candidates[0]
    if _segment_intersection(
        lower_bar.p0,
        lower_bar.p1,
        stem.p0,
        stem.p1,
    ) is not None:
        return None
    lower_ratio = lower_bar.length / max(upper_bar.length, 1e-9)
    if not (PLUS_LEN_RATIO_MIN <= lower_ratio <= PLUS_LEN_RATIO_MAX):
        return None

    radians = math.radians(axis)
    axis_unit = (math.cos(radians), math.sin(radians))
    normal_unit = (-axis_unit[1], axis_unit[0])
    if (
        normal_unit[1] < -1e-9
        or (
            abs(normal_unit[1]) <= 1e-9
            and normal_unit[0] < 0.0
        )
    ):
        normal_unit = (-normal_unit[0], -normal_unit[1])
    lower_midpoint = _segment_midpoint(lower_bar)
    along_offset = abs(
        _dot(lower_midpoint, axis_unit)
        - _dot(intersection, axis_unit)
    )
    if along_offset > max(1.2, scale * PM_ALIGN_FACTOR):
        return None
    lower_side_offset = (
        _dot(lower_midpoint, normal_unit)
        - _dot(intersection, normal_unit)
    )
    if lower_side_offset <= 0.0:
        return None
    separation = lower_side_offset
    gap = separation - stem.length / 2.0
    stroke_reference = _topology_stroke_reference(
        [upper_bar, lower_bar, stem],
        fallback=stroke_w,
    )
    gap_minimum = max(
        0.35,
        scale * PM_GAP_MIN_FACTOR,
        stroke_reference * 1.5,
    )
    gap_maximum = max(2.0, scale * PM_GAP_MAX_FACTOR)
    if not (gap_minimum <= gap <= gap_maximum):
        return None

    role_segments = [
        ("upper_bar", upper_bar),
        ("lower_bar", lower_bar),
        ("stem", stem),
    ]
    role_bindings = _canonical_role_bindings(
        role_segments,
        primitive_ids_by_segment_id=primitive_ids_by_segment_id,
    )
    if role_bindings is None:
        return None
    isolation = _neighbor_ink_isolation(
        role_segments,
        neighbor_segments=neighbor_segments,
        axis_angle_deg=axis,
        scale_baseline=scale,
        stroke_reference=stroke_reference,
    )
    if isolation is None:
        return None
    geometry = {
        "axis_angle_deg": round(axis, 6),
        "scale_baseline": round(scale, 6),
        "bar_stem_angle_error_deg": round(
            _angle_delta(stem.angle_deg, (axis + 90.0) % 180.0),
            6,
        ),
        "bar_pair_angle_error_deg": round(
            _angle_delta(upper_bar.angle_deg, lower_bar.angle_deg),
            6,
        ),
        "bar_fraction": round(bar_fraction, 6),
        "stem_fraction": round(stem_fraction, 6),
        "lower_bar_along_offset_pt": round(along_offset, 6),
        "lower_bar_side_offset_pt": round(lower_side_offset, 6),
        "lower_bar_gap_pt": round(gap, 6),
        "role_lengths_pt": {
            role: round(float(segment.length), 6)
            for role, segment in role_segments
        },
        "isolation": isolation,
    }
    digest_payload = {
        "schema_version": TOPOLOGY_PROOF_SCHEMA_VERSION,
        "role_bindings": role_bindings,
        "geometry": geometry,
    }
    return {
        "schema_version": TOPOLOGY_PROOF_SCHEMA_VERSION,
        "source_roles": [
            {"role": role, "segment_id": int(segment.segment_id)}
            for role, segment in role_segments
        ],
        "role_bindings": role_bindings,
        "axis_angle_deg": round(axis, 6),
        "geometry": geometry,
        "topology_digest": hashlib.sha256(json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _canonical_role_bindings(
    role_segments: list[tuple[str, Segment]],
    *,
    primitive_ids_by_segment_id: dict[int, str] | None,
) -> list[dict[str, Any]] | None:
    if primitive_ids_by_segment_id is not None and not isinstance(
        primitive_ids_by_segment_id,
        dict,
    ):
        return None
    rows: list[dict[str, Any]] = []
    for role, segment in role_segments:
        primitive_id = (
            primitive_ids_by_segment_id.get(segment.segment_id)
            if primitive_ids_by_segment_id is not None
            else (
                f"d{int(segment.drawing_order):06d}_"
                f"i{int(segment.item_index):04d}_l"
            )
        )
        if type(primitive_id) is not str or not primitive_id:
            return None
        rows.append({
            "role": role,
            "primitive_id": primitive_id,
            "drawing_order": int(segment.drawing_order),
            "item_index": int(segment.item_index),
        })
    if len({row["primitive_id"] for row in rows}) != len(rows):
        return None
    return rows


def _neighbor_ink_isolation(
    role_segments: list[tuple[str, Segment]],
    *,
    neighbor_segments: list[Segment] | None,
    axis_angle_deg: float,
    scale_baseline: float,
    stroke_reference: float,
) -> dict[str, Any] | None:
    if neighbor_segments is None:
        return {
            "checked": False,
            "neighbor_count": 0,
            "unused_intersecting_count": 0,
            "envelope_margin_pt": 0.0,
        }
    if (
        not isinstance(neighbor_segments, list)
        or any(
            not isinstance(segment, Segment)
            for segment in neighbor_segments
        )
        or len({segment.segment_id for segment in neighbor_segments})
        != len(neighbor_segments)
    ):
        return None
    role_ids = {
        segment.segment_id for _role, segment in role_segments
    }
    if not role_ids.issubset({
        segment.segment_id for segment in neighbor_segments
    }):
        return None

    radians = math.radians(axis_angle_deg)
    axis_unit = (math.cos(radians), math.sin(radians))
    normal_unit = (-axis_unit[1], axis_unit[0])
    role_points = [
        (
            _dot(point, axis_unit),
            _dot(point, normal_unit),
        )
        for _role, segment in role_segments
        for point in (segment.p0, segment.p1)
    ]
    margin = max(
        0.25,
        scale_baseline * ROLE_HULL_MARGIN_SCALE_FACTOR,
        stroke_reference * 2.0,
    )
    u_min = min(point[0] for point in role_points) - margin
    u_max = max(point[0] for point in role_points) + margin
    v_min = min(point[1] for point in role_points) - margin
    v_max = max(point[1] for point in role_points) + margin
    unused_intersecting = 0
    for segment in neighbor_segments:
        if segment.segment_id in role_ids:
            continue
        points = [
            (
                _dot(point, axis_unit),
                _dot(point, normal_unit),
            )
            for point in (segment.p0, segment.p1)
        ]
        if _segment_intersects_local_rect(
            points[0],
            points[1],
            u_min=u_min,
            v_min=v_min,
            u_max=u_max,
            v_max=v_max,
        ):
            unused_intersecting += 1
    if unused_intersecting:
        return None
    return {
        "checked": True,
        "neighbor_count": len(role_ids),
        "unused_intersecting_count": 0,
        "envelope_margin_pt": round(margin, 6),
    }


def _segment_intersects_local_rect(
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    u_min: float,
    v_min: float,
    u_max: float,
    v_max: float,
) -> bool:
    def _inside(point: tuple[float, float]) -> bool:
        return (
            u_min <= point[0] <= u_max
            and v_min <= point[1] <= v_max
        )

    if _inside(p0) or _inside(p1):
        return True
    corners = (
        (u_min, v_min),
        (u_max, v_min),
        (u_max, v_max),
        (u_min, v_max),
    )
    return any(
        _segment_intersection(
            p0,
            p1,
            corners[index],
            corners[(index + 1) % len(corners)],
        )
        is not None
        for index in range(len(corners))
    )


def detect_plus_minus_from_segments(
    segments: list[Segment],
    anchor_rows: list[dict[str, Any]],
    *,
    page_index: int = 0,
) -> dict[str, Any]:
    contexts = [_anchor_context(index, row) for index, row in enumerate(anchor_rows)]
    endpoint_neighbors = _endpoint_neighbor_map(segments)
    rows: list[dict[str, Any]] = []
    for context in contexts:
        rows.extend(_detect_for_anchor(segments, context, contexts, endpoint_neighbors))

    rows.sort(key=lambda row: (str(row["anchor_id"]), str(row["symbol"]), row["center"]["x"], row["center"]["y"]))
    for index, row in enumerate(rows):
        row["id"] = f"pm_strict_{index:04d}"
        row["page_index"] = int(page_index)

    counts = Counter(row["symbol"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "r22_pm_strict_v1": rows,
        "r22_pm_strict_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "page_index": int(page_index),
            "count": len(rows),
            "counts_by_symbol": dict(sorted(counts.items())),
            "consumer_allowed_count": sum(1 for row in rows if row.get("consumer_allowed")),
        },
    }


def _detect_for_anchor(
    segments: list[Segment],
    context: AnchorContext,
    all_contexts: list[AnchorContext],
    endpoint_neighbors: dict[int, set[int]],
) -> list[dict[str, Any]]:
    local_segments = [
        local
        for segment in segments
        if (local := _local_segment(segment, context)) is not None
    ]
    axis_segments = [segment for segment in local_segments if segment.is_axis and _segment_in_length_band(segment, context)]
    cross_segments = [segment for segment in local_segments if segment.is_cross and _segment_in_length_band(segment, context)]

    rows: list[dict[str, Any]] = []
    used_segment_ids: set[int] = set()
    plus_candidates = _plus_candidates(axis_segments, cross_segments, context)

    for plus_candidate in plus_candidates:
        if used_segment_ids.intersection(plus_candidate["segment_ids"]):
            continue
        extra = _matching_plus_minus_bar(plus_candidate, axis_segments, context)
        if extra is None:
            continue
        row_segments = [
            plus_candidate["axis_segment"].segment,
            plus_candidate["cross_segment"].segment,
            extra.segment,
        ]
        row = _row_from_segments(
            "plus_minus",
            row_segments,
            context,
        )
        if _nearest_anchor_index(_center_tuple(row["center"]), all_contexts) != context.index:
            continue
        rows.append(row)
        used_segment_ids.update(segment.segment_id for segment in row_segments)

    for plus_candidate in plus_candidates:
        if used_segment_ids.intersection(plus_candidate["segment_ids"]):
            continue
        if _has_near_extra_segment(plus_candidate["center_local"], plus_candidate["segment_ids"], local_segments, context):
            continue
        row_segments = [
            plus_candidate["axis_segment"].segment,
            plus_candidate["cross_segment"].segment,
        ]
        row = _row_from_segments(
            "plus",
            row_segments,
            context,
        )
        if _nearest_anchor_index(_center_tuple(row["center"]), all_contexts) != context.index:
            continue
        rows.append(row)
        used_segment_ids.update(segment.segment_id for segment in row_segments)

    for axis_segment in axis_segments:
        segment_id = axis_segment.segment.segment_id
        if segment_id in used_segment_ids:
            continue
        if endpoint_neighbors.get(segment_id):
            continue
        if _has_near_extra_segment(axis_segment.mid, {segment_id}, local_segments, context):
            continue
        row = _row_from_segments("minus", [axis_segment.segment], context)
        if _nearest_anchor_index(_center_tuple(row["center"]), all_contexts) != context.index:
            continue
        rows.append(row)
        used_segment_ids.add(segment_id)

    return rows


def _plus_candidates(
    axis_segments: list[LocalSegment],
    cross_segments: list[LocalSegment],
    context: AnchorContext,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    midpoint_tol = max(0.8, context.scale_baseline * PLUS_MIDPOINT_TOL_FACTOR)
    for axis_segment in axis_segments:
        for cross_segment in cross_segments:
            intersection = _segment_intersection(
                axis_segment.p0,
                axis_segment.p1,
                cross_segment.p0,
                cross_segment.p1,
            )
            if intersection is None:
                continue
            point, axis_fraction, cross_fraction = intersection
            if not (0.20 <= axis_fraction <= 0.80 and 0.20 <= cross_fraction <= 0.80):
                continue
            if _distance(point, axis_segment.mid) > midpoint_tol:
                continue
            if _distance(point, cross_segment.mid) > midpoint_tol:
                continue
            ratio = axis_segment.segment.length / max(cross_segment.segment.length, 1e-9)
            if not (PLUS_LEN_RATIO_MIN <= ratio <= PLUS_LEN_RATIO_MAX):
                continue
            candidates.append({
                "axis_segment": axis_segment,
                "cross_segment": cross_segment,
                "segment_ids": {axis_segment.segment.segment_id, cross_segment.segment.segment_id},
                "center_local": point,
            })
    return candidates


def _matching_plus_minus_bar(
    plus_candidate: dict[str, Any],
    axis_segments: list[LocalSegment],
    context: AnchorContext,
) -> LocalSegment | None:
    axis_segment = plus_candidate["axis_segment"]
    cross_segment = plus_candidate["cross_segment"]
    plus_center = plus_candidate["center_local"]
    align_tol = max(1.2, context.scale_baseline * PM_ALIGN_FACTOR)
    gap_min = max(0.35, context.scale_baseline * PM_GAP_MIN_FACTOR, _stroke_reference(context, [axis_segment, cross_segment]) * 1.5)
    gap_max = max(2.0, context.scale_baseline * PM_GAP_MAX_FACTOR)

    matches: list[tuple[float, LocalSegment]] = []
    for candidate in axis_segments:
        if candidate.segment.segment_id in plus_candidate["segment_ids"]:
            continue
        ratio = candidate.segment.length / max(axis_segment.segment.length, 1e-9)
        if not (PLUS_LEN_RATIO_MIN <= ratio <= PLUS_LEN_RATIO_MAX):
            continue
        if abs(candidate.mid[0] - plus_center[0]) > align_tol:
            continue
        separation = abs(candidate.mid[1] - plus_center[1])
        gap = separation - cross_segment.segment.length / 2.0
        if not (gap_min <= gap <= gap_max):
            continue
        matches.append((abs(candidate.mid[0] - plus_center[0]) + abs(gap - gap_min), candidate))

    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1].segment.segment_id))
    return matches[0][1]


def _has_near_extra_segment(
    center_local: tuple[float, float],
    own_segment_ids: set[int],
    local_segments: list[LocalSegment],
    context: AnchorContext,
) -> bool:
    u_tol = max(8.0, context.scale_baseline * NEAR_EXTRA_U_FACTOR)
    v_tol = max(6.0, context.scale_baseline * NEAR_EXTRA_V_FACTOR)
    for local in local_segments:
        if local.segment.segment_id in own_segment_ids:
            continue
        if not (local.is_axis or local.is_cross):
            continue
        if abs(local.mid[0] - center_local[0]) <= u_tol and abs(local.mid[1] - center_local[1]) <= v_tol:
            return True
    return False


def _local_segment(segment: Segment, context: AnchorContext) -> LocalSegment | None:
    p0 = _to_local(segment.p0, context)
    p1 = _to_local(segment.p1, context)
    mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    if abs(mid[0]) > context.main_limit or abs(mid[1]) > context.cross_limit:
        return None
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0
    return LocalSegment(
        segment=segment,
        p0=p0,
        p1=p1,
        mid=mid,
        local_angle_deg=angle,
        is_axis=_angle_delta(angle, 0.0) <= ANGLE_TOL_DEG,
        is_cross=_angle_delta(angle, 90.0) <= ANGLE_TOL_DEG,
    )


def _segment_in_length_band(local_segment: LocalSegment, context: AnchorContext) -> bool:
    min_len, max_len = _length_band(context)
    return (
        min_len <= local_segment.segment.length <= max_len
        and _stroke_width_matches(local_segment.segment.stroke_width, context.stroke_w)
    )


def _length_band(context: AnchorContext) -> tuple[float, float]:
    return (
        max(1.0, context.scale_baseline * STROKE_LEN_MIN_FACTOR),
        max(2.5, context.scale_baseline * STROKE_LEN_MAX_FACTOR),
    )


def _row_from_segments(
    symbol: str,
    row_segments: list[Segment],
    context: AnchorContext,
) -> dict[str, Any]:
    points = [point for segment in row_segments for point in (segment.p0, segment.p1)]
    bbox = _bbox_from_points(points)
    center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "",
        "symbol": symbol,
        "bbox": _bbox_dict(bbox),
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "axis_angle_deg": round(context.axis_angle_deg, 4),
        "anchor_id": context.anchor_id,
        "segment_ids": sorted(segment.segment_id for segment in row_segments),
        "calibrated_stroke_len": round(float(median(segment.length for segment in row_segments)), 3),
        "source": SOURCE,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _anchor_context(index: int, row: dict[str, Any]) -> AnchorContext:
    center = _anchor_center(row)
    scale = _anchor_scale(row)
    return AnchorContext(
        index=index,
        anchor_id=str(row.get("id") or row.get("anchor_id") or row.get("reference_id") or f"anchor_{index:04d}"),
        center=center,
        axis_angle_deg=float(row.get("axis_angle_deg") or 0.0) % 180.0,
        scale_baseline=scale,
        stroke_w=_optional_float(row.get("stroke_w") or row.get("stroke_width")),
        main_limit=max(scale * 8.0, 64.0),
        cross_limit=max(scale * 1.5, 6.0),
    )


def _anchor_center(row: dict[str, Any]) -> tuple[float, float]:
    center = row.get("center")
    if isinstance(center, dict) and "x" in center and "y" in center:
        return float(center["x"]), float(center["y"])
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        return float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) / 2.0, float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) / 2.0
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0
    return float(row.get("x") or 0.0), float(row.get("y") or 0.0)


def _anchor_scale(row: dict[str, Any]) -> float:
    for key in ("scale_baseline", "glyph_h", "char_h", "height", "font_size"):
        value = _optional_float(row.get(key))
        if value and value > 0:
            return value
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        width = _optional_float(bbox.get("w")) or 0.0
        height = _optional_float(bbox.get("h")) or 0.0
        if max(width, height) > 0:
            return max(width, height)
    return 10.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_local(point: tuple[float, float], context: AnchorContext) -> tuple[float, float]:
    radians = math.radians(context.axis_angle_deg)
    dx = point[0] - context.center[0]
    dy = point[1] - context.center[1]
    return (
        dx * math.cos(radians) + dy * math.sin(radians),
        -dx * math.sin(radians) + dy * math.cos(radians),
    )


def _segment_intersection(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> tuple[tuple[float, float], float, float] | None:
    ax = a1[0] - a0[0]
    ay = a1[1] - a0[1]
    bx = b1[0] - b0[0]
    by = b1[1] - b0[1]
    denom = ax * by - ay * bx
    if abs(denom) < 1e-9:
        return None
    dx = b0[0] - a0[0]
    dy = b0[1] - a0[1]
    t = (dx * by - dy * bx) / denom
    u = (dx * ay - dy * ax) / denom
    if not (-1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6):
        return None
    return (a0[0] + t * ax, a0[1] + t * ay), t, u


def _endpoint_neighbor_map(segments: list[Segment]) -> dict[int, set[int]]:
    if not segments:
        return {}
    endpoints = [point for segment in segments for point in (segment.p0, segment.p1)]
    eps = max(0.1, _median_stroke_width(segments) * 1.5)
    uf = UnionFind(len(endpoints))
    grid: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(endpoints):
        gx = int(math.floor(point[0] / eps))
        gy = int(math.floor(point[1] / eps))
        for nx in range(gx - 1, gx + 2):
            for ny in range(gy - 1, gy + 2):
                for other_index in grid.get((nx, ny), []):
                    if _distance(point, endpoints[other_index]) <= eps:
                        uf.union(index, other_index)
        grid.setdefault((gx, gy), []).append(index)
    nodes: dict[int, list[int]] = {}
    for index in range(len(endpoints)):
        nodes.setdefault(uf.find(index), []).append(index // 2)

    neighbors: dict[int, set[int]] = {segment.segment_id: set() for segment in segments}
    for segment_indices in nodes.values():
        unique = set(segment_indices)
        if len(unique) <= 1:
            continue
        for segment_index in unique:
            segment_id = segments[segment_index].segment_id
            neighbors.setdefault(segment_id, set()).update(
                segments[other].segment_id for other in unique if other != segment_index
            )
    return neighbors


def _median_stroke_width(segments: list[Segment]) -> float:
    widths = [float(segment.stroke_width) for segment in segments if segment.stroke_width and segment.stroke_width > 0]
    return float(median(widths)) if widths else 0.25


def _stroke_reference(context: AnchorContext, segments: list[LocalSegment]) -> float:
    widths = [
        float(segment.segment.stroke_width)
        for segment in segments
        if segment.segment.stroke_width is not None and segment.segment.stroke_width > 0
    ]
    if widths:
        return float(median(widths))
    return context.stroke_w or 0.25


def _stroke_width_matches(segment_width: float | None, anchor_width: float | None) -> bool:
    if segment_width is None or anchor_width is None or segment_width <= 0 or anchor_width <= 0:
        return True
    ratio = segment_width / anchor_width
    return 0.35 <= ratio <= 2.80


def _nearest_anchor_index(center: tuple[float, float], contexts: list[AnchorContext]) -> int:
    return min(
        contexts,
        key=lambda context: (_distance(center, context.center), context.index),
    ).index


def _center_tuple(center: dict[str, float]) -> tuple[float, float]:
    return float(center["x"]), float(center["y"])


def _segment_midpoint(segment: Segment) -> tuple[float, float]:
    return (
        (float(segment.p0[0]) + float(segment.p1[0])) / 2.0,
        (float(segment.p0[1]) + float(segment.p1[1])) / 2.0,
    )


def _dot(
    point: tuple[float, float],
    unit: tuple[float, float],
) -> float:
    return float(point[0]) * float(unit[0]) + float(point[1]) * float(unit[1])


def _topology_stroke_reference(
    segments: list[Segment],
    *,
    fallback: float | None,
) -> float:
    widths = [
        float(segment.stroke_width)
        for segment in segments
        if (
            segment.stroke_width is not None
            and math.isfinite(float(segment.stroke_width))
            and float(segment.stroke_width) > 0.0
        )
    ]
    if widths:
        return float(median(widths))
    if fallback is not None:
        try:
            value = float(fallback)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if math.isfinite(value) and value > 0.0:
            return value
    return 0.25


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_dict(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "x": round(bbox[0], 3),
        "y": round(bbox[1], 3),
        "w": round(bbox[2] - bbox[0], 3),
        "h": round(bbox[3] - bbox[1], 3),
    }


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
