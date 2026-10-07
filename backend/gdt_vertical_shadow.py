"""R7 vertical GD&T frame shadow detection.

Diagnostic-only path: decimal point angle seeds wake up local vertical FCF
frame detection, but outputs are never routed into production consumers.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pdf_analyzer_gdt_lines import (
    _angle_is_axis_aligned,
    _angle_mod_180,
    _bbox_from_points,
    _bbox_iou,
    _detect_axis_gdt_frames_from_lines,
    _oriented_quad_dict,
    _oriented_quad_from_axis_rect,
    _rotate_point,
)


SCHEMA_VERSION = "r7_gdt_vertical_shadow_v1"
SOURCE = "gdt_line_decimal_seeded_vertical_shadow"
CONSUMER_ALLOWED = False

K_SCALE = 10.1
RATIO_MIN = 7.0
RATIO_MAX = 14.0
AXIS_RADIUS_FACTOR = 8.0
CROSS_RADIUS_FACTOR = 2.2
MIN_AXIS_RADIUS = 90.0
MIN_CROSS_RADIUS = 30.0
DEDUPE_IOU = 0.45
CURRENT_OVERLAP_IOU = 0.35
VERTICAL_ANGLE_TOLERANCE = 5.0


def _angle_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def _is_vertical_angle(angle: float) -> bool:
    return _angle_delta(angle, 90.0) <= VERTICAL_ANGLE_TOLERANCE


def _bbox_area(bbox: dict[str, Any]) -> float:
    try:
        return max(0.0, float(bbox.get("w") or 0.0)) * max(
            0.0,
            float(bbox.get("h") or 0.0),
        )
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, dict):
        return False
    return _bbox_area(bbox) > 0.0


def _point_in_xywh(point: dict[str, Any], bbox: dict[str, Any], *, margin: float = 0.0) -> bool:
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
        bx = float(bbox.get("x"))
        by = float(bbox.get("y"))
        bw = float(bbox.get("w"))
        bh = float(bbox.get("h"))
    except (AttributeError, TypeError, ValueError):
        return False
    return bx - margin <= x <= bx + bw + margin and by - margin <= y <= by + bh + margin


def _line_primitives(page: Any) -> list[tuple[float, float, float, float, float]]:
    if page is None:
        return []
    lines: list[tuple[float, float, float, float, float]] = []
    for drawing in page.get_drawings():
        width = float(drawing.get("width", 0) or 0.0)
        for item in drawing.get("items", []) or []:
            if not item or item[0] != "l":
                continue
            p0, p1 = item[1], item[2]
            lines.append((float(p0.x), float(p0.y), float(p1.x), float(p1.y), width))
    return lines


def _annotation_width_candidates(
    lines: list[tuple[float, float, float, float, float]],
) -> list[float]:
    width_counts: Counter[float] = Counter()
    for *_coords, width in lines:
        if 0.1 < width < 0.6:
            width_counts[round(width, 3)] += 1
    if not width_counts:
        return []

    most_common = width_counts.most_common()
    dominant_count = most_common[0][1]
    min_secondary_count = max(500, int(dominant_count * 0.04))
    widths: list[float] = []
    for width, count in most_common:
        if widths:
            if count < min_secondary_count:
                continue
            if width > 0.49:
                continue
            if any(
                abs(width - existing) <= max(width, existing) * 0.15
                for existing in widths
            ):
                continue
        widths.append(width)
        if len(widths) >= 3:
            break
    return widths


def _lines_for_width(
    lines: list[tuple[float, float, float, float, float]],
    width: float,
) -> list[tuple[float, float, float, float]]:
    low = width * 0.85
    high = width * 1.15
    return [
        (x0, y0, x1, y1)
        for x0, y0, x1, y1, line_width in lines
        if low <= line_width <= high
    ]


def _rotated_lines_for_angle(
    lines: list[tuple[float, float, float, float]],
    angle: float,
) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in lines:
        rx0, ry0 = _rotate_point(x0, y0, -angle)
        rx1, ry1 = _rotate_point(x1, y1, -angle)
        out.append((rx0, ry0, rx1, ry1))
    return out


def _seed_angle(dot: dict[str, Any]) -> float | None:
    try:
        return _angle_mod_180(float((dot.get("short_axis") or {}).get("angle_deg")))
    except (AttributeError, TypeError, ValueError):
        return None


def _seed_long(dot: dict[str, Any]) -> float | None:
    try:
        value = float((dot.get("detail") or {}).get("long"))
    except (AttributeError, TypeError, ValueError):
        value = 0.0
    if value <= 0:
        bbox = dot.get("bbox") or {}
        try:
            value = max(float(bbox.get("w") or 0.0), float(bbox.get("h") or 0.0))
        except (AttributeError, TypeError, ValueError):
            value = 0.0
    return value if value > 0 else None


def _local_seed_window(dot: dict[str, Any], angle: float, frame_height: float) -> dict[str, float]:
    center = dot.get("center") or {}
    cx = float(center.get("x") or 0.0)
    cy = float(center.get("y") or 0.0)
    sx, sy = _rotate_point(cx, cy, -angle)
    half_x = max(frame_height * AXIS_RADIUS_FACTOR, MIN_AXIS_RADIUS)
    half_y = max(frame_height * CROSS_RADIUS_FACTOR, MIN_CROSS_RADIUS)
    return {
        "x0": sx - half_x,
        "x1": sx + half_x,
        "y0": sy - half_y,
        "y1": sy + half_y,
    }


def _filter_lines_to_window(
    rotated_lines: list[tuple[float, float, float, float]],
    window: dict[str, float],
) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in rotated_lines:
        lx0 = min(x0, x1)
        lx1 = max(x0, x1)
        ly0 = min(y0, y1)
        ly1 = max(y0, y1)
        if lx1 < window["x0"] or lx0 > window["x1"]:
            continue
        if ly1 < window["y0"] or ly0 > window["y1"]:
            continue
        out.append((x0, y0, x1, y1))
    return out


def _convert_axis_frame_to_page(frame: dict[str, Any], angle: float) -> dict[str, Any]:
    angle = _angle_mod_180(angle)
    if _angle_delta(angle, 0.0) <= 1.0:
        converted = dict(frame)
        converted.setdefault("source", SOURCE)
        return converted
    if not _angle_is_axis_aligned(angle, tolerance=1.0):
        converted = dict(frame)
        converted["source"] = SOURCE
        return converted

    bbox = frame.get("bbox") or {}
    x_left = float(bbox.get("x") or 0.0)
    y_top = float(bbox.get("y") or 0.0)
    x_right = x_left + float(bbox.get("w") or 0.0)
    y_bot = y_top + float(bbox.get("h") or 0.0)
    frame_width = x_right - x_left
    frame_height = y_bot - y_top
    quad_points = _oriented_quad_from_axis_rect(x_left, x_right, y_top, y_bot, angle)
    compartments = []
    for comp in frame.get("compartments") or []:
        cx = float(comp.get("x") or 0.0)
        cy = float(comp.get("y") or 0.0)
        cw = float(comp.get("w") or 0.0)
        ch = float(comp.get("h") or 0.0)
        comp_points = _oriented_quad_from_axis_rect(cx, cx + cw, cy, cy + ch, angle)
        compartments.append(_bbox_from_points(comp_points))

    return {
        **frame,
        "bbox": _bbox_from_points(quad_points),
        "compartments": compartments,
        "angle_deg": round(angle, 3),
        "oriented_quad": _oriented_quad_dict(quad_points, angle, frame_width, frame_height),
        "source": SOURCE,
    }


def _frame_ratio(frame: dict[str, Any], dot: dict[str, Any]) -> float | None:
    long = _seed_long(dot)
    if not long:
        return None
    try:
        return float(frame.get("frame_height") or 0.0) / long
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _seed_accepts_frame(frame: dict[str, Any], dot: dict[str, Any]) -> bool:
    ratio = _frame_ratio(frame, dot)
    if ratio is None or ratio < RATIO_MIN or ratio > RATIO_MAX:
        return False
    center = dot.get("center") or {}
    margin = max(float(frame.get("frame_height") or 0.0) * 0.35, 3.0)
    return _point_in_xywh(center, frame.get("bbox") or {}, margin=margin)


def _annotate_frame(
    frame: dict[str, Any],
    dot: dict[str, Any],
    *,
    angle: float,
    annotation_width: float,
    frame_height_estimate: float,
) -> dict[str, Any]:
    ratio = _frame_ratio(frame, dot)
    return {
        **frame,
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "consumer_allowed": CONSUMER_ALLOWED,
        "source": SOURCE,
        "seed_id": dot.get("id"),
        "seed_center": dot.get("center"),
        "seed_angle_deg": round(angle, 3),
        "seed_dot_long": _seed_long(dot),
        "seed_frame_height_estimate": round(frame_height_estimate, 3),
        "frame_height_dot_long_ratio": None if ratio is None else round(ratio, 4),
        "annotation_width": round(float(annotation_width), 3),
    }


def _dedupe_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def quality(frame: dict[str, Any]) -> tuple[float, float, float, float]:
        comp_count = float(len(frame.get("compartments") or []))
        ratio = frame.get("frame_height_dot_long_ratio")
        ratio_score = -abs(float(ratio) - K_SCALE) if ratio is not None else -99.0
        confidence = float(frame.get("confidence") or 0.0)
        area = _bbox_area(frame.get("bbox") or {})
        return comp_count, ratio_score, confidence, area

    final: list[dict[str, Any]] = []
    for frame in sorted(frames, key=quality, reverse=True):
        bbox = frame.get("bbox") or {}
        if any(_bbox_iou(bbox, existing.get("bbox") or {}) > DEDUPE_IOU for existing in final):
            continue
        final.append(frame)
    return final


def _tag_current_overlap(
    frames: list[dict[str, Any]],
    current_frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    tagged: list[dict[str, Any]] = []
    duplicate_count = 0
    new_count = 0
    current_bboxes = [
        frame.get("bbox") or {}
        for frame in current_frames
        if _valid_bbox(frame.get("bbox") or {})
    ]
    for frame in frames:
        bbox = frame.get("bbox") or {}
        best_iou = 0.0
        if _valid_bbox(bbox):
            for current_bbox in current_bboxes:
                best_iou = max(best_iou, _bbox_iou(bbox, current_bbox))
        overlaps = best_iou > CURRENT_OVERLAP_IOU
        if overlaps:
            duplicate_count += 1
        else:
            new_count += 1
        tagged.append({
            **frame,
            "overlaps_current_frame": overlaps,
            "current_overlap_iou": round(best_iou, 4),
        })
    return tagged, duplicate_count, new_count


def _empty_result(stats: dict[str, Any] | None = None) -> dict[str, Any]:
    base_stats = {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "consumer_allowed_count": 0,
        "decimal_point_count": 0,
        "input_consumer_allowed_decimal_rows": 0,
        "current_frame_count": 0,
        "annotation_widths": [],
        "vertical_seed_count": 0,
        "non_vertical_seed_count": 0,
        "missing_seed_long_count": 0,
        "seed_attempt_count": 0,
        "strict_raw_count": 0,
        "strict_frame_count": 0,
        "duplicate_current_count": 0,
        "new_independent_count": 0,
    }
    if stats:
        base_stats.update(stats)
    return {
        "r7_gdt_vertical_shadow_v1": [],
        "r7_gdt_vertical_shadow_stats_v1": base_stats,
    }


def build_gdt_decimal_seeded_frames_shadow_v1(
    *,
    page: Any,
    decimal_point_rows: list[dict[str, Any]],
    current_frames: list[dict[str, Any]],
    consumer_allowed: bool = False,
    allow_any_seed_angle: bool = False,
) -> dict[str, Any]:
    """Build diagnostic vertical GD&T frames from decimal point angle seeds."""

    if consumer_allowed:
        raise ValueError("R7 GD&T vertical shadow is dump-only; consumer_allowed must be False")

    decimal_rows = list(decimal_point_rows or [])
    current = list(current_frames or [])
    forbidden_input_count = sum(1 for row in decimal_rows if row.get("consumer_allowed") is True)
    allowed_rows = [row for row in decimal_rows if row.get("consumer_allowed") is not True]

    all_lines = _line_primitives(page)
    widths = _annotation_width_candidates(all_lines)
    if not all_lines or not widths:
        return _empty_result({
            "decimal_point_count": len(decimal_rows),
            "input_consumer_allowed_decimal_rows": forbidden_input_count,
            "current_frame_count": len(current),
            "annotation_widths": widths,
        })

    counters: Counter[str] = Counter()
    strict_frames: list[dict[str, Any]] = []
    rotated_cache: dict[tuple[float, float], list[tuple[float, float, float, float]]] = {}

    for dot in allowed_rows:
        angle = _seed_angle(dot)
        if angle is None:
            counters["non_vertical_seed_count"] += 1
            continue
        if not allow_any_seed_angle and not _is_vertical_angle(angle):
            counters["non_vertical_seed_count"] += 1
            continue
        if _is_vertical_angle(angle):
            counters["vertical_seed_count"] += 1
        else:
            counters["non_vertical_seed_count"] += 1
        long = _seed_long(dot)
        if long is None:
            counters["missing_seed_long_count"] += 1
            continue
        frame_height_estimate = K_SCALE * long
        if not (10.0 <= frame_height_estimate <= 40.0):
            counters["missing_seed_long_count"] += 1
            continue

        for width in widths:
            counters["seed_attempt_count"] += 1
            angle_key = round(angle, 3)
            cache_key = (round(width, 3), angle_key)
            if cache_key not in rotated_cache:
                anno_lines = _lines_for_width(all_lines, width)
                rotated_cache[cache_key] = _rotated_lines_for_angle(anno_lines, angle)
            window = _local_seed_window(dot, angle, frame_height_estimate)
            local_lines = _filter_lines_to_window(rotated_cache[cache_key], window)
            if not local_lines:
                continue

            for local_frame in _detect_axis_gdt_frames_from_lines(
                local_lines,
                width,
                angle_deg=angle,
            ):
                page_frame = _convert_axis_frame_to_page(local_frame, angle)
                if not _seed_accepts_frame(page_frame, dot):
                    continue
                strict_frames.append(
                    _annotate_frame(
                        page_frame,
                        dot,
                        angle=angle,
                        annotation_width=width,
                        frame_height_estimate=frame_height_estimate,
                    )
                )

    strict_unique = _dedupe_frames(strict_frames)
    tagged_frames, duplicate_count, new_count = _tag_current_overlap(strict_unique, current)
    consumer_allowed_count = sum(1 for row in tagged_frames if row.get("consumer_allowed") is True)
    stats = {
        "schema_version": SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "consumer_allowed_count": consumer_allowed_count,
        "decimal_point_count": len(decimal_rows),
        "input_consumer_allowed_decimal_rows": forbidden_input_count,
        "current_frame_count": len(current),
        "annotation_widths": widths,
        "vertical_seed_count": counters["vertical_seed_count"],
        "non_vertical_seed_count": counters["non_vertical_seed_count"],
        "allow_any_seed_angle": bool(allow_any_seed_angle),
        "missing_seed_long_count": counters["missing_seed_long_count"],
        "seed_attempt_count": counters["seed_attempt_count"],
        "strict_raw_count": len(strict_frames),
        "strict_frame_count": len(tagged_frames),
        "duplicate_current_count": duplicate_count,
        "new_independent_count": new_count,
    }
    return {
        "r7_gdt_vertical_shadow_v1": tagged_frames,
        "r7_gdt_vertical_shadow_stats_v1": stats,
    }
