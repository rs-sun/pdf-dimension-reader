"""Core decimal-point colon-pair filter.

R21 uses decimal points as trusted reference anchors. Colon-like double dots
must be filtered before layer boundaries so downstream readers do not treat
punctuation as dimension anchors. This module is intentionally independent of
probe scripts.
"""

from __future__ import annotations

import statistics
from typing import Any


CONSUMER_ALLOWED = False


def xywh_from_any(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        if all(key in value for key in ("x", "y", "w", "h")):
            return {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "w": float(value["w"]),
                "h": float(value["h"]),
            }
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            x0 = float(value["x0"])
            y0 = float(value["y0"])
            x1 = float(value["x1"])
            y1 = float(value["y1"])
            return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    except (TypeError, ValueError):
        return None
    return None


def round_xywh(value: dict[str, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {key: round(float(value[key]), 3) for key in ("x", "y", "w", "h")}


def round_point(value: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {"x": round(float(value["x"]), 3), "y": round(float(value["y"]), 3)}
    except (TypeError, ValueError, KeyError):
        return None


def center_of_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x": float(bbox["x"]) + float(bbox["w"]) / 2.0,
        "y": float(bbox["y"]) + float(bbox["h"]) / 2.0,
    }


def decimal_nominal_size(dot: dict[str, Any]) -> float:
    detail = dot.get("detail") or {}
    values: list[float] = []
    for key in ("short", "long", "bbox_w", "bbox_h"):
        try:
            value = float(detail.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    bbox = xywh_from_any(dot.get("bbox"))
    if bbox is not None:
        values.extend([abs(float(bbox["w"])), abs(float(bbox["h"]))])
    return max(values) if values else 1.0


def is_colon_decimal_pair(
    upper: dict[str, Any],
    lower: dict[str, Any],
    *,
    x_factor: float = 0.90,
    gap_min_factor: float = 2.00,
    gap_max_factor: float = 3.20,
    page_median_size: float | None = None,
    require_large_scale: bool = False,
    min_size_to_median: float = 1.20,
    max_pair_size_ratio: float = 1.45,
) -> bool:
    upper_bbox = xywh_from_any(upper.get("bbox"))
    lower_bbox = xywh_from_any(lower.get("bbox"))
    if upper_bbox is None or lower_bbox is None:
        return False
    c1 = upper.get("center") or center_of_bbox(upper_bbox)
    c2 = lower.get("center") or center_of_bbox(lower_bbox)
    try:
        dx = abs(float(c1["x"]) - float(c2["x"]))
        dy = abs(float(c1["y"]) - float(c2["y"]))
    except (TypeError, ValueError, KeyError):
        return False
    upper_size = decimal_nominal_size(upper)
    lower_size = decimal_nominal_size(lower)
    size = max(upper_size, lower_size)
    min_size = min(upper_size, lower_size)
    if min_size <= 0:
        return False
    if size / min_size > float(max_pair_size_ratio):
        return False
    if require_large_scale:
        if page_median_size is None or page_median_size <= 0:
            return False
        if size / float(page_median_size) < float(min_size_to_median):
            return False
    x_limit = max(1.25, size * float(x_factor))
    return x_limit >= dx and size * float(gap_min_factor) <= dy <= size * float(gap_max_factor)


def filter_colon_decimal_pairs(
    dots: list[dict[str, Any]],
    *,
    extra_colon_dots: list[dict[str, Any]] | None = None,
    x_factor: float = 0.90,
    gap_min_factor: float = 2.00,
    gap_max_factor: float = 3.20,
    min_size_to_median: float = 1.20,
    max_pair_size_ratio: float = 1.45,
) -> dict[str, Any]:
    decimal_dots = list(dots or [])
    colon_candidate_dots = list(extra_colon_dots or [])
    decimal_sizes = [decimal_nominal_size(row) for row in decimal_dots if decimal_nominal_size(row) > 0]
    page_median_size = statistics.median(decimal_sizes) if decimal_sizes else None
    require_large_scale = len(decimal_sizes) >= 8
    pair_by_id: dict[str, str] = {}
    pair_rows: list[dict[str, Any]] = []
    candidates = [
        _candidate_dot(row, index=index, origin="decimal")
        for index, row in enumerate(decimal_dots)
    ]
    candidates.extend(
        _candidate_dot(row, index=index, origin="colon_candidate")
        for index, row in enumerate(colon_candidate_dots)
    )
    sorted_dots = sorted(
        candidates,
        key=lambda row: (
            float((row.get("center") or {}).get("x", 0.0)),
            float((row.get("center") or {}).get("y", 0.0)),
        ),
    )
    for idx, first in enumerate(sorted_dots):
        first_id = str(first.get("id") or f"dot_{idx:04d}")
        if first_id in pair_by_id:
            continue
        for jdx in range(idx + 1, len(sorted_dots)):
            second = sorted_dots[jdx]
            second_id = str(second.get("id") or f"dot_{jdx:04d}")
            if second_id in pair_by_id:
                continue
            if first.get("_origin") == "colon_candidate" and second.get("_origin") == "colon_candidate":
                continue
            if is_colon_decimal_pair(
                first,
                second,
                x_factor=x_factor,
                gap_min_factor=gap_min_factor,
                gap_max_factor=gap_max_factor,
                page_median_size=page_median_size,
                require_large_scale=require_large_scale,
                min_size_to_median=min_size_to_median,
                max_pair_size_ratio=max_pair_size_ratio,
            ):
                pair_by_id[first_id] = second_id
                pair_by_id[second_id] = first_id
                pair_rows.append(_colon_pair_row(
                    first,
                    second,
                    pair_id=f"colon_pair_{len(pair_rows):06d}",
                    page_median_size=page_median_size,
                ))
                break

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for idx, dot in enumerate(decimal_dots):
        dot_id = str(dot.get("id") or f"dot_{idx:04d}")
        keep = dot_id not in pair_by_id
        row = {
            "dot_id": dot_id,
            "bbox": round_xywh(xywh_from_any(dot.get("bbox"))),
            "center": round_point(dot.get("center")),
            "keep": keep,
            "paired_dot_id": pair_by_id.get(dot_id, ""),
            "reason": "keep_decimal_point" if keep else "drop_colon_vertical_pair",
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        rows.append(row)
        if keep:
            kept.append(dot)
        else:
            dropped.append(dot)
    return {
        "kept": kept,
        "dropped": dropped,
        "rows": rows,
        "colon_pairs": pair_rows,
        "summary": {
            "input_dot_count": len(decimal_dots),
            "colon_candidate_dot_count": len(colon_candidate_dots),
            "kept_dot_count": len(kept),
            "dropped_colon_dot_count": len(dropped),
            "colon_pair_count": len(pair_rows),
            "page_median_decimal_size": round(float(page_median_size), 4) if page_median_size else None,
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    }


def _candidate_dot(row: dict[str, Any], *, index: int, origin: str) -> dict[str, Any]:
    out = dict(row)
    out["_origin"] = origin
    out["id"] = str(row.get("id") or f"{origin}_{index:04d}")
    bbox = xywh_from_any(out.get("bbox"))
    if bbox is not None and not isinstance(out.get("center"), dict):
        out["center"] = center_of_bbox(bbox)
    return out


def _colon_pair_row(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    pair_id: str,
    page_median_size: float | None,
) -> dict[str, Any]:
    ordered = sorted([first, second], key=lambda row: float((row.get("center") or {}).get("y", 0.0)))
    upper, lower = ordered
    upper_bbox = xywh_from_any(upper.get("bbox"))
    lower_bbox = xywh_from_any(lower.get("bbox"))
    upper_center = upper.get("center") or (center_of_bbox(upper_bbox) if upper_bbox else None)
    lower_center = lower.get("center") or (center_of_bbox(lower_bbox) if lower_bbox else None)
    upper_size = decimal_nominal_size(upper)
    lower_size = decimal_nominal_size(lower)
    size = max(upper_size, lower_size, 1e-6)
    min_size = max(min(upper_size, lower_size), 1e-6)
    dx = abs(float(upper_center["x"]) - float(lower_center["x"])) if upper_center and lower_center else 0.0
    dy = abs(float(upper_center["y"]) - float(lower_center["y"])) if upper_center and lower_center else 0.0
    return {
        "pair_id": pair_id,
        "kind": "colon_pair",
        "point_ids": [str(upper.get("id") or ""), str(lower.get("id") or "")],
        "points": [_point_payload(upper, page_median_size), _point_payload(lower, page_median_size)],
        "bbox": round_xywh(_union_bbox([upper_bbox, lower_bbox])),
        "center": round_point(_union_center([upper_bbox, lower_bbox])),
        "direction": "vertical",
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "dy_over_size": round(dy / size, 4),
        "size_ratio": round(size / min_size, 4),
        "size_to_page_median": (
            round(size / float(page_median_size), 4)
            if page_median_size and page_median_size > 0 else None
        ),
        "source": "decimal_colon_filter",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _point_payload(row: dict[str, Any], page_median_size: float | None) -> dict[str, Any]:
    size = decimal_nominal_size(row)
    return {
        "dot_id": str(row.get("id") or ""),
        "source": str(row.get("source") or ""),
        "origin": str(row.get("_origin") or ""),
        "bbox": round_xywh(xywh_from_any(row.get("bbox"))),
        "center": round_point(row.get("center")),
        "size": round(float(size), 4),
        "size_to_page_median": (
            round(float(size) / float(page_median_size), 4)
            if page_median_size and page_median_size > 0 else None
        ),
    }


def _union_bbox(boxes: list[dict[str, float] | None]) -> dict[str, float] | None:
    valid = [box for box in boxes if box is not None]
    if not valid:
        return None
    x0 = min(float(box["x"]) for box in valid)
    y0 = min(float(box["y"]) for box in valid)
    x1 = max(float(box["x"]) + float(box["w"]) for box in valid)
    y1 = max(float(box["y"]) + float(box["h"]) for box in valid)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _union_center(boxes: list[dict[str, float] | None]) -> dict[str, float] | None:
    bbox = _union_bbox(boxes)
    if bbox is None:
        return None
    return center_of_bbox(bbox)
