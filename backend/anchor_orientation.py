"""Trusted anchor-axis orientation helpers for shadow vector readers."""

from __future__ import annotations

import math
from typing import Any


TRUSTED_AXIS_SOURCES = frozenset({
    "axis_angle_deg",
    "dimension_axis_angle_deg",
    "decimal_point_quad",
    "dot_axis_angle_deg",
    "dot_pair",
    "dot_to_digit_pca",
    "oriented_quad:dot_axis_angle",
})

FALLBACK_AXIS_SOURCES = frozenset({"candidate_bbox", "digit_centroid_pca"})


def resolve_candidate_orientation(
    candidate: dict[str, Any],
    *,
    bbox_orientation: str,
    vertical_threshold_deg: float = 45.0,
) -> dict[str, Any]:
    """Return H/V orientation, preferring explicit trusted anchor angles."""
    bbox_orientation = _normalize_orientation(bbox_orientation)
    axis = trusted_axis_angle(candidate)
    if axis["angle_deg"] is None:
        return {
            "orientation": bbox_orientation,
            "orientation_source": "bbox_aspect" if bbox_orientation in {"H", "V"} else "unknown",
            "bbox_orientation": bbox_orientation,
            "axis_angle_deg": None,
            "axis_angle_source": "",
            "orientation_conflict": False,
            "consumer_allowed": False,
        }
    orientation = orientation_from_axis_angle(
        axis["angle_deg"],
        vertical_threshold_deg=vertical_threshold_deg,
    )
    return {
        "orientation": orientation,
        "orientation_source": "trusted_axis_angle",
        "bbox_orientation": bbox_orientation,
        "axis_angle_deg": round(float(axis["angle_deg"]), 6),
        "axis_angle_source": axis["source"],
        "orientation_conflict": (
            orientation in {"H", "V"}
            and bbox_orientation in {"H", "V"}
            and orientation != bbox_orientation
        ),
        "consumer_allowed": False,
    }


def trusted_axis_angle(candidate: dict[str, Any]) -> dict[str, Any]:
    for key in ("axis_angle_deg", "dimension_axis_angle_deg"):
        angle = _normalized_axis_angle(candidate.get(key))
        if angle is not None:
            return {"angle_deg": angle, "source": key}

    oriented_quad = candidate.get("oriented_quad")
    if isinstance(oriented_quad, dict):
        angle = _normalized_axis_angle(oriented_quad.get("angle_deg"))
        source = str(oriented_quad.get("source") or "")
        source_key = f"oriented_quad:{source}" if source else "oriented_quad"
        if angle is not None and source_key in TRUSTED_AXIS_SOURCES:
            return {"angle_deg": angle, "source": source_key}

    checks = ((candidate.get("evidence_gate") or {}).get("checks") or {})
    if isinstance(checks, dict):
        for key in ("dot_axis_angle_deg", "axis_angle_deg", "dimension_axis_angle_deg"):
            angle = _normalized_axis_angle(checks.get(key))
            if angle is None:
                continue
            source = str(checks.get("dot_axis_angle_source") or key)
            if source in TRUSTED_AXIS_SOURCES or key in TRUSTED_AXIS_SOURCES:
                return {"angle_deg": angle, "source": source}

    axis = candidate.get("axis")
    if isinstance(axis, dict):
        angle = _normalized_axis_angle(axis.get("angle_deg"))
        source = str(axis.get("source") or "")
        if angle is not None and source in TRUSTED_AXIS_SOURCES and source not in FALLBACK_AXIS_SOURCES:
            return {"angle_deg": angle, "source": source}

    for node in candidate.get("source_nodes") or []:
        if not isinstance(node, dict):
            continue
        axis = trusted_axis_angle({key: value for key, value in node.items() if key != "source_nodes"})
        if axis["angle_deg"] is not None:
            return axis

    return {"angle_deg": None, "source": ""}


def orientation_from_axis_angle(angle_deg: float, *, vertical_threshold_deg: float = 45.0) -> str:
    angle = _normalized_axis_angle(angle_deg)
    if angle is None:
        return "unknown"
    return "V" if abs(angle - 90.0) <= float(vertical_threshold_deg) else "H"


def _normalized_axis_angle(value: Any) -> float | None:
    try:
        angle = float(value) % 180.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(angle):
        return None
    return angle


def _normalize_orientation(value: Any) -> str:
    raw = str(value or "").upper()
    return raw if raw in {"H", "V"} else "unknown"
