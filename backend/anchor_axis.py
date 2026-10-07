"""Path B anchor-derived text-run axis dump.

This module is diagnostic-only. It orders vector glyph evidence around
dimension candidates and records anchor-derived text-run axes without changing
candidate gating, OCR, assembler decisions, or final dimensions.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import Any

from anchor_orientation import resolve_candidate_orientation


SCHEMA_VERSION = "vector_anchor_axis_v1"
STATS_SCHEMA_VERSION = "vector_anchor_axis_stats_v1"
CONSUMER_ALLOWED = False

AXIS_PRIORITY = (
    "dot_pair",
    "dot_to_digit_pca",
    "degree_terminus",
    "digit_centroid_pca",
    "candidate_bbox",
)

GLYPH_ANCHOR_TYPES = {
    "dot": "decimal_point",
    "plus_minus": "plus_minus",
    "diameter": "diameter",
    "degree": "degree",
}

PREFIX_ANCHOR_TYPES = {
    "R": "R_prefix",
    "SR": "SR_prefix",
}


def build_vector_anchor_axis_v1(
    *,
    case_id: str = "",
    page_index: int = 0,
    dimension_candidates: list[dict[str, Any]] | None = None,
    anchor_corridor_candidates: list[dict[str, Any]] | None = None,
    anchor_phrase_candidates: list[dict[str, Any]] | None = None,
    vector_glyph_tokens: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = _dedupe_candidates([
        *(dimension_candidates or []),
        *(anchor_corridor_candidates or []),
    ])
    glyph_tokens = list(vector_glyph_tokens or [])
    rows = [
        _row_from_candidate(
            candidate,
            candidate_index=index,
            case_id=case_id,
            page_index=page_index,
            vector_glyph_tokens=glyph_tokens,
            anchor_corridor_candidates=anchor_corridor_candidates or [],
            anchor_phrase_candidates=anchor_phrase_candidates or [],
        )
        for index, candidate in enumerate(candidates)
        if _candidate_bbox(candidate) is not None
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_anchor_axis_v1": rows,
        "vector_anchor_axis_stats_v1": _build_stats(rows),
    }


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped = []
    for index, candidate in enumerate(candidates):
        bbox = _candidate_bbox(candidate)
        key = (
            str(candidate.get("candidate_id") or ""),
            tuple(_round_bbox(bbox).items()) if bbox else index,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _row_from_candidate(
    candidate: dict[str, Any],
    *,
    candidate_index: int,
    case_id: str,
    page_index: int,
    vector_glyph_tokens: list[dict[str, Any]],
    anchor_corridor_candidates: list[dict[str, Any]],
    anchor_phrase_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_bbox = _candidate_bbox(candidate) or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    candidate_id = _candidate_id(candidate, candidate_index, page_index)
    orientation = _candidate_orientation(candidate, raw_bbox)
    warnings: list[str] = []

    glyph_members = _glyph_members_for_candidate(vector_glyph_tokens, raw_bbox, page_index)
    digit_members = [
        member for member in glyph_members
        if member["kind"] == "digit_unknown"
    ]
    digit_heights = [
        float(member["bbox"]["h"])
        for member in digit_members
        if member.get("bbox")
    ]
    nearby_digit_height = median(digit_heights) if digit_heights else 0.0

    anchors: list[dict[str, Any]] = []
    ordered_items: list[dict[str, Any]] = []
    ordered_items.extend(digit_members)

    prefix_items = _prefix_members_for_candidate(
        candidate,
        candidate_id=candidate_id,
        candidate_index=candidate_index,
        raw_bbox=raw_bbox,
        anchor_corridor_candidates=anchor_corridor_candidates,
        anchor_phrase_candidates=anchor_phrase_candidates,
    )
    ordered_items.extend(prefix_items)
    anchors.extend(_anchor_from_member(member) for member in prefix_items)
    if len(prefix_items) >= 2:
        warnings.append("multi_prefix_unsplit")

    for member in glyph_members:
        kind = str(member.get("kind") or "")
        if kind == "digit_unknown":
            continue
        anchor_type = kind
        if anchor_type == "decimal_point" and not _dot_has_digit_neighbor(
            member,
            digit_members,
        ):
            anchor_type = "isolated_dot"
            warnings.append("isolated_dot_no_digit_neighbor")
        if anchor_type == "degree" and not _degree_size_ok(member, digit_members):
            warnings.append("degree_diameter_too_large_for_aspect")
        item = dict(member)
        item["kind"] = anchor_type
        ordered_items.append(item)
        anchors.append(_anchor_from_member(item))

    if sum(1 for anchor in anchors if anchor["type"] == "decimal_point") >= 2:
        warnings.append("multi_anchor_unsplit")

    axis = _choose_axis(
        anchors=anchors,
        digit_members=digit_members,
        raw_bbox=raw_bbox,
        orientation=orientation,
    )
    ordered_members = _ordered_members(ordered_items, axis["angle_deg"], raw_bbox)
    prefix_tokens = [
        "SR" if item["kind"] == "SR_prefix" else "R"
        for item in ordered_members
        if item["kind"] in {"R_prefix", "SR_prefix"}
    ]
    suffix_tokens = [
        "degree"
        for item in ordered_members
        if item["kind"] == "degree"
    ]
    tolerance_split = _tolerance_split(ordered_members)

    warnings = _unique(warnings)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case_id or ""),
        "candidate_id": candidate_id,
        "page_index": int(page_index),
        "raw_bbox": _round_bbox(raw_bbox),
        "candidate_orientation": orientation,
        "anchors": anchors,
        "anchor_count": len(anchors),
        "axis": axis,
        "ordered_members": ordered_members,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "tolerance_split": tolerance_split,
        "warnings": warnings,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _glyph_members_for_candidate(
    tokens: list[dict[str, Any]],
    raw_bbox: dict[str, float],
    page_index: int,
) -> list[dict[str, Any]]:
    members = []
    for index, token in enumerate(tokens):
        if int(token.get("page_index", page_index)) != int(page_index):
            continue
        bbox = _bbox_from_any(token)
        if not bbox:
            continue
        centroid = _bbox_center(bbox)
        if not _point_in_bbox(centroid, raw_bbox, pad=2.0):
            continue
        glyph_type = str(token.get("glyph_type") or "")
        if glyph_type in {"digit_unknown", "digit"}:
            kind = "digit_unknown"
        elif glyph_type in GLYPH_ANCHOR_TYPES:
            kind = GLYPH_ANCHOR_TYPES[glyph_type]
        else:
            continue
        members.append({
            "component_id": _component_id(token, index),
            "kind": kind,
            "centroid": centroid,
            "bbox": bbox,
        })
    return members


def _prefix_members_for_candidate(
    candidate: dict[str, Any],
    *,
    candidate_id: str,
    candidate_index: int,
    raw_bbox: dict[str, float],
    anchor_corridor_candidates: list[dict[str, Any]],
    anchor_phrase_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for token_index, token in enumerate(_iter_text_tokens(candidate)):
        text = _prefix_text(token.get("text"))
        if not text:
            continue
        bbox = _bbox_from_any(token) or _prefix_bbox_from_candidate(raw_bbox, text)
        members.append(_prefix_member(
            text,
            bbox,
            component_id=str(token.get("component_id") or token.get("token_id") or (
                f"{candidate_id}_prefix_{token_index:02d}"
            )),
        ))

    if members:
        return members

    text_run_prefix = _prefix_from_text_run(_candidate_text(candidate))
    if text_run_prefix:
        members.append(_prefix_member(
            text_run_prefix,
            _prefix_bbox_from_candidate(raw_bbox, text_run_prefix),
            component_id=f"{candidate_id}_prefix_text",
        ))
        return members

    source_rows = [
        row for row in [*anchor_corridor_candidates, *anchor_phrase_candidates]
        if _row_refs_candidate(row, candidate_id, candidate_index)
        or _bbox_inside(_bbox_from_any(row.get("corridor_bbox") or row), raw_bbox)
    ]
    for row_index, row in enumerate(source_rows):
        text = _prefix_text(_row_text(row))
        if not text:
            text = _prefix_from_text_run(_candidate_text(candidate))
        if not text:
            continue
        bbox = _bbox_from_any(row.get("anchor_bbox") or row.get("corridor_bbox") or row)
        bbox = bbox or _prefix_bbox_from_candidate(raw_bbox, text)
        members.append(_prefix_member(
            text,
            bbox,
            component_id=str(row.get("candidate_id") or f"{candidate_id}_prefix_row_{row_index:02d}"),
        ))
    return _dedupe_prefix_members(members)


def _iter_text_tokens(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = []
    for key in ("tokens", "text_tokens", "phrase_tokens", "chars"):
        value = candidate.get(key)
        if isinstance(value, list):
            tokens.extend(item for item in value if isinstance(item, dict))
    return tokens


def _prefix_text(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in PREFIX_ANCHOR_TYPES:
        return text
    return None


def _prefix_from_text_run(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    compact = "".join(text.split())
    if compact.startswith("SR") and len(compact) > 2:
        if compact[2].isdigit() or compact[2] in ".+-":
            return "SR"
    if compact.startswith("R") and len(compact) > 1:
        if compact[1].isdigit() or compact[1] in ".+-":
            return "R"
    return _prefix_text(text)


def _candidate_text(candidate: dict[str, Any]) -> str:
    for key in ("text", "raw_text", "normalized_text", "ocr_text", "phrase", "label"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _row_text(row: dict[str, Any]) -> str:
    for key in ("text", "raw_text", "normalized_text", "ocr_text", "phrase", "label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    checks = (row.get("evidence_gate") or {}).get("checks") or {}
    for key in ("text", "raw_text", "normalized_text", "ocr_text", "phrase"):
        value = checks.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _row_refs_candidate(row: dict[str, Any], candidate_id: str, candidate_index: int) -> bool:
    for node in row.get("source_nodes") or []:
        if not isinstance(node, dict):
            continue
        if str(node.get("candidate_id") or "") == candidate_id:
            return True
        try:
            if int(node.get("index")) == int(candidate_index) and node.get("kind") == "dimension_candidate":
                return True
        except (TypeError, ValueError):
            pass
    return False


def _prefix_bbox_from_candidate(raw_bbox: dict[str, float], text: str) -> dict[str, float]:
    width_ratio = 0.20 if text == "R" else 0.28
    return {
        "x": raw_bbox["x"],
        "y": raw_bbox["y"],
        "w": max(1.0, raw_bbox["w"] * width_ratio),
        "h": raw_bbox["h"],
    }


def _prefix_member(text: str, bbox: dict[str, float], *, component_id: str) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "kind": PREFIX_ANCHOR_TYPES[text],
        "centroid": _bbox_center(bbox),
        "bbox": bbox,
    }


def _dedupe_prefix_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for member in members:
        key = (member["kind"], tuple(_round_point(member["centroid"])))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(member)
    return deduped


def _anchor_from_member(member: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(member.get("kind") or ""),
        "component_id": str(member.get("component_id") or ""),
        "centroid": _round_point(member.get("centroid") or (0.0, 0.0)),
    }


def _dot_has_digit_neighbor(
    dot_member: dict[str, Any],
    digit_members: list[dict[str, Any]],
) -> bool:
    if not digit_members:
        return False
    dot_centroid = dot_member["centroid"]
    nearest = min(
        digit_members,
        key=lambda member: _distance(dot_centroid, member["centroid"]),
    )
    digit_height = float((nearest.get("bbox") or {}).get("h") or 0.0)
    if digit_height <= 0.0:
        return False
    return _distance(dot_centroid, nearest["centroid"]) <= 2.0 * digit_height


def _degree_size_ok(
    degree_member: dict[str, Any],
    digit_members: list[dict[str, Any]],
) -> bool:
    if not digit_members:
        return False
    degree_centroid = degree_member["centroid"]
    nearest = min(
        digit_members,
        key=lambda member: _distance(degree_centroid, member["centroid"]),
    )
    digit_height = float((nearest.get("bbox") or {}).get("h") or 0.0)
    bbox = degree_member.get("bbox") or {}
    diameter = max(float(bbox.get("w") or 0.0), float(bbox.get("h") or 0.0))
    return bool(digit_height > 0.0 and diameter < 0.5 * digit_height)


def _choose_axis(
    *,
    anchors: list[dict[str, Any]],
    digit_members: list[dict[str, Any]],
    raw_bbox: dict[str, float],
    orientation: str,
) -> dict[str, Any]:
    fallback_chain: list[str] = []
    valid_dots = [
        anchor for anchor in anchors
        if anchor["type"] == "decimal_point"
    ]
    degrees = [
        anchor for anchor in anchors
        if anchor["type"] == "degree"
    ]
    digit_points = [member["centroid"] for member in digit_members]

    fallback_chain.append("dot_pair")
    if len(valid_dots) >= 2:
        angle = _pca_angle([tuple(anchor["centroid"]) for anchor in valid_dots])
        return _axis_result("dot_pair", angle, 0.96, fallback_chain)

    fallback_chain.append("dot_to_digit_pca")
    if valid_dots and len(digit_points) >= 2:
        points = [*digit_points, *[tuple(anchor["centroid"]) for anchor in valid_dots]]
        angle = _pca_angle(points)
        return _axis_result("dot_to_digit_pca", angle, 0.88, fallback_chain)

    fallback_chain.append("degree_terminus")
    if degrees and digit_points:
        points = [*digit_points, *[tuple(anchor["centroid"]) for anchor in degrees]]
        angle = _pca_angle(points) if len(points) >= 2 else _bbox_axis(raw_bbox, orientation)
        return _axis_result("degree_terminus", angle, 0.82, fallback_chain)

    fallback_chain.append("digit_centroid_pca")
    if len(digit_points) >= 2:
        return _axis_result("digit_centroid_pca", _pca_angle(digit_points), 0.62, fallback_chain)

    fallback_chain.append("candidate_bbox")
    return _axis_result("candidate_bbox", _bbox_axis(raw_bbox, orientation), 0.35, fallback_chain)


def _axis_result(
    source: str,
    angle_deg: float,
    confidence: float,
    fallback_chain: list[str],
) -> dict[str, Any]:
    return {
        "source": source,
        "angle_deg": round(_normalize_angle(angle_deg), 3),
        "confidence": round(float(confidence), 3),
        "fallback_chain": list(fallback_chain),
    }


def _ordered_members(
    members: list[dict[str, Any]],
    angle_deg: float,
    raw_bbox: dict[str, float],
) -> list[dict[str, Any]]:
    axis_min, axis_max, perp_min, perp_max = _projected_bbox_extents(raw_bbox, angle_deg)
    axis_span = max(axis_max - axis_min, 1e-6)
    perp_span = max(perp_max - perp_min, 1e-6)
    projected = []
    for member in members:
        local_x_raw, local_y_raw = _project_point(member["centroid"], angle_deg)
        local_x = (local_x_raw - axis_min) / axis_span
        local_y = (local_y_raw - perp_min) / perp_span
        projected.append((
            local_x,
            local_y,
            {
                "component_id": str(member.get("component_id") or ""),
                "kind": str(member.get("kind") or ""),
                "local_x": round(local_x, 6),
                "local_y": round(local_y, 6),
            },
        ))
    projected.sort(key=lambda item: (item[0], item[1], item[2]["component_id"]))
    return [item[2] for item in projected]


def _tolerance_split(ordered_members: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, member in enumerate(ordered_members):
        if member.get("kind") == "plus_minus":
            return {
                "split_index": index + 1,
                "anchor": "plus_minus",
            }
    return None


def _build_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    axis_source_counts = Counter(str(row.get("axis", {}).get("source") or "unknown") for row in rows)
    warning_counts = Counter(
        warning for row in rows for warning in row.get("warnings") or []
    )
    prefix_token_counts = Counter(
        token for row in rows for token in row.get("prefix_tokens") or []
    )
    suffix_token_counts = Counter(
        token for row in rows for token in row.get("suffix_tokens") or []
    )
    anchor_type_candidate_counts: Counter[str] = Counter()
    for row in rows:
        for anchor_type in {
            str(anchor.get("type") or "unknown")
            for anchor in row.get("anchors") or []
        }:
            anchor_type_candidate_counts[anchor_type] += 1

    multi_row_count = 0
    for row in rows:
        raw_bbox = _bbox_from_any(row.get("raw_bbox"))
        digit_height = _median_digit_member_height(row)
        if raw_bbox and digit_height > 0.0 and raw_bbox["h"] > 1.5 * digit_height:
            multi_row_count += 1

    integer_no_anchor_count = sum(
        1 for row in rows
        if str(row.get("axis", {}).get("source") or "") in {
            "digit_centroid_pca",
            "candidate_bbox",
        }
    )
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "consumer_allowed": CONSUMER_ALLOWED,
        "total_candidates": total,
        "anchor_type_coverage": {
            anchor_type: {
                "candidate_count": count,
                "total_candidates": total,
                "ratio": round(count / total, 6) if total else 0.0,
            }
            for anchor_type, count in sorted(anchor_type_candidate_counts.items())
        },
        "axis_source_counts": dict(sorted(axis_source_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "prefix_token_counts": dict(sorted(prefix_token_counts.items())),
        "suffix_token_counts": dict(sorted(suffix_token_counts.items())),
        "multi_row_bbox_count": multi_row_count,
        "multi_row_bbox_ratio": round(multi_row_count / total, 6) if total else 0.0,
        "integer_no_anchor_candidate_count": integer_no_anchor_count,
        "integer_no_anchor_candidate_ratio": (
            round(integer_no_anchor_count / total, 6) if total else 0.0
        ),
    }


def _median_digit_member_height(row: dict[str, Any]) -> float:
    heights = []
    for member in row.get("ordered_members") or []:
        if member.get("kind") == "digit_unknown":
            heights.append(1.0)
    raw_bbox = _bbox_from_any(row.get("raw_bbox"))
    if not heights or not raw_bbox:
        return 0.0
    return raw_bbox["h"] / max(len(heights), 1)


def _candidate_bbox(candidate: dict[str, Any]) -> dict[str, float] | None:
    for key in ("raw_bbox", "bbox", "corridor_bbox", "anchor_bbox"):
        bbox = _bbox_from_any(candidate.get(key))
        if bbox:
            return bbox
    return _bbox_from_any(candidate)


def _candidate_id(candidate: dict[str, Any], index: int, page_index: int) -> str:
    value = str(candidate.get("candidate_id") or "").strip()
    if value:
        return value
    return f"p{int(page_index) + 1:03d}_anchor_axis_{int(index):06d}"


def _candidate_orientation(candidate: dict[str, Any], bbox: dict[str, float]) -> str:
    bbox_orientation = "H" if bbox["w"] >= bbox["h"] else "V"
    orientation_decision = resolve_candidate_orientation(
        candidate,
        bbox_orientation=bbox_orientation,
    )
    if str(orientation_decision.get("orientation_source") or "") == "trusted_axis_angle":
        return str(orientation_decision["orientation"])
    raw = str(candidate.get("candidate_orientation") or candidate.get("orientation") or "").upper()
    if raw in {"H", "V"}:
        return raw
    for node in candidate.get("source_nodes") or []:
        if not isinstance(node, dict):
            continue
        raw = str(node.get("orientation") or "").upper()
        if raw in {"H", "V"}:
            return raw
    return bbox_orientation


def _component_id(token: dict[str, Any], index: int) -> str:
    for key in ("component_id", "token_id", "candidate_id", "id"):
        value = token.get(key)
        if value is not None:
            return str(value)
    return f"glyph_{index:06d}"


def _bbox_from_any(item: Any) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("bbox"), dict):
        return _bbox_from_any(item["bbox"])
    try:
        if all(key in item for key in ("x", "y", "w", "h")):
            x = float(item["x"])
            y = float(item["y"])
            w = float(item["w"])
            h = float(item["h"])
            if w <= 0.0 or h <= 0.0:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
        if all(key in item for key in ("x0", "y0", "x1", "y1")):
            x0 = float(item["x0"])
            y0 = float(item["y0"])
            x1 = float(item["x1"])
            y1 = float(item["y1"])
            return _bbox_from_xyxy(x0, y0, x1, y1)
        if all(key in item for key in ("x0", "top", "x1", "bottom")):
            x0 = float(item["x0"])
            y0 = float(item["top"])
            x1 = float(item["x1"])
            y1 = float(item["bottom"])
            return _bbox_from_xyxy(x0, y0, x1, y1)
    except (TypeError, ValueError):
        return None
    return None


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> dict[str, float] | None:
    x_min = min(x0, x1)
    y_min = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x_min, "y": y_min, "w": w, "h": h}


def _bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return (
        float(bbox["x"]) + float(bbox["w"]) / 2.0,
        float(bbox["y"]) + float(bbox["h"]) / 2.0,
    )


def _point_in_bbox(
    point: tuple[float, float] | list[float],
    bbox: dict[str, float],
    *,
    pad: float = 0.0,
) -> bool:
    x, y = float(point[0]), float(point[1])
    return (
        bbox["x"] - pad <= x <= bbox["x"] + bbox["w"] + pad
        and bbox["y"] - pad <= y <= bbox["y"] + bbox["h"] + pad
    )


def _bbox_inside(inner: dict[str, float] | None, outer: dict[str, float]) -> bool:
    if not inner:
        return False
    center = _bbox_center(inner)
    return _point_in_bbox(center, outer, pad=2.0)


def _round_bbox(bbox: dict[str, float] | None) -> dict[str, float]:
    if not bbox:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    return {
        "x": round(float(bbox["x"]), 4),
        "y": round(float(bbox["y"]), 4),
        "w": round(float(bbox["w"]), 4),
        "h": round(float(bbox["h"]), 4),
    }


def _round_point(point: Any) -> list[float]:
    return [round(float(point[0]), 4), round(float(point[1]), 4)]


def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _distance(a: tuple[float, float] | list[float], b: tuple[float, float] | list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _pca_angle(points: list[tuple[float, float] | list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0.0 and syy == 0.0:
        return 0.0
    return math.degrees(0.5 * math.atan2(2.0 * sxy, sxx - syy))


def _bbox_axis(raw_bbox: dict[str, float], orientation: str) -> float:
    if orientation == "V":
        return 90.0
    if raw_bbox["h"] > raw_bbox["w"] * 1.2:
        return 90.0
    return 0.0


def _normalize_angle(angle_deg: float) -> float:
    angle = float(angle_deg) % 180.0
    if angle < 0.0:
        angle += 180.0
    return angle


def _project_point(point: tuple[float, float] | list[float], angle_deg: float) -> tuple[float, float]:
    angle = math.radians(_normalize_angle(angle_deg))
    ux = math.cos(angle)
    uy = math.sin(angle)
    px = -uy
    py = ux
    x = float(point[0])
    y = float(point[1])
    return (x * ux + y * uy, x * px + y * py)


def _projected_bbox_extents(
    bbox: dict[str, float],
    angle_deg: float,
) -> tuple[float, float, float, float]:
    corners = [
        (bbox["x"], bbox["y"]),
        (bbox["x"] + bbox["w"], bbox["y"]),
        (bbox["x"], bbox["y"] + bbox["h"]),
        (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
    ]
    projected = [_project_point(point, angle_deg) for point in corners]
    axis_values = [point[0] for point in projected]
    perp_values = [point[1] for point in projected]
    return (
        min(axis_values),
        max(axis_values),
        min(perp_values),
        max(perp_values),
    )
