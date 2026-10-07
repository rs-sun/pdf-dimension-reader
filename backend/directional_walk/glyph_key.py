"""Pure vector grouping and S001 + free-endpoint glyph evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
import json
import math
import struct
from typing import Any


ALLOWED_OPS = frozenset({"l", "c", "qu"})
FLOAT_TOLERANCE = 1e-12
CHAR_QUANTUM = Decimal("0.01")
STEP_QUANTUM = Decimal("0.1")
FREE_ENDPOINT_EPSILON_RATIO = 0.01
GROUP_LINK_RATIO = 0.08
SHAPE_KEY_SIZE_TOLERANCE_PT = 0.015


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decimal_text(value: float, quantum: Decimal) -> str:
    quantized = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    places = max(0, -quantum.as_tuple().exponent)
    return f"{quantized:.{places}f}"


@lru_cache(maxsize=4096)
def _axis_from_binary64(angle_binary64: bytes) -> tuple[float, float]:
    """Return the original trigonometric result for one exact float value."""

    angle = struct.unpack("!d", angle_binary64)[0]
    radians = math.radians(angle)
    return math.cos(radians), math.sin(radians)


def _axis(angle_deg: float) -> tuple[float, float]:
    angle = float(angle_deg)
    if not math.isfinite(angle):
        raise ValueError("axis angle must be finite")
    # Float equality conflates +0.0 and -0.0.  Keying by the IEEE-754 payload
    # preserves the pre-cache bit pattern for every finite input.
    return _axis_from_binary64(struct.pack("!d", angle))


def project_point(point: Sequence[float], angle_deg: float) -> tuple[float, float]:
    dx, dy = _axis(angle_deg)
    x, y = float(point[0]), float(point[1])
    return x * dx + y * dy, -x * dy + y * dx


def unproject_point(point: Sequence[float], angle_deg: float) -> tuple[float, float]:
    dx, dy = _axis(angle_deg)
    u, v = float(point[0]), float(point[1])
    return u * dx - v * dy, u * dy + v * dx


def bbox_union(bboxes: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    rows = [tuple(float(value) for value in bbox) for bbox in bboxes]
    if not rows:
        raise ValueError("bbox union requires at least one bbox")
    return (
        min(row[0] for row in rows),
        min(row[1] for row in rows),
        max(row[2] for row in rows),
        max(row[3] for row in rows),
    )


def bbox_xywh(bbox: Sequence[float]) -> dict[str, float]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _projected_bbox(points: Sequence[Sequence[float]], angle_deg: float) -> tuple[float, float, float, float]:
    projected = [project_point(point, angle_deg) for point in points]
    return (
        min(point[0] for point in projected),
        min(point[1] for point in projected),
        max(point[0] for point in projected),
        max(point[1] for point in projected),
    )


def _bbox_gap(left: Sequence[float], right: Sequence[float]) -> float:
    dx = max(float(left[0]) - float(right[2]), float(right[0]) - float(left[2]), 0.0)
    dy = max(float(left[1]) - float(right[3]), float(right[1]) - float(left[3]), 0.0)
    return math.hypot(dx, dy)


def group_primitives(
    primitives: Sequence[Mapping[str, Any]],
    *,
    axis_angle_deg: float,
    reference_scale: float,
    projected_bboxes: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Form deterministic local connected groups using a scale-relative link."""

    scale = float(reference_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("group reference scale must be finite and positive")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in primitives:
        primitive_id = str(raw.get("primitive_id") or "")
        op = str(raw.get("op") or "")
        points = tuple(
            (float(point[0]), float(point[1])) for point in raw.get("points", ())
        )
        if not primitive_id or primitive_id in seen or op not in ALLOWED_OPS or len(points) < 2:
            continue
        seen.add(primitive_id)
        projected_bbox = (
            None
            if projected_bboxes is None
            else projected_bboxes.get(primitive_id)
        )
        rows.append(
            {
                "primitive_id": primitive_id,
                "op": op,
                "points": points,
                "bbox": bbox_union(((point[0], point[1], point[0], point[1]) for point in points)),
                "uv_bbox": (
                    _projected_bbox(points, axis_angle_deg)
                    if projected_bbox is None
                    else tuple(float(value) for value in projected_bbox)
                ),
            }
        )
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        root = min(left_root, right_root)
        parent[max(left_root, right_root)] = root

    maximum_gap = scale * GROUP_LINK_RATIO
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _bbox_gap(rows[left]["uv_bbox"], rows[right]["uv_bbox"]) <= maximum_gap:
                union(left, right)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(find(index), []).append(row)
    result: list[dict[str, Any]] = []
    for members in grouped.values():
        ordered = tuple(sorted(members, key=lambda row: row["primitive_id"]))
        result.append(
            {
                "primitive_ids": tuple(row["primitive_id"] for row in ordered),
                "primitives": ordered,
                "bbox": bbox_union(row["bbox"] for row in ordered),
                "uv_bbox": bbox_union(row["uv_bbox"] for row in ordered),
            }
        )
    return tuple(sorted(result, key=lambda row: row["primitive_ids"]))


def _free_endpoint_count(
    primitives: Sequence[Mapping[str, Any]],
    *,
    height_raw: float,
) -> int:
    endpoints: list[tuple[float, float]] = []
    for primitive in primitives:
        points = primitive["points"]
        endpoints.extend((points[0], points[-1]))
    parent = list(range(len(endpoints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    epsilon = max(0.0, float(height_raw)) * FREE_ENDPOINT_EPSILON_RATIO
    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            if math.dist(endpoints[left], endpoints[right]) <= epsilon:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    root = min(left_root, right_root)
                    parent[max(left_root, right_root)] = root
    degree: Counter[int] = Counter(find(index) for index in range(len(endpoints)))
    return sum(value == 1 for value in degree.values())


def _simple_glyph_evidence(group: Mapping[str, Any]) -> dict[str, Any]:
    primitives = tuple(group.get("primitives", ()))
    if not primitives:
        raise ValueError("glyph group has no primitives")
    u0, v0, u1, v1 = (float(value) for value in group["uv_bbox"])
    width_raw = u1 - u0
    height_raw = v1 - v0
    counts = Counter(str(row["op"]) for row in primitives)
    free_endpoint_count = _free_endpoint_count(primitives, height_raw=height_raw)
    s001 = [
        width_raw,
        height_raw,
        len(primitives),
        counts["l"],
        counts["c"],
        counts["qu"],
    ]
    shape_key_value = [
        _decimal_text(width_raw, CHAR_QUANTUM),
        _decimal_text(height_raw, CHAR_QUANTUM),
        len(primitives),
        counts["l"],
        counts["c"],
        counts["qu"],
        free_endpoint_count,
    ]
    step_key_value = [
        _decimal_text(width_raw, STEP_QUANTUM),
        _decimal_text(height_raw, STEP_QUANTUM),
        len(primitives),
        counts["l"],
        counts["c"],
        counts["qu"],
    ]
    return {
        "shape_key": canonical_json(shape_key_value),
        "step_key": canonical_json(step_key_value),
        "s001": s001,
        "free_endpoint_count": free_endpoint_count,
        "primitive_ids": tuple(group["primitive_ids"]),
        "bbox": tuple(group["bbox"]),
        "uv_bbox": tuple(group["uv_bbox"]),
        "u_center": float(
            group.get("u_center", (float(group["uv_bbox"][0]) + float(group["uv_bbox"][2])) / 2.0)
        ),
        "points": tuple(
            point for primitive in primitives for point in primitive["points"]
        ),
    }


def glyph_evidence(group: Mapping[str, Any]) -> dict[str, Any]:
    components = tuple(group.get("components", ()))
    if not components:
        return _simple_glyph_evidence(group)
    members = tuple(_simple_glyph_evidence(component) for component in components)
    ordered = tuple(sorted(members, key=lambda row: row["shape_key"]))
    shape_key = canonical_json(
        {"compound": [json.loads(row["shape_key"]) for row in ordered]}
    )
    step_key = canonical_json(
        {
            "compound": sorted(
                (json.loads(row["step_key"]) for row in members),
                key=canonical_json,
            )
        }
    )
    primitives = tuple(group.get("primitives", ()))
    if not primitives:
        raise ValueError("compound glyph group has no primitives")
    return {
        "shape_key": shape_key,
        "step_key": step_key,
        "s001": [value for row in ordered for value in row["s001"]],
        "free_endpoint_count": sum(row["free_endpoint_count"] for row in ordered),
        "primitive_ids": tuple(sorted(str(value) for value in group["primitive_ids"])),
        "bbox": tuple(group["bbox"]),
        "uv_bbox": tuple(group["uv_bbox"]),
        "u_center": float(group["u_center"]),
        "points": tuple(
            point for primitive in primitives for point in primitive["points"]
        ),
    }


def _shape_key_dimension(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        dimension = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return dimension if dimension.is_finite() else None


def _simple_shape_keys_match(
    observed: Any,
    candidate: Any,
) -> bool:
    if (
        not isinstance(observed, list)
        or not isinstance(candidate, list)
        or len(observed) != 7
        or len(candidate) != 7
        or any(type(value) is not int for value in observed[2:])
        or any(type(value) is not int for value in candidate[2:])
        or observed[2:] != candidate[2:]
    ):
        return False
    tolerance = Decimal(str(SHAPE_KEY_SIZE_TOLERANCE_PT))
    for index in (0, 1):
        observed_dimension = _shape_key_dimension(observed[index])
        candidate_dimension = _shape_key_dimension(candidate[index])
        if (
            observed_dimension is None
            or candidate_dimension is None
            or abs(observed_dimension - candidate_dimension) > tolerance
        ):
            return False
    return True


def _shape_keys_match(observed: Any, candidate: Any) -> bool:
    if isinstance(observed, list) or isinstance(candidate, list):
        return _simple_shape_keys_match(observed, candidate)
    if not isinstance(observed, Mapping) or not isinstance(candidate, Mapping):
        return False
    if set(observed) != {"compound"} or set(candidate) != {"compound"}:
        return False
    observed_components = observed["compound"]
    candidate_components = candidate["compound"]
    return (
        isinstance(observed_components, list)
        and isinstance(candidate_components, list)
        and len(observed_components) == len(candidate_components)
        and all(
            _simple_shape_keys_match(observed_component, candidate_component)
            for observed_component, candidate_component in zip(
                observed_components,
                candidate_components,
                strict=True,
            )
        )
    )


def _normalized_character_entry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("character table entry must be an object")
    characters = tuple(sorted({str(value) for value in raw.get("characters", ())}))
    entry_id = str(raw.get("entry_id") or "")
    candidate_entry_ids = tuple(
        sorted(
            {
                str(value)
                for value in raw.get("candidate_entry_ids", (entry_id,))
                if str(value)
            }
        )
    )
    return {
        "entry_id": entry_id,
        "candidate_entry_ids": candidate_entry_ids,
        "characters": characters,
        "support_count": int(raw.get("support_count", 0)),
        "width_disambiguated": bool(raw.get("width_disambiguated", False)),
        "width_evidence": raw.get("width_evidence"),
    }


def _matched_entry_set_id(entry_ids: Sequence[str]) -> str:
    return "r92_char_match_set:" + canonical_json(list(entry_ids))


def lookup_character(
    character_table: Mapping[str, Any],
    *,
    shape_key: str,
) -> dict[str, Any] | None:
    entries = character_table.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("character table entries missing")
    try:
        observed_key = json.loads(str(shape_key))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    matches: list[dict[str, Any]] = []
    for candidate_key, raw in entries.items():
        try:
            candidate = json.loads(str(candidate_key))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if _shape_keys_match(observed_key, candidate):
            matches.append(_normalized_character_entry(raw))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    matched_entry_ids = tuple(sorted({match["entry_id"] for match in matches}))
    candidate_entry_ids = tuple(
        sorted(
            {
                entry_id
                for match in matches
                for entry_id in match["candidate_entry_ids"]
            }
        )
    )
    characters = tuple(
        sorted(
            {
                character
                for match in matches
                for character in match["characters"]
            }
        )
    )
    return {
        "entry_id": _matched_entry_set_id(matched_entry_ids),
        "candidate_entry_ids": candidate_entry_ids,
        "characters": characters,
        "support_count": sum(match["support_count"] for match in matches),
        "width_disambiguated": False,
        "width_evidence": None,
    }


__all__ = [
    "bbox_union",
    "bbox_xywh",
    "canonical_json",
    "glyph_evidence",
    "group_primitives",
    "lookup_character",
    "project_point",
    "SHAPE_KEY_SIZE_TOLERANCE_PT",
    "unproject_point",
]
