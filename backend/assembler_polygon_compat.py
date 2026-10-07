"""组装结果的多边形与轴对齐包围框兼容诊断。记录候选几何的表示方式，但不将诊断多边形送入组装器或去重逻辑。"""

from __future__ import annotations

from collections import Counter
from typing import Any


SCHEMA_VERSION = "assembler_polygon_compat_v1"

REQUIRED_ROW_FIELDS = frozenset({
    "schema_version",
    "dimension_id",
    "aabb_bbox",
    "polygon_or_quad",
    "bbox_source",
    "dedup_decision",
    "claim_decision",
    "compat_fallback_reason",
})


def build_assembler_polygon_compat_v1(
    *,
    dimensions: list[dict[str, Any]] | None = None,
    dimension_candidates: list[dict[str, Any]] | None = None,
    candidate_drop_ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for idx, dim in enumerate(dimensions or []):
        rows.append(_row_from_item(
            item=dim,
            row_kind="final_dimension",
            order=idx,
            dimension_id=_item_id(dim, idx, prefix="dim"),
            candidate_id=_optional_id(dim, ("candidate_id", "source_candidate_id")),
            dedup_decision="kept_final_dimension",
            claim_decision="claimed_by_current_assembler",
        ))

    for idx, candidate in enumerate(dimension_candidates or []):
        candidate_id = _item_id(candidate, idx, prefix="candidate")
        rows.append(_row_from_item(
            item=candidate,
            row_kind="dimension_candidate",
            order=idx,
            dimension_id=candidate_id,
            candidate_id=candidate_id,
            dedup_decision="not_consumed_dump_only",
            claim_decision=(
                "candidate_active"
                if not candidate.get("reject_reason")
                else "candidate_rejected"
            ),
        ))

    for idx, drop in enumerate(candidate_drop_ledger or []):
        candidate_id = _item_id(drop, idx, prefix="drop")
        rows.append(_row_from_item(
            item=drop,
            row_kind="candidate_drop",
            order=idx,
            dimension_id=candidate_id,
            candidate_id=candidate_id,
            dedup_decision="dropped_before_assembler",
            claim_decision="dropped",
            fallback_override=str(
                drop.get("reject_reason")
                or drop.get("reason")
                or drop.get("drop_reason")
                or ""
            ),
        ))

    source_counts = Counter(row["bbox_source"] for row in rows)
    fallback_counts = Counter(
        row["compat_fallback_reason"] or "none"
        for row in rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "assembler_polygon_compat_v1": rows,
        "assembler_polygon_compat_stats_v1": {
            "schema_version": SCHEMA_VERSION,
            "row_count": len(rows),
            "final_dimension_count": sum(
                1 for row in rows if row["row_kind"] == "final_dimension"
            ),
            "candidate_count": sum(
                1 for row in rows if row["row_kind"] == "dimension_candidate"
            ),
            "candidate_drop_count": sum(
                1 for row in rows if row["row_kind"] == "candidate_drop"
            ),
            "bbox_source_counts": dict(sorted(source_counts.items())),
            "fallback_counts": dict(sorted(fallback_counts.items())),
            "schema_field_coverage": _schema_field_coverage(rows),
        },
    }


def _row_from_item(
    *,
    item: dict[str, Any],
    row_kind: str,
    order: int,
    dimension_id: str,
    candidate_id: str | None,
    dedup_decision: str,
    claim_decision: str,
    fallback_override: str | None = None,
) -> dict[str, Any]:
    polygon = _polygon_or_quad(item)
    aabb = _bbox_from_any(item)
    fallback_reason = ""
    bbox_source = "aabb"

    if polygon:
        bbox_source = polygon["kind"]
        if not aabb:
            aabb = _bbox_from_points(polygon["points"])
            fallback_reason = "aabb_derived_from_polygon_or_quad"
    else:
        fallback_reason = "missing_polygon_or_quad"

    if not aabb:
        aabb = {"x": None, "y": None, "w": None, "h": None}
        fallback_reason = fallback_reason or "missing_aabb"

    if fallback_override:
        fallback_reason = fallback_override

    return {
        "schema_version": SCHEMA_VERSION,
        "row_kind": row_kind,
        "row_order": int(order),
        "dimension_id": dimension_id,
        "candidate_id": candidate_id,
        "aabb_bbox": _round_bbox(aabb),
        "polygon_or_quad": polygon,
        "bbox_source": bbox_source,
        "dedup_decision": dedup_decision,
        "claim_decision": claim_decision,
        "compat_fallback_reason": fallback_reason,
    }


def _polygon_or_quad(item: dict[str, Any]) -> dict[str, Any] | None:
    for key, kind in (
        ("polygon", "polygon"),
        ("oriented_quad", "oriented_quad"),
        ("quad", "quad"),
    ):
        raw = item.get(key)
        points = _points_from_any(raw)
        if points:
            return {"kind": kind, "points": points}
    return None


def _points_from_any(raw: Any) -> list[list[float]] | None:
    if isinstance(raw, dict):
        raw = raw.get("points")
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    points: list[list[float]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        points.append([round(x, 3), round(y, 3)])
    if _polygon_area(points) <= 1e-6:
        return None
    return points


def _polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) / 2.0


def _bbox_from_any(item: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(item, dict):
        return None
    nested = item.get("bbox")
    if isinstance(nested, dict):
        return _bbox_from_any(nested)
    try:
        if all(key in item for key in ("x", "y", "w", "h")):
            x = float(item.get("x") or 0.0)
            y = float(item.get("y") or 0.0)
            w = float(item.get("w") or 0.0)
            h = float(item.get("h") or 0.0)
            if w <= 0.0 or h <= 0.0:
                return None
            return {"x": x, "y": y, "w": w, "h": h}
        if all(key in item for key in ("x0", "y0", "x1", "y1")):
            x0 = float(item.get("x0"))
            y0 = float(item.get("y0"))
            x1 = float(item.get("x1"))
            y1 = float(item.get("y1"))
            return _bbox_from_xyxy(x0, y0, x1, y1)
    except (TypeError, ValueError):
        return None
    return None


def _bbox_from_points(points: list[list[float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return _bbox_from_xyxy(min(xs), min(ys), max(xs), max(ys))


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> dict[str, float] | None:
    x = min(x0, x1)
    y = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _round_bbox(bbox: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("x", "y", "w", "h"):
        value = bbox.get(key)
        out[key] = round(float(value), 3) if value is not None else None
    return out


def _item_id(item: dict[str, Any], idx: int, *, prefix: str) -> str:
    value = _optional_id(
        item,
        ("dimension_id", "dim_id", "id", "candidate_id", "source_candidate_id"),
    )
    if value:
        return value
    return f"{prefix}_{idx:06d}"


def _optional_id(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _schema_field_coverage(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    present = 0
    total = len(rows) * len(REQUIRED_ROW_FIELDS)
    for row in rows:
        present += sum(1 for key in REQUIRED_ROW_FIELDS if key in row)
    return round(present / total, 6) if total else 1.0
