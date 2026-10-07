"""R3 shadow evidence/hypothesis dump.

This module is dump-only. It normalizes existing pipeline evidence into a
ledger that can be reviewed without changing assembler behavior or API output.
See docs/restart_plans/R3.md for reason-code semantics and stage gates.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from candidate_builder import normalize_bbox


SCHEMA_VERSION = "evidence_shadow_v1"
PIPELINE_VERSION = "evidence_shadow_v0.1.0"
EVIDENCE_LINK_IOU_THRESHOLD = 0.30
SHORT_TEXT_LINK_IOU_THRESHOLD = 0.30
PARTIAL_TEXT_LINK_IOU_THRESHOLD = 0.05
PARTIAL_TEXT_LINK_EXPANSION_RATIO = 0.20
PARTIAL_TEXT_LINK_MIN_SIGNIFICANT_CHARS = 3
PARTIAL_SHORT_TEXT_LINK_IOU_THRESHOLD = 0.50
PARTIAL_ANGLE_TEXT_LINK_IOU_THRESHOLD = 0.20
CLAIM_REASON_CODES = frozenset({
    "claimed_by_geometry",
    "claimed_by_text",
    "claimed_by_assembler_output",
    "no_supporting_evidence",
})


_REQUIRED_EVIDENCE_FIELDS = {
    "schema_version",
    "evidence_id",
    "kind",
    "bbox",
    "quad",
    "text",
    "value",
    "confidence",
    "source_stage",
    "source_index",
    "raw_ref",
}
_REQUIRED_HYPOTHESIS_FIELDS = {
    "schema_version",
    "hypothesis_id",
    "final_dimension_id",
    "claim",
    "supporting_evidence_ids",
    "drop_ledger",
    "dedup_ledger",
}


def build_evidence_shadow(
    *,
    dimensions: list[dict[str, Any]] | None = None,
    dimension_candidates: list[dict[str, Any]] | None = None,
    candidate_drop_ledger: list[dict[str, Any]] | None = None,
    ocr_results: list[dict[str, Any]] | None = None,
    diameter_glyphs: list[dict[str, Any]] | None = None,
    capsule_candidates: list[dict[str, Any]] | None = None,
    gdt_frames_all: list[dict[str, Any]] | None = None,
    detected_dim_lines: list[dict[str, Any]] | None = None,
    detected_leader_lines: list[dict[str, Any]] | None = None,
    assembler_debug: dict[str, Any] | None = None,
    post_filter_drop_ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic shadow dump from existing pipeline artifacts."""
    evidence: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    def add_evidence(
        *,
        kind: str,
        source_stage: str,
        item: dict[str, Any] | None,
        source_index: int,
        text: str | None = None,
        value: Any = None,
        confidence: Any = None,
        quad: Any = None,
        raw_ref: dict[str, Any] | None = None,
    ) -> str:
        counters[kind] += 1
        evidence_id = f"ev_{_slug(kind)}_{counters[kind]:06d}"
        item = item or {}
        bbox = _best_bbox(item)
        record = {
            "schema_version": SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "kind": kind,
            "bbox": _round_bbox(bbox),
            "quad": _normalize_quad(quad if quad is not None else item.get("oriented_quad")),
            "text": _clean_text(text if text is not None else _item_text(item)),
            "value": value if value is not None else _item_value(item),
            "confidence": _safe_float(
                confidence if confidence is not None else item.get("confidence")
            ),
            "source_stage": source_stage,
            "source_index": source_index,
            "raw_ref": raw_ref or _raw_ref(item),
        }
        evidence.append(record)
        return evidence_id

    for idx, cand in enumerate(dimension_candidates or []):
        add_evidence(
            kind="candidate",
            source_stage="dimension_candidate",
            item=cand,
            source_index=idx,
            raw_ref={
                "candidate_id": cand.get("candidate_id"),
                "type": cand.get("type"),
                "subtype": cand.get("subtype"),
                "priority_hint": cand.get("priority_hint"),
            },
        )

    for idx, row in enumerate(ocr_results or []):
        add_evidence(
            kind="ocr",
            source_stage=str(row.get("source") or row.get("pass") or "ocr"),
            item=row,
            source_index=idx,
            text=row.get("text"),
            confidence=row.get("confidence"),
        )

    for idx, glyph in enumerate(diameter_glyphs or []):
        add_evidence(
            kind="vector_token",
            source_stage="diameter_glyph",
            item=glyph,
            source_index=idx,
            text=glyph.get("text") or glyph.get("symbol") or "diameter",
            confidence=glyph.get("confidence"),
        )

    for idx, capsule in enumerate(capsule_candidates or []):
        add_evidence(
            kind="geometric_anchor",
            source_stage="capsule_candidate",
            item=capsule,
            source_index=idx,
        )

    for idx, frame in enumerate(gdt_frames_all or []):
        add_evidence(
            kind="geometric_anchor",
            source_stage="gdt_frame",
            item=frame,
            source_index=idx,
            text=frame.get("text"),
            confidence=frame.get("confidence"),
        )

    for idx, line in enumerate(detected_dim_lines or []):
        add_evidence(
            kind="geometric_anchor",
            source_stage="dimension_line",
            item=line,
            source_index=idx,
            confidence=line.get("confidence"),
        )

    for idx, line in enumerate(detected_leader_lines or []):
        add_evidence(
            kind="geometric_anchor",
            source_stage="leader_line",
            item=line,
            source_index=idx,
            confidence=line.get("confidence") or line.get("arrow_alignment"),
        )

    drop_entries = _normalize_drop_entries(
        candidate_drop_ledger=candidate_drop_ledger,
        post_filter_drop_ledger=(
            post_filter_drop_ledger
            if post_filter_drop_ledger is not None
            else (assembler_debug or {}).get("post_filter_drop_ledger", [])
        ),
    )

    final_dimension_evidence: list[str] = []
    for idx, dim in enumerate(dimensions or []):
        final_dimension_evidence.append(add_evidence(
            kind="assembler_output",
            source_stage="final_dimension",
            item=dim,
            source_index=idx,
            text=dim.get("text") or dim.get("nominal"),
            confidence=dim.get("confidence"),
            raw_ref={
                "source": dim.get("source"),
                "type": dim.get("type"),
                "nominal": dim.get("nominal"),
            },
        ))

    hypotheses = _build_hypotheses(
        dimensions=dimensions or [],
        evidence=evidence,
        final_dimension_evidence=final_dimension_evidence,
        drop_entries=drop_entries,
    )

    stats = _build_stats(evidence, hypotheses, drop_entries)
    return {
        "evidence_store_v1": evidence,
        "hypothesis_dump_v1": hypotheses,
        "evidence_shadow_stats": stats,
    }


def _build_hypotheses(
    *,
    dimensions: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    final_dimension_evidence: list[str],
    drop_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hypotheses = []
    for idx, dim in enumerate(dimensions):
        dim_bbox = _best_bbox(dim)
        linked_ids = _link_evidence(dim, dim_bbox, evidence)
        if idx < len(final_dimension_evidence):
            linked_ids.append(final_dimension_evidence[idx])
        linked_ids = _dedupe_keep_order(linked_ids)
        reason_code = _claim_reason_code(linked_ids, evidence)
        final_dimension_id = (
            str(dim.get("dimension_id") or dim.get("id") or f"dim_{idx + 1:06d}")
        )
        claim = {
            "reason_code": reason_code,
            "source_stage": "assembler",
            "action": "claimed",
            "text": _clean_text(dim.get("text") or dim.get("nominal")),
            "bbox": _round_bbox(dim_bbox),
            "evidence_ids": linked_ids,
        }
        matched_drops = _attach_evidence_ids(
            _matching_drop_entries(dim_bbox, drop_entries),
            linked_ids,
        )
        hypotheses.append({
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": f"hyp_{idx + 1:06d}",
            "final_dimension_id": final_dimension_id,
            "claim": claim,
            "supporting_evidence_ids": linked_ids,
            "drop_ledger": matched_drops or [{
                "reason_code": reason_code,
                "source_stage": "assembler",
                "action": "kept",
                "evidence_ids": linked_ids,
            }],
            "dedup_ledger": [{
                "reason_code": "dedup_keep",
                "source_stage": "assembler_dedup",
                "action": "kept",
                "evidence_ids": linked_ids,
            }],
        })
    return hypotheses


def _normalize_drop_entries(
    *,
    candidate_drop_ledger: list[dict[str, Any]] | None,
    post_filter_drop_ledger: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    entries = []
    for idx, row in enumerate(candidate_drop_ledger or []):
        reason = _drop_reason(row)
        entries.append({
            "reason_code": reason,
            "missing_reason": reason is None,
            "source_stage": "dimension_candidate",
            "source_index": idx,
            "action": "dropped",
            "bbox": _round_bbox(_best_bbox(row)),
            "text": _clean_text(row.get("text")),
            "evidence_ids": [],
        })
    for idx, row in enumerate(post_filter_drop_ledger or []):
        reason = _drop_reason(row)
        entries.append({
            "reason_code": reason,
            "missing_reason": reason is None,
            "source_stage": "post_filter",
            "source_index": idx,
            "action": "dropped",
            "bbox": _round_bbox(_best_bbox(row)),
            "text": _clean_text(row.get("text") or row.get("nominal")),
            "evidence_ids": [],
        })
    return entries


def _build_stats(
    evidence: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    drop_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    missing_fields = []
    for row in evidence:
        missing = sorted(k for k in _REQUIRED_EVIDENCE_FIELDS if k not in row)
        if missing:
            missing_fields.append({
                "record_id": row.get("evidence_id"),
                "record_type": "evidence",
                "missing": missing,
            })
    complete_hypotheses = 0
    for row in hypotheses:
        missing = sorted(k for k in _REQUIRED_HYPOTHESIS_FIELDS if k not in row)
        if missing:
            missing_fields.append({
                "record_id": row.get("hypothesis_id"),
                "record_type": "hypothesis",
                "missing": missing,
            })
            continue
        claim_reason = (row.get("claim") or {}).get("reason_code")
        if claim_reason not in CLAIM_REASON_CODES:
            missing_fields.append({
                "record_id": row.get("hypothesis_id"),
                "record_type": "hypothesis",
                "missing": ["known_claim_reason_code"],
            })
        if (
            row.get("claim")
            and row.get("drop_ledger")
            and row.get("dedup_ledger")
            and row.get("supporting_evidence_ids")
        ):
            complete_hypotheses += 1

    true_drop_entries = [row for row in drop_entries if row.get("action") == "dropped"]
    for row in true_drop_entries:
        if not row.get("reason_code"):
            missing_fields.append({
                "record_id": f"{row.get('source_stage')}:{row.get('source_index')}",
                "record_type": "drop_ledger",
                "missing": ["reason_code"],
            })
    total_drops = len(true_drop_entries)
    drops_with_reason = sum(1 for row in true_drop_entries if row.get("reason_code"))
    hypothesis_count = len(hypotheses)
    kind_counts = Counter(row.get("kind") or "unknown" for row in evidence)
    evidence_by_id = {
        row.get("evidence_id"): row
        for row in evidence
        if row.get("evidence_id")
    }
    hypotheses_without_external_support = []
    for row in hypotheses:
        external_ids = [
            evidence_id
            for evidence_id in row.get("supporting_evidence_ids", [])
            if (evidence_by_id.get(evidence_id) or {}).get("kind") != "assembler_output"
        ]
        if not external_ids:
            hypotheses_without_external_support.append(row.get("hypothesis_id"))
    external_support_count = hypothesis_count - len(hypotheses_without_external_support)
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "evidence_count": len(evidence),
        "hypothesis_count": hypothesis_count,
        "evidence_kind_counts": dict(sorted(kind_counts.items())),
        "drop_count": total_drops,
        "drops_with_reason": drops_with_reason,
        "drops_missing_reason": total_drops - drops_with_reason,
        "drops_with_reason_ratio": (
            1.0 if total_drops == 0 else round(drops_with_reason / total_drops, 6)
        ),
        "ledger_complete_count": complete_hypotheses,
        "ledger_completeness": (
            1.0
            if hypothesis_count == 0
            else round(complete_hypotheses / hypothesis_count, 6)
        ),
        "external_support_count": external_support_count,
        "external_support_ratio": (
            1.0
            if hypothesis_count == 0
            else round(external_support_count / hypothesis_count, 6)
        ),
        "hypotheses_without_external_support": hypotheses_without_external_support,
        "missing_required_fields": missing_fields,
    }


def _link_evidence(
    dim: dict[str, Any],
    dim_bbox: dict[str, float] | None,
    evidence: list[dict[str, Any]],
) -> list[str]:
    dim_text = _clean_text(dim.get("text") or dim.get("nominal"))
    linked = []
    for row in evidence:
        if row.get("kind") == "assembler_output":
            continue
        row_bbox = row.get("bbox")
        row_text = _clean_text(row.get("text") or row.get("value"))
        iou = _bbox_iou(dim_bbox, row_bbox) if dim_bbox and row_bbox else 0.0
        text_matches = _text_link_matches(
            dim_text,
            row_text,
            iou,
            dim_bbox=dim_bbox,
            row_bbox=row_bbox,
        )
        geometry_matches = iou >= EVIDENCE_LINK_IOU_THRESHOLD
        if row.get("kind") == "ocr":
            if text_matches:
                linked.append(row["evidence_id"])
            continue
        if text_matches or geometry_matches:
            linked.append(row["evidence_id"])
    return linked


def _matching_drop_entries(
    dim_bbox: dict[str, float] | None,
    drop_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not dim_bbox:
        return []
    matched = []
    for row in drop_entries:
        row_bbox = row.get("bbox")
        if row_bbox and _bbox_iou(dim_bbox, row_bbox) >= EVIDENCE_LINK_IOU_THRESHOLD:
            matched.append(row)
    return matched


def _attach_evidence_ids(
    entries: list[dict[str, Any]],
    evidence_ids: list[str],
) -> list[dict[str, Any]]:
    attached = []
    for entry in entries:
        row = dict(entry)
        if not row.get("evidence_ids"):
            row["evidence_ids"] = list(evidence_ids)
        attached.append(row)
    return attached


def _claim_reason_code(linked_ids: list[str], evidence: list[dict[str, Any]]) -> str:
    linked = {row["evidence_id"]: row for row in evidence if row.get("evidence_id") in linked_ids}
    kinds = {row.get("kind") for row in linked.values()}
    if "candidate" in kinds or "geometric_anchor" in kinds or "vector_token" in kinds:
        return "claimed_by_geometry"
    if "ocr" in kinds:
        return "claimed_by_text"
    if "assembler_output" in kinds:
        return "claimed_by_assembler_output"
    return "no_supporting_evidence"


def _drop_reason(row: dict[str, Any]) -> str | None:
    reason = _clean_text(row.get("reject_reason") or row.get("reason"))
    return reason or None


def _text_link_matches(
    dim_text: str,
    row_text: str,
    iou: float,
    *,
    dim_bbox: dict[str, Any] | None = None,
    row_bbox: dict[str, Any] | None = None,
) -> bool:
    dim_norm = _normalize_link_text(dim_text)
    row_norm = _normalize_link_text(row_text)
    if not dim_norm or not row_norm or dim_norm != row_norm:
        if not _partial_text_link_matches(
            dim_norm,
            row_norm,
            iou,
            dim_bbox=dim_bbox,
            row_bbox=row_bbox,
        ):
            return False
        return True
    if min(_significant_char_count(dim_norm), _significant_char_count(row_norm)) <= 2:
        return iou >= SHORT_TEXT_LINK_IOU_THRESHOLD
    return True


def _normalize_link_text(value: str) -> str:
    text = str(value or "").upper()
    replacements = {
        "。": ".",
        "．": ".",
        "，": ".",
        "＋": "+",
        "﹢": "+",
        "－": "-",
        "−": "-",
        "﹣": "-",
        "—": "-",
        "Ø": "⌀",
        "Φ": "⌀",
        "∅": "⌀",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return "".join(text.split())


def _partial_text_link_matches(
    dim_norm: str,
    row_norm: str,
    iou: float,
    *,
    dim_bbox: dict[str, Any] | None,
    row_bbox: dict[str, Any] | None,
) -> bool:
    if not _geometry_allows_partial_text_link(iou, dim_bbox, row_bbox):
        return False

    if (
        row_norm in dim_norm
        and len(row_norm) >= PARTIAL_TEXT_LINK_MIN_SIGNIFICANT_CHARS
    ):
        return True
    if (
        dim_norm in row_norm
        and len(dim_norm) >= PARTIAL_TEXT_LINK_MIN_SIGNIFICANT_CHARS
    ):
        return True
    if (
        row_norm in dim_norm
        and _significant_char_count(row_norm) >= 2
        and iou >= PARTIAL_SHORT_TEXT_LINK_IOU_THRESHOLD
    ):
        return True
    if (
        dim_norm in row_norm
        and _looks_like_angle_fragment(dim_norm)
        and iou >= PARTIAL_ANGLE_TEXT_LINK_IOU_THRESHOLD
    ):
        return True
    if (
        row_norm in dim_norm
        and _looks_like_angle_fragment(row_norm)
        and iou >= PARTIAL_ANGLE_TEXT_LINK_IOU_THRESHOLD
    ):
        return True

    dim_tokens = set(_numeric_link_tokens(dim_norm))
    row_tokens = set(_numeric_link_tokens(row_norm))
    shared_tokens = dim_tokens & row_tokens
    return any(len(token) >= PARTIAL_TEXT_LINK_MIN_SIGNIFICANT_CHARS for token in shared_tokens)


def _geometry_allows_partial_text_link(
    iou: float,
    dim_bbox: dict[str, Any] | None,
    row_bbox: dict[str, Any] | None,
) -> bool:
    if iou >= PARTIAL_TEXT_LINK_IOU_THRESHOLD:
        return True
    return _bbox_center_inside_expanded(
        outer=dim_bbox,
        inner=row_bbox,
        expansion_ratio=PARTIAL_TEXT_LINK_EXPANSION_RATIO,
    )


def _bbox_center_inside_expanded(
    *,
    outer: dict[str, Any] | None,
    inner: dict[str, Any] | None,
    expansion_ratio: float,
) -> bool:
    try:
        ox = float(outer["x"])
        oy = float(outer["y"])
        ow = float(outer["w"])
        oh = float(outer["h"])
        ix = float(inner["x"])
        iy = float(inner["y"])
        iw = float(inner["w"])
        ih = float(inner["h"])
    except (KeyError, TypeError, ValueError):
        return False
    margin_x = max(2.0, ow * expansion_ratio)
    margin_y = max(2.0, oh * expansion_ratio)
    cx = ix + iw / 2.0
    cy = iy + ih / 2.0
    return (
        ox - margin_x <= cx <= ox + ow + margin_x
        and oy - margin_y <= cy <= oy + oh + margin_y
    )


def _numeric_link_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for ch in value:
        if ch.isdigit() or ch in ".+-":
            current.append(ch)
            continue
        _flush_numeric_link_token(tokens, current)
    _flush_numeric_link_token(tokens, current)
    return tokens


def _flush_numeric_link_token(tokens: list[str], current: list[str]) -> None:
    if not current:
        return
    token = "".join(current).strip("+-")
    current.clear()
    if any(ch.isdigit() for ch in token):
        tokens.append(token)


def _significant_char_count(value: str) -> int:
    ignored = {".", ",", " ", "\t", "\n", "\r", "(", ")", "[", "]", "{", "}", "（", "）"}
    return sum(1 for ch in value if ch not in ignored)


def _looks_like_angle_fragment(value: str) -> bool:
    return "°" in value and any(ch.isdigit() for ch in value) and len(value) <= 4


def _best_bbox(item: dict[str, Any] | None) -> dict[str, float] | None:
    if not item:
        return None
    direct = normalize_bbox(item)
    if direct:
        return direct
    for key in (
        "bbox_in_pdf",
        "bbox_pdf",
        "region",
        "text_bbox",
        "candidate_bbox",
        "corridor_bbox",
        "anchor_bbox",
    ):
        value = item.get(key)
        bbox = normalize_bbox(value)
        if bbox:
            return bbox
    return None


def _bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    try:
        ax0 = float(a["x"])
        ay0 = float(a["y"])
        ax1 = ax0 + float(a["w"])
        ay1 = ay0 + float(a["h"])
        bx0 = float(b["x"])
        by0 = float(b["y"])
        bx1 = bx0 + float(b["w"])
        by1 = by0 + float(b["h"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _round_bbox(bbox: dict[str, Any] | None) -> dict[str, float] | None:
    if not bbox:
        return None
    return {
        "x": round(float(bbox.get("x", 0.0)), 3),
        "y": round(float(bbox.get("y", 0.0)), 3),
        "w": round(float(bbox.get("w", 0.0)), 3),
        "h": round(float(bbox.get("h", 0.0)), 3),
    }


def _normalize_quad(quad: Any) -> Any:
    if not quad:
        return None
    if isinstance(quad, dict):
        return quad
    if isinstance(quad, (list, tuple)):
        normalized = []
        for point in quad:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    normalized.append([round(float(point[0]), 3), round(float(point[1]), 3)])
                except (TypeError, ValueError):
                    return None
        return normalized or None
    return None


def _raw_ref(item: dict[str, Any]) -> dict[str, Any]:
    ref = {}
    for key in (
        "candidate_id",
        "source",
        "type",
        "subtype",
        "priority_hint",
        "reject_reason",
        "reason",
        "orientation",
        "line_kind",
    ):
        if key in item and item.get(key) is not None:
            ref[key] = item.get(key)
    return ref


def _item_text(item: dict[str, Any]) -> str:
    return _clean_text(
        item.get("text")
        or item.get("nominal")
        or item.get("value")
        or item.get("label")
        or item.get("symbol")
    )


def _item_value(item: dict[str, Any]) -> Any:
    for key in ("nominal", "value", "parsed_nominal"):
        if item.get(key) is not None:
            return item.get(key)
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "row"


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
