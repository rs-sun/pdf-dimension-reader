"""Strict-vector GD&T review candidate assembly.

The strict path has two deliberately separate steps:

1. capture compact compartment-0 symbol evidence while the pipeline's shared
   PyMuPDF drawing snapshot is still available;
2. after the ordinary R33-M1 phrase chain has resolved its controlled runtime,
   use one in-frame decimal anchor to read one tolerance compartment and build
   a review-only GD&T item.

No OCR, YOLO-B, legacy assembler, datum inversion, dimension-line, arrow, or
leader evidence is imported or called here.

Non-empty vector tolerance reads remain review-visible even when their syntax
needs correction.  Geometry, symbol, empty-read, public-contract and forbidden
runtime failures still close without guessing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Callable

from anchor_orientation import orientation_from_axis_angle
from dimension_syntax import classify_dimension_syntax
from gdt_vector_inversion import (
    CLASS_NAMES,
    CLASS_SYMBOLS,
    GDTVectorInverter,
)
from r33_m1_phrase_runtime import (
    PINNED_TEMPLATE_MARGIN,
    PINNED_TEMPLATE_THRESHOLD,
    R33M1PhraseRuntime,
)
from review_candidates_contract import (
    build_gdt_review_candidate_item,
    is_valid_unsigned_decimal_text,
)
from vector_page_context import VectorPageContext


SYMBOL_EVIDENCE_SCHEMA_VERSION = "strict_gdt_symbol_evidence_v1"
SYMBOL_EVIDENCE_BATCH_SCHEMA_VERSION = "strict_gdt_symbol_evidence_batch_v1"
AUDIT_SCHEMA_VERSION = "strict_gdt_candidate_audit_v1"
AUDIT_FIELD = AUDIT_SCHEMA_VERSION
MIN_SYMBOL_CONFIDENCE = 0.60
DECIMAL_QUAD_SCHEMA_VERSION = "vector_decimal_point_quad_v1"
DECIMAL_QUAD_SOURCE = "qu_rect_ratio"
SYMBOL_METHOD = "vector_inversion"
# Keep the GD&T compartment-reader budget local.  The ordinary phrase reader
# is quarantined independently and must not remain a production import solely
# to supply this constant.
MAX_BASE_PITCH_PRIMITIVE_COUNT = 180

_CORE_SYMBOLS = {
    name: CLASS_SYMBOLS[name]
    for name in CLASS_NAMES[:12]
}
_SYMBOL_DROP_ORDER = (
    "frame_invalid",
    "frame_compartments_invalid",
    "symbol_inversion_failed",
    "symbol_missing_or_low_confidence",
    "symbol_conflict",
    "rect_origin_symbol_not_allowed",
    "duplicate_frame",
)
_CANDIDATE_DROP_ORDER = (
    "decimal_anchor_missing",
    "decimal_anchor_conflict",
    "reader_input_empty",
    "reader_input_complexity_exceeded",
    "decimal_primitive_missing",
    "reader_execution_failed",
    "reader_forbidden_runtime_nonzero",
    "reader_output_invalid",
    "candidate_contract_invalid",
    "duplicate_candidate",
)
_FORBIDDEN_RUNTIME_COUNTER_KEYS = frozenset({
    "invert_datum_call_count",
    "ocr_family_call_count",
    "paddleocr_call_count",
    "rapidocr_call_count",
    "yolo_char_call_count",
    "yolo_b_call_count",
    "assembler_call_count",
    "arrow_detector_call_count",
})


def build_strict_gdt_symbol_evidence(
    *,
    frames: list[dict[str, Any]],
    fitz_page: Any,
    drawing_snapshot: Any,
    trace_id: str,
    page_index: int,
    inverter: GDTVectorInverter | None = None,
) -> dict[str, Any]:
    """Capture one verified core-symbol hit per valid feature-control frame."""
    _require_identity(trace_id, page_index)
    if not isinstance(frames, list):
        raise ValueError("strict_gdt_frames_not_list")
    if fitz_page is None:
        raise ValueError("strict_gdt_fitz_page_missing")

    drops: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    inverter_call_count = 0
    resolved_inverter = inverter
    if frames and resolved_inverter is None:
        if drawing_snapshot is None:
            raise ValueError("strict_gdt_drawing_snapshot_missing")
        resolved_inverter = GDTVectorInverter(
            drawing_snapshot=drawing_snapshot,
        )

    for raw_frame in sorted(frames, key=_frame_sort_key):
        frame = _normalize_frame(raw_frame)
        if frame is None:
            drops["frame_invalid"] += 1
            continue
        compartments = frame["compartments"]
        if len(compartments) < 2:
            drops["frame_compartments_invalid"] += 1
            continue
        frame_digest = _frame_digest(frame)
        try:
            inverter_call_count += 1
            hits = resolved_inverter.invert_in_frame(
                fitz_page,
                frame["bbox"],
                [compartments[0]],
            )
        except Exception:
            drops["symbol_inversion_failed"] += 1
            continue
        qualified = [
            hit
            for hit in (hits if isinstance(hits, list) else [])
            if _valid_core_symbol_hit(hit)
        ]
        if not qualified:
            drops["symbol_missing_or_low_confidence"] += 1
            continue
        if len(qualified) != 1:
            drops["symbol_conflict"] += 1
            continue
        hit = qualified[0]
        if (
            frame["origin"] == "rect"
            and hit["symbol_class"] != 0
        ):
            drops["rect_origin_symbol_not_allowed"] += 1
            continue
        rows.append({
            "schema_version": SYMBOL_EVIDENCE_SCHEMA_VERSION,
            "trace_id": trace_id,
            "page_index": page_index,
            "frame_digest": frame_digest,
            "bbox": frame["bbox"],
            "compartments": compartments,
            "origin": frame["origin"],
            "angle_deg": frame["angle_deg"],
            "symbol": {
                "symbol_class": hit["symbol_class"],
                "symbol_name": hit["symbol_name"],
                "symbol_unicode": hit["symbol_unicode"],
                "confidence": float(hit["confidence"]),
                "method": SYMBOL_METHOD,
                "compartment_index": 0,
            },
            "consumer_allowed": False,
        })

    unique_rows: list[dict[str, Any]] = []
    seen_frames: set[str] = set()
    for row in sorted(rows, key=lambda value: value["frame_digest"]):
        if row["frame_digest"] in seen_frames:
            drops["duplicate_frame"] += 1
            continue
        seen_frames.add(row["frame_digest"])
        unique_rows.append(row)
    audit = _symbol_audit(
        trace_id=trace_id,
        page_index=page_index,
        input_frame_count=len(frames),
        rows=unique_rows,
        drops=drops,
        inverter_call_count=inverter_call_count,
    )
    return {
        "schema_version": SYMBOL_EVIDENCE_BATCH_SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": page_index,
        "rows": unique_rows,
        "audit": audit,
        "consumer_allowed": False,
    }


def build_strict_gdt_review_candidates(
    *,
    symbol_evidence: dict[str, Any],
    decimal_point_rows: list[dict[str, Any]],
    page_context: VectorPageContext,
    runtime: R33M1PhraseRuntime,
    trace_id: str,
    page_index: int,
) -> dict[str, Any]:
    """Read bounded tolerance values and emit exact 12-key GD&T items."""
    _require_identity(trace_id, page_index)
    evidence_rows = _validated_symbol_evidence_rows(
        symbol_evidence,
        trace_id=trace_id,
        page_index=page_index,
    )
    if not isinstance(decimal_point_rows, list):
        raise ValueError("strict_gdt_decimal_rows_not_list")

    decimal_rows = [
        normalized
        for row in decimal_point_rows
        if (normalized := _normalize_decimal_row(row)) is not None
    ]
    drops: Counter[str] = Counter()
    proposals: list[dict[str, Any]] = []
    for evidence in evidence_rows:
        matches: list[tuple[dict[str, Any], int]] = []
        ambiguous_dot = False
        for decimal in decimal_rows:
            compartment_indices = [
                index
                for index, compartment in enumerate(
                    evidence["compartments"]
                )
                if index >= 1
                and _point_in_bbox(decimal["center"], compartment)
            ]
            if len(compartment_indices) > 1:
                ambiguous_dot = True
                break
            if len(compartment_indices) == 1:
                matches.append((decimal, compartment_indices[0]))
        if ambiguous_dot or len(matches) > 1:
            drops["decimal_anchor_conflict"] += 1
            continue
        if not matches:
            drops["decimal_anchor_missing"] += 1
            continue
        decimal, tolerance_index = matches[0]
        proposals.append({
            "evidence": evidence,
            "decimal": decimal,
            "tolerance_compartment_index": tolerance_index,
        })

    if proposals:
        _validate_runtime_and_context(runtime, page_context)
    items: list[dict[str, Any]] = []
    reader_call_count = 0
    template_sha = _template_content_sha256(runtime) if proposals else None
    for proposal in proposals:
        evidence = proposal["evidence"]
        decimal = proposal["decimal"]
        tolerance_index = proposal["tolerance_compartment_index"]
        compartment = evidence["compartments"][tolerance_index]
        reader_items, decimal_item_id = _bounded_reader_items(
            page_context,
            compartment=compartment,
            decimal=decimal,
        )
        if not reader_items:
            drops["reader_input_empty"] += 1
            continue
        if len(reader_items) > MAX_BASE_PITCH_PRIMITIVE_COUNT:
            drops["reader_input_complexity_exceeded"] += 1
            continue
        if decimal_item_id is None:
            drops["decimal_primitive_missing"] += 1
            continue
        reader_phrase_id = (
            f"gdt_{evidence['frame_digest'][:16]}_"
            f"{decimal['id']}"
        )
        reader_call_count += 1
        try:
            result = runtime.read_one(
                reader_items,
                bbox=_bbox_xyxy(compartment),
                orientation=orientation_from_axis_angle(
                    evidence["angle_deg"],
                ),
                template_library=runtime.template_library,
                threshold=PINNED_TEMPLATE_THRESHOLD,
                margin=PINNED_TEMPLATE_MARGIN,
                candidate_axis_angle_deg=evidence["angle_deg"],
                candidate_axis_angle_source="gdt_frame",
                phrase_id=reader_phrase_id,
                source_candidate_evidence=[],
                source_anchor_types=("dot_strict",),
                query_bbox_items=_reader_query_callback(page_context),
                page_context_manifest=page_context.manifest_v1(),
                prepared_context=None,
            )
        except Exception:
            drops["reader_execution_failed"] += 1
            continue
        parsed = _parse_reader_result(
            result,
            input_item_ids={
                str(item["item_id"])
                for item in reader_items
            },
            decimal_item_id=decimal_item_id,
        )
        if parsed["reason"] is not None:
            drops[parsed["reason"]] += 1
            continue
        tolerance_value = parsed["text"]
        syntax = parsed["syntax"]
        if (
            "." not in tolerance_value
            or not is_valid_unsigned_decimal_text(tolerance_value)
        ) and syntax == "ok":
            syntax = "invalid_dimension_syntax"
        symbol = evidence["symbol"]
        try:
            item = build_gdt_review_candidate_item(
                page_index=page_index,
                bbox=evidence["bbox"],
                symbol_name=symbol["symbol_name"],
                symbol_unicode=symbol["symbol_unicode"],
                tolerance_value=tolerance_value,
                frame_digest=evidence["frame_digest"],
                decimal_quad_id=decimal["id"],
                symbol_method=symbol["method"],
                symbol_confidence=symbol["confidence"],
                symbol_compartment_index=0,
                tolerance_compartment_index=tolerance_index,
                reader_phrase_id=reader_phrase_id,
                template_content_sha256=template_sha,
                syntax=syntax,
            )
        except (TypeError, ValueError):
            drops["candidate_contract_invalid"] += 1
            continue
        items.append(item)

    unique_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in sorted(items, key=lambda value: value["candidate_id"]):
        if item["candidate_id"] in seen_ids:
            drops["duplicate_candidate"] += 1
            continue
        seen_ids.add(item["candidate_id"])
        unique_items.append(item)
    return {
        "items": unique_items,
        "audit": _candidate_audit(
            trace_id=trace_id,
            page_index=page_index,
            symbol_evidence=symbol_evidence,
            input_decimal_count=len(decimal_point_rows),
            valid_decimal_count=len(decimal_rows),
            proposal_count=len(proposals),
            reader_call_count=reader_call_count,
            items=unique_items,
            drops=drops,
            template_identity=(
                runtime.template_identity
                if proposals
                else None
            ),
        ),
    }


def empty_strict_gdt_candidate_result(
    *,
    symbol_evidence: dict[str, Any],
    trace_id: str,
    page_index: int,
    reason: str,
) -> dict[str, Any]:
    """Return an explicit non-running audit when the shared runtime is closed."""
    _require_identity(trace_id, page_index)
    rows = _validated_symbol_evidence_rows(
        symbol_evidence,
        trace_id=trace_id,
        page_index=page_index,
    )
    return {
        "items": [],
        "audit": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "trace_id": trace_id,
            "page_index": page_index,
            "status": "unavailable",
            "reason": str(reason),
            "input_frame_count": int(
                symbol_evidence["audit"]["input_frame_count"]
            ),
            "symbol_evidence_count": len(rows),
            "symbol_inverter_call_count": int(
                symbol_evidence["audit"]["symbol_inverter_call_count"]
            ),
            "symbol_drop_count": int(
                symbol_evidence["audit"]["drop_count"]
            ),
            "symbol_drop_reasons": list(
                symbol_evidence["audit"]["drop_reasons"]
            ),
            "input_symbol_evidence_count": len(rows),
            "input_decimal_count": 0,
            "valid_decimal_count": 0,
            "proposal_count": 0,
            "value_reader_call_count": 0,
            "candidate_count": 0,
            "candidate_drop_count": 0,
            "candidate_drop_reasons": [],
            "drop_count": int(symbol_evidence["audit"]["drop_count"]),
            "drop_reasons": list(
                symbol_evidence["audit"]["drop_reasons"]
            ),
            "invert_datum_call_count": 0,
            "ocr_family_call_count": 0,
            "paddleocr_call_count": 0,
            "rapidocr_call_count": 0,
            "yolo_char_call_count": 0,
            "yolo_b_call_count": 0,
            "assembler_call_count": 0,
            "template_identity": None,
            "consumer_allowed": False,
        },
    }


def _normalize_frame(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    bbox = _normalize_bbox(value.get("bbox"))
    raw_compartments = value.get("compartments")
    if bbox is None or not isinstance(raw_compartments, list):
        return None
    compartments = [
        normalized
        for raw in raw_compartments
        if (normalized := _normalize_bbox(raw)) is not None
    ]
    if len(compartments) != len(raw_compartments):
        return None
    if any(not _bbox_within(compartment, bbox) for compartment in compartments):
        return None
    raw_angle = (
        (value.get("oriented_quad") or {}).get("angle_deg")
        if isinstance(value.get("oriented_quad"), dict)
        else value.get("angle_deg", 0.0)
    )
    try:
        angle = float(raw_angle or 0.0) % 180.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(angle):
        return None
    origin = value.get("_origin")
    if origin not in {"line", "rect"}:
        return None
    return {
        "bbox": bbox,
        "compartments": compartments,
        "origin": origin,
        "angle_deg": angle,
    }


def _normalize_decimal_row(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if (
        value.get("schema_version") != DECIMAL_QUAD_SCHEMA_VERSION
        or value.get("source") != DECIMAL_QUAD_SOURCE
        or value.get("consumer_allowed") is not True
        or not isinstance(value.get("id"), str)
        or not value["id"]
    ):
        return None
    bbox = _normalize_bbox(value.get("bbox"))
    center = value.get("center")
    detail = value.get("detail")
    if bbox is None or not isinstance(center, dict) or not isinstance(detail, dict):
        return None
    try:
        cx = float(center["x"])
        cy = float(center["y"])
    except (KeyError, TypeError, ValueError):
        return None
    drawing_order = detail.get("drawing_order")
    item_index = detail.get("item_index")
    if (
        not math.isfinite(cx)
        or not math.isfinite(cy)
        or type(drawing_order) is not int
        or drawing_order < 0
        or type(item_index) is not int
        or item_index < 0
    ):
        return None
    return {
        "id": value["id"],
        "bbox": bbox,
        "center": {"x": cx, "y": cy},
        "drawing_order": drawing_order,
        "item_index": item_index,
    }


def _valid_core_symbol_hit(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    symbol_class = value.get("symbol_class")
    symbol_name = value.get("symbol_name")
    confidence = value.get("confidence")
    return bool(
        type(symbol_class) is int
        and 0 <= symbol_class <= 11
        and symbol_name == CLASS_NAMES[symbol_class]
        and value.get("symbol_unicode") == _CORE_SYMBOLS.get(symbol_name)
        and type(confidence) in {int, float}
        and math.isfinite(float(confidence))
        and float(confidence) >= MIN_SYMBOL_CONFIDENCE
        and value.get("method") == SYMBOL_METHOD
        and value.get("compartment_index") == 0
    )


def _validated_symbol_evidence_rows(
    value: Any,
    *,
    trace_id: str,
    page_index: int,
) -> list[dict[str, Any]]:
    if not (
        isinstance(value, dict)
        and value.get("schema_version")
        == SYMBOL_EVIDENCE_BATCH_SCHEMA_VERSION
        and value.get("trace_id") == trace_id
        and value.get("page_index") == page_index
        and value.get("consumer_allowed") is False
        and isinstance(value.get("rows"), list)
    ):
        raise ValueError("strict_gdt_symbol_evidence_invalid")
    rows = value["rows"]
    for row in rows:
        if not (
            isinstance(row, dict)
            and row.get("schema_version") == SYMBOL_EVIDENCE_SCHEMA_VERSION
            and row.get("trace_id") == trace_id
            and row.get("page_index") == page_index
            and row.get("consumer_allowed") is False
            and _normalize_bbox(row.get("bbox")) == row.get("bbox")
            and isinstance(row.get("compartments"), list)
            and len(row["compartments"]) >= 2
            and _valid_sha(row.get("frame_digest"))
            and _valid_core_symbol_hit(row.get("symbol"))
        ):
            raise ValueError("strict_gdt_symbol_evidence_row_invalid")
    return rows


def _validate_runtime_and_context(
    runtime: Any,
    page_context: Any,
) -> None:
    if not (
        isinstance(runtime, R33M1PhraseRuntime)
        and isinstance(runtime.template_library, dict)
        and callable(runtime.read_one)
        and _valid_template_identity(runtime.template_identity)
    ):
        raise ValueError("strict_gdt_runtime_invalid")
    if not isinstance(page_context, VectorPageContext):
        raise ValueError("strict_gdt_page_context_invalid")
    page_context.assert_integrity()


def _bounded_reader_items(
    page_context: VectorPageContext,
    *,
    compartment: dict[str, float],
    decimal: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    inset = _inset_bbox(compartment)
    primitives = [
        primitive
        for primitive in page_context.query_bbox(_bbox_xyxy(inset))
        if _primitive_center_in_bbox(primitive, inset)
    ]
    items = [_reader_item(primitive, page_context) for primitive in primitives]
    decimal_matches = [
        primitive
        for primitive in primitives
        if (
            primitive.op == "qu"
            and primitive.drawing_order == decimal["drawing_order"]
            and primitive.item_index == decimal["item_index"]
        )
    ]
    decimal_item_id = (
        page_context.item_id(decimal_matches[0])
        if len(decimal_matches) == 1
        else None
    )
    return items, decimal_item_id


def _reader_query_callback(
    page_context: VectorPageContext,
) -> Callable[[Any], list[dict[str, Any]]]:
    def query(value: Any) -> list[dict[str, Any]]:
        bbox = _query_bbox(value)
        if bbox is None:
            return []
        return [
            _reader_item(primitive, page_context)
            for primitive in page_context.query_bbox(bbox)
        ]

    return query


def _parse_reader_result(
    value: Any,
    *,
    input_item_ids: set[str],
    decimal_item_id: str,
) -> dict[str, Any]:
    if not (
        isinstance(value, tuple)
        and len(value) == 4
        and isinstance(value[0], str)
        and isinstance(value[1], list)
        and isinstance(value[2], list)
        and isinstance(value[3], dict)
    ):
        return {
            "text": "",
            "reason": "reader_output_invalid",
            "syntax": "invalid_dimension_syntax",
        }
    text, spans, drops, debug = value
    if _recursive_forbidden_runtime_counter_violation(
        (spans, drops, debug)
    ):
        return {
            "text": text,
            "reason": "reader_forbidden_runtime_nonzero",
            "syntax": classify_dimension_syntax(text, drops=drops),
        }
    if _recursive_true_consumer_flag_count(debug):
        return {
            "text": text,
            "reason": "reader_output_invalid",
            "syntax": classify_dimension_syntax(text, drops=drops),
        }
    if not text:
        return {
            "text": "",
            "reason": "reader_output_invalid",
            "syntax": "invalid_dimension_syntax",
        }
    return {
        "text": text,
        "reason": None,
        "syntax": classify_dimension_syntax(text, drops=drops),
    }


def _reader_item(item: Any, page_context: VectorPageContext) -> dict[str, Any]:
    return {
        "item_id": page_context.item_id(item),
        "op": str(item.op),
        "points": [[float(x), float(y)] for x, y in item.points],
        "bbox": [float(value) for value in item.bbox],
        "draw_type": str(item.draw_type),
        "stroke_width": (
            float(item.stroke_width)
            if item.stroke_width is not None
            else None
        ),
    }


def _candidate_audit(
    *,
    trace_id: str,
    page_index: int,
    symbol_evidence: dict[str, Any],
    input_decimal_count: int,
    valid_decimal_count: int,
    proposal_count: int,
    reader_call_count: int,
    items: list[dict[str, Any]],
    drops: Counter[str],
    template_identity: Any,
) -> dict[str, Any]:
    symbol_audit = symbol_evidence["audit"]
    symbol_drop_rows = list(symbol_audit["drop_reasons"])
    candidate_drop_rows = _drop_rows(drops, _CANDIDATE_DROP_ORDER)
    drop_rows = [*symbol_drop_rows, *candidate_drop_rows]
    total_drop_count = sum(row["count"] for row in drop_rows)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": page_index,
        "status": "ok",
        "reason": None,
        "input_frame_count": int(symbol_audit["input_frame_count"]),
        "symbol_evidence_count": len(symbol_evidence["rows"]),
        "symbol_inverter_call_count": int(
            symbol_audit["symbol_inverter_call_count"]
        ),
        "symbol_drop_count": int(symbol_audit["drop_count"]),
        "symbol_drop_reasons": symbol_drop_rows,
        "input_symbol_evidence_count": len(symbol_evidence["rows"]),
        "input_decimal_count": input_decimal_count,
        "valid_decimal_count": valid_decimal_count,
        "proposal_count": proposal_count,
        "value_reader_call_count": reader_call_count,
        "candidate_count": len(items),
        "candidate_drop_count": sum(
            row["count"] for row in candidate_drop_rows
        ),
        "candidate_drop_reasons": candidate_drop_rows,
        "drop_count": total_drop_count,
        "drop_reasons": drop_rows,
        "invert_datum_call_count": 0,
        "ocr_family_call_count": 0,
        "paddleocr_call_count": 0,
        "rapidocr_call_count": 0,
        "yolo_char_call_count": 0,
        "yolo_b_call_count": 0,
        "assembler_call_count": 0,
        "template_identity": (
            dict(template_identity)
            if isinstance(template_identity, dict)
            else None
        ),
        "consumer_allowed": False,
    }


def _symbol_audit(
    *,
    trace_id: str,
    page_index: int,
    input_frame_count: int,
    rows: list[dict[str, Any]],
    drops: Counter[str],
    inverter_call_count: int,
) -> dict[str, Any]:
    drop_rows = _drop_rows(drops, _SYMBOL_DROP_ORDER)
    return {
        "schema_version": "strict_gdt_symbol_evidence_audit_v1",
        "trace_id": trace_id,
        "page_index": page_index,
        "status": "ok",
        "input_frame_count": input_frame_count,
        "symbol_evidence_count": len(rows),
        "drop_count": sum(row["count"] for row in drop_rows),
        "drop_reasons": drop_rows,
        "symbol_inverter_call_count": inverter_call_count,
        "invert_datum_call_count": 0,
        "consumer_allowed": False,
    }


def _frame_digest(frame: dict[str, Any]) -> str:
    return _digest({
        "schema_version": SYMBOL_EVIDENCE_SCHEMA_VERSION,
        "bbox": frame["bbox"],
        "compartments": frame["compartments"],
        "origin": frame["origin"],
        "angle_deg": frame["angle_deg"],
    })


def _frame_sort_key(value: Any) -> tuple[float, float, float, float, str]:
    bbox = value.get("bbox") if isinstance(value, dict) else None
    normalized = _normalize_bbox(bbox) or {
        "x": math.inf,
        "y": math.inf,
        "w": math.inf,
        "h": math.inf,
    }
    return (
        normalized["y"],
        normalized["x"],
        normalized["h"],
        normalized["w"],
        str(value),
    )


def _normalize_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        x = float(value["x"])
        y = float(value["y"])
        width = float(value["w"])
        height = float(value["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(item) for item in (x, y, width, height))
        or width <= 0.0
        or height <= 0.0
    ):
        return None
    return {"x": x, "y": y, "w": width, "h": height}


def _bbox_within(
    inner: dict[str, float],
    outer: dict[str, float],
    *,
    tolerance: float = 1.0,
) -> bool:
    return bool(
        inner["x"] >= outer["x"] - tolerance
        and inner["y"] >= outer["y"] - tolerance
        and inner["x"] + inner["w"]
        <= outer["x"] + outer["w"] + tolerance
        and inner["y"] + inner["h"]
        <= outer["y"] + outer["h"] + tolerance
    )


def _point_in_bbox(
    point: dict[str, float],
    bbox: dict[str, float],
) -> bool:
    x = float(point["x"])
    y = float(point["y"])
    return bool(
        bbox["x"] < x < bbox["x"] + bbox["w"]
        and bbox["y"] < y < bbox["y"] + bbox["h"]
    )


def _inset_bbox(value: dict[str, float]) -> dict[str, float]:
    pad = min(max(min(value["w"], value["h"]) * 0.04, 0.35), 1.25)
    return {
        "x": value["x"] + pad,
        "y": value["y"] + pad,
        "w": max(0.01, value["w"] - 2.0 * pad),
        "h": max(0.01, value["h"] - 2.0 * pad),
    }


def _bbox_xyxy(
    value: dict[str, float],
) -> tuple[float, float, float, float]:
    return (
        value["x"],
        value["y"],
        value["x"] + value["w"],
        value["y"] + value["h"],
    )


def _query_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        normalized = _normalize_bbox(value)
        return _bbox_xyxy(normalized) if normalized is not None else None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(item) for item in (x0, y0, x1, y1))
        or x1 <= x0
        or y1 <= y0
    ):
        return None
    return x0, y0, x1, y1


def _primitive_center_in_bbox(
    primitive: Any,
    bbox: dict[str, float],
) -> bool:
    x0, y0, x1, y1 = primitive.bbox
    return _point_in_bbox(
        {"x": (x0 + x1) / 2.0, "y": (y0 + y1) / 2.0},
        bbox,
    )


def _valid_template_identity(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == "r33_m1_template_identity_v1"
        and isinstance(value.get("template_version"), str)
        and bool(value["template_version"])
        and _valid_sha(value.get("content_sha256"))
    )


def _template_content_sha256(runtime: R33M1PhraseRuntime) -> str:
    value = runtime.template_identity["content_sha256"]
    if not _valid_sha(value):
        raise ValueError("strict_gdt_template_identity_invalid")
    return value


def _drop_rows(
    drops: Counter[str],
    order: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {"reason": reason, "count": int(drops[reason])}
        for reason in order
        if drops[reason]
    ]


def _recursive_true_consumer_flag_count(value: Any) -> int:
    if isinstance(value, dict):
        return int(
            "consumer_allowed" in value
            and value["consumer_allowed"] is not False
        ) + sum(
            _recursive_true_consumer_flag_count(child)
            for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _recursive_true_consumer_flag_count(child)
            for child in value
        )
    return 0


def _recursive_forbidden_runtime_counter_violation(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key in _FORBIDDEN_RUNTIME_COUNTER_KEYS
                and (type(child) is not int or child != 0)
            ):
                return True
            if _recursive_forbidden_runtime_counter_violation(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _recursive_forbidden_runtime_counter_violation(child)
            for child in value
        )
    return False


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _valid_sha(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_identity(trace_id: Any, page_index: Any) -> None:
    if not isinstance(trace_id, str) or not trace_id or "\x00" in trace_id:
        raise ValueError("strict_gdt_trace_id_invalid")
    if type(page_index) is not int or page_index < 0:
        raise ValueError("strict_gdt_page_index_invalid")
