"""Dump-only vector numeric phrase assembly.

This is the next R2a shell after vector digit matching: it assembles accepted
vector digits and vector symbol glyphs into auditable numeric strings without
calling OCR or routing anything into final dimensions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


SCHEMA_VERSION = "vector_numeric_phrases_v1"
STATS_SCHEMA_VERSION = "vector_numeric_phrase_stats_v1"
CONSUMER_ALLOWED = False
ACCEPTED_DIGIT_DECISIONS = frozenset({"accepted_exact", "accepted_prototype"})
STRUCTURAL_ACCEPTED_DIGIT_DECISION = "accepted_structural"
STRUCTURAL_ACCEPTED_SLOT_SOURCES = frozenset({
    "vector_text_component_plus_minus_anchored_slot",
})
SYMBOL_TEXT_BY_GLYPH_TYPE = {
    "dot": ".",
    "plus_minus": "±",
    "plus": "+",
    "minus": "-",
    "slash": "/",
    "diameter": "⌀",
    "degree": "°",
}
MAX_JOIN_GAP_FACTOR = 1.5
MAX_JOIN_GAP_PT = 18.0
VERTICAL_OVERLAP_RATIO = 0.30

_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_LEADING_DECIMAL_RE = re.compile(r"^\.\d+$")
_NOMINAL_PLUS_MINUS_RE = re.compile(r"^(?:\d+(?:\.\d+)?|\.\d+)±(?:\d+(?:\.\d+)?|\.\d+)$")
_PLUS_MINUS_RE = re.compile(r"^±(?:\d+(?:\.\d+)?|\.\d+)$")
_NOMINAL_ASYMMETRIC_TOLERANCE_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?|\.\d+)[+\-](?:\d+(?:\.\d+)?|\.\d+)/?[+\-](?:\d+(?:\.\d+)?|\.\d+)$"
)
_NOMINAL_SIGNED_TOLERANCE_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?|\.\d+)[+\-](?:\d+(?:\.\d+)?|\.\d+)$"
)
_DIAMETER_SIGNED_TOLERANCE_RE = re.compile(
    r"^⌀(?:\d+(?:\.\d+)?|\.\d+)([+\-])(?:\d+(?:\.\d+)?|\.\d+)$"
)
_DIAMETER_RE = re.compile(
    r"^⌀(?:\d+(?:\.\d+)?|\.\d+)(?:±(?:\d+(?:\.\d+)?|\.\d+))?$"
)
_ANGLE_RE = re.compile(r"^(?:\d+(?:\.\d+)?|\.\d+)°$")


def build_vector_numeric_phrases_v1(
    *,
    vector_digit_matches: list[dict[str, Any]] | None,
    vector_glyph_tokens: list[dict[str, Any]] | None,
    page_index: int = 0,
) -> dict[str, Any]:
    digit_rows = list(vector_digit_matches or [])
    glyph_rows = list(vector_glyph_tokens or [])
    digit_candidates = _digit_candidate_chars(digit_rows)
    digit_chars = _digit_chars(digit_rows)
    symbol_chars = _symbol_chars(glyph_rows)
    chars = digit_chars + symbol_chars
    phrase_rows: list[dict[str, Any]] = []
    rejected_segments = 0
    rejected_reasons: Counter[str] = Counter()
    segments = _character_segments(chars)
    for segment in segments:
        row = _row_from_segment(segment, page_index=page_index, order=len(phrase_rows))
        if row is None:
            rejected_segments += 1
            rejected_reasons[_segment_rejection_reason(segment)] += 1
            continue
        phrase_rows.append(row)
    rescue_rows = _rescued_symbol_phrase_rows(
        segments,
        page_index=page_index,
        start_order=len(phrase_rows),
    )
    phrase_rows.extend(rescue_rows)

    phrase_kind_counts = Counter(str(row.get("phrase_kind") or "unknown") for row in phrase_rows)
    rescue_kind_counts = Counter(
        str(row.get("phrase_kind") or "unknown") for row in rescue_rows
    )
    char_kind_counts = Counter(str(char.get("kind") or "unknown") for char in chars)
    accepted_digit_decisions = Counter(
        str(char.get("decision") or "unknown") for char in digit_chars
    )
    accepted_digit_sources = Counter(
        str(char.get("slot_source") or char.get("source_glyph_type") or "unknown")
        for char in digit_chars
    )
    input_digit_decisions = Counter(
        str(char.get("decision") or "unknown") for char in digit_candidates
    )
    input_digit_sources = Counter(
        str(char.get("slot_source") or char.get("source_glyph_type") or "unknown")
        for char in digit_candidates
    )
    symbol_sources = Counter(
        str(char.get("source_glyph_type") or "unknown") for char in symbol_chars
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "page_index": int(page_index),
        "consumer_allowed": CONSUMER_ALLOWED,
        "vector_numeric_phrases_v1": phrase_rows,
        "vector_numeric_phrase_stats_v1": {
            "schema_version": STATS_SCHEMA_VERSION,
            "page_index": int(page_index),
            "row_count": len(phrase_rows),
            "character_count": len(chars),
            "input_digit_match_count": len(digit_rows),
            "accepted_digit_count": len(digit_chars),
            "symbol_char_count": len(symbol_chars),
            "segment_count": len(segments),
            "rejected_segment_count": rejected_segments,
            "segment_rescue_count": len(rescue_rows),
            "counts_by_phrase_kind": dict(sorted(phrase_kind_counts.items())),
            "counts_by_segment_rescue_kind": dict(
                sorted(rescue_kind_counts.items())
            ),
            "counts_by_char_kind": dict(sorted(char_kind_counts.items())),
            "counts_by_accepted_digit_decision": dict(
                sorted(accepted_digit_decisions.items())
            ),
            "counts_by_accepted_digit_source": dict(
                sorted(accepted_digit_sources.items())
            ),
            "counts_by_input_digit_decision": dict(
                sorted(input_digit_decisions.items())
            ),
            "counts_by_input_digit_source": dict(sorted(input_digit_sources.items())),
            "counts_by_symbol_glyph_type": dict(sorted(symbol_sources.items())),
            "counts_by_rejected_segment_reason": dict(
                sorted(rejected_reasons.items())
            ),
            "symbol_adjacency_by_type": _symbol_adjacency_by_type(
                symbol_chars,
                digit_candidates=digit_candidates,
                accepted_digit_chars=digit_chars,
            ),
            "consumer_allowed_count": sum(
                1 for row in phrase_rows if row.get("consumer_allowed")
            ),
        },
    }


def _digit_chars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    for row in rows:
        decision = str(row.get("decision") or "")
        text = str(row.get("predicted_text") or "")
        if not _digit_row_accepted(row):
            continue
        if len(text) != 1 or not text.isdigit():
            continue
        if bool(row.get("consumer_allowed")):
            continue
        bbox = _bbox_from_any(row)
        if bbox is None:
            continue
        chars.append({
            "kind": "digit",
            "text": text,
            "bbox": bbox,
            "source_match_id": str(row.get("match_id") or ""),
            "source_token_id": str(row.get("source_token_id") or ""),
            "decision": decision,
            "confidence": _float_or(row.get("confidence"), 1.0),
            "source_glyph_type": str(row.get("source_glyph_type") or "digit"),
            "slot_source": _slot_source(row),
        })
    return chars


def _digit_candidate_chars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    for row in rows:
        if bool(row.get("consumer_allowed")):
            continue
        bbox = _bbox_from_any(row)
        if bbox is None:
            continue
        decision = str(row.get("decision") or "")
        text = str(row.get("predicted_text") or "")
        chars.append({
            "kind": "digit_candidate",
            "text": text if len(text) == 1 and text.isdigit() else "",
            "bbox": bbox,
            "source_match_id": str(row.get("match_id") or ""),
            "source_token_id": str(row.get("source_token_id") or ""),
            "decision": decision,
            "confidence": _float_or(row.get("confidence"), 0.0),
            "source_glyph_type": str(row.get("source_glyph_type") or "digit"),
            "slot_source": _slot_source(row),
            "accepted": decision in ACCEPTED_DIGIT_DECISIONS,
            "structural_accepted": _structural_digit_row_accepted(row),
        })
    return chars


def _symbol_chars(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chars: list[dict[str, Any]] = []
    for token in tokens:
        glyph_type = str(token.get("glyph_type") or "")
        text = SYMBOL_TEXT_BY_GLYPH_TYPE.get(glyph_type)
        if text is None:
            continue
        if bool(token.get("consumer_allowed")):
            continue
        bbox = _bbox_from_any(token)
        if bbox is None:
            continue
        chars.append({
            "kind": "symbol",
            "text": text,
            "bbox": bbox,
            "source_match_id": "",
            "source_token_id": str(token.get("token_id") or ""),
            "decision": "",
            "confidence": _float_or(token.get("confidence"), 0.0),
            "source_glyph_type": glyph_type,
        })
    return chars


def _symbol_adjacency_by_type(
    symbol_chars: list[dict[str, Any]],
    *,
    digit_candidates: list[dict[str, Any]],
    accepted_digit_chars: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for symbol in symbol_chars:
        glyph_type = str(symbol.get("source_glyph_type") or "unknown")
        summary = summaries.setdefault(
            glyph_type,
            {
                "symbol_count": 0,
                "same_line_any_digit_match_count": 0,
                "same_line_accepted_digit_count": 0,
                "joinable_any_digit_match_count": 0,
                "joinable_accepted_digit_count": 0,
                "_nearest_any": None,
                "_nearest_accepted": None,
            },
        )
        summary["symbol_count"] += 1
        any_same_line = [
            digit for digit in digit_candidates
            if _same_text_line(symbol["bbox"], digit["bbox"])
        ]
        accepted_same_line = [
            digit for digit in accepted_digit_chars
            if _same_text_line(symbol["bbox"], digit["bbox"])
        ]
        if any_same_line:
            summary["same_line_any_digit_match_count"] += 1
        if accepted_same_line:
            summary["same_line_accepted_digit_count"] += 1
        if any(_joinable_pair(symbol["bbox"], digit["bbox"]) for digit in any_same_line):
            summary["joinable_any_digit_match_count"] += 1
        if any(_joinable_pair(symbol["bbox"], digit["bbox"]) for digit in accepted_same_line):
            summary["joinable_accepted_digit_count"] += 1
        nearest_any = _nearest_distance(symbol, digit_candidates)
        nearest_accepted = _nearest_distance(symbol, accepted_digit_chars)
        summary["_nearest_any"] = _min_optional(summary["_nearest_any"], nearest_any)
        summary["_nearest_accepted"] = _min_optional(
            summary["_nearest_accepted"],
            nearest_accepted,
        )

    finalized: dict[str, dict[str, Any]] = {}
    for glyph_type, summary in sorted(summaries.items()):
        finalized[glyph_type] = {
            "symbol_count": int(summary["symbol_count"]),
            "same_line_any_digit_match_count": int(
                summary["same_line_any_digit_match_count"]
            ),
            "same_line_accepted_digit_count": int(
                summary["same_line_accepted_digit_count"]
            ),
            "joinable_any_digit_match_count": int(
                summary["joinable_any_digit_match_count"]
            ),
            "joinable_accepted_digit_count": int(
                summary["joinable_accepted_digit_count"]
            ),
            "nearest_any_digit_match_distance_min": _round_optional(
                summary["_nearest_any"]
            ),
            "nearest_accepted_digit_distance_min": _round_optional(
                summary["_nearest_accepted"]
            ),
        }
    return finalized


def _character_segments(chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    for char in sorted(chars, key=lambda item: (_center_y(item["bbox"]), item["bbox"]["x"])):
        for line in lines:
            if _same_text_line(char["bbox"], line["bbox"]):
                line["chars"].append(char)
                line["bbox"] = _union_bbox(line["bbox"], char["bbox"])
                break
        else:
            lines.append({"bbox": dict(char["bbox"]), "chars": [char]})

    segments: list[list[dict[str, Any]]] = []
    for line in lines:
        current: list[dict[str, Any]] = []
        for char in sorted(line["chars"], key=lambda item: item["bbox"]["x"]):
            if not current:
                current.append(char)
                continue
            previous = current[-1]
            if _same_text_line(previous["bbox"], char["bbox"]) and _join_gap_ok(
                previous["bbox"],
                char["bbox"],
            ):
                current.append(char)
            else:
                segments.append(current)
                current = [char]
        if current:
            segments.append(current)
    return segments


def _row_from_segment(
    segment: list[dict[str, Any]],
    *,
    page_index: int,
    order: int,
    source: str = "vector_numeric_phrase_builder",
) -> dict[str, Any] | None:
    if not segment:
        return None
    text = "".join(char["text"] for char in segment)
    phrase_kind = _phrase_kind(text)
    if phrase_kind is None:
        return None
    digit_count = sum(1 for char in segment if char["kind"] == "digit")
    if digit_count <= 0:
        return None
    bbox: dict[str, float] | None = None
    for char in segment:
        bbox = _union_bbox(bbox, char["bbox"]) if bbox else dict(char["bbox"])
    if bbox is None:
        return None
    source_match_ids = [
        char["source_match_id"]
        for char in segment
        if char.get("source_match_id")
    ]
    source_token_ids = [
        char["source_token_id"]
        for char in segment
        if char.get("source_token_id")
    ]
    confidence = min(float(char.get("confidence") or 0.0) for char in segment)
    decisions = Counter(
        str(char.get("decision") or "symbol")
        for char in segment
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phrase_id": f"p{int(page_index) + 1:03d}_vnp_{int(order):06d}",
        "page_index": int(page_index),
        "text": text,
        "phrase_kind": phrase_kind,
        "bbox": _round_bbox(bbox),
        "oriented_quad": _quad_from_bbox(bbox),
        "orientation": "H",
        "character_count": len(segment),
        "digit_count": digit_count,
        "symbol_count": len(segment) - digit_count,
        "confidence": round(confidence, 4),
        "source_match_ids": source_match_ids,
        "source_token_ids": source_token_ids,
        "strict_checks": {
            "source": source,
            "accepted_digit_decisions": sorted(ACCEPTED_DIGIT_DECISIONS),
            "structural_accepted_digit_decision": (
                STRUCTURAL_ACCEPTED_DIGIT_DECISION
            ),
            "structural_accepted_slot_sources": sorted(
                STRUCTURAL_ACCEPTED_SLOT_SOURCES
            ),
            "counts_by_decision": dict(sorted(decisions.items())),
            "text_pattern_valid": True,
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "source_segments": [
            {
                "kind": "vector_digit_match" if char["kind"] == "digit" else "vector_glyph_token",
                "match_id": char["source_match_id"],
                "token_id": char["source_token_id"],
                "text": char["text"],
                "glyph_type": char["source_glyph_type"],
            }
            for char in segment
        ],
        "drop_reason": None,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _rescued_symbol_phrase_rows(
    segments: list[list[dict[str, Any]]],
    *,
    page_index: int,
    start_order: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for segment in segments:
        if not _single_plus_minus_segment(segment):
            continue
        right = _nearest_plain_number_segment(
            segments,
            segment,
            side="right",
        )
        if right is None:
            continue
        left = _nearest_plain_number_segment(
            segments,
            segment,
            side="left",
        )
        candidates: list[list[dict[str, Any]]] = []
        if left is not None:
            candidates.append(left + segment + right)
        candidates.append(segment + right)
        for candidate in candidates:
            text = "".join(char["text"] for char in candidate)
            if _phrase_kind(text) not in {
                "nominal_plus_minus_tolerance",
                "plus_minus_tolerance",
            }:
                continue
            key = tuple(
                str(char.get("source_match_id") or char.get("source_token_id") or "")
                for char in candidate
            )
            if key in seen:
                continue
            row = _row_from_segment(
                candidate,
                page_index=page_index,
                order=start_order + len(rows),
                source="vector_numeric_phrase_segment_rescue",
            )
            if row is None:
                continue
            row["strict_checks"]["segment_rescue"] = True
            seen.add(key)
            rows.append(row)
            break
    return rows


def _single_plus_minus_segment(segment: list[dict[str, Any]]) -> bool:
    return (
        len(segment) == 1
        and str(segment[0].get("kind") or "") == "symbol"
        and str(segment[0].get("source_glyph_type") or "") == "plus_minus"
        and str(segment[0].get("text") or "") == "±"
    )


def _nearest_plain_number_segment(
    segments: list[list[dict[str, Any]]],
    source_segment: list[dict[str, Any]],
    *,
    side: str,
) -> list[dict[str, Any]] | None:
    source_bbox = _segment_bbox(source_segment)
    if source_bbox is None:
        return None
    source_center_x = _center_x(source_bbox)
    candidates: list[tuple[float, list[dict[str, Any]]]] = []
    for segment in segments:
        if segment is source_segment:
            continue
        text = "".join(char["text"] for char in segment)
        if _phrase_kind(text) != "plain_number":
            continue
        if side == "right":
            endpoint = segment[0]
            endpoint_center_x = _center_x(endpoint["bbox"])
            if endpoint_center_x <= source_center_x:
                continue
        elif side == "left":
            endpoint = segment[-1]
            endpoint_center_x = _center_x(endpoint["bbox"])
            if endpoint_center_x >= source_center_x:
                continue
        else:
            continue
        if not _joinable_pair(source_bbox, endpoint["bbox"]):
            continue
        candidates.append((abs(endpoint_center_x - source_center_x), segment))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _segment_rejection_reason(segment: list[dict[str, Any]]) -> str:
    if not segment:
        return "empty_segment"
    digit_count = sum(1 for char in segment if char["kind"] == "digit")
    if digit_count <= 0:
        return "no_digit_chars"
    text = "".join(char["text"] for char in segment)
    if _phrase_kind(text) is None:
        return "unsupported_text_pattern"
    bbox = _segment_bbox(segment)
    if bbox is None:
        return "invalid_bbox"
    return "unknown"


def _segment_bbox(segment: list[dict[str, Any]]) -> dict[str, float] | None:
    bbox: dict[str, float] | None = None
    for char in segment:
        char_bbox = char.get("bbox")
        if not isinstance(char_bbox, dict):
            return None
        bbox = _union_bbox(bbox, char_bbox) if bbox else dict(char_bbox)
    return bbox


def _phrase_kind(text: str) -> str | None:
    if _NOMINAL_PLUS_MINUS_RE.match(text):
        return "nominal_plus_minus_tolerance"
    if _NOMINAL_ASYMMETRIC_TOLERANCE_RE.match(text):
        return "nominal_asymmetric_tolerance"
    if _NOMINAL_SIGNED_TOLERANCE_RE.match(text):
        return "nominal_signed_tolerance"
    diameter_signed = _DIAMETER_SIGNED_TOLERANCE_RE.match(text)
    if diameter_signed:
        return (
            "diameter_plus_tolerance"
            if diameter_signed.group(1) == "+"
            else "diameter_minus_tolerance"
        )
    if _NUMBER_RE.match(text) or _LEADING_DECIMAL_RE.match(text):
        return "plain_number"
    if _PLUS_MINUS_RE.match(text):
        return "plus_minus_tolerance"
    if _DIAMETER_RE.match(text):
        return "diameter_number"
    if _ANGLE_RE.match(text):
        return "angle_number"
    return None


def _digit_row_accepted(row: dict[str, Any]) -> bool:
    decision = str(row.get("decision") or "")
    if decision in ACCEPTED_DIGIT_DECISIONS:
        return True
    return _structural_digit_row_accepted(row)


def _structural_digit_row_accepted(row: dict[str, Any]) -> bool:
    return (
        str(row.get("decision") or "") == STRUCTURAL_ACCEPTED_DIGIT_DECISION
        and _slot_source(row) in STRUCTURAL_ACCEPTED_SLOT_SOURCES
    )


def _same_text_line(a: dict[str, float], b: dict[str, float]) -> bool:
    if _vertical_overlap_ratio(a, b) >= VERTICAL_OVERLAP_RATIO:
        return True
    return abs(_center_y(a) - _center_y(b)) <= max(float(a["h"]), float(b["h"])) * 0.65


def _join_gap_ok(left: dict[str, float], right: dict[str, float]) -> bool:
    gap = float(right["x"]) - (float(left["x"]) + float(left["w"]))
    if gap < 0.0:
        return True
    limit = min(MAX_JOIN_GAP_PT, max(float(left["h"]), float(right["h"])) * MAX_JOIN_GAP_FACTOR)
    return gap <= limit


def _joinable_pair(a: dict[str, float], b: dict[str, float]) -> bool:
    left, right = (a, b) if float(a["x"]) <= float(b["x"]) else (b, a)
    return _same_text_line(left, right) and _join_gap_ok(left, right)


def _vertical_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    ay0 = float(a["y"])
    ay1 = ay0 + float(a["h"])
    by0 = float(b["y"])
    by1 = by0 + float(b["h"])
    overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
    return overlap / max(min(float(a["h"]), float(b["h"])), 1e-9)


def _center_y(bbox: dict[str, float]) -> float:
    return float(bbox["y"]) + float(bbox["h"]) / 2.0


def _center_x(bbox: dict[str, float]) -> float:
    return float(bbox["x"]) + float(bbox["w"]) / 2.0


def _center_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return ((_center_x(a) - _center_x(b)) ** 2 + (_center_y(a) - _center_y(b)) ** 2) ** 0.5


def _nearest_distance(
    source: dict[str, Any],
    targets: list[dict[str, Any]],
) -> float | None:
    if not targets:
        return None
    return min(_center_distance(source["bbox"], target["bbox"]) for target in targets)


def _min_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None else None


def _float_or(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _slot_source(row: dict[str, Any]) -> str:
    strict_checks = row.get("strict_checks") if isinstance(row.get("strict_checks"), dict) else {}
    source_checks = (
        row.get("source_glyph_checks")
        if isinstance(row.get("source_glyph_checks"), dict)
        else {}
    )
    return str(
        strict_checks.get("slot_source")
        or source_checks.get("source")
        or row.get("source_glyph_type")
        or "unknown"
    )


def _bbox_from_any(row: dict[str, Any]) -> dict[str, float] | None:
    raw = row.get("bbox") if isinstance(row, dict) else None
    if isinstance(raw, dict):
        return _bbox_from_any(raw)
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            x0, y0, x1, y1 = [float(value) for value in raw[:4]]
        except (TypeError, ValueError):
            return None
        return _bbox_from_xyxy(x0, y0, x1, y1)
    try:
        if all(key in row for key in ("x", "y", "w", "h")):
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])
            return {"x": x, "y": y, "w": w, "h": h} if w > 0 and h > 0 else None
        if all(key in row for key in ("x0", "y0", "x1", "y1")):
            return _bbox_from_xyxy(
                float(row["x0"]),
                float(row["y0"]),
                float(row["x1"]),
                float(row["y1"]),
            )
    except (TypeError, ValueError):
        return None
    return None


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> dict[str, float] | None:
    x = min(x0, x1)
    y = min(y0, y1)
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _union_bbox(left: dict[str, float] | None, right: dict[str, float]) -> dict[str, float]:
    if left is None:
        return dict(right)
    x0 = min(float(left["x"]), float(right["x"]))
    y0 = min(float(left["y"]), float(right["y"]))
    x1 = max(float(left["x"]) + float(left["w"]), float(right["x"]) + float(right["w"]))
    y1 = max(float(left["y"]) + float(left["h"]), float(right["y"]) + float(right["h"]))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {
        "x": round(float(bbox["x"]), 3),
        "y": round(float(bbox["y"]), 3),
        "w": round(float(bbox["w"]), 3),
        "h": round(float(bbox["h"]), 3),
    }


def _quad_from_bbox(bbox: dict[str, float]) -> list[list[float]]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    return [
        [round(x, 3), round(y, 3)],
        [round(x + w, 3), round(y, 3)],
        [round(x + w, 3), round(y + h, 3)],
        [round(x, 3), round(y + h, 3)],
    ]
