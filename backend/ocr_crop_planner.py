"""Candidate-guided OCR crop planning.

This module is intentionally small and feature-flagged by the pipeline. It
turns DimensionCandidate v0 boxes into bounded OCR crops, then returns normal
OCR result rows so the existing dedup and assembler chain can consume them.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

import numpy as np

from ocr_engine_utils import (
    _extract_roi,
    _get_ocr,
    pixel_bbox_to_pdf,
    quad_to_bbox,
    rotate_bbox_back,
)


_DIMENSIONISH_RE = re.compile(
    r"(?ix)"
    r"("
    r"[±+\-/°ØøΦ⌀∅×x]"
    r"|^[RrMm]\s*\d"
    r"|\d+\.\d+"
    r"|\d+\s*(?:H|K|P|G|M|N|R|T|J)\d*\b"
    r"|\b\d{2,3}\b"
    r")"
)
_LONG_ALPHA_RE = re.compile(r"[A-Za-z]{4,}")
_LIMIT_NOTE_RE = re.compile(r"(?i)\b(?:MAX|MIN)\.?\s*[O0]?\.\s*\d")
_EARLY_STOP_MIN_ATTEMPTS = 10
_EARLY_STOP_GRAMMAR_REJECT_RATE = 0.70


def is_dimension_like_candidate_text(text: str) -> bool:
    """Conservative grammar gate for low-priority candidate OCR text."""
    value = _normalize_candidate_text(text)
    if not value or len(value) > 40 or not any(ch.isdigit() for ch in value):
        return False
    upper = value.upper()
    if "GB/T" in upper or "%" in value:
        return False
    if _LIMIT_NOTE_RE.search(value):
        return True
    if re.search(r"[\u3000-\u303f\u3400-\u9fff]", value):
        return False
    if _LONG_ALPHA_RE.search(value):
        return False
    return bool(_DIMENSIONISH_RE.search(value))


def build_candidate_crop_plans(
    candidates: list[dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
    low_priority_budget_min: int = 20,
    low_priority_budget_ratio: float = 0.30,
    max_plan_count: int | None = None,
    diag: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build bounded OCR crop plans from active dimension candidates."""
    eligible: list[tuple[int, int, dict[str, Any]]] = []
    for order, candidate in enumerate(candidates or []):
        if candidate.get("reject_reason"):
            _bump(diag, "skipped_rejected")
            continue
        if candidate.get("type") != "dimension":
            _bump(diag, "skipped_non_dimension")
            continue
        subtype = str(candidate.get("subtype") or "")
        if subtype == "feature_control_frame":
            _bump(diag, "skipped_gdt")
            continue
        bbox = _valid_bbox(candidate.get("bbox"))
        if not bbox:
            _bump(diag, "skipped_invalid_bbox")
            continue
        eligible.append((_score_candidate_for_planning(candidate), order, candidate))

    eligible.sort(key=lambda item: (-item[0], item[1]))
    high_count = sum(1 for _score, _order, c in eligible if c.get("priority_hint") != "low")
    low_budget = max(
        int(low_priority_budget_min),
        int(math.ceil(float(low_priority_budget_ratio) * high_count)),
    )
    low_used = 0
    plans: list[dict[str, Any]] = []

    score_distribution: dict[str, int] = {}
    for planning_score, _order, candidate in eligible:
        score_key = str(planning_score)
        score_distribution[score_key] = score_distribution.get(score_key, 0) + 1
        if max_plan_count is not None and len(plans) >= max(0, int(max_plan_count)):
            _bump(diag, "skipped_plan_cap")
            continue
        priority = str(candidate.get("priority_hint") or "normal")
        if priority == "low":
            if low_used >= low_budget:
                _bump(diag, "skipped_low_priority_budget")
                continue
            low_used += 1
        bbox = _valid_bbox(candidate.get("bbox"))
        if not bbox:
            continue
        crop_bbox = _expanded_crop_bbox(
            bbox,
            subtype=str(candidate.get("subtype") or ""),
            has_capsule=_has_source_kind(candidate, "capsule"),
            page_width=page_width,
            page_height=page_height,
        )
        plan = {
            "candidate_id": candidate.get("candidate_id"),
            "crop_bbox": crop_bbox,
            "rotation_deg": _rotation_for_candidate(candidate, bbox),
            "class_hint": str(candidate.get("subtype") or "unknown"),
            "max_attempts": 1 if priority == "low" else 2,
            "priority_hint": priority,
            "quality_warning_count": _quality_warning_count(candidate),
            "planning_score": planning_score,
        }
        # 显式 crop_id：page 内唯一，与 candidate_id 绑定。
        # 形成 candidate_id → crop_id 的字段级证据链锚点。
        plan["crop_id"] = f"{plan['candidate_id']}_crop_{len(plans):02d}"
        plans.append(plan)

    if diag is not None:
        diag["candidate_count"] = len(candidates or [])
        diag["eligible_count"] = len(eligible)
        diag["plan_count"] = len(plans)
        diag["high_priority_count"] = high_count
        diag["low_priority_budget"] = low_budget
        diag["low_priority_planned"] = low_used
        diag["score_distribution"] = dict(sorted(
            score_distribution.items(),
            key=lambda item: int(item[0]),
            reverse=True,
        ))
    return plans


def run_candidate_guided_ocr(
    page_img,
    crop_plans: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    dpi: int,
    verbose: bool = False,
    deadline: float | None = None,
    deadline_counter_key: str = "stopped_deadline",
    diag: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run OCR for candidate crop plans and return standard OCR rows."""
    if not page_img or not crop_plans:
        return []

    candidates_by_id = {
        c.get("candidate_id"): c
        for c in candidates or []
        if c.get("candidate_id")
    }
    ocr = _get_ocr()
    results: list[dict[str, Any]] = []
    attempts_this_page = 0
    grammar_rejects_this_page = 0
    grammar_accepts_this_page = 0

    for plan in crop_plans:
        if deadline is not None and time.monotonic() >= deadline:
            _bump(diag, deadline_counter_key)
            break
        grammar_checked_this_page = grammar_rejects_this_page + grammar_accepts_this_page
        if (
            attempts_this_page >= _EARLY_STOP_MIN_ATTEMPTS
            and grammar_checked_this_page >= _EARLY_STOP_MIN_ATTEMPTS
            and grammar_accepts_this_page == 0
            and grammar_rejects_this_page / max(1, grammar_checked_this_page) >= _EARLY_STOP_GRAMMAR_REJECT_RATE
        ):
            _bump(diag, "early_stopped_high_reject_rate")
            if diag is not None:
                diag["early_stop_attempts"] = attempts_this_page
                diag["early_stop_grammar_rejects"] = grammar_rejects_this_page
                diag["early_stop_grammar_accepts"] = grammar_accepts_this_page
            break
        candidate = candidates_by_id.get(plan.get("candidate_id"))
        if candidate is None:
            _bump(diag, "missing_candidate")
            continue
        region = plan.get("crop_bbox") or {}
        try:
            roi_img, origin_px, angle_applied, scale_up = _extract_roi(
                page_img,
                region,
                dpi,
                near_angle_deg=float(plan.get("rotation_deg") or 0.0),
            )
        except Exception as exc:
            _record_attempt(candidate, plan, accepted=False, reason=f"crop_error:{exc}")
            _bump(diag, "crop_errors")
            continue

        ocr_calls_before = int(diag.get("ocr_calls") or 0) if diag is not None else 0
        ocr_lines = _run_ocr_attempts(
            ocr,
            roi_img,
            angle_applied=angle_applied,
            max_attempts=int(plan.get("max_attempts") or 1),
            diag=diag,
        )
        ocr_lines = _merge_max_min_suffix_ocr_lines(ocr_lines)
        if diag is not None:
            ocr_calls_after = int(diag.get("ocr_calls") or 0)
            attempts_this_page += max(1, ocr_calls_after - ocr_calls_before)
        else:
            attempts_this_page += 1
        if not ocr_lines:
            _record_attempt(candidate, plan, accepted=False, reason="empty_ocr")
            _bump(diag, "empty_outputs")
            continue

        accepted_for_candidate = 0
        for quad_in_roi, text, conf, output_angle in ocr_lines:
            text = _normalize_candidate_text(text)
            if not text:
                continue
            _bump(diag, "raw_lines")
            conf_floor = 0.15 if any(ch.isdigit() for ch in text) else 0.30
            if float(conf) < conf_floor:
                _record_attempt(
                    candidate,
                    plan,
                    text=text,
                    confidence=float(conf),
                    accepted=False,
                    reason="low_confidence",
                )
                _bump(diag, "rejected_low_conf")
                continue
            if not is_dimension_like_candidate_text(text):
                _record_attempt(
                    candidate,
                    plan,
                    text=text,
                    confidence=float(conf),
                    accepted=False,
                    reason="grammar_reject",
                )
                _bump(diag, "rejected_grammar")
                grammar_rejects_this_page += 1
                continue

            bbox_pdf = _ocr_quad_to_pdf_bbox(
                quad_in_roi,
                origin_px=origin_px,
                dpi=dpi,
                scale_up=scale_up,
                roi_size=roi_img.size,
                angle_applied=output_angle,
            )
            result = {
                "text": text,
                "confidence": float(conf),
                "bbox_in_pdf": bbox_pdf,
                "font_height_approx": bbox_pdf["h"],
                "source_region_id": None,
                "orientation": float(output_angle),
                "low_confidence_region": plan.get("priority_hint") == "low",
                "source": "candidate_guided_ocr",
                "source_candidate_id": candidate.get("candidate_id"),
                "source_crop_id": plan.get("crop_id"),
                "candidate_priority_hint": plan.get("priority_hint"),
                "candidate_quality_warning_count": int(plan.get("quality_warning_count") or 0),
                "candidate_crop_bbox": dict(plan.get("crop_bbox") or {}),
                "candidate_planning_score": int(plan.get("planning_score") or 0),
                "candidate_evidence_summary": dict(candidate.get("evidence_summary") or {}),
                "candidate_has_geometry_evidence": _has_geometry_evidence(candidate),
            }
            results.append(result)
            accepted_for_candidate += 1
            grammar_accepts_this_page += 1
            _record_attempt(
                candidate,
                plan,
                text=text,
                confidence=float(conf),
                accepted=True,
                reason="accepted",
            )
            if not candidate.get("text"):
                candidate["text"] = text
            if verbose:
                print(
                    "[candidate_guided_ocr] "
                    f"candidate={candidate.get('candidate_id')} text={text!r} "
                    f"conf={float(conf):.2f}"
                )

        if accepted_for_candidate:
            _bump(diag, "final_lines", accepted_for_candidate)
        elif not candidate.get("ocr_attempts"):
            _record_attempt(candidate, plan, accepted=False, reason="no_accepted_lines")

    return results


def _run_ocr_attempts(
    ocr,
    roi_img,
    *,
    angle_applied: float,
    max_attempts: int,
    diag: dict[str, Any] | None,
) -> list[tuple[list, str, float, float]]:
    attempts: list[tuple[Any, float]] = []
    if abs(angle_applied) == 90.0:
        rotated = [
            (roi_img.rotate(-90, expand=True, fillcolor=(255, 255, 255)), -90.0),
            (roi_img.rotate(90, expand=True, fillcolor=(255, 255, 255)), 90.0),
        ]
        attempts = rotated[:max(1, min(2, max_attempts))]
    elif abs(angle_applied) > 1e-6:
        attempts = [(roi_img, angle_applied)]
        if max_attempts >= 2:
            attempts.append((
                roi_img.rotate(180, expand=True, fillcolor=(255, 255, 255)),
                angle_applied + 180.0,
            ))
    else:
        attempts = [(roi_img, 0.0)]

    lines: list[tuple[list, str, float, float]] = []
    for image, output_angle in attempts:
        try:
            _bump(diag, "ocr_calls")
            output = ocr.ocr(np.array(image), cls=False)
        except Exception:
            _bump(diag, "ocr_errors")
            continue
        if not output or output[0] is None:
            continue
        for line in output[0]:
            if not line or len(line) < 2:
                continue
            quad, (text, conf) = line[0], line[1]
            lines.append((quad, text, float(conf), output_angle))
    return lines


def _merge_max_min_suffix_ocr_lines(
    lines: list[tuple[list, str, float, float]]
) -> list[tuple[list, str, float, float]]:
    """Merge a numeric OCR line with a separate MAX/MIN limit suffix.

    Paddle often separates compact limit suffixes in unrotated vector-text
    crops.  Joining only a single numeric line with a single standalone suffix
    preserves real MAX/MIN dimensions without broadly rotating text groups.
    """
    if len(lines) < 2 or len(lines) > 3:
        return lines

    numeric_indices: list[int] = []
    suffix_indices: list[int] = []
    for idx, (_quad, text, _conf, _angle) in enumerate(lines):
        value = _normalize_candidate_text(text).upper()
        if re.fullmatch(r"(?:R?\d+\.\d+|\d{2,})", value):
            numeric_indices.append(idx)
        elif re.fullmatch(r"(?:MAX|MIN)\.?", value):
            suffix_indices.append(idx)

    if len(numeric_indices) != 1 or len(suffix_indices) != 1:
        return lines

    num_idx = numeric_indices[0]
    suffix_idx = suffix_indices[0]
    num_quad, num_text, num_conf, num_angle = lines[num_idx]
    suffix_quad, suffix_text, suffix_conf, suffix_angle = lines[suffix_idx]
    if abs(float(num_angle) - float(suffix_angle)) > 1e-6:
        return lines

    merged = (
        _union_quads(num_quad, suffix_quad),
        f"{_normalize_candidate_text(num_text)}{_normalize_candidate_text(suffix_text).upper().rstrip('.')}",
        min(float(num_conf), float(suffix_conf)),
        float(num_angle),
    )
    return [
        merged if idx == num_idx else line
        for idx, line in enumerate(lines)
        if idx != suffix_idx
    ]


def _union_quads(a: list, b: list) -> list[list[float]]:
    points: list[tuple[float, float]] = []
    for quad in (a or [], b or []):
        for pt in quad or []:
            try:
                points.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError, IndexError):
                continue
    if not points:
        return a or b
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _ocr_quad_to_pdf_bbox(
    quad_in_roi: list,
    *,
    origin_px: tuple[int, int],
    dpi: int,
    scale_up: int,
    roi_size: tuple[int, int],
    angle_applied: float,
) -> dict[str, float]:
    quad_unscaled = [[pt[0] / scale_up, pt[1] / scale_up] for pt in quad_in_roi]
    bbox_in_roi = quad_to_bbox(quad_unscaled)
    if angle_applied != 0.0:
        roi_w = int(round(roi_size[0] / max(1, scale_up)))
        roi_h = int(round(roi_size[1] / max(1, scale_up)))
        bbox_in_roi = rotate_bbox_back(bbox_in_roi, angle_applied, roi_w, roi_h)
    bbox_page_px = {
        "x": bbox_in_roi["x"] + origin_px[0],
        "y": bbox_in_roi["y"] + origin_px[1],
        "w": bbox_in_roi["w"],
        "h": bbox_in_roi["h"],
    }
    return pixel_bbox_to_pdf(bbox_page_px, dpi)


def _expanded_crop_bbox(
    bbox: dict[str, float],
    *,
    subtype: str,
    has_capsule: bool,
    page_width: float,
    page_height: float,
) -> dict[str, float]:
    if subtype == "unknown" and has_capsule:
        pads = (0.0, 0.0, 0.0, 0.0)
    elif subtype == "diameter":
        pads = (14.0, 8.0, 12.0, 8.0)
    else:
        pads = (12.0, 8.0, 12.0, 8.0)
    left, top, right, bottom = pads
    return _clamp_bbox(
        {
            "x": bbox["x"] - left,
            "y": bbox["y"] - top,
            "w": bbox["w"] + left + right,
            "h": bbox["h"] + top + bottom,
        },
        page_width=page_width,
        page_height=page_height,
    )


def _rotation_for_bbox(bbox: dict[str, float]) -> float:
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    if width > 0.0 and height >= width * 1.4:
        return 90.0
    return 0.0


def _rotation_for_candidate(candidate: dict[str, Any], bbox: dict[str, float]) -> float:
    for node in candidate.get("source_nodes") or []:
        if node.get("kind") != "dimension_line":
            continue
        orientation = str(node.get("orientation") or "").upper()
        if orientation not in {"V", "D"}:
            continue
        try:
            angle = float(node.get("angle_deg"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(angle):
            continue
        return angle % 180.0
    return _rotation_for_bbox(bbox)


def _valid_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        bbox = {
            "x": float(value.get("x")),
            "y": float(value.get("y")),
            "w": float(value.get("w")),
            "h": float(value.get("h")),
        }
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in bbox.values()):
        return None
    if bbox["w"] <= 0.0 or bbox["h"] <= 0.0:
        return None
    return bbox


def _clamp_bbox(
    bbox: dict[str, float],
    *,
    page_width: float,
    page_height: float,
) -> dict[str, float]:
    x0 = max(0.0, float(bbox["x"]))
    y0 = max(0.0, float(bbox["y"]))
    x1 = min(float(page_width), float(bbox["x"]) + float(bbox["w"]))
    y1 = min(float(page_height), float(bbox["y"]) + float(bbox["h"]))
    return {
        "x": round(x0, 2),
        "y": round(y0, 2),
        "w": round(max(0.0, x1 - x0), 2),
        "h": round(max(0.0, y1 - y0), 2),
    }


def _has_source_kind(candidate: dict[str, Any], kind: str) -> bool:
    return any(node.get("kind") == kind for node in candidate.get("source_nodes") or [])


def _score_candidate_for_planning(candidate: dict[str, Any]) -> int:
    evidence = candidate.get("evidence_summary") or {}
    score = 0
    if evidence.get("has_capsule") or _has_source_kind(candidate, "capsule"):
        score += 3
    if (
        evidence.get("has_dimline")
        or evidence.get("has_arrow")
        or _has_source_kind(candidate, "dimension_line")
    ):
        score += 2
    if evidence.get("has_text_region") or _has_source_kind(candidate, "text_region"):
        score += 2
    if evidence.get("inside_table"):
        if _is_plausible_vector_dimension_phrase(candidate):
            score += 1
        else:
            score -= 5
    if evidence.get("inside_title_block"):
        score -= 5
    if evidence.get("inside_note_zone"):
        score -= 3
    if candidate.get("priority_hint") == "high":
        score += 2
    return score


def _has_geometry_evidence(candidate: dict[str, Any]) -> bool:
    evidence = candidate.get("evidence_summary") or {}
    return bool(
        evidence.get("has_capsule")
        or evidence.get("has_dimline")
        or evidence.get("has_arrow")
        or _has_source_kind(candidate, "capsule")
        or _has_source_kind(candidate, "dimension_line")
    )


def _quality_warning_count(candidate: dict[str, Any]) -> int:
    if "quality_warning_count" in candidate:
        try:
            return int(candidate.get("quality_warning_count") or 0)
        except (TypeError, ValueError):
            return 0
    provenance = candidate.get("provenance") or {}
    warnings = provenance.get("quality_warnings") or []
    return len(warnings) if isinstance(warnings, list) else 0


def _is_plausible_vector_dimension_phrase(candidate: dict[str, Any]) -> bool:
    """Keep compact vector-text dimension phrases from false table masks eligible.

    Dense drawing views can be mistaken for table regions. Keep compact
    numeric phrases eligible for crop inspection rather than demoting all
    such candidates to table priority.
    """
    if str(candidate.get("subtype") or "") != "vector_text_phrase":
        return False
    bbox = _valid_bbox(candidate.get("bbox"))
    if not bbox:
        return False
    width = float(bbox.get("w") or 0.0)
    height = float(bbox.get("h") or 0.0)
    area = width * height
    compact_line = 24.0 <= width <= 125.0 and 8.0 <= height <= 24.0 and 220.0 <= area <= 2400.0
    stacked_tolerance = (
        45.0 <= width <= 95.0
        and 28.0 <= height <= 40.0
        and 1100.0 <= area <= 3400.0
    )
    vertical_stacked_tolerance = (
        8.0 <= width <= 26.0
        and 45.0 <= height <= 90.0
        and 450.0 <= area <= 2200.0
    )
    return compact_line or stacked_tolerance or vertical_stacked_tolerance


def _normalize_candidate_text(text: str) -> str:
    return (
        str(text or "")
        .strip()
        .replace(" ", "")
        .replace("。", ".")
        .replace("．", ".")
    )


def _record_attempt(
    candidate: dict[str, Any],
    plan: dict[str, Any],
    *,
    text: str | None = None,
    confidence: float | None = None,
    accepted: bool,
    reason: str,
) -> None:
    attempt = {
        "source": "candidate_guided_ocr",
        "candidate_id": candidate.get("candidate_id"),
        "priority_hint": plan.get("priority_hint"),
        "quality_warning_count": int(plan.get("quality_warning_count") or 0),
        "accepted": bool(accepted),
        "reason": reason,
    }
    if text is not None:
        attempt["text"] = text
    if confidence is not None:
        attempt["confidence"] = float(confidence)
    candidate.setdefault("ocr_attempts", []).append(attempt)


def _bump(diag: dict[str, Any] | None, key: str, n: int = 1) -> None:
    if diag is not None:
        diag[key] = diag.get(key, 0) + n
