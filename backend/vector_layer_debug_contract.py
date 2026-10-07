"""Compact, versioned diagnostics projected from strict vector stage dumps.

This module is an output-only adapter.  It neither repairs nor mutates the
authoritative R33 chain and its payload is permanently diagnostic-only.
Only stage dumps that are actually present are projected.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from typing import Any


SCHEMA_VERSION = "vector_layer_debug_v1"
COORDINATE_SPACE = "pdf_page_points_top_left"
MAX_OVERLAY_LIMIT = 2000

_STAGE_FIELDS = (
    ("directional_walk", "r92_directional_walk_dump_v1"),
    ("pitch_reader", "vector_pitch_phrase_read_dump_v1"),
    ("quality_gate", "vector_phrase_quality_gate_dump_v1"),
    ("hypothesis", "vector_phrase_hypothesis_dump_v1"),
    ("spatial_dedup", "vector_phrase_spatial_dedup_dump_v1"),
    ("l3", "vector_phrase_l3_shadow_dump_v1"),
    ("l4", "vector_phrase_l4_shadow_dump_v1"),
    ("l5", "vector_phrase_l5_shadow_dump_v1"),
)


def build_vector_layer_debug_v1(
    pipe: dict[str, Any],
    *,
    trace_id: str,
    page_index: int,
    include_overlays: bool,
    overlay_limit: int,
) -> dict[str, Any]:
    """Project compact stage diagnostics without changing chain authority."""
    _validate_call(
        pipe=pipe,
        trace_id=trace_id,
        page_index=page_index,
        include_overlays=include_overlays,
        overlay_limit=overlay_limit,
    )
    bbox_lookup = _BoundedBBoxLookup(
        pipe,
        cache_limit=max(8, min(MAX_OVERLAY_LIMIT, overlay_limit + 8)),
    )
    stages: list[dict[str, Any]] = []
    executed_sources: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for stage_id, field in _STAGE_FIELDS:
        source = pipe.get(field)
        if not isinstance(source, dict):
            continue
        if source.get("schema_version") != field:
            return build_vector_layer_debug_unavailable_v1(
                trace_id=trace_id,
                page_index=page_index,
                reason=f"{stage_id}_schema_version_mismatch",
            )
        _validate_source_identity(
            source,
            trace_id=trace_id,
            page_index=page_index,
        )
        stage = _stage_projection(
            stage_id,
            source,
            pipe=pipe,
            bbox_lookup=bbox_lookup,
        )
        stages.append(_public_stage(stage))
        if stage["status"] == "ok":
            executed_sources.append((stage_id, source, stage))
    if include_overlays:
        overlays, total_overlays = _bounded_overlays(
            executed_sources,
            bbox_lookup=bbox_lookup,
            overlay_limit=overlay_limit,
        )
    else:
        overlays = []
        total_overlays = 0
    overlays_truncated = total_overlays > len(overlays)
    base_status = "unavailable" if not stages else "ok"
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": page_index,
        "coordinate_space": COORDINATE_SPACE,
        "status": base_status,
        "unavailable_reason": (
            "no_executed_vector_layers" if not stages else None
        ),
        "stages": stages,
        "overlays": overlays,
        "limits": {
            "total": total_overlays,
            "returned": len(overlays),
            "truncated": overlays_truncated,
        },
        "authority": {
            "diagnostic_only": True,
            "consumer_allowed": False,
            "release_allowed": False,
        },
    }


def build_vector_layer_debug_unavailable_v1(
    *,
    trace_id: str,
    page_index: int,
    reason: str,
) -> dict[str, Any]:
    """Return the same exact root shape when diagnostics cannot be built."""
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("vector layer debug trace_id invalid")
    if type(page_index) is not int or page_index < 0:
        raise ValueError("vector layer debug page_index invalid")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("vector layer debug unavailable reason invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": page_index,
        "coordinate_space": COORDINATE_SPACE,
        "status": "unavailable",
        "unavailable_reason": _reason_code(reason),
        "stages": [],
        "overlays": [],
        "limits": {"total": 0, "returned": 0, "truncated": False},
        "authority": {
            "diagnostic_only": True,
            "consumer_allowed": False,
            "release_allowed": False,
        },
    }


def _validate_call(
    *,
    pipe: Any,
    trace_id: Any,
    page_index: Any,
    include_overlays: Any,
    overlay_limit: Any,
) -> None:
    if not isinstance(pipe, dict):
        raise ValueError("vector layer debug pipe invalid")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("vector layer debug trace_id invalid")
    if type(page_index) is not int or page_index < 0:
        raise ValueError("vector layer debug page_index invalid")
    if type(include_overlays) is not bool:
        raise ValueError("vector layer debug include_overlays invalid")
    if (
        type(overlay_limit) is not int
        or overlay_limit < 0
        or overlay_limit > MAX_OVERLAY_LIMIT
    ):
        raise ValueError("vector layer debug overlay_limit invalid")


def _validate_source_identity(
    source: dict[str, Any],
    *,
    trace_id: str,
    page_index: int,
) -> None:
    if (
        source.get("trace_id") != trace_id
        or source.get("page_index") != page_index
    ):
        raise ValueError("vector layer debug stage identity mismatch")


def _stage_projection(
    stage_id: str,
    source: dict[str, Any],
    *,
    pipe: dict[str, Any],
    bbox_lookup: "_BoundedBBoxLookup",
) -> dict[str, Any]:
    status = {
        "ok": "ok",
        "not_run": "not_run",
        "unavailable": "unavailable",
        "fail_closed": "error",
        "error": "error",
    }.get(str(source.get("status") or ""), "error")
    first_drop = None
    if status != "ok":
        return _empty_stage(stage_id, status=status)
    if stage_id == "directional_walk":
        row_count = _dict_row_count(source.get("phrases"))
        stats = source.get("stats") if isinstance(source.get("stats"), dict) else {}
        input_count = _count(stats.get("input_anchor_count"), row_count)
        output_count = _count(stats.get("phrase_count"), row_count)
        drop_count = _count(
            stats.get("anchor_failure_count"),
            max(0, input_count - output_count),
        )
        elapsed_ms = _elapsed_ms(pipe, "r92_directional_walk")
    elif stage_id == "pitch_reader":
        input_count = _dict_row_count(source.get("reads"))
        output_count = sum(
            row.get("status") == "read"
            for row in _iter_dict_rows(source.get("reads"))
        )
        drop_count = input_count - output_count
        first_drop = _first_drop(
            (
                row
                for row in _iter_dict_rows(source.get("reads"))
                if row.get("status") != "read"
            ),
            bbox_lookup=bbox_lookup,
            reason_keys=("fail_closed_reason", "status"),
        )
        elapsed_ms = 0.0
    elif stage_id == "quality_gate":
        input_count = _dict_row_count(source.get("rows"))
        output_count = sum(
            row.get("phrase_complete") is True
            for row in _iter_dict_rows(source.get("rows"))
        )
        drop_count = input_count - output_count
        first_drop = _first_drop(
            (
                row
                for row in _iter_dict_rows(source.get("rows"))
                if row.get("phrase_complete") is not True
            ),
            bbox_lookup=bbox_lookup,
            reason_keys=("first_failed_gate", "status"),
        )
        elapsed_ms = 0.0
    elif stage_id == "hypothesis":
        input_count = _dict_row_count(source.get("hypotheses"))
        drop_count = sum(
            row.get("route") in {
                "rejected_pre_route",
                "geometry_quarantine",
            }
            for row in _iter_dict_rows(source.get("hypotheses"))
        )
        output_count = input_count - drop_count
        first_drop = _first_drop(
            (
                _hypothesis_drop_row(row)
                for row in _iter_dict_rows(source.get("hypotheses"))
                if row.get("route") in {
                    "rejected_pre_route",
                    "geometry_quarantine",
                }
            ),
            bbox_lookup=bbox_lookup,
            reason_keys=("reason_code",),
        )
        elapsed_ms = 0.0
    elif stage_id == "spatial_dedup":
        winner_count = _dict_row_count(source.get("winner_hypotheses"))
        duplicate_count = _dict_row_count(source.get("duplicate_ledger"))
        stats = source.get("stats") if isinstance(source.get("stats"), dict) else {}
        input_count = _count(
            stats.get("eligible_input_count"),
            winner_count + duplicate_count,
        )
        output_count = _count(stats.get("winner_count"), winner_count)
        drop_count = _count(
            stats.get("duplicate_suppressed_count"),
            duplicate_count,
        )
        first_drop = _first_drop(
            (
                {
                    **row,
                    "bbox": row.get("loser_bbox"),
                    "reason_code": row.get("reason"),
                }
                for row in _iter_dict_rows(source.get("duplicate_ledger"))
            ),
            bbox_lookup=bbox_lookup,
            reason_keys=("reason_code",),
        )
        elapsed_ms = 0.0
    elif stage_id == "l3":
        dimension_count = _dict_row_count(source.get("L3_dimensions"))
        suspect_count = _dict_row_count(source.get("L3_suspects"))
        routed_count = _dict_row_count(source.get("L3_gdt_routed"))
        input_count = dimension_count + suspect_count + routed_count
        output_count = dimension_count + routed_count
        drop_count = suspect_count
        first_drop = _first_drop(
            _iter_dict_rows(source.get("L3_suspects")),
            bbox_lookup=bbox_lookup,
            reason_keys=("primary_reason", "reason"),
        )
        elapsed_ms = 0.0
    elif stage_id == "l4":
        recovered_count = _dict_row_count(source.get("L4_recovered"))
        unresolved_count = _dict_row_count(source.get("L4_unresolved_suspects"))
        input_count = recovered_count + unresolved_count
        output_count = recovered_count
        drop_count = unresolved_count
        first_drop = _first_drop(
            _iter_dict_rows(source.get("L4_unresolved_suspects")),
            bbox_lookup=bbox_lookup,
            reason_keys=("primary_reason", "recovery_status", "reason"),
        )
        elapsed_ms = 0.0
    elif stage_id == "l5":
        input_count = sum(
            _dict_row_count(source.get(collection))
            for collection in (
                "L5_review_required",
                "L5_high_conf",
                "L5_release_ready",
            )
        )
        output_count = input_count
        drop_count = 0
        elapsed_ms = 0.0
    else:
        input_count = output_count = drop_count = 0
        elapsed_ms = 0.0
    if drop_count > 0 and first_drop is None:
        return _empty_stage(stage_id, status="unavailable")
    return {
        "stage_id": stage_id,
        "status": status,
        "input_count": input_count,
        "output_count": output_count,
        "drop_count": drop_count,
        "elapsed_ms": elapsed_ms,
        "first_drop": first_drop,
    }


def _empty_stage(stage_id: str, *, status: str) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "status": status,
        "input_count": 0,
        "output_count": 0,
        "drop_count": 0,
        "elapsed_ms": 0.0,
        "first_drop": None,
    }


def _public_stage(stage: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: stage[key]
        for key in (
            "stage_id",
            "status",
            "input_count",
            "output_count",
            "drop_count",
            "elapsed_ms",
            "first_drop",
        )
    }
    first_drop = public["first_drop"]
    if isinstance(first_drop, dict):
        public["first_drop"] = {
            key: first_drop[key]
            for key in ("candidate_id", "reason_code", "bbox")
        }
    return public


class _BoundedBBoxLookup:
    """Resolve only requested candidate bboxes; never build a page-wide index."""

    def __init__(self, pipe: dict[str, Any], *, cache_limit: int) -> None:
        source = pipe.get("r92_directional_walk_dump_v1")
        self._regions = (
            source.get("phrases")
            if isinstance(source, dict)
            and isinstance(source.get("phrases"), list)
            else []
        )
        self._cache_limit = max(0, int(cache_limit))
        self._cache: dict[str, dict[str, float] | None] = {}
        self._available_ids: set[str] | None = None

    def _ensure_available_ids(self) -> set[str]:
        if self._available_ids is None:
            available = set()
            for row in self._regions:
                if not isinstance(row, dict):
                    continue
                candidate_id = _candidate_id(row)
                if candidate_id and _bbox(row.get("bbox")) is not None:
                    available.add(candidate_id)
            self._available_ids = available
        return self._available_ids

    def has(self, candidate_id: str) -> bool:
        if not candidate_id:
            return False
        cached = self._cache.get(candidate_id)
        if cached is not None:
            return True
        return candidate_id in self._ensure_available_ids()

    def get(self, candidate_id: str) -> dict[str, float] | None:
        if not candidate_id:
            return None
        if candidate_id in self._cache:
            return self._cache[candidate_id]
        if self._available_ids is not None and candidate_id not in self._available_ids:
            return None
        result = None
        for row in self._regions:
            if not isinstance(row, dict) or _candidate_id(row) != candidate_id:
                continue
            result = _bbox(row.get("bbox"))
            if result is not None:
                break
        if len(self._cache) < self._cache_limit:
            self._cache[candidate_id] = result
        return result

    def resolve_many(
        self,
        candidate_ids,
    ) -> dict[str, dict[str, float]]:
        targets = {
            value
            for value in candidate_ids
            if isinstance(value, str) and value
        }
        results = {
            value: self._cache[value]
            for value in targets
            if value in self._cache and self._cache[value] is not None
        }
        remaining = targets.difference(results)
        if remaining:
            for row in self._regions:
                if not isinstance(row, dict):
                    continue
                candidate_id = _candidate_id(row)
                if candidate_id not in remaining:
                    continue
                bbox = _bbox(row.get("bbox"))
                if bbox is None:
                    continue
                results[candidate_id] = bbox
                remaining.remove(candidate_id)
                if not remaining:
                    break
        for candidate_id in targets:
            if len(self._cache) >= self._cache_limit:
                break
            self._cache[candidate_id] = results.get(candidate_id)
        return results


def _first_drop(
    rows,
    *,
    bbox_lookup: _BoundedBBoxLookup,
    reason_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    best = None
    for row in rows:
        candidate_id = _candidate_id(row)
        reason = next(
            (
                str(row.get(key))
                for key in reason_keys
                if isinstance(row.get(key), str) and row.get(key)
            ),
            None,
        )
        if not candidate_id or not reason:
            continue
        bbox = _bbox(row.get("bbox"))
        if bbox is None and not bbox_lookup.has(candidate_id):
            continue
        candidate = (candidate_id, reason, bbox)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        return None
    raw_candidate_id, raw_reason, bbox = best
    if bbox is None:
        bbox = bbox_lookup.get(raw_candidate_id)
    if bbox is None:
        return None
    return {
        "candidate_id": _stable_token(raw_candidate_id, prefix="candidate"),
        "reason_code": _reason_code(raw_reason),
        "bbox": bbox,
        "_raw_candidate_id": raw_candidate_id,
    }


def _hypothesis_drop_row(row: dict[str, Any]) -> dict[str, Any]:
    geometry = row.get("geometry_gate")
    raw_reasons = geometry.get("reasons") if isinstance(geometry, dict) else None
    reasons = sorted(
        value
        for value in (raw_reasons if isinstance(raw_reasons, list) else [])
        if isinstance(value, str) and value
    )
    route = str(row.get("route") or "")
    reason = (
        reasons[0]
        if reasons
        else "pre_route_gate_failed"
        if route == "rejected_pre_route"
        else route or "hypothesis_rejected"
    )
    return {**row, "reason_code": reason}


def _iter_stage_overlay_specs(stage_id: str, source: dict[str, Any]):
    if stage_id == "directional_walk":
        for row in _iter_dict_rows(source.get("phrases")):
            yield row, "context", _candidate_id(row), row.get("bbox")
    elif stage_id == "pitch_reader":
        for row in _iter_dict_rows(source.get("reads")):
            accepted = row.get("status") == "read"
            label = (
                str(row.get("text") or _candidate_id(row))
                if accepted
                else str(
                    row.get("fail_closed_reason")
                    or row.get("status")
                    or "dropped"
                )
            )
            yield (
                row,
                "accepted" if accepted else "rejected",
                label,
                row.get("bbox"),
            )
    elif stage_id == "quality_gate":
        for row in _iter_dict_rows(source.get("rows")):
            accepted = row.get("phrase_complete") is True
            yield (
                row,
                "accepted" if accepted else "rejected",
                str(
                    row.get("final_text")
                    if accepted
                    else row.get("first_failed_gate")
                    or _candidate_id(row)
                ),
                row.get("bbox"),
            )
    elif stage_id == "hypothesis":
        for row in _iter_dict_rows(source.get("hypotheses")):
            route = str(row.get("route") or "unknown")
            role = (
                "rejected"
                if route in {"rejected_pre_route", "geometry_quarantine"}
                else "context"
                if route == "gdt"
                else "accepted"
            )
            yield (
                row,
                role,
                str(row.get("text") or route),
                row.get("bbox"),
            )
    elif stage_id == "spatial_dedup":
        for row in _iter_dict_rows(source.get("winner_hypotheses")):
            yield (
                row,
                "accepted",
                str(
                    row.get("canonical_text")
                    or row.get("text")
                    or _candidate_id(row)
                ),
                row.get("bbox"),
            )
        for row in _iter_dict_rows(source.get("duplicate_ledger")):
            yield (
                row,
                "rejected",
                str(row.get("reason") or "duplicate_suppressed"),
                row.get("loser_bbox"),
            )
    elif stage_id == "l3":
        for collection, role, fallback in (
            ("L3_dimensions", "accepted", "L3_dimension"),
            ("L3_suspects", "rejected", "L3_suspect"),
            ("L3_gdt_routed", "context", "L3_gdt_routed"),
        ):
            for row in _iter_dict_rows(source.get(collection)):
                yield (
                    row,
                    role,
                    str(
                        row.get("canonical_text")
                        or row.get("source_text")
                        or row.get("primary_reason")
                        or fallback
                    ),
                    row.get("bbox"),
                )
    elif stage_id == "l4":
        for collection, role, fallback in (
            ("L4_recovered", "accepted", "L4_recovered"),
            ("L4_unresolved_suspects", "rejected", "L4_unresolved"),
        ):
            for row in _iter_dict_rows(source.get(collection)):
                yield (
                    row,
                    role,
                    str(
                        row.get("canonical_text")
                        or row.get("source_text")
                        or row.get("primary_reason")
                        or row.get("recovery_status")
                        or fallback
                    ),
                    row.get("bbox"),
                )
    elif stage_id == "l5":
        for collection, role, fallback in (
            ("L5_review_required", "context", "L5_review_required"),
            ("L5_high_conf", "accepted", "L5_high_conf"),
            ("L5_release_ready", "accepted", "L5_release_ready"),
        ):
            for row in _iter_dict_rows(source.get(collection)):
                yield (
                    row,
                    role,
                    str(
                        row.get("canonical_text")
                        or row.get("source_text")
                        or fallback
                    ),
                    row.get("bbox"),
                )


def _bounded_stage_overlay_candidates(
    stage_id: str,
    source: dict[str, Any],
    *,
    first_drop: dict[str, Any] | None,
    first_drop_raw_candidate_id: str | None,
    bbox_lookup: _BoundedBBoxLookup,
    capacity: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
    total = 0
    replaced_first_drop = False

    def candidates():
        nonlocal total, replaced_first_drop
        for row, role, label, raw_bbox in _iter_stage_overlay_specs(
            stage_id,
            source,
        ):
            candidate_id = _candidate_id(row)
            if not candidate_id:
                continue
            bbox = _bbox(raw_bbox)
            if bbox is None and not bbox_lookup.has(candidate_id):
                continue
            total += 1
            normalized_label = str(label)[:160]
            if (
                first_drop_raw_candidate_id is not None
                and candidate_id == first_drop_raw_candidate_id
            ):
                replaced_first_drop = True
                continue
            bbox_key = (
                tuple(bbox[key] for key in ("x", "y", "w", "h"))
                if bbox is not None
                else (math.inf, math.inf, math.inf, math.inf)
            )
            yield {
                "stage_id": stage_id,
                "candidate_id": candidate_id,
                "role": role,
                "bbox": bbox,
                "label": normalized_label,
                "_sort_key": (candidate_id, role, normalized_label, bbox_key),
            }

    candidate_stream = candidates()
    if capacity > 0:
        selected = heapq.nsmallest(
            capacity,
            candidate_stream,
            key=lambda value: value["_sort_key"],
        )
    else:
        for _candidate in candidate_stream:
            pass
        selected = []

    first_drop_candidate = None
    if first_drop is not None:
        if not replaced_first_drop:
            total += 1
        first_drop_candidate = {
            "stage_id": stage_id,
            "candidate_id": first_drop["candidate_id"],
            "role": "first_drop",
            "bbox": first_drop["bbox"],
            "label": first_drop["reason_code"][:160],
            "_sort_key": (
                first_drop["candidate_id"],
                "first_drop",
                first_drop["reason_code"][:160],
                tuple(first_drop["bbox"][key] for key in ("x", "y", "w", "h")),
            ),
        }
    return first_drop_candidate, selected, total


def _bounded_overlays(
    executed_sources: list[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    bbox_lookup: _BoundedBBoxLookup,
    overlay_limit: int,
) -> tuple[list[dict[str, Any]], int]:
    priorities: list[dict[str, Any]] = []
    buckets: list[list[dict[str, Any]]] = []
    total = 0
    for stage_id, source, stage in executed_sources:
        first_drop, selected, stage_total = _bounded_stage_overlay_candidates(
            stage_id,
            source,
            first_drop=stage.get("first_drop"),
            first_drop_raw_candidate_id=(
                stage.get("first_drop", {}).get("_raw_candidate_id")
                if isinstance(stage.get("first_drop"), dict)
                else None
            ),
            bbox_lookup=bbox_lookup,
            capacity=overlay_limit,
        )
        if first_drop is not None:
            priorities.append(first_drop)
        buckets.append(selected)
        total += stage_total

    chosen = priorities[:overlay_limit]
    row_index = 0
    while len(chosen) < overlay_limit:
        added = False
        for bucket in buckets:
            if row_index < len(bucket):
                chosen.append(bucket[row_index])
                added = True
                if len(chosen) >= overlay_limit:
                    break
        if not added:
            break
        row_index += 1

    joined_bboxes = bbox_lookup.resolve_many(
        candidate["candidate_id"]
        for candidate in chosen
        if candidate["bbox"] is None
    )
    overlays = []
    for candidate in chosen:
        bbox = candidate["bbox"] or joined_bboxes.get(candidate["candidate_id"])
        if bbox is None:
            continue
        identity = {
            "stage_id": candidate["stage_id"],
            "candidate_id": candidate["candidate_id"],
            "role": candidate["role"],
            "bbox": bbox,
            "label": candidate["label"],
        }
        overlays.append({
            "overlay_id": "vld_" + hashlib.sha256(json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()[:24],
            "stage_id": candidate["stage_id"],
            "role": candidate["role"],
            "bbox": bbox,
            "label": candidate["label"],
        })
    return overlays, total


def _candidate_id(row: dict[str, Any]) -> str:
    for key in (
        "candidate_id",
        "phrase_id",
        "review_id",
        "loser_phrase_id",
    ):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _stable_token(value: str, *, prefix: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _reason_code(value: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9._-]+",
        "_",
        str(value).strip().lower(),
    ).strip("._-")
    if not normalized or not normalized[0].isalnum():
        normalized = "diagnostic_unavailable"
    return normalized[:128]


def _bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, float] = {}
    for key in ("x", "y", "w", "h"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        number = float(raw)
        if not math.isfinite(number):
            return None
        result[key] = number
    if result["w"] <= 0.0 or result["h"] <= 0.0:
        return None
    return result


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _dict_row_count(value: Any) -> int:
    return sum(1 for _row in _iter_dict_rows(value))


def _iter_dict_rows(value: Any):
    if not isinstance(value, list):
        return
    for row in value:
        if isinstance(row, dict):
            yield row


def _count(value: Any, fallback: int) -> int:
    return value if type(value) is int and value >= 0 else fallback


def _elapsed_ms(pipe: dict[str, Any], key: str) -> float:
    timings = pipe.get("timings")
    value = timings.get(key) if isinstance(timings, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return 0.0
    return round(number * 1000.0, 3)
