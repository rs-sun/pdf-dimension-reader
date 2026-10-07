"""Deterministic spatial/value dedup for routed vector phrase hypotheses."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from typing import Any

from r2p_dimension_format_validator import validate_dimension_text_format
from vector_phrase_chain_validation import (
    request_local_artifact_safety_reasons,
)


DUMP_SCHEMA_VERSION = "vector_phrase_spatial_dedup_dump_v1"
DECISION_SCHEMA_VERSION = "vector_phrase_dedup_decision_v1"
WINNER_SCHEMA_VERSION = "vector_phrase_dedup_winner_v1"
DUPLICATE_SCHEMA_VERSION = "vector_phrase_duplicate_ledger_v1"
CONFLICT_SCHEMA_VERSION = "vector_phrase_spatial_value_conflict_v1"
BRIDGE_SCHEMA_VERSION = "vector_phrase_same_value_bridge_v1"
PINNED_IOU_THRESHOLD = 0.30
GDT_DEDUP_STATUS = "disabled_missing_compartment_ownership"
DEDUP_STATUS = "diagnostic_only_no_suppression"
CONSUMER_ALLOWED = False
REAL_GEOMETRY_SAFETY_REASONS = frozenset({
    "gdt_route_evidence_invalid",
    "read_trace_mismatch",
    "read_page_mismatch",
})


def build_vector_phrase_spatial_dedup_dump(
    *,
    hypothesis_dump: dict[str, Any],
    iou_threshold: float = PINNED_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Record spatial duplicate diagnostics without suppressing hypotheses."""
    try:
        reasons = _input_reasons(hypothesis_dump)
    except (
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        return _failed_dump(
            trace_id=(
                str(hypothesis_dump.get("trace_id") or "")
                if isinstance(hypothesis_dump, dict)
                else ""
            ),
            page_index=(
                hypothesis_dump.get("page_index")
                if isinstance(hypothesis_dump, dict)
                else -1
            ),
            reasons=["hypothesis_input_validation_exception"],
        )
    if not _pinned_threshold(iou_threshold):
        reasons.append("dedup_iou_threshold_unpinned")
    if reasons:
        return _failed_dump(
            trace_id=(
                str(hypothesis_dump.get("trace_id") or "")
                if isinstance(hypothesis_dump, dict)
                else ""
            ),
            page_index=(
                hypothesis_dump.get("page_index")
                if isinstance(hypothesis_dump, dict)
                else -1
            ),
            reasons=reasons,
        )

    rows = sorted(
        (
            _canonicalize_hypothesis_row(copy.deepcopy(row))
            for row in hypothesis_dump["hypotheses"]
        ),
        key=lambda row: (
            str(row.get("phrase_id") or ""),
            str(row.get("physical_core_digest") or ""),
        ),
    )
    input_hypothesis_semantic_digest = _digest(rows)
    canonical_text = {
        id(row): str(
            validate_dimension_text_format(str(row["text"]))["canonical_text"]
        )
        for row in rows
    }
    eligible = [
        row for row in rows if row["eligible_for_spatial_dedup"] is True
    ]
    ranked = sorted(eligible, key=_winner_sort_key)
    clusters: list[dict[str, Any]] = []
    duplicate_by_row_identity: dict[int, dict[str, Any]] = {}
    duplicate_ledger: list[dict[str, Any]] = []
    bridge_ledger: list[dict[str, Any]] = []
    for candidate in ranked:
        compatible_clusters = [
            cluster
            for cluster in clusters
            if _same_duplicate_partition(candidate, cluster["winner"])
            and canonical_text[id(candidate)]
            == canonical_text[id(cluster["winner"])]
            and all(
                _bbox_iou(candidate["bbox"], member["bbox"])
                >= PINNED_IOU_THRESHOLD
                for member in cluster["members"]
            )
        ]
        compatible_cluster_ids = {id(cluster) for cluster in compatible_clusters}
        for cluster in clusters:
            if id(cluster) in compatible_cluster_ids or not (
                _same_duplicate_partition(candidate, cluster["winner"])
                and canonical_text[id(candidate)]
                == canonical_text[id(cluster["winner"])]
            ):
                continue
            overlapping = [
                member
                for member in cluster["members"]
                if _bbox_iou(candidate["bbox"], member["bbox"])
                >= PINNED_IOU_THRESHOLD
            ]
            blocking = [
                member
                for member in cluster["members"]
                if _bbox_iou(candidate["bbox"], member["bbox"])
                < PINNED_IOU_THRESHOLD
            ]
            if overlapping and blocking:
                bridge_ledger.append(_bridge_row(
                    candidate=candidate,
                    cluster=cluster,
                    overlapping=overlapping,
                    blocking=blocking,
                    canonical_text=canonical_text[id(candidate)],
                ))
        if len(compatible_clusters) == 1:
            compatible_cluster = compatible_clusters[0]
            duplicate_winner = compatible_cluster["winner"]
            iou = _bbox_iou(candidate["bbox"], duplicate_winner["bbox"])
            ledger = _duplicate_row(
                winner=duplicate_winner,
                loser=candidate,
                canonical_text=canonical_text[id(candidate)],
                bbox_iou=iou,
            )
            duplicate_ledger.append(ledger)
            duplicate_by_row_identity[id(candidate)] = ledger
            compatible_cluster["members"].append(candidate)
            continue
        if len(compatible_clusters) > 1:
            bridge_ledger.append(_multi_cluster_bridge_row(
                candidate=candidate,
                clusters=compatible_clusters,
                canonical_text=canonical_text[id(candidate)],
            ))
            clusters.append({"winner": candidate, "members": [candidate]})
            continue

        clusters.append({"winner": candidate, "members": [candidate]})

    cluster_by_member_identity = {
        id(member): cluster
        for cluster in clusters
        for member in cluster["members"]
    }
    decisions = [
        _decision_row(
            row=row,
            canonical_text=canonical_text[id(row)],
            duplicate=duplicate_by_row_identity.get(id(row)),
            cluster=cluster_by_member_identity.get(id(row)),
        )
        for row in rows
    ]
    conflicts = _conflict_ledger(
        eligible,
        canonical_text=canonical_text,
    )
    winner_hypotheses = sorted(
        (
            _winner_record(
                cluster=cluster,
                source=member,
                canonical_text=canonical_text[id(member)],
            )
            for cluster in clusters
            for member in cluster["members"]
        ),
        key=lambda row: (
            str(row["phrase_id"]),
            str(row["physical_core_digest"]),
        ),
    )
    duplicate_ledger.sort(key=lambda row: (
        str(row["loser_phrase_id"]),
        str(row["loser_physical_core_digest"]),
    ))
    bridge_ledger.sort(key=lambda row: (
        str(row["candidate_phrase_id"]),
        str(
            row.get("cluster_winner_phrase_id")
            or row.get("compatible_cluster_winner_phrase_ids")
            or ""
        ),
    ))
    payload_for_digest = {
        "input_hypothesis_semantic_digest": str(
            input_hypothesis_semantic_digest
        ),
        "iou_threshold": PINNED_IOU_THRESHOLD,
        "gdt_dedup_status": GDT_DEDUP_STATUS,
        "dedup_status": DEDUP_STATUS,
        "decisions": decisions,
        "winner_hypotheses": winner_hypotheses,
        "duplicate_ledger": duplicate_ledger,
        "spatial_value_conflict_ledger": conflicts,
        "same_value_bridge_ledger": bridge_ledger,
    }
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": str(hypothesis_dump["trace_id"]),
        "page_index": int(hypothesis_dump["page_index"]),
        "page_context_manifest": copy.deepcopy(
            hypothesis_dump["page_context_manifest"]
        ),
        "status": "ok",
        "error": None,
        "input_hypothesis_semantic_digest": str(
            input_hypothesis_semantic_digest
        ),
        "iou_threshold": PINNED_IOU_THRESHOLD,
        "gdt_dedup_status": GDT_DEDUP_STATUS,
        "decisions": decisions,
        "winner_hypotheses": winner_hypotheses,
        "duplicate_ledger": duplicate_ledger,
        "spatial_value_conflict_ledger": conflicts,
        "same_value_bridge_ledger": bridge_ledger,
        "dedup_status": DEDUP_STATUS,
        "stats": {
            "input_hypothesis_count": len(rows),
            "eligible_input_count": len(eligible),
            "winner_count": len(winner_hypotheses),
            "duplicate_suppressed_count": 0,
            "spatial_value_conflict_count": len(conflicts),
            "same_value_bridge_count": len(bridge_ledger),
            "not_eligible_count": len(rows) - len(eligible),
            "feeds_display_l3_count": 0,
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest(payload_for_digest),
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _artifact_safety_reasons(
    value: Any,
    *,
    path: str,
) -> list[str]:
    return request_local_artifact_safety_reasons(value, path=path)


def _input_reasons(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["hypothesis_dump_not_object"]
    reasons: list[str] = []
    if (
        value.get("schema_version") != "vector_phrase_hypothesis_dump_v1"
        or value.get("status") != "ok"
        or value.get("error") is not None
        or not isinstance(value.get("hypotheses"), list)
        or value.get("feeds_display_l3") is not False
        or value.get("ocr_called") is not False
        or value.get("consumer_allowed") is not False
    ):
        reasons.append("hypothesis_dump_contract_invalid")
    if _artifact_safety_reasons(
        value,
        path="hypothesis",
    ):
        reasons.append("hypothesis_artifact_safety_invalid")
    trace_id = str(value.get("trace_id") or "")
    page_index = value.get("page_index")
    if not trace_id or type(page_index) is not int or page_index < 0:
        reasons.append("hypothesis_trace_or_page_invalid")
    if not _manifest_valid(value.get("page_context_manifest"), page_index):
        reasons.append("hypothesis_manifest_invalid")
    rows = value.get("hypotheses")
    if not isinstance(rows, list):
        return sorted(set(reasons))
    for index, row in enumerate(rows):
        row_reasons = _row_reasons(
            row,
            trace_id=trace_id,
            page_index=page_index,
            manifest=value.get("page_context_manifest"),
        )
        reasons.extend(
            f"hypothesis_row:{index}:{reason}" for reason in row_reasons
        )
    return sorted(set(reasons))


def _row_reasons(
    row: Any,
    *,
    trace_id: str,
    page_index: int,
    manifest: Any,
) -> list[str]:
    if not isinstance(row, dict):
        return ["not_object"]
    reasons: list[str] = []
    if (
        row.get("schema_version") != "vector_phrase_hypothesis_v1"
        or row.get("trace_id") != trace_id
        or row.get("page_index") != page_index
        or not str(row.get("phrase_id") or "")
        or type(row.get("physical_core_digest")) is not str
        or not row.get("physical_core_digest")
        or row.get("feeds_display_l3") is not False
        or row.get("ocr_called") is not False
        or row.get("consumer_allowed") is not False
        or "eligible_for_l3_assembly" in row
    ):
        reasons.append("identity_or_safety_invalid")
    if not _bbox_valid_in_page(row.get("bbox"), manifest):
        reasons.append("bbox_invalid")
    if not _oriented_quad_valid(
        row.get("oriented_quad"),
        row.get("bbox"),
        manifest,
    ):
        reasons.append("oriented_quad_invalid")
    if not isinstance(row.get("text"), str) or not isinstance(
        row.get("raw_text"), str
    ):
        reasons.append("text_invalid")
    geometry = row.get("geometry_gate")
    geometry_contract_valid = bool(
        isinstance(geometry, dict)
        and geometry.get("schema_version") == "vector_phrase_geometry_gate_v1"
        and type(geometry.get("passed")) is bool
        and isinstance(geometry.get("reasons"), list)
        and all(
            isinstance(reason, str) and bool(reason)
            for reason in geometry.get("reasons") or []
        )
        and geometry.get("consumer_allowed") is False
    )
    if (
        geometry_contract_valid
        and geometry.get("passed") is True
        and any(
            reason in REAL_GEOMETRY_SAFETY_REASONS
            for reason in geometry.get("reasons") or []
        )
    ):
        geometry_contract_valid = False
    geometry_passed = bool(
        geometry_contract_valid and geometry.get("passed") is True
    )
    if not _gdt_evidence_structure_valid(row.get("gdt_route_evidence")):
        reasons.append("gdt_route_evidence_invalid")
    if not geometry_contract_valid:
        reasons.append("geometry_gate_contract_invalid")
    axis_contract_valid = bool(
        _finite_number(row.get("axis_angle_deg")) is not None
        and isinstance(row.get("axis_angle_source"), str)
        and bool(row.get("axis_angle_source"))
        and type(row.get("axis_trusted")) is bool
    )
    if not axis_contract_valid:
        reasons.append("axis_contract_invalid")
    if geometry_passed and not (
        row.get("axis_trusted") is True
        and isinstance(row.get("axis_angle_source"), str)
        and bool(row.get("axis_angle_source"))
    ):
        reasons.append("eligible_geometry_safety_invalid")
    return reasons


def _gdt_evidence_structure_valid(value: Any) -> bool:
    """Require interpretable GD&T geometry, not replayed set equations."""
    if not isinstance(value, dict) or value.get("consumer_allowed") is not False:
        return False
    for key in (
        "frame_ids",
        "ownership_frame_ids",
        "inside_frame_ids",
        "boundary_frame_ids",
    ):
        raw = value.get(key)
        if not (
            isinstance(raw, list)
            and all(isinstance(item, str) and bool(item) for item in raw)
        ):
            return False
    return True


def _canonicalize_hypothesis_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild advisory route/accounting fields from admitted safety state."""
    raw_source_types = row.get("source_anchor_types")
    source_types = sorted({
        value
        for value in raw_source_types
        if isinstance(value, str) and value
    }) if isinstance(raw_source_types, list) else []
    row["source_anchor_types"] = source_types
    row["key_candidate"] = "capsule" in source_types

    raw_gdt = row.get("gdt_route_evidence")
    list_keys = (
        "frame_ids",
        "ownership_frame_ids",
        "inside_frame_ids",
        "boundary_frame_ids",
    )
    canonical_lists = {
        key: sorted(set(raw_gdt[key]))
        for key in list_keys
    }
    routing_ids = sorted({
        *canonical_lists["frame_ids"],
        *canonical_lists["ownership_frame_ids"],
        *canonical_lists["inside_frame_ids"],
        *canonical_lists["boundary_frame_ids"],
    })
    row["gdt_route_evidence"] = {
        "frame_ids": canonical_lists["frame_ids"],
        "ownership_frame_ids": canonical_lists["ownership_frame_ids"],
        "routing_frame_ids": routing_ids,
        "inside_frame_ids": canonical_lists["inside_frame_ids"],
        "boundary_frame_ids": canonical_lists["boundary_frame_ids"],
        "crosses_frame_boundary": bool(
            canonical_lists["boundary_frame_ids"]
        ),
        "ordinary_eligible": not routing_ids,
        "consumer_allowed": CONSUMER_ALLOWED,
    }

    geometry = copy.deepcopy(row["geometry_gate"])
    geometry_passed = geometry.get("passed") is True
    geometry["status"] = "pass" if geometry_passed else "fail_closed"
    geometry["reasons"] = (
        []
        if geometry_passed
        else list(geometry.get("reasons") or ["pre_route_gate_failed"])
    )
    geometry["consumer_allowed"] = CONSUMER_ALLOWED
    row["geometry_gate"] = geometry

    if geometry_passed:
        route = "gdt" if routing_ids else "ordinary"
        row["three_gates_passed"] = True
        row["first_failed_gate"] = None
        row["ordinary_candidate"] = route == "ordinary"
        row["gdt_candidate"] = route == "gdt"
        row["eligible_for_spatial_dedup"] = True
    else:
        three_gates_passed = row.get("three_gates_passed") is True
        route = (
            "geometry_quarantine"
            if three_gates_passed
            else "rejected_pre_route"
        )
        row["three_gates_passed"] = three_gates_passed
        row["first_failed_gate"] = (
            None
            if three_gates_passed
            else row.get("first_failed_gate")
            if row.get("first_failed_gate") in {
                "coverage",
                "format",
                "boundary",
            }
            else "coverage"
        )
        row["ordinary_candidate"] = False
        row["gdt_candidate"] = False
        row["eligible_for_spatial_dedup"] = False
    row["route"] = route
    row["dedup_status"] = "pending_spatial_value_dedup"
    return row


def _manifest_valid(value: Any, page_index: Any) -> bool:
    expected_keys = {
        "schema_version",
        "pdf_stem",
        "page_num",
        "page_width",
        "page_height",
        "primitive_count",
        "primitive_sha256",
        "consumer_allowed",
    }
    return bool(
        isinstance(value, dict)
        and set(value) == expected_keys
        and type(page_index) is int
        and page_index >= 0
        and value.get("schema_version") == "vector_page_context_manifest_v1"
        and isinstance(value.get("pdf_stem"), str)
        and bool(value.get("pdf_stem"))
        and value.get("page_num") == page_index + 1
        and _positive_number(value.get("page_width"))
        and _positive_number(value.get("page_height"))
        and math.isfinite(
            float(value["page_width"]) * float(value["page_height"])
        )
        and type(value.get("primitive_count")) is int
        and value["primitive_count"] >= 0
        and _valid_sha256(value.get("primitive_sha256"))
        and value.get("consumer_allowed") is False
    )


def _winner_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    source_types = set(str(value) for value in row.get("source_anchor_types") or [])
    anchor_rank = max(
        (
            3 if value in {"dot", "dot_strict"}
            else 2 if value in {"degree", "diameter"}
            else 1 if value == "capsule"
            else 0
            for value in source_types
        ),
        default=0,
    )
    bbox = row["bbox"]
    area = float(bbox["w"]) * float(bbox["h"])
    return (
        -anchor_rank,
        area,
        -int(row.get("key_candidate") is True),
        str(row["phrase_id"]),
    )


def _same_duplicate_partition(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    if left.get("route") == "gdt" or right.get("route") == "gdt":
        return False
    return bool(
        left.get("route") == right.get("route")
        and round(float(left.get("axis_angle_deg")), 6)
        == round(float(right.get("axis_angle_deg")), 6)
        and left.get("axis_angle_source") == right.get("axis_angle_source")
        and _routing_frame_ids(left.get("gdt_route_evidence"))
        == _routing_frame_ids(right.get("gdt_route_evidence"))
    )


def _routing_frame_ids(value: Any) -> tuple[str, ...] | None:
    if not _gdt_evidence_valid(value):
        return None
    return tuple(value["routing_frame_ids"])


def _gdt_evidence_valid(value: Any) -> bool:
    expected_keys = {
        "frame_ids",
        "ownership_frame_ids",
        "routing_frame_ids",
        "inside_frame_ids",
        "boundary_frame_ids",
        "crosses_frame_boundary",
        "ordinary_eligible",
        "consumer_allowed",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return False
    lists: dict[str, list[str]] = {}
    for key in (
        "frame_ids",
        "ownership_frame_ids",
        "routing_frame_ids",
        "inside_frame_ids",
        "boundary_frame_ids",
    ):
        raw = value.get(key)
        if not (
            isinstance(raw, list)
            and all(isinstance(item, str) and bool(item) for item in raw)
            and raw == sorted(set(raw))
        ):
            return False
        lists[key] = raw
    frame_ids = set(lists["frame_ids"])
    ownership_ids = set(lists["ownership_frame_ids"])
    routing_ids = set(lists["routing_frame_ids"])
    return bool(
        frame_ids
        == set(lists["inside_frame_ids"]) | set(lists["boundary_frame_ids"])
        and ownership_ids.issubset(frame_ids)
        and routing_ids == frame_ids | ownership_ids
        and value.get("crosses_frame_boundary")
        is bool(lists["boundary_frame_ids"])
        and value.get("ordinary_eligible") is (not routing_ids)
        and value.get("consumer_allowed") is False
    )


def _bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_x1 = float(left["x"]) + float(left["w"])
    left_y1 = float(left["y"]) + float(left["h"])
    right_x1 = float(right["x"]) + float(right["w"])
    right_y1 = float(right["y"]) + float(right["h"])
    intersection_width = max(
        0.0,
        min(left_x1, right_x1) - max(float(left["x"]), float(right["x"])),
    )
    intersection_height = max(
        0.0,
        min(left_y1, right_y1) - max(float(left["y"]), float(right["y"])),
    )
    intersection = intersection_width * intersection_height
    left_area = float(left["w"]) * float(left["h"])
    right_area = float(right["w"]) * float(right["h"])
    scale = max(left_area, right_area, intersection)
    if scale <= 0.0 or not math.isfinite(scale):
        return 0.0
    normalized_intersection = intersection / scale
    normalized_union = (
        left_area / scale
        + right_area / scale
        - normalized_intersection
    )
    return (
        round(normalized_intersection / normalized_union, 12)
        if normalized_union > 0.0 and math.isfinite(normalized_union)
        else 0.0
    )


def _duplicate_row(
    *,
    winner: dict[str, Any],
    loser: dict[str, Any],
    canonical_text: str,
    bbox_iou: float,
) -> dict[str, Any]:
    return {
        "schema_version": DUPLICATE_SCHEMA_VERSION,
        "winner_phrase_id": str(winner["phrase_id"]),
        "winner_physical_core_digest": str(winner["physical_core_digest"]),
        "winner_bbox": copy.deepcopy(winner["bbox"]),
        "winner_rank": list(_winner_sort_key(winner)),
        "loser_phrase_id": str(loser["phrase_id"]),
        "loser_physical_core_digest": str(loser["physical_core_digest"]),
        "loser_bbox": copy.deepcopy(loser["bbox"]),
        "loser_rank": list(_winner_sort_key(loser)),
        "route": str(winner["route"]),
        "axis_angle_deg": float(winner["axis_angle_deg"]),
        "axis_angle_source": str(winner["axis_angle_source"]),
        "routing_frame_ids": list(
            _routing_frame_ids(winner["gdt_route_evidence"]) or ()
        ),
        "canonical_text": canonical_text,
        "bbox_iou": float(bbox_iou),
        "iou_threshold": PINNED_IOU_THRESHOLD,
        "reason": "same_route_axis_value_spatial_overlap",
        "key_candidate_merged": bool(
            winner.get("key_candidate") is True
            or loser.get("key_candidate") is True
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _bridge_row(
    *,
    candidate: dict[str, Any],
    cluster: dict[str, Any],
    overlapping: list[dict[str, Any]],
    blocking: list[dict[str, Any]],
    canonical_text: str,
) -> dict[str, Any]:
    winner = cluster["winner"]
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "candidate_phrase_id": str(candidate["phrase_id"]),
        "candidate_physical_core_digest": str(
            candidate["physical_core_digest"]
        ),
        "candidate_bbox": copy.deepcopy(candidate["bbox"]),
        "candidate_rank": list(_winner_sort_key(candidate)),
        "cluster_winner_phrase_id": str(winner["phrase_id"]),
        "cluster_winner_physical_core_digest": str(
            winner["physical_core_digest"]
        ),
        "cluster_winner_rank": list(_winner_sort_key(winner)),
        "overlapping_phrase_ids": sorted(
            str(row["phrase_id"]) for row in overlapping
        ),
        "blocking_phrase_ids": sorted(
            str(row["phrase_id"]) for row in blocking
        ),
        "pair_iou": {
            str(row["phrase_id"]): _bbox_iou(candidate["bbox"], row["bbox"])
            for row in sorted(
                cluster["members"],
                key=lambda row: str(row["phrase_id"]),
            )
        },
        "route": str(candidate["route"]),
        "axis_angle_deg": float(candidate["axis_angle_deg"]),
        "axis_angle_source": str(candidate["axis_angle_source"]),
        "routing_frame_ids": list(
            _routing_frame_ids(candidate["gdt_route_evidence"]) or ()
        ),
        "canonical_text": canonical_text,
        "iou_threshold": PINNED_IOU_THRESHOLD,
        "reason": "same_value_overlap_not_complete_link",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _multi_cluster_bridge_row(
    *,
    candidate: dict[str, Any],
    clusters: list[dict[str, Any]],
    canonical_text: str,
) -> dict[str, Any]:
    ordered_clusters = sorted(
        clusters,
        key=lambda cluster: str(cluster["winner"]["phrase_id"]),
    )
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "candidate_phrase_id": str(candidate["phrase_id"]),
        "candidate_physical_core_digest": str(
            candidate["physical_core_digest"]
        ),
        "candidate_bbox": copy.deepcopy(candidate["bbox"]),
        "candidate_rank": list(_winner_sort_key(candidate)),
        "compatible_cluster_winner_phrase_ids": [
            str(cluster["winner"]["phrase_id"])
            for cluster in ordered_clusters
        ],
        "compatible_cluster_iou": {
            str(cluster["winner"]["phrase_id"]): _bbox_iou(
                candidate["bbox"],
                cluster["winner"]["bbox"],
            )
            for cluster in ordered_clusters
        },
        "route": str(candidate["route"]),
        "axis_angle_deg": float(candidate["axis_angle_deg"]),
        "axis_angle_source": str(candidate["axis_angle_source"]),
        "routing_frame_ids": list(
            _routing_frame_ids(candidate["gdt_route_evidence"]) or ()
        ),
        "canonical_text": canonical_text,
        "iou_threshold": PINNED_IOU_THRESHOLD,
        "reason": "same_value_multi_cluster_ambiguity",
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _winner_record(
    *,
    cluster: dict[str, Any],
    source: dict[str, Any],
    canonical_text: str,
) -> dict[str, Any]:
    members = sorted(
        cluster["members"],
        key=lambda row: (
            str(row["phrase_id"]),
            str(row["physical_core_digest"]),
        ),
    )
    return {
        "schema_version": WINNER_SCHEMA_VERSION,
        "trace_id": str(source["trace_id"]),
        "page_index": int(source["page_index"]),
        "phrase_id": str(source["phrase_id"]),
        "physical_core_digest": str(source["physical_core_digest"]),
        "route": str(source["route"]),
        "axis_angle_deg": float(source["axis_angle_deg"]),
        "axis_angle_source": str(source["axis_angle_source"]),
        "bbox": copy.deepcopy(source["bbox"]),
        "text": str(source["text"]),
        "canonical_text": canonical_text,
        "source_winner_key_candidate": source.get("key_candidate") is True,
        "key_candidate": source.get("key_candidate") is True,
        "source_anchor_types": copy.deepcopy(source["source_anchor_types"]),
        "cluster_phrase_ids": [str(row["phrase_id"]) for row in members],
        "cluster_physical_core_digests": [
            str(row["physical_core_digest"]) for row in members
        ],
        "selection_rank": list(_winner_sort_key(source)),
        "selection_policy": "diagnostic_only_no_suppression",
        "source_hypothesis_digest": _digest(source),
        "source_hypothesis": copy.deepcopy(source),
        "dedup_status": "diagnostic_only_carried",
        "eligible_for_post_dedup_routing": True,
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _decision_row(
    *,
    row: dict[str, Any],
    canonical_text: str,
    duplicate: dict[str, Any] | None,
    cluster: dict[str, Any] | None,
) -> dict[str, Any]:
    eligible = row["eligible_for_spatial_dedup"] is True
    if not eligible:
        disposition = "not_eligible"
        winner_phrase_id = None
        bbox_iou = None
        reason = "upstream_not_eligible"
    else:
        disposition = "carried"
        diagnostic_winner = (
            cluster.get("winner") if isinstance(cluster, dict) else None
        )
        winner_phrase_id = str(
            (diagnostic_winner or row)["phrase_id"]
        )
        bbox_iou = (
            duplicate["bbox_iou"] if duplicate is not None else None
        )
        reason = "spatial_dedup_diagnostic_only"
    cluster_members = (
        list(cluster.get("members") or []) if isinstance(cluster, dict) else []
    )
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "trace_id": str(row["trace_id"]),
        "page_index": int(row["page_index"]),
        "phrase_id": str(row["phrase_id"]),
        "physical_core_digest": str(row["physical_core_digest"]),
        "route": str(row["route"]),
        "axis_angle_deg": row.get("axis_angle_deg"),
        "axis_angle_source": row.get("axis_angle_source"),
        "bbox": copy.deepcopy(row["bbox"]),
        "source_text": str(row["text"]),
        "canonical_text": canonical_text,
        "key_candidate": row["key_candidate"] is True,
        "cluster_key_candidate": any(
            member.get("key_candidate") is True for member in cluster_members
        ),
        "cluster_phrase_ids": sorted(
            str(member["phrase_id"]) for member in cluster_members
        ),
        "eligible_for_spatial_dedup": eligible,
        "winner_rank": list(_winner_sort_key(row)) if eligible else None,
        "disposition": disposition,
        "winner_phrase_id": winner_phrase_id,
        "bbox_iou_to_winner": bbox_iou,
        "reason": reason,
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _conflict_ledger(
    rows: list[dict[str, Any]],
    *,
    canonical_text: dict[int, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["phrase_id"]),
            str(row["physical_core_digest"]),
        ),
    )
    for left, right in itertools.combinations(ordered, 2):
        if not _same_duplicate_partition(left, right):
            continue
        iou = _bbox_iou(left["bbox"], right["bbox"])
        left_text = canonical_text[id(left)]
        right_text = canonical_text[id(right)]
        if iou < PINNED_IOU_THRESHOLD or left_text == right_text:
            continue
        result.append({
            "schema_version": CONFLICT_SCHEMA_VERSION,
            "left_phrase_id": str(left["phrase_id"]),
            "left_physical_core_digest": str(left["physical_core_digest"]),
            "left_canonical_text": left_text,
            "left_bbox": copy.deepcopy(left["bbox"]),
            "right_phrase_id": str(right["phrase_id"]),
            "right_physical_core_digest": str(right["physical_core_digest"]),
            "right_canonical_text": right_text,
            "right_bbox": copy.deepcopy(right["bbox"]),
            "route": str(left["route"]),
            "axis_angle_deg": float(left["axis_angle_deg"]),
            "axis_angle_source": str(left["axis_angle_source"]),
            "routing_frame_ids": list(
                _routing_frame_ids(left["gdt_route_evidence"]) or ()
            ),
            "bbox_iou": iou,
            "iou_threshold": PINNED_IOU_THRESHOLD,
            "reason": "spatial_overlap_value_conflict",
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return result


def _bbox_valid_in_page(value: Any, manifest: Any) -> bool:
    if not isinstance(manifest, dict) or not isinstance(value, dict) or set(value) != {
        "x",
        "y",
        "w",
        "h",
    }:
        return False
    numbers = [_finite_number(value.get(key)) for key in ("x", "y", "w", "h")]
    if any(number is None for number in numbers):
        return False
    x, y, width, height = (float(number) for number in numbers)
    page_width = _finite_number(manifest.get("page_width"))
    page_height = _finite_number(manifest.get("page_height"))
    return bool(
        page_width is not None
        and page_height is not None
        and math.isfinite(page_width * page_height)
        and x >= 0.0
        and y >= 0.0
        and width > 0.0
        and height > 0.0
        and math.isfinite(width * height)
        and math.isfinite(x + width)
        and math.isfinite(y + height)
        and x + width <= page_width
        and y + height <= page_height
    )


def _oriented_quad_valid(value: Any, bbox: Any, manifest: Any) -> bool:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and isinstance(bbox, dict)
        and isinstance(manifest, dict)
        and set(bbox) == {"x", "y", "w", "h"}
        and all(
            _finite_number(bbox.get(key)) is not None
            for key in ("x", "y", "w", "h")
        )
        and all(
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(_finite_number(coordinate) is not None for coordinate in point)
            for point in value
        )
    ):
        return False
    area = abs(sum(
        float(value[index][0]) * float(value[(index + 1) % 4][1])
        - float(value[(index + 1) % 4][0]) * float(value[index][1])
        for index in range(4)
    )) / 2.0
    return math.isfinite(area) and area > 0.0


def _positive_number(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and number > 0.0


def _finite_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _pinned_threshold(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and abs(number - PINNED_IOU_THRESHOLD) <= 1e-12


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _failed_dump(
    *,
    trace_id: str,
    page_index: Any,
    reasons: list[str],
) -> dict[str, Any]:
    empty_payload = {
        "decisions": [],
        "winner_hypotheses": [],
        "duplicate_ledger": [],
        "spatial_value_conflict_ledger": [],
        "same_value_bridge_ledger": [],
    }
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": str(trace_id),
        "page_index": page_index if type(page_index) is int else -1,
        "status": "fail_closed",
        "error": {
            "schema_version": "vector_phrase_spatial_dedup_error_v1",
            "code": "invalid_spatial_dedup_input",
            "reasons": sorted(set(reasons)),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        **empty_payload,
        "dedup_status": "not_run",
        "gdt_dedup_status": "not_run",
        "stats": {
            "input_hypothesis_count": 0,
            "eligible_input_count": 0,
            "winner_count": 0,
            "duplicate_suppressed_count": 0,
            "spatial_value_conflict_count": 0,
            "same_value_bridge_count": 0,
            "not_eligible_count": 0,
            "feeds_display_l3_count": 0,
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest(empty_payload),
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
