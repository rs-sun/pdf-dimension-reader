"""Strict post-dedup L3 shadow assembly for the pure-vector mainline."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from directional_walk.phrase_output import validate_directional_walk_dump_v1
from vector_phrase_chain_validation import (
    request_local_artifact_safety_reasons,
)


DUMP_SCHEMA_VERSION = "vector_phrase_l3_shadow_dump_v1"
DIMENSION_SCHEMA_VERSION = "vector_phrase_l3_dimension_v1"
SEMANTICS_SCHEMA_VERSION = "vector_dimension_semantics_v1"
CONSUMER_ALLOWED = False


def build_vector_phrase_l3_shadow_dump(
    *,
    phrase_dump: dict[str, Any] | None = None,
    pitch_read_dump: dict[str, Any] | None = None,
    quality_gate_dump: dict[str, Any] | None = None,
    hypothesis_dump: dict[str, Any],
    spatial_dedup_dump: dict[str, Any],
) -> dict[str, Any]:
    """Assemble carried ordinary hypotheses without enabling final use."""
    trace_id = str(
        hypothesis_dump.get("trace_id") or ""
        if isinstance(hypothesis_dump, dict)
        else ""
    )
    page_index = (
        hypothesis_dump.get("page_index")
        if isinstance(hypothesis_dump, dict)
        else -1
    )
    reasons: list[str] = []
    source_chain_complete = all(
        isinstance(value, dict)
        for value in (
            phrase_dump,
            pitch_read_dump,
            quality_gate_dump,
            hypothesis_dump,
            spatial_dedup_dump,
        )
    )
    if not source_chain_complete:
        reasons.append("authoritative_source_chain_missing")
    else:
        for label, artifact in (
            ("phrase", phrase_dump),
            ("pitch", pitch_read_dump),
            ("quality", quality_gate_dump),
            ("hypothesis", hypothesis_dump),
            ("spatial_dedup", spatial_dedup_dump),
        ):
            if _artifact_safety_reasons(
                artifact,
                path=label,
            ):
                reasons.append(f"{label}_artifact_safety_invalid")
        reasons.extend(_l3_input_structure_reasons(
            phrase_dump=phrase_dump,
            pitch_read_dump=pitch_read_dump,
            quality_gate_dump=quality_gate_dump,
            hypothesis_dump=hypothesis_dump,
            spatial_dedup_dump=spatial_dedup_dump,
        ))
    if reasons:
        return _failed_dump(
            trace_id=trace_id,
            page_index=page_index,
            reasons=reasons,
        )

    dimensions: list[dict[str, Any]] = []
    suspects: list[dict[str, Any]] = []
    gdt_routed: list[dict[str, Any]] = []
    phrase_by_id: dict[str, dict[str, Any]] = {}
    for row in phrase_dump["phrases"]:
        phrase_by_id.setdefault(str(row["phrase_id"]), row)
    quality_by_id: dict[str, dict[str, Any]] = {}
    for row in quality_gate_dump["rows"]:
        quality_by_id.setdefault(str(row["phrase_id"]), row)
    for source in hypothesis_dump["hypotheses"]:
        if source.get("eligible_for_spatial_dedup") is True:
            continue
        source_phrase = phrase_by_id.get(str(source["phrase_id"]))
        quality_row = quality_by_id.get(str(source["phrase_id"]))
        if source_phrase is None or quality_row is None:
            suspects.append(_missing_source_reference_suspect(
                source=source,
                canonical_text=str(source["text"]),
                source_phrase=source_phrase,
                quality_row=quality_row,
                phrase_dump=phrase_dump,
                spatial_dedup_dump=spatial_dedup_dump,
            ))
            continue
        first_failed_gate = source.get("first_failed_gate")
        if first_failed_gate in {"coverage", "format", "boundary"}:
            primary_reason = str(first_failed_gate)
            first_drop_stage = "quality_gate"
        elif source.get("route") == "geometry_quarantine":
            primary_reason = "geometry"
            first_drop_stage = "geometry_gate"
        else:
            primary_reason = "upstream_not_eligible"
            first_drop_stage = "hypothesis_route"
        suspects.append({
            "schema_version": "vector_phrase_l3_suspect_v1",
            "trace_id": str(source["trace_id"]),
            "page_index": int(source["page_index"]),
            "phrase_id": str(source["phrase_id"]),
            "physical_core_digest": str(source["physical_core_digest"]),
            "source_text": str(source["text"]),
            "canonical_text": str(source["text"]),
            "bbox": copy.deepcopy(source["bbox"]),
            "cluster_phrase_ids": [str(source["phrase_id"])],
            "key_candidate": source.get("key_candidate") is True,
            "primary_reason": primary_reason,
            "first_drop_stage": first_drop_stage,
            "quality_gate_digest": str(source["quality_gate_digest"]),
            "quality_gate_reference": _quality_gate_reference(
                quality_row
            ),
            "geometry_gate": copy.deepcopy(source["geometry_gate"]),
            "source_provenance": {
                "phrase_provenance_digest": str(
                    source_phrase["provenance_digest"]
                ),
                "anchor_ids": copy.deepcopy(source_phrase["anchor_ids"]),
                "source_candidate_ids": copy.deepcopy(
                    source_phrase["source_candidate_ids"]
                ),
                "source_region_ids": copy.deepcopy(
                    source_phrase["source_region_ids"]
                ),
                "primitive_run_ids": copy.deepcopy(
                    source_phrase["primitive_run_ids"]
                ),
                "primitive_manifest_sha256": str(
                    phrase_dump["page_context_manifest"][
                        "primitive_sha256"
                    ]
                ),
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            "dedup_evidence": {
                "status": "not_applicable_pre_route",
                "reason": "source_not_eligible_for_spatial_dedup",
                "input_spatial_dedup_semantic_digest": str(
                    spatial_dedup_dump["semantic_digest"]
                ),
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            "l3_shadow_eligible": False,
            "feeds_display_l3": False,
            "ocr_called": False,
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    conflict_by_phrase_id: dict[str, list[dict[str, Any]]] = {}
    for conflict in spatial_dedup_dump[
        "spatial_value_conflict_ledger"
    ]:
        for phrase_id in (
            str(conflict["left_phrase_id"]),
            str(conflict["right_phrase_id"]),
        ):
            conflict_by_phrase_id.setdefault(phrase_id, []).append(conflict)
    bridge_by_phrase_id: dict[str, list[dict[str, Any]]] = {}
    for bridge in spatial_dedup_dump["same_value_bridge_ledger"]:
        phrase_ids = {
            str(bridge.get("candidate_phrase_id") or ""),
            str(bridge.get("cluster_winner_phrase_id") or ""),
            *(
                str(value)
                for value in bridge.get(
                    "compatible_cluster_winner_phrase_ids"
                ) or []
            ),
            *(
                str(value)
                for value in bridge.get("overlapping_phrase_ids") or []
            ),
            *(
                str(value)
                for value in bridge.get("blocking_phrase_ids") or []
            ),
        }
        for phrase_id in sorted(phrase_ids - {""}):
            bridge_by_phrase_id.setdefault(phrase_id, []).append(bridge)
    for winner in spatial_dedup_dump["winner_hypotheses"]:
        source = winner["source_hypothesis"]
        source_phrase = phrase_by_id.get(str(winner["phrase_id"]))
        quality_row = quality_by_id.get(str(winner["phrase_id"]))
        if source_phrase is None or quality_row is None:
            suspects.append(_missing_source_reference_suspect(
                source=source,
                canonical_text=str(winner["canonical_text"]),
                source_phrase=source_phrase,
                quality_row=quality_row,
                phrase_dump=phrase_dump,
                spatial_dedup_dump=spatial_dedup_dump,
            ))
            continue
        if winner.get("route") == "gdt":
            gdt_routed.append({
                "schema_version": "vector_phrase_l3_gdt_shadow_v1",
                "trace_id": str(winner["trace_id"]),
                "page_index": int(winner["page_index"]),
                "phrase_id": str(winner["phrase_id"]),
                "physical_core_digest": str(
                    winner["physical_core_digest"]
                ),
                "source_text": str(winner["text"]),
                "canonical_text": str(winner["canonical_text"]),
                "bbox": copy.deepcopy(winner["bbox"]),
                "oriented_quad": copy.deepcopy(source["oriented_quad"]),
                "axis_angle_deg": float(winner["axis_angle_deg"]),
                "axis_angle_source": str(winner["axis_angle_source"]),
                "cluster_phrase_ids": copy.deepcopy(
                    winner["cluster_phrase_ids"]
                ),
                "routing_frame_ids": copy.deepcopy(
                    source["gdt_route_evidence"]["routing_frame_ids"]
                ),
                "key_candidate": winner.get("key_candidate") is True,
                "quality_gate_reference": _quality_gate_reference(
                    quality_row
                ),
                "geometry_gate": copy.deepcopy(source["geometry_gate"]),
                "source_provenance": {
                    "phrase_provenance_digest": str(
                        source_phrase["provenance_digest"]
                    ),
                    "anchor_ids": copy.deepcopy(source_phrase["anchor_ids"]),
                    "source_candidate_ids": copy.deepcopy(
                        source_phrase["source_candidate_ids"]
                    ),
                    "source_region_ids": copy.deepcopy(
                        source_phrase["source_region_ids"]
                    ),
                    "primitive_run_ids": copy.deepcopy(
                        source_phrase["primitive_run_ids"]
                    ),
                    "primitive_manifest_sha256": str(
                        phrase_dump["page_context_manifest"][
                            "primitive_sha256"
                        ]
                    ),
                    "consumer_allowed": CONSUMER_ALLOWED,
                },
                "dedup_evidence": {
                    "input_spatial_dedup_semantic_digest": str(
                        spatial_dedup_dump["semantic_digest"]
                    ),
                    "cluster_phrase_ids": copy.deepcopy(
                        winner["cluster_phrase_ids"]
                    ),
                    "selection_rank": copy.deepcopy(
                        winner["selection_rank"]
                    ),
                    "selection_policy": str(winner["selection_policy"]),
                    "source_hypothesis_digest": str(
                        winner["source_hypothesis_digest"]
                    ),
                    "consumer_allowed": CONSUMER_ALLOWED,
                },
                "strict_gdt_l3_eligible": False,
                "primary_reason": (
                    "gdt_dedup_disabled_missing_compartment_ownership"
                ),
                "feeds_display_l3": False,
                "ocr_called": False,
                "consumer_allowed": CONSUMER_ALLOWED,
            })
            continue
        if winner.get("route") != "ordinary":
            continue
        winner_audit = _winner_shadow_audit(
            winner=winner,
            source=source,
            source_phrase=source_phrase,
            quality_row=quality_row,
            spatial_dedup_dump=spatial_dedup_dump,
            phrase_dump=phrase_dump,
        )
        cluster_phrase_ids = {
            str(value) for value in winner["cluster_phrase_ids"]
        }
        cluster_conflicts = [
            conflict
            for phrase_id in sorted(cluster_phrase_ids)
            for conflict in conflict_by_phrase_id.get(phrase_id, [])
        ]
        cluster_bridges_by_digest = {
            _digest(bridge): bridge
            for phrase_id in sorted(cluster_phrase_ids)
            for bridge in bridge_by_phrase_id.get(phrase_id, [])
        }
        cluster_bridges = [
            cluster_bridges_by_digest[key]
            for key in sorted(cluster_bridges_by_digest)
        ]
        if cluster_conflicts:
            suspects.append({
                "schema_version": "vector_phrase_l3_suspect_v1",
                "trace_id": str(winner["trace_id"]),
                "page_index": int(winner["page_index"]),
                "phrase_id": str(winner["phrase_id"]),
                "physical_core_digest": str(
                    winner["physical_core_digest"]
                ),
                "source_text": str(winner["text"]),
                "canonical_text": str(winner["canonical_text"]),
                "bbox": copy.deepcopy(winner["bbox"]),
                "cluster_phrase_ids": copy.deepcopy(
                    winner["cluster_phrase_ids"]
                ),
                "key_candidate": winner.get("key_candidate") is True,
                **copy.deepcopy(winner_audit),
                "primary_reason": "spatial_value_conflict",
                "ambiguity_evidence": copy.deepcopy(cluster_conflicts),
                "l3_shadow_eligible": False,
                "feeds_display_l3": False,
                "ocr_called": False,
                "consumer_allowed": CONSUMER_ALLOWED,
            })
            continue
        if cluster_bridges:
            suspects.append({
                "schema_version": "vector_phrase_l3_suspect_v1",
                "trace_id": str(winner["trace_id"]),
                "page_index": int(winner["page_index"]),
                "phrase_id": str(winner["phrase_id"]),
                "physical_core_digest": str(
                    winner["physical_core_digest"]
                ),
                "source_text": str(winner["text"]),
                "canonical_text": str(winner["canonical_text"]),
                "bbox": copy.deepcopy(winner["bbox"]),
                "cluster_phrase_ids": copy.deepcopy(
                    winner["cluster_phrase_ids"]
                ),
                "key_candidate": winner.get("key_candidate") is True,
                **copy.deepcopy(winner_audit),
                "primary_reason": "same_value_bridge_ambiguity",
                "ambiguity_evidence": copy.deepcopy(cluster_bridges),
                "l3_shadow_eligible": False,
                "feeds_display_l3": False,
                "ocr_called": False,
                "consumer_allowed": CONSUMER_ALLOWED,
            })
            continue
        semantics = _dimension_semantics(str(winner["canonical_text"]))
        if semantics is None:
            suspects.append({
                "schema_version": "vector_phrase_l3_suspect_v1",
                "trace_id": str(winner["trace_id"]),
                "page_index": int(winner["page_index"]),
                "phrase_id": str(winner["phrase_id"]),
                "physical_core_digest": str(
                    winner["physical_core_digest"]
                ),
                "source_text": str(winner["text"]),
                "canonical_text": str(winner["canonical_text"]),
                "bbox": copy.deepcopy(winner["bbox"]),
                "cluster_phrase_ids": copy.deepcopy(
                    winner["cluster_phrase_ids"]
                ),
                "key_candidate": winner.get("key_candidate") is True,
                **copy.deepcopy(winner_audit),
                "primary_reason": "semantic_assembly_unsupported",
                "first_drop_stage": "semantic_assembly",
                "l3_shadow_eligible": False,
                "feeds_display_l3": False,
                "ocr_called": False,
                "consumer_allowed": CONSUMER_ALLOWED,
            })
            continue
        cluster_source_provenance = _cluster_source_provenance_rows(
            phrase_ids=winner["cluster_phrase_ids"],
            phrase_by_id=phrase_by_id,
        )
        relevant_duplicate_ledger = [
            copy.deepcopy(row)
            for row in spatial_dedup_dump["duplicate_ledger"]
            if row.get("winner_phrase_id") == winner["phrase_id"]
        ]
        dimensions.append({
            "schema_version": DIMENSION_SCHEMA_VERSION,
            "trace_id": str(winner["trace_id"]),
            "page_index": int(winner["page_index"]),
            "l3_dimension_id": (
                f"l3_{str(winner['physical_core_digest'])[:16]}"
            ),
            "phrase_id": str(winner["phrase_id"]),
            "physical_core_digest": str(winner["physical_core_digest"]),
            "source_text": str(winner["text"]),
            "canonical_text": str(winner["canonical_text"]),
            "dimension_semantics": semantics,
            "bbox": copy.deepcopy(winner["bbox"]),
            "oriented_quad": copy.deepcopy(source["oriented_quad"]),
            "axis_angle_deg": float(winner["axis_angle_deg"]),
            "axis_angle_source": str(winner["axis_angle_source"]),
            "key_candidate": winner.get("key_candidate") is True,
            "three_gates_passed": source["three_gates_passed"] is True,
            "quality_gate_digest": str(source["quality_gate_digest"]),
            "quality_gate_reference": _quality_gate_reference(
                quality_row
            ),
            "geometry_gate": copy.deepcopy(source["geometry_gate"]),
            "gdt_route_evidence": copy.deepcopy(
                source["gdt_route_evidence"]
            ),
            "source_provenance": {
                "phrase_provenance_digest": str(
                    source_phrase["provenance_digest"]
                ),
                "anchor_ids": copy.deepcopy(source_phrase["anchor_ids"]),
                "source_candidate_ids": copy.deepcopy(
                    source_phrase["source_candidate_ids"]
                ),
                "source_region_ids": copy.deepcopy(
                    source_phrase["source_region_ids"]
                ),
                "primitive_run_ids": copy.deepcopy(
                    source_phrase["primitive_run_ids"]
                ),
                "primitive_manifest_sha256": str(
                    phrase_dump["page_context_manifest"]["primitive_sha256"]
                ),
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            "cluster_source_provenance": cluster_source_provenance,
            "dedup_evidence": {
                "input_spatial_dedup_semantic_digest": str(
                    spatial_dedup_dump["semantic_digest"]
                ),
                "cluster_phrase_ids": copy.deepcopy(
                    winner["cluster_phrase_ids"]
                ),
                "cluster_physical_core_digests": copy.deepcopy(
                    winner["cluster_physical_core_digests"]
                ),
                "selection_rank": copy.deepcopy(winner["selection_rank"]),
                "selection_policy": str(winner["selection_policy"]),
                "source_hypothesis_digest": str(
                    winner["source_hypothesis_digest"]
                ),
                "duplicate_ledger": relevant_duplicate_ledger,
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            "l3_shadow_eligible": True,
            "feeds_display_l3": False,
            "ocr_called": False,
            "consumer_allowed": CONSUMER_ALLOWED,
        })

    key_index = _key_index_rows(
        dimensions=dimensions,
        suspects=suspects,
        gdt_routed=gdt_routed,
    )
    trace_relation = "same_trace_phrase_pitch_gate_hypothesis_dedup_l3"
    reader_provenance_status = (
        "contract_only_unreleased_or_unverified_reader"
    )
    source_stage_digests = {
        "phrase_dump_digest": _source_dump_semantic_digest(
            phrase_dump,
            rows_key="phrases",
        ),
        "pitch_read_dump_digest": _source_dump_semantic_digest(
            pitch_read_dump,
            rows_key="reads",
        ),
        "quality_gate_semantic_digest": str(
            quality_gate_dump["semantic_digest"]
        ),
        "hypothesis_semantic_digest": str(
            hypothesis_dump["semantic_digest"]
        ),
        "spatial_dedup_semantic_digest": str(
            spatial_dedup_dump["semantic_digest"]
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    semantic_payload = {
        "trace_relation": trace_relation,
        "reader_provenance_status": reader_provenance_status,
        "recognition_quality_claim_allowed": False,
        "source_stage_digests": source_stage_digests,
        "input_hypothesis_semantic_digest": str(
            hypothesis_dump["semantic_digest"]
        ),
        "input_spatial_dedup_semantic_digest": str(
            spatial_dedup_dump["semantic_digest"]
        ),
        "L3_dimensions": dimensions,
        "L3_suspects": suspects,
        "L3_gdt_routed": gdt_routed,
        "L3_key_index": key_index,
    }
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": int(page_index),
        "page_context_manifest": copy.deepcopy(
            hypothesis_dump["page_context_manifest"]
        ),
        "status": "ok",
        "error": None,
        "trace_relation": trace_relation,
        "reader_provenance_status": reader_provenance_status,
        "recognition_quality_claim_allowed": False,
        "source_stage_digests": source_stage_digests,
        "input_hypothesis_semantic_digest": str(
            hypothesis_dump["semantic_digest"]
        ),
        "input_spatial_dedup_semantic_digest": str(
            spatial_dedup_dump["semantic_digest"]
        ),
        "L3_dimensions": dimensions,
        "L3_suspects": suspects,
        "L3_gdt_routed": gdt_routed,
        "L3_key_index": key_index,
        "promotion_status": "strict_l3_shadow_dump_only",
        "stats": {
            "l3_dimension_count": len(dimensions),
            "l3_suspect_count": len(suspects),
            "l3_gdt_routed_count": len(gdt_routed),
            "l3_key_index_count": len(key_index),
            "spatial_value_conflict_quarantine_count": sum(
                row.get("primary_reason") == "spatial_value_conflict"
                for row in suspects
            ),
            "same_value_bridge_quarantine_count": sum(
                row.get("primary_reason") == "same_value_bridge_ambiguity"
                for row in suspects
            ),
            "semantic_assembly_unsupported_count": sum(
                row.get("primary_reason")
                == "semantic_assembly_unsupported"
                for row in suspects
            ),
            "feeds_display_l3_count": 0,
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest(semantic_payload),
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _dimension_semantics(text: str) -> dict[str, Any] | None:
    chamfer_match = re.fullmatch(
        r"(?P<size>[+\-]?\d{1,3}(?:\.\d{1,3})?)"
        r"(?P<separator>[*xX×])"
        r"(?P<angle>\d{1,3}(?:\.\d{1,3})?)°",
        text,
    )
    if chamfer_match is not None:
        return {
            "schema_version": SEMANTICS_SCHEMA_VERSION,
            "kind": "chamfer",
            "quantity": 1,
            "prefix": None,
            "main_value_text": chamfer_match.group("size"),
            "unit": None,
            "qualifier": None,
            "tolerance": {
                "kind": "none",
                "upper_text": None,
                "lower_text": None,
                "unit": None,
            },
            "chamfer": {
                "separator": chamfer_match.group("separator"),
                "angle_value_text": chamfer_match.group("angle"),
                "angle_unit": "degree",
                "consumer_allowed": CONSUMER_ALLOWED,
            },
            "consumer_allowed": CONSUMER_ALLOWED,
        }
    qualifier = None
    for candidate in ("MAX", "MIN"):
        if text.endswith(candidate):
            qualifier = candidate
            text = text[:-len(candidate)]
            break
    prefix_match = re.fullmatch(
        r"(?:(?P<quantity>[0-9]{1,3})[*xX×])?"
        r"(?P<prefix>Ra|R|Ø)?(?P<body>.+)",
        text,
    )
    if prefix_match is None:
        return None
    quantity_text = prefix_match.group("quantity")
    if quantity_text is not None and int(quantity_text) < 1:
        return None
    value_match = re.fullmatch(
        r"(?P<main>[+\-]?\d{1,3}(?:\.\d{1,3})?)"
        r"(?P<main_degree>°)?"
        r"(?:±(?P<symmetric>\d(?:\.\d{1,3})?)"
        r"(?P<symmetric_degree>°)?"
        r"|(?P<deviations>(?:[+\-]\d(?:\.\d{1,3})?)"
        r"(?:/?[+\-]\d(?:\.\d{1,3})?)?))?",
        prefix_match.group("body"),
    )
    if value_match is None:
        return None
    symmetric = value_match.group("symmetric")
    deviations = value_match.group("deviations")
    if symmetric is not None:
        tolerance_kind = "symmetric"
        upper_text = f"+{symmetric}"
        lower_text = f"-{symmetric}"
    elif deviations is not None:
        parts = re.findall(r"[+\-]\d(?:\.\d{1,3})?", deviations)
        if deviations.replace("/", "") != "".join(parts):
            return None
        positives = [part for part in parts if part.startswith("+")]
        negatives = [part for part in parts if part.startswith("-")]
        if len(positives) > 1 or len(negatives) > 1:
            return None
        upper_text = positives[0] if positives else "0"
        lower_text = negatives[0] if negatives else "0"
        tolerance_kind = (
            "bilateral"
            if positives and negatives
            else "unilateral_upper"
            if positives
            else "unilateral_lower"
        )
    else:
        tolerance_kind = "none"
        upper_text = None
        lower_text = None
    prefix = prefix_match.group("prefix")
    unit = "degree" if value_match.group("main_degree") else None
    kind = "angle" if unit == "degree" else {
        "Ø": "diameter",
        "R": "radius",
        "Ra": "surface_roughness",
    }.get(prefix, "linear")
    return {
        "schema_version": SEMANTICS_SCHEMA_VERSION,
        "kind": kind,
        "quantity": int(quantity_text or 1),
        "prefix": prefix,
        "main_value_text": value_match.group("main"),
        "unit": unit,
        "qualifier": qualifier,
        "tolerance": {
            "kind": tolerance_kind,
            "upper_text": upper_text,
            "lower_text": lower_text,
            "unit": unit if tolerance_kind != "none" else None,
        },
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _key_index_rows(
    *,
    dimensions: list[dict[str, Any]],
    suspects: list[dict[str, Any]],
    gdt_routed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_collection, targets in (
        ("L3_dimensions", dimensions),
        ("L3_gdt_routed", gdt_routed),
        ("L3_suspects", suspects),
    ):
        for target in targets:
            if target.get("key_candidate") is not True:
                continue
            cluster_phrase_ids = (
                target.get("dedup_evidence", {}).get("cluster_phrase_ids")
                if target_collection == "L3_dimensions"
                else target.get("cluster_phrase_ids")
            )
            rows.append({
                "schema_version": "vector_phrase_l3_key_index_v1",
                "trace_id": str(target["trace_id"]),
                "page_index": int(target["page_index"]),
                "key_id": (
                    f"key_{str(target['physical_core_digest'])[:16]}"
                ),
                "phrase_id": str(target["phrase_id"]),
                "physical_core_digest": str(
                    target["physical_core_digest"]
                ),
                "cluster_phrase_ids": copy.deepcopy(
                    list(cluster_phrase_ids or [target["phrase_id"]])
                ),
                "target_collection": target_collection,
                "primary_reason": target.get("primary_reason"),
                "feeds_display_l3": False,
                "ocr_called": False,
                "consumer_allowed": CONSUMER_ALLOWED,
            })
    return sorted(
        rows,
        key=lambda row: (str(row["phrase_id"]), row["target_collection"]),
    )


def _winner_shadow_audit(
    *,
    winner: dict[str, Any],
    source: dict[str, Any],
    source_phrase: dict[str, Any],
    quality_row: dict[str, Any],
    spatial_dedup_dump: dict[str, Any],
    phrase_dump: dict[str, Any],
) -> dict[str, Any]:
    return {
        "quality_gate_digest": str(source["quality_gate_digest"]),
        "quality_gate_reference": _quality_gate_reference(quality_row),
        "geometry_gate": copy.deepcopy(source["geometry_gate"]),
        "source_provenance": {
            "phrase_provenance_digest": str(
                source_phrase["provenance_digest"]
            ),
            "anchor_ids": copy.deepcopy(source_phrase["anchor_ids"]),
            "source_candidate_ids": copy.deepcopy(
                source_phrase["source_candidate_ids"]
            ),
            "source_region_ids": copy.deepcopy(
                source_phrase["source_region_ids"]
            ),
            "primitive_run_ids": copy.deepcopy(
                source_phrase["primitive_run_ids"]
            ),
            "primitive_manifest_sha256": str(
                phrase_dump["page_context_manifest"]["primitive_sha256"]
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "dedup_evidence": {
            "input_spatial_dedup_semantic_digest": str(
                spatial_dedup_dump["semantic_digest"]
            ),
            "cluster_phrase_ids": copy.deepcopy(
                winner["cluster_phrase_ids"]
            ),
            "selection_rank": copy.deepcopy(winner["selection_rank"]),
            "selection_policy": str(winner["selection_policy"]),
            "source_hypothesis_digest": str(
                winner["source_hypothesis_digest"]
            ),
            "duplicate_ledger": [
                copy.deepcopy(row)
                for row in spatial_dedup_dump["duplicate_ledger"]
                if row.get("winner_phrase_id") == winner["phrase_id"]
            ],
            "consumer_allowed": CONSUMER_ALLOWED,
        },
    }


def _cluster_source_provenance_rows(
    *,
    phrase_ids: list[Any],
    phrase_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in phrase_ids:
        phrase_id = str(value)
        source_phrase = phrase_by_id.get(phrase_id)
        if source_phrase is None:
            rows.append({
                "phrase_id": phrase_id,
                "reference_status": "missing",
                "consumer_allowed": CONSUMER_ALLOWED,
            })
            continue
        rows.append({
            "phrase_id": phrase_id,
            "physical_core_digest": str(
                source_phrase["physical_core_digest"]
            ),
            "phrase_provenance_digest": str(
                source_phrase["provenance_digest"]
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        })
    return rows


def _missing_source_reference_suspect(
    *,
    source: dict[str, Any],
    canonical_text: str,
    source_phrase: dict[str, Any] | None,
    quality_row: dict[str, Any] | None,
    phrase_dump: dict[str, Any],
    spatial_dedup_dump: dict[str, Any],
) -> dict[str, Any]:
    """Keep a safe read visible when an internal join reference is absent."""
    missing_references = [
        label
        for label, value in (
            ("phrase", source_phrase),
            ("quality", quality_row),
        )
        if value is None
    ]
    if source.get("first_failed_gate") == "boundary":
        primary_reason = "boundary"
        first_drop_stage = "quality_gate"
    elif (
        source.get("route") == "geometry_quarantine"
        or source.get("geometry_gate", {}).get("passed") is False
    ):
        primary_reason = "geometry"
        first_drop_stage = "geometry_gate"
    else:
        primary_reason = "source_reference_unavailable"
        first_drop_stage = "l3_reference_join"
    if quality_row is None:
        quality_reference = {
            "reference_status": "missing",
            "source_text": str(source.get("text") or ""),
            "consumer_allowed": CONSUMER_ALLOWED,
        }
    else:
        quality_reference = _quality_gate_reference(quality_row)
    if source_phrase is None:
        source_provenance = {
            "reference_status": "missing",
            "anchor_ids": [],
            "source_candidate_ids": [],
            "source_region_ids": [],
            "primitive_run_ids": [],
            "primitive_manifest_sha256": str(
                phrase_dump["page_context_manifest"]["primitive_sha256"]
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        }
    else:
        source_provenance = {
            "phrase_provenance_digest": str(
                source_phrase["provenance_digest"]
            ),
            "anchor_ids": copy.deepcopy(source_phrase["anchor_ids"]),
            "source_candidate_ids": copy.deepcopy(
                source_phrase["source_candidate_ids"]
            ),
            "source_region_ids": copy.deepcopy(
                source_phrase["source_region_ids"]
            ),
            "primitive_run_ids": copy.deepcopy(
                source_phrase["primitive_run_ids"]
            ),
            "primitive_manifest_sha256": str(
                phrase_dump["page_context_manifest"]["primitive_sha256"]
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        }
    return {
        "schema_version": "vector_phrase_l3_suspect_v1",
        "trace_id": str(source["trace_id"]),
        "page_index": int(source["page_index"]),
        "phrase_id": str(source["phrase_id"]),
        "physical_core_digest": str(source["physical_core_digest"]),
        "source_text": str(source["text"]),
        "canonical_text": canonical_text,
        "bbox": copy.deepcopy(source["bbox"]),
        "cluster_phrase_ids": [str(source["phrase_id"])],
        "key_candidate": source.get("key_candidate") is True,
        "primary_reason": primary_reason,
        "first_drop_stage": first_drop_stage,
        "quality_gate_digest": str(
            source.get("quality_gate_digest") or ""
        ),
        "quality_gate_reference": quality_reference,
        "geometry_gate": copy.deepcopy(source["geometry_gate"]),
        "source_provenance": source_provenance,
        "dedup_evidence": {
            "status": "diagnostic_only_reference_missing",
            "missing_references": missing_references,
            "input_spatial_dedup_semantic_digest": str(
                spatial_dedup_dump["semantic_digest"]
            ),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "l3_shadow_eligible": False,
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _quality_gate_reference(row: dict[str, Any]) -> dict[str, Any]:
    validator = row["format_gate"]["evidence"]["validator"]
    coverage_evidence = row["coverage_gate"]["evidence"]
    return {
        "gate_digest": str(row["gate_digest"]),
        "coverage_passed": row["coverage_gate"]["passed"] is True,
        "format_passed": row["format_gate"]["passed"] is True,
        "boundary_passed": row["boundary_gate"]["passed"] is True,
        "phrase_complete": row["phrase_complete"] is True,
        "first_failed_gate": row["first_failed_gate"],
        "format_source_text": str(validator["source_text"]),
        "format_canonical_text": str(validator["canonical_text"]),
        "format_normalization_ledger": copy.deepcopy(
            validator["normalization_ledger"]
        ),
        "raw_text": str(coverage_evidence.get("raw_text") or ""),
        "span_text": str(coverage_evidence.get("span_text") or ""),
        "final_text": str(coverage_evidence.get("final_text") or ""),
        "transform_step_count": int(
            coverage_evidence.get("transform_step_count") or 0
        ),
        "transform_ledger_digest": str(
            coverage_evidence.get("transform_ledger_digest") or ""
        ),
        "transform_replay_text": str(
            coverage_evidence.get("transform_replay_text") or ""
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _artifact_safety_reasons(
    value: Any,
    *,
    path: str,
) -> list[str]:
    reasons = request_local_artifact_safety_reasons(value, path=path)
    if (
        path == "phrase"
        and isinstance(value, dict)
        and value.get("schema_version") == "r92_directional_walk_dump_v1"
    ):
        reasons = [
            reason
            for reason in reasons
            if not (
                reason.startswith(
                    "non_json_value:phrase.phrases["
                )
                and ".walk.cells[" in reason
                and reason.endswith(
                    ".step_evidence.next_shape_keys:tuple"
                )
            )
        ]
    return reasons


def _l3_input_structure_reasons(
    *,
    phrase_dump: dict[str, Any],
    pitch_read_dump: dict[str, Any],
    quality_gate_dump: dict[str, Any],
    hypothesis_dump: dict[str, Any],
    spatial_dedup_dump: dict[str, Any],
) -> list[str]:
    """Check only fields needed to route rows; do not replay stage proofs."""
    trace_id = hypothesis_dump.get("trace_id")
    page_index = hypothesis_dump.get("page_index")
    reasons: list[str] = []
    try:
        validate_directional_walk_dump_v1(phrase_dump)
    except (TypeError, ValueError):
        reasons.append("phrase_structure_invalid")
    stages = (
        (
            "phrase",
            phrase_dump,
            "r92_directional_walk_dump_v1",
            "phrases",
        ),
        (
            "pitch",
            pitch_read_dump,
            "vector_pitch_phrase_read_dump_v1",
            "reads",
        ),
        (
            "quality",
            quality_gate_dump,
            "vector_phrase_quality_gate_dump_v1",
            "rows",
        ),
        (
            "hypothesis",
            hypothesis_dump,
            "vector_phrase_hypothesis_dump_v1",
            "hypotheses",
        ),
        (
            "spatial_dedup",
            spatial_dedup_dump,
            "vector_phrase_spatial_dedup_dump_v1",
            "winner_hypotheses",
        ),
    )
    if not (
        type(trace_id) is str
        and bool(trace_id)
        and type(page_index) is int
        and page_index >= 0
    ):
        reasons.append("trace_or_page_invalid")
    for label, artifact, schema_version, rows_key in stages:
        if not (
            artifact.get("schema_version") == schema_version
            and artifact.get("status") == "ok"
            and artifact.get("trace_id") == trace_id
            and artifact.get("page_index") == page_index
            and isinstance(artifact.get("page_context_manifest"), dict)
            and isinstance(artifact.get(rows_key), list)
            and artifact.get("consumer_allowed") is False
        ):
            reasons.append(f"{label}_structure_invalid")
        if label != "phrase" and artifact.get("ocr_called") is not False:
            reasons.append(f"{label}_structure_invalid")
    if reasons:
        return sorted(set(reasons))

    if not all(
        isinstance(spatial_dedup_dump.get(key), list)
        for key in (
            "duplicate_ledger",
            "spatial_value_conflict_ledger",
            "same_value_bridge_ledger",
        )
    ):
        reasons.append("spatial_dedup_structure_invalid")

    try:
        for quality in quality_gate_dump["rows"]:
            validator = quality["format_gate"]["evidence"]["validator"]
            if not (
                quality["trace_id"] == trace_id
                and quality["page_index"] == page_index
                and type(quality["coverage_gate"]["passed"]) is bool
                and isinstance(quality["coverage_gate"]["evidence"], dict)
                and type(quality["format_gate"]["passed"]) is bool
                and type(quality["boundary_gate"]["passed"]) is bool
                and type(quality["phrase_complete"]) is bool
                and type(validator["source_text"]) is str
                and type(validator["canonical_text"]) is str
                and isinstance(validator["normalization_ledger"], list)
            ):
                reasons.append("quality_row_structure_invalid")
        for source in hypothesis_dump["hypotheses"]:
            if not _route_row_structure_valid(
                source,
                trace_id=trace_id,
                page_index=page_index,
            ):
                reasons.append("hypothesis_row_structure_invalid")
        for winner in spatial_dedup_dump["winner_hypotheses"]:
            if not (
                winner["trace_id"] == trace_id
                and winner["page_index"] == page_index
                and winner["route"] in {"ordinary", "gdt"}
                and isinstance(winner["cluster_phrase_ids"], list)
                and isinstance(
                    winner["cluster_physical_core_digests"],
                    list,
                )
                and isinstance(winner["source_hypothesis"], dict)
                and type(winner["key_candidate"]) is bool
                and _route_row_structure_valid(
                    winner["source_hypothesis"],
                    trace_id=trace_id,
                    page_index=page_index,
                )
            ):
                reasons.append("spatial_winner_structure_invalid")
    except (KeyError, TypeError, ValueError):
        reasons.append("source_row_structure_invalid")
    return sorted(set(reasons))


def _route_row_structure_valid(
    row: dict[str, Any],
    *,
    trace_id: str,
    page_index: int,
) -> bool:
    return bool(
        row.get("trace_id") == trace_id
        and row.get("page_index") == page_index
        and type(row.get("phrase_id")) is str
        and type(row.get("text")) is str
        and isinstance(row.get("bbox"), dict)
        and isinstance(row.get("oriented_quad"), list)
        and isinstance(row.get("geometry_gate"), dict)
        and type(row["geometry_gate"].get("passed")) is bool
        and isinstance(row.get("gdt_route_evidence"), dict)
        and type(row.get("key_candidate")) is bool
        and type(row.get("three_gates_passed")) is bool
        and type(row.get("eligible_for_spatial_dedup")) is bool
        and row.get("consumer_allowed") is False
        and row.get("ocr_called") is False
    )


def _source_dump_semantic_digest(
    value: dict[str, Any],
    *,
    rows_key: str,
) -> str:
    canonical = copy.deepcopy(value)
    canonical[rows_key] = sorted(
        canonical[rows_key],
        key=lambda row: str(row["phrase_id"]),
    )
    return _digest(canonical)


def _failed_dump(
    *,
    trace_id: str,
    page_index: Any,
    reasons: list[str],
) -> dict[str, Any]:
    semantic_payload = {
        "trace_relation": "invalid",
        "reader_provenance_status": "not_verified",
        "recognition_quality_claim_allowed": False,
        "source_stage_digests": {},
        "input_hypothesis_semantic_digest": "",
        "input_spatial_dedup_semantic_digest": "",
        "L3_dimensions": [],
        "L3_suspects": [],
        "L3_gdt_routed": [],
        "L3_key_index": [],
    }
    return {
        "schema_version": DUMP_SCHEMA_VERSION,
        "trace_id": trace_id,
        "page_index": page_index if type(page_index) is int else -1,
        "status": "fail_closed",
        "error": {
            "schema_version": "vector_phrase_l3_shadow_error_v1",
            "code": "invalid_l3_shadow_input",
            "reasons": sorted(set(reasons)),
            "consumer_allowed": CONSUMER_ALLOWED,
        },
        "trace_relation": "invalid",
        "reader_provenance_status": "not_verified",
        "recognition_quality_claim_allowed": False,
        "source_stage_digests": {},
        "input_hypothesis_semantic_digest": "",
        "input_spatial_dedup_semantic_digest": "",
        "L3_dimensions": [],
        "L3_suspects": [],
        "L3_gdt_routed": [],
        "L3_key_index": [],
        "promotion_status": "not_run",
        "stats": {
            "l3_dimension_count": 0,
            "l3_suspect_count": 0,
            "l3_gdt_routed_count": 0,
            "l3_key_index_count": 0,
            "spatial_value_conflict_quarantine_count": 0,
            "same_value_bridge_quarantine_count": 0,
            "semantic_assembly_unsupported_count": 0,
            "feeds_display_l3_count": 0,
            "consumer_allowed_count": 0,
        },
        "semantic_digest": _digest(semantic_payload),
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
