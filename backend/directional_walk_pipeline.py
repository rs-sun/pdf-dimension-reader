"""Production orchestration for the view-free R92 directional-walk core.

The offline R92 shape experiment proved that the detector-owned primitive set
must be the first cell.  The public walker still performs its older spatial
``_anchor_group`` reconstruction, so this retained seam deliberately composes
the walk from the core's pure helpers without mutating or monkeypatching the
core module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable

from decimal_point_quad import normalize_quad_points, side_metrics
from degree_polyline_strict import classify_sld_sequence, segment_angle
from directional_walk import walker as directional_walker
from directional_walk.claim_arbitration import arbitrate_true_subset_claims_v1
from directional_walk.glyph_key import (
    bbox_union,
    glyph_evidence,
    group_primitives,
    lookup_character,
    project_point,
)
from directional_walk.phrase_output import (
    build_dump_v1,
    build_phrase_v1,
    make_termination,
    stable_digest,
    validate_directional_walk_dump_v1,
)
from directional_walk.resolution import (
    resolve_cell_direction_v1,
    resolve_fragment_merges_v1,
    resolve_internal_cell_overlap_v1,
)
from directional_walk.step_table import lookup_step_candidates
from vector_digit_fixed_template import classify_fixed_digit_points
from vector_page_context import VectorPageContext


_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "directional_walk"
_ASSET_MANIFEST = _ASSET_ROOT / "manifest_v1.json"
_ASSET_MANIFEST_SHA256 = 'fc819668a3c0d73f04e8198caadb63bd30007e7e66d11c56ad8219279a27410c'
_ASSET_SHA256 = {'character_table_v1.json': '505fb063fb3da1cd4fe5e3c206a4b598fae9531d273966b7af3cb25adcd2dfff', 'step_table_v1.json': 'c981a98a397da2f3faa46d121b55341cd1becba989633239ea9d96ba807984ef'}
_CONTROLLED_TEMPLATE_SHA256 = ''
_ANCHOR_TYPES = {
    "decimal": ".",
    "degree": "°",
    "diameter": "Ø",
}
_AXIS_SOURCES = {
    "decimal": "decimal_point_quad.short_axis",
    "degree": "degree_polyline_strict.axis_angle_deg",
    "diameter": "pdf_analyzer_extract.axis_angle_deg",
}
_PREFIX_TEMPLATE_SAMPLE_STEP_PT = 0.35
_PREFIX_OUTSIDE_SEARCH_PITCHES = 3.25
_PREFIX_CROSS_AXIS_PAD_RATIO = 0.08
_PREFIX_MAX_PRIMITIVE_SPAN_RATIO = 1.25
_PREFIX_MIN_GLYPH_HEIGHT_RATIO = 0.45
_PREFIX_MAX_GLYPH_HEIGHT_RATIO = 1.20
_PREFIX_SINGLE_EDGE_GAP_PITCH_RANGE = (0.0, 1.60)
_PREFIX_NX_BRIDGE_PITCH_RANGE = (1.20, 1.90)
_PREFIX_NX_PAIR_PITCH_RANGE = (0.80, 1.25)
_PREFIX_ALLOWED_TERMINATION_REASONS = frozenset({
    "no_step_for_prev_key",
    "no_group_at_landing",
})


_PrefixQueryAudit = tuple[Callable[[dict[str, Any]], None], bool]
_PREFIX_QUERY_AUDIT: ContextVar[_PrefixQueryAudit | None] = ContextVar(
    "directional_walk_prefix_query_audit",
    default=None,
)


@contextmanager
def capture_prefix_query_audit_v1(
    observer: Callable[[dict[str, Any]], None],
    *,
    exhaustive: bool = False,
) -> Iterator[None]:
    """Observe prefix locals; optionally compare each one to a full scan."""

    if not callable(observer):
        raise ValueError("prefix query audit observer must be callable")
    token = _PREFIX_QUERY_AUDIT.set((observer, bool(exhaustive)))
    try:
        yield
    finally:
        _PREFIX_QUERY_AUDIT.reset(token)


def build_shape_native_directional_walk_dump_v1(
    *,
    page_context: VectorPageContext,
    decimal_point_rows: Sequence[Mapping[str, Any]],
    degree_rows: Sequence[Mapping[str, Any]],
    diameter_rows: Sequence[Mapping[str, Any]],
    gdt_frames: Sequence[Mapping[str, Any]],
    trace_id: str,
    template_library: Mapping[str, Any],
    content_order_step_binding_enabled: bool = True,
    claim_quality_rank_enabled: bool = True,
) -> dict[str, Any]:
    """Build the native R92 dump from detector-owned shape anchors."""

    if not isinstance(page_context, VectorPageContext):
        raise ValueError("directional_walk_page_context_invalid")
    page_context.assert_integrity()
    if not isinstance(template_library, Mapping):
        raise ValueError("directional_walk_template_library_invalid")
    step_tables, character_table = load_directional_walk_assets_v1()
    manifest = directional_walker._manifest(page_context)
    page_index = int(manifest["page_num"]) - 1
    if page_index < 0:
        raise ValueError("directional_walk_page_index_invalid")
    trace = str(trace_id or "")
    if not trace:
        raise ValueError("directional_walk_trace_id_missing")

    primitives = directional_walker._primitive_rows(page_context)
    anchors, failures = _shape_anchors(
        page_context=page_context,
        decimal_point_rows=decimal_point_rows,
        degree_rows=degree_rows,
        diameter_rows=diameter_rows,
    )
    recognize = _template_recognizer(template_library)
    successes: list[dict[str, Any]] = []
    for raw_anchor in anchors:
        anchor = directional_walker._normalized_anchor(
            raw_anchor,
            page_index=page_index,
        )
        # Canonicalize the binary-float result after normalizing an undirected
        # negative vector so phrase and source evidence compare exactly at the
        # production JSON seam.
        canonical_angle = round(float(anchor["axis_angle_deg"]), 6) % 360.0
        anchor["axis_angle_deg"] = canonical_angle
        anchor["source_anchor_evidence"] = {
            **dict(anchor["source_anchor_evidence"]),
            "axis_angle_deg": canonical_angle,
        }
        success, failure = _walk_shape_anchor(
            primitives=primitives,
            anchor=anchor,
            step_tables=step_tables,
            character_table=character_table,
            template_recognizer=recognize,
            content_order_step_binding_enabled=(
                content_order_step_binding_enabled
            ),
        )
        if failure is not None:
            failures.append(failure)
        elif success is not None:
            successes.append(success)

    for input_order, success in enumerate(successes):
        success["_input_order"] = input_order
    prefix_delivery_before_rows = _counterfactual_success_delivery_rows(
        successes,
        trace_id=trace,
        page_index=page_index,
        gdt_frames=gdt_frames,
    )
    successes, prefix_diagnostics = _discover_outside_prefixes_v1(
        successes=successes,
        primitives=primitives,
        step_tables=step_tables,
        character_table=character_table,
        template_library=template_library,
        template_recognizer=recognize,
    )
    counterfactual_delivery_rows = _counterfactual_success_delivery_rows(
        successes,
        trace_id=trace,
        page_index=page_index,
        gdt_frames=gdt_frames,
    )
    prefix_diagnostics = _bind_prefix_delivery_ledger_v1(
        prefix_diagnostics,
        before_rows=prefix_delivery_before_rows,
        after_rows=counterfactual_delivery_rows,
    )
    internal_overlap_rows = [
        {
            "anchor_id": str(success["anchor"]["anchor_id"]),
            **dict(success["_internal_direction_overlap"]),
        }
        for success in successes
        if "_internal_direction_overlap" in success
    ]
    fragment_resolution = resolve_fragment_merges_v1(successes)
    successes = list(fragment_resolution["successes"])
    fragment_diagnostics = dict(fragment_resolution["diagnostics"])
    successes, transaction_rollbacks = (
        _rollback_undeliverable_fragment_merges(
            successes,
            trace_id=trace,
            page_index=page_index,
            gdt_frames=gdt_frames,
            counterfactual_delivery_rows=counterfactual_delivery_rows,
        )
    )
    fragment_diagnostics.update({
        "internal_direction_overlap_resolved_count": len(
            internal_overlap_rows
        ),
        "internal_direction_overlaps": internal_overlap_rows,
        "transaction_rollback_count": len(transaction_rollbacks),
        "transaction_rollbacks": transaction_rollbacks,
    })
    claim_arbitration_baseline_rows = _counterfactual_success_delivery_rows(
        successes,
        trace_id=trace,
        page_index=page_index,
        gdt_frames=gdt_frames,
    )

    landing_rejections: list[dict[str, Any]] = []
    prepared_claims: list[dict[str, Any]] = []
    for success in successes:
        landing_rejections.extend(success["landing_group_rejections"])
        direction_resolution = resolve_cell_direction_v1(success["cells"])
        phrase = build_phrase_v1(
            trace_id=trace,
            page_index=page_index,
            anchor=success["anchor"],
            cells=direction_resolution["cells"],
            negative_termination=success["negative_termination"],
            positive_termination=success["positive_termination"],
        )
        phrase, gdt_failure = _bind_real_gdt_overlap(
            phrase,
            gdt_frames=gdt_frames,
        )
        direction_row = {
            "anchor_id": str(success["anchor"]["anchor_id"]),
            "source_anchor_ids": list(
                success["anchor"].get(
                    "merged_anchor_ids",
                    [str(success["anchor"]["anchor_id"])],
                )
            ),
            "phrase_id": str(phrase["phrase_id"]),
            "input_text": "".join(
                str(cell["char"]) for cell in success["cells"]
            ),
            "output_text": str(phrase["text"]),
            "direction_source": str(
                direction_resolution["direction_source"]
            ),
            "reversed": bool(direction_resolution["reversed"]),
            "position_signals": list(
                direction_resolution["position_signals"]
            ),
            "criterion_decisions": dict(
                direction_resolution["criterion_decisions"]
            ),
            "criterion_conflict": bool(
                direction_resolution["criterion_conflict"]
            ),
            "syntax_labels": dict(direction_resolution["syntax_labels"]),
            "delivery_status": "pending",
            "consumer_allowed": False,
        }
        prepared_claims.append({
            "success": success,
            "phrase": phrase,
            "gdt_failure": gdt_failure,
            "direction_row": direction_row,
        })

    claim_eligible_indices = [
        index
        for index, prepared in enumerate(prepared_claims)
        if prepared["gdt_failure"] is None
    ]
    eligible_subset_decisions = arbitrate_true_subset_claims_v1([
        {
            "anchor_id": prepared_claims[index]["success"]["anchor"][
                "anchor_id"
            ],
            "input_order": index,
            "primitive_run_ids": prepared_claims[index]["phrase"][
                "primitive_run_ids"
            ],
        }
        for index in claim_eligible_indices
    ])
    subset_decisions: list[dict[str, Any] | None] = [
        None for _prepared in prepared_claims
    ]
    for index, decision in zip(
        claim_eligible_indices,
        eligible_subset_decisions,
        strict=True,
    ):
        subset_decisions[index] = decision
    active_indices = [
        index
        for index, (prepared, subset_decision) in enumerate(zip(
            prepared_claims,
            subset_decisions,
            strict=True,
        ))
        if subset_decision is None and prepared["gdt_failure"] is None
    ]
    if claim_quality_rank_enabled:
        ranked_active = directional_walker._quality_ranked_claim_indices_v1([
            {
                "input_order": index,
                "phrase_id": prepared_claims[index]["phrase"]["phrase_id"],
                "text": prepared_claims[index]["phrase"]["text"],
                "bbox": prepared_claims[index]["phrase"]["bbox"],
                "primitive_run_ids": prepared_claims[index]["phrase"][
                    "primitive_run_ids"
                ],
            }
            for index in active_indices
        ])
        active_indices = [active_indices[index] for index in ranked_active]
    active_index_set = set(active_indices)
    inactive_indices = [
        index
        for index in range(len(prepared_claims))
        if index not in active_index_set
    ]
    arbitration_order = (
        active_indices + inactive_indices
        if claim_quality_rank_enabled
        else list(range(len(prepared_claims)))
    )
    phrases: list[dict[str, Any]] = []
    claimed_primitive_ids: set[str] = set()
    direction_rows_by_index: dict[int, dict[str, Any]] = {}
    claim_failures: dict[int, dict[str, Any]] = {}
    for index in arbitration_order:
        prepared = prepared_claims[index]
        subset_decision = subset_decisions[index]
        success = prepared["success"]
        phrase = prepared["phrase"]
        gdt_failure = prepared["gdt_failure"]
        direction_row = prepared["direction_row"]
        if subset_decision is not None:
            direction_row["delivery_status"] = (
                "subsumed_by_superset_claim"
            )
            direction_row["superset_anchor_id"] = subset_decision[
                "superset_anchor_id"
            ]
            direction_rows_by_index[index] = direction_row
            claim_failures[index] = subset_decision
            continue
        if gdt_failure is not None:
            direction_row["delivery_status"] = "gdt_frame_boundary_crossing"
            direction_rows_by_index[index] = direction_row
            claim_failures[index] = {
                "anchor_id": success["anchor"]["anchor_id"],
                "reason": gdt_failure,
                "claimed_primitive_ids": [],
            }
            continue
        conflicts = claimed_primitive_ids.intersection(
            phrase["primitive_run_ids"]
        )
        if conflicts:
            direction_row["delivery_status"] = "primitive_claim_conflict"
            direction_rows_by_index[index] = direction_row
            claim_failures[index] = {
                "anchor_id": success["anchor"]["anchor_id"],
                "reason": "primitive_claim_conflict",
                "claimed_primitive_ids": [],
                "conflicting_primitive_ids": sorted(conflicts),
                "conflict_scope": "page_phrase_overlap",
            }
            continue
        phrases.append(phrase)
        claimed_primitive_ids.update(phrase["primitive_run_ids"])
        direction_row["delivery_status"] = "delivered"
        direction_rows_by_index[index] = direction_row

    direction_rows = [
        direction_rows_by_index[index]
        for index in range(len(prepared_claims))
    ]
    failures.extend(claim_failures[index] for index in sorted(claim_failures))

    if len(claim_arbitration_baseline_rows) != len(direction_rows):
        raise ValueError("claim arbitration population mismatch")
    paired_delivery_rows = list(zip(
        claim_arbitration_baseline_rows,
        direction_rows,
        strict=True,
    ))
    if any(
        str(baseline["anchor_id"]) != str(live["anchor_id"])
        for baseline, live in paired_delivery_rows
    ):
        raise ValueError("claim arbitration ordering mismatch")
    status_change_rows = [
        {
            "anchor_id": str(live["anchor_id"]),
            "text": str(live["output_text"]),
            "baseline_delivery_status": str(baseline["delivery_status"]),
            "live_delivery_status": str(live["delivery_status"]),
        }
        for baseline, live in paired_delivery_rows
        if baseline["delivery_status"] != live["delivery_status"]
    ]
    released_superset_rows = [
        dict(row)
        for row in status_change_rows
        if row["baseline_delivery_status"] != "delivered"
        and row["live_delivery_status"] == "delivered"
    ]
    delivery_gain_count = len(released_superset_rows)
    delivery_loss_count = sum(
        row["baseline_delivery_status"] == "delivered"
        and row["live_delivery_status"] != "delivered"
        for row in status_change_rows
    )
    delivery_candidate_delta = sum(
        row["delivery_status"] == "delivered"
        for row in direction_rows
    ) - sum(
        row["delivery_status"] == "delivered"
        for row in claim_arbitration_baseline_rows
    )
    subsumed_claim_count = sum(
        decision is not None for decision in subset_decisions
    )
    surviving_claim_count = (
        len(claim_eligible_indices) - subsumed_claim_count
    )
    gdt_ineligible_count = (
        len(prepared_claims) - len(claim_eligible_indices)
    )
    claim_arbitration_diagnostics = {
        "schema_version": "r92_claim_arbitration_diagnostics_v1",
        "baseline_boundary": (
            "post_fragment_resolution_pre_claim_arbitration"
        ),
        "live_boundary": "post_claim_arbitration",
        "input_claim_count": len(prepared_claims),
        "claim_eligible_count": len(claim_eligible_indices),
        "gdt_ineligible_count": gdt_ineligible_count,
        "subsumed_claim_count": subsumed_claim_count,
        "surviving_claim_count": surviving_claim_count,
        "claim_population_balance_closed": (
            len(prepared_claims)
            == gdt_ineligible_count
            + subsumed_claim_count
            + surviving_claim_count
        ),
        "baseline_delivered_claim_count": sum(
            row["delivery_status"] == "delivered"
            for row in claim_arbitration_baseline_rows
        ),
        "live_delivered_claim_count": sum(
            row["delivery_status"] == "delivered"
            for row in direction_rows
        ),
        "delivery_candidate_delta": delivery_candidate_delta,
        "delivery_gain_count": delivery_gain_count,
        "delivery_loss_count": delivery_loss_count,
        "delivery_balance_closed": (
            delivery_candidate_delta
            == delivery_gain_count - delivery_loss_count
        ),
        "status_change_count": len(status_change_rows),
        "replaced_fragment_count": sum(
            baseline["delivery_status"] == "delivered"
            and live["delivery_status"] == "subsumed_by_superset_claim"
            for baseline, live in paired_delivery_rows
        ),
        "released_superset_count": len(released_superset_rows),
        "subsumed_rows": [
            {
                "anchor_id": str(decision["anchor_id"]),
                "text": str(live["output_text"]),
                "superset_anchor_id": str(decision["superset_anchor_id"]),
                "baseline_delivery_status": str(baseline["delivery_status"]),
                "live_delivery_status": str(live["delivery_status"]),
            }
            for decision, (baseline, live) in zip(
                subset_decisions,
                paired_delivery_rows,
                strict=True,
            )
            if decision is not None
        ],
        "released_superset_rows": released_superset_rows,
        "status_change_rows": status_change_rows,
        "consumer_allowed": False,
    }

    fragment_diagnostics = _finalize_fragment_delivery_diagnostics(
        fragment_diagnostics,
        direction_rows=claim_arbitration_baseline_rows,
        counterfactual_delivery_rows=counterfactual_delivery_rows,
        delivered_phrase_count=sum(
            row["delivery_status"] == "delivered"
            for row in claim_arbitration_baseline_rows
        ),
    )
    fragment_diagnostics["delivery_accounting_boundary"] = (
        "post_fragment_resolution_pre_claim_arbitration"
    )

    direction_counts = {
        source: sum(row["direction_source"] == source for row in direction_rows)
        for source in (
            "character_position",
            "draw_order",
            "syntax",
            "kept_default",
        )
    }
    direction_diagnostics = {
        "schema_version": "r92_direction_resolution_diagnostics_v1",
        "participating_walk_count": len(direction_rows),
        "ordering_input_count": len(direction_rows),
        "ordering_output_count": len(direction_rows),
        "ordering_candidate_delta": 0,
        "direction_by_character_position": direction_counts[
            "character_position"
        ],
        "direction_by_character_position_symbols": {
            symbol: sum(
                row["direction_source"] == "character_position"
                and symbol in row["position_signals"]
                for row in direction_rows
            )
            for symbol in ("R", "Ø", "°")
        },
        "direction_by_draw_order": direction_counts["draw_order"],
        "direction_by_syntax": direction_counts["syntax"],
        "direction_kept_default": direction_counts["kept_default"],
        "delivered_walk_count": sum(
            row["delivery_status"] == "delivered" for row in direction_rows
        ),
        "delivery_excluded_walk_count": sum(
            row["delivery_status"] != "delivered" for row in direction_rows
        ),
        "evaluated_reversed_count": sum(
            row["reversed"] for row in direction_rows
        ),
        "actual_reversed_count": sum(
            row["reversed"]
            and row["delivery_status"] == "delivered"
            for row in direction_rows
        ),
        "criterion_conflict_count": sum(
            row["criterion_conflict"] for row in direction_rows
        ),
        "criterion_conflicts": [
            dict(row) for row in direction_rows if row["criterion_conflict"]
        ],
        "rows": direction_rows,
        "consumer_allowed": False,
    }
    prefix_diagnostics = _bind_prefix_final_delivery_v1(
        prefix_diagnostics,
        direction_rows=direction_rows,
    )

    result = build_dump_v1(
        trace_id=trace,
        page_index=page_index,
        page_context_manifest=manifest,
        phrases=phrases,
        anchor_failures=failures,
        additional_diagnostics={
            "shape_native": {
                "schema_version": "r92_shape_native_pipeline_diagnostics_v1",
                "raw_anchor_count": (
                    len(decimal_point_rows)
                    + len(degree_rows)
                    + len(diameter_rows)
                ),
                "walkable_anchor_count": len(anchors),
                "fail_closed_no_axis_count": sum(
                    row.get("reason") == "fail_closed_no_axis"
                    for row in failures
                ),
                "fail_closed_source_lineage_count": sum(
                    row.get("reason")
                    == "fail_closed_source_primitive_lineage"
                    for row in failures
                ),
                "consumer_allowed": False,
            },
            "landing_group_rejections": sorted(
                (dict(row) for row in landing_rejections),
                key=lambda row: (
                    str(row.get("anchor_id") or ""),
                    str(row.get("direction") or ""),
                    int(row.get("attempted_step_index") or 0),
                    tuple(row.get("primitive_ids") or ()),
                ),
            ),
            "low_confidence_step_candidates": (
                directional_walker._low_confidence_step_diagnostics(
                    step_tables
                )
            ),
            "direction_resolution": direction_diagnostics,
            "prefix_discovery": prefix_diagnostics,
            "fragment_merge": fragment_diagnostics,
            "claim_arbitration": claim_arbitration_diagnostics,
        },
    )
    # The HTTP and sealed-contract boundary is JSON.  Normalize tuples emitted
    # by frozen step-table evidence here so semantic hashing is deterministic.
    normalized = json.loads(json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))
    validate_directional_walk_dump_v1(normalized)
    return normalized


def _finalize_fragment_delivery_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    direction_rows: Sequence[Mapping[str, Any]],
    counterfactual_delivery_rows: Sequence[Mapping[str, Any]],
    delivered_phrase_count: int,
) -> dict[str, Any]:
    """Distinguish resolver proposals from merges that changed delivery."""

    result = dict(diagnostics)
    result["merges"] = [
        dict(merge) for merge in diagnostics.get("merges", [])
    ]
    resolver_merge_count = int(result.get("actual_merge_count", 0))
    rollback_by_source_ids = {
        tuple(row["source_anchor_ids"]): row
        for row in result.get("transaction_rollbacks", [])
    }
    actual_merge_count = 0
    delivered_merge_result_count = 0
    for merge in result["merges"]:
        source_ids = {
            str(value) for value in merge.get("source_anchor_ids", [])
        }
        delivery_rows = [
            row
            for row in direction_rows
            if source_ids.issubset(
                {str(value) for value in row.get("source_anchor_ids", [])}
            )
        ]
        delivery_status = (
            min(
                delivery_rows,
                key=lambda row: len(row.get("source_anchor_ids", [])),
            ).get("delivery_status", "not_evaluated")
            if delivery_rows
            else "not_evaluated"
        )
        rollback = rollback_by_source_ids.get(tuple(sorted(source_ids)))
        if not delivery_rows and rollback is not None:
            delivery_status = "rolled_back_" + str(
                rollback["proposal_delivery_status"]
            )
        merge["delivery_status"] = str(delivery_status)
        delivered_merge_result_count += int(delivery_status == "delivered")
        branch_delivery = []
        for raw_branch in merge.get("input_fragment_anchor_ids", []):
            branch_ids = {str(value) for value in raw_branch}
            delivered_rows = [
                row
                for row in counterfactual_delivery_rows
                if row.get("delivery_status") == "delivered"
                and {
                    str(value)
                    for value in row.get("source_anchor_ids", [])
                }.issubset(branch_ids)
            ]
            branch_delivery.append({
                "source_anchor_ids": sorted(branch_ids),
                "delivered_counterfactual_count": len(delivered_rows),
                "has_delivered_counterfactual": bool(delivered_rows),
            })
        merge["counterfactual_input_branch_delivery"] = branch_delivery
        merge["actual_output_reduction_count"] = int(
            delivery_status == "delivered"
            and len(branch_delivery) == 2
            and all(
                row["has_delivered_counterfactual"]
                for row in branch_delivery
            )
        )
        actual_merge_count += merge["actual_output_reduction_count"]
    counterfactual_delivered_phrase_count = sum(
        row.get("delivery_status") == "delivered"
        for row in counterfactual_delivery_rows
    )
    candidate_count_delta = (
        int(delivered_phrase_count) - counterfactual_delivered_phrase_count
    )
    result.update({
        "resolver_merge_count": resolver_merge_count,
        "resolver_output_success_count": result.get(
            "output_success_count",
            0,
        ),
        "actual_merge_count": actual_merge_count,
        "actual_candidate_reduction_count": actual_merge_count,
        "delivered_merge_result_count": delivered_merge_result_count,
        "non_reducing_delivered_merge_count": (
            delivered_merge_result_count - actual_merge_count
        ),
        "excluded_merge_count": (
            resolver_merge_count - delivered_merge_result_count
        ),
        "counterfactual_delivered_phrase_count": (
            counterfactual_delivered_phrase_count
        ),
        "delivered_phrase_count": int(delivered_phrase_count),
        "candidate_count_delta_vs_no_merge": candidate_count_delta,
        "unexplained_delivery_loss_count": max(
            0,
            -candidate_count_delta - actual_merge_count,
        ),
        "counterfactual_delivery_rows": [
            dict(row) for row in counterfactual_delivery_rows
        ],
    })
    return result


def _counterfactual_success_delivery_rows(
    successes: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
    page_index: int,
    gdt_frames: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay old no-fragment-merge delivery for candidate accounting."""

    rows: list[dict[str, Any]] = []
    claimed_primitive_ids: set[str] = set()
    for success in successes:
        direction_resolution = resolve_cell_direction_v1(success["cells"])
        phrase = build_phrase_v1(
            trace_id=trace_id,
            page_index=page_index,
            anchor=success["anchor"],
            cells=direction_resolution["cells"],
            negative_termination=success["negative_termination"],
            positive_termination=success["positive_termination"],
        )
        phrase, gdt_failure = _bind_real_gdt_overlap(
            phrase,
            gdt_frames=gdt_frames,
        )
        delivery_status = "delivered"
        conflicting_primitive_ids: list[str] = []
        if gdt_failure is not None:
            delivery_status = "gdt_frame_boundary_crossing"
        elif conflicts := claimed_primitive_ids.intersection(
            phrase["primitive_run_ids"]
        ):
            delivery_status = "primitive_claim_conflict"
            conflicting_primitive_ids = sorted(conflicts)
        else:
            claimed_primitive_ids.update(phrase["primitive_run_ids"])
        rows.append({
            "anchor_id": str(success["anchor"]["anchor_id"]),
            "source_anchor_ids": list(
                success["anchor"].get(
                    "merged_anchor_ids",
                    [str(success["anchor"]["anchor_id"])],
                )
            ),
            "phrase_id": str(phrase["phrase_id"]),
            "delivery_status": delivery_status,
            "primitive_run_ids": list(phrase["primitive_run_ids"]),
            "conflicting_primitive_ids": conflicting_primitive_ids,
            "consumer_allowed": False,
        })
    return rows


def _rollback_undeliverable_fragment_merges(
    successes: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
    page_index: int,
    gdt_frames: Sequence[Mapping[str, Any]],
    counterfactual_delivery_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Treat each merge as a transaction and restore members on exclusion."""

    rows = [dict(success) for success in successes]
    rollbacks: list[dict[str, Any]] = []
    while True:
        rows.sort(key=lambda row: (
            int(row.get("_input_order", 0)),
            tuple(
                sorted(
                    str(anchor["anchor_id"])
                    for anchor in row.get(
                        "_source_anchors",
                        (row["anchor"],),
                    )
                )
            ),
        ))
        previews = _counterfactual_success_delivery_rows(
            rows,
            trace_id=trace_id,
            page_index=page_index,
            gdt_frames=gdt_frames,
        )
        rollback_index = next(
            (
                index
                for index, (success, preview) in enumerate(
                    zip(rows, previews, strict=True)
                )
                if success.get("_merge_members")
                and preview["delivery_status"] != "delivered"
            ),
            None,
        )
        rollback_reason = None
        displaced_source_anchor_ids: list[list[str]] = []
        if rollback_index is not None:
            rollback_reason = str(
                previews[rollback_index]["delivery_status"]
            )
        else:
            represented_anchor_ids = {
                str(value)
                for preview in previews
                if preview["delivery_status"] == "delivered"
                for value in preview.get("source_anchor_ids", [])
            }
            displaced = [
                row
                for row in counterfactual_delivery_rows
                if row.get("delivery_status") == "delivered"
                and not {
                    str(value)
                    for value in row.get("source_anchor_ids", [])
                }.issubset(represented_anchor_ids)
            ]
            if displaced:
                displaced_source_anchor_ids = [
                    sorted(
                        str(value)
                        for value in row.get("source_anchor_ids", [])
                    )
                    for row in displaced
                ]
                conflict_ids = {
                    str(value)
                    for displaced_row in displaced
                    for preview in previews
                    if set(preview.get("source_anchor_ids", []))
                    == set(displaced_row.get("source_anchor_ids", []))
                    for value in preview.get(
                        "conflicting_primitive_ids",
                        [],
                    )
                }
                rollback_index = next(
                    (
                        index
                        for index, (success, preview) in enumerate(
                            zip(rows, previews, strict=True)
                        )
                        if success.get("_merge_members")
                        and preview["delivery_status"] == "delivered"
                        and conflict_ids.intersection(
                            str(value)
                            for cell in success["cells"]
                            for value in cell["primitive_ids"]
                        )
                    ),
                    None,
                )
                if rollback_index is None:
                    rollback_index = next(
                        (
                            index
                            for index, (success, preview) in enumerate(
                                zip(rows, previews, strict=True)
                            )
                            if success.get("_merge_members")
                            and preview["delivery_status"] == "delivered"
                        ),
                        None,
                    )
                if rollback_index is not None:
                    rollback_reason = "displaced_counterfactual_delivery"
        if rollback_index is None:
            return rows, rollbacks
        proposal = rows.pop(rollback_index)
        source_anchor_ids = sorted(
            str(anchor["anchor_id"])
            for anchor in proposal.get(
                "_source_anchors",
                (proposal["anchor"],),
            )
        )
        members = [
            dict(member) for member in proposal["_merge_members"]
        ]
        rows.extend(members)
        rollbacks.append({
            "source_anchor_ids": source_anchor_ids,
            "proposal_delivery_status": str(rollback_reason),
            "displaced_counterfactual_source_anchor_ids": (
                displaced_source_anchor_ids
            ),
            "restored_fragment_anchor_ids": [
                sorted(
                    str(anchor["anchor_id"])
                    for anchor in member.get(
                        "_source_anchors",
                        (member["anchor"],),
                    )
                )
                for member in members
            ],
            "consumer_allowed": False,
        })


@lru_cache(maxsize=1)
def load_directional_walk_assets_v1() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify the checked-in step and character tables."""

    manifest_bytes = _ASSET_MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != _ASSET_MANIFEST_SHA256:
        raise ValueError("directional_walk_asset_manifest_digest_mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "r92_directional_walk_asset_manifest_v1"
        or manifest.get("consumer_allowed") is not False
    ):
        raise ValueError("directional_walk_asset_manifest_invalid")
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("directional_walk_asset_manifest_invalid")

    loaded: dict[str, dict[str, Any]] = {}
    for filename in ("step_table_v1.json", "character_table_v1.json"):
        identity = assets.get(filename)
        if not isinstance(identity, dict):
            raise ValueError("directional_walk_asset_identity_missing")
        raw = (_ASSET_ROOT / filename).read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if (
            identity.get("sha256") != _ASSET_SHA256[filename]
            or actual_sha256 != _ASSET_SHA256[filename]
        ):
            raise ValueError("directional_walk_asset_digest_mismatch")
        payload = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != identity.get("schema_version")
            or payload.get("consumer_allowed") is not False
        ):
            raise ValueError("directional_walk_asset_payload_invalid")
        loaded[filename] = payload
    return loaded["step_table_v1.json"], loaded["character_table_v1.json"]


def build_directional_walk_pitch_projection_v1(
    *,
    phrase_dump: Mapping[str, Any],
    page_context: VectorPageContext,
) -> dict[str, Any]:
    """Project already-recognized R92 cells into the retained gate envelope."""

    validate_directional_walk_dump_v1(phrase_dump)
    manifest = page_context.manifest_v1()
    if dict(phrase_dump["page_context_manifest"]) != manifest:
        raise ValueError("directional_walk_projection_manifest_mismatch")
    reads: list[dict[str, Any]] = []
    for phrase in phrase_dump["phrases"]:
        cells = phrase["walk"]["cells"]
        spans = [
            {
                "char": str(cell["char"]),
                "slot_item_count": len(cell["primitive_ids"]),
                "primitive_ids": list(cell["primitive_ids"]),
                "reading_index": int(cell["reading_index"]),
                "source_cell_id": str(cell["cell_id"]),
                "decision": "directional_walk_character_table_match",
                "consumer_allowed": False,
            }
            for cell in cells
        ]
        text = str(phrase["text"])
        reader_debug = {
            "schema_version": "r92_directional_walk_pitch_projection_v1",
            "status": "projected",
            "algorithm": "directional_walk_cell_projection",
            "used_item_count": len(phrase["primitive_run_ids"]),
            "filtered_item_count": 0,
            "raw_predicted_text": text,
            "transform_ledger": [],
            "repairs": [],
            "retries": [],
            "consumer_allowed": False,
        }
        reads.append({
            "schema_version": "vector_pitch_phrase_read_v1",
            "source_phrase_schema_version": (
                "r92_directional_walk_phrase_v1"
            ),
            "read_mode": "upstream_directional_walk_projection",
            "trace_id": str(phrase["trace_id"]),
            "phrase_id": str(phrase["phrase_id"]),
            "physical_core_digest": str(phrase["physical_core_digest"]),
            "page_index": int(phrase["page_index"]),
            "status": "read",
            "fail_closed_reason": None,
            "text": text,
            "raw_text": text,
            "bbox": dict(phrase["bbox"]),
            "oriented_quad": list(phrase["oriented_quad"]),
            "orientation": _orientation_from_axis(phrase["axis_angle_deg"]),
            "axis_angle_deg": float(phrase["axis_angle_deg"]),
            "axis_angle_source": str(phrase["axis_angle_source"]),
            "axis_trusted": True,
            "spans": spans,
            "exclusions": [],
            "drops": [],
            "attempts": [],
            "pitch_candidate_count": 0,
            "pitch_origin_attempt_count": 0,
            "slot_window_count": len(spans),
            "template_classification_count": 0,
            "chamfer_count": 0,
            "template_decisions": [],
            "repairs": [],
            "retries": [],
            "coverage": {
                "status": "projected_from_r92_walk",
                "primitive_count": len(phrase["primitive_run_ids"]),
                "span_count": len(spans),
                "exclusion_count": 0,
                "drop_count": 0,
                "consumer_allowed": False,
            },
            "format": _format_result(text),
            "source_candidate_ids": list(phrase["source_candidate_ids"]),
            "source_anchor_ids": list(phrase["anchor_ids"]),
            "source_anchor_evidence": list(phrase["source_anchor_evidence"]),
            "source_anchor_types": sorted({
                str(row["anchor_type"])
                for row in phrase["source_anchor_evidence"]
            }),
            "primitive_ids": list(phrase["primitive_run_ids"]),
            "primitive_manifest": None,
            "reader_call_count": 0,
            "reader_bbox_query_count": 0,
            "reader_debug": reader_debug,
            "feeds_display_l3": False,
            "ocr_called": False,
            "consumer_allowed": False,
        })
    return {
        "schema_version": "vector_pitch_phrase_read_dump_v1",
        "trace_id": str(phrase_dump["trace_id"]),
        "page_index": int(phrase_dump["page_index"]),
        "page_context_manifest": manifest,
        "status": "ok",
        "error": None,
        "reads": reads,
        "duplicate_ledger": [],
        "stats": {
            "input_phrase_count": len(reads),
            "output_read_count": len(reads),
            "successful_read_count": len(reads),
            "fail_closed_read_count": 0,
            "reader_input_complexity_rejected_count": 0,
            "duplicate_suppressed_count": 0,
            "reader_call_count": 0,
            "reader_bbox_query_count": 0,
            "page_context_extraction_call_count": int(
                page_context.runtime_metrics().get("extraction_call_count") or 0
            ),
            "feeds_display_l3_count": 0,
            "consumer_allowed_count": 0,
        },
        "reader_dependency": {
            "schema_version": "r92_directional_walk_projection_dependency_v1",
            "status": "directional_walk_projection_connected",
            "module": "directional_walk_pipeline",
            "callable": "build_directional_walk_pitch_projection_v1",
            "consumer_allowed": False,
        },
        "feeds_display_l3": False,
        "ocr_called": False,
        "consumer_allowed": False,
    }


def _axis_angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _axis_sample_mean(
    angles: Sequence[float],
    weights: Sequence[float],
) -> float:
    x = sum(
        math.cos(math.radians(float(angle) * 2.0)) * float(weight)
        for angle, weight in zip(angles, weights, strict=True)
    )
    y = sum(
        math.sin(math.radians(float(angle) * 2.0)) * float(weight)
        for angle, weight in zip(angles, weights, strict=True)
    )
    return (math.degrees(math.atan2(y, x)) / 2.0) % 180.0


def _axis_resolution_evidence(
    *,
    kind: str,
    detector_row: Mapping[str, Any],
    source_primitives: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if kind == "decimal" and len(source_primitives) == 1:
        points = normalize_quad_points([
            (float(x), float(y))
            for x, y in source_primitives[0]["points"]
        ])
        metrics = None if points is None else side_metrics(points)
        if metrics is None:
            return None
        sides = [float(value) for value in metrics["sides"]]
        angles = [float(value) for value in metrics["angles"]]
        short_index = min(range(4), key=lambda index: sides[index])
        sample_indices = [short_index, (short_index + 2) % 4]
        samples = [angles[index] for index in sample_indices]
        weights = [sides[index] for index in sample_indices]
        emitted = float(detector_row["short_axis"]["angle_deg"])
        serialization_quantum = 0.01
        evidence_kind = "decimal_quad_short_opposite_edges"
    elif kind == "degree" and len(source_primitives) == 8:
        weights = [
            math.dist(
                tuple(primitive["points"][0]),
                tuple(primitive["points"][-1]),
            )
            for primitive in source_primitives
        ]
        angles = [
            segment_angle(
                tuple(primitive["points"][0]),
                tuple(primitive["points"][-1]),
            )
            for primitive in source_primitives
        ]
        classified = classify_sld_sequence(weights)
        sample_indices = [
            index
            for index, value in enumerate(classified["types"])
            if value == "S"
        ]
        if len(sample_indices) != 2:
            return None
        samples = [angles[index] for index in sample_indices]
        weights = [weights[index] for index in sample_indices]
        emitted = float(detector_row["axis_angle_deg"])
        serialization_quantum = 0.0001
        evidence_kind = "degree_strict_sequence_S_edges"
    else:
        return None
    dispersion = max(_axis_angle_delta(sample, emitted) for sample in samples)
    return {
        "kind": evidence_kind,
        "raw_geometry": {
            "source_primitive_ids": sorted(
                str(primitive["primitive_id"])
                for primitive in source_primitives
            ),
            "source": (
                "VectorPageContext primitive points before detector rounding"
            ),
        },
        "sample_edge_indices": sample_indices,
        "sample_angles_deg": samples,
        "sample_lengths_pt": weights,
        "sample_mean_angle_deg": _axis_sample_mean(samples, weights),
        "representative_angle_deg": emitted,
        "detector_emitted_angle_deg": emitted,
        "measured_dispersion_deg": dispersion,
        "detector_serialization_quantum_deg": serialization_quantum,
        "axis_precision_half_width_deg": (
            dispersion + serialization_quantum / 2.0
        ),
    }


def _shape_anchors(
    *,
    page_context: VectorPageContext,
    decimal_point_rows: Sequence[Mapping[str, Any]],
    degree_rows: Sequence[Mapping[str, Any]],
    diameter_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = page_context.manifest_v1()
    page_index = int(manifest["page_num"]) - 1
    lineage = {
        (int(item.drawing_order), int(item.item_index), str(item.op)): (
            page_context.item_id(item)
        )
        for item in page_context.primitives
    }
    primitive_by_id = {
        str(row["primitive_id"]): row
        for row in directional_walker._primitive_rows(page_context)
    }
    anchors: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def add_anchor(
        *,
        kind: str,
        detector_id: str,
        bbox: Mapping[str, Any],
        axis_angle_deg: Any,
        source_schema_version: str,
        source_rows: Sequence[Mapping[str, Any]],
        detector_row: Mapping[str, Any],
    ) -> None:
        stable_id = (
            f"r92_shape:{manifest['pdf_stem']}:p{page_index + 1:03d}:"
            f"{kind}:{detector_id}"
        )
        if axis_angle_deg is None:
            failures.append({
                "anchor_id": stable_id,
                "reason": "fail_closed_no_axis",
                "claimed_primitive_ids": [],
            })
            return
        try:
            angle = float(axis_angle_deg)
        except (TypeError, ValueError):
            angle = math.nan
        if not math.isfinite(angle):
            failures.append({
                "anchor_id": stable_id,
                "reason": "fail_closed_no_axis",
                "claimed_primitive_ids": [],
            })
            return
        requested_in_order: list[tuple[int, int, str]] = []
        for row in source_rows:
            try:
                requested_in_order.append((
                    int(row["drawing_order"]),
                    int(row["item_index"]),
                    str(row["op"]),
                ))
            except (KeyError, TypeError, ValueError):
                requested_in_order.clear()
                break
        requested = set(requested_in_order)
        source_ids = sorted(lineage[key] for key in requested if key in lineage)
        if not requested or len(source_ids) != len(requested):
            failures.append({
                "anchor_id": stable_id,
                "reason": "fail_closed_source_primitive_lineage",
                "claimed_primitive_ids": [],
            })
            return
        normalized_bbox = _bbox_xywh(bbox)
        anchor = {
            "anchor_id": stable_id,
            "source_id": stable_id,
            "source_schema_version": source_schema_version,
            "anchor_type": kind,
            "anchor_bbox": normalized_bbox,
            "axis_angle_deg": angle % 180.0,
            "axis_angle_source": _AXIS_SOURCES[kind],
            "axis_directionality": "undirected_fallback",
            "axis_trusted": True,
            "source_primitive_ids": source_ids,
            "source_candidate_ids": [stable_id],
            "source_region_ids": [],
        }
        ordered_source_primitives = [
            primitive_by_id[lineage[key]] for key in requested_in_order
        ]
        resolution_evidence = _axis_resolution_evidence(
            kind=kind,
            detector_row=detector_row,
            source_primitives=ordered_source_primitives,
        )
        if resolution_evidence is not None:
            anchor["axis_resolution_evidence"] = resolution_evidence
        anchors.append(anchor)

    for row in decimal_point_rows:
        detail = row.get("detail") if isinstance(row, Mapping) else None
        if not isinstance(detail, Mapping):
            detail = {}
        add_anchor(
            kind="decimal",
            detector_id=str(row.get("id") or "decimal_missing_id"),
            bbox=row.get("bbox") or {},
            axis_angle_deg=(row.get("short_axis") or {}).get("angle_deg"),
            source_schema_version=str(
                row.get("schema_version") or "vector_decimal_point_quad_v1"
            ),
            source_rows=({
                "drawing_order": detail.get("drawing_order"),
                "item_index": detail.get("item_index"),
                "op": "qu",
            },),
            detector_row=row,
        )
    for row in degree_rows:
        add_anchor(
            kind="degree",
            detector_id=str(row.get("id") or "degree_missing_id"),
            bbox=row.get("bbox") or {},
            axis_angle_deg=row.get("axis_angle_deg"),
            source_schema_version=str(
                row.get("schema_version") or "vector_degree_polyline_v1"
            ),
            source_rows=tuple(row.get("source_primitives") or ()),
            detector_row=row,
        )
    # Diameter is structurally disabled for this cutover.  Do not let a future
    # detector field or a test fixture accidentally promote it before the
    # separately specified axis/lineage work exists.
    for index, row in enumerate(diameter_rows):
        detector_id = str(row.get("id") or f"diameter_{index:04d}")
        failures.append({
            "anchor_id": (
                f"r92_shape:{manifest['pdf_stem']}:p{page_index + 1:03d}:"
                f"diameter:{detector_id}"
            ),
            "reason": "fail_closed_no_axis",
            "claimed_primitive_ids": [],
        })
    anchors.sort(key=lambda row: row["anchor_id"])
    failures.sort(key=lambda row: (row["anchor_id"], row["reason"]))
    return anchors, failures


def _shape_source_group(
    *,
    primitives: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Use exactly the detector-owned primitives as the anchor cell."""

    source_ids = set(str(value) for value in anchor["source_primitive_ids"])
    selected = sorted(
        (
            dict(row)
            for row in primitives
            if str(row["primitive_id"]) in source_ids
        ),
        key=lambda row: str(row["primitive_id"]),
    )
    if len(selected) != len(source_ids):
        raise ValueError("directional_walk_shape_source_disappeared")
    angle = float(anchor["axis_angle_deg"])
    projected = [
        project_point(point, angle)
        for row in selected
        for point in row["points"]
    ]
    return {
        "primitive_ids": tuple(row["primitive_id"] for row in selected),
        "primitives": tuple(selected),
        "bbox": bbox_union(row["bbox"] for row in selected),
        "uv_bbox": (
            min(point[0] for point in projected),
            min(point[1] for point in projected),
            max(point[0] for point in projected),
            max(point[1] for point in projected),
        ),
    }


def _projected_points_bbox(
    points: Sequence[Sequence[float]],
    *,
    axis_angle_deg: float,
) -> tuple[float, float, float, float]:
    projected = [
        project_point(point, axis_angle_deg)
        for point in points
    ]
    if not projected:
        raise ValueError("prefix discovery requires vector points")
    return (
        min(point[0] for point in projected),
        min(point[1] for point in projected),
        max(point[0] for point in projected),
        max(point[1] for point in projected),
    )


def _prefix_cell_uv_bbox(
    cell: Mapping[str, Any],
    *,
    axis_angle_deg: float,
) -> tuple[float, float, float, float]:
    return _projected_points_bbox(
        cell.get("points", ()),
        axis_angle_deg=axis_angle_deg,
    )


def _prefix_group_u_center(group: Mapping[str, Any]) -> float:
    u0, _v0, u1, _v1 = (float(value) for value in group["uv_bbox"])
    return (u0 + u1) / 2.0


def _prefix_body_pitch(
    cells: Sequence[Mapping[str, Any]],
    *,
    axis_angle_deg: float,
    reference_height: float,
) -> float | None:
    centers = []
    for cell in cells:
        u0, _v0, u1, _v1 = _prefix_cell_uv_bbox(
            cell,
            axis_angle_deg=axis_angle_deg,
        )
        centers.append((u0 + u1) / 2.0)
    gaps = [
        abs(right - left)
        for left, right in zip(centers, centers[1:])
        if reference_height * 0.25
        <= abs(right - left)
        <= reference_height * 1.75
    ]
    if not gaps:
        return None
    result = float(median(gaps))
    return result if math.isfinite(result) and result > 0.0 else None


def _dense_prefix_template_points(
    group: Mapping[str, Any],
) -> list[list[float]]:
    sampled: list[tuple[float, float]] = []
    for primitive in group.get("primitives", ()):
        points = [
            (float(point[0]), float(point[1]))
            for point in primitive.get("points", ())
        ]
        for left, right in zip(points, points[1:]):
            length = math.dist(left, right)
            count = max(
                1,
                int(math.ceil(
                    length / _PREFIX_TEMPLATE_SAMPLE_STEP_PT
                )),
            )
            for index in range(count):
                fraction = index / count
                sampled.append((
                    left[0] + (right[0] - left[0]) * fraction,
                    left[1] + (right[1] - left[1]) * fraction,
                ))
            sampled.append(right)
    result: list[list[float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in sampled:
        key = (int(round(x * 1000.0)), int(round(y * 1000.0)))
        if key in seen:
            continue
        seen.add(key)
        result.append([round(x, 6), round(y, 6)])
    return result


def _controlled_x_template_match(
    group: Mapping[str, Any],
    *,
    template_library: Mapping[str, Any],
) -> dict[str, Any]:
    return classify_fixed_digit_points(
        _dense_prefix_template_points(group),
        template_library=dict(template_library),
        threshold=0.008,
        margin=0.010,
        allowed_labels={"X"},
        base_templates_only=False,
    )


def _single_character_entry(
    evidence: Mapping[str, Any],
    *,
    character_table: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    entry = lookup_character(
        character_table,
        shape_key=str(evidence["shape_key"]),
    )
    if entry is None:
        return None, None
    characters = tuple(entry["characters"])
    if len(characters) != 1:
        return None, dict(entry)
    return str(characters[0]), dict(entry)


def _prefix_local_rows(
    primitives: Sequence[Mapping[str, Any]],
    *,
    claimed: set[str],
    projected_bbox_for: Callable[
        [Mapping[str, Any]],
        tuple[float, float, float, float],
    ],
    first_uv: Sequence[float],
    body_start_u: float,
    reading_sign: int,
    v_padding: float,
    maximum_distance: float,
    maximum_primitive_span: float,
) -> tuple[Mapping[str, Any], ...]:
    """Apply the original prefix primitive predicates in page order."""

    local: list[Mapping[str, Any]] = []
    for primitive in primitives:
        if str(primitive["primitive_id"]) in claimed:
            continue
        u0, v0, u1, v1 = projected_bbox_for(primitive)
        if max(u1 - u0, v1 - v0) > maximum_primitive_span:
            continue
        if v0 < first_uv[1] - v_padding or v1 > first_uv[3] + v_padding:
            continue
        if reading_sign > 0:
            in_axial_window = (
                u1 <= body_start_u
                and u0 >= body_start_u - maximum_distance
            )
        else:
            in_axial_window = (
                u0 >= body_start_u
                and u1 <= body_start_u + maximum_distance
            )
        if in_axial_window:
            local.append(primitive)
    return tuple(local)


def _prefix_candidate_rows(
    local: Sequence[Mapping[str, Any]],
    *,
    projected_bboxes: Mapping[str, Sequence[float]],
    axis_angle_deg: float,
    reference_height: float,
    first_u: float,
    reading_sign: int,
    maximum_distance: float,
) -> list[dict[str, Any]]:
    groups = group_primitives(
        local,
        axis_angle_deg=axis_angle_deg,
        reference_scale=reference_height,
        projected_bboxes=projected_bboxes,
    )
    candidates: list[dict[str, Any]] = []
    for group in groups:
        u0, v0, u1, v1 = (float(value) for value in group["uv_bbox"])
        height_ratio = (v1 - v0) / reference_height
        center_u = (u0 + u1) / 2.0
        outward_distance = (first_u - center_u) * reading_sign
        if not (
            _PREFIX_MIN_GLYPH_HEIGHT_RATIO
            <= height_ratio
            <= _PREFIX_MAX_GLYPH_HEIGHT_RATIO
            and 0.0 < outward_distance <= maximum_distance
        ):
            continue
        candidates.append({
            "group": group,
            "evidence": glyph_evidence(group),
            "outward_distance": outward_distance,
            "height_ratio": height_ratio,
        })
    return sorted(
        candidates,
        key=lambda row: (
            float(row["outward_distance"]),
            tuple(row["group"]["primitive_ids"]),
        ),
    )


def _prefix_candidate_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _prefix_candidate_groups(
    *,
    success: Mapping[str, Any],
    primitives: Sequence[Mapping[str, Any]],
    ordered_cells: Sequence[Mapping[str, Any]],
    axis_angle_deg: float,
    reading_sign: int,
    reference_height: float,
    body_pitch: float,
) -> list[dict[str, Any]]:
    first_uv = _prefix_cell_uv_bbox(
        ordered_cells[0],
        axis_angle_deg=axis_angle_deg,
    )
    first_u = (first_uv[0] + first_uv[2]) / 2.0
    body_start_u = first_uv[0] if reading_sign > 0 else first_uv[2]
    v_padding = reference_height * _PREFIX_CROSS_AXIS_PAD_RATIO
    maximum_distance = body_pitch * _PREFIX_OUTSIDE_SEARCH_PITCHES
    maximum_primitive_span = (
        reference_height * _PREFIX_MAX_PRIMITIVE_SPAN_RATIO
    )
    claimed = {
        str(primitive_id)
        for cell in success["cells"]
        for primitive_id in cell["primitive_ids"]
    }
    if reading_sign > 0:
        query_u0 = body_start_u - maximum_distance
        query_u1 = body_start_u
    else:
        query_u0 = body_start_u
        query_u1 = body_start_u + maximum_distance
    page_bbox = directional_walker._landing_window_page_bbox(
        u0=query_u0,
        v0=first_uv[1] - v_padding,
        u1=query_u1,
        v1=first_uv[3] + v_padding,
        axis_angle_deg=axis_angle_deg,
    )
    rough_rows = directional_walker._query_primitive_rows(
        primitives,
        page_bbox,
    )
    projected_bboxes: dict[str, tuple[float, float, float, float]] = {}
    projected_bbox_lookup = getattr(primitives, "projected_bbox", None)

    def projected_bbox_for(
        primitive: Mapping[str, Any],
    ) -> tuple[float, float, float, float]:
        primitive_id = str(primitive["primitive_id"])
        result = projected_bboxes.get(primitive_id)
        if result is None:
            if callable(projected_bbox_lookup):
                result = projected_bbox_lookup(
                    primitive,
                    axis_angle_deg=axis_angle_deg,
                )
            else:
                result = _projected_points_bbox(
                    primitive.get("points", ()),
                    axis_angle_deg=axis_angle_deg,
                )
            projected_bboxes[primitive_id] = result
        return result

    local = _prefix_local_rows(
        rough_rows,
        claimed=claimed,
        projected_bbox_for=projected_bbox_for,
        first_uv=first_uv,
        body_start_u=body_start_u,
        reading_sign=reading_sign,
        v_padding=v_padding,
        maximum_distance=maximum_distance,
        maximum_primitive_span=maximum_primitive_span,
    )
    candidates = _prefix_candidate_rows(
        local,
        projected_bboxes=projected_bboxes,
        axis_angle_deg=axis_angle_deg,
        reference_height=reference_height,
        first_u=first_u,
        reading_sign=reading_sign,
        maximum_distance=maximum_distance,
    )
    audit = _PREFIX_QUERY_AUDIT.get()
    if audit is not None:
        observer, exhaustive = audit
        indexed_ids = tuple(str(row["primitive_id"]) for row in local)
        indexed_station_ids = tuple(
            tuple(str(value) for value in row["group"]["primitive_ids"])
            for row in candidates
        )
        exhaustive_ids: tuple[str, ...] | None = None
        exhaustive_station_ids: tuple[tuple[str, ...], ...] | None = None
        local_ordered_ids_equal: bool | None = None
        candidate_station_ordered_ids_equal: bool | None = None
        candidate_proposals_equal: bool | None = None
        exhaustive_equal: bool | None = None
        indexed_candidate_proposals_sha256 = _prefix_candidate_rows_sha256(
            candidates
        )
        exhaustive_candidate_proposals_sha256: str | None = None
        if exhaustive:
            exhaustive_local = _prefix_local_rows(
                primitives,
                claimed=claimed,
                projected_bbox_for=projected_bbox_for,
                first_uv=first_uv,
                body_start_u=body_start_u,
                reading_sign=reading_sign,
                v_padding=v_padding,
                maximum_distance=maximum_distance,
                maximum_primitive_span=maximum_primitive_span,
            )
            exhaustive_candidates = _prefix_candidate_rows(
                exhaustive_local,
                projected_bboxes=projected_bboxes,
                axis_angle_deg=axis_angle_deg,
                reference_height=reference_height,
                first_u=first_u,
                reading_sign=reading_sign,
                maximum_distance=maximum_distance,
            )
            exhaustive_ids = tuple(
                str(row["primitive_id"]) for row in exhaustive_local
            )
            exhaustive_station_ids = tuple(
                tuple(
                    str(value) for value in row["group"]["primitive_ids"]
                )
                for row in exhaustive_candidates
            )
            local_ordered_ids_equal = indexed_ids == exhaustive_ids
            candidate_station_ordered_ids_equal = (
                indexed_station_ids == exhaustive_station_ids
            )
            candidate_proposals_equal = candidates == exhaustive_candidates
            exhaustive_candidate_proposals_sha256 = (
                _prefix_candidate_rows_sha256(exhaustive_candidates)
            )
            exhaustive_equal = bool(
                local_ordered_ids_equal
                and candidate_station_ordered_ids_equal
                and candidate_proposals_equal
            )
        event = {
            "schema_version": "r92_prefix_query_audit_v1",
            "anchor_id": str(success["anchor"]["anchor_id"]),
            "axis_angle_deg": float(axis_angle_deg),
            "reading_sign": int(reading_sign),
            "page_query_bbox": tuple(float(value) for value in page_bbox),
            "page_grid_query_active": bool(
                getattr(primitives, "_bbox_index_safe", False)
            ),
            "page_primitive_count": len(primitives),
            "rough_primitive_count": len(rough_rows),
            "indexed_local_primitive_ids": indexed_ids,
            "indexed_candidate_station_primitive_ids": indexed_station_ids,
            "exhaustive_local_primitive_ids": exhaustive_ids,
            "exhaustive_candidate_station_primitive_ids": (
                exhaustive_station_ids
            ),
            "local_ordered_ids_equal": local_ordered_ids_equal,
            "candidate_station_ordered_ids_equal": (
                candidate_station_ordered_ids_equal
            ),
            "candidate_proposals_equal": candidate_proposals_equal,
            "indexed_candidate_proposals_sha256": (
                indexed_candidate_proposals_sha256
            ),
            "exhaustive_candidate_proposals_sha256": (
                exhaustive_candidate_proposals_sha256
            ),
            "exhaustive_equal": exhaustive_equal,
            "consumer_allowed": False,
        }
        observer(event)
        if exhaustive_equal is False:
            raise AssertionError(
                "indexed prefix local differs from exhaustive full scan: "
                f"anchor={event['anchor_id']!r} "
                f"indexed={indexed_ids!r} exhaustive={exhaustive_ids!r} "
                f"indexed_stations={indexed_station_ids!r} "
                f"exhaustive_stations={exhaustive_station_ids!r} "
                "indexed_proposals_sha256="
                f"{indexed_candidate_proposals_sha256} "
                "exhaustive_proposals_sha256="
                f"{exhaustive_candidate_proposals_sha256}"
            )
    return candidates


def _prefix_failure_row(
    *,
    anchor_id: str,
    input_text: str,
    reached_prefix_position: bool,
    candidate_groups: Sequence[Mapping[str, Any]],
    body_pitch: float | None,
    failure_reason: str,
    spacing: Mapping[str, Any] | None = None,
    recognition: Mapping[str, Any] | None = None,
    prefix_direction: str | None = None,
    original_walk_termination: Mapping[str, Any] | None = None,
    walk_termination_eligible: bool | None = None,
    outer_step_candidates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "anchor_id": anchor_id,
        "input_text": input_text,
        "output_text": input_text,
        "reached_prefix_position": reached_prefix_position,
        "recognized_prefix": None,
        "prepended": False,
        "failure_reason": failure_reason,
        "candidate_station_count": len(candidate_groups),
        "candidate_station_primitive_ids": [
            list(row["group"]["primitive_ids"])
            for row in candidate_groups
        ],
        "body_pitch_pt": body_pitch,
        "spacing": None if spacing is None else dict(spacing),
        "recognition": (
            None if recognition is None else dict(recognition)
        ),
        "accepted_prefix_primitive_ids": [],
        "ownership_conflicting_anchor_ids": [],
        "prefix_direction": prefix_direction,
        "walk_termination_eligible": walk_termination_eligible,
        "outer_step_candidate_count": len(outer_step_candidates),
        "outer_step_candidate_entry_ids": [
            str(row["entry_id"]) for row in outer_step_candidates
        ],
        "original_walk_termination": (
            None
            if original_walk_termination is None
            else dict(original_walk_termination)
        ),
        "replacement_walk_termination": None,
        "consumer_allowed": False,
    }


def _discover_prefix_for_success_v1(
    *,
    success: Mapping[str, Any],
    primitives: Sequence[Mapping[str, Any]],
    step_tables: Mapping[str, Any],
    character_table: Mapping[str, Any],
    template_library: Mapping[str, Any],
    template_recognizer: Callable[[dict[str, Any]], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    copied = dict(success)
    copied["cells"] = [dict(cell) for cell in success["cells"]]
    anchor_id = str(success["anchor"]["anchor_id"])
    axis_angle_deg = float(success["anchor"]["axis_angle_deg"])
    resolved = resolve_cell_direction_v1(copied["cells"])
    ordered_cells = list(resolved["cells"])
    input_text = "".join(str(cell["char"]) for cell in ordered_cells)
    if (
        input_text.startswith(("R", "Ø"))
        or (
            len(input_text) >= 2
            and input_text[0].isdigit()
            and input_text[1] == "X"
        )
    ):
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=False,
            candidate_groups=[],
            body_pitch=None,
            failure_reason="already_prefixed",
        )
    if len(ordered_cells) < 2:
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=False,
            candidate_groups=[],
            body_pitch=None,
            failure_reason="insufficient_body_cells",
        )
    cell_uv = [
        _prefix_cell_uv_bbox(cell, axis_angle_deg=axis_angle_deg)
        for cell in ordered_cells
    ]
    first_u = (cell_uv[0][0] + cell_uv[0][2]) / 2.0
    reading_delta = next(
        (
            ((uv[0] + uv[2]) / 2.0) - first_u
            for uv in cell_uv[1:]
            if abs(((uv[0] + uv[2]) / 2.0) - first_u) > 1e-9
        ),
        0.0,
    )
    if reading_delta == 0.0:
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=False,
            candidate_groups=[],
            body_pitch=None,
            failure_reason="reading_axis_direction_unresolved",
        )
    reading_sign = 1 if reading_delta > 0.0 else -1
    prefix_direction = "negative" if reading_sign > 0 else "positive"
    termination_key = f"{prefix_direction}_termination"
    raw_termination = copied.get(termination_key)
    original_termination = (
        dict(raw_termination)
        if isinstance(raw_termination, Mapping)
        else None
    )
    termination_eligible = bool(
        original_termination is not None
        and str(original_termination.get("reason"))
        in _PREFIX_ALLOWED_TERMINATION_REASONS
    )
    if not termination_eligible:
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=False,
            candidate_groups=[],
            body_pitch=None,
            failure_reason="walk_termination_not_prefix_eligible",
            prefix_direction=prefix_direction,
            original_walk_termination=original_termination,
            walk_termination_eligible=False,
        )
    reference_height = float(cell_uv[0][3]) - float(cell_uv[0][1])
    body_pitch = _prefix_body_pitch(
        ordered_cells,
        axis_angle_deg=axis_angle_deg,
        reference_height=reference_height,
    )
    if (
        not math.isfinite(reference_height)
        or reference_height <= 0.0
        or body_pitch is None
    ):
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=False,
            candidate_groups=[],
            body_pitch=body_pitch,
            failure_reason="body_geometry_unresolved",
            prefix_direction=prefix_direction,
            original_walk_termination=original_termination,
            walk_termination_eligible=True,
        )
    candidates = _prefix_candidate_groups(
        success=copied,
        primitives=primitives,
        ordered_cells=ordered_cells,
        axis_angle_deg=axis_angle_deg,
        reading_sign=reading_sign,
        reference_height=reference_height,
        body_pitch=body_pitch,
    )
    if not candidates:
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=False,
            candidate_groups=[],
            body_pitch=body_pitch,
            failure_reason="no_outside_character_station",
            prefix_direction=prefix_direction,
            original_walk_termination=original_termination,
            walk_termination_eligible=True,
        )
    nearest = candidates[0]
    nearest_ratio = float(nearest["outward_distance"]) / body_pitch
    nearest_uv = tuple(float(value) for value in nearest["group"]["uv_bbox"])
    first_uv = cell_uv[0]
    nearest_edge_gap = (
        first_uv[0] - nearest_uv[2]
        if reading_sign > 0
        else nearest_uv[0] - first_uv[2]
    )
    nearest_edge_gap_ratio = max(0.0, nearest_edge_gap) / body_pitch
    single_reached = (
        _PREFIX_SINGLE_EDGE_GAP_PITCH_RANGE[0]
        <= nearest_edge_gap_ratio
        <= _PREFIX_SINGLE_EDGE_GAP_PITCH_RANGE[1]
    )
    pair_ratio = None
    nx_reached = False
    if len(candidates) >= 2:
        pair_ratio = (
            float(candidates[1]["outward_distance"])
            - float(nearest["outward_distance"])
        ) / body_pitch
        nx_reached = (
            _PREFIX_NX_BRIDGE_PITCH_RANGE[0]
            <= nearest_ratio
            <= _PREFIX_NX_BRIDGE_PITCH_RANGE[1]
            and _PREFIX_NX_PAIR_PITCH_RANGE[0]
            <= pair_ratio
            <= _PREFIX_NX_PAIR_PITCH_RANGE[1]
        )
    reached = single_reached or nx_reached
    spacing = {
        "nearest_bridge_over_body_pitch": nearest_ratio,
        "nearest_edge_gap_over_body_pitch": nearest_edge_gap_ratio,
        "outer_pair_gap_over_body_pitch": pair_ratio,
        "single_prefix_geometry_passed": single_reached,
        "nx_prefix_geometry_passed": nx_reached,
    }
    nearest_character, nearest_entry = _single_character_entry(
        nearest["evidence"],
        character_table=character_table,
    )
    x_match = {
        "decision": "template_not_run",
        "status": "nx_prefix_geometry_not_reached",
        "top1_distance": None,
    }
    if nx_reached:
        x_match = _controlled_x_template_match(
            nearest["group"],
            template_library=template_library,
        )
    outer_character = None
    if len(candidates) >= 2:
        outer_character, _outer_entry = _single_character_entry(
            candidates[1]["evidence"],
            character_table=character_table,
        )
    recognition = {
        "nearest_shape_character": nearest_character,
        "outer_shape_character": outer_character,
        "controlled_x_template_decision": str(
            x_match.get("decision") or ""
        ),
        "controlled_x_template_status": str(x_match.get("status") or ""),
        "controlled_x_template_distance": x_match.get("top1_distance"),
        "controlled_x_template_id": x_match.get("top1_template_id"),
        "controlled_x_template_threshold": x_match.get("threshold"),
        "controlled_x_template_margin": x_match.get("margin"),
    }
    selected: list[dict[str, Any]] = []
    recognized_prefix = None
    if (
        single_reached
        and nearest_character in {"R", "Ø"}
        and nearest_entry is not None
    ):
        selected = [{
            **nearest,
            "character": nearest_character,
            "entry": nearest_entry,
            "template_match": None,
        }]
        recognized_prefix = nearest_character
    if not selected:
        failure_reason = "prefix_character_unrecognized"
        if nx_reached and x_match.get("decision") != "accepted_template":
            failure_reason = "controlled_x_template_rejected"
        elif nx_reached and x_match.get("decision") == "accepted_template":
            failure_reason = "controlled_x_requires_shape_authority"
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=reached,
            candidate_groups=candidates,
            body_pitch=body_pitch,
            failure_reason=failure_reason,
            spacing=spacing,
            recognition=recognition,
            prefix_direction=prefix_direction,
            original_walk_termination=original_termination,
            walk_termination_eligible=True,
        )
    selected_primitive_ids = {
        str(primitive_id)
        for row in selected
        for primitive_id in row["evidence"]["primitive_ids"]
    }
    outer_step_candidates = lookup_step_candidates(
        step_tables,
        direction=prefix_direction,
        previous_shape_key=str(selected[0]["evidence"]["step_key"]),
    )
    if outer_step_candidates:
        return copied, _prefix_failure_row(
            anchor_id=anchor_id,
            input_text=input_text,
            reached_prefix_position=True,
            candidate_groups=candidates,
            body_pitch=body_pitch,
            failure_reason="prefix_continuation_requires_walk",
            spacing=spacing,
            recognition=recognition,
            prefix_direction=prefix_direction,
            original_walk_termination=original_termination,
            walk_termination_eligible=True,
            outer_step_candidates=outer_step_candidates,
        )
    anchor_cell = next(
        cell for cell in copied["cells"]
        if cell["walk_direction"] == "anchor"
    )
    anchor_uv = _prefix_cell_uv_bbox(
        anchor_cell,
        axis_angle_deg=axis_angle_deg,
    )
    anchor_u = (anchor_uv[0] + anchor_uv[2]) / 2.0
    phrase_seed = stable_digest({
        "anchor_id": anchor_id,
        "primitive_ids": list(anchor_cell["primitive_ids"]),
        "identity_source": copied["anchor"]["axis_angle_source"],
    })
    direction_counts = {
        direction: max(
            (
                int(cell["step_index"])
                for cell in copied["cells"]
                if cell["walk_direction"] == direction
            ),
            default=0,
        )
        for direction in ("negative", "positive")
    }
    added_cells: list[dict[str, Any]] = []
    replaced_termination = None
    replacement_termination = None
    selected_by_distance = sorted(
        selected,
        key=lambda row: abs(_prefix_group_u_center(row["group"]) - anchor_u),
    )
    for row in selected_by_distance:
        direction = (
            "negative"
            if _prefix_group_u_center(row["group"]) < anchor_u
            else "positive"
        )
        if direction != prefix_direction:
            raise ValueError("prefix direction disagrees with resolved text start")
        direction_counts[direction] += 1
        added_cells.append(directional_walker._accepted_cell(
            group=row["group"],
            evidence=row["evidence"],
            dictionary_entry=row["entry"],
            key_character=str(row["character"]),
            template_recognizer=template_recognizer,
            walk_direction=direction,
            step_index=direction_counts[direction],
            step_evidence=None,
            phrase_seed=phrase_seed,
        ))
        termination_key = f"{direction}_termination"
        replaced_termination = dict(original_termination)
        replacement_termination = make_termination(
            direction=direction,
            reason="no_step_for_prev_key",
            attempted_step_index=direction_counts[direction] + 1,
            shape_key=str(row["evidence"]["step_key"]),
        )
        copied[termination_key] = replacement_termination
    copied["cells"].extend(added_cells)
    copied["cells"].sort(
        key=lambda cell: (
            sum(
                project_point(point, axis_angle_deg)[0]
                for point in cell["points"]
            ) / len(cell["points"]),
            str(cell["cell_id"]),
        )
    )
    output_text = "".join(
        str(cell["char"])
        for cell in resolve_cell_direction_v1(copied["cells"])["cells"]
    )
    return copied, {
        "anchor_id": anchor_id,
        "input_text": input_text,
        "output_text": output_text,
        "reached_prefix_position": True,
        "recognized_prefix": recognized_prefix,
        "prepended": True,
        "failure_reason": None,
        "candidate_station_count": len(candidates),
        "candidate_station_primitive_ids": [
            list(row["group"]["primitive_ids"])
            for row in candidates
        ],
        "body_pitch_pt": body_pitch,
        "spacing": spacing,
        "recognition": recognition,
        "accepted_prefix_primitive_ids": sorted(selected_primitive_ids),
        "ownership_conflicting_anchor_ids": [],
        "prefix_direction": prefix_direction,
        "walk_termination_eligible": True,
        "outer_step_candidate_count": 0,
        "outer_step_candidate_entry_ids": [],
        "original_walk_termination": replaced_termination,
        "replacement_walk_termination": replacement_termination,
        "consumer_allowed": False,
    }


def _discover_outside_prefixes_v1(
    *,
    successes: Sequence[Mapping[str, Any]],
    primitives: Sequence[Mapping[str, Any]],
    step_tables: Mapping[str, Any],
    character_table: Mapping[str, Any],
    template_library: Mapping[str, Any],
    template_recognizer: Callable[[dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_successes: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for success in successes:
        resolved, row = _discover_prefix_for_success_v1(
            success=success,
            primitives=primitives,
            step_tables=step_tables,
            character_table=character_table,
            template_library=template_library,
            template_recognizer=template_recognizer,
        )
        resolved_successes.append(resolved)
        rows.append(row)
    original_claims = [
        {
            str(primitive_id)
            for cell in success["cells"]
            for primitive_id in cell["primitive_ids"]
        }
        for success in successes
    ]
    proposal_claims = [
        {str(value) for value in row["accepted_prefix_primitive_ids"]}
        for row in rows
    ]
    ownership_conflicts: list[set[int]] = [set() for _row in rows]
    for index, prefix_claim in enumerate(proposal_claims):
        if not prefix_claim:
            continue
        for other_index, other_claim in enumerate(original_claims):
            if index != other_index and prefix_claim.intersection(other_claim):
                ownership_conflicts[index].add(other_index)
        for other_index in range(index + 1, len(proposal_claims)):
            if prefix_claim.intersection(proposal_claims[other_index]):
                ownership_conflicts[index].add(other_index)
                ownership_conflicts[other_index].add(index)
    for index, conflicts in enumerate(ownership_conflicts):
        if not conflicts:
            continue
        resolved_successes[index] = dict(successes[index])
        row = dict(rows[index])
        row.update({
            "output_text": row["input_text"],
            "recognized_prefix": None,
            "prepended": False,
            "failure_reason": "prefix_station_ownership_ambiguous",
            "accepted_prefix_primitive_ids": [],
            "ownership_conflicting_anchor_ids": sorted(
                str(successes[other]["anchor"]["anchor_id"])
                for other in conflicts
            ),
            "replacement_walk_termination": None,
        })
        rows[index] = row
    diagnostics = {
        "schema_version": "r92_outside_prefix_discovery_v1",
        "input_success_count": len(successes),
        "output_success_count": len(resolved_successes),
        "candidate_count_delta": len(resolved_successes) - len(successes),
        "reached_prefix_position_count": sum(
            row["reached_prefix_position"] for row in rows
        ),
        "recognized_prefix_count": sum(
            row["recognized_prefix"] is not None for row in rows
        ),
        "prepended_prefix_count": sum(row["prepended"] for row in rows),
        "ownership_conflict_count": sum(bool(row) for row in ownership_conflicts),
        "unique_station_claim_count": sum(
            bool(row["accepted_prefix_primitive_ids"]) for row in rows
        ),
        "rows": rows,
        "consumer_allowed": False,
    }
    return resolved_successes, diagnostics


def _bind_prefix_delivery_ledger_v1(
    diagnostics: Mapping[str, Any],
    *,
    before_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(before_rows) != len(after_rows):
        raise ValueError("prefix delivery population mismatch")
    pairs = list(zip(before_rows, after_rows, strict=True))
    if any(
        str(before["anchor_id"]) != str(after["anchor_id"])
        for before, after in pairs
    ):
        raise ValueError("prefix delivery ordering mismatch")
    by_anchor = {
        str(before["anchor_id"]): (before, after)
        for before, after in pairs
    }
    text_by_anchor = {
        str(row["anchor_id"]): (
            str(row["input_text"]),
            str(row["output_text"]),
        )
        for row in diagnostics["rows"]
    }
    rows = []
    for raw in diagnostics["rows"]:
        row = dict(raw)
        before, after = by_anchor[str(row["anchor_id"])]
        row["delivery_status_before_prefix"] = str(
            before["delivery_status"]
        )
        row["delivery_status_after_prefix"] = str(
            after["delivery_status"]
        )
        rows.append(row)
    status_changes = [
        {
            "anchor_id": str(before["anchor_id"]),
            "input_text": text_by_anchor[str(before["anchor_id"])][0],
            "output_text": text_by_anchor[str(before["anchor_id"])][1],
            "delivery_status_before_prefix": str(before["delivery_status"]),
            "delivery_status_after_prefix": str(after["delivery_status"]),
        }
        for before, after in pairs
        if (
            before["delivery_status"] != after["delivery_status"]
            or before["phrase_id"] != after["phrase_id"]
        )
    ]
    delivered_before = sum(
        row["delivery_status"] == "delivered" for row in before_rows
    )
    delivered_after = sum(
        row["delivery_status"] == "delivered" for row in after_rows
    )
    delivery_gain_count = sum(
        before["delivery_status"] != "delivered"
        and after["delivery_status"] == "delivered"
        for before, after in pairs
    )
    delivery_loss_count = sum(
        before["delivery_status"] == "delivered"
        and after["delivery_status"] != "delivered"
        for before, after in pairs
    )
    result = dict(diagnostics)
    result.update({
        "delivery_accounting_boundary_before": (
            "post_walk_pre_prefix_pre_fragment"
        ),
        "delivery_accounting_boundary_after": (
            "post_prefix_pre_fragment"
        ),
        "delivered_success_count_before_prefix": delivered_before,
        "delivered_success_count_after_prefix": delivered_after,
        "delivery_candidate_delta": delivered_after - delivered_before,
        "delivery_gain_count": delivery_gain_count,
        "delivery_loss_count": delivery_loss_count,
        "delivery_balance_closed": (
            delivered_after - delivered_before
            == delivery_gain_count - delivery_loss_count
        ),
        "delivery_status_change_count": sum(
            before["delivery_status"] != after["delivery_status"]
            for before, after in pairs
        ),
        "delivery_or_text_change_rows": status_changes,
        "rows": rows,
    })
    return result


def _bind_prefix_final_delivery_v1(
    diagnostics: Mapping[str, Any],
    *,
    direction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_by_anchor: dict[str, list[Mapping[str, Any]]] = {}
    for direction_row in direction_rows:
        for anchor_id in direction_row.get("source_anchor_ids", ()):
            final_by_anchor.setdefault(str(anchor_id), []).append(direction_row)
    rows = []
    for raw in diagnostics["rows"]:
        row = dict(raw)
        final_rows = final_by_anchor.get(str(row["anchor_id"]), [])
        statuses = sorted({
            str(final["delivery_status"])
            for final in final_rows
        })
        row["final_delivery_statuses"] = statuses
        row["final_delivered"] = any(
            status == "delivered" for status in statuses
        )
        row["final_output_texts"] = sorted({
            str(final["output_text"])
            for final in final_rows
        })
        row["delivered_at_prefix_boundary"] = (
            str(row["delivery_status_after_prefix"]) == "delivered"
        )
        row["delivered_prefix"] = bool(
            row["prepended"] and row["final_delivered"]
        )
        rows.append(row)
    delivered_at_prefix_boundary = sum(
        row["delivered_at_prefix_boundary"] for row in rows
    )
    delivered_final = sum(row["final_delivered"] for row in rows)
    final_source_delivery_gain_count = sum(
        not row["delivered_at_prefix_boundary"] and row["final_delivered"]
        for row in rows
    )
    final_source_delivery_loss_count = sum(
        row["delivered_at_prefix_boundary"] and not row["final_delivered"]
        for row in rows
    )
    final_source_delivery_change_rows = [
        {
            "anchor_id": str(row["anchor_id"]),
            "output_text": str(row["output_text"]),
            "delivery_status_at_prefix_boundary": str(
                row["delivery_status_after_prefix"]
            ),
            "final_delivery_statuses": list(row["final_delivery_statuses"]),
            "final_output_texts": list(row["final_output_texts"]),
        }
        for row in rows
        if row["delivered_at_prefix_boundary"] != row["final_delivered"]
    ]
    result = dict(diagnostics)
    result.update({
        "final_source_delivery_accounting_boundary_before": (
            "post_prefix_pre_fragment"
        ),
        "final_source_delivery_accounting_boundary_after": (
            "post_claim_direction_resolution"
        ),
        "final_source_delivery_counting_unit": "source_anchor_success",
        "final_candidate_conservation_authority": (
            "fragment_merge_and_claim_arbitration_diagnostics"
        ),
        "delivered_success_count_at_prefix_boundary": (
            delivered_at_prefix_boundary
        ),
        "final_delivered_source_success_count": delivered_final,
        "final_source_delivery_coverage_delta": (
            delivered_final - delivered_at_prefix_boundary
        ),
        "final_source_delivery_gain_count": final_source_delivery_gain_count,
        "final_source_delivery_loss_count": final_source_delivery_loss_count,
        "final_source_delivery_balance_closed": (
            delivered_final - delivered_at_prefix_boundary
            == final_source_delivery_gain_count
            - final_source_delivery_loss_count
        ),
        "final_source_delivery_status_change_count": len(
            final_source_delivery_change_rows
        ),
        "final_source_delivery_change_rows": final_source_delivery_change_rows,
        "delivered_prepended_prefix_count": sum(
            row["delivered_prefix"] for row in rows
        ),
        "delivered_recognized_prefix_count": sum(
            row["recognized_prefix"] is not None
            and row["delivered_prefix"]
            for row in rows
        ),
        "prepended_but_not_delivered_count": sum(
            row["prepended"] and not row["final_delivered"]
            for row in rows
        ),
        "rows": rows,
    })
    return result


def _walk_shape_anchor(
    *,
    primitives: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    step_tables: Mapping[str, Any],
    character_table: Mapping[str, Any],
    template_recognizer: Callable[[dict[str, Any]], Any],
    content_order_step_binding_enabled: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    group = _shape_source_group(primitives=primitives, anchor=anchor)
    anchor_evidence = glyph_evidence(group)
    expected = _ANCHOR_TYPES[str(anchor["anchor_type"])]
    phrase_seed = stable_digest({
        "anchor_id": anchor["anchor_id"],
        "primitive_ids": list(anchor_evidence["primitive_ids"]),
        "identity_source": anchor["axis_angle_source"],
    })
    anchor_cell = directional_walker._accepted_cell(
        group=group,
        evidence=anchor_evidence,
        dictionary_entry={
            "entry_id": f"shape-native:{anchor['anchor_type']}",
            "width_disambiguated": False,
            "width_evidence": None,
        },
        key_character=expected,
        template_recognizer=lambda _evidence: None,
        walk_direction="anchor",
        step_index=0,
        step_evidence=None,
        phrase_seed=phrase_seed,
    )
    initial_claimed = set(anchor_evidence["primitive_ids"])
    negative_cells, negative_stop, negative_rejections = (
        directional_walker._walk_direction(
            direction="negative",
            sign=-1,
            start_group=group,
            start_evidence=anchor_evidence,
            primitives=primitives,
            claimed=set(initial_claimed),
            step_tables=step_tables,
            character_table=character_table,
            template_recognizer=template_recognizer,
            axis_angle_deg=float(anchor["axis_angle_deg"]),
            phrase_seed=phrase_seed,
            anchor_id=str(anchor["anchor_id"]),
            content_order_step_binding_enabled=(
                content_order_step_binding_enabled
            ),
        )
    )
    positive_cells, positive_stop, positive_rejections = (
        directional_walker._walk_direction(
            direction="positive",
            sign=1,
            start_group=group,
            start_evidence=anchor_evidence,
            primitives=primitives,
            claimed=set(initial_claimed),
            step_tables=step_tables,
            character_table=character_table,
            template_recognizer=template_recognizer,
            axis_angle_deg=float(anchor["axis_angle_deg"]),
            phrase_seed=phrase_seed,
            anchor_id=str(anchor["anchor_id"]),
            content_order_step_binding_enabled=(
                content_order_step_binding_enabled
            ),
        )
    )
    negative_claims = {
        value for cell in negative_cells for value in cell["primitive_ids"]
    }
    positive_claims = {
        value for cell in positive_cells for value in cell["primitive_ids"]
    }
    conflicts = sorted(negative_claims & positive_claims)
    cells = list(reversed(negative_cells)) + [anchor_cell] + positive_cells
    internal_overlap = None
    if conflicts:
        cell_resolution = resolve_internal_cell_overlap_v1(cells)
        if not cell_resolution["compatible"]:
            return None, {
                "anchor_id": anchor["anchor_id"],
                "reason": "primitive_claim_conflict",
                "claimed_primitive_ids": [],
                "conflicting_primitive_ids": conflicts,
                "conflict_scope": "independent_direction_overlap",
            }
        cells = list(cell_resolution["cells"])
        claimed_after_resolution = {
            str(value)
            for cell in cells
            for value in cell["primitive_ids"]
        }
        if any(
            claimed_after_resolution.intersection(
                str(value)
                for value in termination.get("primitive_ids", ())
            )
            for termination in (negative_stop, positive_stop)
        ):
            return None, {
                "anchor_id": anchor["anchor_id"],
                "reason": "primitive_claim_conflict",
                "claimed_primitive_ids": [],
                "conflicting_primitive_ids": conflicts,
                "conflict_scope": "independent_direction_overlap",
            }
        internal_overlap = {
            "conflicting_primitive_ids": conflicts,
            "input_cell_count": (
                len(negative_cells) + 1 + len(positive_cells)
            ),
            "output_cell_count": len(cells),
            "deduplicated_cell_count": cell_resolution[
                "deduplicated_cell_count"
            ],
        }
    success = {
        "anchor": anchor,
        "cells": cells,
        "negative_termination": negative_stop,
        "positive_termination": positive_stop,
        "landing_group_rejections": [
            *negative_rejections,
            *positive_rejections,
        ],
    }
    if internal_overlap is not None:
        success["_internal_direction_overlap"] = internal_overlap
    return success, None


def _bind_real_gdt_overlap(
    phrase: dict[str, Any],
    *,
    gdt_frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    phrase_box = _bbox_xyxy(phrase["bbox"])
    inside_ids: list[str] = []
    boundary_ids: list[str] = []
    for index, raw in enumerate(gdt_frames):
        frame = raw.get("frame") if isinstance(raw, Mapping) else None
        frame = frame if isinstance(frame, Mapping) else raw
        if not isinstance(frame, Mapping):
            continue
        frame_box = _optional_bbox_xyxy(frame.get("bbox"))
        if frame_box is None or not _boxes_overlap(phrase_box, frame_box):
            continue
        frame_id = str(
            frame.get("frame_id")
            or raw.get("frame_id")
            or "gdt_frame_"
            + stable_digest({
                "page_index": phrase["page_index"],
                "index": index,
                "bbox": list(frame_box),
            })[:24]
        )
        if _box_contains(frame_box, phrase_box):
            inside_ids.append(frame_id)
        else:
            boundary_ids.append(frame_id)
    if boundary_ids:
        return phrase, "gdt_frame_boundary_crossing"
    ownership = sorted(set(inside_ids))
    phrase["gdt_overlap"] = {
        "frame_ids": ownership,
        "inside_frame_ids": ownership,
        "boundary_frame_ids": [],
        "crosses_frame_boundary": False,
        "ordinary_eligible": not ownership,
        "consumer_allowed": False,
    }
    phrase["gdt_ownership_frame_ids"] = ownership
    terminations = phrase["walk"]["terminations"]
    phrase["physical_core_digest"] = stable_digest({
        "page_index": int(phrase["page_index"]),
        "gdt_ownership_frame_ids": ownership,
        "axis_angle_deg": round(float(phrase["axis_angle_deg"]), 6),
        "axis_directionality": phrase["axis_directionality"],
        "bbox": phrase["bbox"],
        "primitive_run_ids": phrase["primitive_run_ids"],
        "negative_stop_digest": terminations["negative"]["semantic_digest"],
        "positive_stop_digest": terminations["positive"]["semantic_digest"],
    })
    return phrase, None


def _template_recognizer(
    template_library: Mapping[str, Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def recognize(evidence: dict[str, Any]) -> dict[str, Any]:
        return classify_fixed_digit_points(
            [list(point) for point in evidence.get("points", ())],
            template_library=dict(template_library),
            threshold=0.008,
            margin=0.010,
            base_templates_only=False,
        )

    return recognize


def _format_result(text: str) -> dict[str, Any]:
    # Kept local to avoid importing the quarantined reader/segmenter.
    from r2p_dimension_format_validator import validate_dimension_text_format

    return validate_dimension_text_format(text)


def _orientation_from_axis(value: Any) -> str:
    angle = float(value) % 180.0
    return "V" if abs(angle - 90.0) <= 45.0 else "H"


def _bbox_xywh(value: Mapping[str, Any]) -> dict[str, float]:
    x = float(value["x"])
    y = float(value["y"])
    width = float(value["w"])
    height = float(value["h"])
    if (
        not all(math.isfinite(item) for item in (x, y, width, height))
        or width <= 0.0
        or height <= 0.0
    ):
        raise ValueError("directional_walk_anchor_bbox_invalid")
    return {"x": x, "y": y, "w": width, "h": height}


def _bbox_xyxy(value: Mapping[str, Any]) -> tuple[float, float, float, float]:
    box = _bbox_xywh(value)
    return box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]


def _optional_bbox_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        if set(("x", "y", "w", "h")).issubset(value):
            try:
                return _bbox_xyxy(value)
            except (KeyError, TypeError, ValueError):
                return None
        keys = ("x0", "y0", "x1", "y1")
        if all(key in value for key in keys):
            try:
                result = tuple(float(value[key]) for key in keys)
            except (TypeError, ValueError):
                return None
            return result if result[2] > result[0] and result[3] > result[1] else None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        return result if result[2] > result[0] and result[3] > result[1] else None
    return None


def _boxes_overlap(
    left: Sequence[float],
    right: Sequence[float],
) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _box_contains(
    outer: Sequence[float],
    inner: Sequence[float],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


__all__ = [
    "directional_walk_asset_identity_v1",
    "build_directional_walk_pitch_projection_v1",
    "build_shape_native_directional_walk_dump_v1",
    "load_directional_walk_assets_v1",
]


def directional_walk_asset_identity_v1() -> dict[str, Any]:
    """Return the pinned identities after loading every production asset."""

    load_directional_walk_assets_v1()
    return {
        "schema_version": "r92_directional_walk_asset_identity_v1",
        "manifest_sha256": _ASSET_MANIFEST_SHA256,
        "step_table_sha256": _ASSET_SHA256["step_table_v1.json"],
        "character_table_sha256": _ASSET_SHA256["character_table_v1.json"],
        "controlled_template_sha256": _CONTROLLED_TEMPLATE_SHA256,
        "consumer_allowed": False,
    }
