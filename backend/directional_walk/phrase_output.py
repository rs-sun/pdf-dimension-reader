"""Construction helpers for the R92 directional-walk output contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

from .glyph_key import (
    bbox_union,
    bbox_xywh,
    canonical_json,
    project_point,
    unproject_point,
)


TERMINATION_REASONS = frozenset(
    {
        "no_group_at_landing",
        "key_not_in_table",
        "key_ambiguous",
        "post_arbitration_key_ambiguous",
        "no_step_for_prev_key",
    }
)
AXIS_DIRECTIONALITIES = frozenset({"directed", "undirected_fallback"})
AXIS_SIGN_EPSILON = 1e-12
FORBIDDEN_LEGACY_FIELDS = frozenset(
    {
        "view_id",
        "view_partition_key",
        "view_ownership",
        "view_boundary",
        "crosses_boundary",
        "view_bbox",
        "candidate_view_ids",
        "corridor_bbox",
        "datum_1",
        "datum_2",
        "datum_3",
    }
)
SHA256_HEX = frozenset("0123456789abcdef")


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_axis_angle(angle_deg: Any, *, directionality: str) -> float:
    """Normalize a directed axis, or canonicalize an undirected fallback vector."""

    if directionality not in AXIS_DIRECTIONALITIES:
        raise ValueError("axis directionality invalid")
    try:
        angle = float(angle_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("axis angle must be finite") from exc
    if not math.isfinite(angle):
        raise ValueError("axis angle must be finite")
    angle = round(angle % 360.0, 6) % 360.0
    if directionality == "undirected_fallback":
        radians = math.radians(angle)
        dx, dy = math.cos(radians), math.sin(radians)
        if dx < -AXIS_SIGN_EPSILON or (
            abs(dx) <= AXIS_SIGN_EPSILON and dy < 0.0
        ):
            angle = (angle + 180.0) % 360.0
    return angle


def make_termination(
    *,
    direction: str,
    reason: str,
    attempted_step_index: int,
    shape_key: str | None,
    primitive_ids: Sequence[str] = (),
    bbox: Mapping[str, Any] | None = None,
    s001: Sequence[float | int] | None = None,
    free_endpoint_count: int | None = None,
    candidate_dictionary_entry_ids: Sequence[str] = (),
) -> dict[str, Any]:
    payload = {
        "schema_version": "r92_walk_termination_v1",
        "direction": direction,
        "reason": reason,
        "attempted_step_index": int(attempted_step_index),
        "shape_key": shape_key,
        "primitive_ids": sorted({str(value) for value in primitive_ids}),
        "bbox": None if bbox is None else dict(bbox),
        "s001": None if s001 is None else list(s001),
        "free_endpoint_count": free_endpoint_count,
        "candidate_dictionary_entry_ids": sorted(
            {str(value) for value in candidate_dictionary_entry_ids}
        ),
        "consumer_allowed": False,
    }
    return {**payload, "semantic_digest": stable_digest(payload)}


def _oriented_quad(cells: Sequence[Mapping[str, Any]], angle_deg: float) -> list[list[float]]:
    points: list[tuple[float, float]] = []
    for cell in cells:
        bbox = cell["bbox"]
        x0, y0 = float(bbox["x"]), float(bbox["y"])
        x1 = x0 + float(bbox["w"])
        y1 = y0 + float(bbox["h"])
        points.extend(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    projected = [project_point(point, angle_deg) for point in points]
    u0 = min(point[0] for point in projected)
    v0 = min(point[1] for point in projected)
    u1 = max(point[0] for point in projected)
    v1 = max(point[1] for point in projected)
    return [
        list(unproject_point(point, angle_deg))
        for point in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    ]


def build_phrase_v1(
    *,
    trace_id: str,
    page_index: int,
    anchor: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    negative_termination: Mapping[str, Any],
    positive_termination: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_cells: list[dict[str, Any]] = []
    for reading_index, raw in enumerate(cells):
        cell = dict(raw)
        cell["reading_index"] = reading_index
        cell.pop("points", None)
        ordered_cells.append(cell)
    primitive_ids = sorted(
        {
            str(value)
            for cell in ordered_cells
            for value in cell["primitive_ids"]
        }
    )
    bbox_xyxy = bbox_union(
        (
            cell["bbox"]["x"],
            cell["bbox"]["y"],
            cell["bbox"]["x"] + cell["bbox"]["w"],
            cell["bbox"]["y"] + cell["bbox"]["h"],
        )
        for cell in ordered_cells
    )
    axis_directionality = str(anchor["axis_directionality"])
    axis_angle_deg = normalize_axis_angle(
        anchor["axis_angle_deg"],
        directionality=axis_directionality,
    )
    anchor_ids = sorted({str(anchor["anchor_id"])})
    source_candidate_ids = sorted(
        {str(value) for value in anchor.get("source_candidate_ids", ())}
    )
    source_region_ids = sorted(
        {str(value) for value in anchor.get("source_region_ids", ())}
    )
    source_anchor_evidence = [dict(anchor["source_anchor_evidence"])]
    provenance_payload = {
        "anchor_ids": anchor_ids,
        "source_candidate_ids": source_candidate_ids,
        "source_anchor_evidence": source_anchor_evidence,
        "source_region_ids": source_region_ids,
    }
    gdt_frame_ids = sorted(
        {str(value) for value in anchor.get("gdt_ownership_frame_ids", ())}
    )
    bbox = bbox_xywh(bbox_xyxy)
    physical_payload = {
        "page_index": int(page_index),
        "gdt_ownership_frame_ids": gdt_frame_ids,
        "axis_angle_deg": axis_angle_deg,
        "axis_directionality": axis_directionality,
        "bbox": bbox,
        "primitive_run_ids": primitive_ids,
        "negative_stop_digest": negative_termination["semantic_digest"],
        "positive_stop_digest": positive_termination["semantic_digest"],
    }
    phrase_id = "r92_phrase_" + stable_digest(
        {
            "page_index": int(page_index),
            "primitive_run_ids": primitive_ids,
            "anchor_ids": anchor_ids,
        }
    )[:24]
    gdt_overlap = dict(
        anchor.get(
            "gdt_overlap",
            {
                "frame_ids": gdt_frame_ids,
                "inside_frame_ids": gdt_frame_ids,
                "boundary_frame_ids": [],
                "crosses_frame_boundary": False,
                "ordinary_eligible": not gdt_frame_ids,
                "consumer_allowed": False,
            },
        )
    )
    return {
        "schema_version": "r92_directional_walk_phrase_v1",
        "trace_id": trace_id,
        "page_index": int(page_index),
        "phrase_id": phrase_id,
        "physical_core_digest": stable_digest(physical_payload),
        "provenance_digest": stable_digest(provenance_payload),
        "axis_angle_deg": axis_angle_deg,
        "axis_angle_source": str(anchor["axis_angle_source"]),
        "axis_directionality": axis_directionality,
        "axis_trusted": True,
        "bbox": bbox,
        "oriented_quad": _oriented_quad(cells, axis_angle_deg),
        "primitive_run_ids": primitive_ids,
        "anchor_ids": anchor_ids,
        "source_candidate_ids": source_candidate_ids,
        "source_anchor_evidence": source_anchor_evidence,
        "source_region_ids": source_region_ids,
        "text": "".join(str(cell["char"]) for cell in ordered_cells),
        "walk": {
            "schema_version": "r92_directional_walk_evidence_v1",
            "cells": ordered_cells,
            "terminations": {
                "negative": dict(negative_termination),
                "positive": dict(positive_termination),
            },
            "consumer_allowed": False,
        },
        "gdt_overlap": gdt_overlap,
        "gdt_ownership_frame_ids": gdt_frame_ids,
        "consumer_allowed": False,
    }


def build_dump_v1(
    *,
    trace_id: str,
    page_index: int,
    page_context_manifest: Mapping[str, Any],
    phrases: Sequence[Mapping[str, Any]],
    anchor_failures: Sequence[Mapping[str, Any]],
    additional_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in phrases), key=lambda row: row["phrase_id"])
    failures = [dict(row) for row in anchor_failures]
    cells = sum(len(row["walk"]["cells"]) for row in ordered)
    diagnostics = {"anchor_failures": failures}
    if additional_diagnostics is not None:
        diagnostics.update(dict(additional_diagnostics))
    return {
        "schema_version": "r92_directional_walk_dump_v1",
        "trace_id": trace_id,
        "page_index": int(page_index),
        "status": "ok",
        "error": None,
        "page_context_manifest": dict(page_context_manifest),
        "phrases": ordered,
        "stats": {
            "phrase_count": len(ordered),
            "cell_count": cells,
            "matched_cell_count": cells,
            "termination_count": 2 * len(ordered),
            "consumer_allowed_count": 0,
            "anchor_group_unresolved_count": sum(
                row.get("reason") == "anchor_group_unresolved" for row in failures
            ),
            "anchor_group_unbounded_count": sum(
                row.get("reason") == "anchor_group_unbounded" for row in failures
            ),
        },
        "diagnostics": diagnostics,
        "consumer_allowed": False,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_HEX)
    )


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _sorted_unique_strings(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and value == sorted(set(value))
        and (not nonempty or bool(value))
    )


def _validate_bbox(
    value: Any,
    *,
    page_width: float,
    page_height: float,
    allow_degenerate: bool = True,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"x", "y", "w", "h"}:
        raise ValueError("bbox fields invalid")
    if not all(_finite_number(value[key]) for key in ("x", "y", "w", "h")):
        raise ValueError("bbox values invalid")
    width = float(value["w"])
    height = float(value["h"])
    if (
        (width < 0.0 if allow_degenerate else width <= 0.0)
        or (height < 0.0 if allow_degenerate else height <= 0.0)
        or float(value["x"]) < 0.0
        or float(value["y"]) < 0.0
        or float(value["x"]) + width > page_width + 1e-9
        or float(value["y"]) + height > page_height + 1e-9
    ):
        raise ValueError("bbox lies outside the page")


def _walk_objects(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_objects(child)


def _validate_no_legacy_fields(value: Mapping[str, Any]) -> None:
    for row in _walk_objects(value):
        overlap = set(row).intersection(FORBIDDEN_LEGACY_FIELDS)
        if overlap:
            raise ValueError(f"forbidden legacy field: {sorted(overlap)[0]}")
        if "consumer_allowed" in row and row["consumer_allowed"] is not False:
            raise ValueError("consumer_allowed must remain false throughout the tree")


def _validate_termination(
    value: Any,
    *,
    direction: str,
    claimed_primitive_ids: set[str],
    page_width: float,
    page_height: float,
) -> None:
    required = {
        "schema_version",
        "direction",
        "reason",
        "attempted_step_index",
        "shape_key",
        "primitive_ids",
        "bbox",
        "s001",
        "free_endpoint_count",
        "candidate_dictionary_entry_ids",
        "semantic_digest",
        "consumer_allowed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("termination fields invalid")
    if (
        value["schema_version"] != "r92_walk_termination_v1"
        or value["direction"] != direction
        or value["reason"] not in TERMINATION_REASONS
    ):
        raise ValueError("termination reason or identity invalid")
    if type(value["attempted_step_index"]) is not int or value["attempted_step_index"] < 0:
        raise ValueError("termination attempted step invalid")
    if not _sorted_unique_strings(value["primitive_ids"]):
        raise ValueError("termination primitive ids invalid")
    if not _sorted_unique_strings(value["candidate_dictionary_entry_ids"]):
        raise ValueError("termination dictionary entry ids invalid")
    if claimed_primitive_ids.intersection(value["primitive_ids"]):
        raise ValueError("termination evidence was incorrectly claimed")
    reason = value["reason"]
    if reason in {"no_group_at_landing", "no_step_for_prev_key"}:
        if any(
            value[key] is not None
            for key in ("bbox", "s001", "free_endpoint_count")
        ) or value["primitive_ids"] or value["candidate_dictionary_entry_ids"]:
            raise ValueError("empty termination carries observed glyph evidence")
        if reason == "no_group_at_landing" and value["shape_key"] is not None:
            raise ValueError("no-group termination shape key must be null")
        if reason == "no_step_for_prev_key" and not isinstance(value["shape_key"], str):
            raise ValueError("no-step termination must retain the previous key")
    else:
        if not isinstance(value["shape_key"], str) or not value["shape_key"]:
            raise ValueError("observed termination shape key missing")
        if not value["primitive_ids"] or value["bbox"] is None or value["s001"] is None:
            raise ValueError("observed termination evidence incomplete")
        _validate_bbox(value["bbox"], page_width=page_width, page_height=page_height)
        if (
            not isinstance(value["s001"], list)
            or not value["s001"]
            or not all(_finite_number(item) for item in value["s001"])
            or type(value["free_endpoint_count"]) is not int
            or value["free_endpoint_count"] < 0
        ):
            raise ValueError("observed termination S001 evidence invalid")
        if reason == "key_not_in_table" and value["candidate_dictionary_entry_ids"]:
            raise ValueError("unknown key cannot cite dictionary candidates")
    digest_payload = {key: item for key, item in value.items() if key != "semantic_digest"}
    if value["semantic_digest"] != stable_digest(digest_payload):
        raise ValueError("termination semantic digest invalid")


def _validate_source_anchor(
    value: Any,
    *,
    trace_page_index: int,
    axis_angle_deg: float,
    axis_source: str,
    axis_directionality: str,
    primitive_ids: set[str],
    page_width: float,
    page_height: float,
) -> None:
    required = {
        "schema_version",
        "source_schema_version",
        "source_id",
        "anchor_id",
        "page_index",
        "anchor_type",
        "anchor_bbox",
        "axis_angle_deg",
        "axis_angle_source",
        "axis_directionality",
        "axis_trusted",
        "source_primitive_ids",
        "source_region_ids",
        "consumer_allowed",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source anchor evidence fields invalid")
    if (
        value["schema_version"] != "r92_source_anchor_evidence_v1"
        or not isinstance(value["source_schema_version"], str)
        or not value["source_schema_version"]
        or not isinstance(value["source_id"], str)
        or not value["source_id"]
        or not isinstance(value["anchor_id"], str)
        or not value["anchor_id"]
        or value["page_index"] != trace_page_index
        or value["axis_trusted"] is not True
        or float(value["axis_angle_deg"]) != axis_angle_deg
        or value["axis_angle_source"] != axis_source
        or value["axis_directionality"] != axis_directionality
    ):
        raise ValueError("source anchor evidence identity invalid")
    _validate_bbox(
        value["anchor_bbox"],
        page_width=page_width,
        page_height=page_height,
        allow_degenerate=False,
    )
    if not _sorted_unique_strings(value["source_primitive_ids"], nonempty=True):
        raise ValueError("source anchor primitive ids invalid")
    if not set(value["source_primitive_ids"]).issubset(primitive_ids):
        raise ValueError("source anchor primitive is not owned by the phrase")
    if not _sorted_unique_strings(value["source_region_ids"]):
        raise ValueError("source anchor region ids invalid")


def _validate_phrase(
    phrase: Any,
    *,
    trace_id: str,
    page_index: int,
    page_width: float,
    page_height: float,
) -> set[str]:
    required = {
        "schema_version",
        "trace_id",
        "page_index",
        "phrase_id",
        "physical_core_digest",
        "provenance_digest",
        "axis_angle_deg",
        "axis_angle_source",
        "axis_directionality",
        "axis_trusted",
        "bbox",
        "oriented_quad",
        "primitive_run_ids",
        "anchor_ids",
        "source_candidate_ids",
        "source_anchor_evidence",
        "source_region_ids",
        "text",
        "walk",
        "gdt_overlap",
        "gdt_ownership_frame_ids",
        "consumer_allowed",
    }
    if not isinstance(phrase, Mapping) or set(phrase) != required:
        raise ValueError("phrase fields invalid")
    if (
        phrase["schema_version"] != "r92_directional_walk_phrase_v1"
        or phrase["trace_id"] != trace_id
        or phrase["page_index"] != page_index
        or not isinstance(phrase["phrase_id"], str)
        or not phrase["phrase_id"]
        or not _valid_sha256(phrase["physical_core_digest"])
        or not _valid_sha256(phrase["provenance_digest"])
        or phrase["axis_trusted"] is not True
        or not _finite_number(phrase["axis_angle_deg"])
        or not 0.0 <= float(phrase["axis_angle_deg"]) < 360.0
        or not isinstance(phrase["axis_angle_source"], str)
        or not phrase["axis_angle_source"]
        or phrase["axis_directionality"] not in AXIS_DIRECTIONALITIES
    ):
        raise ValueError("phrase identity invalid")
    if float(phrase["axis_angle_deg"]) != normalize_axis_angle(
        phrase["axis_angle_deg"],
        directionality=phrase["axis_directionality"],
    ):
        raise ValueError("phrase axis normalization invalid")
    _validate_bbox(
        phrase["bbox"],
        page_width=page_width,
        page_height=page_height,
        allow_degenerate=False,
    )
    quad = phrase["oriented_quad"]
    if (
        not isinstance(quad, list)
        or len(quad) != 4
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or not all(_finite_number(item) for item in point)
            for point in quad
        )
    ):
        raise ValueError("phrase oriented quad invalid")
    for key, nonempty in (
        ("primitive_run_ids", True),
        ("anchor_ids", True),
        ("source_candidate_ids", False),
        ("source_region_ids", False),
        ("gdt_ownership_frame_ids", False),
    ):
        if not _sorted_unique_strings(phrase[key], nonempty=nonempty):
            raise ValueError(f"phrase {key} invalid")
    primitive_ids = set(phrase["primitive_run_ids"])
    source_rows = phrase["source_anchor_evidence"]
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("source anchor evidence missing")
    for source in source_rows:
        _validate_source_anchor(
            source,
            trace_page_index=page_index,
            axis_angle_deg=float(phrase["axis_angle_deg"]),
            axis_source=phrase["axis_angle_source"],
            axis_directionality=phrase["axis_directionality"],
            primitive_ids=primitive_ids,
            page_width=page_width,
            page_height=page_height,
        )
    walk = phrase["walk"]
    if (
        not isinstance(walk, Mapping)
        or set(walk) != {"schema_version", "cells", "terminations", "consumer_allowed"}
        or walk["schema_version"] != "r92_directional_walk_evidence_v1"
        or not isinstance(walk["cells"], list)
        or not walk["cells"]
    ):
        raise ValueError("walk evidence invalid")
    cell_ids: set[str] = set()
    cell_primitives: set[str] = set()
    cell_bboxes: list[tuple[float, float, float, float]] = []
    characters: list[str] = []
    for index, cell in enumerate(walk["cells"]):
        required_cell = {
            "cell_id",
            "reading_index",
            "walk_direction",
            "step_index",
            "char",
            "key_character",
            "template_character",
            "characters_agree",
            "primitive_ids",
            "bbox",
            "shape_key",
            "s001",
            "free_endpoint_count",
            "dictionary_entry_id",
            "width_disambiguated",
            "width_evidence",
            "step_evidence",
            "consumer_allowed",
        }
        if not isinstance(cell, Mapping) or set(cell) != required_cell:
            raise ValueError("cell fields invalid")
        if (
            not isinstance(cell["cell_id"], str)
            or not cell["cell_id"]
            or cell["cell_id"] in cell_ids
            or cell["reading_index"] != index
            or cell["walk_direction"] not in {"negative", "anchor", "positive"}
            or type(cell["step_index"]) is not int
            or cell["step_index"] < 0
            or not isinstance(cell["key_character"], str)
            or not cell["key_character"]
            or cell["char"] != cell["key_character"]
            or (
                cell["template_character"] is not None
                and (
                    not isinstance(cell["template_character"], str)
                    or not cell["template_character"]
                )
            )
            or cell["characters_agree"]
            is not (
                None
                if cell["template_character"] is None
                else cell["template_character"] == cell["key_character"]
            )
            or not isinstance(cell["shape_key"], str)
            or not cell["shape_key"]
            or not isinstance(cell["dictionary_entry_id"], str)
            or not cell["dictionary_entry_id"]
        ):
            raise ValueError("cell identity or recognizer agreement invalid")
        cell_ids.add(cell["cell_id"])
        if not _sorted_unique_strings(cell["primitive_ids"], nonempty=True):
            raise ValueError("cell primitive ids invalid")
        overlap = cell_primitives.intersection(cell["primitive_ids"])
        if overlap:
            raise ValueError("primitive is claimed by multiple cells")
        cell_primitives.update(cell["primitive_ids"])
        _validate_bbox(cell["bbox"], page_width=page_width, page_height=page_height)
        cell_bboxes.append(
            (
                float(cell["bbox"]["x"]),
                float(cell["bbox"]["y"]),
                float(cell["bbox"]["x"]) + float(cell["bbox"]["w"]),
                float(cell["bbox"]["y"]) + float(cell["bbox"]["h"]),
            )
        )
        if (
            not isinstance(cell["s001"], list)
            or not cell["s001"]
            or not all(_finite_number(item) for item in cell["s001"])
            or type(cell["free_endpoint_count"]) is not int
            or cell["free_endpoint_count"] < 0
            or type(cell["width_disambiguated"]) is not bool
            or (cell["width_disambiguated"] and not isinstance(cell["width_evidence"], Mapping))
            or (not cell["width_disambiguated"] and cell["width_evidence"] is not None)
        ):
            raise ValueError("cell glyph evidence invalid")
        if cell["width_disambiguated"]:
            width_evidence = cell["width_evidence"]
            candidates = width_evidence.get("candidate_characters")
            if (
                not _finite_number(width_evidence.get("measured_width"))
                or float(width_evidence["measured_width"])
                != float(cell["s001"][0])
                or not _sorted_unique_strings(candidates, nonempty=True)
                or width_evidence.get("selected_character") != cell["key_character"]
                or cell["key_character"] not in candidates
            ):
                raise ValueError("cell width evidence invalid")
        characters.append(cell["char"])
    if cell_primitives != primitive_ids:
        raise ValueError("phrase primitive ownership is not the union of cells")
    if phrase["text"] != "".join(characters) or not phrase["text"]:
        raise ValueError("phrase text does not match cells")
    union = bbox_xywh(bbox_union(cell_bboxes))
    if phrase["bbox"] != union:
        raise ValueError("phrase bbox is not the union of cells")
    expected_quad = _oriented_quad(
        walk["cells"],
        float(phrase["axis_angle_deg"]),
    )
    if phrase["oriented_quad"] != expected_quad:
        raise ValueError("phrase oriented quad is inconsistent with cells and axis")
    terminations = walk["terminations"]
    if not isinstance(terminations, Mapping) or set(terminations) != {"negative", "positive"}:
        raise ValueError("walk terminations invalid")
    for direction in ("negative", "positive"):
        _validate_termination(
            terminations[direction],
            direction=direction,
            claimed_primitive_ids=primitive_ids,
            page_width=page_width,
            page_height=page_height,
        )
    provenance_payload = {
        "anchor_ids": phrase["anchor_ids"],
        "source_candidate_ids": phrase["source_candidate_ids"],
        "source_anchor_evidence": phrase["source_anchor_evidence"],
        "source_region_ids": phrase["source_region_ids"],
    }
    if phrase["provenance_digest"] != stable_digest(provenance_payload):
        raise ValueError("phrase provenance digest invalid")
    physical_payload = {
        "page_index": page_index,
        "gdt_ownership_frame_ids": phrase["gdt_ownership_frame_ids"],
        "axis_angle_deg": round(float(phrase["axis_angle_deg"]), 6),
        "axis_directionality": phrase["axis_directionality"],
        "bbox": phrase["bbox"],
        "primitive_run_ids": phrase["primitive_run_ids"],
        "negative_stop_digest": terminations["negative"]["semantic_digest"],
        "positive_stop_digest": terminations["positive"]["semantic_digest"],
    }
    if phrase["physical_core_digest"] != stable_digest(physical_payload):
        raise ValueError("phrase physical digest invalid")
    overlap = phrase["gdt_overlap"]
    overlap_fields = {
        "frame_ids",
        "inside_frame_ids",
        "boundary_frame_ids",
        "crosses_frame_boundary",
        "ordinary_eligible",
        "consumer_allowed",
    }
    if not isinstance(overlap, Mapping) or set(overlap) != overlap_fields:
        raise ValueError("GD&T overlap fields invalid")
    for key in ("frame_ids", "inside_frame_ids", "boundary_frame_ids"):
        if not _sorted_unique_strings(overlap[key]):
            raise ValueError("GD&T frame ids invalid")
    if (
        type(overlap["crosses_frame_boundary"]) is not bool
        or type(overlap["ordinary_eligible"]) is not bool
        or overlap["boundary_frame_ids"]
        or overlap["crosses_frame_boundary"] is not False
    ):
        raise ValueError("GD&T boundary evidence invalid")
    ownership = phrase["gdt_ownership_frame_ids"]
    if (
        overlap["frame_ids"] != ownership
        or overlap["inside_frame_ids"] != ownership
        or overlap["ordinary_eligible"] != (not ownership)
    ):
        raise ValueError("GD&T ownership evidence inconsistent")
    return primitive_ids


def validate_directional_walk_dump_v1(value: Any) -> None:
    """Fail closed unless ``value`` satisfies the corrected Issue 00 contract."""

    required_root_fields = {
        "schema_version",
        "trace_id",
        "page_index",
        "status",
        "error",
        "page_context_manifest",
        "phrases",
        "stats",
        "consumer_allowed",
    }
    if (
        not isinstance(value, Mapping)
        or not required_root_fields.issubset(value)
        or set(value).difference(required_root_fields) not in (set(), {"diagnostics"})
    ):
        raise ValueError("directional-walk root fields invalid")
    _validate_no_legacy_fields(value)
    if (
        value["schema_version"] != "r92_directional_walk_dump_v1"
        or not isinstance(value["trace_id"], str)
        or not value["trace_id"]
        or type(value["page_index"]) is not int
        or value["page_index"] < 0
        or value["status"] not in {"ok", "fail_closed"}
        or not isinstance(value["phrases"], list)
        or not isinstance(value["stats"], Mapping)
        or not isinstance(value.get("diagnostics", {}), Mapping)
    ):
        raise ValueError("directional-walk root identity invalid")
    if value["status"] == "ok" and value["error"] is not None:
        raise ValueError("successful directional-walk dump cannot carry an error")
    if value["status"] == "fail_closed":
        if (
            not isinstance(value["error"], Mapping)
            or not value["error"]
            or value["phrases"]
        ):
            raise ValueError("fail-closed dump cannot partially publish phrases")
    manifest = value["page_context_manifest"]
    manifest_fields = {
        "schema_version",
        "pdf_stem",
        "page_num",
        "page_width",
        "page_height",
        "primitive_count",
        "primitive_sha256",
        "consumer_allowed",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != manifest_fields
        or manifest["schema_version"] != "vector_page_context_manifest_v1"
        or not isinstance(manifest["pdf_stem"], str)
        or manifest["page_num"] != value["page_index"] + 1
        or not _finite_number(manifest["page_width"])
        or not _finite_number(manifest["page_height"])
        or float(manifest["page_width"]) <= 0.0
        or float(manifest["page_height"]) <= 0.0
        or type(manifest["primitive_count"]) is not int
        or manifest["primitive_count"] < 0
        or not _valid_sha256(manifest["primitive_sha256"])
    ):
        raise ValueError("page context manifest invalid")
    phrase_ids: list[str] = []
    all_claimed: set[str] = set()
    for phrase in value["phrases"]:
        claimed = _validate_phrase(
            phrase,
            trace_id=value["trace_id"],
            page_index=value["page_index"],
            page_width=float(manifest["page_width"]),
            page_height=float(manifest["page_height"]),
        )
        if all_claimed.intersection(claimed):
            raise ValueError("primitive is claimed by multiple phrases")
        all_claimed.update(claimed)
        phrase_ids.append(phrase["phrase_id"])
    if phrase_ids != sorted(set(phrase_ids)):
        raise ValueError("phrases are not uniquely sorted by phrase_id")
    stats = value["stats"]
    required_stats = {
        "phrase_count",
        "cell_count",
        "matched_cell_count",
        "termination_count",
        "consumer_allowed_count",
    }
    if not required_stats.issubset(stats):
        raise ValueError("directional-walk stats fields missing")
    cell_count = sum(len(phrase["walk"]["cells"]) for phrase in value["phrases"])
    if (
        stats["phrase_count"] != len(value["phrases"])
        or stats["cell_count"] != cell_count
        or stats["matched_cell_count"] != cell_count
        or stats["termination_count"] != 2 * len(value["phrases"])
        or stats["consumer_allowed_count"] != 0
    ):
        raise ValueError("directional-walk stats do not conserve output")


__all__ = [
    "build_dump_v1",
    "build_phrase_v1",
    "make_termination",
    "normalize_axis_angle",
    "stable_digest",
    "validate_directional_walk_dump_v1",
]
