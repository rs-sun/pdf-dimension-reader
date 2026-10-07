"""Pure, offline R92 directional walking from trusted vector anchors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
import math
from typing import Any

from dimension_syntax import classify_dimension_syntax

from .claim_arbitration import arbitrate_true_subset_claims_v1
from .glyph_key import (
    ALLOWED_OPS,
    bbox_union,
    bbox_xywh,
    canonical_json,
    glyph_evidence,
    group_primitives,
    lookup_character,
    project_point,
    unproject_point,
)
from .phrase_output import (
    build_dump_v1,
    build_phrase_v1,
    make_termination,
    normalize_axis_angle,
    stable_digest,
    validate_directional_walk_dump_v1,
)
from .resolution import resolve_draw_order_direction_v1
from .step_table import lookup_step_candidates


ANCHOR_ENVELOPE_RATIO = 12.0
LANDING_ENVELOPE_RATIO = 6.0
ANCHOR_CHARACTERS = {
    "decimal": ".",
    "degree": "°",
    "diameter": "Ø",
    "Ra·Rmax": "R",
}
UNDIRECTED_AXIS_SOURCES = frozenset({"r69_adapter_axis", "r69_producer_axis"})
_CLAIM_QUALITY_SYNTAX_RANK = {
    "ok": 0,
    "incomplete_separator": 1,
    "unread_glyph_present": 2,
    "invalid_dimension_syntax": 3,
}


_LandingQueryAudit = tuple[Callable[[dict[str, Any]], None], bool]
_LANDING_QUERY_AUDIT: ContextVar[_LandingQueryAudit | None] = ContextVar(
    "directional_walk_landing_query_audit",
    default=None,
)


def _claim_quality_rank_v1(claim: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank one already-eligible claim without granting admission authority."""

    text = claim.get("text")
    bbox = claim.get("bbox")
    phrase_id = claim.get("phrase_id")
    if (
        not isinstance(text, str)
        or not text
        or not isinstance(bbox, Mapping)
        or not isinstance(phrase_id, str)
        or not phrase_id
    ):
        raise ValueError("claim quality rank evidence incomplete")
    width = bbox.get("w")
    height = bbox.get("h")
    if (
        type(width) not in (int, float)
        or type(height) not in (int, float)
        or not math.isfinite(float(width))
        or not math.isfinite(float(height))
        or float(width) < 0.0
        or float(height) < 0.0
    ):
        raise ValueError("claim quality rank bbox invalid")
    syntax = classify_dimension_syntax(text)
    return (
        _CLAIM_QUALITY_SYNTAX_RANK[syntax],
        -len(text),
        float(width) * float(height),
        phrase_id,
    )


def _greedy_claim_survivors_v1(
    indices: Sequence[int],
    primitive_sets: Sequence[frozenset[str]],
) -> tuple[int, ...]:
    claimed: set[str] = set()
    delivered: list[int] = []
    for index in indices:
        primitive_ids = primitive_sets[index]
        if claimed.intersection(primitive_ids):
            continue
        delivered.append(index)
        claimed.update(primitive_ids)
    return tuple(delivered)


def _changed_survivors_have_direct_conflict_matching_v1(
    losses: Sequence[int],
    gains: Sequence[int],
    primitive_sets: Sequence[frozenset[str]],
) -> bool:
    """Require every representative swap to close to one direct conflict."""

    if len(losses) != len(gains):
        return False
    matched_loss_by_gain: dict[int, int] = {}

    def augment(loss: int, visited_gains: set[int]) -> bool:
        for gain in gains:
            if (
                gain in visited_gains
                or not primitive_sets[loss].intersection(primitive_sets[gain])
            ):
                continue
            visited_gains.add(gain)
            prior_loss = matched_loss_by_gain.get(gain)
            if prior_loss is None or augment(prior_loss, visited_gains):
                matched_loss_by_gain[gain] = loss
                return True
        return False

    return all(augment(loss, set()) for loss in losses)


def _quality_ranked_claim_indices_v1(
    claims: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Reorder only cardinality-preserving, one-to-one overlap components.

    The exact primitive overlap graph is established independently of syntax.
    If the quality order would change a component's survivor count, or if its
    changed survivors cannot be paired one-to-one by direct primitive overlap,
    that component retains its existing first-come order.
    """

    primitive_sets: list[frozenset[str]] = []
    input_orders: list[int] = []
    for claim in claims:
        raw_ids = claim.get("primitive_run_ids")
        input_order = claim.get("input_order")
        if (
            not isinstance(raw_ids, (list, tuple))
            or type(input_order) is not int
            or input_order < 0
            or any(not isinstance(value, str) or not value for value in raw_ids)
        ):
            raise ValueError("claim quality order evidence incomplete")
        primitive_ids = frozenset(raw_ids)
        if not primitive_ids or len(primitive_ids) != len(raw_ids):
            raise ValueError("claim quality order primitive ids invalid")
        primitive_sets.append(primitive_ids)
        input_orders.append(input_order)
    if len(set(input_orders)) != len(input_orders):
        raise ValueError("claim quality order input orders must be unique")

    parents = list(range(len(claims)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(claims)):
        for right in range(left + 1, len(claims)):
            if primitive_sets[left].intersection(primitive_sets[right]):
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(claims)):
        components.setdefault(find(index), []).append(index)

    ordered: list[int] = []
    for component in sorted(
        components.values(),
        key=lambda values: min(input_orders[index] for index in values),
    ):
        baseline = sorted(component, key=lambda index: input_orders[index])
        ranked = sorted(component, key=lambda index: _claim_quality_rank_v1(claims[index]))
        baseline_survivors = _greedy_claim_survivors_v1(
            baseline,
            primitive_sets,
        )
        ranked_survivors = _greedy_claim_survivors_v1(ranked, primitive_sets)
        losses = [
            index for index in baseline_survivors
            if index not in ranked_survivors
        ]
        gains = [
            index for index in ranked_survivors
            if index not in baseline_survivors
        ]
        safe = (
            len(baseline_survivors) == len(ranked_survivors)
            and _changed_survivors_have_direct_conflict_matching_v1(
                losses,
                gains,
                primitive_sets,
            )
        )
        ordered.extend(ranked if safe else baseline)
    return tuple(ordered)


@contextmanager
def capture_landing_query_audit_v1(
    observer: Callable[[dict[str, Any]], None],
    *,
    exhaustive: bool = False,
) -> Iterator[None]:
    """Observe indexed landing locals; optionally compare every one to a full scan."""

    if not callable(observer):
        raise ValueError("landing query audit observer must be callable")
    token = _LANDING_QUERY_AUDIT.set((observer, bool(exhaustive)))
    try:
        yield
    finally:
        _LANDING_QUERY_AUDIT.reset(token)


def _primitive_row(page_context: Any, primitive: Any) -> dict[str, Any]:
    return {
        "primitive_id": str(page_context.item_id(primitive)),
        "op": str(primitive.op),
        "points": tuple(primitive.points),
        "bbox": tuple(primitive.bbox),
    }


class _PrimitiveRows(tuple):
    """Tuple-compatible rows retaining the page context's spatial query."""

    def __new__(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        page_context: Any,
    ) -> "_PrimitiveRows":
        instance = super().__new__(cls, rows)
        instance._page_context = page_context
        instance._projected_bboxes = {}
        instance._bbox_index_safe = (
            callable(getattr(page_context, "query_bbox", None))
            and all(
                _primitive_bbox_contains_points(row)
                for row in instance
            )
        )
        return instance

    def __copy__(self) -> "_PrimitiveRows":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> tuple[dict[str, Any], ...]:
        result = tuple(deepcopy(row, memo) for row in self)
        memo[id(self)] = result
        return result

    def __reduce__(self) -> tuple[Any, tuple[tuple[Any, ...]]]:
        # A detached serialized value remains the same ordinary tuple of rows;
        # the live page query is deliberately process-local.
        return tuple, (tuple(self),)

    def query_bbox_rows(
        self,
        bbox: Sequence[float],
    ) -> tuple[dict[str, Any], ...]:
        if not self._bbox_index_safe:
            return tuple(self)
        return tuple(
            _primitive_row(self._page_context, primitive)
            for primitive in self._page_context.query_bbox(tuple(bbox))
        )

    def projected_bbox(
        self,
        primitive: Mapping[str, Any],
        *,
        axis_angle_deg: float,
    ) -> tuple[float, float, float, float]:
        axis_key = float(axis_angle_deg).hex()
        by_primitive = self._projected_bboxes.setdefault(axis_key, {})
        primitive_id = str(primitive["primitive_id"])
        result = by_primitive.get(primitive_id)
        if result is None:
            result = _projected_primitive_bbox(
                primitive,
                axis_angle_deg=axis_angle_deg,
            )
            by_primitive[primitive_id] = result
        return result


def _primitive_bbox_contains_points(primitive: Mapping[str, Any]) -> bool:
    try:
        x0, y0, x1, y1 = (float(value) for value in primitive["bbox"])
        points = tuple(primitive["points"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(points) and all(
        x0 <= float(point[0]) <= x1
        and y0 <= float(point[1]) <= y1
        for point in points
    )


def _manifest(page_context: Any) -> dict[str, Any]:
    assert_integrity = getattr(page_context, "assert_integrity", None)
    if not callable(assert_integrity):
        raise ValueError("vector page context integrity invalid")
    assert_integrity()
    value = page_context.manifest_v1()
    if not isinstance(value, Mapping):
        raise ValueError("page context manifest missing")
    return dict(value)


def _primitive_rows(page_context: Any) -> tuple[dict[str, Any], ...]:
    return _PrimitiveRows(
        tuple(
            _primitive_row(page_context, primitive)
            for primitive in page_context.primitives
        ),
        page_context=page_context,
    )


def _bbox_xyxy(value: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x, y = float(value["x"]), float(value["y"])
    width, height = float(value["w"]), float(value["h"])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("anchor bbox must be positive")
    return x, y, x + width, y + height


def _bbox_intersects(left: Sequence[float], right: Sequence[float]) -> bool:
    return not (
        float(left[2]) < float(right[0])
        or float(right[2]) < float(left[0])
        or float(left[3]) < float(right[1])
        or float(right[3]) < float(left[1])
    )


def _query_primitive_rows(
    primitives: Sequence[Mapping[str, Any]],
    bbox: Sequence[float],
) -> tuple[Mapping[str, Any], ...]:
    query_bbox_rows = getattr(primitives, "query_bbox_rows", None)
    if callable(query_bbox_rows):
        return tuple(query_bbox_rows(bbox))
    return tuple(primitives)


def _landing_window_page_bbox(
    *,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    axis_angle_deg: float,
) -> tuple[float, float, float, float]:
    """Conservative page AABB of all four corners of a rotated UV window."""

    corners = tuple(
        unproject_point(point, axis_angle_deg)
        for point in (
            (float(u0), float(v0)),
            (float(u0), float(v1)),
            (float(u1), float(v0)),
            (float(u1), float(v1)),
        )
    )
    x0 = min(point[0] for point in corners)
    y0 = min(point[1] for point in corners)
    x1 = max(point[0] for point in corners)
    y1 = max(point[1] for point in corners)
    # The inverse rotation repeats rounded trig products used by projection.
    # Padding only this coarse page query makes it conservative at boundaries;
    # the unchanged UV comparisons below remain the recognition authority.
    magnitude = max(
        1.0,
        *(abs(value) for point in corners for value in point),
        abs(float(u0)),
        abs(float(v0)),
        abs(float(u1)),
        abs(float(v1)),
    )
    padding = 32.0 * math.ulp(magnitude)
    return x0 - padding, y0 - padding, x1 + padding, y1 + padding


def _projected_primitive_bbox(
    primitive: Mapping[str, Any],
    *,
    axis_angle_deg: float,
) -> tuple[float, float, float, float]:
    projected = [
        project_point(point, axis_angle_deg)
        for point in primitive["points"]
    ]
    return (
        min(point[0] for point in projected),
        min(point[1] for point in projected),
        max(point[0] for point in projected),
        max(point[1] for point in projected),
    )


def _landing_local_rows(
    primitives: Sequence[Mapping[str, Any]],
    *,
    claimed: set[str],
    projected_bboxes: Mapping[str, Sequence[float]],
    predicted_u: float,
    maximum_glyph_width: float,
    current_v: float,
    cross_axis_radius: float,
) -> tuple[Mapping[str, Any], ...]:
    local: list[Mapping[str, Any]] = []
    for primitive in primitives:
        primitive_id = str(primitive["primitive_id"])
        if primitive_id in claimed:
            continue
        u0, v0, u1, v1 = projected_bboxes[primitive_id]
        if (
            u1 >= predicted_u - maximum_glyph_width
            and u0 <= predicted_u + maximum_glyph_width
            and v0 >= current_v - cross_axis_radius
            and v1 <= current_v + cross_axis_radius
        ):
            local.append(primitive)
    return tuple(local)


def _full_scan_landing_local(
    primitives: Sequence[Mapping[str, Any]],
    *,
    claimed: set[str],
    axis_angle_deg: float,
    predicted_u: float,
    maximum_glyph_width: float,
    current_v: float,
    cross_axis_radius: float,
) -> tuple[Mapping[str, Any], ...]:
    projected_bboxes = {
        str(primitive["primitive_id"]): _projected_primitive_bbox(
            primitive,
            axis_angle_deg=axis_angle_deg,
        )
        for primitive in primitives
        if str(primitive["primitive_id"]) not in claimed
    }
    return _landing_local_rows(
        primitives,
        claimed=claimed,
        projected_bboxes=projected_bboxes,
        predicted_u=predicted_u,
        maximum_glyph_width=maximum_glyph_width,
        current_v=current_v,
        cross_axis_radius=cross_axis_radius,
    )


def _group_primitive_count(
    primitives: Sequence[Mapping[str, Any]],
) -> int:
    seen: set[str] = set()
    count = 0
    for primitive in primitives:
        primitive_id = str(primitive.get("primitive_id") or "")
        if (
            not primitive_id
            or primitive_id in seen
            or str(primitive.get("op") or "") not in ALLOWED_OPS
            or len(tuple(primitive.get("points", ()))) < 2
        ):
            continue
        seen.add(primitive_id)
        count += 1
    return count


def _normalized_anchor(anchor: Mapping[str, Any], *, page_index: int) -> dict[str, Any]:
    anchor_id = str(anchor.get("anchor_id") or "")
    source_id = str(anchor.get("source_id") or "")
    anchor_type = str(anchor.get("anchor_type") or "")
    source_schema_version = str(anchor.get("source_schema_version") or "")
    axis_source = str(anchor.get("axis_angle_source") or "")
    if (
        not anchor_id
        or not source_id
        or anchor_type not in ANCHOR_CHARACTERS
        or not source_schema_version
        or not axis_source
        or anchor.get("axis_trusted") is not True
    ):
        raise ValueError("anchor evidence incomplete")
    raw_angle = float(anchor.get("axis_angle_deg"))
    if not math.isfinite(raw_angle):
        raise ValueError("anchor axis must be finite")
    raw_directionality = anchor.get("axis_directionality")
    if axis_source == "r84_alignment":
        if raw_directionality not in (None, "directed"):
            raise ValueError("r84 axis requires directed directionality")
        directionality = "directed"
    elif axis_source in UNDIRECTED_AXIS_SOURCES:
        if raw_directionality not in (None, "undirected_fallback"):
            raise ValueError("r69 axis requires undirected fallback directionality")
        directionality = "undirected_fallback"
    elif raw_directionality is None:
        directionality = "undirected_fallback"
    else:
        directionality = str(raw_directionality)
    angle = normalize_axis_angle(raw_angle, directionality=directionality)
    source_ids = sorted({str(value) for value in anchor.get("source_primitive_ids", ())})
    if not source_ids:
        raise ValueError("anchor source primitive ids missing")
    bbox = bbox_xywh(_bbox_xyxy(anchor["anchor_bbox"]))
    evidence = {
        "schema_version": "r92_source_anchor_evidence_v1",
        "source_schema_version": source_schema_version,
        "source_id": source_id,
        "anchor_id": anchor_id,
        "page_index": int(page_index),
        "anchor_type": anchor_type,
        "anchor_bbox": bbox,
        "axis_angle_deg": angle,
        "axis_angle_source": axis_source,
        "axis_directionality": directionality,
        "axis_trusted": True,
        "source_primitive_ids": source_ids,
        "source_region_ids": sorted(
            {str(value) for value in anchor.get("source_region_ids", ())}
        ),
        "consumer_allowed": False,
    }
    return {
        **dict(anchor),
        "anchor_id": anchor_id,
        "anchor_type": anchor_type,
        "anchor_bbox": bbox,
        "axis_angle_deg": angle,
        "axis_angle_source": axis_source,
        "axis_directionality": directionality,
        "axis_trusted": True,
        "source_primitive_ids": source_ids,
        "source_anchor_evidence": evidence,
    }


def _anchor_group(
    *,
    primitives: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    anchor_bbox = _bbox_xyxy(anchor["anchor_bbox"])
    source_ids = set(anchor["source_primitive_ids"])
    source_primitives = [row for row in primitives if row["primitive_id"] in source_ids]
    if len(source_primitives) != len(source_ids):
        return None, "anchor_group_unresolved"
    source_scale = max(
        anchor_bbox[2] - anchor_bbox[0],
        anchor_bbox[3] - anchor_bbox[1],
    )
    center_x = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
    center_y = (anchor_bbox[1] + anchor_bbox[3]) / 2.0
    radius = source_scale * ANCHOR_ENVELOPE_RATIO
    envelope = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    local = tuple(
        row
        for row in _query_primitive_rows(primitives, envelope)
        if _bbox_intersects(row["bbox"], envelope)
    )
    groups = group_primitives(
        local,
        axis_angle_deg=float(anchor["axis_angle_deg"]),
        reference_scale=source_scale,
    )
    matches = [group for group in groups if source_ids.issubset(group["primitive_ids"])]
    if len(matches) != 1:
        return None, "anchor_group_unresolved"
    group = matches[0]
    if any(
        row["primitive_id"] in group["primitive_ids"]
        and not (
            envelope[0] <= row["bbox"][0]
            and envelope[1] <= row["bbox"][1]
            and row["bbox"][2] <= envelope[2]
            and row["bbox"][3] <= envelope[3]
        )
        for row in local
    ):
        return None, "anchor_group_unbounded"
    u0, v0, u1, v1 = group["uv_bbox"]
    if max(u1 - u0, v1 - v0) > source_scale * ANCHOR_ENVELOPE_RATIO:
        return None, "anchor_group_unbounded"
    return group, None


def _template_character(
    template_recognizer: Callable[[dict[str, Any]], Any],
    evidence: Mapping[str, Any],
) -> tuple[str | None, Any]:
    result = template_recognizer(dict(evidence))
    if isinstance(result, str):
        return result or None, result
    if not isinstance(result, Mapping):
        return None, result
    decision = str(result.get("decision") or "")
    if decision not in {"accepted_template", "unique", "accepted"}:
        return None, dict(result)
    value = result.get("template_character")
    if value is None:
        value = result.get("predicted_text")
    if value is None:
        value = result.get("character")
    character = str(value) if value is not None else ""
    return character or None, dict(result)


def _accepted_cell(
    *,
    group: Mapping[str, Any],
    evidence: Mapping[str, Any],
    dictionary_entry: Mapping[str, Any],
    key_character: str,
    template_recognizer: Callable[[dict[str, Any]], Any],
    walk_direction: str,
    step_index: int,
    step_evidence: Mapping[str, Any] | None,
    phrase_seed: str,
) -> dict[str, Any]:
    template_character, _template_match = _template_character(
        template_recognizer,
        {
            **dict(evidence),
            "key_character": key_character,
        },
    )
    agreement = (
        None
        if template_character is None
        else template_character == key_character
    )
    bbox = bbox_xywh(evidence["bbox"])
    width_disambiguated = bool(dictionary_entry["width_disambiguated"])
    width_evidence = dictionary_entry["width_evidence"]
    if width_disambiguated:
        if not isinstance(width_evidence, Mapping):
            raise ValueError("width-disambiguated dictionary entry lacks evidence")
        width_evidence = {
            **dict(width_evidence),
            "measured_width": float(evidence["s001"][0]),
        }
    return {
        "cell_id": "r92_cell_" + stable_digest(
            {
                "phrase_seed": phrase_seed,
                "walk_direction": walk_direction,
                "step_index": int(step_index),
                "primitive_ids": list(evidence["primitive_ids"]),
            }
        )[:24],
        "reading_index": -1,
        "walk_direction": walk_direction,
        "step_index": int(step_index),
        "char": key_character,
        "key_character": key_character,
        "template_character": template_character,
        "characters_agree": agreement,
        "primitive_ids": list(evidence["primitive_ids"]),
        "bbox": bbox,
        "shape_key": str(evidence["shape_key"]),
        "s001": list(evidence["s001"]),
        "free_endpoint_count": int(evidence["free_endpoint_count"]),
        "dictionary_entry_id": str(dictionary_entry["entry_id"]),
        "width_disambiguated": width_disambiguated,
        "width_evidence": width_evidence,
        "step_evidence": (
            None
            if step_evidence is None
            else {**dict(step_evidence), "consumer_allowed": False}
        ),
        "consumer_allowed": False,
        "points": list(evidence["points"]),
    }


def _character_table_landing_geometry(
    character_table: Mapping[str, Any],
) -> tuple[float, tuple[tuple[str, ...], ...]]:
    entries = character_table.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("character table entries missing")
    widths: list[float] = []
    compound_specifications: set[tuple[str, ...]] = set()
    for raw_shape_key in entries:
        try:
            value = json.loads(str(raw_shape_key))
        except (TypeError, ValueError) as exc:
            raise ValueError("character table shape key invalid") from exc
        members = value.get("compound") if isinstance(value, Mapping) else None
        if members is None:
            members = [value]
        if not isinstance(members, list) or not members:
            raise ValueError("character table shape key members invalid")
        specification: list[str] = []
        for member in members:
            if not isinstance(member, list) or len(member) < 2:
                raise ValueError("character table shape key component invalid")
            width = float(member[0])
            if not math.isfinite(width) or width < 0.0:
                raise ValueError("character table glyph width invalid")
            widths.append(width)
            specification.append(canonical_json(member))
        if isinstance(value, Mapping):
            compound_specifications.add(tuple(sorted(specification)))
    maximum_width = max(widths, default=0.0)
    if maximum_width <= 0.0:
        raise ValueError("character table has no positive glyph width")
    return maximum_width, tuple(sorted(compound_specifications))


def _group_u_center(group: Mapping[str, Any]) -> float:
    if "u_center" in group:
        return float(group["u_center"])
    u0, _, u1, _ = group["uv_bbox"]
    return (float(u0) + float(u1)) / 2.0


def _cross_residual_group_score(hit: Mapping[str, Any]) -> float | None:
    source_pairs = hit.get("source_candidate_pairs")
    if not isinstance(source_pairs, (list, tuple)) or not source_pairs:
        return None

    residuals: list[float] = []
    for pair in source_pairs:
        if not isinstance(pair, Mapping):
            return None
        residual = pair.get("absolute_cross_center_residual_pt")
        if type(residual) not in (int, float):
            return None
        numeric_residual = float(residual)
        if not math.isfinite(numeric_residual) or numeric_residual < 0.0:
            return None
        residuals.append(numeric_residual)
    return min(residuals)


def _arbitrate_cross_residual_owner(
    recognized: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    ],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None:
    scored: list[
        tuple[
            float,
            tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
        ]
    ] = []
    physical_groups: set[tuple[str, ...]] = set()
    for row in recognized:
        hit, evidence, _entry = row
        primitive_ids = tuple(sorted(str(value) for value in evidence["primitive_ids"]))
        if not primitive_ids or primitive_ids in physical_groups:
            return None
        physical_groups.add(primitive_ids)
        score = _cross_residual_group_score(hit)
        if score is None:
            return None
        scored.append((score, row))
    if len(scored) < 2:
        return None
    best_score = min(score for score, _row in scored)
    winners = [row for score, row in scored if score == best_score]
    return winners[0] if len(winners) == 1 else None


def _combined_component_group(
    groups: Sequence[Mapping[str, Any]],
    *,
    core_group: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = tuple(sorted(groups, key=lambda row: tuple(row["primitive_ids"])))
    primitives_by_id = {
        str(primitive["primitive_id"]): primitive
        for group in ordered
        for primitive in group["primitives"]
    }
    primitives = tuple(primitives_by_id[key] for key in sorted(primitives_by_id))
    return {
        "primitive_ids": tuple(sorted(primitives_by_id)),
        "primitives": primitives,
        "components": ordered,
        "bbox": bbox_union(row["bbox"] for row in ordered),
        "uv_bbox": bbox_union(row["uv_bbox"] for row in ordered),
        "u_center": _group_u_center(core_group),
    }


def _compound_components_form_local_station(
    groups: Sequence[Mapping[str, Any]],
    *,
    core_group: Mapping[str, Any],
    predicted_u: float,
) -> bool:
    """Require real, scale-normalized adjacency before composing a glyph."""

    core_u0, core_v0, core_u1, core_v1 = (
        float(value) for value in core_group["uv_bbox"]
    )
    core_scale = max(core_u1 - core_u0, core_v1 - core_v0)
    if core_scale <= 0.0 or not core_u0 <= predicted_u <= core_u1:
        return False
    for group in groups:
        if group is core_group:
            continue
        u0, v0, u1, v1 = (float(value) for value in group["uv_bbox"])
        if not u0 <= predicted_u <= u1:
            return False
        if min(core_u1, u1) - max(core_u0, u0) <= 0.0:
            return False
        cross_gap = max(core_v0 - v1, v0 - core_v1, 0.0)
        if cross_gap <= 0.0 or cross_gap > core_scale:
            return False
    return True


def _compound_landing_groups(
    groups: Sequence[Mapping[str, Any]],
    *,
    specifications: Sequence[tuple[str, ...]],
    predicted_u: float,
) -> tuple[dict[str, Any], ...]:
    if not specifications:
        return ()

    evidence = tuple(glyph_evidence(group) for group in groups)
    combinations: dict[tuple[str, ...], dict[str, Any]] = {}
    for specification in specifications:
        options = [
            tuple(
                index
                for index, row in enumerate(evidence)
                if row["shape_key"] == component_key
            )
            for component_key in specification
        ]
        if any(not rows for rows in options):
            continue

        def visit(position: int, selected: tuple[int, ...]) -> None:
            if position == len(options):
                selected_groups = tuple(groups[index] for index in selected)
                core = max(
                    selected_groups,
                    key=lambda row: (
                        float(row["uv_bbox"][3]) - float(row["uv_bbox"][1]),
                        len(row["primitive_ids"]),
                        tuple(row["primitive_ids"]),
                    ),
                )
                if not _compound_components_form_local_station(
                    selected_groups,
                    core_group=core,
                    predicted_u=predicted_u,
                ):
                    return
                combined = _combined_component_group(
                    selected_groups,
                    core_group=core,
                )
                combinations[tuple(combined["primitive_ids"])] = combined
                return
            for index in options[position]:
                if index not in selected:
                    visit(position + 1, (*selected, index))

        visit(0, ())
    return tuple(combinations[key] for key in sorted(combinations))


def _landing_groups(
    *,
    primitives: Sequence[Mapping[str, Any]],
    claimed: set[str],
    current_evidence: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    character_table: Mapping[str, Any],
    axis_angle_deg: float,
    sign: int,
    audit_context: Mapping[str, Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    current_uv = current_evidence["uv_bbox"]
    current_u = float(
        current_evidence.get("u_center", (current_uv[0] + current_uv[2]) / 2.0)
    )
    current_v = (float(current_uv[1]) + float(current_uv[3])) / 2.0
    maximum_glyph_width, compound_specifications = _character_table_landing_geometry(
        character_table
    )
    reference_scale = max(
        float(current_uv[2]) - float(current_uv[0]),
        float(current_uv[3]) - float(current_uv[1]),
        1e-6,
    )
    cross_axis_radius = reference_scale * LANDING_ENVELOPE_RATIO
    landing_queries: list[
        tuple[Mapping[str, Any], float, tuple[Mapping[str, Any], ...]]
    ] = []
    unique_rough_rows: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        predicted_u = current_u + sign * float(candidate["step_pt"])
        page_bbox = _landing_window_page_bbox(
            u0=predicted_u - maximum_glyph_width,
            v0=current_v - cross_axis_radius,
            u1=predicted_u + maximum_glyph_width,
            v1=current_v + cross_axis_radius,
            axis_angle_deg=axis_angle_deg,
        )
        rough_rows = tuple(
            primitive
            for primitive in _query_primitive_rows(primitives, page_bbox)
            if str(primitive["primitive_id"]) not in claimed
        )
        landing_queries.append((candidate, predicted_u, rough_rows))
        for primitive in rough_rows:
            unique_rough_rows.setdefault(str(primitive["primitive_id"]), primitive)

    projected_bboxes: dict[str, tuple[float, float, float, float]] = {}
    projected_bbox_lookup = getattr(primitives, "projected_bbox", None)
    for primitive_id, primitive in unique_rough_rows.items():
        if callable(projected_bbox_lookup):
            projected_bboxes[primitive_id] = projected_bbox_lookup(
                primitive,
                axis_angle_deg=axis_angle_deg,
            )
        else:
            projected_bboxes[primitive_id] = _projected_primitive_bbox(
                primitive,
                axis_angle_deg=axis_angle_deg,
            )

    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    source_pairs: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    rejected: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    audit = _LANDING_QUERY_AUDIT.get()
    for candidate_index, (candidate, predicted_u, rough_rows) in enumerate(
        landing_queries
    ):
        local = _landing_local_rows(
            rough_rows,
            claimed=claimed,
            projected_bboxes=projected_bboxes,
            predicted_u=predicted_u,
            maximum_glyph_width=maximum_glyph_width,
            current_v=current_v,
            cross_axis_radius=cross_axis_radius,
        )
        if audit is not None:
            observer, exhaustive = audit
            indexed_ids = tuple(str(row["primitive_id"]) for row in local)
            group_primitive_count = _group_primitive_count(local)
            exhaustive_ids: tuple[str, ...] | None = None
            exhaustive_equal: bool | None = None
            if exhaustive:
                exhaustive_ids = tuple(
                    str(row["primitive_id"])
                    for row in _full_scan_landing_local(
                        primitives,
                        claimed=claimed,
                        axis_angle_deg=axis_angle_deg,
                        predicted_u=predicted_u,
                        maximum_glyph_width=maximum_glyph_width,
                        current_v=current_v,
                        cross_axis_radius=cross_axis_radius,
                    )
                )
                exhaustive_equal = indexed_ids == exhaustive_ids
            event = {
                "schema_version": "r92_landing_query_audit_v1",
                "candidate_index": candidate_index,
                "candidate_entry_id": str(candidate["entry_id"]),
                "walk_context": dict(audit_context or {}),
                "axis_angle_deg": float(axis_angle_deg),
                "predicted_u": predicted_u,
                "rough_primitive_count": len(rough_rows),
                "indexed_local_primitive_ids": indexed_ids,
                "indexed_local_size": len(indexed_ids),
                "group_primitive_count": group_primitive_count,
                "group_pair_evaluation_count": (
                    group_primitive_count * (group_primitive_count - 1) // 2
                ),
                "exhaustive_local_primitive_ids": exhaustive_ids,
                "exhaustive_equal": exhaustive_equal,
                "consumer_allowed": False,
            }
            observer(event)
            if exhaustive_equal is False:
                raise AssertionError(
                    "indexed landing local differs from exhaustive full scan: "
                    f"context={event['walk_context']!r} "
                    f"candidate={event['candidate_entry_id']!r} "
                    f"indexed={indexed_ids!r} exhaustive={exhaustive_ids!r}"
                )
        groups = group_primitives(
            local,
            axis_angle_deg=axis_angle_deg,
            reference_scale=reference_scale,
            projected_bboxes=projected_bboxes,
        )
        compound_groups = _compound_landing_groups(
            groups,
            specifications=compound_specifications,
            predicted_u=predicted_u,
        )
        compound_primitive_ids = {
            primitive_id
            for group in compound_groups
            for primitive_id in group["primitive_ids"]
        }
        landing_groups = [*compound_groups]
        landing_groups.extend(
            group
            for group in groups
            if not compound_primitive_ids.intersection(group["primitive_ids"])
        )
        for group in landing_groups:
            u0, v0, u1, v1 = (float(value) for value in group["uv_bbox"])
            if not (
                u0
                <= predicted_u
                <= u1
            ):
                continue
            measured_long_edge = max(u1 - u0, v1 - v0)
            envelope_limit = reference_scale * LANDING_ENVELOPE_RATIO
            if measured_long_edge > envelope_limit:
                rejection_key = (
                    tuple(group["primitive_ids"]),
                    str(candidate["entry_id"]),
                )
                rejected[rejection_key] = {
                    "group": group,
                    "candidate": dict(candidate),
                    "predicted_u": predicted_u,
                    "predicted_v": current_v,
                    "reference_scale": reference_scale,
                    "measured_long_edge": measured_long_edge,
                    "envelope_limit": envelope_limit,
                }
                continue
            axial_error = abs(_group_u_center(group) - predicted_u)
            key = tuple(group["primitive_ids"])
            source_pair = {
                "step_candidate": dict(candidate),
                "absolute_cross_center_residual_pt": abs(
                    (v0 + v1) / 2.0 - current_v
                ),
            }
            source_pairs.setdefault(key, {})[
                canonical_json(source_pair)
            ] = source_pair
            previous = rows.get(key)
            hit = {"group": group, "candidate": dict(candidate), "axial_error": axial_error}
            if previous is None or (axial_error, candidate["entry_id"]) < (
                previous["axial_error"],
                previous["candidate"]["entry_id"],
            ):
                rows[key] = hit
    for key, hit in rows.items():
        hit["source_candidate_pairs"] = tuple(
            source_pairs[key][pair_key]
            for pair_key in sorted(source_pairs[key])
        )
    return (
        tuple(rows[key] for key in sorted(rows)),
        tuple(rejected[key] for key in sorted(rejected)),
    )


def _landing_group_rejection_evidence(
    rejection: Mapping[str, Any],
    *,
    anchor_id: str,
    direction: str,
    attempted_step_index: int,
    axis_angle_deg: float,
) -> dict[str, Any]:
    group = rejection["group"]
    predicted_uv = (
        float(rejection["predicted_u"]),
        float(rejection["predicted_v"]),
    )
    payload = {
        "schema_version": "r92_landing_group_rejection_v1",
        "anchor_id": anchor_id,
        "direction": direction,
        "attempted_step_index": int(attempted_step_index),
        "reason": "landing_group_unbounded",
        "candidate_step_entry_id": str(rejection["candidate"]["entry_id"]),
        "predicted_point": list(unproject_point(predicted_uv, axis_angle_deg)),
        "predicted_uv": list(predicted_uv),
        "primitive_ids": list(group["primitive_ids"]),
        "bbox": bbox_xywh(group["bbox"]),
        "uv_bbox": [float(value) for value in group["uv_bbox"]],
        "reference_scale": float(rejection["reference_scale"]),
        "envelope_ratio": LANDING_ENVELOPE_RATIO,
        "measured_long_edge": float(rejection["measured_long_edge"]),
        "envelope_limit": float(rejection["envelope_limit"]),
        "consumer_allowed": False,
    }
    return {**payload, "semantic_digest": stable_digest(payload)}


def _termination_for_stop(
    *,
    direction: str,
    reason: str,
    attempted_step_index: int,
    evidence: Mapping[str, Any] | None = None,
    dictionary_entry_ids: Sequence[str] = (),
    previous_step_key: str | None = None,
) -> dict[str, Any]:
    return make_termination(
        direction=direction,
        reason=reason,
        attempted_step_index=attempted_step_index,
        shape_key=(
            previous_step_key
            if reason == "no_step_for_prev_key"
            else None if evidence is None else str(evidence["shape_key"])
        ),
        primitive_ids=() if evidence is None else evidence["primitive_ids"],
        bbox=None if evidence is None else bbox_xywh(evidence["bbox"]),
        s001=None if evidence is None else evidence["s001"],
        free_endpoint_count=(
            None if evidence is None else int(evidence["free_endpoint_count"])
        ),
        candidate_dictionary_entry_ids=dictionary_entry_ids,
    )


def _walk_direction(
    *,
    direction: str,
    sign: int,
    start_group: Mapping[str, Any],
    start_evidence: Mapping[str, Any],
    primitives: Sequence[Mapping[str, Any]],
    claimed: set[str],
    step_tables: Mapping[str, Any],
    character_table: Mapping[str, Any],
    template_recognizer: Callable[[dict[str, Any]], Any],
    axis_angle_deg: float,
    phrase_seed: str,
    anchor_id: str,
    content_order_step_binding_enabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    del start_group
    current = dict(start_evidence)
    cells: list[dict[str, Any]] = []
    landing_rejections: list[dict[str, Any]] = []
    locked_lookup_bucket: str | None = None
    step_index = 0
    while True:
        step_index += 1
        if (
            content_order_step_binding_enabled
            and locked_lookup_bucket is None
            and cells
        ):
            content_order = resolve_draw_order_direction_v1([
                {"primitive_ids": start_evidence["primitive_ids"]},
                *cells,
            ])
            if content_order == "forward":
                locked_lookup_bucket = "positive"
            elif content_order == "reverse":
                locked_lookup_bucket = "negative"
        candidates = lookup_step_candidates(
            step_tables,
            direction=locked_lookup_bucket or direction,
            previous_shape_key=str(current["step_key"]),
        )
        if not candidates:
            return (
                cells,
                _termination_for_stop(
                    direction=direction,
                    reason="no_step_for_prev_key",
                    attempted_step_index=step_index,
                    previous_step_key=str(current["step_key"]),
                ),
                landing_rejections,
            )
        hits, rejected = _landing_groups(
            primitives=primitives,
            claimed=claimed,
            current_evidence=current,
            candidates=candidates,
            character_table=character_table,
            axis_angle_deg=axis_angle_deg,
            sign=sign,
            audit_context={
                "anchor_id": anchor_id,
                "direction": direction,
                "attempted_step_index": step_index,
                "sign": sign,
            },
        )
        landing_rejections.extend(
            _landing_group_rejection_evidence(
                rejection,
                anchor_id=anchor_id,
                direction=direction,
                attempted_step_index=step_index,
                axis_angle_deg=axis_angle_deg,
            )
            for rejection in rejected
        )
        if not hits:
            return (
                cells,
                _termination_for_stop(
                    direction=direction,
                    reason="no_group_at_landing",
                    attempted_step_index=step_index,
                ),
                landing_rejections,
            )
        observed = tuple(
            (
                hit,
                glyph_evidence(hit["group"]),
            )
            for hit in hits
        )
        recognized = tuple(
            (hit, evidence, entry)
            for hit, evidence in observed
            for entry in (
                lookup_character(character_table, shape_key=evidence["shape_key"]),
            )
            if entry is not None
        )
        if not recognized:
            evidence = observed[0][1]
            return (
                cells,
                _termination_for_stop(
                    direction=direction,
                    reason="key_not_in_table",
                    attempted_step_index=step_index,
                    evidence=evidence,
                ),
                landing_rejections,
            )
        owner_arbitrated = False
        if len(recognized) != 1:
            owner = _arbitrate_cross_residual_owner(recognized)
            if owner is not None:
                hit, evidence, entry = owner
                owner_arbitrated = True
            else:
                evidence = recognized[0][1]
                candidate_entry_ids = sorted(
                    {
                        entry_id
                        for _, _, entry in recognized
                        for entry_id in entry["candidate_entry_ids"]
                    }
                )
                return (
                    cells,
                    _termination_for_stop(
                        direction=direction,
                        reason="key_ambiguous",
                        attempted_step_index=step_index,
                        evidence=evidence,
                        dictionary_entry_ids=candidate_entry_ids,
                    ),
                    landing_rejections,
                )
        else:
            hit, evidence, entry = recognized[0]
        characters = tuple(entry["characters"])
        if len(characters) != 1:
            return (
                cells,
                _termination_for_stop(
                    direction=direction,
                    reason=(
                        "post_arbitration_key_ambiguous"
                        if owner_arbitrated
                        else "key_ambiguous"
                    ),
                    attempted_step_index=step_index,
                    evidence=evidence,
                    dictionary_entry_ids=entry["candidate_entry_ids"],
                ),
                landing_rejections,
            )
        cell = _accepted_cell(
            group=hit["group"],
            evidence=evidence,
            dictionary_entry=entry,
            key_character=characters[0],
            template_recognizer=template_recognizer,
            walk_direction=direction,
            step_index=step_index,
            step_evidence=hit["candidate"],
            phrase_seed=phrase_seed,
        )
        cells.append(cell)
        claimed.update(evidence["primitive_ids"])
        current = dict(evidence)


def _walk_anchor(
    *,
    primitives: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    step_tables: Mapping[str, Any],
    character_table: Mapping[str, Any],
    template_recognizer: Callable[[dict[str, Any]], Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    group, failure_reason = _anchor_group(primitives=primitives, anchor=anchor)
    if group is None:
        return None, {
            "anchor_id": anchor["anchor_id"],
            "reason": failure_reason,
            "claimed_primitive_ids": [],
        }
    anchor_evidence = glyph_evidence(group)
    entry = lookup_character(character_table, shape_key=anchor_evidence["shape_key"])
    expected = ANCHOR_CHARACTERS[anchor["anchor_type"]]
    if entry is None or tuple(entry["characters"]) != (expected,):
        return None, {
            "anchor_id": anchor["anchor_id"],
            "reason": "anchor_group_unresolved",
            "claimed_primitive_ids": [],
        }
    phrase_seed = stable_digest(
        {
            "anchor_id": anchor["anchor_id"],
            "primitive_ids": list(anchor_evidence["primitive_ids"]),
        }
    )
    anchor_cell = _accepted_cell(
        group=group,
        evidence=anchor_evidence,
        dictionary_entry=entry,
        key_character=expected,
        template_recognizer=template_recognizer,
        walk_direction="anchor",
        step_index=0,
        step_evidence=None,
        phrase_seed=phrase_seed,
    )
    claimed = set(anchor_evidence["primitive_ids"])
    negative_cells, negative_stop, negative_rejections = _walk_direction(
        direction="negative",
        sign=-1,
        start_group=group,
        start_evidence=anchor_evidence,
        primitives=primitives,
        claimed=claimed,
        step_tables=step_tables,
        character_table=character_table,
        template_recognizer=template_recognizer,
        axis_angle_deg=float(anchor["axis_angle_deg"]),
        phrase_seed=phrase_seed,
        anchor_id=str(anchor["anchor_id"]),
    )
    positive_cells, positive_stop, positive_rejections = _walk_direction(
        direction="positive",
        sign=1,
        start_group=group,
        start_evidence=anchor_evidence,
        primitives=primitives,
        claimed=claimed,
        step_tables=step_tables,
        character_table=character_table,
        template_recognizer=template_recognizer,
        axis_angle_deg=float(anchor["axis_angle_deg"]),
        phrase_seed=phrase_seed,
        anchor_id=str(anchor["anchor_id"]),
    )
    cells = list(reversed(negative_cells)) + [anchor_cell] + positive_cells
    return {
        "anchor": anchor,
        "cells": cells,
        "negative_termination": negative_stop,
        "positive_termination": positive_stop,
        "landing_group_rejections": [
            *negative_rejections,
            *positive_rejections,
        ],
    }, None


def _low_confidence_step_diagnostics(
    step_tables: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    directions = step_tables.get("directions")
    if not isinstance(directions, Mapping):
        raise ValueError("step-table directions missing")
    result: dict[str, list[dict[str, Any]]] = {}
    for direction in ("negative", "positive"):
        table = directions.get(direction)
        if not isinstance(table, Mapping):
            raise ValueError(f"step-table direction missing: {direction}")
        raw_rows = table.get("sparse_candidates", ())
        if not isinstance(raw_rows, (list, tuple)):
            raise ValueError("sparse step candidates must be an array")
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            if (
                not isinstance(raw, Mapping)
                or raw.get("eligible") is not False
                or raw.get("low_confidence") is not True
            ):
                raise ValueError("sparse step candidate flags invalid")
            rows.append(
                {
                    "direction": direction,
                    "prev_key": str(raw.get("prev_key") or ""),
                    "entry_id": str(raw.get("entry_id") or ""),
                    "step_pt": float(raw["step_pt"]),
                    "n": int(raw["n"]),
                    "relative_dispersion": float(raw["relative_dispersion"]),
                    "eligible": False,
                    "low_confidence": True,
                    "consumer_allowed": False,
                }
            )
        result[direction] = sorted(
            rows,
            key=lambda row: (row["prev_key"], row["step_pt"], row["entry_id"]),
        )
    return result


def build_directional_walk_dump_v1(
    *,
    page_context: Any,
    anchors: Sequence[Mapping[str, Any]],
    trace_id: str,
    step_tables: Mapping[str, Any],
    character_table: Mapping[str, Any],
    template_recognizer: Callable[[dict[str, Any]], Any],
    claim_quality_rank_enabled: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, offline R92 walk dump without external I/O."""

    trace = str(trace_id)
    if not trace:
        raise ValueError("trace_id must be non-empty")
    manifest = _manifest(page_context)
    page_num = int(manifest["page_num"])
    page_index = page_num - 1
    if page_index < 0:
        raise ValueError("page context page_num must be positive")
    primitives = _primitive_rows(page_context)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for raw_anchor in sorted(anchors, key=lambda row: str(row.get("anchor_id") or "")):
        anchor = _normalized_anchor(raw_anchor, page_index=page_index)
        success, failure = _walk_anchor(
            primitives=primitives,
            anchor=anchor,
            step_tables=step_tables,
            character_table=character_table,
            template_recognizer=template_recognizer,
        )
        if failure is not None:
            failures.append(failure)
        elif success is not None:
            successes.append(success)
    phrase_rows = [
        (
            row,
            build_phrase_v1(
                trace_id=trace,
                page_index=page_index,
                anchor=row["anchor"],
                cells=row["cells"],
                negative_termination=row["negative_termination"],
                positive_termination=row["positive_termination"],
            ),
        )
        for row in successes
    ]
    subset_decisions = arbitrate_true_subset_claims_v1([
        {
            "anchor_id": row["anchor"]["anchor_id"],
            "input_order": input_order,
            "primitive_run_ids": phrase["primitive_run_ids"],
        }
        for input_order, (row, phrase) in enumerate(phrase_rows)
    ])
    active_indices = [
        index
        for index, decision in enumerate(subset_decisions)
        if decision is None
    ]
    if claim_quality_rank_enabled:
        ranked_active = _quality_ranked_claim_indices_v1([
            {
                "input_order": index,
                "phrase_id": phrase_rows[index][1]["phrase_id"],
                "text": phrase_rows[index][1]["text"],
                "bbox": phrase_rows[index][1]["bbox"],
                "primitive_run_ids": phrase_rows[index][1][
                    "primitive_run_ids"
                ],
            }
            for index in active_indices
        ])
        active_indices = [active_indices[index] for index in ranked_active]
    inactive_indices = [
        index
        for index, decision in enumerate(subset_decisions)
        if decision is not None
    ]
    arbitration_order = (
        active_indices + inactive_indices
        if claim_quality_rank_enabled
        else list(range(len(phrase_rows)))
    )
    phrases: list[dict[str, Any]] = []
    claimed_primitive_ids: set[str] = set()
    claim_failures: dict[int, dict[str, Any]] = {}
    for index in arbitration_order:
        row, phrase = phrase_rows[index]
        subset_decision = subset_decisions[index]
        if subset_decision is not None:
            claim_failures[index] = subset_decision
            continue
        conflicts = claimed_primitive_ids.intersection(phrase["primitive_run_ids"])
        if conflicts:
            claim_failures[index] = {
                "anchor_id": row["anchor"]["anchor_id"],
                "reason": "primitive_claim_conflict",
                "claimed_primitive_ids": [],
                "conflicting_primitive_ids": sorted(conflicts),
            }
            continue
        phrases.append(phrase)
        claimed_primitive_ids.update(phrase["primitive_run_ids"])
    failures.extend(claim_failures[index] for index in sorted(claim_failures))
    result = build_dump_v1(
        trace_id=trace,
        page_index=page_index,
        page_context_manifest=manifest,
        phrases=phrases,
        anchor_failures=failures,
        additional_diagnostics={
            "landing_group_rejections": sorted(
                (
                    dict(rejection)
                    for row in successes
                    for rejection in row["landing_group_rejections"]
                ),
                key=lambda rejection: (
                    rejection["anchor_id"],
                    rejection["direction"],
                    rejection["attempted_step_index"],
                    rejection["primitive_ids"],
                    rejection["candidate_step_entry_id"],
                ),
            ),
            "low_confidence_step_candidates": _low_confidence_step_diagnostics(
                step_tables
            )
        },
    )
    validate_directional_walk_dump_v1(result)
    return result


__all__ = ["build_directional_walk_dump_v1"]
