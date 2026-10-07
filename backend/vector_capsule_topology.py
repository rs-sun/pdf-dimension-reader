"""Fail-closed phrase-local vector capsule topology evidence.

This module does not recognize text and does not release evidence to consumers.
It only proves one closed outer contour made from two axis-compatible rails and
two opposite segmented bows.  Connected chords and branches are retained as
auditable non-boundary evidence, never folded into the outer contour.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

from vector_draw_items import ItemPrimitive
from vector_page_context import VectorPageContext


SCHEMA_VERSION = "vector_capsule_topology_v1"
REASON_SCHEMA_VERSION = "vector_capsule_topology_reason_v1"
DIGEST_SCHEMA_VERSION = "vector_capsule_topology_digest_v1"
AUDIT_SCHEMA_VERSION = "vector_capsule_topology_audit_v1"

DEFAULT_ENDPOINT_TOLERANCE_PT = 0.002
DEFAULT_SEARCH_PADDING_PT = 48.0
DEFAULT_RAIL_ANGLE_TOLERANCE_DEG = 2.0
DEFAULT_MAX_COMPONENT_EDGES = 64
DEFAULT_MAX_CYCLES = 128


@dataclass(frozen=True)
class _Edge:
    primitive: ItemPrimitive
    primitive_id: str
    node_a: int
    node_b: int
    length: float


@dataclass(frozen=True)
class _CompatibleCycle:
    edge_indices: tuple[int, ...]
    ordered_edge_indices: tuple[int, ...]
    ordered_node_indices: tuple[int, ...]
    roles: dict[str, tuple[int, ...]]
    interior_source_ids: tuple[str, ...]


def normalize_vector_capsule_topology_v1(
    topology: Any,
    *,
    expected_page_context_manifest: Any,
    expected_axis_angle_deg: float,
    expected_phrase_primitive_ids: Iterable[str],
) -> dict[str, Any] | None:
    """Return a replayed accepted sidecar or ``None``.

    This is the narrow integration boundary.  It binds the topology to the
    immutable page snapshot, trusted phrase axis, and exact phrase primitive
    universe before exposing the phrase-boundary intersection.
    """

    try:
        if vector_capsule_topology_identity_reasons(topology):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        topology.get("status") != "accepted"
        or topology.get("reason") != "unique_closed_outer_contour"
        or not isinstance(expected_page_context_manifest, dict)
        or expected_page_context_manifest.get("schema_version")
        != "vector_page_context_manifest_v1"
        or topology.get("page_context_primitive_sha256")
        != expected_page_context_manifest.get("primitive_sha256")
        or topology.get("page_num")
        != expected_page_context_manifest.get("page_num")
    ):
        return None

    try:
        expected_axis = _rounded(float(expected_axis_angle_deg))
        phrase_ids = frozenset(
            str(value) for value in expected_phrase_primitive_ids
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(expected_axis)
        or topology.get("axis_angle_deg") != expected_axis
        or not phrase_ids
        or any(not value for value in phrase_ids)
    ):
        return None

    roles = topology.get("roles")
    expected_role_names = {
        "leading_bow",
        "trailing_bow",
        "upper_rail",
        "lower_rail",
    }
    if not isinstance(roles, dict) or set(roles) != expected_role_names:
        return None
    normalized_roles = {
        role: _canonical_primitive_ids(roles.get(role))
        for role in expected_role_names
    }
    if any(not ids for ids in normalized_roles.values()):
        return None
    role_ids = [
        primitive_id
        for ids in normalized_roles.values()
        for primitive_id in ids
    ]
    if len(role_ids) != len(set(role_ids)):
        return None

    boundary_ids = _canonical_primitive_ids(
        topology.get("boundary_primitive_ids")
    )
    phrase_boundary_ids = _canonical_primitive_ids(
        topology.get("phrase_boundary_primitive_ids")
    )
    interior_ids = _canonical_primitive_ids(
        topology.get("interior_phrase_primitive_ids")
    )
    if (
        not boundary_ids
        or not phrase_boundary_ids
        or not interior_ids
        or boundary_ids != sorted(role_ids, key=_primitive_id_key)
        or phrase_boundary_ids != sorted(
            set(boundary_ids).intersection(phrase_ids),
            key=_primitive_id_key,
        )
        or not set(interior_ids).issubset(phrase_ids)
        or set(boundary_ids).intersection(interior_ids)
    ):
        return None

    non_boundary = topology.get("non_boundary_component_primitives")
    if not isinstance(non_boundary, list):
        return None
    non_boundary_ids: list[str] = []
    for row in non_boundary:
        if (
            not isinstance(row, dict)
            or set(row) != {"primitive_id", "reason"}
            or not isinstance(row.get("primitive_id"), str)
            or not row["primitive_id"]
            or row.get("reason")
            not in {
                "internal_chord_not_outer_contour",
                "connected_branch_not_outer_contour",
            }
        ):
            return None
        non_boundary_ids.append(row["primitive_id"])
    if (
        len(non_boundary_ids) != len(set(non_boundary_ids))
        or set(non_boundary_ids).intersection(boundary_ids)
    ):
        return None

    audit = topology.get("audit")
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != AUDIT_SCHEMA_VERSION
        or audit.get("compatible_outer_contour_count") != 1
    ):
        return None
    return json.loads(json.dumps(
        topology,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))


def vector_capsule_topology_identity_reasons(
    topology: Any,
) -> list[str]:
    """Replay the sidecar identity and fail closed on structural tampering."""

    if not isinstance(topology, dict):
        return ["topology_invalid"]
    if topology.get("schema_version") != SCHEMA_VERSION:
        return ["topology_schema_version_invalid"]
    if topology.get("reason_schema_version") != REASON_SCHEMA_VERSION:
        return ["reason_schema_version_invalid"]
    if (
        topology.get("topology_digest_schema_version")
        != DIGEST_SCHEMA_VERSION
    ):
        return ["topology_digest_schema_version_invalid"]
    digest = topology.get("topology_digest")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    ):
        return ["topology_digest_invalid"]

    status = topology.get("status")
    if status == "accepted":
        identity_keys = (
            "schema_version",
            "reason_schema_version",
            "status",
            "reason",
            "page_context_primitive_sha256",
            "page_num",
            "axis_angle_deg",
            "phrase_bbox",
            "roles",
            "boundary_primitive_ids",
            "phrase_boundary_primitive_ids",
            "interior_phrase_primitive_ids",
            "non_boundary_component_primitives",
        )
        identity = {key: topology.get(key) for key in identity_keys}
        expected_digest = _digest({
            **identity,
            "audit": topology.get("audit"),
        })
    elif status == "rejected":
        identity_keys = (
            "schema_version",
            "reason_schema_version",
            "status",
            "reason",
            "roles",
            "boundary_primitive_ids",
            "phrase_boundary_primitive_ids",
            "interior_phrase_primitive_ids",
            "non_boundary_component_primitives",
        )
        identity = {key: topology.get(key) for key in identity_keys}
        expected_digest = _digest({
            **identity,
            "audit": topology.get("audit"),
        })
    else:
        return ["topology_status_invalid"]
    if digest != expected_digest:
        return ["topology_digest_mismatch"]

    if topology.get("consumer_allowed") is not False:
        return ["consumer_allowed_invalid"]
    if status == "accepted":
        expected_capsule_id = (
            f"p{int(topology.get('page_num')):03d}_capsule_"
            f"{digest[:16]}"
        )
        if topology.get("capsule_id") != expected_capsule_id:
            return ["capsule_id_mismatch"]
    elif topology.get("capsule_id") is not None:
        return ["rejected_capsule_id_invalid"]
    return []


def build_vector_capsule_topology_v1(
    *,
    page_context: VectorPageContext,
    phrase_bbox: Iterable[float],
    axis_angle_deg: float,
    phrase_primitive_ids: Iterable[str],
    required_boundary_cut_primitive_ids: Iterable[str] = (),
    endpoint_tolerance_pt: float = DEFAULT_ENDPOINT_TOLERANCE_PT,
    search_padding_pt: float = DEFAULT_SEARCH_PADDING_PT,
    rail_angle_tolerance_deg: float = DEFAULT_RAIL_ANGLE_TOLERANCE_DEG,
    max_component_edges: int = DEFAULT_MAX_COMPONENT_EDGES,
    max_cycles: int = DEFAULT_MAX_CYCLES,
) -> dict[str, Any]:
    """Build one non-consumable, phrase-local capsule topology sidecar.

    The result is always a schema-valid audit record.  Any invalid input,
    ambiguity, incomplete closure, or search-budget breach returns a rejected
    record rather than a partial capsule.
    """

    normalized = _normalize_inputs(
        page_context=page_context,
        phrase_bbox=phrase_bbox,
        axis_angle_deg=axis_angle_deg,
        phrase_primitive_ids=phrase_primitive_ids,
        required_boundary_cut_primitive_ids=(
            required_boundary_cut_primitive_ids
        ),
        endpoint_tolerance_pt=endpoint_tolerance_pt,
        search_padding_pt=search_padding_pt,
        rail_angle_tolerance_deg=rail_angle_tolerance_deg,
        max_component_edges=max_component_edges,
        max_cycles=max_cycles,
    )
    if normalized["reason"] is not None:
        return _rejected(
            reason=str(normalized["reason"]),
            audit=normalized["audit"],
        )

    bbox = normalized["phrase_bbox"]
    padding = normalized["search_padding_pt"]
    query_bbox = (
        max(0.0, bbox[0] - padding),
        max(0.0, bbox[1] - padding),
        min(float(page_context.page_width), bbox[2] + padding),
        min(float(page_context.page_height), bbox[3] + padding),
    )
    audit = {
        **normalized["audit"],
        "query_bbox": [_rounded(value) for value in query_bbox],
    }

    try:
        page_context.assert_integrity()
        queried = page_context.query_bbox(query_bbox)
    except (TypeError, ValueError):
        return _rejected(
            reason="page_context_integrity_invalid",
            audit=audit,
        )

    nodes, edges = _build_endpoint_graph(
        page_context=page_context,
        primitives=queried,
        endpoint_tolerance_pt=normalized["endpoint_tolerance_pt"],
    )
    phrase_ids = normalized["phrase_primitive_ids"]
    required_cut_ids = normalized[
        "required_boundary_cut_primitive_ids"
    ]
    components = _edge_components(edges)
    relevant_components = [
        component
        for component in components
        if any(edges[index].primitive_id in phrase_ids for index in component)
        and len(component) >= 3
        and (
            not required_cut_ids
            or required_cut_ids.issubset({
                edges[index].primitive_id for index in component
            })
        )
    ]

    compatible: list[tuple[_CompatibleCycle, tuple[int, ...]]] = []
    enumerated_cycle_count = 0
    for component in relevant_components:
        if len(component) > normalized["max_component_edges"]:
            return _rejected(
                reason="search_budget_exceeded",
                audit={
                    **audit,
                    "relevant_component_count": len(relevant_components),
                    "oversized_component_edge_count": len(component),
                },
            )
        remaining_cycle_budget = (
            normalized["max_cycles"] - enumerated_cycle_count
        )
        cycles, exhausted = _enumerate_simple_cycles(
            edges=edges,
            component=component,
            max_cycles=max(0, remaining_cycle_budget),
        )
        enumerated_cycle_count += len(cycles)
        if exhausted:
            return _rejected(
                reason="search_budget_exceeded",
                audit={
                    **audit,
                    "relevant_component_count": len(relevant_components),
                    "enumerated_cycle_count": enumerated_cycle_count,
                },
            )
        for cycle in cycles:
            if (
                required_cut_ids
                and not required_cut_ids.issubset({
                    edges[index].primitive_id for index in cycle
                })
            ):
                continue
            classified = _classify_cycle(
                cycle=cycle,
                nodes=nodes,
                edges=edges,
                phrase_ids=phrase_ids,
                axis_angle_deg=normalized["axis_angle_deg"],
                rail_angle_tolerance_deg=normalized[
                    "rail_angle_tolerance_deg"
                ],
            )
            if classified is not None:
                compatible.append((classified, component))

    audit = {
        **audit,
        "queried_line_primitive_count": len(edges),
        "relevant_component_count": len(relevant_components),
        "enumerated_cycle_count": enumerated_cycle_count,
        "compatible_outer_contour_count": len(compatible),
    }
    if not compatible:
        return _rejected(
            reason="no_compatible_closed_outer_contour",
            audit=audit,
        )
    if len(compatible) != 1:
        return _rejected(
            reason="ambiguous_compatible_outer_contours",
            audit=audit,
        )

    cycle, component = compatible[0]
    boundary_edge_indices = set(cycle.edge_indices)
    roles = {
        role: sorted(
            (edges[index].primitive_id for index in indices),
            key=_primitive_id_key,
        )
        for role, indices in cycle.roles.items()
    }
    boundary_ids = sorted(
        (edges[index].primitive_id for index in boundary_edge_indices),
        key=_primitive_id_key,
    )
    phrase_boundary_ids = sorted(
        set(boundary_ids).intersection(phrase_ids),
        key=_primitive_id_key,
    )
    non_boundary = []
    cycle_nodes = set(cycle.ordered_node_indices)
    for index in sorted(
        set(component) - boundary_edge_indices,
        key=lambda edge_index: _primitive_id_key(
            edges[edge_index].primitive_id
        ),
    ):
        edge = edges[index]
        reason = (
            "internal_chord_not_outer_contour"
            if edge.node_a in cycle_nodes and edge.node_b in cycle_nodes
            else "connected_branch_not_outer_contour"
        )
        non_boundary.append({
            "primitive_id": edge.primitive_id,
            "reason": reason,
        })

    identity = {
        "schema_version": SCHEMA_VERSION,
        "reason_schema_version": REASON_SCHEMA_VERSION,
        "status": "accepted",
        "reason": "unique_closed_outer_contour",
        "page_context_primitive_sha256": (
            page_context.manifest_v1()["primitive_sha256"]
        ),
        "page_num": int(page_context.page_num),
        "axis_angle_deg": _rounded(normalized["axis_angle_deg"]),
        "phrase_bbox": [_rounded(value) for value in bbox],
        "roles": roles,
        "boundary_primitive_ids": boundary_ids,
        "phrase_boundary_primitive_ids": phrase_boundary_ids,
        "interior_phrase_primitive_ids": list(
            cycle.interior_source_ids
        ),
        "non_boundary_component_primitives": non_boundary,
    }
    topology_digest = _digest({
        **identity,
        "audit": audit,
    })
    return {
        **identity,
        "capsule_id": (
            f"p{int(page_context.page_num):03d}_capsule_"
            f"{topology_digest[:16]}"
        ),
        "topology_digest_schema_version": DIGEST_SCHEMA_VERSION,
        "topology_digest": topology_digest,
        "audit": audit,
        "consumer_allowed": False,
    }


def _normalize_inputs(
    *,
    page_context: Any,
    phrase_bbox: Iterable[float],
    axis_angle_deg: float,
    phrase_primitive_ids: Iterable[str],
    required_boundary_cut_primitive_ids: Iterable[str],
    endpoint_tolerance_pt: float,
    search_padding_pt: float,
    rail_angle_tolerance_deg: float,
    max_component_edges: int,
    max_cycles: int,
) -> dict[str, Any]:
    audit: dict[str, Any] = {"schema_version": AUDIT_SCHEMA_VERSION}
    try:
        bbox_values = tuple(float(value) for value in phrase_bbox)
        ids = frozenset(str(value) for value in phrase_primitive_ids)
        required_cut_ids = frozenset(
            str(value)
            for value in required_boundary_cut_primitive_ids
        )
        axis = float(axis_angle_deg)
        endpoint_tolerance = float(endpoint_tolerance_pt)
        padding = float(search_padding_pt)
        rail_tolerance = float(rail_angle_tolerance_deg)
    except (TypeError, ValueError, OverflowError):
        return {"reason": "invalid_input", "audit": audit}

    valid_context = isinstance(page_context, VectorPageContext)
    valid_bbox = (
        len(bbox_values) == 4
        and all(math.isfinite(value) for value in bbox_values)
        and bbox_values[0] < bbox_values[2]
        and bbox_values[1] < bbox_values[3]
    )
    valid_scalars = (
        math.isfinite(axis)
        and math.isfinite(endpoint_tolerance)
        and 0.0 < endpoint_tolerance <= 0.01
        and math.isfinite(padding)
        and 0.0 < padding <= 64.0
        and math.isfinite(rail_tolerance)
        and 0.0 < rail_tolerance <= 5.0
        and type(max_component_edges) is int
        and 3 <= max_component_edges <= 256
        and type(max_cycles) is int
        and 1 <= max_cycles <= 1024
        and bool(ids)
        and all(value for value in ids)
        and all(value for value in required_cut_ids)
    )
    if not valid_context or not valid_bbox or not valid_scalars:
        return {"reason": "invalid_input", "audit": audit}

    return {
        "reason": None,
        "phrase_bbox": bbox_values,
        "axis_angle_deg": axis,
        "phrase_primitive_ids": ids,
        "required_boundary_cut_primitive_ids": required_cut_ids,
        "endpoint_tolerance_pt": endpoint_tolerance,
        "search_padding_pt": padding,
        "rail_angle_tolerance_deg": rail_tolerance,
        "max_component_edges": max_component_edges,
        "max_cycles": max_cycles,
        "audit": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "phrase_bbox": [_rounded(value) for value in bbox_values],
            "axis_angle_deg": _rounded(axis),
            "phrase_primitive_count": len(ids),
            "required_boundary_cut_primitive_ids": sorted(
                required_cut_ids,
                key=_primitive_id_key,
            ),
            "endpoint_tolerance_pt": _rounded(endpoint_tolerance),
            "search_padding_pt": _rounded(padding),
            "rail_angle_tolerance_deg": _rounded(rail_tolerance),
            "max_component_edges": max_component_edges,
            "max_cycles": max_cycles,
        },
    }


def _build_endpoint_graph(
    *,
    page_context: VectorPageContext,
    primitives: Iterable[ItemPrimitive],
    endpoint_tolerance_pt: float,
) -> tuple[list[tuple[float, float]], list[_Edge]]:
    nodes: list[tuple[float, float]] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def node_for(point: tuple[float, float]) -> int:
        bucket_x = math.floor(point[0] / endpoint_tolerance_pt)
        bucket_y = math.floor(point[1] / endpoint_tolerance_pt)
        best: tuple[float, int] | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for node_index in buckets.get(
                    (bucket_x + dx, bucket_y + dy),
                    (),
                ):
                    distance = math.dist(point, nodes[node_index])
                    if distance <= endpoint_tolerance_pt and (
                        best is None or (distance, node_index) < best
                    ):
                        best = (distance, node_index)
        if best is not None:
            return best[1]
        node_index = len(nodes)
        nodes.append((float(point[0]), float(point[1])))
        buckets[(bucket_x, bucket_y)].append(node_index)
        return node_index

    edges: list[_Edge] = []
    for primitive in primitives:
        if primitive.op != "l" or len(primitive.points) < 2:
            continue
        point_a = primitive.points[0]
        point_b = primitive.points[-1]
        length = math.dist(point_a, point_b)
        if not math.isfinite(length) or length <= endpoint_tolerance_pt:
            continue
        node_a = node_for(point_a)
        node_b = node_for(point_b)
        if node_a == node_b:
            continue
        edges.append(_Edge(
            primitive=primitive,
            primitive_id=page_context.item_id(primitive),
            node_a=node_a,
            node_b=node_b,
            length=length,
        ))
    return nodes, edges


def _edge_components(edges: list[_Edge]) -> list[tuple[int, ...]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        adjacency[edge.node_a].append(index)
        adjacency[edge.node_b].append(index)
    remaining = set(range(len(edges)))
    components: list[tuple[int, ...]] = []
    while remaining:
        first = min(remaining)
        queue = deque([first])
        component: set[int] = set()
        while queue:
            edge_index = queue.popleft()
            if edge_index in component:
                continue
            component.add(edge_index)
            edge = edges[edge_index]
            for node in (edge.node_a, edge.node_b):
                queue.extend(
                    neighbor
                    for neighbor in adjacency[node]
                    if neighbor not in component
                )
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return components


def _enumerate_simple_cycles(
    *,
    edges: list[_Edge],
    component: tuple[int, ...],
    max_cycles: int,
) -> tuple[list[tuple[int, ...]], bool]:
    if max_cycles <= 0:
        return [], True
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    component_nodes: set[int] = set()
    for edge_index in component:
        edge = edges[edge_index]
        component_nodes.update((edge.node_a, edge.node_b))
        adjacency[edge.node_a].append((edge.node_b, edge_index))
        adjacency[edge.node_b].append((edge.node_a, edge_index))
    for neighbors in adjacency.values():
        neighbors.sort()

    found: dict[tuple[int, ...], tuple[int, ...]] = {}
    exhausted = False
    for start in sorted(component_nodes):
        visited = {start}

        def walk(
            node: int,
            path_edges: list[int],
        ) -> None:
            nonlocal exhausted
            if exhausted:
                return
            for neighbor, edge_index in adjacency[node]:
                if path_edges and edge_index == path_edges[-1]:
                    continue
                if neighbor == start:
                    if len(path_edges) >= 2:
                        key = tuple(sorted((*path_edges, edge_index)))
                        found.setdefault(key, key)
                        if len(found) > max_cycles:
                            exhausted = True
                            return
                    continue
                if neighbor < start or neighbor in visited:
                    continue
                visited.add(neighbor)
                path_edges.append(edge_index)
                walk(neighbor, path_edges)
                path_edges.pop()
                visited.remove(neighbor)

        walk(start, [])
        if exhausted:
            break
    cycles = list(found.values())
    cycles.sort()
    return cycles[:max_cycles], exhausted


def _classify_cycle(
    *,
    cycle: tuple[int, ...],
    nodes: list[tuple[float, float]],
    edges: list[_Edge],
    phrase_ids: frozenset[str],
    axis_angle_deg: float,
    rail_angle_tolerance_deg: float,
) -> _CompatibleCycle | None:
    ordered = _order_cycle(cycle=cycle, edges=edges)
    if ordered is None:
        return None
    ordered_edges, ordered_nodes = ordered
    if len(ordered_edges) < 12:
        return None

    radians = math.radians(axis_angle_deg)
    main = (math.cos(radians), math.sin(radians))
    cross = (-main[1], main[0])
    rail_cosine = math.cos(math.radians(rail_angle_tolerance_deg))
    rail_flags = []
    for edge_index in ordered_edges:
        edge = edges[edge_index]
        point_a = nodes[edge.node_a]
        point_b = nodes[edge.node_b]
        delta = (point_b[0] - point_a[0], point_b[1] - point_a[1])
        alignment = abs(
            (delta[0] * main[0] + delta[1] * main[1]) / edge.length
        )
        rail_flags.append(alignment >= rail_cosine)

    rail_groups = _cyclic_groups(rail_flags, expected=True)
    bow_groups = _cyclic_groups(rail_flags, expected=False)
    if len(rail_groups) != 2 or len(bow_groups) != 2:
        return None

    main_values = [
        _project(nodes[node_index], main)
        for node_index in ordered_nodes[:-1]
    ]
    cross_values = [
        _project(nodes[node_index], cross)
        for node_index in ordered_nodes[:-1]
    ]
    main_span = max(main_values) - min(main_values)
    cross_span = max(cross_values) - min(cross_values)
    if main_span <= 0.0 or cross_span <= 0.0:
        return None

    rail_group_edges = [
        tuple(ordered_edges[index] for index in group)
        for group in rail_groups
    ]
    bow_group_edges = [
        tuple(ordered_edges[index] for index in group)
        for group in bow_groups
    ]
    if any(
        sum(edges[index].length for index in group) < 0.5 * main_span
        for group in rail_group_edges
    ):
        return None
    if any(
        len(group) < 6
        or sum(edges[index].length for index in group) < cross_span
        for group in bow_group_edges
    ):
        return None

    rail_cross_centers = [
        _edge_group_center(
            group=group,
            edges=edges,
            nodes=nodes,
            axis=cross,
        )
        for group in rail_group_edges
    ]
    bow_main_centers = [
        _edge_group_center(
            group=group,
            edges=edges,
            nodes=nodes,
            axis=main,
        )
        for group in bow_group_edges
    ]
    if abs(rail_cross_centers[0] - rail_cross_centers[1]) < (
        0.75 * cross_span
    ):
        return None
    if abs(bow_main_centers[0] - bow_main_centers[1]) < 0.75 * main_span:
        return None

    polygon = [nodes[index] for index in ordered_nodes]
    boundary_ids = {edges[index].primitive_id for index in cycle}
    interior_source_ids = tuple(sorted(
        (
            edge.primitive_id
            for edge in edges
            if edge.primitive_id in phrase_ids
            and edge.primitive_id not in boundary_ids
            and _point_in_polygon(
                _primitive_center(edge.primitive),
                polygon,
            )
        ),
        key=_primitive_id_key,
    ))
    if not interior_source_ids:
        return None

    upper_index = min(
        range(2),
        key=lambda index: rail_cross_centers[index],
    )
    leading_index = min(
        range(2),
        key=lambda index: bow_main_centers[index],
    )
    roles = {
        "leading_bow": bow_group_edges[leading_index],
        "trailing_bow": bow_group_edges[1 - leading_index],
        "upper_rail": rail_group_edges[upper_index],
        "lower_rail": rail_group_edges[1 - upper_index],
    }
    return _CompatibleCycle(
        edge_indices=tuple(sorted(cycle)),
        ordered_edge_indices=tuple(ordered_edges),
        ordered_node_indices=tuple(ordered_nodes),
        roles=roles,
        interior_source_ids=interior_source_ids,
    )


def _order_cycle(
    *,
    cycle: tuple[int, ...],
    edges: list[_Edge],
) -> tuple[list[int], list[int]] | None:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge_index in cycle:
        edge = edges[edge_index]
        adjacency[edge.node_a].append(edge_index)
        adjacency[edge.node_b].append(edge_index)
    if not adjacency or any(len(indices) != 2 for indices in adjacency.values()):
        return None
    start = min(adjacency)
    first_edge = min(adjacency[start])
    ordered_edges: list[int] = []
    ordered_nodes = [start]
    current_node = start
    current_edge = first_edge
    while len(ordered_edges) < len(cycle):
        ordered_edges.append(current_edge)
        edge = edges[current_edge]
        next_node = (
            edge.node_b if edge.node_a == current_node else edge.node_a
        )
        ordered_nodes.append(next_node)
        if next_node == start:
            break
        candidates = [
            index
            for index in adjacency[next_node]
            if index != current_edge
        ]
        if len(candidates) != 1:
            return None
        current_node = next_node
        current_edge = candidates[0]
    if (
        len(ordered_edges) != len(cycle)
        or ordered_nodes[-1] != start
        or len(set(ordered_edges)) != len(cycle)
    ):
        return None
    return ordered_edges, ordered_nodes


def _cyclic_groups(
    flags: list[bool],
    *,
    expected: bool,
) -> list[tuple[int, ...]]:
    if not flags or all(flag == expected for flag in flags):
        return [tuple(range(len(flags)))] if flags else []
    start = next(
        index
        for index, flag in enumerate(flags)
        if flag != expected
    )
    groups: list[list[int]] = []
    current: list[int] = []
    for offset in range(1, len(flags) + 1):
        index = (start + offset) % len(flags)
        if flags[index] == expected:
            current.append(index)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [tuple(group) for group in groups]


def _edge_group_center(
    *,
    group: tuple[int, ...],
    edges: list[_Edge],
    nodes: list[tuple[float, float]],
    axis: tuple[float, float],
) -> float:
    points = []
    for edge_index in group:
        edge = edges[edge_index]
        points.extend((nodes[edge.node_a], nodes[edge.node_b]))
    return sum(_project(point, axis) for point in points) / len(points)


def _primitive_center(item: ItemPrimitive) -> tuple[float, float]:
    return (
        sum(point[0] for point in item.points) / len(item.points),
        sum(point[1] for point in item.points) / len(item.points),
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    inside = False
    x, y = point
    for index in range(len(polygon) - 1):
        x0, y0 = polygon[index]
        x1, y1 = polygon[index + 1]
        if (y0 > y) != (y1 > y):
            crossing_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if crossing_x > x:
                inside = not inside
    return inside


def _project(
    point: tuple[float, float],
    axis: tuple[float, float],
) -> float:
    return point[0] * axis[0] + point[1] * axis[1]


def _rejected(
    *,
    reason: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "reason_schema_version": REASON_SCHEMA_VERSION,
        "status": "rejected",
        "reason": reason,
        "roles": {
            "leading_bow": [],
            "trailing_bow": [],
            "upper_rail": [],
            "lower_rail": [],
        },
        "boundary_primitive_ids": [],
        "phrase_boundary_primitive_ids": [],
        "interior_phrase_primitive_ids": [],
        "non_boundary_component_primitives": [],
    }
    topology_digest = _digest({
        **identity,
        "audit": audit,
    })
    return {
        **identity,
        "capsule_id": None,
        "topology_digest_schema_version": DIGEST_SCHEMA_VERSION,
        "topology_digest": topology_digest,
        "audit": audit,
        "consumer_allowed": False,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _primitive_id_key(value: str) -> tuple[int, int, int, str]:
    try:
        page_text, drawing_text, item_text, op = value.split("_")
        return (
            int(page_text[1:]),
            int(drawing_text[1:]),
            int(item_text[1:]),
            op,
        )
    except (TypeError, ValueError):
        return (2**31 - 1, 2**31 - 1, 2**31 - 1, str(value))


def _canonical_primitive_ids(value: Any) -> list[str] | None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        return None
    return sorted(value, key=_primitive_id_key)


def _rounded(value: float) -> float:
    return round(float(value), 6)
