"""Order-independent primitive-claim arbitration for directional walks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def arbitrate_true_subset_claims_v1(
    claims: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, ...]:
    """Apply proper-subset swaps using each claim's stable ``input_order``."""

    normalized: list[tuple[str, int, frozenset[str]]] = []
    for claim in claims:
        anchor_id = claim.get("anchor_id")
        input_order = claim.get("input_order")
        raw_primitive_ids = claim.get("primitive_run_ids")
        if (
            not isinstance(anchor_id, str)
            or not anchor_id
            or type(input_order) is not int
            or input_order < 0
            or not isinstance(raw_primitive_ids, (list, tuple))
        ):
            raise ValueError("primitive claim evidence incomplete")
        if any(
            not isinstance(value, str) or not value
            for value in raw_primitive_ids
        ):
            raise ValueError("primitive claim ids invalid")
        primitive_ids = tuple(raw_primitive_ids)
        primitive_set = frozenset(primitive_ids)
        if (
            not primitive_set
            or len(primitive_set) != len(primitive_ids)
        ):
            raise ValueError("primitive claim ids invalid")
        normalized.append((anchor_id, input_order, primitive_set))
    if len({row[0] for row in normalized}) != len(normalized):
        raise ValueError("primitive claim anchor ids must be unique")
    if len({row[1] for row in normalized}) != len(normalized):
        raise ValueError("primitive claim input order must be unique")

    decisions: list[dict[str, Any] | None] = [None] * len(normalized)
    strict_supersets_by_index: list[list[int]] = []
    for index, (_anchor_id, _input_order, primitive_ids) in enumerate(
        normalized
    ):
        strict_supersets_by_index.append([
            other_index
            for other_index, (_other_anchor, _other_order, other_ids)
            in enumerate(normalized)
            if primitive_ids < other_ids
        ])

    parents = list(range(len(normalized)))

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

    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            if normalized[left][2].intersection(normalized[right][2]):
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(normalized)):
        components.setdefault(find(index), []).append(index)

    def first_come_delivered(
        component: Sequence[int],
        *,
        skipped: set[int],
    ) -> set[int]:
        claimed: set[str] = set()
        delivered: set[int] = set()
        for index in sorted(
            component,
            key=lambda value: (
                normalized[value][1],
                normalized[value][0],
                value,
            ),
        ):
            if index in skipped:
                continue
            primitive_ids = normalized[index][2]
            if claimed.intersection(primitive_ids):
                continue
            delivered.add(index)
            claimed.update(primitive_ids)
        return delivered

    def witness_key(index: int) -> tuple[int, str, int, int]:
        anchor_id, input_order, primitive_ids = normalized[index]
        return (-len(primitive_ids), anchor_id, input_order, index)

    for component in components.values():
        proposed_subsets = {
            index
            for index in component
            if strict_supersets_by_index[index]
        }
        if not proposed_subsets:
            continue
        baseline_delivered = first_come_delivered(
            component,
            skipped=set(),
        )
        proposed_delivered = first_come_delivered(
            component,
            skipped=proposed_subsets,
        )
        delivered_witnesses: dict[int, int] = {}
        for index in proposed_subsets:
            live_supersets = [
                other_index
                for other_index in proposed_delivered
                if normalized[index][2] < normalized[other_index][2]
            ]
            if not live_supersets:
                break
            delivered_witnesses[index] = min(
                live_supersets,
                key=witness_key,
            )
        else:
            delivered_losses = baseline_delivered - proposed_delivered
            delivered_gains = proposed_delivered - baseline_delivered
            if (
                delivered_losses.issubset(proposed_subsets)
                and delivered_gains
                == {
                    delivered_witnesses[index]
                    for index in delivered_losses
                }
            ):
                for index in proposed_subsets:
                    decisions[index] = {
                        "anchor_id": normalized[index][0],
                        "reason": "subsumed_by_superset_claim",
                        "claimed_primitive_ids": [],
                        "superset_anchor_id": normalized[
                            delivered_witnesses[index]
                        ][0],
                    }
    return tuple(decisions)


__all__ = ["arbitrate_true_subset_claims_v1"]
