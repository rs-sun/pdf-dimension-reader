"""Fail-closed repeated-size evidence inside one decimal height cluster.

The proof is a private sibling of ``vector_decimal_corridor_v1``.  It carries
the complete parent member ledger so every derived cohort value can be
recomputed without trusting summary fields or a digest alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Any


SCHEMA_VERSION = "vector_decimal_repeated_size_cohort_v1"
SOURCE = "decimal_parent_cluster_repeated_size_cohort"
CHAR_HEIGHT_SCHEMA_VERSION = "vector_decimal_char_height_v1"
CORRIDOR_SCHEMA_VERSION = "vector_decimal_corridor_v1"
DECIMAL_QUAD_SCHEMA_VERSION = "vector_decimal_point_quad_v1"
DECIMAL_QUAD_SOURCE = "qu_rect_ratio"
MIN_COMPONENT_SUPPORT = 10
RELATIVE_BAND = 0.02
ABSOLUTE_BAND_PT = 0.02
CONSUMER_ALLOWED = False


def build_decimal_repeated_size_cohort_proofs_v1(
    decimal_quad_rows: list[dict[str, Any]],
    char_height_clusters: list[dict[str, Any]],
    decimal_corridor_rows: list[dict[str, Any]],
    *,
    page_index: int,
) -> list[dict[str, Any]]:
    """Build proofs only for direct tolerance members rejected by global C4."""
    if (
        type(page_index) is not int
        or page_index < 0
        or not isinstance(decimal_quad_rows, list)
        or not isinstance(char_height_clusters, list)
        or not isinstance(decimal_corridor_rows, list)
    ):
        return []
    quad_by_id = _unique_quad_long_edges(decimal_quad_rows, page_index=page_index)
    corridor_by_id = _unique_corridors(decimal_corridor_rows, page_index=page_index)
    if quad_by_id is None or corridor_by_id is None:
        return []

    proofs: list[dict[str, Any]] = []
    seen_cluster_ids: set[str] = set()
    for cluster in char_height_clusters:
        if not isinstance(cluster, dict):
            return []
        cluster_id = cluster.get("cluster_id")
        member_ids = cluster.get("member_quad_ids")
        count = cluster.get("count")
        if (
            cluster.get("schema_version") != CHAR_HEIGHT_SCHEMA_VERSION
            or cluster.get("consumer_allowed") is not False
            or type(cluster_id) is not str
            or not cluster_id
            or cluster_id in seen_cluster_ids
            or type(count) is not int
            or count < 1
            or not isinstance(member_ids, list)
            or count != len(member_ids)
            or any(type(member_id) is not str or not member_id for member_id in member_ids)
            or len(set(member_ids)) != len(member_ids)
            or any(member_id not in quad_by_id for member_id in member_ids)
        ):
            return []
        seen_cluster_ids.add(cluster_id)
        if cluster.get("role") != "tolerance":
            continue
        ledger = sorted(
            (
                {
                    "decimal_quad_id": member_id,
                    "long_edge_pt": quad_by_id[member_id],
                }
                for member_id in member_ids
            ),
            key=lambda row: (row["long_edge_pt"], row["decimal_quad_id"]),
        )
        components = _components(ledger)
        component_by_id = {
            member["decimal_quad_id"]: component
            for component in components
            for member in component
        }
        for member_id in member_ids:
            corridor = corridor_by_id.get(member_id)
            component = component_by_id.get(member_id)
            if (
                corridor is None
                or component is None
                or corridor.get("char_height_cluster_id") != cluster_id
                or corridor.get("size_class") != "tolerance"
                or corridor.get("cluster_membership") != "direct_member_quad_id"
                or corridor.get("cluster_member_count") != count
                or not _failed_global_c4_with_other_conditions(corridor)
            ):
                continue
            proof = _proof_from_component(
                ledger,
                component,
                target_quad_id=member_id,
                cluster_id=cluster_id,
                page_index=page_index,
            )
            if proof is not None:
                proofs.append(proof)
    return sorted(proofs, key=lambda row: row["decimal_quad_id"])


def rebuild_decimal_repeated_size_cohort_proofs_from_corridors_v1(
    decimal_quad_rows: list[dict[str, Any]],
    decimal_corridor_rows: list[dict[str, Any]],
    *,
    page_index: int,
) -> list[dict[str, Any]]:
    """Rebuild private proofs for legacy replays that only persisted corridors."""
    if not isinstance(decimal_corridor_rows, list):
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in decimal_corridor_rows:
        if not isinstance(row, dict):
            return []
        cluster_id = row.get("char_height_cluster_id")
        if type(cluster_id) is not str or not cluster_id:
            return []
        if row.get("cluster_membership") == "direct_member_quad_id":
            grouped.setdefault(cluster_id, []).append(row)
    clusters: list[dict[str, Any]] = []
    for cluster_id, rows in sorted(grouped.items()):
        role_values = [row.get("size_class") for row in rows]
        count_values = [row.get("cluster_member_count") for row in rows]
        member_ids = [row.get("decimal_quad_id") for row in rows]
        if (
            any(type(role) is not str or not role for role in role_values)
            or any(
                type(count) is not int or count < 1
                for count in count_values
            )
            or len(set(role_values)) != 1
            or len(set(count_values)) != 1
            or any(type(member_id) is not str or not member_id for member_id in member_ids)
            or len(set(member_ids)) != len(member_ids)
            or count_values[0] != len(member_ids)
        ):
            return []
        clusters.append({
            "schema_version": CHAR_HEIGHT_SCHEMA_VERSION,
            "cluster_id": cluster_id,
            "role": role_values[0],
            "count": len(member_ids),
            "member_quad_ids": member_ids,
            "consumer_allowed": False,
        })
    return build_decimal_repeated_size_cohort_proofs_v1(
        decimal_quad_rows,
        clusters,
        decimal_corridor_rows,
        page_index=page_index,
    )


def normalize_decimal_repeated_size_cohort_proof(
    value: Any,
) -> dict[str, Any] | None:
    """Recompute a proof from its complete parent ledger, or reject it."""
    expected_keys = {
        "schema_version",
        "page_index",
        "decimal_quad_id",
        "char_height_cluster_id",
        "size_class",
        "cluster_membership",
        "target_long_edge_pt",
        "parent_member_count",
        "parent_member_ledger",
        "component_member_quad_ids",
        "component_member_count",
        "component_median_long_edge_pt",
        "component_band_pt",
        "component_max_relative_error",
        "nearest_external_gap_pt",
        "source",
        "consumer_allowed",
        "proof_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and type(value.get("page_index")) is int
        and value["page_index"] >= 0
        and type(value.get("decimal_quad_id")) is str
        and bool(value["decimal_quad_id"])
        and type(value.get("char_height_cluster_id")) is str
        and bool(value["char_height_cluster_id"])
        and value.get("size_class") == "tolerance"
        and value.get("cluster_membership") == "direct_member_quad_id"
        and type(value.get("parent_member_count")) is int
        and type(value.get("component_member_count")) is int
        and isinstance(value.get("parent_member_ledger"), list)
        and isinstance(value.get("component_member_quad_ids"), list)
        and all(
            _strict_positive_number(value.get(key)) is not None
            for key in (
                "target_long_edge_pt",
                "component_median_long_edge_pt",
                "component_band_pt",
                "nearest_external_gap_pt",
            )
        )
        and _strict_nonnegative_number(
            value.get("component_max_relative_error")
        )
        is not None
        and value.get("source") == SOURCE
        and value.get("consumer_allowed") is False
        and type(value.get("proof_digest")) is str
    ):
        return None

    ledger: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_member in value["parent_member_ledger"]:
        if (
            not isinstance(raw_member, dict)
            or set(raw_member) != {"decimal_quad_id", "long_edge_pt"}
            or type(raw_member.get("decimal_quad_id")) is not str
            or not raw_member["decimal_quad_id"]
            or raw_member["decimal_quad_id"] in seen_ids
            or _strict_positive_number(raw_member.get("long_edge_pt")) is None
        ):
            return None
        seen_ids.add(raw_member["decimal_quad_id"])
        canonical_long_edge = round(float(raw_member["long_edge_pt"]), 6)
        if canonical_long_edge <= 0.0:
            return None
        ledger.append({
            "decimal_quad_id": raw_member["decimal_quad_id"],
            "long_edge_pt": canonical_long_edge,
        })
    if ledger != sorted(
        ledger,
        key=lambda row: (row["long_edge_pt"], row["decimal_quad_id"]),
    ):
        return None
    if value["parent_member_count"] != len(ledger):
        return None
    target_id = value["decimal_quad_id"]
    target_rows = [row for row in ledger if row["decimal_quad_id"] == target_id]
    if len(target_rows) != 1:
        return None
    components = _components(ledger)
    target_component = next(
        (
            component
            for component in components
            if any(row["decimal_quad_id"] == target_id for row in component)
        ),
        None,
    )
    if target_component is None:
        return None
    rebuilt = _proof_from_component(
        ledger,
        target_component,
        target_quad_id=target_id,
        cluster_id=value["char_height_cluster_id"],
        page_index=value["page_index"],
    )
    if rebuilt is None or rebuilt != value:
        return None
    return rebuilt


def _proof_from_component(
    ledger: list[dict[str, Any]],
    component: list[dict[str, Any]],
    *,
    target_quad_id: str,
    cluster_id: str,
    page_index: int,
) -> dict[str, Any] | None:
    parent_count = len(ledger)
    component_count = len(component)
    minimum_support = max(
        MIN_COMPONENT_SUPPORT,
        math.ceil(parent_count * 0.05),
    )
    if component_count < minimum_support or component_count >= parent_count:
        return None
    values = [row["long_edge_pt"] for row in component]
    raw_component_median = float(median(values))
    if not math.isfinite(raw_component_median) or raw_component_median <= 0.0:
        return None
    raw_band = max(
        ABSOLUTE_BAND_PT,
        RELATIVE_BAND * raw_component_median,
    )
    raw_max_relative_error = max(
        abs(edge - raw_component_median) / raw_component_median
        for edge in values
    )
    if (
        any(abs(edge - raw_component_median) > raw_band for edge in values)
        or raw_max_relative_error > RELATIVE_BAND
    ):
        return None
    component_median = round(raw_component_median, 6)
    band = round(raw_band, 6)
    max_relative_error = round(raw_max_relative_error, 6)
    component_ids = [row["decimal_quad_id"] for row in component]
    if target_quad_id not in component_ids:
        return None
    external = [
        row["long_edge_pt"]
        for row in ledger
        if row["decimal_quad_id"] not in set(component_ids)
    ]
    if not external:
        return None
    raw_nearest_external_gap = min(
        abs(edge - raw_component_median) for edge in external
    )
    if raw_nearest_external_gap <= 2.0 * raw_band:
        return None
    nearest_external_gap = round(raw_nearest_external_gap, 6)
    target_long_edge = next(
        row["long_edge_pt"]
        for row in component
        if row["decimal_quad_id"] == target_quad_id
    )
    proof = {
        "schema_version": SCHEMA_VERSION,
        "page_index": page_index,
        "decimal_quad_id": target_quad_id,
        "char_height_cluster_id": cluster_id,
        "size_class": "tolerance",
        "cluster_membership": "direct_member_quad_id",
        "target_long_edge_pt": target_long_edge,
        "parent_member_count": parent_count,
        "parent_member_ledger": ledger,
        "component_member_quad_ids": component_ids,
        "component_member_count": component_count,
        "component_median_long_edge_pt": component_median,
        "component_band_pt": band,
        "component_max_relative_error": max_relative_error,
        "nearest_external_gap_pt": nearest_external_gap,
        "source": SOURCE,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    proof["proof_digest"] = _digest(proof)
    return proof


def _components(
    ledger: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not ledger:
        return []
    components = [[ledger[0]]]
    for member in ledger[1:]:
        previous = components[-1][-1]
        allowed_gap = max(
            ABSOLUTE_BAND_PT,
            RELATIVE_BAND
            * max(previous["long_edge_pt"], member["long_edge_pt"]),
        )
        if member["long_edge_pt"] - previous["long_edge_pt"] <= allowed_gap:
            components[-1].append(member)
        else:
            components.append([member])
    return components


def _unique_quad_long_edges(
    rows: list[dict[str, Any]],
    *,
    page_index: int,
) -> dict[str, float] | None:
    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        quad_id = row.get("id")
        detail = row.get("detail")
        if (
            row.get("schema_version") != DECIMAL_QUAD_SCHEMA_VERSION
            or row.get("source") != DECIMAL_QUAD_SOURCE
            or row.get("consumer_allowed") is not False
            or type(quad_id) is not str
            or not quad_id
            or quad_id in out
            or not isinstance(detail, dict)
            or _strict_positive_number(detail.get("long")) is None
            or (
                "page_index" in row
                and (
                    type(row.get("page_index")) is not int
                    or row["page_index"] != page_index
                )
            )
        ):
            return None
        out[quad_id] = round(float(detail["long"]), 6)
    return out


def _unique_corridors(
    rows: list[dict[str, Any]],
    *,
    page_index: int,
) -> dict[str, dict[str, Any]] | None:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        quad_id = row.get("decimal_quad_id")
        if (
            row.get("schema_version") != CORRIDOR_SCHEMA_VERSION
            or row.get("consumer_allowed") is not False
            or type(row.get("page_index")) is not int
            or row["page_index"] != page_index
            or type(quad_id) is not str
            or not quad_id
            or quad_id in out
        ):
            return None
        out[quad_id] = row
    return out


def _failed_global_c4_with_other_conditions(row: dict[str, Any]) -> bool:
    within = _strict_nonnegative_number(row.get("within_cluster_relative_error"))
    nearest_gap = _strict_nonnegative_number(row.get("nearest_cluster_relative_gap"))
    return bool(
        within is not None
        and within > 0.15
        and type(row.get("cluster_member_count")) is int
        and row["cluster_member_count"] >= 3
        and type(row.get("page_cluster_count")) is int
        and row["page_cluster_count"] >= 2
        and nearest_gap is not None
        and nearest_gap >= 0.20
    )


def _strict_positive_number(value: Any) -> float | None:
    if type(value) not in {int, float} or type(value) is bool:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _strict_nonnegative_number(value: Any) -> float | None:
    if type(value) not in {int, float} or type(value) is bool:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
