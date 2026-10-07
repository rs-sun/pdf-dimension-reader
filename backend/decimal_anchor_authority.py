"""Auditable page-scale authority for one strict decimal-point anchor.

The size class is evidence for semantic routing only.  It must not be used as
an inferred character box: the local phrase builder owns ROI geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from decimal_repeated_size_cohort import (
    SCHEMA_VERSION as REPEATED_SIZE_COHORT_SCHEMA_VERSION,
    normalize_decimal_repeated_size_cohort_proof,
)


SCHEMA_VERSION = "vector_decimal_anchor_authority_v1"
SCHEMA_VERSION_V2 = "vector_decimal_anchor_authority_v2"
CLASSIFICATION_SOURCE = "page_decimal_long_edge_cluster"
AUTHORITATIVE_SIZE_CLASSES = frozenset({"primary", "tolerance"})
KNOWN_SIZE_CLASSES = AUTHORITATIVE_SIZE_CLASSES | frozenset({"aux", "outlier"})
DECIMAL_QUAD_SCHEMA_VERSION = "vector_decimal_point_quad_v1"
DECIMAL_QUAD_SOURCE = "qu_rect_ratio"
DECIMAL_CORRIDOR_SCHEMA_VERSION = "vector_decimal_corridor_v1"
DECIMAL_CORRIDOR_SOURCE = "decimal_char_height_back_inference"
CALIBRATED_DECIMAL_EXTERNAL_METHOD = (
    "nearest_view_bbox_calibrated_decimal_external"
)
CALIBRATED_DECIMAL_EXTERNAL_GROUP_METHOD = (
    "group_consensus_calibrated_decimal_external"
)
CALIBRATED_DECIMAL_EXTERNAL_MAX_DISTANCE_PT = 128.0
CALIBRATED_DECIMAL_EXTERNAL_MAX_DISTANCE_RATIO = 4.0
CONSUMER_ALLOWED = False


def build_decimal_anchor_authority(
    decimal_quad: dict[str, Any],
    decimal_corridor: dict[str, Any],
    *,
    page_index: int,
    repeated_size_cohort_proof: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Join one strict quad to its page-level decimal scale classification."""
    if type(page_index) is not int or page_index < 0:
        return None
    if not (
        decimal_quad.get("schema_version") == DECIMAL_QUAD_SCHEMA_VERSION
        and decimal_quad.get("source") == DECIMAL_QUAD_SOURCE
        and decimal_quad.get("consumer_allowed") is False
        and decimal_corridor.get("schema_version")
        == DECIMAL_CORRIDOR_SCHEMA_VERSION
        and type(decimal_corridor.get("page_index")) is int
        and decimal_corridor.get("page_index") == page_index
        and decimal_corridor.get("source") == DECIMAL_CORRIDOR_SOURCE
        and decimal_corridor.get("consumer_allowed") is False
    ):
        return None
    quad_id = decimal_quad.get("id")
    if (
        type(quad_id) is not str
        or not quad_id
        or type(decimal_corridor.get("decimal_quad_id")) is not str
        or decimal_corridor.get("decimal_quad_id") != quad_id
    ):
        return None
    detail = decimal_quad.get("detail")
    if not isinstance(detail, dict):
        return None
    if not (
        type(detail.get("drawing_order")) is int
        and type(detail.get("item_index")) is int
        and type(decimal_corridor.get("cluster_member_count")) is int
        and type(decimal_corridor.get("page_cluster_count")) is int
        and all(
            type(value) in {int, float} and type(value) is not bool
            for value in (
                detail.get("long"),
                decimal_corridor.get("cluster_long_edge_median"),
                decimal_corridor.get("long_edge_ratio_to_primary"),
                decimal_corridor.get("within_cluster_relative_error"),
                decimal_corridor.get("nearest_cluster_relative_gap"),
                decimal_corridor.get("baseline_angle_deg"),
            )
        )
    ):
        return None
    try:
        drawing_order = int(detail["drawing_order"])
        item_index = int(detail["item_index"])
        dot_long_edge = float(detail["long"])
        cluster_long_edge = float(decimal_corridor["cluster_long_edge_median"])
        ratio_to_primary = float(decimal_corridor["long_edge_ratio_to_primary"])
        cluster_member_count = int(decimal_corridor["cluster_member_count"])
        page_cluster_count = int(decimal_corridor["page_cluster_count"])
        within_cluster_error = float(
            decimal_corridor["within_cluster_relative_error"]
        )
        nearest_cluster_gap = float(
            decimal_corridor["nearest_cluster_relative_gap"]
        )
        baseline_angle = float(decimal_corridor["baseline_angle_deg"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    size_class = decimal_corridor.get("size_class")
    cluster_id = decimal_corridor.get("char_height_cluster_id")
    membership = decimal_corridor.get("cluster_membership")
    if (
        type(size_class) is not str
        or type(cluster_id) is not str
        or type(membership) is not str
        or drawing_order < 0
        or item_index < 0
        or not cluster_id
        or size_class not in KNOWN_SIZE_CLASSES
        or cluster_member_count < 1
        or page_cluster_count < 1
        or not all(
            math.isfinite(value) and value > 0.0
            for value in (dot_long_edge, cluster_long_edge, ratio_to_primary)
        )
        or membership not in {
            "direct_member_quad_id",
            "nearest_cluster_long_edge",
        }
        or not math.isfinite(within_cluster_error)
        or within_cluster_error < 0.0
        or not math.isfinite(nearest_cluster_gap)
        or nearest_cluster_gap < 0.0
        or not math.isfinite(baseline_angle)
    ):
        return None
    page_number = page_index + 1
    authority = {
        "schema_version": SCHEMA_VERSION,
        "page_index": page_index,
        "decimal_quad_id": quad_id,
        "source_primitive_id": (
            f"p{page_number:03d}_d{drawing_order:06d}_"
            f"i{item_index:04d}_qu"
        ),
        "drawing_order": drawing_order,
        "item_index": item_index,
        "size_class": size_class,
        "char_height_cluster_id": cluster_id,
        "cluster_membership": membership,
        "cluster_member_count": cluster_member_count,
        "page_cluster_count": page_cluster_count,
        "dot_long_edge_pt": round(dot_long_edge, 6),
        "cluster_long_edge_median_pt": round(cluster_long_edge, 6),
        "long_edge_ratio_to_primary": round(ratio_to_primary, 6),
        "within_cluster_relative_error": round(within_cluster_error, 6),
        "nearest_cluster_relative_gap": round(nearest_cluster_gap, 6),
        "baseline_angle_deg": round(baseline_angle % 180.0, 6),
        "classification_source": CLASSIFICATION_SOURCE,
        "geometry_consumer_allowed": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    if repeated_size_cohort_proof is not None:
        proof = normalize_decimal_repeated_size_cohort_proof(
            repeated_size_cohort_proof
        )
        if proof is None or not _cohort_proof_matches_authority(
            proof,
            authority,
        ):
            return None
        authority["schema_version"] = SCHEMA_VERSION_V2
        authority["repeated_size_cohort_proof"] = proof
    authority["authority_digest"] = _authority_digest(authority)
    return authority


def normalize_decimal_anchor_authority(value: Any) -> dict[str, Any] | None:
    """Return the canonical authority blob, or ``None`` on any contract drift."""
    if not isinstance(value, dict):
        return None
    base_expected_keys = {
        "schema_version",
        "page_index",
        "decimal_quad_id",
        "source_primitive_id",
        "drawing_order",
        "item_index",
        "size_class",
        "char_height_cluster_id",
        "cluster_membership",
        "cluster_member_count",
        "page_cluster_count",
        "dot_long_edge_pt",
        "cluster_long_edge_median_pt",
        "long_edge_ratio_to_primary",
        "within_cluster_relative_error",
        "nearest_cluster_relative_gap",
        "baseline_angle_deg",
        "classification_source",
        "geometry_consumer_allowed",
        "consumer_allowed",
        "authority_digest",
    }
    schema_version = value.get("schema_version")
    expected_keys = (
        base_expected_keys | {"repeated_size_cohort_proof"}
        if schema_version == SCHEMA_VERSION_V2
        else base_expected_keys
    )
    if set(value) != expected_keys:
        return None
    if not (
        type(value.get("schema_version")) is str
        and type(value.get("page_index")) is int
        and type(value.get("decimal_quad_id")) is str
        and type(value.get("source_primitive_id")) is str
        and type(value.get("drawing_order")) is int
        and type(value.get("item_index")) is int
        and type(value.get("size_class")) is str
        and type(value.get("char_height_cluster_id")) is str
        and type(value.get("cluster_membership")) is str
        and type(value.get("cluster_member_count")) is int
        and type(value.get("page_cluster_count")) is int
        and all(
            type(value.get(key)) in {int, float}
            and type(value.get(key)) is not bool
            for key in (
                "dot_long_edge_pt",
                "cluster_long_edge_median_pt",
                "long_edge_ratio_to_primary",
                "within_cluster_relative_error",
                "nearest_cluster_relative_gap",
                "baseline_angle_deg",
            )
        )
        and type(value.get("classification_source")) is str
        and value.get("geometry_consumer_allowed") is False
        and value.get("consumer_allowed") is False
        and type(value.get("authority_digest")) is str
    ):
        return None
    try:
        canonical = {
            "schema_version": str(value["schema_version"]),
            "page_index": int(value["page_index"]),
            "decimal_quad_id": str(value["decimal_quad_id"]),
            "source_primitive_id": str(value["source_primitive_id"]),
            "drawing_order": int(value["drawing_order"]),
            "item_index": int(value["item_index"]),
            "size_class": str(value["size_class"]),
            "char_height_cluster_id": str(value["char_height_cluster_id"]),
            "cluster_membership": str(value["cluster_membership"]),
            "cluster_member_count": int(value["cluster_member_count"]),
            "page_cluster_count": int(value["page_cluster_count"]),
            "dot_long_edge_pt": round(float(value["dot_long_edge_pt"]), 6),
            "cluster_long_edge_median_pt": round(
                float(value["cluster_long_edge_median_pt"]),
                6,
            ),
            "long_edge_ratio_to_primary": round(
                float(value["long_edge_ratio_to_primary"]),
                6,
            ),
            "within_cluster_relative_error": round(
                float(value["within_cluster_relative_error"]),
                6,
            ),
            "nearest_cluster_relative_gap": round(
                float(value["nearest_cluster_relative_gap"]),
                6,
            ),
            "baseline_angle_deg": round(float(value["baseline_angle_deg"]) % 180.0, 6),
            "classification_source": str(value["classification_source"]),
            "geometry_consumer_allowed": value["geometry_consumer_allowed"],
            "consumer_allowed": value["consumer_allowed"],
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if schema_version == SCHEMA_VERSION_V2:
        proof = normalize_decimal_repeated_size_cohort_proof(
            value.get("repeated_size_cohort_proof")
        )
        if proof is None:
            return None
        canonical["repeated_size_cohort_proof"] = proof
    canonical["authority_digest"] = str(value["authority_digest"])
    expected_primitive_id = (
        f"p{canonical['page_index'] + 1:03d}_"
        f"d{canonical['drawing_order']:06d}_"
        f"i{canonical['item_index']:04d}_qu"
    )
    if (
        canonical["schema_version"] not in {
            SCHEMA_VERSION,
            SCHEMA_VERSION_V2,
        }
        or canonical["page_index"] < 0
        or not canonical["decimal_quad_id"]
        or canonical["source_primitive_id"] != expected_primitive_id
        or canonical["drawing_order"] < 0
        or canonical["item_index"] < 0
        or canonical["size_class"] not in KNOWN_SIZE_CLASSES
        or not canonical["char_height_cluster_id"]
        or canonical["cluster_membership"] not in {
            "direct_member_quad_id",
            "nearest_cluster_long_edge",
        }
        or canonical["cluster_member_count"] < 1
        or canonical["page_cluster_count"] < 1
        or not all(
            math.isfinite(canonical[key]) and canonical[key] > 0.0
            for key in (
                "dot_long_edge_pt",
                "cluster_long_edge_median_pt",
                "long_edge_ratio_to_primary",
            )
        )
        or not math.isfinite(canonical["within_cluster_relative_error"])
        or canonical["within_cluster_relative_error"] < 0.0
        or not math.isfinite(canonical["nearest_cluster_relative_gap"])
        or canonical["nearest_cluster_relative_gap"] < 0.0
        or not math.isfinite(canonical["baseline_angle_deg"])
        or canonical["classification_source"] != CLASSIFICATION_SOURCE
        or canonical["geometry_consumer_allowed"] is not False
        or canonical["consumer_allowed"] is not False
        or canonical["authority_digest"] != _authority_digest(
            {
                key: item
                for key, item in canonical.items()
                if key != "authority_digest"
            }
        )
    ):
        return None
    if canonical["schema_version"] == SCHEMA_VERSION_V2:
        if not _cohort_proof_matches_authority(
            canonical["repeated_size_cohort_proof"],
            canonical,
        ):
            return None
    return canonical


def is_authoritative_decimal_anchor(value: Any) -> bool:
    normalized = normalize_decimal_anchor_authority(value)
    c4_passed = bool(
        normalized is not None
        and (
            normalized["within_cluster_relative_error"] <= 0.15
            or (
                normalized["schema_version"] == SCHEMA_VERSION_V2
                and normalized.get("repeated_size_cohort_proof", {}).get(
                    "schema_version"
                )
                == REPEATED_SIZE_COHORT_SCHEMA_VERSION
            )
        )
    )
    return bool(
        normalized is not None
        and normalized["size_class"] in AUTHORITATIVE_SIZE_CLASSES
        and normalized["cluster_membership"] == "direct_member_quad_id"
        and normalized["cluster_member_count"] >= 3
        and c4_passed
        and (
            normalized["size_class"] != "tolerance"
            or (
                normalized["page_cluster_count"] >= 2
                and normalized["nearest_cluster_relative_gap"] >= 0.20
            )
        )
    )


def _cohort_proof_matches_authority(
    proof: dict[str, Any],
    authority: dict[str, Any],
) -> bool:
    """Bind one deeply normalized proof to every relevant authority field."""
    return bool(
        proof["page_index"] == authority["page_index"]
        and proof["decimal_quad_id"] == authority["decimal_quad_id"]
        and proof["char_height_cluster_id"]
        == authority["char_height_cluster_id"]
        and proof["size_class"] == authority["size_class"] == "tolerance"
        and proof["cluster_membership"]
        == authority["cluster_membership"]
        == "direct_member_quad_id"
        and proof["target_long_edge_pt"] == authority["dot_long_edge_pt"]
        and proof["parent_member_count"]
        == authority["cluster_member_count"]
        and authority["cluster_member_count"] >= 3
        and authority["within_cluster_relative_error"] > 0.15
        and authority["page_cluster_count"] >= 2
        and authority["nearest_cluster_relative_gap"] >= 0.20
    )


def calibrated_decimal_external_view_authority_evidence(
    ownership: Any,
    source_candidate_evidence: Any,
) -> dict[str, Any]:
    """Bind an external-view grant to one audited decimal authority.

    This is shared by the quality gate and hypothesis consumer so neither can
    accept a syntactically valid but unrelated digest.  It intentionally does
    not prove span ownership; the quality gate owns that separate check.
    """
    method = (
        ownership.get("method")
        if isinstance(ownership, dict)
        else None
    )
    required = method in {
        CALIBRATED_DECIMAL_EXTERNAL_METHOD,
        CALIBRATED_DECIMAL_EXTERNAL_GROUP_METHOD,
    }
    if not required:
        return {
            "schema_version": "vector_calibrated_decimal_external_authority_v1",
            "required": False,
            "passed": True,
            "reasons": [],
            "authority_count": 0,
            "invalid_authority_count": 0,
            "matched_authority": None,
            "consumer_allowed": CONSUMER_ALLOWED,
        }

    reasons: list[str] = []
    authorities_by_digest: dict[str, dict[str, Any]] = {}
    digests_by_primitive: dict[str, set[str]] = {}
    invalid_authority_count = 0
    if not isinstance(source_candidate_evidence, list):
        source_candidate_evidence = []
        reasons.append("source_candidate_evidence_missing")
    for source in source_candidate_evidence:
        if not isinstance(source, dict) or "decimal_anchor_authority" not in source:
            continue
        if source.get("decimal_anchor_authority_source") != "dot_reference":
            invalid_authority_count += 1
            continue
        authority = normalize_decimal_anchor_authority(
            source.get("decimal_anchor_authority")
        )
        if authority is None or not is_authoritative_decimal_anchor(authority):
            invalid_authority_count += 1
            continue
        source_page_index = source.get("page_index")
        if (
            source_page_index is not None
            and (
                type(source_page_index) is not int
                or source_page_index != authority["page_index"]
            )
        ):
            invalid_authority_count += 1
            continue
        digest = authority["authority_digest"]
        authorities_by_digest[digest] = authority
        digests_by_primitive.setdefault(
            authority["source_primitive_id"],
            set(),
        ).add(digest)
    if invalid_authority_count:
        reasons.append("source_decimal_authority_invalid")
    if any(len(digests) > 1 for digests in digests_by_primitive.values()):
        reasons.append("source_decimal_authority_conflict")

    if method == CALIBRATED_DECIMAL_EXTERNAL_METHOD:
        external_members = [ownership] if isinstance(ownership, dict) else []
    else:
        external_members = (
            ownership.get("external_members")
            if isinstance(ownership, dict)
            else None
        )
        if not isinstance(external_members, list) or not external_members:
            external_members = []
            reasons.append("external_group_members_invalid")

    ownership_digests: list[str] = []
    matched_authorities: list[dict[str, Any]] = []
    for member in external_members:
        if not isinstance(member, dict):
            reasons.append("external_group_member_invalid")
            continue
        digest = member.get("authority_digest")
        if not _sha256_digest(digest):
            reasons.append("ownership_authority_digest_invalid")
        else:
            ownership_digests.append(digest)
            authority = authorities_by_digest.get(digest)
            if authority is None:
                reasons.append("ownership_authority_digest_unbound")
            else:
                matched_authorities.append(authority)
        reasons.extend(_external_member_distance_reasons(member))

    ownership_digests = sorted(set(ownership_digests))
    matched_authorities = [
        authorities_by_digest[digest]
        for digest in ownership_digests
        if digest in authorities_by_digest
    ]
    # The per-member authority binding and distance checks above are the
    # evidence. The producer's digest list and count only summarize those
    # same rows and cannot invalidate otherwise safe external geometry.

    return {
        "schema_version": "vector_calibrated_decimal_external_authority_v1",
        "required": True,
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "authority_count": len(authorities_by_digest),
        "invalid_authority_count": invalid_authority_count,
        "ownership_authority_digests": ownership_digests,
        "matched_authorities": matched_authorities,
        "matched_authority": (
            matched_authorities[0]
            if len(matched_authorities) == 1
            else None
        ),
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _external_member_distance_reasons(
    member: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if member.get("consumer_allowed") is not False:
        reasons.append("external_member_consumer_flag_invalid")
    distance = _strict_positive_float(member.get("distance_pt"))
    span = _strict_positive_float(member.get("max_distance_pt"))
    ratio = _strict_positive_float(member.get("distance_ratio"))
    max_ratio = _strict_positive_float(member.get("max_distance_ratio"))
    view_candidate_count = member.get("view_candidate_count")
    if (
        distance is None
        or span is None
        or ratio is None
        or max_ratio != CALIBRATED_DECIMAL_EXTERNAL_MAX_DISTANCE_RATIO
        or not (1.0 < ratio <= CALIBRATED_DECIMAL_EXTERNAL_MAX_DISTANCE_RATIO)
        or distance > CALIBRATED_DECIMAL_EXTERNAL_MAX_DISTANCE_PT
        or not math.isclose(
            ratio,
            distance / span,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        reasons.append("external_distance_contract_invalid")
    second_gap = member.get("second_distance_gap_pt")
    if type(view_candidate_count) is not int or view_candidate_count < 1:
        reasons.append("external_view_candidate_count_invalid")
    elif view_candidate_count == 1:
        if second_gap is not None:
            reasons.append("external_second_view_gap_invalid")
    elif second_gap is None:
        reasons.append("external_second_view_gap_invalid")
    else:
        parsed_gap = _strict_positive_float(second_gap)
        if (
            parsed_gap is None
            or span is None
            or parsed_gap + 1e-6 < span
        ):
            reasons.append("external_second_view_gap_invalid")
    return reasons


def _strict_positive_float(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _sha256_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _authority_digest(value: dict[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items() if key != "authority_digest"
    }
    return hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
