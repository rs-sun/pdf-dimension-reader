"""Pure request/response contracts for the strict vector-only path."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from directional_walk.phrase_output import validate_directional_walk_dump_v1
from review_candidates_contract import (
    CHAIN_AUDIT_FIELD,
    CHAIN_DUMP_FIELDS,
    CHAIN_FIELD_KEYS,
    ROOT_KEY,
    validate_review_candidates_v1,
)


OCR_ARTIFACT_KEYS = (
    "ocr_results",
    "candidate_guided_ocr_results",
    "hires_results",
    "strip_results",
    "gdt_comp_results",
    "capsule_ocr_results",
    "dimline_ocr_results",
    "angle_label_ocr_results",
    "radius_label_ocr_results",
)
OCR_ARTIFACT_KEY_SET = frozenset(OCR_ARTIFACT_KEYS)
OCR_RUNTIME_INSTRUMENTATION = (
    "forbidden_pipeline_ocr_entrypoints_engine_acquisition_and_singleton_delta"
)
STRICT_RECURSIVE_DENIAL_KEYS = frozenset({
    "consumer_allowed",
    "display_allowed",
    "release_allowed",
    "feeds_display_l3",
})
STRICT_SUCCESS_CONTRACT_CORE_SCHEMA_VERSION = (
    "sealed_success_contract_core_v1"
)
STRICT_SUCCESS_VALIDATION_SESSION_AUDIT_SCHEMA_VERSION = (
    "strict_success_validation_session_audit_v1"
)


class StrictSuccessValidationSession:
    """Request-local state shared by the two strict success seams."""

    __slots__ = (
        "_core_cache",
        "_core_cache_hit_count",
        "_core_cache_miss_count",
        "_core_validation_count",
        "_whole_envelope_scan_count",
    )

    def __init__(self) -> None:
        self._core_cache: dict[str, tuple[str, ...]] = {}
        self._core_cache_hit_count = 0
        self._core_cache_miss_count = 0
        self._core_validation_count = 0
        self._whole_envelope_scan_count = 0

    def _resolve_core(
        self,
        digest: str | None,
        validate: Any,
    ) -> tuple[str, ...]:
        if digest is not None and digest in self._core_cache:
            self._core_cache_hit_count += 1
            return self._core_cache[digest]
        self._core_cache_miss_count += 1
        self._core_validation_count += 1
        reasons = tuple(validate())
        if digest is not None:
            self._core_cache[digest] = reasons
        return reasons

    def _note_whole_envelope_scan(self) -> None:
        self._whole_envelope_scan_count += 1

    def audit(self) -> dict[str, Any]:
        """Return immutable-by-copy request-local cache/guard counters."""
        return {
            "schema_version": (
                STRICT_SUCCESS_VALIDATION_SESSION_AUDIT_SCHEMA_VERSION
            ),
            "core_cache_hit_count": self._core_cache_hit_count,
            "core_cache_miss_count": self._core_cache_miss_count,
            "core_validation_count": self._core_validation_count,
            "whole_envelope_scan_count": self._whole_envelope_scan_count,
        }


class VectorOnlyStageUnavailable(RuntimeError):
    """A required pure-vector producer failed and must not look like no data."""

    def __init__(
        self,
        *,
        reason: str,
        stage: str,
        trace_id: str,
        page_index: int,
    ) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.stage = str(stage)
        self.trace_id = str(trace_id)
        self.page_index = int(page_index)

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "vector_only_error_v1",
            "code": "vector_only_stage_unavailable",
            "status": "fail_closed",
            "reasons": [self.reason],
            "stage": self.stage,
            "trace_id": self.trace_id,
            "page_index": self.page_index,
            "fallback": False,
            "feeds_display_l3": False,
            "consumer_allowed": False,
        }


def zero_ocr_breakdown() -> dict[str, int]:
    return {
        "base": 0,
        "hires": 0,
        "strip": 0,
        "gdt_comp": 0,
        "capsule": 0,
        "dimline": 0,
        "angle_label": 0,
        "radius_label": 0,
        "candidate_guided": 0,
        "phase2_90": 0,
        "phase2_270": 0,
        "total": 0,
    }


def zero_ocr_runtime_audit() -> dict[str, Any]:
    return {
        "schema_version": "r32_ocr_runtime_audit_v1",
        "initialization_count": 0,
        "initialization_entrypoints": [],
        "singleton_cache_delta_count": 0,
        "process_global_singleton_cache_delta_observed": 0,
        "call_count": 0,
        "called_entrypoints": [],
        "retry_count": 0,
        "artifact_count": 0,
        "artifact_counts": {key: 0 for key in OCR_ARTIFACT_KEYS},
        "instrumentation": OCR_RUNTIME_INSTRUMENTATION,
        "consumer_allowed": False,
    }


def zero_consumer_runtime_audit() -> dict[str, Any]:
    return {
        "schema_version": "r32_consumer_runtime_audit_v1",
        "call_count": 0,
        "called_entrypoints": [],
        "consumer_allowed_count": 0,
        "consumer_allowed": False,
    }


def _sealed_success_contract_core(
    payload: dict[str, Any],
    *,
    expected_trace_id: str,
    expected_page_index: int,
) -> dict[str, Any]:
    direct_fields = (
        "dimensions",
        "ocr_pass_counters",
        "consumer_allowed",
        "display_allowed",
        "release_allowed",
        "view_boxes",
        "strict_yolo_view_status_v1",
        "view_seg_status_v1",
        "r92_directional_walk_dump_v1",
        "r33_m1_phrase_chain_audit_v1",
        "r32_phrase_runtime_audit_v1",
        ROOT_KEY,
        "ocr_breakdown",
        "r32_ocr_runtime_audit_v1",
        "r32_consumer_runtime_audit_v1",
    )
    alias_present = "review_candidates" in payload
    return {
        "schema_version": STRICT_SUCCESS_CONTRACT_CORE_SCHEMA_VERSION,
        "expected_trace_id": expected_trace_id,
        "expected_page_index": expected_page_index,
        "fields": {
            field: payload.get(field)
            for field in direct_fields
        },
        # Presence is contract-significant for the forbidden legacy alias.
        "review_candidates_alias": {
            "present": alias_present,
            "value": payload.get("review_candidates") if alias_present else None,
        },
        # Preserve the validator's existing missing-is-empty semantics while
        # hashing the complete normalized value that it actually validates.
        "ocr_artifacts": {
            key: payload.get(key, [])
            for key in OCR_ARTIFACT_KEYS
        },
    }


def _exact_json_tree_sha256(value: Any) -> str | None:
    """Hash an exact JSON tree, or fail closed by disabling cache reuse.

    The type walk prevents JSON's list/tuple and scalar-subclass coercions from
    making distinct Python contract inputs share a cache key.  A cache miss is
    always safe, so unsupported/cyclic values return ``None`` and are fully
    revalidated.
    """
    stack: list[tuple[Any, bool]] = [(value, False)]
    active_containers: set[int] = set()
    while stack:
        current, exiting = stack.pop()
        current_type = type(current)
        if current_type in {dict, list}:
            identity = id(current)
            if exiting:
                active_containers.remove(identity)
                continue
            if identity in active_containers:
                return None
            active_containers.add(identity)
            stack.append((current, True))
            if current_type is dict:
                if any(type(key) is not str for key in current):
                    return None
                stack.extend((child, False) for child in current.values())
            else:
                stack.extend((child, False) for child in current)
            continue
        if current is None or current_type in {bool, int, str}:
            continue
        if current_type is float and math.isfinite(current):
            continue
        return None
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _validate_vector_only_success_core(
    payload: dict[str, Any],
    *,
    expected_trace_id: str,
    expected_page_index: int,
) -> list[str]:
    """Validate direct strict fields; recursive whole-envelope guards are separate."""
    reasons: list[str] = []
    if payload.get("dimensions") != []:
        reasons.append("dimensions_nonempty_dump_only")
    if payload.get("ocr_pass_counters") != {}:
        reasons.append("ocr_pass_counters_nonzero")
    for field in ("consumer_allowed", "display_allowed", "release_allowed"):
        if payload.get(field) is not False:
            reasons.append(f"{field}_top_level_not_false")

    try:
        from yolo_view_async import (
            StrictYoloViewResult,
            StrictYoloViewUnavailable,
            resolve_strict_yolo_view_source,
        )

        strict_result = StrictYoloViewResult(
            view_boxes=tuple(payload.get("view_boxes") or ()),
            status=dict(payload.get("strict_yolo_view_status_v1") or {}),
        )
        resolved_boxes, resolved_status, _ = resolve_strict_yolo_view_source(
            strict_result,
            trace_id=expected_trace_id,
            page_index=expected_page_index,
        )
        if resolved_boxes != payload.get("view_boxes"):
            reasons.append("strict_yolo_view_boxes_mismatch")
        provided_status = payload.get("view_seg_status_v1")
        status_keys = (
            "schema_version",
            "status",
            "backend",
            "requested_backend",
            "effective_backend",
            "fallback",
            "source",
            "model_exists",
            "view_count",
            "page_index",
            "trace_id",
            "model_sha256",
            "model_sha256_verified",
        )
        if not isinstance(provided_status, dict) or any(
            provided_status.get(key) != resolved_status.get(key)
            for key in status_keys
        ):
            reasons.append("view_seg_status_mismatch")
    except (StrictYoloViewUnavailable, TypeError, ValueError):
        reasons.append("strict_yolo_contract_invalid")
    reasons.extend(_validate_live_phrase_contract(
        payload,
        expected_trace_id=expected_trace_id,
        expected_page_index=expected_page_index,
    ))
    review_candidates = payload.get(ROOT_KEY)
    if validate_review_candidates_v1(review_candidates):
        reasons.append("review_candidates_v1_invalid")
    elif review_candidates.get("trace_id") != expected_trace_id:
        reasons.append("review_candidates_v1_trace_mismatch")
    elif review_candidates.get("page_index") != expected_page_index:
        reasons.append("review_candidates_v1_page_mismatch")
    if "review_candidates" in payload:
        reasons.append("review_candidates_alias_forbidden")
    artifact_counts = {
        key: len(value) if isinstance(value := payload.get(key, []), list) else -1
        for key in OCR_ARTIFACT_KEYS
    }
    if any(count != 0 for count in artifact_counts.values()):
        reasons.append("ocr_artifact_nonzero")

    breakdown = payload.get("ocr_breakdown")
    if not isinstance(breakdown, dict) or any(
        _strict_nonnegative_int(value) != 0 for value in breakdown.values()
    ) or _strict_nonnegative_int(breakdown.get("total")) != 0:
        reasons.append("ocr_breakdown_nonzero")

    runtime = payload.get("r32_ocr_runtime_audit_v1")
    runtime_valid = isinstance(runtime, dict) and _valid_ocr_runtime_audit(runtime)
    if not runtime_valid:
        reasons.append("ocr_runtime_audit_invalid")
    else:
        if any(
            runtime[key] != 0
            for key in (
                "initialization_count",
                "singleton_cache_delta_count",
                "process_global_singleton_cache_delta_observed",
                "call_count",
                "retry_count",
            )
        ):
            reasons.append("ocr_runtime_nonzero")
        expected_artifact_count = sum(max(0, count) for count in artifact_counts.values())
        if (
            runtime.get("artifact_count") != expected_artifact_count
            or runtime.get("artifact_counts") != artifact_counts
        ):
            reasons.append("ocr_runtime_artifact_mismatch")

    consumer = payload.get("r32_consumer_runtime_audit_v1")
    if isinstance(consumer, dict) and (
        (_strict_nonnegative_int(consumer.get("call_count")) or 0) != 0
        or (_strict_nonnegative_int(consumer.get("consumer_allowed_count")) or 0)
        != 0
    ):
        reasons.append("consumer_runtime_nonzero")
    if not isinstance(consumer, dict) or not _valid_consumer_runtime_audit(consumer):
        reasons.append("consumer_runtime_invalid")
    return reasons


def validate_vector_only_success_payload(
    payload: Any,
    *,
    expected_trace_id: str,
    expected_page_index: int,
    validation_session: StrictSuccessValidationSession | None = None,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["vector_only_payload_invalid"]
    if validation_session is not None:
        validation_session._note_whole_envelope_scan()
        sealed_core = _sealed_success_contract_core(
            payload,
            expected_trace_id=expected_trace_id,
            expected_page_index=expected_page_index,
        )
        core_digest = _exact_json_tree_sha256(sealed_core)
        direct_reasons = validation_session._resolve_core(
            core_digest,
            lambda: _validate_vector_only_success_core(
                payload,
                expected_trace_id=expected_trace_id,
                expected_page_index=expected_page_index,
            ),
        )
    else:
        direct_reasons = tuple(_validate_vector_only_success_core(
            payload,
            expected_trace_id=expected_trace_id,
            expected_page_index=expected_page_index,
        ))
    reasons = list(direct_reasons)
    recursive_artifact_count, recursive_denial_counts = (
        _scan_recursive_success_guards(payload)
    )
    if recursive_artifact_count != 0:
        reasons.append("ocr_artifact_nonzero")
    for key in STRICT_RECURSIVE_DENIAL_KEYS:
        if recursive_denial_counts[key]:
            reasons.append(f"{key}_recursive_nonzero")
    return sorted(set(reasons))


def validate_r33_m1_review_api_projection(
    payload: Any,
    *,
    expected_trace_id: str,
    expected_page_index: int,
) -> tuple[str, list[str]]:
    """Validate the unchanged pipeline-to-HTTP review projection.

    Returns ``("ok", [])``, ``("unavailable", [reason])``, or
    ``("violation", reasons)``.  This function validates the public envelope
    and raw-stage safety; it never replays accounting, repairs, aliases, or
    converts a candidate envelope.
    """
    if not isinstance(payload, dict):
        return "violation", ["review_projection_payload_invalid"]
    if "review_candidates" in payload:
        return "violation", ["review_candidates_alias_forbidden"]
    envelope = payload.get(ROOT_KEY)
    schema_reasons = validate_review_candidates_v1(envelope)
    if schema_reasons:
        return "violation", [
            f"review_candidates_v1:{reason}" for reason in schema_reasons
        ]
    if envelope.get("trace_id") != expected_trace_id:
        return "violation", ["review_candidates_v1_trace_mismatch"]
    if envelope.get("page_index") != expected_page_index:
        return "violation", ["review_candidates_v1_page_mismatch"]

    review_audit = envelope["audit"]
    chain_audit = payload.get(CHAIN_AUDIT_FIELD)
    if not isinstance(chain_audit, dict):
        return "violation", ["r33_m1_phrase_chain_audit_missing"]
    identity_reasons = _r33_projection_identity_reasons(
        chain_audit,
        trace_id=expected_trace_id,
        page_index=expected_page_index,
    )
    if identity_reasons:
        return "violation", identity_reasons
    strict_ocr_reasons = _r33_projection_ocr_reasons(
        payload,
        review_audit=review_audit,
        chain_audit=chain_audit,
    )
    if strict_ocr_reasons:
        return "violation", strict_ocr_reasons

    review_status = review_audit.get("status")
    chain_status = chain_audit.get("status")
    if review_status == "unavailable" or chain_status in {
        "unavailable",
        "fail_closed",
    }:
        reasons: list[str] = []
        if review_status != "unavailable":
            reasons.append("review_status_not_unavailable")
        if chain_status not in {"unavailable", "fail_closed"}:
            reasons.append("chain_status_not_unavailable")
        chain_reason = chain_audit.get("reason")
        if not isinstance(chain_reason, str) or not chain_reason:
            reasons.append("chain_unavailable_reason_missing")
        if review_audit.get("unavailable_reason") != chain_reason:
            reasons.append("review_chain_unavailable_reason_mismatch")
        if envelope.get("items") != []:
            reasons.append("unavailable_review_items_nonempty")
        if reasons:
            return "violation", sorted(set(reasons))
        return "unavailable", [chain_reason]

    if review_status != "ok" or chain_status != "ok":
        return "violation", ["review_chain_status_invalid"]
    missing = sorted(CHAIN_FIELD_KEYS - set(payload))
    if missing:
        return "violation", [
            f"review_projection_field_missing:{field}" for field in missing
        ]

    reasons = _r33_projection_chain_reasons(
        payload,
        envelope=envelope,
        chain_audit=chain_audit,
        trace_id=expected_trace_id,
        page_index=expected_page_index,
    )
    return ("violation", reasons) if reasons else ("ok", [])


def _r33_projection_identity_reasons(
    value: dict[str, Any],
    *,
    trace_id: str,
    page_index: int,
) -> list[str]:
    reasons: list[str] = []
    if value.get("schema_version") != "r33_m1_phrase_chain_audit_v1":
        reasons.append("r33_chain_schema_invalid")
    if value.get("trace_id") != trace_id:
        reasons.append("r33_chain_trace_mismatch")
    if value.get("page_index") != page_index:
        reasons.append("r33_chain_page_mismatch")
    if value.get("dimensions_promoted_count") != 0:
        reasons.append("r33_dimensions_promoted_nonzero")
    if value.get("release_allowed") is not False:
        reasons.append("r33_release_allowed_not_false")
    if value.get("consumer_allowed") is not False:
        reasons.append("r33_consumer_allowed_not_false")
    if any(
        _recursive_non_false_count(value, key)
        for key in (
            "consumer_allowed",
            "display_allowed",
            "release_allowed",
            "formal_project_data_allowed",
        )
    ):
        reasons.append("r33_chain_authority_recursive_nonzero")
    return reasons


def _r33_projection_ocr_reasons(
    payload: dict[str, Any],
    *,
    review_audit: dict[str, Any],
    chain_audit: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    zero_keys = (
        "ocr_family_call_count",
        "paddleocr_call_count",
        "rapidocr_call_count",
        "yolo_char_call_count",
        "yolo_b_call_count",
    )
    review_runtime = review_audit.get("runtime")
    if not isinstance(review_runtime, dict) or any(
        review_runtime.get(key) != 0 for key in zero_keys
    ):
        reasons.append("review_ocr_runtime_nonzero")
    if any(
        type(chain_audit.get(key)) is not int
        or chain_audit.get(key) != 0
        for key in zero_keys
    ) or _recursive_named_counter_violation(chain_audit, frozenset(zero_keys)):
        reasons.append("r33_chain_ocr_runtime_nonzero")
    runtime = payload.get("r32_ocr_runtime_audit_v1")
    if not isinstance(runtime, dict) or not _valid_ocr_runtime_audit(runtime):
        reasons.append("strict_ocr_runtime_audit_invalid")
    elif any(runtime.get(key) != 0 for key in (
        "initialization_count",
        "singleton_cache_delta_count",
        "process_global_singleton_cache_delta_observed",
        "call_count",
        "retry_count",
        "artifact_count",
    )):
        reasons.append("strict_ocr_runtime_nonzero")
    breakdown = payload.get("ocr_breakdown")
    if not isinstance(breakdown, dict) or any(
        _strict_nonnegative_int(value) != 0 for value in breakdown.values()
    ):
        reasons.append("strict_ocr_breakdown_nonzero")
    return reasons


def _r33_projection_chain_reasons(
    payload: dict[str, Any],
    *,
    envelope: dict[str, Any],
    chain_audit: dict[str, Any],
    trace_id: str,
    page_index: int,
) -> list[str]:
    reasons: list[str] = []
    review_audit = envelope["audit"]
    if review_audit.get("template_identity") != chain_audit.get(
        "template_identity"
    ):
        reasons.append("review_template_identity_mismatch")

    dumps: dict[str, dict[str, Any]] = {}
    for stage, field in CHAIN_DUMP_FIELDS.items():
        dump = payload.get(field)
        if not isinstance(dump, dict):
            reasons.append(f"review_raw_dump_invalid:{field}")
            continue
        dumps[stage] = dump
        if dump.get("trace_id") != trace_id:
            reasons.append(f"review_raw_dump_trace_mismatch:{field}")
        if dump.get("page_index") != page_index:
            reasons.append(f"review_raw_dump_page_mismatch:{field}")
        if dump.get("status") != "ok":
            reasons.append(f"review_raw_dump_status_invalid:{field}")
        if dump.get("ocr_called") is not False:
            reasons.append(f"review_raw_dump_ocr_nonzero:{field}")
        if dump.get("consumer_allowed") is not False:
            reasons.append(f"review_raw_dump_consumer_nonzero:{field}")
    l5 = dumps.get("l5", {})
    if l5.get("release_allowed") is not False:
        reasons.append("review_l5_release_allowed_not_false")
    if l5.get("feeds_display_l5") is not False:
        reasons.append("review_l5_display_allowed_not_false")
    return sorted(set(reasons))


def _recursive_named_counter_violation(
    value: Any,
    keys: frozenset[str],
) -> bool:
    if isinstance(value, dict):
        return any(
            (
                key in keys
                and (type(child) is not int or child != 0)
            )
            or _recursive_named_counter_violation(child, keys)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _recursive_named_counter_violation(child, keys)
            for child in value
        )
    return False


def _validate_live_phrase_contract(
    payload: dict[str, Any],
    *,
    expected_trace_id: str,
    expected_page_index: int,
) -> list[str]:
    reasons: list[str] = []
    phrase = payload.get("r92_directional_walk_dump_v1")
    if "vector_phrase_region_dump_v1" in payload:
        reasons.append("legacy_phrase_region_alias_present")
    if not isinstance(phrase, dict):
        reasons.append("directional_walk_contract_invalid")
    else:
        try:
            validate_directional_walk_dump_v1(phrase)
        except (TypeError, ValueError):
            reasons.append("directional_walk_contract_invalid")
        if phrase.get("status") != "ok" or phrase.get("error") is not None:
            reasons.append("directional_walk_fail_closed")
        if phrase.get("trace_id") != expected_trace_id:
            reasons.append("directional_walk_trace_mismatch")
        if phrase.get("page_index") != expected_page_index:
            reasons.append("directional_walk_page_mismatch")
    r33_runtime = payload.get("r33_m1_phrase_chain_audit_v1")
    if r33_runtime is not None:
        if not _valid_r33_m1_phrase_chain_audit(
            r33_runtime,
            trace_id=expected_trace_id,
            page_index=expected_page_index,
        ):
            reasons.append("r33_m1_phrase_chain_audit_invalid")

    runtime = payload.get("r32_phrase_runtime_audit_v1")
    if not isinstance(runtime, dict) or not (
        runtime.get("schema_version") == "r32_phrase_runtime_audit_v1"
        and runtime.get("consumer_allowed") is False
    ):
        reasons.append("phrase_runtime_audit_invalid")
    return reasons


def _valid_r33_m1_phrase_chain_audit(
    value: Any,
    *,
    trace_id: str,
    page_index: int,
) -> bool:
    if not isinstance(value, dict):
        return False
    template_identity = value.get("template_identity")
    return bool(
        value.get("schema_version") == "r33_m1_phrase_chain_audit_v1"
        and value.get("trace_id") == trace_id
        and value.get("page_index") == page_index
        and value.get("status") in {"ok", "fail_closed", "unavailable"}
        and (
            (
                isinstance(template_identity, dict)
                and set(template_identity) == {
                    "schema_version",
                    "template_version",
                    "content_sha256",
                }
                and template_identity.get("schema_version")
                == "r33_m1_template_identity_v1"
                and type(template_identity.get("template_version")) is str
                and bool(template_identity.get("template_version"))
                and _valid_sha256(template_identity.get("content_sha256"))
            )
            or (
                template_identity is None
                and value.get("status") in {"fail_closed", "unavailable"}
            )
        )
        and value.get("ocr_family_call_count") == 0
        and value.get("paddleocr_call_count") == 0
        and value.get("rapidocr_call_count") == 0
        and value.get("yolo_char_call_count") == 0
        and value.get("yolo_b_call_count") == 0
        and value.get("dimensions_promoted_count") == 0
        and value.get("release_allowed") is False
        and value.get("consumer_allowed") is False
    )


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _valid_ocr_runtime_audit(runtime: dict[str, Any]) -> bool:
    called = runtime.get("called_entrypoints")
    initialized = runtime.get("initialization_entrypoints")
    counts = {
        key: _strict_nonnegative_int(runtime.get(key))
        for key in (
            "initialization_count",
            "singleton_cache_delta_count",
            "process_global_singleton_cache_delta_observed",
            "call_count",
            "retry_count",
            "artifact_count",
        )
    }
    return bool(
        runtime.get("schema_version") == "r32_ocr_runtime_audit_v1"
        and runtime.get("instrumentation") == OCR_RUNTIME_INSTRUMENTATION
        and runtime.get("consumer_allowed") is False
        and isinstance(called, list)
        and all(isinstance(name, str) for name in called)
        and isinstance(initialized, list)
        and all(isinstance(name, str) for name in initialized)
        and all(value is not None for value in counts.values())
        and counts["call_count"] == len(called)
        and counts["retry_count"] == called.count("run_empty_region_retry")
        and counts["initialization_count"]
        == len(initialized) + counts["singleton_cache_delta_count"]
        and counts["process_global_singleton_cache_delta_observed"]
        == counts["singleton_cache_delta_count"]
        and isinstance(runtime.get("artifact_counts"), dict)
    )


def _valid_consumer_runtime_audit(runtime: dict[str, Any]) -> bool:
    called = runtime.get("called_entrypoints")
    call_count = _strict_nonnegative_int(runtime.get("call_count"))
    allowed_count = _strict_nonnegative_int(runtime.get("consumer_allowed_count"))
    return bool(
        runtime.get("schema_version") == "r32_consumer_runtime_audit_v1"
        and runtime.get("consumer_allowed") is False
        and isinstance(called, list)
        and all(isinstance(name, str) for name in called)
        and call_count is not None
        and call_count == len(called)
        and allowed_count is not None
    )


def _strict_nonnegative_int(value: Any) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _scan_recursive_success_guards(
    value: Any,
) -> tuple[int, dict[str, int]]:
    """Evaluate all recursive strict guards in one complete tree walk."""
    artifact_count = 0
    denial_counts = {
        key: 0
        for key in STRICT_RECURSIVE_DENIAL_KEYS
    }
    stack: list[tuple[Any, str | None]] = [(value, None)]
    while stack:
        current, parent_key = stack.pop()
        if isinstance(current, dict):
            for child_key, child_value in current.items():
                if (
                    child_key in STRICT_RECURSIVE_DENIAL_KEYS
                    and child_value is not False
                ):
                    denial_counts[child_key] += 1
                if (
                    parent_key != "artifact_counts"
                    and child_key in OCR_ARTIFACT_KEY_SET
                ):
                    if isinstance(child_value, (list, tuple)):
                        artifact_count += len(child_value)
                    elif child_value not in (None, ""):
                        artifact_count += 1
                stack.append((child_value, str(child_key)))
        elif isinstance(current, (list, tuple)):
            for child in current:
                stack.append((child, parent_key))
    return artifact_count, denial_counts


def _recursive_non_false_count(value: Any, key: str) -> int:
    if isinstance(value, dict):
        return sum(
            int(child_key == key and child_value is not False)
            + _recursive_non_false_count(child_value, key)
            for child_key, child_value in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_recursive_non_false_count(child, key) for child in value)
    return 0


def _recursive_ocr_artifact_count(value: Any, parent_key: str | None = None) -> int:
    if isinstance(value, dict):
        total = 0
        for child_key, child_value in value.items():
            if parent_key != "artifact_counts" and child_key in OCR_ARTIFACT_KEYS:
                if isinstance(child_value, (list, tuple)):
                    total += len(child_value)
                elif child_value not in (None, ""):
                    total += 1
            total += _recursive_ocr_artifact_count(
                child_value,
                parent_key=str(child_key),
            )
        return total
    if isinstance(value, (list, tuple)):
        return sum(
            _recursive_ocr_artifact_count(child, parent_key=parent_key)
            for child in value
        )
    return 0


def _force_recursive_denials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                False
                if key in STRICT_RECURSIVE_DENIAL_KEYS
                else _force_recursive_denials(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_force_recursive_denials(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_force_recursive_denials(child) for child in value)
    return value


def fail_closed_payload(error_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **_force_recursive_denials(dict(error_payload)),
        "dimensions": [],
        "view_boxes": [],
        "ocr_results": [],
        "ocr_breakdown": zero_ocr_breakdown(),
        "ocr_pass_counters": {},
        "r32_ocr_runtime_audit_v1": zero_ocr_runtime_audit(),
        "r32_consumer_runtime_audit_v1": zero_consumer_runtime_audit(),
        "feeds_display_l3": False,
        "consumer_allowed": False,
        "display_allowed": False,
        "release_allowed": False,
    }
