"""Recursive safety checks shared by strict pure-vector dump stages."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any


_NON_ASCII_IDENTIFIER_PATTERN = re.compile(r"[^a-z0-9]+")


@lru_cache(maxsize=2048)
def _compact_identifier_text(value: str) -> str:
    return _NON_ASCII_IDENTIFIER_PATTERN.sub("", value.lower())


def _compact_identifier(value: Any) -> str:
    return _compact_identifier_text(str(value))


OCR_ARTIFACT_KEYS = frozenset({
    "ocr_results",
    "candidate_guided_ocr_results",
    "hires_results",
    "strip_results",
    "gdt_comp_results",
    "capsule_ocr_results",
    "dimline_ocr_results",
    "angle_label_ocr_results",
    "radius_label_ocr_results",
})
ZERO_RUNTIME_COUNT_KEYS = frozenset({
    "initialization_count",
    "singleton_cache_delta_count",
    "call_count",
    "retry_count",
    "artifact_count",
})
EMPTY_RUNTIME_LIST_KEYS = frozenset({
    "initialization_entrypoints",
    "called_entrypoints",
})
COMPACT_OCR_ARTIFACT_KEYS = frozenset(
    _compact_identifier(value) for value in OCR_ARTIFACT_KEYS
)
COMPACT_ZERO_RUNTIME_COUNT_KEYS = frozenset(
    _compact_identifier(value) for value in ZERO_RUNTIME_COUNT_KEYS
)
COMPACT_EMPTY_RUNTIME_LIST_KEYS = frozenset(
    _compact_identifier(value) for value in EMPTY_RUNTIME_LIST_KEYS
)
FORBIDDEN_RECOGNITION_BACKENDS = frozenset({
    "ocr",
    "opticalcharacterrecognition",
    "paddleocr",
    "rapidocr",
    "tesseract",
    "directedocr",
    "fullpageocr",
    "yolochar",
    "vlm",
})
BACKEND_DESCRIPTOR_KEYS = frozenset({
    "backend",
    "dependency",
    "engine",
    "mode",
    "model",
    "producer",
    "reader",
    "recognizer",
    "recognition_backend",
    "runtime",
    "source",
})
BACKEND_DESCRIPTOR_KEY_ALIASES = frozenset({
    "backendname",
    "provider",
})
COMPACT_BACKEND_DESCRIPTOR_KEYS = frozenset({
    "backend",
    "backendname",
    "dependency",
    "engine",
    "mode",
    "model",
    "producer",
    "provider",
    "reader",
    "recognizer",
    "recognitionbackend",
    "runtime",
    "source",
})
DUMP_ONLY_FALSE_FLAG_KEYS = frozenset({
    "feedsdisplayl3",
    "feedsdisplayl4",
    "feedsdisplayl5",
    "eligibleforl3assembly",
    "l3shadoweligible",
    "releaseallowed",
    "recognitionqualityclaimallowed",
    "finalapproved",
    "consumerapproved",
})
DUMP_ONLY_ZERO_COUNT_KEYS = frozenset({
    "consumerallowedcount",
    "feedsdisplayl3count",
    "feedsdisplayl4count",
    "feedsdisplayl5count",
    "releaseallowedcount",
    "releasereadycount",
    "recognitionqualityclaimallowedcount",
})
FORBIDDEN_RELEASE_ARTIFACT_KEYS = frozenset({
    "releasegate",
    "l5highconf",
    "l5releaseready",
    "finaldimensions",
})
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
MAX_JSON_NESTING_DEPTH = 128


def vector_artifact_safety_reasons(
    value: Any,
    *,
    path: str = "root",
) -> list[str]:
    """Reject recursive OCR, L3, or consumer evidence in dump-only artifacts."""
    reasons: list[str] = []
    try:
        _visit(
            value,
            path=path,
            parent_key=None,
            reasons=reasons,
            ancestor_container_ids=set(),
            depth=0,
            descriptor_context=False,
        )
    except RecursionError:
        reasons.append(f"nesting_recursion_error:{path}")
    except RuntimeError:
        reasons.append(f"artifact_mutated_during_scan:{path}")
    return sorted(set(reasons))


def _visit(
    value: Any,
    *,
    path: str,
    parent_key: str | None,
    reasons: list[str],
    ancestor_container_ids: set[int],
    depth: int,
    descriptor_context: bool,
) -> None:
    if depth > MAX_JSON_NESTING_DEPTH:
        reasons.append(f"nesting_too_deep:{path}")
        return
    if isinstance(value, dict):
        container_id = id(value)
        if container_id in ancestor_container_ids:
            reasons.append(f"cyclic_container:{path}")
            return
        next_ancestors = ancestor_container_ids | {container_id}
        for key, child in value.items():
            if not isinstance(key, str):
                reasons.append(f"non_string_object_key:{path}")
            elif not _utf8_encodable(key):
                reasons.append(f"invalid_unicode_key:{path}")
            child_path = f"{path}.{key}"
            compact_key = _compact_identifier(key)
            compact_parent_key = _compact_identifier(parent_key or "")
            child_descriptor_context = bool(
                descriptor_context
                or compact_key in COMPACT_BACKEND_DESCRIPTOR_KEYS
            )
            if compact_key == "consumerallowed" and child is not False:
                reasons.append(f"consumer_allowed_nonfalse:{child_path}")
            elif (
                compact_key in DUMP_ONLY_FALSE_FLAG_KEYS
                and child is not False
            ):
                reasons.append(f"dump_only_flag_nonfalse:{child_path}")
            elif compact_key in DUMP_ONLY_ZERO_COUNT_KEYS and not (
                type(child) is int and child == 0
            ):
                reasons.append(f"dump_only_count_nonzero:{child_path}")
            elif (
                compact_key in FORBIDDEN_RELEASE_ARTIFACT_KEYS
                and not _empty_artifact(child)
            ):
                reasons.append(f"release_artifact_nonempty:{child_path}")
            elif compact_key == "ocrcalled" and child is not False:
                reasons.append(f"ocr_called_nonfalse:{child_path}")
            elif compact_key in COMPACT_ZERO_RUNTIME_COUNT_KEYS and not (
                type(child) is int and child == 0
            ):
                reasons.append(f"runtime_count_nonzero:{child_path}")
            elif (
                compact_key in COMPACT_EMPTY_RUNTIME_LIST_KEYS
                and child != []
            ):
                reasons.append(f"runtime_entrypoints_nonempty:{child_path}")
            elif compact_key == "artifactcounts" and not _zero_count_mapping(child):
                reasons.append(f"ocr_artifact_counts_nonzero:{child_path}")
            elif (
                compact_parent_key != "artifactcounts"
                and (
                    compact_key in COMPACT_OCR_ARTIFACT_KEYS
                    or compact_key.endswith("ocrresults")
                )
                and not _empty_artifact(child)
            ):
                reasons.append(f"ocr_artifact_nonempty:{child_path}")
            elif (
                "ocr" in compact_key
                and not (
                    compact_parent_key == "fontnames"
                    and type(child) is int
                    and child >= 0
                )
                and not _zero_or_empty_ocr_value(child)
            ):
                reasons.append(f"ocr_field_nonzero:{child_path}")
            _visit(
                child,
                path=child_path,
                parent_key=key,
                reasons=reasons,
                ancestor_container_ids=next_ancestors,
                depth=depth + 1,
                descriptor_context=child_descriptor_context,
            )
    elif isinstance(value, list):
        container_id = id(value)
        if container_id in ancestor_container_ids:
            reasons.append(f"cyclic_container:{path}")
            return
        next_ancestors = ancestor_container_ids | {container_id}
        for index, child in enumerate(value):
            _visit(
                child,
                path=f"{path}[{index}]",
                parent_key=parent_key,
                reasons=reasons,
                ancestor_container_ids=next_ancestors,
                depth=depth + 1,
                descriptor_context=descriptor_context,
            )
    elif isinstance(value, str):
        if not _utf8_encodable(value):
            reasons.append(f"invalid_unicode_scalar:{path}")
            return
        compact = _NON_ASCII_IDENTIFIER_PATTERN.sub("", value.lower())
        descriptor_key = _compact_identifier(parent_key or "")
        if (
            descriptor_context
            or descriptor_key in COMPACT_BACKEND_DESCRIPTOR_KEYS
        ) and any(
            name in compact for name in FORBIDDEN_RECOGNITION_BACKENDS
        ):
            reasons.append(f"forbidden_recognition_backend:{path}")
    elif type(value) is float and not math.isfinite(value):
        reasons.append(f"nonfinite_numeric:{path}")
    elif type(value) is int and abs(value) > MAX_SAFE_JSON_INTEGER:
        reasons.append(f"integer_out_of_safe_range:{path}")
    elif value is None or type(value) in {bool, int, float}:
        return
    else:
        reasons.append(f"non_json_value:{path}:{type(value).__name__}")

def _utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _zero_count_mapping(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and all(type(count) is int and count == 0 for count in value.values())
    )


def _empty_artifact(value: Any) -> bool:
    return value is None or value == [] or value == {} or value == ""


def _zero_or_empty_ocr_value(value: Any) -> bool:
    return bool(
        value is False
        or (type(value) is int and value == 0)
        or _empty_artifact(value)
    )
