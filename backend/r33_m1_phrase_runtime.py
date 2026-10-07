"""R33-M1 controlled runtime and same-request phrase review chain.

Runtime assets are supplied explicitly by configuration or dependency
injection. This module records only the template format/version and content digest; it does not
embed an asset path in request output and it does not make a holdout or release
quality claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vector_digit_fixed_template import load_fixed_digit_template_library
from vector_page_context import VectorPageContext
from directional_walk_pipeline import (
    build_directional_walk_pitch_projection_v1,
)
from vector_phrase_gate import build_vector_phrase_gate_dump
from vector_phrase_l3_shadow import build_vector_phrase_l3_shadow_dump
from vector_phrase_l4_shadow import build_vector_phrase_l4_shadow_dump
from vector_phrase_l5_shadow import build_vector_phrase_l5_shadow_dump
from vector_phrase_spatial_dedup import build_vector_phrase_spatial_dedup_dump
from vector_phrase_to_hypothesis import build_vector_phrase_hypothesis_dump


CHAIN_AUDIT_SCHEMA_VERSION = "r33_m1_phrase_chain_audit_v1"
TEMPLATE_IDENTITY_SCHEMA_VERSION = "r33_m1_template_identity_v1"
PINNED_TEMPLATE_THRESHOLD = 0.008
PINNED_TEMPLATE_MARGIN = 0.010
CONSUMER_ALLOWED = False


class R33M1RuntimeUnavailable(RuntimeError):
    """The explicitly configured M1 phrase runtime could not be loaded."""

    def __init__(
        self,
        reason: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason)
        self.details = dict(details or {})
        super().__init__(self.reason)


ReadOne = Callable[..., tuple[
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]]
PrepareVectorGlyphPage = Callable[..., Any]
PageGlyphShadowSink = Callable[[dict[str, Any]], None]


def read_one_fail_closed(
    *_args: Any,
    **_kwargs: Any,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep the bounded GD&T hook honest after the legacy reader is isolated."""

    return (
        "",
        [],
        [],
        {
            "schema_version": "r33_m1_gdt_reader_fail_closed_v1",
            "status": "fail_closed",
            "reason": "legacy_pitch_reader_quarantined",
            "consumer_allowed": False,
        },
    )


@dataclass(frozen=True)
class R33M1PhraseRuntime:
    """A request-safe handle to one explicitly supplied template runtime."""

    template_library: dict[str, Any]
    template_identity: dict[str, Any]
    read_one: ReadOne = read_one_fail_closed
    prepare_vector_glyph_page: PrepareVectorGlyphPage | None = None
    page_glyph_shadow_sink: PageGlyphShadowSink | None = None


_RUNTIME_CACHE_LOCK = threading.RLock()
_RUNTIME_CACHE: dict[tuple[str, int, int], R33M1PhraseRuntime] = {}
_RUNTIME_CACHE_CONDITION = threading.Condition(_RUNTIME_CACHE_LOCK)
_RUNTIME_LOADS: set[tuple[str, int, int]] = set()
_RUNTIME_FAILURE_CACHE: dict[
    tuple[str, int, int],
    tuple[str, dict[str, Any]],
] = {}


def load_r33_m1_phrase_runtime(
    template_library_path: str | Path,
    *,
    read_one: ReadOne = read_one_fail_closed,
) -> R33M1PhraseRuntime:
    """Load an explicitly configured template library with a small stat cache."""
    raw_path = str(template_library_path or "").strip()
    if not raw_path:
        raise R33M1RuntimeUnavailable("template_library_path_unconfigured")
    if not callable(read_one):
        raise R33M1RuntimeUnavailable("phrase_reader_not_callable")
    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise R33M1RuntimeUnavailable(
            "template_library_unavailable",
            details={"exception_type": type(exc).__name__},
        ) from exc
    if not resolved.is_file():
        raise R33M1RuntimeUnavailable("template_library_not_file")
    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with _RUNTIME_CACHE_CONDITION:
        while True:
            cached = _RUNTIME_CACHE.get(cache_key)
            if cached is not None and cached.read_one is read_one:
                return cached
            cached_failure = _RUNTIME_FAILURE_CACHE.get(cache_key)
            if cached_failure is not None:
                reason, details = cached_failure
                raise R33M1RuntimeUnavailable(
                    reason,
                    details=details,
                )
            if cache_key not in _RUNTIME_LOADS:
                _RUNTIME_LOADS.add(cache_key)
                break
            _RUNTIME_CACHE_CONDITION.wait()

    try:
        try:
            raw_bytes = resolved.read_bytes()
            raw_payload = json.loads(raw_bytes.decode("utf-8"))
            template_library = load_fixed_digit_template_library(resolved)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise R33M1RuntimeUnavailable(
                "template_library_invalid",
                details={"exception_type": type(exc).__name__},
            ) from exc
        if not isinstance(raw_payload, dict):
            raise R33M1RuntimeUnavailable(
                "template_library_payload_not_object"
            )
        debug = (
            template_library.get("debug")
            if isinstance(template_library, dict)
            else None
        )
        if not (
            isinstance(template_library, dict)
            and isinstance(debug, dict)
            and debug.get("status") == "loaded"
            and isinstance(template_library.get("base_templates"), list)
            and bool(template_library["base_templates"])
        ):
            raise R33M1RuntimeUnavailable(
                "template_library_has_no_runtime_templates"
            )
        template_schema_version = str(
            raw_payload.get("schema_version") or ""
        )
        summary = raw_payload.get("summary")
        summary_schema_version = str(
            summary.get("schema_version") or ""
            if isinstance(summary, dict)
            else ""
        )
        identity = {
            "schema_version": TEMPLATE_IDENTITY_SCHEMA_VERSION,
            "template_version": (
                summary_schema_version or template_schema_version
            ),
            "content_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
        runtime = R33M1PhraseRuntime(
            template_library=template_library,
            template_identity=identity,
            read_one=read_one,
        )
    except R33M1RuntimeUnavailable as exc:
        with _RUNTIME_CACHE_CONDITION:
            _RUNTIME_FAILURE_CACHE.clear()
            _RUNTIME_FAILURE_CACHE[cache_key] = (
                exc.reason,
                dict(exc.details),
            )
            _RUNTIME_LOADS.remove(cache_key)
            _RUNTIME_CACHE_CONDITION.notify_all()
        raise
    except BaseException:
        with _RUNTIME_CACHE_CONDITION:
            _RUNTIME_LOADS.discard(cache_key)
            _RUNTIME_CACHE_CONDITION.notify_all()
        raise
    with _RUNTIME_CACHE_CONDITION:
        _RUNTIME_CACHE.clear()
        _RUNTIME_CACHE[cache_key] = runtime
        _RUNTIME_FAILURE_CACHE.clear()
        _RUNTIME_LOADS.remove(cache_key)
        _RUNTIME_CACHE_CONDITION.notify_all()
    return runtime


def build_r33_m1_phrase_chain(
    *,
    phrase_dump: dict[str, Any],
    page_context: VectorPageContext,
    runtime: R33M1PhraseRuntime,
    experimental_primary_pitch_pt: float | None = None,
) -> dict[str, Any]:
    """Run reader -> gates -> hypothesis -> dedup -> L3/L4/L5 once."""
    _validate_runtime(runtime)
    # R92 has already recognized and ordered the cells.  Projecting those
    # authoritative cells is not a second reader invocation, so the ordinary
    # chain deliberately does not call ``runtime.read_one`` here.
    pitch = build_directional_walk_pitch_projection_v1(
        phrase_dump=phrase_dump,
        page_context=page_context,
    )
    quality = build_vector_phrase_gate_dump(
        phrase_dump=phrase_dump,
        pitch_read_dump=pitch,
    )
    hypothesis = build_vector_phrase_hypothesis_dump(
        phrase_dump=phrase_dump,
        pitch_read_dump=pitch,
        quality_gate_dump=quality,
    )
    dedup = build_vector_phrase_spatial_dedup_dump(
        hypothesis_dump=hypothesis,
    )
    l3 = build_vector_phrase_l3_shadow_dump(
        phrase_dump=phrase_dump,
        pitch_read_dump=pitch,
        quality_gate_dump=quality,
        hypothesis_dump=hypothesis,
        spatial_dedup_dump=dedup,
    )
    l4 = build_vector_phrase_l4_shadow_dump(
        phrase_dump=phrase_dump,
        pitch_read_dump=pitch,
        quality_gate_dump=quality,
        hypothesis_dump=hypothesis,
        spatial_dedup_dump=dedup,
        l3_shadow_dump=l3,
    )
    l5 = build_vector_phrase_l5_shadow_dump(
        phrase_dump=phrase_dump,
        pitch_read_dump=pitch,
        quality_gate_dump=quality,
        hypothesis_dump=hypothesis,
        spatial_dedup_dump=dedup,
        l3_shadow_dump=l3,
        l4_shadow_dump=l4,
    )
    stages = {
        "phrase_region": phrase_dump,
        "pitch_reader": pitch,
        "quality_gate": quality,
        "hypothesis": hypothesis,
        "spatial_dedup": dedup,
        "l3": l3,
        "l4": l4,
        "l5": l5,
    }
    audit = _build_chain_audit(
        stages=stages,
        runtime=runtime,
        page_context=page_context,
    )
    _emit_page_glyph_shadow(
        runtime=runtime,
        phrase_dump=phrase_dump,
        page_context=page_context,
        control_reads=list(pitch.get("reads") or []),
    )
    return {
        "vector_pitch_phrase_read_dump_v1": pitch,
        "vector_phrase_quality_gate_dump_v1": quality,
        "vector_phrase_hypothesis_dump_v1": hypothesis,
        "vector_phrase_spatial_dedup_dump_v1": dedup,
        "vector_phrase_l3_shadow_dump_v1": l3,
        "vector_phrase_l4_shadow_dump_v1": l4,
        "vector_phrase_l5_shadow_dump_v1": l5,
        "r33_m1_phrase_chain_audit_v1": audit,
    }


def build_r33_m1_unavailable_audit(
    *,
    trace_id: str,
    page_index: int,
    reason: str,
    details: dict[str, Any] | None = None,
    page_context: VectorPageContext | None = None,
    phrase_dump: dict[str, Any] | None = None,
    status: str = "unavailable",
) -> dict[str, Any]:
    """Return the stable audit envelope used when no runtime is configured."""
    if status not in {"unavailable", "fail_closed"}:
        raise ValueError("invalid R33-M1 unavailable audit status")
    counts = _zero_counts()
    stage_status = {
        "phrase_region": "not_run",
        "pitch_reader": "not_run",
        "quality_gate": "not_run",
        "hypothesis": "not_run",
        "spatial_dedup": "not_run",
        "l3": "not_run",
        "l4": "not_run",
        "l5": "not_run",
    }
    if page_context is not None:
        metrics = page_context.runtime_metrics()
        counts["page_context_build_count"] = 1
        counts["page_context_extraction_call_count"] = int(
            metrics.get("extraction_call_count") or 0
        )
    if isinstance(phrase_dump, dict):
        stage_status["phrase_region"] = str(
            phrase_dump.get("status") or "unknown"
        )
        counts["phrase_region_count"] = len(
            phrase_dump.get("phrases") or []
        )
    semantic_projection = {
        "reason": str(reason),
        "stage_status": stage_status,
        "counts": counts,
    }
    return {
        "schema_version": CHAIN_AUDIT_SCHEMA_VERSION,
        "trace_id": str(trace_id or ""),
        "page_index": int(page_index),
        "status": status,
        "reason": str(reason),
        "details": _json_scalar_details(details or {}),
        "template_identity": None,
        "stage_status": stage_status,
        "counts": counts,
        "semantic_projection_sha256": _digest(semantic_projection),
        "ocr_family_call_count": 0,
        "paddleocr_call_count": 0,
        "rapidocr_call_count": 0,
        "yolo_char_call_count": 0,
        "yolo_b_call_count": 0,
        "dimensions_promoted_count": 0,
        "release_allowed": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _emit_page_glyph_shadow(
    *,
    runtime: R33M1PhraseRuntime,
    phrase_dump: dict[str, Any],
    page_context: VectorPageContext,
    control_reads: list[dict[str, Any]],
) -> None:
    """The legacy page-glyph sidecar is isolated with its old region chain."""

    return


def _build_chain_audit(
    *,
    stages: dict[str, dict[str, Any]],
    runtime: R33M1PhraseRuntime,
    page_context: VectorPageContext,
) -> dict[str, Any]:
    phrase = stages["phrase_region"]
    pitch = stages["pitch_reader"]
    quality = stages["quality_gate"]
    hypothesis = stages["hypothesis"]
    dedup = stages["spatial_dedup"]
    l3 = stages["l3"]
    l4 = stages["l4"]
    l5 = stages["l5"]
    stage_status = {
        name: str(value.get("status") or "unknown")
        for name, value in stages.items()
    }
    all_ok = all(value == "ok" for value in stage_status.values())
    details = _first_failed_stage_details(stages)
    runtime_metrics = page_context.runtime_metrics()
    counts = {
        "page_context_build_count": 1,
        "page_context_extraction_call_count": int(
            runtime_metrics.get("extraction_call_count") or 0
        ),
        "phrase_region_count": len(phrase.get("phrases") or []),
        "reader_call_count": int(
            (pitch.get("stats") or {}).get("reader_call_count") or 0
        ),
        "reader_success_count": int(
            (pitch.get("stats") or {}).get("successful_read_count") or 0
        ),
        "quality_pass_count": int(
            (quality.get("stats") or {}).get("phrase_complete_count") or 0
        ),
        "ordinary_hypothesis_count": int(
            (hypothesis.get("stats") or {}).get("ordinary_candidate_count") or 0
        ),
        "dedup_winner_count": len(dedup.get("winner_hypotheses") or []),
        "l3_dimension_count": len(l3.get("L3_dimensions") or []),
        "l4_recovered_count": len(l4.get("L4_recovered") or []),
        "l5_review_required_count": len(l5.get("L5_review_required") or []),
    }
    semantic_projection = {
        "template_content_sha256": runtime.template_identity[
            "content_sha256"
        ],
        "stage_status": stage_status,
        "counts": counts,
        "pitch_reads": _stage_digest(pitch, rows_key="reads"),
        "quality": str(quality.get("semantic_digest") or ""),
        "hypothesis": str(hypothesis.get("semantic_digest") or ""),
        "dedup": str(dedup.get("semantic_digest") or ""),
        "l3": str(l3.get("semantic_digest") or ""),
        "l4": str(l4.get("semantic_digest") or ""),
        "l5": str(l5.get("semantic_digest") or ""),
    }
    return {
        "schema_version": CHAIN_AUDIT_SCHEMA_VERSION,
        "trace_id": str(phrase.get("trace_id") or ""),
        "page_index": int(phrase.get("page_index") or 0),
        "status": "ok" if all_ok else "fail_closed",
        "reason": None if all_ok else "phrase_chain_stage_failed",
        "details": details,
        "template_identity": dict(runtime.template_identity),
        "stage_status": stage_status,
        "counts": counts,
        "semantic_projection_sha256": _digest(semantic_projection),
        "ocr_family_call_count": 0,
        "paddleocr_call_count": 0,
        "rapidocr_call_count": 0,
        "yolo_char_call_count": 0,
        "yolo_b_call_count": 0,
        "dimensions_promoted_count": 0,
        "release_allowed": False,
        "consumer_allowed": CONSUMER_ALLOWED,
    }


def _first_failed_stage_details(
    stages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for stage_name, stage in stages.items():
        if str(stage.get("status") or "unknown") == "ok":
            continue
        error = stage.get("error")
        raw_reasons = (
            error.get("reasons")
            if isinstance(error, dict)
            else stage.get("reasons")
        )
        reasons = [
            str(reason)
            for reason in raw_reasons or []
            if type(reason) is str and reason
        ] if isinstance(raw_reasons, list) else []
        details: dict[str, Any] = {
            "first_failed_stage": str(stage_name),
        }
        if isinstance(error, dict):
            details["error"] = copy.deepcopy(error)
        if reasons:
            details["reasons"] = reasons
        return details
    return {}


def _validate_runtime(runtime: Any) -> None:
    if not isinstance(runtime, R33M1PhraseRuntime):
        raise R33M1RuntimeUnavailable("phrase_runtime_type_invalid")
    identity = runtime.template_identity
    if not (
        isinstance(runtime.template_library, dict)
        and isinstance(identity, dict)
        and set(identity) == {
            "schema_version",
            "template_version",
            "content_sha256",
        }
        and identity.get("schema_version") == TEMPLATE_IDENTITY_SCHEMA_VERSION
        and _valid_sha256(identity.get("content_sha256"))
        and type(identity.get("template_version")) is str
        and bool(identity["template_version"])
        and callable(runtime.read_one)
    ):
        raise R33M1RuntimeUnavailable("phrase_runtime_identity_invalid")


def _stage_digest(value: dict[str, Any], *, rows_key: str) -> str:
    rows = value.get(rows_key)
    return _digest(rows if isinstance(rows, list) else [])


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_scalar_details(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): child
        for key, child in value.items()
        if (
            child is None
            or type(child) in (str, bool, int)
            or (type(child) is float and math.isfinite(child))
        )
    }


def _zero_counts() -> dict[str, int]:
    return {
        "page_context_build_count": 0,
        "page_context_extraction_call_count": 0,
        "phrase_region_count": 0,
        "reader_call_count": 0,
        "reader_success_count": 0,
        "quality_pass_count": 0,
        "ordinary_hypothesis_count": 0,
        "dedup_winner_count": 0,
        "l3_dimension_count": 0,
        "l4_recovered_count": 0,
        "l5_review_required_count": 0,
    }
