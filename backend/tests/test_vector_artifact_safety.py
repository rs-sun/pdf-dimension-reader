from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from vector_artifact_safety import (
    _compact_identifier,
    vector_artifact_safety_reasons,
)
from vector_phrase_chain_validation import (
    request_local_artifact_safety_reasons,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("PADDLE OCR", "paddleocr"),
        ("V.L.M.", "vlm"),
        ("İ", "i"),
        ("Straße", "strae"),
        ("AＢC", "ac"),
        (1.0, "10"),
    ],
)
def test_compact_identifier_preserves_ascii_byte_semantics(
    value,
    expected,
):
    assert _compact_identifier(value) == expected


def test_precompiled_identifier_sets_preserve_exact_safety_reasons():
    artifact = {
        "Consumer-Allowed": True,
        "INITIALIZATION COUNT": 1,
        "OCR RESULTS": ["glyph"],
        "nested": {"backend": {"name": "PaddleOCR"}},
    }

    assert vector_artifact_safety_reasons(artifact) == [
        "consumer_allowed_nonfalse:root.Consumer-Allowed",
        "forbidden_recognition_backend:root.nested.backend.name",
        "ocr_artifact_nonempty:root.OCR RESULTS",
        "runtime_count_nonzero:root.INITIALIZATION COUNT",
    ]


def test_standalone_safety_scan_detects_in_place_mutation_and_path():
    artifact = {"engine": "vector"}
    assert vector_artifact_safety_reasons(artifact, path="first") == []

    artifact["engine"] = "PaddleOCR"

    assert vector_artifact_safety_reasons(
        artifact,
        path="second",
    ) == ["forbidden_recognition_backend:second.engine"]


def test_native_pdf_font_names_do_not_impersonate_ocr_runtime_fields():
    artifact = {
        "font_profile": {
            "font_names": {
                "OCR-A": 2,
                "OCR-B": 3,
            },
        },
    }

    assert vector_artifact_safety_reasons(artifact) == []
    assert vector_artifact_safety_reasons({
        "ocr_backend": "OCR-B",
    }) == ["ocr_field_nonzero:root.ocr_backend"]
    assert vector_artifact_safety_reasons({
        "backend": "OCR-B",
    }) == ["forbidden_recognition_backend:root.backend"]
    assert vector_artifact_safety_reasons({
        "font_names": {
            "ocr_backend": "paddleocr",
        },
    }) == ["ocr_field_nonzero:root.font_names.ocr_backend"]


def test_standalone_safety_scan_fails_closed_on_concurrent_mutation():
    class _MutatingDict(dict):
        def items(self):
            for index, item in enumerate(super().items()):
                if index == 0:
                    self["added_during_scan"] = True
                yield item

    assert vector_artifact_safety_reasons(
        _MutatingDict({"engine": "vector"}),
        path="artifact",
    ) == ["artifact_mutated_during_scan:artifact"]


def test_request_safety_wrapper_rechecks_in_place_mutation():
    artifact = {"engine": "vector"}
    assert request_local_artifact_safety_reasons(
        artifact,
        path="first",
    ) == []

    artifact["engine"] = "PaddleOCR"

    assert request_local_artifact_safety_reasons(
        artifact,
        path="second",
    ) == ["forbidden_recognition_backend:second.engine"]


def test_request_safety_wrapper_preserves_nested_paths():
    artifact = {"outer": [{"Consumer Allowed": True}]}

    assert request_local_artifact_safety_reasons(
        artifact,
        path="candidate",
    ) == [
        "consumer_allowed_nonfalse:candidate.outer[0].Consumer Allowed"
    ]


def test_request_safety_wrapper_has_no_stage_identity_admission():
    first = {"engine": "vector"}
    second = {"engine": "vector"}

    assert request_local_artifact_safety_reasons(first, path="first") == []
    assert request_local_artifact_safety_reasons(second, path="second") == []


def test_request_safety_wrapper_is_thread_independent():
    artifact = {"nested": {"engine": "PaddleOCR"}}
    with ThreadPoolExecutor(max_workers=1) as executor:
        reasons = executor.submit(
            request_local_artifact_safety_reasons,
            artifact,
            path="thread",
        ).result()

    assert reasons == [
        "forbidden_recognition_backend:thread.nested.engine"
    ]


def test_request_safety_wrapper_returns_a_fresh_reason_list():
    artifact = {"Consumer Allowed": True}
    first = request_local_artifact_safety_reasons(artifact, path="item")
    first.append("caller_mutation")

    assert request_local_artifact_safety_reasons(
        artifact,
        path="item",
    ) == ["consumer_allowed_nonfalse:item.Consumer Allowed"]
