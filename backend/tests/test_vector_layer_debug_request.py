from __future__ import annotations

from vector_layer_debug_request import (
    MAX_OVERLAY_LIMIT,
    REQUEST_SCHEMA_VERSION,
    VectorLayerDebugRequest,
)


def _request(options):
    return {
        "settings": {
            "dimension_path": "vector_only",
            "vector_layer_debug": options,
        },
    }


def _valid_options(**updates):
    value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "include_overlays": True,
        "overlay_limit": MAX_OVERLAY_LIMIT,
    }
    value.update(updates)
    return value


def test_missing_debug_option_preserves_the_existing_request_contract():
    assert VectorLayerDebugRequest.from_request(None) is None
    assert VectorLayerDebugRequest.from_request({}) is None
    assert VectorLayerDebugRequest.from_request({"settings": {}}) is None


def test_valid_debug_option_is_explicit_and_bounded():
    parsed = VectorLayerDebugRequest.from_request(_request(_valid_options()))
    assert parsed == VectorLayerDebugRequest(
        include_overlays=True,
        overlay_limit=MAX_OVERLAY_LIMIT,
        valid=True,
        reason=None,
    )


def test_counts_only_request_requires_zero_overlay_limit():
    parsed = VectorLayerDebugRequest.from_request(_request(_valid_options(
        include_overlays=False,
        overlay_limit=0,
    )))
    assert parsed is not None
    assert parsed.valid is True
    assert parsed.include_overlays is False
    assert parsed.overlay_limit == 0


def test_malformed_debug_options_fail_only_the_diagnostic_sidecar():
    cases = [
        (None, "options_not_object"),
        ({}, "options_keys_invalid"),
        (_valid_options(extra=True), "options_keys_invalid"),
        (_valid_options(schema_version="future"), "schema_version_invalid"),
        (_valid_options(include_overlays=1), "include_overlays_invalid"),
        (_valid_options(overlay_limit=True), "overlay_limit_invalid"),
        (_valid_options(overlay_limit=-1), "overlay_limit_invalid"),
        (_valid_options(overlay_limit=MAX_OVERLAY_LIMIT + 1), "overlay_limit_invalid"),
        (_valid_options(include_overlays=False), "overlay_limit_without_overlays"),
    ]
    for raw, reason in cases:
        parsed = VectorLayerDebugRequest.from_request(_request(raw))
        assert parsed is not None
        assert parsed.valid is False
        assert parsed.reason == reason
        assert parsed.include_overlays is False
        assert parsed.overlay_limit == 0
