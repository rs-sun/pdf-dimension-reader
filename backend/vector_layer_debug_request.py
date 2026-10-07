"""Parse the explicitly requested, bounded vector-layer diagnostic options.

This request is deliberately independent from recognition quality settings.
Malformed diagnostic options never change or block the product pipeline; the
API can report them as an unavailable diagnostic sidecar after product work is
complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUEST_SCHEMA_VERSION = "vector_layer_debug_request_v1"
MAX_OVERLAY_LIMIT = 2000
_OPTION_KEYS = frozenset({
    "schema_version",
    "include_overlays",
    "overlay_limit",
})


@dataclass(frozen=True)
class VectorLayerDebugRequest:
    include_overlays: bool
    overlay_limit: int
    valid: bool = True
    reason: str | None = None

    @classmethod
    def from_request(
        cls,
        data: dict[str, Any] | None,
    ) -> "VectorLayerDebugRequest | None":
        if not isinstance(data, dict):
            return None
        settings = data.get("settings")
        if not isinstance(settings, dict) or "vector_layer_debug" not in settings:
            return None
        raw = settings.get("vector_layer_debug")
        if not isinstance(raw, dict):
            return cls._invalid("options_not_object")
        if frozenset(raw) != _OPTION_KEYS:
            return cls._invalid("options_keys_invalid")
        if raw.get("schema_version") != REQUEST_SCHEMA_VERSION:
            return cls._invalid("schema_version_invalid")
        include_overlays = raw.get("include_overlays")
        if type(include_overlays) is not bool:
            return cls._invalid("include_overlays_invalid")
        overlay_limit = raw.get("overlay_limit")
        if (
            type(overlay_limit) is not int
            or overlay_limit < 0
            or overlay_limit > MAX_OVERLAY_LIMIT
        ):
            return cls._invalid("overlay_limit_invalid")
        if not include_overlays and overlay_limit != 0:
            return cls._invalid("overlay_limit_without_overlays")
        return cls(
            include_overlays=include_overlays,
            overlay_limit=overlay_limit,
        )

    @classmethod
    def _invalid(cls, reason: str) -> "VectorLayerDebugRequest":
        return cls(
            include_overlays=False,
            overlay_limit=0,
            valid=False,
            reason=str(reason),
        )
