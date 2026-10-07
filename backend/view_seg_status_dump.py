"""R4 view segmentation status dump helpers."""

from __future__ import annotations

import os
from typing import Any


SCHEMA_VERSION = "view_seg_status_v1"


def build_view_seg_status_v1(
    raw_status: dict[str, Any] | None,
    *,
    page_index: int,
    view_boxes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Normalize view-seg router telemetry into a stable dump schema."""
    raw_status = dict(raw_status or {})
    requested_backend = str(
        raw_status.get("requested_backend")
        or raw_status.get("backend")
        or "unknown"
    )
    effective_backend = str(
        raw_status.get("effective_backend")
        or raw_status.get("backend")
        or raw_status.get("status")
        or "unknown"
    )
    fallback = bool(raw_status.get("fallback"))
    fallback_reason = str(raw_status.get("fallback_reason") or "")
    if fallback and not fallback_reason:
        fallback_reason = "unknown_fallback"

    status = _status_label(
        requested_backend=requested_backend,
        effective_backend=effective_backend,
        fallback=fallback,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "backend": requested_backend,
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "source": str(raw_status.get("source") or ""),
        "model_path": str(raw_status.get("model_path") or raw_status.get("model") or ""),
        "model_exists": raw_status.get("model_exists"),
        "timeout_s": _float_env("YOLO_VIEW_SUBPROCESS_TIMEOUT", 120.0),
        "view_count": _view_count(raw_status, view_boxes),
        "page_index": int(page_index),
    }


def _status_label(
    *,
    requested_backend: str,
    effective_backend: str,
    fallback: bool,
) -> str:
    if fallback and requested_backend == "yolo" and effective_backend == "watershed":
        return "yolo->watershed_fallback"
    if effective_backend:
        return effective_backend
    return requested_backend or "unknown"


def _view_count(
    raw_status: dict[str, Any],
    view_boxes: list[dict[str, Any]] | None,
) -> int:
    if isinstance(view_boxes, list):
        return len(view_boxes)
    try:
        return int(raw_status.get("view_count") or 0)
    except (TypeError, ValueError):
        return 0


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
