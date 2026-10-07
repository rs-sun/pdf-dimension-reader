"""Request-local execution guard for the strict vector-only API."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import sys
from typing import Any, Callable, Iterator, TypeVar

from vector_only_contract import (
    zero_consumer_runtime_audit,
    zero_ocr_runtime_audit,
)


_F = TypeVar("_F", bound=Callable[..., Any])
_active_audits: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = (
    contextvars.ContextVar("strict_vector_only_runtime_audits", default=None)
)


class VectorOnlyRuntimeForbidden(RuntimeError):
    def __init__(self, *, kind: str, entrypoint: str):
        self.kind = str(kind)
        self.entrypoint = str(entrypoint)
        super().__init__(f"vector-only forbids {self.kind}: {self.entrypoint}")


@contextlib.contextmanager
def strict_vector_only_runtime_guard() -> Iterator[dict[str, dict[str, Any]]]:
    audits = {
        "ocr": zero_ocr_runtime_audit(),
        "consumer": zero_consumer_runtime_audit(),
    }
    token = _active_audits.set(audits)
    try:
        yield audits
    finally:
        _active_audits.reset(token)


def guard_ocr_call(entrypoint: str) -> None:
    audits = _active_audits.get()
    if audits is None:
        return
    audit = audits["ocr"]
    audit["called_entrypoints"].append(str(entrypoint))
    audit["call_count"] += 1
    if entrypoint == "run_empty_region_retry":
        audit["retry_count"] += 1
    raise VectorOnlyRuntimeForbidden(kind="ocr_call", entrypoint=entrypoint)


def guard_ocr_initialization(entrypoint: str) -> None:
    audits = _active_audits.get()
    if audits is None:
        return
    audit = audits["ocr"]
    audit["initialization_entrypoints"].append(str(entrypoint))
    audit["initialization_count"] += 1
    raise VectorOnlyRuntimeForbidden(
        kind="ocr_initialization",
        entrypoint=entrypoint,
    )


def guard_consumer_call(entrypoint: str) -> None:
    audits = _active_audits.get()
    if audits is None:
        return
    audit = audits["consumer"]
    audit["called_entrypoints"].append(str(entrypoint))
    audit["call_count"] += 1
    raise VectorOnlyRuntimeForbidden(kind="consumer", entrypoint=entrypoint)


def guarded_ocr_entrypoint(name: str, function: _F) -> _F:
    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any):
        guard_ocr_call(name)
        return function(*args, **kwargs)

    return guarded  # type: ignore[return-value]


def guarded_consumer_entrypoint(name: str, function: _F) -> _F:
    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any):
        guard_consumer_call(name)
        return function(*args, **kwargs)

    return guarded  # type: ignore[return-value]


def ocr_singleton_snapshot() -> tuple[bool, frozenset[str]]:
    module = sys.modules.get("ocr_engine_utils")
    if module is None:
        return False, frozenset()
    lock = getattr(module, "_ocr_lock", None)
    if lock is None:
        paddle_loaded = getattr(module, "_paddle_ocr", None) is not None
        backends = frozenset(
            str(key) for key in (getattr(module, "_ocr_by_backend", {}) or {})
        )
        return paddle_loaded, backends
    with lock:
        paddle_loaded = getattr(module, "_paddle_ocr", None) is not None
        backends = frozenset(
            str(key) for key in (getattr(module, "_ocr_by_backend", {}) or {})
        )
    return paddle_loaded, backends


def finalize_runtime_audits(
    audits: dict[str, dict[str, Any]],
    *,
    before_ocr_singletons: tuple[bool, frozenset[str]],
    after_ocr_singletons: tuple[bool, frozenset[str]],
) -> None:
    before_paddle, before_backends = before_ocr_singletons
    after_paddle, after_backends = after_ocr_singletons
    delta = int(not before_paddle and after_paddle) + len(
        after_backends - before_backends
    )
    audit = audits["ocr"]
    audit["singleton_cache_delta_count"] = delta
    audit["initialization_count"] += delta
    audit["process_global_singleton_cache_delta_observed"] = delta
