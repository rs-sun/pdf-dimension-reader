"""Strict, request-scoped YOLO view inference for the vector-only path.

The worker receives only immutable PDF bytes and a page index.  It never owns a
``pdfplumber.Page`` or ``fitz.Page`` from the request thread, and it never falls
back to watershed/minesweeper.  Wall-clock measurements are emitted only via a
caller-owned timing sink so semantic payloads remain deterministic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, MutableMapping


STATUS_SCHEMA_VERSION = "strict_yolo_view_status_v1"
ERROR_SCHEMA_VERSION = "strict_yolo_view_error_v1"
VIEW_STATUS_SCHEMA_VERSION = "view_seg_status_v1"
CONSUMER_ALLOWED = False
_executor_state: _ExecutorState | None = None
_executor_lock = threading.Lock()


@dataclass(frozen=True)
class StrictYoloViewResult:
    """Deterministic semantic result for one PDF page."""

    view_boxes: tuple[dict[str, Any], ...]
    status: dict[str, Any]


class StrictYoloViewUnavailable(RuntimeError):
    """Structured fail-closed error for the strict vector-only path."""

    def __init__(self, *, reason: str, trace_id: str, page_index: int):
        self.reason = str(reason)
        self.trace_id = str(trace_id)
        self.page_index = int(page_index)
        super().__init__(f"strict yolo view unavailable: {self.reason}")

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ERROR_SCHEMA_VERSION,
            "code": "yolo_view_unavailable",
            "status": "fail_closed",
            "reasons": [self.reason],
            "trace_id": self.trace_id,
            "page_index": self.page_index,
            "requested_backend": "yolo",
            "effective_backend": "unavailable",
            "fallback": False,
            "feeds_display_l3": False,
            "consumer_allowed": CONSUMER_ALLOWED,
        }


def build_strict_yolo_trace_id(pdf_bytes: bytes, page_index: int) -> str:
    """Build a deterministic, content-addressed page trace identifier."""
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise ValueError("pdf_bytes must be non-empty bytes")
    page_value = int(page_index)
    if page_value < 0:
        raise ValueError("page_index must be >= 0")
    digest = hashlib.sha256(pdf_bytes).hexdigest()[:24]
    return f"r32v1-{digest}-p{page_value + 1:04d}"


def strict_yolo_view_model_path() -> Path:
    explicit = os.environ.get("YOLO_VIEW_MODEL")
    if explicit:
        return Path(explicit).expanduser()
    return (
        Path(__file__).resolve().parent
        / "yolo_views"
        / "models"
        / "view_model.onnx"
    )


def strict_yolo_view_onnx_runner_path() -> Path:
    """Return the fixed Issue-04-owned ONNX runner location."""
    return Path(__file__).resolve().parent / "yolo_views" / "infer_onnx_cli.py"


def strict_yolo_view_expected_sha256() -> str | None:
    configured = os.environ.get("YOLO_VIEW_MODEL_SHA256")
    if configured:
        return configured.strip().lower()
    return None


def strict_yolo_view_timeout_s() -> float:
    try:
        value = float(os.environ.get("YOLO_VIEW_SUBPROCESS_TIMEOUT", "120"))
    except ValueError:
        return 120.0
    return value if math.isfinite(value) and value > 0.0 else 120.0


def _require_strict_onnx_model(
    model_path: str | Path,
    *,
    trace_id: str,
    page_index: int,
) -> Path:
    model = Path(model_path)
    if model.suffix.lower() != ".onnx":
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_model_format_forbidden",
            trace_id=trace_id,
            page_index=page_index,
        )
    fallback_setting = str(os.environ.get("YOLO_ALLOW_PT_FALLBACK") or "").strip().lower()
    if fallback_setting not in {"", "0", "false", "no", "off"}:
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_pt_fallback_forbidden",
            trace_id=trace_id,
            page_index=page_index,
        )
    if not model.is_file():
        raise StrictYoloViewUnavailable(
            reason="model_file_missing",
            trace_id=trace_id,
            page_index=page_index,
        )
    if not strict_yolo_view_onnx_runner_path().is_file():
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_onnx_runner_missing",
            trace_id=trace_id,
            page_index=page_index,
        )
    return model


def resolve_strict_yolo_view_source(
    source: Any,
    *,
    trace_id: str,
    page_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Resolve one task/result and validate the strict no-fallback contract."""
    if type(page_index) is not int or page_index < 0:
        raise StrictYoloViewUnavailable(
            reason="invalid_strict_yolo_page_index",
            trace_id=str(trace_id or ""),
            page_index=0,
        )
    page_value = page_index
    trace_value = str(trace_id or "")
    if source is None:
        raise StrictYoloViewUnavailable(
            reason="missing_strict_yolo_source",
            trace_id=trace_value,
            page_index=page_value,
        )
    if not isinstance(trace_id, str) or not trace_id:
        raise StrictYoloViewUnavailable(
            reason="missing_strict_yolo_trace_id",
            trace_id=trace_value,
            page_index=page_value,
        )
    trace_value = trace_id
    if isinstance(source, StrictYoloViewResult):
        result = source
    else:
        resolver = getattr(source, "result", None)
        if not callable(resolver):
            raise StrictYoloViewUnavailable(
                reason="invalid_strict_yolo_source",
                trace_id=trace_value,
                page_index=page_value,
            )
        try:
            result = resolver()
        except StrictYoloViewUnavailable:
            raise
        except Exception as exc:
            raise StrictYoloViewUnavailable(
                reason="strict_yolo_source_failed",
                trace_id=trace_value,
                page_index=page_value,
            ) from exc
    if not isinstance(result, StrictYoloViewResult):
        raise StrictYoloViewUnavailable(
            reason="invalid_strict_yolo_result",
            trace_id=trace_value,
            page_index=page_value,
        )

    strict_status = dict(result.status) if isinstance(result.status, dict) else {}
    checks = (
        (strict_status.get("schema_version") == STATUS_SCHEMA_VERSION, "strict_yolo_status_schema_mismatch"),
        (str(strict_status.get("trace_id") or "") == trace_value, "strict_yolo_trace_mismatch"),
        (
            _strict_nonnegative_int(strict_status.get("page_index")) == page_value,
            "strict_yolo_page_mismatch",
        ),
        (strict_status.get("requested_backend") == "yolo", "strict_yolo_requested_backend_invalid"),
        (strict_status.get("effective_backend") == "yolo", "strict_yolo_effective_backend_invalid"),
        (strict_status.get("source") == "yolo_views", "strict_yolo_source_invalid"),
        (strict_status.get("fallback") is False, "strict_yolo_fallback_forbidden"),
        (strict_status.get("model_sha256_verified") is True, "strict_yolo_model_unverified"),
        (strict_status.get("consumer_allowed") is False, "strict_yolo_consumer_forbidden"),
    )
    for passed, reason in checks:
        if not passed:
            raise StrictYoloViewUnavailable(
                reason=reason,
                trace_id=trace_value,
                page_index=page_value,
            )
    model_sha256 = str(strict_status.get("model_sha256") or "").lower()
    if (
        len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256)
    ):
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_model_sha256_invalid",
            trace_id=trace_value,
            page_index=page_value,
        )
    try:
        raw_view_boxes = list(result.view_boxes)
    except TypeError as exc:
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_view_structure_invalid",
            trace_id=trace_value,
            page_index=page_value,
        ) from exc
    if any(not isinstance(row, dict) for row in raw_view_boxes):
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_view_structure_invalid",
            trace_id=trace_value,
            page_index=page_value,
        )
    for row in raw_view_boxes:
        if row.get("backend") != "yolo":
            raise StrictYoloViewUnavailable(
                reason="strict_yolo_view_backend_invalid",
                trace_id=trace_value,
                page_index=page_value,
            )
        if row.get("source") != "yolo_views":
            raise StrictYoloViewUnavailable(
                reason="strict_yolo_view_source_invalid",
                trace_id=trace_value,
                page_index=page_value,
            )
        if row.get("fallback") is not False:
            raise StrictYoloViewUnavailable(
                reason="strict_yolo_view_fallback_forbidden",
                trace_id=trace_value,
                page_index=page_value,
            )
    try:
        view_boxes = _canonical_views(raw_view_boxes, page_index=page_value)
    except (TypeError, ValueError) as exc:
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_view_structure_invalid",
            trace_id=trace_value,
            page_index=page_value,
        ) from exc
    if not view_boxes:
        raise StrictYoloViewUnavailable(
            reason="empty_views",
            trace_id=trace_value,
            page_index=page_value,
        )
    if _strict_nonnegative_int(strict_status.get("view_count")) != len(view_boxes):
        raise StrictYoloViewUnavailable(
            reason="strict_yolo_view_count_mismatch",
            trace_id=trace_value,
            page_index=page_value,
        )
    for expected, raw in zip(view_boxes, raw_view_boxes):
        if str(raw.get("view_id") or "") != str(expected["view_id"]):
            raise StrictYoloViewUnavailable(
                reason="strict_yolo_view_id_mismatch",
                trace_id=trace_value,
                page_index=page_value,
            )

    view_status = {
        "schema_version": VIEW_STATUS_SCHEMA_VERSION,
        "status": "yolo",
        "backend": "yolo",
        "requested_backend": "yolo",
        "effective_backend": "yolo",
        "fallback": False,
        "fallback_reason": "",
        "source": "yolo_views",
        "model_path": str(strict_status.get("model_name") or ""),
        "model_exists": True,
        "view_count": len(view_boxes),
        "page_index": page_value,
        "trace_id": trace_value,
        "model_sha256": model_sha256,
        "model_sha256_verified": True,
        "consumer_allowed": CONSUMER_ALLOWED,
    }
    return view_boxes, view_status, strict_status


def _strict_nonnegative_int(value: Any) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


@dataclass
class _ExecutorState:
    pid: int
    executor: ThreadPoolExecutor
    slots: threading.BoundedSemaphore
    max_pending: int
    submit_lock: Any = field(default_factory=threading.Lock, repr=False)
    poisoned: bool = False


class _TaskControl:
    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._process: Any | None = None
        self._stop_claimed = False
        self._stop_failed = False
        self._stop_done = threading.Event()
        self._completed = False

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def attach(self, process: Any) -> None:
        should_stop = False
        with self._lock:
            self._process = process
            if self._cancelled.is_set() and not self._stop_claimed:
                self._stop_claimed = True
                should_stop = True
        if should_stop:
            self._record_stop_result(_terminate_process(process))

    def detach(self, process: Any) -> None:
        with self._lock:
            if self._process is process:
                self._process = None

    def cancel(self) -> bool:
        process = None
        with self._lock:
            if self._completed:
                return False
            first_request = not self._cancelled.is_set()
            self._cancelled.set()
            if self._process is not None and not self._stop_claimed:
                self._stop_claimed = True
                process = self._process
        if process is not None:
            self._record_stop_result(_terminate_process(process))
        return first_request

    def finish_success(self) -> bool:
        with self._lock:
            if self._cancelled.is_set():
                return False
            self._completed = True
            return True

    def stop_for_timeout(self, process: Any) -> bool:
        should_stop = False
        with self._lock:
            if self._process is process and not self._stop_claimed:
                self._stop_claimed = True
                should_stop = True
        if should_stop:
            stopped = _terminate_process(process)
            self._record_stop_result(stopped)
            return stopped
        self._stop_done.wait(timeout=3.0)
        with self._lock:
            return self._stop_done.is_set() and not self._stop_failed

    def _record_stop_result(self, stopped: bool) -> None:
        with self._lock:
            if not stopped:
                self._stop_failed = True
        self._stop_done.set()

    def cancellation_reason(self) -> str:
        with self._lock:
            stop_claimed = self._stop_claimed
        if stop_claimed:
            self._stop_done.wait(timeout=3.0)
        with self._lock:
            if self._stop_claimed and (
                not self._stop_done.is_set() or self._stop_failed
            ):
                return "process_termination_failed"
        return "cancelled"


class _PendingPdfBytes:
    def __init__(self, pdf_bytes: bytes | bytearray):
        self._lock = threading.Lock()
        self._pdf_bytes: bytes | None = bytes(pdf_bytes)

    def take(self) -> bytes | None:
        with self._lock:
            value = self._pdf_bytes
            self._pdf_bytes = None
            return value

    def clear(self) -> None:
        with self._lock:
            self._pdf_bytes = None


class _SubmissionGate:
    """Keep a queued worker inert until ``submit`` has fully committed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._decision: bool | None = None
        self._decided = threading.Event()

    def commit(self) -> None:
        with self._lock:
            if self._decision is None:
                self._decision = True
                self._decided.set()

    def abort(self) -> None:
        with self._lock:
            if self._decision is None:
                self._decision = False
                self._decided.set()

    def wait_for_commit(self) -> bool:
        self._decided.wait()
        with self._lock:
            return self._decision is True


class _SlotLease:
    """Release one pending-request slot at most once across all races."""

    def __init__(self, slots: threading.BoundedSemaphore) -> None:
        self._slots = slots
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._slots.release()


class StrictYoloViewTask:
    """Handle for one request-scoped background inference."""

    def __init__(
        self,
        future: Future[StrictYoloViewResult],
        control: _TaskControl,
        pending_pdf: _PendingPdfBytes,
        trace_id: str,
        page_index: int,
    ):
        self._future = future
        self._control = control
        self._pending_pdf = pending_pdf
        self._trace_id = str(trace_id)
        self._page_index = int(page_index)

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> StrictYoloViewResult:
        try:
            return self._future.result(timeout=timeout)
        except CancelledError as exc:
            raise StrictYoloViewUnavailable(
                reason="cancelled",
                trace_id=self._trace_id,
                page_index=self._page_index,
            ) from exc

    def cancel(self, reason: str = "cancelled") -> bool:
        del reason
        if self._future.done():
            return False
        accepted = self._control.cancel()
        self._pending_pdf.clear()
        self._future.cancel()
        return accepted


def _run_pending_strict_yolo_view(
    *,
    pending_pdf: _PendingPdfBytes,
    submission_gate: _SubmissionGate,
    control: _TaskControl,
    trace_id: str,
    page_index: int,
    **kwargs: Any,
) -> StrictYoloViewResult:
    if not submission_gate.wait_for_commit():
        pending_pdf.clear()
        raise StrictYoloViewUnavailable(
            reason="cancelled",
            trace_id=trace_id,
            page_index=page_index,
        )
    pdf_bytes = pending_pdf.take()
    if pdf_bytes is None:
        raise StrictYoloViewUnavailable(
            reason="cancelled",
            trace_id=trace_id,
            page_index=page_index,
        )
    return run_strict_yolo_view(
        pdf_bytes=pdf_bytes,
        page_index=page_index,
        trace_id=trace_id,
        _control=control,
        **kwargs,
    )


def submit_strict_yolo_view(
    *,
    pdf_bytes: bytes,
    page_index: int,
    trace_id: str,
    model_path: str | Path,
    expected_model_sha256: str | None = None,
    timeout_s: float,
    timing_sink: MutableMapping[str, float] | None = None,
) -> StrictYoloViewTask:
    """Start strict YOLO view inference without blocking the request thread."""
    _require_strict_onnx_model(
        model_path,
        trace_id=trace_id,
        page_index=page_index,
    )
    submitted_at = time.perf_counter()
    timeout_value = float(timeout_s)
    control = _TaskControl()
    pending_pdf = _PendingPdfBytes(pdf_bytes)
    submission_gate = _SubmissionGate()
    try:
        state = _get_executor_state()
    except Exception as exc:
        pending_pdf.clear()
        control.cancel()
        raise StrictYoloViewUnavailable(
            reason="executor_init_failed",
            trace_id=trace_id,
            page_index=page_index,
        ) from exc
    with state.submit_lock:
        if state.poisoned:
            pending_pdf.clear()
            raise StrictYoloViewUnavailable(
                reason="executor_poisoned",
                trace_id=trace_id,
                page_index=page_index,
            )
        if not state.slots.acquire(blocking=False):
            pending_pdf.clear()
            raise StrictYoloViewUnavailable(
                reason="queue_saturated",
                trace_id=trace_id,
                page_index=page_index,
            )
        slot_lease = _SlotLease(state.slots)
        try:
            future = state.executor.submit(
                _run_pending_strict_yolo_view,
                pending_pdf=pending_pdf,
                submission_gate=submission_gate,
                control=control,
                page_index=page_index,
                trace_id=trace_id,
                model_path=model_path,
                expected_model_sha256=expected_model_sha256,
                timeout_s=timeout_s,
                timing_sink=timing_sink,
                _started_at=submitted_at,
                _deadline_monotonic=submitted_at + timeout_value,
            )

            def release_request_slot(_future: Future[StrictYoloViewResult]) -> None:
                pending_pdf.clear()
                slot_lease.release()

            future.add_done_callback(release_request_slot)
            task = StrictYoloViewTask(
                future,
                control,
                pending_pdf,
                trace_id,
                page_index,
            )
            submission_gate.commit()
        except BaseException as exc:
            state.poisoned = True
            submission_gate.abort()
            pending_pdf.clear()
            try:
                control.cancel()
            finally:
                slot_lease.release()
                try:
                    state.executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            if isinstance(exc, Exception):
                raise StrictYoloViewUnavailable(
                    reason="executor_submit_failed",
                    trace_id=trace_id,
                    page_index=page_index,
                ) from exc
            raise
    return task


def _get_executor_state() -> _ExecutorState:
    global _executor_state
    pid = os.getpid()
    with _executor_lock:
        if _executor_state is None or _executor_state.pid != pid:
            try:
                max_workers = int(os.environ.get("STRICT_YOLO_VIEW_WORKERS", "1"))
            except ValueError:
                max_workers = 1
            max_workers = max(1, min(max_workers, 4))
            try:
                max_pending = int(
                    os.environ.get(
                        "STRICT_YOLO_VIEW_MAX_PENDING",
                        str(max_workers * 2),
                    )
                )
            except ValueError:
                max_pending = max_workers * 2
            max_pending = max(max_workers, min(max_pending, max_workers * 8))
            _executor_state = _ExecutorState(
                pid=pid,
                executor=ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="strict-yolo-view",
                ),
                slots=threading.BoundedSemaphore(max_pending),
                max_pending=max_pending,
            )
        return _executor_state


def run_strict_yolo_view(
    *,
    pdf_bytes: bytes,
    page_index: int,
    trace_id: str,
    model_path: str | Path,
    expected_model_sha256: str | None = None,
    timeout_s: float,
    timing_sink: MutableMapping[str, float] | None = None,
    _control: _TaskControl | None = None,
    _started_at: float | None = None,
    _deadline_monotonic: float | None = None,
) -> StrictYoloViewResult:
    """Run isolated YOLO view inference and return a canonical page result."""
    started = float(_started_at) if _started_at is not None else time.perf_counter()
    timeout_value = float(timeout_s)
    deadline = (
        float(_deadline_monotonic)
        if _deadline_monotonic is not None
        else started + timeout_value
    )
    control = _control or _TaskControl()
    if control.is_cancelled():
        raise StrictYoloViewUnavailable(
            reason="cancelled",
            trace_id=trace_id,
            page_index=page_index,
        )
    if time.perf_counter() >= deadline:
        raise StrictYoloViewUnavailable(
            reason="timeout",
            trace_id=trace_id,
            page_index=page_index,
        )
    model = _require_strict_onnx_model(
        model_path,
        trace_id=trace_id,
        page_index=page_index,
    )
    expected_sha256 = expected_model_sha256 or os.environ.get(
        "YOLO_VIEW_MODEL_SHA256"
    )
    if not expected_sha256:
        raise StrictYoloViewUnavailable(
            reason="model_sha256_required",
            trace_id=trace_id,
            page_index=page_index,
        )
    expected_sha256 = str(expected_sha256).strip().lower()
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise StrictYoloViewUnavailable(
            reason="model_sha256_invalid",
            trace_id=trace_id,
            page_index=page_index,
        )
    try:
        model_sha256 = _sha256_file(model)
    except OSError as exc:
        raise StrictYoloViewUnavailable(
            reason="model_read_failed",
            trace_id=trace_id,
            page_index=page_index,
        ) from exc
    if expected_sha256 != model_sha256:
        raise StrictYoloViewUnavailable(
            reason="model_sha256_mismatch",
            trace_id=trace_id,
            page_index=page_index,
        )
    if control.is_cancelled():
        raise StrictYoloViewUnavailable(
            reason="cancelled",
            trace_id=trace_id,
            page_index=page_index,
        )
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        raise ValueError("pdf_bytes must be non-empty bytes")
    if int(page_index) < 0:
        raise ValueError("page_index must be >= 0")
    if not str(trace_id):
        raise ValueError("trace_id must be non-empty")
    if not math.isfinite(timeout_value) or timeout_value <= 0.0:
        raise ValueError("timeout_s must be finite and > 0")

    temp_pdf: str | None = None
    try:
        try:
            handle = tempfile.NamedTemporaryFile(
                prefix="pdf_reader_strict_yolo_view_",
                suffix=".pdf",
                delete=False,
            )
            temp_pdf = handle.name
            try:
                handle.write(bytes(pdf_bytes))
            finally:
                handle.close()
        except OSError as exc:
            raise StrictYoloViewUnavailable(
                reason="temp_pdf_failed",
                trace_id=trace_id,
                page_index=page_index,
            ) from exc

        script = strict_yolo_view_onnx_runner_path()
        cmd = [
            sys.executable,
            str(script),
            "--pdf",
            str(temp_pdf),
            "--page",
            str(int(page_index) + 1),
            "--model",
            str(model),
        ]
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"YOLO_ALLOW_PT_FALLBACK", "YOLO_CONFIG_DIR"}
        }
        if time.perf_counter() >= deadline:
            raise StrictYoloViewUnavailable(
                reason="timeout",
                trace_id=trace_id,
                page_index=page_index,
            )
        if control.is_cancelled():
            raise StrictYoloViewUnavailable(
                reason=control.cancellation_reason(),
                trace_id=trace_id,
                page_index=page_index,
            )
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise StrictYoloViewUnavailable(
                reason="spawn_failed",
                trace_id=trace_id,
                page_index=page_index,
            ) from exc
        control.attach(process)
        try:
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0.0:
                stopped = control.stop_for_timeout(process)
                raise StrictYoloViewUnavailable(
                    reason=(
                        "timeout"
                        if stopped
                        else "process_termination_failed"
                    ),
                    trace_id=trace_id,
                    page_index=page_index,
                )
            stdout, _stderr = process.communicate(timeout=remaining_s)
        except subprocess.TimeoutExpired as exc:
            stopped = control.stop_for_timeout(process)
            raise StrictYoloViewUnavailable(
                reason=(
                    "process_termination_failed"
                    if not stopped
                    else ("cancelled" if control.is_cancelled() else "timeout")
                ),
                trace_id=trace_id,
                page_index=page_index,
            ) from exc
        except StrictYoloViewUnavailable:
            raise
        except Exception as exc:
            stopped = control.stop_for_timeout(process)
            raise StrictYoloViewUnavailable(
                reason=(
                    "process_termination_failed"
                    if not stopped
                    else (
                        "cancelled"
                        if control.is_cancelled()
                        else "subprocess_communication_failed"
                    )
                ),
                trace_id=trace_id,
                page_index=page_index,
            ) from exc
        finally:
            control.detach(process)
            _close_process_pipes(process)
        if control.is_cancelled():
            raise StrictYoloViewUnavailable(
                reason=control.cancellation_reason(),
                trace_id=trace_id,
                page_index=page_index,
            )
        if process.returncode != 0:
            raise StrictYoloViewUnavailable(
                reason="subprocess_failed",
                trace_id=trace_id,
                page_index=page_index,
            )
        try:
            model_sha256_after = _sha256_file(model)
        except OSError as exc:
            raise StrictYoloViewUnavailable(
                reason="model_read_failed",
                trace_id=trace_id,
                page_index=page_index,
            ) from exc
        if model_sha256_after != model_sha256:
            raise StrictYoloViewUnavailable(
                reason="model_sha256_changed_during_inference",
                trace_id=trace_id,
                page_index=page_index,
            )
        try:
            raw_views = json.loads(stdout or "[]")
        except (json.JSONDecodeError, TypeError) as exc:
            raise StrictYoloViewUnavailable(
                reason="invalid_json",
                trace_id=trace_id,
                page_index=page_index,
            ) from exc
        if not isinstance(raw_views, list):
            raise StrictYoloViewUnavailable(
                reason="invalid_output_schema",
                trace_id=trace_id,
                page_index=page_index,
            )
        try:
            canonical_views = _canonical_views(raw_views, page_index=int(page_index))
        except (TypeError, ValueError) as exc:
            raise StrictYoloViewUnavailable(
                reason="invalid_view_structure",
                trace_id=trace_id,
                page_index=page_index,
            ) from exc
        if not canonical_views:
            raise StrictYoloViewUnavailable(
                reason="empty_views",
                trace_id=trace_id,
                page_index=page_index,
            )
        status = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "trace_id": str(trace_id),
            "page_index": int(page_index),
            "requested_backend": "yolo",
            "effective_backend": "yolo",
            "source": "yolo_views",
            "fallback": False,
            "model_name": model.name,
            "model_sha256": model_sha256,
            "model_sha256_verified": True,
            "view_count": len(canonical_views),
            "consumer_allowed": CONSUMER_ALLOWED,
        }
        result = StrictYoloViewResult(
            view_boxes=tuple(canonical_views),
            status=status,
        )
    finally:
        cleanup_error: OSError | None = None
        active_error = sys.exc_info()[1]
        if temp_pdf is not None:
            try:
                Path(temp_pdf).unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = exc
        _emit_runtime_metrics(
            timing_sink,
            started=started,
            cleanup_failed=cleanup_error is not None,
        )
        if cleanup_error is not None and active_error is None:
            raise StrictYoloViewUnavailable(
                reason="temp_pdf_cleanup_failed",
                trace_id=trace_id,
                page_index=page_index,
            ) from cleanup_error
    if not control.finish_success():
        raise StrictYoloViewUnavailable(
            reason=control.cancellation_reason(),
            trace_id=trace_id,
            page_index=page_index,
        )
    return result


def _canonical_views(raw_views: Any, *, page_index: int) -> list[dict[str, Any]]:
    if not isinstance(raw_views, list):
        raise ValueError("YOLO view output must be a list")
    rows: list[dict[str, Any]] = []
    for raw in raw_views:
        if not isinstance(raw, dict):
            raise ValueError("YOLO view rows must be objects")
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["w"])
            height = float(raw["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("YOLO view row has an invalid bbox") from exc
        values = (x, y, width, height)
        if (
            not all(math.isfinite(value) for value in values)
            or x < 0.0
            or y < 0.0
            or width <= 0.0
            or height <= 0.0
        ):
            raise ValueError("YOLO view row has an invalid bbox")
        shape = raw.get("shape")
        if shape not in {"rect", "circle"}:
            raise ValueError("YOLO view row has an invalid shape")
        try:
            confidence = float(raw["conf"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("YOLO view row has an invalid confidence") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO view row has an invalid confidence")
        row = {
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "shape": str(shape),
            "conf": confidence,
            "backend": "yolo",
            "source": "yolo_views",
            "fallback": False,
        }
        if shape == "circle":
            try:
                cx = float(raw["cx"])
                cy = float(raw["cy"])
                radius = float(raw["r"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("YOLO circle view is incomplete") from exc
            if (
                not all(math.isfinite(value) for value in (cx, cy, radius))
                or cx < 0.0
                or cy < 0.0
                or radius <= 0.0
            ):
                raise ValueError("YOLO circle view is invalid")
            row.update({"cx": cx, "cy": cy, "r": radius})
        rows.append(row)
    rows.sort(
        key=lambda row: (
            round(float(row["x"]), 6),
            round(float(row["y"]), 6),
            round(float(row["w"]), 6),
            round(float(row["h"]), 6),
            str(row["shape"]),
        )
    )
    for index, row in enumerate(rows):
        row["view_id"] = f"p{int(page_index) + 1:03d}_view_{index:04d}"
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit_runtime_metrics(
    timing_sink: MutableMapping[str, float] | None,
    *,
    started: float,
    cleanup_failed: bool,
) -> None:
    if timing_sink is None:
        return
    try:
        timing_sink["yolo_view_wall_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        timing_sink["yolo_view_temp_cleanup_error_count"] = float(
            cleanup_failed
        )
    except Exception:
        return


def _terminate_process(process: subprocess.Popen[str]) -> bool:
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=1.0)
        return True
    except subprocess.TimeoutExpired:
        pass
    except OSError:
        pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1.0)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    """Release PIPE descriptors after the worker has left ``communicate``."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is None:
            continue
        try:
            stream.close()
        except Exception:
            pass
