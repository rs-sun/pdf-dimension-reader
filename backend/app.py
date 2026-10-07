"""
app.py — Flask 服务入口

端点:
  POST /analyze        — 分析 PDF 页面，返回结构化尺寸数据
  POST /prelabel_fast  — 秒级矢量预标 dump-only API（默认关闭）
  POST /analyze_stamp  — 手绘章定点 OCR（轻量，不走完整管线）
  GET  /health         — 健康检查
  POST /feedback       — 接收用户校验反馈，追加写入本地 JSONL

重构后：/analyze 的矢量+OCR+装配 orchestration 全部移入 pipeline.run_type_b_pipeline。
本文件只保留：HTTP 参数校验、base64 解码、pdfplumber 页面打开、线程安全 stdout、
错误处理、/analyze_stamp 独立轻量端点、/feedback JSONL 写入、静态文件。
"""

import base64
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile

import fitz  # PyMuPDF
import pdfplumber
from flask import Flask, Response, g, jsonify, request, send_file
from flask_cors import CORS
from openpyxl.utils.exceptions import InvalidFileException
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from final_consume_approval import resolve_final_consume_approval
from pdf_analyzer import detect_pdf_type, extract_type_a, close_fitz_cache
from assembler import assemble
from pipeline import run_type_b_pipeline
from recognition_settings import RecognitionSettings
from vector_layer_debug_contract import (
    build_vector_layer_debug_unavailable_v1,
    build_vector_layer_debug_v1,
)
from vector_layer_debug_request import VectorLayerDebugRequest
from ocr_engine_utils import get_selected_ocr_backend, ocr_engine_capabilities
from view_seg_router import view_seg_capabilities
from vector_prelabel_fast import run_vector_prelabel_fast_profile
from vector_only_contract import (
    StrictSuccessValidationSession,
    VectorOnlyStageUnavailable,
    fail_closed_payload,
    validate_r33_m1_review_api_projection,
    validate_vector_only_success_payload,
)
from review_candidates_contract import (
    CHAIN_AUDIT_FIELD,
    CHAIN_FIELD_KEYS,
    ROOT_KEY as REVIEW_CANDIDATES_ROOT_KEY,
    validate_review_candidates_v1,
)
from strict_gdt_candidates import AUDIT_FIELD as STRICT_GDT_AUDIT_FIELD
from strict_diagnostic_payload import (
    begin_strict_diagnostic_request_v1,
    merge_strict_diagnostic_sidecars_v1,
    publish_strict_pipeline_diagnostics_v1,
    publish_strict_response_diagnostics_v1,
    read_strict_diagnostic_field_v1,
    split_strict_pipeline_product_v1,
    split_strict_response_product_v1,
)
from r33_m1_phrase_runtime import (
    R33M1RuntimeUnavailable,
    load_r33_m1_phrase_runtime,
)
from directional_walk_pipeline import directional_walk_asset_identity_v1
from vector_only_runtime_guard import (
    VectorOnlyRuntimeForbidden,
    finalize_runtime_audits,
    ocr_singleton_snapshot,
    strict_vector_only_runtime_guard,
)
from yolo_view_async import (
    StrictYoloViewUnavailable,
    build_strict_yolo_trace_id,
    strict_yolo_view_expected_sha256,
    strict_yolo_view_model_path,
    strict_yolo_view_onnx_runner_path,
    strict_yolo_view_timeout_s,
    submit_strict_yolo_view,
)
from export_gold_xlsx import build_xlsx_from_bytes
from export_dim_override_consistency import (
    build_export_dim_override_consistency_v1,
    summarize_export_dim_override_consistency,
)
from inspection_importer import parse_uploaded_workbooks
from legacy_stamp_importer import extract_legacy_stamp_marks


# ── 启动期 stdout 捕获 ──────────────────────────────────────────────
# 接下来 YOLO-B / YOLO-A / OCR / config.summary 这些 print 全部 tee 进
# _STARTUP_LOG_BUF。Flask 起来后通过 /startup_log endpoint 给前端读，
# 跟 /analyze 的 debug_log 配合，覆盖「启动 + 调用」两个时间窗的所有日志。
# 用完后恢复原始 stdout，让下面的 _ThreadLocalTee 接管。
_STARTUP_LOG_BUF = io.StringIO()
_STARTUP_LOG_TEXT = None  # 启动结束后由 _close_startup_capture() 定型


class _StartupTee:
    def __init__(self, orig, buf):
        self._orig = orig
        self._buf  = buf
    def write(self, s):
        self._orig.write(s)
        self._buf.write(s)
    def flush(self):
        self._orig.flush()
    def isatty(self):
        return getattr(self._orig, 'isatty', lambda: False)()
    @property
    def encoding(self):
        return getattr(self._orig, 'encoding', 'utf-8')


_orig_stdout_at_startup = sys.stdout
sys.stdout = _StartupTee(_orig_stdout_at_startup, _STARTUP_LOG_BUF)


def _close_startup_capture():
    """启动收尾：定型 _STARTUP_LOG_TEXT，恢复原始 stdout。"""
    global _STARTUP_LOG_TEXT
    sys.stdout.flush()
    _STARTUP_LOG_TEXT = _STARTUP_LOG_BUF.getvalue()
    sys.stdout = _orig_stdout_at_startup


# ── 模型加载边界 ────────────────────────────────────────────────────
# 默认 lazy-load，避免 import app / endpoint 单测时加载 OCR、ONNX 或 Torch。
# 如生产环境希望启动时预热，可显式设置 PDF_READER_EAGER_LOAD_MODELS=1。
_EAGER_LOAD_MODELS = os.environ.get('PDF_READER_EAGER_LOAD_MODELS') == '1'
_gdt_detector = None
_gdt_detector_load_attempted = False
_model_status = {
    'eager_load_models': _EAGER_LOAD_MODELS,
    'gdt_loaded': False,
    'gdt_error': '',
    'ocr_eager_loaded': False,
    'ocr_eager_error': '',
    'ocr_backend': '',
    'ocr_backend_selected': '',
    'yolo_view_backend': os.environ.get('VIEW_SEG_BACKEND', 'yolo'),
}


try:
    from ocr_engine_utils import get_preferred_ocr_backend
    _model_status['ocr_backend'] = get_preferred_ocr_backend()
    print(f"[app] OCR backend configured: {_model_status['ocr_backend']} (lazy)")
except Exception as _e:
    print(f"[app] OCR backend configured lookup failed: {_e}")


def _safe_get_selected_ocr_backend() -> str:
    try:
        from ocr_engine_utils import get_selected_ocr_backend
        return get_selected_ocr_backend()
    except Exception:
        return ''


def gdt_capabilities(gdt_loaded: bool) -> dict:
    return {
        'hybrid': {'available': True, 'reason': 'ok'},
        'vector': {'available': True, 'reason': 'ok'},
        'yolo': {
            'available': bool(gdt_loaded),
            'reason': 'ok' if gdt_loaded else 'model_not_loaded',
        },
    }


def _get_gdt_detector():
    global _gdt_detector, _gdt_detector_load_attempted
    if _gdt_detector_load_attempted:
        return _gdt_detector
    _gdt_detector_load_attempted = True
    try:
        from gdt_detector import GDTDetector
        _gdt_detector = GDTDetector()
        _model_status['gdt_loaded'] = True
        print("[app] YOLO-B GD&T detector loaded")
    except Exception as e:
        _gdt_detector = None
        _model_status['gdt_error'] = str(e)
        print(f"[app] YOLO-B GD&T detector not available: {e}")
    return _gdt_detector


def _cancel_strict_yolo_task(task, reason: str) -> None:
    if task is None:
        return
    try:
        if task.done():
            return
    except Exception:
        pass
    try:
        task.cancel(reason)
    except Exception:
        pass


if _EAGER_LOAD_MODELS:
    try:
        import paddle  # noqa: F401
    except Exception as _e:
        print(f"[app] paddle 预 import 失败（PaddleOCR 仍可能 fallback rapidocr）: {_e}")
    _get_gdt_detector()
    try:
        from ocr_engine_utils import _get_ocr, get_selected_ocr_backend
        _get_ocr()
        _model_status['ocr_eager_loaded'] = True
        _model_status['ocr_backend_selected'] = get_selected_ocr_backend()
        _model_status['ocr_backend'] = _model_status['ocr_backend_selected'] or _model_status['ocr_backend']
        print(f"[app] OCR backend loaded: {_model_status['ocr_backend_selected']}")
    except Exception as e:
        _model_status['ocr_eager_error'] = str(e)
        print(f"[app] OCR backend eager-load 失败（将降级 lazy-load）: {e}")

# ── 启动日志捕获收尾：定型文本 + 恢复原始 stdout ─────────────────────
_close_startup_capture()


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = getattr(Config, 'MAX_UPLOAD_BYTES', 32 * 1024 * 1024)
CORS(app, origins=Config.CORS_ORIGINS)


def _max_request_body_bytes() -> int:
    return int(app.config['MAX_CONTENT_LENGTH'])


@app.before_request
def _begin_analyze_diagnostic_capture():
    """Reset an opt-in sidecar before every analyze outcome, including 413."""
    if request.path == '/analyze':
        begin_strict_diagnostic_request_v1()
    return None


@app.errorhandler(RequestEntityTooLarge)
def _request_body_too_large(_error):
    payload = {
        'schema_version': 'request_error_v1',
        'status': 'error',
        'code': 'request_body_too_large',
        'message': '请求体超过服务端上限',
        'max_request_body_bytes': _max_request_body_bytes(),
        'retryable': False,
    }
    if request.content_length is not None:
        payload['content_length'] = int(request.content_length)
    return jsonify(payload), 413


@app.before_request
def _enforce_request_body_limit_without_content_length():
    """Make the whole-body limit exact for terminated chunked WSGI input.

    Werkzeug cannot distinguish EOF from a max-length cutoff when a request has
    no Content-Length. A valid JSON prefix followed by excess whitespace can
    therefore appear valid at exactly the cutoff. Buffer at most limit + 1
    bytes before Flask parses such a body, then restore a bounded input stream.
    """
    if request.method not in {'POST', 'PUT', 'PATCH'}:
        return None

    limit = _max_request_body_bytes()
    content_length = request.content_length
    if content_length is not None:
        if content_length > limit:
            raise RequestEntityTooLarge()
        return None
    if not request.environ.get('wsgi.input_terminated'):
        return None

    source = request.environ.get('wsgi.input')
    if source is None:
        return None
    chunks = []
    remaining = limit + 1
    while remaining > 0:
        chunk = source.read(min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    body = b''.join(chunks)
    if len(body) > limit:
        raise RequestEntityTooLarge()

    request.environ['wsgi.input'] = io.BytesIO(body)
    request.environ['CONTENT_LENGTH'] = str(len(body))
    request.environ.pop('wsgi.input_terminated', None)
    return None

# Flask 单 worker 模式（见 CLAUDE.md 部署注意），跨线程写 feedback.jsonl 用这把锁
_feedback_lock = threading.Lock()

# ── 运行时计数器（每次 endpoint 命中 +1，前端拉 /runtime_stats 读快照）──
_runtime_stats_lock = threading.Lock()
_runtime_stats = {
    'started_at':          datetime.now().isoformat(),
    'started_ts':          time.time(),
    'analyze_calls':       0,
    'analyze_success_total': 0,
    'analyze_error_total': 0,
    'analyze_success_by_pdf_type': {},
    'analyze_errors_by_status': {},
    'analyze_errors_by_reason': {},
    'analyze_errors_by_status_reason': {},
    'analyze_latency_ms': {
        'last': 0.0,
        'max': 0.0,
    },
    'analyze_last_result': {
        'pdf_type': '',
        'page_number': 0,
        'pdf_pages': 0,
        'dimensions': 0,
        'basic_dimensions': 0,
        'references': 0,
        'gdt_frames': 0,
        'notes': 0,
        'dimension_candidates': 0,
        'candidate_drops': 0,
        'ocr_results': 0,
        'view_boxes': 0,
    },
    'analyze_result_max': {
        'pdf_pages': 0,
        'dimensions': 0,
        'basic_dimensions': 0,
        'references': 0,
        'gdt_frames': 0,
        'notes': 0,
        'dimension_candidates': 0,
        'candidate_drops': 0,
        'ocr_results': 0,
        'view_boxes': 0,
    },
    'analyze_ocr_breakdown_last': {},
    'analyze_ocr_breakdown_total': {},
    'analyze_ocr_pass_counters_last': {},
    'analyze_ocr_pass_counters_total': {},
    'analyze_shadow_stats_last': {},
    'analyze_shadow_stats_total': {},
    'analyze_stamp_calls': 0,
    'feedback_calls': 0,
    'feedback_inserted_total': 0,
    'feedback_error_total': 0,
    'feedback_errors_by_status_reason': {},
    'feedback_last_inserted': 0,
    'export_calls':        0,
    'import_measurements_calls': 0,
    'import_legacy_stamps_calls': 0,
    'http_requests_total': 0,
    'http_request_errors_total': 0,
    'http_requests_by_status': {},
    'http_requests_by_route': {},
    'http_latency_ms': {
        'last': 0.0,
        'max': 0.0,
    },
}


def _bump(key, n=1):
    with _runtime_stats_lock:
        _runtime_stats[key] = _runtime_stats.get(key, 0) + n


def _runtime_stats_snapshot() -> dict:
    with _runtime_stats_lock:
        return copy.deepcopy(_runtime_stats)


def _record_http_metric(endpoint: str, method: str, status_code: int, elapsed_sec: float):
    status = int(status_code)
    status_key = str(status)
    route_key = f'{method} {endpoint}'
    latency_ms = round(max(0.0, float(elapsed_sec)) * 1000.0, 3)
    with _runtime_stats_lock:
        _runtime_stats['http_requests_total'] += 1
        if status >= 400:
            _runtime_stats['http_request_errors_total'] += 1
        status_counts = _runtime_stats.setdefault('http_requests_by_status', {})
        status_counts[status_key] = int(status_counts.get(status_key, 0)) + 1
        latency = _runtime_stats.setdefault('http_latency_ms', {'last': 0.0, 'max': 0.0})
        latency['last'] = latency_ms
        latency['max'] = max(float(latency.get('max') or 0.0), latency_ms)
        by_route = _runtime_stats.setdefault('http_requests_by_route', {})
        route = by_route.setdefault(route_key, {
            'endpoint': endpoint,
            'method': method,
            'count': 0,
            'error_count': 0,
            'status_counts': {},
            'last_latency_ms': 0.0,
            'max_latency_ms': 0.0,
        })
        route['count'] = int(route.get('count') or 0) + 1
        if status >= 400:
            route['error_count'] = int(route.get('error_count') or 0) + 1
        route_status_counts = route.setdefault('status_counts', {})
        route_status_counts[status_key] = int(route_status_counts.get(status_key, 0)) + 1
        route['last_latency_ms'] = latency_ms
        route['max_latency_ms'] = max(float(route.get('max_latency_ms') or 0.0), latency_ms)


def _count_items(value: object) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 0


def _numeric_value(value: object) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _flatten_numeric_stats(value: object, *, prefix: str = '', depth: int = 0) -> dict[str, float | int]:
    if depth > 3:
        return {}
    numeric = _numeric_value(value)
    if numeric is not None:
        return {prefix or 'value': numeric}
    if not isinstance(value, dict):
        return {}
    out: dict[str, float | int] = {}
    for raw_key, raw_value in value.items():
        key = re.sub(r'[^A-Za-z0-9_]+', '_', str(raw_key).strip()).strip('_') or 'unknown'
        nested_key = f'{prefix}_{key}' if prefix else key
        out.update(_flatten_numeric_stats(raw_value, prefix=nested_key, depth=depth + 1))
    return out


def _add_numeric_totals(total_key: str, last_key: str, values: dict[str, float | int]) -> None:
    last = _runtime_stats.setdefault(last_key, {})
    last.clear()
    last.update(values)
    totals = _runtime_stats.setdefault(total_key, {})
    for key, value in values.items():
        totals[key] = totals.get(key, 0) + value


def _pipe_shadow_numeric_stats(
    pipe: dict | None,
    diagnostics: dict | None = None,
) -> dict[str, float | int]:
    if not pipe:
        return {}
    stat_fields = (
        'vector_glyph_stats_v1',
        'anchor_corridor_stats_v1',
        'anchor_phrase_stats_v1',
        'vector_numeric_phrase_stats_v1',
        'r2n_nominal_rescue_shadow_stats_v1',
        'ocr_crop_oriented_quad_stats_v1',
        'ocr_pass_necessity_stats_v1',
        'assembler_polygon_compat_stats_v1',
    )
    out: dict[str, float | int] = {}
    for field in stat_fields:
        field_value = read_strict_diagnostic_field_v1(
            pipe,
            diagnostics,
            field,
        )
        for key, value in _flatten_numeric_stats(field_value).items():
            out[f'{field}_{key}'] = value
    return out


def _record_analyze_success(
    *,
    pdf_type: str,
    page_number: int,
    pdf_pages: int,
    elapsed_sec: float,
    payload: dict,
    pipe: dict | None = None,
    pipeline_diagnostics: dict | None = None,
    response_diagnostics: dict | None = None,
) -> None:
    latency_ms = round(max(0.0, float(elapsed_sec)) * 1000.0, 3)
    response_diagnostics = response_diagnostics or {}

    def _payload_field(key):
        return read_strict_diagnostic_field_v1(
            payload,
            response_diagnostics,
            key,
        )

    result_counts = {
        'pdf_pages': int(pdf_pages),
        'dimensions': _count_items(payload.get('dimensions')),
        'basic_dimensions': _count_items(payload.get('basic_dimensions')),
        'references': _count_items(payload.get('references')),
        'gdt_frames': _count_items(_payload_field('gdt_frames')),
        'notes': _count_items(payload.get('notes')),
        'dimension_candidates': _count_items(
            _payload_field('dimension_candidates')
        ),
        'candidate_drops': _count_items(
            _payload_field('candidate_drop_ledger')
        ),
        'ocr_results': _count_items((pipe or {}).get('ocr_results')),
        'view_boxes': _count_items(payload.get('view_boxes')),
    }
    pipe = pipe or {}
    ocr_breakdown = _flatten_numeric_stats(pipe.get('ocr_breakdown') or payload.get('ocr_breakdown'))
    ocr_pass_counters = _flatten_numeric_stats(
        pipe.get('ocr_pass_counters') or payload.get('ocr_pass_counters')
    )
    shadow_stats = _pipe_shadow_numeric_stats(
        pipe,
        pipeline_diagnostics,
    )
    with _runtime_stats_lock:
        _runtime_stats['analyze_success_total'] += 1
        by_type = _runtime_stats.setdefault('analyze_success_by_pdf_type', {})
        type_key = str(pdf_type or 'unknown')
        by_type[type_key] = int(by_type.get(type_key, 0)) + 1
        latency = _runtime_stats.setdefault('analyze_latency_ms', {'last': 0.0, 'max': 0.0})
        latency['last'] = latency_ms
        latency['max'] = max(float(latency.get('max') or 0.0), latency_ms)
        last = _runtime_stats.setdefault('analyze_last_result', {})
        last.clear()
        last.update({
            'pdf_type': type_key,
            'page_number': int(page_number),
            **result_counts,
        })
        max_counts = _runtime_stats.setdefault('analyze_result_max', {})
        for key, value in result_counts.items():
            max_counts[key] = max(int(max_counts.get(key) or 0), int(value))
        _add_numeric_totals(
            'analyze_ocr_breakdown_total',
            'analyze_ocr_breakdown_last',
            ocr_breakdown,
        )
        _add_numeric_totals(
            'analyze_ocr_pass_counters_total',
            'analyze_ocr_pass_counters_last',
            ocr_pass_counters,
        )
        _add_numeric_totals(
            'analyze_shadow_stats_total',
            'analyze_shadow_stats_last',
            shadow_stats,
        )


def _record_analyze_error(*, status_code: int, reason: str, elapsed_sec: float) -> None:
    latency_ms = round(max(0.0, float(elapsed_sec)) * 1000.0, 3)
    status_key = str(int(status_code))
    reason_key = str(reason or 'unknown')
    with _runtime_stats_lock:
        _runtime_stats['analyze_error_total'] += 1
        by_status = _runtime_stats.setdefault('analyze_errors_by_status', {})
        by_status[status_key] = int(by_status.get(status_key, 0)) + 1
        by_reason = _runtime_stats.setdefault('analyze_errors_by_reason', {})
        by_reason[reason_key] = int(by_reason.get(reason_key, 0)) + 1
        by_status_reason = _runtime_stats.setdefault('analyze_errors_by_status_reason', {})
        combo_key = f'{status_key} {reason_key}'
        combo = by_status_reason.setdefault(combo_key, {
            'status': status_key,
            'reason': reason_key,
            'count': 0,
        })
        combo['count'] = int(combo.get('count') or 0) + 1
        latency = _runtime_stats.setdefault('analyze_latency_ms', {'last': 0.0, 'max': 0.0})
        latency['last'] = latency_ms
        latency['max'] = max(float(latency.get('max') or 0.0), latency_ms)


def _record_feedback_success(inserted: int) -> None:
    with _runtime_stats_lock:
        _runtime_stats['feedback_inserted_total'] += int(inserted)
        _runtime_stats['feedback_last_inserted'] = int(inserted)


def _record_feedback_error(*, status_code: int, reason: str) -> None:
    status_key = str(int(status_code))
    reason_key = str(reason or 'unknown')
    with _runtime_stats_lock:
        _runtime_stats['feedback_error_total'] += 1
        by_status_reason = _runtime_stats.setdefault('feedback_errors_by_status_reason', {})
        combo_key = f'{status_key} {reason_key}'
        combo = by_status_reason.setdefault(combo_key, {
            'status': status_key,
            'reason': reason_key,
            'count': 0,
        })
        combo['count'] = int(combo.get('count') or 0) + 1


def _prom_label(value: object) -> str:
    return str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def _runtime_metrics_text() -> str:
    snap = _runtime_stats_snapshot()
    uptime = int(time.time() - float(snap.get('started_ts') or time.time()))
    lines = [
        '# HELP pdf_reader_uptime_seconds Process uptime in seconds.',
        '# TYPE pdf_reader_uptime_seconds gauge',
        f'pdf_reader_uptime_seconds {uptime}',
        '# HELP pdf_reader_http_requests_total HTTP requests by endpoint, method, and status.',
        '# TYPE pdf_reader_http_requests_total counter',
    ]
    for route in sorted((snap.get('http_requests_by_route') or {}).values(), key=lambda row: (row.get('endpoint'), row.get('method'))):
        endpoint = _prom_label(route.get('endpoint') or 'unknown')
        method = _prom_label(route.get('method') or 'UNKNOWN')
        for status, count in sorted((route.get('status_counts') or {}).items()):
            lines.append(
                f'pdf_reader_http_requests_total{{endpoint="{endpoint}",method="{method}",status="{_prom_label(status)}"}} {int(count)}'
            )
    lines.extend([
        '# HELP pdf_reader_http_request_errors_total HTTP requests with status >= 400.',
        '# TYPE pdf_reader_http_request_errors_total counter',
        f'pdf_reader_http_request_errors_total {int(snap.get("http_request_errors_total") or 0)}',
        '# HELP pdf_reader_http_request_latency_ms_max Max observed HTTP request latency in milliseconds by endpoint and method.',
        '# TYPE pdf_reader_http_request_latency_ms_max gauge',
    ])
    for route in sorted((snap.get('http_requests_by_route') or {}).values(), key=lambda row: (row.get('endpoint'), row.get('method'))):
        endpoint = _prom_label(route.get('endpoint') or 'unknown')
        method = _prom_label(route.get('method') or 'UNKNOWN')
        lines.append(
            f'pdf_reader_http_request_latency_ms_max{{endpoint="{endpoint}",method="{method}"}} {float(route.get("max_latency_ms") or 0.0):.3f}'
        )
    lines.extend([
        '# HELP pdf_reader_analyze_calls_total Analyze endpoint calls.',
        '# TYPE pdf_reader_analyze_calls_total counter',
        f'pdf_reader_analyze_calls_total {int(snap.get("analyze_calls") or 0)}',
        '# HELP pdf_reader_analyze_success_total Successful analyze calls by detected PDF type.',
        '# TYPE pdf_reader_analyze_success_total counter',
    ])
    success_by_type = snap.get('analyze_success_by_pdf_type') or {}
    if not success_by_type:
        lines.append('pdf_reader_analyze_success_total{pdf_type="unknown"} 0')
    else:
        for pdf_type, count in sorted(success_by_type.items()):
            lines.append(
                f'pdf_reader_analyze_success_total{{pdf_type="{_prom_label(pdf_type)}"}} {int(count)}'
            )
    lines.extend([
        '# HELP pdf_reader_analyze_errors_total Failed analyze calls by HTTP status and reason.',
        '# TYPE pdf_reader_analyze_errors_total counter',
    ])
    errors_by_status_reason = snap.get('analyze_errors_by_status_reason') or {}
    if not errors_by_status_reason:
        lines.append('pdf_reader_analyze_errors_total{status="0",reason="none"} 0')
    else:
        for row in sorted(
            errors_by_status_reason.values(),
            key=lambda item: (item.get('status'), item.get('reason')),
        ):
            status = row.get('status') or 'unknown'
            reason = row.get('reason') or 'unknown'
            lines.append(
                f'pdf_reader_analyze_errors_total{{status="{_prom_label(status)}",reason="{_prom_label(reason)}"}} {int(row.get("count") or 0)}'
            )
    analyze_latency = snap.get('analyze_latency_ms') or {}
    lines.extend([
        '# HELP pdf_reader_analyze_latency_ms_last Last analyze latency in milliseconds.',
        '# TYPE pdf_reader_analyze_latency_ms_last gauge',
        f'pdf_reader_analyze_latency_ms_last {float(analyze_latency.get("last") or 0.0):.3f}',
        '# HELP pdf_reader_analyze_latency_ms_max Max observed analyze latency in milliseconds.',
        '# TYPE pdf_reader_analyze_latency_ms_max gauge',
        f'pdf_reader_analyze_latency_ms_max {float(analyze_latency.get("max") or 0.0):.3f}',
        '# HELP pdf_reader_analyze_result_count_last Last analyze output counts by kind.',
        '# TYPE pdf_reader_analyze_result_count_last gauge',
    ])
    last_result = snap.get('analyze_last_result') or {}
    for key, value in sorted(last_result.items()):
        if isinstance(value, (int, float)) and key != 'page_number':
            lines.append(
                f'pdf_reader_analyze_result_count_last{{kind="{_prom_label(key)}"}} {int(value)}'
            )
    lines.extend([
        '# HELP pdf_reader_analyze_result_count_max Max observed analyze output counts by kind.',
        '# TYPE pdf_reader_analyze_result_count_max gauge',
    ])
    for key, value in sorted((snap.get('analyze_result_max') or {}).items()):
        lines.append(
            f'pdf_reader_analyze_result_count_max{{kind="{_prom_label(key)}"}} {int(value)}'
        )
    lines.extend([
        '# HELP pdf_reader_analyze_ocr_breakdown_total OCR breakdown counts accumulated from analyze responses.',
        '# TYPE pdf_reader_analyze_ocr_breakdown_total counter',
    ])
    ocr_breakdown_total = snap.get('analyze_ocr_breakdown_total') or {}
    if not ocr_breakdown_total:
        lines.append('pdf_reader_analyze_ocr_breakdown_total{kind="none"} 0')
    else:
        for key, value in sorted(ocr_breakdown_total.items()):
            lines.append(
                f'pdf_reader_analyze_ocr_breakdown_total{{kind="{_prom_label(key)}"}} {float(value):.6g}'
            )
    lines.extend([
        '# HELP pdf_reader_analyze_ocr_pass_counter_total OCR pass counters accumulated from analyze responses.',
        '# TYPE pdf_reader_analyze_ocr_pass_counter_total counter',
    ])
    ocr_pass_total = snap.get('analyze_ocr_pass_counters_total') or {}
    if not ocr_pass_total:
        lines.append('pdf_reader_analyze_ocr_pass_counter_total{kind="none"} 0')
    else:
        for key, value in sorted(ocr_pass_total.items()):
            lines.append(
                f'pdf_reader_analyze_ocr_pass_counter_total{{kind="{_prom_label(key)}"}} {float(value):.6g}'
            )
    lines.extend([
        '# HELP pdf_reader_analyze_shadow_stat_total Shadow diagnostic numeric stats accumulated from analyze responses.',
        '# TYPE pdf_reader_analyze_shadow_stat_total counter',
    ])
    shadow_total = snap.get('analyze_shadow_stats_total') or {}
    if not shadow_total:
        lines.append('pdf_reader_analyze_shadow_stat_total{kind="none"} 0')
    else:
        for key, value in sorted(shadow_total.items()):
            lines.append(
                f'pdf_reader_analyze_shadow_stat_total{{kind="{_prom_label(key)}"}} {float(value):.6g}'
            )
    lines.extend([
        '# HELP pdf_reader_feedback_calls_total Feedback endpoint calls.',
        '# TYPE pdf_reader_feedback_calls_total counter',
        f'pdf_reader_feedback_calls_total {int(snap.get("feedback_calls") or 0)}',
        '# HELP pdf_reader_feedback_inserted_total Feedback records appended to JSONL.',
        '# TYPE pdf_reader_feedback_inserted_total counter',
        f'pdf_reader_feedback_inserted_total {int(snap.get("feedback_inserted_total") or 0)}',
        '# HELP pdf_reader_feedback_errors_total Feedback endpoint errors by status and reason.',
        '# TYPE pdf_reader_feedback_errors_total counter',
    ])
    feedback_errors = snap.get('feedback_errors_by_status_reason') or {}
    if not feedback_errors:
        lines.append('pdf_reader_feedback_errors_total{status="0",reason="none"} 0')
    else:
        for row in sorted(feedback_errors.values(), key=lambda item: (item.get('status'), item.get('reason'))):
            status = row.get('status') or 'unknown'
            reason = row.get('reason') or 'unknown'
            lines.append(
                f'pdf_reader_feedback_errors_total{{status="{_prom_label(status)}",reason="{_prom_label(reason)}"}} {int(row.get("count") or 0)}'
            )
    lines.extend([
        '# HELP pdf_reader_feedback_last_inserted Last feedback inserted record count.',
        '# TYPE pdf_reader_feedback_last_inserted gauge',
        f'pdf_reader_feedback_last_inserted {int(snap.get("feedback_last_inserted") or 0)}',
    ])
    return '\n'.join(lines) + '\n'


@app.before_request
def _mark_request_start():
    g.pdf_reader_request_started_ts = time.time()


@app.after_request
def _record_request_metric(response):
    started = getattr(g, 'pdf_reader_request_started_ts', None)
    elapsed = time.time() - started if started is not None else 0.0
    _record_http_metric(
        endpoint=str(request.endpoint or 'unknown'),
        method=str(request.method or 'UNKNOWN'),
        status_code=int(response.status_code),
        elapsed_sec=elapsed,
    )
    return response


# ── 线程安全 stdout 代理（多请求并发各自独立捕获 print 输出） ────────
_tee_local = threading.local()


class _ThreadLocalTee:
    """安装到 sys.stdout 的全局代理。

    每个线程可通过 ``_tee_local.log_buffer = io.StringIO()`` 开启日志
    捕获；未设置时 print 正常输出到原始 stdout。
    """

    def __init__(self, original):
        self._original = original

    def write(self, text):
        self._original.write(text)
        buf = getattr(_tee_local, 'log_buffer', None)
        if buf is not None:
            buf.write(text)

    def flush(self):
        self._original.flush()
        buf = getattr(_tee_local, 'log_buffer', None)
        if buf is not None:
            buf.flush()

    def isatty(self):
        return hasattr(self._original, 'isatty') and self._original.isatty()

    @property
    def encoding(self):
        return getattr(self._original, 'encoding', 'utf-8')


sys.stdout = _ThreadLocalTee(sys.__stdout__)


# ── 静态文件 ──────────────────────────────────────────────────────

_PRODUCT_STATIC_ASSETS = frozenset({
    'index.html',
    'styles.css',
    'app.js',
    'app_analysis.js',
    'app_cursor.js',
    'app_diagnosis.js',
    'app_interaction.js',
    'app_io.js',
    'app_sidebar.js',
    'app_utils.js',
    'api_client.js',
    'canvas_renderer.js',
    'config.js',
    'dbscan.js',
    'dimension_stamp_projection.js',
    'history_snapshot.js',
    'import_export_transaction.js',
    'inspection_matching.js',
    'legacy_pdf_transport.js',
    'motion_preferences.js',
    'page_coordinates.js',
    'reanalysis_transaction.js',
    'review_candidate_promotion.js',
    'review_candidates_v1.js',
    'review_keyboard.js',
    'review_workspace_guard.js',
    'snapshot_pdf_renderer.js',
    'vector_debug_drawer.js',
    'vector_layer_debug_v1.js',
    'stamp.js',
    'state.js',
    'app_sidebar/control_panel.js',
    'app_sidebar/export_affordance.js',
    'app_sidebar/expand_form.js',
    'app_sidebar/list.js',
    'app_sidebar/popup.js',
    'app_sidebar/review_workbench.js',
    'app_sidebar/shared_templates.js',
    'app_sidebar/symbol_palette.js',
    'libs/jszip.min.js',
    'libs/pdf-lib.min.js',
    'libs/pdf.min.js',
    'libs/pdf.worker.min.js',
})


def _resolve_product_static_asset(filename: str) -> Path | None:
    """Resolve one explicitly published frontend asset inside the frontend root."""
    logical_name = str(filename or '')
    if (
        logical_name not in _PRODUCT_STATIC_ASSETS
        or '\\' in logical_name
        or '%' in logical_name
    ):
        return None
    try:
        frontend_root = Path(Config.FRONTEND_DIR).resolve(strict=True)
        candidate = frontend_root
        for part in Path(logical_name).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(frontend_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _serve_product_static_asset(filename: str):
    path = _resolve_product_static_asset(filename)
    if path is None:
        return 'Not Found', 404
    return send_file(path)

@app.route('/')
def serve_index():
    return _serve_product_static_asset('index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return _serve_product_static_asset(filename)


# ── 健康检查 ──────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'pdf_reader',
        'uptime_sec': int(time.time() - _runtime_stats['started_ts']),
    })


def _feedback_log_check() -> dict:
    path = Config.FEEDBACK_LOG_PATH
    directory = os.path.dirname(path) or '.'
    ok = os.path.isdir(directory) and os.access(directory, os.W_OK)
    return {
        'name': 'feedback_log_directory_writable',
        'ok': ok,
        'reason_code': None if ok else 'feedback_log_directory_unwritable',
        'path': path,
        'directory': directory,
    }


def _final_consume_check() -> dict:
    decision = resolve_final_consume_approval()
    check = {
        'name': 'final_consume_requires_explicit_approval',
        'ok': decision.policy_valid,
        **decision.as_dict(),
        # Preserve the existing readiness field for production smoke readers.
        'approved': decision.explicit_approval,
    }
    return check


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_runtime_assets_check() -> dict:
    model_path = strict_yolo_view_model_path()
    expected_sha256 = str(strict_yolo_view_expected_sha256() or '').lower()
    onnx_runner_path = strict_yolo_view_onnx_runner_path()
    fallback_setting = str(os.environ.get('YOLO_ALLOW_PT_FALLBACK') or '').strip().lower()
    pt_fallback_requested = fallback_setting not in {'', '0', 'false', 'no', 'off'}
    actual_sha256 = None
    reason_code = None

    if model_path.suffix.lower() != '.onnx':
        reason_code = 'strict_yolo_model_format_forbidden'
    elif pt_fallback_requested:
        reason_code = 'strict_yolo_pt_fallback_forbidden'
    elif not model_path.is_file():
        reason_code = 'strict_yolo_model_missing'
    elif not onnx_runner_path.is_file():
        reason_code = 'strict_yolo_onnx_runner_missing'
    elif (
        len(expected_sha256) != 64
        or any(character not in '0123456789abcdef' for character in expected_sha256)
    ):
        reason_code = 'strict_yolo_model_sha256_unconfigured'
    else:
        try:
            actual_sha256 = _sha256_file(model_path)
        except OSError:
            reason_code = 'strict_yolo_model_unreadable'
        if reason_code is None and actual_sha256 != expected_sha256:
            reason_code = 'strict_yolo_model_sha256_mismatch'

    return {
        'name': 'strict_runtime_assets',
        'ok': reason_code is None,
        'reason_code': reason_code,
        'model_path': str(model_path),
        'expected_model_sha256': expected_sha256 or None,
        'actual_model_sha256': actual_sha256,
        'onnx_runner_path': str(onnx_runner_path),
        'pt_fallback_requested': pt_fallback_requested,
    }


def _r33_m1_template_runtime_check() -> dict:
    """Validate the same controlled template runtime used by vector-only analyze."""
    reason_code = None
    template_version = None
    content_sha256 = None
    try:
        runtime = load_r33_m1_phrase_runtime(
            Config.R33_M1_TEMPLATE_LIBRARY_PATH,
        )
    except R33M1RuntimeUnavailable as exc:
        reason_code = exc.reason
    except Exception:
        # Readiness must stay structured and must not expose loader internals.
        # Controlled input failures already carry a specific public reason.
        reason_code = 'template_runtime_unavailable'
    else:
        identity = (
            runtime.template_identity
            if isinstance(runtime.template_identity, dict)
            else {}
        )
        template_version = str(identity.get('template_version') or '').strip() or None
        content_sha256 = str(identity.get('content_sha256') or '').strip().lower() or None
        if (
            template_version is None
            or content_sha256 is None
            or len(content_sha256) != 64
            or any(character not in '0123456789abcdef' for character in content_sha256)
        ):
            reason_code = 'template_runtime_identity_invalid'
            template_version = None
            content_sha256 = None

    return {
        'name': 'r33_m1_template_runtime',
        'ok': reason_code is None,
        'reason_code': reason_code,
        'template_version': template_version,
        'content_sha256': content_sha256,
    }


def _directional_walk_assets_check() -> dict:
    """Load and digest-check the exact assets used by the R92 producer."""

    reason_code = None
    identity = None
    try:
        identity = directional_walk_asset_identity_v1()
    except Exception:
        reason_code = "directional_walk_assets_invalid"
    if reason_code is None:
        try:
            runtime = load_r33_m1_phrase_runtime(
                Config.R33_M1_TEMPLATE_LIBRARY_PATH,
            )
        except R33M1RuntimeUnavailable:
            # The existing template check owns unavailable/unconfigured
            # reporting.  Bind identities only when both sides loaded.
            pass
        except Exception:
            pass
        else:
            runtime_identity = (
                runtime.template_identity
                if isinstance(runtime.template_identity, dict)
                else {}
            )
            if runtime_identity.get("content_sha256") != identity.get(
                "controlled_template_sha256"
            ):
                reason_code = "directional_walk_template_identity_mismatch"
    return {
        "name": "directional_walk_assets",
        "ok": reason_code is None,
        "reason_code": reason_code,
        "identity": identity,
    }


def _strict_forbidden_runtime_check(
    *,
    ocr_snapshot: tuple[bool, frozenset[str]] | None = None,
) -> dict:
    paddle_loaded, ocr_backends = (
        ocr_singleton_snapshot()
        if ocr_snapshot is None
        else ocr_snapshot
    )
    forbidden_components = []
    if paddle_loaded:
        forbidden_components.append('ocr:paddle')
    forbidden_components.extend(
        f'ocr:{backend}' for backend in sorted(ocr_backends)
    )
    if _model_status.get('ocr_eager_loaded') and not ocr_backends:
        forbidden_components.append('ocr:eager')
    if _model_status.get('gdt_loaded') or _gdt_detector is not None:
        forbidden_components.append('yolo_b')
    configured_view_backend = str(_model_status.get('yolo_view_backend') or '')
    if configured_view_backend != 'yolo':
        forbidden_components.append(
            f'yolo_a_fallback:{configured_view_backend or "unknown"}'
        )

    return {
        'name': 'strict_forbidden_runtime',
        'ok': not forbidden_components,
        'reason_code': (
            None if not forbidden_components else 'strict_forbidden_runtime_loaded'
        ),
        'forbidden_components': forbidden_components,
        'configured_view_backend': configured_view_backend,
    }


def _vector_only_preloaded_runtime_failure(
    runtime_check: dict,
    *,
    trace_id: str | None,
    page_index: int,
) -> tuple[dict, str]:
    reason_code = str(
        runtime_check.get('reason_code')
        or 'strict_forbidden_runtime_loaded'
    )
    payload = fail_closed_payload({
        'schema_version': 'vector_only_error_v1',
        'code': 'vector_only_preloaded_runtime_forbidden',
        'status': 'fail_closed',
        'recovery': 'restart_backend_required',
        'reasons': [reason_code],
        'forbidden_components': list(
            runtime_check.get('forbidden_components') or []
        ),
        'trace_id': trace_id,
        'page_index': page_index,
        'requested_backend': 'yolo',
        'effective_backend': 'unavailable',
        'fallback': False,
    })
    return payload, reason_code


def _r33_m1_review_projection_failure(
    source: dict,
    *,
    projection_status: str,
    reasons: list[str],
    trace_id: str,
    page_index: int,
) -> tuple[dict, str]:
    """Build an error response without repairing or translating candidates."""
    code = (
        'r33_m1_review_projection_unavailable'
        if projection_status == 'unavailable'
        else 'r33_m1_review_projection_violation'
    )
    chain_audit = source.get(CHAIN_AUDIT_FIELD)
    error_payload = {
        'schema_version': 'r33_m1_review_projection_error_v1',
        'code': code,
        'status': 'fail_closed',
        'stage': 'api_projection',
        'reasons': sorted(set(str(reason) for reason in reasons)),
        'trace_id': trace_id,
        'page_index': page_index,
        'requested_backend': 'yolo',
        'effective_backend': 'unavailable',
        'fallback': False,
    }
    if (
        projection_status == 'unavailable'
        and isinstance(chain_audit, dict)
        and isinstance(chain_audit.get('details'), dict)
        and chain_audit['details']
    ):
        error_payload['details'] = copy.deepcopy(chain_audit['details'])
    payload = fail_closed_payload(error_payload)
    envelope = source.get(REVIEW_CANDIDATES_ROOT_KEY)
    if validate_review_candidates_v1(envelope) == []:
        payload[REVIEW_CANDIDATES_ROOT_KEY] = envelope
    if projection_status == 'unavailable' and isinstance(chain_audit, dict):
        payload[CHAIN_AUDIT_FIELD] = chain_audit
    return payload, code


def _eager_model_check() -> dict:
    eager_errors = []
    if _model_status.get('ocr_eager_error'):
        eager_errors.append({'component': 'ocr', 'error': _model_status['ocr_eager_error']})
    if _model_status.get('gdt_error') and _EAGER_LOAD_MODELS:
        eager_errors.append({'component': 'gdt', 'error': _model_status['gdt_error']})
    return {
        'name': 'eager_model_load',
        'ok': not eager_errors,
        'reason_code': None if not eager_errors else 'eager_model_load_failed',
        'eager_load_models': bool(_EAGER_LOAD_MODELS),
        'errors': eager_errors,
    }


def _transport_limits_payload() -> dict:
    return {
        'schema_version': 'transport_limits_v1',
        'legacy_json_base64': {
            'supported': True,
            'max_request_body_bytes': _max_request_body_bytes(),
            'size_includes': 'entire_http_body',
            'encoding': 'rfc4648_base64_in_utf8_json',
        },
    }


def _readiness_payload() -> dict:
    max_request_body_bytes = _max_request_body_bytes()
    checks = [
        {
            'name': 'upload_limit_configured',
            'ok': max_request_body_bytes > 0,
            'reason_code': (
                None
                if max_request_body_bytes > 0
                else 'upload_limit_invalid'
            ),
            'max_upload_bytes': max_request_body_bytes,
        },
        _feedback_log_check(),
        _final_consume_check(),
        _strict_runtime_assets_check(),
        _r33_m1_template_runtime_check(),
        _directional_walk_assets_check(),
        _strict_forbidden_runtime_check(),
        _eager_model_check(),
    ]
    ready = all(bool(check.get('ok')) for check in checks)
    reason_codes = [
        str(check['reason_code'])
        for check in checks
        if not check.get('ok') and check.get('reason_code')
    ]
    return {
        'status': 'ready' if ready else 'not_ready',
        'service': 'pdf_reader',
        'app_env': Config.APP_ENV,
        'production_mode': bool(Config.PRODUCTION_MODE),
        'reason_codes': reason_codes,
        'checks': checks,
        'transport_limits': _transport_limits_payload(),
        'components': {
            'ocr_backend': _safe_get_selected_ocr_backend() or _model_status['ocr_backend'],
            'ocr_engines': ocr_engine_capabilities(),
            'view_seg': view_seg_capabilities(),
            'gdt': gdt_capabilities(bool(_model_status['gdt_loaded'])),
            'gdt_loaded': bool(_model_status['gdt_loaded']),
            'gdt_error': _model_status['gdt_error'],
            'yolo_view_backend': _model_status['yolo_view_backend'],
        },
    }


@app.route('/readyz', methods=['GET'])
def readyz():
    payload = _readiness_payload()
    status_code = 200 if payload['status'] == 'ready' else 503
    return jsonify(payload), status_code


# ── 启动日志（YOLO-B / YOLO-A / OCR backend / config 摘要 等）──────────
@app.route('/startup_log', methods=['GET'])
def get_startup_log():
    """前端 download-log 按钮拉这个，把后端启动日志一起塞进 JSON 归档。"""
    text = _STARTUP_LOG_TEXT or ''
    return jsonify({
        'log':   text,
        'bytes': len(text),
        'lines': text.count('\n'),
        'components': {
            'yolo_b_loaded':   _model_status['gdt_loaded'],
            'yolo_b_error':    _model_status['gdt_error'],
            'yolo_a_backend':  _model_status['yolo_view_backend'],
            'ocr_eager_loaded': _model_status['ocr_eager_loaded'],
            'ocr_eager_error':  _model_status['ocr_eager_error'],
            'ocr_backend':     (_safe_get_selected_ocr_backend()
                                or _model_status['ocr_backend']),
            'ocr_backend_selected': _safe_get_selected_ocr_backend(),
            'eager_load_models': _model_status['eager_load_models'],
        },
    })


# ── 运行时计数器快照（自启动以来每个 endpoint 命中数）──────────────────
@app.route('/runtime_stats', methods=['GET'])
def get_runtime_stats():
    import platform as _plat
    snap = _runtime_stats_snapshot()
    snap['uptime_sec'] = int(time.time() - snap['started_ts'])
    snap['platform']   = _plat.platform()
    snap['python']     = sys.version.split()[0]
    snap['pid']        = os.getpid()
    return jsonify(snap)


@app.route('/metrics', methods=['GET'])
def get_metrics():
    return Response(
        _runtime_metrics_text(),
        status=200,
        mimetype='text/plain; version=0.0.4; charset=utf-8',
    )


# ── HITL 反馈：JSONL append-only 存储（从 SQLite 迁移过来）─────
#
# 背景：之前用 SQLite，但整个 backend 从来没 SELECT 过这张表（write-only）。
# SQLite 单 worker 假设、无 WAL、无 migration，换到多 worker
# 部署就坏。改成 JSONL append-only：
#   - 跨线程: threading.Lock 串行化（Flask 单 worker，见 CLAUDE.md）
#   - 无 schema 演进问题（每行独立 JSON）
#   - 离线分析：pandas.read_json('feedback.jsonl', lines=True)
#   - 存量 feedback.db 保留在磁盘但不再写入（以后手动导出或 ignore）

_FEEDBACK_ALLOWED_ACTIONS = {'keep', 'modify', 'delete', 'add'}
_FEEDBACK_MAX_ACTIONS = 5000


def _feedback_text(value, fallback: str = '') -> str:
    if value is None:
        return fallback
    return str(value)


@app.route('/feedback', methods=['POST'])
def feedback():
    """接收用户校验反馈，append 到 JSONL 日志。

    请求体包含 pdf_name、page、timestamp 和 actions 列表。
    每个 action 可携带 stamp_id、nominal、source 或 bbox，按操作类型提供。
    action 字段: keep / modify / delete / add
    """
    _bump('feedback_calls')
    data = request.get_json(silent=True)
    if not data:
        _record_feedback_error(status_code=400, reason='missing_json')
        return jsonify({'status': 'error', 'message': 'JSON body required'}), 400

    actions = data.get('actions', [])
    if not isinstance(actions, list):
        _record_feedback_error(status_code=400, reason='invalid_actions')
        return jsonify({'status': 'error', 'message': 'actions 必须是数组'}), 400
    if len(actions) > _FEEDBACK_MAX_ACTIONS:
        _record_feedback_error(status_code=400, reason='too_many_actions')
        return jsonify({'status': 'error', 'message': f'actions 数量超过 {_FEEDBACK_MAX_ACTIONS}'}), 400

    if not actions:
        _record_feedback_success(0)
        return jsonify({'status': 'ok', 'inserted': 0})

    pdf_name = _feedback_text(data.get('pdf_name'), 'unknown') or 'unknown'
    timestamp = _feedback_text(data.get('timestamp'), '')
    try:
        page = _parse_int_field(data, 'page', 0, min_value=0)
    except ValueError as e:
        _record_feedback_error(status_code=400, reason='invalid_page')
        return jsonify({'status': 'error', 'message': str(e)}), 400

    log_path = Config.FEEDBACK_LOG_PATH
    created_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    records = []
    for idx, act in enumerate(actions):
        if not isinstance(act, dict):
            _record_feedback_error(status_code=400, reason='invalid_action_item')
            return jsonify({'status': 'error', 'message': f'actions[{idx}] 必须是对象'}), 400
        action = act.get('action', 'keep')
        if action not in _FEEDBACK_ALLOWED_ACTIONS:
            _record_feedback_error(status_code=400, reason='invalid_action')
            return jsonify({
                'status': 'error',
                'message': f'actions[{idx}].action 无效: {action}',
            }), 400
        bbox = act.get('bbox')
        if bbox is not None and not isinstance(bbox, dict):
            _record_feedback_error(status_code=400, reason='invalid_bbox')
            return jsonify({'status': 'error', 'message': f'actions[{idx}].bbox 必须是对象或 null'}), 400
        records.append({
            'pdf_name': pdf_name,
            'page': page,
            'timestamp': timestamp,
            'stamp_id': _feedback_text(act.get('stamp_id'), ''),
            'action': action,
            'nominal': _feedback_text(act.get('nominal'), ''),
            'source': _feedback_text(act.get('source'), ''),
            'confidence': _feedback_text(act.get('confidence'), ''),
            'bbox': bbox,
            'user_correction': _feedback_text(act.get('user_correction'), ''),
            'created_at': created_at,
        })

    try:
        with _feedback_lock:
            with open(log_path, 'a', encoding='utf-8') as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        _record_feedback_success(len(records))
        return jsonify({'status': 'ok', 'inserted': len(records)})
    except Exception as e:
        app.logger.error('feedback write error: %s', e)
        _record_feedback_error(status_code=500, reason='write_error')
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── POST /export ─────────────────────────────────────────────────
#
# 把 pipeline 输出转成 QA 格式的 xlsx（"尺寸表" sheet）并直接以
# binary attachment 方式返回。复用全局 _gdt_detector 单例，避免每次
# 请求都重新加载 YOLO 模型。

# 文件名 sanitize：剥非 ASCII，再把非 [\w\-_.] 的字符替换为 '_'，
# 防止 Content-Disposition header 注入 / 浏览器解码异常。
_FILENAME_SAFE_RE = re.compile(r'[^\w\-_.]')


def _sanitize_filename(name: str, fallback: str = 'drawing') -> str:
    if not name:
        return fallback
    # 只保留 ASCII 可打印字符
    ascii_only = name.encode('ascii', 'ignore').decode('ascii').strip()
    cleaned = _FILENAME_SAFE_RE.sub('_', ascii_only).strip('._')
    return cleaned or fallback


def _upload_basename(name: str | None, fallback: str) -> str:
    normalized = (name or '').replace('\\', '/')
    return os.path.basename(normalized) or fallback


def _json_error(message: str, status_code: int):
    return jsonify({'status': 'error', 'message': message}), status_code


def _production_strict_only_payload(
    *,
    endpoint: str,
    reason: str,
    message: str,
    requested_dimension_path=None,
) -> dict:
    """Build the shared fail-closed response for production recognition gates."""
    return fail_closed_payload({
        'schema_version': 'vector_only_error_v1',
        'code': 'production_mode_requires_vector_only',
        'status': 'fail_closed',
        'reasons': [reason],
        'message': message,
        'endpoint': endpoint,
        'required_dimension_path': 'vector_only',
        'requested_dimension_path': requested_dimension_path,
        'trace_id': None,
        'page_index': None,
        'requested_backend': 'unavailable',
        'effective_backend': 'unavailable',
        'fallback': False,
    })


def _production_analyze_rejection(data) -> tuple[dict, str] | None:
    """Require an exact raw vector_only request before any recognition work."""
    if not Config.PRODUCTION_MODE:
        return None

    if data is None:
        # Preserve the existing missing/malformed JSON 400 contract.
        return None
    if not isinstance(data, dict):
        reason = 'production_mode_json_object_required'
        return (
            _production_strict_only_payload(
                endpoint='/analyze',
                reason=reason,
                message='生产模式要求请求体为 JSON 对象并显式使用 vector_only',
            ),
            reason,
        )

    raw_settings = data.get('settings')
    requested_path = (
        raw_settings.get('dimension_path')
        if isinstance(raw_settings, dict)
        else None
    )
    reported_path = requested_path if isinstance(requested_path, str) else None
    if not isinstance(raw_settings, dict):
        reason = 'production_mode_settings_required'
        message = '生产模式要求显式提供 settings.dimension_path=vector_only'
    elif 'dimension_path' not in raw_settings:
        reason = 'production_mode_dimension_path_required'
        message = '生产模式要求显式提供 dimension_path=vector_only'
    elif requested_path == 'vector_only':
        return None
    elif requested_path in ('ocr_only', 'vector_plus_ocr'):
        reason = 'production_mode_non_vector_path_forbidden'
        message = '生产模式禁止非 vector_only 识别路径'
    else:
        reason = 'production_mode_dimension_path_invalid'
        message = '生产模式不接受非法 dimension_path；必须显式使用 vector_only'

    return (
        _production_strict_only_payload(
            endpoint='/analyze',
            reason=reason,
            message=message,
            requested_dimension_path=reported_path,
        ),
        reason,
    )


def _production_legacy_recognition_route_rejection(
    endpoint: str,
) -> tuple[dict, str] | None:
    """Block production routes that can only enter a legacy recognition path."""
    if not Config.PRODUCTION_MODE:
        return None
    reason = 'production_mode_legacy_recognition_route_forbidden'
    return (
        _production_strict_only_payload(
            endpoint=endpoint,
            reason=reason,
            message='生产模式禁止该非 strict 识别入口',
        ),
        reason,
    )


def _production_export_rejection(data) -> tuple[dict, str] | None:
    """Allow only the reviewed-dimensions, no-recognition export in production."""
    if not Config.PRODUCTION_MODE or data is None:
        return None
    if not isinstance(data, dict):
        reason = 'production_mode_json_object_required'
        return (
            _production_strict_only_payload(
                endpoint='/export',
                reason=reason,
                message='生产模式要求导出请求体为 JSON 对象',
            ),
            reason,
        )
    if data.get('dims') is None:
        return _production_legacy_recognition_route_rejection('/export')
    return None


def _is_measurement_workbook_parse_error(exc: Exception) -> bool:
    return isinstance(exc, (BadZipFile, InvalidFileException, ValueError))


def _parse_int_field(data: dict, name: str, default=None,
                     *, min_value=None, max_value=None):
    raw = data.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f'{name} 必须是整数')
    if isinstance(raw, float):
        if not math.isfinite(raw) or not raw.is_integer():
            raise ValueError(f'{name} 必须是整数')
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{name} 必须是整数')
    if min_value is not None and value < min_value:
        raise ValueError(f'{name} 必须 >= {min_value}')
    if max_value is not None and value > max_value:
        raise ValueError(f'{name} 必须 <= {max_value}')
    return value


def _parse_bool_field(data: dict, name: str, default=False):
    raw = data.get(name, default)
    if isinstance(raw, bool):
        return raw
    if raw in (None, ''):
        return bool(default)
    raise ValueError(f'{name} 必须是布尔值')


def _decode_pdf_base64(pdf_b64: str) -> bytes:
    if not isinstance(pdf_b64, str) or not pdf_b64.strip():
        raise ValueError('缺少 pdf_base64 字段')
    try:
        return base64.b64decode(pdf_b64, validate=True)
    except Exception as exc:
        raise ValueError('pdf_base64 解码失败') from exc


def _read_pdf_upload_file(*, fallback_filename: str = 'drawing.pdf') -> tuple[str, bytes]:
    storage = request.files.get('file')
    if storage is None:
        raise ValueError('请上传 PDF 文件')

    filename = _upload_basename(storage.filename, fallback_filename)
    if not filename.lower().endswith('.pdf'):
        raise ValueError(f'不支持的文件类型: {filename}')

    max_upload_bytes = int(getattr(Config, 'MAX_UPLOAD_BYTES', 32 * 1024 * 1024))
    pdf_bytes = storage.read()
    if not pdf_bytes:
        raise ValueError(f'文件为空: {filename}')
    if len(pdf_bytes) > max_upload_bytes:
        raise OverflowError(f'上传文件超过大小限制: {max_upload_bytes} bytes')
    return filename, pdf_bytes


def _parse_part_no(value) -> str:
    if value is None or value == '':
        return 'drawing'
    if not isinstance(value, str):
        raise ValueError('part_no 必须是字符串')
    return value.strip() or 'drawing'


def _validate_dims_override(value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError('dims 必须是数组')
    if len(value) > 5000:
        raise ValueError('dims 数量超过 5000')
    if not all(isinstance(item, dict) for item in value):
        raise ValueError('dims 每一项必须是对象')
    return value


@app.route('/export', methods=['POST'])
def export_xlsx():
    """
    请求体 (JSON):
      {
        "pdf_base64": "<base64>",
        "page": 1,             // 1-based, 不传默认 1
        "all_pages": false,    // 可选, true 时跑所有页
        "part_no": "drawing",  // 可选, 不传时填 'drawing'
        "dims": [...]          // 开发模式可选；生产模式必填，
                               // 调用方直接给定 dim list，
                               // 传则跳过 pipeline 重跑，直接用这些 dim
                               // 写 xlsx（前端 ZIP 导出走这条路径，
                               // 保留用户校对后的删除/修改/添加结果）。
                               // all_pages=true 时每项用 page 标识所属页。
      }

    响应:
      200  binary xlsx, Content-Disposition attachment
      400  请求体或参数无效
      422  生产 strict-only 策略拒绝或 PDF 解析失败
      500  pipeline 内部异常
    """
    _bump('export_calls')
    # ── 1. 解析请求 ─────────────────────────────────────────────
    data = request.get_json(silent=True)
    production_rejection = _production_export_rejection(data)
    if production_rejection is not None:
        payload, _reason = production_rejection
        return jsonify(payload), 422
    if not data:
        return _json_error('请求体必须是 JSON', 400)

    pdf_b64 = data.get('pdf_base64')

    try:
        page_number = _parse_int_field(data, 'page', 1, min_value=1)
        all_pages = _parse_bool_field(data, 'all_pages', False)
        part_no_raw = _parse_part_no(data.get('part_no'))
        dims_override = _validate_dims_override(data.get('dims'))
        pdf_bytes = _decode_pdf_base64(pdf_b64)
    except ValueError as e:
        return _json_error(str(e), 400)

    # ── 3. 预校验 PDF 可打开（快速失败，跟 /analyze 风格一致） ──
    # dims_override 路径不跑 pipeline，但仍验证 PDF bytes 真能打开，防止坏包静默出表。
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as _probe:
            total_pages = len(_probe.pages)
            if not all_pages and page_number > total_pages:
                return _json_error(f'页码 {page_number} 超出范围（共 {total_pages} 页）', 400)
            if dims_override is not None:
                for index, dim in enumerate(dims_override):
                    raw_page = dim.get('page', dim.get('page_number', dim.get('pageIndex', page_number)))
                    try:
                        dim_page = int(raw_page)
                    except (TypeError, ValueError):
                        return _json_error(f'dims[{index}].page 必须是正整数', 400)
                    if dim_page < 1 or dim_page > total_pages:
                        return _json_error(
                            f'dims[{index}].page {dim_page} 超出范围（共 {total_pages} 页）',
                            400,
                        )
    except Exception as e:
        return _json_error(f'PDF 解析失败: {e}', 422)

    # ── 4. 跑 pipeline + 生成 xlsx ────────────────────────────
    try:
        xlsx_bytes = build_xlsx_from_bytes(
            pdf_bytes=pdf_bytes,
            part_no=part_no_raw,
            page_number=page_number,
            all_pages=all_pages,
            gdt_detector=None if dims_override is not None else _get_gdt_detector(),
            dims_override=dims_override,
            document_page_count=total_pages,
        )
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error('export error:\n%s', tb)
        return _json_error('导出内部错误', 500)
    finally:
        close_fitz_cache()

    # ── 5. 返回 binary attachment ──────────────────────────────
    safe_name = _sanitize_filename(part_no_raw)
    filename = f'{safe_name}_dim_table.xlsx'
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"',
        'Content-Length': str(len(xlsx_bytes)),
    }
    if Config.EXPORT_DIM_OVERRIDE_CONSISTENCY and dims_override is not None:
        dump = build_export_dim_override_consistency_v1(
            dims_override=dims_override,
            xlsx_bytes=xlsx_bytes,
            pipeline_dims=None,
            pipeline_rerun_count=0,
        )
        headers['X-Export-Dim-Override-Consistency'] = json.dumps(
            summarize_export_dim_override_consistency(dump),
            ensure_ascii=True,
            separators=(',', ':'),
        )
    return Response(
        xlsx_bytes,
        status=200,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers=headers,
    )


# ── POST /prelabel_fast ───────────────────────────────────────────

@app.route('/prelabel_fast', methods=['POST'])
def prelabel_fast():
    """
    仅供非生产诊断；生产模式由 strict-only 策略禁用。

    Multipart form:
      file: PDF upload
      page: 1-based page number, default 1
    """
    t_start = time.time()
    if not Config.VECTOR_PRELABEL_API_ENABLED:
        return _json_error('VECTOR_PRELABEL_API_ENABLED is false; /prelabel_fast is disabled', 403)
    production_rejection = _production_legacy_recognition_route_rejection(
        '/prelabel_fast'
    )
    if production_rejection is not None:
        payload, _reason = production_rejection
        return jsonify(payload), 422

    try:
        page_number = _parse_int_field(request.form, 'page', 1, min_value=1)
        filename, pdf_bytes = _read_pdf_upload_file()
    except OverflowError as e:
        return _json_error(str(e), 413)
    except ValueError as e:
        return _json_error(str(e), 400)

    try:
        pdf_file = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as e:
        return _json_error(f'PDF 解析失败: {e}', 422)

    try:
        if page_number < 1 or page_number > len(pdf_file.pages):
            return _json_error(f'页码 {page_number} 超出范围（共 {len(pdf_file.pages)} 页）', 400)

        page_index = page_number - 1
        payload = run_vector_prelabel_fast_profile(
            pdf_bytes=pdf_bytes,
            page=pdf_file.pages[page_index],
            page_index=page_index,
            verbose=Config.VERBOSE,
        )
        response_payload = dict(payload)
        response_payload['status'] = 'success'
        response_payload['request'] = {
            'filename': filename,
            'page': page_number,
            'page_index': page_index,
            'elapsed_sec': round(time.time() - t_start, 3),
        }
        return jsonify(response_payload)
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error('prelabel_fast error:\n%s', tb)
        return _json_error(f'预标内部错误: {e}', 500)
    finally:
        pdf_file.close()


# ── POST /import_measurements ─────────────────────────────────────

@app.route('/import_measurements', methods=['POST'])
def import_measurements():
    """
    Multipart form upload for M3 inspection result import.

    Form fields:
      files: one or more .xls/.xlsx workbooks

    Response:
      {
        "status": "success",
        "workbooks": [...],
        "records": [...],
        "warnings": [...],
        "stats": {...}
      }
    """
    _bump('import_measurements_calls')
    uploaded = request.files.getlist('files') or request.files.getlist('file')
    if not uploaded:
        return jsonify({'status': 'error', 'message': '请上传 .xls/.xlsx 测量结果文件'}), 400

    files = []
    for storage in uploaded:
        filename = _upload_basename(storage.filename, 'measurement.xlsx')
        if not filename.lower().endswith(('.xls', '.xlsx', '.xlsm')):
            return jsonify({'status': 'error', 'message': f'不支持的文件类型: {filename}'}), 400
        data = storage.read()
        if not data:
            return jsonify({'status': 'error', 'message': f'文件为空: {filename}'}), 400
        files.append((filename, data))

    try:
        return jsonify(parse_uploaded_workbooks(files))
    except Exception as e:
        if _is_measurement_workbook_parse_error(e):
            return jsonify({'status': 'error', 'message': f'测量结果文件解析失败: {e}'}), 422
        tb = traceback.format_exc()
        app.logger.error('import_measurements error:\n%s', tb)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── POST /import_legacy_stamps ─────────────────────────────────────

@app.route('/import_legacy_stamps', methods=['POST'])
def import_legacy_stamps():
    """
    Multipart upload for old stamped sequence PDFs.

    This endpoint does not run OCR. It extracts PDF text spans and annotation
    geometry from old manually numbered drawings and returns page-local marks
    that the frontend converts into editable Stamp objects.
    """
    _bump('import_legacy_stamps_calls')
    storage = request.files.get('file')
    if storage is None:
        return jsonify({'status': 'error', 'message': '请上传旧序号 PDF'}), 400
    filename = _upload_basename(storage.filename, 'legacy_stamps.pdf')
    if not filename.lower().endswith('.pdf'):
        return jsonify({'status': 'error', 'message': f'不支持的文件类型: {filename}'}), 400
    pdf_bytes = storage.read()
    if not pdf_bytes:
        return jsonify({'status': 'error', 'message': f'文件为空: {filename}'}), 400

    try:
        result = extract_legacy_stamp_marks(pdf_bytes)
        result['filename'] = filename
        return jsonify(result)
    except (fitz.FileDataError, ValueError) as e:
        return jsonify({'status': 'error', 'message': f'PDF 解析失败: {e}'}), 422
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error('import_legacy_stamps error:\n%s', tb)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── POST /analyze ─────────────────────────────────────────────────

def _literal_vector_layer_debug_unavailable_v1(
    *,
    trace_id,
    page_index,
    reason,
):
    """Last-resort JSON literal; diagnostic failures must never escape."""
    safe_trace_id = (
        trace_id
        if isinstance(trace_id, str) and trace_id
        else 'diagnostic_trace_unavailable'
    )
    safe_page_index = (
        page_index
        if type(page_index) is int and page_index >= 0
        else 0
    )
    safe_reason = re.sub(
        r'[^a-z0-9._-]+',
        '_',
        reason if isinstance(reason, str) else 'diagnostic_unavailable',
    ).strip('._-') or 'diagnostic_unavailable'
    return {
        'schema_version': 'vector_layer_debug_v1',
        'trace_id': safe_trace_id,
        'page_index': safe_page_index,
        'coordinate_space': 'pdf_page_points_top_left',
        'status': 'unavailable',
        'unavailable_reason': safe_reason[:128],
        'stages': [],
        'overlays': [],
        'limits': {'total': 0, 'returned': 0, 'truncated': False},
        'authority': {
            'diagnostic_only': True,
            'consumer_allowed': False,
            'release_allowed': False,
        },
    }


def _attach_vector_layer_debug_v1_no_throw(
    response_payload,
    *,
    debug_request,
    pipe,
    strict_trace_id,
    pdf_bytes,
    page_index,
    is_vector_only,
    unavailable_reason=None,
):
    """Attach opt-in diagnostics without changing product control flow."""
    if debug_request is None:
        return
    fallback_trace_id = (
        strict_trace_id
        if isinstance(strict_trace_id, str) and strict_trace_id
        else f'diagnostic-page-{page_index if type(page_index) is int else 0}'
    )
    debug_trace_id = fallback_trace_id
    failure_reason = 'debug_projection_failed'
    try:
        debug_trace_id = str(
            strict_trace_id
            or build_strict_yolo_trace_id(pdf_bytes, page_index)
        )
        if not debug_request.valid:
            failure_reason = str(
                debug_request.reason or 'debug_request_invalid'
            )
            payload = build_vector_layer_debug_unavailable_v1(
                trace_id=debug_trace_id,
                page_index=page_index,
                reason=failure_reason,
            )
        elif unavailable_reason is not None:
            failure_reason = unavailable_reason
            payload = build_vector_layer_debug_unavailable_v1(
                trace_id=debug_trace_id,
                page_index=page_index,
                reason=failure_reason,
            )
        elif not is_vector_only:
            failure_reason = 'vector_only_required'
            payload = build_vector_layer_debug_unavailable_v1(
                trace_id=debug_trace_id,
                page_index=page_index,
                reason=failure_reason,
            )
        else:
            payload = build_vector_layer_debug_v1(
                pipe,
                trace_id=debug_trace_id,
                page_index=page_index,
                include_overlays=debug_request.include_overlays,
                overlay_limit=debug_request.overlay_limit,
            )
        if not isinstance(payload, dict):
            raise ValueError('vector layer debug payload invalid')
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except Exception:
        try:
            app.logger.exception('vector_layer_debug_v1 projection failed')
        except Exception:
            pass
        try:
            payload = build_vector_layer_debug_unavailable_v1(
                trace_id=debug_trace_id,
                page_index=page_index,
                reason=failure_reason,
            )
            if not isinstance(payload, dict):
                raise ValueError('vector layer debug fallback invalid')
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except Exception:
            payload = _literal_vector_layer_debug_unavailable_v1(
                trace_id=debug_trace_id,
                page_index=page_index,
                reason=failure_reason,
            )
    try:
        response_payload['vector_layer_debug_v1'] = payload
    except Exception:
        return


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    请求体 (JSON):
      {
        "pdf_base64": "<base64>", "page": 1, "dpi": 200,
        "settings": {"dimension_path": "vector_only"}
      }

    响应体:
      {
        "status": "success",
        "pdf_type": "A" | "B" | "BC",
        "page_width": float, "page_height": float,
        "dimensions": [...], "references": [...], "basic_dimensions": [...],
        "gdt_frames": [...],
        "view_boxes": [...], "timings": {...}, "debug_log": "..."
      }
    """
    _bump('analyze_calls')
    t_start = time.time()

    log_buffer = io.StringIO()

    # ── 1. 解析请求 ─────────────────────────────────────────────
    data = request.get_json(silent=True)
    production_rejection = _production_analyze_rejection(data)
    if production_rejection is not None:
        payload, reason = production_rejection
        _record_analyze_error(
            status_code=422,
            reason=reason,
            elapsed_sec=time.time() - t_start,
        )
        return jsonify(payload), 422
    if not data:
        _record_analyze_error(
            status_code=400,
            reason='missing_json',
            elapsed_sec=time.time() - t_start,
        )
        return _json_error('请求体必须是 JSON', 400)

    pdf_b64 = data.get('pdf_base64')
    recognition_settings = RecognitionSettings.from_request(data)
    try:
        vector_layer_debug_request = VectorLayerDebugRequest.from_request(data)
    except Exception:
        vector_layer_debug_request = VectorLayerDebugRequest(
            include_overlays=False,
            overlay_limit=0,
            valid=False,
            reason='debug_request_parse_failed',
        )
    try:
        page_number = _parse_int_field(data, 'page', 1, min_value=1)
        dpi = _parse_int_field(data, 'dpi', Config.DPI_DEFAULT,
                               min_value=72, max_value=600)
        pdf_bytes = _decode_pdf_base64(pdf_b64)
    except ValueError as e:
        _record_analyze_error(
            status_code=400,
            reason='invalid_request',
            elapsed_sec=time.time() - t_start,
        )
        if recognition_settings is not None and recognition_settings.is_vector_only:
            raw_page = data.get('page', 1)
            page_index = (
                raw_page - 1
                if type(raw_page) is int and raw_page >= 1
                else None
            )
            payload = fail_closed_payload({
                'schema_version': 'vector_only_error_v1',
                'code': 'vector_only_invalid_request',
                'status': 'fail_closed',
                'reasons': ['invalid_request'],
                'message': str(e),
                'trace_id': None,
                'page_index': page_index,
                'requested_backend': 'yolo',
                'effective_backend': 'unavailable',
                'fallback': False,
            })
            return jsonify(payload), 400
        return _json_error(str(e), 400)

    strict_yolo_task = None
    strict_yolo_trace_id = None
    strict_yolo_timing_sink = {}
    strict_runtime_audits = None
    strict_runtime_baseline = None
    strict_fitz_doc = None
    if recognition_settings is not None and recognition_settings.is_vector_only:
        strict_yolo_trace_id = build_strict_yolo_trace_id(
            pdf_bytes,
            page_number - 1,
        )
        strict_runtime_baseline = ocr_singleton_snapshot()
        runtime_preflight = _strict_forbidden_runtime_check(
            ocr_snapshot=strict_runtime_baseline,
        )
        if not runtime_preflight.get('ok'):
            payload, reason_code = _vector_only_preloaded_runtime_failure(
                runtime_preflight,
                trace_id=strict_yolo_trace_id,
                page_index=page_number - 1,
            )
            _record_analyze_error(
                status_code=503,
                reason=reason_code,
                elapsed_sec=time.time() - t_start,
            )
            return jsonify(payload), 503
        strict_requested_view_seg = recognition_settings.invalid_fields.get(
            'view_seg',
            recognition_settings.view_seg,
        )
        if (
            'view_seg' in recognition_settings.invalid_fields
            or recognition_settings.view_seg not in (None, 'yolo')
        ):
            payload = fail_closed_payload({
                'schema_version': 'vector_only_error_v1',
                'code': 'vector_only_requires_yolo_view',
                'status': 'fail_closed',
                'reasons': ['vector_only_requires_yolo_view'],
                'trace_id': strict_yolo_trace_id,
                'page_index': page_number - 1,
                'requested_backend': strict_requested_view_seg,
                'effective_backend': 'unavailable',
                'fallback': False,
                'feeds_display_l3': False,
                'consumer_allowed': False,
            })
            _record_analyze_error(
                status_code=422,
                reason='vector_only_requires_yolo_view',
                elapsed_sec=time.time() - t_start,
            )
            return jsonify(payload), 422
        try:
            strict_yolo_task = submit_strict_yolo_view(
                pdf_bytes=pdf_bytes,
                page_index=page_number - 1,
                trace_id=strict_yolo_trace_id,
                model_path=strict_yolo_view_model_path(),
                expected_model_sha256=strict_yolo_view_expected_sha256(),
                timeout_s=strict_yolo_view_timeout_s(),
                timing_sink=strict_yolo_timing_sink,
            )
        except StrictYoloViewUnavailable as exc:
            _record_analyze_error(
                status_code=503,
                reason=exc.reason,
                elapsed_sec=time.time() - t_start,
            )
            return jsonify(fail_closed_payload(exc.as_payload())), 503
        except Exception as exc:
            wrapped = StrictYoloViewUnavailable(
                reason='strict_yolo_submit_failed',
                trace_id=strict_yolo_trace_id,
                page_index=page_number - 1,
            )
            _record_analyze_error(
                status_code=503,
                reason=wrapped.reason,
                elapsed_sec=time.time() - t_start,
            )
            app.logger.error('strict yolo submit failed: %s', exc)
            return jsonify(fail_closed_payload(wrapped.as_payload())), 503

    # ── 2. base64 解码 ──────────────────────────────────────────

    # ── 3. 打开 PDF ─────────────────────────────────────────────
    try:
        pdf_file = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as e:
        _cancel_strict_yolo_task(strict_yolo_task, 'invalid_pdf')
        _record_analyze_error(
            status_code=422,
            reason='invalid_pdf',
            elapsed_sec=time.time() - t_start,
        )
        if recognition_settings is not None and recognition_settings.is_vector_only:
            payload = fail_closed_payload({
                'schema_version': 'vector_only_error_v1',
                'code': 'vector_only_invalid_pdf',
                'status': 'fail_closed',
                'reasons': ['invalid_pdf'],
                'message': f'PDF 解析失败: {e}',
                'trace_id': strict_yolo_trace_id,
                'page_index': page_number - 1,
                'requested_backend': 'yolo',
                'effective_backend': 'unavailable',
                'fallback': False,
                'feeds_display_l3': False,
                'consumer_allowed': False,
            })
            return jsonify(payload), 422
        return _json_error(f'PDF 解析失败: {e}', 422)

    # 捕获 pipeline print 输出到 log_buffer（线程安全）。放在参数/PDF
    # 校验之后，保证所有早退路径不会残留 thread-local buffer。
    _tee_local.log_buffer = log_buffer

    try:
        if page_number < 1 or page_number > len(pdf_file.pages):
            _record_analyze_error(
                status_code=400,
                reason='page_out_of_range',
                elapsed_sec=time.time() - t_start,
            )
            if recognition_settings is not None and recognition_settings.is_vector_only:
                payload = fail_closed_payload({
                    'schema_version': 'vector_only_error_v1',
                    'code': 'vector_only_page_out_of_range',
                    'status': 'fail_closed',
                    'reasons': ['page_out_of_range'],
                    'message': (
                        f'页码 {page_number} 超出范围（共 {len(pdf_file.pages)} 页）'
                    ),
                    'trace_id': strict_yolo_trace_id,
                    'page_index': page_number - 1,
                    'requested_backend': 'yolo',
                    'effective_backend': 'unavailable',
                    'fallback': False,
                    'feeds_display_l3': False,
                    'consumer_allowed': False,
                })
                return jsonify(payload), 400
            return _json_error(f'页码 {page_number} 超出范围（共 {len(pdf_file.pages)} 页）', 400)

        page = pdf_file.pages[page_number - 1]
        page_index = page_number - 1

        # ── 4. 类型探测 ───────────────────────────────────────
        pdf_type = detect_pdf_type(page)
        page_width = float(page.width)
        page_height = float(page.height)

        # ── 5. 按类型分派 ─────────────────────────────────────
        if pdf_type == 'A':
            if recognition_settings is not None and recognition_settings.is_vector_only:
                response = fail_closed_payload({
                    'schema_version': 'vector_only_error_v1',
                    'code': 'vector_only_pdf_type_unsupported',
                    'status': 'fail_closed',
                    'reasons': ['vector_only_pdf_type_a_unsupported'],
                    'trace_id': strict_yolo_trace_id,
                    'page_index': page_index,
                    'requested_backend': 'yolo',
                    'effective_backend': 'unavailable',
                    'fallback': False,
                    'feeds_display_l3': False,
                    'consumer_allowed': False,
                })
                _record_analyze_error(
                    status_code=422,
                    reason='vector_only_pdf_type_a_unsupported',
                    elapsed_sec=time.time() - t_start,
                )
                _attach_vector_layer_debug_v1_no_throw(
                    response,
                    debug_request=vector_layer_debug_request,
                    pipe=None,
                    strict_trace_id=strict_yolo_trace_id,
                    pdf_bytes=pdf_bytes,
                    page_index=page_index,
                    is_vector_only=True,
                    unavailable_reason='pdf_type_a_not_supported',
                )
                return jsonify(response), 422
            # Type A: 直接矢量提取 + 语义组装
            type_a_result = extract_type_a(page)
            assembled = assemble(pdf_type='A', type_a_result=type_a_result)
            response = {
                'status': 'success',
                'pdf_type': pdf_type,
                'page_width': page_width,
                'page_height': page_height,
                'dimensions': assembled.get('dimensions', []),
                'references': assembled.get('references', []),
                'basic_dimensions': assembled.get('basic_dimensions', []),
                'gdt_frames': assembled.get('gdt_frames', []),
                'notes': [],
                'view_boxes': [],
                'timings': {'total': round(time.time() - t_start, 3)},
                'debug_log': log_buffer.getvalue(),
            }
            _attach_vector_layer_debug_v1_no_throw(
                response,
                debug_request=vector_layer_debug_request,
                pipe=None,
                strict_trace_id=None,
                pdf_bytes=pdf_bytes,
                page_index=page_index,
                is_vector_only=False,
                unavailable_reason='pdf_type_a_not_supported',
            )
            _record_analyze_success(
                pdf_type=pdf_type,
                page_number=page_number,
                pdf_pages=len(pdf_file.pages),
                elapsed_sec=time.time() - t_start,
                payload=response,
            )
            _tee_local.log_buffer = None
            return jsonify(response)

        # Type B / BC：主管线
        is_vector_only = (
            recognition_settings is not None
            and recognition_settings.is_vector_only
        )
        if is_vector_only:
            guard_entry_snapshot = ocr_singleton_snapshot()
            guard_entry_check = _strict_forbidden_runtime_check(
                ocr_snapshot=guard_entry_snapshot,
            )
            if not guard_entry_check.get('ok'):
                payload, reason_code = _vector_only_preloaded_runtime_failure(
                    guard_entry_check,
                    trace_id=strict_yolo_trace_id,
                    page_index=page_index,
                )
                _record_analyze_error(
                    status_code=503,
                    reason=reason_code,
                    elapsed_sec=time.time() - t_start,
                )
                return jsonify(payload), 503
            strict_fitz_doc = fitz.open(
                stream=pdf_bytes,
                filetype='pdf',
            )
        pipeline_kwargs = dict(
            pdf_bytes=pdf_bytes,
            page=page,
            page_index=page_index,
            dpi=dpi,
            gdt_detector=(
                None
                if recognition_settings is not None and recognition_settings.is_vector_only
                else _get_gdt_detector()
            ),
            verbose=Config.VERBOSE,
            settings=recognition_settings,
            strict_yolo_source=strict_yolo_task,
            strict_yolo_trace_id=strict_yolo_trace_id,
            shared_fitz_doc=strict_fitz_doc,
        )
        strict_pipeline_diagnostics = {}
        if is_vector_only:
            pipeline_kwargs['strict_diagnostics_out'] = (
                strict_pipeline_diagnostics
            )
            strict_success_validation_session = StrictSuccessValidationSession()
            with strict_vector_only_runtime_guard() as strict_runtime_audits:
                try:
                    pipe = run_type_b_pipeline(**pipeline_kwargs)
                finally:
                    finalize_runtime_audits(
                        strict_runtime_audits,
                        before_ocr_singletons=strict_runtime_baseline,
                        after_ocr_singletons=ocr_singleton_snapshot(),
                    )
            defensive_pipeline_diagnostics = {}
            pipe = split_strict_pipeline_product_v1(
                pipe,
                diagnostics_out=defensive_pipeline_diagnostics,
            )
            strict_pipeline_diagnostics = (
                merge_strict_diagnostic_sidecars_v1(
                    strict_pipeline_diagnostics,
                    defensive_pipeline_diagnostics,
                )
            )
            pipe['r32_ocr_runtime_audit_v1'] = dict(
                strict_runtime_audits['ocr']
            )
            pipe['r32_consumer_runtime_audit_v1'] = dict(
                strict_runtime_audits['consumer']
            )
            review_projection_status, review_projection_reasons = (
                validate_r33_m1_review_api_projection(
                    pipe,
                    expected_trace_id=strict_yolo_trace_id,
                    expected_page_index=page_index,
                )
            )
            if review_projection_status != 'ok':
                payload, reason_code = _r33_m1_review_projection_failure(
                    pipe,
                    projection_status=review_projection_status,
                    reasons=review_projection_reasons,
                    trace_id=strict_yolo_trace_id,
                    page_index=page_index,
                )
                _record_analyze_error(
                    status_code=503,
                    reason=reason_code,
                    elapsed_sec=time.time() - t_start,
                )
                return jsonify(payload), 503
            contract_reasons = validate_vector_only_success_payload(
                pipe,
                expected_trace_id=strict_yolo_trace_id,
                expected_page_index=page_index,
                validation_session=strict_success_validation_session,
            )
            if contract_reasons:
                payload = fail_closed_payload({
                    'schema_version': 'vector_only_error_v1',
                    'code': 'vector_only_contract_violation',
                    'status': 'fail_closed',
                    'reasons': contract_reasons,
                    'trace_id': strict_yolo_trace_id,
                    'page_index': page_index,
                    'requested_backend': 'yolo',
                    'effective_backend': 'unavailable',
                    'fallback': False,
                    'feeds_display_l3': False,
                    'consumer_allowed': False,
                })
                payload['r32_ocr_runtime_audit_v1'] = dict(
                    strict_runtime_audits['ocr']
                )
                payload['r32_consumer_runtime_audit_v1'] = dict(
                    strict_runtime_audits['consumer']
                )
                _record_analyze_error(
                    status_code=503,
                    reason='vector_only_contract_violation',
                    elapsed_sec=time.time() - t_start,
                )
                return jsonify(payload), 503
        else:
            pipe = run_type_b_pipeline(**pipeline_kwargs)

        def _pipe_field(key, default=None):
            return read_strict_diagnostic_field_v1(
                pipe,
                strict_pipeline_diagnostics,
                key,
                default,
            )

        if strict_yolo_timing_sink:
            pipe.setdefault('timings', {}).update(strict_yolo_timing_sink)

        # ── 6. 诊断汇总 ───────────────────────────────────────
        dimensions = pipe['dimensions']
        basic_dimensions = pipe.get('basic_dimensions', [])
        notes = pipe.get('notes', [])
        ocr_results = pipe['ocr_results']
        breakdown = pipe['ocr_breakdown']
        h_count = sum(1 for r in ocr_results if abs(r.get('orientation', 0)) <= 45)
        v_count = sum(1 for r in ocr_results if abs(r.get('orientation', 0)) > 45)
        src_counts = dict(Counter(d.get('source', '?') for d in dimensions))

        print(f"[/analyze] timings: {pipe['timings']}")
        print("\n=== 诊断汇总 ===")
        print(f"OCR: base={breakdown['base']} hires=+{breakdown['hires']} "
              f"strip=+{breakdown['strip']} gdt_comp=+{breakdown['gdt_comp']} "
              f"capsule=+{breakdown['capsule']} dimline=+{breakdown['dimline']} "
              f"angle_label=+{breakdown.get('angle_label', 0)} "
              f"radius_label=+{breakdown.get('radius_label', 0)} "
              f"total={breakdown['total']}")
        print(f"OCR orientation: H={h_count} V={v_count}")
        print(f"dimensions: {len(dimensions)} sources={src_counts}")
        if basic_dimensions:
            print(f"basic_dimensions: {len(basic_dimensions)} (not numbered)")
        if notes:
            print(f"notes: {len(notes)}")
        if Config.VERBOSE:
            for i, d in enumerate(dimensions):
                b = d.get('bbox', {})
                print(f"  [{i+1}] {d.get('nominal','')} src={d.get('source','')} "
                      f"bbox=({b.get('x',0):.0f},{b.get('y',0):.0f}) "
                      f"orient={d.get('orientation',0)} "
                      f"tol={d.get('upper_tol','')}/{d.get('lower_tol','')}")

        _tee_local.log_buffer = None
        response_payload = {
            'status': 'success',
            'pdf_type': pdf_type,
            'page_width': page_width,
            'page_height': page_height,
            'dimensions': dimensions,
            'references': pipe['references'],
            'basic_dimensions': basic_dimensions,
            'gdt_frames': _pipe_field('gdt_frames', []),
            'notes': notes,
            'dimension_candidates': _pipe_field('dimension_candidates', []),
            'candidate_drop_ledger': _pipe_field('candidate_drop_ledger', []),
            'candidate_debug_stats': _pipe_field('candidate_debug_stats', {}),
            'assembler_debug': pipe.get('assembler_debug', {}),
            'post_filter_drop_ledger': pipe.get('post_filter_drop_ledger', []),
            'post_filter_debug_stats': pipe.get('post_filter_debug_stats', {}),
            'view_boxes': pipe['view_boxes'],
            'timings': pipe['timings'],
            # R14.2 perf 诊断：把 pipeline 已有但未透传的两个结构化字段塞进
            # response，前端 download-log 能直接做聚合分析（不再 grep debug_log）
            'ocr_breakdown': pipe.get('ocr_breakdown', {}),
            'ocr_pass_counters': pipe.get('ocr_pass_counters', {}),
            'debug_log': log_buffer.getvalue(),
        }
        if is_vector_only:
            response_payload.update({
                'consumer_allowed': False,
                'display_allowed': False,
                'release_allowed': False,
            })
        if recognition_settings is not None:
            is_vector_only = recognition_settings.is_vector_only
            response_payload['capabilities'] = {
                'ocr_engines': ocr_engine_capabilities(),
                'view_seg': view_seg_capabilities(),
                'gdt': gdt_capabilities(
                    False if is_vector_only else bool(_model_status['gdt_loaded'])
                ),
            }
            response_payload['effective_settings'] = {
                'dimension_path': recognition_settings.dimension_path,
                'ocr_engine_effective': (
                    None
                    if is_vector_only
                    else get_selected_ocr_backend()
                ),
                'view_seg': 'yolo' if is_vector_only else recognition_settings.view_seg,
                'gdt_backend': recognition_settings.gdt_backend,
            }
            response_payload['settings_warnings'] = recognition_settings.invalid_fields
        if 'view_seg_status_v1' in pipe:
            response_payload['view_seg_status_v1'] = pipe.get('view_seg_status_v1', {})
        if 'strict_yolo_view_status_v1' in pipe:
            response_payload['strict_yolo_view_status_v1'] = pipe.get(
                'strict_yolo_view_status_v1', {}
            )
        if 'r32_ocr_runtime_audit_v1' in pipe:
            response_payload['r32_ocr_runtime_audit_v1'] = pipe.get(
                'r32_ocr_runtime_audit_v1', {}
            )
        if 'r32_consumer_runtime_audit_v1' in pipe:
            response_payload['r32_consumer_runtime_audit_v1'] = pipe.get(
                'r32_consumer_runtime_audit_v1', {}
            )
        if 'r92_directional_walk_dump_v1' in pipe:
            response_payload['r92_directional_walk_dump_v1'] = pipe.get(
                'r92_directional_walk_dump_v1', {}
            )
        if 'r32_phrase_runtime_audit_v1' in pipe:
            response_payload['r32_phrase_runtime_audit_v1'] = pipe.get(
                'r32_phrase_runtime_audit_v1', {}
            )
        if is_vector_only:
            for field in sorted(CHAIN_FIELD_KEYS):
                if field in pipe:
                    response_payload[field] = pipe[field]
            if REVIEW_CANDIDATES_ROOT_KEY in pipe:
                response_payload[REVIEW_CANDIDATES_ROOT_KEY] = pipe[
                    REVIEW_CANDIDATES_ROOT_KEY
                ]
            if (
                STRICT_GDT_AUDIT_FIELD in pipe
                or STRICT_GDT_AUDIT_FIELD in strict_pipeline_diagnostics
            ):
                response_payload[STRICT_GDT_AUDIT_FIELD] = _pipe_field(
                    STRICT_GDT_AUDIT_FIELD,
                    {},
                )
        if 'vector_glyph_v1' in pipe:
            response_payload['vector_glyph_v1'] = pipe.get('vector_glyph_v1', [])
            response_payload['vector_glyph_stats_v1'] = pipe.get('vector_glyph_stats_v1', {})
        if 'vector_numeric_phrases_v1' in pipe:
            response_payload['vector_numeric_phrases_v1'] = pipe.get(
                'vector_numeric_phrases_v1', []
            )
            response_payload['vector_numeric_phrase_stats_v1'] = pipe.get(
                'vector_numeric_phrase_stats_v1', {}
            )
        if 'vector_dimension_hypotheses_v1' in pipe:
            response_payload['vector_dimension_hypotheses_v1'] = pipe.get(
                'vector_dimension_hypotheses_v1', []
            )
            response_payload['vector_dimension_hypothesis_stats_v1'] = pipe.get(
                'vector_dimension_hypothesis_stats_v1', {}
            )
        if (
            'anchor_corridor_candidates_v1' in pipe
            or 'anchor_corridor_candidates_v1' in strict_pipeline_diagnostics
        ):
            response_payload['anchor_corridor_candidates_v1'] = _pipe_field(
                'anchor_corridor_candidates_v1', []
            )
            response_payload['anchor_corridor_stats_v1'] = _pipe_field(
                'anchor_corridor_stats_v1', {}
            )
            response_payload['anchor_phrase_candidates_v1'] = _pipe_field(
                'anchor_phrase_candidates_v1', []
            )
            response_payload['anchor_phrase_stats_v1'] = _pipe_field(
                'anchor_phrase_stats_v1', {}
            )
        if 'r2n_nominal_rescue_shadow_v1' in pipe:
            response_payload['r2n_nominal_rescue_shadow_v1'] = pipe.get(
                'r2n_nominal_rescue_shadow_v1',
                [],
            )
            response_payload['r2n_nominal_rescue_shadow_stats_v1'] = pipe.get(
                'r2n_nominal_rescue_shadow_stats_v1',
                {},
            )
        if 'vector_times_candidates_v1' in pipe:
            response_payload['vector_times_raw_cross_candidates_v1'] = pipe.get(
                'vector_times_raw_cross_candidates_v1',
                [],
            )
            response_payload['vector_times_candidates_v1'] = pipe.get(
                'vector_times_candidates_v1',
                [],
            )
            response_payload['vector_times_detector_stats_v1'] = pipe.get(
                'vector_times_detector_stats_v1',
                {},
            )
        if 'ocr_crop_oriented_quad_v1' in pipe:
            response_payload['ocr_crop_oriented_quad_v1'] = pipe.get(
                'ocr_crop_oriented_quad_v1', []
            )
            response_payload['ocr_crop_oriented_quad_stats_v1'] = pipe.get(
                'ocr_crop_oriented_quad_stats_v1', {}
            )
        if 'ocr_pass_necessity_v1' in pipe:
            response_payload['ocr_pass_necessity_v1'] = pipe.get(
                'ocr_pass_necessity_v1', []
            )
            response_payload['ocr_pass_necessity_stats_v1'] = pipe.get(
                'ocr_pass_necessity_stats_v1', {}
            )
        if 'assembler_polygon_compat_v1' in pipe:
            response_payload['assembler_polygon_compat_v1'] = pipe.get(
                'assembler_polygon_compat_v1', []
            )
            response_payload['assembler_polygon_compat_stats_v1'] = pipe.get(
                'assembler_polygon_compat_stats_v1', {}
            )
        if 'vector_final_consume_stats_v1' in pipe:
            response_payload['vector_final_dimensions_v1'] = pipe.get(
                'vector_final_dimensions_v1', []
            )
            response_payload['vector_final_consume_stats_v1'] = pipe.get(
                'vector_final_consume_stats_v1', {}
            )
            response_payload['vector_final_consume_ledger_v1'] = pipe.get(
                'vector_final_consume_ledger_v1', []
            )

        _attach_vector_layer_debug_v1_no_throw(
            response_payload,
            debug_request=vector_layer_debug_request,
            pipe=pipe,
            strict_trace_id=strict_yolo_trace_id,
            pdf_bytes=pdf_bytes,
            page_index=page_index,
            is_vector_only=is_vector_only,
        )
        strict_response_diagnostics = {}
        if is_vector_only:
            response_payload = split_strict_response_product_v1(
                response_payload,
                diagnostics_out=strict_response_diagnostics,
            )
        if is_vector_only:
            review_projection_status, review_projection_reasons = (
                validate_r33_m1_review_api_projection(
                    response_payload,
                    expected_trace_id=strict_yolo_trace_id,
                    expected_page_index=page_index,
                )
            )
            if review_projection_status != 'ok':
                payload, reason_code = _r33_m1_review_projection_failure(
                    response_payload,
                    projection_status=review_projection_status,
                    reasons=review_projection_reasons,
                    trace_id=strict_yolo_trace_id,
                    page_index=page_index,
                )
                _record_analyze_error(
                    status_code=503,
                    reason=reason_code,
                    elapsed_sec=time.time() - t_start,
                )
                return jsonify(payload), 503
            projection_runtime_snapshot = ocr_singleton_snapshot()
            projection_runtime_check = _strict_forbidden_runtime_check(
                ocr_snapshot=projection_runtime_snapshot,
            )
            if not projection_runtime_check.get('ok'):
                payload, reason_code = _vector_only_preloaded_runtime_failure(
                    projection_runtime_check,
                    trace_id=strict_yolo_trace_id,
                    page_index=page_index,
                )
                payload['r32_ocr_runtime_audit_v1'] = dict(
                    strict_runtime_audits['ocr']
                )
                payload['r32_consumer_runtime_audit_v1'] = dict(
                    strict_runtime_audits['consumer']
                )
                _record_analyze_error(
                    status_code=503,
                    reason=reason_code,
                    elapsed_sec=time.time() - t_start,
                )
                return jsonify(payload), 503
            projection_reasons = validate_vector_only_success_payload(
                response_payload,
                expected_trace_id=strict_yolo_trace_id,
                expected_page_index=page_index,
                validation_session=strict_success_validation_session,
            )
            if projection_reasons:
                payload = fail_closed_payload({
                    'schema_version': 'vector_only_error_v1',
                    'code': 'vector_only_contract_violation',
                    'status': 'fail_closed',
                    'reasons': projection_reasons,
                    'trace_id': strict_yolo_trace_id,
                    'page_index': page_index,
                    'requested_backend': 'yolo',
                    'effective_backend': 'unavailable',
                    'fallback': False,
                })
                payload['r32_ocr_runtime_audit_v1'] = dict(
                    strict_runtime_audits['ocr']
                )
                payload['r32_consumer_runtime_audit_v1'] = dict(
                    strict_runtime_audits['consumer']
                )
                _record_analyze_error(
                    status_code=503,
                    reason='vector_only_contract_violation',
                    elapsed_sec=time.time() - t_start,
                )
                return jsonify(payload), 503
        _record_analyze_success(
            pdf_type=pdf_type,
            page_number=page_number,
            pdf_pages=len(pdf_file.pages),
            elapsed_sec=time.time() - t_start,
            payload=response_payload,
            pipe=pipe,
            pipeline_diagnostics=strict_pipeline_diagnostics,
            response_diagnostics=strict_response_diagnostics,
        )
        response = jsonify(response_payload)
        if is_vector_only:
            publish_strict_pipeline_diagnostics_v1(
                strict_pipeline_diagnostics
            )
            publish_strict_response_diagnostics_v1(
                strict_response_diagnostics
            )
        return response

    except VectorOnlyStageUnavailable as e:
        payload = fail_closed_payload(e.as_payload())
        if strict_runtime_audits is not None:
            payload['r32_ocr_runtime_audit_v1'] = dict(
                strict_runtime_audits['ocr']
            )
            payload['r32_consumer_runtime_audit_v1'] = dict(
                strict_runtime_audits['consumer']
            )
        _record_analyze_error(
            status_code=503,
            reason=e.reason,
            elapsed_sec=time.time() - t_start,
        )
        return jsonify(payload), 503

    except VectorOnlyRuntimeForbidden as e:
        payload = fail_closed_payload({
            'schema_version': 'vector_only_error_v1',
            'code': 'vector_only_runtime_forbidden',
            'status': 'fail_closed',
            'reasons': [f'{e.kind}:{e.entrypoint}'],
            'trace_id': strict_yolo_trace_id,
            'page_index': page_number - 1,
            'requested_backend': 'yolo',
            'effective_backend': 'unavailable',
            'fallback': False,
            'feeds_display_l3': False,
            'consumer_allowed': False,
        })
        if strict_runtime_audits is not None:
            payload['r32_ocr_runtime_audit_v1'] = dict(
                strict_runtime_audits['ocr']
            )
            payload['r32_consumer_runtime_audit_v1'] = dict(
                strict_runtime_audits['consumer']
            )
        _record_analyze_error(
            status_code=503,
            reason='vector_only_runtime_forbidden',
            elapsed_sec=time.time() - t_start,
        )
        return jsonify(payload), 503

    except StrictYoloViewUnavailable as e:
        _record_analyze_error(
            status_code=503,
            reason=e.reason,
            elapsed_sec=time.time() - t_start,
        )
        payload = fail_closed_payload(e.as_payload())
        if strict_runtime_audits is not None:
            payload['r32_ocr_runtime_audit_v1'] = dict(
                strict_runtime_audits['ocr']
            )
            payload['r32_consumer_runtime_audit_v1'] = dict(
                strict_runtime_audits['consumer']
            )
        return jsonify(payload), 503

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error('analyze error:\n%s', tb)
        if recognition_settings is not None and recognition_settings.is_vector_only:
            payload = fail_closed_payload({
                'schema_version': 'vector_only_error_v1',
                'code': 'vector_only_internal_error',
                'status': 'fail_closed',
                'reasons': ['vector_only_internal_error'],
                'trace_id': strict_yolo_trace_id,
                'page_index': page_number - 1,
                'requested_backend': 'yolo',
                'effective_backend': 'unavailable',
                'fallback': False,
                'feeds_display_l3': False,
                'consumer_allowed': False,
            })
            if strict_runtime_audits is not None:
                payload['r32_ocr_runtime_audit_v1'] = dict(
                    strict_runtime_audits['ocr']
                )
                payload['r32_consumer_runtime_audit_v1'] = dict(
                    strict_runtime_audits['consumer']
                )
            _record_analyze_error(
                status_code=500,
                reason='vector_only_internal_error',
                elapsed_sec=time.time() - t_start,
            )
            return jsonify(payload), 500
        _record_analyze_error(
            status_code=500,
            reason='internal_error',
            elapsed_sec=time.time() - t_start,
        )
        return _json_error('分析内部错误', 500)

    finally:
        _cancel_strict_yolo_task(strict_yolo_task, 'request_finished')
        if strict_fitz_doc is not None:
            try:
                strict_fitz_doc.close()
            except Exception:
                pass
        _tee_local.log_buffer = None
        close_fitz_cache()
        pdf_file.close()


# ── POST /analyze_stamp ────────────────────────────────────────
#
# 手绘章定点 OCR，不走完整管线。前端 fire-and-forget 调用，失败静默。
# 复用 PaddleOCR 单例（_get_ocr），与 /analyze 共享 _ocr_lock 串行化。
# 生产模式由 strict-only 策略在解析 PDF 前禁用。

@app.route('/analyze_stamp', methods=['POST'])
def analyze_stamp():
    _bump('analyze_stamp_calls')
    data = request.get_json(silent=True)
    if not data:
        return _json_error('请求体必须是 JSON', 400)

    production_rejection = _production_legacy_recognition_route_rejection(
        '/analyze_stamp'
    )
    if production_rejection is not None:
        payload, _reason = production_rejection
        return jsonify(payload), 422

    pdf_b64 = data.get('pdf_base64')
    try:
        pdf_bytes = _decode_pdf_base64(pdf_b64)
        page_number = _parse_int_field(data, 'page', 1, min_value=1)
        dpi = _parse_int_field(data, 'dpi', Config.DPI_HIRES,
                               min_value=72, max_value=600)
    except ValueError as e:
        return _json_error(str(e), 400)
    page_index = page_number - 1
    crop = data.get('crop_bbox') or {}
    try:
        cx = float(crop['x']); cy = float(crop['y'])
        cw = float(crop['w']); ch = float(crop['h'])
    except (KeyError, TypeError, ValueError):
        return _json_error('缺少 crop_bbox {x,y,w,h}', 400)
    if not all(math.isfinite(v) for v in (cx, cy, cw, ch)) or cw <= 0 or ch <= 0:
        return _json_error('crop_bbox 必须是有限数值且 w/h > 0', 400)

    # PDF 坐标 (72 DPI) → 渲染像素坐标
    scale = dpi / 72.0
    px = int(cx * scale)
    py = int(cy * scale)
    pw = max(1, int(cw * scale))
    ph = max(1, int(ch * scale))

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return _json_error(f'PDF 解析失败: {e}', 422)

    try:
        if page_index < 0 or page_index >= doc.page_count:
            return _json_error('页码越界', 400)
        page = doc[page_index]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 裁 crop 区域（+2px 余量防边界截断）
        x0 = max(0, px - 2)
        y0 = max(0, py - 2)
        x1 = min(img.width,  px + pw + 2)
        y1 = min(img.height, py + ph + 2)
        if x1 <= x0 or y1 <= y0:
            return _json_error('crop 区域无效', 400)
        roi = img.crop((x0, y0, x1, y1))

        from ocr_engine import _get_ocr
        import numpy as np
        ocr = _get_ocr()
        raw = ocr.ocr(np.array(roi), cls=False)

        texts = []
        if raw and raw[0]:
            for line in raw[0]:
                bbox_pts, (txt, conf) = line
                texts.append({'text': txt, 'conf': float(conf)})

        merged = ' '.join(t['text'] for t in texts) if texts else ''

        return jsonify({
            'status': 'success',
            'text': merged.strip(),
            'candidates': texts,
        })

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error('analyze_stamp error:\n%s', tb)
        return _json_error('手绘章 OCR 内部错误', 500)

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


# ── 入口 ────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 多线程模式：threaded=True 允许前端并行发送多页请求
    # 线程安全保障：_fitz_cache 用 threading.local、PaddleOCR 用 _ocr_lock 串行化、
    #   stdout 用 _ThreadLocalTee 线程隔离日志捕获
    # use_reloader=False：Flask stat reloader 在 PaddleOCR 加载模型时会
    # kill 掉子进程触发 RemoteDisconnected；debug 由 Config.FLASK_DEBUG 控制
    print(f"[app] config: {Config.summary()}")
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT,
            debug=Config.FLASK_DEBUG, threaded=True, use_reloader=False)
