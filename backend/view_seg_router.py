"""view_seg_router — 按环境变量分发 view 切分到不同 backend。

用法：在 pipeline.py 改 import:
    from view_seg_router import segment_views

环境变量 VIEW_SEG_BACKEND:
  yolo        (default)  R13+ YOLO11n 微调模型（view_model.pt），生产路径
  watershed              R0-R10 watershed 算法（fallback / 兜底）
  minesweeper            R9-R10 几何算法（experimental）

YOLO backend 默认加载 yolo_views/models/view_model.pt。
通过 YOLO_VIEW_MODEL 环境变量覆盖路径。

容错：YOLO 加载/推理失败时会 latch 到 watershed，避免整个 /analyze 链路崩。
"""
import os
import json
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path

# 一旦 yolo backend 失败（ultralytics 缺 / 模型缺 / 推理 panic），
# latch 到 watershed，不再反复尝试。重启进程后重试。
_yolo_failed = False
_last_yolo_error = ""
_last_yolo_model = ""
_last_view_seg_status = {}
_VIEW_SEG_BACKENDS = ("yolo", "watershed", "minesweeper")


def _annotate_views(views, *, backend, source, fallback=False,
                    model=None, fallback_reason=""):
    annotated = []
    for view in views or []:
        item = dict(view)
        item.setdefault("backend", backend)
        item.setdefault("source", source)
        item.setdefault("fallback", bool(fallback))
        if model:
            item.setdefault("model", str(model))
        if fallback_reason:
            item.setdefault("fallback_reason", str(fallback_reason))
        annotated.append(item)
    return annotated


def _model_exists(model):
    if not model:
        return False
    try:
        return Path(model).exists()
    except TypeError:
        return False


def _can_import(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _default_yolo_model_path():
    explicit = os.environ.get("YOLO_VIEW_MODEL")
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parent / "yolo_views" / "models" / "view_model.pt"


def view_seg_capabilities() -> dict:
    """Return view segmentation backend availability without loading models."""
    yolo_model_ok = _model_exists(_default_yolo_model_path())
    yolo_import_ok = _can_import("ultralytics")
    yolo_ok = bool(yolo_model_ok and yolo_import_ok)
    return {
        "yolo": {
            "available": yolo_ok,
            "reason": (
                "ok"
                if yolo_ok
                else ("model_file_missing" if not yolo_model_ok else "ultralytics_not_installed")
            ),
        },
        "watershed": {"available": True, "reason": "ok"},
        "minesweeper": {"available": True, "reason": "experimental"},
    }


def _set_status(**status):
    global _last_view_seg_status
    _last_view_seg_status = {
        key: value
        for key, value in status.items()
        if value not in (None, "")
    }


def _detect_views_yolo_subprocess(page, *, model_path, pdf_bytes=None, page_index=None):
    """Run .pt YOLO-view inference outside the OCR process."""
    page_number = int(page_index) + 1 if page_index is not None else int(getattr(page, "page_number", 1) or 1)
    temp_pdf = None
    pdf_path = None
    if pdf_bytes:
        tmp = tempfile.NamedTemporaryFile(prefix="pdf_reader_yolo_view_", suffix=".pdf", delete=False)
        try:
            tmp.write(pdf_bytes)
            temp_pdf = tmp.name
            pdf_path = temp_pdf
        finally:
            tmp.close()
    else:
        pdf_obj = getattr(page, "pdf", None)
        pdf_path = getattr(pdf_obj, "path", None)
        if not pdf_path:
            pdf_path = getattr(getattr(pdf_obj, "stream", None), "name", None)

    if not pdf_path:
        raise RuntimeError("YOLO .pt subprocess requires PDF bytes or a PDF path")

    script = Path(__file__).resolve().parent / "yolo_views" / "infer_cli.py"
    env = dict(os.environ)
    env.setdefault("YOLO_ALLOW_PT_FALLBACK", "1")
    env.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "Ultralytics"))
    cmd = [
        sys.executable,
        str(script),
        "--pdf", str(pdf_path),
        "--page", str(page_number),
        "--model", str(model_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=float(os.environ.get("YOLO_VIEW_SUBPROCESS_TIMEOUT", "120")),
        )
        return json.loads(proc.stdout or "[]")
    finally:
        if temp_pdf:
            try:
                Path(temp_pdf).unlink(missing_ok=True)
            except Exception:
                pass


def get_last_view_seg_status():
    """Return telemetry for the last segment_views call, even when no views exist."""
    return dict(_last_view_seg_status)


def _segment_minesweeper(page):
    from view_seg_minesweeper import segment_views as ms_seg
    return ms_seg(page)


def _segment_watershed(page):
    from view_segmenter import segment_views as ws_seg
    return ws_seg(page)


def _segment_yolo(page, *, model_path, pdf_bytes=None, page_index=None):
    from yolo_views.infer import detect_views_yolo
    if str(model_path).endswith(".pt") and os.environ.get("YOLO_VIEW_PT_INPROCESS") != "1":
        return _detect_views_yolo_subprocess(
            page,
            model_path=model_path,
            pdf_bytes=pdf_bytes,
            page_index=page_index,
        )
    return detect_views_yolo(page, model_path=model_path)


def segment_views(page, *args, backend: str | None = None, **kwargs):
    global _yolo_failed, _last_yolo_error, _last_yolo_model
    backend = (backend or os.environ.get("VIEW_SEG_BACKEND", "yolo")).lower()
    if backend not in _VIEW_SEG_BACKENDS:
        backend = "yolo"

    if backend == "minesweeper":
        # ms_seg signature 不接受 pdf_bytes/page_index，吞掉避免 TypeError
        annotated = _annotate_views(
            _segment_minesweeper(page),
            backend="minesweeper",
            source="view_seg_minesweeper",
        )
        _set_status(
            requested_backend=backend,
            effective_backend="minesweeper",
            source="view_seg_minesweeper",
            fallback=False,
            view_count=len(annotated),
        )
        return annotated

    if backend == "watershed":
        # ws_seg signature 不接受 pdf_bytes/page_index，吞掉避免 TypeError
        annotated = _annotate_views(
            _segment_watershed(page),
            backend="watershed",
            source="view_segmenter",
        )
        _set_status(
            requested_backend=backend,
            effective_backend="watershed",
            source="view_segmenter",
            fallback=False,
            view_count=len(annotated),
        )
        return annotated

    # 默认 yolo
    if not _yolo_failed:
        try:
            from yolo_views.infer import resolve_model_path
            model_path = os.environ.get("YOLO_VIEW_MODEL")
            resolved_model = resolve_model_path(model_path)
            _last_yolo_model = str(resolved_model)
            views = _segment_yolo(
                page,
                model_path=resolved_model,
                pdf_bytes=kwargs.get("pdf_bytes"),
                page_index=kwargs.get("page_index"),
            )
            annotated = _annotate_views(
                views,
                backend="yolo",
                source="yolo_views",
                fallback=False,
                model=resolved_model,
            )
            _set_status(
                requested_backend=backend,
                effective_backend="yolo",
                source="yolo_views",
                fallback=False,
                model=_last_yolo_model,
                model_exists=_model_exists(resolved_model),
                view_count=len(annotated),
            )
            return annotated
        except Exception as e:
            print(f"[view_seg_router] yolo 失败: {e}; 后续请求降级 watershed")
            _yolo_failed = True
            _last_yolo_error = str(e)

    # ws_seg signature 不接受 pdf_bytes/page_index，吞掉避免 TypeError
    annotated = _annotate_views(
        _segment_watershed(page),
        backend="watershed",
        source="view_segmenter",
        fallback=(backend == "yolo"),
        fallback_reason=_last_yolo_error if backend == "yolo" else "",
    )
    _set_status(
        requested_backend=backend,
        effective_backend="watershed",
        source="view_segmenter",
        fallback=(backend == "yolo"),
        fallback_reason=_last_yolo_error if backend == "yolo" else "",
        model=_last_yolo_model if backend == "yolo" else "",
        model_exists=_model_exists(_last_yolo_model) if backend == "yolo" else None,
        view_count=len(annotated),
    )
    return annotated
