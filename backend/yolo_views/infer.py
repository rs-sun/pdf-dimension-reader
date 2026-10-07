#!/usr/bin/env python3
"""infer.py — YOLO-A view 检测推理（onnxruntime + ONNX 模型）。

ONNX 推理将模型运行依赖与训练工具分开；调用方提供获授权的模型资产。

  导出方式（在裸 Python 进程，不要在 import paddle 的进程里）：
    python -c "from ultralytics import YOLO; \
               YOLO('models/view_model.pt').export(format='onnx', opset=12, simplify=True)"
  生成的 .onnx 会出现在跟 .pt 同一目录。

输入: PDF page (pdfplumber.Page)
输出: list[dict] {x, y, w, h, conf, shape}（PDF pt 坐标）
      shape='rect'：默认；shape='circle' 则附带 cx, cy, r

集成到 view_segmenter:
  from yolo_views.infer import detect_views_yolo
  views = detect_views_yolo(page, model_path='models/view_model.onnx')

YOLO bbox 检测后复用 _detect_circle_shape 圆形判据，
为局部放大圆 / 圆形端盖等"圆形视图"补 shape/cx/cy/r 字段，前端按 shape 分支渲染。
"""
import sys
import os
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image

# 父目录加 sys.path 让 view_seg_common / view_segmenter 能 import（infer.py 在
# yolo_views/ 子目录里被相对导入）
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

YOLO_VIEWS = Path(__file__).parent
LONG_EDGE_PX = 1600  # 跟 render_pdf.py 一致（与训练时 pre-letterbox 输入对齐）

# ── 单例缓存 ─────────────────────────────────────────────────────
_session = None
_session_path = None
_session_input_name = None
_session_input_h = 640
_session_input_w = 640


def _resolve_default_model_path():
    """优先 .onnx（onnxruntime 主路径，跟 paddle 共存）；
    没有 .onnx 时 fallback 到 .pt，由 router 走子进程模式避免撞 PaddleOCR。
    """
    names = [
        "view_model.onnx",
        "view_model_compat.onnx",
        "view_model.pt",
        "view_model_compat.pt",
    ]
    for name in names:
        p = YOLO_VIEWS / "models" / name
        if p.exists():
            return p
    return YOLO_VIEWS / "models" / "view_model.onnx"


def resolve_model_path(model_path=None):
    """Resolve an explicit or default YOLO-view model path for telemetry."""
    return Path(model_path) if model_path else _resolve_default_model_path()


def get_yolo_model(model_path=None):
    """模块级单例：第一次调用加载，后续复用。
    .onnx → onnxruntime（CUDA EP 优先，CPU EP fallback）。
    .pt   → ultralytics YOLO（仅在裸进程能用；主进程会跟 paddle 撞车）。
    """
    global _session, _session_path, _session_input_name
    global _session_input_h, _session_input_w
    model_path = resolve_model_path(model_path)
    mp = str(model_path)
    if _session is not None and _session_path == mp:
        return _session

    if mp.endswith('.onnx'):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(mp, providers=providers)
        inp = sess.get_inputs()[0]
        # input shape 一般是 [1, 3, H, W]；H/W 可能是 int 或 'dynamic' string
        shape = inp.shape
        h = shape[2] if isinstance(shape[2], int) else 640
        w = shape[3] if isinstance(shape[3], int) else 640
        _session = sess
        _session_path = mp
        _session_input_name = inp.name
        _session_input_h = h
        _session_input_w = w
        print(f"[yolo_a] onnx loaded: {mp}, input={shape}, EP={sess.get_providers()}")
        return sess

    # .pt fallback（不应在主进程被走到，只为独立测试）
    from ultralytics import YOLO
    sess = YOLO(mp)
    _session = sess
    _session_path = mp
    print(f"[yolo_a] WARN: 加载 .pt 仅适合裸进程；主进程跟 paddle 共存会撞车: {mp}")
    return sess


def _render_page_to_image(page) -> Image.Image:
    """将 PDF page 渲染为适配模型输入的 PIL Image。"""
    pw, ph = float(page.width), float(page.height)
    scale = LONG_EDGE_PX / max(pw, ph)
    dpi = int(scale * 72)
    img = page.to_image(resolution=dpi).original.convert("RGB")
    w, h = img.size
    if max(w, h) != LONG_EDGE_PX:
        ratio = LONG_EDGE_PX / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img


def _letterbox(img_pil: Image.Image, new_h: int, new_w: int):
    """letterbox：保持长宽比缩放，padding 灰色到 new_h × new_w。
    返回 (NCHW float32 array, scale, dx, dy)。
    """
    w, h = img_pil.size
    scale = min(new_w / w, new_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = img_pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (new_w, new_h), (114, 114, 114))
    dx = (new_w - nw) // 2
    dy = (new_h - nh) // 2
    canvas.paste(img_resized, (dx, dy))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]       # NCHW
    return arr, scale, dx, dy


# ── 圆形视图判据 ────────────────────────────────────────────────
_BUCKETED_CACHE = {}  # id(page) → bucketed ndarray，避免一页多 bbox 重复算


def _compute_page_bucketed(page):
    """复用 view_segmenter 的 _extract_segments + _discretize + _score_grid 三件套，
    输出 bucketed 网格供 _detect_circle_shape 用。CELL=1，与 _detect_circle_shape
    的 default cell=1 对齐。结果按 page 对象 id 缓存（同一页多个 bbox 共用）。
    """
    key = id(page)
    if key in _BUCKETED_CACHE:
        return _BUCKETED_CACHE[key]
    from view_segmenter import _extract_segments, _discretize, _score_grid
    pw, ph = float(page.width), float(page.height)
    rows = int(np.ceil(ph)); cols = int(np.ceil(pw))
    segs = _extract_segments(page)
    has = _discretize(segs, rows, cols)
    bucketed = _score_grid(has)
    _BUCKETED_CACHE[key] = bucketed
    # 缓存大小防爆：只留最近 4 页
    if len(_BUCKETED_CACHE) > 4:
        _BUCKETED_CACHE.pop(next(iter(_BUCKETED_CACHE)))
    return bucketed


def _classify_view_shape(view_xywh, bucketed):
    """对 yolo 输出的 {x, y, w, h} bbox 调 view_seg_common._detect_circle_shape。
    返回 None（保持矩形）或 dict {shape:'circle', cx, cy, r}。
    """
    from view_seg_common import _detect_circle_shape
    x, y, w, h = view_xywh['x'], view_xywh['y'], view_xywh['w'], view_xywh['h']
    view_dict = {
        'x0': float(x), 'y0': float(y),
        'x1': float(x + w), 'y1': float(y + h),
    }
    return _detect_circle_shape(view_dict, bucketed, cell=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """简易 NMS。boxes (N,4) xyxy 像素坐标。"""
    if len(boxes) == 0:
        return []
    keep = []
    idx = np.argsort(-scores)
    while len(idx) > 0:
        i = idx[0]
        keep.append(int(i))
        if len(idx) == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        idx = rest[iou < iou_threshold]
    return keep


def detect_views_yolo(page, model_path=None, conf=0.25, iou_nms=0.5):
    """跟原版同签名同语义：对 pdfplumber.Page 跑 YOLO-A，返回 view bbox（PDF pt）。"""
    sess = get_yolo_model(model_path)

    pw, ph = float(page.width), float(page.height)
    img_pil = _render_page_to_image(page)
    iw, ih = img_pil.size

    # ── ultralytics fallback（仅 .pt 路径，主进程不应走）────────────
    if not str(_session_path).endswith('.onnx'):
        results = sess(img_pil, conf=conf, iou=iou_nms, verbose=False)
        out_list = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cf = float(b.conf[0])
                sx, sy = pw / iw, ph / ih
                out_list.append({
                    "x": round(x1 * sx, 2),
                    "y": round(y1 * sy, 2),
                    "w": round((x2 - x1) * sx, 2),
                    "h": round((y2 - y1) * sy, 2),
                    "shape": "rect",
                    "conf": round(cf, 3),
                })
        _augment_shapes(out_list, page)
        return out_list

    # ── onnxruntime 主路径 ────────────────────────────────────────
    inp, scale, dx, dy = _letterbox(img_pil, _session_input_h, _session_input_w)
    out = sess.run(None, {_session_input_name: inp})[0]

    # YOLOv8/11 输出: (1, 4+nc, num_anchors) 或 (1, num_anchors, 4+nc)
    # nc = 1 (单类 view) → 通道数 5
    if out.ndim == 3:
        # 判断 layout：channel-first 还是 anchor-first
        if out.shape[1] < out.shape[2]:
            arr = out[0].T  # (na, 5)
        else:
            arr = out[0]    # (na, 5)
    else:
        arr = out  # (na, 5)

    if arr.shape[1] < 5:
        return []  # defensive

    boxes_xywh = arr[:, :4]
    cls_scores = arr[:, 4:]
    scores = cls_scores.max(axis=1) if cls_scores.shape[1] > 1 else cls_scores[:, 0]

    mask = scores >= conf
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return []

    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    keep = _nms(boxes_xyxy, scores, iou_nms)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    # 反 letterbox：onnx_input 坐标 → 1600px 原图坐标
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dx) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dy) / scale

    # 1600px 原图 → PDF pt
    sx = pw / iw
    sy = ph / ih

    out_list = []
    for (x1, y1, x2, y2), cf in zip(boxes_xyxy, scores):
        out_list.append({
            "x": round(float(x1 * sx), 2),
            "y": round(float(y1 * sy), 2),
            "w": round(float((x2 - x1) * sx), 2),
            "h": round(float((y2 - y1) * sy), 2),
            "shape": "rect",
            "conf": round(float(cf), 3),
        })
    _augment_shapes(out_list, page)
    return out_list


def _augment_shapes(out_list, page):
    """对每个 yolo bbox 补 shape 信息：圆形覆盖为 shape='circle' + cx/cy/r。
    bucketed 计算失败（page 不可读 / 解析异常）时静默退化为全 rect。
    """
    if not out_list:
        return
    try:
        bucketed = _compute_page_bucketed(page)
    except Exception as e:
        print(f"[yolo_a] shape classify 跳过（bucketed 计算失败）: {e}")
        return
    for vb in out_list:
        try:
            shape_info = _classify_view_shape(vb, bucketed)
        except Exception:
            shape_info = None
        if shape_info:
            vb['shape'] = 'circle'
            vb['cx'] = round(float(shape_info['cx']), 2)
            vb['cy'] = round(float(shape_info['cy']), 2)
            vb['r']  = round(float(shape_info['r']), 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 infer.py <pdf_path> [page_idx=0] [model_path]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    page_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        views = detect_views_yolo(page, model_path=model_path)
    print(f"detected {len(views)} view(s):")
    for i, v in enumerate(views):
        print(f"  v{i+1}: bbox=({v['x']:.0f},{v['y']:.0f},"
              f"{v['x']+v['w']:.0f},{v['y']+v['h']:.0f}) conf={v['conf']:.2f}")
