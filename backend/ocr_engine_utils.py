"""ocr_engine_utils.py — OCR 工具函数 + PaddleOCR 单例

从 ocr_engine.py 拆出。"""

import math
import io
import platform as _plat
import re as _re_module
import threading
from contextlib import contextmanager
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

from vector_only_runtime_guard import guard_ocr_call, guard_ocr_initialization

# 调试输出控制：由 app.py（默认 False）或 debug 脚本（设为 True）控制
VERBOSE = False

# ---------------------------------------------------------------------------
# 全局 PaddleOCR 单例（延迟初始化，避免每次请求重新加载模型）
# 线程安全：_ocr_lock 保护初始化 + 推理（PaddleOCR 非线程安全）
# ---------------------------------------------------------------------------

_paddle_ocr = None
_ocr_lock = threading.Lock()
_selected_backend = None
_ocr_by_backend = {}
_ocr_override = threading.local()


def get_selected_ocr_backend() -> str:
    """返回 _get_ocr() 实际加载的 backend 名（'rapidocr' / 'paddleocr' / ''）。"""
    return _selected_backend or ''


def get_preferred_ocr_backend() -> str:
    """Return the backend that lazy OCR will try first without loading OCR models."""
    import os as _os
    is_arm_mac = (_plat.system() == 'Darwin' and _plat.machine() == 'arm64')
    explicit = _os.environ.get('OCR_BACKEND', '').strip().lower()
    if explicit == 'rapidocr':
        return 'rapidocr'
    if explicit == 'paddleocr' and not is_arm_mac:
        return 'paddleocr'
    return 'rapidocr' if is_arm_mac else 'paddleocr'


def _can_import(module_name: str) -> bool:
    import importlib.util as _importlib_util
    try:
        return _importlib_util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def ocr_engine_capabilities() -> dict:
    """Return local OCR engine availability without loading OCR models."""
    is_arm_mac = (_plat.system() == 'Darwin' and _plat.machine() == 'arm64')
    rapid_ok = _can_import('rapidocr')
    if is_arm_mac:
        paddle = {'available': False, 'reason': 'arm_mac_paddle_segfault'}
    elif not _can_import('paddleocr') and not _can_import('paddle'):
        paddle = {'available': False, 'reason': 'paddleocr_not_installed'}
    else:
        paddle = {'available': True, 'reason': 'ok'}
    return {
        'rapidocr': {
            'available': rapid_ok,
            'reason': 'ok' if rapid_ok else 'rapidocr_not_installed',
        },
        'paddle': paddle,
        'yolo_char': {'available': False, 'reason': 'not_built'},
    }


@contextmanager
def ocr_backend_override(backend: str | None):
    """Temporarily route OCR loading to a request-selected backend."""
    prev = getattr(_ocr_override, 'backend', None)
    _ocr_override.backend = backend
    try:
        yield
    finally:
        _ocr_override.backend = prev


def _is_invalid_ocr_image(img) -> bool:
    if img is None:
        return True
    shape = getattr(img, 'shape', None)
    if shape is not None:
        if len(shape) < 2:
            return True
        try:
            return int(shape[0]) <= 0 or int(shape[1]) <= 0
        except (TypeError, ValueError):
            return True
    size = getattr(img, 'size', None)
    if size is not None and not isinstance(size, (int, float)):
        try:
            return int(size[0]) <= 0 or int(size[1]) <= 0
        except (TypeError, ValueError, IndexError):
            return True
    return False


class _RapidOCRShim:
    """
    PaddleOCR API 兼容 shim，内部调用 RapidOCR 原生 API。

    此适配器仅服务 legacy/diagnostic OCR 调用，严格矢量路径不调用它。
    通过 OCR_BACKEND=rapidocr 配置选择后端，平台默认选择由加载器决定。
    """

    def __init__(self):
        guard_ocr_initialization("_RapidOCRShim")
        from rapidocr import RapidOCR
        # 压制 rapidocr 3.x 的 INFO / WARNING 噪声（GD&T 空 ROI 会大量刷屏）。
        # 不能只在 import 后 setLevel：RapidOCR() 构造时会按配置把 logger
        # 改回 info，所以必须通过它自己的 params 覆盖默认配置。
        self._rapid = RapidOCR(params={"Global.log_level": "error"})

    def ocr(self, img, cls=False):
        if _is_invalid_ocr_image(img):
            return [None]
        # rapidocr 3.x 返回 RapidOCROutput(boxes=ndarray[N,4,2], txts=tuple, scores=tuple)
        # use_cls=False 跳过角度分类器（工程图 ROI 都是水平/已旋转）
        res = self._rapid(img, use_cls=False)
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return [None]
        lines = []
        for box, text, conf in zip(res.boxes, res.txts, res.scores):
            # box: ndarray shape (4,2) → list of [x, y]
            bbox = [[float(pt[0]), float(pt[1])] for pt in box]
            lines.append([bbox, (text, float(conf))])
        return [lines]


class _ThreadSafeOCR:
    """线程安全 OCR 包装器：用 _ocr_lock 串行化所有推理调用。

    所有现有 ``ocr.ocr(img, cls=False)`` 调用点无需修改。
    """

    def __init__(self, impl):
        self._impl = impl

    def ocr(self, img, cls=False):
        guard_ocr_call("_ThreadSafeOCR.ocr")
        with _ocr_lock:
            return self._impl.ocr(img, cls=cls)


def _init_paddle_ocr():
    """Instantiate PaddleOCR. 默认 GPU 优先；GPU 加载失败自动降 CPU 重试。

    显式关 GPU：`PADDLE_USE_GPU=0`（或 false/no）。
    探测 paddle 是 CPU 版时直接走 CPU 路径，不浪费一次失败的 GPU 加载。
    """
    guard_ocr_initialization("_init_paddle_ocr")
    import os as _os
    from paddleocr import PaddleOCR
    _models_root = _os.path.join(_os.path.dirname(__file__), 'models')

    # 启动时打印 CUDA 编译、GPU 数量和期望配置，便于诊断实际运行后端。
    cuda_compiled = False
    try:
        import paddle  # type: ignore
        cuda_compiled = paddle.is_compiled_with_cuda()
        gpu_count = paddle.device.cuda.device_count() if cuda_compiled else 0
        print(f"[ocr] paddle: cuda_compiled={cuda_compiled}, gpu_count={gpu_count}")
    except Exception as e:
        print(f"[ocr] paddle device 探测失败: {e}")

    user_force_cpu = _os.environ.get('PADDLE_USE_GPU', '').strip().lower() in ('0', 'false', 'no')

    common = dict(
        use_angle_cls=False, lang='en', show_log=False,
        det_model_dir=_os.environ.get(
            'PADDLE_DET_MODEL',
            _os.path.join(_models_root, 'det_en')),
        rec_model_dir=_os.environ.get(
            'PADDLE_REC_MODEL',
            _os.path.join(_models_root, 'rec_en')),
    )

    # 默认走 GPU；仅当 paddle 不带 CUDA 或用户显式 PADDLE_USE_GPU=0 时直接走 CPU
    if cuda_compiled and not user_force_cpu:
        try:
            ocr = PaddleOCR(use_gpu=True, **common)
            print("[ocr] paddle: GPU mode loaded")
            return ocr
        except Exception as e:
            print(f"[ocr] paddle GPU 加载失败 → fallback CPU: {e}")

    if not cuda_compiled and not user_force_cpu:
        print("[ocr] WARN: paddlepaddle 是 CPU 版（请装 paddlepaddle-gpu 才能 GPU 加速）")

    ocr = PaddleOCR(use_gpu=False, **common)
    print("[ocr] paddle: CPU mode loaded")
    return ocr


def _get_ocr():
    guard_ocr_initialization("_get_ocr")
    override = getattr(_ocr_override, 'backend', None)
    if override is None:
        return _get_ocr_default()
    return _get_ocr_for_backend(override)


def _get_ocr_for_backend(backend: str):
    guard_ocr_initialization("_get_ocr_for_backend")
    is_arm_mac = (_plat.system() == 'Darwin' and _plat.machine() == 'arm64')
    if backend == 'paddleocr' and is_arm_mac:
        backend = 'rapidocr'
    if backend not in ('rapidocr', 'paddleocr'):
        backend = get_preferred_ocr_backend()

    global _selected_backend
    if backend in _ocr_by_backend:
        _selected_backend = backend
        return _ocr_by_backend[backend]

    with _ocr_lock:
        if backend not in _ocr_by_backend:
            raw = _RapidOCRShim() if backend == 'rapidocr' else _init_paddle_ocr()
            _ocr_by_backend[backend] = _ThreadSafeOCR(raw)
        _selected_backend = backend
    return _ocr_by_backend[backend]


def _get_ocr_default():
    guard_ocr_initialization("_get_ocr_default")
    """惰性获取 OCR 单例 — 浮动切换 rapidocr / paddleocr。

    选择顺序：
      1. OCR_BACKEND=rapidocr  → 优先 rapidocr，失败回退 paddleocr（除 ARM Mac）
      2. OCR_BACKEND=paddleocr → 优先 paddleocr，失败回退 rapidocr
      3. 未设                  → 优先 paddleocr（GPU 主路径），失败回退 rapidocr
                                 （ARM Mac 上 paddle segfault，强制 rapidocr）

    ARM Mac 红线：paddlepaddle 在 darwin/arm64 跑推理 segfault，
    所以在该平台禁用 paddleocr 回退（哪怕用户 explicit 设了 paddleocr）。
    """
    global _paddle_ocr
    if _paddle_ocr is None:
        with _ocr_lock:
            if _paddle_ocr is None:  # double-check after acquiring lock
                import os as _os
                import platform as _plat
                _is_arm_mac = (_plat.system() == 'Darwin' and _plat.machine() == 'arm64')
                _explicit = _os.environ.get('OCR_BACKEND', '').lower()

                debug_ocr = VERBOSE or _os.environ.get('OCR_DEBUG', '').strip().lower() in ('1', 'true', 'yes')
                if debug_ocr:
                    print(f"[ocr-debug] OCR_BACKEND env = {repr(_os.environ.get('OCR_BACKEND', '<unset>'))}")
                    print(f"[ocr-debug] _is_arm_mac = {_is_arm_mac}, _explicit = {repr(_explicit)}")

                if _explicit == 'rapidocr':
                    order = (['rapidocr', 'paddleocr'] if not _is_arm_mac else ['rapidocr'])
                else:
                    # explicit=paddleocr 或 未设：默认走 paddleocr（GPU 主路径）
                    order = (['paddleocr', 'rapidocr'] if not _is_arm_mac else ['rapidocr'])

                if debug_ocr:
                    print(f"[ocr-debug] order = {order}")

                raw = None
                errs = []
                global _selected_backend
                for backend in order:
                    try:
                        if backend == 'rapidocr':
                            raw = _RapidOCRShim()
                        else:
                            raw = _init_paddle_ocr()
                        _selected_backend = backend
                        print(f'[ocr] backend selected: {backend}')
                        break
                    except Exception as e:
                        # 失败原因打印到 stdout，让 _STARTUP_LOG_TEXT / 终端都能看到
                        # 哪个 backend 挂了、为什么挂，方便诊断（之前只塞 errs 列表，
                        # 在没全部失败时永远看不到，导致 PaddleOCR 静默降级到 rapidocr）
                        msg = f'{backend}: {type(e).__name__}: {e}'
                        errs.append(msg)
                        if debug_ocr:
                            print(f'[ocr] {backend} 加载失败 → 尝试下一个: {msg}')
                        # 第一次 paddleocr 失败时打印 traceback 让根因可见
                        if debug_ocr and backend == 'paddleocr':
                            import traceback as _tb
                            _tb.print_exc()
                        continue

                if raw is None:
                    raise RuntimeError(
                        '没有可用的 OCR 后端。尝试过：\n  - ' + '\n  - '.join(errs) +
                        '\n请安装 rapidocr (`pip install rapidocr`) 或 paddleocr。'
                    )

                _paddle_ocr = _ThreadSafeOCR(raw)
    return _paddle_ocr


# ---------------------------------------------------------------------------
# Step 1: 渲染 PDF 页面为高分辨率位图
# ---------------------------------------------------------------------------

def render_page_to_image(pdf_bytes: bytes, page_index: int, dpi: int = 200) -> Image.Image:
    """
    使用 PyMuPDF 将 PDF 指定页渲染为 PIL Image。

    参数:
        pdf_bytes: PDF 文件的二进制内容
        page_index: 0-based 页码
        dpi: 渲染分辨率，默认 200 DPI

    返回:
        PIL.Image.Image (RGB 模式)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        # PDF 原始坐标是 72 DPI，缩放倍数 = dpi / 72
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Step 2: 坐标变换工具
# ---------------------------------------------------------------------------

def pdf_bbox_to_pixel(bbox: dict, dpi: int = 200) -> tuple:
    """
    将 PDF 原始坐标（72 DPI，pdfplumber 坐标系：左上角原点，Y 向下）
    转换为渲染图像的像素坐标。

    参数:
        bbox: {'x': float, 'y': float, 'w': float, 'h': float}  —— PDF 坐标
        dpi: 渲染 DPI

    返回:
        (px, py, pw, ph) —— 整数像素坐标
    """
    scale = dpi / 72.0
    px = int(bbox['x'] * scale)
    py = int(bbox['y'] * scale)
    pw = int(bbox['w'] * scale)
    ph = int(bbox['h'] * scale)
    return px, py, pw, ph


def pixel_bbox_to_pdf(pixel_bbox: dict, dpi: int = 200) -> dict:
    """
    将像素坐标（相对于页面左上角）转回 PDF 坐标（72 DPI）。

    参数:
        pixel_bbox: {'x': int, 'y': int, 'w': int, 'h': int}

    返回:
        {'x': float, 'y': float, 'w': float, 'h': float}
    """
    scale = 72.0 / dpi
    return {
        'x': pixel_bbox['x'] * scale,
        'y': pixel_bbox['y'] * scale,
        'w': pixel_bbox['w'] * scale,
        'h': pixel_bbox['h'] * scale,
    }



def quad_to_bbox(quad: list) -> dict:
    """
    将四点多边形转换为轴对齐包围盒 {'x', 'y', 'w', 'h'}（像素）。
    """
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    x = min(xs)
    y = min(ys)
    return {'x': x, 'y': y, 'w': max(xs) - x, 'h': max(ys) - y}


def rotate_bbox_back(bbox_in_rotated: dict, angle_deg: float,
                     roi_w: int, roi_h: int) -> dict:
    """
    将旋转后 ROI 内的 bbox 反向旋转回原始 ROI 坐标。

    参数:
        bbox_in_rotated: {'x', 'y', 'w', 'h'} —— 旋转后图像内坐标
        angle_deg: 旋转角度（正值 = 逆时针，PIL rotate 的 expand=True）
        roi_w, roi_h: 原始（未旋转）ROI 的像素尺寸

    返回:
        {'x', 'y', 'w', 'h'} —— 原始 ROI 坐标系内的 bbox
    """
    if angle_deg == 0:
        return bbox_in_rotated

    x_ocr = bbox_in_rotated['x']
    y_ocr = bbox_in_rotated['y']
    w_ocr = bbox_in_rotated['w']
    h_ocr = bbox_in_rotated['h']

    # 将 PIL 角度（正=CCW）转换为顺时针方向，统一判断分支
    # PIL rotate(-90) = CW90, rotate(90) = CCW90 = CW270
    cw_angle = int(-angle_deg) % 360

    if cw_angle == 90:
        # CW90 reverse: rotated (x',y') → orig (y', H-x')
        return {
            'x': y_ocr,
            'y': roi_h - x_ocr - w_ocr,
            'w': h_ocr,
            'h': w_ocr,
        }
    elif cw_angle == 270:
        # CW270 reverse: rotated (x',y') → orig (W-y', x')
        return {
            'x': roi_w - y_ocr - h_ocr,
            'y': x_ocr,
            'w': h_ocr,
            'h': w_ocr,
        }

    # 通用角度回退（非 90/270）
    rad = math.radians(angle_deg)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))
    rot_w = int(roi_w * cos_a + roi_h * sin_a)
    rot_h = int(roi_w * sin_a + roi_h * cos_a)

    cx_rot = x_ocr + w_ocr / 2 - rot_w / 2
    cy_rot = y_ocr + h_ocr / 2 - rot_h / 2

    rad_back = -rad
    cx_orig = cx_rot * math.cos(rad_back) - cy_rot * math.sin(rad_back) + roi_w / 2
    cy_orig = cx_rot * math.sin(rad_back) + cy_rot * math.cos(rad_back) + roi_h / 2

    # The OCR box is axis-aligned in the rotated image.  After mapping the
    # centre back into the original ROI, expand width/height to the enclosing
    # axis-aligned box in the original coordinate system.
    w_out = w_ocr * cos_a + h_ocr * sin_a
    h_out = w_ocr * sin_a + h_ocr * cos_a

    return {
        'x': cx_orig - w_out / 2,
        'y': cy_orig - h_out / 2,
        'w': w_out,
        'h': h_out,
    }


def _rotation_to_horizontal(angle_deg: float) -> float:
    """Return the PIL rotation that makes a 0..180° text/line angle horizontal."""
    normalized = float(angle_deg or 0.0) % 180.0
    if normalized <= 90.0:
        return -normalized
    return 180.0 - normalized


# ---------------------------------------------------------------------------
# Step 3: 单个 ROI 图像预处理
# ---------------------------------------------------------------------------

def _extract_roi(page_img: Image.Image, region: dict, dpi: int,
                 near_angle_deg: float = 0.0) -> tuple:
    """
    从页面图像中切出单个 text_region 对应的 ROI，并做旋转预处理。

    参数:
        page_img: 整页 PIL Image
        region: {'x', 'y', 'w', 'h', ...} PDF 坐标
        dpi: 渲染 DPI
        near_angle_deg: 附近斜线角度（度，0 = 不旋转）

    返回:
        (roi_img, origin_px, angle_applied)
        roi_img: PIL Image（已旋转）
        origin_px: (px, py) ROI 在页面像素坐标中的左上角（旋转前）
        angle_applied: 实际应用的旋转角度
    """
    px, py, pw, ph = pdf_bbox_to_pixel(region, dpi)
    page_w, page_h = page_img.size

    normalized = float(near_angle_deg or 0.0) % 180.0
    is_angled = not (normalized < 10 or normalized > 170 or 80 < normalized < 100)

    # 动态 padding：斜向文字按长边扩，避免旋平前裁掉字头/字尾。
    if is_angled:
        pad = max(int(max(pw, ph) * 0.6), 14)
    else:
        pad = max(int(ph * 0.4), 10)

    left   = max(0, px - pad)
    top    = max(0, py - pad)
    right  = min(page_w, px + pw + pad)
    bottom = min(page_h, py + ph + pad)

    roi = page_img.crop((left, top, right, bottom))
    origin_px = (left, top)

    # 极小 ROI 上采样（PaddleOCR DBNet 需要一定分辨率）
    roi_w, roi_h = roi.size
    if roi_w * roi_h < 100:
        scale_up = 3
    elif roi_w * roi_h < 400:
        scale_up = 2
    else:
        scale_up = 1

    if scale_up > 1:
        roi = roi.resize((roi_w * scale_up, roi_h * scale_up), Image.LANCZOS)

    # 旋转处理：竖直文字标记 angle_applied=90，由 run_directed_ocr 做双向尝试。
    # 斜向文字按附近尺寸线真实角度旋平；不要把斜线 snap 回水平，否则
    # 对斜向紧凑文字按方向旋转 ROI，避免仅尝试轴向读取。
    angle_applied = 0.0
    if 80 < normalized < 100:
        # 竖直文字：不在此处旋转，交由调用方做双向尝试
        angle_applied = 90.0
    elif not (normalized < 10 or normalized > 170):
        angle_applied = _rotation_to_horizontal(normalized)
        roi = roi.rotate(angle_applied, expand=True, fillcolor=(255, 255, 255))

    return roi, origin_px, angle_applied, scale_up


# ---------------------------------------------------------------------------
# 辅助：L1 斜线角度索引
# ---------------------------------------------------------------------------

def _build_angle_index(l1_lines: list) -> list:
    """
    将 L1 层线段列表转换为带角度信息的结构，
    便于按空间位置快速查找最近斜线。

    返回:
        [{'cx', 'cy', 'angle_deg', 'length'}, ...]
    """
    index = []
    for ln in l1_lines:
        x0, y0 = ln.get('x0', 0), ln.get('y0', 0)
        x1, y1 = ln.get('x1', 0), ln.get('y1', 0)
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 5:  # 忽略极短线段
            continue
        angle_deg = math.degrees(math.atan2(dy, dx)) % 180
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        index.append({'cx': cx, 'cy': cy, 'angle_deg': angle_deg, 'length': length})
    return index


def _find_near_angle(region: dict, angle_index: list, search_radius: float = 50.0,
                     region_id: int = -1, verbose: bool = True) -> float:
    """
    在 region 附近 search_radius PDF 单位范围内，
    查找最长的非水平/垂直 L1 线段角度。水平/竖直按轴向返回，
    真实斜线保留原始角度，交给 ROI 旋转 OCR。
    """
    # 快速 fallback：region 自身是竖直形态（h >> w），不依赖附近线段判断
    if region.get('h', 0) > 0 and region.get('w', 0) > 0:
        aspect_ratio = region['h'] / region['w']
        if aspect_ratio >= 1.4:
            if verbose:
                print(f'[angle] region {region_id}: aspect_ratio={aspect_ratio:.1f} → vertical, snapped=90°')
            return 90.0

    cx = region['x'] + region['w'] / 2
    cy = region['y'] + region['h'] / 2

    best_angle = None
    best_length = 0.0
    for entry in angle_index:
        dist = math.hypot(entry['cx'] - cx, entry['cy'] - cy)
        if dist > search_radius:
            continue
        a = entry['angle_deg']
        if entry['length'] > best_length:
            best_length = entry['length']
            best_angle = a

    if best_angle is None:
        if verbose:
            print(f'[angle] region {region_id}: no nearby diagonal → 0°')
        return 0.0

    snap_tolerance = 15.0
    if best_angle < snap_tolerance or best_angle > 180.0 - snap_tolerance:
        if verbose:
            print(f'[angle] region {region_id}: raw={best_angle:.1f}° → snapped=0°')
        return 0.0
    if abs(best_angle - 90.0) <= snap_tolerance:
        if verbose:
            print(f'[angle] region {region_id}: raw={best_angle:.1f}° → snapped=90°')
        return 90.0

    if verbose:
        print(f'[angle] region {region_id}: diagonal={best_angle:.1f}°')
    return float(best_angle)


# ---------------------------------------------------------------------------
# bbox IoU / containment 工具
# ---------------------------------------------------------------------------

def _bbox_iou(a: dict, b: dict) -> float:
    """计算两个 bbox 的 IoU（Intersection over Union）。"""
    ax1, ay1 = a['x'], a['y']
    ax2, ay2 = ax1 + a['w'], ay1 + a['h']
    bx1, by1 = b['x'], b['y']
    bx2, by2 = bx1 + b['w'], by1 + b['h']

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = a['w'] * a['h']
    area_b = b['w'] * b['h']
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_containment(inner: dict, outer: dict) -> float:
    """inner 落在 outer 内的面积占 inner 自身面积的比例（0-1）。"""
    ix1 = max(inner['x'], outer['x'])
    iy1 = max(inner['y'], outer['y'])
    ix2 = min(inner['x'] + inner['w'], outer['x'] + outer['w'])
    iy2 = min(inner['y'] + inner['h'], outer['y'] + outer['h'])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_inner = inner['w'] * inner['h']
    return inter / area_inner if area_inner > 0 else 0.0


# ---------------------------------------------------------------------------
# 工程字符正则 + 公差工具
# ---------------------------------------------------------------------------

# 有效工程字符正则（数字、字母、工程符号 + GD&T 12 类符号）
# 与 gdt_detector.CLASS_SYMBOLS 保持同步，避免 dedup_ocr_engineering 误判含 GD&T
# 符号的 compartment OCR 结果有效长度偏低。
_ENGINEERING_CHARS_RE = _re_module.compile(r'[^0-9A-Za-z±+.\-⌀∅Ø°×xX⊕⊘⊥∥◎○⌭—∠⌓⌒▱≡]')
# 公差符号
_TOLERANCE_CHARS = frozenset('±')

R2P_DIGIT_RESTRICTED_OCR_ENV = 'R2P_DIGIT_RESTRICTED_OCR'
_R2P_DIGIT_RESTRICTED_SCHEMA = 'r2p_digit_restricted_ocr_v1'
_R2P_DIGIT_RESTRICTED_ALLOWED = frozenset('0123456789.,')
_R2P_DIGIT_RESTRICTED_FALLBACK_RE = _re_module.compile(r'[MmXx×-]')
_R2P_DIGIT_RESTRICTED_TRANSLATION = str.maketrans({
    '，': ',',
    '。': '.',
    '－': '-',
    '﹣': '-',
    '−': '-',
})


def is_r2p_digit_restricted_ocr_enabled() -> bool:
    """Runtime flag for WS-A shadow OCR filtering. Default is off."""
    import os as _os
    raw = _os.environ.get(R2P_DIGIT_RESTRICTED_OCR_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def build_r2p_digit_restricted_ocr_shadow(text: str, backend: str | None = None) -> dict:
    """
    Build dump-only metadata for the WS-A digit-restricted OCR experiment.

    This is deliberately post-recognition and does not mutate the OCR text that
    feeds the current assembler/final dimensions. Tokens with M/x/×/- are kept
    as full-dictionary fallbacks because they often encode thread/count/range
    syntax rather than pure numeric residuals.
    """
    original = str(text or '').strip()
    compact = _re_module.sub(r'\s+', '', original).translate(
        _R2P_DIGIT_RESTRICTED_TRANSLATION)
    selected_backend = backend or get_selected_ocr_backend() or get_preferred_ocr_backend()

    if not compact:
        status = 'blank'
        restricted_text = ''
        fallback_reason = 'blank_token'
    elif _R2P_DIGIT_RESTRICTED_FALLBACK_RE.search(compact):
        status = 'fallback_full_dictionary_token'
        restricted_text = original
        fallback_reason = 'contains_m_x_multiply_or_minus'
    else:
        restricted_text = ''.join(
            ch for ch in compact if ch in _R2P_DIGIT_RESTRICTED_ALLOWED)
        if not restricted_text:
            status = 'fallback_full_dictionary_token'
            restricted_text = original
            fallback_reason = 'empty_after_digit_filter'
        elif restricted_text == compact:
            status = 'unchanged_allowed'
            fallback_reason = None
        else:
            status = 'filtered'
            fallback_reason = None

    payload = {
        'schema_version': _R2P_DIGIT_RESTRICTED_SCHEMA,
        'mode': 'shadow_post_filter',
        'consumer_allowed': False,
        'backend': selected_backend,
        'original_text': original,
        'restricted_text': restricted_text,
        'status': status,
        'would_change_text': restricted_text != original,
        'env_flag': R2P_DIGIT_RESTRICTED_OCR_ENV,
    }
    if fallback_reason:
        payload['fallback_reason'] = fallback_reason
    return payload


def attach_r2p_digit_restricted_ocr_shadow(result: dict, text: str) -> dict:
    """Attach WS-A shadow metadata when enabled; leave default output byte-equal."""
    if is_r2p_digit_restricted_ocr_enabled():
        result['_r2p_digit_restricted_ocr'] = build_r2p_digit_restricted_ocr_shadow(text)
    return result


def _effective_eng_len(text: str) -> int:
    """计算有效工程字符数（数字+字母+工程符号，过滤噪声字符）。"""
    return len(_ENGINEERING_CHARS_RE.sub('', text))


def _has_tolerance(text: str) -> bool:
    """是否包含公差符号。"""
    return bool(_TOLERANCE_CHARS & set(text)) or '+/-' in text
