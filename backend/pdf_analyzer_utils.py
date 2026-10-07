"""
pdf_analyzer_utils.py — PDF 矢量解析共享工具函数

包含:
  - 常量与符号映射
  - fitz (PyMuPDF) 线程本地缓存
  - 曲线几何小工具 (_curve_bbox_wh, _curve_center)
  - IoU 计算 (_iou, _rect_iou_xyxy)
  - 线宽自适应阈值 (_adaptive_width_thresholds)
  - 标注线宽检测 (detect_annotation_linewidth)

从 pdf_analyzer.py 拆出。所有子模块均可导入此文件，无循环依赖。
"""

import math
import re
import threading

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# 常量与符号映射
# ---------------------------------------------------------------------------

SYMBOL_MAP = {
    '#': '±',
    '$': '°',
}

# GD&T / 工程图常用前缀正则
PREFIX_PATTERNS = [
    (re.compile(r'^[⌀∅Øφ]'), '⌀', 'diameter'),
    (re.compile(r'^R(?=\d)'), 'R', 'radius'),
    (re.compile(r'^M(?=\d)'), 'M', 'thread'),
    (re.compile(r'^C(?=\d)'), 'C', 'chamfer'),
    (re.compile(r'^\d+[×xX]'), None, 'repeat'),  # Optional quantity prefix.
]


# ---------------------------------------------------------------------------
# 共享工具函数
# ---------------------------------------------------------------------------

def detect_annotation_linewidth(all_lines, thin_thick):
    """从线段集合中检测 annotation 层的主 linewidth。"""
    width_counts = {}
    for ln in all_lines:
        w = ln.get('lineWidth', 0)
        if 0.01 < w < thin_thick:
            wr = round(w, 3)
            width_counts[wr] = width_counts.get(wr, 0) + 1
    if not width_counts:
        return None
    return max(width_counts, key=width_counts.get)


# ---------------------------------------------------------------------------
# 曲线几何工具
# ---------------------------------------------------------------------------

def _count_tiny_curves(curves):
    """统计满足"微小"条件的曲线数量（包围盒面积 < 100，即 w<15 且 h<15 大约）。"""
    count = 0
    for c in curves:
        w, h = _curve_bbox_wh(c)
        if w < 15 and h < 15:
            count += 1
    return count


def _curve_bbox_wh(curve):
    """从 pdfplumber curve 对象提取包围盒宽高。"""
    x0 = curve.get('x0', curve.get('pts', [[0]])[0][0] if curve.get('pts') else 0)
    x1 = curve.get('x1', x0)
    y0 = curve.get('top', curve.get('pts', [[0, 0]])[0][1] if curve.get('pts') else 0)
    y1 = curve.get('bottom', y0)
    if curve.get('pts'):
        xs = [p[0] for p in curve['pts']]
        ys = [p[1] for p in curve['pts']]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    return abs(x1 - x0), abs(y1 - y0)


def _curve_center(curve):
    """返回曲线包围盒中心 (cx, cy)。"""
    if curve.get('pts'):
        xs = [p[0] for p in curve['pts']]
        ys = [p[1] for p in curve['pts']]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x0 = curve.get('x0', 0)
    x1 = curve.get('x1', x0)
    y0 = curve.get('top', 0)
    y1 = curve.get('bottom', y0)
    return (x0 + x1) / 2, (y0 + y1) / 2


# ---------------------------------------------------------------------------
# IoU 计算
# ---------------------------------------------------------------------------

def _iou(r1, r2):
    """计算两个 region dict {x, y, w, h} 的 IoU。"""
    x1 = max(r1['x'], r2['x'])
    y1 = max(r1['y'], r2['y'])
    x2 = min(r1['x'] + r1['w'], r2['x'] + r2['w'])
    y2 = min(r1['y'] + r1['h'], r2['y'] + r2['h'])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = r1['w'] * r1['h'] + r2['w'] * r2['h'] - inter
    return inter / union if union > 0 else 0


def _rect_iou_xyxy(a, b):
    """Compute IoU between two [x0, y0, x1, y1] rects."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 线宽自适应阈值
# ---------------------------------------------------------------------------

def _adaptive_width_thresholds(lines):
    """
    自适应计算线宽分层阈值 — 双峰总长度法。
    按线宽分组计算每组的线段总长度，取总长度排名前二的两个线宽值的算术平均值作为 thin_thick。
    """

    zero_cutoff = 0.01

    nonzero_lines = [ln for ln in lines if ln['lineWidth'] > zero_cutoff]
    if not nonzero_lines:
        return {'zero_cutoff': zero_cutoff, 'thin_thick': 0.5, 'method': 'fallback'}

    # 按线宽分组，计算每组总长度
    length_by_width = {}
    for ln in nonzero_lines:
        w = round(ln['lineWidth'], 4)
        seg_len = math.hypot(ln['x1'] - ln['x0'], ln['y1'] - ln['y0'])
        length_by_width[w] = length_by_width.get(w, 0.0) + seg_len

    unique_widths = sorted(length_by_width.keys())

    if len(unique_widths) == 0:
        return {'zero_cutoff': zero_cutoff, 'thin_thick': 0.5, 'method': 'fallback'}

    if len(unique_widths) == 1:
        thin_thick = unique_widths[0] * 1.5
        return {'zero_cutoff': zero_cutoff, 'thin_thick': thin_thick, 'method': 'dual_peak_length'}

    # 找总长度排名前二的两个线宽值
    sorted_by_length = sorted(length_by_width.items(), key=lambda kv: kv[1], reverse=True)
    w1 = sorted_by_length[0][0]
    w2 = sorted_by_length[1][0]
    thin_thick = (w1 + w2) / 2.0

    print(f"[_adaptive_width_thresholds] top2 by total length: w={w1} (len={sorted_by_length[0][1]:.0f}), w={w2} (len={sorted_by_length[1][1]:.0f})")
    print(f"[_adaptive_width_thresholds] thin_thick = {thin_thick:.4f}")

    return {'zero_cutoff': zero_cutoff, 'thin_thick': thin_thick, 'method': 'dual_peak_length'}


# ---------------------------------------------------------------------------
# PyMuPDF (fitz) 线程本地缓存
# ---------------------------------------------------------------------------

_fitz_local = threading.local()  # 线程本地缓存（替代旧的全局 _fitz_doc_cache）


def _get_fitz_page(plumber_page):
    """从 pdfplumber page 获取对应的 fitz Page 对象（线程本地缓存）。"""
    try:
        pdf_obj = plumber_page.pdf
        stream = pdf_obj.stream
        if not hasattr(stream, 'read'):
            return None
        stream_id = id(stream)
        cache = getattr(_fitz_local, 'doc_cache', None)
        if cache is not None and cache[0] == stream_id:
            fitz_doc = cache[1]
        else:
            if cache is not None:
                try:
                    cache[1].close()
                except Exception:
                    pass
            stream.seek(0)
            pdf_bytes = stream.read()
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            _fitz_local.doc_cache = (stream_id, fitz_doc)
        page_num = plumber_page.page_number - 1
        return fitz_doc[page_num]
    except Exception as e:
        print(f"[warn] _get_fitz_page failed: {e}")
        return None


def close_fitz_cache():
    """关闭缓存的 fitz.Document。在请求结束时调用（线程安全）。"""
    cache = getattr(_fitz_local, 'doc_cache', None)
    if cache is not None:
        try:
            cache[1].close()
        except Exception:
            pass
        _fitz_local.doc_cache = None
