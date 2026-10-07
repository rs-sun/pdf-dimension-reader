"""
pdf_analyzer_extract.py — 基础检测与提取

包含:
  - detect_pdf_type()              — PDF 类型探测
  - extract_diameter_glyphs()      — 直径符号几何检测
  - extract_lines_by_width()       — 自适应线宽分层
  - filter_annotation_segments_by_connectivity() — 连通分量预过滤

从 pdf_analyzer.py 拆出。依赖 pdf_analyzer_utils，不依赖其他 pdf_analyzer_* 子模块。
"""

import math
from collections import defaultdict

import numpy as np

from pdf_analyzer_utils import (
    _count_tiny_curves,
    _get_fitz_page,
    _adaptive_width_thresholds,
)


# ---------------------------------------------------------------------------
# Step 0: 类型探测
# ---------------------------------------------------------------------------

def detect_pdf_type(page):
    """
    判断 pdfplumber Page 对象的 PDF 类型。

    返回: "A" | "B" | "BC"
      A  — 完整矢量（有原生文字对象，chars > 50）
      B  — 文字转曲（chars = 0，微小曲线密集）
      BC — 混合（有大面积扫描位图）
    """
    # --- 1. 检查原生字符数量 ---
    chars = page.chars
    if len(chars) > 50:
        return "A"

    # --- 2. 检查是否有大面积图片 ---
    images = page.images
    page_area = page.width * page.height
    for img in images:
        img_w = img.get('width', 0) or (img.get('x1', 0) - img.get('x0', 0))
        img_h = img.get('height', 0) or (img.get('bottom', 0) - img.get('top', 0))
        # pdfplumber image 对象有 x0, top, x1, bottom
        try:
            img_area = (img['x1'] - img['x0']) * (img['bottom'] - img['top'])
        except (KeyError, TypeError):
            img_area = img_w * img_h
        if page_area > 0 and img_area / page_area > 0.40:
            return "BC"

    # --- 3. 统计微小曲线占比 ---
    curves = page.curves
    if curves:
        tiny = _count_tiny_curves(curves)
        ratio = tiny / len(curves)
        if ratio > 0.60:
            return "B"

    return "B"  # 默认


# ---------------------------------------------------------------------------
# 直径符号几何检测
# ---------------------------------------------------------------------------

def extract_diameter_glyphs(page, *, drawing_snapshot=None):
    """
    从矢量层识别 diameter 符号的几何位置。

    PaddleOCR 字典不含 diameter，所有直径标注都丢失前缀。从矢量层识别 diameter 符号
    （小闭合圆 + 穿过圆心的对角线），绕开 OCR 限制。

    混合数据源：
    - 圆：pdfplumber page.curves（正常暴露）
    - 斜线：PyMuPDF page.get_drawings()（部分斜线嵌在复合绘图里，
            pdfplumber 完全看不见）+ pdfplumber page.lines 后备

    判定（中点对中心强约束）：
    - 圆：闭合曲线 pts>=4，linewidth 0.2-0.85，尺寸 3-12pt，近似方形 (<=1.4)
    - 斜线：长度 2.5-25pt，dx>1 且 dy>1（真正斜向）
    - 匹配：斜线中点距圆心 < 3pt 且斜线长度 >= max(diameter*0.6, 3)

    返回: [{'cx','cy','w','h','x0','y0','x1','y1'}]
    坐标系：pdfplumber (左上原点)
    """
    curves = page.curves or []
    if not curves:
        return []

    drawings = None
    fitz_page = None if drawing_snapshot is not None else _get_fitz_page(page)
    if drawing_snapshot is not None:
        drawings = list(drawing_snapshot)
    elif fitz_page is not None:
        try:
            drawings = list(fitz_page.get_drawings())
        except Exception as e:
            print(f'[warn] extract_diameter_glyphs fitz.get_drawings failed: {e}')

    # ---- 1) 圆候选：pdfplumber curves ----
    circle_candidates = []
    for c in curves:
        pts = c.get('pts') or []
        if not (4 <= len(pts) <= 6):
            continue
        lw = c.get('linewidth', c.get('lineWidth', 0)) or 0
        if not (0.2 <= lw <= 0.85):
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w = x1 - x0
        h = y1 - y0
        if not (9.0 <= w <= 12.0 and 9.0 <= h <= 12.0):
            continue
        smaller = min(w, h)
        larger = max(w, h)
        if larger / max(smaller, 0.1) > 1.4:
            continue
        circle_candidates.append({
            'geometry': (x0, y0, x1, y1, w, h),
            'source_primitives': _diameter_circle_source_primitives(
                c,
                drawings,
            ),
        })

    if not circle_candidates:
        return []

    # ---- 2) 斜线：fitz.get_drawings 优先 + pdfplumber page.lines 后备 ----
    diag_endpoints = []  # [(lx0, ly0, lx1, ly1)]
    diag_source_primitives = []

    # 2a) PyMuPDF 复合绘图
    if drawings is not None:
        try:
            for drawing_order, draw in enumerate(drawings):
                lw = draw.get('width', 0) or 0
                if lw > 0 and not (0.15 <= lw <= 1.0):
                    continue
                items = draw.get('items') or []
                for item_index, item in enumerate(items):
                    if not item or len(item) < 3 or item[0] != 'l':
                        continue
                    try:
                        p1 = item[1]
                        p2 = item[2]
                        lx0, ly0 = float(p1.x), float(p1.y)
                        lx1, ly1 = float(p2.x), float(p2.y)
                    except Exception:
                        continue
                    dx = abs(lx1 - lx0)
                    dy = abs(ly1 - ly0)
                    length = math.hypot(dx, dy)
                    if not (2.5 <= length <= 25.0):
                        continue
                    if dx < 1.0 or dy < 1.0:
                        continue
                    diag_endpoints.append((lx0, ly0, lx1, ly1))
                    diag_source_primitives.append({
                        'drawing_order': int(drawing_order),
                        'item_index': int(item_index),
                        'op': 'l',
                        'role': 'slash_line',
                    })
        except Exception as e:
            print(f'[warn] extract_diameter_glyphs fitz.get_drawings failed: {e}')

    # 2b) pdfplumber page.lines 后备
    for ln in (page.lines or []):
        lx0 = ln.get('x0', 0)
        ly0 = ln.get('y0', ln.get('top', 0))
        lx1 = ln.get('x1', 0)
        ly1 = ln.get('y1', ln.get('bottom', 0))
        dx = abs(lx1 - lx0)
        dy = abs(ly1 - ly0)
        length = math.hypot(dx, dy)
        if not (2.5 <= length <= 25.0):
            continue
        if dx < 1.0 or dy < 1.0:
            continue
        lw = ln.get('linewidth', ln.get('lineWidth', 0)) or 0
        if lw > 0 and not (0.15 <= lw <= 1.0):
            continue
        diag_endpoints.append((lx0, ly0, lx1, ly1))
        diag_source_primitives.append(None)

    if not diag_endpoints:
        return []

    # ---- 3) 几何匹配（numpy 广播：中点距圆心 < 3pt + 长度 >= diameter*0.6）----
    circles_np = np.array([
        [(x0 + x1) / 2, (y0 + y1) / 2, max(w, h)]
        for candidate in circle_candidates
        for (x0, y0, x1, y1, w, h) in [candidate['geometry']]
    ])
    diags_np = np.array(diag_endpoints)

    d_mx = (diags_np[:, 0] + diags_np[:, 2]) / 2  # (M,)
    d_my = (diags_np[:, 1] + diags_np[:, 3]) / 2
    d_len = np.sqrt(
        (diags_np[:, 2] - diags_np[:, 0]) ** 2
        + (diags_np[:, 3] - diags_np[:, 1]) ** 2
    )

    c_cx = circles_np[:, 0:1]   # (N, 1)
    c_cy = circles_np[:, 1:2]
    c_diam = circles_np[:, 2:3]

    dist_to_center = np.sqrt(
        (d_mx - c_cx) ** 2 + (d_my - c_cy) ** 2
    )  # (N, M)
    min_line_len = np.maximum(c_diam * 0.6, 3.0)  # (N, 1)

    valid = (dist_to_center < 3.0) & (d_len >= min_line_len)
    has_match = valid.any(axis=1)

    glyphs = []
    seen = set()
    for idx, candidate in enumerate(circle_candidates):
        x0, y0, x1, y1, w, h = candidate['geometry']
        if not has_match[idx]:
            continue
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        key = (round(cx * 2), round(cy * 2))
        if key in seen:
            continue
        seen.add(key)
        glyph = {
            'cx': cx, 'cy': cy, 'w': w, 'h': h,
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
        }
        matched_indices = [
            match_index
            for match_index in range(len(diag_source_primitives))
            if bool(valid[idx, match_index])
        ]
        matched_lineage = {
            (
                int(source['drawing_order']),
                int(source['item_index']),
                str(source['op']),
            ): source
            for match_index, source in enumerate(diag_source_primitives)
            if bool(valid[idx, match_index])
            and isinstance(source, dict)
        }
        matched_geometry = {
            _line_endpoint_signature(diag_endpoints[match_index])
            for match_index in matched_indices
        }
        circle_lineage = candidate.get('source_primitives')
        if (
            isinstance(circle_lineage, list)
            and len(circle_lineage) == 4
            and len(matched_lineage) == 1
            and None not in matched_geometry
            and len(matched_geometry) == 1
        ):
            glyph['source_primitives'] = [
                *[dict(row) for row in circle_lineage],
                dict(next(iter(matched_lineage.values()))),
            ]
        glyphs.append(glyph)

    return glyphs


def _diameter_circle_source_primitives(curve, drawings):
    """Join one pdfplumber cubic path to one exact snapshot drawing."""
    signature = _pdfplumber_cubic_signature(curve.get('path'))
    if signature is None or not isinstance(drawings, list):
        return None
    matches = []
    for drawing_order, drawing in enumerate(drawings):
        items = drawing.get('items') or []
        candidate_signature = _snapshot_cubic_signature(items)
        if candidate_signature != signature:
            continue
        matches.append([
            {
                'drawing_order': int(drawing_order),
                'item_index': int(item_index),
                'op': 'c',
                'role': 'circle_curve',
            }
            for item_index in range(len(items))
        ])
    return matches[0] if len(matches) == 1 else None


def _pdfplumber_cubic_signature(path):
    if not isinstance(path, (list, tuple)) or not path:
        return None
    current = None
    start = None
    rows = []
    for command in path:
        if not isinstance(command, (list, tuple)) or not command:
            return None
        op = str(command[0])
        if op == 'm' and len(command) == 2 and current is None:
            current = _point_signature(command[1])
            start = current
            if current is None:
                return None
            continue
        if op != 'c' or len(command) != 4 or current is None:
            return None
        control_1 = _point_signature(command[1])
        control_2 = _point_signature(command[2])
        end = _point_signature(command[3])
        if None in {control_1, control_2, end}:
            return None
        rows.append((current, control_1, control_2, end))
        current = end
    if len(rows) != 4 or current != start:
        return None
    return tuple(rows)


def _snapshot_cubic_signature(items):
    if not isinstance(items, (list, tuple)) or len(items) != 4:
        return None
    rows = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            return None
        if str(item[0]) != 'c':
            return None
        points = tuple(_point_signature(value) for value in item[1:5])
        if any(point is None for point in points):
            return None
        rows.append(points)
    if any(rows[index][3] != rows[(index + 1) % 4][0] for index in range(4)):
        return None
    return tuple(rows)


def _point_signature(value):
    try:
        if hasattr(value, 'x') and hasattr(value, 'y'):
            x, y = float(value.x), float(value.y)
        else:
            x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    # pdfplumber and PyMuPDF expose the same path through different float
    # widths; 0.001 pt is the shared serialized coordinate precision.
    return round(x, 3), round(y, 3)


def _line_endpoint_signature(value):
    try:
        first = _point_signature((value[0], value[1]))
        second = _point_signature((value[2], value[3]))
    except (TypeError, IndexError):
        return None
    if first is None or second is None:
        return None
    return tuple(sorted((first, second)))


# ---------------------------------------------------------------------------
# 自适应线宽分层
# ---------------------------------------------------------------------------

def extract_lines_by_width(page, *, drawing_snapshot=None):
    """
    自适应线宽分层，使用 PyMuPDF (fitz) page.get_drawings() 获取 CTM 修正后的真实线宽。

    返回:
    {
      'l0_lines': [...],   # lineWidth ~= 0，文字笔画（排除）
      'l1_lines': [...],   # 细线，尺寸线/延伸线（保留，最重要）
      'l2_lines': [...],   # 中线，剖面线/中心线
      'l3_lines': [...],   # 粗线，轮廓线
      'all_lines': [...],
      'thresholds': {'thin_thick': float},  # 分层阈值
    }
    """
    fitz_page = None if drawing_snapshot is not None else _get_fitz_page(page)

    all_lines = []
    if drawing_snapshot is not None or fitz_page is not None:
        drawings = (
            drawing_snapshot
            if drawing_snapshot is not None
            else fitz_page.get_drawings()
        )
        for d in drawings:
            w = d.get('width', 0) or 0
            for item in d.get('items', []):
                if item[0] == 'l':  # 线段
                    p1, p2 = item[1], item[2]
                    all_lines.append({
                        'x0': p1.x, 'y0': p1.y,
                        'x1': p2.x, 'y1': p2.y,
                        'lineWidth': w,
                    })
                elif item[0] == 're':  # 矩形（4条边）
                    pass
    else:
        for ln in page.lines:
            all_lines.append({
                'x0': ln.get('x0', 0),
                'y0': ln.get('top', 0),
                'x1': ln.get('x1', 0),
                'y1': ln.get('bottom', 0),
                'lineWidth': ln.get('linewidth', ln.get('width', 0)),
            })

    if not all_lines:
        return {'l0_lines': [], 'l1_lines': [], 'l2_lines': [], 'l3_lines': [],
                'all_lines': [], 'thresholds': {'thin_thick': 0.5}}

    # 自适应分层
    thresholds = _adaptive_width_thresholds(all_lines)
    t_zero = thresholds['zero_cutoff']
    t_thin_thick = thresholds['thin_thick']

    l0, l1, l2, l3 = [], [], [], []
    for ln in all_lines:
        w = ln['lineWidth']
        if w <= t_zero:
            l0.append(ln)
        elif w <= t_thin_thick:
            l1.append(ln)
        elif w <= t_thin_thick * 3.0:
            l2.append(ln)
        else:
            l3.append(ln)

    # -- L1 hatching purge: 按角度分桶清洗剖面填充线 --
    before_count = len(l1)

    if len(l1) > 1000:
        # -- numpy 向量化路径 --
        arr = np.array([[ln['x0'], ln['y0'], ln['x1'], ln['y1']] for ln in l1])
        dx = arr[:, 2] - arr[:, 0]
        dy = arr[:, 3] - arr[:, 1]
        lengths = np.hypot(dx, dy)
        angles_rad = np.arctan2(dy, dx)
        angles_deg = np.degrees(angles_rad)
        angles_deg[angles_deg < 0] += 180
        angles_int = np.round(angles_deg).astype(int)

        # 分类掩码
        long_mask = lengths >= 80
        ortho_mask = (angles_int <= 2) | (angles_int >= 178) | ((angles_int >= 88) & (angles_int <= 92))
        ortho_long_mask = ortho_mask & ~long_mask & (lengths >= 20)
        ortho_short_mask = ortho_mask & ~long_mask & (lengths < 20)
        remainder_mask = ~long_mask & ~ortho_mask

        # kept = 长线 + 正交长线
        kept_mask = long_mask | ortho_long_mask

        # remainder 按角度分桶判定
        rem_indices = np.where(remainder_mask)[0]
        rem_angles = angles_int[rem_indices]
        rem_lengths = lengths[rem_indices]

        purged_buckets = []
        if len(rem_indices) > 0:
            unique_angles, inverse = np.unique(rem_angles, return_inverse=True)
            purge_flags = np.zeros(len(rem_indices), dtype=bool)
            for i, angle in enumerate(unique_angles):
                bucket_mask = inverse == i
                bucket_count = bucket_mask.sum()
                if bucket_count > 50:
                    avg_len = rem_lengths[bucket_mask].mean()
                    if avg_len < 40:
                        purge_flags[bucket_mask] = True
                        purged_buckets.append(int(angle))
            rem_kept = rem_indices[~purge_flags]
            kept_mask[rem_kept] = True

        # 正交短线池判定
        ortho_short_indices = np.where(ortho_short_mask)[0]
        if len(ortho_short_indices) > 200:
            print(f"[L1 hatching purge] ortho short lines discarded: {len(ortho_short_indices)} (too many, likely geometry)")
        else:
            kept_mask[ortho_short_indices] = True
            print(f"[L1 hatching purge] ortho short lines kept: {len(ortho_short_indices)} (small pool, likely real)")

        kept_indices = np.where(kept_mask)[0]
        kept = [l1[i] for i in kept_indices]

    else:
        # -- 原始 Python 循环路径 --
        kept = []
        remainder = []
        ortho_short = []

        for ln in l1:
            dx = ln['x1'] - ln['x0']
            dy = ln['y1'] - ln['y0']
            length = math.hypot(dx, dy)

            if length >= 80:
                kept.append(ln)
                continue

            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 180
            angle = round(angle)

            if angle <= 2 or angle >= 178 or (88 <= angle <= 92):
                if length >= 20:
                    kept.append(ln)
                else:
                    ortho_short.append(ln)
                continue

            remainder.append((ln, angle, length))

        buckets = defaultdict(list)
        for ln, angle, length in remainder:
            buckets[angle].append((ln, length))

        purged_buckets = []
        for angle, items in sorted(buckets.items()):
            if len(items) > 50:
                avg_len = sum(l for _, l in items) / len(items)
                if avg_len < 40:
                    purged_buckets.append(angle)
                    continue
            for ln, _ in items:
                kept.append(ln)

        if len(ortho_short) > 200:
            print(f"[L1 hatching purge] ortho short lines discarded: {len(ortho_short)} (too many, likely geometry)")
        else:
            kept.extend(ortho_short)
            print(f"[L1 hatching purge] ortho short lines kept: {len(ortho_short)} (small pool, likely real)")

    l1 = kept
    print(f"[L1 hatching purge] before: {before_count}, after: {len(l1)}, removed: {before_count - len(l1)}")
    print(f"[L1 hatching purge] purged angle buckets: {purged_buckets}")

    return {
        'l0_lines': l0,
        'l1_lines': l1,
        'l2_lines': l2,
        'l3_lines': l3,
        'all_lines': all_lines,
        'thresholds': thresholds,
    }


# ---------------------------------------------------------------------------
# 连通分量预过滤
# ---------------------------------------------------------------------------

def filter_annotation_segments_by_connectivity(
    fitz_page,
    annotation_lw,
    *,
    drawing_snapshot=None,
):
    """
    用 Union-Find 对 annotation 线段建连通分量，按跨度自适应阈值
    过滤大连通分量（剖面线），保留小连通分量（文字笔画）。

    参数:
        fitz_page: PyMuPDF page 对象
        annotation_lw: annotation 层的主 linewidth

    返回:
        (retained_segments, diag_info)
        retained_segments: list of dict {'x0','y0','x1','y1'}
        diag_info: dict with diagnostic counters
    """
    tol = annotation_lw * 0.15
    lw_lo = annotation_lw - tol
    lw_hi = annotation_lw + tol

    # 1. 提取该 linewidth 下的所有 line items
    segments = []
    drawings = (
        drawing_snapshot
        if drawing_snapshot is not None
        else fitz_page.get_drawings()
    )
    for d in drawings:
        w = d.get('width') or 0
        if not (lw_lo <= w <= lw_hi):
            continue
        for item_type, *pts in d['items']:
            if item_type == 'l':  # line
                p1, p2 = pts
                segments.append((p1.x, p1.y, p2.x, p2.y))

    total_segs = len(segments)
    if total_segs == 0:
        return [], {'total_segs': 0, 'components': 0, 'threshold': 0,
                    'retained': 0, 'discarded': 0}

    # 2. Union-Find (dict-based with path compression)
    ep_tol = max(annotation_lw * 1.5, 0.5)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Hash endpoints into buckets
    bucket_to_segs = defaultdict(list)
    for i, (x0, y0, x1, y1) in enumerate(segments):
        parent[i] = i
        for x, y in ((x0, y0), (x1, y1)):
            key = (round(x / ep_tol), round(y / ep_tol))
            bucket_to_segs[key].append(i)

    # Union segments sharing a bucket
    for key, seg_ids in bucket_to_segs.items():
        for j in range(1, len(seg_ids)):
            union(seg_ids[0], seg_ids[j])

    # Also check adjacent buckets for robustness
    checked = set()
    for key in list(bucket_to_segs.keys()):
        for dk in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nk = (key[0] + dk[0], key[1] + dk[1])
            pair = (min(key, nk), max(key, nk))
            if pair in checked or nk not in bucket_to_segs:
                continue
            checked.add(pair)
            for si in bucket_to_segs[key]:
                for sj in bucket_to_segs[nk]:
                    if find(si) == find(sj):
                        continue
                    sx0, sy0, sx1, sy1 = segments[si]
                    tx0, ty0, tx1, ty1 = segments[sj]
                    for (ax, ay) in ((sx0, sy0), (sx1, sy1)):
                        for (bx, by) in ((tx0, ty0), (tx1, ty1)):
                            if abs(ax - bx) <= ep_tol and abs(ay - by) <= ep_tol:
                                union(si, sj)

    # 3. Compute component spans
    components = defaultdict(list)
    for i in range(total_segs):
        components[find(i)].append(i)

    spans = []
    for root, members in components.items():
        xs, ys = [], []
        for i in members:
            x0, y0, x1, y1 = segments[i]
            xs.extend((x0, x1))
            ys.extend((y0, y1))
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        spans.append((root, span))

    num_components = len(spans)
    span_values = [s for _, s in spans]

    # 4. Adaptive threshold via histogram
    if not span_values:
        threshold = 0
    else:
        bin_width = 2.0
        max_span = max(span_values)
        n_bins = max(1, int(math.ceil(max_span / bin_width)))
        hist = [0] * n_bins
        for sv in span_values:
            b = min(int(sv / bin_width), n_bins - 1)
            hist[b] += 1

        threshold = None
        scan_limit = min(n_bins, 50)
        peak_bin = 0
        peak_count = 0
        for b in range(scan_limit):
            if hist[b] > peak_count:
                peak_count = hist[b]
                peak_bin = b
        if peak_count >= 3:
            valley_threshold = max(peak_count * 0.10, 1)
            for b in range(peak_bin + 1, scan_limit):
                if hist[b] <= valley_threshold:
                    if b + 1 >= n_bins or hist[b + 1] <= valley_threshold:
                        threshold = (b + 1) * bin_width
                        break

        if threshold is None:
            sorted_spans = sorted(span_values)
            idx_95 = int(len(sorted_spans) * 0.95)
            threshold = sorted_spans[min(idx_95, len(sorted_spans) - 1)]

        threshold = min(threshold, 100.0)

    # 5. Filter by threshold
    retained_roots = set()
    for root, span in spans:
        if span <= threshold:
            retained_roots.add(root)

    retained = []
    discarded_count = 0
    for i in range(total_segs):
        if find(i) in retained_roots:
            x0, y0, x1, y1 = segments[i]
            retained.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
        else:
            discarded_count += 1

    diag_info = {
        'total_segs': total_segs,
        'components': num_components,
        'threshold': round(threshold, 1),
        'retained': len(retained),
        'discarded': discarded_count,
    }

    return retained, diag_info
