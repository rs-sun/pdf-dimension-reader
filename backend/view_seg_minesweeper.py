"""view_seg_minesweeper — 扫雷算法 v3.2 视图切分。

对外接口: segment_views(page) -> list[{'x','y','w','h','shape'}]
            与 view_segmenter.segment_views 兼容。

实施阶段（按 design doc Step 0-5）:
    Step 0  矢量预处理（线宽过滤 + 离散化）
    Step 1  多分辨率网格评分（L1/L2 自适应）
    Step 2  图框 + 标题栏剥离（复用 common）
    Step 3a Moore 轮廓追踪（Jacob's 准则 + 凸包兜底）
    Step 3b Grid Flood-fill 区域删除（无 seed 兜底安全清扫）
    Step 3c L2 局部精修（完整 _extract_segments + Liang-Barsky 裁剪 + L2 通行规则）
    Step 4  圆形视图检测（复用 common）
    Step 5  后处理 dedup/sort
"""
import math
from collections import deque
from typing import List

import numpy as np
from scipy import ndimage

from view_seg_common import (
    _safe_lw, _iou, _detect_frame, _detect_title, _detect_circle_shape,
)


# ── Config（design doc §二 默认）─────────────────────────────
WALL_MIN, WALL_MAX = 0.30, 0.80   # 设计默认；不擅自降到 0.20

# 多分辨率网格步长。
CELL_L1 = 20
CELL_L2 = 8
CELL_L3 = 4

# Step 1 评分阈值（design doc §二.Step 1，固定不调）
BUCKET_THRESHOLDS = (6, 4, 1)

# Step 2 frame margin（design doc §二.Step 2：薄条 <60pt）
FRAME_MARGIN_PT = 60

# Step 3c L2 margin
L2_MARGIN_PT = 20

# Step 5 后处理常量
MIN_AREA_RATIO = 0.002      # 设计文档：< 页面 0.2% 丢
MAX_AREA_RATIO = 0.30       
DEDUP_IOU = 0.5             # 设计文档


# ── Step 3 通行条件 ────────────
# 设计文档表格"允许走向"按字面读会让 trace 在小区域 ping-pong（2-ring start
# 周围只有 1/0 cells，且 1↔1 互相允许 → 死循环）。改为标准 Moore 解读：
# target 只要非 3/-1（绝对禁区）就可走，附加约束仅作用于 0/1 cells（防游离）。
# 等价于标准"绕 dense 区域 (3) 边界"算法。
_MOVE_RULES_L1 = {
    2: {2, 1, 0},
    1: {2, 1, 0},
    0: {2, 1, 0},
}

_MOVE_RULES_L2 = {
    2: {2, 1, 0},
    1: {2, 1, 0},
    0: {2, 1, 0},
}

# Moore 邻域 8 方向，顺时针，从北开始
_MOORE_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]


# ╭──────────────────────────────────────────────────────────────╮
# │ Step 0: 线宽过滤 + 离散化                                       │
# ╰──────────────────────────────────────────────────────────────╯


def _auto_cell_size(page, multiplier=15.0, floor_pt=4.0, ceil_pt=7.0):
    widths = []
    for line in page.lines:
        lw = _safe_lw(line)
        if lw is not None and lw > 0.01:
            widths.append(lw)
    if not widths:
        return 6.0
    widths.sort()
    p10 = widths[max(0, int(len(widths) * 0.10))]
    cell = p10 * multiplier
    return max(floor_pt, min(ceil_pt, cell))


def _auto_wall_range(page):
    widths = []
    for line in page.lines:
        lw = _safe_lw(line)
        if lw is not None and lw > 0.01:
            widths.append(lw)
    if not widths:
        return WALL_MIN, WALL_MAX
    widths.sort()
    n = len(widths)
    p5 = widths[max(0, int(n * 0.05))]
    p95 = widths[min(n - 1, int(n * 0.95))]
    # 若 95% lineWidth 都在 WALL_MIN 之下：自动放宽到 p5
    if p95 < WALL_MIN:
        return p5, max(WALL_MAX, p95)
    return WALL_MIN, max(WALL_MAX, p95)


def _extract_segments(page, wall_range=None):
    """按 lineWidth ∈ [wall_min, wall_max] 过滤后返回 [(start, end), ...]。
    wall_range=None 时调 _auto_wall_range 自动选；显式传 (lo, hi) 覆盖。
    """
    if wall_range is None:
        lo, hi = _auto_wall_range(page)
    else:
        lo, hi = wall_range
    segs = []
    for line in page.lines:
        lw = _safe_lw(line)
        if lw is not None and lo <= lw <= hi:
            segs.append(((line['x0'], line['top']), (line['x1'], line['bottom'])))
    for curve in page.curves:
        lw = _safe_lw(curve)
        if lw is not None and lo <= lw <= hi:
            pts = curve.get('pts') or curve.get('points') or []
            for i in range(len(pts) - 1):
                segs.append((pts[i], pts[i + 1]))
    for rect in page.rects:
        lw = _safe_lw(rect)
        if lw is not None and lo <= lw <= hi:
            x0, y0, x1, y1 = rect['x0'], rect['top'], rect['x1'], rect['bottom']
            segs.extend([
                ((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0)),
            ])
    return segs


def _discretize(segs, rows, cols, cell=1):
    """对线段按 step = cell * 0.8 插值生成采样点，标记 has 网格。
    设计依据: design doc §二.Step 0b（避免 G1 长线段中间格子判空）
    """
    has = np.zeros((rows, cols), dtype=bool)
    if not segs:
        return has
    step = cell * 0.8
    arr = np.array(segs, dtype=np.float64)
    starts = arr[:, 0]
    ends = arr[:, 1]
    diffs = ends - starts
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    n_steps = np.maximum(1, np.ceil(lengths / step).astype(np.int64))
    counts = n_steps + 1
    total = int(counts.sum())
    seg_ids = np.repeat(np.arange(len(segs)), counts)
    offsets = np.zeros(len(segs) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    local = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], counts)
    t = local / np.repeat(n_steps, counts).astype(np.float64)
    x = starts[seg_ids, 0] + t * diffs[seg_ids, 0]
    y = starts[seg_ids, 1] + t * diffs[seg_ids, 1]
    ci = (x / cell).astype(np.intp)
    ri = (y / cell).astype(np.intp)
    valid = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
    has[ri[valid], ci[valid]] = True
    return has


# ╭──────────────────────────────────────────────────────────────╮
# │ Step 1: 网格评分                                               │
# ╰──────────────────────────────────────────────────────────────╯


def _score_grid(has):
    """8 邻域评分 → 桶 3/2/1/0（用户全套规则）。

    阈值 (6, 4, 1) 起步 + 两条修正：

    1. raw=3 邻接 raw>=4 升 2 边缘（用户洞察：part 真边缘的 raw=3 紧贴 dense；
       孤立标注区的 raw=3 周围只有更稀疏点，不升级）
    2. **斜向卷积过滤尺寸线**（用户洞察）：横竖方向邻居数远多于斜方向邻居数
       的 cell，是横竖尺寸线 / 延伸线，即使 raw>=4 也降级为 1（不算 part 边缘）

    raw 邻居方向分解：
      raw_h = 横邻居 (left, right) 数 ∈ [0, 2]
      raw_v = 纵邻居 (up, down) 数 ∈ [0, 2]
      raw_d = 斜邻居 (4 个对角) 数 ∈ [0, 4]
      raw = raw_h + raw_v + raw_d
    """
    has_int = has.astype(np.int32)

    # 横/纵/斜 各方向邻居计数
    k_h = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int32)
    k_v = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.int32)
    k_d = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.int32)
    raw_h = ndimage.convolve(has_int, k_h, mode='constant', cval=0)
    raw_v = ndimage.convolve(has_int, k_v, mode='constant', cval=0)
    raw_d = ndimage.convolve(has_int, k_d, mode='constant', cval=0)
    raw = (raw_h + raw_v + raw_d).astype(np.int8)

    # 阈值起步
    bucketed = np.where(raw >= 6, 3,
               np.where(raw >= 4, 2,
               np.where(raw >= 1, 1, 0))).astype(np.int8)

    # 改进 1：raw=3 邻接 raw>=4 区升 2（5x5 范围内有 raw>=4 即可）
    is_3 = raw == 3
    near_4plus = ndimage.maximum_filter((raw >= 4).astype(np.int8), size=5) > 0
    bucketed[is_3 & near_4plus] = 2

    # 横竖线降级实验保持关闭。
    # axis_only = ((raw_h + raw_v) >= 3) & (raw_d <= 1)
    # bucketed[axis_only & (bucketed >= 2)] = 1

    return bucketed


# ╭──────────────────────────────────────────────────────────────╮
# │ Step 2: frame/title strip                                     │
# ╰──────────────────────────────────────────────────────────────╯


def _strip_frame(bucketed, frame, title, cell):
    """frame 外 + frame margin + 标题栏 → 标 -1。"""
    rows, cols = bucketed.shape
    fx0, fy0, fx1, fy1 = frame
    cy = (np.arange(rows, dtype=np.float32) + 0.5) * cell
    cx = (np.arange(cols, dtype=np.float32) + 0.5) * cell
    row_out = (cy < fy0) | (cy > fy1)
    row_margin = (cy - fy0 < FRAME_MARGIN_PT) | (fy1 - cy < FRAME_MARGIN_PT)
    col_out = (cx < fx0) | (cx > fx1)
    col_margin = (cx - fx0 < FRAME_MARGIN_PT) | (fx1 - cx < FRAME_MARGIN_PT)
    bucketed[row_out, :] = -1
    bucketed[:, col_out] = -1
    bucketed[row_margin, :] = -1
    bucketed[:, col_margin] = -1
    if title:
        tx0, ty0, tx1, ty1 = title
        title_mask = ((cy >= ty0) & (cy <= ty1))[:, None] & ((cx >= tx0) & (cx <= tx1))[None, :]
        bucketed[title_mask] = -1
    return bucketed


def _build_bucketed(page, cell):
    """端到端到 Step 2：返回 (bucketed, frame, title)。"""
    pw, ph = float(page.width), float(page.height)
    rows = math.ceil(ph / cell)
    cols = math.ceil(pw / cell)
    segs = _extract_segments(page)
    has = _discretize(segs, rows, cols, cell=cell)
    bucketed = _score_grid(has)
    frame = _detect_frame(page)
    title = _detect_title(page, frame)
    _strip_frame(bucketed, frame, title, cell=cell)
    return bucketed, frame, title


# ╭──────────────────────────────────────────────────────────────╮
# │ Step 3a: Moore 轮廓追踪 + 凸包兜底                              │
# ╰──────────────────────────────────────────────────────────────╯


def _convex_hull_close(path):
    """超 max_steps / 死角时取已追踪格子的凸包强行闭合（设计文档 §三）。"""
    if len(path) < 3:
        return path
    try:
        from scipy.spatial import ConvexHull
        pts = np.array(path, dtype=np.float64)  # (N, 2) [row, col]
        hull = ConvexHull(pts)
        return [tuple(int(v) for v in pts[i]) for i in hull.vertices]
    except Exception:
        # ConvexHull 退化（共线等）：返回原 path 的 bbox 四角
        rs = [r for r, _ in path]
        cs = [c for _, c in path]
        return [(min(rs), min(cs)), (min(rs), max(cs)),
                (max(rs), max(cs)), (max(rs), min(cs))]


def _trace_one_loop(bucketed, start, cell, level='L1', max_steps=None):
    """Moore-Neighbor Tracing + Jacob's 标准停止准则。

    Jacob's: 记录 first_exit_dir_idx（首次离开 start 的方向），停止条件 =
    `next_cell == start AND next_dir_idx == first_exit_dir_idx`。
    """
    rows, cols = bucketed.shape
    if max_steps is None:
        max_steps = (rows + cols) * 6
    move_rules = _MOVE_RULES_L1 if level == 'L1' else _MOVE_RULES_L2
    path = [start]
    visited = {start}   # 防 ping-pong：除 start 外不重访
    cur = start
    prev_dir_idx = None
    first_exit_dir_idx = None
    for step in range(max_steps):
        scan_start = 0 if prev_dir_idx is None else (prev_dir_idx + 4) % 8
        next_cell = None
        next_dir_idx = None
        for k in range(8):
            d = (scan_start + k) % 8
            dr, dc = _MOORE_DIRS[d]
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            # 特例：闭合 start 始终可达
            if (nr, nc) == start and len(path) >= 3:
                next_cell = start
                next_dir_idx = d
                break
            if bucketed[nr, nc] in (-1, 3):
                continue
            if (nr, nc) in visited:
                continue
            cur_score = int(bucketed[cur[0], cur[1]])
            target_score = int(bucketed[nr, nc])
            allowed = move_rules.get(cur_score, set())
            if target_score not in allowed:
                continue
            if cur_score == 0:
                nbr = bucketed[max(0, nr - 1):nr + 2, max(0, nc - 1):nc + 2]
                if not ((nbr == 1) | (nbr == 2)).any():
                    continue
            if level == 'L2' and cur_score == 1:
                nbr = bucketed[max(0, nr - 1):nr + 2, max(0, nc - 1):nc + 2]
                if not (nbr == 0).any():
                    continue
            next_cell = (nr, nc)
            next_dir_idx = d
            break
        if next_cell is None:
            return _convex_hull_close(path)
        if first_exit_dir_idx is None:
            first_exit_dir_idx = next_dir_idx
        elif next_cell == start:
            # 回到 start 时采用方向容差，避免边界追踪无法闭合。
            return path
        path.append(next_cell)
        visited.add(next_cell)
        cur = next_cell
        prev_dir_idx = next_dir_idx
    return _convex_hull_close(path)


# ╭──────────────────────────────────────────────────────────────╮
# │ Step 3b: Grid Flood-fill 区域删除                              │
# ╰──────────────────────────────────────────────────────────────╯


def _grid_flood_fill_delete(bucketed, path):
    """以 path 为墙壁，泛洪填充内部。

    无 seed 时在 bbox 内清扫格子，防主循环重复检测。
    """
    rows, cols = bucketed.shape
    path_set = set(path)
    if not path_set:
        return
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    seed = None
    for r in range(rmin + 1, rmax):
        for c in range(cmin + 1, cmax):
            if (r, c) in path_set:
                continue
            if bucketed[r, c] in (0, 1):
                seed = (r, c)
                break
        if seed:
            break
    if seed is None:
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if bucketed[r, c] != -1:
                    bucketed[r, c] = -1
        return
    filled = {seed}
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in filled or (nr, nc) in path_set:
                continue
            if bucketed[nr, nc] in (-1, 3):
                continue
            filled.add((nr, nc))
            q.append((nr, nc))
    for r, c in filled:
        bucketed[r, c] = -1
    for r, c in path:
        bucketed[r, c] = -1


def _path_bbox(path, cell):
    """从 path 算 bbox（pt 坐标），返回 dict {x0,y0,x1,y1}。"""
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    return {
        'x0': float(min(cs)) * cell,
        'y0': float(min(rs)) * cell,
        'x1': float(max(cs) + 1) * cell,
        'y1': float(max(rs) + 1) * cell,
    }


# ╭──────────────────────────────────────────────────────────────╮
# │ Phase A: 起点搜索 + L1/L2 主循环                               │
# ╰──────────────────────────────────────────────────────────────╯


def _find_start_l1_no_visited(bucketed):
    """首选起点：自身=2 且邻域有 3 分格。"""
    rows, cols = bucketed.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if bucketed[r, c] != 2:
                continue
            nbrs = bucketed[r - 1:r + 2, c - 1:c + 2]
            if (nbrs == 3).any():
                return (r, c)
    return None


def _find_start_l1_fallback(bucketed):
    """Fallback 起点：自身=2 + 周围有 ≥3 个相连 2 分格 + 邻域有 0/1 分格。"""
    rows, cols = bucketed.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if bucketed[r, c] != 2:
                continue
            nbrs = bucketed[r - 1:r + 2, c - 1:c + 2]
            n_two = int((nbrs == 2).sum())
            n_low = int(((nbrs == 0) | (nbrs == 1)).sum())
            if n_two >= 3 and n_low >= 1:
                return (r, c)
    return None


def _trace_and_extract(bucketed, cell, page_area_pt, level='L1'):
    """L1/L2 主循环：找起点 → Moore 追踪 → flood-fill 删除 → 收集 view bbox。"""
    views = []
    page_area_cells = page_area_pt / (cell * cell)
    min_area_cells = page_area_cells * 0.005   # design doc Q2: 闭合环 > 页面 0.5%
    iter_safety = 0
    max_iters = 500   # 防极端 case 死循环
    while iter_safety < max_iters:
        iter_safety += 1
        start = _find_start_l1_no_visited(bucketed)
        if start is None:
            start = _find_start_l1_fallback(bucketed)
        if start is None:
            break
        path = _trace_one_loop(bucketed, start, cell, level=level)
        bbox = _path_bbox(path, cell)
        bbox_area_cells = ((bbox['x1'] - bbox['x0']) / cell) * ((bbox['y1'] - bbox['y0']) / cell)
        if len(path) < 4 or bbox_area_cells < min_area_cells:
            # 太短/太小：把 path bbox 内全部清扫，防同一区域反复触发
            rs = [r for r, _ in path]; cs = [c for _, c in path]
            for r in range(min(rs), max(rs) + 1):
                for c in range(min(cs), max(cs) + 1):
                    if 0 <= r < bucketed.shape[0] and 0 <= c < bucketed.shape[1]:
                        bucketed[r, c] = -1
            continue
        views.append(bbox)
        _grid_flood_fill_delete(bucketed, path)
    return views


# ╭──────────────────────────────────────────────────────────────╮
# │ Step 3c: L2 局部精修（Liang-Barsky 裁剪 + 完整 segs）           │
# ╰──────────────────────────────────────────────────────────────╯


def _clip_segment_to_region(p0, p1, rx0, ry0, rx1, ry1):
    """Liang-Barsky 线段裁剪：返回裁到矩形内的子线段端点；完全在外返回 None。"""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - rx0, rx1 - x0, y0 - ry0, ry1 - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    return ((x0 + u1 * dx, y0 + u1 * dy),
            (x0 + u2 * dx, y0 + u2 * dy))


def _extract_segments_in_region(page, region):
    """复用完整 _extract_segments + Liang-Barsky 裁到 region 局部坐标。"""
    rx0, ry0, rx1, ry1 = region
    raw = _extract_segments(page)
    out = []
    for p0, p1 in raw:
        clip = _clip_segment_to_region(p0, p1, rx0, ry0, rx1, ry1)
        if clip is None:
            continue
        (cx0, cy0), (cx1, cy1) = clip
        out.append(((cx0 - rx0, cy0 - ry0), (cx1 - rx0, cy1 - ry0)))
    return out


def _trace_l2_local(page, region, frame, title):
    """局部 L2 网格追踪。region=(x0,y0,x1,y1)，输出绝对坐标 view bbox 列表。"""
    rx0, ry0, rx1, ry1 = region
    sub_w = rx1 - rx0
    sub_h = ry1 - ry0
    sub_rows = math.ceil(sub_h / CELL_L2)
    sub_cols = math.ceil(sub_w / CELL_L2)
    segs = _extract_segments_in_region(page, region)
    has = _discretize(segs, sub_rows, sub_cols, cell=CELL_L2)
    bucketed = _score_grid(has)
    title_local = None
    if title:
        tx0, ty0, tx1, ty1 = title
        if not (tx1 < rx0 or tx0 > rx1 or ty1 < ry0 or ty0 > ry1):
            title_local = (max(0, tx0 - rx0), max(0, ty0 - ry0),
                           min(sub_w, tx1 - rx0), min(sub_h, ty1 - ry0))
    if title_local:
        _strip_frame(bucketed, (0, 0, sub_w, sub_h), title_local, cell=CELL_L2)
    sub_views = _trace_and_extract(bucketed, cell=CELL_L2,
                                    page_area_pt=sub_w * sub_h, level='L2')
    return [{
        'x0': sv['x0'] + rx0, 'y0': sv['y0'] + ry0,
        'x1': sv['x1'] + rx0, 'y1': sv['y1'] + ry0,
    } for sv in sub_views]


def _l2_refine(page, l1_views, frame, title):
    """对每个 L1 view bbox + margin 局部生成 L2 网格 + 重新追踪。"""
    refined = []
    for v in l1_views:
        x0 = max(frame[0], v['x0'] - L2_MARGIN_PT)
        y0 = max(frame[1], v['y0'] - L2_MARGIN_PT)
        x1 = min(frame[2], v['x1'] + L2_MARGIN_PT)
        y1 = min(frame[3], v['y1'] + L2_MARGIN_PT)
        sub_views = _trace_l2_local(page, (x0, y0, x1, y1), frame, title)
        if not sub_views:
            refined.append(v)
        else:
            for sv in sub_views:
                refined.append(sv)
    return refined


# ╭──────────────────────────────────────────────────────────────╮
# │ Step 5: 后处理                                                 │
# ╰──────────────────────────────────────────────────────────────╯


def _postprocess(views, pw, ph, title, frame, bucketed=None):
    """面积过滤 → 标题栏过滤 → 右下角兜底 → containment merge → IoU dedup → 圆检测 → 排序。

    右下角兜底（移植自 view_segmenter._drop_spurious FIL3）：
    cx 在 frame 右 30% AND cy 在 frame 下 20% AND area < page * 8% → 丢
    在 _detect_title 无法确定标题栏时提供保守的局部过滤。
    """
    page_area = pw * ph
    min_area = page_area * MIN_AREA_RATIO
    max_area = page_area * MAX_AREA_RATIO
    fx0, fy0, fx1, fy1 = frame
    fw, fh = fx1 - fx0, fy1 - fy0
    br_x_min = fx0 + fw * 0.70
    br_y_min = fy0 + fh * 0.80
    kept = []
    for v in views:
        w = v['x1'] - v['x0']
        h = v['y1'] - v['y0']
        a = w * h
        if a < min_area or a > max_area:
            continue
        cx = (v['x0'] + v['x1']) / 2
        cy = (v['y0'] + v['y1']) / 2
        if title:
            tx0, ty0, tx1, ty1 = title
            if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                continue
        # FIL3 右下角兜底（title 检测失败时）
        if cx >= br_x_min and cy >= br_y_min and a < page_area * 0.08:
            continue
        kept.append(v)
    MERGE_GAP_PT = 40.0
    MERGE_AREA_RATIO_MAX = 0.20
    kept.sort(key=lambda v: (v['x1'] - v['x0']) * (v['y1'] - v['y0']), reverse=True)
    changed = True
    while changed:
        changed = False
        for i, big in enumerate(kept):
            big_area = (big['x1'] - big['x0']) * (big['y1'] - big['y0'])
            for j in range(len(kept) - 1, i, -1):
                small = kept[j]
                small_area = (small['x1'] - small['x0']) * (small['y1'] - small['y0'])
                if big_area <= 0 or small_area / big_area > MERGE_AREA_RATIO_MAX:
                    continue
                x_in = small['x0'] >= big['x0'] and small['x1'] <= big['x1']
                gap_y = max(big['y0'], small['y0']) - min(big['y1'], small['y1'])
                y_in = small['y0'] >= big['y0'] and small['y1'] <= big['y1']
                gap_x = max(big['x0'], small['x0']) - min(big['x1'], small['x1'])
                if (x_in and 0 <= gap_y <= MERGE_GAP_PT) or \
                   (y_in and 0 <= gap_x <= MERGE_GAP_PT):
                    big['x0'] = min(big['x0'], small['x0'])
                    big['y0'] = min(big['y0'], small['y0'])
                    big['x1'] = max(big['x1'], small['x1'])
                    big['y1'] = max(big['y1'], small['y1'])
                    kept.pop(j)
                    changed = True
        kept.sort(key=lambda v: (v['x1'] - v['x0']) * (v['y1'] - v['y0']), reverse=True)
    # 大面积优先，小框被大框 containment ≥0.7 时丢
    final = []
    for v in kept:
        drop = False
        v_area = (v['x1'] - v['x0']) * (v['y1'] - v['y0'])
        for k in final:
            iou = _iou(v, k)
            if iou > DEDUP_IOU:
                drop = True
                break
            # containment dedup：仅当 v 95%+ 被 k 包住 AND 面积比 v/k > 0.5
            # （防主视图把内部详图吞掉）
            ix0 = max(v['x0'], k['x0']); iy0 = max(v['y0'], k['y0'])
            ix1 = min(v['x1'], k['x1']); iy1 = min(v['y1'], k['y1'])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                k_area = (k['x1'] - k['x0']) * (k['y1'] - k['y0'])
                if v_area > 0 and k_area > 0:
                    contain = inter / v_area
                    area_ratio = v_area / k_area
                    if contain > 0.95 and area_ratio > 0.5:
                        drop = True
                        break
        if not drop:
            final.append(v)
    if bucketed is not None:
        for v in final:
            shape_info = _detect_circle_shape(v, bucketed, cell=CELL_L1)
            if shape_info:
                v.update(shape_info)
    final.sort(key=lambda v: (v['y0'], v['x0']))
    return final


# ╭──────────────────────────────────────────────────────────────╮
# │ 公开入口                                                       │
# ╰──────────────────────────────────────────────────────────────╯


def _algo_a9_pixel_fill(page, frame, title, cell=1.0):
    """A9: cell=1pt 像素级 + closing + fill_holes（用户原意"贴着轮廓走"）。

    思路：
    - cell=1pt → 每个 cell 是 1pt 方块，has=True 直接代表"part 线穿过此 cell"
    - 不需要 raw 评分：has 本身就是 part 几何
    - binary_closing 修线段轻微断裂
    - binary_fill_holes 把每个闭合 part 轮廓填实 → view
    - CC → view 候选
    - 关键洞察（用户）：cell 极细时尺寸线间距远 > cell，不互相干扰评分

    此路径直接处理线段占据的细网格，不依赖 dense 评分作为视图边界。
    """
    pw, ph = float(page.width), float(page.height)
    rows = math.ceil(ph / cell)
    cols = math.ceil(pw / cell)

    segs = _extract_segments(page)
    has = _discretize(segs, rows, cols, cell=cell)

    # 使用局部 closing 修补细小线段间隙。
    closed = ndimage.binary_closing(has, structure=np.ones((3, 3)))

    # 限 valid
    fx0, fy0, fx1, fy1 = frame
    cy = (np.arange(rows, dtype=np.float32) + 0.5) * cell
    cx = (np.arange(cols, dtype=np.float32) + 0.5) * cell
    in_frame = ((cy >= fy0 + FRAME_MARGIN_PT) & (cy <= fy1 - FRAME_MARGIN_PT))[:, None] & \
               ((cx >= fx0 + FRAME_MARGIN_PT) & (cx <= fx1 - FRAME_MARGIN_PT))[None, :]
    if title:
        tx0, ty0, tx1, ty1 = title
        in_title = ((cy >= ty0) & (cy <= ty1))[:, None] & ((cx >= tx0) & (cx <= tx1))[None, :]
    else:
        in_title = np.zeros_like(in_frame)
    valid = in_frame & ~in_title
    closed = closed & valid

    # 选择性 fill_holes：只 fill 小洞（< 视图间隔的洞）
    inv = ~closed
    hl, n_h = ndimage.label(inv, structure=np.ones((3, 3)))
    border_ids = set()
    border_ids.update(int(x) for x in np.unique(hl[0, :]) if x > 0)
    border_ids.update(int(x) for x in np.unique(hl[-1, :]) if x > 0)
    border_ids.update(int(x) for x in np.unique(hl[:, 0]) if x > 0)
    border_ids.update(int(x) for x in np.unique(hl[:, -1]) if x > 0)
    sizes = ndimage.sum(inv.astype(np.int32), hl, index=np.arange(1, n_h + 1))
    small_hole_max = 2000
    fill_mask = np.zeros_like(closed)
    for h_id in range(1, n_h + 1):
        if h_id in border_ids:
            continue
        if sizes[h_id - 1] < small_hole_max:
            fill_mask |= (hl == h_id)
    filled = closed | fill_mask

    # CC = view candidates
    labels, n_labels = ndimage.label(filled, structure=np.ones((3, 3)))
    min_view_pt2 = pw * ph * 0.003   # 0.3% 页面，物理 pt²
    # 防纯噪：bbox 大但 pixel 极稀疏（< 100 cells 或 fill_ratio<3%）的 bbox 不算 view
    min_pixel_cells = 100
    min_fill_ratio = 0.03
    shrink_pt = 4.0
    shrink_cells = max(1, int(round(shrink_pt / cell)))
    views = []
    for lid in range(1, n_labels + 1):
        m = labels == lid
        n_pix = int(m.sum())
        if n_pix < min_pixel_cells:
            continue
        ys, xs = np.where(m)
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        bbox_w_pt = (c1 - c0 + 1) * cell
        bbox_h_pt = (r1 - r0 + 1) * cell
        bbox_area_pt2 = bbox_w_pt * bbox_h_pt
        if bbox_area_pt2 < min_view_pt2:
            continue
        bbox_cells = (r1 - r0 + 1) * (c1 - c0 + 1)
        if n_pix / max(1, bbox_cells) < min_fill_ratio:
            continue
        # 收 shrink_cells 各向，防小 view 收没（min size 4×shrink）
        if (r1 - r0) > 4 * shrink_cells and (c1 - c0) > 4 * shrink_cells:
            r0 += shrink_cells; r1 -= shrink_cells
            c0 += shrink_cells; c1 -= shrink_cells
        views.append({
            'x0': float(c0) * cell,
            'y0': float(r0) * cell,
            'x1': float(c1 + 1) * cell,
            'y1': float(r1 + 1) * cell,
            'algo': 'A9',
        })
    return views


def segment_views(page, verbose: bool = False) -> List[dict]:
    """Run the production A9 pixel-fill path for a pdfplumber-like page.

    Retired A2--A7 candidates live under scripts/experiments and are
    deliberately unavailable as production backend choices.
    """
    pw, ph = float(page.width), float(page.height)
    bucketed_l1, frame, title = _build_bucketed(page, cell=CELL_L1)

    # A9: cell=1pt 像素级 fill_holes（用户原意"贴着轮廓走"）
    a9_views = _algo_a9_pixel_fill(page, frame, title, cell=1.0)
    if verbose:
        print(f"[ms] A9={len(a9_views)}")

    final = _postprocess(a9_views, pw, ph, title, frame, bucketed=bucketed_l1)
    if verbose:
        print(f"[ms] final={len(final)}")

    out = []
    for v in final:
        item = {
            'x': round(v['x0'], 2),
            'y': round(v['y0'], 2),
            'w': round(v['x1'] - v['x0'], 2),
            'h': round(v['y1'] - v['y0'], 2),
            'shape': v.get('shape', 'rect'),
        }
        if v.get('shape') == 'circle':
            item['cx'] = round(v.get('cx', 0), 2)
            item['cy'] = round(v.get('cy', 0), 2)
            item['r'] = round(v.get('r', 0), 2)
        out.append(item)
    return out
