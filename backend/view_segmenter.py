"""Segment drawing page regions with geometric image-space boundaries.

Views supply spatial grouping and clipping; they do not identify part features.
"""

import math
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

# ── Config（与 minesweeper v5 一致）──────────────────────────────────────
CELL = 1
WALL_MIN, WALL_MAX = 0.30, 0.80
BUCKET_THRESHOLDS = (6, 4, 1)
MARKER_MIN_CELLS = 200
MARKER_CLOSE_RADIUS = 20
FRAME_MARGIN_CELLS = 60
MIN_DIM_PT = 50
DEDUP_IOU = 0.5
CONTAIN_THRESH = 0.80
DENSE_MIN_CELLS = 320
ABSORB_GAP_PT = 30
# 保守的相邻视图合并门，避免较小独立视图被吸入。
ABSORB_SIZE_RATIO = 0.05
PARTIAL_OVERLAP_THRESH = 0.30
THIN_STRIP_ASPECT = 5.0
THIN_STRIP_DENSE_DENSITY = 0.08
THIN_STRIP_NARROW_ASPECT = 3.0
THIN_STRIP_NARROW_MAX_MIN_DIM = 100

# 清 SPUR 伪视图过滤参数（GT 对照校准）
# ── 规则 FIL1 低密度小方块:area<1%页面 AND dense_ratio<5% AND aspect<1.8 → 丢
#    抓典型伪视图；aspect 门槛保护真正的瘦视图不被误伤。
SPUR_TINY_AREA_RATIO = 0.01
SPUR_TINY_DENSE_RATIO_MAX = 0.05
SPUR_TINY_ASPECT_MAX = 1.8
# ── 规则 FIL2 低密度长条:aspect≥3 AND density<6% AND 长边≥28%页面对应维度 → 丢
#    过滤稀疏的说明区长条候选。
SPUR_LONGSTRIP_ASPECT = 3.0
SPUR_LONGSTRIP_DENSITY = 0.06
SPUR_LONGSTRIP_LONG_SIDE_RATIO = 0.28


def _safe_lw(obj):
    lw = obj.get('linewidth') or obj.get('width')
    if lw is None:
        return None
    try:
        return float(lw)
    except (TypeError, ValueError):
        return None


def _extract_segments(page):
    segs = []
    for line in page.lines:
        lw = _safe_lw(line)
        if lw and WALL_MIN <= lw <= WALL_MAX:
            segs.append(((line['x0'], line['top']), (line['x1'], line['bottom'])))
    for curve in page.curves:
        lw = _safe_lw(curve)
        if lw and WALL_MIN <= lw <= WALL_MAX:
            pts = curve.get('pts') or curve.get('points') or []
            for i in range(len(pts) - 1):
                segs.append((pts[i], pts[i + 1]))
    for rect in page.rects:
        lw = _safe_lw(rect)
        if lw and WALL_MIN <= lw <= WALL_MAX:
            x0, y0, x1, y1 = rect['x0'], rect['top'], rect['x1'], rect['bottom']
            segs.extend([
                ((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0)),
            ])
    return segs


def _discretize(segs, rows, cols):
    has = np.zeros((rows, cols), dtype=bool)
    if not segs:
        return has
    step = CELL * 0.8
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
    ci = x.astype(np.intp)
    ri = y.astype(np.intp)
    valid = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
    has[ri[valid], ci[valid]] = True
    return has


def _score_grid(has):
    kernel = np.ones((3, 3), dtype=np.int32)
    kernel[1, 1] = 0
    raw = ndimage.convolve(has.astype(np.int32), kernel,
                           mode='constant', cval=0).astype(np.int8)
    t3, t2, t1 = BUCKET_THRESHOLDS
    bucketed = np.where(raw >= t3, 3,
                np.where(raw >= t2, 2,
                np.where(raw >= t1, 1, 0))).astype(np.int8)
    return bucketed


def _detect_frame(page):
    """
    返回 (fx0, fy0, fx1, fy1) 图框矩形。

    加合理性检查。旧逻辑 `hl[0]/hl[-1]` 盲取最顶/最底水平长线,
    当长水平线都聚在页边时，frame 可能收缩为局部小区域，
    后续 frame 剥离会误遮蔽页面内容。
    修法:frame 的宽高必须 ≥ 页面 60%,否则回退到整页 (0,0,pw,ph)。
    """
    pw, ph = page.width, page.height
    hl, vl = [], []
    for line in page.lines:
        x0, y0, x1, y1 = line['x0'], line['top'], line['x1'], line['bottom']
        if abs(y1 - y0) < 2 and abs(x1 - x0) > pw * 0.5:
            hl.append(min(y0, y1))
        if abs(x1 - x0) < 2 and abs(y1 - y0) > ph * 0.5:
            vl.append(min(x0, x1))
    hl.sort()
    vl.sort()
    fx0 = vl[0] if vl else 0
    fy0 = hl[0] if hl else 0
    fx1 = vl[-1] if vl else pw
    fy1 = hl[-1] if hl else ph
    # 合理性检查:frame 太小说明线段分布 degenerate(如全挤在一边),回退整页
    if (fx1 - fx0) < pw * 0.6 or (fy1 - fy0) < ph * 0.6:
        return (0.0, 0.0, float(pw), float(ph))
    return (fx0, fy0, fx1, fy1)


def _detect_title(page, frame):
    """
    探测右下角标题栏(回到原版判据)。对部分图返回 None;
    对 None 的情况,由下游 _drop_bottom_right 通用规则兜底。
    """
    fx0, fy0, fx1, fy1 = frame
    fw, fh = fx1 - fx0, fy1 - fy0
    cands = []
    for line in page.lines:
        x0, y0, x1, y1 = line['x0'], line['top'], line['x1'], line['bottom']
        if abs(y1 - y0) < 2 and abs(x1 - x0) > fw * 0.15:
            my = (y0 + y1) / 2
            if my > fy0 + fh * 0.70:
                cands.append(my)
    if not cands:
        return None
    cands.sort()
    title_top = cands[0]
    vcands = []
    for line in page.lines:
        x0, y0, x1, y1 = line['x0'], line['top'], line['x1'], line['bottom']
        if abs(x1 - x0) < 2 and abs(y1 - y0) > fh * 0.05:
            mx = (x0 + x1) / 2
            if mx > fx0 + fw * 0.4 and min(y0, y1) >= title_top - 10:
                vcands.append(mx)
    title_left = min(vcands) if vcands else fx0 + fw * 0.55
    return (title_left, title_top, fx1, fy1)


def _strip_frame(bucketed, frame, title):
    rows, cols = bucketed.shape
    fx0, fy0, fx1, fy1 = frame
    margin_pt = FRAME_MARGIN_CELLS * CELL
    cy = (np.arange(rows, dtype=np.float32) + 0.5) * CELL
    cx = (np.arange(cols, dtype=np.float32) + 0.5) * CELL
    row_out = (cy < fy0) | (cy > fy1)
    row_margin = (cy - fy0 < margin_pt) | (fy1 - cy < margin_pt)
    col_out = (cx < fx0) | (cx > fx1)
    col_margin = (cx - fx0 < margin_pt) | (fx1 - cx < margin_pt)
    bucketed[row_out, :] = -1
    bucketed[:, col_out] = -1
    bucketed[row_margin, :] = -1
    bucketed[:, col_margin] = -1
    if title:
        tx0, ty0, tx1, ty1 = title
        row_title = (cy >= ty0) & (cy <= ty1)
        col_title = (cx >= tx0) & (cx <= tx1)
        title_mask = row_title[:, None] & col_title[None, :]
        bucketed[title_mask] = -1
    return bucketed


def _run_watershed(bucketed):
    marker_mask = bucketed >= 3
    close_r = MARKER_CLOSE_RADIUS
    yr, xr = np.ogrid[-close_r:close_r + 1, -close_r:close_r + 1]
    struct = (yr * yr + xr * xr <= close_r * close_r)
    marker_closed = ndimage.binary_closing(marker_mask, structure=struct)
    markers, _ = ndimage.label(marker_closed, structure=np.ones((3, 3)))
    sizes = np.bincount(markers.ravel())
    small = np.zeros(len(sizes), dtype=bool)
    small[1:] = sizes[1:] < MARKER_MIN_CELLS
    if small.any():
        markers[small[markers]] = 0
    markers, _ = ndimage.label(markers > 0, structure=np.ones((3, 3)))
    elevation = np.where(bucketed >= 0, 3 - bucketed, 4).astype(np.float32)
    ws_mask = bucketed >= 1
    return watershed(elevation, markers, mask=ws_mask)


def _detect_circle_shape(view, bucketed):
    """
    判断 view 是否为圆形视图(工程图里的局部放大圆 / 圆形端盖投影等)。

    算法: 在 view bbox 内切圆,统计 dense(bucket>=2)cells 落在圆内 vs 圆外的比例。
    矩形视图: 四角也有 part 线段 → 圆内比例 ~80%(内切圆本就占矩形 π/4≈78.5%)
    圆形视图: 四角为空白 → 圆内比例 ≥90%
    判据: aspect<1.3 AND in_circle_ratio ≥ 0.90 AND cells ≥ 200
    """
    x0 = int(view['x0'] / CELL); x1 = int(view['x1'] / CELL)
    y0 = int(view['y0'] / CELL); y1 = int(view['y1'] / CELL)
    w = x1 - x0; h = y1 - y0
    if w <= 0 or h <= 0: return None
    ar = max(w, h) / max(1, min(w, h))
    if ar > 1.3: return None
    sub = bucketed[y0:y1, x0:x1]
    dense_mask = sub >= 2
    n_dense = int(dense_mask.sum())
    if n_dense < 200: return None
    # 内切圆
    cx = w / 2; cy = h / 2
    r = min(w, h) / 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    in_circle = dist2 <= r*r
    n_in = int((dense_mask & in_circle).sum())
    ratio = n_in / n_dense if n_dense > 0 else 0
    # 算法结合(两个独立信号):
    #   1) in_circle_ratio: bbox 内切圆内 dense 占总 dense 比率
    #   2) corner_ratio: 四角 (距 bbox 角 r*0.3) 的 dense 密度
    # 真圆形视图:in_circle ≥ 0.93 AND 四角 dense 密度 < 2%
    # 矩形视图:in_circle 80~85% (因内切圆占矩形 78.5%) AND 四角有 part
    if ratio < 0.93: return None
    # 四角 dense 密度必须都 <2%
    for (cxc, cyc) in [(0, 0), (w, 0), (0, h), (w, h)]:
        corner_mask = (xx - cxc)**2 + (yy - cyc)**2 <= (r * 0.3)**2
        n_corner_total = int(corner_mask.sum())
        if n_corner_total == 0: continue
        n_corner_dense = int((dense_mask & corner_mask).sum())
        if n_corner_dense / n_corner_total > 0.02:
            return None
    return {
        'shape': 'circle',
        'cx': (x0 + cx) * CELL,
        'cy': (y0 + cy) * CELL,
        'r':  r * CELL,
    }


def _estimate_thin_thick(page):
    """
    估计细/中线宽分界阈值: 取 page 所有线段 lineWidth 的按 total-length 加权直方图,
    找双峰间谷值。简化版:直接取所有 linewidth 的中位数作为近似。
    """
    widths = []
    for line in page.lines:
        lw = line.get('linewidth') or line.get('width') or 0
        try:
            lw = float(lw)
            if lw > 0.01: widths.append(lw)
        except: pass
    for curve in page.curves:
        lw = curve.get('linewidth') or curve.get('width') or 0
        try:
            lw = float(lw)
            if lw > 0.01: widths.append(lw)
        except: pass
    if not widths:
        return 0.4
    # 返回中位数本身作为"part 轮廓至少这么粗"的下界
    # 短细线段(hatching/文字笔画)linewidth 通常 < 中位数,part 轮廓 ≥ 中位数
    widths.sort()
    return widths[len(widths)//2]


def _extract_views(ws, bucketed):
    n_regions = int(ws.max())
    views = []
    for rid in range(1, n_regions + 1):
        region = ws == rid
        dense = region & (bucketed >= 2)
        n_dense = int(dense.sum())
        if n_dense < DENSE_MIN_CELLS:
            continue
        ys, xs = np.where(dense)
        views.append({
            'x0': float(xs.min()) * CELL,
            'y0': float(ys.min()) * CELL,
            'x1': float(xs.max() + 1) * CELL,
            'y1': float(ys.max() + 1) * CELL,
            'dense_cells': n_dense,
        })
    return views


def _tighten_to_long_wide_lines(views, page, thin_thick=None, min_length=15):
    """
    算法结合: watershed 识别 part 所在区域 + 区内 L2+ 粗线段端点外包络 shrink bbox。

    思路:
      - watershed dense bbox 是"所有 bucket≥2 cells 外接矩形",含尺寸延伸线端点外扩
      - part 真实轮廓由粗线段(L2/L3, linewidth ≥ thin_thick)组成;尺寸线是细线(L1)
      - 在 view 区内取 lineWidth ≥ thin_thick 且 length ≥ min_length 的线段,它们端点的
        min/max = part 轮廓紧框,不会被细尺寸线端点撑开

    保守:若区内合格线段端点 <8 个,不 shrink(避免小详图被砍空)。
      新 bbox 只在原 dense bbox 内收缩,不外扩。
    """
    if thin_thick is None:
        return views
    cutoff = thin_thick  # >= median 为 part 轮廓候选
    # 收集所有 L2+ 长线段(线 + 曲线 + 矩形四边)
    segs = []
    for line in page.lines:
        lw = line.get('linewidth') or line.get('width') or 0
        try: lw = float(lw)
        except: continue
        if lw < cutoff: continue
        x0, y0, x1, y1 = line['x0'], line['top'], line['x1'], line['bottom']
        if math.hypot(x1-x0, y1-y0) >= min_length:
            segs.append((x0, y0, x1, y1))
    for curve in page.curves:
        lw = curve.get('linewidth') or curve.get('width') or 0
        try: lw = float(lw)
        except: continue
        if lw < cutoff: continue
        pts = curve.get('pts') or curve.get('points') or []
        for i in range(len(pts) - 1):
            try:
                p1, p2 = pts[i], pts[i+1]
                a, b = float(p1[0]), float(p1[1])
                c, d = float(p2[0]), float(p2[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.hypot(c-a, d-b) >= min_length:
                segs.append((a, b, c, d))
    for rect in page.rects:
        lw = rect.get('linewidth') or rect.get('width') or 0
        try: lw = float(lw)
        except: continue
        if lw < cutoff: continue
        x0, y0, x1, y1 = rect['x0'], rect['top'], rect['x1'], rect['bottom']
        if x1-x0 >= min_length:
            segs.append((x0, y0, x1, y0))
            segs.append((x0, y1, x1, y1))
        if y1-y0 >= min_length:
            segs.append((x0, y0, x0, y1))
            segs.append((x1, y0, x1, y1))

    if not segs:
        return views

    seg_arr = np.array(segs, dtype=np.float64)  # (N, 4)
    out = []
    for v in views:
        vx0, vy0, vx1, vy1 = v['x0'], v['y0'], v['x1'], v['y1']
        # 线段中点在 view 内的保留
        mx = (seg_arr[:, 0] + seg_arr[:, 2]) / 2
        my = (seg_arr[:, 1] + seg_arr[:, 3]) / 2
        inside = (mx >= vx0) & (mx <= vx1) & (my >= vy0) & (my <= vy1)
        n_in = int(inside.sum())
        if n_in < 4:  # 端点 <8 个, 不 shrink
            out.append(v); continue
        xs = np.concatenate([seg_arr[inside, 0], seg_arr[inside, 2]])
        ys = np.concatenate([seg_arr[inside, 1], seg_arr[inside, 3]])
        # 只在原 dense bbox 内收缩
        new_x0 = max(vx0, float(xs.min()))
        new_y0 = max(vy0, float(ys.min()))
        new_x1 = min(vx1, float(xs.max()))
        new_y1 = min(vy1, float(ys.max()))
        # 合理性: shrink 后 bbox 不能退化成线/点
        if new_x1 - new_x0 < MIN_DIM_PT or new_y1 - new_y0 < MIN_DIM_PT:
            out.append(v); continue
        out.append({
            'x0': new_x0, 'y0': new_y0, 'x1': new_x1, 'y1': new_y1,
            'dense_cells': v.get('dense_cells', 0),
        })
    return out


def _iou(a, b):
    ix0 = max(a['x0'], b['x0'])
    iy0 = max(a['y0'], b['y0'])
    ix1 = min(a['x1'], b['x1'])
    iy1 = min(a['y1'], b['y1'])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a['x1'] - a['x0']) * (a['y1'] - a['y0'])
    ub = (b['x1'] - b['x0']) * (b['y1'] - b['y0'])
    denom = ua + ub - inter
    return inter / denom if denom > 0 else 0.0


def _absorb_edge_fragments(views, gap_pt=ABSORB_GAP_PT, size_ratio=ABSORB_SIZE_RATIO):
    views = sorted(views,
                   key=lambda v: (v['x1'] - v['x0']) * (v['y1'] - v['y0']),
                   reverse=True)
    absorbed = set()
    for i in range(len(views)):
        if i in absorbed:
            continue
        vi = views[i]
        for j in range(i + 1, len(views)):
            if j in absorbed:
                continue
            vj = views[j]
            ai = (vi['x1'] - vi['x0']) * (vi['y1'] - vi['y0'])
            aj = (vj['x1'] - vj['x0']) * (vj['y1'] - vj['y0'])
            if aj > ai * size_ratio:
                continue
            gap_x = max(0, max(vi['x0'], vj['x0']) - min(vi['x1'], vj['x1']))
            gap_y = max(0, max(vi['y0'], vj['y0']) - min(vi['y1'], vj['y1']))
            gap = max(gap_x, gap_y)
            if gap > gap_pt:
                continue
            vi['x0'] = min(vi['x0'], vj['x0'])
            vi['y0'] = min(vi['y0'], vj['y0'])
            vi['x1'] = max(vi['x1'], vj['x1'])
            vi['y1'] = max(vi['y1'], vj['y1'])
            vi['dense_cells'] = vi.get('dense_cells', 0) + vj.get('dense_cells', 0)
            absorbed.add(j)
    return [v for i, v in enumerate(views) if i not in absorbed]


def _merge_partial_overlap(views, thresh=PARTIAL_OVERLAP_THRESH):
    """
    合并「大框部分吞小框」的情况：两框有实质重叠但既不构成
    IoU 去重（<0.5）也不构成 CONTAIN_THRESH（<0.8），典型场景是
    两个有较大交集的相邻区域被算作独立视图。
    规则：重叠区占小框面积 >= thresh → 把小框吞进大框。
    """
    views = sorted(views,
                   key=lambda v: (v['x1'] - v['x0']) * (v['y1'] - v['y0']),
                   reverse=True)
    merged = set()
    for i in range(len(views)):
        if i in merged:
            continue
        vi = views[i]
        for j in range(i + 1, len(views)):
            if j in merged:
                continue
            vj = views[j]
            ix0 = max(vi['x0'], vj['x0'])
            iy0 = max(vi['y0'], vj['y0'])
            ix1 = min(vi['x1'], vj['x1'])
            iy1 = min(vi['y1'], vj['y1'])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            aj = (vj['x1'] - vj['x0']) * (vj['y1'] - vj['y0'])
            if aj > 0 and inter / aj >= thresh:
                vi['x0'] = min(vi['x0'], vj['x0'])
                vi['y0'] = min(vi['y0'], vj['y0'])
                vi['x1'] = max(vi['x1'], vj['x1'])
                vi['y1'] = max(vi['y1'], vj['y1'])
                vi['dense_cells'] = vi.get('dense_cells', 0) + vj.get('dense_cells', 0)
                merged.add(j)
    return [v for i, v in enumerate(views) if i not in merged]


def _drop_thin_strips(views, aspect=THIN_STRIP_ASPECT, density=THIN_STRIP_DENSE_DENSITY,
                       verbose=False):
    """
    丢弃两类伪视图：
    1. 纵横比 >= 5.0 且 dense 密度 < 8% — 明显的标注带/引线群
    2. 短边 < 100pt 且纵横比 >= 3.0 **且 dense 密度 < 15%** — 窄条带
       追加 dense_ratio<0.15 门槛。原先仅靠纵横比一刀切，会误删
       短边窄但内容充实的局部详图（detail view / section）。
    """
    kept = []
    dropped_strip = []     # rule 1 命中
    dropped_narrow = []    # rule 2 命中
    for v in views:
        w = v['x1'] - v['x0']
        h = v['y1'] - v['y0']
        if w <= 0 or h <= 0:
            continue
        ar = max(w, h) / max(1.0, min(w, h))
        area = w * h
        dense_ratio = v.get('dense_cells', 0) / area if area > 0 else 0
        if ar >= aspect and dense_ratio < density:
            dropped_strip.append((v, ar, dense_ratio))
            continue
        # 放宽：短边窄 + 高纵横比 + **低密度** 才删
        if (min(w, h) < THIN_STRIP_NARROW_MAX_MIN_DIM and
                ar >= THIN_STRIP_NARROW_ASPECT and
                dense_ratio < 0.15):
            dropped_narrow.append((v, ar, dense_ratio))
            continue
        kept.append(v)
    if verbose:
        for v, ar, dr in dropped_strip:
            print(f"[view_seg][drop_strip] bbox=({v['x0']:.0f},{v['y0']:.0f},"
                  f"{v['x1']:.0f},{v['y1']:.0f}) aspect={ar:.2f} density={dr:.2%}")
        for v, ar, dr in dropped_narrow:
            print(f"[view_seg][drop_narrow] bbox=({v['x0']:.0f},{v['y0']:.0f},"
                  f"{v['x1']:.0f},{v['y1']:.0f}) aspect={ar:.2f} density={dr:.2%}")
    return kept


def _drop_spurious(views, pw, ph, frame, verbose=False):
    """
    通用过滤,不做针对性特调:
      FIL2 长条: aspect≥3 + dense_ratio<6% + 长边≥28%页面对应维度 → 丢
             (跨大半页的说明文字区横条)
      FIL3 右下角块: view center 在图框右下 30%x20% 区域 AND area < 页面 8% → 丢
             (标题栏检测失效时的兜底规则，按位置和大小上限筛选，
              不依赖 title 几何)
    """
    fx0, fy0, fx1, fy1 = frame
    fw, fh = fx1 - fx0, fy1 - fy0
    br_x_min = fx0 + fw * 0.70   # 右 30%
    br_y_min = fy0 + fh * 0.80   # 下 20%
    pa = pw * ph
    kept = []
    dropped = []
    for v in views:
        w = v['x1'] - v['x0']
        h = v['y1'] - v['y0']
        if w <= 0 or h <= 0:
            continue
        area = w * h
        dense_cells = v.get('dense_cells', 0)
        dense_ratio = dense_cells / area if area > 0 else 0
        ar = max(w, h) / max(1.0, min(w, h))
        cx = (v['x0'] + v['x1']) / 2
        cy = (v['y0'] + v['y1']) / 2
        # FIL2 长条
        long_side = max(w, h)
        corresponding = pw if w >= h else ph
        if (ar >= SPUR_LONGSTRIP_ASPECT and
                dense_ratio < SPUR_LONGSTRIP_DENSITY and
                long_side >= corresponding * SPUR_LONGSTRIP_LONG_SIDE_RATIO):
            dropped.append((v, 'longstrip', ar, dense_ratio, dense_cells))
            continue
        # FIL3 右下角小块(标题栏碎片兜底)
        if cx >= br_x_min and cy >= br_y_min and area < pa * 0.08:
            dropped.append((v, 'bottomright', ar, dense_ratio, dense_cells))
            continue
        kept.append(v)
    if verbose:
        for v, rule, ar, dr, dc in dropped:
            print(f"[view_seg][drop_{rule}] bbox=({v['x0']:.0f},{v['y0']:.0f},"
                  f"{v['x1']:.0f},{v['y1']:.0f}) aspect={ar:.2f} density={dr:.2%} dense={dc}")
    return kept


def _merge_adjacent_similar(views, gap_pt=40, overlap_ratio=0.70, area_ratio=0.60):
    """
    合并相邻大小相近的 views(主视图被 watershed 竖/横切成两半的场景)。

    条件(水平邻接示例,垂直对称):
      - 水平 gap ≤ gap_pt (可以是负值,即稍重叠)
      - 垂直方向 overlap 比例 ≥ overlap_ratio (相对较小那个 view 的高度)
      - 较小 area / 较大 area ≥ area_ratio (尺寸量级接近,排除"小碎片贴大视图")

    """
    # 按 area desc 尝试两两合并,迭代直到不变
    changed = True
    while changed:
        changed = False
        views = sorted(views, key=lambda v: (v['x1']-v['x0'])*(v['y1']-v['y0']), reverse=True)
        i = 0
        while i < len(views):
            j = i + 1
            while j < len(views):
                a, b = views[i], views[j]
                aw, ah = a['x1']-a['x0'], a['y1']-a['y0']
                bw, bh = b['x1']-b['x0'], b['y1']-b['y0']
                aa, ab = aw*ah, bw*bh
                smaller, larger = (aa, ab) if aa <= ab else (ab, aa)
                if smaller / larger < area_ratio:
                    j += 1; continue
                # 水平相邻: x 方向 gap,y 方向 overlap
                gap_x = max(a['x0'], b['x0']) - min(a['x1'], b['x1'])
                y_ov  = max(0, min(a['y1'], b['y1']) - max(a['y0'], b['y0']))
                # 垂直相邻
                gap_y = max(a['y0'], b['y0']) - min(a['y1'], b['y1'])
                x_ov  = max(0, min(a['x1'], b['x1']) - max(a['x0'], b['x0']))
                min_h = min(ah, bh); min_w = min(aw, bw)
                horiz = (gap_x <= gap_pt) and (min_h > 0) and (y_ov / min_h >= overlap_ratio)
                vert  = (gap_y <= gap_pt) and (min_w > 0) and (x_ov / min_w >= overlap_ratio)
                if horiz or vert:
                    merged = {
                        'x0': min(a['x0'], b['x0']), 'y0': min(a['y0'], b['y0']),
                        'x1': max(a['x1'], b['x1']), 'y1': max(a['y1'], b['y1']),
                        'dense_cells': a.get('dense_cells', 0) + b.get('dense_cells', 0),
                    }
                    views[i] = merged
                    views.pop(j)
                    changed = True
                    continue
                j += 1
            i += 1
    return views


def _dedup_pass(views):
    views = sorted(views,
                   key=lambda v: (v['x1'] - v['x0']) * (v['y1'] - v['y0']),
                   reverse=True)
    kept = []
    for v in views:
        drop = False
        for k in kept:
            ix0 = max(v['x0'], k['x0'])
            iy0 = max(v['y0'], k['y0'])
            ix1 = min(v['x1'], k['x1'])
            iy1 = min(v['y1'], k['y1'])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                area_v = (v['x1'] - v['x0']) * (v['y1'] - v['y0'])
                if area_v > 0 and inter / area_v > CONTAIN_THRESH:
                    drop = True
                    break
            if _iou(v, k) > DEDUP_IOU:
                drop = True
                break
        if not drop:
            kept.append(v)
    return kept


def _postprocess(views, pw, ph, title, verbose=False, frame=None, page=None, thin_thick=None, bucketed=None):
    min_area = pw * ph * 0.002
    filtered = []
    removed_small = removed_narrow = removed_title = 0
    for v in views:
        w = v['x1'] - v['x0']
        h = v['y1'] - v['y0']
        if w * h < min_area:
            removed_small += 1
            continue
        if w < MIN_DIM_PT or h < MIN_DIM_PT:
            removed_narrow += 1
            continue
        if title:
            cx = (v['x0'] + v['x1']) / 2
            cy = (v['y0'] + v['y1']) / 2
            if title[0] <= cx <= title[2] and title[1] <= cy <= title[3]:
                removed_title += 1
                continue
        filtered.append(v)

    if verbose:
        print(f"[view_seg][pre-proc] in={len(views)} after_size={len(filtered)} "
              f"(drop_small={removed_small} drop_narrow={removed_narrow} drop_title={removed_title})")

    kept = _dedup_pass(filtered)
    if verbose: print(f"[view_seg][dedup_1] -> {len(kept)}")
    kept = _absorb_edge_fragments(kept)
    if verbose: print(f"[view_seg][absorb ] -> {len(kept)}")
    kept = _merge_partial_overlap(kept)
    if verbose: print(f"[view_seg][merge  ] -> {len(kept)}")
    kept = _merge_adjacent_similar(kept)
    if verbose: print(f"[view_seg][merge_adj] -> {len(kept)}")
    kept = _drop_thin_strips(kept, verbose=verbose)
    if verbose: print(f"[view_seg][strip  ] -> {len(kept)}")
    eff_frame = frame if frame is not None else (0.0, 0.0, float(pw), float(ph))
    kept = _drop_spurious(kept, pw, ph, eff_frame, verbose=verbose)
    if verbose: print(f"[view_seg][spur   ] -> {len(kept)}")
    kept = _dedup_pass(kept)
    if verbose: print(f"[view_seg][dedup_2] -> {len(kept)}")
    # 最后 shrink: 算法结合 — L2+ 粗线段端点外包络
    if page is not None and thin_thick is not None:
        kept = _tighten_to_long_wide_lines(kept, page, thin_thick=thin_thick)
        if verbose: print(f"[view_seg][tighten] -> {len(kept)}")
    # 圆形视图检测(用 bucket dense cells 在内切圆内的比例判)
    if bucketed is not None:
        for v in kept:
            shape_info = _detect_circle_shape(v, bucketed)
            if shape_info:
                v.update(shape_info)
        n_circle = sum(1 for v in kept if v.get('shape') == 'circle')
        if verbose: print(f"[view_seg][circle ] {n_circle} of {len(kept)} views are circles")
    final = kept
    final.sort(key=lambda v: (v['y0'], v['x0']))
    return final


def segment_views(page, verbose=False):
    """
    返回视图 bbox 列表（pdfplumber 坐标系，左上原点）。

    每个条目: {'x': float, 'y': float, 'w': float, 'h': float}
    调用失败或无视图时返回空列表。

    Args:
        page: pdfplumber.Page
        verbose: 打印每个过滤阶段的 in/out 计数，便于诊断过删
    """
    pw = float(page.width)
    ph = float(page.height)
    segs = _extract_segments(page)
    if not segs:
        print(f"[view_seg] no segments in [{WALL_MIN},{WALL_MAX}]pt range")
        return []
    rows = math.ceil(ph / CELL)
    cols = math.ceil(pw / CELL)
    has = _discretize(segs, rows, cols)
    bucketed = _score_grid(has)
    frame = _detect_frame(page)
    title = _detect_title(page, frame)
    _strip_frame(bucketed, frame, title)
    ws = _run_watershed(bucketed)
    raw = _extract_views(ws, bucketed)
    thin_thick = _estimate_thin_thick(page)
    final = _postprocess(raw, pw, ph, title, verbose=verbose, frame=frame,
                          page=page, thin_thick=thin_thick, bucketed=bucketed)
    print(f"[view_seg] {len(segs):,} segs → {len(raw)} raw → {len(final)} final views")
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
            item['r']  = round(v.get('r', 0), 2)
        out.append(item)
    return out
