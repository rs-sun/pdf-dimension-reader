"""view_seg_common — view_segmenter / view_seg_minesweeper 共享几何工具。

包含图框检测、标题栏检测、IoU、圆形视图判别、线宽提取。

注意：dense cells 计算依赖外部 `bucketed` 数组传入；本模块不持 grid 状态。
"""
import numpy as np


def _safe_lw(obj):
    lw = obj.get('linewidth') or obj.get('width')
    if lw is None:
        return None
    try:
        return float(lw)
    except (TypeError, ValueError):
        return None


def _iou(a, b):
    """IoU on bbox dict {x0,y0,x1,y1} or any with these keys."""
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


def _detect_frame(page):
    """返回 (fx0, fy0, fx1, fy1) 图框矩形。

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
    if (fx1 - fx0) < pw * 0.6 or (fy1 - fy0) < ph * 0.6:
        return (0.0, 0.0, float(pw), float(ph))
    return (fx0, fy0, fx1, fy1)


def _detect_title(page, frame):
    """探测右下角标题栏(回到原版判据)。对部分图返回 None;
    对 None 的情况,由下游 _drop_bottom_right 通用规则兜底。
    """
    fx0, fy0, fx1, fy1 = frame
    fw, fh = fx1 - fx0, fy1 - fy0
    cands = []
    for line in page.lines:
        x0, y0, x1, y1 = line['x0'], line['top'], line['x1'], line['bottom']
        if abs(y1 - y0) < 2 and abs(x1 - x0) > fw * 0.15:
            my = (y0 + y1) / 2
            # 跳过 my 离 frame 底边/顶边 < 5pt 的（重合的图框线，不是 title 内线）
            if my > fy0 + fh * 0.70 and (fy1 - my) > 5:
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


def _detect_circle_shape(view, bucketed, cell=1):
    """判断 view 是否为圆形视图(工程图里的局部放大圆 / 圆形端盖投影等)。

    算法: 在 view bbox 内切圆,统计 dense(bucket>=2)cells 落在圆内 vs 圆外的比例。
    矩形视图: 四角也有 part 线段 → 圆内比例 ~80%(内切圆本就占矩形 π/4≈78.5%)
    圆形视图: 四角为空白 → 圆内比例 ≥90%
    判据: aspect<1.3 AND in_circle_ratio ≥ 0.93 AND cells ≥ 200 AND 四角 dense<2%
    """
    x0 = int(view['x0'] / cell); x1 = int(view['x1'] / cell)
    y0 = int(view['y0'] / cell); y1 = int(view['y1'] / cell)
    w = x1 - x0; h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    ar = max(w, h) / max(1, min(w, h))
    if ar > 1.3:
        return None
    sub = bucketed[y0:y1, x0:x1]
    dense_mask = sub >= 2
    n_dense = int(dense_mask.sum())
    if n_dense < 200:
        return None
    cx = w / 2; cy = h / 2
    r = min(w, h) / 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    in_circle = dist2 <= r * r
    n_in = int((dense_mask & in_circle).sum())
    ratio = n_in / n_dense if n_dense > 0 else 0
    if ratio < 0.93:
        return None
    for (cxc, cyc) in [(0, 0), (w, 0), (0, h), (w, h)]:
        corner_mask = (xx - cxc) ** 2 + (yy - cyc) ** 2 <= (r * 0.3) ** 2
        n_corner_total = int(corner_mask.sum())
        if n_corner_total == 0:
            continue
        n_corner_dense = int((dense_mask & corner_mask).sum())
        if n_corner_dense / n_corner_total > 0.02:
            return None
    return {
        'shape': 'circle',
        'cx': (x0 + cx) * cell,
        'cy': (y0 + cy) * cell,
        'r': r * cell,
    }
