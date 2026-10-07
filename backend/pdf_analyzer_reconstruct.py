"""pdf_analyzer_reconstruct.py — 标注/轮廓线提取 + 线段矩形重建

从 pdf_analyzer.py 拆出。"""

import bisect
import math
from collections import defaultdict

from pdf_analyzer_utils import _rect_iou_xyxy


class _PreparedVerticalCoverage:
    """Immutable x-sorted broad phase for repeated vertical-line queries."""

    def __init__(self, lines):
        self._lines = lines
        entries = []
        self._fallback = False
        for ordinal, line in enumerate(lines):
            try:
                raw_x = line['x0']
            except (KeyError, TypeError):
                self._fallback = True
                break
            if (
                not isinstance(raw_x, (int, float))
                or not math.isfinite(float(raw_x))
            ):
                self._fallback = True
                break
            entries.append((float(raw_x), ordinal, line))
        if self._fallback:
            self._entries = ()
            self._xs = ()
        else:
            entries.sort(key=lambda entry: (entry[0], entry[1]))
            self._entries = tuple(entries)
            self._xs = tuple(entry[0] for entry in entries)

    def near_x(self, target_x, x_tol):
        if (
            self._fallback
            or not isinstance(target_x, (int, float))
            or not isinstance(x_tol, (int, float))
            or not math.isfinite(float(target_x))
            or not math.isfinite(float(x_tol))
            or float(x_tol) < 0.0
        ):
            return self._lines
        left = bisect.bisect_left(self._xs, float(target_x) - float(x_tol))
        right = bisect.bisect_right(self._xs, float(target_x) + float(x_tol))
        candidates = self._entries[left:right]
        return [
            entry[2]
            for entry in sorted(candidates, key=lambda entry: entry[1])
        ]


def _prepare_vertical_coverage(v_lines):
    return _PreparedVerticalCoverage(v_lines)


def get_annotation_and_contour_lines(page, *, drawing_snapshot=None):
    """
    从 fitz page.get_drawings() 提取 H/V 线段，只保留 annotation 和 contour 两层。
    层选择基于 linewidth 分布的前两个峰值（按 total_length 排序）。

    Args:
        page: fitz.Page 对象

    Returns:
        (h_lines, v_lines)，每条线段格式 {"x0","y0","x1","y1","width"}
    """
    drawings = (
        drawing_snapshot
        if drawing_snapshot is not None
        else page.get_drawings()
    )
    all_segments = []  # (x0, y0, x1, y1, width)

    for d in drawings:
        w = d.get('width', 0) or 0
        for item in d.get('items', []):
            if item[0] == 'l':
                p0, p1 = item[1], item[2]
                all_segments.append((p0.x, p0.y, p1.x, p1.y, w))

    if not all_segments:
        return [], []

    # 分为 H 和 V
    h_all = []
    v_all = []
    for x0, y0, x1, y1, w in all_segments:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dy < 1.5 and dx > 3:
            h_all.append({'x0': min(x0, x1), 'y0': (y0 + y1) / 2,
                          'x1': max(x0, x1), 'y1': (y0 + y1) / 2, 'width': w})
        elif dx < 1.5 and dy > 3:
            v_all.append({'x0': (x0 + x1) / 2, 'y0': min(y0, y1),
                          'x1': (x0 + x1) / 2, 'y1': max(y0, y1), 'width': w})

    # 统计 linewidth 分布（按 total_length 排序找前两个峰值）
    width_length = defaultdict(float)  # width_bucket → total_length
    for seg in h_all + v_all:
        length = math.hypot(seg['x1'] - seg['x0'], seg['y1'] - seg['y0'])
        bucket = round(seg['width'], 2)
        width_length[bucket] += length

    # 排除 width ≈ 0（文字笔画填充）
    width_length = {w: l for w, l in width_length.items() if w > 0.01}

    if not width_length:
        print("[get_anno_contour] 无非零 linewidth 线段")
        return [], []

    # 按 total_length 降序，取前两个峰值
    sorted_widths = sorted(width_length.items(), key=lambda x: -x[1])
    peak_widths = [sw[0] for sw in sorted_widths[:2]]

    print(f"[get_anno_contour] linewidth peaks (by total_length): "
          f"{[(w, round(l, 0)) for w, l in sorted_widths[:5]]}")
    print(f"[get_anno_contour] selected peaks: {peak_widths}")

    # 只保留属于这两个峰值（± 0.05 容差）的线段
    def _in_peaks(w):
        return any(abs(w - pw) <= 0.05 for pw in peak_widths)

    h_lines = [s for s in h_all if _in_peaks(s['width'])]
    v_lines = [s for s in v_all if _in_peaks(s['width'])]

    print(f"[get_anno_contour] H lines: {len(h_all)} → {len(h_lines)}, "
          f"V lines: {len(v_all)} → {len(v_lines)}")

    return h_lines, v_lines


def _collective_v_coverage(v_lines, target_x, x_tol, top_y, bot_y):
    """
    收集 x 在 target_x ± x_tol 内的所有 V 线段，
    计算它们对 [top_y, bot_y] 区间的集体覆盖率。
    使用 interval merge 防止重叠重复计算。
    """
    # 1. 收集候选 V 线段在 [top_y, bot_y] 内的有效区间
    intervals = []
    candidate_lines = (
        v_lines.near_x(target_x, x_tol)
        if isinstance(v_lines, _PreparedVerticalCoverage)
        else v_lines
    )
    for vs in candidate_lines:
        if abs(vs['x0'] - target_x) > x_tol:
            continue
        seg_top = max(vs['y0'], top_y)
        seg_bot = min(vs['y1'], bot_y)
        if seg_bot > seg_top:
            intervals.append((seg_top, seg_bot))

    if not intervals:
        return 0.0

    # 2. Interval merge（标准区间合并）
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # 3. 总覆盖长度 / 框高
    total = sum(e - s for s, e in merged)
    frame_h = bot_y - top_y
    return total / frame_h if frame_h > 0 else 0.0


def reconstruct_rectangles_from_lines(h_lines, v_lines,
                                       y_tol=1.5, x_tol=1.5,
                                       min_width=10, min_height=5,
                                       max_height=25):
    """
    从 H/V 线段中重建闭合矩形。

    Args:
        h_lines: list of dict, 每个 {"x0","y0","x1","y1","width"}，水平线段
        v_lines: list of dict, 每个 {"x0","y0","x1","y1","width"}，竖直线段
        y_tol: float, 共线合并和配对的 y 容差（默认 1.5pt）
        x_tol: float, 端点对齐容差（默认 1.5pt）
        min_width: float, 最小矩形宽度
        min_height: float, 最小矩形高度
        max_height: float, 最大矩形高度（超过的仍返回但标记 oversized=True）

    Returns:
        list of dict, 每个 = {
            "bbox": [x0, y0, x1, y1],
            "width": float,
            "height": float,
            "interior_v_lines": list,
            "compartments": int,
            "oversized": bool,
            "has_interior_h": bool,
        }
    """
    if not h_lines or not v_lines:
        print(f"[rect_reconstruct] 输入不足: H={len(h_lines)}, V={len(v_lines)}")
        return []

    n_h_before = len(h_lines)
    n_v_before = len(v_lines)

    # ── Step 0: 共线合并 (Collinear Merge) ──
    h_merged = _collinear_merge_h(h_lines, y_tol, x_tol)
    v_merged = _collinear_merge_v(v_lines, x_tol, y_tol)
    v_coverage = _prepare_vertical_coverage(v_merged)

    print(f"[rect_reconstruct] collinear merge: H {n_h_before}→{len(h_merged)}, "
          f"V {n_v_before}→{len(v_merged)}")

    # ── Step 1: 水平线对配对 ──
    # 按 y 排序
    h_sorted = sorted(h_merged, key=lambda s: s['y0'])
    pairs = []

    for i in range(len(h_sorted)):
        h_a = h_sorted[i]
        for j in range(i + 1, len(h_sorted)):
            h_b = h_sorted[j]
            dy = abs(h_b['y0'] - h_a['y0'])
            if dy < min_height:
                continue
            if dy > 50:
                break  # h_sorted by y, no more valid pairs

            # x 范围重叠 ≥ 80% of shorter
            overlap_left = max(h_a['x0'], h_b['x0'])
            overlap_right = min(h_a['x1'], h_b['x1'])
            if overlap_right <= overlap_left:
                continue
            overlap_len = overlap_right - overlap_left
            shorter_len = min(h_a['x1'] - h_a['x0'], h_b['x1'] - h_b['x0'])
            if shorter_len <= 0:
                continue
            if overlap_len / shorter_len < 0.8:
                continue

            top_y = min(h_a['y0'], h_b['y0'])
            bot_y = max(h_a['y0'], h_b['y0'])
            pairs.append({
                'top_y': top_y, 'bot_y': bot_y,
                'x_lo': overlap_left, 'x_hi': overlap_right,
            })

    print(f"[rect_reconstruct] H-line pairs: {len(pairs)}")

    # ── Step 2: 闭合检测（多段集体覆盖） ──
    CLOSE_X_TOL = 3.0  # 闭合检测的 x 容差（比共线合并的 1.5 更宽松）
    closed_rects = []
    for pair in pairs:
        top_y, bot_y = pair['top_y'], pair['bot_y']
        x_lo, x_hi = pair['x_lo'], pair['x_hi']

        left_cov = _collective_v_coverage(v_coverage, x_lo, CLOSE_X_TOL, top_y, bot_y)
        right_cov = _collective_v_coverage(v_coverage, x_hi, CLOSE_X_TOL, top_y, bot_y)
        if left_cov < 0.5 or right_cov < 0.5:
            continue

        rect_w = x_hi - x_lo
        if rect_w < min_width:
            continue

        closed_rects.append({
            'x_lo': x_lo, 'x_hi': x_hi,
            'top_y': top_y, 'bot_y': bot_y,
        })

    print(f"[rect_reconstruct] closed rects: {len(closed_rects)}")

    # ── Step 3: 内部结构分析（多段集体覆盖） ──
    results = []
    for cr in closed_rects:
        x_lo, x_hi = cr['x_lo'], cr['x_hi']
        top_y, bot_y = cr['top_y'], cr['bot_y']
        frame_h = bot_y - top_y
        frame_w = x_hi - x_lo

        # 收集框内所有 V 线段的 x 坐标，按 x 聚类（间距 < 3pt 合并）
        interior_vx_raw = []
        for vs in v_merged:
            vx = vs['x0']
            if vx <= x_lo + x_tol or vx >= x_hi - x_tol:
                continue
            interior_vx_raw.append(vx)

        interior_vx_raw.sort()
        # x 聚类：间距 < 3pt 合并为同一竖线位置
        x_clusters = []  # list of list of x values
        for vx in interior_vx_raw:
            if x_clusters and abs(vx - x_clusters[-1][-1]) < 3.0:
                x_clusters[-1].append(vx)
            else:
                x_clusters.append([vx])

        # 对每个聚类位置计算集体覆盖率
        deduped_vx = []
        for cluster in x_clusters:
            cluster_x = sum(cluster) / len(cluster)  # 聚类中心
            cov = _collective_v_coverage(v_coverage, cluster_x, 3.0, top_y, bot_y)
            # 薄框内部分隔符可能是短桩，因此采用相应的局部覆盖率门。
            # 高框（表格等）维持严格标准
            min_cov = 0.12 if frame_h < 15 else 0.5
            if cov >= min_cov:
                deduped_vx.append(round(cluster_x, 2))

        # 所有竖线含左右边界
        all_vx = [x_lo] + deduped_vx + [x_hi]
        compartments = len(all_vx) - 1

        # 内部水平线检查 — 薄框（< 15pt）不可能是多行表格，跳过检查
        has_interior_h = False
        interior_h_count = 0
        if frame_h >= 15:
            for hs in h_merged:
                hy = hs['y0']
                if hy <= top_y + 2 or hy >= bot_y - 2:
                    continue
                h_lo = hs['x0']
                h_hi = hs['x1']
                # x 覆盖 ≥ 50% 框宽
                cover_left = max(h_lo, x_lo)
                cover_right = min(h_hi, x_hi)
                if cover_right - cover_left >= frame_w * 0.5:
                    interior_h_count += 1
            has_interior_h = interior_h_count > 0

        results.append({
            'bbox': [round(x_lo, 2), round(top_y, 2), round(x_hi, 2), round(bot_y, 2)],
            'width': round(frame_w, 2),
            'height': round(frame_h, 2),
            'interior_v_lines': [round(vx, 2) for vx in deduped_vx],
            'compartments': compartments,
            'oversized': frame_h > max_height,
            'has_interior_h': has_interior_h,
            'interior_h_count': interior_h_count,
        })

    # ── Step 4: 去重 (IoU > 0.7 → 保留 compartments 更多的) ──
    results.sort(key=lambda r: -r['compartments'])
    final = []
    for r in results:
        is_dup = False
        for f in final:
            iou = _rect_iou_xyxy(r['bbox'], f['bbox'])
            if iou > 0.7:
                is_dup = True
                break
        if not is_dup:
            final.append(r)

    print(f"[rect_reconstruct] pre-dedup: {len(results)}, post-dedup: {len(final)}")
    for r in final:
        tag = 'OVERSIZED' if r['oversized'] else ''
        tag += ' HAS_INT_H' if r['has_interior_h'] else ''
        print(f"  [{r['compartments']}comp] "
              f"bbox=({r['bbox'][0]:.0f},{r['bbox'][1]:.0f},{r['bbox'][2]:.0f},{r['bbox'][3]:.0f}) "
              f"{r['width']:.0f}x{r['height']:.0f} "
              f"int_v={len(r['interior_v_lines'])} {tag}")

    return final


def _collinear_merge_h(h_lines, y_tol, x_tol):
    """合并共线的水平线段（y 接近、x 连续/重叠）。"""
    if not h_lines:
        return []
    # 统一方向：确保 x0 < x1
    normed = []
    for s in h_lines:
        x0, x1 = min(s['x0'], s['x1']), max(s['x0'], s['x1'])
        normed.append({'x0': x0, 'y0': s['y0'], 'x1': x1, 'y1': s['y0'], 'width': s['width']})

    # 按 (quantized_y, x0) 排序
    normed.sort(key=lambda s: (round(s['y0'] / y_tol) * y_tol, s['x0']))

    merged = [dict(normed[0])]
    for seg in normed[1:]:
        prev = merged[-1]
        if abs(seg['y0'] - prev['y0']) < y_tol and seg['x0'] <= prev['x1'] + x_tol:
            # 合并
            prev['x1'] = max(prev['x1'], seg['x1'])
            prev['y0'] = (prev['y0'] + seg['y0']) / 2
            prev['y1'] = prev['y0']
        else:
            merged.append(dict(seg))
    return merged


def _collinear_merge_v(v_lines, x_tol, y_tol):
    """合并共线的竖直线段（x 接近、y 连续/重叠）。"""
    if not v_lines:
        return []
    # 统一方向：确保 y0 < y1
    normed = []
    for s in v_lines:
        y0, y1 = min(s['y0'], s['y1']), max(s['y0'], s['y1'])
        normed.append({'x0': s['x0'], 'y0': y0, 'x1': s['x0'], 'y1': y1, 'width': s['width']})

    normed.sort(key=lambda s: (round(s['x0'] / x_tol) * x_tol, s['y0']))

    merged = [dict(normed[0])]
    for seg in normed[1:]:
        prev = merged[-1]
        if abs(seg['x0'] - prev['x0']) < x_tol and seg['y0'] <= prev['y1'] + y_tol:
            prev['y1'] = max(prev['y1'], seg['y1'])
            prev['x0'] = (prev['x0'] + seg['x0']) / 2
            prev['x1'] = prev['x0']
        else:
            merged.append(dict(seg))
    return merged
