#!/usr/bin/env python3

import math
import numpy as np
from sklearn.cluster import DBSCAN


def detect_arrows(annotation_lines, long_line_threshold=20, arrow_eps=3.0,
                  arrow_min_samples=5, short_line_cutoff=3.0,
                  bind_dist_threshold=None, verbose=False):
    """
    从 annotation 层线段中检测箭头并配对定位尺寸线。

    参数:
        annotation_lines: list of dict {x0, y0, x1, y1, lineWidth}
            所有 annotation linewidth 层的线段
        long_line_threshold: float
            长线段最小长度（候选尺寸线/延伸线）
        arrow_eps: float
            DBSCAN 聚类 eps 参数
        arrow_min_samples: int
            DBSCAN 聚类 min_samples 参数
        short_line_cutoff: float
            短线段/长线段分界（pt）
        bind_dist_threshold: float | None
            箭头尖端绑定长线端点的最大距离。None 时随 arrow_eps 取一个
            保守默认值；需要容忍转曲 PDF 端点错位时，由调用方显式放宽。

    返回:
        {
            'arrows': list of dict {
                'tip': (x, y),
                'base_center': (x, y),
                'direction': (dx, dy),
                'bbox': {x, y, w, h},
                'bound_line_idx': int,
            },
            'dimension_lines': list of dict {
                'start': (x, y),
                'end': (x, y),
                'midpoint': (x, y),
                'length': float,
                'orientation': 'H' | 'V' | 'D',
                'arrow_a': dict,
                'arrow_b': dict,
                'line_idx': int,
            },
            'long_lines': list of dict {x0, y0, x1, y1, length, orientation},
        }
    """
    if not annotation_lines:
        return {'arrows': [], 'dimension_lines': [], 'long_lines': []}
    if bind_dist_threshold is None:
        bind_dist_threshold = max(3.0, arrow_eps)

    # Step 1: 分离短线段和长线段
    short_lines = []
    long_lines = []

    for i, ln in enumerate(annotation_lines):
        x0, y0 = ln['x0'], ln['y0']
        x1, y1 = ln['x1'], ln['y1']
        seg_len = math.hypot(x1 - x0, y1 - y0)

        if seg_len < short_line_cutoff:
            # 短线段：箭头填充 or 文字笔画
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            short_lines.append({
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'mx': mx, 'my': my, 'length': seg_len, 'idx': i,
            })
        elif seg_len >= long_line_threshold:
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            if dy < 1.5 and dx > 5:
                orient = 'H'
            elif dx < 1.5 and dy > 5:
                orient = 'V'
            else:
                orient = 'D'
            long_lines.append({
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'length': seg_len, 'orientation': orient, 'idx': i,
            })

    if not short_lines:
        return {'arrows': [], 'dimension_lines': [], 'long_lines': long_lines}

    # Step 2: 短线段 DBSCAN 聚类
    centers = np.array([[s['mx'], s['my']] for s in short_lines])
    db = DBSCAN(eps=arrow_eps, min_samples=arrow_min_samples).fit(centers)
    labels = db.labels_

    # 收集 clusters
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(idx)

    # 预计算长线段端点，用于后续绑定
    long_endpoints = []
    for li, ll in enumerate(long_lines):
        long_endpoints.append((ll['x0'], ll['y0'], li, 'p0'))
        long_endpoints.append((ll['x1'], ll['y1'], li, 'p1'))

    # Step 3: 箭头特征识别
    arrows = []
    for label, indices in clusters.items():
        cluster_pts = [short_lines[i] for i in indices]

        # 计算 bbox
        all_x = []
        all_y = []
        for s in cluster_pts:
            all_x.extend([s['x0'], s['x1']])
            all_y.extend([s['y0'], s['y1']])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        w = max_x - min_x
        h = max_y - min_y

        # 避免零宽高
        if w < 0.01 and h < 0.01:
            continue

        max_dim = max(w, h)
        min_dim = min(w, h) if min(w, h) > 0.01 else 0.01
        aspect = max_dim / min_dim

        # 箭头打分条件
        cond_aspect = aspect > 2.0
        cond_max = max_dim < 15
        cond_min = min_dim < 5

        if not (cond_aspect and cond_max and cond_min):
            continue

        # 计算质心
        cx = sum(s['mx'] for s in cluster_pts) / len(cluster_pts)
        cy = sum(s['my'] for s in cluster_pts) / len(cluster_pts)

        # 找所有端点中离质心最远的点（极值点 = 潜在尖端）
        extremes = []
        for s in cluster_pts:
            for px, py in [(s['x0'], s['y0']), (s['x1'], s['y1'])]:
                d = math.hypot(px - cx, py - cy)
                extremes.append((px, py, d))
        extremes.sort(key=lambda e: e[2], reverse=True)

        # 尝试绑定到长线段端点
        best_bind = None
        best_bind_dist = float('inf')
        best_bind_extreme = None

        for ex_x, ex_y, _ in extremes[:6]:  # 检查前几个极值点
            for ep_x, ep_y, li, ep_name in long_endpoints:
                d = math.hypot(ex_x - ep_x, ex_y - ep_y)
                if d < bind_dist_threshold and d < best_bind_dist:
                    best_bind_dist = d
                    best_bind = (li, ep_name)
                    best_bind_extreme = (ex_x, ex_y)

        bound_line_idx = best_bind[0] if best_bind else -1

        # 确定尖端和方向
        if best_bind_extreme:
            tip = best_bind_extreme
        else:
            # 未绑定：取最远极值点
            tip = (extremes[0][0], extremes[0][1])

        # 底边中点 = bbox 中离 tip 最远的一端的中点
        # 判断 tip 在 bbox 的哪一端
        if w > h:
            # 水平方向为主
            if abs(tip[0] - min_x) < abs(tip[0] - max_x):
                # tip 在左端，base 在右端
                base_center = (max_x, (min_y + max_y) / 2)
            else:
                base_center = (min_x, (min_y + max_y) / 2)
        else:
            # 竖直方向为主
            if abs(tip[1] - min_y) < abs(tip[1] - max_y):
                base_center = ((min_x + max_x) / 2, max_y)
            else:
                base_center = ((min_x + max_x) / 2, min_y)

        # 方向向量 base_center → tip
        dx = tip[0] - base_center[0]
        dy = tip[1] - base_center[1]
        norm = math.hypot(dx, dy)
        if norm < 0.001:
            direction = (1.0, 0.0)
        else:
            direction = (dx / norm, dy / norm)

        arrows.append({
            'tip': tip,
            'base_center': base_center,
            'direction': direction,
            'bbox': {'x': min_x, 'y': min_y, 'w': w, 'h': h},
            'bound_line_idx': bound_line_idx,
            'cluster_size': len(cluster_pts),
            '_bind_endpoint': best_bind[1] if best_bind else None,
            '_bind_line_local_idx': best_bind[0] if best_bind else None,
        })

    # Step 5: 箭头配对 → dimension line
    dimension_lines = []

    # 按 bound_line_idx 分组箭头
    arrows_by_line = {}
    for arrow in arrows:
        li = arrow['bound_line_idx']
        if li >= 0:
            arrows_by_line.setdefault(li, []).append(arrow)

    for li, bound_arrows in arrows_by_line.items():
        if len(bound_arrows) < 2:
            continue

        ll = long_lines[li]

        # 找绑定到 p0 和 p1 的箭头
        p0_arrows = [a for a in bound_arrows if a['_bind_endpoint'] == 'p0']
        p1_arrows = [a for a in bound_arrows if a['_bind_endpoint'] == 'p1']

        if not p0_arrows or not p1_arrows:
            continue

        arrow_a = p0_arrows[0]
        arrow_b = p1_arrows[0]

        # 检查两箭头方向相反（dot product < -0.5）
        dot = (arrow_a['direction'][0] * arrow_b['direction'][0] +
               arrow_a['direction'][1] * arrow_b['direction'][1])
        if dot >= -0.5:
            continue

        # 确认为 dimension line
        start = arrow_a['tip']
        end = arrow_b['tip']
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        length = math.hypot(end[0] - start[0], end[1] - start[1])

        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dy < 1.5:
            orient = 'H'
        elif dx < 1.5:
            orient = 'V'
        else:
            orient = 'D'

        dimension_lines.append({
            'start': start,
            'end': end,
            'midpoint': midpoint,
            'length': length,
            'orientation': orient,
            'arrow_a': arrow_a,
            'arrow_b': arrow_b,
            'line_idx': ll['idx'],
        })

    if verbose:
        print(f"[arrow_detector] short_lines: {len(short_lines)}, "
              f"long_lines: {len(long_lines)}, "
              f"clusters: {len(clusters)}, "
              f"arrows: {len(arrows)}, "
              f"dimension_lines: {len(dimension_lines)}")

    return {
        'arrows': arrows,
        'dimension_lines': dimension_lines,
        'long_lines': long_lines,
    }
