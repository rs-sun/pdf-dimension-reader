"""pdf_analyzer_capsule.py — 跑道框检测（三阶段弧线配对）

从 pdf_analyzer.py 拆出。"""

import math
import re

import numpy as np
import fitz

from pdf_analyzer_utils import _curve_bbox_wh, _curve_center, _get_fitz_page


def _detect_capsules_legacy(page):
    """
    旧版跑道框检测（单路径方式）。作为新算法的 fallback。
    """
    curves = page.curves
    capsules = []

    for c in curves:
        pts = c.get('pts', [])
        if len(pts) <= 8:
            continue

        w, h = _curve_bbox_wh(c)
        area = w * h
        if area < 100:
            continue
        if area < 200 or area > 5000:
            continue
        if h < 0.01:
            continue
        aspect = w / h
        if aspect < 2.5:
            continue

        cx, cy = _curve_center(c)
        capsules.append({
            'x': cx - w / 2,
            'y': cy - h / 2,
            'w': w,
            'h': h,
        })

    return capsules


def detect_capsules(page, *, drawing_snapshot=None):
    """
    从 PyMuPDF page.drawings 中检测跑道框（关键尺寸的椭圆形边框）。
    三阶段弧线对聚合算法：
      Phase 1: 提取近似半圆弧的 Bezier 曲线段
      Phase 2: 跨 drawing 配对弧线
      Phase 3: 去重与输出

    如果新算法结果为空，fallback 到旧版单路径检测。

    返回: [{'x': float, 'y': float, 'w': float, 'h': float}, ...]
    """
    from itertools import combinations

    fitz_page = None if drawing_snapshot is not None else _get_fitz_page(page)
    if drawing_snapshot is None and fitz_page is None:
        print("[detect_capsules] fitz page unavailable, using legacy fallback")
        return _detect_capsules_legacy(page)

    drawings = (
        tuple(drawing_snapshot)
        if drawing_snapshot is not None
        else tuple(fitz_page.get_drawings())
    )

    # === Phase 1: 弧线候选提取 ===

    # 预收集全页 line endpoints（用于 Phase 2 桥接检测）
    # 空间网格索引把端点哈希到 CELL_SIZE 网格，避免每次查询全量扫描。
    _GRID_CELL = 8.0  # pt — 略大于 tolerance(6.0)，保证查询只需检查 3x3 邻域
    _ep_grid = {}  # (gx, gy) → list of fitz.Point endpoints
    for drawing in drawings:
        for item in drawing["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                for p in (p1, p2):
                    gk = (int(p.x // _GRID_CELL), int(p.y // _GRID_CELL))
                    _ep_grid.setdefault(gk, []).append(p)

    arc_candidates = []

    for d_idx, drawing in enumerate(drawings):
        items = drawing["items"]
        i = 0
        while i < len(items):
            if items[i][0] != "c":
                i += 1
                continue

            # 收集连续的 Bezier 段（同一 drawing 内首尾相连的）
            bezier_chain = [items[i]]
            j = i + 1
            while j < len(items) and items[j][0] == "c":
                prev_end = bezier_chain[-1][4]  # p4 of previous
                curr_start = items[j][1]        # p1 of current
                if abs(prev_end - curr_start) < 2.0:
                    bezier_chain.append(items[j])
                    j += 1
                else:
                    break
            i = j

            # 计算弧参数
            if len(bezier_chain) == 1:
                seg = bezier_chain[0]
                p0, p1, p2, p3 = seg[1], seg[2], seg[3], seg[4]
                chord = p3 - p0
                chord_mid = (p0 + p3) * 0.5
                control_mid = (p1 + p2) * 0.5
                sag = control_mid - chord_mid
            else:
                # 多段合并弧
                p0 = bezier_chain[0][1]   # 首段 p1
                p3 = bezier_chain[-1][4]  # 末段 p4
                chord = p3 - p0
                chord_mid = (p0 + p3) * 0.5
                # 所有中间控制点的平均
                all_controls = []
                for seg in bezier_chain:
                    all_controls.append(seg[2])
                    all_controls.append(seg[3])
                ctrl_avg = fitz.Point(
                    sum(c.x for c in all_controls) / len(all_controls),
                    sum(c.y for c in all_controls) / len(all_controls),
                )
                sag = ctrl_avg - chord_mid

            sag_len = abs(sag)
            chord_len = abs(chord)

            if sag_len < 0.5 or chord_len < 2:
                continue

            radius = (chord_len ** 2 / (8 * sag_len)) + (sag_len / 2)
            arc_angle = 2 * math.asin(min(chord_len / (2 * radius), 1.0))

            # 筛选：近似半圆弧 [140°, 220°]，半径 [3pt, 60pt]
            if arc_angle < 2.44 or arc_angle > 3.84:
                continue
            if radius < 3 or radius > 60:
                continue

            center = chord_mid + sag * ((radius - sag_len) / sag_len)
            opening_dir = sag * (-1.0 / sag_len)

            ep_a = bezier_chain[0][1]    # 弧起点
            ep_b = bezier_chain[-1][4]   # 弧终点

            # 计算包围盒
            all_pts = []
            for seg in bezier_chain:
                all_pts.extend([seg[1], seg[2], seg[3], seg[4]])
            xs = [p.x for p in all_pts]
            ys = [p.y for p in all_pts]
            bbox = fitz.Rect(min(xs), min(ys), max(xs), max(ys))

            arc_candidates.append({
                "center": center,
                "radius": radius,
                "arc_angle": arc_angle,
                "ep_a": ep_a,
                "ep_b": ep_b,
                "opening_dir": opening_dir,
                "bbox": bbox,
                "drawing_idx": d_idx,
            })

    print(f"[detect_capsules] Phase 1: {len(arc_candidates)} arc candidates from {len(drawings)} drawings")

    # === Phase 2: 弧线配对 ===

    def _check_endpoint_evidence(arc_a, arc_b, grid, cell_size, tolerance=6.0):
        """检查两弧的4个端点附近是否有线段端头存在（至少3个有evidence）。
        空间网格查询代替全量线段扫描 — O(1) per endpoint."""
        tol_sq = tolerance * tolerance
        endpoints = [
            arc_a["ep_a"], arc_a["ep_b"],
            arc_b["ep_a"], arc_b["ep_b"],
        ]
        evidence_count = 0
        need = 3  # 至少 3 个端点有 evidence
        for ep in endpoints:
            gx = int(ep.x // cell_size)
            gy = int(ep.y // cell_size)
            found = False
            # 搜索 3x3 邻域格子
            for dx in (-1, 0, 1):
                if found:
                    break
                for dy in (-1, 0, 1):
                    bucket = grid.get((gx + dx, gy + dy))
                    if not bucket:
                        continue
                    for lp in bucket:
                        ddx = lp.x - ep.x
                        ddy = lp.y - ep.y
                        if ddx * ddx + ddy * ddy < tol_sq:
                            found = True
                            break
                    if found:
                        break
            if found:
                evidence_count += 1
            else:
                # 早退：剩余端点不够凑 3 个
                remaining = len(endpoints) - endpoints.index(ep) - 1
                if evidence_count + remaining < need:
                    return False
        return evidence_count >= need

    MAX_CAPSULE_LENGTH = 200  # pt — 两弧线中心距上限（AABB 粗筛）

    capsule_candidates = []

    for i, j in combinations(range(len(arc_candidates)), 2):
        arc_a = arc_candidates[i]
        arc_b = arc_candidates[j]

        # 条件 0（AABB 粗筛）：bbox 中心距 > MAX_CAPSULE_LENGTH → 不可能属于同一跑道框
        ba, bb = arc_a["bbox"], arc_b["bbox"]
        cx_a = (ba.x0 + ba.x1) / 2
        cy_a = (ba.y0 + ba.y1) / 2
        cx_b = (bb.x0 + bb.x1) / 2
        cy_b = (bb.y0 + bb.y1) / 2
        if math.hypot(cx_a - cx_b, cy_a - cy_b) > MAX_CAPSULE_LENGTH:
            continue

        # 条件 1：半径相近
        r_avg = (arc_a["radius"] + arc_b["radius"]) / 2
        if abs(arc_a["radius"] - arc_b["radius"]) > r_avg * 0.2:
            continue

        # 条件 2：开口方向相反（dot < -0.7）
        dot = (arc_a["opening_dir"].x * arc_b["opening_dir"].x +
               arc_a["opening_dir"].y * arc_b["opening_dir"].y)
        if dot > -0.7:
            continue

        # 条件 3：中心距合理
        center_dist = abs(arc_a["center"] - arc_b["center"])
        if center_dist < r_avg * 0.3 or center_dist > r_avg * 50:
            continue

        # 条件 4：端点闭合校验
        tolerance = 4.0
        match_direct = (
            abs(arc_a["ep_a"] - arc_b["ep_a"]) < tolerance and
            abs(arc_a["ep_b"] - arc_b["ep_b"]) < tolerance
        )
        match_cross = (
            abs(arc_a["ep_a"] - arc_b["ep_b"]) < tolerance and
            abs(arc_a["ep_b"] - arc_b["ep_a"]) < tolerance
        )

        if not match_direct and not match_cross:
            if not _check_endpoint_evidence(arc_a, arc_b, _ep_grid, _GRID_CELL, 6.0):
                continue

        # 通过所有条件 → 合成跑道框
        capsule_bbox = arc_a["bbox"] | arc_b["bbox"]
        # 计算定向尺寸（不依赖 AABB，斜向 capsule 的 AABB 接近正方形会被误杀）
        ac = arc_a["center"]
        bc = arc_b["center"]
        _oriented_long = center_dist + 2 * r_avg   # capsule 实际长度
        _oriented_short = 2 * r_avg                 # capsule 实际宽度（直径）
        _angle_deg = math.degrees(math.atan2(bc.y - ac.y, bc.x - ac.x)) % 180
        capsule_candidates.append({
            "bbox": capsule_bbox,
            "radius": r_avg,
            "arc_indices": (i, j),
            "oriented_long": _oriented_long,
            "oriented_short": _oriented_short,
            "angle": _angle_deg,
        })

    print(f"[detect_capsules] Phase 2: {len(capsule_candidates)} raw pairs")

    # === Phase 3: 去重与输出 ===

    capsule_candidates.sort(key=lambda c: c["bbox"].get_area(), reverse=True)

    final_capsules = []
    for cap in capsule_candidates:
        duplicate = False
        for existing in final_capsules:
            intersection = cap["bbox"] & existing["bbox"]
            if intersection.is_empty:
                continue
            inter_area = intersection.get_area()
            union_area = cap["bbox"].get_area() + existing["bbox"].get_area() - inter_area
            if union_area > 0 and inter_area / union_area > 0.5:
                duplicate = True
                break
        if not duplicate:
            w, h = cap["bbox"].width, cap["bbox"].height
            if w > 6 and h > 4:
                # 用定向尺寸做形状校验（斜向 capsule 的 AABB 接近正方形，用 AABB 会误杀）
                _ol = cap.get("oriented_long", max(w, h))
                _os = cap.get("oriented_short", min(w, h))
                if _ol > 200:
                    continue  # 过大
                if _os > 60:
                    continue  # 过胖
                if _os > 0 and _ol / _os < 2.0:
                    continue  # 不够细长
                final_capsules.append(cap)

    # Fallback 到旧逻辑
    legacy_results = _detect_capsules_legacy(page)

    result = [{"x": c["bbox"].x0, "y": c["bbox"].y0,
               "w": c["bbox"].width, "h": c["bbox"].height,
               "angle": c.get("angle", 0.0),
               "oriented_long": c.get("oriented_long"),
               "oriented_short": c.get("oriented_short"),
               "angle_source": "arc_pair",
               "source": "arc_pair"}
              for c in final_capsules]

    if not result:
        result = legacy_results

    print(f"[detect_capsules] Phase 3: {len(capsule_candidates)} before filter → {len(final_capsules)} after filter (legacy fallback: {len(legacy_results)})")

    # F1: 单椭圆 path fallback (CAD 一笔画椭圆 = {c: 2} 签名，
    # 弧线对配对逻辑漏检)
    ellipse_results = _detect_ellipse_paths_from_drawings(drawings)
    if ellipse_results:
        # IoU 去重 (避免跟 Phase 3 / legacy 输出重复)
        merged = list(result)
        for ec in ellipse_results:
            dup = False
            for existing in merged:
                if _bbox_iou(ec, existing) > 0.5:
                    dup = True
                    break
            if not dup:
                merged.append(ec)
        n_added = len(merged) - len(result)
        if n_added:
            print(f"[detect_capsules] F1 ellipse fallback: +{n_added} "
                  f"(from {len(ellipse_results)} ellipse paths)")
        result = merged

    # F2: polyline capsule fallback.  Some converted CAD drawings flatten the
    # rounded capsule ends into many short straight line segments inside a much
    # larger drawing path.  The Bezier-only detector above cannot see those.
    line_poly_results = _detect_line_poly_capsules_from_drawings(drawings)
    if line_poly_results:
        merged = list(result)
        for lc in line_poly_results:
            dup = False
            for existing in merged:
                if _bbox_iou(lc, existing) > 0.5:
                    dup = True
                    break
            if not dup:
                merged.append(lc)
        n_added = len(merged) - len(result)
        if n_added:
            print(f"[detect_capsules] F2 line-poly fallback: +{n_added} "
                  f"(from {len(line_poly_results)} polyline components)")
        result = merged

    # 前置过滤：丢弃明显假阳性 capsule。
    # 用 min(w, h) 取代旧的 cap['h'] —— 旧版本只看 AABB 高度对横向跑道有效，
    # 竖向跑道 (h>>w) 时 cap['h'] 始终大但 cap['w']<8 是真窄，不应被放过；
    # 对横向跑道 min(w,h)=h 行为不变。
    pre_filter_count = len(result)
    result = [cap for cap in result if min(cap['w'], cap['h']) >= 8]
    if pre_filter_count != len(result):
        print(f"[detect_capsules] F4 short-axis filter (min(w,h)>=8): "
              f"{pre_filter_count} → {len(result)}")

    return result


def _bbox_iou(a, b):
    """两个 capsule dict ({x,y,w,h}) 的 IoU。"""
    ax0, ay0 = a['x'], a['y']
    ax1, ay1 = ax0 + a['w'], ay0 + a['h']
    bx0, by0 = b['x'], b['y']
    bx1, by1 = bx0 + b['w'], by0 + b['h']
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aarea = a['w'] * a['h']
    barea = b['w'] * b['h']
    union = aarea + barea - inter
    return inter / union if union > 0 else 0.0


def _detect_ellipse_paths(page):
    """检测一笔画椭圆 / 胶囊 path（cubic-only，2 段或 4 段闭合）。

    很多 CAD 软件画 KEY 跑道框是"一笔画椭圆/胶囊"，而非"两弧 + 两直线"组合。
    Phase 1/2/3 的弧线对配对会全漏。这个 fallback 直接扫 cubic-only 闭合 path：
      - 2 段：椭圆上下两半（最常见）
      - 4 段：椭圆四 quarter（CAD 也常见，每段 ~115° / 90°）

    过滤：
      - 尺寸 10pt ≤ max(w,h) ≤ 200pt (排除噪声 + 视图边界)
      - aspect ≥ 2.0 或 ≤ 0.5 (跑道形 / 椭圆形，排除接近圆形)
    """
    fitz_page = _get_fitz_page(page)
    if fitz_page is None:
        return []
    return _detect_ellipse_paths_from_drawings(fitz_page.get_drawings())


def _detect_ellipse_paths_from_drawings(drawings):
    """Detect one-path capsules from an existing immutable drawing snapshot."""
    out = []
    for d in drawings:
        items = d.get('items', [])
        # 接受 2 或 4 段闭合 cubic（覆盖 CAD 一笔画椭圆 / 4-quarter 椭圆）
        if len(items) not in (2, 4):
            continue
        if not all(it[0] == 'c' for it in items):
            continue
        rect = d.get('rect')
        if rect is None:
            continue
        w, h = rect.width, rect.height
        if max(w, h) > 200 or max(w, h) < 10:
            continue
        if min(w, h) < 4:
            continue
        aspect = w / max(h, 0.01)
        if not (aspect >= 2.0 or aspect <= 0.5):
            continue
        out.append({
            'x': rect.x0, 'y': rect.y0,
            'w': w, 'h': h,
            'angle': 90.0 if h > w else 0.0,
            'source': 'ellipse_path',
        })
    return out


def _detect_line_poly_capsules(page):
    """Detect capsules whose rounded ends are flattened into line segments."""
    fitz_page = _get_fitz_page(page)
    if fitz_page is None:
        return []
    return _detect_line_poly_capsules_from_drawings(fitz_page.get_drawings())


def _detect_line_poly_capsules_from_drawings(drawings):
    """Detect line-only capsule components from PyMuPDF drawing dictionaries.

    CAD conversions sometimes approximate a racetrack outline as:
      many short line segments for one rounded end,
      two long straight sides,
      many short line segments for the opposite rounded end.

    The component may live inside a huge drawing path with hundreds of other
    lines, so this builds endpoint-connected components instead of trusting
    drawing-level bboxes.
    """

    def _segment_length(seg):
        return abs(seg[1] - seg[0])

    def _bbox_for_segments(segs):
        xs = []
        ys = []
        for p0, p1 in segs:
            xs.extend([p0.x, p1.x])
            ys.extend([p0.y, p1.y])
        return fitz.Rect(min(xs), min(ys), max(xs), max(ys))

    def _merge_candidates(out, cap):
        for existing in out:
            if _bbox_iou(cap, existing) > 0.5:
                return
        out.append(cap)

    def _has_parallel_sides(segs, rect, vertical):
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        w, h = rect.width, rect.height
        tol = 2.0
        if vertical:
            needed = h * 0.45
            left = right = False
            for p0, p1 in segs:
                if abs(p0.x - p1.x) > tol or _segment_length((p0, p1)) < needed:
                    continue
                x = (p0.x + p1.x) / 2.0
                if abs(x - x0) <= tol:
                    left = True
                if abs(x - x1) <= tol:
                    right = True
            return left and right
        needed = w * 0.45
        top = bottom = False
        for p0, p1 in segs:
            if abs(p0.y - p1.y) > tol or _segment_length((p0, p1)) < needed:
                continue
            y = (p0.y + p1.y) / 2.0
            if abs(y - y0) <= tol:
                top = True
            if abs(y - y1) <= tol:
                bottom = True
        return top and bottom

    def _has_segmented_round_ends(segs, rect, vertical):
        w, h = rect.width, rect.height
        short_axis = min(w, h)
        zone = max(short_axis * 0.75, 8.0)
        max_arc_seg = max(short_axis * 0.45, 4.0)
        if vertical:
            near_start = near_end = 0
            for p0, p1 in segs:
                length = _segment_length((p0, p1))
                if length > max_arc_seg:
                    continue
                cy = (p0.y + p1.y) / 2.0
                if cy <= rect.y0 + zone:
                    near_start += 1
                if cy >= rect.y1 - zone:
                    near_end += 1
            return near_start >= 5 and near_end >= 5
        near_start = near_end = 0
        for p0, p1 in segs:
            length = _segment_length((p0, p1))
            if length > max_arc_seg:
                continue
            cx = (p0.x + p1.x) / 2.0
            if cx <= rect.x0 + zone:
                near_start += 1
            if cx >= rect.x1 - zone:
                near_end += 1
        return near_start >= 5 and near_end >= 5

    def _classify_component(segs):
        if len(segs) < 12:
            return None
        rect = _bbox_for_segments(segs)
        w, h = rect.width, rect.height
        if max(w, h) < 18 or max(w, h) > 200:
            return None
        if min(w, h) < 8:
            return None
        ratio = max(w, h) / max(min(w, h), 0.01)
        if ratio < 2.0:
            return None
        vertical = h > w
        if not _has_parallel_sides(segs, rect, vertical):
            return None
        if not _has_segmented_round_ends(segs, rect, vertical):
            return None
        return {
            'x': rect.x0,
            'y': rect.y0,
            'w': w,
            'h': h,
            'angle': 90.0 if vertical else 0.0,
            'source': 'line_poly_capsule',
        }

    out = []
    endpoint_tol = 1.5
    grid_cell = endpoint_tol * 2.0

    for drawing in drawings:
        segs = []
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            p0, p1 = item[1], item[2]
            if abs(p1 - p0) < 0.1:
                continue
            segs.append((p0, p1))
        if len(segs) < 12:
            continue

        parent = list(range(len(segs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        endpoint_grid = {}
        for si, (p0, p1) in enumerate(segs):
            for p in (p0, p1):
                gx = int(p.x // grid_cell)
                gy = int(p.y // grid_cell)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for other_i, other_p in endpoint_grid.get((gx + dx, gy + dy), []):
                            ddx = p.x - other_p.x
                            ddy = p.y - other_p.y
                            if ddx * ddx + ddy * ddy <= endpoint_tol * endpoint_tol:
                                union(si, other_i)
                endpoint_grid.setdefault((gx, gy), []).append((si, p))

        groups = {}
        for si, seg in enumerate(segs):
            groups.setdefault(find(si), []).append(seg)

        for comp in groups.values():
            cap = _classify_component(comp)
            if cap:
                _merge_candidates(out, cap)

    return out
