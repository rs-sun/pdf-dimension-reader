"""pdf_analyzer_cluster.py — 文字区域聚类 + 矩形提取

从 pdf_analyzer.py 拆出。"""

import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from sklearn.cluster import DBSCAN
import fitz
from PIL import Image, ImageDraw

from pdf_analyzer_utils import _curve_bbox_wh, _curve_center, _iou, _get_fitz_page


def cluster_text_regions(page, debug_plot=False, debug_output_path=None,
                         annotation_width=None, extra_points=None,
                         enable_numeric_symbol_seeds=True,
                         enable_plus_minus_symbol_seeds=True,
                         plus_minus_shadow_sink=None):
    """
    对页面中的微小曲线（文字笔画）做 DBSCAN 聚类，定位文字区域。

    参数:
        page: pdfplumber Page 对象
        debug_plot: 是否生成调试可视化
        debug_output_path: 调试图输出路径
        annotation_width: float or None — 保留供外部模块使用，
            本函数不再纳入 annotation 短线段到 DBSCAN。
        extra_points: list of (x, y) tuples — 额外的点（如连通分量过滤后的线段中点），
            追加到微小曲线中心点后一起跑 DBSCAN。
        enable_numeric_symbol_seeds: bool — 是否把贴近文字主体的小数点 / ±
            seed 合并回文字区域。默认开启；调试脚本可关闭做 A/B 回归。
        enable_plus_minus_symbol_seeds: bool — 是否允许 legacy 全页 ± 检测结果
            成为 seed 并扩张文字区域。strict 路径关闭；legacy 默认保持开启。
        plus_minus_shadow_sink: list or None — 可选诊断账本。记录 legacy 全页
            ± 几何候选，但记录本身永不获得消费权限。

    返回:
    [
      {
        'x': float, 'y': float, 'w': float, 'h': float,
        'curve_count': int,
        'avg_curve_size': float,
        'confidence': 'high' | 'low'
      },
      ...
    ]
    """
    curves = page.curves
    page_area = page.width * page.height

    # 1. 筛选微小曲线
    tiny_curves = []
    for c in curves:
        w, h = _curve_bbox_wh(c)
        if w < 15 and h < 15:
            tiny_curves.append(c)

    if not tiny_curves and not extra_points:
        print("[info] No tiny curves found — likely Type A or empty page.")
        if plus_minus_shadow_sink is not None:
            _add_attached_numeric_symbol_seed_regions(
                page,
                [],
                enable_decimal_symbol_seeds=False,
                enable_plus_minus_symbol_seeds=False,
                plus_minus_shadow_sink=plus_minus_shadow_sink,
            )
        return []

    # 2. 提取中心点（微小曲线 + extra_points）
    curve_centers = [_curve_center(c) for c in tiny_curves] if tiny_curves else []
    if extra_points:
        all_centers = curve_centers + list(extra_points)
    else:
        all_centers = curve_centers
    centers = np.array(all_centers)

    # 3. 计算 eps：微小曲线高度中位数 × 2.0
    heights = []
    for c in tiny_curves:
        _, h = _curve_bbox_wh(c)
        if h > 0:
            heights.append(h)
    if heights:
        eps = statistics.median(heights) * 2.0
    else:
        eps = 4.0  # 回退默认值
    eps = max(eps, 1.0)

    # 4. DBSCAN 聚类
    db = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = db.labels_

    # 5. 对每个簇计算包围盒
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_dict[label].append(idx)

    n_tiny = len(tiny_curves)

    regions = []
    for label, indices in cluster_dict.items():
        if label == -1:
            continue  # 噪点（min_samples=1 时理论上不会有，保险起见）

        all_pts = []
        curve_areas = []
        curve_count = 0
        for i in indices:
            if i < n_tiny:
                c = tiny_curves[i]
                w, h = _curve_bbox_wh(c)
                cx, cy = _curve_center(c)
                all_pts.append((cx - w/2, cy - h/2))
                all_pts.append((cx + w/2, cy + h/2))
                curve_areas.append(w * h)
                curve_count += 1
            else:
                # extra_point: use center directly with minimal extent
                px, py = all_centers[i]
                all_pts.append((px - 0.5, py - 0.5))
                all_pts.append((px + 0.5, py + 0.5))
                curve_count += 1

        if not all_pts:
            continue

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = 2.0
        rx = min(xs) - margin
        ry = min(ys) - margin
        rx1 = max(xs) + margin
        ry1 = max(ys) + margin
        rw = rx1 - rx
        rh = ry1 - ry

        avg_area = statistics.mean(curve_areas) if curve_areas else 0
        region_area = rw * rh

        # 置信度评估
        confidence = 'high'
        if page_area > 0 and region_area / page_area > 0.10:
            confidence = 'low'  # 区域太大，可能是密集剖面线
        if avg_area > 50:
            confidence = 'low'  # 曲线平均面积太大，可能是几何图形

        regions.append({
            'x': rx, 'y': ry,
            'w': rw, 'h': rh,
            'curve_count': curve_count,
            'avg_curve_size': round(avg_area, 2),
            'confidence': confidence,
        })

    # 按 y 然后 x 排序，便于阅读
    regions.sort(key=lambda r: (round(r['y'] / 5) * 5, r['x']))

    # 6. 合并相邻小 region（字符级 → 词级）
    regions = _merge_adjacent_regions(regions)

    # 6a. Track B: 短线段中点补充 DBSCAN（双轨）
    # 从 page.lines 中提取长度 < 3pt 的线段中点，与微小曲线中点合并重跑 DBSCAN
    short_line_mids = []
    for ln in page.lines:
        lx0, ly0 = ln.get('x0', 0), ln.get('y0', 0)
        lx1, ly1 = ln.get('x1', 0), ln.get('y1', 0)
        length = math.sqrt((lx1 - lx0) ** 2 + (ly1 - ly0) ** 2)
        if length < 3.0:
            short_line_mids.append(((lx0 + lx1) / 2, (ly0 + ly1) / 2))

    if short_line_mids:
        # 合并微小曲线中心 + 短线段中点
        track_b_centers = list(all_centers) + short_line_mids
        if len(track_b_centers) >= 2:
            track_b_arr = np.array(track_b_centers)
            db_b = DBSCAN(eps=eps, min_samples=1).fit(track_b_arr)
            labels_b = db_b.labels_

            cluster_dict_b = defaultdict(list)
            for idx, label in enumerate(labels_b):
                if label >= 0:
                    cluster_dict_b[label].append(track_b_centers[idx])

            regions_b = []
            for label, pts in cluster_dict_b.items():
                if len(pts) < 5:
                    continue  # 少于 5 个点的簇不太可能是文字
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                margin = 2.0
                rx = min(xs) - margin
                ry = min(ys) - margin
                rw = max(xs) - min(xs) + 2 * margin
                rh = max(ys) - min(ys) + 2 * margin
                if rw * rh < 100:
                    continue  # 面积 < 100 pt² 的区域不是文字
                regions_b.append({
                    'x': rx, 'y': ry, 'w': rw, 'h': rh,
                    'curve_count': len(pts),
                    'avg_curve_size': 0,
                    'confidence': 'high',
                    'source': 'track_b',
                })

            regions_b.sort(key=lambda r: (round(r['y'] / 5) * 5, r['x']))
            regions_b = _merge_adjacent_regions(regions_b)

            # 仅追加 IoU < 0.3 的新区域
            n_added = 0
            for rb in regions_b:
                max_iou = max((_iou(rb, ra) for ra in regions), default=0)
                if max_iou < 0.3:
                    regions.append(rb)
                    n_added += 1
            if n_added:
                print(f"  [DBSCAN-TrackB] added {n_added} new regions from {len(short_line_mids)} short lines")

    # 6b. 大 region 警告
    for r in regions:
        area = r['w'] * r['h']
        if area > 5000:
            print(f"  [DBSCAN-WARN] large region: bbox=({r['x']:.0f},{r['y']:.0f},"
                  f"{r['w']:.0f}x{r['h']:.0f}) area={area:.0f}")

    # 6c. 多行 / 横向大间隙 region 二次切分（codex 排单 #4）
    # 对 h > 30 的 region 用内部 tiny_curve 的 1D y 投影找空白带切分
    regions = _split_multiline_regions(regions, tiny_curves)
    regions = _split_wide_regions(regions, tiny_curves)
    if enable_numeric_symbol_seeds or plus_minus_shadow_sink is not None:
        regions = _add_attached_numeric_symbol_seed_regions(
            page,
            regions,
            enable_decimal_symbol_seeds=enable_numeric_symbol_seeds,
            enable_plus_minus_symbol_seeds=(
                enable_numeric_symbol_seeds and enable_plus_minus_symbol_seeds
            ),
            plus_minus_shadow_sink=plus_minus_shadow_sink,
        )

    # 7. Debug 可视化
    if debug_plot:
        _save_debug_image(page, regions, debug_output_path)

    return regions


def _region_xyxy(region):
    return (
        float(region.get('x', 0.0)),
        float(region.get('y', 0.0)),
        float(region.get('x', 0.0)) + float(region.get('w', 0.0)),
        float(region.get('y', 0.0)) + float(region.get('h', 0.0)),
    )


def _bbox_gap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _expanded_region_from_bbox(bbox, *, subtype, min_size=10.0, margin=2.0,
                               curve_count=1):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    width = max((x1 - x0) + 2.0 * margin, min_size)
    height = max((y1 - y0) + 2.0 * margin, min_size)
    return {
        'x': cx - width / 2.0,
        'y': cy - height / 2.0,
        'w': width,
        'h': height,
        'curve_count': curve_count,
        'avg_curve_size': 0,
        'confidence': 'high',
        'source': 'numeric_symbol_seed',
        'subtype': subtype,
        'raw_symbol_bbox': {
            'x': x0,
            'y': y0,
            'w': x1 - x0,
            'h': y1 - y0,
        },
    }


def _axis_overlap_amount(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _expanded_xyxy(bbox, margin):
    return (
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin,
    )


class _RegionBBoxSpatialIndex:
    """Conservative page-scoped broad phase for region bbox lookups."""

    def __init__(self, regions, *, cell_size=64.0, max_cells_per_bbox=256):
        self._cell_size = float(cell_size)
        self._max_cells_per_bbox = int(max_cells_per_bbox)
        self._count = len(regions)
        self._cells = defaultdict(list)
        self._overflow = []
        for idx, region in enumerate(regions):
            bbox = _region_xyxy(region)
            cell_range = self._cell_range(bbox)
            if cell_range is None:
                self._overflow.append(idx)
                continue
            x0, y0, x1, y1 = cell_range
            for cell_x in range(x0, x1 + 1):
                for cell_y in range(y0, y1 + 1):
                    self._cells[(cell_x, cell_y)].append(idx)

    def _cell_range(self, bbox):
        x0, y0, x1, y1 = bbox
        if (
            not all(math.isfinite(value) for value in bbox)
            or x1 < x0
            or y1 < y0
        ):
            return None
        cell_x0 = math.floor(x0 / self._cell_size)
        cell_y0 = math.floor(y0 / self._cell_size)
        cell_x1 = math.floor(x1 / self._cell_size)
        cell_y1 = math.floor(y1 / self._cell_size)
        cell_count = (cell_x1 - cell_x0 + 1) * (cell_y1 - cell_y0 + 1)
        if cell_count > self._max_cells_per_bbox:
            return None
        return cell_x0, cell_y0, cell_x1, cell_y1

    def query(self, bbox):
        cell_range = self._cell_range(bbox)
        if cell_range is None:
            return range(self._count)
        x0, y0, x1, y1 = cell_range
        candidates = set(self._overflow)
        for cell_x in range(x0, x1 + 1):
            for cell_y in range(y0, y1 + 1):
                candidates.update(self._cells.get((cell_x, cell_y), ()))
        return sorted(candidates)


def _nearest_text_anchor_index(
    candidate,
    regions,
    max_gap=18.0,
    overlap_margin=6.0,
    *,
    spatial_index=None,
):
    cb = _region_xyxy(candidate)
    best_idx = None
    best_score = float('inf')
    ecb = _expanded_xyxy(cb, overlap_margin)
    region_indices = (
        spatial_index.query(_expanded_xyxy(cb, max_gap))
        if spatial_index is not None
        else range(len(regions))
    )
    for idx in region_indices:
        region = regions[idx]
        if region.get('source') == 'numeric_symbol_seed':
            continue
        width = float(region.get('w') or 0.0)
        height = float(region.get('h') or 0.0)
        area = width * height
        # 小点/单短线自身不能当 anchor；需要贴到已有文字主体上。
        if area < 80.0 or (width < 8.0 and height < 8.0):
            continue
        rb = _region_xyxy(region)
        gap = _bbox_gap(cb, rb)
        if gap > max_gap:
            continue
        # Require same row/column neighborhood. Pure diagonal proximity is too
        # weak for tiny punctuation-like seeds.
        if (
            _axis_overlap_amount(ecb[0], ecb[2], rb[0], rb[2]) <= 0.0
            and _axis_overlap_amount(ecb[1], ecb[3], rb[1], rb[3]) <= 0.0
        ):
            continue
        score = gap + abs((cb[1] + cb[3]) / 2.0 - (rb[1] + rb[3]) / 2.0) * 0.05
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _covering_useful_region_index(candidate, regions, *, spatial_index=None):
    cb = _region_xyxy(candidate)
    cx = (cb[0] + cb[2]) / 2.0
    cy = (cb[1] + cb[3]) / 2.0
    region_indices = (
        spatial_index.query((cx - 3.0, cy - 3.0, cx + 3.0, cy + 3.0))
        if spatial_index is not None
        else range(len(regions))
    )
    for idx in region_indices:
        region = regions[idx]
        if region.get('source') == 'numeric_symbol_seed':
            continue
        width = float(region.get('w') or 0.0)
        height = float(region.get('h') or 0.0)
        area = width * height
        if area < 80.0:
            continue
        rb = _region_xyxy(region)
        if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
            return idx
        inflated = _expanded_xyxy(rb, 3.0)
        if inflated[0] <= cx <= inflated[2] and inflated[1] <= cy <= inflated[3]:
            return idx
    return None


def _covered_by_useful_region(candidate, regions, *, spatial_index=None):
    return _covering_useful_region_index(
        candidate,
        regions,
        spatial_index=spatial_index,
    ) is not None


def _line_xyxy(line):
    x0 = float(line.get('x0', 0.0))
    x1 = float(line.get('x1', x0))
    # pdfplumber y0/y1 are bottom-origin; text regions use top-origin.
    # Prefer top/bottom here so numeric symbol seeds share the same coordinate
    # system as DBSCAN regions.
    y0 = float(line.get('top', line.get('y0', 0.0)))
    y1 = float(line.get('bottom', line.get('y1', y0)))
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _angle_delta_deg(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _short_line_angle_deg(line):
    if len(line) > 7:
        return float(line[7])
    x0, y0, x1, y1 = line[:4]
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0


def _looks_like_plus_minus(group):
    if len(group) != 3:
        return False
    angles = [_short_line_angle_deg(line) for line in group]
    centers = [(line[4], line[5]) for line in group]
    lengths = [float(line[6]) if len(line) > 6 else math.hypot(line[2] - line[0], line[3] - line[1]) for line in group]
    if min(lengths) < 3.0 or max(lengths) > 12.0:
        return False
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            pair_delta = _angle_delta_deg(angles[i], angles[j])
            if pair_delta > 25.0:
                continue
            cross_delta = min(_angle_delta_deg(angles[i], angles[k]),
                              _angle_delta_deg(angles[j], angles[k]))
            pair_len = (lengths[i] + lengths[j]) / 2.0
            if pair_len <= 0:
                continue
            if abs(lengths[i] - lengths[j]) / pair_len > 0.35:
                continue
            if cross_delta < 55.0:
                if not (pair_len * 1.18 <= lengths[k] <= pair_len * 2.2):
                    continue
            radians = math.radians(angles[i])
            ux, uy = math.cos(radians), math.sin(radians)
            nx, ny = -uy, ux
            q1 = centers[i][0] * nx + centers[i][1] * ny
            q2 = centers[j][0] * nx + centers[j][1] * ny
            q3 = centers[k][0] * nx + centers[k][1] * ny
            p1 = centers[i][0] * ux + centers[i][1] * uy
            p2 = centers[j][0] * ux + centers[j][1] * uy
            p3 = centers[k][0] * ux + centers[k][1] * uy
            q_low, q_high = sorted((q1, q2))
            separation = q_high - q_low
            if separation < 2.0 or separation > 18.0:
                continue
            if not (q_low - 3.0 <= q3 <= q_high + 3.0):
                continue
            if abs(p3 - ((p1 + p2) / 2.0)) > max(10.0, separation * 1.5):
                continue
            return True
    return False


def _merge_seed_into_region(region, seed):
    rx0, ry0, rx1, ry1 = _region_xyxy(region)
    sx0, sy0, sx1, sy1 = _region_xyxy(seed)
    nx0, ny0, nx1, ny1 = min(rx0, sx0), min(ry0, sy0), max(rx1, sx1), max(ry1, sy1)
    region['x'] = nx0
    region['y'] = ny0
    region['w'] = nx1 - nx0
    region['h'] = ny1 - ny0
    region['curve_count'] = int(region.get('curve_count') or 0) + int(seed.get('curve_count') or 0)
    attached = list(region.get('attached_numeric_symbol_seeds') or [])
    attached_seed = {
        'subtype': seed.get('subtype'),
        'x': seed.get('x'),
        'y': seed.get('y'),
        'w': seed.get('w'),
        'h': seed.get('h'),
    }
    if isinstance(seed.get('raw_symbol_bbox'), dict):
        attached_seed['raw_symbol_bbox'] = dict(seed['raw_symbol_bbox'])
    if isinstance(seed.get('raw_symbol_segments'), list):
        attached_seed['raw_symbol_segments'] = [
            dict(row)
            for row in seed['raw_symbol_segments']
            if isinstance(row, dict)
        ]
    attached.append(attached_seed)
    region['attached_numeric_symbol_seeds'] = attached
    region['numeric_symbol_seed_count'] = len(attached)
    region['numeric_symbol_seed_subtypes'] = sorted({
        str(item.get('subtype') or '') for item in attached if item.get('subtype')
    })


def _record_numeric_seed_in_region(region, seed):
    attached = list(region.get('attached_numeric_symbol_seeds') or [])
    attached_seed = {
        'subtype': seed.get('subtype'),
        'x': seed.get('x'),
        'y': seed.get('y'),
        'w': seed.get('w'),
        'h': seed.get('h'),
        'covered_by_region': True,
    }
    if isinstance(seed.get('raw_symbol_bbox'), dict):
        attached_seed['raw_symbol_bbox'] = dict(seed['raw_symbol_bbox'])
    if isinstance(seed.get('raw_symbol_segments'), list):
        attached_seed['raw_symbol_segments'] = [
            dict(row)
            for row in seed['raw_symbol_segments']
            if isinstance(row, dict)
        ]
    attached.append(attached_seed)
    region['attached_numeric_symbol_seeds'] = attached
    region['numeric_symbol_seed_count'] = len(attached)
    region['numeric_symbol_seed_subtypes'] = sorted({
        str(item.get('subtype') or '') for item in attached if item.get('subtype')
    })


def _mark_dot_seed_only_region(seed):
    region = dict(seed)
    region['confidence'] = 'low'
    region['source'] = 'numeric_symbol_seed'
    region['role'] = 'dot_seed_only'
    region['dot_seed_only'] = True
    region['excluded_from_dbscan_merge'] = True
    region['numeric_symbol_seed_count'] = 1
    region['numeric_symbol_seed_subtypes'] = ['decimal_point_candidate']
    region.setdefault('attached_numeric_symbol_seeds', [])
    return region


def _add_attached_numeric_symbol_seed_regions(
    page,
    regions,
    *,
    enable_decimal_symbol_seeds=True,
    enable_plus_minus_symbol_seeds=True,
    plus_minus_shadow_sink=None,
):
    """Recover decimal-point and ± seeds near already detected text.

    This deliberately does not recover standalone '+' or '-' because short
    drawing strokes match those shapes too easily. The region must attach to
    an existing text region, so these seeds expand recall without becoming a
    full-page line detector.
    """
    if not regions and plus_minus_shadow_sink is None:
        return regions

    seeds = []
    orphan_dot_seeds = []
    covered_dot_count = 0
    region_spatial_index = _RegionBBoxSpatialIndex(regions)

    curves_for_seeding = (
        (getattr(page, 'curves', []) or [])
        if regions and enable_decimal_symbol_seeds
        else []
    )
    for curve in curves_for_seeding:
        # DEPRECATED: this broad curve-size decimal dot seed is legacy evidence.
        # Use decimal_point_quad.vector_decimal_point_quad_v1 for authoritative
        # decimal-point counts; keep this only as compatibility/recall bridge.
        w, h = _curve_bbox_wh(curve)
        area = w * h
        if w <= 4.5 and h <= 4.5 and area <= 20.0:
            if min(w, h) <= 0.0 or max(w, h) / min(w, h) > 1.8:
                continue
            cx, cy = _curve_center(curve)
            candidate = _expanded_region_from_bbox(
                (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
                subtype='decimal_point_candidate',
                min_size=10.0,
                margin=2.0,
                curve_count=1,
            )
            covering_idx = _covering_useful_region_index(
                candidate,
                regions,
                spatial_index=region_spatial_index,
            )
            if covering_idx is not None:
                _record_numeric_seed_in_region(regions[covering_idx], candidate)
                covered_dot_count += 1
                continue
            anchor_idx = _nearest_text_anchor_index(
                candidate,
                regions,
                max_gap=10.0,
                spatial_index=region_spatial_index,
            )
            if anchor_idx is not None:
                seeds.append((anchor_idx, candidate))
            else:
                orphan_dot_seeds.append(_mark_dot_seed_only_region(candidate))

    shadow_deduped_seeds = []
    if plus_minus_shadow_sink is not None:
        for anchor_idx, seed in seeds:
            if not any(
                _iou(seed, existing) > 0.70
                for _existing_idx, existing in shadow_deduped_seeds
            ):
                shadow_deduped_seeds.append((anchor_idx, seed))

    short_lines = []
    page_height = float(getattr(page, 'height', 0.0) or 0.0)
    for line in getattr(page, 'lines', []) or []:
        x0, y0, x1, y1 = _line_xyxy(line)
        raw_x0 = float(line.get('x0', x0))
        raw_x1 = float(line.get('x1', x1))
        raw_y0 = float(line.get('y0', line.get('top', y0)))
        raw_y1 = float(line.get('y1', line.get('bottom', y1)))
        if page_height > 0 and ('y0' in line or 'y1' in line):
            raw_y0 = page_height - raw_y0
            raw_y1 = page_height - raw_y1
        length = math.hypot(raw_x1 - raw_x0, raw_y1 - raw_y0)
        if 2.0 <= length <= 18.0:
            angle = math.degrees(math.atan2(raw_y1 - raw_y0, raw_x1 - raw_x0)) % 180.0
            short_lines.append((x0, y0, x1, y1, (x0 + x1) / 2.0, (y0 + y1) / 2.0, length, angle))

    if len(short_lines) >= 3:
        centers = np.array([(row[4], row[5]) for row in short_lines])
        labels = DBSCAN(eps=10.0, min_samples=1).fit(centers).labels_
        grouped = defaultdict(list)
        for line, label in zip(short_lines, labels):
            grouped[int(label)].append(line)

        for raw_group in grouped.values():
            # 当前只补三笔画 ±，不补单独 + / -，也不把更复杂的碎线簇当符号。
            if len(raw_group) < 3 or len(raw_group) > 8:
                continue
            candidate_groups = [raw_group] if len(raw_group) == 3 else combinations(raw_group, 3)
            for group in candidate_groups:
                group = list(group)
                if not _looks_like_plus_minus(group):
                    continue
                xs = [line[0] for line in group] + [line[2] for line in group]
                ys = [line[1] for line in group] + [line[3] for line in group]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < 4.0 or height < 4.0:
                    continue
                if max(width, height) > 22.0 or width * height > 360.0:
                    continue
                candidate = _expanded_region_from_bbox(
                    bbox,
                    subtype='plus_minus_candidate',
                    min_size=10.0,
                    margin=2.0,
                    curve_count=len(group),
                )
                candidate['raw_symbol_segments'] = sorted(
                    [
                        {
                            'x0': float(line[0]),
                            'y0': float(line[1]),
                            'x1': float(line[2]),
                            'y1': float(line[3]),
                        }
                        for line in group
                    ],
                    key=lambda row: (
                        row['x0'],
                        row['y0'],
                        row['x1'],
                        row['y1'],
                    ),
                )
                shadow_row = None
                if plus_minus_shadow_sink is not None:
                    shadow_row = {
                        'schema_version': 'legacy_plus_minus_full_page_shadow_v1',
                        'candidate_index': len(plus_minus_shadow_sink),
                        'detector': 'pdf_analyzer_cluster._looks_like_plus_minus',
                        'raw_symbol_bbox': dict(candidate['raw_symbol_bbox']),
                        'raw_symbol_segments': [
                            dict(row) for row in candidate['raw_symbol_segments']
                        ],
                        'consumer_allowed': False,
                        'seed_emitted': False,
                        'region_expanded': False,
                        'legacy_seed_eligible': False,
                        'legacy_seed_rejection_reason': None,
                    }
                    plus_minus_shadow_sink.append(shadow_row)
                covered_by_region = _covered_by_useful_region(
                    candidate,
                    regions,
                    spatial_index=region_spatial_index,
                )
                if covered_by_region:
                    if shadow_row is not None:
                        shadow_row['legacy_seed_rejection_reason'] = (
                            'covered_by_text_region'
                        )
                    continue
                anchor_idx = _nearest_text_anchor_index(
                    candidate,
                    regions,
                    spatial_index=region_spatial_index,
                )
                if anchor_idx is None:
                    if shadow_row is not None:
                        shadow_row['legacy_seed_rejection_reason'] = (
                            'no_text_anchor'
                        )
                    continue
                if shadow_row is not None:
                    duplicate = any(
                        _iou(candidate, existing) > 0.70
                        for _existing_idx, existing in shadow_deduped_seeds
                    )
                    if duplicate:
                        shadow_row['legacy_seed_rejection_reason'] = (
                            'dedup_iou'
                        )
                    else:
                        shadow_row['legacy_seed_eligible'] = True
                        shadow_deduped_seeds.append((anchor_idx, candidate))
                if not enable_plus_minus_symbol_seeds:
                    continue
                seeds.append((anchor_idx, candidate))

    if not seeds and not orphan_dot_seeds and covered_dot_count <= 0:
        return regions

    deduped = []
    for anchor_idx, seed in seeds:
        duplicate = False
        for existing_idx, existing in deduped:
            if _iou(seed, existing) > 0.70:
                duplicate = True
                break
        if not duplicate:
            deduped.append((anchor_idx, seed))

    counts = Counter()
    merged = [dict(region) for region in regions]
    for anchor_idx, seed in deduped:
        _merge_seed_into_region(merged[anchor_idx], seed)
        counts[str(seed.get('subtype') or 'unknown')] += 1
    for seed in orphan_dot_seeds:
        duplicate = False
        for region in merged:
            if _iou(seed, region) > 0.70:
                duplicate = True
                break
        if not duplicate:
            merged.append(seed)
            counts['decimal_point_orphan'] += 1
    merged.sort(key=lambda r: (round(r['y'] / 5) * 5, r['x']))
    if covered_dot_count:
        counts['decimal_point_covered'] += covered_dot_count
    if deduped or orphan_dot_seeds or covered_dot_count:
        detail = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"  [DBSCAN-NumericSeed] merged {len(deduped)} attached decimal/± seeds ({detail})")
    return merged


_REGION_MERGE_CELL_WIDTH = 64.0
_REGION_MERGE_CELL_HEIGHT = 128.0


def _iter_nearby_small_region_pairs(regions: list, small_idx: list):
    """Yield the only small-region pairs that can satisfy the merge gates.

    For well-formed regions (``0 <= w,h < 20``), the existing merge rules imply
    strict centre-distance bounds of less than 50pt horizontally and 95pt
    vertically.  A 64x128pt grid therefore provides a conservative broad phase:
    every mergeable pair is in the same or an immediately adjacent cell.

    Candidate positions are sorted before yielding so the legacy all-pairs order
    is retained.  Malformed/non-finite regions use the full-pair fallback rather
    than changing their historical behaviour.
    """
    cells = defaultdict(list)
    positions = {}
    fallback_positions = set()

    for position, region_idx in enumerate(small_idx):
        region = regions[region_idx]
        width = region['w']
        height = region['h']
        center_x = region['x'] + width / 2
        center_y = region['y'] + height / 2
        if (
            0 <= width < 20
            and 0 <= height < 20
            and math.isfinite(center_x)
            and math.isfinite(center_y)
        ):
            cell = (
                math.floor(center_x / _REGION_MERGE_CELL_WIDTH),
                math.floor(center_y / _REGION_MERGE_CELL_HEIGHT),
            )
            cells[cell].append(position)
            positions[position] = cell
        else:
            fallback_positions.add(position)

    for position, region_idx in enumerate(small_idx):
        if position in fallback_positions:
            candidate_positions = range(position + 1, len(small_idx))
        else:
            cell_x, cell_y = positions[position]
            candidates = {
                candidate_position
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for candidate_position in cells.get((cell_x + dx, cell_y + dy), ())
                if candidate_position > position
            }
            candidates.update(
                candidate_position
                for candidate_position in fallback_positions
                if candidate_position > position
            )
            candidate_positions = sorted(candidates)

        for candidate_position in candidate_positions:
            yield region_idx, small_idx[candidate_position]


def _merge_adjacent_regions(regions: list) -> list:
    """
    将相邻的小 region 合并成词级别的文字块。
    只对 w < 20 且 h < 20 的小 region 做合并，大 region 保持不变。

    合并条件（水平）：
      - y 中心差 < max(两者高度) × 0.6
      - 水平间距 (B.x - (A.x + A.w)) < max(两者高度) × 1.5，且 > 0

    合并条件（竖直）：
      - x 中心差 < max(两者宽度) × 1.2
      - 竖直间距 (B.y - (A.y + A.h)) < max(两者宽度) × 1.5，且 > 0

    使用并查集（Union-Find）管理连通分量，最终对每个分量计算包围盒。
    """
    n = len(regions)
    if n == 0:
        return regions

    # 分离小 region 和大 region；声明为 seed-only 的小点不参与词级合并。
    small_idx = [
        i for i, r in enumerate(regions)
        if not r.get('excluded_from_dbscan_merge') and r['w'] < 20 and r['h'] < 20
    ]
    large_idx = [
        i for i, r in enumerate(regions)
        if r.get('excluded_from_dbscan_merge') or not (r['w'] < 20 and r['h'] < 20)
    ]

    if not small_idx:
        return regions

    # 并查集
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 空间粗筛后仍使用原有条件精筛；不改变阈值、并查集或输出顺序。
    current_i = None
    for i, j in _iter_nearby_small_region_pairs(regions, small_idx):
        if i != current_i:
            current_i = i
            ri = regions[i]
            ci_x = ri['x'] + ri['w'] / 2
            ci_y = ri['y'] + ri['h'] / 2
            # 各向异性：高宽比 > 2.5 的竖直文字 region，y 方向阈值放大 2.5 倍
            ri_vertical = (ri['h'] / max(ri['w'], 0.01)) > 2.5

        rj = regions[j]
        cj_x = rj['x'] + rj['w'] / 2
        cj_y = rj['y'] + rj['h'] / 2
        rj_vertical = (rj['h'] / max(rj['w'], 0.01)) > 2.5

        max_h = max(ri['h'], rj['h'])
        max_w = max(ri['w'], rj['w'])

        # 水平合并：确保 j 在 i 右边（或交换方向）
        for (left, right) in [(ri, rj), (rj, ri)]:
            h_gap = right['x'] - (left['x'] + left['w'])
            if 0 < h_gap < max_h * 1.5:
                y_diff = abs(ci_y - cj_y)
                if y_diff < max_h * 0.6:
                    union(i, j)
                    break

        if find(i) == find(j):
            continue

        # 竖直合并：确保 j 在 i 下边（或交换方向）
        # 各向异性：两个 region 任一为竖直文字时，y 方向阈值 × 2.5
        v_multiplier = 2.5 if (ri_vertical or rj_vertical) else 1.0
        for (top, bottom) in [(ri, rj), (rj, ri)]:
            v_gap = bottom['y'] - (top['y'] + top['h'])
            if 0 < v_gap < max_w * 1.5 * v_multiplier:
                x_diff = abs(ci_x - cj_x)
                if x_diff < max_w * 1.2:
                    union(i, j)
                    break

    # 按连通分量分组
    groups = defaultdict(list)
    for i in small_idx:
        groups[find(i)].append(i)

    # 对每个分量计算合并后的包围盒
    merged_small = []
    for root, members in groups.items():
        if len(members) == 1:
            merged_small.append(regions[members[0]])
            continue

        sub = [regions[i] for i in members]
        x0 = min(r['x'] for r in sub)
        y0 = min(r['y'] for r in sub)
        x1 = max(r['x'] + r['w'] for r in sub)
        y1 = max(r['y'] + r['h'] for r in sub)

        total_curves = sum(r['curve_count'] for r in sub)
        total_area = sum(r['curve_count'] * r['avg_curve_size'] for r in sub)
        avg_size = total_area / total_curves if total_curves > 0 else 0

        conf_rank = {'high': 1, 'low': 0}
        best_conf = max(sub, key=lambda r: conf_rank.get(r['confidence'], 0))['confidence']

        merged_small.append({
            'x': x0, 'y': y0,
            'w': x1 - x0, 'h': y1 - y0,
            'curve_count': total_curves,
            'avg_curve_size': round(avg_size, 2),
            'confidence': best_conf,
        })

    merged = merged_small + [regions[i] for i in large_idx]
    merged.sort(key=lambda r: (round(r['y'] / 5) * 5, r['x']))

    print(f"[merge] 合并前: {len(regions)} regions → 合并后: {len(merged)} regions")
    return merged


def _split_multiline_regions(regions, tiny_curves, min_height=30, gap_factor=1.5):
    """对 h > min_height 的 region 用内部 curve 1D y 投影找空白带做二次切分。

    局部几何启发式：
      - 候选门槛: region.h > min_height（默认 30pt）
      - 切分阈值: 自适应 = median(region 内 curve_h) × gap_factor
      - 子 region h < 5pt 丢弃（噪点）
      - 切分后 < 2 个子 region 等价没切，保留原 region

    track_b 来源的 region（短线段中点合成，无真实 tiny_curve 对应）跳过。
    """
    if not tiny_curves:
        return regions

    curve_meta = []
    for c in tiny_curves:
        cx, cy = _curve_center(c)
        cw, ch = _curve_bbox_wh(c)
        curve_meta.append((cx, cy, cw, ch))

    out = []
    split_count = 0
    for r in regions:
        if r.get('source') == 'track_b' or r['h'] <= min_height:
            out.append(r); continue
        rx0, ry0 = r['x'], r['y']
        rx1, ry1 = rx0 + r['w'], ry0 + r['h']
        in_curves = [m for m in curve_meta
                     if rx0 <= m[0] <= rx1 and ry0 <= m[1] <= ry1]
        if len(in_curves) < 4:
            out.append(r); continue

        in_sorted = sorted(in_curves, key=lambda t: t[1])
        heights = [t[3] for t in in_sorted if t[3] > 0]
        if not heights:
            out.append(r); continue
        median_h = statistics.median(heights)
        threshold = median_h * gap_factor

        groups = [[in_sorted[0]]]
        for i in range(1, len(in_sorted)):
            gap = in_sorted[i][1] - in_sorted[i - 1][1]
            if gap > threshold:
                groups.append([])
            groups[-1].append(in_sorted[i])

        if len(groups) <= 1:
            out.append(r); continue

        new_subs = []
        for grp in groups:
            xs = [t[0] - t[2] / 2 for t in grp] + [t[0] + t[2] / 2 for t in grp]
            ys = [t[1] - t[3] / 2 for t in grp] + [t[1] + t[3] / 2 for t in grp]
            sx = min(xs) - 2
            sy = min(ys) - 2
            sw = max(xs) - min(xs) + 4
            sh = max(ys) - min(ys) + 4
            if sh < 5:
                continue
            new_subs.append({
                'x': sx, 'y': sy, 'w': sw, 'h': sh,
                'curve_count': len(grp),
                'avg_curve_size': r.get('avg_curve_size', 0),
                'confidence': r.get('confidence', 'high'),
                'source': 'split_multiline',
            })

        if len(new_subs) < 2:
            out.append(r)
        else:
            split_count += 1
            out.extend(new_subs)

    if split_count:
        sub_total = sum(1 for r in out if r.get('source') == 'split_multiline')
        print(f"[split] {split_count} multi-line regions split into {sub_total} sub-regions")
    return out


def _split_wide_regions(regions, tiny_curves, min_width=120, max_height=35,
                        gap_factor=4.0):
    """Split very wide one-line regions when internal curves form separated islands.

    The gap threshold is intentionally large: normal spaces inside one
    dimension should remain intact, while two separate callouts accidentally
    clustered into one ROI become separate OCR crops.
    """
    if not tiny_curves:
        return regions

    curve_meta = []
    for c in tiny_curves:
        cx, cy = _curve_center(c)
        cw, ch = _curve_bbox_wh(c)
        curve_meta.append((cx, cy, cw, ch))

    out = []
    split_count = 0
    for r in regions:
        if (r.get('source') == 'track_b' or
                r['w'] <= min_width or r['h'] > max_height):
            out.append(r); continue

        rx0, ry0 = r['x'], r['y']
        rx1, ry1 = rx0 + r['w'], ry0 + r['h']
        in_curves = [m for m in curve_meta
                     if rx0 <= m[0] <= rx1 and ry0 <= m[1] <= ry1]
        if len(in_curves) < 6:
            out.append(r); continue

        widths = [m[2] for m in in_curves if m[2] > 0]
        if not widths:
            out.append(r); continue

        median_w = statistics.median(widths)
        threshold = max(median_w * gap_factor, r['h'] * 1.8, 18.0)
        in_sorted = sorted(in_curves, key=lambda t: t[0])
        groups = [[in_sorted[0]]]
        for i in range(1, len(in_sorted)):
            gap = in_sorted[i][0] - in_sorted[i - 1][0]
            if gap > threshold:
                groups.append([])
            groups[-1].append(in_sorted[i])

        if len(groups) <= 1:
            out.append(r); continue

        new_subs = []
        for grp in groups:
            if len(grp) < 2:
                continue
            xs = [t[0] - t[2] / 2 for t in grp] + [t[0] + t[2] / 2 for t in grp]
            ys = [t[1] - t[3] / 2 for t in grp] + [t[1] + t[3] / 2 for t in grp]
            sx = min(xs) - 2
            sy = min(ys) - 2
            sw = max(xs) - min(xs) + 4
            sh = max(ys) - min(ys) + 4
            if sw < 5 or sh < 5:
                continue
            new_subs.append({
                'x': sx, 'y': sy, 'w': sw, 'h': sh,
                'curve_count': len(grp),
                'avg_curve_size': r.get('avg_curve_size', 0),
                'confidence': r.get('confidence', 'high'),
                'source': 'split_wide',
            })

        if len(new_subs) < 2:
            out.append(r)
        else:
            split_count += 1
            out.extend(new_subs)

    if split_count:
        sub_total = sum(1 for r in out if r.get('source') == 'split_wide')
        print(f"[split_wide] {split_count} wide regions split into {sub_total} sub-regions")
    return out


def _save_debug_image(plumber_page, regions, output_path):
    """用 PyMuPDF 渲染页面，用 Pillow 绘制文字区域包围盒，保存为 JPEG。"""
    try:
        fitz_page = _get_fitz_page(plumber_page)
        if fitz_page is None:
            print("[warn] Cannot render debug image: fitz page unavailable.")
            return

        DPI = 150
        scale = DPI / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        draw = ImageDraw.Draw(img)
        for r in regions:
            # 将 PDF 坐标（72 DPI）转换为图像像素坐标
            px = r['x'] * scale
            py = r['y'] * scale
            px1 = (r['x'] + r['w']) * scale
            py1 = (r['y'] + r['h']) * scale
            color = 'red' if r['confidence'] == 'high' else 'orange'
            draw.rectangle([px, py, px1, py1], outline=color, width=2)

        img.save(output_path, 'JPEG', quality=85)
        print(f"debug_task2.jpg saved to {output_path}")
    except Exception as e:
        print(f"[warn] Debug image save failed: {e}")


def extract_rects(page):
    """
    从 pdfplumber page.rects 提取并分类矩形。

    返回:
    {
      'arrow_candidates': [...],   # max(w,h) < 3
      'gdt_frames': [...],         # 小矩形水平排列组合
      'datum_candidates': [...],   # 独立小方形
      'table_cells': [...],        # 中等矩形
      'frame_borders': [...],      # 大矩形（图框）
    }
    """
    rects_raw = page.rects
    # 也检查 pdfplumber 的 rect_edges（某些版本）
    categorized = {
        'arrow_candidates': [],
        'gdt_frames': [],
        'datum_candidates': [],
        'table_cells': [],
        'frame_borders': [],
        '_small_rects': [],  # 临时存小矩形，用于 GD&T 检测
    }

    for r in rects_raw:
        x0 = r.get('x0', 0)
        y0 = r.get('top', 0)
        x1 = r.get('x1', x0)
        y1 = r.get('bottom', y0)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        max_dim = max(w, h)
        aspect = max(w, h) / max(min(w, h), 0.01)

        rect_info = {
            'x': x0, 'y': y0, 'w': w, 'h': h,
            'x1': x1, 'y1': y1,
        }

        if max_dim < 3:
            categorized['arrow_candidates'].append(rect_info)
        elif max_dim < 20:
            categorized['_small_rects'].append(rect_info)
        elif max_dim < 200:
            categorized['table_cells'].append(rect_info)
        else:
            categorized['frame_borders'].append(rect_info)

    # 检测水平排列的小矩形 → GD&T 控制框
    small = categorized.pop('_small_rects', [])
    gdt_groups, datum_candidates = _detect_gdt_groups(small)
    categorized['gdt_frames'] = gdt_groups
    categorized['datum_candidates'] = datum_candidates

    # 表格区域检测（供 OCR 跳过使用）
    categorized['table_regions'] = _detect_table_regions(page)

    # Fallback: 当 page.rects 中没有大矩形时，从线段中推断图框边界
    if not categorized['frame_borders']:
        page_w, page_h = float(page.width), float(page.height)
        min_len_x = page_w * 0.7  # 水平长线 > 页面宽度 70%
        min_len_y = page_h * 0.7  # 竖直长线 > 页面高度 70%

        # 收集候选长线段
        h_lines = []  # 水平线 (近乎水平，角度 < 5°)
        v_lines = []  # 竖直线 (近乎竖直，角度 > 85°)

        for ln in page.lines:
            x0, y0 = ln.get('x0', 0), ln.get('top', 0)
            x1, y1 = ln.get('x1', 0), ln.get('bottom', 0)
            dx, dy = abs(x1 - x0), abs(y1 - y0)

            if dx > min_len_x and dy < dx * 0.1:  # 水平
                h_lines.append({'pos': (y0 + y1) / 2, 'x0': min(x0, x1), 'x1': max(x0, x1)})
            elif dy > min_len_y and dx < dy * 0.1:  # 竖直
                v_lines.append({'pos': (x0 + x1) / 2, 'y0': min(y0, y1), 'y1': max(y0, y1)})

        if h_lines and v_lines:
            # 取最靠近页面四边的线
            top_line    = min(h_lines, key=lambda l: l['pos'])
            bottom_line = max(h_lines, key=lambda l: l['pos'])
            left_line   = min(v_lines, key=lambda l: l['pos'])
            right_line  = max(v_lines, key=lambda l: l['pos'])

            fx = left_line['pos']
            fy = top_line['pos']
            fw = right_line['pos'] - fx
            fh = bottom_line['pos'] - fy

            # 合理性检查：合成框面积 > 页面面积 50%
            if fw > 0 and fh > 0 and (fw * fh) > (page_w * page_h * 0.5):
                categorized['frame_borders'].append({
                    'x': fx, 'y': fy, 'w': fw, 'h': fh,
                    'x1': fx + fw, 'y1': fy + fh,
                    'source': 'line_fallback',
                })
                print(f"[rects] frame_border fallback from lines: "
                      f"({fx:.0f}, {fy:.0f}) {fw:.0f}x{fh:.0f}")

    # 标题栏检测：工业图纸标题栏位于图框右下角
    if categorized['frame_borders']:
        fb = categorized['frame_borders'][0]
        fb_x, fb_y, fb_w, fb_h = fb['x'], fb['y'], fb['w'], fb['h']
        # 标题栏候选区域：右 35%、下 25%
        tb_x_min = fb_x + fb_w * 0.65
        tb_y_min = fb_y + fb_h * 0.75
        tb_x_max = fb_x + fb_w
        tb_y_max = fb_y + fb_h

        # 在候选区域内查找线段
        tb_line_count = 0
        for ln in page.lines:
            lx0, ly0 = ln.get('x0', 0), ln.get('top', 0)
            lx1, ly1 = ln.get('x1', 0), ln.get('bottom', 0)
            mid_x = (lx0 + lx1) / 2
            mid_y = (ly0 + ly1) / 2
            if tb_x_min <= mid_x <= tb_x_max and tb_y_min <= mid_y <= tb_y_max:
                dx, dy = abs(lx1 - lx0), abs(ly1 - ly0)
                if (dx > 5 and dy < dx * 0.2) or (dy > 5 and dx < dy * 0.2):
                    tb_line_count += 1

        if tb_line_count >= 3:
            tb_w = tb_x_max - tb_x_min
            tb_h = tb_y_max - tb_y_min
            categorized['frame_borders'].append({
                'x': tb_x_min, 'y': tb_y_min, 'w': tb_w, 'h': tb_h,
                'x1': tb_x_max, 'y1': tb_y_max,
                'source': 'title_block',
            })
            print(f"[rects] title_block detected: "
                  f"({tb_x_min:.0f}, {tb_y_min:.0f}) {tb_w:.0f}x{tb_h:.0f}, "
                  f"{tb_line_count} lines inside")

    return categorized


def _detect_gdt_groups(small_rects):
    """
    检测水平紧密排列的小矩形组（间距 < 2px），合并为 GD&T 控制框。
    其余独立的接近正方形小矩形 → 基准符号框候选。
    """
    if not small_rects:
        return [], []

    # 按 y 坐标分行（y 差距 < 5 视为同行）
    rows = defaultdict(list)
    for r in small_rects:
        row_key = round(r['y'] / 3)
        rows[row_key].append(r)

    gdt_frames = []
    datum_candidates = []

    for row_key, rects_in_row in rows.items():
        if len(rects_in_row) < 2:
            r = rects_in_row[0]
            # 独立小矩形且接近正方形 → 基准候选
            aspect = max(r['w'], r['h']) / max(min(r['w'], r['h']), 0.01)
            if aspect <= 1.5:
                datum_candidates.append(r)
            continue

        # 按 x 排序
        sorted_row = sorted(rects_in_row, key=lambda r: r['x'])

        # 检测连续紧密排列的矩形
        group = [sorted_row[0]]
        groups = []
        for i in range(1, len(sorted_row)):
            prev = group[-1]
            curr = sorted_row[i]
            gap = curr['x'] - prev['x1']
            if gap < 2.0:
                group.append(curr)
            else:
                if len(group) >= 2:
                    groups.append(group)
                elif group:
                    # 单个矩形
                    r = group[0]
                    aspect = max(r['w'], r['h']) / max(min(r['w'], r['h']), 0.01)
                    if aspect <= 1.5:
                        datum_candidates.append(r)
                group = [curr]
        if len(group) >= 2:
            groups.append(group)
        elif group:
            r = group[0]
            aspect = max(r['w'], r['h']) / max(min(r['w'], r['h']), 0.01)
            if aspect <= 1.5:
                datum_candidates.append(r)

        for g in groups:
            # 合并包围盒
            gx = min(r['x'] for r in g)
            gy = min(r['y'] for r in g)
            gx1 = max(r['x1'] for r in g)
            gy1 = max(r['y1'] for r in g)
            gdt_frames.append({
                'x': gx, 'y': gy,
                'w': gx1 - gx, 'h': gy1 - gy,
                'rect_count': len(g),
                'type': 'gdt_frame',
            })

    return gdt_frames, datum_candidates


def _detect_table_regions(page) -> list:
    """
    检测页面中的表格区域，返回 [{'x','y','w','h'}, ...]。
    优先使用 pdfplumber 内置检测，失效时用密度 fallback。
    """
    regions = []

    # 方案 A: pdfplumber 内置表格检测
    try:
        tables = page.find_tables()
        for t in tables:
            x0, y0, x1, y1 = t.bbox
            regions.append({
                'x': x0 - 5, 'y': y0 - 5,
                'w': (x1 - x0) + 10, 'h': (y1 - y0) + 10,
            })
    except Exception as e:
        print(f"[table] pdfplumber find_tables failed: {e}")

    regions.extend(_line_grid_table_regions(page))

    # 方案 C: 如果前两种都没找到表格，用密度 fallback
    if not regions:
        regions = _density_based_table_detection(page)

    regions = _dedupe_table_regions(_filter_table_regions(regions, page))

    print(f"[table] 检测到 {len(regions)} 个表格区域")
    for i, r in enumerate(regions):
        print(f"  table[{i}]: ({r['x']:.0f}, {r['y']:.0f}) {r['w']:.0f}x{r['h']:.0f}")

    return regions


def _filter_table_regions(regions: list, page) -> list:
    """Drop drawing-frame-sized false tables before they suppress real grids."""
    page_w, page_h = float(page.width), float(page.height)
    page_area = page_w * page_h
    out = []
    for region in regions:
        try:
            x = float(region.get('x') or 0.0)
            y = float(region.get('y') or 0.0)
            w = float(region.get('w') or 0.0)
            h = float(region.get('h') or 0.0)
        except Exception:
            continue
        if w <= 0.0 or h <= 0.0:
            continue
        area = w * h
        if page_area > 0.0 and area / page_area > 0.65:
            continue
        covers_frame = (
            x <= 80.0
            and y <= 80.0
            and x + w >= page_w - 80.0
            and y + h >= page_h - 80.0
        )
        if covers_frame:
            continue
        if page_w > 0.0 and page_h > 0.0:
            if (w / page_w > 0.94 and h / page_h > 0.75) or (
                w / page_w > 0.80 and h / page_h > 0.85
            ):
                continue
        out.append(region)
    return out


def _line_grid_table_regions(page) -> list:
    page_w, page_h = float(page.width), float(page.height)
    h_lines = []
    v_lines = []
    for ln in page.lines:
        x0 = float(ln.get('x0', 0) or 0)
        x1 = float(ln.get('x1', x0) or x0)
        y0 = float(ln.get('top', ln.get('y0', 0)) or 0)
        y1 = float(ln.get('bottom', ln.get('y1', y0)) or y0)
        lx0, lx1 = sorted((x0, x1))
        ly0, ly1 = sorted((y0, y1))
        dx = lx1 - lx0
        dy = ly1 - ly0
        if dx >= 80.0 and dy <= 2.0:
            h_lines.append({'x0': lx0, 'x1': lx1, 'y': (ly0 + ly1) / 2.0})
        elif dy >= 35.0 and dx <= 2.0:
            v_lines.append({'x': (lx0 + lx1) / 2.0, 'y0': ly0, 'y1': ly1})

    if len(h_lines) < 3 or len(v_lines) < 2:
        return []

    parent = list(range(len(h_lines) + len(v_lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    tol = 3.0
    for hi, h in enumerate(h_lines):
        for vi, v in enumerate(v_lines):
            if (
                h['x0'] - tol <= v['x'] <= h['x1'] + tol
                and v['y0'] - tol <= h['y'] <= v['y1'] + tol
            ):
                union(hi, len(h_lines) + vi)

    groups = defaultdict(lambda: {'h': [], 'v': []})
    for hi, h in enumerate(h_lines):
        groups[find(hi)]['h'].append(h)
    for vi, v in enumerate(v_lines):
        groups[find(len(h_lines) + vi)]['v'].append(v)

    regions = []
    page_area = page_w * page_h
    for group in groups.values():
        hs = group['h']
        vs = group['v']
        if len(hs) < 3 or len(vs) < 2:
            continue
        x0 = min([h['x0'] for h in hs] + [v['x'] for v in vs])
        x1 = max([h['x1'] for h in hs] + [v['x'] for v in vs])
        y0 = min([h['y'] for h in hs] + [v['y0'] for v in vs])
        y1 = max([h['y'] for h in hs] + [v['y1'] for v in vs])
        w = x1 - x0
        h = y1 - y0
        area = w * h
        if area < 5000.0:
            continue
        if page_area > 0 and area / page_area > 0.65:
            continue
        regions.append({'x': x0 - 5, 'y': y0 - 5, 'w': w + 10, 'h': h + 10})
    return regions


def _dedupe_table_regions(regions: list) -> list:
    out = []
    for region in sorted(regions, key=lambda r: float(r.get('w', 0)) * float(r.get('h', 0)), reverse=True):
        if any(_table_region_contains(existing, region) or _iou(existing, region) > 0.80 for existing in out):
            continue
        out.append(region)
    return out


def _table_region_contains(large: dict, small: dict) -> bool:
    lx0, ly0 = float(large['x']), float(large['y'])
    lx1, ly1 = lx0 + float(large['w']), ly0 + float(large['h'])
    sx0, sy0 = float(small['x']), float(small['y'])
    sx1, sy1 = sx0 + float(small['w']), sy0 + float(small['h'])
    return lx0 <= sx0 and ly0 <= sy0 and lx1 >= sx1 and ly1 >= sy1


def _density_based_table_detection(page) -> list:
    """
    基于微小曲线空间密度的表格区域检测（pdfplumber fallback）。

    步骤：
    1. 将页面划分为 grid_size × grid_size 网格
    2. 统计每格内微小曲线数
    3. 密度超过全页均值 3 倍的格子标记为热区
    4. 连通域合并，面积 >= 6 格的连通域视为表格
    """
    page_w, page_h = float(page.width), float(page.height)
    grid_size = 50  # PDF 点，约 17.6 mm

    cols = max(1, int(page_w / grid_size))
    rows = max(1, int(page_h / grid_size))

    grid = [[0] * cols for _ in range(rows)]
    small_curve_count = 0

    for curve in page.curves:
        # 坐标提取：优先用 pts，fallback 用 x0/top 等字段
        pts = curve.get('pts')
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
        else:
            x0 = curve.get('x0', 0)
            x1 = curve.get('x1', x0)
            y0 = curve.get('top', curve.get('y0', 0))
            y1 = curve.get('bottom', curve.get('y1', y0))
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            bbox_w = abs(x1 - x0)
            bbox_h = abs(y1 - y0)

        if bbox_w < 15 and bbox_h < 15:
            col = min(cols - 1, max(0, int(cx / grid_size)))
            row = min(rows - 1, max(0, int(cy / grid_size)))
            grid[row][col] += 1
            small_curve_count += 1

    if small_curve_count == 0:
        return []

    avg_density = small_curve_count / (rows * cols)
    threshold = max(avg_density * 3, 5)

    hot = [[grid[r][c] > threshold for c in range(cols)] for r in range(rows)]

    visited = [[False] * cols for _ in range(rows)]
    regions = []

    for r in range(rows):
        for c in range(cols):
            if hot[r][c] and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                cells = [(r, c)]
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and hot[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            cells.append((nr, nc))

                if len(cells) >= 6:
                    min_r = min(cell[0] for cell in cells)
                    max_r = max(cell[0] for cell in cells)
                    min_c = min(cell[1] for cell in cells)
                    max_c = max(cell[1] for cell in cells)
                    regions.append({
                        'x': min_c * grid_size - 5,
                        'y': min_r * grid_size - 5,
                        'w': (max_c - min_c + 1) * grid_size + 10,
                        'h': (max_r - min_r + 1) * grid_size + 10,
                    })

    return regions
