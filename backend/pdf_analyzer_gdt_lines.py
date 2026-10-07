"""pdf_analyzer_gdt_lines.py — GD&T 特征控制框线段网格检测

从 pdf_analyzer.py 拆出。"""

import math
from collections import Counter


def _angle_mod_180(angle):
    value = float(angle) % 180.0
    if abs(value - 180.0) < 1e-6:
        return 0.0
    return value


def _angle_is_axis_aligned(angle, tolerance=5.0):
    angle = _angle_mod_180(angle)
    return (
        angle <= tolerance
        or angle >= 180.0 - tolerance
        or abs(angle - 90.0) <= tolerance
    )


def _rotate_point(x, y, angle_deg):
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return (x * c - y * s, x * s + y * c)


def _bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return {
        "x": round(x0, 2),
        "y": round(y0, 2),
        "w": round(x1 - x0, 2),
        "h": round(y1 - y0, 2),
    }


def _oriented_quad_from_axis_rect(x_left, x_right, y_top, y_bot, angle_deg):
    local_points = [
        (x_left, y_top),
        (x_right, y_top),
        (x_right, y_bot),
        (x_left, y_bot),
    ]
    return [_rotate_point(x, y, angle_deg) for x, y in local_points]


def _oriented_quad_dict(points, angle_deg, frame_width, frame_height):
    angle = _angle_mod_180(angle_deg)
    return {
        "coord_space": "page_pdf",
        "points": [[round(x, 3), round(y, 3)] for x, y in points],
        "angle_deg": round(angle, 3),
        "long_length": round(float(frame_width), 3),
        "short_length": round(float(frame_height), 3),
        "source": "gdt_frame",
    }


def _axis_compartment_to_page_bbox(compartment, angle_deg):
    points = _oriented_quad_from_axis_rect(
        compartment["x"],
        compartment["x"] + compartment["w"],
        compartment["y"],
        compartment["y"] + compartment["h"],
        angle_deg,
    )
    return _bbox_from_points(points)


def _detect_axis_gdt_frames_from_lines(anno_lines, annotation_width, *, angle_deg=0.0):
    """Detect FCF frames after the page has been rotated into a local axis."""
    h_lines = []
    v_lines = []

    for x0, y0, x1, y1 in anno_lines:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        length = math.hypot(dx, dy)

        if dy < 1.5 and length > 20:
            h_lines.append((min(x0, x1), max(x0, x1), (y0 + y1) / 2))
        elif dx < 1.5 and length > 12:
            v_lines.append(((x0 + x1) / 2, min(y0, y1), max(y0, y1)))

    if not h_lines:
        return []

    frame_height_min = 14.0
    frame_height_max = 26.0
    y_tolerance = 2.0

    sorted_lines = sorted(h_lines, key=lambda line: line[2])
    h_clusters = []
    current = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        if abs(line[2] - current[-1][2]) <= y_tolerance:
            current.append(line)
        else:
            h_clusters.append(current)
            current = [line]
    h_clusters.append(current)

    h_merged = []
    for cluster in h_clusters:
        y_avg = sum(line[2] for line in cluster) / len(cluster)
        intervals = sorted((line[0], line[1]) for line in cluster)
        merged_intervals = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged_intervals[-1][1] + 5:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
            else:
                merged_intervals.append([start, end])
        for x_left, x_right in merged_intervals:
            if x_right - x_left > 25:
                h_merged.append((x_left, x_right, y_avg))

    frame_candidates = []
    for i in range(len(h_merged)):
        for j in range(i + 1, len(h_merged)):
            h1 = h_merged[i]
            h2 = h_merged[j]
            dy = abs(h2[2] - h1[2])
            if dy < frame_height_min or dy > frame_height_max:
                continue
            overlap_left = max(h1[0], h2[0])
            overlap_right = min(h1[1], h2[1])
            if overlap_right <= overlap_left:
                continue
            if overlap_right - overlap_left < 20:
                continue
            frame_candidates.append({
                "x_left": overlap_left,
                "x_right": overlap_right,
                "y_top": min(h1[2], h2[2]),
                "y_bot": max(h1[2], h2[2]),
                "height": dy,
            })

    results = []
    for fc in frame_candidates:
        x_left = fc["x_left"]
        x_right = fc["x_right"]
        y_top = fc["y_top"]
        y_bot = fc["y_bot"]
        fh = fc["height"]

        ep_tol = max(fh * 0.20, 12.0)
        len_tol = max(fh * 0.40, 15.0)
        full_height_v = []
        for vx, vy_top, vy_bot in v_lines:
            if vx < x_left - 2 or vx > x_right + 2:
                continue
            v_len = vy_bot - vy_top
            if abs(v_len - fh) > len_tol:
                continue
            if abs(vy_top - y_top) > ep_tol and abs(vy_bot - y_top) > ep_tol:
                continue
            if abs(vy_bot - y_bot) > ep_tol and abs(vy_top - y_bot) > ep_tol:
                continue
            full_height_v.append(vx)

        full_height_v.sort()
        deduped_v = []
        for vx in full_height_v:
            if not deduped_v or abs(vx - deduped_v[-1]) > 2:
                deduped_v.append(vx)
            else:
                deduped_v[-1] = (deduped_v[-1] + vx) / 2
        if len(deduped_v) < 2:
            continue

        compartments = []
        for k in range(len(deduped_v) - 1):
            comp_x = deduped_v[k]
            comp_w = deduped_v[k + 1] - deduped_v[k]
            if comp_w < 5:
                continue
            axis_compartment = {
                "x": round(comp_x, 2),
                "y": round(y_top, 2),
                "w": round(comp_w, 2),
                "h": round(fh, 2),
            }
            if _angle_is_axis_aligned(angle_deg, tolerance=1.0):
                compartments.append(axis_compartment)
            else:
                compartments.append(_axis_compartment_to_page_bbox(
                    axis_compartment,
                    angle_deg,
                ))

        if len(compartments) < 2:
            continue
        if any(comp["w"] < 8 for comp in compartments):
            continue

        frame_width = x_right - x_left
        if frame_width < 50:
            continue
        if compartments[0]["w"] > 45 and _angle_is_axis_aligned(angle_deg, tolerance=1.0):
            continue

        confidence = 0.3
        first_width = deduped_v[1] - deduped_v[0]
        if 12 <= first_width <= 35:
            confidence += 0.3
        narrow_count = sum(
            1 for idx in range(len(deduped_v) - 1)
            if 12 <= deduped_v[idx + 1] - deduped_v[idx] <= 22
        )
        if narrow_count >= 1:
            confidence += 0.2
        if 18 <= fh <= 22:
            confidence += 0.2
        confidence = round(min(confidence, 1.0), 2)
        if confidence < 0.6:
            continue

        if _angle_is_axis_aligned(angle_deg, tolerance=1.0):
            bbox = {
                "x": round(x_left, 2),
                "y": round(y_top, 2),
                "w": round(frame_width, 2),
                "h": round(fh, 2),
            }
            frame = {
                "bbox": bbox,
                "compartments": compartments,
                "frame_height": round(fh, 2),
                "line_width": round(annotation_width, 3),
                "confidence": confidence,
            }
        else:
            quad_points = _oriented_quad_from_axis_rect(
                x_left,
                x_right,
                y_top,
                y_bot,
                angle_deg,
            )
            bbox = _bbox_from_points(quad_points)
            frame = {
                "bbox": bbox,
                "compartments": compartments,
                "frame_height": round(fh, 2),
                "line_width": round(annotation_width, 3),
                "confidence": confidence,
                "angle_deg": round(_angle_mod_180(angle_deg), 3),
                "oriented_quad": _oriented_quad_dict(
                    quad_points,
                    angle_deg,
                    frame_width,
                    fh,
                ),
                "source": "gdt_line_rotated",
            }
        results.append(frame)

    return results


def _rotated_angle_candidates(anno_lines):
    buckets = {}
    for x0, y0, x1, y1 in anno_lines:
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length < 12.0:
            continue
        angle = _angle_mod_180(math.degrees(math.atan2(dy, dx)))
        if _angle_is_axis_aligned(angle):
            continue
        bucket = round(angle / 2.0) * 2.0
        bucket = _angle_mod_180(bucket)
        stats = buckets.setdefault(bucket, {"count": 0, "weight": 0.0})
        stats["count"] += 1
        stats["weight"] += length

    candidates = []
    for angle, stats in sorted(
        buckets.items(),
        key=lambda item: (item[1]["weight"], item[1]["count"]),
        reverse=True,
    )[:10]:
        if stats["count"] < 4 and stats["weight"] < 120.0:
            continue
        for candidate in (angle, angle - 90.0):
            candidate = _angle_mod_180(candidate)
            if _angle_is_axis_aligned(candidate):
                continue
            if all(abs(candidate - existing) > 3.0 for existing in candidates):
                candidates.append(candidate)
        if len(candidates) >= 6:
            break
    return candidates[:6]


def _detect_rotated_gdt_frames(anno_lines, annotation_width):
    frames = []
    for angle in _rotated_angle_candidates(anno_lines):
        rad = math.radians(-angle)
        cosine = math.cos(rad)
        sine = math.sin(rad)
        rotated_lines = []
        for x0, y0, x1, y1 in anno_lines:
            rx0 = x0 * cosine - y0 * sine
            ry0 = x0 * sine + y0 * cosine
            rx1 = x1 * cosine - y1 * sine
            ry1 = x1 * sine + y1 * cosine
            rotated_lines.append((rx0, ry0, rx1, ry1))
        frames.extend(_detect_axis_gdt_frames_from_lines(
            rotated_lines,
            annotation_width,
            angle_deg=angle,
        ))
    return frames


def _line_records_from_drawings(drawings):
    """Flatten one drawing snapshot exactly once for all width passes."""
    all_lines = []
    for drawing in drawings:
        width = drawing.get("width", 0) or 0
        for item in drawing.get("items", []):
            if item[0] != "l":
                continue
            p0, p1 = item[1], item[2]
            all_lines.append((p0.x, p0.y, p1.x, p1.y, width))
    return tuple(all_lines)


def _bbox_iou(a, b):
    ax1 = a["x"] + a["w"]
    ay1 = a["y"] + a["h"]
    bx1 = b["x"] + b["w"]
    by1 = b["y"] + b["h"]
    ix0 = max(a["x"], b["x"])
    iy0 = max(a["y"], b["y"])
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _frame_shared_horizontal_edge(a, b, *, tolerance=1.5):
    ab = a.get("bbox") or {}
    bb = b.get("bbox") or {}
    if not ab or not bb:
        return False
    a_edges = [float(ab["y"]), float(ab["y"]) + float(ab["h"])]
    b_edges = [float(bb["y"]), float(bb["y"]) + float(bb["h"])]
    if not any(abs(a_edge - b_edge) <= tolerance for a_edge in a_edges for b_edge in b_edges):
        return False
    ax0 = float(ab["x"])
    ax1 = ax0 + float(ab["w"])
    bx0 = float(bb["x"])
    bx1 = bx0 + float(bb["w"])
    overlap = min(ax1, bx1) - max(ax0, bx0)
    if overlap <= 0:
        return False
    return overlap / max(min(ax1 - ax0, bx1 - bx0), 1e-6) >= 0.60


def _append_nonduplicate_frames(primary, secondary, *, iou_threshold=0.35, shared_edges=False):
    final = list(primary)
    for frame in sorted(secondary, key=lambda item: -item.get("confidence", 0.0)):
        if any(
            _bbox_iou(frame["bbox"], existing["bbox"]) > iou_threshold
            or (shared_edges and _frame_shared_horizontal_edge(frame, existing))
            for existing in final
        ):
            continue
        final.append(frame)
    return final


def detect_gdt_frames_from_lines(
    page,
    annotation_width=None,
    frame_borders=None,
    *,
    drawing_snapshot=None,
    enable_r23_multiline=False,
    dedupe_shared_edges=False,
    dedupe_iou_threshold=0.5,
    drop_log=None,
    _prepared_line_records=None,
):
    # ── Phase 1: 收集标注线宽的线段 ──────────────────────────────
    if _prepared_line_records is None:
        drawings = (
            tuple(drawing_snapshot)
            if drawing_snapshot is not None
            else tuple(page.get_drawings())
        )
        all_lines = _line_records_from_drawings(drawings)
    else:
        all_lines = tuple(_prepared_line_records)

    def _dedup_frames_by_iou(frames):
        """Deduplicate frames found from one or more annotation-line widths."""
        frames = sorted(frames, key=lambda r: -r["confidence"])
        final_frames = []
        for r in frames:
            is_dup = False
            for f in final_frames:
                if (
                    _bbox_iou(r["bbox"], f["bbox"]) > dedupe_iou_threshold
                    or (dedupe_shared_edges and _frame_shared_horizontal_edge(r, f))
                ):
                    is_dup = True
                    break
            if not is_dup:
                final_frames.append(r)
        return final_frames

    def _log_drop(gate, fc, **detail):
        if drop_log is None:
            return
        row = {
            "gate": str(gate),
            "bbox": {
                "x": round(float(fc.get("x_left", 0.0)), 2),
                "y": round(float(fc.get("y_top", 0.0)), 2),
                "w": round(float(fc.get("x_right", 0.0)) - float(fc.get("x_left", 0.0)), 2),
                "h": round(float(fc.get("height", 0.0)), 2),
            },
            "params": {
                "enable_r23_multiline": bool(enable_r23_multiline),
                "dedupe_shared_edges": bool(dedupe_shared_edges),
                "dedupe_iou_threshold": float(dedupe_iou_threshold),
            },
        }
        row.update(detail)
        drop_log.append(row)

    # 自动检测标注线宽（默认扫描主线宽；若存在足量次要细线宽，也单独扫描后合并）。
    if annotation_width is None:
        width_counts = Counter()
        for _, _, _, _, w in all_lines:
            if 0.1 < w < 0.6:
                bucket = round(w, 3)
                width_counts[bucket] += 1
        if width_counts:
            most_common = width_counts.most_common()
            dominant_count = most_common[0][1]
            min_secondary_count = max(500, int(dominant_count * 0.04))
            candidate_widths = []
            for width, count in most_common:
                if candidate_widths:
                    if count < min_secondary_count:
                        continue
                    # Avoid adding a thick secondary line band to thinner
                    # annotation widths; a dominant thick style remains valid.
                    if width > 0.49:
                        continue
                    if any(abs(width - existing) <= max(width, existing) * 0.15
                           for existing in candidate_widths):
                        continue
                candidate_widths.append(width)
                if len(candidate_widths) >= 3:
                    break
            if len(candidate_widths) > 1:
                combined = []
                for width in candidate_widths:
                    combined.extend(detect_gdt_frames_from_lines(
                        page,
                        annotation_width=width,
                        frame_borders=frame_borders,
                        enable_r23_multiline=enable_r23_multiline,
                        dedupe_shared_edges=dedupe_shared_edges,
                        dedupe_iou_threshold=dedupe_iou_threshold,
                        drop_log=drop_log,
                        _prepared_line_records=all_lines,
                    ))
                final = _dedup_frames_by_iou(combined)
                print(f"[detect_gdt_frames] auto annotation_widths="
                      f"{','.join(f'{w:.3f}' for w in candidate_widths)} "
                      f"combined={len(combined)}, final={len(final)}")
                return final
            annotation_width = candidate_widths[0]
        else:
            print("[detect_gdt_frames] 无法自动检测标注线宽，返回空")
            return []

    # 容差 ±15% 过滤
    w_lo = annotation_width * 0.85
    w_hi = annotation_width * 1.15
    anno_lines = [
        (x0, y0, x1, y1)
        for x0, y0, x1, y1, w in all_lines
        if w_lo <= w <= w_hi
    ]
    rotated_results = _detect_rotated_gdt_frames(anno_lines, annotation_width)

    # ── Phase 2: 分离水平线和垂直线 ─────────────────────────────
    h_lines = []  # (x_left, x_right, y)
    v_lines = []  # (x, y_top, y_bottom)

    for x0, y0, x1, y1 in anno_lines:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        length = math.hypot(dx, dy)

        if dy < 1.5 and length > 20:  # 水平线，长度 > 20pt
            h_lines.append((min(x0, x1), max(x0, x1), (y0 + y1) / 2))
        elif dx < 1.5 and length > 12:  # 垂直线，长度 > 12pt
            v_lines.append(((x0 + x1) / 2, min(y0, y1), max(y0, y1)))

    if not h_lines and not rotated_results:
        print("[detect_gdt_frames] 无水平线，返回空")
        return []
    if not h_lines:
        final = _append_nonduplicate_frames([], rotated_results)
        print(f"[gdt_filter] rotated results: {len(rotated_results)}")
        print(f"[detect_gdt_frames] annotation_width={annotation_width:.3f}, "
              f"anno_lines={len(anno_lines)}, h_merged=0, "
              f"frame_candidates=0, final={len(final)}")
        return final

    # ── Phase 3: 寻找水平线对（框的顶边和底边） ──────────────────
    FRAME_HEIGHT_MIN = 14.0
    FRAME_HEIGHT_MAX = 26.0
    Y_TOLERANCE = 2.0

    def _cluster_h_by_y(lines, tol=Y_TOLERANCE):
        """将 y 坐标接近的水平线合并为逻辑行。"""
        sorted_lines = sorted(lines, key=lambda l: l[2])
        clusters = []
        current = [sorted_lines[0]]
        for line in sorted_lines[1:]:
            if abs(line[2] - current[-1][2]) <= tol:
                current.append(line)
            else:
                clusters.append(current)
                current = [line]
        clusters.append(current)
        return clusters

    h_clusters = _cluster_h_by_y(h_lines)

    # 每个 cluster 合并重叠的 x 区间
    h_merged = []
    for cluster in h_clusters:
        y_avg = sum(l[2] for l in cluster) / len(cluster)
        intervals = sorted([(l[0], l[1]) for l in cluster])
        merged_intervals = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged_intervals[-1][1] + 5:  # 允许 5pt 间隙
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
            else:
                merged_intervals.append([start, end])
        for x_left, x_right in merged_intervals:
            if x_right - x_left > 25:  # 最小框宽
                h_merged.append((x_left, x_right, y_avg))

    # 寻找配对：两条 H 线，高度差在 [14, 26]，x 范围重叠 > 70%
    frame_candidates = []
    for i in range(len(h_merged)):
        for j in range(i + 1, len(h_merged)):
            h1 = h_merged[i]
            h2 = h_merged[j]

            y_top = min(h1[2], h2[2])
            y_bot = max(h1[2], h2[2])
            dy = y_bot - y_top
            overlap_left = max(h1[0], h2[0])
            overlap_right = min(h1[1], h2[1])
            if overlap_right <= overlap_left:
                continue

            overlap_len = overlap_right - overlap_left
            shorter_len = min(h1[1] - h1[0], h2[1] - h2[0])
            # 放宽重叠要求：交集 > 20pt 即可（原来要求 70%，复合公差框会漏检）
            if overlap_len < 20:
                continue

            internal_ys = sorted(
                line[2]
                for line in h_merged
                if y_top + Y_TOLERANCE < line[2] < y_bot - Y_TOLERANCE
                and min(line[1], overlap_right) - max(line[0], overlap_left) >= 20
            )
            row_bounds = [y_top, *internal_ys, y_bot]
            row_count = max(len(row_bounds) - 1, 1)
            row_height = dy / row_count
            if not enable_r23_multiline:
                if dy < FRAME_HEIGHT_MIN or dy > FRAME_HEIGHT_MAX:
                    continue
                row_bounds = [y_top, y_bot]
                row_count = 1
                row_height = dy
            else:
                if row_count < 1 or row_count > 3:
                    continue
                if row_height < FRAME_HEIGHT_MIN or row_height > FRAME_HEIGHT_MAX:
                    continue

            frame_candidates.append({
                "x_left": overlap_left,
                "x_right": overlap_right,
                "y_top": y_top,
                "y_bot": y_bot,
                "height": y_bot - y_top,
                "row_height": row_height,
                "row_count": row_count,
                "row_bounds": row_bounds,
            })

    # ── Phase 4: 匹配垂直分隔线 → 确认 GD&T 框 ─────────────────
    results = []
    _gf_total = len(frame_candidates)
    _gf_drop_vlines = 0
    _gf_drop_compartment_count = 0
    _gf_drop_narrow_comp = 0
    _gf_drop_short_frame = 0
    _gf_drop_wide_first_comp = 0
    _gf_drop_low_confidence = 0

    for fc in frame_candidates:
        x_left = fc["x_left"]
        x_right = fc["x_right"]
        y_top = fc["y_top"]
        y_bot = fc["y_bot"]
        fh = fc["height"]
        row_height = fc.get("row_height", fh)
        row_bounds = list(fc.get("row_bounds") or [y_top, y_bot])

        # 找全高垂直线：端点容差比例法（15% 框高，最低 3pt）
        ep_tol = max(row_height * 0.20, 12.0)
        len_tol = max(row_height * 0.40, 15.0)  # 长度容差稍宽
        matched_v = []
        for vx, vy_top, vy_bot in v_lines:
            if vx < x_left - 2 or vx > x_right + 2:
                continue
            v_len = vy_bot - vy_top
            spans = [(y_top, y_bot)]
            if enable_r23_multiline and len(row_bounds) > 2:
                spans = [
                    (row_bounds[start], row_bounds[end])
                    for start in range(len(row_bounds) - 1)
                    for end in range(start + 1, len(row_bounds))
                ]
            if not any(
                abs(v_len - (span_bot - span_top)) <= len_tol
                and not (abs(vy_top - span_top) > ep_tol and abs(vy_bot - span_top) > ep_tol)
                and not (abs(vy_bot - span_bot) > ep_tol and abs(vy_top - span_bot) > ep_tol)
                for span_top, span_bot in spans
            ):
                continue
            matched_v.append((vx, vy_top, vy_bot))

        # 去重（x 容差 2pt）
        full_height_v = sorted(vx for vx, _vy_top, _vy_bot in matched_v)
        deduped_v = []
        for vx in full_height_v:
            if not deduped_v or abs(vx - deduped_v[-1]) > 2:
                deduped_v.append(vx)
            else:
                deduped_v[-1] = (deduped_v[-1] + vx) / 2

        # 至少 2 条全高 V 线 = 至少 1 个 compartment
        if len(deduped_v) < 2:
            _gf_drop_vlines += 1
            _log_drop("full_height_v_lt_2", fc, full_height_v_count=len(deduped_v))
            continue

        # 构建 compartment
        compartments = []
        for k in range(len(deduped_v) - 1):
            comp_x = deduped_v[k]
            comp_w = deduped_v[k + 1] - deduped_v[k]
            if comp_w < 5:
                continue
            compartments.append({
                "x": round(comp_x, 2),
                "y": round(y_top, 2),
                "w": round(comp_w, 2),
                "h": round(fh, 2),
            })

        if len(compartments) < 2:
            _gf_drop_compartment_count += 1
            _log_drop("compartments_lt_2", fc, compartment_count=len(compartments), full_height_v_count=len(deduped_v))
            if _gf_drop_compartment_count <= 15:
                print(f"  [gdt_comp_kill] bbox=({x_left:.0f},{y_top:.0f},"
                      f"{x_right - x_left:.0f}x{fh:.0f}) "
                      f"compartments={len(compartments)} "
                      f"full_height_v={len(deduped_v)}")
            continue

        # 过滤假阳性：任何 compartment 宽度 < 8pt 的不是真 GD&T 框
        if any(c["w"] < 8 for c in compartments):
            _gf_drop_narrow_comp += 1
            _log_drop("narrow_comp_lt_8pt", fc, min_compartment_width=min(c["w"] for c in compartments))
            continue

        # 对过短的双格结构应用保守宽度门，减少非公差框几何的干扰。
        frame_width = x_right - x_left
        if frame_width < 50:
            _gf_drop_short_frame += 1
            _log_drop("frame_width_lt_50pt", fc, frame_width=round(frame_width, 2))
            continue

        # 第一格应是 GD&T 符号格；异常宽度可能表示检测从中间格切入。
        if compartments[0]["w"] > 45:
            _gf_drop_wide_first_comp += 1
            _log_drop("first_compartment_gt_45pt", fc, first_compartment_width=compartments[0]["w"])
            continue

        # 置信度评分
        confidence = 0.3  # 有 ≥2 compartment 的基础分
        if 12 <= compartments[0]["w"] <= 35:
            confidence += 0.3  # 第一格宽度像 ⊕ 符号格
        narrow_count = sum(1 for c in compartments if 12 <= c["w"] <= 22)
        if narrow_count >= 1:
            confidence += 0.2  # 有窄格（datum 引用）
        if 18 <= row_height <= 22:
            confidence += 0.2  # 典型框高

        confidence = round(min(confidence, 1.0), 2)
        if confidence < 0.6:
            _gf_drop_low_confidence += 1
            _log_drop("confidence_lt_0_6", fc, confidence=confidence)
            continue


        row_compartments = []
        if enable_r23_multiline and len(row_bounds) > 2:
            for row_index in range(len(row_bounds) - 1):
                row_top = row_bounds[row_index]
                row_bot = row_bounds[row_index + 1]
                row_v = sorted(
                    vx for vx, vy_top, vy_bot in matched_v
                    if vy_top <= row_top + ep_tol and vy_bot >= row_bot - ep_tol
                )
                deduped_row_v = []
                for vx in row_v:
                    if not deduped_row_v or abs(vx - deduped_row_v[-1]) > 2:
                        deduped_row_v.append(vx)
                    else:
                        deduped_row_v[-1] = (deduped_row_v[-1] + vx) / 2
                row_comps = []
                for k in range(len(deduped_row_v) - 1):
                    comp_x = deduped_row_v[k]
                    comp_w = deduped_row_v[k + 1] - deduped_row_v[k]
                    if comp_w >= 5:
                        row_comps.append({
                            "x": round(comp_x, 2),
                            "y": round(row_top, 2),
                            "w": round(comp_w, 2),
                            "h": round(row_bot - row_top, 2),
                        })
                row_compartments.append(row_comps)

        result = {
            "bbox": {
                "x": round(x_left, 2),
                "y": round(y_top, 2),
                "w": round(x_right - x_left, 2),
                "h": round(fh, 2),
            },
            "compartments": compartments,
            "frame_height": round(fh, 2),
            "line_width": round(annotation_width, 3),
            "confidence": confidence,
        }
        if enable_r23_multiline and len(row_bounds) > 2:
            result["row_count"] = len(row_bounds) - 1
            result["row_bounds"] = [round(float(value), 2) for value in row_bounds]
            result["row_compartments"] = row_compartments
            result["source"] = "gdt_line_r23_multiline"
        results.append(result)

    # 去重：IoU > 0.5 取 confidence 高的
    final = _append_nonduplicate_frames(
        _dedup_frames_by_iou(results),
        rotated_results,
        iou_threshold=0.35,
        shared_edges=dedupe_shared_edges,
    )

    print(f"[gdt_filter] frame_candidates: {_gf_total}")
    print(f"[gdt_filter] drop full_height_v < 2: {_gf_drop_vlines} → {_gf_total - _gf_drop_vlines}")
    _after_vlines = _gf_total - _gf_drop_vlines
    print(f"[gdt_filter] drop compartments < 2: {_gf_drop_compartment_count} → {_after_vlines - _gf_drop_compartment_count}")
    _after_comp = _after_vlines - _gf_drop_compartment_count
    print(f"[gdt_filter] drop narrow_comp < 8pt: {_gf_drop_narrow_comp} → {_after_comp - _gf_drop_narrow_comp}")
    _after_narrow = _after_comp - _gf_drop_narrow_comp
    print(f"[gdt_filter] drop frame_width < 50pt: {_gf_drop_short_frame} → {_after_narrow - _gf_drop_short_frame}")
    _after_short = _after_narrow - _gf_drop_short_frame
    print(f"[gdt_filter] drop first_compartment > 45pt: {_gf_drop_wide_first_comp} → {_after_short - _gf_drop_wide_first_comp}")
    _after_first = _after_short - _gf_drop_wide_first_comp
    print("[gdt_filter] title_block drop disabled for line-origin FCF")
    print(f"[gdt_filter] drop confidence < 0.6: {_gf_drop_low_confidence} → {_after_first - _gf_drop_low_confidence}")
    print(f"[gdt_filter] rotated results: {len(rotated_results)}")
    print(f"[gdt_filter] pre-dedup results: {len(results)}, post-dedup final: {len(final)}")
    print(f"[detect_gdt_frames] annotation_width={annotation_width:.3f}, "
          f"anno_lines={len(anno_lines)}, h_merged={len(h_merged)}, "
          f"frame_candidates={len(frame_candidates)}, final={len(final)}")

    return final
