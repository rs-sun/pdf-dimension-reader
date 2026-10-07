"""
assembler_loaders_dim.py — 尺寸装载器 + GD&T 链式合并

从 assembler_loaders.py 拆出。包含：
  - _load_datum_boxes()
  - _load_basic_dimensions()
  - _in_exclusion_zone()
  - _load_dimension_lines()
  - _merge_gdt_chain()
"""

import math
import re

from assembler_utils import (
    VERBOSE, METADATA_BLACKLIST_PATTERNS,
    _bbox_center, _point_in_bbox,
    _fixup_ocr_text, _extract_prefix,
    _is_gdt_fragment, _quick_dim_score,
    _parse_tolerance_inline,
)


# 装载兼容基准框。



def _datum_rect_from_reconstructed(rr: dict) -> dict | None:
    if rr.get('compartments') != 1:
        return None
    if rr.get('has_interior_h'):
        return None
    bbox = rr.get('bbox') or []
    if len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    w = float(x1) - float(x0)
    h = float(y1) - float(y0)
    if not (8.0 <= w <= 45.0 and 8.0 <= h <= 45.0):
        return None
    aspect = max(w, h) / max(min(w, h), 0.01)
    if aspect > 1.65:
        return None
    return {'x': float(x0), 'y': float(y0), 'w': w, 'h': h}


def _dedup_rect_candidates(rects: list[dict]) -> list[dict]:
    out = []
    for rect in rects:
        rcx, rcy = _bbox_center(rect)
        duplicate = False
        for prev in out:
            pcx, pcy = _bbox_center(prev)
            if (abs(rcx - pcx) <= 3.0 and abs(rcy - pcy) <= 3.0
                    and abs(rect['w'] - prev['w']) <= 4.0
                    and abs(rect['h'] - prev['h']) <= 4.0):
                duplicate = True
                break
        if not duplicate:
            out.append(rect)
    return out


def _load_datum_boxes(ocr_results: list, datum_candidates: list, out_refs: list,
                      reconstructed_rects: list | None = None):
    """
    对每个基准符号框候选，检查内部是否有单个大写字母。
    """
    boxes = [
        {'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h']}
        for b in (datum_candidates or [])
        if all(k in b for k in ('x', 'y', 'w', 'h'))
    ]
    for rr in (reconstructed_rects or []):
        rect = _datum_rect_from_reconstructed(rr)
        if rect:
            boxes.append(rect)

    for box in _dedup_rect_candidates(boxes):
        texts_inside = []
        for r in ocr_results:
            if r['_claimed']:
                continue
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            if _point_in_bbox(cx, cy, box, margin=2.0):
                texts_inside.append(r)

        for r in texts_inside:
            t = r['text'].strip()
            if re.match(r'^[A-Z]$', t):
                r['_claimed'] = True
                out_refs.append({
                    'text': t,
                    'type': 'datum_reference',
                    'bbox': {'x': box['x'], 'y': box['y'],
                             'w': box['w'], 'h': box['h']},
                    'numbering_excluded': True,
                })
                break


# 识别理论精确尺寸。



_BASIC_DIM_PLAIN_NUMERIC_RE = re.compile(
    r'^\s*(?:\d{1,2}\s*[xX×]\s*)?\d{1,4}(?:\.\d{1,4})?\s*(?:°|º)?\s*$'
)


def _is_plain_basic_dimension_text(text: str) -> bool:
    """Basic dimensions are boxed exact nominal values, not toleranced dims."""
    value = _fixup_ocr_text(str(text or '')).strip()
    if not _BASIC_DIM_PLAIN_NUMERIC_RE.fullmatch(value):
        return False
    compact = re.sub(r'\s+', '', value).replace('×', 'X').replace('x', 'X')
    compact = re.sub(r'^\d{1,2}X', '', compact)
    compact = compact.replace('º', '°').rstrip('°')
    if re.search(r'[±+\-⌀∅Ø⊘φΦRrxX×]', compact):
        return False
    return True


def _load_basic_dimensions(ocr_results: list, table_cells: list,
                           gdt_frame_candidates: list, frame_borders: list,
                           out_dims: list, reconstructed_rects: list = None,
                           out_refs: list = None):
    """检测由紧贴矩形线框包围的理论精确尺寸。候选来自表格单元格或单格重建矩形，按文字中心、面积与候选来源检查匹配。"""
    if not ocr_results:
        return

    # 排除集合仅包含形位框，页面外框不作为内容排除区。
    
    exclude_rects = []
    for gf in (gdt_frame_candidates or []):
        exclude_rects.append(gf)

    def _is_excluded(cx, cy):
        for er in exclude_rects:
            if (er['x'] <= cx <= er['x'] + er['w'] and
                    er['y'] <= cy <= er['y'] + er['h']):
                return True
        return False

    # 统一矩形候选格式并记录来源；根据来源选择小数严格性，不要求全部框值带长小数。
    
    
    
    candidate_rects = []  

    # 接纳表格单元格来源候选。
    for tc in (table_cells or []):
        tc_cx, tc_cy = _bbox_center(tc)
        if not _is_excluded(tc_cx, tc_cy):
            candidate_rects.append((tc, False))

    # 接纳重建单格细框候选。
    _BASIC_DIM_MAX_H = 60.0
    n_from_recon = 0
    for rr in (reconstructed_rects or []):
        if rr['compartments'] != 1:
            continue
        if rr['has_interior_h']:
            continue
        if rr['height'] > _BASIC_DIM_MAX_H:
            continue
        bbox_xyxy = rr['bbox']  # 矩形坐标按两角点表示。
        x0, y0, x1, y1 = bbox_xyxy
        rect = {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}
        rc_cx = (x0 + x1) / 2
        rc_cy = (y0 + y1) / 2
        if not _is_excluded(rc_cx, rc_cy):
            candidate_rects.append((rect, False))
            n_from_recon += 1

    if VERBOSE:
        n_table = len(candidate_rects) - n_from_recon
        print(f"[basic_dim] candidates: {len(candidate_rects)} "
              f"(table_cells={n_table}, reconstructed={n_from_recon}, "
              f"plain numeric only)")

    if not candidate_rects:
        return

    # 精度判定
    _THREE_DECIMAL_RE = re.compile(r'\d+\.\d{3,}')

    n_found = 0
    for r in ocr_results:
        if r.get('_claimed', False):
            continue
        bbox = r['bbox_in_pdf']
        ocr_cx, ocr_cy = _bbox_center(bbox)
        ocr_area = max(bbox['w'] * bbox['h'], 1.0)
        ocr_text = (r.get('text') or '').strip()
        is_high_precision = bool(_THREE_DECIMAL_RE.search(ocr_text))
        is_plain_numeric = _is_plain_basic_dimension_text(ocr_text)

        for rect, strict in candidate_rects:
            if strict and not is_high_precision:
                continue
            if not is_plain_numeric:
                continue
            if not _point_in_bbox(ocr_cx, ocr_cy, rect, margin=5.0):
                continue
            rect_area = rect['w'] * rect['h']
            if rect_area > ocr_area * 14:
                continue

            # paren-reference fast-skip：basic-dim 内部如果是 (N) 形态也丢弃
            # （basic_dimension 整体在 _post_filter_all 也会被 source 黑名单清掉，
            #  但提早跳过可以省 _claimed flag 的污染）
            _raw_bd = r.get('_raw_text', r['text']).strip()
            if re.match(r'^\([\d.]+\)$', _raw_bd):
                if VERBOSE:
                    print(f"    [paren_ref] skip basic_dim ref: '{_raw_bd}'")
                continue

            r['_claimed'] = True
            text = r['text'].strip()
            cleaned, prefix, dim_type = _extract_prefix(text)
            basic_dim = {
                'text': text,
                'nominal': cleaned if cleaned else text,
                'upper_tol': None,
                'lower_tol': None,
                'type': 'basic_dimension',
                'prefix': prefix,
                'bbox': {'x': rect['x'], 'y': rect['y'],
                         'w': rect['w'], 'h': rect['h']},
                'confidence': 'high',
                'source': 'basic_dimension',
                'inspection_role': 'theoretical_exact',
                'numbering_excluded': True,
                '_raw_text': _raw_bd,
            }
            out_dims.append(basic_dim)
            if out_refs is not None:
                out_refs.append({
                    'text': text,
                    'type': 'basic_dimension',
                    'bbox': dict(basic_dim['bbox']),
                    'inspection_role': 'theoretical_exact',
                    'numbering_excluded': True,
                })
            n_found += 1
            if VERBOSE:
                print(f"[basic_dim] text='{text}' "
                      f"bbox=({bbox['x']:.0f},{bbox['y']:.0f},{bbox['w']:.0f}x{bbox['h']:.0f}) "
                      f"— 被矩形 ({rect['x']:.0f},{rect['y']:.0f},"
                      f"{rect['w']:.0f}x{rect['h']:.0f}) 包围")
            break

    if VERBOSE:
        print(f"[basic_dim] 识别 {n_found} 个 Basic Dimension")


# 兼容路径的尺寸线关联。



def _in_exclusion_zone(bbox, exclusion_zones):
    """检查 region 的中心点是否落在任何排除区内。"""
    cx, cy = _bbox_center(bbox)
    for zone in exclusion_zones:
        if zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]:
            return True
    return False


def _ocr_orientation_angle(r):
    try:
        return abs(float(r.get('orientation', 0))) % 180.0
    except (TypeError, ValueError):
        return 0.0


def _is_vertical_ocr_result(r):
    angle = _ocr_orientation_angle(r)
    return 45.0 < angle < 135.0


def _dimline_axis_distance(dl, cx, cy):
    """Return a score for pairing OCR to a detected dimension line.

    For diagonal lines, midpoint distance is too blunt: a closer unrelated text
    blob can beat a real label that sits along the diagonal.  Score diagonal
    candidates mainly by perpendicular distance, with a small along-line cost
    once the text drifts past the line body.
    """
    orient = dl.get('orientation')
    if orient != 'D':
        mx, my = dl['midpoint']
        return math.hypot(cx - mx, cy - my), None

    start = dl.get('start')
    end = dl.get('end')
    if not start or not end:
        mx, my = dl['midpoint']
        return math.hypot(cx - mx, cy - my), None

    sx, sy = start
    ex, ey = end
    line_len = float(dl.get('length') or math.hypot(ex - sx, ey - sy) or 0.0)
    if line_len <= 0:
        mx, my = dl['midpoint']
        return math.hypot(cx - mx, cy - my), None

    mx, my = dl.get('midpoint') or ((sx + ex) / 2, (sy + ey) / 2)
    ux = (ex - sx) / line_len
    uy = (ey - sy) / line_len
    vx = cx - mx
    vy = cy - my
    along = abs(vx * ux + vy * uy)
    perp = abs(vx * (-uy) + vy * ux)
    beyond = max(0.0, along - line_len * 0.65)
    score = perp + beyond * 0.35
    return score, {'perp': perp, 'along': along}


def _load_dimension_lines(ocr_results: list, l1_lines: list, out_dims: list,
                          page_width: float = 0.0, page_height: float = 0.0,
                          frame_borders: list = None,
                          detected_dim_lines: list = None):
    """兼容路径中沿线段法线搜索未认领文字，结合排除区、距离和文本特征装载候选；优先使用检测到的尺寸线，缺失时采用细线段候选。"""
    # 构建采用两角点表示的排除区。
    exclusion_zones = []

    # 标题栏排除区
    if frame_borders:
        for fb in frame_borders:
            if fb.get("source") == "title_block":
                exclusion_zones.append((
                    fb["x"], fb["y"],
                    fb["x"] + fb["w"], fb["y"] + fb["h"],
                ))

    # 按固定边距比例排除页面边缘。
    if page_width > 0 and page_height > 0:
        mx_edge = page_width * 0.05
        my_edge = page_height * 0.05
        exclusion_zones.append((0, 0, page_width, my_edge))                        # top
        exclusion_zones.append((0, page_height - my_edge, page_width, page_height)) # bottom
        exclusion_zones.append((0, 0, mx_edge, page_height))                        # left
        exclusion_zones.append((page_width - mx_edge, 0, page_width, page_height))  # right

    if VERBOSE:
        print(f"[_load_dimension_lines] exclusion_zones: {len(exclusion_zones)} zones")

    # --- 箭头检测器优先路径 ---
    n_arrow_claimed = 0
    if detected_dim_lines:
        if VERBOSE:
            print(f"[_load_dimension_lines] arrow-detected dim lines: {len(detected_dim_lines)}")
        for dl_idx, dl in enumerate(detected_dim_lines):
            mx, my = dl['midpoint']
            search_dist = dl['length'] * 0.4
            search_r = max(search_dist, 50)
            if dl.get('orientation') == 'D':
                search_r = max(search_r, 75)
            orient = dl['orientation']

            if VERBOSE:
                print(f"[arrow_diag] dimline #{dl_idx+1}: midpoint=({mx:.1f},{my:.1f}) dir={orient} len={dl['length']:.1f} search_r={search_r:.1f}")

            best_r = None
            best_dist = float('inf')
            best_axis = None
            _diag_candidates = [] if VERBOSE else None
            for r in ocr_results:
                cx, cy = _bbox_center(r['bbox_in_pdf'])
                d, axis_dist = _dimline_axis_distance(dl, cx, cy)
                if d > search_r:
                    continue
                reason = None
                if r.get('_claimed', False):
                    reason = 'already_claimed'
                elif r.get('_leader_line_evidence'):
                    reason = 'leader_ocr_only'
                else:
                    is_vert = _is_vertical_ocr_result(r)
                    if orient == 'H' and is_vert:
                        reason = f'orient_mismatch(dimH, ocr_vert={r.get("orientation",0)})'
                    elif orient == 'V' and not is_vert:
                        d = d * 2.0  # 距离惩罚
                if VERBOSE:
                    _diag_candidates.append((r['text'], cx, cy, d, r.get('orientation', 0), reason))
                if reason:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_r = r
                    best_axis = axis_dist

            if VERBOSE:
                if _diag_candidates:
                    for txt, cx, cy, d, ori, rej in _diag_candidates:
                        status = f'REJECTED: {rej}' if rej else 'CANDIDATE'
                        if VERBOSE:
                            print(f"  candidate: text='{txt}' center=({cx:.1f},{cy:.1f}) dist={d:.1f} orient={ori} -> {status}")
                else:
                    if VERBOSE:
                        print(f"  (no candidates in range)")

            if best_r:
                _text_q = best_r['text'].strip()
                if not re.search(r'\d', _text_q):
                    continue
                _alpha_tokens = re.findall(r'[a-zA-Z]{2,}', _text_q)
                _legit_words = {'MAX', 'MIN', 'TYP', 'REF', 'BSC', 'NOM'}
                if len([w for w in _alpha_tokens if w.upper() not in _legit_words]) >= 3:
                    continue
                if re.match(r'^\d{3,}$', _text_q) and '.' not in _text_q:
                    _has_nearby_tol = False
                    for _oth in ocr_results:
                        if _oth is best_r:
                            continue
                        _ot = _oth.get('text', '').strip()
                        if re.match(r'^[±+\-]', _ot):
                            _od = math.hypot(
                                best_r['bbox_in_pdf']['x'] - _oth['bbox_in_pdf']['x'],
                                best_r['bbox_in_pdf']['y'] - _oth['bbox_in_pdf']['y'])
                            if _od < 50:
                                _has_nearby_tol = True
                                break
                    if not _has_nearby_tol:
                        continue

                if VERBOSE:
                    b = best_r['bbox_in_pdf']
                    if VERBOSE:
                        print(f"  -> ACCEPTED: text='{best_r['text']}' dist={best_dist:.1f}")
                # paren-reference fast-skip：raw OCR 是 (N) 形态直接当参考尺寸丢弃
                _raw_q = best_r.get('_raw_text', best_r['text'])
                if re.match(r'^\([\d.]+\)$', _raw_q.strip()):
                    if VERBOSE:
                        print(f"    [paren_ref] skip dimline ref: '{_raw_q}'")
                    best_r['_claimed'] = True  # 防止后续 score 路径再次处理
                    continue
                best_r['_dimension_line_evidence'] = {
                    'line_index': dl_idx,
                    'distance': round(best_dist, 2),
                    'orientation': orient,
                    'line_length': round(float(dl.get('length') or 0), 2),
                    'kind': 'paired_dimension_line',
                }
                if best_axis:
                    best_r['_dimension_line_evidence']['perp_distance'] = round(best_axis['perp'], 2)
                    best_r['_dimension_line_evidence']['along_distance'] = round(best_axis['along'], 2)
                n_arrow_claimed += 1

        if VERBOSE:
            print(f"[_load_dimension_lines] arrow-path claimed: {n_arrow_claimed}")

    n_candidates = 0
    n_blocked = 0
    n_blacklist = 0
    n_claimed = 0
    n_claimed_h = 0
    n_claimed_v = 0

    def _is_vertical_ocr(r):
        """根据 OCR 方向信息判断文本是否竖直。"""
        return _is_vertical_ocr_result(r)

    # 缺少检测结果时采用原始细线段候选。
    for line in l1_lines:
        x0, y0 = line.get('x0', 0), line.get('y0', 0)
        x1, y1 = line.get('x1', 0), line.get('y1', 0)
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 5:
            continue

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx < 1.5 and dy > 5:
            line_axis = 'v'
        elif dy < 1.5 and dx > 5:
            line_axis = 'h'
        else:
            line_axis = None

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        search_dist = length * 0.3

        found = []
        for r in ocr_results:
            if r['_claimed']:
                continue
            if line_axis == 'h' and _is_vertical_ocr(r):
                continue
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            dist = math.hypot(cx - mx, cy - my)
            if line_axis == 'v' and not _is_vertical_ocr(r):
                dist *= 2.0
            if dist <= search_dist:
                found.append((dist, r))

        if not found:
            continue

        found.sort(key=lambda t: t[0])
        best_dist, best = found[0]
        text = best['text'].strip()

        if not re.search(r'\d', text):
            continue
        _alpha_tok = re.findall(r'[a-zA-Z]{2,}', text)
        _legit_w = {'MAX', 'MIN', 'TYP', 'REF', 'BSC', 'NOM'}
        if len([w for w in _alpha_tok if w.upper() not in _legit_w]) >= 3:
            continue
        if re.match(r'^\d{3,}$', text) and '.' not in text:
            _has_nearby_tol = False
            for _oth in ocr_results:
                if _oth is best:
                    continue
                _ot = _oth.get('text', '').strip()
                if re.match(r'^[±+\-]', _ot):
                    _od = math.hypot(
                        best['bbox_in_pdf']['x'] - _oth['bbox_in_pdf']['x'],
                        best['bbox_in_pdf']['y'] - _oth['bbox_in_pdf']['y'])
                    if _od < 50:
                        _has_nearby_tol = True
                        break
            if not _has_nearby_tol:
                continue

        n_candidates += 1

        if _in_exclusion_zone(best['bbox_in_pdf'], exclusion_zones):
            n_blocked += 1
            continue

        if page_width > 0 and page_height > 0:
            region_cx, region_cy = _bbox_center(best['bbox_in_pdf'])
            ref_right_limit = page_width * 0.65
            if frame_borders:
                for fb in frame_borders:
                    if fb.get("source") == "title_block":
                        ref_right_limit = fb["x"]
                        break
            if (region_cx > page_width * 0.50 and
                    region_cy > page_height * 0.78 and
                    region_cx < ref_right_limit):
                n_blocked += 1
                continue

        if not text:
            continue

        if re.match(r'^\d{1,2}$', text):
            bbox_cx, bbox_cy = _bbox_center(best['bbox_in_pdf'])
            if (page_height > 0 and bbox_cy > page_height * 0.80
                    and page_width > 0 and bbox_cx > page_width * 0.45):
                continue

        if any(pat.search(text) for pat in METADATA_BLACKLIST_PATTERNS):
            n_blacklist += 1
            continue

        should_claim = False

        if best_dist < 10:
            if any(c in text for c in '.±⌀°⊕⊘∅Ø⊥∥◎○⌭—∠⌓⌒▱≡'):
                should_claim = True
            elif any(c.isdigit() for c in text):
                stripped = text.replace(' ', '')
                if len(stripped) == 1 and stripped.isdigit():
                    pass
                elif stripped.isdigit() and len(stripped) >= 4:
                    pass
                else:
                    should_claim = True
            elif best_dist < 6 and len(text) >= 1:
                should_claim = True

        if not should_claim and _quick_dim_score(best['text']) >= 10:
            stripped = text.replace(' ', '')
            if len(stripped) == 1 and stripped.isdigit():
                pass
            elif stripped.isdigit() and len(stripped) >= 4:
                pass
            else:
                should_claim = True

        if not should_claim:
            continue

        # paren-reference fast-skip：raw OCR 是 (N) 形态直接当参考尺寸丢弃
        _raw_q = best.get('_raw_text', best['text'])
        if re.match(r'^\([\d.]+\)$', _raw_q.strip()):
            if VERBOSE:
                print(f"    [paren_ref] skip L1-fallback ref: '{_raw_q}'")
            best['_claimed'] = True
            continue

        n_claimed += 1
        if _is_vertical_ocr(best):
            n_claimed_v += 1
        else:
            n_claimed_h += 1
        best['_claimed'] = True
        text_full = _fixup_ocr_text(best['text'])
        text_clean, prefix, dim_type = _extract_prefix(text_full)
        upper_tol, lower_tol = _parse_tolerance_inline(text_clean)
        nominal_raw = re.sub(r'[±\+\-]\s*[\d.]+.*$', '', text_clean).strip() if (upper_tol or lower_tol) else text_clean

        out_dims.append({
            'text': text_full,
            'nominal': nominal_raw,
            'upper_tol': upper_tol,
            'lower_tol': lower_tol,
            'type': dim_type,
            'prefix': prefix,
            'bbox': best['bbox_in_pdf'],
            'confidence': 'medium',
            'source': 'ocr',
            'orientation': best.get('orientation', 0),
            '_raw_text': _raw_q,
        })

    if VERBOSE:
        print(f"[_load_dimension_lines] candidates near dimlines: {n_candidates}")
    if VERBOSE:
        print(f"[_load_dimension_lines] blocked by exclusion: {n_blocked}")
    if VERBOSE:
        print(f"[_load_dimension_lines] blocked by blacklist: {n_blacklist}")
    if VERBOSE:
        print(f"[_load_dimension_lines] claimed: {n_claimed} (H={n_claimed_h}, V={n_claimed_v})")


# 评分前合并横向形位框链。



def _merge_gdt_chain(items: list) -> list:
    """
    GD&T 横向链式合并：将水平相邻、高度相近、至少一端含 GD&T 特征的碎片合并。
    迭代执行直到无更多合并发生。
    """
    changed = True
    while changed:
        changed = False
        n = len(items)
        merged_flags = [False] * n
        new_items = []

        for i in range(n):
            if merged_flags[i]:
                continue
            a = items[i]
            ba = a['bbox_in_pdf']
            best_j = -1
            best_gap = float('inf')

            for j in range(i + 1, n):
                if merged_flags[j]:
                    continue
                b = items[j]
                bb = b['bbox_in_pdf']

                # 检查纵向中心对齐。
                _, cy_a = _bbox_center(ba)
                _, cy_b = _bbox_center(bb)
                min_h = min(ba['h'], bb['h'])
                if min_h <= 0 or abs(cy_a - cy_b) > min_h * 0.3:
                    continue

                # 检查横向间隔。
                max_w = max(ba['w'], bb['w'])
                gap_x = min(
                    abs(bb['x'] - (ba['x'] + ba['w'])),
                    abs(ba['x'] - (bb['x'] + bb['w'])),
                )
                if gap_x > max_w * 0.8:
                    continue

                # 检查高度比例。
                ratio = ba['h'] / bb['h'] if bb['h'] > 0 else 999
                if not (0.7 <= ratio <= 1.3):
                    continue

                # 要求候选携带形位特征。
                if not (_is_gdt_fragment(a['text']) or _is_gdt_fragment(b['text'])):
                    continue

                if gap_x < best_gap:
                    best_gap = gap_x
                    best_j = j

            if best_j >= 0:
                b_item = items[best_j]
                bb = b_item['bbox_in_pdf']
                x0 = min(ba['x'], bb['x'])
                y0 = min(ba['y'], bb['y'])
                x1 = max(ba['x'] + ba['w'], bb['x'] + bb['w'])
                y1 = max(ba['y'] + ba['h'], bb['y'] + bb['h'])
                if ba['x'] <= bb['x']:
                    merged_text = a['text'].strip() + ' ' + b_item['text'].strip()
                else:
                    merged_text = b_item['text'].strip() + ' ' + a['text'].strip()

                merged = {**a,
                          'text': merged_text,
                          'bbox_in_pdf': {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0},
                          'font_height_approx': max(
                              a.get('font_height_approx', 0),
                              b_item.get('font_height_approx', 0)),
                          }
                new_items.append(merged)
                merged_flags[i] = True
                merged_flags[best_j] = True
                changed = True
            else:
                new_items.append(a)

        items = new_items

    return items
