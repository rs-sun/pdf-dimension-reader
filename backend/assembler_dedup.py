"""
assembler_dedup.py — 去重 / 孤儿公差回收 / 字号配对

从 assembler.py 拆出。包含：
  - _recover_orphan_tolerances()
  - _upgrade_tol_if_safe()
  - _dedup_dimensions()
  - _font_size_pairing()
"""

import math
import re
import statistics

from assembler_utils import (
    VERBOSE, _CONF_ORDER,
    _bbox_center, _bbox_iou, _levenshtein, _is_vertical_orient, _union_bbox,
)


# 孤儿公差合并时允许的 parent-to-pair/orphan 最大欧式距离（pt）
_ORPHAN_MAX_EUCLID = 50.0


def _normalize_signed_tolerance_text(tol: str) -> str:
    """规范化带符号的 OCR 公差碎片，合并符号与数值之间的空白。"""
    raw = (tol or '').strip().replace('。', '.')
    raw = re.sub(r'(?<=\d)[Oo](?=\d)', '0', raw)
    m = re.fullmatch(r'([+\-])\s*((?:\d+(?:\.\d*)?|\.\d+))(°?)', raw)
    if not m:
        return raw
    value = m.group(2)
    if value.startswith('.'):
        value = '0' + value
    elif value.endswith('.'):
        value = value[:-1]
    return f'{m.group(1)}{value}{m.group(3)}'


def _tol_abs_value(tol: str) -> str | None:
    text = _normalize_signed_tolerance_text(tol)
    if not text:
        return None
    if text[0] in '+-':
        text = text[1:]
    if text.endswith('°'):
        text = text[:-1]
    try:
        return f"{float(text):g}"
    except ValueError:
        return None


def _tol_display_value(tol: str) -> str:
    text = _normalize_signed_tolerance_text(tol)
    if text and text[0] in '+-':
        return text[1:]
    return text


def _unsigned_tol_value(text: str) -> str | None:
    """规范化满足幅值限制的无符号小数碎片，供缺失公差符号的兼容推断使用。"""
    raw = (text or '').strip().replace(' ', '')
    raw = raw.replace('O', '0').replace('o', '0')
    if not re.fullmatch(r'(?:0?\.\d+|0\.\d+)', raw):
        return None
    if raw.startswith('.'):
        raw = '0' + raw
    try:
        value = float(raw)
    except ValueError:
        return None
    if not (0.0 < value < 1.0):
        return None
    return raw


def _is_unsigned_tol_fragment(dim: dict) -> bool:
    if dim.get('upper_tol') or dim.get('lower_tol') or dim.get('prefix'):
        return False
    text = (dim.get('text') or '').strip()
    if not text or text[0] in '+-':
        return False
    return _unsigned_tol_value(text) is not None


def _unsigned_dim_has_signed_tol_parent_signal(dim: dict, signed_tols: list[dict]) -> bool:
    """根据附近带符号公差的位置，判断无符号小数应作为公称值还是缺失符号的公差碎片。"""
    if not _is_unsigned_tol_fragment(dim):
        return False
    if dim.get('source') == 'gdt_line_frame':
        return False
    parent_bbox = dim.get('bbox') or {}
    if not all(k in parent_bbox for k in ('x', 'y', 'w', 'h')):
        return False
    parent_family = _orientation_family(dim)
    for tol in signed_tols:
        tol_text = _normalize_signed_tolerance_text(tol.get('text', ''))
        if not tol_text.startswith(('+', '-')):
            continue
        if not _tol_reasonable_for_parent(dim, [tol_text]):
            continue
        tol_bbox = tol.get('bbox') or {}
        if not all(k in tol_bbox for k in ('x', 'y', 'w', 'h')):
            continue
        ok, _dist = _parent_ok_for_single(
            tol_bbox,
            parent_bbox,
            max_euclid=45.0,
        )
        if not ok:
            continue
        tol_family = _orientation_family(tol)
        if parent_family != 'unknown' and tol_family != 'unknown' and parent_family != tol_family:
            continue
        return True
    return False


def _unsigned_fragment_has_left_parent_pair_signal(
    dim: dict,
    signed_tols: list[dict],
    dimensions: list[dict],
) -> bool:
    """结合堆叠公差位置与左侧公称值，判断无符号小数是否是缺失负号的下公差。"""
    value = _unsigned_tol_value(dim.get('text', ''))
    if value is None:
        return False
    ub = dim.get('bbox') or {}
    if not all(k in ub for k in ('x', 'y', 'w', 'h')):
        return False

    for signed in signed_tols:
        signed_text = _normalize_signed_tolerance_text(signed.get('text', ''))
        if not signed_text.startswith(('+', '-')):
            continue
        sb = signed.get('bbox') or {}
        if not all(k in sb for k in ('x', 'y', 'w', 'h')):
            continue
        inferred = _infer_tolerance_pair(
            -1,
            signed,
            signed_text[0],
            -2,
            dim,
            None,
        )
        if inferred is None:
            continue
        _pi, plus_o, _mi, minus_o, _priority = inferred
        pair_bbox = _union_bbox([plus_o['bbox'], minus_o['bbox']])
        pair_cx, pair_cy = _bbox_center(pair_bbox)

        for parent in dimensions:
            if parent is dim or parent is signed:
                continue
            text = (parent.get('text') or '').strip()
            if not text or text.startswith(('+', '-')):
                continue
            if parent.get('source') == 'gdt_line_frame':
                continue
            if parent.get('upper_tol') or parent.get('lower_tol'):
                continue
            nominal_val = _numeric_abs(parent.get('nominal') or text)
            if nominal_val is None or nominal_val < 1.0:
                continue
            if not _tol_reasonable_for_parent(parent, [plus_o.get('text', ''), minus_o.get('text', '')]):
                continue
            pb = parent.get('bbox') or {}
            if not all(k in pb for k in ('x', 'y', 'w', 'h')):
                continue
            pcx, pcy = _bbox_center(pb)
            dx = pcx - pair_cx
            dy = pcy - pair_cy
            if dx >= 0:
                continue
            dist = math.hypot(dx, dy)
            if dist > 60.0:
                continue
            if abs(dy) > max(pair_bbox.get('h', 0.0), pb.get('h', 0.0)) * 1.15:
                continue
            return True
    return False


def _signed_tol_copy(dim: dict, sign: str) -> dict:
    value = _unsigned_tol_value(dim.get('text', ''))
    signed = dict(dim)
    if value is not None:
        signed['text'] = f'{sign}{value}'
        signed['nominal'] = signed['text']
        signed['_sign_inferred'] = True
    return signed


def _symmetric_tol_value(dim: dict) -> str | None:
    """识别只有对称公差、没有公称值的独立碎片，返回其绝对值，供后续并入邻近公称值。"""
    if dim.get('prefix') or dim.get('_merged'):
        return None
    nominal = (dim.get('nominal') or '').strip()
    text = (dim.get('text') or '').strip()
    text_compact = re.sub(r'\s+', '', text)

    m_text = re.fullmatch(r'±(\d+(?:\.\d+)?)(?:°)?', text_compact)
    if m_text and not nominal:
        return m_text.group(1)

    if nominal and nominal not in {text, text_compact}:
        return None
    upper = _tol_abs_value(dim.get('upper_tol'))
    lower = _tol_abs_value(dim.get('lower_tol'))
    if upper and lower and upper == lower and (
            not nominal or re.fullmatch(r'[±\s\d.°]+', text)):
        return upper
    return None


def _format_tolerance_text(base: str, upper_tol: str | None, lower_tol: str | None) -> str:
    ut = _normalize_signed_tolerance_text(upper_tol)
    lt = _normalize_signed_tolerance_text(lower_tol)
    if ut and lt:
        u_abs = _tol_abs_value(ut)
        l_abs = _tol_abs_value(lt)
        if u_abs is not None and u_abs == l_abs:
            return f"{base} ±{_tol_display_value(ut)}"
        return f"{base} {ut}/{lt}"
    if ut:
        return f"{base} {ut}"
    if lt:
        return f"{base} {lt}"
    return base


def _sync_tol_into_text(parent: dict) -> None:
    """回填孤立公差后，依据公称值和前缀重建显示文本，使结构化字段与文本保持一致。"""
    ut = (parent.get('upper_tol') or '').strip()
    lt = (parent.get('lower_tol') or '').strip()
    if not ut and not lt:
        return
    nominal = (parent.get('nominal') or '').strip()
    if not nominal:
        return
    prefix = (parent.get('prefix') or '').strip()
    base = f"{prefix} {nominal}".strip() if prefix else nominal
    parent['text'] = _format_tolerance_text(base, ut, lt)


def _angle_nominal_key(text: str) -> str | None:
    compact = re.sub(r'\s+', '', text or '')
    m = re.match(r'^(\d+(?:\.\d+)?)°', compact)
    return m.group(1) if m else None


def _angle_parent_text(text: str) -> tuple[str, float] | None:
    compact = re.sub(r'\s+', '', text or '')
    m = re.fullmatch(r'(?:(\d{1,2})[xX×])?(\d{1,3}(?:\.\d+)?)°', compact)
    if not m:
        return None
    try:
        value = float(m.group(2))
    except ValueError:
        return None
    if value <= 5.0:
        return None
    prefix = f'{m.group(1)}X' if m.group(1) else ''
    return f'{prefix}{m.group(2)}°', value


def _angle_tolerance_value(text: str) -> str | None:
    compact = re.sub(r'\s+', '', text or '').replace('º', '°')
    m = re.fullmatch(r'(?:±|\+)?(\d(?:\.\d+)?)°', compact)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    if not (0.0 < value <= 5.0):
        return None
    return str(int(value)) if abs(value - int(value)) < 1e-6 else str(value).rstrip('0').rstrip('.')


def _merge_angle_tolerance_fragments(dimensions: list) -> list:
    """将邻近的小角度公差碎片并回兼容路径中的角度公称值。"""
    if not dimensions:
        return dimensions

    parents = []
    fragments = []
    for idx, dim in enumerate(dimensions):
        text = (dim.get('text') or '').strip()
        if dim.get('upper_tol') or dim.get('lower_tol'):
            continue
        parent = _angle_parent_text(text)
        if parent is not None:
            parents.append((idx, dim, parent[0]))
            continue
        tol_value = _angle_tolerance_value(text)
        if tol_value is not None:
            fragments.append((idx, dim, tol_value))

    if not parents or not fragments:
        return dimensions

    drop_indices: set[int] = set()
    for fi, frag, tol_value in fragments:
        if fi in drop_indices:
            continue
        fb = frag.get('bbox') or {}
        if not all(k in fb for k in ('x', 'y', 'w', 'h')):
            continue
        fcx, fcy = _bbox_center(fb)
        best = None
        for pi, parent, parent_text in parents:
            if pi == fi or pi in drop_indices:
                continue
            pb = parent.get('bbox') or {}
            if not all(k in pb for k in ('x', 'y', 'w', 'h')):
                continue
            pcx, pcy = _bbox_center(pb)
            dx = fcx - pcx
            dy = fcy - pcy
            parent_vertical = _is_vertical_orient(parent.get('orientation'))
            frag_vertical = _is_vertical_orient(frag.get('orientation'))
            h = max(float(pb.get('h', 0.0) or 0.0),
                    float(fb.get('h', 0.0) or 0.0),
                    1.0)
            w = max(float(pb.get('w', 0.0) or 0.0),
                    float(fb.get('w', 0.0) or 0.0),
                    1.0)
            if parent_vertical and frag_vertical:
                if abs(dx) > max(24.0, w * 1.6):
                    continue
                if abs(dy) > max(45.0, h * 2.4):
                    continue
            else:
                if dx <= 0:
                    continue
                if dx > max(45.0, h * 2.8):
                    continue
                if abs(dy) > max(14.0, h * 0.75):
                    continue
            dist = math.hypot(dx, dy)
            if best is None or dist < best[0]:
                best = (dist, pi, parent, parent_text)
        if best is None:
            continue
        _, _pi, parent, parent_text = best
        parent['text'] = f'{parent_text}±{tol_value}°'
        parent['nominal'] = parent_text
        parent['upper_tol'] = f'+{tol_value}'
        parent['lower_tol'] = f'-{tol_value}'
        parent['bbox'] = _union_bbox([parent['bbox'], frag['bbox']])
        parent['_angle_tolerance_merged'] = True
        drop_indices.add(fi)

    if not drop_indices:
        return dimensions
    return [dim for idx, dim in enumerate(dimensions) if idx not in drop_indices]


# ---------------------------------------------------------------------------
# 孤儿公差回收
# ---------------------------------------------------------------------------

def _classify_pair_layout(plus_o: dict, minus_o: dict) -> str:
    """判定 (+tol, -tol) pair 的内部布局：
       - 'stacked'      两 tol 上下堆叠
       - 'side-by-side' 两 tol 左右并列
       - 'mixed'        方向不明确（差距过小）
    """
    pcx, pcy = _bbox_center(plus_o['bbox'])
    mcx, mcy = _bbox_center(minus_o['bbox'])
    dx = abs(pcx - mcx)
    dy = abs(pcy - mcy)
    small = max(min(plus_o['bbox']['h'], minus_o['bbox']['h']) * 0.5, 2.0)
    if dy > dx * 1.5 and dy > small:
        return 'stacked'
    if dx > dy * 1.5 and dx > small:
        return 'side-by-side'
    return 'mixed'


def _ordered_pair_indices(a: dict, b: dict, layout: str) -> tuple[int, int] | None:
    """Return first/second index for the sign convention (+ before -)."""
    acx, acy = _bbox_center(a['bbox'])
    bcx, bcy = _bbox_center(b['bbox'])
    if layout == 'side-by-side':
        if abs(acx - bcx) < 1.0:
            return None
        return (0, 1) if acx < bcx else (1, 0)
    if layout == 'stacked':
        if abs(acy - bcy) < 1.0:
            return None
        return (0, 1) if acy < bcy else (1, 0)
    return None


def _infer_tolerance_pair(
    left_idx: int,
    left_dim: dict,
    left_sign: str | None,
    right_idx: int,
    right_dim: dict,
    right_sign: str | None,
) -> tuple[int, dict, int, dict, int] | None:
    """Infer a (+tol, -tol) pair, allowing one OCR sign to be missing."""
    layout = _classify_pair_layout(left_dim, right_dim)
    order = _ordered_pair_indices(left_dim, right_dim, layout)
    if order is None:
        return None

    dims = [left_dim, right_dim]
    indices = [left_idx, right_idx]
    signs = [left_sign, right_sign]
    first, second = order

    plus_pos = None
    minus_pos = None
    for pos, sign in enumerate(signs):
        if sign == '+':
            plus_pos = pos
        elif sign == '-':
            minus_pos = pos

    if plus_pos is not None and minus_pos is not None:
        priority = 0
    elif plus_pos is not None:
        minus_pos = second if plus_pos == first else None
        priority = 1
    elif minus_pos is not None:
        plus_pos = first if minus_pos == second else None
        priority = 1
    else:
        return None

    if plus_pos is None or minus_pos is None or plus_pos == minus_pos:
        return None

    plus_dim = dims[plus_pos] if signs[plus_pos] == '+' else _signed_tol_copy(dims[plus_pos], '+')
    minus_dim = dims[minus_pos] if signs[minus_pos] == '-' else _signed_tol_copy(dims[minus_pos], '-')
    return indices[plus_pos], plus_dim, indices[minus_pos], minus_dim, priority


def _parent_ok_for_pair(pair_bbox: dict, parent_bbox: dict, layout: str,
                        max_euclid: float = _ORPHAN_MAX_EUCLID) -> tuple:
    """根据并列、堆叠或不明确的布局，检查公称值与公差对的相对方向和中心距离，返回是否可配对及距离。"""
    pair_cx, pair_cy = _bbox_center(pair_bbox)
    pair_w, pair_h = pair_bbox['w'], pair_bbox['h']

    qcx, qcy = _bbox_center(parent_bbox)
    qw, qh = parent_bbox['w'], parent_bbox['h']

    dx = qcx - pair_cx
    dy = qcy - pair_cy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > max_euclid:
        return (False, dist)

    if layout == 'side-by-side':
        # (a) parent 在 pair 左侧（水平尺寸惯例，parent 横排或方形）
        left_ok = (dx < 0) and (abs(dy) < max(pair_h, qh) * 0.9)
        # 只有竖排公称值允许上下方向的横排公差对，避免放行水平布局的邻项干扰。
        
        
        parent_is_vertical = qh > qw * 1.5
        vert_ok = (parent_is_vertical
                   and abs(dx) < max(pair_w, qw) * 1.2
                   and abs(dy) < max(pair_h, qh) * 2.0)
        return (left_ok or vert_ok, dist)

    if layout == 'stacked':
        # (a) parent 在 pair 左侧（ISO 风格），y 对齐
        left_ok = (dx < 0) and (abs(dy) < max(pair_h, qh) * 1.0)
        # 检查公称值与堆叠公差中心轴的对齐及纵向距离。
        center_ok = (abs(dx) < max(pair_w, qw) * 0.8) and (abs(dy) < max(pair_h, qh) * 1.5)
        return (left_ok or center_ok, dist)

    # mixed: 纯欧式兜底
    return (True, dist)


def _bbox_text_size(dim: dict) -> float:
    """Return an orientation-stable text size proxy for pairing decisions."""
    bbox = dim.get('bbox') or {}
    w = float(bbox.get('w') or 0.0)
    h = float(bbox.get('h') or 0.0)
    fh = float(dim.get('_font_height') or 0.0)
    if _is_vertical_orient(dim.get('orientation', 0)):
        return min(v for v in (w, h, fh) if v > 0) if any(v > 0 for v in (w, h, fh)) else 0.0
    if fh > 0 and h > 0:
        return min(fh, h)
    return h if h > 0 else min(w, fh) if min(w, fh) > 0 else max(w, fh)


def _orientation_family(dim: dict) -> str:
    orient = dim.get('orientation', None)
    if orient is None:
        return 'unknown'
    try:
        orient = float(orient)
    except (TypeError, ValueError):
        return 'unknown'
    if abs(abs(orient) - 90.0) < 20.0:
        return 'vertical'
    if abs(orient) < 20.0 or abs(abs(orient) - 180.0) < 20.0:
        return 'horizontal'
    return 'angled'


def _numeric_abs(text: str | None) -> float | None:
    m = re.search(r'-?\d+(?:\.\d+)?', str(text or ''))
    if not m:
        return None
    try:
        return abs(float(m.group(0)))
    except ValueError:
        return None


def _tol_reasonable_for_parent(parent: dict, tol_texts: list[str]) -> bool:
    nominal_val = _numeric_abs(parent.get('nominal') or parent.get('text'))
    if nominal_val is None or nominal_val <= 0:
        return True
    limit = max(nominal_val * 2.0, 0.5) if nominal_val < 5 else max(nominal_val * 0.5, 3.0)
    for tol_text in tol_texts:
        tol_val = _numeric_abs(tol_text)
        if tol_val is not None and tol_val >= limit:
            return False
    return True


def _font_size_penalty(parent: dict, tol_dims: list[dict]) -> float:
    parent_size = _bbox_text_size(parent)
    tol_sizes = [_bbox_text_size(t) for t in tol_dims]
    tol_sizes = [s for s in tol_sizes if s > 0]
    if parent_size <= 0 or not tol_sizes:
        return 0.0
    tol_size = sum(tol_sizes) / len(tol_sizes)
    ratio = tol_size / parent_size
    if 0.35 <= ratio <= 0.95:
        return -4.0
    if 0.95 < ratio <= 1.20:
        return 0.0
    if 0.22 <= ratio < 0.35:
        return 4.0
    return 12.0


def _orientation_penalty(parent: dict, tol_dims: list[dict]) -> float:
    parent_family = _orientation_family(parent)
    tol_families = {_orientation_family(t) for t in tol_dims}
    tol_families.discard('unknown')
    if parent_family == 'unknown' or not tol_families:
        return 0.0
    if parent_family in tol_families:
        return -2.0
    # Vertical dimensions sometimes carry horizontal +/- pairs above/below the
    # nominal; keep this as a soft signal instead of a hard rejection.
    if parent_family == 'vertical' and 'horizontal' in tol_families:
        return 7.0
    return 8.0


def _pair_sign_order_penalty(plus_o: dict, minus_o: dict, layout: str) -> float:
    pcx, pcy = _bbox_center(plus_o['bbox'])
    mcx, mcy = _bbox_center(minus_o['bbox'])
    if layout == 'side-by-side':
        return -3.0 if pcx <= mcx else 14.0
    if layout == 'stacked':
        return -4.0 if pcy <= mcy else 14.0
    return 0.0


def _pair_relative_bonus(pair_bbox: dict, parent_bbox: dict, layout: str) -> float:
    pair_cx, pair_cy = _bbox_center(pair_bbox)
    qcx, qcy = _bbox_center(parent_bbox)
    dx = qcx - pair_cx
    dy = qcy - pair_cy
    qw = float(parent_bbox.get('w') or 0.0)
    qh = float(parent_bbox.get('h') or 0.0)
    if layout == 'side-by-side':
        if qh > qw * 1.5 and abs(dx) < max(pair_bbox['w'], qw) * 1.2:
            return -3.0
        if dx < 0 and abs(dy) < max(pair_bbox['h'], qh) * 0.8:
            return -4.0
    if layout == 'stacked':
        if abs(dx) < max(pair_bbox['w'], qw) * 0.8:
            return -4.0
        if dx < 0 and abs(dy) < max(pair_bbox['h'], qh) * 0.9:
            return -3.0
    return 0.0


def _pair_parent_score(pair_bbox: dict, plus_o: dict, minus_o: dict,
                       parent: dict, layout: str, dist: float) -> float:
    score = dist
    score += _pair_sign_order_penalty(plus_o, minus_o, layout)
    score += _pair_relative_bonus(pair_bbox, parent['bbox'], layout)
    score += _font_size_penalty(parent, [plus_o, minus_o])
    score += _orientation_penalty(parent, [plus_o, minus_o])
    return score


def _parent_ok_for_single(orphan_bbox: dict, parent_bbox: dict,
                          max_euclid: float = _ORPHAN_MAX_EUCLID) -> tuple:
    """依据孤立公差的长轴检查公称值的搜索方向和对齐关系；弱方向采用更紧的距离限制。"""
    ocx, ocy = _bbox_center(orphan_bbox)
    ow, oh = orphan_bbox['w'], orphan_bbox['h']

    qcx, qcy = _bbox_center(parent_bbox)
    qw, qh = parent_bbox['w'], parent_bbox['h']

    dx = qcx - ocx
    dy = qcy - ocy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > max_euclid:
        return (False, dist)

    if ow >= oh:  # 横排孤立公差优先匹配左侧对齐的公称值。
        
        main_ok = (dx < 0) and (abs(dy) < max(oh, qh) * 1.0)
        # 弱方向：x 对齐 + y 稍远（竖排布局反向落到横排 orphan 的情况）
        weak_ok = (abs(dx) < max(ow, qw) * 0.7) and (abs(dy) < max(oh, qh) * 1.5)
        return (main_ok or weak_ok, dist)
    else:  # 竖排 orphan
        # 放宽：parent 在 orphan 上方或下方均可（竖排标注里两种都常见），
        # 核心判据改为 "x 对齐 + y 在 max_euclid 距离内"。
        aligned_x = abs(dx) < max(ow, qw) * 1.2
        aligned_y = abs(dy) < max(oh, qh) * 1.5
        main_ok = aligned_x and aligned_y
        # 弱方向：y 对齐 + x 稍远（罕见，但 gdt_compartment 剥离后可能出现）
        weak_ok = (abs(dy) < max(oh, qh) * 0.7) and (abs(dx) < max(ow, qw) * 1.5)
        return (main_ok or weak_ok, dist)


def _single_parent_score(orphan: dict, parent: dict, dist: float) -> float:
    score = dist
    ob = orphan['bbox']
    pb = parent['bbox']
    ocx, ocy = _bbox_center(ob)
    pcx, pcy = _bbox_center(pb)
    dx = pcx - ocx
    dy = pcy - ocy
    ow, oh = float(ob.get('w') or 0.0), float(ob.get('h') or 0.0)
    pw, ph = float(pb.get('w') or 0.0), float(pb.get('h') or 0.0)
    if ow >= oh:
        if dx < 0 and abs(dy) < max(oh, ph) * 0.8:
            score -= 4.0
    else:
        if abs(dx) < max(ow, pw) * 1.0:
            score -= 3.0
    score += _font_size_penalty(parent, [orphan])
    score += _orientation_penalty(parent, [orphan])
    return score


def _pick_best_with_ambig_guard(candidates: list):
    """从距离项或评分项中选择候选，并以相邻排名的差距检查歧义；评分包含符号顺序、字号、方向和相对布局。"""
    if not candidates:
        return None
    if len(candidates[0]) == 2:
        candidates = [(dist, dist, idx) for dist, idx in candidates]
    candidates.sort(key=lambda t: (t[0], t[1]))
    best = candidates[0]
    if len(candidates) >= 2:
        abs_gap = abs(candidates[1][1] - best[1])
        rel_gap = abs_gap / max(abs(best[1]), 1.0)
        if rel_gap < 0.15 and abs_gap < 5.0:
            return None
    return best[1], best[2]


def _recover_orphan_tolerances(dimensions: list, page_width: float = 0.0, page_height: float = 0.0) -> list:
    """依据符号、相对布局、字号和方向回收孤立公差；候选关系不明确时拒绝合并。"""
    # 显式符号优先；无符号推断仅接纳满足幅值限制的小数碎片。
    _orphan_pat = re.compile(r'^[+\-]\s*\.?\d')
    orphans = []
    explicit_orphans = set()
    symmetric_orphans = {}
    tolerance_fragments = []  # (idx, dim, sign-or-None)
    parents = []

    if VERBOSE:
        print(f"[s27.orphan] ENTRY: dimensions={len(dimensions)}")
        for _i, _d in enumerate(dimensions):
            _t = _d.get('text', '').strip()
            _b = _d.get('bbox', {})
            _fh = _d.get('_font_height', 0)
            _or = _d.get('orientation', 0)
            _src = _d.get('source', '')
            print(f"[s27.orphan.cand] [{_i}] text='{_t}' bbox=({_b.get('x',0):.0f},{_b.get('y',0):.0f},{_b.get('w',0):.0f}x{_b.get('h',0):.0f}) fh={_fh:.1f} orient={_or} src={_src}")

    explicit_signed_tols = [
        dim for dim in dimensions
        if _orphan_pat.match((dim.get('text') or '').strip()) and not dim.get('_merged')
    ]

    for i, dim in enumerate(dimensions):
        text = dim.get('text', '').strip()
        sym_value = _symmetric_tol_value(dim)
        if sym_value is not None:
            orphans.append(i)
            symmetric_orphans[i] = sym_value
        elif _orphan_pat.match(text) and not dim.get('_merged'):
            orphans.append(i)
            explicit_orphans.add(i)
            tolerance_fragments.append((i, dim, _normalize_signed_tolerance_text(text)[0]))
        elif _is_unsigned_tol_fragment(dim) and not dim.get('_merged'):
            if (_unsigned_dim_has_signed_tol_parent_signal(dim, explicit_signed_tols)
                    and not _unsigned_fragment_has_left_parent_pair_signal(
                        dim,
                        explicit_signed_tols,
                        dimensions,
                    )):
                parents.append(i)
            else:
                orphans.append(i)
                tolerance_fragments.append((i, dim, None))
        # 父标称值：包含数字、不以+/-开头、不是纯字母
        elif (re.search(r'\d', text)
              and not text.startswith(('+', '-'))
              and not re.match(r'^[a-zA-Z\s]+$', text)
              and dim.get('source') != 'gdt_line_frame'):
            parents.append(i)

    if not tolerance_fragments and not symmetric_orphans:
        return dimensions

    # 排除页面边缘的短整数宿主，避免页码等标号吸收公差。
    if page_width > 0 and page_height > 0:
        _valid_parents = []
        for pi in parents:
            p = dimensions[pi]
            pb = p.get('bbox', {})
            pcx = pb.get('x', 0) + pb.get('w', 0) / 2
            pcy = pb.get('y', 0) + pb.get('h', 0) / 2
            _in_edge = (pcx < page_width * 0.05 or pcx > page_width * 0.95 or
                        pcy < page_height * 0.05 or pcy > page_height * 0.95)
            if _in_edge:
                _pnom = p.get('nominal', '').strip()
                _ptext = p.get('text', '').strip()
                if re.match(r'^\d{1,3}$', _pnom or _ptext):
                    if VERBOSE:
                        print(f"[s27.orphan.parent_drop] edge short-int '{_ptext}' at ({pcx:.0f},{pcy:.0f})")
                    continue
            _valid_parents.append(pi)
        parents = _valid_parents

    if VERBOSE:
        print(f"[orphan_tol] orphans={len(orphans)}, parents={len(parents)}")

    # === 预配对：空间上紧邻的 +/- 对 ===
    plus_orphans = [(i, dimensions[i]) for i in explicit_orphans if dimensions[i]['text'].strip().startswith('+')]
    minus_orphans = [(i, dimensions[i]) for i in explicit_orphans if dimensions[i]['text'].strip().startswith('-')]
    paired_groups = []
    used_indices = set()
    pair_candidates = []

    # Build sign-aware pair candidates. Priority keeps explicit +/- pairs first,
    # then one inferred sign, then two inferred signs.
    for left_pos, (li, left_dim, left_sign) in enumerate(tolerance_fragments):
        lcx, lcy = _bbox_center(left_dim['bbox'])
        for ri, right_dim, right_sign in tolerance_fragments[left_pos + 1:]:
            rcx, rcy = _bbox_center(right_dim['bbox'])
            dx = abs(lcx - rcx)
            dy = abs(lcy - rcy)
            if dx >= 30 or dy >= 25:
                continue
            inferred = _infer_tolerance_pair(
                li, left_dim, left_sign,
                ri, right_dim, right_sign,
            )
            if inferred is None:
                continue
            pi, po, mi, mo, priority = inferred
            pair_candidates.append((priority, math.hypot(dx, dy), pi, po, mi, mo))

    for _priority, _dist, pi, po, mi, mo in sorted(pair_candidates, key=lambda t: (t[0], t[1])):
        if pi in used_indices or mi in used_indices:
            continue
        paired_groups.append((pi, po, mi, mo))
        used_indices.add(mi)
        used_indices.add(pi)

    # Legacy fallback for any explicit pair that slipped through inference.
    for pi, po in plus_orphans:
        if pi in used_indices:
            continue
        pcx, pcy = _bbox_center(po['bbox'])
        for mi, mo in minus_orphans:
            if mi in used_indices or mi == pi:
                continue
            mcx, mcy = _bbox_center(mo['bbox'])
            dx = abs(pcx - mcx)
            dy = abs(pcy - mcy)
            if dx < 30 and dy < 25:
                paired_groups.append((pi, po, mi, mo))
                used_indices.add(mi)
                used_indices.add(pi)
                break

    if VERBOSE:
        print(f"[orphan_tol] pre-paired {len(paired_groups)} tolerance pairs")

    merged_indices = set()

    # ── 预配对 +/- 处理：方向感知搜 parent ─────────────────────────
    for pi, plus_o, mi, minus_o in paired_groups:
        layout = _classify_pair_layout(plus_o, minus_o)
        pair_bbox = _union_bbox([plus_o['bbox'], minus_o['bbox']])

        # 收集所有方向合格的候选 parent
        candidates = []  # list of (score, dist, parent_idx)
        for ppi in parents:
            if ppi in merged_indices:
                continue
            p = dimensions[ppi]
            if p.get('upper_tol') or p.get('lower_tol'):
                continue
            if not _tol_reasonable_for_parent(
                    p, [plus_o.get('text', ''), minus_o.get('text', '')]):
                if VERBOSE:
                    print(f"  [pair_search] reject tol_unreasonable vs '{p.get('nominal','')}'")
                continue
            ok, dist = _parent_ok_for_pair(pair_bbox, p['bbox'], layout)
            score = _pair_parent_score(pair_bbox, plus_o, minus_o, p, layout, dist) if ok else dist
            if VERBOSE:
                print(f"  [pair_search] pair layout={layout} vs '{p.get('nominal','')}' "
                      f"dist={dist:.1f} score={score:.1f} dir_ok={ok}")
            if ok:
                candidates.append((score, dist, ppi))

        pick = _pick_best_with_ambig_guard(candidates)
        if pick is None:
            if VERBOSE:
                if not candidates:
                    print(f"  [pair_search] FAILED: no direction-compatible parent (layout={layout})")
                else:
                    b_score, b_dist, b_idx = candidates[0]
                    s_score = candidates[1][0]
                    pn = dimensions[b_idx].get('nominal', '')
                    rel_gap = (s_score - b_score) / max(abs(b_score), 1.0)
                    print(f"  [pair_ambig] pair layout={layout} -> "
                          f"REJECTED: best='{pn}' score={b_score:.1f} d={b_dist:.1f} "
                          f"second_score={s_score:.1f} rel_gap={rel_gap:.2f}")
            continue
        best_dist, best_parent_idx = pick

        # 合并
        parent = dimensions[best_parent_idx]
        upper_val = _normalize_signed_tolerance_text(plus_o.get('text', ''))
        lower_val = _normalize_signed_tolerance_text(minus_o.get('text', ''))
        parent['upper_tol'] = upper_val
        parent['lower_tol'] = lower_val
        _sync_tol_into_text(parent)
        parent['bbox'] = _union_bbox([parent['bbox'], plus_o['bbox'], minus_o['bbox']])
        merged_indices.add(pi)
        merged_indices.add(mi)
        if VERBOSE:
            print(f"  paired '{upper_val}'/'{lower_val}' -> parent '{parent.get('nominal', '')}' "
                  f"layout={layout} dist={best_dist:.1f}")

    # ── 对称 ± orphan 处理：同样按方向/字号/朝向找 parent ─────────────
    for oi, sym_value in symmetric_orphans.items():
        if oi in used_indices:
            continue
        orphan = dimensions[oi]
        ob = orphan['bbox']
        upper_val = f"+{sym_value}"
        lower_val = f"-{sym_value}"

        candidates = []
        for pi in parents:
            if pi in merged_indices:
                continue
            if pi == oi:
                continue
            p = dimensions[pi]
            if p.get('upper_tol') or p.get('lower_tol'):
                continue
            if not _tol_reasonable_for_parent(p, [upper_val, lower_val]):
                continue
            ok, dist = _parent_ok_for_single(ob, p['bbox'])
            if ok:
                candidates.append((_single_parent_score(orphan, p, dist), dist, pi))

        pick = _pick_best_with_ambig_guard(candidates)
        if pick is None:
            if VERBOSE:
                if not candidates:
                    print(f"  [sym_search] orphan '{orphan.get('text', '').strip()}' "
                          f"-> no direction-compatible parent")
                else:
                    b_score, b_dist, b_idx = candidates[0]
                    s_score = candidates[1][0]
                    pn = dimensions[b_idx].get('nominal', '')
                    rel_gap = (s_score - b_score) / max(abs(b_score), 1.0)
                    print(f"  [sym_ambig] orphan '{orphan.get('text', '').strip()}' -> "
                          f"REJECTED: best='{pn}' score={b_score:.1f} d={b_dist:.1f} "
                          f"second_score={s_score:.1f} rel_gap={rel_gap:.2f}")
            continue

        best_dist, best_parent = pick
        parent = dimensions[best_parent]
        parent['upper_tol'] = upper_val
        parent['lower_tol'] = lower_val
        parent['bbox'] = _union_bbox([parent['bbox'], orphan['bbox']])
        merged_indices.add(oi)
        _sync_tol_into_text(parent)
        if VERBOSE:
            print(f"  symmetric orphan '±{sym_value}' -> parent '{parent.get('nominal', '')}' "
                  f"dist={best_dist:.1f}")

    # ── 单 orphan 处理：按 orphan 长轴判方向 ──────────────────────
    remaining_orphans = [oi for oi in explicit_orphans if oi not in used_indices]
    for oi in remaining_orphans:
        orphan = dimensions[oi]
        ob = orphan['bbox']

        candidates = []
        for pi in parents:
            if pi in merged_indices:
                continue
            p = dimensions[pi]
            tol_text = _normalize_signed_tolerance_text(orphan.get('text', ''))
            if tol_text.startswith('+') and p.get('upper_tol'):
                continue
            if tol_text.startswith('-') and p.get('lower_tol'):
                continue
            if not _tol_reasonable_for_parent(p, [tol_text]):
                continue
            ok, dist = _parent_ok_for_single(ob, p['bbox'])
            if ok:
                candidates.append((_single_parent_score(orphan, p, dist), dist, pi))

        pick = _pick_best_with_ambig_guard(candidates)
        if pick is None:
            if VERBOSE:
                if not candidates:
                    print(f"  [single_search] orphan '{orphan['text'].strip()}' "
                          f"-> no direction-compatible parent")
                else:
                    b_score, b_dist, b_idx = candidates[0]
                    s_score = candidates[1][0]
                    pn = dimensions[b_idx].get('nominal', '')
                    rel_gap = (s_score - b_score) / max(abs(b_score), 1.0)
                    print(f"  [single_ambig] orphan '{orphan['text'].strip()}' -> "
                          f"REJECTED: best='{pn}' score={b_score:.1f} d={b_dist:.1f} "
                          f"second_score={s_score:.1f} rel_gap={rel_gap:.2f}")
            continue
        best_dist, best_parent = pick

        parent = dimensions[best_parent]
        tol_text = _normalize_signed_tolerance_text(orphan.get('text', ''))
        if tol_text.startswith('+') and not parent.get('upper_tol'):
            parent['upper_tol'] = tol_text
            parent['bbox'] = _union_bbox([parent['bbox'], orphan['bbox']])
            merged_indices.add(oi)
            _sync_tol_into_text(parent)
            if VERBOSE:
                print(f"  single orphan '{tol_text}' -> parent '{parent.get('nominal', '')}' "
                      f"dist={best_dist:.1f}")
        elif tol_text.startswith('-') and not parent.get('lower_tol'):
            parent['lower_tol'] = tol_text
            parent['bbox'] = _union_bbox([parent['bbox'], orphan['bbox']])
            merged_indices.add(oi)
            _sync_tol_into_text(parent)
            if VERBOSE:
                print(f"  single orphan '{tol_text}' -> parent '{parent.get('nominal', '')}' "
                      f"dist={best_dist:.1f}")

    # 只有父尺寸已有上公差且缺少下公差时，才尝试补入邻近的无符号下公差。
    
    
    
    remaining_unsigned_orphans = [
        oi for oi, _dim, sign in tolerance_fragments
        if sign is None and oi not in used_indices and oi not in merged_indices
    ]
    for oi in remaining_unsigned_orphans:
        orphan = dimensions[oi]
        value = _unsigned_tol_value(orphan.get('text', ''))
        if value is None:
            continue
        lower_val = f'-{value}'
        ob = orphan['bbox']

        candidates = []
        for pi in parents:
            if pi in merged_indices or pi == oi:
                continue
            p = dimensions[pi]
            if not p.get('upper_tol') or p.get('lower_tol'):
                continue
            if not _tol_reasonable_for_parent(p, [lower_val]):
                continue
            ok, dist = _parent_ok_for_single(ob, p['bbox'])
            if ok:
                candidates.append((_single_parent_score(orphan, p, dist), dist, pi))

        pick = _pick_best_with_ambig_guard(candidates)
        if pick is None:
            if VERBOSE:
                print(f"  [unsigned_lower_search] orphan '{orphan['text'].strip()}' "
                      f"-> no unambiguous upper-tol parent")
            continue

        best_dist, best_parent = pick
        parent = dimensions[best_parent]
        parent['lower_tol'] = lower_val
        parent['bbox'] = _union_bbox([parent['bbox'], orphan['bbox']])
        merged_indices.add(oi)
        _sync_tol_into_text(parent)
        if VERBOSE:
            print(f"  unsigned lower orphan '{lower_val}' -> parent '{parent.get('nominal', '')}' "
                  f"dist={best_dist:.1f}")

    if VERBOSE:
        _surv_orphans = [oi for oi in orphans if oi not in merged_indices]
        print(f"[s27.orphan] EXIT: merged={len(merged_indices)} survived={len(_surv_orphans)}")
        for _oi in _surv_orphans:
            _d = dimensions[_oi]
            _t = _d.get('text', '').strip()
            _b = _d.get('bbox', {})
            print(f"[s27.orphan.survived] [{_oi}] text='{_t}' bbox=({_b.get('x',0):.0f},{_b.get('y',0):.0f},{_b.get('w',0):.0f}x{_b.get('h',0):.0f}) -> will become fake dim unless filtered later")

    if merged_indices:
        dimensions = [d for i, d in enumerate(dimensions) if i not in merged_indices]
    return dimensions


# ---------------------------------------------------------------------------
# 公差升级辅助
# ---------------------------------------------------------------------------

def _upgrade_tol_if_safe(existing: dict, candidate: dict) -> None:
    """去重时仅在公称值相同且已有项缺少公差时回填候选公差；空值或不一致的公称值不执行升级。"""
    nom_c = re.sub(r'\s+', '', candidate.get('nominal', '') or '')
    nom_e = re.sub(r'\s+', '', existing.get('nominal', '') or '')
    if not nom_c or nom_c != nom_e:
        return
    c_ut = candidate.get('upper_tol')
    c_lt = candidate.get('lower_tol')
    e_ut = existing.get('upper_tol')
    e_lt = existing.get('lower_tol')
    if (c_ut or c_lt) and not (e_ut or e_lt):
        existing['upper_tol'] = c_ut
        existing['lower_tol'] = c_lt
        existing['text'] = candidate.get('text', existing['text'])


def _dedup_nominal_key(dim: dict) -> str | None:
    text = (dim.get('nominal') or dim.get('text') or '').strip()
    if not text:
        return None
    prefix = (dim.get('prefix') or '').strip().upper()
    if not prefix:
        m = re.match(r'^\s*(\d+\s*[xX×]\s*)?([R⌀∅Ø⊘φΦ])', dim.get('text', ''))
        if m:
            prefix = (m.group(2) or '').upper()
    number = re.search(r'-?\d+(?:\.\d+)?', text)
    if not number:
        return None
    try:
        value = float(number.group(0))
    except ValueError:
        return None
    return f"{prefix}:{value:g}"


def _has_distinct_nominal(candidate: dict, existing: dict) -> bool:
    cand_key = _dedup_nominal_key(candidate)
    exist_key = _dedup_nominal_key(existing)
    return bool(cand_key and exist_key and cand_key != exist_key)


def _dedup_family(dim: dict) -> str:
    text = (dim.get('text') or '').strip()
    prefix = (dim.get('prefix') or '').strip().upper()
    dim_type = (dim.get('type') or '').strip().lower()
    if '°' in text or dim_type == 'angle':
        return 'angle'
    if prefix == 'R' or dim_type == 'radius' or re.match(r'^\s*R', text, re.IGNORECASE):
        return 'radius'
    return 'other'


def _same_tight_family(candidate: dict, existing: dict) -> bool:
    family = _dedup_family(candidate)
    return family in {'angle', 'radius'} and family == _dedup_family(existing)


def _same_family_dist_threshold(candidate: dict, existing: dict, max_side: float,
                                fallback: float) -> float:
    if _same_tight_family(candidate, existing):
        return max(max_side * 0.55, 24.0)
    return fallback


# ---------------------------------------------------------------------------
# IoU / 文本 / nominal 去重
# ---------------------------------------------------------------------------

def _dedup_dimensions(dims: list) -> list:
    """按置信度降序执行贪心去重，结合重叠、中心距离、文字和公称值关系判断重复。"""
    # confidence 降序排列，确保高置信度项优先保留
    sorted_dims = sorted(dims, key=lambda d: _CONF_ORDER.get(d['confidence'], 0), reverse=True)
    kept = []
    for candidate in sorted_dims:
        bbox_c = candidate['bbox']
        cand_src = candidate.get('source', '')
        duplicate = False
        for existing in kept:
            # 条件 A：标准 IoU（适合正常大小 bbox）
            if _bbox_iou(bbox_c, existing['bbox']) > 0.3:
                if _has_distinct_nominal(candidate, existing):
                    continue
                _upgrade_tol_if_safe(existing, candidate)
                duplicate = True
                break
            # 条件 B：中心距 + 文本匹配（修复窄 bbox IoU 失效）
            # 豁免 gdt_line_frame 之间的文本匹配去重：不同位置的 GD&T 框是不同特征，
            # 即使 compartment OCR 失败后都显示 "⊕ (unread)" 也不应合并
            exist_src = existing.get('source', '')
            if cand_src == 'gdt_line_frame' and exist_src == 'gdt_line_frame':
                continue
            cx_c = bbox_c['x'] + bbox_c['w'] / 2
            cy_c = bbox_c['y'] + bbox_c['h'] / 2
            bx = existing['bbox']
            cx_e = bx['x'] + bx['w'] / 2
            cy_e = bx['y'] + bx['h'] / 2
            dist = math.hypot(cx_c - cx_e, cy_c - cy_e)
            max_side = max(bbox_c['w'], bbox_c['h'], bx['w'], bx['h'])
            t_c = re.sub(r'\s+', '', candidate.get('text', ''))
            t_e = re.sub(r'\s+', '', existing.get('text', ''))
            lev_dist = _levenshtein(t_c, t_e)
            angle_key_c = _angle_nominal_key(t_c)
            angle_key_e = _angle_nominal_key(t_e)
            angle_nominal_conflict = (
                angle_key_c is not None
                and angle_key_e is not None
                and angle_key_c != angle_key_e
            )
            dist_threshold = max(max_side * 1.25, 45) if lev_dist == 0 else max_side
            dist_threshold = _same_family_dist_threshold(
                candidate, existing, max_side, dist_threshold)
            if (not angle_nominal_conflict
                    and not _has_distinct_nominal(candidate, existing)
                    and dist < dist_threshold and lev_dist <= 2):
                _upgrade_tol_if_safe(existing, candidate)
                duplicate = True
                break
            # 在局部距离范围内按相同公称值去重。
            nom_c = re.sub(r'\s+', '', candidate.get('nominal', ''))
            nom_e = re.sub(r'\s+', '', existing.get('nominal', ''))
            if nom_c and nom_e and nom_c == nom_e:
                # 处理容器中分裂读取产生的重复碎片。
                if (candidate.get('source') == 'geometric_anchor'
                        and existing.get('source') == 'geometric_anchor'
                        and dist < 150):
                    c_tols = {k: candidate.get(k) for k in ('upper_tol', 'lower_tol')
                              if candidate.get(k)}
                    e_tols = {k: existing.get(k) for k in ('upper_tol', 'lower_tol')
                              if existing.get(k)}
                    if (c_tols and e_tols
                            and set(c_tols.keys()).issubset(set(e_tols.keys()))
                            and all(c_tols[k] == e_tols.get(k) for k in c_tols)):
                        duplicate = True  # candidate 是 existing 的子集
                        break
                    if (c_tols and e_tols
                            and set(e_tols.keys()).issubset(set(c_tols.keys()))
                            and all(e_tols[k] == c_tols.get(k) for k in e_tols)):
                        # existing 是 candidate 的子集 -- 就地升级 existing
                        existing['upper_tol'] = candidate.get('upper_tol')
                        existing['lower_tol'] = candidate.get('lower_tol')
                        existing['text'] = candidate.get('text', existing['text'])
                        duplicate = True
                        break
                # 限制相同公称值的去重邻域，保留真实重复标注。
                
                nominal_threshold = _same_family_dist_threshold(
                    candidate, existing, max_side, max(max_side, 50))
                if dist < nominal_threshold:
                    _upgrade_tol_if_safe(existing, candidate)
                    duplicate = True
                    break
        if not duplicate:
            kept.append(candidate)

    if VERBOSE:
        kept_ids = {id(d) for d in kept}
        gdt_dropped = [d for d in dims
                       if d.get('source') == 'gdt_line_frame' and id(d) not in kept_ids]
        if gdt_dropped:
            print(f"[gdt_dedup] dropped {len(gdt_dropped)} gdt_line_frame dims")
            for d in gdt_dropped[:20]:
                bb = d.get('bbox', {})
                print(f"[gdt_dedup] DROP text={d.get('text','')!r} "
                      f"bbox=({bb.get('x',0):.0f},{bb.get('y',0):.0f},{bb.get('w',0):.0f}x{bb.get('h',0):.0f})")

    return kept


# ---------------------------------------------------------------------------
# 字号配对
# ---------------------------------------------------------------------------

def _font_pair_tol_like(dim: dict) -> bool:
    text = (dim.get('text') or dim.get('nominal') or '').strip()
    return bool(re.match(r'^[±]\s*\d', text) or re.match(r'^[+\-]\s*\.?\d', text))


def _font_pair_tol_text(dim: dict) -> str:
    text = (dim.get('text') or '').strip()
    if _font_pair_tol_like({'text': text}):
        return text
    return (dim.get('nominal') or text).strip()


def _font_pair_parent_like(dim: dict) -> bool:
    text = (dim.get('text') or '').strip()
    nominal = (dim.get('nominal') or '').strip()
    if _font_pair_tol_like(dim):
        return False
    if not re.search(r'\d', nominal or text):
        return False
    if re.match(r'^(?:MIN|MAX|WITH|ALL|PLANE|RIBS)\b', text, re.IGNORECASE):
        return False
    if re.search(r'[\u4e00-\u9fff]{2,}', text):
        return False
    return bool(
        dim.get('prefix') or
        re.match(r'^(?:\d+\s*[xX×]\s*)?(?:R|RO|⌀|∅|Ø|O)?\s*\d', text, re.IGNORECASE) or
        re.match(r'^\d+(?:\.\d+)?$', nominal)
    )


def _font_size_pairing(dimensions: list) -> list:
    """
    对尚未配对公差的尺寸候选，按字号代理值（_font_height）分层，
    尝试将 S 层小字合并到附近 M 层标称值。
    """
    # 只对 source='ocr' 且未配对的候选做配对
    candidates = [
        d for d in dimensions
        if d.get('source') == 'ocr'
        and d.get('_font_height', 0) > 0
        and (
            _font_pair_tol_like(d) or
            (not d.get('upper_tol') and not d.get('lower_tol'))
        )
    ]

    if VERBOSE:
        print(f"[s27.fontpair] ENTRY: total_dims={len(dimensions)} candidates(ocr,unpaired,fh>0)={len(candidates)}")

    if not candidates:
        if VERBOSE:
            print(f"[s27.fontpair] no candidates, skip")
        return dimensions

    heights = [d['_font_height'] for d in candidates]
    try:
        median_h = statistics.median(heights)
    except statistics.StatisticsError:
        if VERBOSE:
            print(f"[s27.fontpair] median failed, skip")
        return dimensions

    # 分层
    for d in candidates:
        h = d['_font_height']
        if _font_pair_tol_like(d):
            d['_size_layer'] = 'S'
        elif h > median_h * 1.4:
            d['_size_layer'] = 'L'
        elif h < median_h * 0.75:
            d['_size_layer'] = 'S'
        else:
            d['_size_layer'] = 'M'

    m_dims = [
        d for d in candidates
        if d.get('_size_layer') == 'M' or (
            d.get('_size_layer') == 'L' and _font_pair_parent_like(d)
        )
    ]
    s_dims = [d for d in candidates if d.get('_size_layer') == 'S']
    used_s = set()

    if VERBOSE:
        print(f"[s27.fontpair] median_h={median_h:.2f} L={sum(1 for d in candidates if d.get('_size_layer')=='L')} M={len(m_dims)} S={len(s_dims)}")
        for _i, _d in enumerate(s_dims):
            _b = _d['bbox']
            _or = _d.get('orientation', 0)
            print(f"[s27.fontpair.S] [{_i}] text='{_d.get('text','').strip()}' bbox=({_b['x']:.0f},{_b['y']:.0f}) fh={_d.get('_font_height',0):.1f} orient={_or}")
        for _i, _d in enumerate(m_dims):
            _b = _d['bbox']
            _or = _d.get('orientation', 0)
            print(f"[s27.fontpair.M] [{_i}] text='{_d.get('text','').strip()}' bbox=({_b['x']:.0f},{_b['y']:.0f},{_b['w']:.0f}x{_b['h']:.0f}) fh={_d.get('_font_height',0):.1f} orient={_or}")

    # --- 竖直文本预处理 pass ---
    for si, sd in enumerate(s_dims):
        if si in used_s:
            continue
        s_orient = sd.get('orientation', 0)
        if not _is_vertical_orient(s_orient):
            continue

        s_cx, s_cy = _bbox_center(sd['bbox'])
        best_mi = None
        best_dy = float('inf')

        for mi, md_v in enumerate(m_dims):
            m_orient = md_v.get('orientation', 0)
            if not _is_vertical_orient(m_orient):
                continue
            if not re.search(r'\d', md_v.get('nominal', '')):
                continue

            m_cx, m_cy = _bbox_center(md_v['bbox'])
            bh = max(sd.get('_font_height', 8), md_v.get('_font_height', 8))

            dx = abs(m_cx - s_cx)
            dy = abs(m_cy - s_cy)

            if dx < bh * 0.5 and dy < bh * 1.5:
                if dy < best_dy:
                    best_mi = mi
                    best_dy = dy

        if best_mi is not None:
            md_v = m_dims[best_mi]
            st = _font_pair_tol_text(sd)

            _tol_ok = True
            try:
                _tv = abs(float(re.sub(r'[^\d.\-]', '', st)))
                _nv = abs(float(re.sub(r'[^\d.\-]', '', md_v.get('nominal', ''))))
                _limit = max(_nv * 2.0, 0.5) if _nv < 5 else max(_nv * 0.5, 3.0)
                if _tv >= _limit:
                    _tol_ok = False
            except (ValueError, ZeroDivisionError):
                pass
            if not _tol_ok:
                if VERBOSE:
                    print(f"[s27.fontpair.vert] REJECT tol_unreasonable: S='{st}' M='{md_v.get('nominal','')}' dy={best_dy:.1f}")
                continue

            m_sym = re.match(r'^[±]\s*([\d.]+)$', st)
            if m_sym:
                v = m_sym.group(1)
                md_v['upper_tol'] = f"+{v}"
                md_v['lower_tol'] = f"-{v}"
                md_v['text'] = f"{md_v.get('prefix') or ''}{md_v['nominal']} \u00b1{v}".strip()
                md_v['bbox'] = _union_bbox([md_v['bbox'], sd['bbox']])
                used_s.add(si)
                if VERBOSE:
                    print(f"[s27.fontpair.vert] PAIR_SYM: S='{st}' -> M='{md_v.get('nominal','')}' dy={best_dy:.1f}")
                continue

            m_stack = re.match(r'^([+\-][\d.]+)$', st)
            if m_stack:
                val = m_stack.group(1)
                if val.startswith('+') and not md_v.get('upper_tol'):
                    md_v['upper_tol'] = val
                    md_v['bbox'] = _union_bbox([md_v['bbox'], sd['bbox']])
                    _sync_tol_into_text(md_v)
                    used_s.add(si)
                    if VERBOSE:
                        print(f"[s27.fontpair.vert] PAIR_UPPER: S='{st}' -> M='{md_v.get('nominal','')}' dy={best_dy:.1f}")
                elif val.startswith('-') and not md_v.get('lower_tol'):
                    md_v['lower_tol'] = val
                    md_v['bbox'] = _union_bbox([md_v['bbox'], sd['bbox']])
                    _sync_tol_into_text(md_v)
                    used_s.add(si)
                    if VERBOSE:
                        print(f"[s27.fontpair.vert] PAIR_LOWER: S='{st}' -> M='{md_v.get('nominal','')}' dy={best_dy:.1f}")
            else:
                if VERBOSE:
                    print(f"[s27.fontpair.vert] REJECT no_stack_match: S='{st}' M='{md_v.get('nominal','')}'")
        else:
            if VERBOSE:
                print(f"[s27.fontpair.vert] NO_M_FOUND: S='{sd.get('text','').strip()}' at ({s_cx:.0f},{s_cy:.0f})")

    # 执行水平字号配对。
    for md in m_dims:
        if not re.search(r'\d', md.get('nominal', '')):
            continue

        bh = md['bbox']['h'] if md['bbox']['h'] > 0 else md['_font_height']
        bw = md['bbox']['w'] if md['bbox']['w'] > 0 else 20

        for si, sd in enumerate(s_dims):
            if si in used_s:
                continue
            if _is_vertical_orient(sd.get('orientation', 0)):
                continue
            dx = sd['bbox']['x'] - (md['bbox']['x'] + md['bbox']['w'])
            dy_center = _bbox_center(sd['bbox'])[1] - _bbox_center(md['bbox'])[1]

            if not (-bw * 0.3 <= dx <= bw * 1.5 and abs(dy_center) <= bh * 1.5):
                continue

            st = _font_pair_tol_text(sd)

            try:
                _tv = abs(float(re.sub(r'[^\d.\-]', '', st)))
                _nv = abs(float(re.sub(r'[^\d.\-]', '', md.get('nominal', ''))))
                _limit = max(_nv * 2.0, 0.5) if _nv < 5 else max(_nv * 0.5, 3.0)
                if _tv >= _limit:
                    continue
            except (ValueError, ZeroDivisionError):
                pass

            m_sym = re.match(r'^[±]\s*([\d.]+)$', st)
            if m_sym:
                v = m_sym.group(1)
                md['upper_tol'] = f"+{v}"
                md['lower_tol'] = f"-{v}"
                md['text'] = f"{md.get('prefix') or ''}{md['nominal']} \u00b1{v}".strip()
                md['bbox'] = _union_bbox([md['bbox'], sd['bbox']])
                used_s.add(si)
                if VERBOSE:
                    print(f"[s27.fontpair.horiz] PAIR_SYM: S='{st}' -> M='{md.get('nominal','')}' dx={dx:.1f} dy={dy_center:.1f}")
                continue

            m_stack = re.match(r'^([+\-][\d.]+)$', st)
            if m_stack:
                val = m_stack.group(1)
                if val.startswith('+') and not md.get('upper_tol'):
                    md['upper_tol'] = val
                    md['bbox'] = _union_bbox([md['bbox'], sd['bbox']])
                    _sync_tol_into_text(md)
                    used_s.add(si)
                    if VERBOSE:
                        print(f"[s27.fontpair.horiz] PAIR_UPPER: S='{st}' -> M='{md.get('nominal','')}' dx={dx:.1f} dy={dy_center:.1f}")
                elif val.startswith('-') and not md.get('lower_tol'):
                    md['lower_tol'] = val
                    md['bbox'] = _union_bbox([md['bbox'], sd['bbox']])
                    _sync_tol_into_text(md)
                    used_s.add(si)
                    if VERBOSE:
                        print(f"[s27.fontpair.horiz] PAIR_LOWER: S='{st}' -> M='{md.get('nominal','')}' dx={dx:.1f} dy={dy_center:.1f}")

    # 将 S 层已配对的候选从 dimensions 中移除
    paired_s = {id(s_dims[i]) for i in used_s}
    result = [d for d in dimensions if id(d) not in paired_s]
    if VERBOSE:
        print(f"[s27.fontpair] EXIT: paired_s={len(used_s)}/{len(s_dims)} result={len(result)}")
    return result
