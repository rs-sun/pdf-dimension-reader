"""
pdf_analyzer_type_a.py — Type A PDF 提取（有原生文字对象）

extract_type_a() 入口 + 全部 helper 函数。
独立模块，极少修改。从 pdf_analyzer.py 拆出。
"""

import math
import re
import statistics

from pdf_analyzer_utils import SYMBOL_MAP, PREFIX_PATTERNS


def extract_type_a(page):
    """
    从 Type A PDF 页面（有原生文字）提取尺寸信息。

    返回:
    {
      "pdf_type": "A",
      "page_width": float,
      "page_height": float,
      "dimensions": [...],
      "references": [...],
      "gdt_frames": [...],
      "raw_chars": [...],
      "raw_lines": [...],
    }
    """
    page_w = page.width
    page_h = page.height

    # 1. 提取并清洗字符
    raw_chars = _extract_and_map_chars(page)

    # 2. 提取尺寸线（L1 层细线）
    raw_lines = _extract_lines_pdfplumber(page)

    # 3. 字符聚类 -> 文本块
    text_blocks = _cluster_chars_into_blocks(raw_chars)

    # 4. 字号分层
    text_blocks = _layer_by_fontsize(text_blocks)

    # 5. 标称值-公差配对 + 前缀吸附
    dimensions = _pair_nominal_tolerance(text_blocks)

    # 6. 基准符号和 GD&T（简单版，后续 assembler.py 会细化）
    references = _extract_datum_refs(text_blocks)
    gdt_frames = _extract_gdt_frames_from_rects(page)

    return {
        "pdf_type": "A",
        "page_width": page_w,
        "page_height": page_h,
        "dimensions": dimensions,
        "references": references,
        "gdt_frames": gdt_frames,
        "raw_chars": raw_chars,
        "raw_lines": raw_lines,
    }


def _extract_and_map_chars(page):
    """提取字符，做符号映射，标记 (cid:N)。"""
    result = []
    for ch in page.chars:
        c = ch.get('text', '')
        # 符号映射
        mapped = SYMBOL_MAP.get(c, c)
        # 检测 (cid:N) 未知字形
        is_unknown = bool(re.match(r'^\(cid:\d+\)$', c))
        result.append({
            'char': mapped,
            'original': c,
            'unknown_symbol': is_unknown,
            'x': ch.get('x0', 0),
            'y': ch.get('top', 0),
            'x1': ch.get('x1', 0),
            'y1': ch.get('bottom', 0),
            'size': ch.get('size', 0),
            'fontname': ch.get('fontname', ''),
            'orientation': _char_orientation(ch),
        })
    return result


def _char_orientation(ch):
    """Infer pdfplumber char orientation in degrees when matrix data exists."""
    matrix = ch.get('matrix')
    if matrix and len(matrix) >= 2:
        try:
            angle = math.degrees(math.atan2(float(matrix[1]), float(matrix[0])))
            while angle <= -180:
                angle += 360
            while angle > 180:
                angle -= 360
            if abs(angle) < 1:
                return 0.0
            return round(angle, 3)
        except (TypeError, ValueError):
            pass
    if ch.get('upright') is False:
        return 90.0
    return 0.0


def _extract_lines_pdfplumber(page):
    """从 pdfplumber 提取线段，不做宽度过滤（Type A 用于参考）。"""
    lines = []
    for ln in page.lines:
        lines.append({
            'x0': ln.get('x0', 0),
            'y0': ln.get('top', 0),
            'x1': ln.get('x1', 0),
            'y1': ln.get('bottom', 0),
            'lineWidth': ln.get('linewidth', ln.get('width', 0)),
        })
    return lines


def _cluster_chars_into_blocks(chars):
    """
    将字符按空间邻近性聚成文本块。
    同一块内的字符水平间距 < 字符宽度x1.5，或竖向间距 < 字符高度x1.5。
    """
    if not chars:
        return []

    # 按 y 坐标分行，再按 x 坐标排序
    sorted_chars = sorted(chars, key=lambda c: (round(c['y'] / 2) * 2, c['x']))

    blocks = []
    current_block = [sorted_chars[0]]

    for ch in sorted_chars[1:]:
        prev = current_block[-1]
        ch_w = max(abs(ch['x1'] - ch['x']), 1)
        ch_h = max(abs(ch['y1'] - ch['y']), 1)
        dx = ch['x'] - prev['x1']
        dy = abs(ch['y'] - prev['y'])

        # 同行逻辑
        if dy < ch_h * 1.5 and dx < ch_w * 2.0 and dx > -ch_w * 0.5:
            current_block.append(ch)
        else:
            blocks.append(_make_block(current_block))
            current_block = [ch]

    if current_block:
        blocks.append(_make_block(current_block))

    return blocks


def _make_block(chars):
    """从字符列表构造文本块字典。"""
    text = ''.join(c['char'] for c in chars)
    x0 = min(c['x'] for c in chars)
    y0 = min(c['y'] for c in chars)
    x1 = max(c['x1'] for c in chars)
    y1 = max(c['y1'] for c in chars)
    sizes = [c['size'] for c in chars if c['size'] > 0]
    avg_size = statistics.mean(sizes) if sizes else 0
    fontnames = sorted({c['fontname'] for c in chars if c.get('fontname')})
    orientations = [c.get('orientation', 0.0) for c in chars]
    orientation = statistics.median(orientations) if orientations else 0.0
    has_unknown = any(c['unknown_symbol'] for c in chars)
    return {
        'text': text,
        'x': x0, 'y': y0, 'x1': x1, 'y1': y1,
        'w': x1 - x0, 'h': y1 - y0,
        'fontsize': avg_size,
        'fontnames': fontnames,
        'orientation': orientation,
        'has_unknown_symbol': has_unknown,
        'layer': None,  # 填充 by _layer_by_fontsize
    }


def _layer_by_fontsize(blocks):
    """
    按字号将文本块分层：L（大，标题），M（中，标称值），S（小，公差）。
    使用分位数自适应分层。
    """
    if not blocks:
        return blocks

    sizes = [b['fontsize'] for b in blocks if b['fontsize'] > 0]
    if not sizes:
        for b in blocks:
            b['layer'] = 'M'
        return blocks

    median = statistics.median(sizes)
    for b in blocks:
        s = b['fontsize']
        if s > median * 1.5:
            b['layer'] = 'L'
        elif s < median * 0.8:
            b['layer'] = 'S'
        else:
            b['layer'] = 'M'

    return blocks


def _orientation_family(orientation):
    try:
        angle = float(orientation)
    except (TypeError, ValueError):
        return 'unknown'
    if abs(abs(angle) - 90.0) < 20.0:
        return 'vertical'
    if abs(angle) < 20.0 or abs(abs(angle) - 180.0) < 20.0:
        return 'horizontal'
    return 'angled'


def _type_a_tol_score(mb, sb, upper, lower, dx, dy_center):
    """Score a Type A tolerance block against a nominal block."""
    bh = mb['h'] if mb['h'] > 0 else 10
    score = abs(dx) + abs(dy_center) * 0.8

    m_family = _orientation_family(mb.get('orientation', 0))
    s_family = _orientation_family(sb.get('orientation', 0))
    if m_family != 'unknown' and s_family != 'unknown':
        if m_family == s_family:
            score -= 2.0
        elif m_family == 'vertical' and s_family == 'horizontal':
            score += 6.0
        else:
            score += 8.0

    m_size = mb.get('fontsize') or 0
    s_size = sb.get('fontsize') or 0
    if m_size > 0 and s_size > 0:
        ratio = s_size / m_size
        if 0.35 <= ratio <= 0.95:
            score -= 2.0
        elif ratio < 0.25 or ratio > 1.25:
            score += 6.0

    if upper and not lower and dy_center > bh * 0.35:
        score += 4.0
    if lower and not upper and dy_center < -bh * 0.35:
        score += 4.0
    if upper and lower and abs(dy_center) < bh * 0.7:
        score -= 1.0
    return score


def _pair_nominal_tolerance(blocks):
    """
    对 M 层文本块（候选标称值）在右侧和右上/右下搜索 S 层公差文本块，
    并做前缀吸附。
    返回 dimensions 列表。
    """
    dimensions = []
    m_blocks = [b for b in blocks if b['layer'] == 'M']
    s_blocks = [b for b in blocks if b['layer'] == 'S']
    used_s = set()

    # 按 M 层块筛选：过滤掉纯文字（多词英文），只保留含数字的
    for mb in m_blocks:
        text = mb['text'].strip()
        if not re.search(r'\d', text):
            continue
        # 检查是否是纯英文多词（标题/注释）
        if re.match(r'^[A-Za-z ]{5,}$', text):
            continue

        nominal_text, prefix, dim_type = _extract_prefix(text)
        if not nominal_text:
            continue

        # 搜索附近的 S 层公差块
        upper_tol = None
        lower_tol = None
        tol_text = None

        bh = mb['h'] if mb['h'] > 0 else 10
        bw = mb['w'] if mb['w'] > 0 else 20

        candidates = []
        for i, sb in enumerate(s_blocks):
            if i in used_s:
                continue
            # 搜索框：右侧和上下 +/-0.8 个字高
            dx = sb['x'] - mb['x1']
            dy_center = (sb['y'] + sb['y1']) / 2 - (mb['y'] + mb['y1']) / 2
            if -bw * 0.5 <= dx <= bw * 2.0 and abs(dy_center) <= bh * 1.5:
                st = sb['text'].strip()
                # 对称公差仅接受明确的 ±/# 符号；单独 +x 是上差，不是 ±x。
                if re.search(r'[±#]', st):
                    tol_val = re.sub(r'[±#\s]', '', st)
                    try:
                        v = float(tol_val)
                        ut = f"+{v}"
                        lt = f"-{v}"
                        score = _type_a_tol_score(mb, sb, ut, lt, dx, dy_center)
                        candidates.append((score, i, ut, lt, st))
                    except ValueError:
                        pass
                    continue
                # 堆叠公差（单独 + 或 -）
                m_tol = re.match(r'^([+\-][\d.]+)$', st)
                if m_tol:
                    val = m_tol.group(1)
                    ut = val if val.startswith('+') else None
                    lt = val if val.startswith('-') else None
                    score = _type_a_tol_score(mb, sb, ut, lt, dx, dy_center)
                    candidates.append((score, i, ut, lt, st))

        for _score, i, ut, lt, st in sorted(candidates, key=lambda item: (item[0], item[1])):
            if i in used_s:
                continue
            if ut and lt:
                if upper_tol or lower_tol:
                    continue
                upper_tol = ut
                lower_tol = lt
                tol_text = st
                used_s.add(i)
                break
            if ut:
                if upper_tol:
                    continue
                upper_tol = ut
                used_s.add(i)
            elif lt:
                if lower_tol:
                    continue
                lower_tol = lt
                used_s.add(i)
            if upper_tol and lower_tol:
                break  # 上下公差都找到了，停止搜索

        bbox = {'x': mb['x'], 'y': mb['y'], 'w': mb['w'], 'h': mb['h']}

        full_text = (prefix or '') + nominal_text
        if upper_tol and lower_tol:
            full_text += f" {upper_tol}/{lower_tol}"
        elif upper_tol:
            full_text += f" {upper_tol}"
        elif lower_tol:
            full_text += f" {lower_tol}"

        dim = {
            'text': full_text,
            'nominal': nominal_text,
            'upper_tol': upper_tol,
            'lower_tol': lower_tol,
            'type': dim_type,
            'prefix': prefix,
            'bbox': bbox,
            'confidence': 'high' if mb['fontsize'] > 0 else 'medium',
            'source': 'vector',
            'has_unknown_symbol': mb['has_unknown_symbol'],
            'orientation': mb.get('orientation', 0),
            '_font_height': mb.get('fontsize', 0),
            'fontnames': mb.get('fontnames', []),
        }
        dimensions.append(dim)

    return dimensions


def _extract_prefix(text):
    """
    从文本中提取工程前缀，返回 (cleaned_text, prefix_str, type_str)。
    """
    text = text.strip()
    for pat, sym, dtype in PREFIX_PATTERNS:
        m = pat.match(text)
        if m:
            remainder = text[m.end():]
            prefix = sym if sym else m.group(0)
            return remainder, prefix, dtype
    return text, None, 'linear'


def _extract_datum_refs(blocks):
    """提取基准符号文本块（单个大写字母）。"""
    refs = []
    for b in blocks:
        t = b['text'].strip()
        if re.match(r'^[A-Z]$', t) and b['layer'] == 'L':
            refs.append({
                'text': t,
                'type': 'datum',
                'bbox': {'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h']},
            })
    return refs


def _extract_gdt_frames_from_rects(page):
    """从矩形中简单提取 GD&T 框候选（小矩形水平排列）。"""
    # 延迟导入避免循环：extract_rects 属于 cluster/rect 分类模块。
    from pdf_analyzer_cluster import extract_rects
    result = extract_rects(page)
    return result.get('gdt_frames', [])
