"""
assembler_scoring.py — 评分 + 后过滤

从 assembler.py 拆出。包含：
  - _post_filter_all()  全局后置过滤
  - _score_and_filter()  OCR 结果打分
"""

import math
import re
from collections import Counter, defaultdict

from assembler_utils import (
    VERBOSE, SCORE_THRESHOLD, METADATA_BLACKLIST_PATTERNS, FRAME_KEYWORDS,
    _ALLCAPS_LONG_PATTERN, _INDUSTRIAL_SYMBOLS,
    _SECTION_LABEL_PATTERN, _GDT_SYMBOL_CHARS, _RATIO_PATTERN, _WORD_PATTERN,
    _bbox_center, _point_in_bbox, _fixup_ocr_text, _fixup_fc_capsule_text, _extract_prefix,
    _parse_nominal, _is_junk_by_symbol_ratio, _quick_dim_score,
    _parse_tolerance_inline, _bbox_iou_min, _union_bbox,
)
from assembler_dedup import _merge_angle_tolerance_fragments


# 右侧说明区域的通用关键词。
_RIGHT_ANNOTATION_PATTERNS = [
    re.compile(r'REFERENCE\s+DEFINITION', re.IGNORECASE),
    re.compile(r'^NOTE', re.IGNORECASE),
    re.compile(r'PERPENDICULAR\s+TO', re.IGNORECASE),
    re.compile(r'ROTATED\s+.*AROUND', re.IGNORECASE),
    re.compile(r'DEFINED\s+BY', re.IGNORECASE),
    re.compile(r'MEASURING\s+POINT', re.IGNORECASE),
    re.compile(r'^IMPORTANT\s+CHARACTERISTIC', re.IGNORECASE),
    re.compile(r'^SPECIAL\s+CHARACTERISTIC', re.IGNORECASE),
    re.compile(r'^CPK\s+VALUE', re.IGNORECASE),
]

# 左下说明区域的通用关键词。
_LEFT_ANNOTATION_PATTERNS = [
    re.compile(r'^NO\s+BURR', re.IGNORECASE),
    re.compile(r'CONSIDERATION', re.IGNORECASE),
    re.compile(r'HEIGHT\s+IS\s+ALLOWED', re.IGNORECASE),
    re.compile(r'PIN\s+WINDOW', re.IGNORECASE),
]

_MERGED_DIM_GDT_GUARD_CHARS = (_GDT_SYMBOL_CHARS - set('⌀∅Ø⊘')) | {'□'}
_MERGED_DIM_TOKEN_RE = re.compile(
    r'(?<![\w.])'
    r'(?P<text>'
    r'(?:\d+\s*[xX×]\s*)?'
    r'(?:[⌀∅ØφΦR]\s*)?'
    r'\d+(?:\.\d+)?'
    r'(?:\s*(?:±|[+\-])\s*\d+(?:\.\d+)?'
    r'(?:\s*/?\s*[+\-]\s*\d+(?:\.\d+)?)?'
    r')?'
    r'(?:°)?'
    r')'
)


def _compact_dim_text(text: str) -> str:
    return re.sub(r'\s+', '', text or '')


def _has_dimension_signal(text: str) -> bool:
    stripped = text.strip()
    return (
        bool(re.search(r'(?:±|[+\-])\s*\d', stripped)) or
        bool(re.match(r'^\d+\s*[xX×]', stripped)) or
        bool(re.match(r'^(?:\d+\s*[xX×]\s*)?[⌀∅ØφΦR]', stripped)) or
        '°' in stripped
    )


def _gdt_allows_integer_value(dim: dict, numbers: list[str]) -> bool:
    """仅在形位框携带基准上下文时接纳整数公差值，以区分框内值和孤立数字碎片。"""
    if any('.' in n for n in numbers):
        return True

    has_positive_integer = False
    for n in numbers:
        try:
            if float(n) > 0:
                has_positive_integer = True
                break
        except ValueError:
            continue
    if not has_positive_integer:
        return False

    datum = str(dim.get('datum') or '').strip()
    if datum:
        return True

    text = str(dim.get('text') or '')
    return bool(re.search(r'\b[A-Z](?:\([MLP]\))?\b', text))


def _candidate_geometry_evidence(row: dict) -> bool:
    evidence = row.get('candidate_evidence_summary') or {}
    return bool(
        row.get('candidate_has_geometry_evidence')
        or row.get('_candidate_geometry_evidence')
        or evidence.get('has_capsule')
        or evidence.get('has_dimline')
        or evidence.get('has_arrow')
    )


def _normalize_merged_dimension_text(text: str) -> str:
    # PaddleOCR occasionally reads the second ± in a merged line as "t".
    text = re.sub(r'(?<=\d)\s*[tT]\s*(?=\d)', '±', text or '')
    text = _fixup_fc_capsule_text(text)
    # 保留同一数值内部被空格拆分的小数片段。
    text = re.sub(r'(?<![\d.])(\d+)\s+\.(?=\d)', r'\1.', text)
    # 把直径符号误读形成的短前缀噪声与后续数值作为同一 token 处理。
    
    
    text = re.sub(
        r'^\s*([⌀∅ØφΦ])\s*\d?\s+[口□O0]\s+'
        r'(\d+(?:\.\d+)?\s*(?:±|[+\-])\s*\d+(?:\.\d+)?)',
        r'\1\2',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'^\s*[Dd]\s*2\.(\d)(\s*(?:±|[+\-])\s*0?\.\d+)',
        r'⌀\1\2',
        text,
    )
    text = re.sub(r'^\s*[Dd]\s+(?=[⌀∅ØφΦ])', '', text)
    text = re.sub(r'^\s*([⌀∅ØφΦ])0(\d)(?=\s*(?:±|[+\-]))', r'\1\2', text)
    return re.sub(r'\s+', ' ', text).strip()


def _split_merged_dimension_text(text: str) -> list[dict] | None:
    """Split OCR strings that contain multiple independent dimensions.

    This is intentionally conservative: distance-to-vector/label is not a
    pass/fail rule here. We only split when the OCR text itself contains
    multiple dimension-shaped spans, or one strong span plus trailing OCR junk.
    """
    if any(ch in text for ch in _MERGED_DIM_GDT_GUARD_CHARS):
        return None

    normalized = _normalize_merged_dimension_text(text)
    if not normalized:
        return None
    normalized_compact = _compact_dim_text(normalized)
    if re.fullmatch(
            r'(?:\d{1,2}[xX×])?\d{1,3}(?:\.\d+)?°(?:±|[+\-])\d(?:\.\d+)?°?',
            normalized_compact):
        return None

    candidates = []
    for m in _MERGED_DIM_TOKEN_RE.finditer(normalized):
        seg_text = m.group('text').strip()
        if not seg_text or not _has_dimension_signal(seg_text):
            continue
        candidates.append({
            'text': seg_text,
            'start': m.start('text'),
            'end': m.end('text'),
        })

    if not candidates:
        return None

    repeat_prefix_match = re.match(r'^\s*(\d+\s*[xX×])\s+', normalized)
    if repeat_prefix_match and len(candidates) > 1:
        repeat_prefix = repeat_prefix_match.group(1).replace('×', 'X').upper()
        for candidate in candidates[1:]:
            if not re.match(r'^\d+\s*[xX×]\s+', candidate['text']):
                candidate['text'] = f"{repeat_prefix} {candidate['text']}"

    if len(candidates) == 1:
        only = candidates[0]
        if _compact_dim_text(only['text']) == _compact_dim_text(normalized):
            if _compact_dim_text(normalized) != _compact_dim_text(text):
                only = dict(only)
                only['start'] = 0
                only['end'] = len(normalized)
                return [only]
            return None
        if normalized[:only['start']].strip():
            return None
        # 在完整尺寸 token 后过滤附加 OCR 杂片，合法双边公差由完整 token 规则处理。
        
        
        if not normalized[only['end']:].strip():
            return None
        return candidates

    return candidates


def _repair_cached_split_repeat_prefix(dim: dict) -> dict:
    """为缓存的拆分尺寸同步倍数前缀，使派生尺寸字段与原始候选结构保持一致。"""
    source = str(dim.get('source') or '')
    if not source.endswith('_split'):
        return dim

    text = str(dim.get('text') or '').strip()
    if not text or re.match(r'^\d+\s*[xX×]\s+', text):
        return dim

    split_from = str(dim.get('_split_from_text') or '').strip()
    if not re.match(r'^\s*\d+\s*[xX×]\s+', split_from):
        return dim

    segments = _split_merged_dimension_text(split_from)
    if not segments:
        return dim

    text_compact = _compact_dim_text(text)
    for segment in segments:
        seg_text = segment.get('text') or ''
        m = re.match(r'^\s*\d+\s*[xX×]\s+', seg_text)
        if not m:
            continue
        seg_without_prefix = seg_text[m.end():]
        if _compact_dim_text(seg_without_prefix) != text_compact:
            continue
        repaired = dict(dim)
        repaired.update(_dimension_fields_from_text(seg_text))
        return repaired

    return dim


def _repeat_prefix_fragment_candidates(ocr_rows: list[dict]) -> list[dict]:
    candidates = []
    for row in _merge_symbol_numeric_fragments(ocr_rows):
        evidence = row.get('_fragment_merge_evidence') or {}
        if str(evidence.get('kind') or '').startswith('repeat_prefix_'):
            candidates.append(row)
    return candidates


def _attach_repeat_prefix_fragments(dimensions: list[dict], ocr_rows: list[dict]) -> list[dict]:
    """按空间、字号和方向关系，将独立倍数前缀并入已构建的尺寸。"""
    if not dimensions or not ocr_rows:
        return dimensions

    candidates = _repeat_prefix_fragment_candidates(ocr_rows)
    if not candidates:
        return dimensions

    def _norm_no_prefix(text: str) -> str:
        fixed = _fixup_ocr_text(str(text or '').strip())
        fixed = re.sub(r'^\s*[2-9]\d?\s*[xX×]\s+', '', fixed)
        return re.sub(r'[\s/]+', '', fixed).upper()

    def _candidate_prefix(text: str) -> str | None:
        m = re.match(r'^\s*([2-9]\d?)\s*[xX×]\s+', str(text or '').strip())
        if not m:
            return None
        return f'{m.group(1)}X'

    def _center_dist(a: dict, b: dict) -> float:
        ax, ay = _bbox_center(a)
        bx, by = _bbox_center(b)
        return math.hypot(ax - bx, ay - by)

    def _spatial_match(dim_bbox: dict, cand_bbox: dict) -> tuple[bool, float]:
        if not all(k in dim_bbox for k in ('x', 'y', 'w', 'h')):
            return False, float('inf')
        if not all(k in cand_bbox for k in ('x', 'y', 'w', 'h')):
            return False, float('inf')
        dist = _center_dist(dim_bbox, cand_bbox)
        if _bbox_iou_min(dim_bbox, cand_bbox) > 0.08:
            return True, dist
        limit = max(36.0, min(110.0, max(dim_bbox.get('w', 0), dim_bbox.get('h', 0)) * 0.9))
        return dist <= limit, dist

    out = []
    for dim in dimensions:
        if dim.get('prefix'):
            out.append(dim)
            continue
        dim_text = str(dim.get('text') or '').strip()
        if not dim_text or not re.search(r'(?:±|[+\-])\s*\d', dim_text):
            out.append(dim)
            continue
        dim_norm = _norm_no_prefix(dim_text)
        dim_bbox = dim.get('bbox') or {}

        best = None
        for cand in candidates:
            cand_text = str(cand.get('text') or '').strip()
            prefix = _candidate_prefix(cand_text)
            if not prefix:
                continue
            if _norm_no_prefix(cand_text) != dim_norm:
                continue
            cand_bbox = cand.get('bbox_in_pdf') or cand.get('bbox') or {}
            ok, dist = _spatial_match(dim_bbox, cand_bbox)
            if not ok:
                continue
            if best is None or dist < best[0]:
                best = (dist, prefix, cand)

        if best is None:
            out.append(dim)
            continue

        _, prefix, cand = best
        repaired = dict(dim)
        repaired['text'] = f"{prefix} {dim_text}"
        repaired['prefix'] = prefix
        repaired['type'] = 'repeat'
        cand_bbox = cand.get('bbox_in_pdf') or cand.get('bbox') or {}
        if all(k in dim_bbox for k in ('x', 'y', 'w', 'h')) and all(k in cand_bbox for k in ('x', 'y', 'w', 'h')):
            repaired['bbox'] = _union_bbox([dim_bbox, cand_bbox])
        repaired['_repeat_prefix_fragment_evidence'] = cand.get('_fragment_merge_evidence')
        out.append(repaired)

    return out


def _bbox_for_text_span(bbox: dict, orientation, start: int, end: int, total_len: int) -> dict:
    if total_len <= 0 or end <= start:
        return dict(bbox)
    try:
        orient = float(orientation or 0)
    except (TypeError, ValueError):
        orient = 0.0
    is_vertical = abs(orient) >= 45
    start_ratio = max(0.0, min(1.0, start / total_len))
    end_ratio = max(start_ratio, min(1.0, end / total_len))
    out = dict(bbox)
    if is_vertical:
        y0 = bbox['y'] + bbox['h'] * start_ratio
        y1 = bbox['y'] + bbox['h'] * end_ratio
        out['y'] = y0
        out['h'] = max(1.0, y1 - y0)
    else:
        x0 = bbox['x'] + bbox['w'] * start_ratio
        x1 = bbox['x'] + bbox['w'] * end_ratio
        out['x'] = x0
        out['w'] = max(1.0, x1 - x0)
    return out


def _dimension_fields_from_text(text: str) -> dict:
    text_clean, prefix, dim_type = _extract_prefix(text)
    upper_tol, lower_tol = _parse_tolerance_inline(text_clean)
    nominal_raw = (
        re.sub(r'[±\+\-]\s*[\d.]+.*$', '', text_clean).strip()
        if (upper_tol or lower_tol) else
        text_clean.strip()
    )
    return {
        'text': text,
        'nominal': nominal_raw,
        'upper_tol': upper_tol,
        'lower_tol': lower_tol,
        'type': dim_type,
        'prefix': prefix,
    }


def _merge_symbol_numeric_fragments(unclaimed: list) -> list:
    """兼容路径中结合工程前缀与数值碎片保守构建候选。"""
    merged = []
    used_pairs = set()

    def _conf(row: dict) -> float:
        try:
            return float(row.get('confidence') or row.get('conf') or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _union_bbox(a: dict, b: dict) -> dict:
        x0 = min(a['x'], b['x'])
        y0 = min(a['y'], b['y'])
        x1 = max(a['x'] + a['w'], b['x'] + b['w'])
        y1 = max(a['y'] + a['h'], b['y'] + b['h'])
        return {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}

    def _bbox_has_keys(bbox: dict) -> bool:
        return all(k in bbox for k in ('x', 'y', 'w', 'h'))

    def _repeat_prefix(text: str) -> str | None:
        m = re.fullmatch(r'([2-9]\d?)\s*[xX×]', str(text or '').strip())
        if not m:
            return None
        return f'{m.group(1)}X'

    def _has_repeat_prefix(text: str) -> bool:
        return bool(re.match(r'^\s*[2-9]\d?\s*[xX×]\s+', str(text or '').strip()))

    def _repeat_prefixed_nominal(text: str) -> tuple[str, str] | None:
        fixed = _fixup_ocr_text(str(text or '').strip())
        m = re.fullmatch(
            r'([2-9]\d?)\s*[xX×]\s+'
            r'((?:[⌀∅ØφΦR]\s*)?\d+(?:\.\d+)?(?:°)?)',
            fixed,
        )
        if not m:
            return None
        return f'{m.group(1)}X', m.group(2)

    def _dimension_like_without_repeat(text: str) -> bool:
        fixed = _fixup_ocr_text(str(text or '').strip())
        if not fixed or _has_repeat_prefix(fixed):
            return False
        if not re.search(r'(?:±|[+\-])\s*\d', fixed):
            return False
        compact = re.sub(r'\s+', '', fixed)
        return bool(re.fullmatch(
            r'(?:[⌀∅ØφΦR]\s*)?\d+(?:\.\d+)?(?:°)?'
            r'(?:±|[+\-])\d+(?:\.\d+)?'
            r'(?:/?[+\-]\d+(?:\.\d+)?)?(?:°)?',
            compact,
        ))

    def _nominal_like_without_repeat(text: str) -> bool:
        fixed = _fixup_ocr_text(str(text or '').strip())
        if _has_repeat_prefix(fixed):
            return False
        return bool(re.fullmatch(r'(?:[⌀∅ØφΦR]\s*)?\d+(?:\.\d+)?(?:°)?', fixed))

    def _tol_fragment(text: str) -> str | None:
        fixed = _fixup_ocr_text(str(text or '').strip())
        fixed = fixed.replace('。', '.')
        compact = re.sub(r'\s+', '', fixed)
        if re.fullmatch(r'(?:±|[+\-])0?\.\d+(?:°)?', compact):
            return compact
        if re.fullmatch(r'0?\.\d+(?:°)?', compact):
            return compact
        return None

    def _orientation_compatible(a, b) -> bool:
        diff = abs((_orientation_degrees(a) - _orientation_degrees(b) + 90.0) % 180.0 - 90.0)
        return diff <= 25.0

    def _aligned_repeat_prefix(
        prefix_bbox: dict,
        value_bbox: dict,
        prefix_orientation,
        value_orientation,
    ) -> tuple[bool, float]:
        if not (_bbox_has_keys(prefix_bbox) and _bbox_has_keys(value_bbox)):
            return False, float('inf')
        if not _orientation_compatible(prefix_orientation, value_orientation):
            return False, float('inf')
        pcx, pcy = _bbox_center(prefix_bbox)
        vcx, vcy = _bbox_center(value_bbox)
        vertical = _is_vertical_orientation(value_orientation)
        dist = math.hypot(vcx - pcx, vcy - pcy)
        if vertical:
            gap = min(
                abs(prefix_bbox['y'] - (value_bbox['y'] + value_bbox['h'])),
                abs(value_bbox['y'] - (prefix_bbox['y'] + prefix_bbox['h'])),
            )
            x_aligned = abs(pcx - vcx) <= max(prefix_bbox['w'], value_bbox['w'], 1.0) + 12.0
            return (x_aligned and gap <= 42.0), dist
        gap = value_bbox['x'] - (prefix_bbox['x'] + prefix_bbox['w'])
        y_aligned = abs(pcy - vcy) <= max(prefix_bbox['h'], value_bbox['h'], 1.0) + 8.0
        return (y_aligned and -4.0 <= gap <= 60.0), dist

    def _near_tolerance_fragment(value_bbox: dict, tol_bbox: dict, orientation) -> tuple[bool, float]:
        if not (_bbox_has_keys(value_bbox) and _bbox_has_keys(tol_bbox)):
            return False, float('inf')
        vcx, vcy = _bbox_center(value_bbox)
        tcx, tcy = _bbox_center(tol_bbox)
        vertical = _is_vertical_orientation(orientation)
        dist = math.hypot(tcx - vcx, tcy - vcy)
        if vertical:
            x_ok = abs(tcx - vcx) <= max(value_bbox['w'], tol_bbox['w'], 1.0) + 22.0
            y_gap = min(
                abs(tol_bbox['y'] - (value_bbox['y'] + value_bbox['h'])),
                abs(value_bbox['y'] - (tol_bbox['y'] + tol_bbox['h'])),
            )
            return (x_ok and y_gap <= 44.0), dist
        y_ok = abs(tcy - vcy) <= max(value_bbox['h'], tol_bbox['h'], 1.0) + 12.0
        x_gap = tol_bbox['x'] - (value_bbox['x'] + value_bbox['w'])
        return (y_ok and -8.0 <= x_gap <= 72.0), dist

    for pi, prefix_row in enumerate(unclaimed):
        prefix_text = (prefix_row.get('text') or '').strip()
        if not re.fullmatch(r'[Rr⌀∅ØφΦ]', prefix_text):
            continue
        prefix_bbox = prefix_row.get('bbox_in_pdf') or {}
        if not all(k in prefix_bbox for k in ('x', 'y', 'w', 'h')):
            continue
        if _conf(prefix_row) < 0.75:
            continue
        pcx, pcy = _bbox_center(prefix_bbox)

        best = None
        for ni, number_row in enumerate(unclaimed):
            if ni == pi:
                continue
            number_text = (number_row.get('text') or '').strip()
            if not re.fullmatch(r'\d{1,2}(?:\.\d+)?', number_text):
                continue
            number_bbox = number_row.get('bbox_in_pdf') or {}
            if not all(k in number_bbox for k in ('x', 'y', 'w', 'h')):
                continue
            if _conf(number_row) < 0.75:
                continue
            ncx, ncy = _bbox_center(number_bbox)
            dist = math.hypot(ncx - pcx, ncy - pcy)
            if dist > 30:
                continue
            if abs(ncx - pcx) > 28 or abs(ncy - pcy) > 28:
                continue
            if best is None or dist < best[0]:
                best = (dist, ni, number_row)

        if best is None:
            continue
        dist, ni, number_row = best
        pair_key = tuple(sorted((pi, ni)))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        number_text = (number_row.get('text') or '').strip()
        merged_text = f'{prefix_text.upper() if prefix_text.lower() == "r" else prefix_text}{number_text}'
        number_bbox = number_row['bbox_in_pdf']
        merged_bbox = _union_bbox(prefix_bbox, number_bbox)
        merged.append({
            **prefix_row,
            'text': merged_text,
            '_raw_text': f'{prefix_text}{number_text}',
            'bbox_in_pdf': merged_bbox,
            'source': 'ocr_fragment_merge',
            'confidence': min(_conf(prefix_row), _conf(number_row)),
            'orientation': prefix_row.get('orientation', number_row.get('orientation', 0)),
            '_fragment_merge_evidence': {
                'kind': 'symbol_numeric',
                'distance': round(dist, 2),
                'parts': [prefix_text, number_text],
            },
        })

    used_prefix_full_pairs = set()
    for pi, prefix_row in enumerate(unclaimed):
        prefix_text = _fixup_ocr_text((prefix_row.get('text') or '').strip())
        prefix_compact = re.sub(r'\s+', '', prefix_text)
        m_prefix = re.fullmatch(
            r'([RrCc⌀∅ØφΦ])(\d+(?:\.\d+)?)(?:±|[+\-])0\.?',
            prefix_compact,
        )
        if not m_prefix:
            continue
        prefix_bbox = prefix_row.get('bbox_in_pdf') or {}
        if not all(k in prefix_bbox for k in ('x', 'y', 'w', 'h')):
            continue
        pcx, pcy = _bbox_center(prefix_bbox)
        symbol = m_prefix.group(1)
        nominal = m_prefix.group(2)

        for ni, number_row in enumerate(unclaimed):
            if ni == pi:
                continue
            number_text = _fixup_ocr_text((number_row.get('text') or '').strip())
            number_compact = re.sub(r'\s+', '', number_text)
            m_number = re.fullmatch(
                rf'{re.escape(nominal)}((?:±|[+\-])0?\.\d+)',
                number_compact,
            )
            if not m_number:
                continue
            number_bbox = number_row.get('bbox_in_pdf') or {}
            if not all(k in number_bbox for k in ('x', 'y', 'w', 'h')):
                continue
            ncx, ncy = _bbox_center(number_bbox)
            dist = math.hypot(ncx - pcx, ncy - pcy)
            if dist > 36:
                continue

            pair_key = (pi, ni)
            if pair_key in used_prefix_full_pairs:
                continue
            used_prefix_full_pairs.add(pair_key)
            merged_text = f'{symbol.upper() if symbol.lower() == "r" else symbol}{nominal}{m_number.group(1)}'
            merged.append({
                **prefix_row,
                'text': merged_text,
                '_raw_text': f'{prefix_text}|{number_text}',
                'bbox_in_pdf': _union_bbox(prefix_bbox, number_bbox),
                'source': 'ocr_fragment_merge',
                'confidence': min(_conf(prefix_row), _conf(number_row)),
                'orientation': prefix_row.get('orientation', number_row.get('orientation', 0)),
                '_fragment_merge_evidence': {
                    'kind': 'truncated_symbol_with_full_numeric_tolerance',
                    'parts': [prefix_text, number_text],
                    'distance': round(dist, 2),
                },
            })

    used_repeat_pairs = set()
    for pi, prefix_row in enumerate(unclaimed):
        prefix = _repeat_prefix(_fixup_ocr_text((prefix_row.get('text') or '').strip()))
        if not prefix:
            continue
        prefix_bbox = prefix_row.get('bbox_in_pdf') or {}
        if not _bbox_has_keys(prefix_bbox):
            continue
        if _conf(prefix_row) < 0.75:
            continue

        best_full = None
        for vi, value_row in enumerate(unclaimed):
            if vi == pi:
                continue
            value_text = _fixup_ocr_text((value_row.get('text') or '').strip())
            if not _dimension_like_without_repeat(value_text):
                continue
            value_bbox = value_row.get('bbox_in_pdf') or {}
            ok, dist = _aligned_repeat_prefix(
                prefix_bbox,
                value_bbox,
                prefix_row.get('orientation', 0),
                value_row.get('orientation', prefix_row.get('orientation', 0)),
            )
            if not ok:
                continue
            if best_full is None or dist < best_full[0]:
                best_full = (dist, vi, value_row, value_text)

        if best_full is not None:
            dist, vi, value_row, value_text = best_full
            pair_key = (pi, vi)
            if pair_key not in used_repeat_pairs:
                used_repeat_pairs.add(pair_key)
                value_bbox = value_row['bbox_in_pdf']
                merged.append({
                    **value_row,
                    'text': f'{prefix} {value_text}',
                    '_raw_text': f"{prefix_row.get('text', '')}|{value_row.get('text', '')}",
                    'bbox_in_pdf': _union_bbox(prefix_bbox, value_bbox),
                    'source': 'ocr_fragment_merge',
                    'confidence': min(_conf(prefix_row), _conf(value_row)),
                    'orientation': value_row.get('orientation', prefix_row.get('orientation', 0)),
                    '_fragment_merge_evidence': {
                        'kind': 'repeat_prefix_with_full_dimension',
                        'distance': round(dist, 2),
                        'parts': [prefix_row.get('text', ''), value_row.get('text', '')],
                    },
                })

        for vi, value_row in enumerate(unclaimed):
            if vi == pi:
                continue
            value_text = _fixup_ocr_text((value_row.get('text') or '').strip())
            if not _nominal_like_without_repeat(value_text):
                continue
            value_bbox = value_row.get('bbox_in_pdf') or {}
            orientation = value_row.get('orientation', prefix_row.get('orientation', 0))
            ok, dist_prefix = _aligned_repeat_prefix(
                prefix_bbox,
                value_bbox,
                prefix_row.get('orientation', 0),
                orientation,
            )
            if not ok:
                continue

            tol_matches = []
            for ti, tol_row in enumerate(unclaimed):
                if ti in {pi, vi}:
                    continue
                tol_text = _tol_fragment(tol_row.get('text') or '')
                if not tol_text:
                    continue
                tol_bbox = tol_row.get('bbox_in_pdf') or {}
                ok_tol, dist_tol = _near_tolerance_fragment(value_bbox, tol_bbox, orientation)
                if not ok_tol:
                    continue
                tol_matches.append((dist_tol, ti, tol_row, tol_text))

            signed = [item for item in tol_matches if re.match(r'^(?:±|[+\-])', item[3])]
            if not signed:
                continue
            signed.sort(key=lambda item: item[0])
            chosen = signed[:2]
            if len(chosen) == 1:
                bare = [
                    item for item in tol_matches
                    if item[1] != chosen[0][1] and not re.match(r'^(?:±|[+\-])', item[3])
                ]
                if bare:
                    bare.sort(key=lambda item: item[0])
                    chosen.append(bare[0])

            chosen.sort(key=lambda item: (
                (item[2].get('bbox_in_pdf') or {}).get('y', 0.0),
                (item[2].get('bbox_in_pdf') or {}).get('x', 0.0),
            ))
            merged_text = _fixup_ocr_text(
                ' '.join([prefix, value_text] + [item[3] for item in chosen])
            )
            if not _dimension_like_without_repeat(re.sub(r'^\s*[2-9]\d?\s*[xX×]\s+', '', merged_text)):
                continue
            set_key = tuple([pi, vi] + [item[1] for item in chosen])
            if set_key in used_repeat_pairs:
                continue
            used_repeat_pairs.add(set_key)
            merged_bbox = _union_bbox(prefix_bbox, value_bbox)
            for _, _, tol_row, _ in chosen:
                merged_bbox = _union_bbox(merged_bbox, tol_row['bbox_in_pdf'])
            merged.append({
                **value_row,
                'text': merged_text,
                '_raw_text': '|'.join(
                    [str(prefix_row.get('text', '')), str(value_row.get('text', ''))]
                    + [str(item[2].get('text', '')) for item in chosen]
                ),
                'bbox_in_pdf': merged_bbox,
                'source': 'ocr_fragment_merge',
                'confidence': min([_conf(prefix_row), _conf(value_row)] + [_conf(item[2]) for item in chosen]),
                'orientation': orientation,
                '_fragment_merge_evidence': {
                    'kind': 'repeat_prefix_with_numeric_tolerance_fragments',
                    'distance': round(dist_prefix, 2),
                    'parts': [prefix_row.get('text', ''), value_row.get('text', '')]
                             + [item[2].get('text', '') for item in chosen],
                },
            })

    for vi, value_row in enumerate(unclaimed):
        value_text = _fixup_ocr_text((value_row.get('text') or '').strip())
        prefixed_nominal = _repeat_prefixed_nominal(value_text)
        if not prefixed_nominal:
            continue
        prefix, nominal_text = prefixed_nominal
        value_bbox = value_row.get('bbox_in_pdf') or {}
        if not _bbox_has_keys(value_bbox):
            continue
        orientation = value_row.get('orientation', 0)

        tol_matches = []
        for ti, tol_row in enumerate(unclaimed):
            if ti == vi:
                continue
            tol_text = _tol_fragment(tol_row.get('text') or '')
            if not tol_text or not re.match(r'^(?:±|[+\-])', tol_text):
                continue
            tol_bbox = tol_row.get('bbox_in_pdf') or {}
            ok_tol, dist_tol = _near_tolerance_fragment(value_bbox, tol_bbox, orientation)
            if not ok_tol:
                continue
            if not _orientation_compatible(orientation, tol_row.get('orientation', orientation)):
                continue
            tol_matches.append((dist_tol, ti, tol_row, tol_text))
        if not tol_matches:
            continue

        tol_matches.sort(key=lambda item: item[0])
        chosen = tol_matches[:2]
        chosen.sort(key=lambda item: (
            (item[2].get('bbox_in_pdf') or {}).get('y', 0.0),
            (item[2].get('bbox_in_pdf') or {}).get('x', 0.0),
        ))
        merged_text = _fixup_ocr_text(
            ' '.join([prefix, nominal_text] + [item[3] for item in chosen])
        )
        if not _dimension_like_without_repeat(re.sub(r'^\s*[2-9]\d?\s*[xX×]\s+', '', merged_text)):
            continue
        set_key = tuple([vi] + [item[1] for item in chosen])
        if set_key in used_repeat_pairs:
            continue
        used_repeat_pairs.add(set_key)
        merged_bbox = value_bbox
        for _, _, tol_row, _ in chosen:
            merged_bbox = _union_bbox(merged_bbox, tol_row['bbox_in_pdf'])
        merged.append({
            **value_row,
            'text': merged_text,
            '_raw_text': '|'.join([str(value_row.get('text', ''))] + [str(item[2].get('text', '')) for item in chosen]),
            'bbox_in_pdf': merged_bbox,
            'source': 'ocr_fragment_merge',
            'confidence': min([_conf(value_row)] + [_conf(item[2]) for item in chosen]),
            'orientation': orientation,
            '_fragment_merge_evidence': {
                'kind': 'repeat_prefixed_nominal_with_tolerance_fragment',
                'parts': [value_row.get('text', '')] + [item[2].get('text', '') for item in chosen],
                'distance': round(chosen[0][0], 2),
            },
        })

    used_diameter_fragment_sets = set()
    for ci, combo_row in enumerate(unclaimed):
        combo_text = _fixup_ocr_text((combo_row.get('text') or '').strip())
        combo_compact = re.sub(r'\s+', '', combo_text)
        m_combo = re.fullmatch(r'([QqOo0口⌀∅ØφΦ])(\d{1,3})\.?', combo_compact)
        if not m_combo:
            continue
        combo_bbox = combo_row.get('bbox_in_pdf') or {}
        if not all(k in combo_bbox for k in ('x', 'y', 'w', 'h')):
            continue
        ccx, ccy = _bbox_center(combo_bbox)

        for di, dec_row in enumerate(unclaimed):
            if di == ci:
                continue
            dec_text = _fixup_ocr_text((dec_row.get('text') or '').strip())
            dec_compact = re.sub(r'\s+', '', dec_text)
            m_dec = re.fullmatch(r'\.(\d{1,3})(?:±)?', dec_compact)
            if not m_dec:
                continue
            dec_bbox = dec_row.get('bbox_in_pdf') or {}
            if not all(k in dec_bbox for k in ('x', 'y', 'w', 'h')):
                continue
            dcx, dcy = _bbox_center(dec_bbox)
            if dec_bbox['x'] < combo_bbox['x'] - 2:
                continue
            if dec_bbox['x'] - (combo_bbox['x'] + combo_bbox['w']) > 24:
                continue
            if abs(dcy - ccy) > 14:
                continue

            for ti, tol_row in enumerate(unclaimed):
                if ti in {ci, di}:
                    continue
                tol_text = _fixup_ocr_text((tol_row.get('text') or '').strip())
                tol_compact = re.sub(r'\s+', '', tol_text)
                if not re.fullmatch(r'(?:±|[+\-])0?\.\d+', tol_compact):
                    continue
                tol_bbox = tol_row.get('bbox_in_pdf') or {}
                if not all(k in tol_bbox for k in ('x', 'y', 'w', 'h')):
                    continue
                tcx, tcy = _bbox_center(tol_bbox)
                if tol_bbox['x'] < dec_bbox['x'] - 2:
                    continue
                if tol_bbox['x'] - (dec_bbox['x'] + dec_bbox['w']) > 28:
                    continue
                if abs(tcy - dcy) > 14:
                    continue

                set_key = (ci, di, ti)
                if set_key in used_diameter_fragment_sets:
                    continue
                used_diameter_fragment_sets.add(set_key)
                merged_bbox = _union_bbox(_union_bbox(combo_bbox, dec_bbox), tol_bbox)
                merged_text = f'⌀{m_combo.group(2)}.{m_dec.group(1)}{tol_compact}'
                merged.append({
                    **combo_row,
                    'text': merged_text,
                    '_raw_text': f'{combo_text}|{dec_text}|{tol_text}',
                    'bbox_in_pdf': merged_bbox,
                    'source': 'ocr_fragment_merge',
                    'confidence': min(_conf(combo_row), _conf(dec_row), _conf(tol_row)),
                    'orientation': combo_row.get('orientation', 0),
                    '_fragment_merge_evidence': {
                        'kind': 'diameter_combo_decimal_tolerance_fragments',
                        'parts': [combo_text, dec_text, tol_text],
                    },
                })

    for si, symbol_row in enumerate(unclaimed):
        symbol_text = _fixup_ocr_text((symbol_row.get('text') or '').strip())
        symbol_compact = re.sub(r'\s+', '', symbol_text)
        if symbol_compact not in {'Q', 'q', 'O', 'o', '0', '口', '⌀', '∅', 'Ø', 'φ', 'Φ'}:
            continue
        symbol_bbox = symbol_row.get('bbox_in_pdf') or {}
        if not all(k in symbol_bbox for k in ('x', 'y', 'w', 'h')):
            continue
        if _conf(symbol_row) < 0.4:
            continue
        scx, scy = _bbox_center(symbol_bbox)

        for ii, int_row in enumerate(unclaimed):
            if ii == si:
                continue
            int_text = _fixup_ocr_text((int_row.get('text') or '').strip())
            int_compact = re.sub(r'\s+', '', int_text)
            m_int = re.fullmatch(r'(\d{1,3})\.?', int_compact)
            if not m_int:
                continue
            int_bbox = int_row.get('bbox_in_pdf') or {}
            if not all(k in int_bbox for k in ('x', 'y', 'w', 'h')):
                continue
            icx, icy = _bbox_center(int_bbox)
            if not (symbol_bbox['x'] - 2 <= int_bbox['x'] <= symbol_bbox['x'] + symbol_bbox['w'] + 32):
                continue
            if abs(icy - scy) > 14:
                continue

            for di, dec_row in enumerate(unclaimed):
                if di in {si, ii}:
                    continue
                dec_text = _fixup_ocr_text((dec_row.get('text') or '').strip())
                dec_compact = re.sub(r'\s+', '', dec_text)
                m_dec = re.fullmatch(r'\.(\d{1,3})(?:±)?', dec_compact)
                if not m_dec:
                    continue
                dec_bbox = dec_row.get('bbox_in_pdf') or {}
                if not all(k in dec_bbox for k in ('x', 'y', 'w', 'h')):
                    continue
                dcx, dcy = _bbox_center(dec_bbox)
                if dec_bbox['x'] < int_bbox['x'] - 2:
                    continue
                if dec_bbox['x'] - (int_bbox['x'] + int_bbox['w']) > 24:
                    continue
                if abs(dcy - icy) > 14:
                    continue

                for ti, tol_row in enumerate(unclaimed):
                    if ti in {si, ii, di}:
                        continue
                    tol_text = _fixup_ocr_text((tol_row.get('text') or '').strip())
                    tol_compact = re.sub(r'\s+', '', tol_text)
                    if not re.fullmatch(r'(?:±|[+\-])0?\.\d+', tol_compact):
                        continue
                    tol_bbox = tol_row.get('bbox_in_pdf') or {}
                    if not all(k in tol_bbox for k in ('x', 'y', 'w', 'h')):
                        continue
                    tcx, tcy = _bbox_center(tol_bbox)
                    if tol_bbox['x'] < dec_bbox['x'] - 2:
                        continue
                    if tol_bbox['x'] - (dec_bbox['x'] + dec_bbox['w']) > 28:
                        continue
                    if abs(tcy - dcy) > 14:
                        continue

                    set_key = (si, ii, di, ti)
                    if set_key in used_diameter_fragment_sets:
                        continue
                    used_diameter_fragment_sets.add(set_key)
                    merged_bbox = _union_bbox(
                        _union_bbox(_union_bbox(symbol_bbox, int_bbox), dec_bbox),
                        tol_bbox,
                    )
                    merged_text = f'⌀{m_int.group(1)}.{m_dec.group(1)}{tol_compact}'
                    merged.append({
                        **symbol_row,
                        'text': merged_text,
                        '_raw_text': f'{symbol_text}|{int_text}|{dec_text}|{tol_text}',
                        'bbox_in_pdf': merged_bbox,
                        'source': 'ocr_fragment_merge',
                        'confidence': min(
                            _conf(symbol_row),
                            _conf(int_row),
                            _conf(dec_row),
                            _conf(tol_row),
                        ),
                        'orientation': symbol_row.get('orientation', int_row.get('orientation', 0)),
                        '_fragment_merge_evidence': {
                            'kind': 'diameter_decimal_tolerance_fragments',
                            'parts': [symbol_text, int_text, dec_text, tol_text],
                        },
                    })

    used_zero_tol_pairs = set()
    for ti, tol_row in enumerate(unclaimed):
        tol_text = _fixup_ocr_text((tol_row.get('text') or '').strip())
        tol_compact = re.sub(r'\s+', '', tol_text)
        m_tol = re.fullmatch(r'0(±|[+\-])(0?\.\d+)', tol_compact)
        if not m_tol:
            continue
        tol_bbox = tol_row.get('bbox_in_pdf') or {}
        if not all(k in tol_bbox for k in ('x', 'y', 'w', 'h')):
            continue
        tcx, tcy = _bbox_center(tol_bbox)
        sign_text = m_tol.group(1)
        tol_value = m_tol.group(2)

        for di, digit_row in enumerate(unclaimed):
            if di == ti:
                continue
            digit_text = _fixup_ocr_text((digit_row.get('text') or '').strip())
            if not re.fullmatch(r'[1-9]', digit_text):
                continue
            digit_bbox = digit_row.get('bbox_in_pdf') or {}
            if not all(k in digit_bbox for k in ('x', 'y', 'w', 'h')):
                continue
            dcx, dcy = _bbox_center(digit_bbox)
            if not (tol_bbox['x'] - 2 <= dcx <= tol_bbox['x'] + tol_bbox['w'] + 2):
                continue
            rel_x = (dcx - tol_bbox['x']) / max(tol_bbox['w'], 1e-6)
            if not 0.22 <= rel_x <= 0.76:
                continue
            if abs(dcy - tcy) > max(tol_bbox['h'], digit_bbox['h'], 1.0):
                continue

            pair_key = (ti, di)
            if pair_key in used_zero_tol_pairs:
                continue
            used_zero_tol_pairs.add(pair_key)
            decimal_text = f'0.{digit_text}'
            merged_text = f'{decimal_text} {sign_text}{tol_value}'
            merged_bbox = _union_bbox(tol_bbox, digit_bbox)
            merged.append({
                **tol_row,
                'text': merged_text,
                '_raw_text': f'{tol_text}|{digit_text}',
                'bbox_in_pdf': merged_bbox,
                'source': 'ocr_fragment_merge',
                'confidence': min(_conf(tol_row), _conf(digit_row)),
                'orientation': tol_row.get('orientation', digit_row.get('orientation', 0)),
                '_fragment_merge_evidence': {
                    'kind': 'zero_tolerance_with_internal_digit',
                    'parts': [tol_text, digit_text],
                    'digit_rel_x': round(rel_x, 3),
                },
            })

    used_triples = set()
    for zi, zero_row in enumerate(unclaimed):
        zero_text = _fixup_ocr_text((zero_row.get('text') or '').strip())
        if not re.fullmatch(r'[0Oo]', zero_text):
            continue
        zero_bbox = zero_row.get('bbox_in_pdf') or {}
        if not all(k in zero_bbox for k in ('x', 'y', 'w', 'h')):
            continue
        zcx, zcy = _bbox_center(zero_bbox)

        for di, digit_row in enumerate(unclaimed):
            if di == zi:
                continue
            digit_text = _fixup_ocr_text((digit_row.get('text') or '').strip())
            if not re.fullmatch(r'\d', digit_text):
                continue
            digit_bbox = digit_row.get('bbox_in_pdf') or {}
            if not all(k in digit_bbox for k in ('x', 'y', 'w', 'h')):
                continue
            dcx, dcy = _bbox_center(digit_bbox)
            if digit_bbox['x'] < zero_bbox['x'] + zero_bbox['w'] - 2:
                continue
            if digit_bbox['x'] - (zero_bbox['x'] + zero_bbox['w']) > 18:
                continue
            if abs(dcy - zcy) > 16:
                continue

            for ti, tol_row in enumerate(unclaimed):
                if ti in {zi, di}:
                    continue
                tol_text = _fixup_ocr_text((tol_row.get('text') or '').strip())
                tol_compact = re.sub(r'\s+', '', tol_text)
                if not re.fullmatch(r'(?:±|[+\-])0?\.\d+', tol_compact):
                    continue
                tol_bbox = tol_row.get('bbox_in_pdf') or {}
                if not all(k in tol_bbox for k in ('x', 'y', 'w', 'h')):
                    continue
                tcx, tcy = _bbox_center(tol_bbox)
                if tol_bbox['x'] < digit_bbox['x'] - 4:
                    continue
                if tol_bbox['x'] - (digit_bbox['x'] + digit_bbox['w']) > 22:
                    continue
                if abs(tcy - dcy) > 18:
                    continue

                triple_key = (zi, di, ti)
                if triple_key in used_triples:
                    continue
                used_triples.add(triple_key)
                decimal_text = f'0.{digit_text}'
                merged_text = f'{decimal_text} {tol_compact}'
                merged_bbox = _union_bbox(_union_bbox(zero_bbox, digit_bbox), tol_bbox)
                merged.append({
                    **zero_row,
                    'text': merged_text,
                    '_raw_text': f'{zero_text}{digit_text}{tol_text}',
                    'bbox_in_pdf': merged_bbox,
                    'source': 'ocr_fragment_merge',
                    'confidence': min(_conf(zero_row), _conf(digit_row), _conf(tol_row)),
                    'orientation': zero_row.get('orientation', 0),
                    '_fragment_merge_evidence': {
                        'kind': 'leading_zero_decimal_tolerance',
                        'parts': [zero_text, digit_text, tol_text],
                        'zero_digit_gap': round(digit_bbox['x'] - (zero_bbox['x'] + zero_bbox['w']), 2),
                        'digit_tol_gap': round(tol_bbox['x'] - (digit_bbox['x'] + digit_bbox['w']), 2),
                    },
                })

    if not merged:
        return unclaimed
    return merged + unclaimed


def _has_incomplete_zero_tolerance(dim: dict) -> bool:
    text = (dim.get('text') or '').strip()
    tol_values = [
        str(dim.get('upper_tol') or '').strip(),
        str(dim.get('lower_tol') or '').strip(),
    ]
    if any(re.fullmatch(r'[+\-]0\.?', tol) for tol in tol_values if tol):
        return True
    return bool(re.search(r'(?:±|[+\-])\s*0\.?\s*$', text))


def _is_zero_nominal_with_real_tolerance(dim: dict, nom: str, text: str) -> bool:
    """仅在同一尺寸携带有效非零公差时接纳显式零公称值；不放行孤立零值。"""
    if not re.fullmatch(r'[0Oo]', (nom or '').strip()):
        return False
    if not re.search(r'(?:±|[+\-])\s*0?\.\d', text or ''):
        return False
    for key in ('upper_tol', 'lower_tol'):
        tol = str(dim.get(key) or '').strip()
        if not tol:
            continue
        val = _parse_nominal(tol.lstrip('+-'))
        if val is not None and val > 0:
            return True
    return False


def _normalize_zero_nominal_tolerance(dim: dict, text: str) -> str:
    dim['nominal'] = '0'
    normalized = re.sub(r'^\s*[Oo](?=\s*(?:±|[+\-]))', '0', text or '', count=1)
    if normalized:
        dim['text'] = normalized
    return normalized or text


def _is_repeated_letter_datum(text: str) -> bool:
    return bool(re.fullmatch(r'([A-Z])\1{1,2}', str(text or '').strip()))


def _repair_gdt_repeated_datum_tails_from_peers(dimensions: list[dict]) -> list[dict]:
    """在重复字母基准链中，仅当邻近同值框提供唯一的缺失引用时补齐末尾基准；不推断普通单字母基准。"""
    if not dimensions:
        return dimensions

    full_by_key: defaultdict[tuple, list[dict]] = defaultdict(list)
    for dim in dimensions:
        if dim.get('source') != 'gdt_line_frame':
            continue
        d1 = str(dim.get('datum_1') or '').strip()
        d2 = str(dim.get('datum_2') or '').strip()
        d3 = str(dim.get('datum_3') or '').strip()
        if not (_is_repeated_letter_datum(d1) and _is_repeated_letter_datum(d2) and _is_repeated_letter_datum(d3)):
            continue
        key = (
            int(dim.get('page') or 1),
            str(dim.get('prefix') or ''),
            str(dim.get('nominal') or '').replace(' ', ''),
            d1,
            d2,
        )
        full_by_key[key].append(dim)

    if not full_by_key:
        return dimensions

    repaired_out = []
    for dim in dimensions:
        if dim.get('source') != 'gdt_line_frame' or str(dim.get('datum_3') or '').strip():
            repaired_out.append(dim)
            continue
        d1 = str(dim.get('datum_1') or '').strip()
        d2 = str(dim.get('datum_2') or '').strip()
        if not (_is_repeated_letter_datum(d1) and _is_repeated_letter_datum(d2)):
            repaired_out.append(dim)
            continue
        key = (
            int(dim.get('page') or 1),
            str(dim.get('prefix') or ''),
            str(dim.get('nominal') or '').replace(' ', ''),
            d1,
            d2,
        )
        peers = full_by_key.get(key) or []
        if not peers:
            repaired_out.append(dim)
            continue

        bbox = dim.get('bbox') or {}
        close_third_datums = set()
        if all(k in bbox for k in ('x', 'y', 'w', 'h')):
            cx, cy = _bbox_center(bbox)
        else:
            cx = cy = None
        for peer in peers:
            peer_bbox = peer.get('bbox') or {}
            if cx is not None and all(k in peer_bbox for k in ('x', 'y', 'w', 'h')):
                pcx, pcy = _bbox_center(peer_bbox)
                if math.hypot(cx - pcx, cy - pcy) > 220.0:
                    continue
            close_third_datums.add(str(peer.get('datum_3') or '').strip())

        if len(close_third_datums) != 1:
            repaired_out.append(dim)
            continue

        third = next(iter(close_third_datums))
        repaired = dict(dim)
        repaired['datum_3'] = third
        repaired['datum'] = ' / '.join([d1, d2, third])
        text = str(repaired.get('text') or '').strip()
        if text and not re.search(rf'(?<![A-Z]){re.escape(third)}(?![A-Z])', text):
            repaired['text'] = f'{text} {third}'
        repaired['_gdt_datum_tail_repair'] = {
            'kind': 'peer_repeated_datum_tail',
            'added': third,
            'basis': f'{d1}/{d2}/{third}',
        }
        repaired_out.append(repaired)

    return repaired_out


def _normalize_leading_decimal_dimension(dim: dict) -> None:
    """恢复缺失前导零且带公差或工程前缀的小数候选；独立公差碎片仍需过滤。"""
    nom = str(dim.get('nominal') or '').strip()
    if not re.fullmatch(r'\.\d+(?:°)?', nom):
        return

    has_tol = bool(dim.get('upper_tol') or dim.get('lower_tol'))
    prefix = str(dim.get('prefix') or '').strip()
    prefix_evidence = bool(re.search(r'[R⌀∅ØφΦ]', prefix, re.IGNORECASE))
    if not (has_tol or prefix_evidence):
        return

    text = str(dim.get('text') or '')
    # 不恢复来自邻近较大尺寸的重复公差碎片。
    
    if re.search(r'(?:±|[+\-])\s*(?:±|[+\-])', text):
        return

    normalized_nom = '0' + nom
    dim['nominal'] = normalized_nom
    if text:
        dim['text'] = re.sub(r'(?<!\d)\.(?=\d)', '0.', text, count=1)
    if dim.get('_raw_text'):
        dim['_raw_text'] = re.sub(
            r'(?<!\d)\.(?=\d)',
            '0.',
            str(dim.get('_raw_text')),
            count=1,
        )


def _normalize_duplicate_diameter_prefix_fragment(dim: dict) -> None:
    """Drop duplicated OCR surrogate before a clean vector-injected diameter."""
    text = str(dim.get('text') or '').strip()
    raw_text = str(dim.get('_raw_text') or '').strip()
    if not text or not raw_text:
        return

    raw_compact = _compact_dim_text(raw_text)
    if not re.fullmatch(
            r'[⌀∅ØφΦ]\d+(?:\.\d+)?'
            r'(?:(?:±|[+\-])\d+(?:\.\d+)?'
            r'(?:/[+\-]\d+(?:\.\d+)?)?)?',
            raw_compact):
        return

    text_compact = _compact_dim_text(text)
    if not re.fullmatch(r'[QqOo0口]' + re.escape(raw_compact), text_compact):
        return

    dim.update(_dimension_fields_from_text(raw_text))
    dim['_raw_text'] = raw_text


def _normalize_trailing_ocr_letter_noise(dim: dict) -> None:
    """Drop a lone trailing OCR letter after a numeric geometric dimension."""
    text = str(dim.get('text') or '').strip()
    if not text:
        return

    match = re.fullmatch(r'((?:[⌀∅ØφΦR]\s*)?\d+(?:\.\d+)?(?:°)?)\s+[Oo]', text)
    if not match:
        return

    source = str(dim.get('source') or '')
    has_geometry = bool(
        source.startswith('geometric_anchor')
        or dim.get('_dimension_line_evidence')
        or dim.get('_leader_line_evidence')
        or dim.get('_strip_ocr_evidence')
        or _candidate_geometry_evidence(dim)
    )
    if not has_geometry:
        return

    cleaned = match.group(1)
    dim.update(_dimension_fields_from_text(cleaned))
    dim['_raw_text'] = text


def _normalize_ocr_gdt_note_dimension(dim: dict) -> None:
    """Convert explicit OCR GD&T note text into a dimension-shaped frame."""
    if dim.get('source') == 'gdt_line_frame':
        return
    text = str(dim.get('text') or '').strip()
    if '位置度' not in text:
        return

    match = re.search(
        r'位置度\s*((?:[0O]\s*)?[.。]\s*\d+|\d+\s*[.。]\s*\d+|\d+)',
        text,
    )
    if not match:
        return
    value = re.sub(r'\s+', '', match.group(1)).replace('。', '.').replace('O', '0')
    if value.startswith('.'):
        value = '0' + value
    parsed = _parse_nominal(value)
    if parsed is None or parsed <= 0 or parsed > 5:
        return

    dim.update({
        'text': f'⊕ {value}',
        'nominal': value,
        'upper_tol': None,
        'lower_tol': None,
        'type': 'gdt',
        'prefix': '⊕',
    })
    dim['source'] = 'gdt_line_frame'
    dim['_origin'] = 'ocr_note'
    dim['_raw_text'] = text


def _orientation_degrees(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_vertical_orientation(value) -> bool:
    angle = abs(_orientation_degrees(value)) % 180.0
    return 45.0 <= angle <= 135.0


def _extract_max_min_limit(text: str) -> tuple[str, str] | None:
    normalized = (text or '').upper()
    normalized = normalized.replace('Ｏ', 'O').replace('０', '0').replace('。', '.')
    normalized = re.sub(r'\bMINO(?=[.\d])', 'MIN0', normalized)
    normalized = re.sub(r'\bMAXO(?=[.\d])', 'MAX0', normalized)
    m = re.search(r'\b(MAX|MIN|M)\s*[. ]*\s*(\d+(?:\.\d+)?)', normalized)
    if not m:
        return None
    kind = 'MIN' if m.group(1) == 'MIN' else 'MAX'
    return kind, m.group(2)


def _is_max_min_suffix_dimension(text: str) -> bool:
    normalized = (text or '').upper()
    normalized = normalized.replace('Ｏ', 'O').replace('０', '0').replace('。', '.')
    normalized = re.sub(r'\s+', '', normalized)
    return bool(re.fullmatch(r'(?:R?\d+\.\d+|\d{2,})(?:MAX|MIN)\.?', normalized))


def _expanded_merged_dimensions(dimensions: list) -> list:
    expanded = []
    for dim in dimensions:
        dim = _repair_cached_split_repeat_prefix(dim)
        text = (dim.get('text') or '').strip()
        bbox = dim.get('bbox') or {}
        if not text or not all(k in bbox for k in ('x', 'y', 'w', 'h')):
            expanded.append(dim)
            continue
        if dim.get('source') == 'gdt_line_frame':
            expanded.append(dim)
            continue
        segments = _split_merged_dimension_text(text)
        if not segments:
            expanded.append(dim)
            continue

        normalized = _normalize_merged_dimension_text(text)
        source = dim.get('source') or 'unknown'
        split_source = source if source.endswith('_split') else f'{source}_split'
        for seg in segments:
            new_dim = dict(dim)
            new_dim.update(_dimension_fields_from_text(seg['text']))
            new_dim['bbox'] = _bbox_for_text_span(
                bbox,
                dim.get('orientation', 0),
                seg['start'],
                seg['end'],
                len(normalized),
            )
            new_dim['source'] = split_source
            new_dim['_split_from_text'] = text
            new_dim.setdefault('_raw_text', dim.get('_raw_text', text))
            expanded.append(new_dim)
    return expanded


# ---------------------------------------------------------------------------
# 全局后置过滤
# ---------------------------------------------------------------------------

def _post_filter_all(
    dimensions: list,
    page_width: float = 0.0,
    page_height: float = 0.0,
    debug: dict | None = None,
) -> list:
    """全局后置过滤：对所有 source 的 dimension 应用质量门槛（含 geometric_anchor/dimline）。"""
    # 仅匹配裸数字括号，不接纳字母前缀或倍数括号。
    _PAREN_REF_RE = re.compile(r'^\([\d.]+\)$')
    filtered = []
    dimensions = _expanded_merged_dimensions(dimensions)
    dimensions = _repair_gdt_repeated_datum_tails_from_peers(dimensions)
    dimensions = _merge_angle_tolerance_fragments(dimensions)
    drop_reasons = Counter()
    drop_examples = defaultdict(list)
    post_filter_ledger = []

    def _drop(reason: str, dim: dict) -> bool:
        bb = dim.get('bbox') or {}
        nom = (dim.get('nominal') or '').strip()
        post_filter_ledger.append({
            'reason': reason,
            'text': dim.get('text', ''),
            'bbox': {
                'x': bb.get('x', 0),
                'y': bb.get('y', 0),
                'w': bb.get('w', 0),
                'h': bb.get('h', 0),
            },
            'source': dim.get('source', ''),
            'confidence': dim.get('confidence'),
            'nominal': nom,
            'parsed_nominal': _parse_nominal(nom) if nom else None,
            'upper_tol': dim.get('upper_tol'),
            'lower_tol': dim.get('lower_tol'),
            'type': dim.get('type'),
            'prefix': dim.get('prefix'),
            'raw_text': dim.get('_raw_text'),
            'origin': dim.get('_origin'),
            'orientation': dim.get('orientation'),
            'font_height': dim.get('_font_height'),
        })
        if VERBOSE:
            drop_reasons[reason] += 1
            if len(drop_examples[reason]) < 8:
                bb = dim.get('bbox', {})
                drop_examples[reason].append(
                    f"{dim.get('source','?')}:{dim.get('text','')!r}"
                    f"@({bb.get('x',0):.0f},{bb.get('y',0):.0f})"
                )
        return True

    for dim in dimensions:
        text = dim.get('text', '').strip()
        _normalize_ocr_gdt_note_dimension(dim)
        text = dim.get('text', '').strip()

        # Basic Dimension（坐标/理论精确尺寸）不输出为检测尺寸
        # 它们是 GD&T 参考值，不是加工尺寸。同时 claim 对应的 OCR 让 scored path 也不重复输出
        if dim.get('source') == 'basic_dimension':
            if VERBOSE:
                print(f"    [BD] exclude basic dimension: '{text}'")
            _drop('basic_dimension', dim)
            continue

        # 参考尺寸 (N) 多点拦截 — text / nominal / _raw_text 任一命中即丢弃
        # 工程图惯例 "(数字)" = REF 参考尺寸，不参与公差检验，必须不输出 stamp。
        # 现实：PaddleOCR 经常漏读首/尾括号，靠 _raw_text 兜底覆盖 OCR 截断场景。
        # 仅对 ocr / dimension_line / scored 这类常规尺寸生效，gdt_line_frame 豁免
        # （GD&T 框内的 (M)/(L)/(P) 不属此类）
        if dim.get('source') != 'gdt_line_frame':
            _nom_raw = (dim.get('nominal') or '').strip()
            _raw_t = (dim.get('_raw_text') or '').strip()
            if (_PAREN_REF_RE.match(text) or
                    _PAREN_REF_RE.match(_nom_raw) or
                    _PAREN_REF_RE.match(_raw_t)):
                if VERBOSE:
                    print(f"    [paren_ref] drop reference dim: text='{text}' "
                          f"nom='{_nom_raw}' raw='{_raw_t}'")
                _drop('paren_reference', dim)
                continue

        # 过滤散列的小写字母和数字序列。
        if re.match(r'^[a-z]\s+[a-z]\s+\d', text):
            _drop('lowercase_sparse_garble', dim)
            continue
        # 过滤长文本中的过密空白。
        if '   ' in text and len(text) > 10:
            _drop('multi_space_garble', dim)
            continue
        # 过滤连续编号碎片。
        if re.match(r'^[A-Z]\d\s+[A-Z]\d', text):
            _drop('numbered_fragment', dim)
            continue
        # 过滤稀疏数字与字母混合碎片。
        if re.match(r'^0\s+\d\s+[A-Z]$', text):
            _drop('sparse_digit_letter', dim)
            continue
        # 按非预期字符占比过滤长文本；已有几何形位框按其独立规则处理。
        
        if len(text) > 5 and dim.get('source') != 'gdt_line_frame':
            # 符号白名单覆盖分类器支持的形位符号。
            _allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,+-⌀±°⊕◎⊥∥○⌭∠⌓⌒▱≡—()/')
            junk_chars = sum(1 for c in text if c not in _allowed)
            if junk_chars / len(text) > 0.25:
                _drop('junk_char_ratio', dim)
                continue

        # 按符号占比过滤乱码，形位内容采用独立门控。
        if dim.get('source') != 'gdt_line_frame' and _is_junk_by_symbol_ratio(text):
            _drop('symbol_ratio', dim)
            continue

        nom = dim.get('nominal', '').strip()
        _is_gdt_frame = dim.get('source') == 'gdt_line_frame'
        if not _is_gdt_frame:
            _normalize_duplicate_diameter_prefix_fragment(dim)
            _normalize_leading_decimal_dimension(dim)
            _normalize_trailing_ocr_letter_noise(dim)
            text = dim.get('text', '').strip()
            nom = dim.get('nominal', '').strip()

        # 裸低置信度数值采用保守过滤，保留带公差或显式工程前缀的候选。
        
        
        _src = dim.get('source', '')
        _near_tol_fragment_evidence = bool(dim.get('_near_tolerance_fragment_evidence'))
        _geometry_evidence = bool(
            dim.get('_dimension_line_evidence')
            or dim.get('_leader_line_evidence')
            or dim.get('_strip_ocr_evidence')
            or _candidate_geometry_evidence(dim)
        )
        if (_src.startswith('ocr')
                and str(dim.get('confidence') or '').lower() == 'low'
                and not _is_gdt_frame):
            _has_tol = bool(
                dim.get('upper_tol') or dim.get('lower_tol') or
                re.search(r'(?:±|[+\-])\s*\d', text)
            )
            _has_prefix = bool(dim.get('prefix') or re.match(
                r'^(?:\d+\s*[xX×]|[⌀∅Ø⊘φΦR])',
                text.strip(),
            ))
            if (not _has_tol and not _has_prefix
                    and re.fullmatch(r'\d+(?:\.\d+)?', text.strip())):
                _bb = dim.get('bbox') or {}
                _cx = _bb.get('x', 0) + _bb.get('w', 0) / 2
                _cy = _bb.get('y', 0) + _bb.get('h', 0) / 2
                _edge = False
                if page_width > 0 and page_height > 0:
                    _edge = (_cx < page_width * 0.05 or _cx > page_width * 0.95 or
                             _cy < page_height * 0.05 or _cy > page_height * 0.95)
                _vertical_text = _is_vertical_orientation(dim.get('orientation'))
                _parsed_numeric = _parse_nominal(text.strip())
                _decimal_dim_candidate = (
                    '.' in text
                    and not _edge
                    and _parsed_numeric is not None
                    and abs(_parsed_numeric) >= 0.01
                    and abs(_parsed_numeric) <= 99
                    and (
                        (8 <= _bb.get('w', 0) <= 70 and 7 <= _bb.get('h', 0) <= 35)
                        or (_vertical_text
                            and 5 <= _bb.get('w', 0) <= 35
                            and 7 <= _bb.get('h', 0) <= 70)
                    )
                )
                _vertical_short_integer_candidate = (
                    _vertical_text
                    and not _edge
                    and re.fullmatch(r'\d{1,2}', text.strip())
                    and 6 <= _bb.get('w', 0) <= 30
                    and 6 <= _bb.get('h', 0) <= 30
                )
                _geometry_short_integer_candidate = (
                    _geometry_evidence
                    and not _edge
                    and re.fullmatch(r'\d{1,2}', text.strip())
                )
                if not (
                    _decimal_dim_candidate
                    or _vertical_short_integer_candidate
                    or _geometry_short_integer_candidate
                ):
                    if not _near_tol_fragment_evidence:
                        _drop('low_conf_naked_number', dim)
                        continue
            if (not _has_tol
                    and str(dim.get('type') or '').lower() == 'repeat'
                    and re.fullmatch(r'\d+\s*[xX×]\s*\d+(?:\.\d+)?', text.strip())):
                _bb = dim.get('bbox') or {}
                _vertical_repeat_candidate = (
                    _is_vertical_orientation(dim.get('orientation'))
                    and 8 <= _bb.get('w', 0) <= 70
                    and 7 <= _bb.get('h', 0) <= 80
                )
                if not _vertical_repeat_candidate:
                    _drop('low_conf_repeat_without_tol', dim)
                    continue

        if (_src.startswith('ocr')
                and str(dim.get('confidence') or '').lower() in {'medium', 'low'}
                and not _is_gdt_frame
                and not (dim.get('upper_tol') or dim.get('lower_tol') or dim.get('prefix'))
                and not _geometry_evidence
                and not _near_tol_fragment_evidence
                and re.fullmatch(r'\d{1,2}', text.strip())):
            if not (_is_vertical_orientation(dim.get('orientation'))
                    and str(dim.get('confidence') or '').lower() == 'low'):
                _drop('medium_low_short_int_without_evidence', dim)
                continue

        if (_src.startswith('ocr')
                and str(dim.get('confidence') or '').lower() in {'medium', 'low'}
                and not _is_gdt_frame
                and str(dim.get('type') or '').lower() == 'chamfer'
                and str(dim.get('prefix') or '').upper() == 'C'
                and not (dim.get('upper_tol') or dim.get('lower_tol'))
                and not _geometry_evidence
                and not _near_tol_fragment_evidence
                and re.fullmatch(r'C\s*\d{1,2}', text.strip(), flags=re.IGNORECASE)):
            _drop('chamfer_integer_without_evidence', dim)
            continue

        if (_src.startswith('ocr')
                and str(dim.get('confidence') or '').lower() == 'low'
                and not _is_gdt_frame
                and str(dim.get('type') or '').lower() == 'radius'
                and str(dim.get('prefix') or '').upper() == 'R'
                and not (dim.get('upper_tol') or dim.get('lower_tol'))
                and not _geometry_evidence
                and not _near_tol_fragment_evidence
                and re.fullmatch(r'R\s*\d{1,2}', text.strip(), flags=re.IGNORECASE)):
            _drop('low_conf_integer_radius_without_evidence', dim)
            continue

        if (_src.startswith('ocr')
                and str(dim.get('confidence') or '').lower() in {'medium', 'low'}
                and not _is_gdt_frame
                and not (dim.get('upper_tol') or dim.get('lower_tol'))
                and not _geometry_evidence
                and not _near_tol_fragment_evidence
                and re.fullmatch(r'\d+(?:\.\d+)?\s*±\s*', text.strip())):
            _drop('incomplete_pm_tolerance_fragment', dim)
            continue

        if not _is_gdt_frame and _has_incomplete_zero_tolerance(dim):
            _drop('incomplete_zero_tolerance', dim)
            continue

        if (not _is_gdt_frame
                and _src.endswith('_split')
                and not (dim.get('upper_tol') or dim.get('lower_tol'))
                and re.fullmatch(r'\d{1,3}°', text)
                and (_parse_nominal(text) or 999.0) <= 5.0
                and re.search(r'[+\-]\s*\d|°.*°', str(dim.get('_split_from_text') or ''))
                and not _geometry_evidence):
            _drop('angle_split_fragment', dim)
            continue

        if (not _is_gdt_frame
                and _src.endswith('_split')
                and not (dim.get('upper_tol') or dim.get('lower_tol'))
                and not dim.get('prefix')
                and not _geometry_evidence
                and re.fullmatch(r'\d{1,3}(?:\.\d+)?°', text)):
            _bb = dim.get('bbox') or {}
            _split_from = str(dim.get('_split_from_text') or '')
            _narrow_vertical_sliver = (
                0 < _bb.get('w', 0) <= 12
                and _bb.get('h', 0) >= max(14.0, _bb.get('w', 0) * 1.8)
            )
            if _narrow_vertical_sliver and re.search(r'°\s*\d', _split_from):
                _drop('narrow_angle_split_fragment', dim)
                continue

        if (not _is_gdt_frame
                and _src.startswith('ocr')
                and str(dim.get('confidence') or '').lower() in {'medium', 'low'}
                and not dim.get('upper_tol')
                and dim.get('lower_tol')
                and not dim.get('prefix')
                and not _geometry_evidence
                and re.fullmatch(r'0\.\d+\s*-\s*0\.\d+', text)):
            _drop('standalone_lower_tol_decimal_fragment', dim)
            continue

        if (not _is_gdt_frame
                and _src.startswith('ocr')
                and not (dim.get('upper_tol') or dim.get('lower_tol'))
                and not dim.get('prefix')
                and not _geometry_evidence
                and re.fullmatch(r'0\.\d+', text)):
            _small_val = _parse_nominal(text)
            if _small_val is not None and _small_val < 0.1:
                _drop('standalone_tolerance_decimal_fragment', dim)
                continue

        if _is_gdt_frame:
            _gdt_nums = re.findall(r'\d+(?:\.\d+)?', text)
            if '(unread)' in text.lower() or not _gdt_nums:
                _drop('gdt_unread_or_no_number', dim)
                continue
            if not _gdt_allows_integer_value(dim, _gdt_nums):
                _drop('gdt_no_decimal_number', dim)
                continue

        # 已认领形位框在公称值解析失败、跨格多值、非法字母或陌生词混入时降级为未读状态，保留其几何候选。
        
        
        
        
        
        
        
        
        
        if _is_gdt_frame and nom:
            _garbage = _parse_nominal(nom) is None
            if not _garbage and re.search(r'\s', nom):
                _num_tokens = re.findall(r'\d+\.?\d*', nom)
                if len(_num_tokens) > 1:
                    _garbage = True
            if not _garbage:
                _residual = re.sub(r'[0-9A-Z.\s+\-±°⌀∅ØRMCxX()/φΦ]', '', nom, flags=re.IGNORECASE)
                if _residual:
                    _garbage = True
            # text 层乱码（nom 碰巧能 parse 出单个数但 text 有 bleed-through 杂字）
            if not _garbage:
                _text_nums = re.findall(r'\d+\.?\d*', text)
                if len(set(_text_nums)) >= 2:
                    _garbage = True
            if not _garbage:
                for _tok in re.split(r'\s+', text.strip()):
                    if len(_tok) < 4:
                        continue
                    if re.fullmatch(r'\(?[MLPmlpFf]\)?', _tok):
                        continue
                    if re.match(r'^[⌀⊘∅ØR]?\d', _tok):
                        # 直径、半径或数字开头的 token 按工程词汇处理。
                        continue
                    _letters = re.findall(r'[A-Za-z]', _tok)
                    if len(_letters) >= 2:
                        _garbage = True
                        break
            if _garbage:
                # rect-origin 框走到这一步 = 几何弱证据 + OCR 乱码，
                # 必定是假阳性（标题栏/参数表），直接丢弃；line-origin 保留给人工校验
                if dim.get('_origin') == 'rect':
                    if VERBOSE:
                        print(f"    [post_filter] drop rect-origin garbled gdt: '{nom}'")
                    _drop('gdt_rect_origin_garble', dim)
                    continue
                _sym_match = re.match(r'^\s*([⌀⊘∅Ø⊕◎▱⊥∥○⌭∠⌓⌒≡—])', text)
                _sym = _sym_match.group(1) if _sym_match else 'GD&T'
                dim['text'] = f"{_sym} (unread)"
                dim['nominal'] = ''
                nom = ''
                text = dim['text']

        # 空 nominal：非 gdt_line_frame 视为孤立公差碎片丢弃
        _is_limit_dim = str(dim.get('type') or '').startswith('limit_')
        if not nom and not _is_gdt_frame and not _is_limit_dim:
            _drop('empty_nominal', dim)
            continue

        # 以下 nominal 格式校验仅适用于非 gdt_line_frame
        if not _is_gdt_frame:
            # 开括号起始的公称值视为截断碎片。
            if nom and nom[0] in '()[]{}':
                _drop('nominal_starts_bracket', dim)
                continue
            # 符号起始的公称值视为尚未合并的公差。
            if nom and nom[0] in '+-':
                _drop('nominal_starts_sign', dim)
                continue
            # 公称值必须能解析为合法数值。
            if nom and _parse_nominal(nom) is None:
                if _is_zero_nominal_with_real_tolerance(dim, nom, text):
                    text = _normalize_zero_nominal_tolerance(dim, text)
                    nom = '0'
                else:
                    _limit = _extract_max_min_limit(text)
                    if _limit:
                        _kind, _value = _limit
                        dim['type'] = 'limit_min' if _kind == 'MIN' else 'limit_max'
                        dim['prefix'] = _kind
                        dim['limit_value'] = _value
                        dim['nominal'] = ''
                        nom = ''
                    else:
                        _drop('nominal_parse_failed', dim)
                        continue
            # 过滤空格分隔的多数值公称字段。
            if nom and re.search(r'\s', nom):
                _num_tokens = re.findall(r'\d+\.?\d*', nom)
                if len(_num_tokens) > 1:
                    _drop('nominal_multi_number_tokens', dim)
                    continue
            # 过滤括号不平衡的公称字段。
            if nom and (nom.count('(') != nom.count(')') or nom.count('[') != nom.count(']')):
                _drop('nominal_unbalanced_brackets', dim)
                continue
            # 过滤含非法字母的公称字段。
            if nom:
                _limit = _extract_max_min_limit(text)
                if _limit:
                    _kind, _value = _limit
                    dim['type'] = 'limit_min' if _kind == 'MIN' else 'limit_max'
                    dim['prefix'] = _kind
                    dim['limit_value'] = _value
                    dim['nominal'] = ''
                    nom = ''
                else:
                    _legal = re.sub(r'[0-9.\s+\-±°⌀∅ØRMCxX()/φΦ]', '', nom, flags=re.IGNORECASE)
                    if _legal and not _is_max_min_suffix_dimension(nom):
                        _drop('nominal_illegal_letters', dim)
                        continue

        # 无数字的长文本按说明标签过滤，已认领形位框保留未读状态供人工校验。
        
        
        if (len(text) >= 8 and not re.search(r'\d', text)
                and dim.get('source') != 'gdt_line_frame'):
            _drop('long_text_without_number', dim)
            continue

        # 对各普通尺寸来源应用元数据黑名单；形位内容采用独立词汇规则。
        
        
        if _src != 'gdt_line_frame':  
            _check_text = text
            # 检查元数据前先剥离末尾公差，避免把合法公差解释为标号。
            _core = re.sub(r'\s+[+\-±][\d./\-+]+.*$', '', _check_text).strip()
            _check_candidates = {_check_text, _core, nom}
            _hit = False
            for _ct in _check_candidates:
                if not _ct:
                    continue
                for pat in METADATA_BLACKLIST_PATTERNS:
                    if pat.search(_ct):
                        _hit = True
                        break
                if _hit:
                    break
            if _hit:
                _drop('metadata_blacklist', dim)
                continue

        # 排除页面边缘的短整数碎片。
        
        _has_prefix = bool(dim.get('prefix'))
        _has_tol = bool(dim.get('upper_tol') or dim.get('lower_tol'))
        _has_symbol = bool(re.search(r'[⌀±°⊕◎R⊥∥○⌭∠⌓⌒▱≡—⊘∅Ø]', text))
        if page_width > 0 and page_height > 0 and _src != 'gdt_line_frame':
            _b = dim.get('bbox', {})
            _cx = _b.get('x', 0) + _b.get('w', 0) / 2
            _cy = _b.get('y', 0) + _b.get('h', 0) / 2
            _in_edge = (_cx < page_width * 0.05 or _cx > page_width * 0.95 or
                        _cy < page_height * 0.05 or _cy > page_height * 0.95)
            if _in_edge:
                # 无前缀、公差和工程符号的短整数按边界碎片处理。
                if (re.match(r'^\d{1,3}$', nom or text) and
                        not _has_prefix and not _has_tol and not _has_symbol):
                    _drop('edge_short_integer', dim)
                    continue

        # 按小包围框和单字符条件排除 OCR 噪点。
        if (_src == 'ocr' and dim.get('confidence') == 'low'
                and not _has_prefix and not _has_tol and not _has_symbol
                and re.match(r'^\d$', nom or text)):
            _bb = dim.get('bbox', {})
            if (_bb.get('w', 99) < 15 and _bb.get('h', 99) < 10
                    and not _is_vertical_orientation(dim.get('orientation'))):
                _drop('small_single_digit_noise', dim)
                continue

        # -- 误报过滤 --
        if not _is_gdt_frame:
            # 过滤小数点位于边缘的截断数值。
            _stripped_nom = (nom or text).strip()
            if _stripped_nom and (re.match(r'^\.\d', _stripped_nom) or
                                  re.match(r'^\d+\.\s*$', _stripped_nom)):
                if VERBOSE:
                    print(f"    [FP1] drop fragment: '{text}'")
                _drop('fp_decimal_fragment', dim)
                continue
            # 过滤超过整数长度限制的裸数值。
            if nom and re.match(r'^\d{5,}$', nom):
                if VERBOSE:
                    print(f"    [FP2] drop large integer: '{text}'")
                _drop('fp_large_integer', dim)
                continue
            # 按括号结构过滤参考尺寸。
            
            if re.match(r'^\([\d.]+\)$', _stripped_nom or text.strip()):
                if VERBOSE:
                    print(f"    [FP4] drop ref dim in parens: '{text}'")
                _drop('fp_parenthesized_reference', dim)
                continue
            # 无公差和工程符号的长小数按坐标表内容处理。
            if (not _has_tol and not _has_symbol
                    and _src in ('ocr', 'scored', 'hires_retry')
                    and re.match(r'^\d+\.\d{3,}$', _stripped_nom)):
                if VERBOSE:
                    print(f"    [FP5] drop 3-decimal coordinate: '{text}'")
                _drop('fp_three_decimal_coordinate', dim)
                continue
            # 无工程前缀或公差的低置信度裸数字采用保守过滤。
            
            if (_src == 'ocr' and not _has_tol and not _has_symbol
                    and dim.get('confidence') == 'low'):
                _has_nx = bool(re.match(r'^\d+[xX]\s', text))
                _has_suffix = bool(re.search(r'[MmHhKk]$', nom or ''))
                if not _has_nx and not _has_suffix:
                    if (re.match(r'^\d{1,2}$', _stripped_nom)
                            and not _is_vertical_orientation(dim.get('orientation'))):
                        if VERBOSE:
                            print(f"    [FP3a] drop bare short int (low conf): '{text}'")
                        _drop('fp_low_conf_short_int', dim)
                        continue

        # 检查公差幅值，排除异常拼接。
        _drop_dim = False
        if nom:
            _nom_val = _parse_nominal(nom)
            if _nom_val is not None and _nom_val > 0:
                for _tk in ('upper_tol', 'lower_tol'):
                    _tv = dim.get(_tk)
                    if not _tv:
                        continue
                    _tval = _parse_nominal(str(_tv).lstrip('+-'))
                    if _tval is None:
                        continue
                    if _tval > _nom_val * 10:
                        _drop_dim = True
                        break
                    if _tval > max(_nom_val * 0.5, 1.0):
                        if (re.search(r'[+\-]\s*\d{2,}$', text)
                                and not re.search(r'[+\-]\s*0?\.', text)):
                            _drop_dim = True
                            break
                        dim[_tk] = None
        if _drop_dim:
            _drop('tolerance_sanity_drop', dim)
            continue

        filtered.append(dim)

    def _bbox_keys_present(bbox: dict) -> bool:
        return all(k in bbox for k in ('x', 'y', 'w', 'h'))

    def _repeat_prefix_from_dim(dim: dict) -> str | None:
        prefix = str(dim.get('prefix') or '').strip()
        m = re.match(r'^([2-9]\d?)\s*[xX×]', prefix)
        if not m:
            m = re.match(r'^\s*([2-9]\d?)\s*[xX×]\s+', str(dim.get('text') or ''))
        if not m:
            return None
        return f'{m.group(1)}X'

    def _has_tol(dim: dict) -> bool:
        return bool(
            dim.get('upper_tol') or dim.get('lower_tol') or
            re.search(r'(?:±|[+\-])\s*\d', str(dim.get('text') or ''))
        )

    def _covered_by_repeat_tolerance_dimension(dim: dict, repeat_tol_dims: list[dict]) -> bool:
        if dim.get('source') == 'gdt_line_frame' or _has_tol(dim):
            return False
        nominal = str(dim.get('nominal') or '').strip()
        if not nominal or _parse_nominal(nominal) is None:
            return False
        bbox = dim.get('bbox') or {}
        if not _bbox_keys_present(bbox):
            return False
        dim_prefix = _repeat_prefix_from_dim(dim)
        dcx, dcy = _bbox_center(bbox)

        for full in repeat_tol_dims:
            if full is dim:
                continue
            if str(full.get('nominal') or '').strip() != nominal:
                continue
            full_prefix = _repeat_prefix_from_dim(full)
            if not full_prefix:
                continue
            if dim_prefix and dim_prefix != full_prefix:
                continue
            full_bbox = full.get('bbox') or {}
            if not _bbox_keys_present(full_bbox):
                continue
            if _bbox_iou_min(bbox, full_bbox) >= 0.70:
                return True
            if _point_in_bbox(dcx, dcy, full_bbox, margin=3.0):
                return True
        return False

    repeat_tol_dims = [
        dim for dim in filtered
        if dim.get('source') != 'gdt_line_frame'
        and _repeat_prefix_from_dim(dim)
        and _has_tol(dim)
    ]
    if repeat_tol_dims:
        deduped = []
        for dim in filtered:
            if _covered_by_repeat_tolerance_dimension(dim, repeat_tol_dims):
                _drop('covered_by_repeat_tolerance_dimension', dim)
                continue
            deduped.append(dim)
        filtered = deduped

    def _covered_by_gdt_line_frame(dim: dict, gdt_dims: list[dict]) -> bool:
        if dim.get('source') == 'gdt_line_frame' or _has_tol(dim):
            return False
        if str(dim.get('source') or '') not in {'ocr', 'ocr_split', 'hires_retry'}:
            return False
        nominal = str(dim.get('nominal') or '').strip()
        nominal_value = _parse_nominal(nominal)
        if nominal_value is None:
            return False
        bbox = dim.get('bbox') or {}
        if not _bbox_keys_present(bbox):
            return False
        dcx, dcy = _bbox_center(bbox)
        for gdt in gdt_dims:
            gdt_nominal = _parse_nominal(str(gdt.get('nominal') or '').strip())
            if gdt_nominal is None or abs(gdt_nominal - nominal_value) > 1e-6:
                continue
            gdt_bbox = gdt.get('bbox') or {}
            if not _bbox_keys_present(gdt_bbox):
                continue
            if _bbox_iou_min(bbox, gdt_bbox) >= 0.70:
                return True
            if _point_in_bbox(dcx, dcy, gdt_bbox, margin=3.0):
                return True
        return False

    gdt_dims = [
        dim for dim in filtered
        if dim.get('source') == 'gdt_line_frame'
        and _parse_nominal(str(dim.get('nominal') or '').strip()) is not None
    ]
    if gdt_dims:
        deduped = []
        for dim in filtered:
            if _covered_by_gdt_line_frame(dim, gdt_dims):
                _drop('covered_by_gdt_line_frame', dim)
                continue
            deduped.append(dim)
        filtered = deduped

    if VERBOSE:
        src_before = Counter(d.get('source', '?') for d in dimensions)
        src_after = Counter(d.get('source', '?') for d in filtered)
        drops = {s: src_before[s] - src_after.get(s, 0)
                 for s in src_before if src_before[s] != src_after.get(s, 0)}
        if drops:
            print(f"[post_filter] drops by source: {drops}")
        if drop_reasons:
            print(f"[post_filter] drops by reason: {dict(drop_reasons.most_common())}")
            for reason, examples in drop_examples.items():
                print(f"[post_filter.reason] {reason}: " + " | ".join(examples))
        kept_ids = {id(d) for d in filtered}
        gdt_dropped = [d for d in dimensions
                       if d.get('source') == 'gdt_line_frame' and id(d) not in kept_ids]
        for d in gdt_dropped[:30]:
            bb = d.get('bbox', {})
            print(f"[post_filter] DROP gdt_line_frame text={d.get('text','')!r} "
                  f"bbox=({bb.get('x',0):.0f},{bb.get('y',0):.0f},{bb.get('w',0):.0f}x{bb.get('h',0):.0f})")

    if debug is not None:
        debug['post_filter_drop_ledger'] = post_filter_ledger
        debug['post_filter_debug_stats'] = {
            'drop_count': len(post_filter_ledger),
            'drop_reason_counts': dict(Counter(d['reason'] for d in post_filter_ledger)),
            'drop_source_counts': dict(Counter(d['source'] for d in post_filter_ledger)),
        }

    return filtered


# ---------------------------------------------------------------------------
# OCR 打分
# ---------------------------------------------------------------------------

def _score_and_filter(
    unclaimed: list,
    frame_bboxes: list,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> list:
    """
    对未认领 OCR 结果打分，score >= SCORE_THRESHOLD 的纳入尺寸候选。
    """
    unclaimed = _merge_symbol_numeric_fragments(unclaimed)
    if VERBOSE:
        print(f"[score] frame_bboxes={len(frame_bboxes)}, page={page_width}x{page_height}, unclaimed={len(unclaimed)}")

    # -- 右侧注释区检测 --
    annotation_region_indices = set()
    if page_width > 0:
        right_threshold = page_width * 0.6
        hint_indices = []
        for i, r in enumerate(unclaimed):
            text = r['text'].strip()
            if not text:
                continue
            cx, _ = _bbox_center(r['bbox_in_pdf'])
            if cx <= right_threshold:
                continue
            for pat in _RIGHT_ANNOTATION_PATTERNS:
                if pat.search(text):
                    hint_indices.append(i)
                    break
        if len(hint_indices) >= 3:
            hint_ys = [_bbox_center(unclaimed[idx]['bbox_in_pdf'])[1]
                       for idx in hint_indices]
            y_min = min(hint_ys) - 20
            y_max = max(hint_ys) + 20
            for i, r in enumerate(unclaimed):
                cx, cy = _bbox_center(r['bbox_in_pdf'])
                if cx > right_threshold and y_min <= cy <= y_max:
                    annotation_region_indices.add(i)
            if annotation_region_indices and VERBOSE:
                print(f"[score] right annotation region: {len(annotation_region_indices)} regions, "
                      f"y=[{y_min:.0f}, {y_max:.0f}]")

    # -- 左下注释区检测 --
    if page_width > 0 and page_height > 0:
        left_threshold = page_width * 0.25
        bottom_threshold = page_height * 0.5
        left_hint_indices = []
        for i, r in enumerate(unclaimed):
            text = r['text'].strip()
            if not text:
                continue
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            if cx >= left_threshold or cy <= bottom_threshold:
                continue
            for pat in _LEFT_ANNOTATION_PATTERNS:
                if pat.search(text):
                    left_hint_indices.append(i)
                    break
        if len(left_hint_indices) >= 2:
            lx_min = min(unclaimed[i]['bbox_in_pdf']['x'] for i in left_hint_indices) - 30
            ly_min = min(unclaimed[i]['bbox_in_pdf']['y'] for i in left_hint_indices) - 30
            lx_max = max(unclaimed[i]['bbox_in_pdf']['x'] + unclaimed[i]['bbox_in_pdf']['w'] for i in left_hint_indices) + 30
            ly_max = max(unclaimed[i]['bbox_in_pdf']['y'] + unclaimed[i]['bbox_in_pdf']['h'] for i in left_hint_indices) + 30
            left_ann_count = 0
            for i, r in enumerate(unclaimed):
                cx, cy = _bbox_center(r['bbox_in_pdf'])
                if lx_min <= cx <= lx_max and ly_min <= cy <= ly_max:
                    annotation_region_indices.add(i)
                    left_ann_count += 1
            if left_ann_count and VERBOSE:
                print(f"[score] left annotation region: {left_ann_count} regions, "
                      f"bbox=[{lx_min:.0f},{ly_min:.0f},{lx_max:.0f},{ly_max:.0f}]")

    out = []

    def _confidence_from_score(score: int, source_row: dict) -> str:
        level = 'high' if score >= 80 else 'medium' if score >= 40 else 'low'
        if (source_row.get('source') == 'candidate_guided_ocr'
                and source_row.get('candidate_priority_hint') == 'low'):
            evidence = source_row.get('candidate_evidence_summary') or {}
            has_geometry_evidence = bool(
                source_row.get('candidate_has_geometry_evidence')
                or evidence.get('has_capsule')
                or evidence.get('has_dimline')
                or evidence.get('has_arrow')
            )
            if not has_geometry_evidence:
                return 'low'
            if level == 'high':
                return 'medium'
        return level

    for ri, r in enumerate(unclaimed):
        text = r['text'].strip()
        if not text:
            continue

        # -- 硬性预过滤（黑名单） --
        if len(text) == 1 and not text.isalnum():
            continue
        if text in ('0', '00'):
            continue
        if any(pat.search(text) for pat in METADATA_BLACKLIST_PATTERNS):
            continue

        # paren-reference fast-skip：raw OCR 或当前 text 是 `(N)` 形态直接丢弃
        # 工程图惯例 (N) = 参考尺寸 REF，不参与公差检验，必须不输出 stamp。
        _raw_q = r.get('_raw_text', text).strip()
        if (re.match(r'^\([\d.]+\)$', text) or
                re.match(r'^\([\d.]+\)$', _raw_q)):
            if VERBOSE:
                print(f"    [paren_ref] skip score-path ref: '{text}' raw='{_raw_q}'")
            continue

        if re.fullmatch(r'(?i)(?:MAX|MIN|M)\.\d', text.strip()):
            if VERBOSE:
                print(f"    [minmax_fragment] skip score-path fragment: '{text}'")
            continue

        # 规范兼容路径的正负公差符号变体。
        text = _fixup_ocr_text(text)
        try:
            _orient_abs = abs(float(r.get('orientation') or 0))
        except (TypeError, ValueError):
            _orient_abs = 0.0
        if (r.get('source') in {'capsule_ocr', 'strip_ocr', 'dimline_strip_ocr'}
                or _orient_abs >= 45):
            text = _fixup_fc_capsule_text(text)
        r['text'] = text

        # 微小面积过滤
        bbox = r['bbox_in_pdf']
        if bbox['w'] * bbox['h'] < 50:
            continue

        # 单字母边缘过滤
        if re.match(r'^[A-Z]$', text) and page_width > 0 and page_height > 0:
            cx, cy = _bbox_center(bbox)
            margin_x = page_width * 0.05
            margin_y = page_height * 0.05
            if (cx < margin_x or cx > page_width - margin_x or
                    cy < margin_y or cy > page_height - margin_y):
                continue

        score = 0
        _score_rules = []

        # 注释区降分
        if ri in annotation_region_indices:
            score -= 50
            _score_rules.append('-50(annotation_region)')

        # 正向信号
        if '±' in text:
            score += 50; _score_rules.append('+50(has_±)')
        if re.search(r'\d\.?\d*\s*t\s*\d', text):
            score += 40; _score_rules.append('+40(ocr_t_as_±)')
        if '⌀' in text:
            score += 50; _score_rules.append('+50(has_⌀)')
        angle_label_evidence = bool(
            r.get('_angle_label_evidence')
            or r.get('source') == 'angle_label_ocr'
        )
        if '°' in text:
            score += 40; _score_rules.append('+40(has_°)')
            if not angle_label_evidence:
                cx0, cy0 = _bbox_center(r['bbox_in_pdf'])
                for nb in unclaimed:
                    if nb is r:
                        continue
                    if _SECTION_LABEL_PATTERN.match(nb['text'].strip()):
                        nx, ny = _bbox_center(nb['bbox_in_pdf'])
                        if math.hypot(nx - cx0, ny - cy0) <= 120:
                            score -= 60; _score_rules.append('-60(°_near_section_label)')
                            break
        if re.match(r'^\d+\.?\d*$', text) and '°' not in text and '±' not in text:
            cx0, cy0 = _bbox_center(bbox)
            _is_vert = abs(r.get('orientation', 0)) >= 45
            for nb in unclaimed:
                if nb is r:
                    continue
                if _SECTION_LABEL_PATTERN.match(nb['text'].strip()):
                    nx, ny = _bbox_center(nb['bbox_in_pdf'])
                    if math.hypot(nx - cx0, ny - cy0) <= 120:
                        _penalty = 15 if _is_vert else 40
                        score -= _penalty; _score_rules.append(f'-{_penalty}(decimal_near_section_label{"_vert" if _is_vert else ""})')
                        break

        if 'MAX' in text.upper() or 'MIN' in text.upper():
            score += 40; _score_rules.append('+40(has_MAX/MIN)')
        if re.search(r'[+\-]\s*[\d.]+', text):
            score += 45; _score_rules.append('+45(has_+/-_digit)')
        if re.match(r'^[\d.]+$', text):
            score += 30; _score_rules.append('+30(pure_number)')
        dimline_evidence = r.get('_dimension_line_evidence')
        if dimline_evidence:
            dimline_kind = dimline_evidence.get('kind') if isinstance(dimline_evidence, dict) else None
            # Defensive: current writers route leader evidence to
            # _leader_line_evidence; this fail-safe blocks future leader-kind
            # rows from receiving paired dimension-line boost.
            if dimline_kind == 'single_arrow_leader':
                _score_rules.append('+0(leader_evidence_ocr_only)')
            else:
                score += 25; _score_rules.append('+25(dimline_evidence)')
        if r.get('_strip_ocr_evidence') and not dimline_evidence:
            score += 10; _score_rules.append('+10(strip_ocr_evidence)')
        candidate_geometry = _candidate_geometry_evidence(r)
        if candidate_geometry:
            score += 25; _score_rules.append('+25(candidate_geometry_evidence)')
        if r.get('_leader_line_evidence'):
            _leader_text_like_dim = bool(
                re.search(r'[RrCcMm⌀∅ØφΦ°±]', text)
                or re.search(r'[+\-]\s*[\d.]+', text)
                or re.search(r'(?:MAX|MIN)\.?$', text, re.IGNORECASE)
                or re.match(r'^\d+\s*[×xX]\s*[\d.R(O⌀∅ØφΦ]', text)
            )
            if not _leader_text_like_dim:
                score -= 30; _score_rules.append('-30(leader_weak_text)')
        if re.match(r'^\d+\s*[×xX][\d.]+', text):
            score += 35; _score_rules.append('+35(NxM_pattern)')
        if re.match(r'^\d+[Xx]\s*[\d.R(O⌀]', text):
            score += 35; _score_rules.append('+35(NX_prefix)')
        if re.match(r'^R[\d.]+', text):
            score += 35; _score_rules.append('+35(R_prefix)')
        if re.match(r'^[⌀∅ØφM][\d.]+', text):
            score += 45; _score_rules.append('+45(symbol_prefix)')
        # 识别表面粗糙度前缀。
        if re.match(r'^R(?:a|z|max)[\d.]+$', text, re.IGNORECASE):
            score += 35; _score_rules.append('+35(surface_rough_prefix)')
        # 最大值词汇的缩写恢复限定于满足数字结构的候选，避免与单位和短碎片混淆。
        
        
        if re.match(r'^(?:\d+\.\d+|\d{2,})\s*(?:MAX|MIN|m)\.?$', text):
            score += 25; _score_rules.append('+25(max_suffix_remnant)')
        # 降低括号参考尺寸的评分。
        if re.match(r'^\([\d.]+\)$', text):
            score -= 30; _score_rules.append('-30(ref_dim_parens)')
        if re.match(r'^\(\d{3,}\)$', text):
            score -= 30; _score_rules.append('-30(parens_3+digit_int)')
        if re.match(r'^[\d.]+\s+[\d.]+$', text):
            score += 30; _score_rules.append('+30(num_space_num)')
        if re.match(r'^[\d.]+\s*[:;]\s*[\d.]+$', text):
            score += 30; _score_rules.append('+30(num_colon_num)')

        # 负向信号
        if re.match(r'^\d{1,2}$', text):
            score -= 10; _score_rules.append('-10(1-2_digit_int)')
        if re.match(r'^\d{4,}$', text):
            score -= 15; _score_rules.append('-15(4+digit_int)')
        if _RATIO_PATTERN.match(text):
            score -= 60; _score_rules.append('-60(ratio_pattern)')
        if _WORD_PATTERN.match(text):
            score -= 40; _score_rules.append('-40(word_pattern)')
        if _ALLCAPS_LONG_PATTERN.match(text) and not any(c in text for c in _INDUSTRIAL_SYMBOLS):
            score -= 70; _score_rules.append('-70(allcaps_long)')
        _en_tokens = [tok for tok in text.split() if re.match(r'[A-Za-z]{2,}', tok)]
        if len(_en_tokens) >= 3:
            score -= 50; _score_rules.append('-50(multi_word_en)')
        if text.rstrip().endswith('.') or text.rstrip().endswith(','):
            score -= 20; _score_rules.append('-20(trailing_punct)')
        upper_text = text.upper()
        for kw in FRAME_KEYWORDS:
            if kw in upper_text:
                score -= 50; _score_rules.append(f'-50(frame_kw:{kw})')
                break

        # 图框标题栏区域减分
        if frame_bboxes:
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            page_area = page_width * page_height if page_width > 0 and page_height > 0 else 0
            for fb in frame_bboxes:
                if page_area > 0 and (fb['w'] * fb['h']) > page_area * 0.4:
                    continue
                if _point_in_bbox(cx, cy, fb):
                    score -= 40; _score_rules.append('-40(in_title_block)')
                    break

        # 图框编号过滤
        if re.match(r'^\d{1,2}$', text) and frame_bboxes:
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            for fb in frame_bboxes:
                fx, fy, fw, fh = fb['x'], fb['y'], fb['w'], fb['h']
                dist_to_edge = min(
                    abs(cx - fx),
                    abs(cx - (fx + fw)),
                    abs(cy - fy),
                    abs(cy - (fy + fh)),
                )
                if dist_to_edge < 15:
                    score -= 30; _score_rules.append('-30(near_frame_edge)')
                    break

        # 低置信度 region 减分
        if r.get('low_confidence_region'):
            score -= 20; _score_rules.append('-20(low_conf_region)')

        # 图框边缘区域减分
        if page_width > 0 and page_height > 0:
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            is_vertical = abs(r.get('orientation', 0)) >= 45
            left_thresh  = 0.005 if is_vertical else 0.02
            right_thresh = 0.005 if is_vertical else 0.02
            top_thresh   = 0.02
            bot_thresh   = 0.02
            if (cx < page_width * left_thresh or
                    cx > page_width * (1 - right_thresh) or
                    cy < page_height * top_thresh or
                    cy > page_height * (1 - bot_thresh)):
                score -= 25; _score_rules.append('-25(page_edge_2%)')

        # GD&T 碎片数字降分
        if re.match(r'^\d*\.\d+$', text):
            try:
                _val = float(text)
            except ValueError:
                _val = 999
            if _val < 2.0:
                cx0, cy0 = _bbox_center(bbox)
                for nb in unclaimed:
                    if nb is r:
                        continue
                    nt = nb['text'].strip()
                    if any(c in nt for c in _GDT_SYMBOL_CHARS) or re.match(r'^[A-Z]$', nt):
                        nx, ny = _bbox_center(nb['bbox_in_pdf'])
                        if math.hypot(nx - cx0, ny - cy0) <= 30:
                            score -= 30; _score_rules.append('-30(gdt_fragment)')
                            break

        # 角度标注特征降分
        _angle_clean = text.strip().replace('°', '').replace('\u3002', '').replace(' ', '')
        try:
            _angle_val = float(_angle_clean)
        except ValueError:
            _angle_val = None
        if (_angle_val is not None and 0 < _angle_val < 90
                and '°' in text
                and not any(c in text for c in '±')
                and re.match(r'^[\d.]+°$', text.strip())):
            try:
                _conf = float(r.get('confidence') or r.get('conf') or 0.0)
            except (TypeError, ValueError):
                _conf = 0.0
            if _conf < 0.75:
                score -= 40; _score_rules.append('-40(angle_like_low_conf)')

        # 图框网格编号过滤
        if re.match(r'^\d{1,2}$', text) and page_width > 0 and page_height > 0:
            cx, cy = _bbox_center(bbox)
            margin_x = page_width * 0.05
            margin_y = page_height * 0.05
            if (cx < margin_x or cx > page_width - margin_x or
                    cy < margin_y or cy > page_height - margin_y):
                score -= 50; _score_rules.append('-50(grid_number_edge)')

        # 公差碎片邻近加分
        if re.match(r'^[\d.]+$', text) and score < SCORE_THRESHOLD:
            cx0, cy0 = _bbox_center(bbox)
            for nb in unclaimed:
                if nb is r:
                    continue
                nt = _fixup_ocr_text(nb['text'].strip())
                if re.match(r'^[+\-]\s*[\d.]+', nt):
                    nb_bbox = nb['bbox_in_pdf']
                    nx, ny = _bbox_center(nb_bbox)
                    dist = math.hypot(nx - cx0, ny - cy0)
                    same_row = abs(ny - cy0) <= max(
                        float(bbox.get('h') or 0.0),
                        float(nb_bbox.get('h') or 0.0),
                    ) * 0.75
                    horizontal_gap = float(nb_bbox.get('x') or 0.0) - (
                        float(bbox.get('x') or 0.0) + float(bbox.get('w') or 0.0)
                    )
                    gap_limit = max(18.0, min(
                        28.0,
                        max(float(bbox.get('w') or 0.0),
                            float(nb_bbox.get('w') or 0.0)) * 1.15,
                    ))
                    same_row_gap_ok = (
                        same_row
                        and -max(4.0, float(bbox.get('w') or 0.0) * 0.25) <= horizontal_gap <= gap_limit
                    )
                    if dist <= 30 or same_row_gap_ok:
                        score += 25; _score_rules.append(f'+25(tol_fragment_boost<-{nt})')
                        r['_near_tolerance_fragment_evidence'] = {
                            'text': nt,
                            'distance': round(dist, 2),
                            'source': nb.get('source'),
                        }
                        if VERBOSE:
                            print(f"  [score] +25 tol_fragment_boost '{text}' <- '{nt}' dist={dist:.1f}")
                        break

        # 输出可选评分诊断。
        if text.strip() == '958':
            if VERBOSE:
                print(f"[score_debug_958] score={score} details: {' '.join(_score_rules)}")

        # [score_detail]
        if VERBOSE:
            _cx, _cy = _bbox_center(bbox)
            _orient = r.get('orientation', 0)
            _rules_str = ' '.join(_score_rules) if _score_rules else '(no rules triggered)'
            _verdict = 'PASS' if score >= SCORE_THRESHOLD else 'FAIL'
            print(f"[score_detail] text='{text}' pos=({_cx:.0f},{_cy:.0f}) orient={_orient} | {_rules_str} | total={score} -> {_verdict}")

        # [DIAG-MISS]
        if VERBOSE and any(k in r.get('text', '') for k in ['2X', '7X', '1.87']):
            print(f"  [DIAG-MISS] score: '{r['text']}' score={score}")

        if score < SCORE_THRESHOLD:
            continue

        merged_segments = _split_merged_dimension_text(text)
        if merged_segments:
            normalized_merged = _normalize_merged_dimension_text(text)
            for seg in merged_segments:
                seg_text = seg['text']
                dim_fields = _dimension_fields_from_text(seg_text)
                out.append({
                    **dim_fields,
                    'bbox': _bbox_for_text_span(
                        r['bbox_in_pdf'],
                        r.get('orientation', 0),
                        seg['start'],
                        seg['end'],
                        len(normalized_merged),
                    ),
                    'confidence': _confidence_from_score(score, r),
                    'source': 'ocr_split',
                    'orientation': r.get('orientation', 0),
                    '_score': score,
                    '_font_height': r.get('font_height_approx', 0),
                    '_paired': False,
                    '_raw_text': r.get('_raw_text', text),
                    '_split_from_text': text,
                    '_dimension_line_evidence': r.get('_dimension_line_evidence'),
                    '_strip_ocr_evidence': r.get('_strip_ocr_evidence'),
                    '_leader_line_evidence': r.get('_leader_line_evidence'),
                    '_candidate_geometry_evidence': candidate_geometry,
                    '_near_tolerance_fragment_evidence': r.get('_near_tolerance_fragment_evidence'),
                })
            continue

        # 拆分空白分隔的公称值与对称公差。
        if re.match(r'^[\d.]+\s+[\d.]+$', text):
            parts = text.split()
            if len(parts) == 2:
                text_clean_parsed, prefix_override, type_override = _extract_prefix(parts[0])
                tol_val = parts[1]
                _tol_reasonable = True
                try:
                    _tv = abs(float(tol_val))
                    _nv = abs(float(re.sub(r'[^\d.\-]', '', text_clean_parsed)))
                    _limit = max(_nv * 2.0, 0.5) if _nv < 5 else max(_nv * 0.5, 3.0)
                    if _tv >= _limit:
                        _tol_reasonable = False
                except (ValueError, ZeroDivisionError):
                    pass
                if _tol_reasonable:
                    out.append({
                        'text': text,
                        'nominal': text_clean_parsed,
                        'upper_tol': f'+{tol_val}',
                        'lower_tol': f'-{tol_val}',
                        'type': type_override,
                        'prefix': prefix_override,
                        'bbox': r['bbox_in_pdf'],
                        'confidence': _confidence_from_score(score, r),
                        'source': 'ocr',
                        'orientation': r.get('orientation', 0),
                        '_score': score,
                        '_font_height': r.get('font_height_approx', 0),
                        '_paired': False,
                        '_raw_text': r.get('_raw_text', text),
                        '_dimension_line_evidence': r.get('_dimension_line_evidence'),
                        '_strip_ocr_evidence': r.get('_strip_ocr_evidence'),
                        '_leader_line_evidence': r.get('_leader_line_evidence'),
                        '_candidate_geometry_evidence': candidate_geometry,
                        '_near_tolerance_fragment_evidence': r.get('_near_tolerance_fragment_evidence'),
                    })
                    continue

        # 拆分标点分隔的公称值与公差。
        if re.match(r'^[\d.]+[:;][\d.]+$', text):
            parts = re.split(r'[:;]', text)
            if len(parts) == 2:
                text_clean_parsed, prefix_override, type_override = _extract_prefix(parts[0])
                tol_val = parts[1]
                out.append({
                    'text': text,
                    'nominal': text_clean_parsed,
                    'upper_tol': f'+{tol_val}',
                    'lower_tol': f'-{tol_val}',
                    'type': type_override,
                    'prefix': prefix_override,
                    'bbox': r['bbox_in_pdf'],
                    'confidence': _confidence_from_score(score, r),
                    'source': 'ocr',
                    'orientation': r.get('orientation', 0),
                    '_score': score,
                    '_font_height': r.get('font_height_approx', 0),
                    '_paired': False,
                        '_raw_text': r.get('_raw_text', text),
                        '_dimension_line_evidence': r.get('_dimension_line_evidence'),
                        '_strip_ocr_evidence': r.get('_strip_ocr_evidence'),
                        '_leader_line_evidence': r.get('_leader_line_evidence'),
                        '_candidate_geometry_evidence': candidate_geometry,
                        '_near_tolerance_fragment_evidence': r.get('_near_tolerance_fragment_evidence'),
                    })
                continue

        # 解析前缀 + 公差
        text_clean, prefix, dim_type = _extract_prefix(text)
        upper_tol, lower_tol = _parse_tolerance_inline(text_clean)
        nominal_raw = re.sub(r'[±\+\-]\s*[\d.]+.*$', '', text_clean).strip() if (upper_tol or lower_tol) else text_clean
        nominal_raw = nominal_raw.strip()

        confidence_level = _confidence_from_score(score, r)

        out.append({
            'text': text,
            'nominal': nominal_raw,
            'upper_tol': upper_tol,
            'lower_tol': lower_tol,
            'type': dim_type,
            'prefix': prefix,
            'bbox': r['bbox_in_pdf'],
            'confidence': confidence_level,
            'source': 'ocr',
            'orientation': r.get('orientation', 0),
            '_score': score,
            '_font_height': r.get('font_height_approx', 0),
            '_paired': False,
            '_raw_text': r.get('_raw_text', text),
            '_dimension_line_evidence': r.get('_dimension_line_evidence'),
            '_strip_ocr_evidence': r.get('_strip_ocr_evidence'),
            '_leader_line_evidence': r.get('_leader_line_evidence'),
            '_candidate_geometry_evidence': candidate_geometry,
            '_near_tolerance_fragment_evidence': r.get('_near_tolerance_fragment_evidence'),
        })

    return out
