"""
assembler_loaders.py — 几何锚点装载器（capsule / GDT）

从 assembler.py 拆出。包含：
  - _load_capsules()
  - _load_gdt_frames()
  - build_augmented_gdt_frames()
  - _load_gdt_line_frames()
  - _merge_yolo_gdt()

尺寸装载器（datum / basic_dim / dimline / chain merge）见 assembler_loaders_dim.py。
"""

import re

from assembler_utils import (
    VERBOSE,
    _bbox_center, _point_in_bbox, _bbox_iou_min,
    _fixup_ocr_text, _fixup_fc_capsule_text, _nxfont_correction, _extract_prefix, _parse_nominal,
    _parse_tolerance_inline, _union_bbox,
)



_GDT_VOCAB_GLYPHS = {
    '口', '□', '⊕', '⌀', '∅', 'Ø', 'φ', 'Φ', '◎', '⊥', '∥',
    '⌒', '⌓', '○', '↗', 'O', '十', 'X', '一',
    'Ⓜ', 'Ⓛ', 'Ⓟ', 'Ⓕ', 'Ⓣ', 'Ⓢ', 'Ⓒ',
}

_RE_GDT_NUMERIC_OPT_MOD = re.compile(r'^\d+(\.\d+)?(?:[MLP]|\([MLP]\))?$')
# 直径数值允许显式材料修饰符和公差后缀。

_RE_GDT_DIAM_NUMERIC = re.compile(r'^[⌀∅ØφΦ]\d+(\.\d+)?(?:[MLP]|\([MLP]\))?([±+\-]\d+(\.\d+)?)?$')
_RE_GDT_COMPOSITE_ZONE = re.compile(r'^\d+(\.\d+)?CZ-?[A-Z]?$')
# 基准词汇只接受单字母或重复字母结构，排除普通标题和材料词。


_RE_GDT_DATUM = re.compile(r'^(?:[A-Z]|([A-Z])\1{1,2})(\([MLP]\))?$')
_RE_GDT_SIGNED_NUMERIC = re.compile(r'^[±+\-]?\d+(\.\d+)?$')
_RE_GDT_COMPACT_VALUE_DATUM = re.compile(
    r'^(?:\d+\s*[xX×]\s*)?(?://|/|∥)?[⌀∅ØφΦ]?'
    r'\d+\.\d+[A-Z]{1,3}(?:\([MLP]\))?$'
)

_GDT_MODIFIER_GLYPHS = {
    'M': 'Ⓜ',
    'L': 'Ⓛ',
    'P': 'Ⓟ',
    'F': 'Ⓕ',
    'T': 'Ⓣ',
    'S': 'Ⓢ',
    'C': 'Ⓒ',
}


def _is_gdt_vocab_token(s: str) -> bool:
    """判断 token 是否属于兼容形位框词汇：数值、直径修饰数值、复合区域标记、基准引用或显式修饰符。"""
    if not s:
        return False
    s = _normalize_gdt_ocr_token(s.strip().rstrip('.,;:'))
    if not s:
        return False
    if s in _GDT_VOCAB_GLYPHS:
        return True
    if _RE_GDT_NUMERIC_OPT_MOD.match(s):
        return True
    if _RE_GDT_DIAM_NUMERIC.match(s):
        return True
    if _RE_GDT_COMPOSITE_ZONE.match(s):
        return True
    if _RE_GDT_DATUM.match(s):
        return True
    if _RE_GDT_SIGNED_NUMERIC.match(s):
        return True
    if _RE_GDT_COMPACT_VALUE_DATUM.match(s):
        return True
    return False


def _normalize_gdt_ocr_token(s: str) -> str:
    """Repair compact GD&T compartment OCR tokens before vocabulary checks."""
    token = str(s or '').strip()
    if not token:
        return ''
    upper = token.upper()
    if re.fullmatch(r'[H工]{2,3}', upper):
        return upper.replace('工', 'H')
    if re.fullmatch(r'C[O0]|[O0]C', upper):
        return 'CC'
    return token


def _normalize_gdt_compartment_text(text: str) -> str:
    tokens = [
        _normalize_gdt_ocr_token(tok)
        for tok in str(text or '').split()
    ]
    return ' '.join(tok for tok in tokens if tok)


def _has_gdt_vocab_content(text: str) -> bool:
    """True if any whitespace-separated token in `text` is GD&T vocab."""
    if not text:
        return False
    return any(_is_gdt_vocab_token(t) for t in text.split())


def _has_gdt_decimal_rows(rows: list[dict]) -> bool:
    """True if OCR rows contain a decimal GD&T tolerance token."""
    for row in rows or []:
        text = _normalize_gdt_compartment_text(str(row.get('text', '') or ''))
        if _split_compact_gdt_value_datum(text):
            return True
        if re.search(r'[⌀∅ØφΦ]?\s*\d+\.\d+', text):
            return True
    return False


def _repair_gdt_decimal_surrogate_tokens(parts: list[str]) -> tuple[list[str], str]:
    """在已分词的形位框内容中修复被识别为方框字符的小数点。"""
    repaired: list[str] = []
    recovered_tol = ''
    i = 0
    while i < len(parts):
        cur = str(parts[i] or '').strip()
        nxt = str(parts[i + 1] or '').strip() if i + 1 < len(parts) else ''
        tail = str(parts[i + 2] or '').strip() if i + 2 < len(parts) else ''
        tail_digit = _gdt_single_digit_surrogate(tail)
        if cur == '0' and nxt in {'口', '□', '一'} and tail_digit:
            value = f'0.{tail_digit}'
            repaired.append(value)
            if not recovered_tol:
                recovered_tol = value
            i += 3
            continue
        repaired.append(cur)
        i += 1
    return repaired, recovered_tol


def _gdt_modifier_glyphs_from_text(*parts: str) -> list[str]:
    """从 OCR 文本提取显式形位修饰符；需要圈定符号或矢量命中的修饰项不得由普通基准字母代替。"""
    found: list[str] = []
    for raw in parts:
        text = str(raw or '').upper()
        for glyph in _GDT_MODIFIER_GLYPHS.values():
            if glyph in text and glyph not in found:
                found.append(glyph)
        for m in re.finditer(r'\d+(?:\.\d+)?\s*\(?([MLP])\)?\b', text):
            glyph = _GDT_MODIFIER_GLYPHS.get(m.group(1))
            if glyph and glyph not in found:
                found.append(glyph)
        for m in re.finditer(r'\b[A-Z]{1,3}\(([MLP])\)', text):
            glyph = _GDT_MODIFIER_GLYPHS.get(m.group(1))
            if glyph and glyph not in found:
                found.append(glyph)
    return found


def _gdt_datum_slots_from_parts(parts: list[str]) -> list[str]:
    """Return ordered datum reference slots, allowing multi-letter datums."""
    slots: list[str] = []
    for raw in parts:
        for tok in re.split(r'[\s/]+', str(raw or '').strip()):
            tok = tok.strip().strip('.,;:')
            if not tok:
                continue
            tok = _normalize_gdt_ocr_token(tok).upper()
            if _RE_GDT_DATUM.match(tok):
                datum = tok.split('(', 1)[0]
                if datum not in slots:
                    slots.append(datum)
    return slots[:3]


def _normalize_gdt_tolerance_value(value: str) -> str:
    """Normalize OCR variants of a GD&T numeric tolerance value."""
    text = str(value or '').strip()
    if not text:
        return ''
    prefix = ''
    if text[0] in '⌀∅ØφΦ':
        prefix, text = text[0], text[1:]
    text = re.sub(r'^[0O]{2,}(?=\.)', '0', text)
    return prefix + text


def _split_compact_gdt_value_datum(text: str) -> tuple[str, str] | None:
    """将粘连的形位值与基准引用拆分；处理范围限定为含小数值的形位框语法。"""
    raw = str(text or '').strip().upper()
    if not raw:
        return None
    candidates = [re.sub(r'\s+', '', raw)]
    candidates.extend(tok for tok in re.split(r'\s+', raw) if tok)
    for compact in candidates:
        compact = re.sub(r'^\d+\s*[X×]\s*', '', compact)
        compact = re.sub(r'^(?://|/|∥)+', '', compact)
        m = re.fullmatch(
            r'([⌀∅ØφΦ]?)(\d+\.\d+)([A-Z]{1,3}(?:\([MLP]\))?)',
            compact,
        )
        if not m:
            continue
        datum = _normalize_gdt_ocr_token(m.group(3)).upper()
        if datum in {'M', 'L', 'P'}:
            continue
        if not _RE_GDT_DATUM.match(datum):
            continue
        value = f"{m.group(1)}{m.group(2)}" if m.group(1) else m.group(2)
        value = _normalize_gdt_tolerance_value(value)
        return value, datum
    return None


def _repair_truncated_repeated_datum_tail(
    datum_parts: list[str],
    all_parts: list[str],
) -> tuple[list[str], list[str]]:
    """在已有重复字母基准链中，修复末尾引用缺失的重复字母；普通单字母链不作补齐。"""
    tokens: list[tuple[str, str]] = []
    for raw in datum_parts:
        for tok in re.split(r'[\s/]+', str(raw or '').strip()):
            tok = _normalize_gdt_ocr_token(tok.strip().strip('.,;:')).upper()
            if not tok or not _RE_GDT_DATUM.match(tok):
                continue
            m = re.fullmatch(r'([A-Z]{1,3})(\([MLP]\))?', tok)
            if not m:
                continue
            tokens.append((tok, m.group(1)))
    if len(tokens) < 2:
        return datum_parts, all_parts

    previous_bases = [base for _, base in tokens[:-1]]
    last_token, last_base = tokens[-1]
    if len(last_base) != 1:
        return datum_parts, all_parts
    if not previous_bases or not all(len(base) >= 2 and len(set(base)) == 1 for base in previous_bases):
        return datum_parts, all_parts
    repeated_lengths = {len(base) for base in previous_bases}
    if len(repeated_lengths) != 1:
        return datum_parts, all_parts

    repaired_token = last_base * repeated_lengths.pop()
    repaired_datums = [tok for tok, _ in tokens[:-1]] + [repaired_token]
    repaired_parts = list(all_parts)
    for idx in range(len(repaired_parts) - 1, -1, -1):
        if _normalize_gdt_ocr_token(str(repaired_parts[idx]).strip().strip('.,;:')).upper() == last_token:
            repaired_parts[idx] = repaired_token
            break
    return repaired_datums, repaired_parts


def _dedupe_gdt_datum_parts(parts: list[str]) -> list[str]:
    """Drop repeated OCR hits for the same datum compartment."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in parts:
        text = str(raw or '').strip()
        if not text:
            continue
        tokens = []
        for tok in re.split(r'[\s/]+', text):
            norm = _normalize_gdt_ocr_token(tok.strip().strip('.,;:')).upper()
            if norm and norm not in {'M', 'L', 'P'} and _RE_GDT_DATUM.match(norm):
                tokens.append(norm)
        if tokens:
            for tok in tokens:
                if tok in seen:
                    continue
                seen.add(tok)
                out.append(tok)
            continue
        key = re.sub(r'\s+', ' ', _normalize_gdt_ocr_token(text).upper())
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _merge_modifier_glyphs(existing: str, additions: list[str]) -> str:
    glyphs = [g for g in re.split(r'\s+', str(existing or '').strip()) if g]
    for glyph in additions:
        if glyph and glyph not in glyphs:
            glyphs.append(glyph)
    return ' '.join(glyphs)


def _gdt_single_digit_surrogate(token: str) -> str:
    """Return a single digit for compact GD&T OCR fragments."""
    if re.fullmatch(r'\d', token or ''):
        return token
    if token in {'一', 'I', 'l', '|'}:
        return '1'
    return ''


def _bbox_has_keys(bbox: dict) -> bool:
    return all(k in bbox for k in ('x', 'y', 'w', 'h'))


def _is_vertical_gdt_ocr_row(row: dict) -> bool:
    bbox = row.get('bbox_in_pdf') or row.get('bbox') or {}
    if not _bbox_has_keys(bbox):
        return False
    try:
        orient = float(row.get('orientation') or 0.0)
    except (TypeError, ValueError):
        orient = 0.0
    if abs(abs(orient) - 90.0) <= 25.0:
        return True
    return bbox.get('h', 0) >= max(18.0, bbox.get('w', 0) * 1.6)


def _vertical_gdt_value_token(text: str) -> tuple[str, str, str] | None:
    """Return (value_text, nominal, modifier_glyphs) for narrow vertical FCF OCR."""
    raw = str(text or '').strip()
    if not raw:
        return None
    fixed = _fixup_ocr_text(raw).replace('。', '.')
    compact = re.sub(r'\s+', '', fixed).upper()
    # Keep this synthetic path narrow: require explicit diameter or material
    # modifier evidence, otherwise ordinary vertical decimals are too easy to
    # promote into position frames.
    has_diameter = bool(re.search(r'[⌀∅ØΦφ]', compact))
    has_modifier = bool(re.search(r'\(?[MLP]\)?$', compact))
    if not (has_diameter or has_modifier):
        return None
    m = re.fullmatch(
        r'[⌀∅ØΦφ]?(0?\.\d+|\d+\.\d+)(?:\(?([MLP])\)?)?',
        compact,
    )
    if not m:
        return None
    nominal = m.group(1)
    if nominal.startswith('.'):
        nominal = f'0{nominal}'
    try:
        nominal_value = float(nominal)
    except ValueError:
        return None
    if not (0.01 <= nominal_value <= 2.0):
        return None
    modifier = m.group(2) or ''
    value_text = f'{nominal}{modifier}' if modifier else nominal
    modifier_glyphs = _gdt_modifier_glyphs_from_text(value_text)
    return value_text, nominal, ' '.join(modifier_glyphs)


def _vertical_repeated_datum_token(text: str) -> str | None:
    token = _normalize_gdt_ocr_token(str(text or '').strip().strip('.,;:')).upper()
    token = re.sub(r'\s+', '', token)
    if re.fullmatch(r'[8B]{2}', token):
        return 'BB'
    if re.fullmatch(r'([A-Z])\1', token):
        return token
    return None


def _recover_vertical_gdt_ocr_frames(ocr_results: list, existing_dims: list) -> list:
    """兼容路径中根据同列的公差、直径或材料修饰与重复字母基准内容，构建竖向形位框候选。"""
    recovered: list[dict] = []

    def _has_existing(nominal: str, datums: list[str], bbox: dict) -> bool:
        cx, cy = _bbox_center(bbox)
        for dim in existing_dims + recovered:
            if dim.get('source') != 'gdt_line_frame':
                continue
            if str(dim.get('nominal') or '').strip() != nominal:
                continue
            slots = [dim.get('datum_1'), dim.get('datum_2'), dim.get('datum_3')]
            if [str(s or '') for s in slots] != datums:
                continue
            db = dim.get('bbox') or {}
            if not _bbox_has_keys(db):
                return True
            dcx, dcy = _bbox_center(db)
            if abs(dcx - cx) <= 140 and abs(dcy - cy) <= 180:
                return True
        return False

    for value_row in ocr_results:
        if value_row.get('_claimed'):
            continue
        if not _is_vertical_gdt_ocr_row(value_row):
            continue
        parsed_value = _vertical_gdt_value_token(value_row.get('text', ''))
        if not parsed_value:
            continue
        value_text, nominal, modifier = parsed_value
        value_bbox = value_row.get('bbox_in_pdf') or {}
        vcx, vcy = _bbox_center(value_bbox)

        datum_hits = []
        for row in ocr_results:
            if row is value_row or row.get('_claimed'):
                continue
            if not _is_vertical_gdt_ocr_row(row):
                continue
            datum = _vertical_repeated_datum_token(row.get('text', ''))
            if not datum:
                continue
            bbox = row.get('bbox_in_pdf') or {}
            dcx, dcy = _bbox_center(bbox)
            if abs(dcx - vcx) > max(28.0, value_bbox.get('w', 0) + 14.0):
                continue
            if not (dcy < vcy - 4.0 and (vcy - dcy) <= 150.0):
                continue
            datum_hits.append((dcy, datum, row))

        datum_hits.sort(key=lambda item: -item[0])
        datums = []
        used_rows = []
        for _dcy, datum, row in datum_hits:
            if datum in datums:
                continue
            datums.append(datum)
            used_rows.append(row)
            if len(datums) == 3:
                break
        if len(datums) != 3:
            continue

        merged_bbox = _union_bbox([value_bbox] + [row['bbox_in_pdf'] for row in used_rows])
        if _has_existing(nominal, datums, merged_bbox):
            continue

        recovered.append({
            'text': ' '.join(['⊕', value_text] + datums),
            'nominal': nominal,
            'upper_tol': '',
            'lower_tol': '',
            'datum': ' / '.join(datums),
            'datum_1': datums[0],
            'datum_2': datums[1],
            'datum_3': datums[2],
            'modifier': modifier,
            'type': 'gdt',
            'prefix': '⊕',
            'symbol_name': 'position',
            'symbol_unicode': '⊕',
            'symbol_method': 'ocr_synthetic_vertical',
            'gdt_symbol_confidence': 0.8,
            'bbox': merged_bbox,
            'confidence': 'high',
            'source': 'gdt_line_frame',
            'orientation': -90,
            '_origin': 'synthetic_vertical_ocr',
        })
        value_row['_claimed'] = True
        for row in used_rows:
            row['_claimed'] = True

    return recovered


def _load_capsules(ocr_results: list, capsules: list, out_dims: list):
    """
    对每个跑道框，找落在其内的 OCR 结果。
    confidence = 'high'，source = 'geometric_anchor'。
    """
    if VERBOSE:
        print(f"[capsule] 输入: {len(capsules)} capsules, {len(ocr_results)} OCR")
    for cap in capsules:
        texts_inside = []
        for r in ocr_results:
            if r['_claimed']:
                continue
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            if _point_in_bbox(cx, cy, cap, margin=2.0):
                texts_inside.append(r)
                r['_claimed'] = True

        if VERBOSE:
            print(f"[capsule] checking capsule at ({cap['x']:.0f}, {cap['y']:.0f}, {cap['w']:.0f}x{cap['h']:.0f}), texts_inside={len(texts_inside)}")
        if not texts_inside:
            continue

        # 合并同一跑道框内的文字（按 x 坐标排序）
        texts_inside.sort(key=lambda r: r['bbox_in_pdf']['x'])

        # 多结果容器过滤非定向来源的单字符噪声，保留定向读取的单字符。
        
        
        _has_multi_char = any(len(r.get('text', '').strip()) > 1 for r in texts_inside)
        if _has_multi_char:
            _noise_sources = {'rotated_ocr', 'strip_ocr'}
            _filtered = []
            for r in texts_inside:
                _t = r.get('text', '').strip()
                if (len(_t) == 1
                        and r.get('source', '') in _noise_sources
                        and not re.match(r'[⌀±°+\-]', _t)):
                    if VERBOSE:
                        _rb = r['bbox_in_pdf']
                        print(f"[capsule] DROP noise '{_t}' src={r.get('source','')} "
                              f"bbox=({_rb['x']:.0f},{_rb['y']:.0f},{_rb['w']:.0f}x{_rb['h']:.0f})")
                    r['_claimed'] = False  # 释放回未认领池
                    continue
                _filtered.append(r)
            texts_inside = _filtered
            if not texts_inside:
                continue

        # 同一容器内以相等或子串关系去除重复和残缺读取，再组装文字。
        
        
        
        
        
        
        if len(texts_inside) > 1:
            _kept_indices = set()
            # 长文本优先保留；等长时保留原始 x-sort 顺序靠前者
            _order = sorted(range(len(texts_inside)),
                            key=lambda i: (-len(texts_inside[i].get('text', '').strip()), i))
            for i in _order:
                ta = texts_inside[i].get('text', '').strip()
                if not ta:
                    continue
                is_sub_of_kept = False
                for j in _kept_indices:
                    tb = texts_inside[j].get('text', '').strip()
                    if tb and ta in tb:
                        is_sub_of_kept = True
                        break
                if not is_sub_of_kept:
                    _kept_indices.add(i)
            if len(_kept_indices) < len(texts_inside):
                _dropped_texts = []
                _new_texts = []
                for i, r in enumerate(texts_inside):
                    if i in _kept_indices:
                        _new_texts.append(r)
                    else:
                        r['_claimed'] = False  # 释放回未认领池
                        _dropped_texts.append(r.get('text', '').strip())
                if VERBOSE and _dropped_texts:
                    print(f"[capsule] DROP substring/dup: {_dropped_texts} "
                          f"kept={[r.get('text','').strip() for r in _new_texts]}")
                texts_inside = _new_texts
                if not texts_inside:
                    continue

        # 存在完整尺寸 token 时，先移除孤立功能标记碎片，避免其混入公称值。
        
        
        
        if len(texts_inside) > 1:
            _has_dim_signal = any(
                re.search(r'(?:±|[+\-]\s*\d|\d+\.\d+)', r.get('text', ''))
                for r in texts_inside
            )
            if _has_dim_signal:
                _filtered = []
                for r in texts_inside:
                    _compact = re.sub(r'\s+', '', r.get('text', '')).upper()
                    _compact = _compact.replace('|', '/').replace('\\', '/')
                    if re.fullmatch(r'\d?F/?[CD]', _compact):
                        r['_claimed'] = False
                        continue
                    _filtered.append(r)
                texts_inside = _filtered
                if not texts_inside:
                    continue

        combined_text = ' '.join(r['text'] for r in texts_inside)

        # 跳过空白过密或非预期字符占比过高的文字。
        if '   ' in combined_text:
            for r in texts_inside:
                r['_claimed'] = False  # 释放回未认领池
            continue
        _non_ascii = sum(1 for c in combined_text if ord(c) > 127 and c not in '⌀±°⊕◎⊥∥○⌭∠⌓⌒▱≡—⊘∅Ø')
        if len(combined_text) > 3 and _non_ascii / len(combined_text) > 0.3:
            for r in texts_inside:
                r['_claimed'] = False
            continue

        # 前置正则：必须含多位数字或小数或 GD&T 符号，纯单数字+乱码不够
        if not re.search(r'\d{2,}|\d+\.\d|[⌀⊕±∅Ø°⊥∥◎○⌭∠⌓⌒▱≡—⊘]', combined_text):
            for r in texts_inside:
                r['_claimed'] = False
            continue

        # OCR 系统性误读修复（t→±、粘连拆分等），覆盖竖直旋转 OCR 路径
        combined_text = _fixup_ocr_text(combined_text)
        combined_text = _fixup_fc_capsule_text(combined_text)
        combined_text = _nxfont_correction(combined_text)

        combined_text, prefix, dim_type = _extract_prefix(combined_text)

        # 计算合并 bbox（union）
        all_bboxes = [r['bbox_in_pdf'] for r in texts_inside]
        union_bbox = _union_bbox(all_bboxes) if len(all_bboxes) > 1 else all_bboxes[0]

        nominal_raw = combined_text.strip()
        # 修复容器合并后直接粘连的小数公称值和公差。
        _glue_m = re.match(r'^(\d+\.\d+)(0\.\d+)$', nominal_raw)
        if _glue_m:
            nominal_raw = _glue_m.group(1) + '±' + _glue_m.group(2)
        upper_tol, lower_tol = _parse_tolerance_inline(nominal_raw)
        if upper_tol or lower_tol:
            nominal_raw = re.sub(r'[±\+\-]\s*[\d.]+.*$', '', nominal_raw).strip()

        # 公差剥离后若没有公称值，释放文字供孤立公差路径处理。
        
        if not nominal_raw:
            for r in texts_inside:
                r['_claimed'] = False
            continue

        # 无公差的小幅值容器内容按兼容过滤规则排除。
        
        nom_val = _parse_nominal(nominal_raw)
        if nom_val is not None and abs(nom_val) < 1.0 and not upper_tol and not lower_tol:
            for r in texts_inside:
                r['_claimed'] = False
            continue

        out_dims.append({
            'text': (prefix or '') + combined_text,
            'nominal': nominal_raw,
            'upper_tol': upper_tol,
            'lower_tol': lower_tol,
            'type': dim_type,
            'prefix': prefix,
            'bbox': union_bbox,
            'confidence': 'high',
            'source': 'geometric_anchor',
            'orientation': texts_inside[0].get('orientation', 0) if texts_inside else 0,
            # 跑道框 = 重点尺寸（KEY）。capsule 形状识别跟颜色无关，黑白图纸同样识别。
            # 前端 dimensionsToStamps 读 dim.is_key 渲染成蓝色跑道，导出 Excel B 列写 'Y'。
            'is_key': True,
        })


# 装载形位控制框。



def _load_gdt_frames(ocr_results: list, gdt_frames: list, out_gdt: list):
    """
    对每个 GD&T 控制框，收集内部文字，按 x 排序拼接。
    type = 'gd_t'
    """
    for frame in gdt_frames:
        texts_inside = []
        for r in ocr_results:
            if r['_claimed']:
                continue
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            if _point_in_bbox(cx, cy, frame, margin=1.0):
                texts_inside.append(r)
                r['_claimed'] = True

        if not texts_inside:
            continue

        texts_inside.sort(key=lambda r: r['bbox_in_pdf']['x'])
        combined = ' '.join(r['text'] for r in texts_inside)

        out_gdt.append({
            'text': combined,
            'type': 'gd_t',
            'bbox': {'x': frame['x'], 'y': frame['y'],
                     'w': frame['w'], 'h': frame['h']},
        })


# 认领线段网格检测产生的形位控制框。



def build_augmented_gdt_frames(
    gdt_frames_from_lines: list,
    reconstructed_rects: list = None,
    frame_borders: list = None,
    page_width: float = None,
    page_height: float = None,
) -> list:
    """合并线段检测与矩形重建产生的形位框候选，返回新列表。仅接纳满足框结构过滤的重建矩形，保持后续装载与符号检测的候选边界一致。"""
    # 线段检测框默认 origin='line'；若调用方已经 augment 过则保留原值（幂等）。
    # 用 `{'_origin': ..., **f}` 让 f 的键覆盖默认值，避免 mutate 输入。
    all_frames = [{'_origin': 'line', **f} for f in (gdt_frames_from_lines or [])]
    if not reconstructed_rects:
        return all_frames

    # 通用重建矩形只有满足单行控制框几何条件时才增补，排除标题栏与多行表格。
    
    
    
    GDT_MIN_HEIGHT = 17.0
    GDT_MAX_HEIGHT = 28.0
    GDT_MIN_WIDTH = 50.0
    GDT_MAX_WIDTH = 260.0
    GDT_MAX_ASPECT = 13.0
    GDT_FIRST_COMP_MIN = 10.0
    GDT_FIRST_COMP_MAX = 45.0
    # Fallback only for drawings where extract_rects() failed to surface a
    # title_block.  The ratios intentionally describe the conventional lower
    # right title area and are not used when an explicit title_block exists.
    TITLE_FALLBACK_X_RATIO = 0.68
    TITLE_FALLBACK_Y_RATIO = 0.72

    def _in_title_block(bbox_xyxy: list[float]) -> bool:
        if not frame_borders:
            return False
        x0, y0, x1, y1 = bbox_xyxy
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        for fb in frame_borders:
            if fb.get('source') != 'title_block':
                continue
            if (fb.get('x', 0) <= cx <= fb.get('x', 0) + fb.get('w', 0)
                    and fb.get('y', 0) <= cy <= fb.get('y', 0) + fb.get('h', 0)):
                return True
        return False

    def _in_bottom_right_title_band(bbox_xyxy: list[float]) -> bool:
        if not page_width or not page_height:
            return False
        if any(fb.get('source') == 'title_block' for fb in (frame_borders or [])):
            return False
        x0, y0, x1, y1 = bbox_xyxy
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        return (
            cx >= page_width * TITLE_FALLBACK_X_RATIO
            and cy >= page_height * TITLE_FALLBACK_Y_RATIO
        )

    gdt_from_rects = []
    for r in reconstructed_rects:
        if r['compartments'] < 2 or r['compartments'] > 8:
            continue
        rw = float(r.get('width', 0) or 0)
        rh = float(r.get('height', 0) or 0)
        if rw < GDT_MIN_WIDTH or rw > GDT_MAX_WIDTH:
            continue
        if rh < GDT_MIN_HEIGHT or rh > GDT_MAX_HEIGHT:
            continue
        if rw / max(rh, 1e-6) > GDT_MAX_ASPECT:
            continue
        if r.get('has_interior_h'):
            continue
        if _in_title_block(r['bbox']) or _in_bottom_right_title_band(r['bbox']):
            continue
        vx = sorted([float(x) for x in r.get('interior_v_lines', [])])
        if not vx:
            continue
        first_w = vx[0] - float(r['bbox'][0])
        if first_w < GDT_FIRST_COMP_MIN or first_w > GDT_FIRST_COMP_MAX:
            continue
        gdt_from_rects.append(r)

    # NMS by containment: 嵌套框去重，保留大框
    to_remove = set()
    for i in range(len(gdt_from_rects)):
        for j in range(i + 1, len(gdt_from_rects)):
            if i in to_remove or j in to_remove:
                continue
            bi = gdt_from_rects[i]['bbox']
            bj = gdt_from_rects[j]['bbox']
            ix0 = max(bi[0], bj[0])
            iy0 = max(bi[1], bj[1])
            ix1 = min(bi[2], bj[2])
            iy1 = min(bi[3], bj[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
            area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
            smaller_area = min(area_i, area_j)
            if smaller_area > 0 and inter / smaller_area > 0.8:
                if area_i < area_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    gdt_from_rects = [r for idx, r in enumerate(gdt_from_rects) if idx not in to_remove]

    for rr in gdt_from_rects:
        bbox_xyxy = rr['bbox']
        x0, y0, x1, y1 = bbox_xyxy
        fw = x1 - x0
        fh = y1 - y0

        all_vx = [x0] + rr['interior_v_lines'] + [x1]
        comps = []
        for k in range(len(all_vx) - 1):
            comp_x = all_vx[k]
            comp_w = all_vx[k + 1] - all_vx[k]
            if comp_w >= 5:
                comps.append({
                    'x': round(comp_x, 2),
                    'y': round(y0, 2),
                    'w': round(comp_w, 2),
                    'h': round(fh, 2),
                })
        if len(comps) < 2:
            continue

        new_frame = {
            'bbox': {'x': round(x0, 2), 'y': round(y0, 2),
                     'w': round(fw, 2), 'h': round(fh, 2)},
            'compartments': comps,
            'frame_height': round(fh, 2),
            'line_width': 0.0,
            'confidence': 0.5,
            '_origin': 'rect',
        }

        is_dup = False
        for existing in all_frames:
            eb = existing['bbox']
            ix0 = max(new_frame['bbox']['x'], eb['x'])
            iy0 = max(new_frame['bbox']['y'], eb['y'])
            ix1 = min(new_frame['bbox']['x'] + new_frame['bbox']['w'], eb['x'] + eb['w'])
            iy1 = min(new_frame['bbox']['y'] + new_frame['bbox']['h'], eb['y'] + eb['h'])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                area_a = new_frame['bbox']['w'] * new_frame['bbox']['h']
                area_b = eb['w'] * eb['h']
                iou = inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0
                if iou > 0.5:
                    is_dup = True
                    break
        if not is_dup:
            all_frames.append(new_frame)

    return all_frames


def _load_gdt_line_frames(
    gdt_frames: list,
    ocr_results: list,
    page_width: float,
    page_height: float,
    reconstructed_rects: list = None,
    capsule_candidates: list = None,
    frame_borders: list = None,
) -> list:
    """
    将 GD&T 线段网格框内的 OCR 文本认领为 dimension。

    Args:
        gdt_frames: detect_gdt_frames_from_lines() 的返回值，每个含
                    bbox, compartments, frame_height, line_width, confidence。
                    入口脚本若已预先 build_augmented_gdt_frames，这里的列表
                    就已经包含 reconstructed_rects 派生的候选。
        ocr_results: 已清洗的 OCR 结果列表（带 _claimed 标记）
        page_width, page_height: 页面尺寸
        reconstructed_rects: reconstruct_rectangles_from_lines() 的返回值（可选）；
                             若 gdt_frames 已被入口脚本 augment，这里的结果会在
                             IoU 去重中被吸收，不会产生重复。

    Returns:
        list of dict: 认领产出的 dimension 列表
    """
    # build_augmented_gdt_frames 本身就是幂等的（IoU 去重）：入口脚本若已 augment
    # 过，此处再次 augment 也不会产生重复 frame。严禁 mutate 调用方传入的 gdt_frames。
    all_frames = build_augmented_gdt_frames(
        gdt_frames, reconstructed_rects, frame_borders=frame_borders,
        page_width=page_width, page_height=page_height)
    if VERBOSE:
        print(f"[_load_gdt_line_frames] frames after augment: {len(all_frames)} "
              f"(raw={len(gdt_frames)}, rect_input={len(reconstructed_rects or [])})")

    EXPAND = 15.0  # bbox 扩展量 (pt) — 宽松以捕获偏移的 OCR
    BELOW_SEARCH = 30.0  # 框下方搜索距离 (pt)
    IOU_THRESHOLD = 0.1  # 部分重叠匹配阈值
    dims_out = []

    for fi, frame in enumerate(all_frames):
        fb = frame['bbox']
        comps = frame.get('compartments', [])
        if len(comps) < 2:
            if VERBOSE:
                print(f"  [gdt_line_frame #{fi}] SKIP: compartments={len(comps)} < 2, "
                      f"bbox=({fb['x']:.0f},{fb['y']:.0f},{fb['w']:.0f}x{fb['h']:.0f})")
            continue

        # 收集落入框内扩展区或与框相交的文字。
        expanded_bbox = {
            'x': fb['x'] - EXPAND,
            'y': fb['y'] - EXPAND,
            'w': fb['w'] + 2 * EXPAND,
            'h': fb['h'] + 2 * EXPAND,
        }
        below_bbox = {
            'x': fb['x'] - 5,
            'y': fb['y'] + fb['h'],
            'w': fb['w'] + 10,
            'h': BELOW_SEARCH,
        }
        texts_inside = []
        claimed_inside_candidates = []
        for r in ocr_results:
            cx, cy = _bbox_center(r['bbox_in_pdf'])
            ocr_bb = r['bbox_in_pdf']
            claimed_by_prior = bool(r.get('_claimed'))
            if (
                claimed_by_prior
                and r.get('source') == 'gdt_compartment_ocr'
                and _point_in_bbox(cx, cy, fb)
            ):
                texts_inside.append(r)
                continue
            if claimed_by_prior:
                # Keep a narrow fallback pool for claimed strip/capsule OCR.
                # It is merged only if the unclaimed frame read lacks a decimal
                # tolerance, so clean frames do not get polluted by neighbours.
                if (
                    frame.get('_origin') == 'line'
                    and _point_in_bbox(cx, cy, fb, margin=5.0)
                    and _has_gdt_vocab_content(r.get('text', ''))
                ):
                    claimed_inside_candidates.append(r)
                continue
            # Primary: center in expanded bbox
            if _point_in_bbox(cx, cy, expanded_bbox):
                texts_inside.append(r)
            # Secondary: partial IoU overlap with expanded bbox (min-area IoU for sensitivity)
            elif _bbox_iou_min(ocr_bb, expanded_bbox) > IOU_THRESHOLD:
                texts_inside.append(r)
            # Tertiary: directly below frame
            elif _point_in_bbox(cx, cy, below_bbox):
                texts_inside.append(r)

        # 来自 reconstructed_rects 的候选框，如果 compartment 内 OCR
        # 零命中，大概率是假阳性（标题栏格 / 参数表单元格 / 剖面填充交汇处），
        # 直接丢弃不留噪声。线段检测框即使 unread 也保留（几何证据强）。
        if frame.get('_origin') == 'rect' and not texts_inside:
            if VERBOSE:
                print(f"  [gdt_line_frame #{fi}] DROP: rect-origin & zero OCR hits, "
                      f"bbox=({fb['x']:.0f},{fb['y']:.0f},{fb['w']:.0f}x{fb['h']:.0f})")
            continue
        if frame.get('_origin') == 'rect':
            _refined = [r for r in texts_inside if _has_gdt_vocab_content(r.get('text', ''))]
            if VERBOSE and len(_refined) < len(texts_inside):
                _dropped = [r.get('text', '') for r in texts_inside if r not in _refined]
                print(f"  [gdt_line_frame #{fi}] rect vocab refine: {len(texts_inside)} -> "
                      f"{len(_refined)} ocr (drop non-GD&T: {_dropped})")
            texts_inside = _refined
            if not texts_inside:
                if VERBOSE:
                    print(f"  [gdt_line_frame #{fi}] DROP: rect-origin & zero GD&T-vocab OCR, "
                          f"bbox=({fb['x']:.0f},{fb['y']:.0f},{fb['w']:.0f}x{fb['h']:.0f})")
                continue

        if frame.get('_origin') == 'line':
            tight_y_top = fb['y'] - 5
            tight_y_bot = fb['y'] + fb['h'] + 5
            _refined = []
            for r in texts_inside:
                _, cy = _bbox_center(r['bbox_in_pdf'])
                in_tight = (tight_y_top <= cy <= tight_y_bot)
                if in_tight and _has_gdt_vocab_content(r['text']):
                    _refined.append(r)
            if VERBOSE and len(_refined) < len(texts_inside):
                _dropped = [r['text'] for r in texts_inside if r not in _refined]
                print(f"  [gdt_line_frame #{fi}] D' refine: {len(texts_inside)} -> "
                      f"{len(_refined)} ocr (drop noise/oob: {_dropped})")
            texts_inside = _refined
            if not texts_inside:
                _claimed_refined_for_empty = []
                for r in claimed_inside_candidates:
                    _, cy = _bbox_center(r['bbox_in_pdf'])
                    in_tight = (tight_y_top <= cy <= tight_y_bot)
                    if in_tight and _has_gdt_vocab_content(r['text']):
                        _claimed_refined_for_empty.append(r)
                if _has_gdt_decimal_rows(_claimed_refined_for_empty):
                    # Leave texts_inside empty for the normal compartment pass;
                    # the post-pass below will synthesize value/datum from the
                    # claimed rows. This keeps clean readable frames unchanged.
                    pass
                else:
                    if VERBOSE:
                        print(f"  [gdt_line_frame #{fi}] DROP: line-origin & zero "
                              f"GD&T-vocab OCR after refine, "
                              f"bbox=({fb['x']:.0f},{fb['y']:.0f},{fb['w']:.0f}x{fb['h']:.0f})")
                    continue
            if not texts_inside and not _has_gdt_decimal_rows(claimed_inside_candidates):
                if VERBOSE:
                    print(f"  [gdt_line_frame #{fi}] DROP: line-origin & zero "
                          f"GD&T-vocab OCR after refine, "
                          f"bbox=({fb['x']:.0f},{fb['y']:.0f},{fb['w']:.0f}x{fb['h']:.0f})")
                continue

        # 将文字分配到对应格子。
        comp_texts = {i: [] for i in range(len(comps))}
        for r in texts_inside:
            cx = _bbox_center(r['bbox_in_pdf'])[0]
            # 找最近的 compartment（按 x 中心距离）
            best_i, best_dist = 0, float('inf')
            for i, c in enumerate(comps):
                c_cx, _ = _bbox_center(c)
                dist = abs(cx - c_cx)
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            comp_texts[best_i].append(r)

        # 依据符号格和显式复合区域标记确定兼容形位类别，再组装文本。
        
        
        
        
        
        
        gdt_symbol = '⊕'
        for _ti in texts_inside:
            _txt = _ti['text']
            if '口' in _txt or '□' in _txt or re.search(r'\d+CZ', _txt):
                gdt_symbol = '□'
                break
            if '∥' in _txt or '//' in _txt:
                gdt_symbol = '∥'

        # 解析各 compartment
        tolerance_value = ''
        datum_parts = []
        all_parts = [gdt_symbol]

        for i in range(len(comps)):
            c = comps[i]
            c_w = c['w']
            # 排序本格内文本
            comp_texts[i].sort(key=lambda r: r['bbox_in_pdf']['x'])
            txt = _normalize_gdt_compartment_text(
                ' '.join(r['text'] for r in comp_texts[i]).strip()
            )
            compact_value_datum = _split_compact_gdt_value_datum(txt)

            if i == 0:
                # 符号格中的漂入数值保留为待处理内容，避免误解释为直径修饰。
                
                
                if compact_value_datum:
                    if not tolerance_value:
                        tolerance_value = compact_value_datum[0]
                        all_parts.append(compact_value_datum[0])
                    if compact_value_datum[1] not in datum_parts:
                        datum_parts.append(compact_value_datum[1])
                    all_parts.append(compact_value_datum[1])
                    continue
                # 较宽格子允许携带直径修饰。
                if c_w > 25 and txt:
                    # 检查是否包含 ⌀ 类符号
                    if re.search(r'[⌀∅ØφΦO0]', txt):
                        all_parts.append('⌀')
                continue

            if i == 1 and c_w < 22:
                # 后续格子可能包含直径修饰或数值片段。
                if compact_value_datum:
                    tolerance_value = compact_value_datum[0]
                    if compact_value_datum[1] not in datum_parts:
                        datum_parts.append(compact_value_datum[1])
                    all_parts.extend(compact_value_datum)
                    continue
                if txt and re.search(r'[⌀∅ØφΦO0]', txt) and not re.search(r'\d', txt):
                    all_parts.append('⌀')
                    continue
                elif txt and re.search(r'\d', txt):
                    # 包含数字，视为公差值
                    tolerance_value = txt
                    all_parts.append(txt)
                    continue
                elif not txt:
                    continue
                else:
                    # A narrow pre-tolerance cell can OCR the symbol/diameter
                    # separator as a stray letter such as X/D. Datum references
                    # belong after the tolerance value, so keep this out of the
                    # datum tail.
                    continue

            # 后续格：尝试区分公差数值 vs datum 引用
            if compact_value_datum:
                if not tolerance_value:
                    tolerance_value = compact_value_datum[0]
                    all_parts.append(compact_value_datum[0])
                if compact_value_datum[1] not in datum_parts:
                    datum_parts.append(compact_value_datum[1])
                all_parts.append(compact_value_datum[1])
            elif not tolerance_value and re.search(r'\d', txt):
                # 公差数值格
                tolerance_value = txt
                all_parts.append(txt)
            elif txt:
                # datum 引用格（如 "C", "A", "B", "A(M)"）
                datum_parts.append(txt)
                all_parts.append(txt)

        fallback_claimed_rows_used = []
        if not tolerance_value and claimed_inside_candidates:
            fallback_rows = []
            tight_y_top = fb['y'] - 5
            tight_y_bot = fb['y'] + fb['h'] + 5
            for r in claimed_inside_candidates:
                _, cy = _bbox_center(r['bbox_in_pdf'])
                in_tight = (tight_y_top <= cy <= tight_y_bot)
                if in_tight and _has_gdt_vocab_content(r['text']):
                    fallback_rows.append(r)
            if _has_gdt_decimal_rows(fallback_rows):
                fallback_value = ''
                fallback_datums = []
                for r in sorted(fallback_rows, key=lambda row: row['bbox_in_pdf']['x']):
                    txt = _normalize_gdt_compartment_text(str(r.get('text', '') or ''))
                    compact_value_datum = _split_compact_gdt_value_datum(txt)
                    if compact_value_datum:
                        if not fallback_value:
                            fallback_value = compact_value_datum[0]
                        if compact_value_datum[1] not in fallback_datums:
                            fallback_datums.append(compact_value_datum[1])
                        fallback_claimed_rows_used.append(r)
                        continue
                    if not fallback_value and re.search(r'\d+\.\d+', txt):
                        fallback_value = _normalize_gdt_tolerance_value(txt)
                        fallback_claimed_rows_used.append(r)
                        continue
                    datum = _normalize_gdt_ocr_token(txt.strip().strip('.,;:')).upper()
                    if datum and datum not in {'M', 'L', 'P'} and _RE_GDT_DATUM.match(datum):
                        if datum not in fallback_datums:
                            fallback_datums.append(datum)
                        fallback_claimed_rows_used.append(r)
                if fallback_value:
                    tolerance_value = fallback_value
                    datum_parts.extend(fallback_datums)
                    all_parts = [gdt_symbol, fallback_value, *datum_parts]
                    texts_inside.extend(fallback_claimed_rows_used)

        # 同格文字先拆分，再按相等字符串和重复数值去重；仅处理当前单行框结构。
        
        
        
        
        
        
        
        _seen_tokens = set()
        _seen_numbers = set()
        _cleaned_parts = []
        for _part in all_parts:
            for _tok in _part.split():
                if not _tok or _tok in _seen_tokens:
                    continue
                _num_m = re.search(r'\d+\.?\d*', _tok)
                _num = _num_m.group(0) if _num_m else None
                if _num is not None and _num in _seen_numbers:
                    continue
                _cleaned_parts.append(_tok)
                _seen_tokens.add(_tok)
                if _num is not None:
                    _seen_numbers.add(_num)
        all_parts, _recovered_decimal_tol = _repair_gdt_decimal_surrogate_tokens(_cleaned_parts)
        if _recovered_decimal_tol and (
                not tolerance_value or not re.search(r'\d+\.\d+', tolerance_value)):
            tolerance_value = _recovered_decimal_tol
            # The surrogate glyph and trailing digit were OCR fragments of the
            # tolerance value, not real datum compartments.
            datum_parts = [
                p for p in datum_parts
                if not re.fullmatch(r'[口□一]?\s*\d?', str(p or '').strip())
            ]
        datum_parts, all_parts = _repair_truncated_repeated_datum_tail(datum_parts, all_parts)
        datum_parts = _dedupe_gdt_datum_parts(datum_parts)

        assembled_text = ' '.join(all_parts)
        # 修复形位数值前导零被识别为括号的兼容模式。
        assembled_text = assembled_text.replace(').', '0.')
        # 将分裂的前导零和复合区域数值重新合并，限定于明确的形位框 token 结构。
        
        
        
        
        
        
        _LEADING_ZERO_CZ = re.compile(r'\b0(?:\s+口)?\s+(\d+CZ)')
        assembled_text = _LEADING_ZERO_CZ.sub(r'0.\1', assembled_text)
        if tolerance_value:
            tolerance_value = _LEADING_ZERO_CZ.sub(r'0.\1', tolerance_value)
            tolerance_value = _normalize_gdt_tolerance_value(tolerance_value)
        if not assembled_text.strip() or assembled_text.strip() == gdt_symbol:
            # 框内完全没有可用 OCR 文本
            if frame.get('_origin') == 'rect':
                if VERBOSE:
                    print(f"  [gdt_line_frame #{fi}] DROP: rect-origin & no compartment text, "
                          f"bbox=({fb['x']:.0f},{fb['y']:.0f},{fb['w']:.0f}x{fb['h']:.0f})")
                continue
            assembled_text = 'GD&T (unread)'

        if VERBOSE:
            print(f"  [gdt_line_frame #{fi}] bbox=({fb['x']:.0f},{fb['y']:.0f},{fb['w']:.0f}x{fb['h']:.0f}) "
                  f"comps={len(comps)} ocr_in={len(texts_inside)} text='{assembled_text}'")

        # 文本质量检查：过滤明显不是 GD&T 内容的匹配
        _gdt_text_clean = assembled_text.replace(gdt_symbol, '').strip()
        if len(_gdt_text_clean) > 20:  # 过滤超过框内容长度限制的文字。
            if VERBOSE:
                print(f"    → FILTERED: text too long ({len(_gdt_text_clean)} chars)")
            continue
        if re.search(r'[#@!]', _gdt_text_clean):  # 含乱码符号
            if VERBOSE:
                print(f"    → FILTERED: junk symbols")
            continue
        if re.search(r'SHEET|DRAWING|SCALE|DATE', _gdt_text_clean, re.IGNORECASE):  # 标题栏误匹配
            if VERBOSE:
                print(f"    → FILTERED: title block match")
            continue
        # 倍数与截断括号结构不作为框值；线段框保留未读状态，重建矩形按过滤规则排除。
        
        if re.search(r'\d+X\s*[\(⌀]', _gdt_text_clean):
            if frame.get('_origin') == 'rect':
                if VERBOSE:
                    print(f"  [gdt_line_frame #{fi}] DROP: rect-origin count-dim text")
                continue
            assembled_text = 'GD&T (unread)'

        # 提取 nominal（公差数值中的纯数字部分）
        nominal = ''
        if tolerance_value:
            m = re.search(r'[\d.]+', tolerance_value)
            if m:
                nominal = m.group(0)
        datum_slots = _gdt_datum_slots_from_parts(datum_parts)
        modifier_glyphs = _gdt_modifier_glyphs_from_text(tolerance_value, *datum_parts)

        # A feature-control frame is one semantic dimension.  Earlier builds
        # emitted one result per datum compartment to satisfy a position-only
        # evaluator, but that made the UI look broken: the same GD&T text was
        # repeated over A/B/C cells as separate boxes.  Keep the full frame
        # bbox here so the visible output matches the drawing.
        dims_out.append({
            'text': assembled_text,
            'nominal': nominal,
            'upper_tol': '',
            'lower_tol': '',
            'datum': ' / '.join(datum_parts),
            'datum_1': datum_slots[0] if len(datum_slots) > 0 else '',
            'datum_2': datum_slots[1] if len(datum_slots) > 1 else '',
            'datum_3': datum_slots[2] if len(datum_slots) > 2 else '',
            'modifier': ' '.join(modifier_glyphs),
            'type': 'gdt',
            'prefix': gdt_symbol,
            'bbox': {'x': fb['x'], 'y': fb['y'], 'w': fb['w'], 'h': fb['h']},
            'confidence': 'high',
            'source': 'gdt_line_frame',
            'orientation': 0,
            '_origin': frame.get('_origin', 'line'),
        })

        # 标记已认领的 OCR region
        for r in texts_inside:
            r['_claimed'] = True

    _RE_CZ_VALUE_TOKEN = re.compile(r'^\d+(\.\d+)?CZ-?[A-Z]?$')
    for r in ocr_results:
        if r.get('_claimed'):
            continue
        text = r.get('text', '').strip()
        if not _RE_CZ_VALUE_TOKEN.match(text):
            continue
        # 提取数值前缀作为公称值。
        m = re.match(r'^(\d+(?:\.\d+)?)', text)
        if not m:
            continue
        nominal = m.group(1)
        # 仅捕获复合区域标记之后的基准后缀，不把标记本身当基准。
        
        
        datum_m = re.search(r'CZ-?([A-Z])$', text)
        datum_txt = datum_m.group(1) if datum_m else ''
        bb = r.get('bbox_in_pdf', {})
        if not bb:
            continue
        synth_text = f'□ {text}'
        dims_out.append({
            'text': synth_text,
            'nominal': nominal,
            'upper_tol': '',
            'lower_tol': '',
            'datum': datum_txt,
            'datum_1': datum_txt,
            'datum_2': '',
            'datum_3': '',
            'modifier': '',
            'type': 'gdt',
            'prefix': '□',
            'symbol_name': 'surface_profile',
            'symbol_unicode': '⌓',
            'symbol_method': 'ocr_synthetic_cz',
            'gdt_symbol_confidence': 0.85,
            'bbox': {'x': bb.get('x', 0), 'y': bb.get('y', 0),
                     'w': bb.get('w', 0), 'h': bb.get('h', 0)},
            'confidence': 'high',
            'source': 'gdt_line_frame',
            'orientation': r.get('orientation', 0),
            '_origin': 'synthetic_cz',
        })
        r['_claimed'] = True
        if VERBOSE:
            print(f"  [gdt_synthetic_cz] CREATE □ '{text}' nominal={nominal} datum={datum_txt!r} "
                  f"bbox=({bb.get('x'):.0f},{bb.get('y'):.0f},{bb.get('w'):.0f}x{bb.get('h'):.0f})")

    vertical_dims = _recover_vertical_gdt_ocr_frames(ocr_results, dims_out)
    if vertical_dims:
        dims_out.extend(vertical_dims)
        if VERBOSE:
            print(f"  [gdt_synthetic_vertical] CREATE {len(vertical_dims)} vertical OCR frames")

    # 包围框被跑道容器几何包含时，同步设置重点尺寸标志。
    
    
    
    
    
    if capsule_candidates:
        margin = 4.0
        n_marked = 0
        for d in dims_out:
            db = d.get('bbox', {})
            dx0, dy0 = db.get('x', 0), db.get('y', 0)
            dx1, dy1 = dx0 + db.get('w', 0), dy0 + db.get('h', 0)
            for cap in capsule_candidates:
                cx0, cy0 = cap['x'], cap['y']
                cx1, cy1 = cx0 + cap['w'], cy0 + cap['h']
                if (cx0 - margin <= dx0 and cy0 - margin <= dy0 and
                        cx1 + margin >= dx1 and cy1 + margin >= dy1):
                    d['is_key'] = True
                    n_marked += 1
                    break
        if VERBOSE and n_marked:
            print(f"[_load_gdt_line_frames] capsule-wrapped KEY: {n_marked}/{len(dims_out)}")

    return dims_out


def _merge_yolo_gdt(gdt_line_dims, gdt_frames_from_lines, yolo_gdt_results):
    """按包围框对应关系，将兼容符号分类结果写入形位尺寸文本；没有匹配结果的项保留未读占位状态。"""
    if not yolo_gdt_results:
        return

    if len(gdt_frames_from_lines) != len(yolo_gdt_results):
        if VERBOSE:
            print(f"[_merge_yolo_gdt] WARNING: frame count ({len(gdt_frames_from_lines)}) "
                  f"!= YOLO result count ({len(yolo_gdt_results)}), truncating to min")

    def _bbox_key(bb):
        return (
            round(float(bb.get("x", 0)), 2),
            round(float(bb.get("y", 0)), 2),
            round(float(bb.get("w", 0)), 2),
            round(float(bb.get("h", 0)), 2),
        )

    # 为每个 gdt_frames_from_lines 框建立符号结果映射；用四元 bbox key
    # 避免同一 x/y 附近的相邻框互相覆盖。
    yolo_by_bbox = {}
    for frame, yolo_det in zip(gdt_frames_from_lines, yolo_gdt_results):
        if yolo_det["symbol_class"] < 0:
            continue
        fb = frame["bbox"]
        yolo_by_bbox[_bbox_key(fb)] = yolo_det

    replaced = 0
    for dim in gdt_line_dims:
        bb = dim.get("bbox", {})
        yolo_det = yolo_by_bbox.get(_bbox_key(bb))
        if not yolo_det:
            continue

        symbol_name = yolo_det["symbol_name"]
        symbol_unicode = yolo_det["symbol_unicode"]
        symbol_conf = yolo_det["confidence"]
        symbol_method = yolo_det.get("method", "yolo")

        # Replace the leading OCR/placeholder symbol with the YOLO-classified
        # symbol. OCR may emit □/口 for profile symbols before YOLO corrects it.
        old_text = dim.get("text", "")
        if "⊕" in old_text:
            dim["text"] = old_text.replace("⊕", symbol_unicode, 1)
        elif old_text.strip() == "GD&T (unread)":
            # compartment OCR 完全失败时，至少把 YOLO 识出的符号写回 text，
            # 前端不再看到千篇一律的 "GD&T (unread)"
            dim["text"] = f"{symbol_unicode} (unread)"
        else:
            dim["text"] = re.sub(
                r'^\s*[口□⊕◎○⌭—∠⌓⌒▱≡⊥∥]',
                symbol_unicode,
                old_text,
                count=1,
            )
        if dim.get("prefix") in {
            "口", "□", "⊕", "◎", "○", "⌭", "—", "∠", "⌓", "⌒", "▱", "≡", "⊥", "∥",
        }:
            dim["prefix"] = symbol_unicode

        # 写入 YOLO 元数据
        dim["symbol_name"] = symbol_name
        dim["symbol_unicode"] = symbol_unicode
        dim["symbol_method"] = symbol_method
        dim["gdt_symbol_confidence"] = symbol_conf
        if symbol_method == "yolo":
            dim["yolo_confidence"] = symbol_conf
        elif symbol_method == "vector_inversion":
            dim["vector_confidence"] = symbol_conf

        frame_symbols = yolo_det.get("frame_symbols") or []
        if frame_symbols:
            dim["gdt_frame_symbols"] = frame_symbols
            modifier_glyphs = []
            for s in frame_symbols:
                try:
                    cls = int(s.get("symbol_class", -1))
                except (TypeError, ValueError):
                    cls = -1
                if 12 <= cls <= 18:
                    modifier_glyphs.append(s.get("symbol_unicode", ""))
            if modifier_glyphs:
                dim["modifier"] = _merge_modifier_glyphs(dim.get("modifier", ""), modifier_glyphs)
                dim["gdt_modifiers"] = dim["modifier"]
        replaced += 1

    if VERBOSE:
        print(f"[yolo_merge] {replaced}/{len(gdt_line_dims)} GD&T dims updated with YOLO symbols")
