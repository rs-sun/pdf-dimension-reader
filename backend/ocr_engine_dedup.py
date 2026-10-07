"""ocr_engine_dedup.py — OCR 去重 + 工程去重 + ⌀ 注入

从 ocr_engine.py 拆出。"""

import re as _re_module

import numpy as np

from ocr_engine_utils import (
    VERBOSE, _bbox_iou, _bbox_containment,
    _ENGINEERING_CHARS_RE, _effective_eng_len, _has_tolerance,
)


def dedup_ocr_by_iou(ocr_results, iou_threshold=0.5):
    """按包围框重叠去重。排序优先考虑有效工程字符长度、公差内容、几何来源和置信度，避免残缺读取覆盖完整读取。"""
    if len(ocr_results) <= 1:
        return ocr_results

    _GEO = {'capsule_ocr', 'dimline_strip_ocr', 'gdt_compartment_ocr'}

    def _key(r):
        text = r.get('text', '')
        eng_len = _effective_eng_len(text)
        has_tol = _has_tolerance(text)
        is_geo = r.get('source', '') in _GEO
        return (-eng_len, -int(has_tol), -int(is_geo), -r.get('confidence', 0))

    sorted_results = sorted(ocr_results, key=_key)
    kept = []
    for candidate in sorted_results:
        cb = candidate['bbox_in_pdf']
        is_dup = False
        for existing in kept:
            if _bbox_iou(cb, existing['bbox_in_pdf']) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)

    return kept


def dedup_ocr_engineering(ocr_results, iou_threshold=0.5, verbose=None):
    """融合多源 OCR 结果并执行工程去重；优先保留更完整的工程字符和公差内容，以几何来源打破并列。返回去重结果。"""
    if verbose is None:
        verbose = VERBOSE
    if len(ocr_results) <= 1:
        return ocr_results

    # 几何源优先级（位置更精确的源排在后面作为 tiebreaker）
    _GEO_SOURCES = {'capsule_ocr', 'dimline_strip_ocr', 'gdt_compartment_ocr'}

    # 预计算每个结果的排序键
    def _sort_key(r):
        text = r.get('text', '')
        eng_len = _effective_eng_len(text)
        has_tol = _has_tolerance(text)
        is_geo = r.get('source', '') in _GEO_SOURCES
        # 排序：eng_len 降序, has_tol 降序, is_geo 降序
        return (-eng_len, -int(has_tol), -int(is_geo))

    sorted_results = sorted(ocr_results, key=_sort_key)
    kept = []
    n_deduped = 0

    for candidate in sorted_results:
        cb = candidate['bbox_in_pdf']
        cand_eng = _effective_eng_len(candidate.get('text', ''))
        is_dup = False
        for existing in kept:
            eb = existing['bbox_in_pdf']
            dup_reason = None
            if _bbox_iou(cb, eb) > iou_threshold:
                dup_reason = 'iou'
            # 用包含关系过滤残缺读取；形位格内容仅在携带完整读取未包含的信息时保留。
            
            
            
            
            
            
            
            
            elif (_bbox_containment(cb, eb) > 0.9
                  and _effective_eng_len(existing.get('text', '')) > cand_eng
                  and not (
                      candidate.get('source', '') == 'gdt_compartment_ocr'
                      and candidate.get('text', '').strip() not in existing.get('text', '')
                  )):
                dup_reason = 'contain'
            if dup_reason:
                is_dup = True
                if verbose:
                    print(f"[s27.dedup] DROP '{candidate.get('text','')}' "
                          f"(src={candidate.get('source','')}, "
                          f"eng={cand_eng}, "
                          f"tol={_has_tolerance(candidate.get('text',''))}) "
                          f"-> KEEP '{existing.get('text','')}' "
                          f"(src={existing.get('source','')}, "
                          f"eng={_effective_eng_len(existing.get('text',''))}, "
                          f"tol={_has_tolerance(existing.get('text',''))}) "
                          f"reason={dup_reason}")
                n_deduped += 1
                break
        if not is_dup:
            kept.append(candidate)

    if verbose and n_deduped:
        print(f"[s27.dedup] {len(ocr_results)} -> {len(kept)} ({n_deduped} removed)")

    return kept


# ---------------------------------------------------------------------------
# ⌀ 前缀回填（绕开 PaddleOCR 字典不含 ⌀ 的问题）
# ---------------------------------------------------------------------------

_DIAM_ALREADY_RE = _re_module.compile(r'[⌀∅Øφ]')
_LEADING_REPEAT_RE = _re_module.compile(r'^(\s*\d+\s*[×xX]\s*)')
# 在数值前缀上下文中替换误读的直径字符，避免重复前缀；普通标识按原规则排除。



_LEADING_Q_DIAM_RE = _re_module.compile(r'^(\s*)[Qqg]\s*(?=\d)')
# 严格全文校验：主数 + 可选（度数 / ± 公差 / 签名公差 [带可选 /签名公差]
# / 公差等级字母 / 间隔的第二个数）。用整串匹配 ($), 任何乱码（# ( ) 不成对字母簇等）都被拒绝。
_VALID_DIM_TEXT_RE = _re_module.compile(
    r'^\s*'
    r'\d+(?:\.\d+)?'
    r'(?:'
    r'\s*°'
    r'|\s*±\s*\d+(?:\.\d+)?'
    r'|\s*[+\-]\s*\d+(?:\.\d+)?(?:\s*/\s*[+\-]\s*\d+(?:\.\d+)?)?'
    r'|\s*[HhKkPpGgMmNnRrTtJj]\d*'
    r'|\s+\d+(?:\.\d+)?'
    r')*'
    r'\s*$'
)


def inject_diameter_prefix(ocr_results, glyphs, verbose=None):
    """兼容路径中，将矢量直径符号与满足完整尺寸语法的 OCR 文字空间配对。按倍数前缀和误读前缀决定插入或替换位置，保留已有符号，拒绝孤立零值；原地更新并返回结果。"""
    if verbose is None:
        verbose = VERBOSE
    if not ocr_results or not glyphs:
        return ocr_results

    glyph_arr = np.array([[g['cx'], g['cy'], g['w'], g['h']] for g in glyphs])
    n_injected = 0
    matched_glyph_idx = set()

    for r in ocr_results:
        text = r.get('text', '')
        if not text or _DIAM_ALREADY_RE.search(text):
            continue
        stripped = text.strip()
        if stripped in ('0', '+0', '-0'):
            continue
        prefix_match = _LEADING_REPEAT_RE.match(text)
        # 按倍数、误读前缀和普通前缀情况选择符号注入位置。
        inject_mode = 'prepend'
        q_match = None
        if prefix_match:
            tail = text[prefix_match.end():]
            # nX 之后再判 Q 误读
            q_match_tail = _LEADING_Q_DIAM_RE.match(tail)
            if q_match_tail:
                # 把 nX 后面的 Q 替换为 ⌀
                tail_after_q = tail[q_match_tail.end():]
                if not _VALID_DIM_TEXT_RE.match(tail_after_q):
                    continue
                inject_mode = 'replace_q_after_prefix'
                q_match = q_match_tail
            elif not _VALID_DIM_TEXT_RE.match(tail):
                continue
            else:
                inject_mode = 'prefix'
        else:
            q_match = _LEADING_Q_DIAM_RE.match(text)
            if q_match:
                tail_after_q = text[q_match.end():]
                if not _VALID_DIM_TEXT_RE.match(tail_after_q):
                    continue
                inject_mode = 'replace_q'
            elif not _VALID_DIM_TEXT_RE.match(text):
                continue

        bb = r.get('bbox_in_pdf') or {}
        bx = bb.get('x', 0)
        by = bb.get('y', 0)
        bw = bb.get('w', 0)
        bh = bb.get('h', 0)
        if bw <= 0 or bh <= 0:
            continue
        by_center = by + bh / 2

        # 倍数场景采用扩展横向匹配范围。
        if prefix_match:
            x_right = bw + 5.0
        else:
            x_right = max(12.0, bw * 0.35)
        dx = glyph_arr[:, 0] - bx
        dy = glyph_arr[:, 1] - by_center
        x_ok = (dx >= -15.0) & (dx <= x_right)
        y_ok = np.abs(dy) <= (bh / 2 + 2.0)
        vertical_layout = bh >= bw * 1.4
        try:
            vertical_layout = vertical_layout or abs(abs(float(r.get('orientation') or 0.0)) - 90.0) <= 15.0
        except (TypeError, ValueError):
            pass
        if vertical_layout:
            glyph_dx_center = glyph_arr[:, 0] - (bx + bw / 2.0)
            vx_ok = np.abs(glyph_dx_center) <= (bw / 2.0 + 8.0)
            vy_ok = (glyph_arr[:, 1] >= by - 15.0) & (glyph_arr[:, 1] <= by + bh + 18.0)
            matched = np.where((x_ok & y_ok) | (vx_ok & vy_ok))[0]
        else:
            matched = np.where(x_ok & y_ok)[0]
        if len(matched) == 0:
            continue

        matched_glyph_idx.update(matched.tolist())
        if inject_mode == 'prefix':
            new_text = text[:prefix_match.end()] + '⌀' + text[prefix_match.end():]
        elif inject_mode == 'replace_q':
            # 把开头的 Q（含前导空格）整体替换为 ⌀
            new_text = '⌀' + text[q_match.end():]
        elif inject_mode == 'replace_q_after_prefix':
            # nX <space> Q<digits> → nX <space> ⌀<digits>
            head = text[:prefix_match.end()]
            tail_after_q = text[prefix_match.end() + q_match.end():]
            new_text = head + '⌀' + tail_after_q
        else:
            new_text = '⌀' + text
        if verbose:
            print(f"[diam_inject] '{text}' -> '{new_text}' "
                  f"bbox=({bx:.0f},{by:.0f},{bw:.0f}x{bh:.0f}) "
                  f"src={r.get('source','')}")
        r['text'] = new_text
        r['diameter_injected'] = True
        n_injected += 1

    if verbose:
        print(f"[diam_inject] injected ⌀ into {n_injected}/{len(ocr_results)} OCR results "
              f"(from {len(glyphs)} vector glyphs)")
        # 诊断未匹配符号时输出局部候选和拒绝原因。
        for gi, g in enumerate(glyphs):
            if gi in matched_glyph_idx:
                continue
            gx, gy = g['cx'], g['cy']
            cands = []
            for r in ocr_results:
                bb = r.get('bbox_in_pdf') or {}
                bx, by, bw, bh = bb.get('x',0), bb.get('y',0), bb.get('w',0), bb.get('h',0)
                if bw <= 0 or bh <= 0:
                    continue
                by_center = by + bh/2
                cdx = gx - bx
                cdy = gy - by_center
                dist = (cdx*cdx + cdy*cdy) ** 0.5
                cands.append((dist, cdx, cdy, bx, by, bw, bh, r.get('text',''), r.get('source','')))
            cands.sort()
            print(f"[diam_inject][MISS] glyph#{gi} cx={gx:.0f} cy={gy:.0f}")
            for c in cands[:5]:
                dist, cdx, cdy, bx, by, bw, bh, ctext, csrc = c
                # 判断为啥拒绝（使用实际生效的 nX 放宽 bound）
                pm = _LEADING_REPEAT_RE.match(ctext)
                x_right = (bw + 5.0) if pm else max(12.0, bw * 0.35)
                reason = []
                if cdx < -15.0 or cdx > x_right:
                    reason.append(f"x_fail(dx={cdx:+.0f}, bound=[-15,{x_right:+.0f}])")
                if abs(cdy) > bh/2 + 2.0:
                    reason.append(f"y_fail(dy={cdy:+.0f}, bound={bh/2+2.0:.0f})")
                if _DIAM_ALREADY_RE.search(ctext):
                    reason.append("already_has_diam")
                if ctext.strip() in ('0','+0','-0'):
                    reason.append("zero_only")
                # 模拟 inject 路径走的实际 test_str（含 Q 误读路径）
                test_str = ctext[pm.end():] if pm else ctext
                qm = _LEADING_Q_DIAM_RE.match(test_str)
                if qm:
                    test_str = test_str[qm.end():]
                if not _VALID_DIM_TEXT_RE.match(test_str):
                    reason.append(f"text_fail('{test_str}')")
                if not reason:
                    reason.append("UNKNOWN")
                print(f"[diam_inject][MISS.cand] d={dist:6.0f} dx={cdx:+5.0f} dy={cdy:+4.0f} bbox=({bx:.0f},{by:.0f},{bw:.0f}x{bh:.0f}) text='{ctext[:30]}' src={csrc} -> {';'.join(reason)}")

    return ocr_results
