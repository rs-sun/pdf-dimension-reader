#!/usr/bin/env python3

import argparse
import io
import os
import re
import sys
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# OCR 兼容模式的默认选择只在命令行入口设置。
# Flask 会将本模块作为库导入，因此模块导入不能改写主服务的环境配置。
# 调用方显式设置的 OCR_BACKEND 始终优先于命令行默认值。

import openpyxl
import pdfplumber

from pipeline import run_type_b_pipeline

# Lazy gdt detector load — keep module import light for Flask tests/startup.
_GDT_DETECTOR = None
_GDT_DETECTOR_LOAD_ATTEMPTED = False


def _get_default_gdt_detector():
    global _GDT_DETECTOR, _GDT_DETECTOR_LOAD_ATTEMPTED
    if _GDT_DETECTOR_LOAD_ATTEMPTED:
        return _GDT_DETECTOR
    _GDT_DETECTOR_LOAD_ATTEMPTED = True
    try:
        from gdt_detector import GDTDetector
        _GDT_DETECTOR = GDTDetector()
    except Exception:
        _GDT_DETECTOR = None
    return _GDT_DETECTOR

# GD&T 文本 → 中文名的 unicode 回退映射（symbol_name 为空时用）
_UNI_TO_CN = {
    '⊕': '位置度', '⌖': '位置度',
    '◎': '同心度', '◉': '同心度',
    '≡': '对称度',
    '∥': '平行度', '⊥': '垂直度', '∠': '倾斜度',
    '○': '圆度', '⊙': '圆柱度',
    '▱': '平面度', '▱': '直线度', '—': '直线度',
    '⌓': '面轮廓度', '⌒': '线轮廓度',
    '△': '面轮廓度', '□': '面轮廓度',
    '↗': '圆跳动', '↯': '全跳动',
}


HEADERS = ['NO.', '重点尺寸', '名义值', '上差/UTL', '下差/LTL', 1, 2, 3]

DETAIL_HEADERS = [
    'NO.', '重点尺寸', '名义值', '上公差', '下公差', '上限', '下限',
    '样件1实测值', '样件2实测值', '样件3实测值',
    '样件4实测值', '样件5实测值', '样件6实测值',
    'OK/NOK', '测量设备', '原始测量项名称', '来源文件', '匹配状态',
    'stamp_uuid', '页码', '尺寸类型', '前缀', '后端来源', 'source_row',
    '单位',
]


def _parse_float(s: Optional[str]) -> Optional[float]:
    if s is None: return None
    if isinstance(s, (int, float)): return float(s)
    m = re.search(r'[-+]?\d+(?:\.\d+)?', str(s))
    if not m: return None
    try:
        v = float(m.group())
        # Preserve sign if string had leading '-' or '+'
        if str(s).strip().startswith('-') and v > 0: v = -v
        return v
    except Exception:
        return None


_GDT_SYMBOL_CN = {
    'position': '位置度',      'symmetry': '对称度',
    'concentricity': '同心度', 'coaxiality': '同轴度',
    'parallelism': '平行度',   'perpendicularity': '垂直度',
    'angularity': '倾斜度',    'circularity': '圆度',
    'cylindricity': '圆柱度',  'flatness': '平面度',
    'straightness': '直线度',  'line_profile': '线轮廓度',
    'surface_profile': '面轮廓度',
    'runout': '圆跳动',        'total_runout': '全跳动',
}


def _strip_leading_zero(s: str) -> str:
    """去除多位整数部分的前导零，保留小数零位。"""
    return re.sub(r'\b0+(\d+\.\d+)', r'\1', str(s))


def _clean_number(s):
    """Normalize a numeric string: strip leading zeros in multi-digit ints."""
    if s is None:
        return ''
    return _strip_leading_zero(str(s).strip())


def _strip_multiplier_prefix(prefix: str) -> str:
    """移除倍数前缀，保留形状、直径和粗糙度等其余工程前缀。"""
    if not prefix:
        return ''
    s = re.sub(r'^\s*\d+\s*X\s*', '', str(prefix), flags=re.IGNORECASE)
    return s.strip()


def _format_nominal(dim: dict) -> str:
    """构建统一的名义值字符串：去除倍数前缀，规范直径符号与前导零，并格式化公差。"""
    nom = _clean_number(dim.get('nominal', '') or '')
    up  = dim.get('upper_tol')
    lo  = dim.get('lower_tol')
    # 倍数前缀按 gold 习惯去掉，保留材质 / 形状前缀
    prefix = _strip_multiplier_prefix(dim.get('prefix') or '')
    # ⌀ → Ø 统一（gold 用 Ø）
    prefix = prefix.replace('⌀', 'Ø')
    dtype = dim.get('type', '')
    unit = str(dim.get('unit') or '').strip()

    def with_unit(value: Any) -> str:
        text = str(value or '')
        if not text or unit != 'degree' or text.endswith('°'):
            return text
        return f'{text}°'

    # ── 非 GD&T 类型 ────────────────────────────────────────
    if dtype != 'gdt':
        if dtype == 'surface_rough':
            return f'{prefix}{nom}'.strip()
        if up is None and lo is None:
            return f'{prefix}{with_unit(nom)}'.strip()
        up_str = str(up).strip() if up is not None else ''
        lo_str = str(lo).strip() if lo is not None else ''
        # 对称公差 → ±
        if up_str.startswith('+') and lo_str.startswith('-') and up_str[1:] == lo_str[1:]:
            return f'{prefix}{with_unit(nom)}±{with_unit(up_str[1:])}'
        # 非对称 / 单边
        pieces = [f'{prefix}{with_unit(nom)}']
        if up_str:
            pieces.append(
                with_unit(up_str)
                if up_str.startswith(('+', '-'))
                else f'+{with_unit(up_str)}'
            )
        if lo_str:
            pieces.append(
                with_unit(lo_str)
                if lo_str.startswith(('+', '-'))
                else f'-{with_unit(lo_str)}'
            )
        return ''.join(pieces)

    # 按结构化框字段格式化形位文本，兼容单引用行与聚合行。
    
    return _format_gdt(dim)


def _format_gdt(dim: dict) -> str:
    """Format one (possibly pre-aggregated) GD&T dim row."""
    sym_name = dim.get('symbol_name', '')
    sym_uni  = dim.get('symbol_unicode', '') or ''
    text     = dim.get('text', '') or ''
    # 优先用 symbol_name -> 中文
    cn = _GDT_SYMBOL_CN.get(sym_name, '')
    if not cn:
        # symbol_unicode 回退
        cn = _UNI_TO_CN.get(sym_uni, '')
    if not cn:
        # 从 text 里找已知 unicode 符号
        for u, c in _UNI_TO_CN.items():
            if u in text:
                cn = c
                break
    if not cn:
        cn = sym_uni or '□'

    zone = dim.get('_zone') or _extract_gdt_zone(dim)
    zone_str = f'{zone:g}' if zone is not None else '?'

    # 直径修饰 — text 含 ⌀/Ø/"D" 紧跟数字
    has_diameter = bool(re.search(r'[⌀Ø∅]', text)) or 'Ø' in text
    # 按受限数值上下文判断误读的直径前缀。
    if not has_diameter and re.search(r'\bD\s+\d', text):
        has_diameter = True
    dia_mark = 'Ø' if has_diameter else ''

    # 修饰符 — 优先用结构化字段，再回退到 text。
    modifier = ''
    modifier_src = f"{dim.get('modifier', '')} {dim.get('gdt_modifiers', '')} {text}"
    for glyph, letter in {
        'Ⓜ': 'M', 'Ⓛ': 'L', 'Ⓟ': 'P', 'Ⓕ': 'F', 'Ⓣ': 'T', 'Ⓢ': 'S', 'Ⓒ': 'C',
    }.items():
        if glyph in modifier_src:
            modifier = letter
            break
    if not modifier:
        if re.search(r'\d+(?:\.\d+)?\s*\(?M\)?\b|\b[A-Z]{1,3}\(M\)', modifier_src):
            modifier = 'M'
        elif re.search(r'\d+(?:\.\d+)?\s*\(?L\)?\b|\b[A-Z]{1,3}\(L\)', modifier_src):
            modifier = 'L'
        elif re.search(r'\d+(?:\.\d+)?\s*\(?P\)?\b|\b[A-Z]{1,3}\(P\)', modifier_src):
            modifier = 'P'

    # datum：聚合多行则传入 _datums；否则取单 datum
    datums = dim.get('_datums')
    if datums is None:
        datums = [
            dim.get('datum_1', ''),
            dim.get('datum_2', ''),
            dim.get('datum_3', ''),
        ]
        if not any(datums):
            d = (dim.get('datum') or '').strip()
            datums = re.split(r'[\s/]+', d) if d else []
    # 过滤 OCR 噪声 datum（允许 A/B/C，也允许 RR/SS/TT 这类重复字母基准）
    datums = [
        str(d).strip()
        for d in datums
        if re.match(r'^(?:[A-Z]|([A-Z])\1{1,2})$', str(d).strip())
    ]
    # 不重复
    seen = set(); clean = []
    for d in datums:
        if d not in seen:
            clean.append(d); seen.add(d)
    datum_part = f'-{"".join(clean)}' if clean else ''

    return f'{cn}{dia_mark}{zone_str}{modifier}{datum_part}'


def _extract_gdt_zone(dim: dict):
    """Extract zone value from a GD&T dim, trying nominal/tol and falling back to text."""
    for candidate in (dim.get('nominal'), dim.get('upper_tol'), dim.get('lower_tol')):
        if candidate is None or candidate == '': continue
        c = str(candidate).strip().lstrip('+-')
        # 移除数值末尾的显式修饰符。
        c = re.sub(r'[MLmlPp]$', '', c)
        if c:
            try: return abs(float(c))
            except Exception: continue
    # Fallback: extract first float from text
    text = dim.get('text', '') or ''
    m = re.search(r'\d+(?:\.\d+)?', text)
    if m:
        try: return float(m.group())
        except Exception: pass
    return None


def _collapse_gdt_rows(dims: list) -> list:
    """按共同框文本收集基准字段，并为对应的形位行生成一致的框级字符串；不合并或删除原始行。"""
    # 先按 text 建 datum 索引
    text_to_datums = {}
    for d in dims:
        if d.get('type') != 'gdt': continue
        key = (_first_present(d, ['page', 'page_number', 'pageIndex'], 1), d.get('text', ''))
        datum = (d.get('datum') or '').strip()
        if datum:
            text_to_datums.setdefault(key, []).append(datum)

    out = []
    for d in dims:
        if d.get('type') != 'gdt':
            out.append(d); continue
        enriched = dict(d)
        key = (_first_present(d, ['page', 'page_number', 'pageIndex'], 1), d.get('text', ''))
        enriched['_datums'] = text_to_datums.get(key, [])
        enriched['_zone'] = _extract_gdt_zone(enriched)
        out.append(enriched)
    return out


def _compute_limits(dim: dict) -> tuple[Optional[float], Optional[float]]:
    """Return (UTL, LTL) as floats or (None, None)."""
    nom = _parse_float(dim.get('nominal'))
    up  = _parse_float(dim.get('upper_tol'))
    lo  = _parse_float(dim.get('lower_tol'))
    utl = (nom + up) if nom is not None and up is not None else None
    ltl = (nom + lo) if nom is not None and lo is not None else None  # lo 通常已含负号
    return utl, ltl


def _first_present(d: dict, keys: list[str], default=None):
    for k in keys:
        v = d.get(k)
        if v is not None and v != '':
            return v
    return default


def _detail_nominal(dim: dict):
    """返回不含工程前缀与公差文字的名义值，供明细表使用。"""
    v = _first_present(dim, ['nominal_clean', 'nominal_value', 'nominal'])
    if v is None:
        return ''
    parsed = _parse_float(v)
    if parsed is not None:
        return parsed
    return _clean_number(v)


def _detail_tol(v):
    if v is None or v == '':
        return None
    parsed = _parse_float(v)
    return parsed if parsed is not None else str(v).strip()


def _detail_limits(dim: dict) -> tuple[Optional[float], Optional[float]]:
    """Prefer explicit imported limits; otherwise compute from nominal/tolerances."""
    utl = _parse_float(_first_present(dim, ['utl', 'UTL', 'upper_limit', 'upperLimit']))
    ltl = _parse_float(_first_present(dim, ['ltl', 'LTL', 'lower_limit', 'lowerLimit']))
    if utl is not None or ltl is not None:
        return utl, ltl
    return _compute_limits(dim)


def _detail_actuals(dim: dict) -> list:
    """Normalize actual measurement values to six sample columns."""
    raw = _first_present(dim, ['actuals', 'measurements', 'sample_values', 'samples'], [])
    out = [None] * 6
    if isinstance(raw, list):
        for i, item in enumerate(raw[:6]):
            if isinstance(item, dict):
                idx = item.get('sample') or item.get('sample_index') or item.get('index') or (i + 1)
                try:
                    pos = int(idx) - 1
                except Exception:
                    pos = i
                if 0 <= pos < 6:
                    out[pos] = _parse_float(_first_present(item, ['value', 'actual', 'measurement']))
            else:
                out[i] = _parse_float(item)
    for i in range(6):
        v = _first_present(dim, [
            f'sample_{i+1}', f'sample{i+1}', f'actual_{i+1}', f'actual{i+1}',
            f'measure_{i+1}', f'measure{i+1}',
        ])
        if v is not None:
            out[i] = _parse_float(v)
    return out


def _detail_result(dim: dict, actuals: list, utl, ltl) -> str:
    explicit = _first_present(dim, ['inspection_result', 'result', 'ok_nok', 'status'])
    if explicit:
        s = str(explicit).strip().upper()
        if s in {'OK', 'NOK', 'NG'}:
            return 'NOK' if s == 'NG' else s
        if str(explicit).strip() in {'未测量', '临界'}:
            return str(explicit).strip()
    values = [v for v in actuals if isinstance(v, (int, float))]
    if not values:
        return '未测量'
    if utl is None or ltl is None:
        return 'UNKNOWN'
    return 'OK' if all(ltl <= v <= utl for v in values) else 'NOK'


def _append_detail_row(ws, no, dim: dict):
    utl, ltl = _detail_limits(dim)
    actuals = _detail_actuals(dim)
    result = _detail_result(dim, actuals, utl, ltl)
    ws.append([
        no,
        'Y' if dim.get('is_key') else None,
        _detail_nominal(dim),
        _detail_tol(dim.get('upper_tol')),
        _detail_tol(dim.get('lower_tol')),
        utl,
        ltl,
        *actuals,
        result,
        _first_present(dim, ['equipment', 'measurement_equipment', 'measure_equip'], ''),
        _first_present(dim, ['raw_item_name', 'raw_measurement_name', 'raw_name', 'text'], ''),
        _first_present(dim, ['source_file', 'measurement_file', 'file_name'], ''),
        _first_present(dim, ['match_status'], 'not_imported'),
        _first_present(dim, ['uuid', 'stamp_uuid'], ''),
        _first_present(dim, ['page', 'page_number', 'pageIndex'], ''),
        _first_present(dim, ['dim_type', 'type'], ''),
        _first_present(dim, ['prefix'], ''),
        _first_present(dim, ['source'], ''),
        _first_present(dim, ['source_row'], ''),
        _first_present(dim, ['unit'], ''),
    ])


def _export_no(dim: dict, fallback):
    no = _first_present(dim, ['no', 'stamp_no', 'stamp_id', 'id'])
    if no is None or no == '':
        return fallback
    return str(no)


def _sort_key(dim: dict) -> tuple:
    """Sort top-to-bottom, then left-to-right by bbox."""
    bb = dim.get('bbox') or {}
    y = bb.get('y', 0) or 0
    x = bb.get('x', 0) or 0
    # 按量化纵向位置分行，同一行按横向位置排序。
    row_bucket = int(y // 15)
    return (row_bucket, x)


def _run_page(pdf_bytes: bytes, page, page_index: int, gdt_detector=None) -> list:
    """Run pipeline on one page, return sorted dim list with GD&T collapsed.

    gdt_detector 优先使用调用方传入的实例（避免重复加载模型）；
    传 None 时回退到模块级 _GDT_DETECTOR。
    """
    detector = gdt_detector if gdt_detector is not None else _get_default_gdt_detector()
    result = run_type_b_pipeline(
        pdf_bytes=pdf_bytes, page=page, page_index=page_index,
        dpi=200, gdt_detector=detector, verbose=False,
    )
    dims = result.get('dimensions', []) or []
    dims = _collapse_gdt_rows(dims)
    dims = sorted(dims, key=_sort_key)
    return dims


def build_xlsx_from_bytes(pdf_bytes: bytes, part_no: str,
                          page_number: Optional[int] = None,
                          all_pages: bool = False,
                          gdt_detector=None,
                          dims_override: Optional[list] = None,
                          document_page_count: Optional[int] = None) -> bytes:
    """将内存 PDF 或调用方提供的校对尺寸转换为工作簿字节。校对覆盖路径保持输入顺序，并仅在已有记录时写入测量明细；文档页数可用于包含空白页的页清单。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '尺寸表'
    ws.append(HEADERS)
    ws.append([None, None, part_no])
    detail_ws = wb.create_sheet('数据明细')
    detail_ws.append(DETAIL_HEADERS)

    if dims_override is not None:
        # 前端校对路径：直接用传入的 dim list 写表，不跑 pipeline、不开 pdfplumber。
        # all_pages=True 时 dims 已是统一全文档快照，按 page 字段加分隔行。
        # 不调 _sort_key——尊重调用方（前端）传入的顺序。前端 stamp.id 按
        # 视图分组 + 行优先排序，backend bbox 全局排序与之不一致；让前端排好
        # 顺序后传，行号 NO 与 stamp PDF 上的序号对齐。
        dims = _collapse_gdt_rows(dims_override)
        row_counter = 0
        previous_page = None
        for d in dims:
            raw_page = _first_present(d, ['page', 'page_number', 'pageIndex'], page_number or 1)
            try:
                item_page = int(raw_page)
            except (TypeError, ValueError):
                raise ValueError(f'dim page must be a positive integer; received {raw_page!r}')
            if item_page < 1:
                raise ValueError(f'dim page must be a positive integer; received {raw_page!r}')
            if all_pages and previous_page is not None and item_page != previous_page:
                ws.append([None, None, f'第{item_page}页 / page{item_page}'])
            previous_page = item_page
            row_counter += 1
            no = _export_no(d, row_counter)
            nom_str = _format_nominal(d)
            utl, ltl = _compute_limits(d)
            # B 列「重点尺寸」：前端传 is_key=True 时写 'Y'，否则空
            key_mark = 'Y' if d.get('is_key') else None
            ws.append([no, key_mark, nom_str, utl, ltl, None, None, None])
            _append_detail_row(detail_ws, no, d)

        if all_pages:
            page_ws = wb.create_sheet('页清单')
            page_ws.append(['页码', '条目数'])
            counts_by_page = {}
            for dim in dims:
                page = int(_first_present(
                    dim,
                    ['page', 'page_number', 'pageIndex'],
                    page_number or 1,
                ))
                counts_by_page[page] = counts_by_page.get(page, 0) + 1
            if document_page_count:
                export_pages = range(1, int(document_page_count) + 1)
            else:
                export_pages = sorted(counts_by_page)
            for page in export_pages:
                page_ws.append([page, counts_by_page.get(page, 0)])
            page_ws.freeze_panes = 'A2'
            page_ws.column_dimensions['A'].width = 10
            page_ws.column_dimensions['B'].width = 12
    else:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_pdf:
            total_pages = len(plumber_pdf.pages)
            if all_pages:
                pages_to_run = list(range(total_pages))
            else:
                pn = (page_number or 1) - 1
                if pn < 0 or pn >= total_pages:
                    raise ValueError(f'page_number {page_number or 1} out of range (1-{total_pages})')
                pages_to_run = [pn]

            row_counter = 0
            for pidx in pages_to_run:
                page = plumber_pdf.pages[pidx]
                dims = _run_page(pdf_bytes, page, pidx, gdt_detector=gdt_detector)
                # 非首页加分隔行（保持 gold 排版风格）
                if pidx > pages_to_run[0] and dims:
                    ws.append([None, None, f'第{pidx+1}页 / page{pidx+1}'])
                for d in dims:
                    row_counter += 1
                    nom_str = _format_nominal(d)
                    utl, ltl = _compute_limits(d)
                    ws.append([row_counter, None, nom_str, utl, ltl, None, None, None])

    ws.freeze_panes = 'A3'
    for col, width in [('A', 6), ('B', 10), ('C', 22), ('D', 12), ('E', 12),
                       ('F', 8), ('G', 8), ('H', 8)]:
        ws.column_dimensions[col].width = width

    detail_ws.freeze_panes = 'A2'
    detail_widths = {
        'A': 8, 'B': 10, 'C': 12, 'D': 10, 'E': 10, 'F': 12, 'G': 12,
        'H': 12, 'I': 12, 'J': 12, 'K': 12, 'L': 12, 'M': 12,
        'N': 10, 'O': 14, 'P': 28, 'Q': 24, 'R': 16,
        'S': 34, 'T': 8, 'U': 14, 'V': 10, 'W': 14, 'X': 10,
        'Y': 10,
    }
    for col, width in detail_widths.items():
        detail_ws.column_dimensions[col].width = width

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_xlsx(pdf_path: str, page_number: Optional[int] = None,
               out_path: Optional[str] = None, all_pages: bool = False) -> str:
    """读取文件，调用内存工作簿构建器并写入输出文件；兼容单页和全文档命令行调用。"""
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    part_no = os.path.splitext(os.path.basename(pdf_path))[0]
    xlsx_bytes = build_xlsx_from_bytes(
        pdf_bytes=pdf_bytes,
        part_no=part_no,
        page_number=page_number,
        all_pages=all_pages,
        gdt_detector=None,  # 让内部回退到模块级 _GDT_DETECTOR
    )

    if out_path is None:
        out_path = os.path.splitext(pdf_path)[0] + '_gold_draft.xlsx'
    with open(out_path, 'wb') as f:
        f.write(xlsx_bytes)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pdf_path')
    p.add_argument('--page', type=int, default=None, help='1-based 单页；不传默认页 1')
    p.add_argument('--all-pages', action='store_true', help='跑所有页，页间加分隔行')
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    out = build_xlsx(args.pdf_path, args.page, args.out, all_pages=args.all_pages)
    wb = openpyxl.load_workbook(out, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = sum(1 for r in range(3, ws.max_row + 1) if ws.cell(r, 1).value is not None)
    print(f'written: {out}')
    print(f'  dim rows: {rows}')
    print(f'  part_no (C2): {ws.cell(2, 3).value}')
    print(f'  all_pages: {args.all_pages}')


if __name__ == '__main__':
    # 命令行入口采用兼容默认值，允许调用方显式环境配置覆盖。
    
    os.environ.setdefault('OCR_BACKEND', 'rapidocr')
    main()
