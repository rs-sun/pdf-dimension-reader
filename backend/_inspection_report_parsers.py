"""Semantic parsers for supported inspection workbook layouts.

The input is deliberately format-neutral ParsedSheet data. Binary Excel
handling belongs to _inspection_workbook_decoder.
"""

from __future__ import annotations

import re
from typing import Any, Optional

if __package__:
    from ._inspection_workbook_decoder import ParsedSheet
else:
    from _inspection_workbook_decoder import ParsedSheet


_NUM_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')
_NO_PREFIX_RE = re.compile(r'^\s*(\d+(?:\s*-\s*\d+)?)\s*#\s*(.+?)\s*$')
_NO_LABEL_RE = re.compile(r'^\s*(\d+)(?:\s*-\s*(\d+))?\s*$')
_LIMIT_HEADER_TOKENS = (
    'utl', 'ltl', 'usl', 'lsl', 'upper limit', 'lower limit',
    '上限', '下限', '最大极限', '最小极限',
)
_DEVIATION_HEADER_TOKENS = (
    '上差', '下差', '上偏差', '下偏差', '偏差',
    'upper tol', 'lower tol', 'upper tolerance', 'lower tolerance',
)


def _norm_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _parse_float(value: Any) -> Optional[float]:
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = (
        str(value)
        .replace(',', '')
        .replace('＋', '+')
        .replace('－', '-')
        .replace('−', '-')
        .replace('﹣', '-')
        .strip()
    )
    if not text:
        return None
    match = _NUM_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group())
    except Exception:
        return None


def _is_sample_number(value: Any, expected: int) -> bool:
    n = _parse_float(value)
    return n is not None and int(n) == expected and abs(n - expected) < 1e-9


def _normal_result(value: Any) -> str:
    text = _norm_text(value).upper().replace(' ', '')
    if text in {'OK', '合格'}:
        return 'OK'
    if text in {'NOK', 'NG', '不合格'}:
        return 'NOK'
    if text in {'未测量', 'UNMEASURED'}:
        return 'UNMEASURED'
    if text in {'临界', 'CRITICAL'}:
        return 'CRITICAL'
    return ''


def _parse_no_label(value: Any) -> Optional[str]:
    text = _norm_text(value)
    if not text:
        return None
    match = _NO_LABEL_RE.match(text)
    if match:
        head = str(int(match.group(1)))
        tail = match.group(2)
        return f'{head}-{int(tail)}' if tail is not None else head
    n = _parse_float(value)
    if n is None:
        return None
    return str(int(n)) if float(n).is_integer() else str(n)


def _compute_result(actuals: list[dict], utl: Optional[float], ltl: Optional[float], explicit: Any = None) -> str:
    explicit_result = _normal_result(explicit)
    values = [a['value'] for a in actuals if isinstance(a.get('value'), (int, float))]
    if not values:
        return explicit_result or 'UNMEASURED'
    if utl is None or ltl is None:
        return explicit_result or 'UNKNOWN'
    computed = 'OK' if all(ltl <= v <= utl for v in values) else 'NOK'
    return computed


def _add_limit(nominal: Optional[float], tol: Optional[float]) -> Optional[float]:
    if nominal is None or tol is None:
        return None
    return round(nominal + tol, 10)


def _strip_leading_quantity(text: str) -> str:
    return re.sub(r'^\s*\d+\s*[*xX×]\s*', '', text)


def _find_nominal_number(text: str) -> tuple[Optional[float], int]:
    source = _strip_leading_quantity(text)
    prefixed_patterns = (
        r'\bM\s*([-+]?\d+(?:\.\d+)?)',
        r'(?:SR|sr)\s*([-+]?\d+(?:\.\d+)?)',
        r'[⌀∅ØφΦRr]\s*([-+]?\d+(?:\.\d+)?)',
    )
    for pattern in prefixed_patterns:
        match = re.search(pattern, source)
        if match:
            return float(match.group(1)), match.end(1)

    match = _NUM_RE.search(source)
    if not match:
        return None, 0
    return float(match.group()), match.end()


def _parse_item_nominal_tol(item_text: Any, utl: Optional[float] = None, ltl: Optional[float] = None):
    text = (
        _norm_text(item_text)
        .replace('＋', '+')
        .replace('－', '-')
        .replace('−', '-')
        .replace('﹣', '-')
    )
    if not text:
        return None, None, None

    # Drop a leading report-item label before parsing the nominal value.
    no_match = _NO_PREFIX_RE.match(text)
    if no_match:
        text = no_match.group(2)

    parse_text = _strip_leading_quantity(text)
    nominal, nominal_end = _find_nominal_number(parse_text)
    upper_tol = None
    lower_tol = None

    sym = re.search(r'±\s*(\d+(?:\.\d+)?)', parse_text)
    if sym:
        tol = float(sym.group(1))
        upper_tol = tol
        lower_tol = -tol
    else:
        # Parse signed asymmetric tolerances following the nominal value.
        if nominal is not None:
            tail = parse_text[nominal_end:]
            signed = re.findall(r'([+-]\s*\d+(?:\.\d+)?)', tail)
            if signed:
                parsed = [float(s.replace(' ', '')) for s in signed]
                positives = [v for v in parsed if v >= 0]
                negatives = [v for v in parsed if v < 0]
                if positives:
                    upper_tol = positives[0]
                if negatives:
                    lower_tol = negatives[0]
                if upper_tol is None and lower_tol is not None:
                    upper_tol = 0.0
                if lower_tol is None and upper_tol is not None:
                    lower_tol = 0.0

    if nominal is not None and utl is not None and upper_tol is None:
        upper_tol = utl - nominal
    if nominal is not None and ltl is not None and lower_tol is None:
        lower_tol = ltl - nominal
    return nominal, upper_tol, lower_tol


def _dimension_limit_mode(sheet: ParsedSheet, header_row: int) -> tuple[str, Optional[str]]:
    labels = []
    for r in range(header_row, min(sheet.nrows, header_row + 4)):
        labels.append(_norm_text(sheet.cell(r, 3)))
        labels.append(_norm_text(sheet.cell(r, 4)))
    text = ' '.join(labels).lower()
    has_limit = any(token in text for token in _LIMIT_HEADER_TOKENS)
    has_deviation = any(token in text for token in _DEVIATION_HEADER_TOKENS)
    if has_limit:
        warning = None
        if has_deviation:
            warning = f'{sheet.name}: 上/下差列同时包含 UTL/LTL，已按绝对上限/下限处理'
        return 'limit', warning
    if has_deviation:
        return 'deviation', None
    return 'limit', f'{sheet.name}: 未能判断 D/E 列是极限值还是偏差，已按绝对上限/下限处理'


def _record_id(source_hash: str, sheet: str, row: int, col: int = 0) -> str:
    return f'{source_hash}:{sheet}:{row + 1}:{col + 1}'


def _sheet_by_name(sheets: list[ParsedSheet], name: str) -> Optional[ParsedSheet]:
    lower = name.lower()
    for sheet in sheets:
        if sheet.name.lower() == lower:
            return sheet
    return None


def _detect_source_type(sheets: list[ParsedSheet]) -> str:
    names = {s.name.lower() for s in sheets}
    if '03' in names:
        return 'dimension_report'
    for sheet in sheets:
        if _find_calypso_report_header(sheet) is not None:
            return 'calypso_report'
    if {'main', 'detail'} & names or 'baogaozhengli' in names:
        return 'measurement_result'
    return 'unknown'


def _extract_detail_meta(sheet: Optional[ParsedSheet]) -> dict:
    meta = {}
    if sheet is None:
        return meta
    key_map = {
        '文件开始号': 'file_start_no',
        'path:': 'calypso_network_path',
        '文件名称': 'calypso_file',
        'filename end:': 'filename_suffix',
        'extension:': 'extension',
        '件数': 'sample_count',
        '结果文件路径': 'calypso_result_path',
    }
    for r in range(sheet.nrows):
        key = _norm_text(sheet.cell(r, 0))
        if key in key_map:
            value = sheet.cell(r, 1)
            if key == '件数':
                value = _parse_float(value)
                if value is not None:
                    value = int(value)
            meta[key_map[key]] = value
    return meta


def _actuals_from_cells(sheet: ParsedSheet, row: int, cols: list[int]) -> list[dict]:
    actuals = []
    for i, col in enumerate(cols, start=1):
        value = _parse_float(sheet.cell(row, col))
        if value is not None:
            actuals.append({'sample': i, 'value': value})
    return actuals


def _dimension_report_sample_cols(sheet: ParsedSheet, header_row: int) -> list[int]:
    for r in range(header_row, min(sheet.nrows, header_row + 5)):
        cols = []
        expected = 1
        for c in range(5, min(sheet.ncols, 40)):
            if _is_sample_number(sheet.cell(r, c), expected):
                cols.append(c)
                expected += 1
                continue
            if cols:
                break
        if cols:
            return cols
    return list(range(5, min(sheet.ncols, 11)))


def _parse_dimension_report(sheet: ParsedSheet, source_file: str, source_hash: str) -> tuple[list[dict], list[str]]:
    warnings = []
    records = []
    header_row = None
    for r in range(min(sheet.nrows, 30)):
        row_text = ' '.join(_norm_text(sheet.cell(r, c)) for c in range(min(sheet.ncols, 14)))
        if ('NO' in row_text.upper() and ('检验项目' in row_text or 'ITEM' in row_text.upper())):
            header_row = r
            break
    if header_row is None:
        warnings.append(f'{sheet.name}: 未找到尺寸报告表头')
        return records, warnings

    limit_mode, mode_warning = _dimension_limit_mode(sheet, header_row)
    if mode_warning:
        warnings.append(mode_warning)

    sample_cols = _dimension_report_sample_cols(sheet, header_row)
    result_col = (sample_cols[-1] + 1) if sample_cols else 11
    equipment_col = result_col + 1
    current_no = None
    current_parent = None
    for r in range(header_row + 1, sheet.nrows):
        no_raw = sheet.cell(r, 0)
        item_text = _norm_text(sheet.cell(r, 2))
        upper_raw = _parse_float(sheet.cell(r, 3))
        lower_raw = _parse_float(sheet.cell(r, 4))
        actuals = _actuals_from_cells(sheet, r, sample_cols)
        equipment = _norm_text(sheet.cell(r, equipment_col))
        explicit_result = sheet.cell(r, result_col)

        # Template subheaders and empty rows.
        row_label = ' '.join(_norm_text(sheet.cell(r, c)) for c in range(0, min(13, sheet.ncols)))
        if not item_text and not actuals and upper_raw is None and lower_raw is None:
            continue
        if '名义值' in row_label and any(token in row_label for token in ('上差', '上限', 'UTL')):
            continue

        no_label = _parse_no_label(no_raw)
        continuation_of = None
        if no_label is not None:
            current_no = no_label
            current_parent = _record_id(source_hash, sheet.name, r)
        elif current_no is not None:
            continuation_of = current_parent
        if not item_text and current_no is None:
            continue

        if limit_mode == 'deviation':
            nominal, parsed_upper_tol, parsed_lower_tol = _parse_item_nominal_tol(item_text)
            upper_tol = upper_raw if upper_raw is not None else parsed_upper_tol
            lower_tol = lower_raw if lower_raw is not None else parsed_lower_tol
            utl = _add_limit(nominal, upper_tol)
            ltl = _add_limit(nominal, lower_tol)
        else:
            utl = upper_raw
            ltl = lower_raw
            nominal, upper_tol, lower_tol = _parse_item_nominal_tol(item_text, utl, ltl)
        rec = {
            'record_id': _record_id(source_hash, sheet.name, r),
            'source_file': source_file,
            'source_type': 'dimension_report',
            'source_sheet': sheet.name,
            'source_row': r + 1,
            'no': current_no,
            'is_key': bool(_norm_text(sheet.cell(r, 1)).upper() in {'Y', 'YES', '1', 'TRUE'}),
            'item_text': item_text,
            'nominal': nominal,
            'upper_tol': upper_tol,
            'lower_tol': lower_tol,
            'utl': utl,
            'ltl': ltl,
            'actuals': actuals,
            'result': _compute_result(actuals, utl, ltl, explicit_result),
            'raw_result': _normal_result(explicit_result) or _norm_text(explicit_result),
            'equipment': equipment,
            'raw_item_name': item_text,
            'continuation_of': continuation_of,
            'match_status': 'unmatched',
        }
        records.append(rec)
    return records, warnings


def _find_calypso_report_header(sheet: ParsedSheet) -> Optional[dict]:
    for r in range(min(sheet.nrows, 60)):
        labels = [_norm_text(sheet.cell(r, c)).lower() for c in range(min(sheet.ncols, 12))]
        if 'characteristic' not in labels or 'actual' not in labels:
            continue
        mapping = {}
        for c, label in enumerate(labels):
            if label == 'characteristic':
                mapping['characteristic'] = c
            elif label == 'actual':
                mapping['actual'] = c
            elif label == 'nominal':
                mapping['nominal'] = c
            elif label in {'upper tol', 'upper tolerance'}:
                mapping['upper_tol'] = c
            elif label in {'lower tol', 'lower tolerance'}:
                mapping['lower_tol'] = c
            elif label == 'deviation':
                mapping['deviation'] = c
        if {'characteristic', 'actual', 'nominal', 'upper_tol', 'lower_tol'} <= set(mapping):
            mapping['row'] = r
            return mapping
    return None


def _extract_calypso_report_meta(sheet: ParsedSheet) -> dict:
    meta = {}
    for r in range(min(sheet.nrows, 20)):
        for c in range(sheet.ncols):
            label = _norm_text(sheet.cell(r, c)).lower()
            if label == 'measurement plan':
                meta['measurement_plan'] = _norm_text(sheet.cell(r + 1, c))
            elif label == 'date':
                meta['date_raw'] = sheet.cell(r + 1, c)
            elif label == 'time':
                meta['time_raw'] = sheet.cell(r + 1, c)
            elif label == 'part no.':
                meta['part_no'] = _norm_text(sheet.cell(r + 1, c))
            elif label == 'operator':
                meta['operator'] = _norm_text(sheet.cell(r + 1, c))
            elif label == 'cmm':
                meta['cmm'] = _norm_text(sheet.cell(r + 1, c))
    return meta


def _parse_calypso_report(sheet: ParsedSheet, source_file: str, source_hash: str) -> tuple[list[dict], dict, list[str]]:
    warnings = []
    header = _find_calypso_report_header(sheet)
    if header is None:
        return [], {}, [f'{sheet.name}: 未找到 Calypso Report 表头']

    records = []
    for r in range(header['row'] + 1, sheet.nrows):
        raw_name = _norm_text(sheet.cell(r, header['characteristic']))
        actual = _parse_float(sheet.cell(r, header['actual']))
        nominal = _parse_float(sheet.cell(r, header['nominal']))
        upper_tol = _parse_float(sheet.cell(r, header['upper_tol']))
        lower_tol = _parse_float(sheet.cell(r, header['lower_tol']))
        if not raw_name and actual is None and nominal is None:
            continue
        if not raw_name:
            continue

        no, item_text, parsed_nominal, parsed_upper_tol, parsed_lower_tol = _parse_measurement_item_name(raw_name)
        if nominal is None:
            nominal = parsed_nominal
        if upper_tol is None:
            upper_tol = parsed_upper_tol
        if lower_tol is None:
            lower_tol = parsed_lower_tol
        utl = _add_limit(nominal, upper_tol)
        ltl = _add_limit(nominal, lower_tol)
        actuals = [{'sample': 1, 'value': actual}] if actual is not None else []
        records.append({
            'record_id': _record_id(source_hash, sheet.name, r),
            'source_file': source_file,
            'source_type': 'calypso_report',
            'source_sheet': sheet.name,
            'source_row': r + 1,
            'no': no,
            'is_key': False,
            'item_text': item_text,
            'nominal': nominal,
            'upper_tol': upper_tol,
            'lower_tol': lower_tol,
            'utl': utl,
            'ltl': ltl,
            'actuals': actuals,
            'result': _compute_result(actuals, utl, ltl),
            'raw_result': '',
            'equipment': 'CMM',
            'raw_item_name': raw_name,
            'continuation_of': None,
            'match_status': 'unmatched',
            'deviation': _parse_float(sheet.cell(r, header.get('deviation', -1))),
        })
    if not records:
        warnings.append('未从 Calypso Report 中识别出测量记录')
    return records, _extract_calypso_report_meta(sheet), warnings


def _sample_cols_after_header(sheet: ParsedSheet, row: int, col: int) -> list[int]:
    cols = []
    expected = 1
    for c in range(col + 1, min(sheet.ncols, col + 12)):
        if _is_sample_number(sheet.cell(row, c), expected):
            cols.append(c)
            expected += 1
            continue
        if cols:
            break
    return cols


def _find_measurement_blocks(sheet: ParsedSheet) -> list[tuple[int, int, list[int]]]:
    blocks = []
    for r in range(sheet.nrows):
        for c in range(sheet.ncols):
            sample_cols = _sample_cols_after_header(sheet, r, c)
            if len(sample_cols) >= 2:
                blocks.append((r, c, sample_cols))
    return blocks


def _parse_measurement_item_name(raw_name: str):
    no = None
    item = raw_name
    match = _NO_PREFIX_RE.match(raw_name)
    if match:
        no = match.group(1)
        item = match.group(2)
    nominal, upper_tol, lower_tol = _parse_item_nominal_tol(item)
    return no, item, nominal, upper_tol, lower_tol


def _parse_measurement_sheet(sheet: ParsedSheet, source_file: str, source_hash: str) -> tuple[list[dict], list[str]]:
    warnings = []
    records = []
    blocks = _find_measurement_blocks(sheet)
    if not blocks:
        return records, warnings

    header_positions = {(r, c) for r, c, _ in blocks}
    for header_row, item_col, sample_cols in blocks:
        blank_run = 0
        for r in range(header_row + 1, sheet.nrows):
            if (r, item_col) in header_positions:
                break
            raw_name = _norm_text(sheet.cell(r, item_col))
            actuals = _actuals_from_cells(sheet, r, sample_cols)
            if not raw_name and not actuals:
                blank_run += 1
                if blank_run >= 2:
                    break
                continue
            blank_run = 0
            if not raw_name:
                continue
            no, item_text, nominal, upper_tol, lower_tol = _parse_measurement_item_name(raw_name)
            records.append({
                'record_id': _record_id(source_hash, sheet.name, r, item_col),
                'source_file': source_file,
                'source_type': 'measurement_result',
                'source_sheet': sheet.name,
                'source_row': r + 1,
                'source_col': item_col + 1,
                'no': no,
                'is_key': False,
                'item_text': item_text,
                'nominal': nominal,
                'upper_tol': upper_tol,
                'lower_tol': lower_tol,
                'utl': None,
                'ltl': None,
                'actuals': actuals,
                'result': _compute_result(actuals, None, None),
                'raw_result': '',
                'equipment': 'CMM',
                'raw_item_name': raw_name,
                'continuation_of': None,
                'match_status': 'unmatched',
            })
    return records, warnings


def _parse_measurement_result(sheets: list[ParsedSheet], source_file: str, source_hash: str) -> tuple[list[dict], dict, list[str]]:
    warnings = []
    meta = _extract_detail_meta(_sheet_by_name(sheets, 'detail'))
    candidates = []
    for preferred in ('main', 'BAOGAOZHENGLI'):
        sheet = _sheet_by_name(sheets, preferred)
        if sheet is not None:
            candidates.append(sheet)
    if not candidates:
        candidates = sheets

    records = []
    for sheet in candidates:
        parsed, sheet_warnings = _parse_measurement_sheet(sheet, source_file, source_hash)
        warnings.extend(sheet_warnings)
        records.extend(parsed)
    if not records:
        warnings.append('未从测量结果 workbook 中识别出测量记录')
    return records, meta, warnings


def parse_inspection_sheets(
    sheets: list[ParsedSheet],
    source_file: str,
    source_hash: str,
) -> tuple[str, list[dict], dict, list[str]]:
    """Recognize and parse decoded sheets into inspection-domain records."""
    source_type = _detect_source_type(sheets)
    warnings = []
    meta = {}

    if source_type == 'dimension_report':
        sheet = _sheet_by_name(sheets, '03') or sheets[0]
        records, warnings = _parse_dimension_report(sheet, source_file, source_hash)
    elif source_type == 'calypso_report':
        report_sheet = next((s for s in sheets if _find_calypso_report_header(s) is not None), sheets[0])
        records, report_meta, warnings = _parse_calypso_report(report_sheet, source_file, source_hash)
        meta.update(report_meta)
    elif source_type == 'measurement_result':
        records, detail_meta, warnings = _parse_measurement_result(sheets, source_file, source_hash)
        meta.update(detail_meta)
    else:
        records = []
        warnings.append('无法识别 workbook 类型')

    return source_type, records, meta, warnings
