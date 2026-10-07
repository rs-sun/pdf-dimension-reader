"""Parse M3 measurement/result Excel files into clean JSON.

This public facade coordinates two internal seams:

* _inspection_workbook_decoder turns xls/xlsx/xlsm bytes into format-neutral
  sheets and enforces workbook resource limits.
* _inspection_report_parsers recognizes supported report layouts and produces
  inspection-domain records.

The parser deliberately returns rows without matching them to stamps. Matching
is a frontend/product-layer concern because users can edit stamp ids and
positions.
"""

from __future__ import annotations

import hashlib
import os

if __package__:
    from ._inspection_report_parsers import (
        _parse_item_nominal_tol,
        parse_inspection_sheets,
    )
    from ._inspection_workbook_decoder import (
        ParsedSheet,
        decode_workbook_sheets,
        validate_sheet_bounds,
    )
else:
    from _inspection_report_parsers import (
        _parse_item_nominal_tol,
        parse_inspection_sheets,
    )
    from _inspection_workbook_decoder import (
        ParsedSheet,
        decode_workbook_sheets,
        validate_sheet_bounds,
    )


_MAX_SHEET_ROWS = 50_000
_MAX_SHEET_COLS = 256
_MAX_WORKBOOK_SHEETS = 100


def _source_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _load_sheets(filename: str, data: bytes) -> list[ParsedSheet]:
    """Compatibility wrapper around the physical workbook decoder."""
    return decode_workbook_sheets(filename, data)


def _validate_sheet_bounds(sheets: list[ParsedSheet], filename: str) -> None:
    """Apply the facade's configurable workbook resource limits."""
    validate_sheet_bounds(
        sheets,
        filename,
        max_workbook_sheets=_MAX_WORKBOOK_SHEETS,
        max_sheet_rows=_MAX_SHEET_ROWS,
        max_sheet_cols=_MAX_SHEET_COLS,
    )


def parse_workbook_bytes(data: bytes, filename: str) -> dict:
    """Parse one uploaded Excel workbook into the M3 inspection JSON shape."""
    source_file = os.path.basename(filename) or 'measurement.xlsx'
    source_hash = _source_hash(data)
    sheets = _load_sheets(source_file, data)
    _validate_sheet_bounds(sheets, source_file)
    source_type, records, report_meta, warnings = parse_inspection_sheets(
        sheets,
        source_file,
        source_hash,
    )
    meta = {'source_hash': source_hash}
    meta.update(report_meta)

    return {
        'source_file': source_file,
        'source_type': source_type,
        'sheets': [sheet.name for sheet in sheets],
        'records': records,
        'meta': meta,
        'warnings': warnings,
        'stats': {
            'record_count': len(records),
            'nok_count': sum(1 for record in records if record.get('result') == 'NOK'),
            'unmeasured_count': sum(
                1 for record in records if record.get('result') == 'UNMEASURED'
            ),
        },
    }


def parse_uploaded_workbooks(files: list[tuple[str, bytes]]) -> dict:
    """Parse and aggregate multiple uploaded inspection workbooks."""
    workbooks = []
    records = []
    warnings = []
    for filename, data in files:
        parsed = parse_workbook_bytes(data, filename)
        workbooks.append({
            key: parsed[key]
            for key in (
                'source_file',
                'source_type',
                'sheets',
                'meta',
                'stats',
                'warnings',
            )
        })
        records.extend(parsed['records'])
        for warning in parsed.get('warnings', []):
            warnings.append(f'{parsed["source_file"]}: {warning}')
    return {
        'status': 'success',
        'workbooks': workbooks,
        'records': records,
        'warnings': warnings,
        'stats': {
            'file_count': len(workbooks),
            'record_count': len(records),
            'nok_count': sum(1 for record in records if record.get('result') == 'NOK'),
            'unmeasured_count': sum(
                1 for record in records if record.get('result') == 'UNMEASURED'
            ),
        },
    }
