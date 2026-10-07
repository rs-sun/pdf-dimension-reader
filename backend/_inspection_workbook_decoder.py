"""Physical Excel decoding adapters for inspection imports.

This module stops at a format-neutral list of :class:`ParsedSheet` objects.
Report recognition and measurement semantics belong to
``_inspection_report_parsers``.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Any
from zipfile import BadZipFile

import openpyxl
from openpyxl.utils.exceptions import InvalidFileException


@dataclass
class ParsedSheet:
    name: str
    nrows: int
    ncols: int
    cell: Any


def _sheet_values_openpyxl(workbook) -> list[ParsedSheet]:
    sheets = []
    for worksheet in workbook.worksheets:
        values = [list(row) for row in worksheet.iter_rows(values_only=True)]
        nrows = len(values)
        ncols = max((len(row) for row in values), default=0)

        def cell(row, col, rows=values):
            if row < 0 or col < 0 or row >= len(rows) or col >= len(rows[row]):
                return None
            return rows[row][col]

        sheets.append(ParsedSheet(worksheet.title, nrows, ncols, cell))
    return sheets


def _sheet_values_calamine(filename: str, data: bytes) -> list[ParsedSheet]:
    try:
        from python_calamine import CalamineError, CalamineWorkbook
    except ImportError as exc:
        raise RuntimeError('解析 .xls 需要安装 python-calamine==0.8.2') from exc

    # A ZIP workbook must retain its .xlsx/.xlsm extension and openpyxl path.
    if data.startswith(b'PK'):
        raise ValueError(f'{filename}: .xls 文件必须是二进制 Excel 工作簿，请使用正确扩展名')

    sheets = []
    try:
        with CalamineWorkbook.from_filelike(io.BytesIO(data)) as workbook:
            for name in workbook.sheet_names:
                worksheet = workbook.get_sheet_by_name(name)
                # Coordinates must remain absolute: report headers can start
                # after leading empty rows or columns.
                values = worksheet.to_python(skip_empty_area=False)
                for row in values:
                    for col, value in enumerate(row):
                        # Calamine converts formatted date serials into typed
                        # values. Reject them rather than guess a serial epoch
                        # or let report numeric parsing read part of a date.
                        if isinstance(value, (date, time, timedelta)):
                            raise ValueError(
                                f'{filename}: .xls 包含日期、时间或时长单元格，'
                                '请另存为 .xlsx 后导入'
                            )
                        if isinstance(value, bool):
                            row[col] = int(value)
                nrows = len(values)
                ncols = max((len(row) for row in values), default=0)

                def cell(row, col, rows=values):
                    if row < 0 or col < 0 or row >= len(rows) or col >= len(rows[row]):
                        return None
                    return rows[row][col]

                sheets.append(ParsedSheet(name, nrows, ncols, cell))
    except CalamineError as exc:
        raise _workbook_parse_value_error(filename, exc) from exc
    return sheets


def _workbook_parse_value_error(filename: str, exc: Exception) -> ValueError:
    return ValueError(f'{filename}: 无法解析测量结果工作簿 ({exc})')


def decode_workbook_sheets(filename: str, data: bytes) -> list[ParsedSheet]:
    """Decode supported workbook bytes without applying report semantics."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == '.xls':
        return _sheet_values_calamine(filename, data)
    if ext in {'.xlsx', '.xlsm'}:
        try:
            workbook = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
        except (BadZipFile, InvalidFileException) as exc:
            raise _workbook_parse_value_error(filename, exc) from exc
        return _sheet_values_openpyxl(workbook)
    raise ValueError(f'不支持的测量文件类型: {ext or filename}')


def validate_sheet_bounds(
    sheets: list[ParsedSheet],
    filename: str,
    *,
    max_workbook_sheets: int,
    max_sheet_rows: int,
    max_sheet_cols: int,
) -> None:
    """Reject decoded sheets that exceed the import resource budget."""
    if len(sheets) > max_workbook_sheets:
        raise ValueError(f'{filename}: sheet 数量过多 ({len(sheets)} > {max_workbook_sheets})')
    for sheet in sheets:
        if sheet.nrows > max_sheet_rows:
            raise ValueError(f'{filename}/{sheet.name}: 行数过多 ({sheet.nrows} > {max_sheet_rows})')
        if sheet.ncols > max_sheet_cols:
            raise ValueError(f'{filename}/{sheet.name}: 列数过多 ({sheet.ncols} > {max_sheet_cols})')
