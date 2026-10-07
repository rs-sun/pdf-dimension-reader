"""Read ordinary glyphs from blockfont-compatible Siemens NX ``.fnx`` fonts.

The compiled FNX container can hold thousands of CJK entries.  This module
deliberately emits only printable ASCII glyphs: it is a small, deterministic
source of vector topology for numbers, Latin letters, and common drawing
symbols, not a Chinese-font converter.  Only the single-stroke LM/LD record
dialect used by blockfont and the Western subset of ``chinesef_fs`` is in
scope; unsupported FNX glyph dialects fail closed.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any


SCHEMA_VERSION = "nx_fnx_ordinary_glyph_profile_v1"
INDEX_OFFSET = 0x200
FIXED_POINT_SCALE = 1_000_000
ORDINARY_ASCII_CODES = frozenset(range(0x20, 0x7F))


class NxFnxFormatError(ValueError):
    """Raised when an FNX source cannot safely produce an ordinary profile."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def parse_nx_fnx_bytes(source: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Return a canonical, JSON-safe profile for printable ASCII glyphs.

    Non-ASCII index entries are never decoded into glyph topology.  This keeps
    the same API safe for ``blockfont.fnx`` and for Chinese FNX containers when
    only their embedded Western subset is useful.
    """
    data = bytes(source)
    if len(data) < 64:
        raise NxFnxFormatError("header_truncated")
    if data[:4] != b"FONT":
        raise NxFnxFormatError("magic_invalid")

    endian, byte_order = _detect_byte_order(data)
    declared_size = _u32(data, 8, endian=endian)

    metrics = list(struct.unpack_from(f"{endian}4I", data, 16))
    normalization_basis = metrics[0]
    if normalization_basis <= 0:
        raise NxFnxFormatError("normalization_basis_invalid")
    scale_parameters = list(struct.unpack_from(f"{endian}3I", data, 48))
    source_character_count = _u32(data, 60, endian=endian)
    table_end = INDEX_OFFSET + source_character_count * 8
    if table_end > len(data):
        raise NxFnxFormatError("character_index_truncated")

    index_rows: list[tuple[int, int]] = []
    record_ranges: list[tuple[int, int]] = []
    seen_codes: set[int] = set()
    seen_offsets: set[int] = set()
    for index in range(source_character_count):
        code, offset = struct.unpack_from(
            f"{endian}2I",
            data,
            INDEX_OFFSET + index * 8,
        )
        if code in seen_codes:
            raise NxFnxFormatError("character_code_duplicate")
        seen_codes.add(code)
        _validate_glyph_offset(data, offset=offset, table_end=table_end)
        if offset in seen_offsets:
            raise NxFnxFormatError("glyph_offset_duplicate")
        seen_offsets.add(offset)
        instruction_count = _u32(data, offset, endian=endian)
        record_end = offset + 4 + instruction_count * 4
        if record_end > len(data):
            raise NxFnxFormatError("glyph_record_truncated")
        index_rows.append((code, offset))
        record_ranges.append((offset, record_end))

    previous_end = table_end
    for offset, record_end in sorted(record_ranges):
        if offset < previous_end:
            raise NxFnxFormatError("glyph_record_overlap")
        previous_end = record_end

    glyphs: list[dict[str, Any]] = []
    for code, offset in index_rows:
        if code not in ORDINARY_ASCII_CODES:
            continue
        glyphs.append(
            _decode_glyph(
                data,
                code=code,
                offset=offset,
                table_end=table_end,
                normalization_basis=normalization_basis,
                endian=endian,
            )
        )

    glyphs.sort(key=lambda row: row["code"])
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "format": "siemens_nx_fnx",
        "glyph_record_dialect": "blockfont_compatible_single_stroke_lm_ld",
        "byte_order": byte_order,
        "declared_size_bytes": declared_size,
        "header_metrics_units": metrics,
        "scale_parameters": scale_parameters,
        "normalization_basis_units": normalization_basis,
        "source_character_count": source_character_count,
        "selected_character_count": len(glyphs),
        "ordinary_character_scope": {
            "minimum_code": min(ORDINARY_ASCII_CODES),
            "maximum_code": max(ORDINARY_ASCII_CODES),
            "non_ascii_topology_decoded": False,
            "chinese_expansion_allowed": False,
        },
        "glyphs": glyphs,
    }
    profile_semantic = {
        key: value
        for key, value in semantic.items()
        if key not in {
            "byte_order",
            "declared_size_bytes",
            "source_character_count",
        }
    }
    return {
        **semantic,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "profile_sha256": hashlib.sha256(
            _canonical_json(profile_semantic)
        ).hexdigest(),
    }


def _decode_glyph(
    data: bytes,
    *,
    code: int,
    offset: int,
    table_end: int,
    normalization_basis: int,
    endian: str,
) -> dict[str, Any]:
    _validate_glyph_offset(data, offset=offset, table_end=table_end)
    instruction_count = _u32(data, offset, endian=endian)
    record_end = offset + 4 + instruction_count * 4
    if record_end > len(data):
        raise NxFnxFormatError("glyph_record_truncated")

    strokes: list[list[list[int]]] = []
    operations: list[str] = []
    for instruction_index in range(instruction_count):
        packed = _u32(
            data,
            offset + 4 + instruction_index * 4,
            endian=endian,
        )
        opcode = packed >> 28
        if opcode not in (0, 1):
            raise NxFnxFormatError("glyph_opcode_unsupported")
        x = _signed14(packed & 0x3FFF)
        y = _signed14((packed >> 14) & 0x3FFF)
        if opcode == 1:
            operations.append("M")
            strokes.append([[x, y]])
        else:
            if not strokes:
                raise NxFnxFormatError("glyph_draw_before_move")
            operations.append("L")
            strokes[-1].append([x, y])

    points = [point for stroke in strokes for point in stroke]
    bounds = (
        [
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        ]
        if points
        else None
    )
    strokes_uH = [
        [
            [
                _to_fixed(point[0], normalization_basis),
                _to_fixed(point[1], normalization_basis),
            ]
            for point in stroke
        ]
        for stroke in strokes
    ]
    return {
        "code": code,
        "character": chr(code),
        "instruction_count": instruction_count,
        "operations": "".join(operations),
        "stroke_count": len(strokes),
        "segment_count": sum(max(0, len(stroke) - 1) for stroke in strokes),
        "closed_stroke_count": sum(
            len(stroke) >= 3 and stroke[0] == stroke[-1]
            for stroke in strokes
        ),
        "bounds_units": bounds,
        "bounds_uH": (
            [_to_fixed(value, normalization_basis) for value in bounds]
            if bounds is not None
            else None
        ),
        "strokes_units": strokes,
        "strokes_uH": strokes_uH,
    }


def _validate_glyph_offset(data: bytes, *, offset: int, table_end: int) -> None:
    if offset < table_end or offset % 4 != 0 or offset + 4 > len(data):
        raise NxFnxFormatError("glyph_offset_invalid")


def _signed14(value: int) -> int:
    return value - 0x4000 if value & 0x2000 else value


def _to_fixed(value: int, basis: int) -> int:
    magnitude = (abs(value) * FIXED_POINT_SCALE + basis // 2) // basis
    return -magnitude if value < 0 else magnitude


def _u32(data: bytes, offset: int, *, endian: str) -> int:
    return struct.unpack_from(f"{endian}I", data, offset)[0]


def _detect_byte_order(data: bytes) -> tuple[str, str]:
    little_matches = struct.unpack_from("<I", data, 8)[0] == len(data)
    big_matches = struct.unpack_from(">I", data, 8)[0] == len(data)
    if little_matches == big_matches:
        reason = (
            "byte_order_ambiguous"
            if little_matches
            else "declared_size_mismatch"
        )
        raise NxFnxFormatError(reason)
    return ("<", "little") if little_matches else (">", "big")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = (
    "NxFnxFormatError",
    "ORDINARY_ASCII_CODES",
    "SCHEMA_VERSION",
    "parse_nx_fnx_bytes",
)
