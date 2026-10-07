"""Extract numbered technical requirement notes from PDF/OCR text."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any


_ANCHOR_RE = re.compile(
    r"(技术\s*要求|technical\s+requirements|(?:general\s+)?notes?\s*[:：]?)",
    re.IGNORECASE,
)
_TITLE_STOP_RE = re.compile(
    r"(材料|图号|比例|标题栏|更改|设计|审核|批准|"
    r"\bmat(?:erial)?\.?\b|\bdrawing\s+no\.?\b|\bscale\b|\bpart\s+no\.?\b)",
    re.IGNORECASE,
)
_ITEM_PATTERNS = (
    re.compile(r"^\s*([1-9]\d{0,2})\s*(?:[、。．)）]|\.(?!\d))\s*"),
    re.compile(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s*"),
    re.compile(r"^\s*\(([1-9]\d{0,2})\)\s*"),
    re.compile(r"^\s*([A-Z])\s*[.)）]\s+"),
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def extract_pdf_text_words(fitz_page) -> list[dict[str, Any]]:
    """Extract native PDF word boxes in top-left PDF coordinates."""
    words: list[dict[str, Any]] = []
    try:
        for idx, word in enumerate(fitz_page.get_text("words")):
            x0, y0, x1, y1, text, *_ = word
            words.append({
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": str(text),
                "word_index": idx,
            })
    except Exception:
        return []
    return words


def extract_notes(
    fitz_page,
    pdf_words=None,
    ocr_results=None,
    pdf_bytes: bytes | None = None,
    page_index: int = 0,
    dpi: int = 120,
) -> list[dict]:
    words = [_normalize_word(word) for word in (pdf_words if pdf_words is not None else extract_pdf_text_words(fitz_page))]
    words = [word for word in words if word]
    if words:
        notes = _extract_notes_from_words(fitz_page, words, source="pdf_text")
        if notes:
            return notes

    if pdf_bytes:
        note_ocr_words = _extract_note_ocr_words(pdf_bytes, page_index, fitz_page, dpi=dpi)
        if note_ocr_words:
            notes = _extract_notes_from_words(fitz_page, note_ocr_words, source="ocr")
            if notes:
                return notes

    ocr_words = [_normalize_ocr_result(item) for item in (ocr_results or [])]
    ocr_words = [word for word in ocr_words if word]
    if ocr_words:
        return _extract_notes_from_words(fitz_page, ocr_words, source="ocr")
    return []


def _extract_notes_from_words(fitz_page, words: list[dict[str, Any]], source: str) -> list[dict]:
    page_width, page_height = _page_size(fitz_page, words)
    median_word_height = _median([w["y1"] - w["y0"] for w in words], default=10.0)
    lines = _cluster_lines(words, y_tolerance=max(2.0, median_word_height * 0.5))
    if not lines:
        return []

    anchor_idx = _find_anchor_line(lines, page_width)
    if anchor_idx is None:
        return []

    median_line_height = _median([line["y1"] - line["y0"] for line in lines], default=median_word_height)
    gap_limit = max(8.0, median_line_height * 2.5)
    candidate_lines = _collect_note_lines(lines, anchor_idx, page_width, page_height, gap_limit)
    return _split_items(candidate_lines, source=source)


def _page_size(fitz_page, words: list[dict[str, Any]]) -> tuple[float, float]:
    rect = getattr(fitz_page, "rect", None)
    if rect is not None:
        try:
            return float(rect.width), float(rect.height)
        except Exception:
            pass
    return (
        max((w["x1"] for w in words), default=0.0),
        max((w["y1"] for w in words), default=0.0),
    )


def _normalize_word(raw: Any) -> dict[str, Any] | None:
    try:
        if isinstance(raw, dict):
            text = str(raw.get("text") or "").strip()
            x0 = float(raw.get("x0"))
            y0 = float(raw.get("y0"))
            x1 = float(raw.get("x1"))
            y1 = float(raw.get("y1"))
        else:
            x0, y0, x1, y1, text, *_ = raw
            text = str(text).strip()
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    except Exception:
        return None
    if not text or not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text}


def _normalize_ocr_result(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    bbox = raw.get("bbox_in_pdf") or raw.get("bbox") or raw.get("bbox_pt")
    try:
        if isinstance(bbox, dict):
            x0 = float(bbox.get("x"))
            y0 = float(bbox.get("y"))
            if "w" in bbox and "h" in bbox:
                x1 = x0 + float(bbox.get("w"))
                y1 = y0 + float(bbox.get("h"))
            else:
                x1 = float(bbox.get("x1"))
                y1 = float(bbox.get("y1"))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        else:
            return None
    except Exception:
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text}


def _median(values: list[float], default: float) -> float:
    clean = [float(v) for v in values if v and math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else default


def _cluster_lines(words: list[dict[str, Any]], y_tolerance: float) -> list[dict[str, Any]]:
    rows: list[list[dict[str, Any]]] = []
    centers: list[float] = []
    for word in sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2.0, w["x0"])):
        cy = (word["y0"] + word["y1"]) / 2.0
        if rows and abs(cy - centers[-1]) <= y_tolerance:
            rows[-1].append(word)
            centers[-1] = sum((w["y0"] + w["y1"]) / 2.0 for w in rows[-1]) / len(rows[-1])
        else:
            rows.append([word])
            centers.append(cy)

    lines = []
    for row in rows:
        ordered = sorted(row, key=lambda w: w["x0"])
        for segment in _split_row_segments(ordered):
            lines.append({
                "text": _join_words(segment),
                "x0": min(w["x0"] for w in segment),
                "y0": min(w["y0"] for w in segment),
                "x1": max(w["x1"] for w in segment),
                "y1": max(w["y1"] for w in segment),
            })
    return lines


def _split_row_segments(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not words:
        return []
    median_height = _median([w["y1"] - w["y0"] for w in words], default=10.0)
    gap_limit = max(45.0, median_height * 6.0)
    segments: list[list[dict[str, Any]]] = [[words[0]]]
    for word in words[1:]:
        gap = word["x0"] - segments[-1][-1]["x1"]
        if gap > gap_limit:
            segments.append([word])
        else:
            segments[-1].append(word)
    return segments


def _join_words(words: list[dict[str, Any]]) -> str:
    out = ""
    prev = ""
    for word in words:
        text = word["text"]
        if not out:
            out = text
        elif _no_space_between(prev, text):
            out += text
        else:
            out += " " + text
        prev = text
    return out.strip()


def _no_space_between(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if right[0] in "，。；：、,.;:)）]】":
        return True
    if left[-1] == "、" and _CJK_RE.search(right[0]):
        return True
    return bool(_CJK_RE.search(left[-1]) and _CJK_RE.search(right[0]))


def _find_anchor_line(lines: list[dict[str, Any]], page_width: float) -> int | None:
    hits = []
    for idx, line in enumerate(lines):
        if not _ANCHOR_RE.search(line["text"]):
            continue
        cx = (line["x0"] + line["x1"]) / 2.0
        is_chinese_anchor = bool(re.search(r"技术\s*要求", line["text"]))
        is_right_half = cx > page_width * 0.5
        hits.append((not is_chinese_anchor, not is_right_half, line["y0"], idx))
    if not hits:
        return None
    hits.sort()
    return hits[0][3]


def _collect_note_lines(
    lines: list[dict[str, Any]],
    anchor_idx: int,
    page_width: float,
    page_height: float,
    gap_limit: float,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    anchor = lines[anchor_idx]
    anchor_right_half = (anchor["x0"] + anchor["x1"]) / 2.0 > page_width * 0.5
    prev_bottom = lines[anchor_idx]["y1"]
    for idx in range(anchor_idx, len(lines)):
        line = dict(lines[idx])
        if idx != anchor_idx:
            line_cx = (line["x0"] + line["x1"]) / 2.0
            if anchor_right_half and line_cx <= page_width * 0.5:
                continue
            if not anchor_right_half and line_cx > page_width * 0.5:
                continue
        text = _strip_anchor_prefix(line["text"]) if idx == anchor_idx else line["text"].strip()
        if not text:
            continue
        if line["y0"] > page_height * 0.95:
            break
        if idx != anchor_idx and collected and line["y0"] - prev_bottom > gap_limit:
            break
        if idx != anchor_idx and _TITLE_STOP_RE.search(text) and not _is_item_start(text):
            break
        line["text"] = text
        collected.append(line)
        prev_bottom = line["y1"]
    return collected


def _strip_anchor_prefix(text: str) -> str:
    return _ANCHOR_RE.sub("", text, count=1).strip(" :-：")


def _is_item_start(text: str) -> bool:
    return any(pattern.match(text) for pattern in _ITEM_PATTERNS)


def _split_items(lines: list[dict[str, Any]], source: str) -> list[dict]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    preamble: dict[str, Any] | None = None
    seq = 0
    for line in lines:
        text = line["text"].strip()
        if not text:
            continue
        if _is_item_start(text):
            if current:
                items.append(current)
            if preamble:
                items.append(preamble)
                preamble = None
            seq += 1
            current = {
                "seq": seq,
                "lines": [text],
                "bbox_first_line": _bbox(line),
                "bbox_all": _bbox(line),
            }
            continue
        if current is None:
            if preamble is None:
                preamble = {
                    "seq": 0,
                    "lines": [text],
                    "bbox_first_line": _bbox(line),
                    "bbox_all": _bbox(line),
                }
            else:
                preamble["lines"].append(text)
                preamble["bbox_all"] = _union_bbox(preamble["bbox_all"], _bbox(line))
            continue
        current["lines"].append(text)
        current["bbox_all"] = _union_bbox(current["bbox_all"], _bbox(line))
    if current:
        items.append(current)
    elif preamble:
        items.append(preamble)

    return [
        {
            "seq": item["seq"],
            "text": "\n".join(item["lines"]),
            "bbox_first_line": item["bbox_first_line"],
            "bbox_all": item["bbox_all"],
            "source": source,
        }
        for item in items
    ]


def _bbox(line: dict[str, Any]) -> list[float]:
    return [
        round(float(line["x0"]), 2),
        round(float(line["y0"]), 2),
        round(float(line["x1"]), 2),
        round(float(line["y1"]), 2),
    ]


def _union_bbox(a: list[float], b: list[float]) -> list[float]:
    return [
        round(min(a[0], b[0]), 2),
        round(min(a[1], b[1]), 2),
        round(max(a[2], b[2]), 2),
        round(max(a[3], b[3]), 2),
    ]
