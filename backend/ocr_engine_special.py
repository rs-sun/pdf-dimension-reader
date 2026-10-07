"""ocr_engine_special.py — 跑道框 OCR + 尺寸线条带 OCR + 公差微框 OCR

从 ocr_engine.py 拆出。"""

import re
import time
import math
import numpy as np
import fitz
from PIL import Image

from ocr_engine_utils import (
    VERBOSE, _get_ocr, render_page_to_image,
    pdf_bbox_to_pixel, pixel_bbox_to_pdf, quad_to_bbox,
    _bbox_iou, rotate_bbox_back, _rotation_to_horizontal,
    attach_r2p_digit_restricted_ocr_shadow,
)


def _font_height_proxy(bbox_pdf: dict) -> float:
    """Use the shorter side as font-height proxy after angled AABB expansion."""
    try:
        w = float(bbox_pdf.get('w', 0.0))
        h = float(bbox_pdf.get('h', 0.0))
    except (TypeError, ValueError):
        return 0.0
    if w > 0 and h > 0:
        return min(w, h)
    return max(w, h)


def _strip_text_strength(text: str, *, orient: str, is_leader: bool) -> int:
    """Rank strip OCR text so weak fragments do not stop angled retries early."""
    value = re.sub(r'\s+', '', str(text or '').strip())
    if not value or not re.search(r'\d', value):
        return -10

    score = 0
    upper = value.upper()
    if re.match(r'^[RrCc⌀∅ØφΦ]\d', value):
        score += 6
    if any(sym in value for sym in ('±', '+', '-')):
        score += 4
    if '°' in value:
        score += 4
    if 'MAX' in upper or 'MIN' in upper:
        score += 3
    if re.search(r'\d+\.\d+', value):
        score += 2
    if re.match(r'^\d{1,2}(?:\.\d+)?$', value):
        score -= 3 if is_leader else 1
    if orient == 'D' and is_leader and not re.search(r'^[RrCc⌀∅ØφΦ]|\+|-|±|°|MAX|MIN', value):
        score -= 2
    return score


def _strong_strip_text_found(results: list, *, orient: str, is_leader: bool) -> bool:
    threshold = 5 if is_leader or orient == 'D' else 3
    return any(
        _strip_text_strength(r.get('text', ''), orient=orient, is_leader=is_leader) >= threshold
        for r in results
    )


def _bbox_union(a: dict, b: dict) -> dict:
    x0 = min(float(a.get('x', 0.0)), float(b.get('x', 0.0)))
    y0 = min(float(a.get('y', 0.0)), float(b.get('y', 0.0)))
    x1 = max(float(a.get('x', 0.0)) + float(a.get('w', 0.0)),
             float(b.get('x', 0.0)) + float(b.get('w', 0.0)))
    y1 = max(float(a.get('y', 0.0)) + float(a.get('h', 0.0)),
             float(b.get('y', 0.0)) + float(b.get('h', 0.0)))
    return {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}


def _normalize_degree_glyphs(text: str) -> str:
    value = re.sub(r'\s+', '', str(text or '').strip())
    value = value.replace('º', '°').replace('。', '°')
    value = re.sub(r'(?<=\d)[oO](?=$|[^\w])', '°', value)
    return value


def _extract_angle_main_text(text: str) -> str | None:
    """Return a normalized angle token with an optional quantity prefix."""
    value = _normalize_degree_glyphs(text)
    if not value or '°' not in value:
        return None
    if value.startswith(('+', '-', '±')) or re.search(r'\d°\d', value):
        return None
    match = re.search(r'(?:(?P<count>\d{1,2})[xX×])?(?P<angle>\d{1,3}(?:\.\d+)?)°', value)
    if not match:
        return None
    try:
        angle_val = float(match.group('angle'))
    except ValueError:
        return None
    if not (0.0 < angle_val <= 180.0):
        return None
    prefix = ''
    if match.group('count'):
        prefix = f"{match.group('count')}X"
    angle_text = match.group('angle').rstrip('.')
    return f'{prefix}{angle_text}°'


def _angle_tolerance_suffix_from_lines(lines: list[dict]) -> tuple[str | None, dict | None]:
    """Find a tiny ±N° tolerance fragment from OCR attempts around an angle."""
    saw_sign = False
    numeric_candidates: list[tuple[float, float, dict | None]] = []
    for line in lines:
        text = _normalize_degree_glyphs(line.get('text', ''))
        if not text:
            continue
        main_text = _extract_angle_main_text(text)
        if main_text:
            main_match = re.search(r'(\d{1,3}(?:\.\d+)?)°', main_text)
            try:
                main_value = float(main_match.group(1)) if main_match else 999.0
            except ValueError:
                main_value = 999.0
            if main_value > 5.0:
                continue
        if any(ch in text for ch in ('±', '+')):
            saw_sign = True
        match = re.search(r'(?:[±+])?(\d(?:\.\d+)?)°?$', text)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 0.0 < value <= 5.0:
            numeric_candidates.append((float(line.get('confidence') or 0.0), value,
                                       line.get('bbox_in_pdf')))
    if not numeric_candidates:
        return None, None
    numeric_candidates.sort(key=lambda item: (-item[0], item[1]))
    conf, value, bbox = numeric_candidates[0]
    if not saw_sign and conf < 0.85:
        return None, None
    value_text = str(int(value)) if abs(value - int(value)) < 1e-6 else str(value).rstrip('0').rstrip('.')
    return f'±{value_text}°', bbox


def _angle_suffix_near_main(main_bbox: dict, suffix_bbox: dict | None) -> bool:
    if not suffix_bbox:
        return False
    mcx = float(main_bbox.get('x', 0.0)) + float(main_bbox.get('w', 0.0)) / 2.0
    mcy = float(main_bbox.get('y', 0.0)) + float(main_bbox.get('h', 0.0)) / 2.0
    scx = float(suffix_bbox.get('x', 0.0)) + float(suffix_bbox.get('w', 0.0)) / 2.0
    scy = float(suffix_bbox.get('y', 0.0)) + float(suffix_bbox.get('h', 0.0)) / 2.0
    dx = scx - mcx
    dy = abs(scy - mcy)
    h = max(float(main_bbox.get('h', 0.0) or 0.0),
            float(suffix_bbox.get('h', 0.0) or 0.0),
            1.0)
    return dx > 0.0 and dx <= max(45.0, h * 2.8) and dy <= max(14.0, h * 0.75)


def _score_angle_label_region(region: dict) -> float | None:
    try:
        x = float(region.get('x') or 0.0)
        y = float(region.get('y') or 0.0)
        w = float(region.get('w') or 0.0)
        h = float(region.get('h') or 0.0)
    except (TypeError, ValueError):
        return None
    area = w * h
    if x < 45.0 or y < 75.0:
        return None
    if w < 5.0 or h < 5.0 or w > 75.0 or h > 36.0:
        return None
    if area < 55.0 or area > 1400.0:
        return None

    curves = float(region.get('curve_count') or 0.0)
    ratio = max(w, h) / max(min(w, h), 1.0)
    score = 0.0
    if 12.0 <= w <= 38.0 and 10.0 <= h <= 25.0:
        score += 42.0
    if 80.0 <= area <= 800.0:
        score += 28.0
    if 2.0 <= curves <= 10.0:
        score += 18.0
    if 1.0 <= ratio <= 5.5:
        score += 10.0
    # Two digits plus a degree glyph are often a near-square compact cluster.
    if 14.0 <= w <= 26.0 and 14.0 <= h <= 22.0 and 2.0 <= curves <= 4.0:
        score += 18.0
    if str(region.get('confidence') or '').lower() == 'low':
        score += 4.0
    if str(region.get('source') or '') == 'track_b':
        score -= 20.0
    if w > 55.0:
        score -= 12.0
    return score


def _select_angle_label_regions(text_regions: list, max_regions: int) -> list[tuple[int, dict, float]]:
    scored = []
    for idx, region in enumerate(text_regions or []):
        score = _score_angle_label_region(region)
        if score is None or score <= 0.0:
            continue
        scored.append((score, idx, region))
    scored.sort(key=lambda item: (
        -item[0],
        float(item[2].get('y') or 0.0),
        float(item[2].get('x') or 0.0),
    ))
    if max_regions and max_regions > 0:
        scored = scored[:max_regions]
    return [(idx, region, score) for score, idx, region in scored]


_RADIUS_REJECT_WORDS_RE = re.compile(
    r'(?:CIRCLE|BOTTOM|DETAIL|SECTION|SCALE|VIEW|RIBS?|MIN|MAX)',
    re.IGNORECASE,
)


def _normalize_radius_glyphs(text: str) -> str:
    value = re.sub(r'\s+', '', str(text or '').strip())
    if not value:
        return ''
    value = value.replace('×', 'X').replace('x', 'X')
    value = value.replace('：', ':').replace('﹕', ':')
    value = value.replace('＋', '+').replace('﹢', '+')
    value = value.replace('－', '-').replace('﹣', '-')
    value = value.replace('+/-', '±').replace('+-', '±')
    value = value.replace('土', '±').replace('士', '±')
    value = re.sub(r'(?<=\d)[,，](?=\d)', '.', value)
    value = re.sub(r'(?i)(R\d{1,3}):(?=\d(?:±|\+|-|$))', r'\1.', value)
    # Repair a restricted terminal glyph confusion in a decimal fraction.
    value = re.sub(r'(?<=\d\.\d)[FfSs](?=$|[^A-Za-z0-9])', '5', value)
    # Normalize an O/zero confusion after a radius prefix.
    value = re.sub(r'(?i)R[oO](?=[.,]?\d)', 'R0', value)
    return value


def _clean_decimal_token(token: str) -> str | None:
    value = str(token or '').replace(',', '.').strip()
    value = value.rstrip('.')
    if not value:
        return None
    if value.startswith('.'):
        value = '0' + value
    if not re.fullmatch(r'\d{1,3}(?:\.\d{1,4})?', value):
        return None
    return value


def _extract_radius_label_text(text: str) -> str | None:
    """Return a normalized radius label with optional quantity and tolerance."""
    value = _normalize_radius_glyphs(text)
    if not value or len(value) > 48:
        return None
    if _RADIUS_REJECT_WORDS_RE.search(value):
        return None

    match = re.search(
        r'(?<![A-Za-z0-9])(?:(?P<count>[1-9]\d?)X)?'
        r'R(?P<nom>(?:\d{1,3}(?:\.\d{1,4})?|\.\d{1,4}))'
        r'(?P<tail>[^A-Za-z]{0,18})',
        value,
        re.IGNORECASE,
    )
    if not match:
        return None

    nominal = _clean_decimal_token(match.group('nom'))
    if not nominal:
        return None
    try:
        nominal_val = float(nominal)
    except ValueError:
        return None
    if not (0.0 < nominal_val <= 500.0):
        return None

    prefix = f"{match.group('count')}X" if match.group('count') else ''
    text_out = f'{prefix}R{nominal}'

    tail = match.group('tail') or ''
    # Treat colon/plus as OCR variants of ± only when there is not an explicit
    # asymmetric minus value in the same tiny tail.
    if '-' not in tail:
        tol_match = re.search(
            r'(?:±(?P<pm>(?:\d{1,2}(?:\.\d{1,4})?|\.\d{1,4}))|'
            r'[:+](?P<sep>(?:\d{1,2}\.\d{1,4}|\.\d{1,4})))',
            tail,
        )
        if tol_match:
            tol = _clean_decimal_token(tol_match.group('pm') or tol_match.group('sep'))
            if tol:
                try:
                    tol_val = float(tol)
                except ValueError:
                    tol_val = 999.0
                if 0.0 < tol_val <= 10.0:
                    text_out = f'{text_out}±{tol}'

    return text_out


def _radius_has_dangling_tolerance_marker(text: str) -> bool:
    value = _normalize_radius_glyphs(text)
    match = re.search(
        r'(?<![A-Za-z0-9])(?:[1-9]\d?X)?'
        r'R(?:\d{1,3}(?:\.\d{1,4})?|\.\d{1,4})(?P<tail>[^A-Za-z]{0,18})',
        value,
        re.IGNORECASE,
    )
    if not match:
        return False
    tail = match.group('tail') or ''
    return bool(re.search(r'(?:±|:|\+)$', tail))


def _radius_tolerance_suffix_from_lines(
    lines: list[dict],
    main_bbox: dict,
) -> tuple[str | None, dict | None]:
    candidates: list[tuple[float, float, str, dict | None]] = []
    for line in lines:
        text = _normalize_radius_glyphs(line.get('text', ''))
        if not text or 'R' in text.upper():
            continue
        match = re.search(r'(?:±|\+)?(?P<tol>(?:\d{1,2}(?:\.\d{1,4})?|\.\d{1,4}))$', text)
        if not match:
            continue
        tol = _clean_decimal_token(match.group('tol'))
        if not tol:
            continue
        try:
            tol_val = float(tol)
        except ValueError:
            continue
        if not (0.0 < tol_val <= 10.0):
            continue
        bbox = line.get('bbox_in_pdf')
        if not _radius_suffix_near_main(main_bbox, bbox):
            continue
        candidates.append((float(line.get('confidence') or 0.0), tol_val, tol, bbox))

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _conf, _tol_val, tol, bbox = candidates[0]
    return f'±{tol}', bbox


def _radius_suffix_near_main(main_bbox: dict, suffix_bbox: dict | None) -> bool:
    if not suffix_bbox:
        return False
    mcx = float(main_bbox.get('x', 0.0)) + float(main_bbox.get('w', 0.0)) / 2.0
    mcy = float(main_bbox.get('y', 0.0)) + float(main_bbox.get('h', 0.0)) / 2.0
    scx = float(suffix_bbox.get('x', 0.0)) + float(suffix_bbox.get('w', 0.0)) / 2.0
    scy = float(suffix_bbox.get('y', 0.0)) + float(suffix_bbox.get('h', 0.0)) / 2.0
    dx = scx - mcx
    dy = abs(scy - mcy)
    h = max(float(main_bbox.get('h', 0.0) or 0.0),
            float(suffix_bbox.get('h', 0.0) or 0.0),
            1.0)
    return -4.0 <= dx <= max(52.0, h * 3.5) and dy <= max(18.0, h * 1.15)


def _score_radius_label_region(region: dict) -> float | None:
    try:
        x = float(region.get('x') or 0.0)
        y = float(region.get('y') or 0.0)
        w = float(region.get('w') or 0.0)
        h = float(region.get('h') or 0.0)
    except (TypeError, ValueError):
        return None
    if x < 45.0 or y < 75.0:
        return None
    if w < 7.0 or h < 5.0 or w > 130.0 or h > 118.0:
        return None
    source = str(region.get('source') or '')
    if h > 58.0 and source != 'track_b':
        return None
    area = w * h
    if area < 55.0 or area > 5200.0:
        return None

    curves = float(region.get('curve_count') or 0.0)
    ratio = max(w, h) / max(min(w, h), 1.0)
    score = 0.0
    if 16.0 <= w <= 88.0 and 8.0 <= h <= 35.0:
        score += 40.0
    if 24.0 <= w <= 115.0 and 7.0 <= h <= 24.0:
        score += 28.0
    if source == 'track_b' and w <= 42.0 and 42.0 <= h <= 115.0:
        score += 72.0
    if source == 'split_multiline' and h <= 36.0:
        score += 10.0
    if 2.0 <= curves <= 60.0:
        score += 13.0
    if 80.0 <= area <= 3000.0:
        score += 12.0
    if 1.0 <= ratio <= 8.0:
        score += 8.0
    if 18.0 <= w <= 62.0 and 12.0 <= h <= 30.0:
        score += 14.0
    if not source and 38.0 <= w <= 82.0 and 10.0 <= h <= 21.0 and curves >= 20.0:
        score += 34.0
    if not source and 28.0 <= w <= 48.0 and 20.0 <= h <= 32.0 and curves <= 8.0:
        score += 48.0
    if source == 'track_b' and w <= 42.0 and h >= 70.0:
        score += 42.0
    if curves > 80.0 and source != 'track_b':
        score -= 16.0
    if source == 'track_b' and h < 35.0 and w < 45.0:
        score -= 70.0
    if w > 105.0:
        score -= 10.0
    return score


def _select_radius_label_regions(text_regions: list, max_regions: int) -> list[tuple[int, dict, float]]:
    scored = []
    for idx, region in enumerate(text_regions or []):
        score = _score_radius_label_region(region)
        if score is None or score <= 0.0:
            continue
        scored.append((score, idx, region))
    scored.sort(key=lambda item: (
        -item[0],
        float(item[2].get('y') or 0.0),
        float(item[2].get('x') or 0.0),
    ))
    if max_regions and max_regions > 0:
        scored = scored[:max_regions]
    return [(idx, region, score) for score, idx, region in scored]


def _radius_ocr_attempts(region: dict) -> list[tuple[float, float]]:
    try:
        w = float(region.get('w') or 0.0)
        h = float(region.get('h') or 0.0)
        curves = float(region.get('curve_count') or 0.0)
    except (TypeError, ValueError):
        w = h = 0.0
        curves = 0.0
    source = str(region.get('source') or '')
    attempts: list[tuple[float, float]] = []

    def _add(angle: float, pad_pt: float):
        item = (float(angle), float(pad_pt))
        if item not in attempts:
            attempts.append(item)

    if source == 'track_b' or (h > w * 2.0 and h > 38.0):
        for angle in (-30.0, -45.0, -60.0, 30.0, 45.0, 0.0):
            _add(angle, 35.0 if angle else 24.0)
    elif w > h * 2.1:
        for angle, pad in ((0.0, 18.0), (0.0, 35.0), (-30.0, 35.0),
                           (30.0, 35.0), (-45.0, 35.0), (45.0, 35.0)):
            _add(angle, pad)
        if curves >= 35.0 and 42.0 <= w <= 72.0 and h <= 19.0:
            for angle in (30.0, 45.0, -30.0):
                _add(angle, 70.0)
    else:
        for angle, pad in ((0.0, 16.0), (-45.0, 24.0), (-30.0, 35.0),
                           (-60.0, 35.0), (30.0, 35.0), (45.0, 35.0)):
            _add(angle, pad)
    return attempts


def _ocr_pdf_region(
    ocr,
    page_img: Image.Image,
    region: dict,
    *,
    dpi: int,
    pad_pt: float,
    angle: float = 0.0,
) -> list[dict]:
    scale = dpi / 72.0
    page_w, page_h = page_img.size
    try:
        crop_bbox = {
            'x': float(region.get('x') or 0.0) - pad_pt,
            'y': float(region.get('y') or 0.0) - pad_pt,
            'w': float(region.get('w') or 0.0) + 2 * pad_pt,
            'h': float(region.get('h') or 0.0) + 2 * pad_pt,
        }
    except (TypeError, ValueError):
        return []
    px, py, pw, ph = pdf_bbox_to_pixel(crop_bbox, dpi)
    px0 = max(0, px)
    py0 = max(0, py)
    px1 = min(page_w, px + pw)
    py1 = min(page_h, py + ph)
    if px1 - px0 < 8 or py1 - py0 < 8:
        return []

    roi = page_img.crop((px0, py0, px1, py1))
    roi_for_ocr = roi.rotate(angle, expand=True, fillcolor=(255, 255, 255)) if angle else roi
    ocr_output = ocr.ocr(np.array(roi_for_ocr), cls=False)
    if not ocr_output or ocr_output[0] is None:
        return []

    rows = []
    for line in ocr_output[0]:
        if not line or len(line) < 2:
            continue
        quad, (text, conf) = line[0], line[1]
        if not text:
            continue
        bbox_in_roi = quad_to_bbox(quad)
        if angle:
            bbox_in_roi = rotate_bbox_back(bbox_in_roi, angle, px1 - px0, py1 - py0)
        bbox_page_px = {
            'x': bbox_in_roi['x'] + px0,
            'y': bbox_in_roi['y'] + py0,
            'w': bbox_in_roi['w'],
            'h': bbox_in_roi['h'],
        }
        rows.append({
            'text': str(text).strip(),
            'confidence': float(conf),
            'bbox_in_pdf': pixel_bbox_to_pdf(bbox_page_px, dpi),
            'angle_applied': float(angle),
        })
    return rows


def run_radius_label_ocr(
    page_img: Image.Image,
    text_regions: list,
    existing_ocr: list,
    dpi: int = 200,
    verbose: bool = True,
    deadline: float | None = None,
    max_regions: int = 220,
    diag: dict = None,
) -> list:
    """Target compact/rotated vector-text regions that look like radius labels."""
    if not text_regions:
        return []

    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    candidates = _select_radius_label_regions(text_regions, max_regions=max_regions)
    if diag is not None:
        diag['candidate_count'] = len(text_regions or [])
        diag['planned_count'] = len(candidates)
        diag['max_regions'] = max_regions
    if not candidates:
        return []

    ocr = _get_ocr()
    results = []
    seen: set[tuple[str, int, int]] = set()

    for region_id, region, score in candidates:
        if deadline is not None and time.monotonic() >= deadline:
            _bump('stopped_deadline')
            break

        best: tuple[float, str, dict, dict, float, float] | None = None
        for angle, pad_pt in _radius_ocr_attempts(region):
            if deadline is not None and time.monotonic() >= deadline:
                _bump('stopped_deadline')
                break
            try:
                _bump('ocr_calls')
                lines = _ocr_pdf_region(
                    ocr, page_img, region, dpi=dpi, pad_pt=pad_pt, angle=angle)
            except Exception:
                _bump('ocr_errors')
                continue
            if not lines:
                _bump('empty_outputs')
                continue
            _bump('raw_lines', len(lines))

            for line in lines:
                conf = float(line.get('confidence') or 0.0)
                if conf < 0.40:
                    _bump('rejected_low_conf')
                    continue
                text = _extract_radius_label_text(line.get('text', ''))
                if not text:
                    _bump('rejected_no_radius')
                    continue
                bbox_pdf = line['bbox_in_pdf']
                if '±' not in text:
                    suffix, suffix_bbox = _radius_tolerance_suffix_from_lines(lines, bbox_pdf)
                    if suffix:
                        text = f'{text}{suffix}'
                        if suffix_bbox:
                            bbox_pdf = _bbox_union(bbox_pdf, suffix_bbox)
                    elif _radius_has_dangling_tolerance_marker(line.get('text', '')):
                        _bump('rejected_dangling_tolerance')
                        continue
                rank = conf * 100.0 + min(len(text), 24)
                if '±' in text:
                    rank += 55.0
                if re.match(r'^\d{1,2}X', text):
                    rank += 8.0
                if best is None or rank > best[0]:
                    best = (rank, text, bbox_pdf, line, angle, pad_pt)
            if best and '±' in best[1]:
                break

        if not best:
            continue

        _rank, text, bbox_pdf, line, angle, pad_pt = best
        key = (text, round(float(bbox_pdf.get('x', 0.0))), round(float(bbox_pdf.get('y', 0.0))))
        if key in seen:
            _bump('rejected_duplicate')
            continue
        seen.add(key)
        results.append({
            'text': text,
            'confidence': float(line.get('confidence') or 0.0),
            'bbox_in_pdf': bbox_pdf,
            'font_height_approx': _font_height_proxy(bbox_pdf),
            'source_region_id': region_id,
            'orientation': float(angle),
            'low_confidence_region': False,
            'source': 'radius_label_ocr',
            '_radius_label_evidence': {
                'region_score': round(float(score), 2),
                'region_id': region_id,
                'angle': float(angle),
                'pad_pt': float(pad_pt),
                'raw_text': str(line.get('text', '')).strip(),
            },
        })
        _bump('accepted')

    if verbose:
        print(f"[radius_label_ocr] {len(results)} radius labels from {len(candidates)} regions")
    return results


def run_angle_label_ocr(
    page_img: Image.Image,
    text_regions: list,
    existing_ocr: list,
    dpi: int = 200,
    verbose: bool = True,
    deadline: float | None = None,
    max_regions: int = 220,
    diag: dict = None,
) -> list:
    """Target compact vector-text regions that look like angle labels."""
    if not text_regions:
        return []

    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    candidates = _select_angle_label_regions(text_regions, max_regions=max_regions)
    if diag is not None:
        diag['candidate_count'] = len(text_regions or [])
        diag['planned_count'] = len(candidates)
        diag['max_regions'] = max_regions
    if not candidates:
        return []

    ocr = _get_ocr()
    results = []
    seen: set[tuple[str, int, int]] = set()

    for region_id, region, score in candidates:
        if deadline is not None and time.monotonic() >= deadline:
            _bump('stopped_deadline')
            break
        try:
            _bump('ocr_calls')
            base_lines = _ocr_pdf_region(
                ocr, page_img, region, dpi=dpi, pad_pt=10.0, angle=0.0)
        except Exception:
            _bump('ocr_errors')
            continue
        if not base_lines:
            _bump('empty_outputs')
            continue
        _bump('raw_lines', len(base_lines))

        main_lines = []
        for line in base_lines:
            text = _extract_angle_main_text(line.get('text', ''))
            if not text:
                continue
            if float(line.get('confidence') or 0.0) < 0.45:
                _bump('rejected_low_conf')
                continue
            main_lines.append((text, line))
        if not main_lines:
            _bump('rejected_no_degree')
            continue

        for text, line in main_lines:
            bbox_pdf = line['bbox_in_pdf']
            suffix, suffix_bbox = _angle_tolerance_suffix_from_lines(base_lines)
            if suffix and not _angle_suffix_near_main(bbox_pdf, suffix_bbox):
                suffix = None
                suffix_bbox = None
            if suffix is None:
                tol_lines = []
                try:
                    for angle, pad_pt in ((0.0, 35.0), (90.0, 18.0), (-90.0, 18.0)):
                        if deadline is not None and time.monotonic() >= deadline:
                            _bump('stopped_deadline')
                            break
                        _bump('ocr_calls')
                        tol_lines.extend(_ocr_pdf_region(
                            ocr, page_img, region, dpi=dpi, pad_pt=pad_pt, angle=angle))
                        suffix, suffix_bbox = _angle_tolerance_suffix_from_lines(tol_lines)
                        if suffix and not _angle_suffix_near_main(bbox_pdf, suffix_bbox):
                            suffix = None
                            suffix_bbox = None
                        if suffix:
                            break
                except Exception:
                    _bump('ocr_errors')
                    tol_lines = []
                _bump('tolerance_raw_lines', len(tol_lines))
            if suffix and '±' not in text:
                text = f'{text}{suffix}'
                if suffix_bbox:
                    bbox_pdf = _bbox_union(bbox_pdf, suffix_bbox)

            key = (text, round(float(bbox_pdf.get('x', 0.0))), round(float(bbox_pdf.get('y', 0.0))))
            if key in seen:
                _bump('rejected_duplicate')
                continue
            seen.add(key)
            result = {
                'text': text,
                'confidence': float(line.get('confidence') or 0.0),
                'bbox_in_pdf': bbox_pdf,
                'font_height_approx': _font_height_proxy(bbox_pdf),
                'source_region_id': region_id,
                'orientation': 0.0,
                'low_confidence_region': False,
                'source': 'angle_label_ocr',
                '_angle_label_evidence': {
                    'region_score': round(float(score), 2),
                    'region_id': region_id,
                },
            }
            results.append(result)
            _bump('accepted')

    if verbose:
        print(f"[angle_label_ocr] {len(results)} angle labels from {len(candidates)} regions")
    return results


def run_capsule_ocr(
    page_img: Image.Image,
    capsules: list,
    existing_ocr: list,
    dpi: int = 200,
    verbose: bool = True,
    deadline: float | None = None,
    diag: dict = None,
) -> list:
    """
    对 capsule（跑道框）内部区域做独立 OCR，不依赖 DBSCAN 覆盖。

    参数:
        page_img: PIL Image，整页渲染图（dpi 对应分辨率）
        capsules: detect_capsules() 的输出 [{'x','y','w','h'}, ...]（PDF 坐标）
        existing_ocr: 已有 OCR 结果列表（用于覆盖检查，避免重复识别）
        dpi: 渲染 DPI
        verbose: 是否打印诊断日志

    返回:
        list of dict，新增的 OCR 结果（格式与 run_directed_ocr 输出一致）
    """
    if not capsules:
        return []

    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    ocr = _get_ocr()
    results = []
    n_skipped_covered = 0

    def _capsule_angle(cap: dict) -> float:
        angle = float(cap.get('angle', 0.0) or 0.0) % 180.0
        # Older fallbacks did not always store an angle for vertical capsules.
        # Infer it from the AABB so vertical racetracks are rotated for OCR.
        try:
            w = float(cap.get('w', 0.0) or 0.0)
            h = float(cap.get('h', 0.0) or 0.0)
        except (TypeError, ValueError):
            w = h = 0.0
        if (angle < 5 or angle > 175) and h > w * 1.8 and h > 20:
            return 90.0
        return angle

    for ci, cap in enumerate(capsules):
        if deadline is not None and time.monotonic() >= deadline:
            _bump('stopped_deadline')
            break
        # 内缩 2pt 避免边框弧线干扰 OCR
        shrink = 2.0
        crop_bbox = {
            'x': cap['x'] + shrink,
            'y': cap['y'] + shrink,
            'w': max(cap['w'] - 2 * shrink, 5),
            'h': max(cap['h'] - 2 * shrink, 5),
        }

        # 斜向 capsule 的 AABB 远大于实际框体，周围无关 OCR 结果
        # 会误触 has_good_ocr 导致跳过。斜向 capsule 强制重新 OCR（带旋转）
        capsule_angle = _capsule_angle(cap)
        _needs_rotation = not (capsule_angle < 5 or capsule_angle > 175)

        if not _needs_rotation:
            # 水平 capsule：检查是否已被 existing_ocr 覆盖。
            # 竖向/斜向 capsule 即使已有碎片，也要强制旋平再 OCR。
            has_good_ocr = False
            for er in existing_ocr:
                eb = er.get('bbox_in_pdf', er.get('bbox', None))
                if eb is None:
                    continue
                ecx = eb['x'] + eb['w'] / 2
                ecy = eb['y'] + eb['h'] / 2
                if (cap['x'] <= ecx <= cap['x'] + cap['w'] and
                        cap['y'] <= ecy <= cap['y'] + cap['h']):
                    import re as _re
                    if _re.search(r'\d', er.get('text', '')):
                        has_good_ocr = True
                        break

            if has_good_ocr:
                n_skipped_covered += 1
                continue
        else:
            if verbose:
                print(f"  [capsule_ocr] capsule {ci}: rotated target ({capsule_angle:.0f}°), "
                      f"force OCR (skip coverage check)")

        # 切 ROI
        px, py, pw, ph = pdf_bbox_to_pixel(crop_bbox, dpi)
        page_w, page_h = page_img.size
        pad = 4  # 小 padding 防止边缘切割
        left = max(0, px - pad)
        top = max(0, py - pad)
        right = min(page_w, px + pw + pad)
        bottom = min(page_h, py + ph + pad)

        if right - left < 8 or bottom - top < 8:
            continue

        roi = page_img.crop((left, top, right, bottom))
        origin_px = (left, top)

        # 上采样极小 ROI
        roi_w, roi_h = roi.size
        if roi_w * roi_h < 100:
            scale_up = 3
        elif roi_w * roi_h < 400:
            scale_up = 2
        else:
            scale_up = 1
        if scale_up > 1:
            roi = roi.resize((roi_w * scale_up, roi_h * scale_up), Image.LANCZOS)

        # capsule 有 angle 时旋转到水平再 OCR（覆盖斜向跑道框）
        capsule_angle = _capsule_angle(cap)
        _needs_rotation = not (capsule_angle < 5 or capsule_angle > 175)
        angle_applied = 0.0
        if _needs_rotation:
            # 旋转到水平：负角度使 capsule 长轴对齐水平
            rot_angle = _rotation_to_horizontal(capsule_angle)
            roi_rot = roi.rotate(rot_angle, expand=True, fillcolor=(255, 255, 255))
            try:
                _bump('ocr_calls')
                ocr_output = ocr.ocr(np.array(roi_rot), cls=False)
                angle_applied = rot_angle
            except Exception:
                _bump('ocr_errors')
                ocr_output = None
            # 旋转版失败则回退原图
            if not ocr_output or ocr_output[0] is None:
                try:
                    _bump('ocr_calls')
                    ocr_output = ocr.ocr(np.array(roi), cls=False)
                    angle_applied = 0.0
                except Exception:
                    _bump('ocr_errors')
                    continue
            if verbose:
                print(f"  [capsule_ocr] capsule {ci}: angle={capsule_angle:.0f}° → "
                      f"rotated {rot_angle:.0f}°")
        else:
            try:
                _bump('ocr_calls')
                ocr_output = ocr.ocr(np.array(roi), cls=False)
            except Exception:
                _bump('ocr_errors')
                continue

        if not ocr_output or ocr_output[0] is None:
            _bump('empty_outputs')
            continue

        for line in ocr_output[0]:
            if not line or len(line) < 2:
                continue
            quad, (text, conf) = line[0], line[1]
            _bump('raw_lines')
            # capsule 有几何锚点确认，置信度门控降至 0.15
            if not text or conf < 0.15:
                _bump('rejected_low_conf')
                continue

            # 还原 scale_up
            quad_unscaled = [[pt[0] / scale_up, pt[1] / scale_up] for pt in quad]
            bbox_in_roi = quad_to_bbox(quad_unscaled)
            if angle_applied != 0.0:
                bbox_in_roi = rotate_bbox_back(bbox_in_roi, angle_applied,
                                               roi_w, roi_h)

            # ROI 坐标 → 页面像素坐标 → PDF 坐标
            bbox_page_px = {
                'x': bbox_in_roi['x'] + origin_px[0],
                'y': bbox_in_roi['y'] + origin_px[1],
                'w': bbox_in_roi['w'],
                'h': bbox_in_roi['h'],
            }
            bbox_pdf = pixel_bbox_to_pdf(bbox_page_px, dpi)

            text_clean = text.strip()
            result = {
                'text': text_clean,
                'confidence': float(conf),
                'bbox_in_pdf': bbox_pdf,
                'font_height_approx': _font_height_proxy(bbox_pdf),
                'source_region_id': -1,
                'orientation': float(capsule_angle if _needs_rotation else 0.0),
                'low_confidence_region': False,
                'source': 'capsule_ocr',
            }
            results.append(attach_r2p_digit_restricted_ocr_shadow(result, text_clean))

    if verbose:
        print(f"[capsule_ocr] {len(capsules)} capsules, "
              f"{n_skipped_covered} already covered, "
              f"{len(results)} new OCR results")

    return results


# ---------------------------------------------------------------------------
# 尺寸线条带 OCR（箭头配对驱动）
# ---------------------------------------------------------------------------

def run_dimline_strip_ocr(
    page_img: Image.Image,
    dimension_lines: list,
    existing_ocr: list,
    dpi: int = 200,
    verbose: bool = True,
    deadline: float | None = None,
    diag: dict = None,
) -> list:
    """
    对箭头配对检测到的尺寸线，沿线方向生成 OCR 条带，直接裁切送 OCR。

    条带参数:
    - 中心 = 尺寸线中点
    - 方向 = 与尺寸线平行
    - 长度 = 尺寸线全长（短线 <30pt 时两端各延伸 50pt）
    - 宽度 = 自适应，锚定 existing_ocr 中位数文字高度 × 1.5

    参数:
        page_img: PIL Image，整页渲染图
        dimension_lines: arrow_detector.detect_arrows()['dimension_lines']
            每个 dict 含 start, end, midpoint, length, orientation ('H'/'V'/'D')
        existing_ocr: 已有 OCR 结果列表（用于估算文字高度 + 覆盖检查）
        dpi: 渲染 DPI
        verbose: bool

    返回:
        list of dict，新增的 OCR 结果（source='dimline_strip_ocr'）
    """
    if not dimension_lines:
        return []

    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    # 估算全图中位数文字高度（从已有高置信度 OCR 结果）
    text_heights = sorted([
        r['bbox_in_pdf']['h'] for r in existing_ocr
        if r.get('confidence', 0) > 0.6 and 2 < r['bbox_in_pdf']['h'] < 30
    ])
    if text_heights:
        median_text_h = text_heights[len(text_heights) // 2]
    else:
        median_text_h = 8.0  # 合理默认值
    # 条带半宽 = 文字高度 × 1.5（覆盖标称值 + 可能的上下公差）
    strip_half_w = median_text_h * 1.5
    cross_strip_half_w = max(strip_half_w, median_text_h * 3.0, 24.0)
    # Converted engineering drawings often place vertical/angled dimension text
    # next to the extension line rather than directly on top of the dimension
    # line.  Try a few parallel strips on both sides, but stop once a target
    # produces a usable OCR result so this remains bounded.
    side_offsets_default = [0.0]
    side_offsets_cross = [0.0, 60.0, -60.0, 120.0, -120.0]

    if verbose:
        print(f"[dimline_strip_ocr] {len(dimension_lines)} dim lines, "
              f"median_text_h={median_text_h:.1f}pt, strip_half_w={strip_half_w:.1f}pt")

    ocr = _get_ocr()
    results = []
    scale = dpi / 72.0
    page_w, page_h = page_img.size

    for di, dl in enumerate(dimension_lines):
        if deadline is not None and time.monotonic() >= deadline:
            _bump('stopped_deadline')
            break
        sx, sy = dl['start']
        ex, ey = dl['end']
        mx, my = dl['midpoint']
        line_len = dl['length']
        orient = dl['orientation']
        line_kind = dl.get('line_kind') or 'paired_dimension_line'
        is_leader = line_kind == 'single_arrow_leader'
        line_angle = math.degrees(math.atan2(ey - sy, ex - sx)) % 180.0

        # 短尺寸线（<30pt）: 文字在外侧，两端延伸 50pt
        extend = 50.0 if line_len < 30 else 0.0

        ux = (ex - sx) / line_len if line_len else 1.0
        uy = (ey - sy) / line_len if line_len else 0.0
        px = -uy
        py = ux
        offsets = side_offsets_cross if orient in {'V', 'D'} else side_offsets_default
        line_results = []
        weak_fallback_results = []
        weak_fallback_strength = -999

        for side_offset in offsets:
            offset_results = []
            # 按线方向/法线方向扩成窄条带，再取其 AABB。对 H/V/D 使用同一几何，
            # 这样 side_offset 能自然表达“尺寸线两侧”的搜索。
            a = (
                sx - ux * extend + px * side_offset,
                sy - uy * extend + py * side_offset,
            )
            b = (
                ex + ux * extend + px * side_offset,
                ey + uy * extend + py * side_offset,
            )
            half_w = cross_strip_half_w if orient in {'V', 'D'} else strip_half_w
            corners = [
                (a[0] + px * half_w, a[1] + py * half_w),
                (a[0] - px * half_w, a[1] - py * half_w),
                (b[0] + px * half_w, b[1] + py * half_w),
                (b[0] - px * half_w, b[1] - py * half_w),
            ]
            x0 = min(c[0] for c in corners)
            x1 = max(c[0] for c in corners)
            y0 = min(c[1] for c in corners)
            y1 = max(c[1] for c in corners)

            # 裁切像素坐标
            px0 = max(0, int(x0 * scale))
            py0 = max(0, int(y0 * scale))
            px1 = min(page_w, int(x1 * scale))
            py1 = min(page_h, int(y1 * scale))

            if px1 - px0 < 8 or py1 - py0 < 8:
                continue

            roi = page_img.crop((px0, py0, px1, py1))
            origin_px = (px0, py0)

            # 竖直/斜向条带：旋转到水平送 OCR（PaddleOCR 对水平文字效果最好）
            angle_applied = 0.0
            def _best_conf(ocr_out):
                if not ocr_out or not ocr_out[0]:
                    return 0.0
                confs = [l[1][1] for l in ocr_out[0] if l and len(l) >= 2]
                return max(confs) if confs else 0.0

            if orient == 'V':
                # 双向旋转取最优
                roi_cw = roi.rotate(-90, expand=True, fillcolor=(255, 255, 255))
                roi_ccw = roi.rotate(90, expand=True, fillcolor=(255, 255, 255))
                try:
                    _bump('ocr_calls', 2)
                    out_cw = ocr.ocr(np.array(roi_cw), cls=False)
                    out_ccw = ocr.ocr(np.array(roi_ccw), cls=False)
                except Exception:
                    _bump('ocr_errors')
                    continue

                if _best_conf(out_cw) >= _best_conf(out_ccw):
                    ocr_output = out_cw
                    angle_applied = -90.0
                else:
                    ocr_output = out_ccw
                    angle_applied = 90.0
            elif orient == 'D':
                rot_angle = _rotation_to_horizontal(line_angle)
                roi_rot = roi.rotate(rot_angle, expand=True, fillcolor=(255, 255, 255))
                try:
                    _bump('ocr_calls')
                    ocr_output = ocr.ocr(np.array(roi_rot), cls=False)
                    angle_applied = rot_angle
                except Exception:
                    _bump('ocr_errors')
                    continue

                # 斜向旋平失败时回退原 AABB，避免少数 OCR 后端对旋转图退化。
                if not ocr_output or ocr_output[0] is None:
                    try:
                        _bump('ocr_calls')
                        ocr_output = ocr.ocr(np.array(roi), cls=False)
                        angle_applied = 0.0
                    except Exception:
                        _bump('ocr_errors')
                        continue
            else:
                try:
                    _bump('ocr_calls')
                    ocr_output = ocr.ocr(np.array(roi), cls=False)
                except Exception:
                    _bump('ocr_errors')
                    continue

            if not ocr_output or ocr_output[0] is None:
                _bump('empty_outputs')
                continue

            for line in ocr_output[0]:
                if not line or len(line) < 2:
                    continue
                quad, (text, conf) = line[0], line[1]
                _bump('raw_lines')
                conf_floor = 0.3
                if not text or conf < conf_floor:
                    _bump('rejected_low_conf')
                    continue
                text_stripped = text.strip()
                if (orient == 'D' and line_len < 50
                        and re.match(r'^\d+(?:\.\d+)?$', text_stripped)):
                    _bump('rejected_short_diagonal_plain_number')
                    continue

                bbox_in_roi = quad_to_bbox(quad)

                # 反向旋转（竖直条带）
                if angle_applied != 0.0:
                    orig_roi_w = px1 - px0
                    orig_roi_h = py1 - py0
                    bbox_in_roi = rotate_bbox_back(bbox_in_roi, angle_applied,
                                                   orig_roi_w, orig_roi_h)

                # ROI 像素 → 页面像素 → PDF 坐标
                bbox_page_px = {
                    'x': bbox_in_roi['x'] + origin_px[0],
                    'y': bbox_in_roi['y'] + origin_px[1],
                    'w': bbox_in_roi['w'],
                    'h': bbox_in_roi['h'],
                }
                bbox_pdf = pixel_bbox_to_pdf(bbox_page_px, dpi)

                result = {
                    'text': text_stripped,
                    'confidence': float(conf),
                    'bbox_in_pdf': bbox_pdf,
                    'font_height_approx': _font_height_proxy(bbox_pdf),
                    'source_region_id': -1,
                    'orientation': float(line_angle if orient == 'D' else angle_applied),
                    'low_confidence_region': False,
                    'source': 'dimline_strip_ocr',
                }
                evidence = {
                    'line_index': di,
                    'distance': round(abs(float(side_offset)), 2),
                    'normal_offset': round(float(side_offset), 2),
                    'orientation': orient,
                    'line_length': round(float(line_len or 0), 2),
                    'line_angle': round(float(line_angle), 2),
                    'kind': line_kind,
                    'source': 'dimline_strip_ocr',
                }
                if is_leader:
                    result['_leader_line_evidence'] = evidence
                else:
                    result['_strip_ocr_evidence'] = evidence
                    result['_dimension_line_evidence'] = evidence
                offset_results.append(
                    attach_r2p_digit_restricted_ocr_shadow(result, text_stripped))
            if not offset_results:
                continue
            if orient not in {'V', 'D'}:
                line_results = offset_results
                break
            best_offset_strength = max(
                _strip_text_strength(
                    r.get('text', ''),
                    orient=orient,
                    is_leader=is_leader,
                )
                for r in offset_results
            )
            if best_offset_strength > weak_fallback_strength:
                weak_fallback_strength = best_offset_strength
                weak_fallback_results = offset_results
            if _strong_strip_text_found(offset_results, orient=orient, is_leader=is_leader):
                line_results = offset_results
                break
        if not line_results and weak_fallback_results:
            line_results = weak_fallback_results
        results.extend(line_results)

    if verbose:
        print(f"[dimline_strip_ocr] {len(results)} new OCR results "
              f"from {len(dimension_lines)} dimension lines")

    return results
