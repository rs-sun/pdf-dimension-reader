"""Text-region prioritization for expensive OCR fallback passes."""

from __future__ import annotations


def score_text_region_for_ocr(
    region: dict,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> float:
    """Score text regions by how likely they are to contain a dimension token.

    The fallback OCR passes are capped for A0 drawings, so raw reading order or
    area-descending order spends too much budget on title/block fragments.  This
    score keeps compact vector-text clusters, split multiline glyphs, and
    track-B regions near the front while demoting page-edge frame artifacts.
    """
    w = float(region.get('w') or 0.0)
    h = float(region.get('h') or 0.0)
    x = float(region.get('x') or 0.0)
    y = float(region.get('y') or 0.0)
    area = w * h
    score = 0.0

    if 4.5 <= w <= 110.0 and 4.5 <= h <= 80.0:
        score += 40.0
    if 35.0 <= area <= 2200.0:
        score += 25.0

    if max(w, h) > 0.0:
        ratio = max(w, h) / max(min(w, h), 1e-6)
        if 1.0 <= ratio <= 9.0:
            score += 18.0
        if 1.7 <= ratio <= 7.0:
            score += 8.0

    curve_count = float(region.get('curve_count') or 0.0)
    if 1.0 <= curve_count <= 14.0:
        score += 12.0
    if curve_count >= 2.0:
        score += 4.0
    if curve_count >= 4.0:
        score += 4.0

    source = region.get('source')
    if source in ('split_multiline', 'track_b'):
        score += 12.0
    if source == 'track_b':
        score += 8.0
    if str(region.get('confidence') or '').lower() == 'high':
        score += 4.0

    # Single-glyph vector splits are common, but they are often fragments of a
    # larger dimension. Keep them eligible while letting whole-token boxes of
    # similar quality run first under a capped OCR budget.
    if source == 'split_multiline' and curve_count <= 1.0 and 55.0 <= area <= 180.0:
        score -= 16.0
        score += 4.0
    if curve_count <= 1.0 and 7.0 <= w <= 18.0 and 6.0 <= h <= 13.0:
        score += 4.0
    if 8.0 <= w <= 16.0 and 18.0 <= h <= 45.0:
        score += 12.0

    if page_width > 0.0 and page_height > 0.0:
        cx = x + w / 2.0
        cy = y + h / 2.0
        if (cx < page_width * 0.015 or cx > page_width * 0.985 or
                cy < page_height * 0.025 or cy > page_height * 0.975):
            score -= 90.0

    if w < 3.5 or h < 3.5:
        score -= 50.0

    return score


def sort_text_regions_for_ocr(
    text_regions: list[dict],
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> list[dict]:
    """Return copies of text regions ordered for capped fallback OCR passes."""
    ranked = []
    for idx, region in enumerate(text_regions or []):
        copied = dict(region)
        copied.setdefault('_source_region_id', idx)
        area = float(copied.get('w') or 0.0) * float(copied.get('h') or 0.0)
        ranked.append((
            -score_text_region_for_ocr(copied, page_width, page_height),
            -area,
            idx,
            copied,
        ))
    ranked.sort()
    return [region for _, _, _, region in ranked]
