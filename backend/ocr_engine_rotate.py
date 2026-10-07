"""ocr_engine_rotate.py — 多方向旋转 OCR + 坐标变换

从 ocr_engine.py 拆出。"""

import time
import numpy as np
from PIL import Image

from ocr_engine_utils import (
    VERBOSE, _get_ocr, quad_to_bbox, pixel_bbox_to_pdf, pdf_bbox_to_pixel,
    _bbox_iou, attach_r2p_digit_restricted_ocr_shadow,
)


def transform_region_cw90(region, page_w, page_h):
    """将 region bbox 从 0° 坐标系变换到顺时针 90° 坐标系。
    公式: x' = H - (y + h), y' = x, w' = h, h' = w
    新页面尺寸: (H, W)
    """
    return {
        **region,
        'x': page_h - (region['y'] + region['h']),
        'y': region['x'],
        'w': region['h'],
        'h': region['w'],
    }


def transform_region_cw270(region, page_w, page_h):
    """将 region bbox 从 0° 坐标系变换到顺时针 270° 坐标系。
    公式: x' = y, y' = W - (x + w), w' = h, h' = w
    新页面尺寸: (H, W)
    """
    return {
        **region,
        'x': region['y'],
        'y': page_w - (region['x'] + region['w']),
        'w': region['h'],
        'h': region['w'],
    }


def transform_ocr_back_cw90(ocr_result, page_w, page_h):
    """将 90° 副本的 OCR 结果 bbox 反变换回 0° 坐标系。
    反公式: x = y', y = H - (x' + w'), w = h', h = w'
    """
    b = ocr_result['bbox_in_pdf']
    new_b = {
        'x': b['y'],
        'y': page_h - (b['x'] + b['w']),
        'w': b['h'],
        'h': b['w'],
    }
    return {
        **ocr_result,
        'bbox_in_pdf': new_b,
        'font_height_approx': new_b['h'],
        'orientation': 90.0,
    }


def transform_ocr_back_cw270(ocr_result, page_w, page_h):
    """将 270° 副本的 OCR 结果 bbox 反变换回 0° 坐标系。
    反公式: x = W - (y' + h'), y = x', w = h', h = w'
    """
    b = ocr_result['bbox_in_pdf']
    new_b = {
        'x': page_w - (b['y'] + b['h']),
        'y': b['x'],
        'w': b['h'],
        'h': b['w'],
    }
    return {
        **ocr_result,
        'bbox_in_pdf': new_b,
        'font_height_approx': new_b['h'],
        'orientation': -90.0,
    }


def run_rotated_ocr(
    page_img,
    text_regions,
    dpi=200,
    diag: dict = None,
    deadline: float | None = None,
    max_regions: int | None = None,
):
    """
    对已旋转的页面图像做纯水平 OCR。

    不做 per-region 旋转、不做 strip_ocr、不做 _find_near_angle。
    只做：裁切 ROI → padding → PaddleOCR → 返回结果列表。

    参数:
        page_img: PIL.Image，已旋转的页面位图
        text_regions: list of dict {x, y, w, h, ...}，已坐标变换的 region
        dpi: int

    返回:
        list of dict，与 run_directed_ocr 输出格式一致。
        bbox_in_pdf 是旋转后坐标系中的坐标，调用方负责反变换回 0°。
    """
    if not text_regions:
        return []

    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    _bump('regions_total', len(text_regions))
    if max_regions is not None and max_regions > 0 and len(text_regions) > max_regions:
        _bump('capped_count', len(text_regions) - max_regions)
        text_regions = text_regions[:max_regions]

    ocr = _get_ocr()
    results = []
    page_w_px, page_h_px = page_img.size

    for region_id, region in enumerate(text_regions):
        if deadline is not None and time.monotonic() >= deadline:
            _bump('stopped_deadline')
            break
        # 跳过大面积区域
        region_area = region['w'] * region['h']
        page_area_pdf = (page_w_px / (dpi / 72.0)) * (page_h_px / (dpi / 72.0))
        if region_area > page_area_pdf * 0.01:
            _bump('skipped_region_too_big')
            continue

        px, py, pw, ph = pdf_bbox_to_pixel(region, dpi)
        pad = max(int(ph * 0.4), 10)

        left   = max(0, px - pad)
        top    = max(0, py - pad)
        right  = min(page_w_px, px + pw + pad)
        bottom = min(page_h_px, py + ph + pad)

        if right - left < 2 or bottom - top < 2:
            _bump('skipped_region_too_small')
            continue

        _bump('regions_processed')
        roi = page_img.crop((left, top, right, bottom))
        origin_px = (left, top)

        # 极小 ROI 上采样
        roi_w, roi_h = roi.size
        if roi_w * roi_h < 100:
            scale_up = 3
        elif roi_w * roi_h < 400:
            scale_up = 2
        else:
            scale_up = 1
        if scale_up > 1:
            roi = roi.resize((roi_w * scale_up, roi_h * scale_up), Image.LANCZOS)

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
            if not text or conf < 0.3:
                _bump('rejected_low_conf')
                continue

            # 还原 scale_up
            quad_unscaled = [[pt[0] / scale_up, pt[1] / scale_up] for pt in quad]
            bbox_in_roi = quad_to_bbox(quad_unscaled)

            # ROI 坐标 → 页面像素坐标
            bbox_page_px = {
                'x': bbox_in_roi['x'] + origin_px[0],
                'y': bbox_in_roi['y'] + origin_px[1],
                'w': bbox_in_roi['w'],
                'h': bbox_in_roi['h'],
            }
            bbox_pdf = pixel_bbox_to_pdf(bbox_page_px, dpi)

            # 过滤：OCR 文字中心必须落在原始 region 内
            text_cx = bbox_pdf['x'] + bbox_pdf['w'] / 2
            text_cy = bbox_pdf['y'] + bbox_pdf['h'] / 2
            if not (region['x'] <= text_cx <= region['x'] + region['w'] and
                    region['y'] <= text_cy <= region['y'] + region['h']):
                _bump('rejected_center_outside')
                continue

            _bump('final_lines')
            text_clean = text.strip()
            result = {
                'text': text_clean,
                'confidence': float(conf),
                'bbox_in_pdf': bbox_pdf,
                'font_height_approx': bbox_pdf['h'],
                'source_region_id': region.get('_source_region_id', region_id),
                'orientation': 0.0,  # 在旋转图上是水平的，调用方设置真实方向
                'low_confidence_region': False,
                'source': 'rotated_ocr',
            }
            results.append(attach_r2p_digit_restricted_ocr_shadow(result, text_clean))

    return results
