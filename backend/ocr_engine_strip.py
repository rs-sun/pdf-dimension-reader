"""ocr_engine_strip.py — 条带 OCR + GD&T 分格 OCR

从 ocr_engine.py 拆出。"""

import math
import re
import time
import numpy as np
from PIL import Image

import fitz

from ocr_engine_utils import (
    VERBOSE, _get_ocr, render_page_to_image,
    pdf_bbox_to_pixel, pixel_bbox_to_pdf, quad_to_bbox,
    rotate_bbox_back, _extract_roi,
    _build_angle_index, _find_near_angle,
    _bbox_iou, _bbox_containment,
    attach_r2p_digit_restricted_ocr_shadow,
)


def _ocr_output_dimension_score(ocr_out) -> float:
    """Prefer OCR attempts that contain a complete-looking dimension token."""
    if not ocr_out or not ocr_out[0]:
        return 0.0
    best = 0.0
    for line in ocr_out[0]:
        if not line or len(line) < 2:
            continue
        text, conf = line[1]
        value = re.sub(r'\s+', '', str(text or '').strip())
        if not value:
            continue
        score = float(conf or 0.0)
        if re.search(r'\d', value):
            score += 1.0
        if re.match(r'^[RrCc⌀∅ØφΦ]\d', value):
            score += 4.0
        if re.search(r'\d+\.\d+', value):
            score += 1.5
        if any(sym in value for sym in ('±', '+', '-')):
            score += 2.0
        if '°' in value:
            score += 2.0
        if value[0] in '.+±-' or value[-1] == '.':
            score -= 2.5
        best = max(best, score)
    return best


def run_strip_ocr(
    page,
    l1_lines: list,
    existing_ocr_results: list,
    pdf_bytes: bytes,
    page_index: int,
    frame_borders: list = None,
    dpi: int = 200,
    verbose: bool = True,
    deadline: float | None = None,
    diag: dict = None,
) -> list:
    """
    v2: 对现有 OCR 结果中"可疑残缺"的条目，扩大区域重新 OCR 补充。

    参数:
        page: pdfplumber page 对象（用于获取页面尺寸）
        l1_lines: L1 层细线段列表（本版本未使用，保留接口兼容）
        existing_ocr_results: run_directed_ocr 的输出
        pdf_bytes: PDF 文件二进制内容
        page_index: 0-based 页码
        frame_borders: frame_borders 列表（含 title_block，本版本未使用）
        dpi: 渲染 DPI
        verbose: 是否打印诊断日志

    返回:
        补充 OCR 结果列表，格式与 run_directed_ocr 的输出一致
    """
    import re as _re

    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    if not existing_ocr_results:
        if verbose:
            print("[strip_ocr] 无现有 OCR 结果，跳过")
        return []

    page_width = float(page.width)
    page_height = float(page.height)

    # ── Step 1: 计算高置信度结果的 median_height ──
    high_conf_heights = sorted([
        r['bbox_in_pdf']['h'] for r in existing_ocr_results
        if r.get('confidence', 0) > 0.8 and r['bbox_in_pdf']['h'] > 0
    ])
    if high_conf_heights:
        median_height = high_conf_heights[len(high_conf_heights) // 2]
    else:
        median_height = 12.0

    if verbose:
        print(f"[strip_ocr] 高置信度 median_height: {median_height:.1f}pt")

    # ── Step 1: 筛选可疑残缺 OCR 结果 ──
    suspects = []
    for r in existing_ocr_results:
        text = r.get('text', '').strip()
        bbox = r['bbox_in_pdf']
        conf = r.get('confidence', 1.0)
        reasons = []

        # a) bbox 宽度极窄（字符极少）
        if bbox['w'] < median_height * 1.5:
            reasons.append('narrow_bbox')

        # b) 文本以截断碎片特征开头/结尾
        if text and (text[0] in '.+±-' or text[-1] == '.'):
            reasons.append('truncated_fragment')

        # c) 文本只有 1-2 字符且包含数字
        if len(text) <= 2 and _re.search(r'\d', text):
            reasons.append('short_numeric')

        # d) 低置信度且包含数字
        if conf < 0.6 and _re.search(r'\d', text):
            reasons.append('low_conf_numeric')

        if reasons:
            suspects.append((r, reasons))

    if verbose:
        print(f"[strip_ocr] 可疑残缺 OCR: {len(suspects)} 个")
        for r, reasons in suspects:
            print(f"  text='{r['text']}' reasons={reasons} "
                  f"bbox=({r['bbox_in_pdf']['x']:.0f},{r['bbox_in_pdf']['y']:.0f},"
                  f"{r['bbox_in_pdf']['w']:.0f}x{r['bbox_in_pdf']['h']:.0f})")

    if not suspects:
        return []

    # ── Step 2 & 3: 对每个可疑结果生成扩大区域并做 OCR ──
    page_img = render_page_to_image(pdf_bytes, page_index, dpi)
    ocr = _get_ocr()
    results = []

    for r, reasons in suspects:
        if deadline is not None and time.monotonic() >= deadline:
            _bump('stopped_deadline')
            break
        bbox = r['bbox_in_pdf']
        orientation = r.get('orientation', 0.0)
        is_vertical = abs(abs(orientation) - 90) < 15

        # 计算扩大区域
        if is_vertical:
            # 竖直文字：上下大扩展，左右小扩展
            expand_h = bbox['h'] * 2
            expand_w = bbox['w'] * 0.5
        else:
            # 水平文字：左右大扩展，上下小扩展
            expand_w = bbox['w'] * 2
            expand_h = bbox['h'] * 0.5
        if re.search(r'[RrCc]', text) and (text.endswith('.') or '±' in text):
            # Radius/chamfer callouts are often diagonal but can enter this
            # pass as a vertical-looking fragment. Widen both axes so the
            # original, unrotated retry can see the missing tolerance tail.
            expand_w = max(expand_w, bbox['w'] * 2.2, median_height * 6.0, 45.0)
            expand_h = max(expand_h, bbox['h'] * 1.5, median_height * 4.0, 35.0)

        ex = max(0, bbox['x'] - expand_w)
        ey = max(0, bbox['y'] - expand_h)
        ex2 = min(page_width, bbox['x'] + bbox['w'] + expand_w)
        ey2 = min(page_height, bbox['y'] + bbox['h'] + expand_h)
        expanded_bbox = {'x': ex, 'y': ey, 'w': ex2 - ex, 'h': ey2 - ey}

        if expanded_bbox['w'] < 1 or expanded_bbox['h'] < 1:
            continue

        # 切 ROI
        px, py, pw, ph = pdf_bbox_to_pixel(expanded_bbox, dpi)
        page_w, page_h = page_img.size

        pad = max(int(ph * 0.3), 8)
        left   = max(0, px - pad)
        top    = max(0, py - pad)
        right  = min(page_w, px + pw + pad)
        bottom = min(page_h, py + ph + pad)

        roi = page_img.crop((left, top, right, bottom))
        origin_px = (left, top)

        roi_w, roi_h = roi.size
        if roi_w * roi_h < 100:
            scale_up = 3
        elif roi_w * roi_h < 400:
            scale_up = 2
        else:
            scale_up = 1
        if scale_up > 1:
            roi = roi.resize((roi_w * scale_up, roi_h * scale_up), Image.LANCZOS)

        # OCR（竖直文字双向旋转取最优）
        plain_retry_target = None
        try:
            if is_vertical:
                roi_cw = roi.rotate(-90, expand=True, fillcolor=(255, 255, 255))
                roi_ccw = roi.rotate(90, expand=True, fillcolor=(255, 255, 255))
                plain_retry_enabled = bool(re.search(r'[RrCc]', text))
                if plain_retry_enabled:
                    target_match = re.match(
                        r'\s*([RrCc])\s*(\d+(?:\.\d+)?)',
                        text,
                    )
                    if target_match:
                        plain_retry_target = target_match.groups()
                _bump('ocr_calls', 2 + (1 if plain_retry_enabled else 0))
                out_cw = ocr.ocr(np.array(roi_cw), cls=False)
                out_ccw = ocr.ocr(np.array(roi_ccw), cls=False)

                def _best_conf(ocr_out):
                    if not ocr_out or not ocr_out[0]:
                        return 0.0
                    confs = [line[1][1] for line in ocr_out[0]
                             if line and len(line) >= 2]
                    return max(confs) if confs else 0.0

                attempts = [
                    (out_cw, -90.0),
                    (out_ccw, 90.0),
                ]
                if plain_retry_enabled:
                    out_plain = ocr.ocr(np.array(roi), cls=False)
                    attempts.append((out_plain, 0.0))
                if plain_retry_enabled:
                    ocr_output, angle_applied = max(
                        attempts,
                        key=lambda item: (
                            _ocr_output_dimension_score(item[0]),
                            _best_conf(item[0]),
                        ),
                    )
                elif _best_conf(out_cw) >= _best_conf(out_ccw):
                    ocr_output = out_cw
                    angle_applied = -90.0
                else:
                    ocr_output = out_ccw
                    angle_applied = 90.0
            else:
                _bump('ocr_calls')
                ocr_output = ocr.ocr(np.array(roi), cls=False)
                angle_applied = 0.0
        except Exception as e:
            _bump('ocr_errors')
            if verbose:
                print(f"[warn] strip_ocr failed for '{r['text']}': {e}")
            continue

        if not ocr_output or ocr_output[0] is None:
            _bump('empty_outputs')
            continue

        # 原始可疑结果的 bbox 中心（用于去重）
        orig_cx = bbox['x'] + bbox['w'] / 2
        orig_cy = bbox['y'] + bbox['h'] / 2

        for line in ocr_output[0]:
            if not line or len(line) < 2:
                continue
            quad_in_roi, (text, conf) = line[0], line[1]
            _bump('raw_lines')
            # GD&T 有几何锚点确认，置信度门控降至 0.15
            if not text or conf < 0.15:
                _bump('rejected_low_conf')
                continue

            # 还原 scale_up
            quad_unscaled = [[pt[0] / scale_up, pt[1] / scale_up] for pt in quad_in_roi]
            bbox_in_roi = quad_to_bbox(quad_unscaled)

            # 反向旋转
            if angle_applied != 0.0:
                orig_roi_w = right - left
                orig_roi_h = bottom - top
                bbox_in_roi = rotate_bbox_back(bbox_in_roi, angle_applied,
                                               orig_roi_w, orig_roi_h)

            # ROI 坐标 → 页面像素坐标 → PDF 坐标
            bbox_page_px = {
                'x': bbox_in_roi['x'] + origin_px[0],
                'y': bbox_in_roi['y'] + origin_px[1],
                'w': bbox_in_roi['w'],
                'h': bbox_in_roi['h'],
            }
            bbox_pdf = pixel_bbox_to_pdf(bbox_page_px, dpi)

            # 去重：文本相同且中心距 < 5pt → 丢弃
            new_cx = bbox_pdf['x'] + bbox_pdf['w'] / 2
            new_cy = bbox_pdf['y'] + bbox_pdf['h'] / 2
            dist = math.hypot(new_cx - orig_cx, new_cy - orig_cy)
            text_clean = text.strip()
            if angle_applied == 0.0 and plain_retry_target:
                compact_retry_text = re.sub(r'\s+', '', text_clean)
                _symbol, _nominal = plain_retry_target
                if not (
                    compact_retry_text.startswith(_nominal)
                    or re.match(r'^[RrCc]\d', compact_retry_text)
                ):
                    _bump('rejected_plain_retry_off_target')
                    continue
            if text_clean == r['text'].strip() and dist < 5.0:
                continue

            result = {
                'text': text_clean,
                'confidence': float(conf),
                'bbox_in_pdf': bbox_pdf,
                'font_height_approx': bbox_pdf['h'],
                'source_region_id': -1,
                'orientation': float(orientation),
                'low_confidence_region': False,
                'source': 'strip_ocr',
            }
            results.append(attach_r2p_digit_restricted_ocr_shadow(result, text_clean))

    if verbose:
        print(f"[strip_ocr] 补充结果: {len(results)} 个")
        for r in results:
            print(f"  text='{r['text']}' conf={r['confidence']:.2f} "
                  f"bbox=({r['bbox_in_pdf']['x']:.0f},{r['bbox_in_pdf']['y']:.0f},"
                  f"{r['bbox_in_pdf']['w']:.0f}x{r['bbox_in_pdf']['h']:.0f})")

    return results


# ---------------------------------------------------------------------------
# GD&T compartment 强制 OCR
# ---------------------------------------------------------------------------

def run_gdt_compartment_ocr(
    page_img: Image.Image,
    gdt_frames: list,
    existing_ocr: list,
    dpi: int = 200,
    verbose: bool = True,
    pdf_bytes: bytes = None,
    page_index: int = 0,
    hires_dpi: int = 400,
    deadline: float | None = None,
    diag: dict = None,
) -> list:
    """
    对已检出的 GD&T 框，按 compartment 切分后强制 OCR。

    当提供 pdf_bytes 时，对每个 cell 用 PyMuPDF clip 局部渲染 hires_dpi（400），
    绕过整页低 DPI 限制，显著提升小号文字（基准字母、公差值）识别率。

    参数:
        page_img: PIL Image，整页渲染图（当 pdf_bytes 不可用时的 fallback）
        gdt_frames: list of dict，每个 dict 包含:
            - bbox: dict {x, y, w, h} 或 list [x0, y0, x1, y1] (PDF 坐标)
            - compartments: int 或 list of dict {x, y, w, h}
            - interior_v_lines: list of float (可选，内部竖线 x 坐标)
        existing_ocr: list of dict，已有 OCR 结果（含 bbox_in_pdf）
        dpi: int，page_img 的渲染 DPI（用于 fallback 路径的坐标换算）
        verbose: bool
        pdf_bytes: PDF 二进制内容（提供时启用高 DPI 局部渲染）
        page_index: 0-based 页码
        hires_dpi: 局部渲染 DPI（默认 300，5pt 文字 → ~21px 高）

    返回:
        list of dict，新增的 OCR 结果（格式与 run_directed_ocr 输出一致）
    """
    if not gdt_frames:
        return []

    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    ocr = _get_ocr()
    results = []
    frames_with_new = 0
    total_cells_ocred = 0

    # 高 DPI 渲染：打开 fitz 文档（请求级，用完关闭）
    fitz_doc = None
    fitz_page = None
    use_hires = pdf_bytes is not None
    if use_hires:
        fitz_doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        try:
            fitz_page = fitz_doc[page_index]
        except Exception:
            fitz_doc.close()
            fitz_doc = None
            raise
        render_dpi = hires_dpi
    else:
        render_dpi = dpi

    for frame in gdt_frames:
        # 统一 bbox 格式为 {x, y, w, h}
        fb = frame.get('bbox', None)
        if fb is None:
            continue
        if isinstance(fb, list):
            # [x0, y0, x1, y1]
            fb = {'x': fb[0], 'y': fb[1], 'w': fb[2] - fb[0], 'h': fb[3] - fb[1]}

        # 构建 compartment cells
        comps = frame.get('compartments', None)
        if isinstance(comps, list) and len(comps) >= 2:
            # 已有 compartment 结构 [{x, y, w, h}, ...]
            cells = comps
        elif isinstance(comps, int) and comps >= 2:
            # 用 interior_v_lines 或等分
            v_lines = frame.get('interior_v_lines', [])
            all_vx = [fb['x']] + sorted(v_lines) + [fb['x'] + fb['w']]
            cells = []
            for k in range(len(all_vx) - 1):
                cw = all_vx[k + 1] - all_vx[k]
                if cw >= 5:
                    cells.append({
                        'x': all_vx[k], 'y': fb['y'],
                        'w': cw, 'h': fb['h'],
                    })
        else:
            continue

        if len(cells) < 2:
            continue

        frame_has_new = False
        for cell in cells:
            if deadline is not None and time.monotonic() >= deadline:
                _bump('stopped_deadline')
                break
            if cell['w'] < 5 or cell['h'] < 5:
                continue

            # 检查是否已被 existing_ocr 覆盖 (IoA >= 0.8)
            covered = False
            for er in existing_ocr:
                eb = er.get('bbox_in_pdf', er.get('bbox', None))
                if eb is None:
                    continue
                # intersection
                ix0 = max(cell['x'], eb['x'])
                iy0 = max(cell['y'], eb['y'])
                ix1 = min(cell['x'] + cell['w'], eb['x'] + eb['w'])
                iy1 = min(cell['y'] + cell['h'], eb['y'] + eb['h'])
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                inter = (ix1 - ix0) * (iy1 - iy0)
                eb_area = eb['w'] * eb['h']
                if eb_area > 0 and inter / eb_area >= 0.8:
                    covered = True
                    break

            if covered:
                continue

            # 切 ROI：cell bbox + 3pt padding（小号 GD&T 文字容易被边缘裁掉）
            pad_pt = 3.0
            crop_x0 = cell['x'] - pad_pt
            crop_y0 = cell['y'] - pad_pt
            crop_x1 = cell['x'] + cell['w'] + pad_pt
            crop_y1 = cell['y'] + cell['h'] + pad_pt

            if use_hires:
                # 高 DPI 局部渲染：用 PyMuPDF clip 直接渲染 cell 区域
                clip_rect = fitz.Rect(crop_x0, crop_y0, crop_x1, crop_y1)
                mat = fitz.Matrix(hires_dpi / 72.0, hires_dpi / 72.0)
                try:
                    pix = fitz_page.get_pixmap(matrix=mat, clip=clip_rect)
                    if pix.width < 5 or pix.height < 5:
                        continue
                    roi = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except Exception:
                    if fitz_doc:
                        fitz_doc.close()
                        fitz_doc = None
                    raise
                # clip 渲染的原点就是 crop 区域左上角，无需 page 级偏移
                # bbox 换算以 crop 区域为参照
                origin_pdf = (crop_x0, crop_y0)
                scale_up = 1  # 已经是高 DPI，不需要上采样
            else:
                # 原 fallback：从整页 page_img 裁切
                crop_bbox = {'x': crop_x0, 'y': crop_y0,
                             'w': crop_x1 - crop_x0, 'h': crop_y1 - crop_y0}
                px, py, pw, ph = pdf_bbox_to_pixel(crop_bbox, dpi)
                page_w, page_h = page_img.size
                left = max(0, px)
                top = max(0, py)
                right = min(page_w, px + pw)
                bottom = min(page_h, py + ph)

                if right - left < 5 or bottom - top < 5:
                    continue

                roi = page_img.crop((left, top, right, bottom))
                origin_pdf = (crop_x0, crop_y0)

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

            # GD&T 图像预处理：转灰度 → 自适应二值化 → 去除细框线干扰
            roi_arr = np.array(roi)
            if len(roi_arr.shape) == 3:
                gray = np.mean(roi_arr, axis=2).astype(np.uint8)
            else:
                gray = roi_arr
            # 使用固定灰度阈值生成黑白 ROI，随后恢复模型所需的通道数。
            binary = np.where(gray < 160, 0, 255).astype(np.uint8)
            # 转回 RGB（PaddleOCR 需要 3 通道）
            roi_clean = np.stack([binary, binary, binary], axis=2)

            # OCR（水平方向，不旋转）—— 先试预处理版，失败回退原图
            try:
                _bump('ocr_calls')
                ocr_output = ocr.ocr(roi_clean, cls=False)
            except Exception:
                _bump('ocr_errors')
                ocr_output = None
            if not ocr_output or ocr_output[0] is None:
                # 预处理可能过度清除，回退到原图
                try:
                    _bump('ocr_calls')
                    ocr_output = ocr.ocr(roi_arr, cls=False)
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
                # GD&T 有几何锚点确认，置信度门控降至 0.15
                if not text or conf < 0.15:
                    _bump('rejected_low_conf')
                    continue

                # 还原 scale_up + 像素坐标 → PDF 坐标
                quad_unscaled = [[pt[0] / scale_up, pt[1] / scale_up] for pt in quad]
                bbox_in_roi = quad_to_bbox(quad_unscaled)

                # ROI 像素坐标 → PDF 坐标
                # render_dpi 是实际渲染 DPI（高 DPI 或全局 DPI）
                pdf_scale = 72.0 / render_dpi
                bbox_pdf = {
                    'x': origin_pdf[0] + bbox_in_roi['x'] * pdf_scale,
                    'y': origin_pdf[1] + bbox_in_roi['y'] * pdf_scale,
                    'w': bbox_in_roi['w'] * pdf_scale,
                    'h': bbox_in_roi['h'] * pdf_scale,
                }

                result = {
                    'text': text.strip(),
                    'confidence': float(conf),
                    'bbox_in_pdf': bbox_pdf,
                    'font_height_approx': bbox_pdf['h'],
                    'source_region_id': -1,
                    'orientation': 0.0,
                    'low_confidence_region': False,
                    'source': 'gdt_compartment_ocr',
                }
                results.append(attach_r2p_digit_restricted_ocr_shadow(result, text))
                frame_has_new = True

            total_cells_ocred += 1

        if frame_has_new:
            frames_with_new += 1

    if fitz_doc:
        fitz_doc.close()

    if verbose:
        dpi_info = f"hires={hires_dpi}" if use_hires else f"dpi={dpi}"
        print(f"[gdt_compartment_ocr] 补充 OCR: {len(results)} 个"
              f"（来自 {frames_with_new} 个 GD&T 框的 {total_cells_ocred} 个格子, {dpi_info}）")

    return results
