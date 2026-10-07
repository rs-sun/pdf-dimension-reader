"""ocr_engine_core.py — 主 OCR: run_directed_ocr + run_empty_region_retry

从 ocr_engine.py 拆出。"""

import math
import re
import time
import numpy as np
import fitz  # PyMuPDF — used by run_empty_region_retry
from PIL import Image

from config import Config
from ocr_engine_utils import (
    VERBOSE, _get_ocr, render_page_to_image,
    pdf_bbox_to_pixel, pixel_bbox_to_pdf, quad_to_bbox,
    rotate_bbox_back, _extract_roi,
    _build_angle_index, _find_near_angle,
    _bbox_iou,
    attach_r2p_digit_restricted_ocr_shadow,
)
from ocr_region_priority import score_text_region_for_ocr


def run_directed_ocr(
    pdf_bytes: bytes,
    page_index: int,
    text_regions: list,
    l1_lines: list = None,
    dpi: int = 200,
    table_regions: list = None,
    verbose: bool = True,
    diag: dict = None,
    deadline: float | None = None,
    skip_regions: list | None = None,
) -> list:
    """
    对 text_regions（来自 cluster_text_regions 的输出）做批量定向 OCR。

    参数:
        pdf_bytes: PDF 文件二进制内容
        page_index: 0-based 页码
        text_regions: [{'x','y','w','h','curve_count','avg_curve_size',...}, ...]
                      pdfplumber 坐标系（左上角原点，Y 向下）
        l1_lines: L1 层细线段列表（用于检测斜向标注），可为 None
        dpi: 渲染 DPI
        table_regions: 表格区域列表（来自 extract_rects()['table_regions']），落在
                       表格内的 text_region 会被跳过，避免识别大量无关数值
        skip_regions: 更高优先级 OCR 已覆盖的 PDF bbox；text_region 中心落入
                      这些 bbox 时跳过，避免重复 OCR 同一块内容

    返回:
        [
          {
            'text': str,
            'confidence': float,          # PaddleOCR 原始置信度
            'bbox_in_pdf': {'x','y','w','h'},  # PDF 坐标系 bbox
            'font_height_approx': float,   # 字号代理（bbox 高度，PDF 单位）
            'source_region_id': int,       # 对应 text_regions 的下标
            'orientation': float,          # 旋转角度（度），0 = 水平
            'low_confidence_region': bool, # region 自身是否被标记为低置信
          },
          ...
        ]
    """
    if not text_regions:
        return []

    # codex 二审 P3 telemetry：可选 diag 累加，不传则不收集（不动行为）
    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    def _overlap_ratio(a, b):
        ax0, ay0 = float(a['x']), float(a['y'])
        ax1, ay1 = ax0 + float(a['w']), ay0 + float(a['h'])
        bx0, by0 = float(b['x']), float(b['y'])
        bx1, by1 = bx0 + float(b['w']), by0 + float(b['h'])
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        area = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
        return ((ix1 - ix0) * (iy1 - iy0)) / area

    def _point_in_bbox(x, y, bbox):
        return (
            float(bbox.get('x', 0.0)) <= x <= float(bbox.get('x', 0.0)) + float(bbox.get('w', 0.0))
            and float(bbox.get('y', 0.0)) <= y <= float(bbox.get('y', 0.0)) + float(bbox.get('h', 0.0))
        )

    def _best_conf(ocr_out):
        if not ocr_out or not ocr_out[0]:
            return 0.0
        confs = [line[1][1] for line in ocr_out[0] if line and len(line) >= 2]
        return max(confs) if confs else 0.0

    _bump('regions_total', len(text_regions))
    skip_regions = [
        bbox for bbox in (skip_regions or [])
        if isinstance(bbox, dict)
        and float(bbox.get('w', 0.0) or 0.0) > 0.0
        and float(bbox.get('h', 0.0) or 0.0) > 0.0
    ]
    if diag is not None:
        diag['candidate_guided_skip_regions'] = len(skip_regions)

    # 渲染整页为高分辨率位图
    page_img = render_page_to_image(pdf_bytes, page_index, dpi)
    page_w_pdf = page_img.size[0] / (dpi / 72.0)
    page_h_pdf = page_img.size[1] / (dpi / 72.0)
    page_area = page_w_pdf * page_h_pdf
    table_regions_raw = list(table_regions or [])
    table_regions_effective = []
    table_skip_max_area_ratio = getattr(
        Config, 'OCR_TABLE_SKIP_MAX_AREA_RATIO', 0.50)
    for tb in table_regions_raw:
        tb_w = float(tb.get('w', 0) or 0)
        tb_h = float(tb.get('h', 0) or 0)
        area_ratio = (tb_w * tb_h) / max(1e-6, page_area)
        spans_page = (
            tb_w >= page_w_pdf * 0.80
            and tb_h >= page_h_pdf * 0.80
        )
        if area_ratio >= table_skip_max_area_ratio or spans_page:
            _bump('ignored_large_table_regions')
            continue
        table_regions_effective.append(tb)
    if diag is not None:
        diag['table_overlap_threshold'] = 0.80
        diag['table_regions_raw'] = len(table_regions_raw)
        diag['table_regions_effective'] = len(table_regions_effective)
        diag['table_skip_max_area_ratio'] = table_skip_max_area_ratio

    # 构建 L1 斜线角度索引（用于斜向处理）
    angle_index = _build_angle_index(l1_lines or [])

    ocr = _get_ocr()
    results = []

    for region_id, region in enumerate(text_regions):
        source_region_id = region.get('_source_region_id', region_id)
        if deadline is not None and time.monotonic() >= deadline:
            _bump('stopped_deadline')
            break
        if skip_regions:
            cx = float(region['x']) + float(region['w']) / 2.0
            cy = float(region['y']) + float(region['h']) / 2.0
            if any(_point_in_bbox(cx, cy, bbox) for bbox in skip_regions):
                _bump('skipped_candidate_guided_overlap')
                continue
        # 诊断：打印 region bbox 和像素尺寸
        _px_w = region['w'] * (dpi / 72.0)
        _px_h = region['h'] * (dpi / 72.0)
        if verbose:
            print(f"  [OCR-region {source_region_id}] bbox=({region['x']:.0f},{region['y']:.0f},"
                  f"{region['w']:.0f}x{region['h']:.0f}) px=({_px_w:.0f}x{_px_h:.0f})")

        # 跳过大面积文字块（技术说明、注释等），不浪费 OCR 资源
        region_area = region['w'] * region['h']
        if region_area > page_area * 0.01:
            _bump('skipped_region_too_big')
            continue

        # 跳过大部分落在表格区域内的 region（避免识别大量无意义数值）。
        # 旧逻辑只看中心点，边缘标注中心误入表格就会被丢；现在按 region
        # 面积重叠比例判定，并透传 counters 供诊断。
        if table_regions_effective:
            max_overlap = max((_overlap_ratio(region, tb) for tb in table_regions_effective), default=0.0)
            if max_overlap > 0:
                _bump('table_overlap_regions')
                if diag is not None:
                    diag['table_overlap_ratio_sum'] = diag.get('table_overlap_ratio_sum', 0.0) + max_overlap
            if max_overlap >= 0.80:
                _bump('skipped_in_table')
                _bump('skipped_in_table_overlap')
                continue
        _bump('regions_processed')

        low_conf_region = region.get('low_confidence', False)

        # 查找附近斜线角度（snap 到有效角度后返回 0 或 90）
        near_angle = _find_near_angle(region, angle_index, region_id=source_region_id, verbose=verbose)

        # 切 ROI + 旋转预处理
        roi_img, origin_px, angle_applied, scale_up = _extract_roi(
            page_img, region, dpi, near_angle_deg=near_angle
        )

        # 转 numpy array 给 PaddleOCR
        roi_np = np.array(roi_img)

        # 运行 PaddleOCR
        try:
            if angle_applied == 90.0:
                # 竖直文字：尝试两个旋转方向，取置信度更高的结果
                roi_cw  = roi_img.rotate(-90, expand=True, fillcolor=(255, 255, 255))
                roi_ccw = roi_img.rotate( 90, expand=True, fillcolor=(255, 255, 255))
                _bump('ocr_calls', 2)
                out_cw  = ocr.ocr(np.array(roi_cw),  cls=False)
                out_ccw = ocr.ocr(np.array(roi_ccw), cls=False)

                conf_cw  = _best_conf(out_cw)
                conf_ccw = _best_conf(out_ccw)
                if conf_cw >= conf_ccw:
                    ocr_output   = out_cw
                    angle_applied = -90.0
                else:
                    ocr_output   = out_ccw
                    angle_applied = 90.0
            elif angle_applied != 0.0:
                # Diagonal text can be rotated onto the baseline in either
                # reading direction.  Try the 180° mate and keep the stronger
                # OCR result so upside-down diagonal dimensions are not lost.
                roi_180 = roi_img.rotate(180, expand=True, fillcolor=(255, 255, 255))
                _bump('ocr_calls', 2)
                out_base = ocr.ocr(roi_np, cls=False)
                out_180 = ocr.ocr(np.array(roi_180), cls=False)
                if _best_conf(out_base) >= _best_conf(out_180):
                    ocr_output = out_base
                else:
                    ocr_output = out_180
                    angle_applied = angle_applied + 180.0
            else:
                _bump('ocr_calls')
                ocr_output = ocr.ocr(roi_np, cls=False)
        except Exception as e:
            _bump('ocr_errors')
            print(f"[warn] PaddleOCR failed on region {source_region_id}: {e}")
            continue

        if not ocr_output or ocr_output[0] is None:
            _bump('empty_outputs')
            continue

        for line in ocr_output[0]:
            # line = [[[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, conf)]
            if not line or len(line) < 2:
                continue
            quad_in_roi, (text, conf) = line[0], line[1]

            if not text:
                continue
            # 工程文本（含数字）置信度门控降至 0.15，其余保持 0.3
            _bump('raw_lines')
            import re as _re_conf
            _conf_floor = 0.15 if _re_conf.search(r'\d', text) else 0.3
            if conf < _conf_floor:
                _bump('rejected_low_conf')
                if verbose and conf >= 0.10:
                    print(f"  [OCR-low-conf] region={source_region_id} text='{text.strip()}' "
                          f"conf={conf:.2f} floor={_conf_floor} (DROPPED)")
                continue

            # 1. 还原 scale_up
            quad_in_roi_unscaled = [
                [pt[0] / scale_up, pt[1] / scale_up] for pt in quad_in_roi
            ]

            # 2. 如果做了旋转，把 bbox 反向旋转回未旋转坐标
            bbox_in_roi = quad_to_bbox(quad_in_roi_unscaled)
            if angle_applied != 0.0:
                # 计算原始 ROI 尺寸（pad 后的尺寸）
                pdf_px, pdf_py, pdf_pw, pdf_ph = pdf_bbox_to_pixel(region, dpi)
                pad = max(int(pdf_ph * 0.4), 10)
                orig_roi_w = min(page_img.size[0], pdf_px + pdf_pw + pad) - max(0, pdf_px - pad)
                orig_roi_h = min(page_img.size[1], pdf_py + pdf_ph + pad) - max(0, pdf_py - pad)
                bbox_in_roi = rotate_bbox_back(bbox_in_roi, angle_applied,
                                               orig_roi_w, orig_roi_h)

            # 3. bbox 从 ROI 坐标 → 页面像素坐标
            bbox_page_px = {
                'x': bbox_in_roi['x'] + origin_px[0],
                'y': bbox_in_roi['y'] + origin_px[1],
                'w': bbox_in_roi['w'],
                'h': bbox_in_roi['h'],
            }

            # 4. 页面像素坐标 → PDF 坐标
            bbox_pdf = pixel_bbox_to_pdf(bbox_page_px, dpi)

            # 过滤：OCR 文字中心必须落在原始 region 内
            # 水平文字也给 3pt 容差（DBSCAN region 边界不完美，0pt 太严格）
            text_cx = bbox_pdf['x'] + bbox_pdf['w'] / 2
            text_cy = bbox_pdf['y'] + bbox_pdf['h'] / 2
            _abs_angle = abs(angle_applied) % 360
            _margin = 10.0 if _abs_angle in (90, 270) else (12.0 if _abs_angle > 10 else 3.0)
            if not (region['x'] - _margin <= text_cx <= region['x'] + region['w'] + _margin and
                    region['y'] - _margin <= text_cy <= region['y'] + region['h'] + _margin):
                _bump('rejected_center_outside')
                continue

            _bump('final_lines')
            result = {
                'text': text.strip(),
                'confidence': float(conf),
                'bbox_in_pdf': bbox_pdf,
                'font_height_approx': bbox_pdf['h'],
                'source_region_id': source_region_id,
                'orientation': float(angle_applied),
                'low_confidence_region': bool(low_conf_region),
                'source': 'directed_ocr',
            }
            results.append(attach_r2p_digit_restricted_ocr_shadow(result, text))
            if verbose:
                print(f"[s27.directed_ocr] region={source_region_id} text='{text.strip()}' conf={float(conf):.2f} orient={float(angle_applied)} bbox=({bbox_pdf['x']:.0f},{bbox_pdf['y']:.0f},{bbox_pdf['w']:.0f}x{bbox_pdf['h']:.0f})")

    # --- P0 诊断日志 ---
    if verbose:
        vert_results = [r for r in results if abs(r['orientation']) > 45]
        horiz_results = [r for r in results if abs(r['orientation']) <= 45]
        print(f"[ocr_diag] 总 OCR 结果: {len(results)}, 水平: {len(horiz_results)}, 竖直: {len(vert_results)}")
        for r in vert_results:
            print(f"  [vert] text='{r['text']}' conf={r['confidence']:.2f} orient={r['orientation']} bbox=({r['bbox_in_pdf']['x']:.0f},{r['bbox_in_pdf']['y']:.0f})")

    return results


# ---------------------------------------------------------------------------
# 空 region 高分辨率重试（directed_ocr 后补充）
# ---------------------------------------------------------------------------

def run_empty_region_retry(
    pdf_bytes: bytes,
    page_index: int,
    text_regions: list,
    existing_ocr_results: list,
    hires_dpi: int = 300,
    verbose: bool = True,
    diag: dict = None,
    max_retry: int = 80,
    deadline: float | None = None,
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> list:
    """
    对 directed_ocr 未产出任何结果的 DBSCAN region，用高 DPI 局部渲染重试。

    此 legacy 重试通过局部高分辨率渲染和二值化预处理检查尚未读取的区域。

    策略：
    - 仅针对 directed_ocr 后 0 个 OCR 结果的 region（不重试已有结果的）
    - 用 PyMuPDF clip 局部渲染到 hires_dpi (300)
    - 置信度阈值 0.10（比 directed_ocr 的 0.15 更宽容，因为 hires 图像质量更好）
    - 加二值化预处理（参考 run_gdt_compartment_ocr）
    - 同时尝试 90 度旋转版（竖直文字 fallback）

    参数:
        pdf_bytes: PDF 二进制内容
        page_index: 0-based 页码
        text_regions: DBSCAN text_regions 列表
        existing_ocr_results: run_directed_ocr 的输出（用于判断哪些 region 已有结果）
        hires_dpi: 局部渲染 DPI（默认 300）
        verbose: 诊断日志

    返回:
        list of dict，格式与 run_directed_ocr 一致（source='hires_retry'）
    """
    import re as _re_retry

    # codex 二审 P3 telemetry：可选 diag 累加，不传则不收集（不动行为）
    def _bump(key, n=1):
        if diag is not None:
            diag[key] = diag.get(key, 0) + n

    if not text_regions or not pdf_bytes:
        return []

    # Step 1: 找出哪些 region 已有 directed_ocr 结果
    regions_with_results = set()
    for r in existing_ocr_results:
        rid = r.get('source_region_id')
        if rid is not None:
            regions_with_results.add(rid)

    _MAX_RETRY = max_retry
    empty_regions = []
    for rid, region in enumerate(text_regions):
        if rid in regions_with_results:
            _bump('skipped_already_has_results')
            continue
        # 跳过大面积 region（与 directed_ocr 一致的 1% 门控）
        if region['w'] * region['h'] > 5000:
            _bump('skipped_region_too_big')
            continue
        # 跳过极小 region（可能是噪点）
        if region['w'] < 3 or region['h'] < 3:
            _bump('skipped_region_too_small')
            continue
        empty_regions.append((rid, region))

    if verbose:
        print(f"[hires_retry] {len(regions_with_results)}/{len(text_regions)} regions "
              f"have directed_ocr results; {len(empty_regions)} empty regions to retry "
              f"at {hires_dpi} DPI (cap={_MAX_RETRY})")

    if not empty_regions:
        return []

    # 安全阀：超过上限时优先 OCR 疑似尺寸 region。
    # 面积排序可能使紧凑文字区域排在预算截断之后。
    if len(empty_regions) > _MAX_RETRY:
        _bump('capped_count', len(empty_regions) - _MAX_RETRY)
        empty_regions.sort(key=lambda x: (
            -score_text_region_for_ocr(x[1], page_width, page_height),
            -(x[1]['w'] * x[1]['h']),
            x[0],
        ))
        empty_regions = empty_regions[:_MAX_RETRY]
        if verbose:
            print(f"[hires_retry] capped to {_MAX_RETRY} regions")
    _bump('regions_processed', len(empty_regions))

    results = []
    # Step 3: 打开 fitz 文档做局部渲染
    fitz_doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    try:
        fitz_page = fitz_doc[page_index]

        ocr = _get_ocr()

        for rid, region in empty_regions:
            if deadline is not None and time.monotonic() >= deadline:
                _bump('stopped_deadline')
                break
            # 宽裕 padding（15pt，比 directed_ocr 的 2pt margin 大得多）
            pad = 15.0
            clip_x0 = max(0, region['x'] - pad)
            clip_y0 = max(0, region['y'] - pad)
            clip_x1 = region['x'] + region['w'] + pad
            clip_y1 = region['y'] + region['h'] + pad

            clip_rect = fitz.Rect(clip_x0, clip_y0, clip_x1, clip_y1)
            mat = fitz.Matrix(hires_dpi / 72.0, hires_dpi / 72.0)
            pix = fitz_page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)

            if pix.width < 5 or pix.height < 5:
                continue

            roi = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            roi_np = np.array(roi)

            # 二值化预处理：灰度 → 阈值 160 → 黑白
            if len(roi_np.shape) == 3:
                gray = np.mean(roi_np, axis=2).astype(np.uint8)
            else:
                gray = roi_np
            binary = np.where(gray < 160, 0, 255).astype(np.uint8)
            roi_clean = np.stack([binary, binary, binary], axis=2)

            # OCR：先试二值化版，成功就用；失败回退原图（与 gdt_compartment_ocr 一致）。
            # all_lines 现在是 [(line, angle_applied), ...]，angle 用来在下面做 bbox 反变换。
            all_lines = []
            try:
                _bump('ocr_calls')
                ocr_output = ocr.ocr(roi_clean, cls=False)
            except Exception:
                _bump('ocr_errors')
                ocr_output = None
            if not ocr_output or ocr_output[0] is None:
                try:
                    _bump('ocr_calls')
                    ocr_output = ocr.ocr(roi_np, cls=False)
                except Exception:
                    _bump('ocr_errors')
                    ocr_output = None
            if ocr_output and ocr_output[0]:
                for line in ocr_output[0]:
                    all_lines.append((line, 0.0))

            # 竖直 region (h > w*2.5)：对 -90 / +90 双向 OCR 取最高 conf。
            # 旧版本只跑 -90 且不 bbox 反变换 → 竖向 dim 漏读 + 命中后旋转 quad 被当成
            # 未旋转 ROI 坐标解释，污染中心门控和 PDF 坐标。orientation 标 ±90 让下游
            # 正确识别竖向文本。PIL.rotate(angle) 是 CCW 正方向；rotate_bbox_back 输入
            # 是同样的 angle 约定，传 ROI 原始（未旋转）尺寸。
            if not all_lines and region.get('h', 0) > region.get('w', 1) * 2.5:
                best_lines = []
                best_angle = 0.0
                best_conf = -1.0
                for try_angle in (-90.0, 90.0):
                    roi_rot = roi.rotate(try_angle, expand=True, fillcolor=(255, 255, 255))
                    try:
                        _bump('ocr_calls')
                        rot_output = ocr.ocr(np.array(roi_rot), cls=False)
                    except Exception:
                        _bump('ocr_errors')
                        continue
                    if not rot_output or not rot_output[0]:
                        continue
                    cands = [l for l in rot_output[0] if l and len(l) >= 2]
                    if not cands:
                        continue
                    agg = max((float(l[1][1]) for l in cands), default=-1.0)
                    if agg > best_conf:
                        best_conf = agg
                        best_lines = cands
                        best_angle = try_angle
                for line in best_lines:
                    all_lines.append((line, best_angle))

            if not all_lines:
                _bump('empty_outputs')
                if verbose:
                    print(f"  [hires_retry] region={rid} ({region['x']:.0f},{region['y']:.0f},"
                          f"{region['w']:.0f}x{region['h']:.0f}) → no results even at {hires_dpi} DPI")
                continue

            # 去重：同 region 内按文本去重（hires + 原图可能重复）
            seen_texts = set()
            for entry in all_lines:
                line, angle_applied = entry
                if not line or len(line) < 2:
                    continue
                quad, (text, conf) = line[0], line[1]
                _bump('raw_lines')
                if not text or conf < 0.10:
                    _bump('rejected_low_conf')
                    continue

                text_clean = text.strip()
                if text_clean in seen_texts:
                    _bump('rejected_dup')
                    continue
                seen_texts.add(text_clean)

                # 坐标变换：clip 渲染的像素原点 = clip_rect 左上角。
                # quad 来自 OCR，可能在旋转 ROI 坐标系下；如有旋转用 rotate_bbox_back
                # 反变换到未旋转 ROI 坐标，再算 PDF 坐标，否则中心门控会用错坐标系
                bbox_in_roi = quad_to_bbox(quad)
                if angle_applied != 0.0:
                    bbox_in_roi = rotate_bbox_back(
                        bbox_in_roi, angle_applied, pix.width, pix.height
                    )
                scale = 72.0 / hires_dpi
                bbox_pdf = {
                    'x': clip_x0 + bbox_in_roi['x'] * scale,
                    'y': clip_y0 + bbox_in_roi['y'] * scale,
                    'w': bbox_in_roi['w'] * scale,
                    'h': bbox_in_roi['h'] * scale,
                }

                # 宽容的中心门控：5pt 容差
                text_cx = bbox_pdf['x'] + bbox_pdf['w'] / 2
                text_cy = bbox_pdf['y'] + bbox_pdf['h'] / 2
                if not (region['x'] - 5.0 <= text_cx <= region['x'] + region['w'] + 5.0 and
                        region['y'] - 5.0 <= text_cy <= region['y'] + region['h'] + 5.0):
                    _bump('rejected_center_outside')
                    if verbose:
                        print(f"  [hires_retry] region={rid} text='{text_clean}' conf={conf:.2f} "
                              f"REJECTED (center outside region)")
                    continue

                _bump('final_lines')
                if verbose:
                    print(f"  [hires_retry] region={rid} text='{text_clean}' conf={conf:.2f} "
                          f"bbox=({bbox_pdf['x']:.0f},{bbox_pdf['y']:.0f},"
                          f"{bbox_pdf['w']:.0f}x{bbox_pdf['h']:.0f})")

                result = {
                    'text': text_clean,
                    'confidence': float(conf),
                    'bbox_in_pdf': bbox_pdf,
                    'font_height_approx': bbox_pdf['h'],
                    'source_region_id': rid,
                    'orientation': float(angle_applied),
                    'low_confidence_region': False,
                    'source': 'hires_retry',
                }
                results.append(attach_r2p_digit_restricted_ocr_shadow(result, text_clean))
    finally:
        fitz_doc.close()

    if verbose:
        print(f"[hires_retry] recovered {len(results)} additional OCR results "
              f"from {len(empty_regions)} empty regions")

    return results
