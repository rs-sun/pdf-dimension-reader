"""兼容语义组装入口。将文本和几何候选送入装载、评分、公差配对与去重模块，输出统一的尺寸、参考与形位框结构。该兼容链不定义严格矢量路径。"""

import math
import re

# ---------------------------------------------------------------------------
# 从子模块导入（同时充当 re-export，外部调用者 from assembler import X 仍可用）
# ---------------------------------------------------------------------------

from assembler_utils import (
    DIM_MIN, DIM_MAX, SCORE_THRESHOLD, FRAME_KEYWORDS,
    METADATA_BLACKLIST_PATTERNS,
    _GDT_SYMBOL_CHARS, _CONF_ORDER,
    _bbox_center, _is_vertical_orient, _point_in_bbox,
    _extract_prefix, _parse_nominal, _nxfont_correction,
    _fixup_ocr_text, _clean_dimension_text,
    _validate_engineering,
    _bbox_iou_min, _levenshtein, _bbox_iou,
    _is_junk_by_symbol_ratio, _union_bbox,
    _parse_tolerance_inline, _is_gdt_fragment, _quick_dim_score,
)

from assembler_scoring import (
    _attach_repeat_prefix_fragments,
    _expanded_merged_dimensions,
    _post_filter_all,
    _score_and_filter,
)

from assembler_loaders import (
    _load_capsules, _load_gdt_frames, build_augmented_gdt_frames,
    _load_gdt_line_frames, _merge_yolo_gdt,
)

from assembler_loaders_dim import (
    _load_datum_boxes, _load_basic_dimensions,
    _in_exclusion_zone, _load_dimension_lines, _merge_gdt_chain,
)

from assembler_dedup import (
    _recover_orphan_tolerances, _upgrade_tol_if_safe,
    _dedup_dimensions, _font_size_pairing,
    _merge_angle_tolerance_fragments,
)


# 入口传播诊断开关，保持各子模块的局部绑定一致。






VERBOSE = False


def _sync_verbose():
    """将本模块的 VERBOSE 传播到所有子模块。"""
    import assembler_utils as _u, assembler_scoring as _s
    import assembler_loaders as _l, assembler_loaders_dim as _ld
    import assembler_dedup as _d
    _u.VERBOSE = VERBOSE
    _s.VERBOSE = VERBOSE
    _l.VERBOSE = VERBOSE
    _ld.VERBOSE = VERBOSE
    _d.VERBOSE = VERBOSE


# ---------------------------------------------------------------------------
# Type A 语义组装
# ---------------------------------------------------------------------------

def assemble_type_a(type_a_result: dict) -> dict:
    """将原生文本分析结果规范化为统一输出，执行文本清洗、前缀纠正和数值合理性校验。"""
    _sync_verbose()

    raw_dims = type_a_result.get('dimensions', [])
    references = type_a_result.get('references', [])
    gdt_frames = type_a_result.get('gdt_frames', [])

    dimensions = []
    for d in raw_dims:
        nominal_raw = _clean_dimension_text(d.get('nominal', ''))
        nominal_raw = _nxfont_correction(nominal_raw)

        upper_tol = d.get('upper_tol')
        lower_tol = d.get('lower_tol')
        prefix = d.get('prefix')
        dim_type = d.get('type', 'linear')

        nominal_val = _parse_nominal(nominal_raw)
        validity = _validate_engineering(nominal_val, upper_tol, lower_tol)

        # 构造 text
        text_parts = []
        if prefix:
            text_parts.append(prefix)
        text_parts.append(nominal_raw)
        if upper_tol and lower_tol:
            if upper_tol == f"+{abs(float(lower_tol.replace('-', '').replace('+', ''))):.10g}" or \
               (upper_tol.lstrip('+') == lower_tol.lstrip('-')):
                # 对称公差
                val = upper_tol.lstrip('+')
                text_parts.append(f"±{val}")
            else:
                text_parts.append(f"{upper_tol}/{lower_tol}")
        elif upper_tol:
            text_parts.append(upper_tol)
        elif lower_tol:
            text_parts.append(lower_tol)

        full_text = ' '.join(text_parts)

        confidence = d.get('confidence', 'medium')
        if validity == 'low':
            confidence = 'low'

        dimensions.append({
            'text': full_text,
            'nominal': nominal_raw,
            'upper_tol': upper_tol,
            'lower_tol': lower_tol,
            'type': dim_type,
            'prefix': prefix,
            'bbox': d.get('bbox', {'x': 0, 'y': 0, 'w': 0, 'h': 0}),
            'confidence': confidence,
            'source': 'vector',
            'orientation': d.get('orientation', 0),
            '_font_height': d.get('_font_height', d.get('fontsize', 0)),
            'fontnames': d.get('fontnames', []),
            'has_unknown_symbol': d.get('has_unknown_symbol', False),
        })

    return {
        'dimensions': dimensions,
        'references': references,
        'gdt_frames': gdt_frames,
    }


# ---------------------------------------------------------------------------
# Type B 语义组装（OCR + 几何融合）— 编排入口
# ---------------------------------------------------------------------------

def assemble_type_b(
    ocr_results: list,
    capsule_candidates: list,
    gdt_frame_candidates: list,
    datum_candidates: list,
    l1_lines: list,
    frame_borders: list,
    page_width: float,
    page_height: float,
    gdt_frames_from_lines: list = None,
    page_fitz=None,
    yolo_gdt: list = None,
    table_cells: list = None,
    reconstructed_rects: list = None,
    detected_dim_lines: list = None,
    pdf_bytes: bytes = None,
    page_index: int = 0,
) -> dict:
    """组装兼容路径的文字、容器、线段与形位框候选。输入包括页面几何、OCR 结果与可选重建矩形；返回尺寸、参考和形位框列表。"""
    _sync_verbose()

    import time as _time
    _timings = {}
    if VERBOSE:
        print(f"[assemble] capsules={len(capsule_candidates)}")

    # [DIAG-MISS] 漏检尺寸诊断
    if VERBOSE:
        _miss_kw = ['2X', '7X', '⌀6', '⌀ 6', '1.87']
        for r in ocr_results:
            t = r.get('text', '')
            if any(k in t for k in _miss_kw):
                if VERBOSE:
                    print(f"  [DIAG-MISS] OCR入口: '{t}' orient={r.get('orientation',0)} "
                          f"bbox=({r['bbox_in_pdf']['x']:.0f},{r['bbox_in_pdf']['y']:.0f}) "
                          f"claimed={r.get('_claimed',False)}")

    # 先做 OCR 文本清洗 + NX 纠错
    # _raw_text 保留**清洗前**的原始 OCR 文本，供下游 paren-reference 拦截使用：
    # PaddleOCR 偶尔会在 _clean_dimension_text 之后才丢括号，但更多场景是 OCR
    # 引擎本身就漏读首/尾括号——两种都要靠装载链上的 _raw_text 兜底。
    cleaned_ocr = []
    for r in ocr_results:
        raw_text = r.get('text', '')
        text = _clean_dimension_text(raw_text)
        text = _nxfont_correction(text)
        if not text:
            continue
        cleaned_ocr.append({**r, 'text': text, '_raw_text': raw_text,
                            '_claimed': False})

    # 装载容器几何锚点。
    dimensions = []
    gdt_frames_out = []
    references_out = []

    t0 = _time.time()
    _load_capsules(cleaned_ocr, capsule_candidates, dimensions)
    _timings['1a_capsules'] = round(_time.time() - t0, 4)

    t0 = _time.time()
    _load_gdt_frames(cleaned_ocr, gdt_frame_candidates, gdt_frames_out)
    _timings['1b_gdt_rect'] = round(_time.time() - t0, 4)

    # GD&T 线段网格框认领（新增管线）
    t0 = _time.time()
    if gdt_frames_from_lines or reconstructed_rects:
        gdt_line_dims = _load_gdt_line_frames(
            gdt_frames_from_lines or [], cleaned_ocr, page_width, page_height,
            reconstructed_rects=reconstructed_rects,
            capsule_candidates=capsule_candidates,
            frame_borders=frame_borders,
        )
        # YOLO-B 合并：将 YOLO 检出的符号类别写入 GD&T dimension
        if yolo_gdt and gdt_frames_from_lines:
            _merge_yolo_gdt(gdt_line_dims, gdt_frames_from_lines, yolo_gdt)
        dimensions.extend(gdt_line_dims)
        if VERBOSE:
            print(f"[assemble] gdt_line_frame_claimed={len(gdt_line_dims)}")
    else:
        if VERBOSE:
            print("[assemble] gdt_line_frame_claimed=0 (no gdt_frames_from_lines)")
    _timings['1b2_gdt_line'] = round(_time.time() - t0, 4)

    t0 = _time.time()
    _load_datum_boxes(cleaned_ocr, datum_candidates, references_out,
                      reconstructed_rects=reconstructed_rects)
    _timings['1c_datum'] = round(_time.time() - t0, 4)

    # 识别被矩形框包围的理论精确尺寸。
    t0 = _time.time()
    _load_basic_dimensions(cleaned_ocr, table_cells or [], gdt_frame_candidates,
                           frame_borders, dimensions,
                           reconstructed_rects=reconstructed_rects,
                           out_refs=references_out)
    _timings['1c2_basic_dim'] = round(_time.time() - t0, 4)

    t0 = _time.time()
    _load_dimension_lines(cleaned_ocr, l1_lines, dimensions,
                          page_width=page_width, page_height=page_height,
                          frame_borders=frame_borders,
                          detected_dim_lines=detected_dim_lines)
    _timings['1d_dimlines'] = round(_time.time() - t0, 4)

    # 将相邻未认领的形位碎片合并成框。
    t0 = _time.time()
    unclaimed_pre = [r for r in cleaned_ocr if not r['_claimed']]
    unclaimed_pre = _merge_gdt_chain(unclaimed_pre)
    _timings['1e_gdt_chain'] = round(_time.time() - t0, 4)

    # 评分前回收孤立公差，避免丢失其宿主公称值。
    t0 = _time.time()
    dimensions = _recover_orphan_tolerances(dimensions, page_width, page_height)
    _timings['1f_orphan_tol'] = round(_time.time() - t0, 4)

    # 对未认领文字进行评分。
    t0 = _time.time()
    unclaimed = unclaimed_pre
    frame_bboxes = frame_borders or []
    scored_dims = _score_and_filter(unclaimed, frame_bboxes, page_width, page_height)
    dimensions.extend(scored_dims)
    _timings['2_scoring'] = round(_time.time() - t0, 4)

    # 按字号配对并组装尚未合并的公差。
    t0 = _time.time()
    dimensions = _font_size_pairing(dimensions)
    _timings['3_pairing'] = round(_time.time() - t0, 4)

    # 本编排不执行额外公差微框 OCR。

    # 检查组装字段的数值合理性。
    t0 = _time.time()
    for d in dimensions:
        nominal_val = _parse_nominal(d.get('nominal', ''))
        validity = _validate_engineering(
            nominal_val, d.get('upper_tol'), d.get('lower_tol')
        )
        if validity == 'low' and d.get('confidence') != 'low':
            d['confidence'] = 'low'
    _timings['4_validate'] = round(_time.time() - t0, 4)

    # 在局部邻域内，把被更完整候选包含的短碎片标记为重复。
    
    t0 = _time.time()
    _frag_remove = set()
    for i, dim_a in enumerate(dimensions):
        text_a = dim_a.get('text', '').strip()
        compact_a = re.sub(r'\s+', '', text_a)
        _is_short_numeric_fragment = bool(re.fullmatch(r'\d{1,3}', compact_a))
        _is_short_symbol_fragment = bool(
            re.fullmatch(r'[⌀∅ØφΦR]\d{1,2}', compact_a, flags=re.IGNORECASE)
            and not (dim_a.get('upper_tol') or dim_a.get('lower_tol'))
        )
        if not text_a or not (_is_short_numeric_fragment or _is_short_symbol_fragment):
            continue
        bbox_a = dim_a.get('bbox', {})
        cx_a = bbox_a.get('x', 0) + bbox_a.get('w', 0) / 2
        cy_a = bbox_a.get('y', 0) + bbox_a.get('h', 0) / 2
        for j, dim_b in enumerate(dimensions):
            if i == j:
                continue
            text_b = dim_b.get('text', '').strip()
            compact_b = re.sub(r'\s+', '', text_b)
            if compact_a not in compact_b or len(compact_b) <= len(compact_a):
                continue
            bbox_b = dim_b.get('bbox', {})
            cx_b = bbox_b.get('x', 0) + bbox_b.get('w', 0) / 2
            cy_b = bbox_b.get('y', 0) + bbox_b.get('h', 0) / 2
            if math.hypot(cx_a - cx_b, cy_a - cy_b) < 50:
                _frag_remove.add(i)
                break
    if _frag_remove:
        dimensions = [d for i, d in enumerate(dimensions) if i not in _frag_remove]

    # 拆出合并文字中的尺寸片段，使其参与公差回收。
    dimensions = _expanded_merged_dimensions(dimensions)
    dimensions = _attach_repeat_prefix_fragments(dimensions, cleaned_ocr)

    # 再次回收评分阶段新产生的公差碎片。
    dimensions = _recover_orphan_tolerances(dimensions, page_width, page_height)

    # 对各来源执行统一后置过滤。
    assembler_debug = {}
    dimensions = _post_filter_all(dimensions, page_width, page_height, debug=assembler_debug)
    dimensions = _merge_angle_tolerance_fragments(dimensions)

    # 按重叠关系去重并保留优先候选。
    dimensions = _dedup_dimensions(dimensions)
    _timings['5_dedup'] = round(_time.time() - t0, 4)

    if VERBOSE:
        print(f"[assembler] step timings: {_timings}")

    # 清理内部调试字段，不泄漏到 API 输出
    _internal_keys = ['_score', '_font_height', '_paired', '_size_layer',
                      '_merged', '_claimed', '_raw_text',
                      '_near_tolerance_fragment_evidence',
                      '_repeat_prefix_fragment_evidence',
                      '_angle_tolerance_merged']
    for d in dimensions:
        for k in _internal_keys:
            d.pop(k, None)
    basic_dimensions_out = [
        ref for ref in references_out
        if ref.get('type') == 'basic_dimension'
    ]

    return {
        'dimensions': dimensions,
        'references': references_out,
        'basic_dimensions': basic_dimensions_out,
        'gdt_frames': gdt_frames_out,
        'assembler_debug': assembler_debug,
    }


# ---------------------------------------------------------------------------
# 主入口：统一调度（给 app.py 调用）
# ---------------------------------------------------------------------------

def assemble(
    pdf_type: str,
    # Type A 参数
    type_a_result: dict = None,
    # Type B 参数
    ocr_results: list = None,
    capsule_candidates: list = None,
    gdt_frame_candidates: list = None,
    datum_candidates: list = None,
    l1_lines: list = None,
    frame_borders: list = None,
    page_width: float = 0.0,
    page_height: float = 0.0,
    gdt_frames_from_lines: list = None,
    page_fitz=None,
    yolo_gdt: list = None,
    table_cells: list = None,
    reconstructed_rects: list = None,
    detected_dim_lines: list = None,
    pdf_bytes: bytes = None,
    page_index: int = 0,
) -> dict:
    """
    统一入口：根据 pdf_type 调用对应的组装器。

    返回:
        {'dimensions': [...], 'references': [...], 'gdt_frames': [...]}
    """
    if pdf_type == 'A' and type_a_result:
        return assemble_type_a(type_a_result)

    # Type B / BC
    return assemble_type_b(
        ocr_results=ocr_results or [],
        capsule_candidates=capsule_candidates or [],
        gdt_frame_candidates=gdt_frame_candidates or [],
        datum_candidates=datum_candidates or [],
        l1_lines=l1_lines or [],
        frame_borders=frame_borders or [],
        page_width=page_width,
        page_height=page_height,
        gdt_frames_from_lines=gdt_frames_from_lines or [],
        page_fitz=page_fitz,
        yolo_gdt=yolo_gdt,
        table_cells=table_cells or [],
        reconstructed_rects=reconstructed_rects or [],
        detected_dim_lines=detected_dim_lines,
        pdf_bytes=pdf_bytes,
        page_index=page_index,
    )
