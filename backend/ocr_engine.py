"""
ocr_engine.py — 定向 OCR 引擎（orchestrator）

拆分：实际逻辑分散在 6 个子模块中，本文件只做 re-export。

子模块依赖 DAG:
  ocr_engine_utils    ← OCR 单例 + 坐标变换 + bbox 工具
  ocr_engine_core     ← run_directed_ocr + run_empty_region_retry（依赖 utils）
  ocr_engine_strip    ← run_strip_ocr + run_gdt_compartment_ocr（依赖 utils）
  ocr_engine_special  ← run_capsule_ocr + run_dimline_strip_ocr（依赖 utils）
  ocr_engine_rotate   ← 旋转变换 + run_rotated_ocr（依赖 utils）
  ocr_engine_dedup    ← 去重 + inject_diameter_prefix（依赖 utils）
"""

# 调试输出控制：debug 脚本通过 `import ocr_engine; ocr_engine.VERBOSE = True` 开启
VERBOSE = False


def _sync_verbose():
    """将 orchestrator 的 VERBOSE 传播到所有子模块。"""
    import ocr_engine_utils as _u
    import ocr_engine_core as _c
    import ocr_engine_strip as _st
    import ocr_engine_special as _sp
    import ocr_engine_rotate as _r
    import ocr_engine_dedup as _d
    _u.VERBOSE = VERBOSE
    _c.VERBOSE = VERBOSE
    _st.VERBOSE = VERBOSE
    _sp.VERBOSE = VERBOSE
    _r.VERBOSE = VERBOSE
    _d.VERBOSE = VERBOSE


# ── ocr_engine_utils ───────────────────────────────────────────
from ocr_engine_utils import (
    _RapidOCRShim,
    _ThreadSafeOCR,
    _get_ocr,
    render_page_to_image,
    pdf_bbox_to_pixel,
    pixel_bbox_to_pdf,
    quad_to_bbox,
    rotate_bbox_back,
    _extract_roi,
    _build_angle_index,
    _find_near_angle,
    _bbox_iou,
    _bbox_containment,
    _ENGINEERING_CHARS_RE,
    _effective_eng_len,
    _has_tolerance,
)

# ── ocr_engine_core ────────────────────────────────────────────
from ocr_engine_core import (
    run_directed_ocr,
    run_empty_region_retry,
)

# ── ocr_engine_strip ──────────────────────────────────────────
from ocr_engine_strip import (
    run_strip_ocr,
    run_gdt_compartment_ocr,
)

# ── ocr_engine_special ────────────────────────────────────────
from ocr_engine_special import (
    run_capsule_ocr,
    run_dimline_strip_ocr,
    run_angle_label_ocr,
    run_radius_label_ocr,
)

# ── ocr_engine_rotate ─────────────────────────────────────────
from ocr_engine_rotate import (
    transform_region_cw90,
    transform_region_cw270,
    transform_ocr_back_cw90,
    transform_ocr_back_cw270,
    run_rotated_ocr,
)

# ── ocr_engine_dedup ──────────────────────────────────────────
from ocr_engine_dedup import (
    dedup_ocr_by_iou,
    dedup_ocr_engineering,
    inject_diameter_prefix,
)
