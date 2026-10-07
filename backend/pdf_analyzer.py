"""
pdf_analyzer.py — PDF 矢量解析核心模块（orchestrator）

拆分：实际逻辑分散在 7 个子模块中，本文件只做 re-export，
保证所有外部调用者（app.py / debug_*.py / diag_*.py / test_*.py / arrow_detector.py / score_dump.py）
的 `from pdf_analyzer import X` 继续工作。

子模块依赖 DAG（单向）:
  pdf_analyzer_utils          ← 常量 + 工具函数 + fitz 缓存
  pdf_analyzer_type_a         ← Type A 提取（依赖 utils）
  pdf_analyzer_extract        ← 类型探测 + 线段提取 + ⌀ 检测（依赖 utils）
  pdf_analyzer_cluster        ← DBSCAN 聚类 + 矩形分类（依赖 utils）
  pdf_analyzer_capsule        ← 跑道框检测（依赖 utils）
  pdf_analyzer_gdt_lines      ← GD&T 线段框检测（无内部依赖）
  pdf_analyzer_reconstruct    ← 线段矩形重建（依赖 utils）
"""

# ── pdf_analyzer_utils ──────────────────────────────────────────
from pdf_analyzer_utils import (
    SYMBOL_MAP,
    PREFIX_PATTERNS,
    detect_annotation_linewidth,
    _count_tiny_curves,
    _curve_bbox_wh,
    _curve_center,
    _iou,
    _rect_iou_xyxy,
    _adaptive_width_thresholds,
    _fitz_local,
    _get_fitz_page,
    close_fitz_cache,
)

# ── pdf_analyzer_type_a ────────────────────────────────────────
from pdf_analyzer_type_a import extract_type_a

# ── pdf_analyzer_extract ───────────────────────────────────────
from pdf_analyzer_extract import (
    detect_pdf_type,
    extract_diameter_glyphs,
    extract_lines_by_width,
    filter_annotation_segments_by_connectivity,
)

# ── pdf_analyzer_cluster ───────────────────────────────────────
from pdf_analyzer_cluster import (
    cluster_text_regions,
    _merge_adjacent_regions,
    extract_rects,
)

# ── pdf_analyzer_capsule ──────────────────────────────────────
from pdf_analyzer_capsule import detect_capsules

# ── pdf_analyzer_gdt_lines ────────────────────────────────────
from pdf_analyzer_gdt_lines import detect_gdt_frames_from_lines

# ── pdf_analyzer_reconstruct ──────────────────────────────────
from pdf_analyzer_reconstruct import (
    get_annotation_and_contour_lines,
    reconstruct_rectangles_from_lines,
)
