"""
config.py — 后端集中配置

目的: 把分散在 app.py / debug_*.py / 各模块里的常量 / env 变量集中到一个类。
原则: 只放需要跨模块或需要运行时可调的配置。子模块内部私用阈值不强行挪。

使用:
    from config import Config
    dpi = Config.DPI_DEFAULT
    if Config.VERBOSE: print(...)

或通过环境变量覆写：
    APP_VERBOSE=1 python app.py
    OCR_DPI_DEFAULT=300 python app.py
"""

import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


_OCR_DEEP_SCAN = _env_bool('OCR_DEEP_SCAN', False)


class Config:
    # ── App / production profile ─────────────────────────────────
    APP_ENV = os.environ.get('APP_ENV', 'development').strip().lower() or 'development'
    PRODUCTION_MODE = _env_bool('PDF_READER_PRODUCTION_MODE', False)
    PRODUCTION_FINAL_CONSUME_APPROVED = _env_bool(
        'PRODUCTION_FINAL_CONSUME_APPROVED',
        False,
    )
    VECTOR_PRELABEL_FAST = _env_bool('VECTOR_PRELABEL_FAST', False)
    VECTOR_PRELABEL_API_ENABLED = _env_bool('VECTOR_PRELABEL_API_ENABLED', False)

    # ── OCR / 渲染 ────────────────────────────────────────────────
    DPI_DEFAULT = _env_int('OCR_DPI_DEFAULT', 200)   # 主管线默认 DPI
    DPI_HIRES   = _env_int('OCR_DPI_HIRES', 300)    # 空 region retry + GD&T compartment
    DPI_YOLO    = _env_int('OCR_DPI_YOLO', 150)     # YOLO-B 推理
    OCR_MAX_SECONDS = _env_int('OCR_MAX_SECONDS', 180)  # 单页 OCR pass 总预算；<=0 表示关闭
    # Dense drawings can spend the whole budget in broad directed OCR. Keep
    # time for GDT/capsule/strip/hires passes, which carry vertical text and
    # position-tolerance recovery.
    OCR_DIRECTED_RESERVE_SECONDS = _env_int('OCR_DIRECTED_RESERVE_SECONDS', 60)
    OCR_HIRES_RETRY_CAP = _env_int('OCR_HIRES_RETRY_CAP', 80)
    OCR_DEEP_SCAN = _OCR_DEEP_SCAN
    OCR_PHASE2_ENABLED = _env_bool('OCR_PHASE2_ENABLED', _OCR_DEEP_SCAN)
    OCR_PHASE2_MAX_REGIONS = _env_int('OCR_PHASE2_MAX_REGIONS', 160)  # 每个 90/270 方向；<=0 不截断
    OCR_TABLE_SKIP_MAX_AREA_RATIO = float(os.environ.get('OCR_TABLE_SKIP_MAX_AREA_RATIO', '0.50'))
    OCR_TABLE_SKIP_VIEW_OVERLAP_RATIO = _env_float('OCR_TABLE_SKIP_VIEW_OVERLAP_RATIO', 0.50)
    OCR_TABLE_SKIP_MIN_CELL_COUNT = _env_int('OCR_TABLE_SKIP_MIN_CELL_COUNT', 4)

    # ── 去重阈值 ──────────────────────────────────────────────────
    OCR_DEDUP_IOU        = 0.5   # dedup_ocr_by_iou / dedup_ocr_engineering
    DIM_DEDUP_IOU        = 0.3   # assembler._dedup_dimensions 条件 A
    GDT_FRAME_IOU        = 0.5   # build_augmented_gdt_frames 去重
    GDT_FRAME_CONTAIN    = 0.8   # build_augmented_gdt_frames containment NMS

    # ── Verbose / 日志 ────────────────────────────────────────────
    VERBOSE = _env_bool('APP_VERBOSE', False)         # /analyze 逐 dim 打印
    VERBOSE_ASSEMBLER = _env_bool('ASSEMBLER_VERBOSE', False)
    VERBOSE_OCR = _env_bool('OCR_VERBOSE', False)
    CANDIDATE_EXPORT_ENABLED = _env_bool('CANDIDATE_EXPORT_ENABLED', True)
    CANDIDATE_PDF_TEXT_WORDS_ENABLED = _env_bool(
        'CANDIDATE_PDF_TEXT_WORDS_ENABLED',
        True,
    )
    CANDIDATE_TEXT_REGION_FALLBACK_ENABLED = _env_bool(
        'CANDIDATE_TEXT_REGION_FALLBACK_ENABLED',
        False,
    )
    CANDIDATE_VECTOR_CONTEXT_DENSITY_ENABLED = _env_bool(
        'CANDIDATE_VECTOR_CONTEXT_DENSITY_ENABLED',
        True,
    )
    CANDIDATE_VECTOR_CONTEXT_DENSITY_WARNING_THRESHOLD = _env_float(
        'CANDIDATE_VECTOR_CONTEXT_DENSITY_WARNING_THRESHOLD',
        _env_float('CANDIDATE_VECTOR_CONTEXT_DENSITY_REJECT_THRESHOLD', 100.0),
    )
    CANDIDATE_VECTOR_CONTEXT_DENSITY_PAD = _env_float(
        'CANDIDATE_VECTOR_CONTEXT_DENSITY_PAD',
        45.0,
    )
    CANDIDATE_VECTOR_CONTEXT_DENSITY_INNER_PAD = _env_float(
        'CANDIDATE_VECTOR_CONTEXT_DENSITY_INNER_PAD',
        3.0,
    )
    # Lower values caught more noise but HITL also demoted true dimensions;
    # keep below-5pt settings experimental unless a new drawing set proves safe.
    CANDIDATE_VECTOR_CONTEXT_DENSITY_MIN_LENGTH = _env_float(
        'CANDIDATE_VECTOR_CONTEXT_DENSITY_MIN_LENGTH',
        5.0,
    )
    ANCHOR_CORRIDOR_CANDIDATES_V1_DUMP = _env_bool(
        'ANCHOR_CORRIDOR_CANDIDATES_V1_DUMP',
        False,
    )
    ANCHOR_CORRIDOR_PRECISION_FILTERS_ENABLED = _env_bool(
        'ANCHOR_CORRIDOR_PRECISION_FILTERS_ENABLED',
        False,
    )
    # Connectivity-derived text-cluster points are a recall aid for sparse
    # vector text. Dense hatching/geometry pages can retain tens of thousands
    # of segments; injecting them into DBSCAN makes text clustering quadratic.
    TEXT_CLUSTER_EXTRA_POINTS_MAX_ABSOLUTE = _env_int(
        'TEXT_CLUSTER_EXTRA_POINTS_MAX_ABSOLUTE',
        10000,
    )
    TEXT_CLUSTER_EXTRA_POINTS_MAX_TINY_RATIO = _env_float(
        'TEXT_CLUSTER_EXTRA_POINTS_MAX_TINY_RATIO',
        2.0,
    )
    # Candidate-guided OCR runs before broad OCR and focuses on geometry-backed
    # dimension candidates. It is capped by plan count and local seconds below.
    CANDIDATE_GUIDED_OCR_ENABLED = _env_bool('CANDIDATE_GUIDED_OCR', True)
    # R1.2 dump-only in the current phase. When enabled, the pipeline emits
    # proposed oriented-quad crop telemetry while OCR still consumes AABB crops.
    OCR_CROP_USE_ORIENTED_QUAD = _env_bool('OCR_CROP_USE_ORIENTED_QUAD', False)
    OCR_LOW_PRIORITY_BUDGET_MIN = _env_int('OCR_LOW_PRIORITY_BUDGET_MIN', 20)
    OCR_LOW_PRIORITY_BUDGET_RATIO = _env_float('OCR_LOW_PRIORITY_BUDGET_RATIO', 0.30)
    OCR_CANDIDATE_GUIDED_MAX_PLANS = _env_int('OCR_CANDIDATE_GUIDED_MAX_PLANS', 200)
    OCR_CANDIDATE_GUIDED_MAX_SECONDS = _env_int('OCR_CANDIDATE_GUIDED_MAX_SECONDS', 30)
    OCR_CANDIDATE_GUIDED_MIN_REMAINING_SECONDS = _env_int(
        'OCR_CANDIDATE_GUIDED_MIN_REMAINING_SECONDS',
        30,
    )
    OCR_ANGLE_LABEL_ENABLED = _env_bool('OCR_ANGLE_LABEL_ENABLED', True)
    OCR_ANGLE_LABEL_MAX_REGIONS = _env_int('OCR_ANGLE_LABEL_MAX_REGIONS', 220)
    OCR_ANGLE_LABEL_MAX_SECONDS = _env_int('OCR_ANGLE_LABEL_MAX_SECONDS', 18)
    OCR_RADIUS_LABEL_ENABLED = _env_bool('OCR_RADIUS_LABEL_ENABLED', True)
    OCR_RADIUS_LABEL_MAX_REGIONS = _env_int('OCR_RADIUS_LABEL_MAX_REGIONS', 220)
    OCR_RADIUS_LABEL_MAX_SECONDS = _env_int('OCR_RADIUS_LABEL_MAX_SECONDS', 20)
    # R2b pass-level kill switches. All default on and are used only for
    # flag-gated A/B runs after necessity dumps are reviewed.
    OCR_PASS_DIRECTED_ENABLED = _env_bool('OCR_PASS_DIRECTED_ENABLED', True)
    OCR_PASS_DIMLINE_ENABLED = _env_bool('OCR_PASS_DIMLINE_ENABLED', True)
    OCR_PASS_GDT_COMP_ENABLED = _env_bool('OCR_PASS_GDT_COMP_ENABLED', True)
    OCR_PASS_CAPSULE_ENABLED = _env_bool('OCR_PASS_CAPSULE_ENABLED', True)
    OCR_PASS_STRIP_ENABLED = _env_bool('OCR_PASS_STRIP_ENABLED', True)
    OCR_PASS_HIRES_RETRY_ENABLED = _env_bool('OCR_PASS_HIRES_RETRY_ENABLED', True)
    OCR_PASS_PHASE2_90_ENABLED = _env_bool('OCR_PASS_PHASE2_90_ENABLED', True)
    OCR_PASS_PHASE2_270_ENABLED = _env_bool('OCR_PASS_PHASE2_270_ENABLED', True)

    # Converted-PDF arrowheads are often several points apart after vector
    # flattening. 5pt keeps the detector local while recovering more paired
    # vertical/diagonal dimension-line targets on dense drawings.
    ARROW_DETECTOR_EPS = _env_float('ARROW_DETECTOR_EPS', 5.0)
    ARROW_DETECTOR_MIN_SAMPLES = _env_int('ARROW_DETECTOR_MIN_SAMPLES', 5)

    # -- Vector Semantic Graph Phase 0 ---------------------------------------
    # Audit/debug-only by default. Consumer flags are deliberately separated so
    # future phases cannot accidentally route semantic evidence into OCR,
    # table skip, strip targets, assembler scoring, or final dimensions.
    VECTOR_SEMANTICS_ENABLED = _env_bool('VECTOR_SEMANTICS_ENABLED', False)
    VECTOR_SEMANTICS_DEBUG_ONLY = _env_bool('VECTOR_SEMANTICS_DEBUG_ONLY', True)
    VECTOR_SEMANTICS_EXPORT_OBJECTS = _env_bool('VECTOR_SEMANTICS_EXPORT_OBJECTS', True)
    VECTOR_SEMANTICS_EXPORT_PRIMITIVES = _env_bool('VECTOR_SEMANTICS_EXPORT_PRIMITIVES', False)
    VECTOR_SEMANTICS_USE_FOR_TABLE_SKIP = _env_bool('VECTOR_SEMANTICS_USE_FOR_TABLE_SKIP', False)
    VECTOR_SEMANTICS_USE_FOR_OCR_PRIORITY = _env_bool('VECTOR_SEMANTICS_USE_FOR_OCR_PRIORITY', False)
    VECTOR_SEMANTICS_USE_FOR_STRIP_TARGETS = _env_bool('VECTOR_SEMANTICS_USE_FOR_STRIP_TARGETS', False)
    VECTOR_SEMANTICS_USE_FOR_ASSEMBLER_EVIDENCE = _env_bool(
        'VECTOR_SEMANTICS_USE_FOR_ASSEMBLER_EVIDENCE',
        False,
    )
    VECTOR_GLYPH_OCR_ENABLED = _env_bool('VECTOR_GLYPH_OCR_ENABLED', False)
    VECTOR_GLYPH_LAYER_DUMP = _env_bool('VECTOR_GLYPH_LAYER_DUMP', False)
    VECTOR_GLYPH_DIGIT_SLOTS_DUMP = _env_bool('VECTOR_GLYPH_DIGIT_SLOTS_DUMP', False)
    VECTOR_GLYPH_ANCHOR_AXIS_DUMP = _env_bool(
        'VECTOR_GLYPH_ANCHOR_AXIS_DUMP',
        False,
    )
    VECTOR_DIGIT_MATCH_DUMP = _env_bool('VECTOR_DIGIT_MATCH_DUMP', False)
    VECTOR_DIGIT_PROTOTYPE_LIBRARY = os.environ.get(
        'VECTOR_DIGIT_PROTOTYPE_LIBRARY',
        '',
    ).strip()
    VECTOR_DIGIT_PROTOTYPE_THRESHOLD = _env_float(
        'VECTOR_DIGIT_PROTOTYPE_THRESHOLD',
        0.20,
    )
    VECTOR_DIGIT_PROTOTYPE_MARGIN = _env_float(
        'VECTOR_DIGIT_PROTOTYPE_MARGIN',
        0.03,
    )
    VECTOR_NUMERIC_PHRASE_DUMP = _env_bool('VECTOR_NUMERIC_PHRASE_DUMP', False)
    # R33-M1 internal ordinary-dimension review runtime.  The application must
    # supply a controlled template library explicitly; production code never
    # falls back to tests/, runs/, scratch output, or a built-in asset path.
    R33_M1_TEMPLATE_LIBRARY_PATH = os.environ.get(
        'R33_M1_TEMPLATE_LIBRARY_PATH',
        '',
    ).strip()
    VECTOR_DIMENSION_HYPOTHESIS_DUMP = _env_bool(
        'VECTOR_DIMENSION_HYPOTHESIS_DUMP',
        False,
    )
    VECTOR_GLYPH_CONSUME_FINAL = _env_bool('VECTOR_GLYPH_CONSUME_FINAL', False)
    VECTOR_DECIMAL_POINT_QUAD_DUMP = _env_bool('VECTOR_DECIMAL_POINT_QUAD_DUMP', False)
    DECIMAL_POINT_QUAD_REPLACE_LEGACY = _env_bool('DECIMAL_POINT_QUAD_REPLACE_LEGACY', False)
    VECTOR_DEGREE_POLYLINE_DUMP = _env_bool('VECTOR_DEGREE_POLYLINE_DUMP', False)
    DEGREE_POLYLINE_REPLACE_LEGACY = _env_bool('DEGREE_POLYLINE_REPLACE_LEGACY', False)
    VECTOR_DECIMAL_CORRIDOR_DUMP = _env_bool('VECTOR_DECIMAL_CORRIDOR_DUMP', False)
    VECTOR_TIMES_DETECTOR_DUMP = _env_bool('VECTOR_TIMES_DETECTOR_DUMP', False)
    # R2n dump-only nominal rescue. This may emit shadow candidates from vector
    # numeric phrase evidence, but it must not alter OCR, assembler, or final
    # dimensions while final-consume remains unapproved.
    R2N_NOMINAL_RESCUE_SHADOW_DUMP = _env_bool(
        'R2N_NOMINAL_RESCUE_SHADOW_DUMP',
        False,
    )

    # -- R3 evidence/hypothesis shadow --------------------------------------
    # Dump-only by default. The consume flag is reserved for a later gated
    # assembler replacement stage and must not change final dimensions while 0.
    EVIDENCE_SHADOW_DUMP = _env_bool('EVIDENCE_SHADOW_DUMP', False)
    EVIDENCE_SHADOW_CONSUME_FINAL = _env_bool(
        'EVIDENCE_SHADOW_CONSUME_FINAL',
        False,
    )
    # R7 dump-only. Classifies already-assembled dimensions with capsule
    # geometry; it must never write production is_key until the 95% gate.
    R7_KEY_SHADOW_DUMP = _env_bool('R7_KEY_SHADOW_DUMP', False)
    R7_ROTATED_RUNWAY_SHADOW_DUMP = _env_bool(
        'R7_ROTATED_RUNWAY_SHADOW_DUMP',
        False,
    )
    R7_GDT_VERTICAL_SHADOW_DUMP = _env_bool(
        'R7_GDT_VERTICAL_SHADOW_DUMP',
        False,
    )
    R21_LAYERED_VECTOR_DIMENSION_DUMP = _env_bool(
        'R21_LAYERED_VECTOR_DIMENSION_DUMP',
        False,
    )
    # R41 条件性实验：已入库，未采纳。默认关闭；只允许把页级 primary
    # char_width 数值作为额外 attempt 的 pitch，不授予或合成任何 authority。
    R41_PRIMARY_PITCH_ATTEMPT_EXPERIMENT = _env_bool(
        'R41_PRIMARY_PITCH_ATTEMPT_EXPERIMENT',
        False,
    )
    VIEW_SEG_STATUS_DUMP = _env_bool('VIEW_SEG_STATUS_DUMP', False)
    # R2b dump-only. Emits OCR pass necessity counters but does not disable any
    # pass; per-pass OFF flags remain a later gated stage.
    OCR_PASS_NECESSITY_DUMP = _env_bool('OCR_PASS_NECESSITY_DUMP', False)
    # R1.3 preflight dump-only. Emits polygon/AABB compatibility ledger without
    # feeding polygon geometry into assembler or dedup.
    ASSEMBLER_POLYGON_COMPAT = _env_bool('ASSEMBLER_POLYGON_COMPAT', False)
    EXPORT_DIM_OVERRIDE_CONSISTENCY = _env_bool(
        'EXPORT_DIM_OVERRIDE_CONSISTENCY',
        False,
    )

    # ── Flask ────────────────────────────────────────────────────
    FLASK_PORT = _env_int('PORT', 5000)
    FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    FLASK_DEBUG = _env_bool('FLASK_DEBUG', not PRODUCTION_MODE)
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').strip() or '*'
    # Whole HTTP request body; this is not an original-PDF byte limit.
    MAX_UPLOAD_BYTES = _env_int('MAX_UPLOAD_BYTES', 32 * 1024 * 1024)

    # ── YOLO-B ───────────────────────────────────────────────────
    GDT_BACKEND = os.environ.get('GDT_BACKEND', 'hybrid').strip().lower()
    GDT_VECTOR_CONF_THRESHOLD = _env_float('GDT_VECTOR_CONF_THRESHOLD', 0.6)
    YOLO_CONF_THRESHOLD = 0.3
    YOLO_MODEL_REL_PATH = os.environ.get(
        'YOLO_MODEL_PATH',
        'yolo_gdt/gdt_crop_v1/weights/best.onnx',
    )

    # ── 路径 ─────────────────────────────────────────────────────
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..'))
    # feedback 写入使用 JSONL append-only（历史上曾用 SQLite，但只写不读，已迁移）
    FEEDBACK_LOG_PATH = os.environ.get(
        'FEEDBACK_LOG_PATH',
        os.path.join(BACKEND_DIR, 'feedback.jsonl'),
    )

    @classmethod
    def summary(cls) -> str:
        """启动时打印的配置摘要。"""
        lines = [
            f"env={cls.APP_ENV} production_mode={cls.PRODUCTION_MODE} debug={cls.FLASK_DEBUG}",
            f"DPI default={cls.DPI_DEFAULT} hires={cls.DPI_HIRES} yolo={cls.DPI_YOLO}",
            f"OCR max_sec={cls.OCR_MAX_SECONDS} hires_cap={cls.OCR_HIRES_RETRY_CAP} "
            f"directed_reserve={cls.OCR_DIRECTED_RESERVE_SECONDS} "
            f"deep_scan={cls.OCR_DEEP_SCAN} phase2={cls.OCR_PHASE2_ENABLED} "
            f"phase2_cap={cls.OCR_PHASE2_MAX_REGIONS}",
            f"candidate_export={cls.CANDIDATE_EXPORT_ENABLED} "
            f"text_region_fallback={cls.CANDIDATE_TEXT_REGION_FALLBACK_ENABLED} "
            f"candidate_guided_ocr={cls.CANDIDATE_GUIDED_OCR_ENABLED}",
            f"VERBOSE app={cls.VERBOSE} assembler={cls.VERBOSE_ASSEMBLER} ocr={cls.VERBOSE_OCR}",
            f"Flask {cls.FLASK_HOST}:{cls.FLASK_PORT} cors={cls.CORS_ORIGINS}",
        ]
        return ' | '.join(lines)
