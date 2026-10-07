// js/state.js
import { createEmptyVectorDebugWorkspace } from './vector_layer_debug_v1.js';
import { legacyJsonBase64MaxRequestBodyBytes } from './legacy_pdf_transport.js';

// 调试日志环形缓冲上限（不限会无限增长）
export const DEBUG_LOG_CAPS = Object.freeze({
    analyzeRuns:   10,    // debug_log 单页可几百 KB，10 条已经 ~5MB
    stampOcrCalls: 500,
    exportCalls:   100,
    runtimeErrors: 200,
});

// 环形缓冲 push：超过 cap 时砍掉最早的
export function pushBounded(arr, entry, cap) {
    arr.push(entry);
    if (arr.length > cap) arr.splice(0, arr.length - cap);
}

export const DOCUMENT_IMPORT_BUSY_MESSAGE = '识别任务运行中，请等待完成后再加载新文件。';
export const DOCUMENT_LOADING_MESSAGE = '文档正在加载，请等待完成后再开始解析。';

const STRICT_RUNTIME_REASON_MESSAGES = Object.freeze({
    strict_forbidden_runtime_loaded:
        '本进程已加载 OCR 或严格模式禁用组件，需重启后端后恢复严格模式',
    vector_only_preloaded_runtime_forbidden:
        '本进程已加载 OCR，需重启后端后恢复严格模式',
});

export function formatStrictRuntimeReason(reason) {
    return STRICT_RUNTIME_REASON_MESSAGES[String(reason || '')] || '';
}

export function rejectDocumentImportWhileAnalysisRunning(
    source,
    notify = globalThis.alert,
) {
    if (source?.isAnalysisRunning !== true) return false;
    if (typeof notify === 'function') notify(DOCUMENT_IMPORT_BUSY_MESSAGE);
    return true;
}

export function advanceDocumentGeneration(source) {
    const current = Number.isSafeInteger(source?.documentGeneration)
        && source.documentGeneration >= 0
        ? source.documentGeneration
        : 0;
    if (current >= Number.MAX_SAFE_INTEGER) {
        throw new RangeError('documentGeneration exhausted');
    }
    source.documentGeneration = current + 1;
    return source.documentGeneration;
}

// 极其轻量的 UUID 生成引擎（v4 标准）
export function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

export function createEmptyInspectionState() {
    return {
        active: false,
        importedAt: null,
        sourceFiles: [],
        records: [],
        byStampUuid: {},
        unmatchedRecords: [],
        warnings: [],
        stats: {
            fileCount: 0,
            recordCount: 0,
            matchedStampCount: 0,
            unmatchedRecordCount: 0,
            skippedRecordCount: 0,
            nokCount: 0,
            unmeasuredCount: 0,
            keyNokCount: 0,
        },
        filter: 'all',
        selectedRecordId: null,
    };
}

// R33-M1 人工复核工作区。API envelope 保持原样存放；人工动作单独记账，
// 只有显式确认动作才允许由 review workbench 创建正式 stamp。
// 该状态不会被工程归档导出，避免待复核候选进入正式项目数据。
export function createEmptyReviewWorkspace() {
    return {
        envelopes: [],
        decisions: {},
        selectedCandidateId: null,
    };
}

// 会话级旁路：不进 history、工程归档或人工复核工作区。
export function createEmptyVectorDebugState() {
    return {
        enabled: false,
        drawerOpen: false,
        workspace: createEmptyVectorDebugWorkspace(),
        error: null,
    };
}

// 默认配色单源（reset 按钮 / state 初始化 / preset 都从这里读）
// 方案 B「稍饱和」: 比低饱和 Linear/Apple 风提一档，比 Bootstrap Material 风柔和一档
//   普通 / 矢量 / GD&T → 砖红    #C23340   (原 #B85C63 提饱和 ~15pp)
//   跑道框（KEY）     → 深钢蓝  #2E5AAE   (原 #4C6FB3 加深)
//   表面粗糙度         → 深金    #C49A1E
export const DEFAULT_TYPE_COLORS = Object.freeze({
    'geometric_anchor': '#2E5AAE',
    'gdt_line_frame':   '#C23340',
    'ocr':              '#C23340',
    'vector':           '#C23340',
    'surface_roughness': '#C49A1E',
    'note':             '#1e7ad6',
    '_default':         '#585E66',
});

// 从 localStorage 加载用户配色；失败/无记录回退 DEFAULT_TYPE_COLORS
function _loadTypeColors() {
    try {
        const raw = localStorage.getItem('typeColors');
        if (!raw) return { ...DEFAULT_TYPE_COLORS };
        const saved = JSON.parse(raw);
        // 合并：以默认为基底，保留 saved 里能识别的 key（防止旧版本 schema 错乱）
        const merged = { ...DEFAULT_TYPE_COLORS };
        for (const k of Object.keys(DEFAULT_TYPE_COLORS)) {
            if (typeof saved[k] === 'string' && /^#[0-9a-f]{6}$/i.test(saved[k])) {
                merged[k] = saved[k];
            }
        }
        return merged;
    } catch (_e) {
        return { ...DEFAULT_TYPE_COLORS };
    }
}

// 保存用户配色到 localStorage（picker change 后调用）
export function saveTypeColors() {
    try {
        localStorage.setItem('typeColors', JSON.stringify(state.typeColors));
    } catch (_e) { /* 配额或隐私模式：静默失败 */ }
}

// 识别后端选项。默认纯矢量 + rapidocr，刷新不丢。
export const DIMENSION_PATHS = ['vector_only', 'vector_plus_ocr', 'ocr_only'];
export const OCR_ENGINES = ['rapidocr', 'paddle', 'yolo_char'];
export const VIEW_SEG_BACKENDS = ['yolo', 'watershed', 'minesweeper'];
export const GDT_BACKENDS = ['hybrid', 'vector', 'yolo'];
export const RECOMMENDED_RECOGNITION_SETTINGS = Object.freeze({
    dimension_path: 'vector_only',
    ocr_engine: 'rapidocr',
    view_seg: 'yolo',
    gdt_backend: 'hybrid',
});

export function getRecommendedRecognitionSettings() {
    return { ...RECOMMENDED_RECOGNITION_SETTINGS };
}

export function computeDimensionPathRowStates(
    strictViewSegCapability = null,
    strictTemplateCapability = null,
) {
    const strictFailures = [strictViewSegCapability, strictTemplateCapability]
        .map(capability => capability || { available: false, reason: 'unknown' })
        .filter(capability => !capability.available)
        .map(capability => capability.reason || 'unknown');
    return DIMENSION_PATHS.map(path => ({
        path,
        disabled: path === 'vector_only' && strictFailures.length > 0,
        note: path === 'vector_only' ? (strictFailures[0] || '') : '',
        reasons: path === 'vector_only' ? [...strictFailures] : [],
    }));
}

export function computeOcrEngineRowStates(dimensionPath, capabilities) {
    const caps = capabilities || {};
    const ocrOff = dimensionPath === 'vector_only';
    return OCR_ENGINES.map((engine) => {
        const cap = caps[engine] || { available: false, reason: 'unknown' };
        if (ocrOff) {
            return { engine, disabled: true, note: '当前模式不跑 OCR' };
        }
        return {
            engine,
            disabled: !cap.available,
            note: cap.available ? '' : cap.reason,
        };
    });
}

export function computeViewSegRowStates(
    capabilities,
    dimensionPath = null,
    strictYoloCapability = null,
) {
    const caps = capabilities || {};
    const strictVectorOnly = dimensionPath === 'vector_only';
    return VIEW_SEG_BACKENDS.map((engine) => {
        const cap = (
            strictVectorOnly && engine === 'yolo'
                ? strictYoloCapability
                : caps[engine]
        ) || { available: false, reason: 'unknown' };
        if (strictVectorOnly && engine !== 'yolo') {
            return {
                engine,
                disabled: true,
                note: 'vector_only_requires_yolo_view',
            };
        }
        return {
            engine,
            disabled: !cap.available,
            note: cap.available ? (cap.reason === 'experimental' ? 'experimental' : '') : cap.reason,
        };
    });
}

export const STRICT_GDT_BACKEND_COMPATIBILITY_NOTE =
    '严格矢量模式下 GD&T 由矢量主链接管，此选项仅对兼容模式生效';

export function computeGdtBackendRowStates(
    capabilities,
    dimensionPath = null,
) {
    const caps = capabilities || {};
    const strictVectorOnly = dimensionPath === 'vector_only';
    return GDT_BACKENDS.map((engine) => {
        const cap = caps[engine] || { available: false, reason: 'unknown' };
        if (strictVectorOnly) {
            return {
                engine,
                disabled: true,
                note: STRICT_GDT_BACKEND_COMPATIBILITY_NOTE,
            };
        }
        return {
            engine,
            disabled: !cap.available,
            note: cap.available ? '' : cap.reason,
        };
    });
}

// Compatibility export for existing consumers and tests. New UI code should
// pass the active dimension path through computeGdtBackendRowStates.
export function computeGdtRowStates(capabilities) {
    return computeGdtBackendRowStates(capabilities);
}

function _loadRecognitionSettings() {
    const defaults = getRecommendedRecognitionSettings();
    try {
        const raw = localStorage.getItem('recognitionSettings');
        if (raw) {
            const parsed = JSON.parse(raw);
            if (parsed) {
                return {
                    dimension_path: DIMENSION_PATHS.includes(parsed.dimension_path)
                        ? parsed.dimension_path
                        : defaults.dimension_path,
                    ocr_engine: OCR_ENGINES.includes(parsed.ocr_engine)
                        ? parsed.ocr_engine
                        : defaults.ocr_engine,
                    view_seg: VIEW_SEG_BACKENDS.includes(parsed.view_seg)
                        ? parsed.view_seg
                        : defaults.view_seg,
                    gdt_backend: GDT_BACKENDS.includes(parsed.gdt_backend)
                        ? parsed.gdt_backend
                        : defaults.gdt_backend,
                };
            }
        }
    } catch (_e) { /* 回退默认 */ }
    return defaults;
}

export function saveRecognitionSettings() {
    try {
        localStorage.setItem('recognitionSettings', JSON.stringify(state.recognitionSettings));
    } catch (_e) { /* 忽略持久化失败 */ }
}

export function resetRecognitionSettingsToRecommended() {
    state.recognitionSettings = getRecommendedRecognitionSettings();
    saveRecognitionSettings();
}

function _isCapabilityMap(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function _isCapabilityEntry(value) {
    return (
        _isCapabilityMap(value)
        && typeof value.available === 'boolean'
        && typeof value.reason === 'string'
    );
}

function _isCompleteCapabilityAxis(capabilities, expectedKeys) {
    return (
        _isCapabilityMap(capabilities)
        && expectedKeys.every(key => _isCapabilityEntry(capabilities[key]))
    );
}

export function hasCompleteRecognitionCapabilities(source = state) {
    const axes = [
        [source?.ocrEngineCapabilities, OCR_ENGINES],
        [source?.viewSegCapabilities, VIEW_SEG_BACKENDS],
        [source?.gdtCapabilities, GDT_BACKENDS],
    ];
    return (
        axes.every(([capabilities, expectedKeys]) => (
            _isCompleteCapabilityAxis(capabilities, expectedKeys)
        ))
        && _isCapabilityEntry(source?.strictViewSegCapability)
        && _isCapabilityEntry(source?.strictTemplateCapability)
    );
}

function _copyCapabilityAxis(capabilities) {
    return Object.fromEntries(
        Object.entries(capabilities).map(([key, value]) => [key, { ...value }]),
    );
}

function _strictViewSegCapabilityFromPayload(payload) {
    const checks = Array.isArray(payload?.checks) ? payload.checks : [];
    const strictAssets = checks.find(check => (
        _isCapabilityMap(check) && check.name === 'strict_runtime_assets'
    ));
    const strictRuntime = checks.find(check => (
        _isCapabilityMap(check) && check.name === 'strict_forbidden_runtime'
    ));
    if (
        !strictAssets
        || typeof strictAssets.ok !== 'boolean'
        || !strictRuntime
        || typeof strictRuntime.ok !== 'boolean'
    ) return null;
    const failedCheck = !strictAssets.ok
        ? strictAssets
        : (!strictRuntime.ok ? strictRuntime : null);
    return {
        available: failedCheck === null,
        reason: failedCheck === null
            ? 'ok'
            : (typeof failedCheck.reason_code === 'string' && failedCheck.reason_code
                ? failedCheck.reason_code
                : 'strict_runtime_unavailable'),
    };
}

function _strictTemplateCapabilityFromPayload(payload) {
    const checks = Array.isArray(payload?.checks) ? payload.checks : [];
    const check = checks.find(item => (
        _isCapabilityMap(item) && item.name === 'r33_m1_template_runtime'
    ));
    if (!check || typeof check.ok !== 'boolean') return null;
    return {
        available: check.ok,
        reason: check.ok
            ? 'ok'
            : (typeof check.reason_code === 'string' && check.reason_code
                ? check.reason_code
                : 'template_runtime_unavailable'),
    };
}

export function updateRecognitionCapabilitiesFromPayload(
    payload,
    target = state,
    { requireReadiness = false } = {},
) {
    const caps = payload?.capabilities || payload?.components;
    if (!_isCapabilityMap(caps)) return false;
    if (
        !_isCompleteCapabilityAxis(caps.ocr_engines, OCR_ENGINES)
        || !_isCompleteCapabilityAxis(caps.view_seg, VIEW_SEG_BACKENDS)
        || !_isCompleteCapabilityAxis(caps.gdt, GDT_BACKENDS)
    ) return false;

    const strictViewSegCapability = _strictViewSegCapabilityFromPayload(payload);
    const strictTemplateCapability = _strictTemplateCapabilityFromPayload(payload);
    if (
        requireReadiness
        && (strictViewSegCapability === null || strictTemplateCapability === null)
    ) return false;

    // Commit every required readiness axis as one snapshot. Analyze responses
    // legitimately carry only the legacy axes, so non-readiness callers may
    // still refresh those without changing either strict authority.
    target.ocrEngineCapabilities = _copyCapabilityAxis(caps.ocr_engines);
    target.viewSegCapabilities = _copyCapabilityAxis(caps.view_seg);
    target.gdtCapabilities = _copyCapabilityAxis(caps.gdt);
    if (strictViewSegCapability) {
        target.strictViewSegCapability = strictViewSegCapability;
    }
    if (strictTemplateCapability) {
        target.strictTemplateCapability = strictTemplateCapability;
    }
    if (requireReadiness || Object.hasOwn(payload, 'transport_limits')) {
        target.legacyJsonBase64MaxRequestBodyBytes = legacyJsonBase64MaxRequestBodyBytes(payload);
    }
    return strictViewSegCapability !== null && strictTemplateCapability !== null;
}

function _recognitionCapabilityErrorMessage(error) {
    if (typeof error?.message === 'string' && error.message.trim()) {
        return error.message.trim();
    }
    const fallback = String(error || '').trim();
    return fallback || 'recognition_capabilities_fetch_failed';
}

/**
 * Build a single-flight readiness capability loader.
 *
 * Failed or incomplete responses stay retryable. HTTP 503 is intentionally
 * accepted when it carries the normal readiness JSON: a deployment can be
 * not-ready while still reporting which recognition components are present.
 */
export function createRecognitionCapabilitiesLoader({
    endpoint = '/readyz',
    fetchImpl = (...args) => globalThis.fetch(...args),
    targetState = state,
    onSettled = () => {},
} = {}) {
    let inFlight = null;
    let activeToken = null;

    function refresh({ force = false } = {}) {
        if (inFlight) return inFlight;
        if (
            !force
            && targetState.recognitionCapabilitiesStatus === 'ready'
            && hasCompleteRecognitionCapabilities(targetState)
        ) {
            targetState.recognitionCapabilitiesStatus = 'ready';
            targetState.recognitionCapabilitiesError = null;
            return Promise.resolve({ status: 'cached', httpStatus: null });
        }

        targetState.recognitionCapabilitiesStatus = 'loading';
        targetState.recognitionCapabilitiesError = null;
        // Starting a real freshness request invalidates both strict authorities
        // immediately. Keep vector_only fail-closed throughout the loading gap.
        targetState.strictViewSegCapability = null;
        targetState.strictTemplateCapability = null;
        targetState.legacyJsonBase64MaxRequestBodyBytes = null;
        const token = Symbol('recognition-capabilities-request');
        activeToken = token;

        // Defer fetch by one microtask so the single-flight lock is installed
        // before a synchronous fetch wrapper/interceptor can re-enter refresh.
        const request = Promise.resolve()
            .then(() => fetchImpl(endpoint, {
                headers: { Accept: 'application/json' },
            }))
            .then(async response => {
                if (!response || typeof response.json !== 'function') {
                    throw new Error('recognition_capabilities_invalid_response');
                }
                const httpStatus = Number.isInteger(response.status)
                    ? response.status
                    : null;
                if (
                    httpStatus === null
                    || !((httpStatus >= 200 && httpStatus < 300) || httpStatus === 503)
                ) {
                    throw new Error(`recognition_capabilities_http_${httpStatus ?? 'unknown'}`);
                }
                const payload = await response.json();
                if (!updateRecognitionCapabilitiesFromPayload(
                    payload,
                    targetState,
                    { requireReadiness: true },
                )) {
                    throw new Error('recognition_capabilities_incomplete');
                }
                targetState.recognitionCapabilitiesStatus = 'ready';
                targetState.recognitionCapabilitiesError = null;
                return {
                    status: 'ready',
                    httpStatus,
                };
            })
            .catch(error => {
                const message = _recognitionCapabilityErrorMessage(error);
                targetState.recognitionCapabilitiesStatus = 'error';
                targetState.recognitionCapabilitiesError = message;
                // Strict mode has no fallback. A failed freshness check must
                // not leave old positive runtime authorities visible as current.
                targetState.strictViewSegCapability = null;
                targetState.strictTemplateCapability = null;
                return { status: 'error', error: message, httpStatus: null };
            })
            .finally(() => {
                if (activeToken === token) {
                    inFlight = null;
                    activeToken = null;
                }
                try {
                    onSettled({
                        status: targetState.recognitionCapabilitiesStatus,
                        error: targetState.recognitionCapabilitiesError,
                    });
                } catch (_error) {
                    // Rendering failures must not poison capability cache state.
                }
            });
        inFlight = request;
        return inFlight;
    }

    return Object.freeze({
        refresh,
        isInFlight: () => inFlight !== null,
    });
}

// 全局状态树（TDD 3.3 权威字段表，★ = v2.1 新增）
export const state = {
    // ── 文档信息 ────────────────────────────────────────────────────────────
    docInfo: { filename: 'Untitled', version: 1 },

    // ── 核心数据 ────────────────────────────────────────────────────────────
    stamps:   [],   // Stamp 对象数组，每个 Stamp 必须含 uuid 字段
    deletedStamps: [],  // HITL: 被用户删除的 stamp（导出时提交 feedback）
    reviewWorkspace: createEmptyReviewWorkspace(),
    vectorDebugState: createEmptyVectorDebugState(),
    clusters: [],   // Cluster 对象数组（DBSCAN 输出，TDD 2.2.2）
    inspection: createEmptyInspectionState(),
    sidebarTab: 'dimensions',

    // 视图框（蓝色方框，每个 DBSCAN cluster 一个，可手动拖动/缩放）
    // { id, pageIndex, x, y, w, h, shape, cx?, cy?, r?, isEdited }  — PDF 全局坐标
    //   shape:        'rect'（默认）| 'circle'（YOLO-A 检出 + watershed 圆形判据二次分类）
    //   cx, cy, r:    仅 shape==='circle' 时存在；圆心 + 半径，PDF pt 坐标
    //   老 ZIP 兼容:  无 shape 字段视为 'rect'（app_io.js 载入时不强制兜底，渲染层 default 'rect'）
    viewBoxes: [],
    // 理论正确尺寸 / basic dimension：方框内纯数字，显示在画布上，但不参与 stamp 编号。
    // { id, pageIndex, text, bbox, inspectionRole, numberingExcluded }
    basicDimensions: [],
    // 基准标识：方框内单字母 datum reference，显示但不参与 stamp 编号。
    // { id, pageIndex, text, bbox }
    datumReferences: [],
    // R14.5 A1：当前 hover 在边缘 ±6 内的 viewBox id；非 null 时 drawViewBoxesLayer 显示该 vb 的 8 句柄
    hoverViewBoxId: null,

    // ── 视图物理状态 ─────────────────────────────────────────────────────────
    pdfWidth:         0,
    pdfHeight:        0,
    zoom:             1.0,
    panX:             0,
    panY:             0,
    rotation:         0,
    // 每页 document-global 几何：start/width/height + sourceWidth/sourceHeight/rotation。
    pageOffsets:      [],
    totalPages:       1,
    currentPdfBuffer: null,
    // 会话级文档身份代数；每次 PDF / JSON / ZIP 成功提交时递增。
    // 不进入工程归档或 undo/redo，只用于让异步事务拒绝跨文档提交。
    documentGeneration: 0,
    isDocumentImportRunning: false,
    miniScale:        1.0,

    // ── 交互状态机 ───────────────────────────────────────────────────────────
    keys:       { space: false, ctrl: false, alt: false, shift: false },
    historyStack:  [],
    historyIndex:  -1,
    baseTool:      'hand',
    stampActive:   false,
    drawPhase:     0,
    tempP1:        null,
    tempType:      'normal',
    // typeColors 从 localStorage 加载（见 _loadTypeColors），单源 DEFAULT_TYPE_COLORS
    typeColors: _loadTypeColors(),
    // 识别后端选项（Phase 1：尺寸主识别轴），默认纯矢量，localStorage 持久化
    recognitionSettings: _loadRecognitionSettings(),
    ocrEngineCapabilities: null,
    viewSegCapabilities: null,
    gdtCapabilities: null,
    strictViewSegCapability: null,
    strictTemplateCapability: null,
    legacyJsonBase64MaxRequestBodyBytes: null,
    recognitionCapabilitiesStatus: 'idle',
    recognitionCapabilitiesError: null,
    drawMode:  'click',
    drawOrder: 'arrow_first',

    // ── 解析范围 ─────────────────────────────────────────────────────────────
    // 'current' = 仅当前页，'all' = 全部页。
    // 默认串行解析全部页；保留当前页选项，供长任务或兼容模式降低阻塞。
    analyzeScope: 'all',
    isAnalysisRunning: false,
    analysisAbortController: null,

    // ── 显示过滤（HITL）────────────────────────────────────────────────
    // true 时画布 + 侧栏隐藏 isAutoGenerated && !confirmed && aiData.confidence==='low' 的 stamp
    // 目的：默认低置信度 stamp 较多会把画布糊满，让用户先聚焦 high/medium AI 输出。
    // 已 confirmed 的 low conf 仍显示（用户已经校验过）。
    hideLowConfidence: false,

    // ── 选中与拖拽 ───────────────────────────────────────────────────────────
    selectedUUIDs: [],
    hoverNode:     null,
    dragCache:     null,
    selectionBox:  null,
    lastMousePt:   null,

    // ── L1 内存缓存池（Key: 页码, Value: 原始 boxes 数组）─────────────────────
    l1Cache:    new Map(),

    // ── 多页解析追踪（已智能解析过的页码集合）────────────────────────────────
    currentAnalyzedPages: new Set(),

    // ── DBSCAN 当前参数 ──────────────────────────────────────────────────────
    currentEps: null,

    // ★ A-03: EPS 滑块生命周期锁定
    // Stage 1 完成时记录 historyIndex 基准值；Stage 2 任意写操作后自动锁定
    clusteringLocked:    false,
    clusteringBaseIndex: -1,

    // ── 侧边栏展开状态 ─────────────────────────────────────────────────────────
    activeEditUUID: null,   // 当前展开编辑的条目 UUID，null 表示全部折叠

    // ── 调试日志：导出当前会话的诊断记录 ─────────────────────────────────────
    lastDebugLog:   null,   // 兼容字段：等价 analyzeRuns[last].debug_log
    lastDimensions: null,   // 兼容字段：等价 analyzeRuns[last].dimensions
    lastBasicDimensions: null,
    // 每次 /analyze 调用一条：{ ts, page, debug_log, dimensions, basic_dimensions, view_boxes,
    //   timings, ocr_breakdown, ocr_pass_counters, durationMs, ok, error }
    analyzeRuns:    [],
    // 每次 /analyze_stamp 调用一条：{ ts, page, cropBbox, stampUuid, durationMs, ok, response, error }
    stampOcrCalls:  [],
    // 每次 /export 调用一条：{ ts, page, allPages, partNo, dimsCount, silent, durationMs, ok, error }
    exportCalls:    [],
    // window.error / unhandledrejection 捕获：{ ts, type, message, source, line, col, stack }
    runtimeErrors:  [],
    // 后端启动日志（GET /startup_log）：{ log, bytes, lines, components, fetchedAt }
    serverStartupLog: null,
    // 后端运行时计数快照（GET /runtime_stats）：{ analyze_calls, ..., uptime_sec, fetchedAt }
    serverRuntimeStats: null,

    // ── 定时器句柄 ───────────────────────────────────────────────────────────
    renderTimer: null,
    resizeTimer: null,
};
