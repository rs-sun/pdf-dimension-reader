// app_sidebar/control_panel.js — 控制面板（颜色配置 + 选中章面板）
// 颜色入口给用户改 source → color 映射；改完调 list.updateSidebar 刷新侧边栏。

import {
    state,
    saveTypeColors,
    saveRecognitionSettings,
    resetRecognitionSettingsToRecommended,
    createRecognitionCapabilitiesLoader,
    OCR_ENGINES,
    VIEW_SEG_BACKENDS,
    GDT_BACKENDS,
    computeDimensionPathRowStates,
    computeOcrEngineRowStates,
    computeViewSegRowStates,
    computeGdtBackendRowStates,
    formatStrictRuntimeReason,
} from '../state.js';
import { CONFIG } from '../config.js';
import { resolveIsKey } from '../canvas_renderer.js';
import { escHtml, pushHistory } from '../app_utils.js';
import { updateSidebar } from './list.js';

// ═══════════════════════════════════════════════════════════════════════════════
// Late-binding callbacks (避免与 app.js 形成循环依赖)
// ═══════════════════════════════════════════════════════════════════════════════

let _syncColorToggleBtn = null;
let _updateTransform = null;

const _recognitionCapabilitiesLoader = createRecognitionCapabilitiesLoader({
    endpoint: `${CONFIG.API_BASE_URL}/readyz`,
    onSettled: () => {
        initRecognitionSettingsRows();
        initOcrEngineRows({ refreshCapabilities: false });
        initViewSegRows({ refreshCapabilities: false });
        initGdtRows({ refreshCapabilities: false });
    },
});

export function registerSidebarCallbacks(syncColorToggleBtn, updateTransform) {
    _syncColorToggleBtn = syncColorToggleBtn;
    _updateTransform = updateTransform;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 控制面板：颜色配置 + 选中面板
// ═══════════════════════════════════════════════════════════════════════════════

// 控制面板里给用户改的颜色入口。state.DEFAULT_TYPE_COLORS 仍保留所有
// source 的默认色，stamp 渲染走 getStampColor 不受影响。
const SOURCE_LABELS = {
    'geometric_anchor': '跑道框（重要特性）',
    'ocr':              '普通尺寸',
    'surface_roughness': '表面粗糙度',
};

export function openControlPanel() {
    const panel = document.getElementById('control-panel');
    const backdrop = document.getElementById('control-panel-backdrop');
    panel.style.display = 'block';
    if (backdrop) backdrop.style.display = 'block';
    // Opening the panel is an explicit freshness boundary. Install the
    // single-flight request first so the controls immediately show loading.
    void refreshRecognitionCapabilities({ force: true });
    initColorRows();
    updateSelectedPanel();
    initRecognitionSettingsRows();
    initOcrEngineRows({ refreshCapabilities: false });
    initViewSegRows({ refreshCapabilities: false });
    initGdtRows({ refreshCapabilities: false });
}
export function closeControlPanel() {
    document.getElementById('control-panel').style.display = 'none';
    const backdrop = document.getElementById('control-panel-backdrop');
    if (backdrop) backdrop.style.display = 'none';
}

export function initColorRows() {
    const rowsDiv = document.getElementById('color-rows');
    if (!rowsDiv) return;
    rowsDiv.innerHTML = '';
    for (const [source, label] of Object.entries(SOURCE_LABELS)) {
        const cur = (state.typeColors || {})[source] || '#6c757d';
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:6px;';
        row.innerHTML = `<input type="color" value="${cur}" data-source="${source}"
            style="width:32px;height:24px;border:1px solid #ccc;border-radius:3px;cursor:pointer;padding:0;">
            <span>${label}</span>`;
        rowsDiv.appendChild(row);
        row.querySelector('input[type="color"]').addEventListener('input', (e) => {
            state.typeColors[e.target.dataset.source] = e.target.value;
            saveTypeColors();   // 持久化到 localStorage
            if (_syncColorToggleBtn) _syncColorToggleBtn();
            if (_updateTransform) _updateTransform();
            updateSidebar();
        });
    }
}

// 识别路径标签 + 人话取舍说明（顺序同 DIMENSION_PATHS）
const DIMENSION_PATH_LABELS = {
    vector_only:     ['纯矢量', '最快、几乎不耗算力，直接从 PDF 矢量层读数。可能漏读或读错一部分。'],
    vector_plus_ocr: ['矢量+OCR', '矢量打底、再用 OCR 补漏，读得更全。要跑 OCR，较慢、耗算力。矢量合并需服务端授权，当前未开启。'],
    ocr_only:        ['仅 OCR（老办法）', '完全靠 OCR，读得最全但最慢最耗算力，且依赖 OCR 引擎可用。'],
};

const OCR_ENGINE_LABELS = {
    rapidocr: ['rapidocr', '跨平台、纯 CPU，速度中等。本机（Apple Silicon）默认就是它。'],
    paddle: ['paddle', '有 NVIDIA 显卡时 GPU 加速最快。苹果芯片不支持（会崩），本机灰掉。'],
    yolo_char: ['YOLO 字符识别', '待建占位：计划用 YOLO 直接认 0-9 / a-z / A-Z。本期灰显、不可选。'],
};

const VIEW_SEG_LABELS = {
    yolo: ['YOLO-A', '训练好的模型，分图最准。需要模型文件在。'],
    watershed: ['watershed', '传统算法兜底，不需要模型；复杂/密集图可能切不准。'],
    minesweeper: ['minesweeper', '实验性几何算法，不稳，仅尝鲜。'],
};

const GDT_LABELS = {
    hybrid: ['矢量+YOLO', '矢量 + YOLO 结合，较平衡。'],
    vector: ['纯矢量', '从线条反推符号，快、不耗算力；竖排 / 重叠框可能漏。'],
    yolo: ['YOLO-B', '模型识别符号，需要模型文件、耗算力。'],
};

function formatOcrCapabilityReason(reason) {
    if (reason === 'ok') return '';
    if (reason === 'arm_mac_paddle_segfault') return '苹果芯片禁用，避免 Paddle 推理崩溃';
    if (reason === 'paddleocr_not_installed') return '未安装 PaddleOCR';
    if (reason === 'rapidocr_not_installed') return '未安装 rapidocr';
    if (reason === 'not_built') return '待建';
    if (reason === 'unknown') return '等待后端能力清单';
    return String(reason || '不可用');
}

export function formatViewSegCapabilityReason(reason) {
    const strictRuntimeReason = formatStrictRuntimeReason(reason);
    if (strictRuntimeReason) return strictRuntimeReason;
    if (reason === 'ok') return '';
    if (reason === 'experimental') return '实验性';
    if (reason === 'vector_only_requires_yolo_view') return '纯矢量模式只支持 YOLO-A，不会回退传统分图';
    if (reason === 'strict_yolo_model_missing') return '严格模式 ONNX 模型缺失';
    if (reason === 'strict_yolo_onnx_runner_missing') return '严格模式 ONNX 运行器缺失';
    if (reason === 'strict_yolo_model_sha256_unconfigured') return '严格模式模型 SHA-256 未配置';
    if (reason === 'strict_yolo_model_sha256_mismatch') return '严格模式模型 SHA-256 不匹配';
    if (reason === 'strict_yolo_model_format_forbidden') return '严格模式只允许 ONNX 模型';
    if (reason === 'strict_yolo_pt_fallback_forbidden') return '严格模式禁止 PT fallback';
    if (reason === 'strict_yolo_model_unreadable') return '严格模式 ONNX 模型不可读';
    if (reason === 'strict_runtime_assets_unavailable') return '严格模式运行资产不可用';
    if (reason === 'strict_runtime_unavailable') return '严格模式运行环境不可用';
    if (reason === 'template_library_path_unconfigured') return '严格模式数字模板未配置';
    if (reason === 'template_library_unavailable') return '严格模式数字模板文件不可用';
    if (reason === 'template_library_not_file') return '严格模式数字模板路径不是文件';
    if (reason === 'template_library_invalid') return '严格模式数字模板无效';
    if (reason === 'template_library_payload_not_object') return '严格模式数字模板格式无效';
    if (reason === 'template_library_has_no_runtime_templates') return '严格模式数字模板不含可运行模板';
    if (reason === 'template_runtime_identity_invalid') return '严格模式数字模板身份无效';
    if (reason === 'template_runtime_unavailable') return '严格模式数字模板运行时不可用';
    if (reason === 'phrase_reader_not_callable') return '严格模式短语读取器不可用';
    if (reason === 'model_file_missing') return '模型文件缺失';
    if (reason === 'ultralytics_not_installed') return '未安装 ultralytics';
    if (reason === 'unknown') return '等待后端能力清单';
    return String(reason || '不可用');
}

function formatGdtCapabilityReason(reason) {
    if (reason === 'ok') return '';
    if (reason === 'model_not_loaded') return 'YOLO-B 模型未加载';
    if (reason === 'unknown') return '等待后端能力清单';
    return String(reason || '不可用');
}

export function refreshRecognitionCapabilities(options) {
    return _recognitionCapabilitiesLoader.refresh(options);
}

export function formatTransportLimitText(maxRequestBodyBytes) {
    if (!Number.isSafeInteger(maxRequestBodyBytes) || maxRequestBodyBytes <= 0) {
        return '服务端旧版传输上限：后端未报告；提交时仍会按 HTTP 状态处理';
    }
    const mib = (maxRequestBodyBytes / (1024 * 1024)).toFixed(2);
    return `服务端旧版传输上限：${mib} MiB（包含 JSON 与 base64；不是原始 PDF 大小）`;
}

export function syncAnalyzeScopeHintVisibility(
    dimensionPath = state.recognitionSettings.dimension_path,
) {
    const hint = document.getElementById('analyze-scope-ocr-hint');
    if (!hint) return;
    hint.style.display = ['vector_plus_ocr', 'ocr_only'].includes(dimensionPath)
        ? ''
        : 'none';
}

export function initRecognitionSettingsRows() {
    syncAnalyzeScopeHintVisibility();
    const box = document.getElementById('recognition-settings-rows');
    if (!box) return;
    box.innerHTML = '';
    const resetRow = document.createElement('div');
    resetRow.style.cssText = 'display:flex;justify-content:flex-end;gap:8px;margin-bottom:10px;';
    const isLoading = state.recognitionCapabilitiesStatus === 'loading';
    resetRow.innerHTML = `<button type="button" id="refresh-recognition-capabilities"
        ${isLoading ? 'disabled' : ''}
        style="padding:4px 10px;border:1px solid #bbb;border-radius:4px;background:#fff;cursor:${isLoading ? 'wait' : 'pointer'};font-size:12px;">
        ${isLoading ? '检测中…' : '重新检测后端'}
    </button>
    <button type="button" id="reset-recognition-settings"
        style="padding:4px 10px;border:1px solid #bbb;border-radius:4px;background:#fff;cursor:pointer;font-size:12px;">
        恢复推荐默认
    </button>`;
    box.appendChild(resetRow);
    resetRow.querySelector('#refresh-recognition-capabilities').addEventListener('click', () => {
        void refreshRecognitionCapabilities({ force: true });
        initRecognitionSettingsRows();
    });
    resetRow.querySelector('#reset-recognition-settings').addEventListener('click', () => {
        resetRecognitionSettingsToRecommended();
        initRecognitionSettingsRows();
        initOcrEngineRows({ refreshCapabilities: false });
        initViewSegRows({ refreshCapabilities: false });
        initGdtRows({ refreshCapabilities: false });
    });
    if (state.recognitionCapabilitiesStatus === 'error') {
        const status = document.createElement('div');
        status.style.cssText = 'margin:-2px 0 10px;color:#b42318;font-size:11px;';
        status.textContent = '后端能力检测失败；可点击“重新检测后端”重试。';
        if (state.recognitionCapabilitiesError) {
            status.title = state.recognitionCapabilitiesError;
        }
        box.appendChild(status);
    }
    const transportStatus = document.createElement('div');
    transportStatus.style.cssText = 'margin:-2px 0 10px;color:#666;font-size:11px;';
    transportStatus.textContent = formatTransportLimitText(
        state.legacyJsonBase64MaxRequestBodyBytes,
    );
    box.appendChild(transportStatus);
    const cur = state.recognitionSettings.dimension_path;
    const pathRows = computeDimensionPathRowStates(
        state.strictViewSegCapability,
        state.strictTemplateCapability,
    );
    for (const rowState of pathRows) {
        const path = rowState.path;
        const [label, desc] = DIMENSION_PATH_LABELS[path];
        const disabled = Boolean(rowState.disabled);
        const note = (rowState.reasons || [])
            .map(formatViewSegCapabilityReason)
            .filter(Boolean)
            .join('；');
        const row = document.createElement('label');
        row.style.cssText = [
            'display:block',
            'margin-bottom:8px',
            `cursor:${disabled ? 'not-allowed' : 'pointer'}`,
            `opacity:${disabled ? '0.55' : '1'}`,
        ].join(';');
        row.innerHTML = `<span style="display:flex;align-items:center;gap:6px;">
            <input type="radio" name="dimension-path" value="${path}"
                ${path === cur ? 'checked' : ''}
                ${disabled ? 'disabled' : ''}>
            <strong>${escHtml(label)}</strong></span>
            <span style="display:block;margin-left:22px;color:#888;font-size:11px;">${escHtml(desc)}</span>
            ${note ? `<span style="display:block;margin-left:22px;color:#b26b00;font-size:11px;">当前不可用：${escHtml(note)}</span>` : ''}`;
        box.appendChild(row);
        row.querySelector('input').addEventListener('change', (e) => {
            if (!e.target.checked || e.target.disabled) return;
            state.recognitionSettings.dimension_path = e.target.value;
            saveRecognitionSettings();
            syncAnalyzeScopeHintVisibility();
            initOcrEngineRows({ refreshCapabilities: false });
            initViewSegRows({ refreshCapabilities: false });
            initGdtRows({ refreshCapabilities: false });
        });
    }
}

export function initOcrEngineRows({ refreshCapabilities = true } = {}) {
    const box = document.getElementById('ocr-engine-rows');
    if (!box) return;
    if (refreshCapabilities) void refreshRecognitionCapabilities();
    box.innerHTML = '';
    const rows = computeOcrEngineRowStates(
        state.recognitionSettings.dimension_path,
        state.ocrEngineCapabilities,
    );
    const current = OCR_ENGINES.includes(state.recognitionSettings.ocr_engine)
        ? state.recognitionSettings.ocr_engine
        : 'rapidocr';

    for (const rowState of rows) {
        const [label, desc] = OCR_ENGINE_LABELS[rowState.engine];
        const disabled = Boolean(rowState.disabled);
        const note = rowState.note ? formatOcrCapabilityReason(rowState.note) : '';
        const row = document.createElement('label');
        row.style.cssText = [
            'display:block',
            'margin-bottom:8px',
            `cursor:${disabled ? 'not-allowed' : 'pointer'}`,
            `opacity:${disabled ? '0.55' : '1'}`,
        ].join(';');
        row.innerHTML = `<span style="display:flex;align-items:center;gap:6px;">
            <input type="radio" name="ocr-engine" value="${rowState.engine}"
                ${rowState.engine === current ? 'checked' : ''}
                ${disabled ? 'disabled' : ''}>
            <strong>${escHtml(label)}</strong></span>
            <span style="display:block;margin-left:22px;color:#888;font-size:11px;">${escHtml(desc)}</span>
            ${note ? `<span style="display:block;margin-left:22px;color:#b26b00;font-size:11px;">${escHtml(note)}</span>` : ''}`;
        box.appendChild(row);
        row.querySelector('input').addEventListener('change', (e) => {
            if (!e.target.checked || e.target.disabled) return;
            state.recognitionSettings.ocr_engine = e.target.value;
            saveRecognitionSettings();
        });
    }
}

export function initViewSegRows({ refreshCapabilities = true } = {}) {
    const box = document.getElementById('view-seg-rows');
    if (!box) return;
    if (refreshCapabilities) void refreshRecognitionCapabilities();
    box.innerHTML = '';
    const dimensionPath = state.recognitionSettings.dimension_path;
    const rows = computeViewSegRowStates(
        state.viewSegCapabilities,
        dimensionPath,
        state.strictViewSegCapability,
    );
    const savedCurrent = VIEW_SEG_BACKENDS.includes(state.recognitionSettings.view_seg)
        ? state.recognitionSettings.view_seg
        : 'yolo';
    const current = dimensionPath === 'vector_only' ? 'yolo' : savedCurrent;

    for (const rowState of rows) {
        const [label, desc] = VIEW_SEG_LABELS[rowState.engine];
        const disabled = Boolean(rowState.disabled);
        const note = rowState.note ? formatViewSegCapabilityReason(rowState.note) : '';
        const row = document.createElement('label');
        row.style.cssText = [
            'display:block',
            'margin-bottom:8px',
            `cursor:${disabled ? 'not-allowed' : 'pointer'}`,
            `opacity:${disabled ? '0.55' : '1'}`,
        ].join(';');
        row.innerHTML = `<span style="display:flex;align-items:center;gap:6px;">
            <input type="radio" name="view-seg" value="${rowState.engine}"
                ${rowState.engine === current ? 'checked' : ''}
                ${disabled ? 'disabled' : ''}>
            <strong>${escHtml(label)}</strong></span>
            <span style="display:block;margin-left:22px;color:#888;font-size:11px;">${escHtml(desc)}</span>
            ${note ? `<span style="display:block;margin-left:22px;color:#b26b00;font-size:11px;">${escHtml(note)}</span>` : ''}`;
        box.appendChild(row);
        row.querySelector('input').addEventListener('change', (e) => {
            if (!e.target.checked || e.target.disabled) return;
            state.recognitionSettings.view_seg = e.target.value;
            saveRecognitionSettings();
        });
    }
}

export function initGdtRows({ refreshCapabilities = true } = {}) {
    const box = document.getElementById('gdt-rows');
    if (!box) return;
    if (refreshCapabilities) void refreshRecognitionCapabilities();
    box.innerHTML = '';
    const rows = computeGdtBackendRowStates(
        state.gdtCapabilities,
        state.recognitionSettings.dimension_path,
    );
    const current = GDT_BACKENDS.includes(state.recognitionSettings.gdt_backend)
        ? state.recognitionSettings.gdt_backend
        : 'hybrid';

    for (const rowState of rows) {
        const [label, desc] = GDT_LABELS[rowState.engine];
        const disabled = Boolean(rowState.disabled);
        const note = rowState.note ? formatGdtCapabilityReason(rowState.note) : '';
        const row = document.createElement('label');
        row.style.cssText = [
            'display:block',
            'margin-bottom:8px',
            `cursor:${disabled ? 'not-allowed' : 'pointer'}`,
            `opacity:${disabled ? '0.55' : '1'}`,
        ].join(';');
        row.innerHTML = `<span style="display:flex;align-items:center;gap:6px;">
            <input type="radio" name="gdt-backend" value="${rowState.engine}"
                ${rowState.engine === current ? 'checked' : ''}
                ${disabled ? 'disabled' : ''}>
            <strong>${escHtml(label)}</strong></span>
            <span style="display:block;margin-left:22px;color:#888;font-size:11px;">${escHtml(desc)}</span>
            ${note ? `<span style="display:block;margin-left:22px;color:#b26b00;font-size:11px;">${escHtml(note)}</span>` : ''}`;
        box.appendChild(row);
        row.querySelector('input').addEventListener('change', (e) => {
            if (!e.target.checked || e.target.disabled) return;
            state.recognitionSettings.gdt_backend = e.target.value;
            saveRecognitionSettings();
        });
    }
}

export function updateSelectedPanel() {
    const infoDiv = document.getElementById('selected-stamp-info');
    if (!infoDiv) return;
    if (!state.selectedUUIDs || state.selectedUUIDs.length === 0) {
        infoDiv.innerHTML = '<span style="color:#999;">未选中任何尺寸</span>';
        return;
    }
    const s = state.stamps.find(st => st.uuid === state.selectedUUIDs[0]);
    if (!s) {
        infoDiv.innerHTML = '<span style="color:#999;">未选中任何尺寸</span>';
        return;
    }
    // 两色按钮从 state.typeColors 实时读，不再硬编码
    const normalHex = state.typeColors.ocr              || '#B85C63';
    const keyHex    = state.typeColors.geometric_anchor || '#4C6FB3';
    const isKey     = resolveIsKey(s);
    const hasManualOverride = (s._colorOverride != null) || (s.isKeyOverride === true || s.isKeyOverride === false);
    const sourceLabel = SOURCE_LABELS[s.source] || s.source || '未知';
    infoDiv.innerHTML = `
        <div style="margin-bottom:6px;">编号: <strong>#${escHtml(s.id)}</strong></div>
        <div style="margin-bottom:6px;">类型: ${escHtml(sourceLabel)}</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <span>颜色：</span>
            <button class="sel-color-btn" data-role="normal"
                style="width:28px;height:22px;background:${normalHex};border:2px solid ${!isKey?'#333':'#ccc'};border-radius:4px;cursor:pointer;" title="普通"></button>
            <button class="sel-color-btn" data-role="key"
                style="width:28px;height:22px;background:${keyHex};border:2px solid ${isKey?'#333':'#ccc'};border-radius:4px;cursor:pointer;" title="重点（跑道框）"></button>
            ${hasManualOverride ? '<button class="sel-color-reset" style="padding:2px 8px;font-size:11px;cursor:pointer;" title="恢复 AI 判定默认">恢复默认</button>' : ''}
        </div>`;
    infoDiv.querySelectorAll('.sel-color-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // 改用 isKeyOverride 而非 _colorOverride，使用户颜色配置能生效
            s.isKeyOverride = (btn.dataset.role === 'key');
            delete s._colorOverride;  // 覆盖掉可能粘住的 preset override
            pushHistory();
            if (_updateTransform) _updateTransform();
            updateSidebar();
            updateSelectedPanel();
        });
    });
    // "恢复默认"按钮清除所有颜色覆盖
    const resetBtn = infoDiv.querySelector('.sel-color-reset');
    if (resetBtn) resetBtn.addEventListener('click', () => {
        delete s._colorOverride;
        delete s.isKeyOverride;
        pushHistory();
        if (_updateTransform) _updateTransform();
        updateSidebar();
        updateSelectedPanel();
    });
}
