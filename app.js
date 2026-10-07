// app.js — orchestrator
// 拆分后的职责：DOM 事件绑定（工具栏 / 快捷键 / 侧边栏 / 小地图 / 控制面板）
// + commitStamp（手绘落章）+ 兼容模式手绘章静默 OCR + 跨模块晚绑定注入
//
// 子模块：
//   app_utils.js        — history / 动画 / showLoading / toHalfWidthAndClean
//   app_cursor.js       — 动态光标 + 工具栏状态同步 + 颜色按钮同步
//   app_analysis.js     — EPS slider + dimensionsToStamps
//   app_sidebar.js      — 侧边栏 / 内联编辑 / 排序 / 控制面板
//   app_interaction.js  — 画布交互（hit-testing / mousedown/move/up / 滚轮 / 视图框重路由）
//   app_io.js           — 文件载入 / 导出 ZIP / HITL feedback
//
// 循环依赖策略：子模块之间用 `registerXxxCallbacks()` 晚绑定，在本文件底部集中注入。

import { CONFIG } from './config.js';
import { state, generateUUID, DEFAULT_TYPE_COLORS, saveTypeColors, pushBounded, DEBUG_LOG_CAPS } from './state.js';
import {
    createStampOcrRequestGate,
    runStampOCR,
    fetchServerStartupLog,
    fetchServerRuntimeStats,
} from './api_client.js';
import { buildDiagnosis } from './app_diagnosis.js';
import {
    els, renderBaseCanvases, updateTransform, syncDrawLayerResolution,
    flashStamp, resolveIsKey,
} from './canvas_renderer.js';
import { initCrop } from './stamp.js';
import {
    documentGlobalBboxToPageLocal,
    getPageGeometry,
} from './page_coordinates.js';

// ── 子模块 ────────────────────────────────────────────────────────────────────
import {
    pushHistory, _restoreSnapshot,
    showLoading, hideLoading, showTransientNotice,
    stepZoom, teleportToMinimap,
    panTo,
    registerCallbacks as registerUtilsCallbacks,
} from './app_utils.js';
import {
    _syncColorToggleBtn, updateCursor, syncToolbarFromSelection,
    updateToolbarState, registerCursorCallbacks,
} from './app_cursor.js';
import {
    initBarPosition, updateEpsSlider,
    dimensionsToStamps,
    _reindexBySpace,
} from './app_analysis.js';
import {
    openControlPanel, closeControlPanel, initColorRows,
    refreshRecognitionCapabilities,
    updateSelectedPanel, updateSidebar,
    idSortLogic, _commitExpandedEdit,
    registerSidebarCallbacks,
    registerReviewCandidateCallbacks,
    getReviewNavigationCandidateIds,
    selectReviewCandidateInWorkbench,
    focusReviewCandidateEditor,
    cancelReviewCandidateEdit,
    confirmReviewCandidateFromWorkbench,
    dismissReviewCandidateFromWorkbench,
    sortOrder, setSortOrder,
} from './app_sidebar.js';
import { registerInteractionCallbacks } from './app_interaction.js';
import './app_io.js';  // 副作用：绑定 file-upload + btn-submit
import {
    confirmReviewCandidate,
    editReviewCandidate,
    listReviewCandidateEntries,
} from './review_candidates_v1.js';
import {
    buildReviewedFormalDimension,
    buildReviewConfirmationPayload,
    buildReviewProvenance,
} from './review_candidate_promotion.js';
import {
    resolveReviewShortcut,
    resolveReviewShortcutTarget,
    shouldPreserveNativeKeyboardTarget,
} from './review_keyboard.js';
import { initVectorDebugUI } from './vector_debug_drawer.js';
import { prefersReducedMotion } from './motion_preferences.js';

// ═══════════════════════════════════════════════════════════════════════════════
// 初始化
// ═══════════════════════════════════════════════════════════════════════════════
pdfjsLib.GlobalWorkerOptions.workerSrc = 'libs/pdf.worker.min.js';

window.addEventListener('load', () => {
    initBarPosition();
    initVectorDebugUI();
    // Prime strict capability authority and the legacy whole-body transport
    // limit before the first user-initiated analysis.
    void refreshRecognitionCapabilities();
    // 后端启动日志：页面打开就拉一次。后端没起来时静默 null，不阻断 UI。
    fetchServerStartupLog();
});
window.addEventListener('resize', () => {
    const ed = document.querySelector('.inline-editor');
    if (ed) ed.blur();
    syncDrawLayerResolution();
    updateSidebar();
    clearTimeout(state.resizeTimer);
    state.resizeTimer = setTimeout(() => updateTransform(), 100);
});

// ═══════════════════════════════════════════════════════════════════════════════
// Undo / Redo
// ═══════════════════════════════════════════════════════════════════════════════

document.getElementById('btn-undo').addEventListener('click', () => {
    if (state.historyIndex <= 0) return;
    state.historyIndex--;
    _restoreSnapshot(state.historyStack[state.historyIndex]);
    document.getElementById('btn-undo').disabled = state.historyIndex <= 0;
    document.getElementById('btn-redo').disabled = false;
    updateEpsSlider();
    updateSidebar(); syncDrawLayerResolution(); syncToolbarFromSelection();
});
document.getElementById('btn-redo').addEventListener('click', () => {
    if (state.historyIndex >= state.historyStack.length - 1) return;
    state.historyIndex++;
    _restoreSnapshot(state.historyStack[state.historyIndex]);
    document.getElementById('btn-undo').disabled = false;
    document.getElementById('btn-redo').disabled =
        state.historyIndex >= state.historyStack.length - 1;
    updateEpsSlider();
    updateSidebar(); syncDrawLayerResolution(); syncToolbarFromSelection();
});

// ═══════════════════════════════════════════════════════════════════════════════
// 工具切换
// ═══════════════════════════════════════════════════════════════════════════════

// 视图 / 编辑 分段控件：两段互斥高亮。点击任一段切到对应模式
function _setToolMode(mode) {
    state.baseTool = mode; state.stampActive = false;
    document.querySelectorAll('#tool-mode-group .seg').forEach(seg => {
        seg.classList.toggle('active', seg.dataset.mode === mode);
    });
    document.getElementById('tool-stamp-toggle').classList.remove('active');
    updateToolbarState(); updateCursor(); syncDrawLayerResolution();
}

document.querySelectorAll('#tool-mode-group .seg').forEach(seg => {
    seg.addEventListener('click', () => _setToolMode(seg.dataset.mode));
});

document.getElementById('tool-stamp-toggle').addEventListener('click', () => {
    state.stampActive = !state.stampActive;
    if (state.stampActive) {
        // 激活标记 → 强制切到 pointer 底层（供 mousedown 捕获），分段高亮同步
        state.baseTool = 'pointer';
        state.drawPhase = 0; state.tempP1 = null;
        document.querySelectorAll('#tool-mode-group .seg').forEach(seg => {
            seg.classList.toggle('active', seg.dataset.mode === 'pointer');
        });
    }
    document.getElementById('tool-stamp-toggle').classList.toggle('active', state.stampActive);
    updateToolbarState(); updateCursor(); syncDrawLayerResolution();
});

// ═══════════════════════════════════════════════════════════════════════════════
// 控制面板入口 + 颜色预设
// ═══════════════════════════════════════════════════════════════════════════════

// 单击切换：普通(红) ↔ 重点(蓝)。
// 双语义：
//   1) 没选中 stamp 时，切 state.tempType 决定"下一笔"画的颜色（保留旧行为）
//   2) 有选中 stamp 时，把所有选中 stamp 切到反向（红↔蓝），改写已存在的而不是
//      只改"下一笔"。
// R14.3 修：与 sidebar / popup 内 stamp-id-badge 点击切换一致——走 isKeyOverride
// 路径，否则 canvas 用 resolveIsKey() 看 isKeyOverride 优先，写 s.is_key 不生效。
// 老 handler 写 s.is_key（canvas 不看）+ 不动 isKeyOverride + 不清 _colorOverride，
// 用户点了视觉无变化，就是这次 bug 报告的根因。
document.getElementById('global-color-toggle').addEventListener('click', () => {
    if (state.selectedUUIDs && state.selectedUUIDs.length > 0) {
        const first = state.stamps.find(s => s.uuid === state.selectedUUIDs[0]);
        if (!first) return;
        // 取第一个 stamp 的真实 KEY 状态（含 override + backend 默认），整批翻转
        const wasKey = resolveIsKey(first);
        const nextIsKey = !wasKey;
        state.selectedUUIDs.forEach(uuid => {
            const s = state.stamps.find(st => st.uuid === uuid);
            if (s) {
                s.isKeyOverride = nextIsKey;
                s.type = nextIsKey ? 'key' : 'normal';   // sync legacy type 字段
                delete s._colorOverride;                  // 解除任何过往颜色锁
            }
        });
        pushHistory();
        updateSidebar();
        syncDrawLayerResolution();
        state.tempType = nextIsKey ? 'key' : 'normal';
    } else {
        state.tempType = (state.tempType === 'key') ? 'normal' : 'key';
    }
    _syncColorToggleBtn();
    updateCursor();
});

document.getElementById('btn-open-panel')?.addEventListener('click', () => {
    const panel = document.getElementById('control-panel');
    if (panel.style.display === 'none') openControlPanel();
    else closeControlPanel();
});
document.getElementById('btn-close-panel')?.addEventListener('click', closeControlPanel);
document.getElementById('control-panel-backdrop')?.addEventListener('click', closeControlPanel);

// 调试日志按钮显隐开关（localStorage 持久化；默认显示）
(() => {
    const LS_KEY = 'showDebugLogBtn';
    const toggle = document.getElementById('toggle-debug-log');
    const btn    = document.getElementById('btn-download-log');
    if (!toggle || !btn) return;
    const saved = localStorage.getItem(LS_KEY);
    const show  = saved === null ? true : saved === '1';
    toggle.checked    = show;
    btn.style.display = show ? '' : 'none';
    toggle.addEventListener('change', () => {
        btn.style.display = toggle.checked ? '' : 'none';
        localStorage.setItem(LS_KEY, toggle.checked ? '1' : '0');
    });
})();

// 只导出已确认项 开关（localStorage 持久化；默认关闭）
// btn-submit 读 checkbox.checked 作为过滤开关，不在这里改导出逻辑
(() => {
    const LS_KEY = 'exportConfirmedOnly';
    const toggle = document.getElementById('toggle-export-confirmed-only');
    if (!toggle) return;
    const saved = localStorage.getItem(LS_KEY);
    toggle.checked = saved === '1';
    toggle.addEventListener('change', () => {
        localStorage.setItem(LS_KEY, toggle.checked ? '1' : '0');
    });
})();

// HITL #3: 点击侧栏行 → 画布定位 + 闪烁高亮（localStorage 持久化；默认开启）
// app_sidebar.js 的单选行点击 handler 读 localStorage['sidebarClickTeleport'] !== '0' 判断
(() => {
    const LS_KEY = 'sidebarClickTeleport';
    const toggle = document.getElementById('toggle-sidebar-click-teleport');
    if (!toggle) return;
    const saved = localStorage.getItem(LS_KEY);
    // 默认开启：saved === null 或 saved === '1' 都算开
    toggle.checked = saved !== '0';
    toggle.addEventListener('change', () => {
        localStorage.setItem(LS_KEY, toggle.checked ? '1' : '0');
    });
})();

// S44h: 隐藏低置信度 AI 标记（localStorage 持久化；默认关闭）
(() => {
    const LS_KEY = 'hideLowConfidence';
    const toggle = document.getElementById('toggle-hide-low-conf');
    if (!toggle) return;
    const saved = localStorage.getItem(LS_KEY);
    toggle.checked = saved === '1';
    state.hideLowConfidence = toggle.checked;
    toggle.addEventListener('change', () => {
        localStorage.setItem(LS_KEY, toggle.checked ? '1' : '0');
        state.hideLowConfidence = toggle.checked;
        updateSidebar();
        syncDrawLayerResolution();
    });
})();

document.getElementById('btn-reset-colors')?.addEventListener('click', () => {
    // 从 DEFAULT_TYPE_COLORS 读单源，不再重复硬编码 hex
    state.typeColors = { ...DEFAULT_TYPE_COLORS };
    saveTypeColors();
    initColorRows();
    _syncColorToggleBtn();
    updateTransform();
    updateSidebar();
});

document.querySelectorAll('input[name="draw-mode"]').forEach(r => {
    r.addEventListener('change', e => { state.drawMode = e.target.value; });
});
document.querySelectorAll('input[name="draw-order"]').forEach(r => {
    r.addEventListener('change', e => {
        state.drawOrder = e.target.value;
        updateCursor();
    });
});
document.getElementById('panel-font-size')?.addEventListener('input', e => {
    const f = document.getElementById('default-font');
    if (f) f.value = e.target.value;
    updateCursor();
});
document.getElementById('panel-shape-ratio')?.addEventListener('input', e => {
    const r = document.getElementById('shape-ratio');
    if (r) r.value = e.target.value;
    updateCursor();
});
// 工具条里的输入框：刷新光标预览 + 实时写回选中 stamp（修：之前选中 stamp
// 后改字号/圈比无效，因为只调 updateCursor 不改 s.fs/s.ratio）
function _applyFsToSelected(fs) {
    if (!Number.isFinite(fs) || fs <= 0) return;
    if (state.selectedUUIDs.length === 0) return;
    state.selectedUUIDs.forEach(uuid => {
        const s = state.stamps.find(st => st.uuid === uuid);
        if (s) s.fs = fs;
    });
    updateSidebar();
    syncDrawLayerResolution();
}
function _applyRatioToSelected(ratio) {
    if (!Number.isFinite(ratio) || ratio <= 0) return;
    if (state.selectedUUIDs.length === 0) return;
    state.selectedUUIDs.forEach(uuid => {
        const s = state.stamps.find(st => st.uuid === uuid);
        if (s) s.ratio = ratio;
    });
    updateSidebar();
    syncDrawLayerResolution();
}
// R14.3.4：current-id input 上下拖拽 / 直接输入时，写回选中 stamp 的 id
//   只单选时生效（多选改 id 会撞重，没意义）；空字符串跳过
//   写回后：updateSidebar 让侧边栏刷新；syncDrawLayer 让画布章序号刷新；
//   selection-change CustomEvent 让 popup header ID 徽章重渲染
function _applyIdToSelected(idStr) {
    const v = String(idStr || '').trim();
    if (!v) return;
    if (state.selectedUUIDs.length !== 1) return;
    const s = state.stamps.find(st => st.uuid === state.selectedUUIDs[0]);
    if (!s || s.id === v) return;
    s.id = v;
    updateSidebar();
    syncDrawLayerResolution();
    document.dispatchEvent(new CustomEvent('pdf-reader:selection-change'));
}
document.getElementById('current-id')?.addEventListener('input', e => {
    _applyIdToSelected(e.target.value);
    updateCursor();
});
document.getElementById('current-id')?.addEventListener('change', () => {
    if (state.selectedUUIDs.length === 1) pushHistory();
});
document.getElementById('default-font')?.addEventListener('input', e => {
    _applyFsToSelected(parseInt(e.target.value, 10));
    updateCursor();
});
document.getElementById('default-font')?.addEventListener('change', () => {
    if (state.selectedUUIDs.length > 0) pushHistory();
});
document.getElementById('shape-ratio')?.addEventListener('input', e => {
    _applyRatioToSelected(parseFloat(e.target.value));
    updateCursor();
});
document.getElementById('shape-ratio')?.addEventListener('change', () => {
    if (state.selectedUUIDs.length > 0) pushHistory();
});

// ── R14.4 长按上下拖拽改数值（Adobe / Figma scrubber 风格） ──────────────
//   工具栏 3 个 input（序号 / 字号 / 圈比）支持：
//     - 按住后向上拖 = 变大，向下拖 = 变小
//     - 没达到 3px 阈值时仍是普通 click → focus → 输入文字
//     - 拖动期间 dispatch 'input'，松手 dispatch 'change'，
//       串到上面已绑的 _applyFsToSelected / _applyRatioToSelected / updateCursor
//   pixelStep = 屏幕每 1px 改多少；向下拖 dy 为正 → 取负
function attachDragScrubber(input, opts = {}) {
    if (!input) return;
    const {
        pixelStep     = 0.5,
        integer       = false,
        decimals      = 2,
        min           = -Infinity,
        max           = Infinity,
        dragThreshold = 3,
    } = opts;
    let startY = 0, startVal = 0, dragging = false;
    const fmt = (n) => integer ? String(Math.round(n)) : n.toFixed(decimals);

    function onMove(e) {
        const dy = e.clientY - startY;
        if (!dragging) {
            if (Math.abs(dy) < dragThreshold) return;
            dragging = true;
            input.blur();  // 阻止 input 文本 caret 干扰
            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';
        }
        // R14.3.1 用户反馈：往下=加，往上=减（dy>0 = 鼠标向下移动 → 加）
        let v = startVal + dy * pixelStep;
        if (v < min) v = min;
        if (v > max) v = max;
        if (integer) v = Math.round(v);
        input.value = fmt(v);
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    function onUp() {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
        if (dragging) {
            input.dispatchEvent(new Event('change', { bubbles: true }));
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
        dragging = false;
    }
    input.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        const v = parseFloat(input.value);
        if (!Number.isFinite(v)) return;  // current-id 多选时是 '-'，跳过
        startY = e.clientY;
        startVal = v;
        dragging = false;
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    });
    // 视觉暗示：hover 时变 ns-resize（输入框还能 click 进入文字 caret）
    input.style.cursor = 'ns-resize';
    input.title = (input.title || '') + (input.title ? '\n' : '') + '上下拖动改值';
}
// R14.3.1 用户反馈：拖拽速度太快"动一下就乱飞"，整体减速 ~3x
attachDragScrubber(document.getElementById('default-font'),
    { pixelStep: 1/8, integer: true, min: 8, max: 96 });    // 拖 8px 改 1（原 3px）
attachDragScrubber(document.getElementById('shape-ratio'),
    { pixelStep: 0.002, integer: false, decimals: 2, min: 0.2, max: 2.0 });  // 5px 改 0.01（原 2px）
attachDragScrubber(document.getElementById('current-id'),
    { pixelStep: 0.1, integer: true, min: 1, max: 9999 });  // 拖 10px 改 1（原 2.5px）
// 控制面板 panel 输入也对选中生效（已通过 input 同步 default-font / shape-ratio
// 的 value，再触发它们的 input 事件即可串到上面的写回逻辑）
document.getElementById('panel-font-size')?.addEventListener('input', () => {
    document.getElementById('default-font')?.dispatchEvent(new Event('input'));
});
// change 单独转发，串到 default-font 的 change 处理器以触发 pushHistory
document.getElementById('panel-font-size')?.addEventListener('change', () => {
    document.getElementById('default-font')?.dispatchEvent(new Event('change'));
});
document.getElementById('panel-shape-ratio')?.addEventListener('input', () => {
    document.getElementById('shape-ratio')?.dispatchEvent(new Event('input'));
});
// change 单独转发，串到 shape-ratio 的 change 处理器以触发 pushHistory
document.getElementById('panel-shape-ratio')?.addEventListener('change', () => {
    document.getElementById('shape-ratio')?.dispatchEvent(new Event('change'));
});

// ═══════════════════════════════════════════════════════════════════════════════
// 调试日志下载：导出当前会话的诊断记录，供本地分析。
// ═══════════════════════════════════════════════════════════════════════════════

// window.error / unhandledrejection 全局捕获，进 state.runtimeErrors
window.addEventListener('error', (e) => {
    pushBounded(state.runtimeErrors, {
        ts: new Date().toISOString(),
        type: 'error',
        message: e.message || String(e),
        source: e.filename || null,
        line:   e.lineno   || null,
        col:    e.colno    || null,
        stack:  e.error?.stack || null,
    }, DEBUG_LOG_CAPS.runtimeErrors);
});
window.addEventListener('unhandledrejection', (e) => {
    const r = e.reason;
    pushBounded(state.runtimeErrors, {
        ts: new Date().toISOString(),
        type: 'unhandledrejection',
        message: r?.message || String(r),
        source: null, line: null, col: null,
        stack:  r?.stack || null,
    }, DEBUG_LOG_CAPS.runtimeErrors);
});

document.getElementById('btn-download-log').addEventListener('click', async () => {
    const hasAnyTrace =
        state.analyzeRuns.length   > 0 ||
        state.stampOcrCalls.length > 0 ||
        state.exportCalls.length   > 0 ||
        state.runtimeErrors.length > 0 ||
        state.stamps.length        > 0 ||
        state.serverStartupLog     != null;
    if (!hasAnyTrace) {
        alert('没有可用的调试记录。请先打开图纸或运行一次解析。');
        return;
    }

    // 下载前再刷一次后端运行时计数 + 启动日志（启动日志可能页面打开时后端还没起来）
    await Promise.allSettled([
        fetchServerRuntimeStats(),
        state.serverStartupLog ? Promise.resolve() : fetchServerStartupLog(),
    ]);

    const stampsAuto    = state.stamps.filter(s => s.isAutoGenerated).length;
    const stampsManual  = state.stamps.length - stampsAuto;
    const stampsConfirm = state.stamps.filter(s => s.confirmed).length;
    const vbEdited      = state.viewBoxes.filter(v => v.isEdited).length;

    // R14.2 perf 聚合：跨多页 /analyze runs 算 per-stage avg/min/max/sum，
    // 排序输出 top stages 让"哪个阶段是大头"一眼可见。
    // 输入：state.analyzeRuns[].timings (后端 pipeline.timings dict, 18+ keys)
    //      state.analyzeRuns[].ocr_breakdown (per-pass region count)
    const perf = (() => {
        const stageAgg = {};
        let nWithTimings = 0;
        for (const r of state.analyzeRuns) {
            if (!r.timings) continue;
            nWithTimings++;
            for (const [k, v] of Object.entries(r.timings)) {
                if (typeof v !== 'number') continue;  // 跳过 _render_calls 这种 list 字段
                if (!stageAgg[k]) {
                    stageAgg[k] = { runs: 0, sum: 0, min: Infinity, max: -Infinity };
                }
                const a = stageAgg[k];
                a.runs++; a.sum += v;
                if (v < a.min) a.min = v;
                if (v > a.max) a.max = v;
            }
        }
        const per_stage = {};
        for (const [k, a] of Object.entries(stageAgg)) {
            per_stage[k] = {
                runs: a.runs,
                sum: +a.sum.toFixed(3),
                avg: +(a.sum / a.runs).toFixed(3),
                min: +a.min.toFixed(3),
                max: +a.max.toFixed(3),
            };
        }
        // top_stages：按 avg 降序，排除 total / vector_extract（聚合值，会重复计算）
        // 和下划线开头的 meta 字段（render_count 这种）
        const top_stages = Object.entries(per_stage)
            .filter(([k, _]) => k !== 'total' && k !== 'vector_extract' && !k.startsWith('_'))
            .sort((a, b) => b[1].avg - a[1].avg)
            .slice(0, 10)
            .map(([k, v]) => ({
                stage: k, avg: v.avg, max: v.max, sum: v.sum, runs: v.runs,
            }));
        // ocr_breakdown 跨 runs 累加（看哪个 OCR pass 产出最多 region）
        const ocr_breakdown_sum = {};
        for (const r of state.analyzeRuns) {
            const b = r.ocr_breakdown;
            if (!b) continue;
            for (const [k, v] of Object.entries(b)) {
                if (typeof v !== 'number') continue;
                ocr_breakdown_sum[k] = (ocr_breakdown_sum[k] || 0) + v;
            }
        }
        // ocr_pass_counters 跨 runs 合并（directed: regions_total / processed / skipped 等）
        const ocr_pass_counters_sum = {};
        for (const r of state.analyzeRuns) {
            const c = r.ocr_pass_counters;
            if (!c) continue;
            for (const [pass, kv] of Object.entries(c)) {
                if (!ocr_pass_counters_sum[pass]) ocr_pass_counters_sum[pass] = {};
                for (const [k, v] of Object.entries(kv || {})) {
                    if (typeof v !== 'number') continue;
                    ocr_pass_counters_sum[pass][k] = (ocr_pass_counters_sum[pass][k] || 0) + v;
                }
            }
        }
        return {
            runs_with_timings: nWithTimings,
            per_stage,
            top_stages,
            ocr_breakdown_sum,
            ocr_pass_counters_sum,
        };
    })();

    // R14.5：诊断 sanity-check（_buildDiagnosis 在 perf 之上跑，输出嵌 index.json）
    const diagnosis = buildDiagnosis(state, perf);

    // 公共顶层元数据（小，约 1-3KB），单独成 index.json
    const indexObj = {
        schema:      'pdf_reader_debug_v2',
        exported_at: new Date().toISOString(),
        doc: {
            filename:   state.docInfo?.filename || null,
            version:    state.docInfo?.version  || null,
            totalPages: state.totalPages,
            pdfWidth:   state.pdfWidth,
            pdfHeight:  state.pdfHeight,
            rotation:   state.rotation,
            pdfBytes:   state.currentPdfBuffer ? state.currentPdfBuffer.byteLength : 0,
        },
        summary: {
            analyze_runs_n:   state.analyzeRuns.length,
            stamp_ocr_n:      state.stampOcrCalls.length,
            export_calls_n:   state.exportCalls.length,
            runtime_errors_n: state.runtimeErrors.length,
            stamps_total:     state.stamps.length,
            stamps_auto:      stampsAuto,
            stamps_manual:    stampsManual,
            stamps_confirmed: stampsConfirm,
            deleted_stamps:   state.deletedStamps.length,
            view_boxes:       state.viewBoxes.length,
            view_boxes_edited: vbEdited,
            analyzed_pages:   Array.from(state.currentAnalyzedPages || []),
        },
        perf,
        diagnosis,
        env: {
            ua:       navigator.userAgent,
            language: navigator.language,
            screen:   { w: screen.width,        h: screen.height        },
            viewport: { w: window.innerWidth,   h: window.innerHeight   },
            dpr:      window.devicePixelRatio || 1,
        },
        // 各分片文件路径声明，方便后处理工具按段读
        files: {
            'analyze_runs.json':    'analyze_runs[] (含 debug_log 文本 + dimensions，最大段)',
            'stamps.json':          'current_state（stamps / deletedStamps / viewBoxes / clusters 等）',
            'stamp_ocr_calls.json': '手绘章 /analyze_stamp 调用日志',
            'server_log.json':      'server_startup_log + server_runtime_stats',
            'errors.json':          'runtime_errors + export_calls',
        },
    };

    const fnSafe = (state.docInfo?.filename || 'untitled').replace(/[^\w.\-]+/g, '_').replace(/\.pdf$/i, '');
    const tsSafe = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const baseName = `debug_${fnSafe}_${tsSafe}`;

    // R14.5：拆 zip 多文件，避免 mega-JSON 全量加载
    //   index.json   : 顶层 meta + summary + perf + diagnosis（必读，<10KB）
    //   analyze_runs : 大头（debug_log 字符串 + dimensions）
    //   stamps       : current_state
    //   其余分文件
    if (typeof window.JSZip !== 'function') {
        // 兜底：JSZip 不可用时退回 mega-JSON（不应发生，index.html 已加载）
        const mega = { ...indexObj,
            analyze_runs: state.analyzeRuns,
            stamp_ocr_calls: state.stampOcrCalls,
            export_calls: state.exportCalls,
            runtime_errors: state.runtimeErrors,
            current_state: {
                stamps: state.stamps, deletedStamps: state.deletedStamps,
                viewBoxes: state.viewBoxes, clusters: state.clusters,
                analyzeScope: state.analyzeScope, drawMode: state.drawMode,
                drawOrder: state.drawOrder, tempType: state.tempType,
                hideLowConfidence: state.hideLowConfidence, currentEps: state.currentEps,
            },
            server_startup_log: state.serverStartupLog,
            server_runtime_stats: state.serverRuntimeStats,
        };
        const blob = new Blob([JSON.stringify(mega, null, 2)], { type: 'application/json' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = `${baseName}.json`; a.click();
        URL.revokeObjectURL(url);
        return;
    }

    const zip = new window.JSZip();
    zip.file(`${baseName}/index.json`,            JSON.stringify(indexObj, null, 2));
    zip.file(`${baseName}/analyze_runs.json`,     JSON.stringify(state.analyzeRuns, null, 2));
    zip.file(`${baseName}/stamps.json`,           JSON.stringify({
        stamps:        state.stamps,
        deletedStamps: state.deletedStamps,
        viewBoxes:     state.viewBoxes,
        clusters:      state.clusters,
        analyzeScope:  state.analyzeScope,
        drawMode:      state.drawMode,
        drawOrder:     state.drawOrder,
        tempType:      state.tempType,
        hideLowConfidence: state.hideLowConfidence,
        currentEps:    state.currentEps,
    }, null, 2));
    zip.file(`${baseName}/stamp_ocr_calls.json`,  JSON.stringify(state.stampOcrCalls, null, 2));
    zip.file(`${baseName}/server_log.json`,       JSON.stringify({
        server_startup_log:   state.serverStartupLog,
        server_runtime_stats: state.serverRuntimeStats,
    }, null, 2));
    zip.file(`${baseName}/errors.json`,           JSON.stringify({
        runtime_errors: state.runtimeErrors,
        export_calls:   state.exportCalls,
    }, null, 2));

    const zipBlob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 } });
    const url = URL.createObjectURL(zipBlob);
    const a   = document.createElement('a');
    a.href = url; a.download = `${baseName}.zip`; a.click();
    URL.revokeObjectURL(url);
});

// ═══════════════════════════════════════════════════════════════════════════════
// EPS Slider（复用 l1Cache，不重新请求后端）
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('input', async (e) => {
    if (e.target.id !== 'eps-slider') return;
    if (state.clusteringLocked) return;
    if (state.isAnalysisRunning) return;

    const curPage = parseInt(document.getElementById('page-indicator').innerText.split('/')[0]);
    const cachedDims = state.l1Cache.get(curPage);
    if (!cachedDims) return;

    state.currentEps = parseFloat(e.target.value);

    state.stamps = state.stamps.filter(s => !(s.isAutoGenerated && s.pageIndex === curPage));

    const pageViewBoxes = state.viewBoxes.filter(v => v.pageIndex === curPage);
    const { stamps: newStamps, clusters } = dimensionsToStamps(cachedDims, curPage, pageViewBoxes);
    state.clusters = clusters;
    state.stamps   = [...state.stamps, ...newStamps];

    pushHistory();
    state.clusteringLocked    = false;
    state.clusteringBaseIndex = state.historyIndex;
    updateEpsSlider();
    updateSidebar();
    syncDrawLayerResolution();
});

// ═══════════════════════════════════════════════════════════════════════════════
// 键盘快捷键 + 快捷键提示层
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('keydown', e => {
    if (shouldPreserveNativeKeyboardTarget({
        tagName: e.target?.tagName,
        isContentEditable: Boolean(e.target?.isContentEditable),
        isNativeInteractive: Boolean(
            e.target?.closest?.('button, summary, a[href], [role="button"]'),
        ),
    })) return;

    if (e.code === 'Space')           { e.preventDefault(); state.keys.space = true; updateCursor(); }
    if (e.key === 'Control' || e.key === 'Meta') state.keys.ctrl = true;
    if (e.key === 'Alt')              state.keys.alt = true;
    if (e.key === 'Shift')            { state.keys.shift = true; updateCursor(); }

    if (e.key === '?') { e.preventDefault(); _toggleShortcutsOverlay(); return; }

    if (e.key === 'Escape') {
        if (_isShortcutsOverlayOpen()) { _toggleShortcutsOverlay(false); return; }
        if (state.stampActive && state.drawPhase > 0) {
            state.drawPhase = 0; state.tempP1 = null;
            updateCursor();
            syncDrawLayerResolution();
        } else {
            state.selectedUUIDs = [];
            syncToolbarFromSelection(); updateSidebar(); syncDrawLayerResolution();
        }
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && state.selectedUUIDs.length > 0) {
        // Backspace 在 input/textarea 里有"退格"语义；用户在编辑字段时不应触发删 stamp
        const t = e.target;
        const isEditing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
        if (isEditing) return;
        const toDelete = state.selectedUUIDs.filter(uuid => {
            const s = state.stamps.find(st => st.uuid === uuid);
            return s && !s.confirmed;
        });
        if (toDelete.length === 0) return;
        e.preventDefault();   // 防止 Backspace 触发浏览器后退
        const removing = state.stamps.filter(s => toDelete.includes(s.uuid) && s.isAutoGenerated);
        state.deletedStamps.push(...removing);
        state.stamps = state.stamps.filter(s => !toDelete.includes(s.uuid));
        state.selectedUUIDs = [];
        pushHistory(); updateSidebar(); syncDrawLayerResolution(); syncToolbarFromSelection();
    }
    // HITL #1: 方向键微调选中章的 circleCenter（1pt / Shift 5pt；已 confirmed 不动）
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)
        && state.selectedUUIDs.length > 0 && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const step = e.shiftKey ? 5 : 1;
        const dx = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step : 0;
        const dy = e.key === 'ArrowUp'   ? -step : e.key === 'ArrowDown'  ? step : 0;
        let moved = 0;
        state.selectedUUIDs.forEach(uuid => {
            const s = state.stamps.find(st => st.uuid === uuid);
            if (!s || s.confirmed) return;
            s.circleCenter.absX += dx;
            s.circleCenter.absY += dy;
            // routingPath 首端 = circleCenter，一起更新保持引线拓扑
            if (s.routingPath && s.routingPath.length >= 2) {
                s.routingPath[0].absX += dx;
                s.routingPath[0].absY += dy;
            }
            moved++;
        });
        if (moved > 0) {
            e.preventDefault();
            pushHistory(); syncDrawLayerResolution(); updateSidebar();
        }
    }
    // HITL #2: N 键跳到下一个未确认项（按 id 升序，到末尾 wrap 到第一个）
    if ((e.key === 'n' || e.key === 'N') && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
        if (state.stamps.length === 0) return;
        const sorted = [...state.stamps].sort(idSortLogic);
        const unconfirmed = sorted.filter(s => !s.confirmed);
        if (unconfirmed.length === 0) return;
        e.preventDefault();
        const curUuid = state.selectedUUIDs.length > 0 ? state.selectedUUIDs[0] : null;
        const curIdx  = curUuid ? sorted.findIndex(s => s.uuid === curUuid) : -1;
        let next = sorted.slice(curIdx + 1).find(s => !s.confirmed);
        if (!next) next = unconfirmed[0];  // wrap 到第一个未确认
        if (!next || next.uuid === curUuid) return;
        state.selectedUUIDs = [next.uuid];
        syncToolbarFromSelection(); updateSidebar(); syncDrawLayerResolution();
        panTo(next); flashStamp(next.uuid);
    }
    // HITL #3: A/D/S 三键流式确认
    //   A → 上一个章；D → 下一个章；S → 确认当前 + 跳到下一个未确认
    //   排序 / 可见性与 sidebar 完全一致（idSortLogic + hideLowConfidence 过滤）
    if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey
        && (e.key === 'a' || e.key === 'A'
         || e.key === 'd' || e.key === 'D'
         || e.key === 's' || e.key === 'S')) {
        const _isLowConfAi = s => (s.isAutoGenerated && !s.confirmed && s.aiData?.confidence === 'low');
        const visible = state.hideLowConfidence
            ? state.stamps.filter(s => !_isLowConfAi(s))
            : state.stamps;
        const sorted = [...visible].sort(idSortLogic);
        if (sorted.length === 0) return;

        const key = e.key.toLowerCase();
        const curUuid = state.selectedUUIDs.length > 0 ? state.selectedUUIDs[0] : null;
        const curIdx  = curUuid ? sorted.findIndex(s => s.uuid === curUuid) : -1;

        let nextIdx = -1;
        if (key === 'a') {
            // 上一个：已是首项就停在首项；无选中默认落到首项
            if (curIdx === -1)      nextIdx = 0;
            else if (curIdx > 0)    nextIdx = curIdx - 1;
            else                    nextIdx = 0;
        } else if (key === 'd') {
            // 下一个：已是末项就停在末项；无选中默认落到首项
            if (curIdx === -1)                       nextIdx = 0;
            else if (curIdx < sorted.length - 1)     nextIdx = curIdx + 1;
            else                                     nextIdx = curIdx;
        } else { // 's'
            // Fix 2 (P2)：复用 _commitExpandedEdit 走统一路径，确保 activeEditUUID 同步推进，
            // 否则旧逻辑只设 confirmed，updateSidebar 仍按旧 activeEditUUID 渲染（已确认行的
            // 编辑区还在，但选中已跳到下一项），UI 不一致。
            // _commitExpandedEdit 内部已 pushHistory + updateSidebar + syncDrawLayerResolution
            // + 推进 activeEditUUID/selectedUUIDs 到下一未确认项，所以这里不再重复。
            //
            // Fix 1 (P2 follow-up)：S 必须保留"curIdx+1 起 wrap 找未确认"的顺序语义。
            //   _commitExpandedEdit 默认实现是"sorted 中第一个未确认（排除自身）"——
            //   总是跳第一个未确认，破坏顺序确认体验。
            //   这里先按 startFrom = curIdx+1 wrap 算出 nextUUID，作为第 4 参数传入，
            //   让 commit 用我们指定的下一项；sidebar OK 等不传 nextUUID 的调用方维持旧行为。
            if (curIdx >= 0 && !sorted[curIdx].confirmed) {
                // 计算 next：从 curIdx+1 开始找未确认；找不到从头找（wrap）
                const startFrom = curIdx + 1;
                let nextSearch = sorted.findIndex((st, i) => i >= startFrom && !st.confirmed);
                if (nextSearch === -1) {
                    nextSearch = sorted.findIndex((st, i) => i < curIdx && !st.confirmed);
                }
                const nextUUID = nextSearch >= 0 ? sorted[nextSearch].uuid : null;
                _commitExpandedEdit(sorted[curIdx].uuid, sorted, undefined, nextUUID);
                // _commitExpandedEdit 已把 selectedUUIDs 推进到 nextUUID；这里再读一次。
                const newSelUuid = state.selectedUUIDs[0];
                nextIdx = newSelUuid ? sorted.findIndex(st => st.uuid === newSelUuid) : -1;
                if (nextIdx === -1) nextIdx = curIdx;  // 全部 confirmed，待在原地
            } else {
                // 没选中 / 当前已 confirmed：直接找一个未确认项跳过去（保留原 fallback 行为）
                const startFrom = curIdx >= 0 ? curIdx + 1 : 0;
                nextIdx = sorted.findIndex((s, i) => i >= startFrom && !s.confirmed);
                if (nextIdx === -1) nextIdx = sorted.findIndex(s => !s.confirmed);
                if (nextIdx === -1) nextIdx = curIdx;
            }
        }

        e.preventDefault();
        if (nextIdx >= 0 && nextIdx < sorted.length && nextIdx !== curIdx) {
            const target = sorted[nextIdx];
            state.selectedUUIDs = [target.uuid];
            panTo(target);
            flashStamp(target.uuid);
            syncToolbarFromSelection();
            // Fix 2：S 路径下 _commitExpandedEdit 已 updateSidebar / syncDrawLayerResolution，
            // 这里只在 A/D 分支再 render；S 分支跳过避免 double render。
            if (key !== 's') {
                updateSidebar();
                syncDrawLayerResolution();
            }
        }
    }
    if (e.ctrlKey || e.metaKey) {
        if (e.code === 'KeyZ') { e.preventDefault(); document.getElementById('btn-undo').click(); }
        if (e.code === 'KeyY') { e.preventDefault(); document.getElementById('btn-redo').click(); }
    }
});
document.addEventListener('keyup', e => {
    if (e.code === 'Space')           { state.keys.space = false; updateCursor(); }
    if (e.key === 'Control' || e.key === 'Meta') state.keys.ctrl = false;
    if (e.key === 'Alt')              state.keys.alt = false;
    if (e.key === 'Shift')            { state.keys.shift = false; updateCursor(); }
});

// ── 快捷键提示层（纯 UI 叠加）────────────────────────────────────────────
let _shortcutsOverlay = null;
function _buildShortcutsOverlay() {
    if (_shortcutsOverlay) return _shortcutsOverlay;
    const root = document.createElement('div');
    root.id = 'shortcuts-overlay';
    root.style.cssText =
        'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.55);' +
        'display:none;align-items:center;justify-content:center;' +
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;';
    const card = document.createElement('div');
    card.style.cssText =
        'background:#fff;border-radius:12px;padding:24px 28px;' +
        'box-shadow:0 12px 36px rgba(0,0,0,0.25);min-width:460px;max-width:620px;' +
        'max-height:85vh;overflow:auto;';
    const row = (k, v) =>
        `<tr><td style="padding:5px 0;color:#666;width:170px;white-space:nowrap;">${k}</td>` +
        `<td style="padding:5px 0;">${v}</td></tr>`;
    const section = (title) =>
        `<tr><td colspan="2" style="padding-top:14px;padding-bottom:4px;font-weight:600;` +
        `border-top:1px solid #eee;color:#333;">${title}</td></tr>`;
    card.innerHTML =
        '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px;">' +
            '<h2 style="margin:0;font-size:17px;">快捷键</h2>' +
            '<span style="font-size:12px;color:#999;">按 ? 或 Esc 关闭</span>' +
        '</div>' +
        '<table style="width:100%;font-size:13px;border-collapse:collapse;"><tbody>' +
            section('画布') +
            row('Space + 左键 / 中键', '平移画布') +
            row('Ctrl/⌘ + 滚轮', '缩放') +
            row('Ctrl/⌘ + Z / Y', '撤销 / 重做') +
            row('Delete / Backspace', '删除未确认标记 / 拒绝选中的待复核候选') +
            row('N', '跳到下一个未确认项（按 ID 升序 + 高亮）') +
            row('Esc', '取消当前盖章 / 清空选区') +
            section('盖章') +
            row('左键', '普通章（红）') +
            row('右键 / Shift + 左键', '关键章（蓝）') +
            row('Shift（按住）', '光标预告关键章颜色') +
            section('侧边栏列表') +
            row('↑ / ↓', '切换条目') +
            row('A / D', '上一个 / 下一个尺寸（全局，输入框聚焦时不响应）') +
            row('S', '确认当前 + 跳到下一未确认') +
            row('Enter', '进入编辑 / 从编辑中确认条目') +
            row('Esc', '取消当前编辑') +
            section('帮助') +
            row('?', '打开 / 关闭本提示层') +
        '</tbody></table>';
    root.appendChild(card);
    root.addEventListener('click', (e) => {
        if (e.target === root) _toggleShortcutsOverlay(false);
    });
    document.body.appendChild(root);
    _shortcutsOverlay = root;
    return root;
}
function _toggleShortcutsOverlay(show) {
    const el = _buildShortcutsOverlay();
    const willShow = (show === undefined) ? (el.style.display !== 'flex') : show;
    el.style.display = willShow ? 'flex' : 'none';
}
function _isShortcutsOverlayOpen() {
    return _shortcutsOverlay && _shortcutsOverlay.style.display === 'flex';
}
document.getElementById('btn-shortcuts')?.addEventListener('click', () => _toggleShortcutsOverlay());

// ═══════════════════════════════════════════════════════════════════════════════
// 兼容模式手绘章静默 OCR（strict 由共享 gate 拦截；commitStamp 尾部调用）
// ═══════════════════════════════════════════════════════════════════════════════

async function _performStampOCR(s) {
    if (!s || s.isAutoGenerated) return;
    if (!s.crop) return;

    const tipX = s.arrowTip.absX, tipY = s.arrowTip.absY;
    const x = tipX + s.crop.l;
    const y = tipY + s.crop.t;
    const w = s.crop.r - s.crop.l;
    const h = s.crop.b - s.crop.t;
    if (w <= 1 || h <= 1) return;

    const pageIdx = s.pageIndex || 1;
    const page = getPageGeometry(state.pageOffsets, pageIdx);
    const localCrop = documentGlobalBboxToPageLocal({ x, y, w, h }, page);

    // HITL #8: 标记"OCR 已请求"，供渲染层判断是否显示失败红点
    const live0 = state.stamps.find(st => st.uuid === s.uuid);
    if (live0) { live0.ocrRequested = true; syncDrawLayerResolution(); }

    const result = await runStampOCR(pageIdx, localCrop, 300, s.uuid);

    const live = state.stamps.find(st => st.uuid === s.uuid);
    if (!live) return;
    live.aiData = live.aiData || { type:'', nominal:'', tolerance:'', symbol:'', datum:'', remarks:'' };
    if (result && result.text) {
        live.aiData.nominal = result.text;
        live.aiData.remarks = (result.candidates || []).map(c => c.text).join(' | ');
    } else {
        // 没识别到文字：保留 ocrRequested=true 让红点常亮，nominal 不动（用户可手填）
    }

    updateSidebar();
    updateSelectedPanel();
    syncDrawLayerResolution();
}

const _requestStampOCR = createStampOcrRequestGate({
    getDimensionPath: () => state.recognitionSettings?.dimension_path,
    requestOcr: _performStampOCR,
    notify: showTransientNotice,
});

// ═══════════════════════════════════════════════════════════════════════════════
// 手动绘制 Stamp（TDD 3.1 全字段 + A-02 cropAdjusted=false）
// ═══════════════════════════════════════════════════════════════════════════════

function commitStamp(p1, p2) {
    const s = {
        uuid:            generateUUID(),
        id:              document.getElementById('current-id').value,
        type:            state.tempType,
        fs:              parseInt(document.getElementById('default-font').value),
        ratio:           parseFloat(document.getElementById('shape-ratio').value) || 0.60,
        isAutoGenerated: false,
        circleCenter:    { absX: p1.x, absY: p1.y },
        arrowTip:        { absX: p2.x, absY: p2.y },
        arrowTipManual:  true,
        crop:            null,
        cropAdjusted:    false,
        pageIndex:       (() => {
            for (let i = 0; i < state.pageOffsets.length; i++) {
                const po = state.pageOffsets[i];
                if (p2.y >= po.start && p2.y <= po.start + po.height) return i + 1;
            }
            return 1;
        })(),
        status:          'pending',
        confirmed:       false,
        aiData:          { type:'', nominal:'', tolerance:'', symbol:'', datum:'', remarks:'' },
    };
    initCrop(s);
    state.stamps.push(s);

    const inc = document.getElementById('increment-mode').value;
    if (inc === 'auto') {
        // fill: 从 1 开始找"当前命名空间"里第一个未占用整数（初稿推荐）
        // next: 取当前命名空间最大号 + 1（改稿阶段推荐，避免旧号被复用）
        const cur = document.getElementById('current-id').value;
        const m = cur.match(/^(.*?)(\d+)([^\d]*)$/);
        if (m) {
            const prefix = m[1], suffix = m[3];
            const used = new Set();
            for (const st of state.stamps) {
                const sm = String(st.id).match(/^(.*?)(\d+)([^\d]*)$/);
                if (sm && sm[1] === prefix && sm[3] === suffix) {
                    used.add(parseInt(sm[2]));
                }
            }
            const strategy = document.querySelector('input[name="inc-strategy"]:checked')?.value || 'fill';
            let next;
            if (strategy === 'next') {
                let maxN = 0;
                for (const n of used) if (n > maxN) maxN = n;
                next = maxN + 1;
            } else {
                next = 1;
                while (used.has(next)) next++;
            }
            document.getElementById('current-id').value = prefix + next + suffix;
        }
    }
    pushHistory(); updateSidebar(); syncDrawLayerResolution();
    updateCursor();

    _requestStampOCR(s);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 缩放 & 导航
// ═══════════════════════════════════════════════════════════════════════════════

document.getElementById('btn-zoom-out').onclick   = () => stepZoom(-1);
document.getElementById('btn-zoom-in').onclick    = () => stepZoom(1);
document.getElementById('btn-actual-size').onclick = () => {
    const wsRect = els.workspace.getBoundingClientRect();
    state.panX = wsRect.width/2  - (wsRect.width/2  - state.panX) / state.zoom;
    state.panY = wsRect.height/2 - (wsRect.height/2 - state.panY) / state.zoom;
    state.zoom = 1.0; updateTransform();
};
document.getElementById('btn-fit-page').onclick   = () => {
    const wsRect = els.workspace.getBoundingClientRect();
    const cur    = parseInt(document.getElementById('page-indicator').innerText.split('/')[0]) - 1;
    if (!state.pageOffsets[cur]) return;
    const ph = state.pageOffsets[cur].height;
    state.zoom = Math.min((wsRect.width-60)/state.pdfWidth, (wsRect.height-60)/ph);
    state.panX = (wsRect.width  - state.pdfWidth * state.zoom) / 2;
    state.panY = -state.pageOffsets[cur].start * state.zoom +
                 (wsRect.height - ph * state.zoom) / 2;
    updateTransform();
};

document.getElementById('btn-prev-page').onclick = () => {
    const cur = parseInt(document.getElementById('page-indicator').innerText.split('/')[0]);
    if (cur > 1) { state.panY = -state.pageOffsets[cur-2].start * state.zoom + 30; updateTransform(); }
};
document.getElementById('btn-next-page').onclick = () => {
    const cur = parseInt(document.getElementById('page-indicator').innerText.split('/')[0]);
    if (cur < state.totalPages) { state.panY = -state.pageOffsets[cur].start * state.zoom + 30; updateTransform(); }
};

// ═══════════════════════════════════════════════════════════════════════════════
// 小地图
// ═══════════════════════════════════════════════════════════════════════════════

let isDraggingMm = false;
els.minimap.addEventListener('mousedown', e => { isDraggingMm = true; teleportToMinimap(e); });
window.addEventListener('mousemove', e => { if (isDraggingMm) teleportToMinimap(e); });
window.addEventListener('mouseup',   () => { isDraggingMm = false; });

// ═══════════════════════════════════════════════════════════════════════════════
// 侧边栏 tab / 键盘导航 / 全部确认 / 重编号 / 排序
// ═══════════════════════════════════════════════════════════════════════════════

const sbContainer = document.getElementById('sidebar-container');
const sbTab       = document.getElementById('sidebar-tab');
sbTab.addEventListener('click', () => {
    // 测量切换前高度（"数据列表"4 字 vs "收起"2 字，writing-mode vertical-rl 下高度不同）
    const fromH = sbTab.offsetHeight;

    // CSS 规范：transition 一旦启动，后续改 transition-* 属性不影响当前动画；
    // 必须在 classList.toggle 触发 right 变化前写入本次 motion 参数。
    const reduceMotion = prefersReducedMotion();
    let easing;
    let durMs;
    if (reduceMotion) {
        easing = 'linear';
        durMs = '1';
    } else {
        // 默认手感是有意设计：每次使用略有不同的时长与过冲，保持原参数不变。
        const dur = 290 + (Math.random() - 0.5) * 40;             // 270-310ms
        const c1x = 0.30 + (Math.random() - 0.5) * 0.10;         // 0.25-0.35
        const c1y = 1.25 + (Math.random() - 0.5) * 0.12;         // 1.19-1.31（>1 = 弹跳峰值）
        const c2x = 0.50 + (Math.random() - 0.5) * 0.16;         // 0.42-0.58
        easing = `cubic-bezier(${c1x.toFixed(3)},${c1y.toFixed(3)},${c2x.toFixed(3)},1)`;
        durMs = dur.toFixed(0);
    }
    sbContainer.style.transition = `right ${durMs}ms ${easing}`;

    sbContainer.classList.toggle('active');
    const active = sbContainer.classList.contains('active');
    const label = sbTab.querySelector('.sidebar-tab-label');
    if (label) label.textContent = active ? '收起' : '数据列表';
    // 切 <use href> 替代 CSS transform（SVG 在 writing-mode:vertical-rl 下 rotate 不生效）
    const use = sbTab.querySelector('.sidebar-tab-chevron use');
    if (use) use.setAttribute('href', active ? '#i-chevron-right' : '#i-chevron-left');

    if (reduceMotion) {
        // 零/近零 transition 不保证派发 transitionend，必须即时恢复自然高度。
        sbTab.style.height = '';
        sbTab.style.transition = '';
    } else {
        // tab 高度：先锁当前高到 inline style → 触发 reflow → 过渡到目标高
        sbTab.style.transition = 'none';
        sbTab.style.height = 'auto';
        const toH = sbTab.offsetHeight;
        sbTab.style.height = fromH + 'px';
        void sbTab.offsetHeight;                                 // 强制 reflow
        sbTab.style.transition = `height ${durMs}ms ${easing}, background var(--t-med) var(--easing)`;
        sbTab.style.height = toH + 'px';

        // 过渡结束清理 inline height 让后续自然流动（否则下次测 fromH 会锁死）
        const cleanup = (e) => {
            if (e.target !== sbTab || e.propertyName !== 'height') return;
            sbTab.style.height = '';
            sbTab.style.transition = '';
            sbTab.removeEventListener('transitionend', cleanup);
        };
        sbTab.addEventListener('transitionend', cleanup);
    }

    updateTransform();
});

document.getElementById('stamp-list').addEventListener('keydown', async e => {
    if (state.sidebarTab === 'review') {
        const shortcutTarget = resolveReviewShortcutTarget({
            key: e.key,
            isListTarget: e.target === e.currentTarget,
            isEditor: Boolean(
                e.target?.matches?.('.review-primary-input, .review-tolerance'),
            ),
        });
        // Buttons, checkboxes, trace <summary>, and editing/navigation keys
        // not owned by the workbench keep their native browser behavior.
        if (shortcutTarget === 'native') return;
        const selectedCandidateId = state.reviewWorkspace.selectedCandidateId;
        const selectedEntry = listReviewCandidateEntries(state.reviewWorkspace).find(
            entry => entry.item.candidate_id === selectedCandidateId,
        );
        const intent = resolveReviewShortcut({
            key: e.key,
            candidateIds: getReviewNavigationCandidateIds(),
            selectedCandidateId,
            selectedStatus: selectedEntry?.status || null,
            isEditing: shortcutTarget === 'editor',
        });
        if (intent.action === 'none') return;

        e.preventDefault();
        e.stopPropagation();
        const reviewList = e.currentTarget;
        try {
            if (intent.action === 'select') {
                selectReviewCandidateInWorkbench(intent.candidateId);
                reviewList.focus();
            } else if (intent.action === 'edit') {
                focusReviewCandidateEditor(intent.candidateId);
            } else if (intent.action === 'cancel-edit') {
                cancelReviewCandidateEdit(intent.candidateId);
                reviewList.focus();
            } else if (intent.action === 'confirm') {
                await confirmReviewCandidateFromWorkbench(intent.candidateId);
                reviewList.focus();
            } else if (intent.action === 'dismiss') {
                dismissReviewCandidateFromWorkbench(intent.candidateId);
                reviewList.focus();
            }
        } catch (error) {
            alert(`复核快捷键操作失败：${error.message || error}`);
            reviewList.focus();
        }
        return;
    }
    const sorted = [...state.stamps].sort(idSortLogic);
    if (sorted.length === 0) return;

    const curIdx = state.selectedUUIDs.length === 1
        ? sorted.findIndex(s => s.uuid === state.selectedUUIDs[0]) : -1;

    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const next = e.key === 'ArrowDown'
            ? Math.min(curIdx + 1, sorted.length - 1)
            : Math.max(curIdx - 1, 0);
        state.selectedUUIDs = [sorted[next].uuid];
        syncToolbarFromSelection(); updateSidebar(); syncDrawLayerResolution();
    } else if (e.key === 'Enter' && curIdx >= 0) {
        e.preventDefault();
        const s = sorted[curIdx];
        if (s && !s.confirmed && state.activeEditUUID !== s.uuid) {
            state.activeEditUUID = s.uuid;
            updateSidebar();
            const li = document.querySelector(`#stamp-list li[data-uuid="${s.uuid}"]`);
            if (li) {
                li.scrollIntoView({ block: 'nearest' });
                li.querySelector('.field-nominal')?.focus();
            }
        } else if (state.activeEditUUID === s?.uuid) {
            _commitExpandedEdit(s.uuid, sorted);
        }
    } else if (e.key === 'Escape') {
        if (state.activeEditUUID) {
            state.activeEditUUID = null;
            updateSidebar();
        }
    }
});

document.getElementById('btn-confirm-all')?.addEventListener('click', () => {
    if (state.stamps.length === 0) return;
    state.stamps.forEach(s => {
        s.confirmed = true;
        if (s.status === 'error') s.status = 'success';
    });
    state.activeEditUUID = null;
    pushHistory(); updateSidebar();
});

document.getElementById('btn-reindex').addEventListener('click', () => {
    if (state.stamps.length === 0) return;
    if (!confirm('将按图纸空间位置重排未确认的 AI 自动章，保留已确认/手绘编号，是否继续？')) return;
    _reindexBySpace();
    pushHistory(); updateSidebar(); syncDrawLayerResolution(); syncToolbarFromSelection();
});
document.getElementById('btn-sort-order').addEventListener('click', () => {
    const next = sortOrder === 'asc' ? 'desc' : 'asc';
    setSortOrder(next);
    document.getElementById('btn-sort-order').innerText =
        next === 'asc' ? '↓ 正序' : '↑ 倒序';
    updateSidebar();
});

// ═══════════════════════════════════════════════════════════════════════════════
// 重新解析按钮（侧边栏入口）
// ═══════════════════════════════════════════════════════════════════════════════

// 侧边栏「重新解析」按钮 → 强制当前页，复用 btn-analyze 的完整流程
document.getElementById('btn-run-ai').addEventListener('click', () => {
    if (!state.currentPdfBuffer) return alert('请先打开图纸文档！');
    if (state.isAnalysisRunning) return;
    const prevScope = state.analyzeScope;
    state.analyzeScope = 'current';
    document.getElementById('btn-analyze').click();
    // btn-analyze 异步跑，radio UI 不动；恢复 state 让后续工具栏点击继续听 radio
    queueMicrotask(() => { state.analyzeScope = prevScope; });
});

function _nextReviewedStampId() {
    const used = new Set(
        state.stamps
            .map(stamp => Number.parseInt(String(stamp.id || ''), 10))
            .filter(value => Number.isInteger(value) && value > 0),
    );
    let next = 1;
    while (used.has(next)) next += 1;
    return String(next);
}

async function _confirmReviewCandidateIntoProject(candidateId, values) {
    const entry = listReviewCandidateEntries(state.reviewWorkspace).find(
        candidate => candidate.item.candidate_id === candidateId,
    );
    if (!entry) throw new RangeError(`待复核候选不存在：${candidateId}`);
    if (entry.status === 'confirmed') {
        return state.stamps.find(
            stamp => stamp.uuid === entry.decision?.formal_stamp_uuid,
        ) || null;
    }
    if (entry.status === 'dismissed') throw new TypeError('已删除候选不能确认');

    // First validate/edit only the isolated review workspace. No formal project
    // state is touched until a complete reviewed stamp has been built.
    const editedWorkspace = editReviewCandidate(
        state.reviewWorkspace,
        candidateId,
        values,
    );
    const pageIndex = entry.envelope.page_index + 1;
    const formalDimension = buildReviewedFormalDimension(entry.item, values);
    const pageViewBoxes = state.viewBoxes.filter(box => box.pageIndex === pageIndex);
    const promoted = dimensionsToStamps(
        [formalDimension],
        pageIndex,
        pageViewBoxes,
    ).stamps;
    if (promoted.length !== 1) {
        throw new TypeError('候选位置无法转换为正式标记');
    }
    const stamp = promoted[0];
    stamp.uuid = stamp.uuid || generateUUID();
    stamp.id = _nextReviewedStampId();
    stamp.confirmed = true;
    stamp.status = 'success';
    stamp.isAutoGenerated = true;
    stamp.source = 'reviewed_vector';
    stamp.reviewProvenance = buildReviewProvenance(entry, values);
    const confirmedWorkspace = confirmReviewCandidate(
        editedWorkspace,
        candidateId,
        buildReviewConfirmationPayload(entry.item, values, stamp.uuid),
    );

    state.stamps = [...state.stamps, stamp];
    state.reviewWorkspace = confirmedWorkspace;
    state.selectedUUIDs = [stamp.uuid];
    state.activeEditUUID = null;
    pushHistory();
    updateSidebar();
    syncDrawLayerResolution();
    syncToolbarFromSelection();
    return stamp;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 子模块晚绑定：打破循环依赖
//   app_utils       需要：updateSidebar / syncToolbarFromSelection / updateSelectedPanel / updateEpsSlider
//   app_cursor      需要：updateSelectedPanel
//   app_sidebar     需要：_syncColorToggleBtn（app_cursor）/ updateTransform（canvas_renderer）
//   app_interaction 需要：commitStamp（本文件）
// ═══════════════════════════════════════════════════════════════════════════════

registerUtilsCallbacks(updateSidebar, syncToolbarFromSelection, updateSelectedPanel, updateEpsSlider);
registerCursorCallbacks(updateSelectedPanel);
registerSidebarCallbacks(_syncColorToggleBtn, updateTransform);
registerReviewCandidateCallbacks({
    confirmCandidate: _confirmReviewCandidateIntoProject,
});
registerInteractionCallbacks(commitStamp, _requestStampOCR);

// 首次加载把顶部颜色按钮同步到当前 typeColors
_syncColorToggleBtn();
