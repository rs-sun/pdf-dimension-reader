// Session-only UI adapter for vector_layer_debug_v1.
// It never imports or mutates stamps/review/history/export code.

import { state, createEmptyVectorDebugState } from './state.js';
import {
    VECTOR_DEBUG_MAX_WORKSPACE_OVERLAYS,
    VECTOR_DEBUG_MAX_WORKSPACE_PAGES,
    VECTOR_LAYER_DEBUG_ROOT_KEY,
    VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS,
    cloneBoundedVectorLayerDebugV1,
    createEmptyVectorDebugWorkspace,
    selectVectorDebugWorkspacePage,
    validateVectorLayerDebugV1,
} from './vector_layer_debug_v1.js';
import {
    getPageGeometry,
    pageLocalBboxToDocumentGlobal,
} from './page_coordinates.js';

const ROLE_COLORS = Object.freeze({
    accepted: '#28c982',
    rejected: '#ff6b6b',
    first_drop: '#ff9f43',
    context: '#4e9af1',
});
const CAPTURE_META = new WeakMap();
const MAX_REPORTED_ERRORS = 20;

function safeBoolean(value, key) {
    try {
        return value?.[key] === true;
    } catch (_error) {
        return false;
    }
}

function cloneDebugState(value) {
    const workspace = value?.workspace || createEmptyVectorDebugWorkspace();
    if (!workspace || !Array.isArray(workspace.pages)) {
        throw new TypeError('vector debug workspace invalid');
    }
    return {
        enabled: safeBoolean(value, 'enabled'),
        drawerOpen: safeBoolean(value, 'drawerOpen'),
        // Pages are immutable snapshots created by capture. Structural sharing
        // keeps a 50-page session O(P), rather than cloning all overlays per page.
        workspace,
        error: typeof value?.error === 'string' ? value.error : null,
    };
}

function captureErrorEntry(code, error = null) {
    return Object.freeze({
        kind: 'error',
        message: `${code}${error ? `:${error?.name || 'Error'}` : ''}`,
    });
}

function capturePageEntry(payload, notice = null) {
    return Object.freeze({ kind: 'page', payload, notice });
}

function capturedOverlayCount(batch) {
    let total = 0;
    for (const entry of batch.values()) {
        if (entry?.kind === 'page' && Array.isArray(entry.payload?.overlays)) {
            total += entry.payload.overlays.length;
        }
    }
    return total;
}

export function captureVectorDebugResponse(stagedByPage, pageNumber, response) {
    let next;
    try {
        next = stagedByPage instanceof Map ? new Map(stagedByPage) : new Map();
    } catch (_error) {
        next = new Map();
    }
    const priorMeta = stagedByPage instanceof Map
        ? CAPTURE_META.get(stagedByPage)
        : null;
    const meta = {
        skippedPageCount: priorMeta?.skippedPageCount || 0,
        captureFailureCount: priorMeta?.captureFailureCount || 0,
    };
    try {
        const page = Number(pageNumber);
        if (!Number.isInteger(page) || page < 1) {
            meta.captureFailureCount += 1;
            CAPTURE_META.set(next, meta);
            return next;
        }
        if (!next.has(page) && next.size >= VECTOR_DEBUG_MAX_WORKSPACE_PAGES) {
            meta.skippedPageCount += 1;
            CAPTURE_META.set(next, meta);
            return next;
        }
        const payload = response && typeof response === 'object'
            ? response[VECTOR_LAYER_DEBUG_ROOT_KEY]
            : undefined;
        if (payload === undefined || payload === null) {
            next.set(page, captureErrorEntry('debug_payload_missing'));
            CAPTURE_META.set(next, meta);
            return next;
        }
        const reasons = validateVectorLayerDebugV1(payload, {
            expectedPageIndex: page - 1,
        });
        if (reasons.length > 0) {
            next.set(page, captureErrorEntry(`debug_payload_invalid:${reasons.join(',')}`));
            CAPTURE_META.set(next, meta);
            return next;
        }

        const previous = next.get(page);
        const previousCount = previous?.kind === 'page'
            ? previous.payload.overlays.length
            : 0;
        const used = capturedOverlayCount(next) - previousCount;
        const remaining = Math.max(0, VECTOR_DEBUG_MAX_WORKSPACE_OVERLAYS - used);
        const overlayLimit = Math.min(
            payload.overlays.length,
            VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS,
            remaining,
        );
        const bounded = cloneBoundedVectorLayerDebugV1(payload, overlayLimit);
        const notice = overlayLimit < payload.overlays.length
            ? 'debug_overlay_session_budget_truncated'
            : null;
        next.set(page, capturePageEntry(bounded, notice));
    } catch (error) {
        meta.captureFailureCount += 1;
        try {
            const page = Number(pageNumber);
            if (Number.isInteger(page) && page >= 1
                    && (next.has(page) || next.size < VECTOR_DEBUG_MAX_WORKSPACE_PAGES)) {
                next.set(page, captureErrorEntry('debug_capture_failed', error));
            }
        } catch (_ignored) { /* diagnostic-only */ }
    }
    CAPTURE_META.set(next, meta);
    return next;
}

function pushBoundedError(errors, message) {
    if (errors.length < MAX_REPORTED_ERRORS) errors.push(message);
}

function stagedPage(stagedByPage, pageNumber, errors) {
    let entry;
    try {
        entry = stagedByPage?.get?.(pageNumber);
    } catch (error) {
        pushBoundedError(errors, `第 ${pageNumber} 页诊断读取失败: ${error?.name || 'Error'}`);
        return null;
    }
    if (entry?.kind === 'error') {
        pushBoundedError(errors, `第 ${pageNumber} 页诊断旁路失败: ${entry.message}`);
        return null;
    }
    if (entry?.kind === 'page') {
        if (entry.notice) pushBoundedError(errors, `第 ${pageNumber} 页: ${entry.notice}`);
        return entry.payload;
    }
    if (entry === null || entry === undefined) {
        pushBoundedError(errors, `第 ${pageNumber} 页未返回分层诊断`);
        return null;
    }

    // Compatibility for tests/callers which supply a raw payload directly.
    const reasons = validateVectorLayerDebugV1(entry, {
        expectedPageIndex: Number(pageNumber) - 1,
    });
    if (reasons.length > 0) {
        pushBoundedError(
            errors,
            `第 ${pageNumber} 页诊断合同无效: ${reasons.join(',')}`,
        );
        return null;
    }
    return cloneBoundedVectorLayerDebugV1(
        entry,
        Math.min(entry.overlays.length, VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS),
    );
}

function mergeBoundedWorkspace(workspace, incomingByIndex, errors) {
    const byPage = new Map();
    for (const page of workspace.pages) {
        if (Number.isInteger(page?.page_index) && !byPage.has(page.page_index)) {
            byPage.set(page.page_index, page);
        }
    }
    for (const [pageIndex, page] of incomingByIndex) byPage.set(pageIndex, page);
    const incomingPages = [...incomingByIndex.values()].sort(
        (left, right) => left.page_index - right.page_index,
    );
    const existingPages = [...byPage.values()]
        .filter(page => !incomingByIndex.has(page.page_index))
        .sort((left, right) => left.page_index - right.page_index);
    const prioritized = [...incomingPages, ...existingPages];
    if (prioritized.length > VECTOR_DEBUG_MAX_WORKSPACE_PAGES) {
        pushBoundedError(
            errors,
            `分层诊断页数预算已满，省略 ${prioritized.length - VECTOR_DEBUG_MAX_WORKSPACE_PAGES} 页`,
        );
    }

    // New results remain inspectable: older session pages/overlays yield to the
    // current request, then the retained pages are sorted for the selector.
    const retainedByPriority = prioritized.slice(0, VECTOR_DEBUG_MAX_WORKSPACE_PAGES);
    let overlayRemaining = VECTOR_DEBUG_MAX_WORKSPACE_OVERLAYS;
    let overlayTrimmed = false;
    const allowedByPage = new Map();
    for (const page of retainedByPriority) {
        if (!Array.isArray(page.overlays)) {
            throw new TypeError('vector debug page overlays invalid');
        }
        const allowed = Math.min(page.overlays.length, overlayRemaining);
        overlayRemaining -= allowed;
        allowedByPage.set(page.page_index, allowed);
    }
    const pages = retainedByPriority
        .sort((left, right) => left.page_index - right.page_index)
        .map(page => {
            const allowed = allowedByPage.get(page.page_index) ?? 0;
            if (allowed === page.overlays.length) return page;
            overlayTrimmed = true;
            return cloneBoundedVectorLayerDebugV1(page, allowed);
        });
    if (overlayTrimmed) {
        pushBoundedError(errors, '分层诊断 overlay 会话预算已截断');
    }

    let selectedPageIndex = workspace.selectedPageIndex;
    let selectedStageId = workspace.selectedStageId;
    let selectedPage = pages.find(page => page.page_index === selectedPageIndex);
    if (!selectedPage) {
        selectedPage = pages[0] || null;
        selectedPageIndex = selectedPage?.page_index ?? null;
        selectedStageId = selectedPage?.stages?.[0]?.stage_id ?? null;
    } else if (!selectedPage.stages?.some(stage => stage.stage_id === selectedStageId)) {
        selectedStageId = selectedPage.stages?.[0]?.stage_id ?? null;
    }
    return { pages, selectedPageIndex, selectedStageId };
}

// Commit only pages the ordinary/review transaction itself accepted. Missing or
// invalid diagnostics are recorded in this sidecar state and never change its
// product outcome.
export function commitVectorDebugOutcome(debugState, { stagedByPage, outcome }) {
    const next = cloneDebugState(debugState);
    if (!next.enabled) return next;
    const succeededPages = Array.isArray(outcome?.succeededPages)
        ? outcome.succeededPages
        : [];
    const errors = [];
    const meta = stagedByPage instanceof Map ? CAPTURE_META.get(stagedByPage) : null;
    if (meta?.skippedPageCount) {
        pushBoundedError(
            errors,
            `分层诊断请求页数预算已满，省略 ${meta.skippedPageCount} 页`,
        );
    }
    if (meta?.captureFailureCount) {
        pushBoundedError(
            errors,
            `分层诊断捕获失败 ${meta.captureFailureCount} 次`,
        );
    }
    const incomingByIndex = new Map();
    for (const pageNumber of succeededPages) {
        if (meta?.skippedPageCount
                && stagedByPage instanceof Map
                && !stagedByPage.has(pageNumber)) {
            // Analyze All deliberately omitted this diagnostic request after the
            // shared page cap. The single aggregate message above is sufficient.
            continue;
        }
        const payload = stagedPage(stagedByPage, pageNumber, errors);
        if (payload) incomingByIndex.set(payload.page_index, payload);
    }
    next.workspace = mergeBoundedWorkspace(next.workspace, incomingByIndex, errors);
    next.error = errors.length > 0 ? errors.join('\n') : null;
    return next;
}

export function isolateVectorDebugOutcome(debugState, args) {
    try {
        return commitVectorDebugOutcome(debugState, args);
    } catch (error) {
        const fallback = createEmptyVectorDebugState();
        fallback.enabled = safeBoolean(debugState, 'enabled');
        fallback.drawerOpen = safeBoolean(debugState, 'drawerOpen');
        fallback.error = `分层诊断旁路失败: ${error?.name || 'Error'}`;
        return fallback;
    }
}

export function projectVectorDebugBboxToViewport(
    bbox,
    pageGeometry,
    { zoom, panX, panY },
) {
    const global = pageLocalBboxToDocumentGlobal(bbox, pageGeometry);
    const scale = Number(zoom);
    const xOffset = Number(panX);
    const yOffset = Number(panY);
    if (![scale, xOffset, yOffset].every(Number.isFinite) || scale <= 0) {
        throw new TypeError('invalid viewport transform');
    }
    return {
        x: global.x * scale + xOffset,
        y: global.y * scale + yOffset,
        w: global.w * scale,
        h: global.h * scale,
    };
}

function currentDebugPage() {
    const workspace = state.vectorDebugState.workspace;
    return workspace.pages.find(
        page => page.page_index === workspace.selectedPageIndex,
    ) || workspace.pages[0] || null;
}

function drawVectorDebugOverlayUnsafe() {
    const canvas = document.getElementById('vector-debug-layer');
    const workspaceElement = document.getElementById('workspace');
    if (!canvas || !workspaceElement) return;
    const debug = state.vectorDebugState;
    const visible = debug.enabled && debug.drawerOpen;
    canvas.hidden = !visible;
    if (!visible) return;

    const rect = workspaceElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext('2d');
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const page = currentDebugPage();
    if (!page || page.status !== 'ok') return;
    const selectedStage = state.vectorDebugState.workspace.selectedStageId;
    let geometry;
    try {
        geometry = getPageGeometry(state.pageOffsets, page.page_index + 1);
    } catch (_error) {
        return;
    }
    for (const overlay of page.overlays) {
        if (selectedStage && overlay.stage_id !== selectedStage) continue;
        let box;
        try {
            box = projectVectorDebugBboxToViewport(overlay.bbox, geometry, state);
        } catch (_error) {
            continue;
        }
        const color = ROLE_COLORS[overlay.role] || ROLE_COLORS.context;
        ctx.strokeStyle = color;
        ctx.lineWidth = overlay.role === 'first_drop' ? 2 : 1.25;
        ctx.setLineDash(overlay.role === 'context' ? [4, 3] : []);
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        if (overlay.label && box.w >= 12 && box.h >= 5) {
            ctx.setLineDash([]);
            ctx.fillStyle = color;
            ctx.font = '11px system-ui, sans-serif';
            ctx.fillText(String(overlay.label).slice(0, 48), box.x, box.y - 3);
        }
    }
    ctx.setLineDash([]);
}

function drawVectorDebugOverlay() {
    try {
        drawVectorDebugOverlayUnsafe();
    } catch (_error) {
        try {
            const canvas = typeof document === 'undefined'
                ? null
                : document.getElementById('vector-debug-layer');
            if (canvas) canvas.hidden = true;
        } catch (_ignored) { /* diagnostic-only */ }
    }
}

function runVectorDebugUiAction(action) {
    try {
        action();
    } catch (error) {
        try {
            state.vectorDebugState.error = (
                `分层诊断操作失败: ${error?.name || 'Error'}`
            );
        } catch (_ignored) { /* diagnostic-only */ }
    }
    renderVectorDebugUI();
}

function renderPageSelector(page) {
    const select = document.getElementById('vector-debug-page-select');
    if (!select) return;
    select.replaceChildren();
    const pages = Array.isArray(state.vectorDebugState?.workspace?.pages)
        ? state.vectorDebugState.workspace.pages
        : [];
    for (const candidate of pages) {
        const option = document.createElement('option');
        option.value = String(candidate.page_index);
        option.textContent = `第 ${candidate.page_index + 1} 页`;
        select.append(option);
    }
    select.disabled = pages.length === 0;
    if (page) select.value = String(page.page_index);
}

function renderStageList(page) {
    const list = document.getElementById('vector-debug-stage-list');
    const firstDrop = document.getElementById('vector-debug-first-drop');
    if (!list || !firstDrop) return;
    list.replaceChildren();
    firstDrop.textContent = '';
    firstDrop.hidden = true;
    if (!page) return;
    const selected = state.vectorDebugState.workspace.selectedStageId;
    for (const stage of page.stages) {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'vector-debug-stage-row';
        row.classList.toggle('is-selected', stage.stage_id === selected);
        const name = document.createElement('span');
        name.textContent = stage.stage_id;
        const status = document.createElement('small');
        status.className = 'stage-status';
        status.textContent = stage.status;
        name.append(document.createElement('br'), status);
        const flow = document.createElement('span');
        flow.textContent = `${stage.input_count} → ${stage.output_count}`;
        const dropped = document.createElement('span');
        dropped.textContent = String(stage.drop_count);
        const elapsed = document.createElement('span');
        elapsed.textContent = `${stage.elapsed_ms.toFixed(2)} ms`;
        row.append(name, flow, dropped, elapsed);
        row.addEventListener('click', () => {
            runVectorDebugUiAction(() => {
                state.vectorDebugState.workspace.selectedPageIndex = page.page_index;
                state.vectorDebugState.workspace.selectedStageId = stage.stage_id;
            });
        });
        list.append(row);
    }
    const stage = page.stages.find(row => row.stage_id === selected);
    if (stage?.first_drop) {
        firstDrop.hidden = false;
        firstDrop.textContent = [
            `首个丢弃：${stage.first_drop.candidate_id}`,
            `原因：${stage.first_drop.reason_code}`,
        ].join('\n');
    }
}

function renderVectorDebugUIUnsafe() {
    const debug = state.vectorDebugState;
    const toggle = document.getElementById('toggle-vector-layer-debug');
    const open = document.getElementById('btn-open-vector-debug');
    const drawer = document.getElementById('vector-debug-drawer');
    const status = document.getElementById('vector-debug-status');
    const summary = document.getElementById('vector-debug-summary');
    if (toggle) toggle.checked = debug.enabled;
    if (open) open.disabled = !debug.enabled;
    if (drawer) drawer.hidden = !(debug.enabled && debug.drawerOpen);
    const page = currentDebugPage();
    if (status) {
        status.textContent = debug.error
            || (page
                ? `第 ${page.page_index + 1} 页 · ${page.status}`
                : '尚未运行诊断');
    }
    if (summary) {
        summary.textContent = page
            ? `trace ${page.trace_id} · overlay ${page.limits.returned}/${page.limits.total}`
                + (page.limits.truncated ? '（已截断）' : '')
            : '勾选诊断后重新解析页面，结果才会出现在这里。';
    }
    renderPageSelector(page);
    renderStageList(page);
    drawVectorDebugOverlay();
}

export function renderVectorDebugUI() {
    try {
        renderVectorDebugUIUnsafe();
    } catch (error) {
        try {
            state.vectorDebugState.error = (
                `分层诊断界面失败: ${error?.name || 'Error'}`
            );
        } catch (_ignored) { /* diagnostic-only */ }
        try {
            if (typeof document !== 'undefined') {
                const drawer = document.getElementById('vector-debug-drawer');
                const canvas = document.getElementById('vector-debug-layer');
                if (drawer) drawer.hidden = true;
                if (canvas) canvas.hidden = true;
            }
        } catch (_ignored) { /* diagnostic-only */ }
    }
}

export function applyVectorDebugOutcome(stagedByPage, outcome) {
    try {
        state.vectorDebugState = isolateVectorDebugOutcome(state.vectorDebugState, {
            stagedByPage,
            outcome,
        });
    } catch (_error) {
        try {
            state.vectorDebugState = createEmptyVectorDebugState();
            state.vectorDebugState.error = '分层诊断旁路失败: Error';
        } catch (_ignored) { /* diagnostic-only */ }
    }
    try { renderVectorDebugUI(); } catch (_ignored) { /* diagnostic-only */ }
}

function initVectorDebugUIUnsafe() {
    const toggle = document.getElementById('toggle-vector-layer-debug');
    const open = document.getElementById('btn-open-vector-debug');
    const close = document.getElementById('btn-close-vector-debug');
    const pageSelect = document.getElementById('vector-debug-page-select');
    toggle?.addEventListener('change', () => {
        runVectorDebugUiAction(() => {
            if (!toggle.checked) {
                state.vectorDebugState = createEmptyVectorDebugState();
            } else {
                state.vectorDebugState.enabled = true;
            }
        });
    });
    open?.addEventListener('click', () => {
        runVectorDebugUiAction(() => {
            if (!state.vectorDebugState.enabled) return;
            const controlPanel = document.getElementById('control-panel');
            const controlBackdrop = document.getElementById('control-panel-backdrop');
            if (controlPanel) controlPanel.style.display = 'none';
            if (controlBackdrop) controlBackdrop.style.display = 'none';
            state.vectorDebugState.drawerOpen = true;
        });
    });
    close?.addEventListener('click', () => {
        runVectorDebugUiAction(() => {
            state.vectorDebugState.drawerOpen = false;
        });
    });
    pageSelect?.addEventListener('change', () => {
        runVectorDebugUiAction(() => {
            const pageIndex = Number(pageSelect.value);
            state.vectorDebugState.workspace = selectVectorDebugWorkspacePage(
                state.vectorDebugState.workspace,
                pageIndex,
            );
        });
    });
    document.addEventListener('pdf-reader:transform-update', drawVectorDebugOverlay);
    window.addEventListener('resize', drawVectorDebugOverlay);
    renderVectorDebugUI();
}

export function initVectorDebugUI() {
    try {
        initVectorDebugUIUnsafe();
    } catch (error) {
        try {
            state.vectorDebugState.error = (
                `分层诊断初始化失败: ${error?.name || 'Error'}`
            );
        } catch (_ignored) { /* diagnostic-only */ }
        try { renderVectorDebugUI(); } catch (_ignored) { /* diagnostic-only */ }
    }
}
