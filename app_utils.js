// app_utils.js -- 工具函数 + 历史管理 + 缩放/导航

import { state } from './state.js';
import { els, updateTransform, syncDrawLayerResolution } from './canvas_renderer.js';
import {
    createHistorySnapshot as _createHistorySnapshot,
    restoreHistorySnapshot,
} from './history_snapshot.js';
import { prefersReducedMotion } from './motion_preferences.js';

// ═══════════════════════════════════════════════════════════════════════════════
// Late-binding callbacks (avoid circular deps with app_sidebar / app_cursor)
// ═══════════════════════════════════════════════════════════════════════════════

let _updateSidebar = null;
let _syncToolbarFromSelection = null;
let _updateSelectedPanel = null;
let _updateEpsSlider = null;

export function registerCallbacks(updateSidebar, syncToolbarFromSelection, updateSelectedPanel, updateEpsSlider) {
    _updateSidebar = updateSidebar;
    _syncToolbarFromSelection = syncToolbarFromSelection;
    _updateSelectedPanel = updateSelectedPanel;
    _updateEpsSlider = updateEpsSlider;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 常量
// ═══════════════════════════════════════════════════════════════════════════════

const ZOOM_STEPS = [0.25,0.33,0.5,0.67,0.75,0.9,1.0,1.1,1.25,1.5,1.75,
                    2.0,2.5,3.0,4.0,5.0,6.0,8.0,12.0,16.0,24.0,32.0];

// ═══════════════════════════════════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════════════════════════════════

export function escHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/** D-01 修复版：显式保留工业符号 Unicode 区段（TDD 第九章）*/
export function toHalfWidthAndClean(str) {
    let h = str.replace(/[\uFF01-\uFF5E]/g,
        c => String.fromCharCode(c.charCodeAt(0) - 0xFEE0)
    ).replace(/\u3000/g, ' ');
    return h.replace(
        /[^\x20-\x7E\u4e00-\u9fa5\u00B0-\u00BF\u00D7\u00D8\u00F8\u03A6\u03C6\u2200-\u22FF\u2300-\u23FF\u2500-\u257F\u25A0-\u25FF\u27C0-\u27EF\u2A00-\u2AFF±⌀∅Ø⌖⟂◎△▷□°′″⊥∠⏤↗→←↑↓φΦ×]/g,
        ''
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loading 遮罩（TDD 6.4，DOM 由 T-13 提供）+ 伪进度计时器
// ═══════════════════════════════════════════════════════════════════════════════

let _loadingStart = 0;
let _loadingTimer = null;
let _loadingBaseText = '处理中...';
let _transientNoticeTimer = null;

export function showTransientNotice(message, { durationMs = 3200 } = {}) {
    const text = String(message || '').trim();
    if (!text) return;
    let notice = document.getElementById('transient-notice');
    if (!notice) {
        notice = document.createElement('div');
        notice.id = 'transient-notice';
        notice.className = 'transient-notice';
        notice.setAttribute('role', 'status');
        notice.setAttribute('aria-live', 'polite');
        notice.setAttribute('aria-atomic', 'true');
        document.body.appendChild(notice);
    }
    notice.textContent = text;
    notice.classList.add('is-visible');
    if (_transientNoticeTimer) clearTimeout(_transientNoticeTimer);
    _transientNoticeTimer = setTimeout(() => {
        notice.classList.remove('is-visible');
        _transientNoticeTimer = null;
    }, durationMs);
}

export function showLoading(text = '处理中...') {
    const el = document.getElementById('global-loading');
    if (!el) return;
    const t = el.querySelector('.loading-text');
    _loadingBaseText = text;
    if (t) t.textContent = text;
    el.style.display = 'flex';

    // 伪进度：后端无 SSE 之前，至少给用户"没死"的信号
    _loadingStart = Date.now();
    if (_loadingTimer) clearInterval(_loadingTimer);
    _loadingTimer = setInterval(() => {
        const t2 = el.querySelector('.loading-text');
        if (!t2) return;
        const sec = Math.floor((Date.now() - _loadingStart) / 1000);
        t2.textContent = `${_loadingBaseText}  已用 ${sec}s`;
    }, 1000);
}
export function hideLoading() {
    const el = document.getElementById('global-loading');
    if (el) el.style.display = 'none';
    if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 历史管理（含 A-03 clusteringLocked 联动）
// ═══════════════════════════════════════════════════════════════════════════════

export function createHistorySnapshot(source = state) {
    return _createHistorySnapshot(source);
}

export function replaceCurrentHistorySnapshot() {
    if (state.historyIndex < 0 || state.historyIndex >= state.historyStack.length) return;
    state.historyStack[state.historyIndex] = _createHistorySnapshot(state);
}

export function pushHistory() {
    state.historyStack = state.historyStack.slice(0, state.historyIndex + 1);
    state.historyStack.push(_createHistorySnapshot(state));
    if (state.historyStack.length > 50) state.historyStack.shift();
    state.historyIndex = state.historyStack.length - 1;

    // [A-03] Stage 2 任意写操作使 EPS Slider 锁定
    if (state.clusteringBaseIndex >= 0 &&
        state.historyIndex > state.clusteringBaseIndex) {
        state.clusteringLocked = true;
    }

    document.getElementById('btn-undo').disabled = state.historyIndex <= 0;
    document.getElementById('btn-redo').disabled =
        state.historyIndex >= state.historyStack.length - 1;
    autoSaveToLocal();
    _updateEpsSlider?.();
}

export function commitStampTypeChange(stamp, nextType) {
    if (!stamp || stamp.aiData?.type === nextType) return false;
    if (!stamp.aiData) stamp.aiData = {};
    stamp.aiData.type = nextType;
    pushHistory();
    return true;
}

function autoSaveToLocal() {
    if (!state.docInfo.filename || state.docInfo.filename === 'Untitled') return;
    try {
        localStorage.setItem(
            `pdf_stamps_${state.docInfo.filename}`,
            JSON.stringify(state.stamps)
        );
    } catch (e) {
        if (e.name === 'QuotaExceededError' || e.name === 'NS_ERROR_DOM_QUOTA_REACHED') {
            try {
                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k && k.startsWith('pdf_stamps_')) { localStorage.removeItem(k); break; }
                }
                localStorage.setItem(
                    `pdf_stamps_${state.docInfo.filename}`,
                    JSON.stringify(state.stamps)
                );
            } catch { /* 存储耗尽，忽略 */ }
        }
    }
}

export function _restoreSnapshot(snap) {
    restoreHistorySnapshot(state, snap);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 缩放
// ═══════════════════════════════════════════════════════════════════════════════

export function stepZoom(direction) {
    let idx = ZOOM_STEPS.findIndex(z => z > state.zoom - 0.01);
    if (direction < 0) idx = Math.max(0, idx - 1);
    else idx = Math.min(ZOOM_STEPS.length-1, state.zoom >= ZOOM_STEPS[idx] ? idx+1 : idx);
    const factor = ZOOM_STEPS[idx] / state.zoom;
    const wsRect = els.workspace.getBoundingClientRect();
    state.panX = wsRect.width/2  - (wsRect.width/2  - state.panX) * factor;
    state.panY = wsRect.height/2 - (wsRect.height/2 - state.panY) * factor;
    state.zoom *= factor; updateTransform();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 小地图
// ═══════════════════════════════════════════════════════════════════════════════

export function teleportToMinimap(e) {
    const rect = els.miniContainer.getBoundingClientRect();
    const cx   = Math.max(0, Math.min(e.clientX - rect.left, rect.width))  / state.miniScale;
    const cy   = Math.max(0, Math.min(e.clientY - rect.top,  rect.height)) / state.miniScale;
    const wsRect = els.workspace.getBoundingClientRect();
    state.panX = wsRect.width/2  - cx * state.zoom;
    state.panY = wsRect.height/2 - cy * state.zoom;
    updateTransform();
}

// ── panTo：将画布视口平移居中到 stamp 的 arrowTip（动画版）─────────────────────

let _panToAnimId = null;

// 外部中断（app.js mousedown 需要在用户重新操作时立即终止动画）
export function cancelPanTo() {
    if (_panToAnimId) {
        cancelAnimationFrame(_panToAnimId);
        _panToAnimId = null;
    }
}

export function panTo(stamp) {
    if (!stamp) return;

    // 取消上一次未完成的动画
    if (_panToAnimId) {
        cancelAnimationFrame(_panToAnimId);
        _panToAnimId = null;
    }

    const ws = els.workspace;
    // 补偿侧边栏宽度
    const targetPanX = ws.clientWidth / 2 - stamp.arrowTip.absX * state.zoom;
    const targetPanY = ws.clientHeight / 2 - stamp.arrowTip.absY * state.zoom;

    if (prefersReducedMotion()) {
        state.panX = targetPanX;
        state.panY = targetPanY;
        updateTransform();
        return;
    }

    const startPanX = state.panX;
    const startPanY = state.panY;
    const duration = 300; // ms
    const startTime = performance.now();

    function animate(now) {
        if (prefersReducedMotion()) {
            state.panX = targetPanX;
            state.panY = targetPanY;
            _panToAnimId = null;
            updateTransform();
            return;
        }

        let t = (now - startTime) / duration;
        if (t >= 1) t = 1;

        // ease-out: 1 - (1-t)^3
        const ease = 1 - Math.pow(1 - t, 3);

        state.panX = startPanX + (targetPanX - startPanX) * ease;
        state.panY = startPanY + (targetPanY - startPanY) * ease;
        updateTransform();

        if (t < 1) {
            _panToAnimId = requestAnimationFrame(animate);
        } else {
            _panToAnimId = null;
        }
    }

    _panToAnimId = requestAnimationFrame(animate);
}
