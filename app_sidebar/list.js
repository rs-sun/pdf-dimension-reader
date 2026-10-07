// app_sidebar/list.js — 侧边栏主列表渲染 + 排序
// updateSidebar 是核心入口；末尾会调 popup.updateStampPopup 同步浮窗。
// _commitExpandedEdit 走 expand_form.js（本文件初始化时通过 _registerUpdateSidebarForCommit
// 把 updateSidebar 注入回去，打破 expand_form ↔ list 的直接循环）。

import { state } from '../state.js';
import { syncDrawLayerResolution, flashStamp, resolveIsKey } from '../canvas_renderer.js';
import { escHtml, pushHistory, panTo } from '../app_utils.js';
import { syncToolbarFromSelection } from '../app_cursor.js';
import { _buildCollapseHeaderHtml, _buildConfirmRowHtml } from './shared_templates.js';
import {
    _buildExpandHtml,
    _commitExpandedEdit,
    finishStampTypeChangeEditing,
    getIsEditingField, setIsEditingField,
    _registerUpdateSidebarForCommit,
} from './expand_form.js';
import { _showSymbolPalette, _FIELD_TO_CTX } from './symbol_palette.js';
import { updateStampPopup } from './popup.js';
import { getExportAffordance } from './export_affordance.js';
import {
    dismissReviewCandidate,
    editReviewCandidate,
    listReviewCandidateEntries,
    pendingReviewCandidateCount,
    selectReviewCandidate,
} from '../review_candidates_v1.js';
import {
    getPageGeometry,
    pageLocalBboxToDocumentGlobal,
} from '../page_coordinates.js';
import {
    REVIEW_GDT_DATUM_NOTICE,
    buildReviewCandidateValues,
    isActionableReviewEntry,
    reviewCandidateCurrentValues,
    reviewCandidateType,
    reviewGdtSymbolChineseName,
    runSelectedReviewCandidateActions,
    selectedReviewCandidateIds,
    syncReviewSelectionDom,
    toggleReviewCandidateSelection,
} from './review_workbench.js';

// ═══════════════════════════════════════════════════════════════════════════════
// Module-level state — 排序方向
// ═══════════════════════════════════════════════════════════════════════════════

export let sortOrder = 'asc';   // 侧边栏排序方向

let _confirmReviewCandidate = null;
let _checkedReviewCandidateIds = new Set();
let _reviewCandidateDrafts = new Map();
let _reviewBatchBusy = false;
let _reviewSelectionDocumentGeneration = null;

export function setSortOrder(v) { sortOrder = v; }

export function registerReviewCandidateCallbacks({ confirmCandidate }) {
    _confirmReviewCandidate = confirmCandidate;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 侧边栏排序辅助
// ═══════════════════════════════════════════════════════════════════════════════

/** 计算自适应同行判定阈值：所有 stamp 平均 crop 高度的 0.5 倍，fallback 30 */
function _getSortRowThreshold() {
    const stamps = state.stamps;
    if (stamps.length === 0) return 30;
    let sum = 0, count = 0;
    for (const s of stamps) {
        if (s.crop && s.crop.b !== undefined && s.crop.t !== undefined) {
            sum += Math.abs(s.crop.b - s.crop.t);
            count++;
        }
    }
    return count > 0 ? (sum / count) * 0.5 : 30;
}

/** 侧边栏显示排序（纯显示用，不负责分配 ID）
 *  S37c: 视图框感知排序 —— 先按所属视图框 (y,x) 分组，再在组内按 stamp (y,x) 行阈值排；
 *  孤儿（不属于任何视图框的 stamp）排到所有视图之后。
 */
export function naturalSortLogic(a, b) {
    const aNote = a?.type === 'note';
    const bNote = b?.type === 'note';
    if (aNote !== bNote) return aNote ? 1 : -1;

    const vbs = (typeof state !== 'undefined' && Array.isArray(state.viewBoxes)) ? state.viewBoxes : [];
    const _vbOf = (s) => {
        const vid = s.viewBoxId;
        if (!vid) return null;
        return vbs.find(v => v.id === vid) || null;
    };
    const vbA = _vbOf(a);
    const vbB = _vbOf(b);

    // 孤儿永远排在视图之后（不受 asc/desc 影响）
    const priA = vbA ? 0 : 1;
    const priB = vbB ? 0 : 1;
    if (priA !== priB) return priA - priB;

    // 两个 stamp 都在视图内，但属于不同视图：按视图框 (y,x) 比较（行阈值 80pt）
    if (vbA && vbB && vbA.id !== vbB.id) {
        const vy = vbA.y - vbB.y;
        if (Math.abs(vy) > 80)
            return sortOrder === 'asc' ? vy : -vy;
        const vx = vbA.x - vbB.x;
        return sortOrder === 'asc' ? vx : -vx;
    }

    // 同一视图内 或 两个孤儿：按 stamp (y,x) + 自适应行阈值
    const ay = a.sortY ?? a.arrowTip.absY;
    const by = b.sortY ?? b.arrowTip.absY;
    const ax = a.sortX ?? a.arrowTip.absX;
    const bx = b.sortX ?? b.arrowTip.absX;
    const threshold = _getSortRowThreshold();
    if (Math.abs(ay - by) > threshold)
        return sortOrder === 'asc' ? ay - by : by - ay;
    return sortOrder === 'asc' ? ax - bx : bx - ax;
}

/** 侧边栏/键盘导航用：按 stamp.id 数值升序 */
export function idSortLogic(a, b) {
    const na = parseInt(a.id) || 0;
    const nb = parseInt(b.id) || 0;
    if (na !== nb) return na - nb;
    return (a.id || '').localeCompare(b.id || '');
}

const _INSPECTION_FILTERS = [
    ['all', '全部'],
    ['ng', 'NG'],
    ['critical', '临界'],
    ['unmeasured', '未测量'],
    ['key_ng', 'KEY NG'],
];

function _setSidebarMode(activeTab) {
    state.sidebarTab = activeTab;
    document.querySelectorAll('.sidebar-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === activeTab);
    });
    const isDimensions = activeTab === 'dimensions';
    const isReview = activeTab === 'review';
    const title = isReview ? '待人工复核'
        : (isDimensions ? '标记列表' : '检验结果');
    document.getElementById('sb-title-label').textContent = title;
    document.querySelector('.sb-progress-wrap')?.style.setProperty('display', isDimensions ? 'flex' : 'none');
    document.getElementById('btn-run-ai')?.style.setProperty('display', isDimensions ? 'inline-flex' : 'none');
    document.getElementById('btn-confirm-all')?.style.setProperty('display', isDimensions ? 'inline-flex' : 'none');
    document.getElementById('btn-reindex')?.style.setProperty('display', isDimensions ? 'inline-flex' : 'none');
    document.getElementById('btn-sort-order')?.style.setProperty('display', isDimensions ? 'inline-flex' : 'none');
    document.getElementById('review-batch-actions')?.toggleAttribute('hidden', !isReview);
}

function _reviewStatusLabel(status) {
    return {
        pending: '待复核',
        edited: '已编辑',
        confirmed: '已确认',
        dismissed: '已拒绝',
    }[status] || status;
}

function _sortedReviewEntries() {
    return listReviewCandidateEntries(state.reviewWorkspace).sort((left, right) => {
        const priority = { pending: 0, edited: 0, confirmed: 1, dismissed: 2 };
        const statusDiff = (priority[left.status] ?? 3) - (priority[right.status] ?? 3);
        if (statusDiff !== 0) return statusDiff;
        const pageDiff = left.envelope.page_index - right.envelope.page_index;
        if (pageDiff !== 0) return pageDiff;
        return left.item.candidate_id.localeCompare(right.item.candidate_id);
    });
}

function _reviewEntry(candidateId) {
    return _sortedReviewEntries().find(
        entry => entry.item.candidate_id === candidateId,
    ) || null;
}

function _reviewRow(candidateId) {
    return [...(document.getElementById('stamp-list')
        ?.querySelectorAll('.review-candidate-row') || [])].find(
        row => row.dataset.candidateId === candidateId,
    ) || null;
}

function _selectReviewRowWithoutRender(candidateId, listEl) {
    if (!candidateId) return;
    state.reviewWorkspace = selectReviewCandidate(
        state.reviewWorkspace,
        candidateId,
    );
    state.selectedUUIDs = [];
    listEl.querySelectorAll('.review-candidate-row').forEach(row => {
        row.classList.toggle(
            'selected',
            row.dataset.candidateId === candidateId,
        );
    });
}

function _reconcileReviewSelection(entries = _sortedReviewEntries()) {
    const documentGeneration = Number.isSafeInteger(state.documentGeneration)
        ? state.documentGeneration
        : null;
    if (_reviewSelectionDocumentGeneration !== documentGeneration) {
        _checkedReviewCandidateIds = new Set();
        _reviewCandidateDrafts = new Map();
        _reviewSelectionDocumentGeneration = documentGeneration;
    }
    _checkedReviewCandidateIds = new Set(
        selectedReviewCandidateIds(entries, _checkedReviewCandidateIds),
    );
    const actionableIds = new Set(
        entries
            .filter(isActionableReviewEntry)
            .map(entry => entry.item.candidate_id),
    );
    for (const candidateId of _reviewCandidateDrafts.keys()) {
        if (!actionableIds.has(candidateId)) {
            _reviewCandidateDrafts.delete(candidateId);
        }
    }
    return _checkedReviewCandidateIds;
}

function _syncReviewBatchControls(entries = _sortedReviewEntries()) {
    const selectedIds = selectedReviewCandidateIds(
        entries,
        _checkedReviewCandidateIds,
    );
    const count = document.getElementById('review-selection-count');
    if (count) count.textContent = `已选 ${selectedIds.length}`;
    for (const id of ['btn-review-confirm-selected', 'btn-review-dismiss-selected']) {
        const button = document.getElementById(id);
        if (button) button.disabled = _reviewBatchBusy || selectedIds.length === 0;
    }
}

export function getReviewNavigationCandidateIds() {
    return _sortedReviewEntries().map(entry => entry.item.candidate_id);
}

export function getReviewSelection() {
    const entries = _sortedReviewEntries();
    _reconcileReviewSelection(entries);
    return selectedReviewCandidateIds(entries, _checkedReviewCandidateIds);
}

export function selectReviewCandidateInWorkbench(candidateId, { locate = false } = {}) {
    const entry = _reviewEntry(candidateId);
    if (!entry) return null;
    state.reviewWorkspace = selectReviewCandidate(state.reviewWorkspace, candidateId);
    state.selectedUUIDs = [];
    if (locate) panTo(_reviewLocation(entry).anchor);
    updateSidebar();
    syncDrawLayerResolution();
    return entry;
}

export function focusReviewCandidateEditor(candidateId) {
    const input = _reviewRow(candidateId)?.querySelector('.review-primary-input');
    if (!input || input.disabled) return false;
    input.focus();
    input.select();
    return true;
}

export function cancelReviewCandidateEdit(candidateId) {
    const entry = _reviewEntry(candidateId);
    const row = _reviewRow(candidateId);
    if (!entry || !row) return false;
    _reviewCandidateDrafts.delete(candidateId);
    const values = reviewCandidateCurrentValues(entry);
    if (reviewCandidateType(entry) === 'gdt') {
        const input = row.querySelector('.review-gdt-tolerance');
        if (input) input.value = values.tolerance_value;
    } else {
        const nominal = row.querySelector('.review-nominal');
        const tolerance = row.querySelector('.review-tolerance');
        const upper = row.querySelector('.review-upper-tolerance');
        const lower = row.querySelector('.review-lower-tolerance');
        if (nominal) nominal.value = values.nominal;
        if (tolerance) tolerance.value = values.symmetric_tolerance ?? '';
        if (upper) upper.value = values.upper_tolerance ?? '';
        if (lower) lower.value = values.lower_tolerance ?? '';
    }
    row.querySelector(':focus')?.blur();
    return true;
}

export function readReviewCandidateValues(candidateId) {
    const entry = _reviewEntry(candidateId);
    if (!entry) throw new RangeError(`unknown review candidate: ${candidateId}`);
    const current = reviewCandidateCurrentValues(entry);
    const draft = _reviewCandidateDrafts.get(candidateId) || {};
    const row = _reviewRow(candidateId);
    if (reviewCandidateType(entry) === 'gdt') {
        return buildReviewCandidateValues(entry, {
            tolerance_value: (
                row?.querySelector('.review-gdt-tolerance')?.value
                ?? draft.tolerance_value
                ?? current.tolerance_value
            ),
        });
    }
    return buildReviewCandidateValues(entry, {
        nominal: (
            row?.querySelector('.review-nominal')?.value
            ?? draft.nominal
            ?? current.nominal
        ),
        symmetric_tolerance: (
            row?.querySelector('.review-tolerance')?.value
            ?? draft.symmetric_tolerance
            ?? current.symmetric_tolerance
            ?? ''
        ),
        upper_tolerance: (
            row?.querySelector('.review-upper-tolerance')?.value
            ?? draft.upper_tolerance
            ?? current.upper_tolerance
            ?? ''
        ),
        lower_tolerance: (
            row?.querySelector('.review-lower-tolerance')?.value
            ?? draft.lower_tolerance
            ?? current.lower_tolerance
            ?? ''
        ),
    });
}

function _captureReviewCandidateDraft(candidateId) {
    const entry = _reviewEntry(candidateId);
    const row = _reviewRow(candidateId);
    if (!entry || !row || !isActionableReviewEntry(entry)) return;
    if (reviewCandidateType(entry) === 'gdt') {
        _reviewCandidateDrafts.set(candidateId, {
            tolerance_value: row.querySelector('.review-gdt-tolerance')?.value ?? '',
        });
        return;
    }
    _reviewCandidateDrafts.set(candidateId, {
        nominal: row.querySelector('.review-nominal')?.value ?? '',
        symmetric_tolerance: row.querySelector('.review-tolerance')?.value ?? '',
        upper_tolerance: row.querySelector('.review-upper-tolerance')?.value ?? '',
        lower_tolerance: row.querySelector('.review-lower-tolerance')?.value ?? '',
    });
}

function _saveReviewCandidate(candidateId) {
    state.reviewWorkspace = editReviewCandidate(
        state.reviewWorkspace,
        candidateId,
        readReviewCandidateValues(candidateId),
    );
    _reviewCandidateDrafts.delete(candidateId);
    pushHistory();
    updateSidebar();
    syncDrawLayerResolution();
}

export async function confirmReviewCandidateFromWorkbench(candidateId) {
    if (!_confirmReviewCandidate) throw new TypeError('人工复核入口尚未初始化');
    const entry = _reviewEntry(candidateId);
    if (!entry || !isActionableReviewEntry(entry)) {
        throw new TypeError('候选当前不可确认');
    }
    const result = await _confirmReviewCandidate(
        candidateId,
        readReviewCandidateValues(candidateId),
    );
    _checkedReviewCandidateIds.delete(candidateId);
    _reviewCandidateDrafts.delete(candidateId);
    _reconcileReviewSelection();
    updateSidebar();
    return result;
}

export function dismissReviewCandidateFromWorkbench(candidateId) {
    const entry = _reviewEntry(candidateId);
    if (!entry || !isActionableReviewEntry(entry)) {
        throw new TypeError('候选当前不可拒绝');
    }
    state.reviewWorkspace = dismissReviewCandidate(
        state.reviewWorkspace,
        candidateId,
    );
    _checkedReviewCandidateIds.delete(candidateId);
    _reviewCandidateDrafts.delete(candidateId);
    _reconcileReviewSelection();
    pushHistory();
    updateSidebar();
    syncDrawLayerResolution();
    return candidateId;
}

export async function confirmSelectedReviewCandidates() {
    const entries = _sortedReviewEntries();
    _reconcileReviewSelection(entries);
    const candidateIds = selectedReviewCandidateIds(
        entries,
        _checkedReviewCandidateIds,
    );
    const valueSnapshots = new Map(
        candidateIds.map(candidateId => [
            candidateId,
            readReviewCandidateValues(candidateId),
        ]),
    );
    _reviewBatchBusy = true;
    _syncReviewBatchControls(entries);
    let result;
    try {
        result = await runSelectedReviewCandidateActions(
            entries,
            candidateIds,
            async candidateId => {
                if (!_confirmReviewCandidate) {
                    throw new TypeError('人工复核入口尚未初始化');
                }
                await _confirmReviewCandidate(
                    candidateId,
                    valueSnapshots.get(candidateId),
                );
                _checkedReviewCandidateIds.delete(candidateId);
                _reviewCandidateDrafts.delete(candidateId);
            },
        );
    } finally {
        _reviewBatchBusy = false;
        _reconcileReviewSelection();
        updateSidebar();
        syncDrawLayerResolution();
    }
    return result;
}

export function dismissSelectedReviewCandidates() {
    const entries = _sortedReviewEntries();
    _reconcileReviewSelection(entries);
    const candidateIds = selectedReviewCandidateIds(
        entries,
        _checkedReviewCandidateIds,
    );
    const succeeded = [];
    const failed = [];
    let nextWorkspace = state.reviewWorkspace;
    for (const candidateId of candidateIds) {
        try {
            nextWorkspace = dismissReviewCandidate(nextWorkspace, candidateId);
            _checkedReviewCandidateIds.delete(candidateId);
            _reviewCandidateDrafts.delete(candidateId);
            succeeded.push(candidateId);
        } catch (error) {
            failed.push({ candidateId, error });
        }
    }
    if (succeeded.length) {
        state.reviewWorkspace = nextWorkspace;
        pushHistory();
    }
    _reconcileReviewSelection();
    updateSidebar();
    syncDrawLayerResolution();
    return { attempted: candidateIds, succeeded, failed };
}

function _reviewLocation(entry) {
    const page = getPageGeometry(state.pageOffsets, entry.envelope.page_index + 1);
    const bbox = pageLocalBboxToDocumentGlobal(entry.item.bbox, page);
    return {
        bbox,
        anchor: {
            arrowTip: {
                absX: bbox.x + bbox.w / 2,
                absY: bbox.y + bbox.h / 2,
            },
        },
    };
}

function _reviewTraceText(entry) {
    return JSON.stringify({
        trace_id: entry.envelope.trace_id,
        item_trace: entry.item.trace,
        audit: entry.envelope.audit,
    }, null, 2);
}

function _reviewCandidateBodyHtml(entry, resolved) {
    const id = entry.item.candidate_id;
    const values = _reviewCandidateDrafts.get(id)
        || reviewCandidateCurrentValues(entry);
    if (reviewCandidateType(entry) === 'gdt') {
        const chineseName = reviewGdtSymbolChineseName(entry.item.symbol_name);
        return `
            <div class="review-gdt-symbol-row">
                <span class="review-gdt-symbol" aria-hidden="true">${escHtml(entry.item.symbol_unicode)}</span>
                <span class="review-gdt-symbol-name">
                    <strong>${escHtml(chineseName)}</strong>
                    <small>${escHtml(entry.item.symbol_name)}</small>
                </span>
            </div>
            <div class="review-value-row review-gdt-value-row">
                <label>公差值
                    <input class="review-gdt-tolerance review-primary-input"
                        data-candidate-id="${escHtml(id)}"
                        value="${escHtml(values.tolerance_value)}"
                        inputmode="decimal" ${resolved ? 'disabled' : ''}>
                </label>
            </div>
            <div class="review-datum-note" role="note">${REVIEW_GDT_DATUM_NOTICE}</div>
        `;
    }
    const asymmetric = values.upper_tolerance !== null
        || values.lower_tolerance !== null;
    const semanticHtml = `
        <div class="review-semantic-fields" aria-label="只读尺寸语义">
            ${entry.item.prefix === null ? '' : `
                <span class="review-semantic-chip review-prefix">前缀 ${escHtml(entry.item.prefix)}</span>
            `}
            ${entry.item.unit === null ? '' : `
                <span class="review-semantic-chip review-unit">单位 ${entry.item.unit === 'degree' ? '°' : escHtml(entry.item.unit)}</span>
            `}
            ${entry.item.quantity > 1 ? `
                <span class="review-semantic-chip review-quantity">倍数 ${entry.item.quantity}X</span>
            ` : ''}
        </div>
    `;
    const toleranceHtml = asymmetric
        ? `
            <label>上公差
                <input class="review-upper-tolerance" data-candidate-id="${escHtml(id)}"
                    value="${escHtml(values.upper_tolerance ?? '')}"
                    inputmode="decimal" ${resolved ? 'disabled' : ''}>
            </label>
            <label>下公差
                <input class="review-lower-tolerance" data-candidate-id="${escHtml(id)}"
                    value="${escHtml(values.lower_tolerance ?? '')}"
                    inputmode="decimal" ${resolved ? 'disabled' : ''}>
            </label>
        `
        : `
            <label>± 公差
                <input class="review-tolerance" data-candidate-id="${escHtml(id)}"
                    value="${escHtml(values.symmetric_tolerance ?? '')}"
                    inputmode="decimal" placeholder="无" ${resolved ? 'disabled' : ''}>
            </label>
        `;
    return `
        <div class="review-source-text">${escHtml(entry.item.trace?.canonical_text || '')}</div>
        ${semanticHtml}
        <div class="review-value-row">
            <label>主值
                <input class="review-nominal review-primary-input"
                    data-candidate-id="${escHtml(id)}"
                    value="${escHtml(values.nominal)}"
                    inputmode="decimal" ${resolved ? 'disabled' : ''}>
            </label>
            ${toleranceHtml}
        </div>
    `;
}

function _bindReviewBatchActions() {
    const confirmButton = document.getElementById('btn-review-confirm-selected');
    if (confirmButton) {
        confirmButton.onclick = async () => {
            const result = await confirmSelectedReviewCandidates();
            if (result.failed.length) {
                const first = result.failed[0].error;
                alert(`有 ${result.failed.length} 条确认失败：${first?.message || first}`);
            }
        };
    }
    const dismissButton = document.getElementById('btn-review-dismiss-selected');
    if (dismissButton) {
        dismissButton.onclick = () => {
            const result = dismissSelectedReviewCandidates();
            if (result.failed.length) {
                const first = result.failed[0].error;
                alert(`有 ${result.failed.length} 条拒绝失败：${first?.message || first}`);
            }
        };
    }
}

function _renderReviewSidebar(listEl) {
    const entries = _sortedReviewEntries();
    _reconcileReviewSelection(entries);
    const pending = pendingReviewCandidateCount(state.reviewWorkspace);
    document.getElementById('sb-count').innerText = `(${pending}/${entries.length})`;
    const tabCount = document.getElementById('review-tab-count');
    if (tabCount) tabCount.textContent = String(pending);
    _syncReviewBatchControls(entries);
    _bindReviewBatchActions();
    listEl.innerHTML = '';
    document.getElementById('stamp-popup')?.style.setProperty('display', 'none');

    if (!entries.length) {
        const hasEnvelope = Array.isArray(state.reviewWorkspace?.envelopes)
            && state.reviewWorkspace.envelopes.length > 0;
        listEl.innerHTML = `<li class="inspection-empty">${hasEnvelope
            ? '本次解析没有待复核候选。'
            : '尚无待人工复核候选。'}</li>`;
        return;
    }

    for (const entry of entries) {
        const id = entry.item.candidate_id;
        const resolved = entry.status === 'confirmed' || entry.status === 'dismissed';
        const selected = state.reviewWorkspace.selectedCandidateId === id;
        const checked = _checkedReviewCandidateIds.has(id);
        const row = document.createElement('li');
        row.className = `review-candidate-row ${entry.status}`
            + `${selected ? ' selected' : ''}${checked ? ' batch-selected' : ''}`;
        row.dataset.candidateId = id;
        const page = entry.envelope.page_index + 1;
        const formalStamp = entry.decision?.formal_stamp_uuid
            ? state.stamps.find(stamp => stamp.uuid === entry.decision.formal_stamp_uuid)
            : null;
        const resolution = entry.status === 'confirmed'
            ? `已进入正式项目${formalStamp?.id ? ` #${escHtml(formalStamp.id)}` : ''}`
            : (entry.status === 'dismissed' ? '本会话已拒绝；重复解析不会复活' : '');
        row.innerHTML = `
            <div class="review-candidate-head">
                <div class="review-candidate-head-main">
                    <label class="review-select-label" title="${resolved ? '已处理候选不能加入批量操作' : '加入批量操作'}">
                        <input class="review-select-checkbox" type="checkbox"
                            data-candidate-id="${escHtml(id)}"
                            aria-label="选择第 ${page} 页候选 ${escHtml(id)}"
                            ${checked ? 'checked' : ''} ${resolved ? 'disabled' : ''}>
                    </label>
                    <button class="review-locate-btn" type="button" data-candidate-id="${escHtml(id)}"
                        title="定位到图纸候选位置">第 ${page} 页 · 定位</button>
                </div>
                <span class="review-status-badge ${entry.status}">${_reviewStatusLabel(entry.status)}</span>
            </div>
            ${_reviewCandidateBodyHtml(entry, resolved)}
            ${resolution ? `<div class="review-resolution">${resolution}</div>` : ''}
            ${resolved ? '' : `
                <div class="review-actions">
                    <button class="review-save-btn" type="button" data-candidate-id="${escHtml(id)}">保存修改</button>
                    <button class="review-confirm-btn" type="button" data-candidate-id="${escHtml(id)}">确认入项目</button>
                    <button class="review-dismiss-btn" type="button" data-candidate-id="${escHtml(id)}">拒绝候选</button>
                </div>`}
            <details class="review-audit-details">
                <summary>完整 trace / audit</summary>
                <pre>${escHtml(_reviewTraceText(entry))}</pre>
            </details>
        `;
        listEl.appendChild(row);
    }

    listEl.querySelectorAll('.review-select-checkbox').forEach(checkbox => {
        checkbox.addEventListener('focus', event => {
            _selectReviewRowWithoutRender(
                event.currentTarget.dataset.candidateId,
                listEl,
            );
        });
        checkbox.addEventListener('change', event => {
            const candidateId = event.currentTarget.dataset.candidateId;
            _checkedReviewCandidateIds = toggleReviewCandidateSelection(
                _checkedReviewCandidateIds,
                candidateId,
                event.currentTarget.checked,
            );
            _reconcileReviewSelection();
            _selectReviewRowWithoutRender(candidateId, listEl);
            syncReviewSelectionDom(listEl, _checkedReviewCandidateIds);
            _syncReviewBatchControls();
            syncDrawLayerResolution();
        });
    });

    listEl.querySelectorAll('.review-value-row input').forEach(input => {
        input.addEventListener('input', event => {
            _captureReviewCandidateDraft(
                event.currentTarget.dataset.candidateId,
            );
        });
        input.addEventListener('focus', event => {
            const candidateId = event.currentTarget.dataset.candidateId;
            if (!candidateId) return;
            _selectReviewRowWithoutRender(candidateId, listEl);
        });
    });

    listEl.querySelectorAll('.review-locate-btn').forEach(button => {
        button.addEventListener('click', event => {
            const candidateId = event.currentTarget.dataset.candidateId;
            const entry = _reviewEntry(candidateId);
            if (!entry) return;
            _selectReviewRowWithoutRender(candidateId, listEl);
            panTo(_reviewLocation(entry).anchor);
            syncDrawLayerResolution();
        });
    });

    listEl.querySelectorAll('.review-save-btn').forEach(button => {
        button.addEventListener('click', event => {
            const candidateId = event.currentTarget.dataset.candidateId;
            try {
                _saveReviewCandidate(candidateId);
            } catch (error) {
                alert(`候选值格式无效：${error.message || error}`);
            }
        });
    });

    listEl.querySelectorAll('.review-dismiss-btn').forEach(button => {
        button.addEventListener('click', event => {
            const candidateId = event.currentTarget.dataset.candidateId;
            try {
                dismissReviewCandidateFromWorkbench(candidateId);
            } catch (error) {
                alert(`拒绝失败：${error.message || error}`);
            }
        });
    });

    listEl.querySelectorAll('.review-confirm-btn').forEach(button => {
        button.addEventListener('click', async event => {
            const candidateId = event.currentTarget.dataset.candidateId;
            event.currentTarget.disabled = true;
            try {
                await confirmReviewCandidateFromWorkbench(candidateId);
            } catch (error) {
                alert(`确认失败，正式项目未改变：${error.message || error}`);
                event.currentTarget.disabled = false;
            }
        });
    });

    const selectedRow = [...listEl.querySelectorAll('.review-candidate-row')].find(
        row => row.dataset.candidateId === state.reviewWorkspace.selectedCandidateId,
    );
    selectedRow?.scrollIntoView({ block: 'nearest' });
}

function _bindSidebarTabs() {
    document.querySelectorAll('.sidebar-tab-btn').forEach(btn => {
        btn.onclick = () => {
            state.sidebarTab = btn.dataset.tab || 'dimensions';
            updateSidebar();
        };
    });
}

function _resultClass(result) {
    const r = String(result || '').toUpperCase();
    if (r === 'NOK' || r === 'NG') return 'nok';
    if (r === 'OK') return 'ok';
    if (r === 'UNMEASURED' || r === '未测量') return 'unmeasured';
    return 'unknown';
}

function _resultLabel(result) {
    const r = String(result || '').toUpperCase();
    if (r === 'NOK' || r === 'NG') return 'NG';
    if (r === 'OK') return 'OK';
    if (r === 'CRITICAL') return '临界';
    if (r === 'UNMEASURED') return '未测量';
    return r || 'UNKNOWN';
}

function _fmtCellValue(v) {
    if (v === null || v === undefined || v === '') return '-';
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(4)));
    return String(v);
}

function _actualsText(actuals) {
    const list = Array.isArray(actuals) ? actuals : [];
    if (!list.length) return '实测值：-';
    return '实测值：' + list.map((a, idx) => {
        if (a && typeof a === 'object') return `${a.sample || idx + 1}=${_fmtCellValue(a.value)}`;
        return `${idx + 1}=${_fmtCellValue(a)}`;
    }).join('  ');
}

function _buildNoteExpandHtml(s) {
    const uid = escHtml(s.uuid);
    const text = escHtml(s.text || s.aiData?._raw_text || s.aiData?.nominal || '');
    return `
        <div class="stamp-expand" data-dim-type="技术要求">
            <div class="ex-row">
                <textarea class="field-note-text ex-grow" data-uuid="${uid}" rows="4"
                    title="技术要求条目">${text}</textarea>
            </div>
        </div>`;
}

function _inspectionItems(visibleStamps) {
    const inspection = state.inspection || {};
    const byStampUuid = inspection.byStampUuid || {};
    const items = visibleStamps.map(stamp => ({
        kind: 'stamp',
        stamp,
        inspection: byStampUuid[stamp.uuid] || stamp.inspection || {
            stampUuid: stamp.uuid,
            stampId: stamp.id,
            result: 'UNMEASURED',
            matchStatus: inspection.active ? 'not_found' : 'not_imported',
            records: [],
            actuals: [],
            isKey: resolveIsKey(stamp),
        },
    }));
    (inspection.unmatchedRecords || []).forEach(record => {
        items.push({ kind: 'record', record, inspection: {
            result: record.result || 'UNKNOWN',
            matchStatus: 'unmatched',
            records: [record],
            actuals: record.actuals || [],
            rawItemName: record.raw_item_name || record.item_text || '',
            sourceFile: record.source_file || '',
            utl: record.utl ?? null,
            ltl: record.ltl ?? null,
        }});
    });
    return items;
}

function _passesInspectionFilter(item) {
    const filter = state.inspection?.filter || 'all';
    const result = String(item.inspection?.result || '').toUpperCase();
    if (filter === 'ng') return result === 'NOK' || result === 'NG';
    if (filter === 'critical') return result === 'CRITICAL';
    if (filter === 'unmeasured') return result === 'UNMEASURED' || item.inspection?.matchStatus === 'not_found';
    if (filter === 'key_ng') return item.kind === 'stamp' && resolveIsKey(item.stamp) && (result === 'NOK' || result === 'NG');
    return true;
}

function _renderInspectionSidebar(visibleStamps, listEl) {
    const inspection = state.inspection || {};
    const stats = inspection.stats || {};
    document.getElementById('sb-count').innerText = `(${stats.recordCount || 0})`;
    listEl.innerHTML = '';

    if (!inspection.active) {
        listEl.innerHTML = '<li class="inspection-empty">尚未导入测量结果。</li>';
        updateStampPopup();
        return;
    }

    const panel = document.createElement('li');
    panel.className = 'inspection-panel';
    const filterHtml = _INSPECTION_FILTERS.map(([key, label]) => `
        <button class="inspection-filter-btn${(inspection.filter || 'all') === key ? ' active' : ''}" type="button" data-filter="${key}">${label}</button>
    `).join('');
    panel.innerHTML = `
        <div class="inspection-stats">
            <div class="inspection-stat"><strong>${stats.matchedStampCount || 0}</strong><span>已匹配</span></div>
            <div class="inspection-stat"><strong>${stats.nokCount || 0}</strong><span>NG</span></div>
            <div class="inspection-stat"><strong>${stats.unmeasuredCount || 0}</strong><span>未测量</span></div>
            <div class="inspection-stat"><strong>${stats.keyNokCount || 0}</strong><span>KEY NG</span></div>
        </div>
        <div class="inspection-filters">${filterHtml}</div>
    `;
    listEl.appendChild(panel);

    const items = _inspectionItems(visibleStamps).filter(_passesInspectionFilter);
    if (!items.length) {
        const empty = document.createElement('li');
        empty.className = 'inspection-empty';
        empty.textContent = '当前筛选下没有记录。';
        listEl.appendChild(empty);
    }

    items.forEach(item => {
        const row = document.createElement('li');
        const inspectionInfo = item.inspection || {};
        const resultCls = _resultClass(inspectionInfo.result);
        const selected = inspection.selectedRecordId
            && (inspectionInfo.records || []).some(r => r.record_id === inspection.selectedRecordId);
        row.className = `inspection-row ${resultCls}${selected ? ' selected' : ''}`;
        if (item.kind === 'stamp') row.dataset.uuid = item.stamp.uuid;
        const title = item.kind === 'stamp'
            ? `#${escHtml(item.stamp.id)} ${escHtml(inspectionInfo.rawItemName || item.stamp.aiData?.nominal || '未测量')}`
            : `未匹配 ${escHtml(item.record?.no ? '#' + item.record.no : '')} ${escHtml(inspectionInfo.rawItemName || '测量记录')}`;
        const source = inspectionInfo.sourceFile ? ` · ${escHtml(inspectionInfo.sourceFile)}` : '';
        row.innerHTML = `
            <div class="inspection-row-head">
                <div class="inspection-row-title">${title}</div>
                <span class="inspection-badge ${resultCls}">${_resultLabel(inspectionInfo.result)}</span>
            </div>
            <div class="inspection-row-meta">
                上限 ${escHtml(_fmtCellValue(inspectionInfo.utl))} / 下限 ${escHtml(_fmtCellValue(inspectionInfo.ltl))}
                · ${escHtml(inspectionInfo.matchStatus || 'unmatched')}${source}
            </div>
            <div class="inspection-row-values">${escHtml(_actualsText(inspectionInfo.actuals))}</div>
        `;
        row.addEventListener('click', () => {
            const firstRecord = (inspectionInfo.records || [])[0];
            if (firstRecord?.record_id) {
                state.inspection.selectedRecordId = firstRecord.record_id;
            } else if (item.kind === 'record' && item.record?.record_id) {
                state.inspection.selectedRecordId = `record:${item.record.record_id}`;
            } else {
                state.inspection.selectedRecordId = `stamp:${item.stamp?.uuid || ''}`;
            }
            if (item.kind === 'stamp') {
                state.selectedUUIDs = [item.stamp.uuid];
                panTo(item.stamp);
                flashStamp(item.stamp.uuid);
                syncToolbarFromSelection();
                syncDrawLayerResolution();
            }
            updateSidebar();
        });
        listEl.appendChild(row);
    });

    listEl.querySelectorAll('.inspection-filter-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            state.inspection.filter = e.currentTarget.dataset.filter || 'all';
            updateSidebar();
        });
    });
    updateStampPopup();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 侧边栏主渲染
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * 渲染侧边栏列表（三段式：粘性头部 / 全量滚动列表 / 粘性底部）
 * 折叠态: [状态点] [ID] [摘要] [定位⊕] [删除×]
 * 展开态: 手风琴，最多一条同时展开（state.activeEditUUID）
 */
export function updateSidebar() {
    if (getIsEditingField()) return;
    // S44h: 和 canvas 渲染保持一致，隐藏低置信度 AI 标记（已 confirmed 除外）
    const _isLowConfAi = s => (s.isAutoGenerated
                                && !s.confirmed
                                && s.aiData?.confidence === 'low');
    const visibleStamps = state.hideLowConfidence
        ? state.stamps.filter(s => !_isLowConfAi(s))
        : state.stamps;
    const tabCount = document.getElementById('review-tab-count');
    if (tabCount) tabCount.textContent = String(pendingReviewCandidateCount(state.reviewWorkspace));
    _bindSidebarTabs();
    const activeTab = state.sidebarTab || 'dimensions';
    _setSidebarMode(activeTab);
    const listEl = document.getElementById('stamp-list');
    if (activeTab === 'review') {
        _renderReviewSidebar(listEl);
        return;
    }
    if (activeTab === 'inspection') {
        _renderInspectionSidebar(visibleStamps, listEl);
        return;
    }
    const total     = visibleStamps.length;
    const confirmed = visibleStamps.filter(s => s.confirmed).length;

    // ── 头部计数 & 进度条 ────────────────────────────────────────────────────
    document.getElementById('sb-count').innerText         = `(${total})`;
    document.getElementById('sb-confirmed-count').textContent = confirmed;
    document.getElementById('sb-total-count').textContent     = total;
    const pct = total > 0 ? (confirmed / total * 100) : 0;
    document.getElementById('sb-progress-fill').style.width   = pct + '%';

    const sorted = [...visibleStamps].sort((a, b) =>
        sortOrder === 'asc' ? idSortLogic(a, b) : idSortLogic(b, a)
    );
    listEl.innerHTML = '';

    sorted.forEach(s => {
        const li = document.createElement('li');
        const isSelected = state.selectedUUIDs.includes(s.uuid);
        const isExpanded = state.activeEditUUID === s.uuid;
        li.className  = `stamp-item${isSelected ? ' selected' : ''}${s.confirmed ? ' confirmed-row' : ''}`;
        li.dataset.uuid = s.uuid;

        // Bug 4：折叠行头部抽到 _buildCollapseHeaderHtml
        // D2：OK 按钮挪到 .stamp-confirm-row 独立行（仅展开 + 未确认时渲染）
        const confirmRowHtml = (isExpanded && !s.confirmed) ? _buildConfirmRowHtml(s) : '';
        li.innerHTML = `
            ${_buildCollapseHeaderHtml(s)}
            ${isExpanded ? (s.type === 'note' ? _buildNoteExpandHtml(s) : _buildExpandHtml(s)) : ''}
            ${confirmRowHtml}
        `;
        listEl.appendChild(li);
    });

    // ── 事件绑定 ─────────────────────────────────────────────────────────────

    // 单击折叠行：选中（单选/多选）
    listEl.querySelectorAll('.stamp-fold-row').forEach(row => {
        row.addEventListener('click', e => {
            if (e.target.classList.contains('delete-btn') ||
                e.target.classList.contains('expand-confirm-btn')) return;
            const uuid = row.closest('li').dataset.uuid;
            if (e.ctrlKey || e.metaKey) {
                const pos = state.selectedUUIDs.indexOf(uuid);
                if (pos > -1) state.selectedUUIDs.splice(pos, 1);
                else state.selectedUUIDs.push(uuid);
            } else {
                state.selectedUUIDs = [uuid];
                // 控制面板开关控制：点击单选行 → 画布定位 + 闪烁高亮
                const teleportEnabled = localStorage.getItem('sidebarClickTeleport') !== '0';
                if (teleportEnabled) {
                    const s = state.stamps.find(st => st.uuid === uuid);
                    if (s) { panTo(s); flashStamp(uuid); }
                }
            }
            syncToolbarFromSelection(); updateSidebar(); syncDrawLayerResolution();
        });
    });

    // 单击摘要：展开/折叠编辑（单选时触发）
    listEl.querySelectorAll('.stamp-summary').forEach(span => {
        span.addEventListener('click', e => {
            e.stopPropagation();
            const uuid = span.closest('li').dataset.uuid;
            const s    = state.stamps.find(st => st.uuid === uuid);
            if (!s || s.confirmed) return;
            state.activeEditUUID = state.activeEditUUID === uuid ? null : uuid;
            state.selectedUUIDs  = [uuid];
            syncToolbarFromSelection(); updateSidebar(); syncDrawLayerResolution();
            const expandedLi = listEl.querySelector(`li[data-uuid="${uuid}"]`);
            if (expandedLi) expandedLi.scrollIntoView({ block: 'nearest' });
        });
    });

    // 双击折叠行：展开编辑（多选时不展开）
    listEl.querySelectorAll('.stamp-fold-row').forEach(row => {
        row.addEventListener('dblclick', e => {
            if (e.target.classList.contains('delete-btn')) return;
            const uuid = row.closest('li').dataset.uuid;
            const s    = state.stamps.find(st => st.uuid === uuid);
            if (!s || s.confirmed) return;
            if (state.selectedUUIDs.length > 1) return;  // 多选时不展开
            state.activeEditUUID = state.activeEditUUID === uuid ? null : uuid;
            state.selectedUUIDs  = [uuid];
            syncToolbarFromSelection(); updateSidebar(); syncDrawLayerResolution();
            // 展开后滚入视图
            const expandedLi = listEl.querySelector(`li[data-uuid="${uuid}"]`);
            if (expandedLi) expandedLi.scrollIntoView({ block: 'nearest' });
        });
    });

    // Bug 5：状态点 → 兼任 ⊕ 定位按钮（不阻断冒泡，让行 click 也能跑，重复 panTo 无副作用）
    listEl.querySelectorAll('.status-dot').forEach(dot => {
        dot.addEventListener('click', e => {
            const uuid = e.currentTarget.dataset.uuid;
            const s = state.stamps.find(st => st.uuid === uuid);
            if (s) { panTo(s); flashStamp(s.uuid); }
        });
    });

    // 徽章点击切换普通/重点——改为设 isKeyOverride（让颜色真正变），不再只改 stamp.type
    listEl.querySelectorAll('.stamp-id-badge').forEach(badge => {
        badge.addEventListener('click', e => {
            e.stopPropagation();
            const uuid = e.currentTarget.dataset.uuid;
            const stamp = state.stamps.find(st => st.uuid === uuid);
            if (stamp) {
                if (stamp.type === 'note') return;
                const wasKey = resolveIsKey(stamp);
                stamp.isKeyOverride = !wasKey;
                stamp.type = stamp.isKeyOverride ? 'key' : 'normal';  // 同步 legacy type 字段
                delete stamp._colorOverride;  // 清掉 preset 粘滞
                pushHistory();
                updateSidebar();
                syncDrawLayerResolution();
            }
        });
    });

    // 删除按钮
    listEl.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            const uuid = e.currentTarget.dataset.uuid;
            const s    = state.stamps.find(st => st.uuid === uuid);
            if (!s || s.confirmed) return;
            if (s.isAutoGenerated) state.deletedStamps.push(s);
            if (state.activeEditUUID === uuid) state.activeEditUUID = null;
            state.stamps        = state.stamps.filter(st => st.uuid !== uuid);
            state.selectedUUIDs = state.selectedUUIDs.filter(u => u !== uuid);
            pushHistory(); updateSidebar(); syncDrawLayerResolution(); syncToolbarFromSelection();
        });
    });

    // 展开态：aiData 字段 blur → 静默写回 state
    const fieldMap = {
        '.field-type': 'type', '.field-prefix': 'prefix',
        '.field-roughness': 'prefix',   // 表面粗糙度的 Ra/Rz 也存 aiData.prefix
        '.field-nominal': 'nominal',
        '.field-upper-tol': 'upper_tol', '.field-lower-tol': 'lower_tol',
        '.field-symbol': 'symbol',
        '.field-datum1': 'datum_1', '.field-datum2': 'datum_2', '.field-datum3': 'datum_3',
        '.field-modifier': 'modifier',
        '.field-note': 'remarks',
        '.field-note-text': 'text',
    };
    Object.entries(fieldMap).forEach(([cls, key]) => {
        listEl.querySelectorAll(cls).forEach(input => {
            input.addEventListener('focus', () => { setIsEditingField(true); });
            input.addEventListener('blur', e => {
                const uuid = e.target.dataset.uuid;
                const s    = state.stamps.find(st => st.uuid === uuid);
                if (!s) { setIsEditingField(false); return; }
                if (!s.aiData) s.aiData = { type:'',nominal:'',tolerance:'',symbol:'',datum:'',remarks:'' };
                const newVal = e.target.value;
                if (key === 'text') {
                    if (s.text !== newVal) {
                        s.text = newVal;
                        s.aiData.nominal = newVal;
                        s.aiData._raw_text = newVal;
                        pushHistory();
                        syncDrawLayerResolution();
                    }
                    setIsEditingField(false);
                    return;
                }
                if (s.aiData[key] !== newVal) {
                    s.aiData[key] = newVal;
                    pushHistory();
                    syncDrawLayerResolution();
                }
                setIsEditingField(false);
            });
        });
    });

    // 展开态：类型切换 → 重建字段区域（保留已填数据）
    listEl.querySelectorAll('.field-type').forEach(sel => {
        sel.addEventListener('change', e => {
            const uuid = e.target.dataset.uuid;
            const s = state.stamps.find(st => st.uuid === uuid);
            if (!s) { setIsEditingField(false); return; }
            // 先保留当前展开项，再解除 editing gate 并重建类型驱动字段。
            state.activeEditUUID = uuid;
            finishStampTypeChangeEditing(s, e.target.value, updateSidebar);
        });
    });

    // 折叠态：ai-field 内联编辑
    // P2.3：scope 限于 sidebar listEl，避免漏到 popup 头部摘要区造成双绑
    listEl.querySelectorAll('.ai-field').forEach(input => {
        input.addEventListener('focus', () => { setIsEditingField(true); });
        input.addEventListener('blur', (e) => {
            const uuid = e.target.dataset.uuid;
            const field = e.target.dataset.field;
            const newVal = e.target.value.trim();
            const stamp = state.stamps.find(st => st.uuid === uuid);
            if (stamp && stamp.aiData && stamp.aiData[field] !== newVal) {
                stamp.aiData[field] = newVal;
                if (stamp.type === 'note' && field === 'nominal') {
                    stamp.text = newVal;
                    stamp.aiData._raw_text = newVal;
                }
                pushHistory();
                syncDrawLayerResolution();
            }
            setIsEditingField(false);
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.target.blur();
                const currentLi = e.target.closest('li');
                const nextLi = currentLi ? currentLi.nextElementSibling : null;
                if (nextLi) {
                    const nextInput = nextLi.querySelector(`.${e.target.classList[1]}`);
                    if (nextInput) {
                        setTimeout(() => nextInput.focus(), 50);
                    }
                }
            }
        });
    });

    // 展开态：符号面板触发按钮（符号字段旁的 ⌨ 按钮）
    listEl.querySelectorAll('.sym-palette-btn').forEach(btn => {
        btn.addEventListener('mousedown', e => {
            e.preventDefault(); // 阻止 input 失去焦点
        });
        btn.addEventListener('click', e => {
            e.stopPropagation();
            const symInput = btn.previousElementSibling;
            _showSymbolPalette(btn, symInput, 'symbol');
        });
    });

    // 展开态：所有字段获得焦点时自动弹出上下文符号面板
    for (const [fieldCls, ctxKey] of Object.entries(_FIELD_TO_CTX)) {
        listEl.querySelectorAll(`.${fieldCls}`).forEach(input => {
            input.addEventListener('focus', () => {
                _showSymbolPalette(input, input, ctxKey);
            });
        });
    }

    // 展开态：✓ 确认按钮
    listEl.querySelectorAll('.expand-confirm-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            const uuid = e.currentTarget.dataset.uuid;
            _commitExpandedEdit(uuid, sorted);
        });
    });

    // 画布 → 侧边栏：自动滚动选中项入视野
    const firstSel = listEl.querySelector('.stamp-item.selected');
    if (firstSel) firstSel.scrollIntoView({ block: 'nearest' });

    // 导出展示态：按钮行为由 app_io.js 决定，这里只让 affordance 与真实策略一致。
    const submitBtn   = document.getElementById('btn-submit');
    if (submitBtn) {
        const affordance = getExportAffordance(state.stamps);
        submitBtn.classList.toggle('export-empty', affordance.state === 'empty');
        submitBtn.classList.toggle('export-warning', affordance.state === 'warning');
        submitBtn.title = affordance.title;
    }
    // 同步选中章悬浮编辑窗（编辑中时只重定位不重建，保留焦点）
    updateStampPopup();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 注入 updateSidebar 回调到 expand_form.js（打破循环依赖：commit → list 刷新）
// ═══════════════════════════════════════════════════════════════════════════════

_registerUpdateSidebarForCommit(updateSidebar);
