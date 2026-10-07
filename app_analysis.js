// app_analysis.js — 解析入口 + DBSCAN 路由 + dimensionsToStamps

import { CONFIG } from './config.js';
import {
    state,
    generateUUID,
    DOCUMENT_LOADING_MESSAGE,
} from './state.js';
import { syncDrawLayerResolution } from './canvas_renderer.js';
import { dbscan } from './dbscan.js';
import { generateRoutedStamps } from './stamp.js';
import { createVectorDebugRequestBudget, runAnalysis } from './api_client.js';
import {
    showLoading,
    hideLoading,
    pushHistory,
    replaceCurrentHistorySnapshot,
} from './app_utils.js';
import { updateCursor, syncToolbarFromSelection } from './app_cursor.js';
import { updateSidebar, naturalSortLogic } from './app_sidebar.js';
import {
    getPageGeometry,
    pageLocalBboxToDocumentGlobal,
    pageLocalPointToDocumentGlobal,
} from './page_coordinates.js';
import { executeReanalysisTransaction } from './reanalysis_transaction.js';
import { executeReviewCandidatesTransaction } from './review_candidates_v1.js';
import {
    applyVectorDebugOutcome,
    captureVectorDebugResponse,
} from './vector_debug_drawer.js';
import {
    buildStampDimensionProjection,
} from './dimension_stamp_projection.js';

// ═══════════════════════════════════════════════════════════════════════════════
// 浮动工具条 初始位置 + 拖拽
// ═══════════════════════════════════════════════════════════════════════════════

export function initBarPosition() {
    const optBar = document.getElementById('stamp-options');
    optBar.style.top  = '65px';
    // left 居中由 CSS left:50% + transform:translateX(-50%) 处理
}

// 浮动工具条拖拽
let _barDragging = false, _barDragOX = 0, _barDragOY = 0;
const _optBar = document.getElementById('stamp-options');
_optBar.addEventListener('mousedown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    // 关键：offsetLeft 和 CSS left 都在 offsetParent 坐标系；CSS translateX(-50%) 不反映在 offsetLeft 里。
    // 想保留视觉位置 → 把 transform 的平移量折进新 left/top 再清 transform。
    // 之前用 rect.left（viewport 坐标）写 CSS left 会跳到 offsetParent.left + offsetLeft 处（本例 body，但任意 positioned 祖先都会错位）。
    const m = new DOMMatrix(getComputedStyle(_optBar).transform);
    const newLeft = _optBar.offsetLeft + m.e;   // m.e = transform.translateX
    const newTop  = _optBar.offsetTop  + m.f;   // m.f = transform.translateY
    _optBar.style.transform = 'none';
    _optBar.style.left = newLeft + 'px';
    _optBar.style.top  = newTop  + 'px';
    _barDragging = true;
    _barDragOX   = e.clientX - newLeft;
    _barDragOY   = e.clientY - newTop;
    _optBar.classList.add('dragging');
});
window.addEventListener('mousemove', e => {
    if (!_barDragging) return;
    _optBar.style.left = (e.clientX - _barDragOX) + 'px';
    _optBar.style.top  = (e.clientY - _barDragOY) + 'px';
});
window.addEventListener('mouseup', () => {
    _barDragging = false;
    _optBar.classList.remove('dragging');
});

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1 辅助：EPS Slider 同步
// ═══════════════════════════════════════════════════════════════════════════════

/** 读取当前 eps 值（Slider 或默认计算值）*/
export function getCurrentEps() {
    const slider = document.getElementById('eps-slider');
    if (slider && !isNaN(parseFloat(slider.value))) return parseFloat(slider.value);
    return state.pdfWidth * CONFIG.DBSCAN_EPS_FACTOR;
}

/** 更新 Slider 属性及 disabled 状态（A-03）*/
export function updateEpsSlider() {
    const slider = document.getElementById('eps-slider');
    if (!slider) return;
    if (state.pdfWidth > 0) {
        slider.min   = String(state.pdfWidth * CONFIG.DBSCAN_EPS_MIN_FACTOR);
        slider.max   = String(state.pdfWidth * CONFIG.DBSCAN_EPS_MAX_FACTOR);
        if (state.currentEps !== null) slider.value = String(state.currentEps);
        else if (!slider.value || parseFloat(slider.value) === 0)
            slider.value = String(state.pdfWidth * CONFIG.DBSCAN_EPS_FACTOR);
    }
    slider.disabled = state.clusteringLocked;
    // [A-03] tooltip
    slider.title = state.clusteringLocked
        ? '已有手动修改，如需重新切分请重新执行全图扫描'
        : `视图切分间距 (当前: ${parseFloat(slider.value || 0).toFixed(1)})`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1 核心：DBSCAN + L1-2 路由排版（可被 EPS Slider 复用）
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1：dimensionsToStamps + 辅助函数
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * 将后端 /analyze 返回的 dimensions 数组转换为 Stamp 对象数组。
 *
 * 坐标转换：后端返回 PDF 原始坐标（pdfplumber，左上角原点，PDF 点单位），
 * 与 state.pageOffsets 坐标系一致（css scale=1.0 的 PDF 点）。
 * 加上 pageOffset.start 变为全局连续坐标。
 *
 * @param {Array}  dimensions  后端返回的 dimension 对象数组
 * @param {number} pageIndex   1-indexed 页码
 * @returns {Stamp[]}
 */

function _findOwningViewBox(physBbox, pageViewBoxes) {
    // 返回包含 bbox 中心的最小视图框；无任何视图框包含则返回 null
    const cx = physBbox.x + physBbox.w / 2;
    const cy = physBbox.y + physBbox.h / 2;
    const containers = pageViewBoxes.filter(vb =>
        cx >= vb.x && cx <= vb.x + vb.w && cy >= vb.y && cy <= vb.y + vb.h
    );
    if (containers.length === 0) return null;
    containers.sort((a, b) => a.w * a.h - b.w * b.h);
    return containers[0];
}

function _hasValue(value) {
    return value !== null && value !== undefined && String(value).trim() !== '';
}

function _firstPresent(...values) {
    for (const value of values) {
        if (value !== null && value !== undefined) return value;
    }
    return undefined;
}

function _firstNonBlank(...values) {
    for (const value of values) {
        if (_hasValue(value)) return value;
    }
    return '';
}

function _toText(value) {
    if (value === null || value === undefined) return '';
    return String(value);
}

function _normalizeBbox(raw) {
    if (!raw) return null;
    let box = null;
    if (Array.isArray(raw) && raw.length >= 4) {
        const [a, b, c, d] = raw.slice(0, 4).map(Number);
        if (![a, b, c, d].every(Number.isFinite)) return null;
        if (c > a && d > b) {
            box = { x: a, y: b, w: c - a, h: d - b };
        } else {
            box = { x: a, y: b, w: c, h: d };
        }
    } else if (typeof raw === 'object') {
        if ('x' in raw && 'y' in raw && 'w' in raw && 'h' in raw) {
            box = {
                x: Number(raw.x),
                y: Number(raw.y),
                w: Number(raw.w),
                h: Number(raw.h),
            };
        } else if ('x0' in raw && 'y0' in raw && 'x1' in raw && 'y1' in raw) {
            const x0 = Number(raw.x0), y0 = Number(raw.y0);
            const x1 = Number(raw.x1), y1 = Number(raw.y1);
            box = {
                x: Math.min(x0, x1),
                y: Math.min(y0, y1),
                w: Math.abs(x1 - x0),
                h: Math.abs(y1 - y0),
            };
        }
    }
    if (!box || ![box.x, box.y, box.w, box.h].every(Number.isFinite)) return null;
    if (box.w <= 0 || box.h <= 0) return null;
    return box;
}

function _normalizeDimension(dim) {
    const bbox = _normalizeBbox(dim?.bbox ?? dim?.bbox_pt);
    if (!bbox) return null;
    const text = _firstNonBlank(
        dim.text,
        dim.raw_text,
        dim._raw_text,
        dim.nominal_text,
        dim.nominal
    );
    if (!_hasValue(text)) return null;
    return {
        ...dim,
        bbox,
        text: _toText(text),
        nominal: _firstPresent(dim.nominal, dim.nominal_text, text),
    };
}

function _viewBoxFromBackend(vb, pageIndex) {
    const page = getPageGeometry(state.pageOffsets, pageIndex);
    const localBbox = _normalizeBbox(vb);
    if (!localBbox) return null;
    const bbox = pageLocalBboxToDocumentGlobal(localBbox, page);
    const localCenter = {
        x: Number.isFinite(Number(vb.cx)) ? Number(vb.cx) : localBbox.x + localBbox.w / 2,
        y: Number.isFinite(Number(vb.cy)) ? Number(vb.cy) : localBbox.y + localBbox.h / 2,
    };
    const center = pageLocalPointToDocumentGlobal(localCenter, page);
    return {
        id:        generateUUID(),
        pageIndex,
        x: bbox.x,
        y: bbox.y,
        w: bbox.w,
        h: bbox.h,
        // R14.5 A3：透传 backend 圆形视图字段（shape='rect'|'circle'，circle 时附 cx/cy/r）
        shape: vb.shape || 'rect',
        ...(vb.shape === 'circle' ? { cx: center.x, cy: center.y, r: Number(vb.r) } : {}),
        isEdited:  false,
    };
}

/**
 * 行聚类排序：先把元素按 y 分行（同行 y 差 < tolerance），行间按 y 升序，
 * 行内按 x 升序。避免 "同行容差 + cx 兜底" 不满足传递性导致的跨行错序。
 *
 * @param {Array} items
 * @param {(it:any)=>number} getY
 * @param {(it:any)=>number} getX
 * @param {number} tolerance  同行 y 差阈值（行内允许的最大 y 跨度）
 * @returns {Array} 按 行→列 平铺的顺序
 */
function _sortIntoRows(items, getY, getX, tolerance) {
    if (items.length === 0) return [];
    const sorted = [...items].sort((a, b) => getY(a) - getY(b));
    const rows = [];
    for (const it of sorted) {
        const lastRow = rows[rows.length - 1];
        if (lastRow) {
            let rowMaxY = getY(lastRow[0]);
            for (let k = 1; k < lastRow.length; k++) {
                const y = getY(lastRow[k]);
                if (y > rowMaxY) rowMaxY = y;
            }
            if (getY(it) - rowMaxY < tolerance) {
                lastRow.push(it);
                continue;
            }
        }
        rows.push([it]);
    }
    for (const row of rows) row.sort((a, b) => getX(a) - getX(b));
    return rows.flat();
}

/**
 * 行排序阈值：基于已有 stamp 的平均 crop 高度的一半，默认 30pt。
 * 同时被 dimensionsToStamps 和 naturalSortLogic（app.js 侧边栏排序）使用。
 */
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

export function dimensionsToStamps(dimensions, pageIndex, pageViewBoxes = []) {
    const page = getPageGeometry(state.pageOffsets, pageIndex);
    const sourceDimensions = Array.isArray(dimensions) ? dimensions : [];

    // Step 1: 准备所有 physBbox
    const items = [];   // [{dim, physBbox}]
    for (const rawDim of sourceDimensions) {
        const dim = _normalizeDimension(rawDim);
        if (!dim) continue;
        const b = dim.bbox;
        // R14.5 B2: 真实 OCR bbox（贴合文字），用于 sourceBbox / dimBbox / 路由占位
        const physBbox = pageLocalBboxToDocumentGlobal(b, page);
        // R14.5 B2: 章身 PNG 裁剪专用 bbox，竖直文字 < 20pt 时居中扩展，保证缩略图清晰
        // 黄框/避让用真实 physBbox，crop 用 cropBbox，两路解耦
        const cropBbox = { x: physBbox.x, y: physBbox.y, w: physBbox.w, h: physBbox.h };
        if (dim.orientation && Math.abs(dim.orientation) > 45) {
            if (cropBbox.w < 20) {
                const expand = (20 - cropBbox.w) / 2;
                cropBbox.x -= expand;
                cropBbox.w = 20;
            }
        }
        items.push({ dim, physBbox, cropBbox });
    }
    if (items.length === 0) return { stamps: [], clusters: [] };

    // Step 2: 按视图框分组；未被任何视图框包含的尺寸走 DBSCAN 兜底
    const viewBoxGroups = new Map();  // viewBoxId → items[]
    const orphans = [];
    for (const it of items) {
        const vb = _findOwningViewBox(it.physBbox, pageViewBoxes);
        if (vb) {
            if (!viewBoxGroups.has(vb.id)) viewBoxGroups.set(vb.id, []);
            viewBoxGroups.get(vb.id).push(it);
        } else {
            orphans.push(it);
        }
    }

    const groupList = [];

    // 每个命中的视图框成一组（离心基准 = 视图框中心；边界 = 视图框矩形）
    for (const vb of pageViewBoxes) {
        const grpItems = viewBoxGroups.get(vb.id);
        if (!grpItems || grpItems.length === 0) continue;
        groupList.push({
            viewBoxId: vb.id,
            items:     grpItems,
            cx:        vb.x + vb.w / 2,
            cy:        vb.y + vb.h / 2,
            bounds: {
                minX: vb.x,
                minY: vb.y,
                maxX: vb.x + vb.w,
                maxY: vb.y + vb.h,
            },
        });
    }

    // 孤儿尺寸走 DBSCAN 兜底（如无视图框覆盖或 Type A 无视图）
    if (orphans.length > 0) {
        const orphanPoints = orphans.map(it => ({
            x: it.physBbox.x + it.physBbox.w / 2,
            y: it.physBbox.y + it.physBbox.h / 2,
        }));
        const eps = getCurrentEps();
        const labels = dbscan(orphanPoints, eps, 1);
        const orphanGroups = new Map();
        let noiseId = -100;
        for (let i = 0; i < labels.length; i++) {
            let lbl = labels[i];
            if (lbl === -1) { lbl = noiseId--; }
            if (!orphanGroups.has(lbl)) orphanGroups.set(lbl, []);
            orphanGroups.get(lbl).push(i);
        }
        for (const indices of orphanGroups.values()) {
            const grpItems = indices.map(i => orphans[i]);
            const bboxes = grpItems.map(it => it.physBbox);
            const cx = bboxes.reduce((s, b) => s + b.x + b.w / 2, 0) / bboxes.length;
            const cy = bboxes.reduce((s, b) => s + b.y + b.h / 2, 0) / bboxes.length;
            groupList.push({
                viewBoxId: null,
                items:     grpItems,
                cx, cy,
                bounds: {
                    minX: Math.min(...bboxes.map(b => b.x)),
                    minY: Math.min(...bboxes.map(b => b.y)),
                    maxX: Math.max(...bboxes.map(b => b.x + b.w)),
                    maxY: Math.max(...bboxes.map(b => b.y + b.h)),
                },
            });
        }
    }

    // 排序：行聚类（同行按 x，行间按 y；视图框高度普遍 > 100pt，用 100pt tolerance 稳健区分上下行）
    const sortedGroupList = _sortIntoRows(groupList, g => g.cy, g => g.cx, 100);
    groupList.length = 0;
    groupList.push(...sortedGroupList);

    console.log(`[dimensionsToStamps] ${items.length} items → ${groupList.length} groups (viewBoxes=${pageViewBoxes.length}, orphans=${orphans.length})`);

    // Step 5: 按组调用路由，维护全局连续 ID
    const allStamps = [];
    const synthClusters = [];
    let globalId = 1;
    const sharedOccupancy = [];   // 跨组共享 AABB 占位列表
    const sharedLeaderLines = []; // 跨组共享引线列表

    // 5e: 预先把所有 dim 的 physBbox 推入 occupancy，让章圆心避让 PDF 文本。
    // R14.5 B1: 每个 entry 附 _selfDim 引用，让 stamp.js 路由自家 stamp 时
    // 通过引用比较跳过自家 dim bbox（避免竖排文本 dim 半高 > 章 AABB 半宽时
    // 30/45pt 候选位被自家撞，导致 stamp 被推到 65pt 档非自然方向）。
    for (const it of items) {
        const b = it.physBbox;
        if (b.w > 0 && b.h > 0) {
            sharedOccupancy.push({ x: b.x, y: b.y, w: b.w, h: b.h, _selfDim: b });
        }
    }

    for (const group of groupList) {
        // 组内行聚类：同行 y 差 < threshold 视为一行，行间 y 升序、行内 x 升序
        const rowThreshold = _getSortRowThreshold();
        const sortedItems = _sortIntoRows(
            group.items,
            it => it.physBbox.y + it.physBbox.h / 2,
            it => it.physBbox.x + it.physBbox.w / 2,
            rowThreshold
        );

        const sortedBboxes = sortedItems.map(it => it.physBbox);
        const sortedCropBboxes = sortedItems.map(it => it.cropBbox);
        const sortedDims = sortedItems.map(it => it.dim);

        // 自动章按尺寸簇或视图中心向外排布；viewBox 只提供空间分组和排布基准。
        const stamps = generateRoutedStamps(
            sortedBboxes, group.cx, group.cy,
            sharedOccupancy, sharedLeaderLines
        );

        // 分配全局连续 ID 并填充 dimension 数据
        for (let k = 0; k < stamps.length; k++) {
            const s = stamps[k];
            const sourceIndex = Number.isInteger(s._sourceIndex) ? s._sourceIndex : k;
            const dim = sortedDims[sourceIndex];
            const physBbox = sortedBboxes[sourceIndex];
            const cropBbox = sortedCropBboxes[sourceIndex];  // R14.5 B2: PNG 裁剪护栏（≥20pt），与 physBbox 解耦
            delete s._sourceIndex;

            s.id = String(globalId++);

            // arrowTip 沿用 stamp.js → calculateArrowTip 的视线交点
            // （从 circleCenter 指向 bbox 中心的射线与 bbox 边界的交点 + 8pt 外推）
            // 旧版本"4 边中点最近吸附"会把斜向接近的引线硬拉到正交边中点，造成偏箭和路径折断
            s.arrowTipManual = false;

            // 连线最短距离保障 80pt → 40pt，对齐 stamp.js 新 BASE_DISTANCES[0]=30pt 起步
            const linkDist = Math.hypot(
                s.circleCenter.absX - s.arrowTip.absX,
                s.circleCenter.absY - s.arrowTip.absY
            );
            if (linkDist < 40 && linkDist > 0.1) {
                const dx = s.circleCenter.absX - s.arrowTip.absX;
                const dy = s.circleCenter.absY - s.arrowTip.absY;
                const scale = 40 / linkDist;
                s.circleCenter.absX = s.arrowTip.absX + dx * scale;
                s.circleCenter.absY = s.arrowTip.absY + dy * scale;
            }

            s.sortY = (physBbox.h > 0) ? physBbox.y + physBbox.h / 2 : s.arrowTip.absY;
            s.sortX = (physBbox.w > 0) ? physBbox.x + physBbox.w / 2 : s.arrowTip.absX;
            s.pageIndex = pageIndex;
            s.viewBoxId = group.viewBoxId;
            delete s.perimeterArc;
            delete s.perimeterSide;
            s.sourceBbox = { x: physBbox.x, y: physBbox.y, w: physBbox.w, h: physBbox.h };
            // R14.5 B2：保留 PNG 裁剪带护栏的 bbox（竖直 dim 已被扩到 ≥20pt 宽），
            // _reRouteViewBox 重建 s.crop 时用 cropSourceBbox 而非 sourceBbox，
            // 保证 reroute 后竖直章身缩略图不丢护栏（清晰度退化）
            s.cropSourceBbox = { x: cropBbox.x, y: cropBbox.y, w: cropBbox.w, h: cropBbox.h };
            s.status = 'success';
            s.confirmed = false;
            if (dim.source === 'surface_roughness') {
                s.id = '';
                s.label = dim.text;
            }

            // GD&T 元字段必须穿过同一正式章投影，让 ZIP/Excel 后端可由
            // _format_gdt 合成读者文本；datum 在首批严格复核中保持空值。
            Object.assign(s, buildStampDimensionProjection(dim));

            if (physBbox.w > 0 && physBbox.h > 0) {
                const pad = 4;
                // R14.5 B2: s.crop 用 cropBbox（带 20pt 护栏，PNG 缩略图清晰）
                s.crop = {
                    l: cropBbox.x - s.arrowTip.absX - pad,
                    r: cropBbox.x + cropBbox.w - s.arrowTip.absX + pad,
                    t: cropBbox.y - s.arrowTip.absY - pad,
                    b: cropBbox.y + cropBbox.h - s.arrowTip.absY + pad,
                };
                // R14.4 B 路线：dimBbox 绝对坐标，跟 arrowTip 解绑
                // R14.5 B2: dimBbox 用真实 physBbox（贴合文字，不被竖直 20pt 护栏撑大）
                s.dimBbox = {
                    x: physBbox.x - pad,
                    y: physBbox.y - pad,
                    w: physBbox.w + pad * 2,
                    h: physBbox.h + pad * 2,
                };
                s.cropAdjusted = true;
            }
        }

        allStamps.push(...stamps);
        synthClusters.push({
            clusterId: group.viewBoxId ?? `orphan_${synthClusters.length}`,
            center:    { x: group.cx, y: group.cy },
            bounds:    group.bounds,
        });
    }

    // 收集真正被使用的 viewBoxId，供调用方过滤空视图框
    const usedViewBoxIds = new Set();
    for (const g of groupList) {
        if (g.viewBoxId) usedViewBoxIds.add(g.viewBoxId);
    }

    return { stamps: allStamps, clusters: synthClusters, usedViewBoxIds };
}

export function basicDimensionsToOverlays(basicDimensions, pageIndex) {
    const page = getPageGeometry(state.pageOffsets, pageIndex);
    const sourceItems = Array.isArray(basicDimensions) ? basicDimensions : [];
    return sourceItems.map(raw => {
        const dim = _normalizeDimension(raw);
        if (!dim) return null;
        return {
            id: generateUUID(),
            pageIndex,
            text: dim.text,
            bbox: pageLocalBboxToDocumentGlobal(dim.bbox, page),
            localBbox: { ...dim.bbox },
            source: 'basic_dimension',
            inspectionRole: dim.inspection_role || dim.inspectionRole || 'theoretical_exact',
            numberingExcluded: dim.numbering_excluded !== false,
        };
    }).filter(Boolean);
}

export function datumReferencesToOverlays(references, pageIndex) {
    const page = getPageGeometry(state.pageOffsets, pageIndex);
    const sourceItems = Array.isArray(references) ? references : [];
    return sourceItems
        .filter(ref => ref?.type === 'datum_reference')
        .map(raw => {
            const ref = _normalizeDimension(raw);
            if (!ref) return null;
            return {
                id: generateUUID(),
                pageIndex,
                text: ref.text,
                bbox: pageLocalBboxToDocumentGlobal(ref.bbox, page),
                source: 'datum_reference',
            };
        })
        .filter(Boolean);
}

function _normalizeNoteBbox(raw) {
    if (!Array.isArray(raw) || raw.length < 4) return null;
    const [x0, y0, x1, y1] = raw.slice(0, 4).map(Number);
    if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
    if (x1 <= x0 || y1 <= y0) return null;
    return { x0, y0, x1, y1 };
}

function notesToStamps(notes, pageIndex) {
    const page = getPageGeometry(state.pageOffsets, pageIndex);
    const sourceItems = Array.isArray(notes) ? notes : [];
    return sourceItems.map((note, idx) => {
        const first = _normalizeNoteBbox(note?.bbox_first_line);
        const all = _normalizeNoteBbox(note?.bbox_all) || first;
        const text = _toText(note?.text).trim();
        if (!first || !all || !text) return null;
        const firstGlobal = pageLocalBboxToDocumentGlobal({
            x: first.x0,
            y: first.y0,
            w: first.x1 - first.x0,
            h: first.y1 - first.y0,
        }, page);
        const sourceBbox = pageLocalBboxToDocumentGlobal({
            x: all.x0,
            y: all.y0,
            w: all.x1 - all.x0,
            h: all.y1 - all.y0,
        }, page);
        const lineY = firstGlobal.y + firstGlobal.h / 2;
        const circleX = Math.max(8, firstGlobal.x - 30);
        const arrowTip = { absX: firstGlobal.x, absY: lineY };
        const circleCenter = { absX: circleX, absY: lineY };
        const pad = 4;
        return {
            uuid: generateUUID(),
            id: '',
            type: 'note',
            fs: 12,
            ratio: 0.60,
            isAutoGenerated: true,
            source: 'note',
            noteSource: note.source || 'pdf_text',
            pageIndex,
            viewBoxId: null,
            circleCenter,
            arrowTip,
            routingPath: [circleCenter, arrowTip],
            arrowTipManual: false,
            sourceBbox,
            cropSourceBbox: { ...sourceBbox },
            dimBbox: {
                x: sourceBbox.x - pad,
                y: sourceBbox.y - pad,
                w: sourceBbox.w + pad * 2,
                h: sourceBbox.h + pad * 2,
            },
            crop: {
                l: sourceBbox.x - arrowTip.absX - pad,
                r: sourceBbox.x + sourceBbox.w - arrowTip.absX + pad,
                t: sourceBbox.y - arrowTip.absY - pad,
                b: sourceBbox.y + sourceBbox.h - arrowTip.absY + pad,
            },
            cropAdjusted: true,
            status: 'success',
            confirmed: false,
            sortX: sourceBbox.x,
            sortY: sourceBbox.y + idx * 0.01,
            text,
            bboxFirstLine: [
                firstGlobal.x,
                firstGlobal.y,
                firstGlobal.x + firstGlobal.w,
                firstGlobal.y + firstGlobal.h,
            ],
            bboxAll: [
                sourceBbox.x,
                sourceBbox.y,
                sourceBbox.x + sourceBbox.w,
                sourceBbox.y + sourceBbox.h,
            ],
            dimType: '技术要求',
            aiData: {
                type: '技术要求',
                nominal: text,
                tolerance: '',
                upper_tol: '',
                lower_tol: '',
                symbol: '',
                datum: '',
                remarks: '',
                confidence: 'high',
                _raw_type: 'note',
                _raw_text: text,
            },
        };
    }).filter(Boolean);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 智能解析统一入口（范围由 state.analyzeScope 决定）
// 'current' = 仅当前页；'all' = 全部页（逐页 await，不增加客户端并发）。
// strict YOLO 默认单 worker；取消会阻止后续页并放弃本批暂存，但服务端
// 当前在途页可能仍需运行到请求清理，因此保留范围选项供长任务降载。
// ═══════════════════════════════════════════════════════════════════════════════

function _stageAnalysisResult(result, pageIndex) {
    const newViewBoxes = (result.view_boxes || [])
        .map(vb => _viewBoxFromBackend(vb, pageIndex))
        .filter(Boolean);
    const { stamps, clusters } = dimensionsToStamps(
        result.dimensions,
        pageIndex,
        newViewBoxes,
    );
    return {
        stamps: stamps.concat(notesToStamps(result.notes, pageIndex)),
        viewBoxes: newViewBoxes,
        basicDimensions: basicDimensionsToOverlays(result.basic_dimensions, pageIndex),
        datumReferences: datumReferencesToOverlays(result.references, pageIndex),
        clusters: clusters.map(cluster => ({ ...cluster, pageIndex })),
        cacheValue: Array.isArray(result.dimensions) ? result.dimensions : [],
    };
}

function _commitAnalysisHistory() {
    _reindexBySpace();
    state.clusteringLocked = false;
    pushHistory();
    state.clusteringBaseIndex = state.historyIndex;
}

function _refreshAfterAnalysis({ review = false } = {}) {
    if (review) state.sidebarTab = 'review';
    updateEpsSlider();
    updateSidebar();
    syncDrawLayerResolution();
}

function _usesReviewCandidatesV1() {
    return state.recognitionSettings.dimension_path === 'vector_only';
}

function _commitReviewHistory() {
    pushHistory();
}

function _reportStaleAnalysisOutcome(outcome) {
    if (outcome?.status !== 'stale') return false;
    alert('解析结果属于已切换的文档，已放弃');
    return true;
}

export async function _analyzeCurrentPage(btn, origHTML, signal) {
    // 解析触发后立刻清焦点，避免按钮持续显示 :focus 蓝框造成"未恢复"视觉错觉
    btn?.blur();
    const curPage = parseInt(document.getElementById('page-indicator').innerText.split('/')[0]);
    const reviewMode = _usesReviewCandidatesV1();

    // 防呆：当前页已有 AI Stamps 时确认
    const hasAiStampsOnPage = state.stamps.some(s => s.isAutoGenerated && s.pageIndex === curPage);
    if (!reviewMode && hasAiStampsOnPage) {
        if (!confirm(
            `重新解析将清除第 ${curPage} 页的 AI 自动标记并重置序号，是否继续？\n` +
            '（其他页标记和手工绘制的标记将被保留）'
        )) return;
    }

    showLoading(`矢量解析中 (第 ${curPage} 页)`);

    try {
        let stagedVectorDebug = new Map();
        const common = {
            state,
            pages: [curPage],
            signal,
            analyzePage: async pageIndex => {
                const result = await runAnalysis(pageIndex, { signal });
                stagedVectorDebug = captureVectorDebugResponse(
                    stagedVectorDebug,
                    pageIndex,
                    result,
                );
                return result;
            },
            beforeCommit: replaceCurrentHistorySnapshot,
        };
        const outcome = reviewMode
            ? await executeReviewCandidatesTransaction({
                ...common,
                pushHistory: _commitReviewHistory,
            })
            : await executeReanalysisTransaction({
                ...common,
                stagePage: _stageAnalysisResult,
                pushHistory: _commitAnalysisHistory,
            });
        if (_reportStaleAnalysisOutcome(outcome)) return outcome;
        applyVectorDebugOutcome(stagedVectorDebug, outcome);
        if (outcome.status === 'success') _refreshAfterAnalysis({ review: reviewMode });
        return outcome;
    } finally {
        // 修：之前函数末尾没调 hideLoading，导致解析完成后"解析中..."loading
        // 永远不消失。现在用 try/finally 保证无论成功/失败/早返回都 hide。
        hideLoading();
    }
}

// 空间重排：只重排未确认的 AI 自动章，保留已确认/手绘编号。
// 解析完成、批量增删后也复用这条路径，避免自动章和人工章抢号。
export function _reindexBySpace() {
    if (state.stamps.length === 0) return;
    const shouldReindex = s => s.isAutoGenerated && !s.confirmed;
    const reservedIds = new Set(
        state.stamps
            .filter(s => !shouldReindex(s))
            .map(s => String(s.id || '').trim())
            .filter(Boolean)
    );
    const sorted = [...state.stamps].sort(naturalSortLogic);
    let nextId = 1;
    for (const stamp of sorted) {
        if (!shouldReindex(stamp)) continue;
        while (reservedIds.has(String(nextId))) nextId += 1;
        stamp.id = String(nextId);
        reservedIds.add(stamp.id);
        nextId += 1;
    }
    state.stamps = sorted;
}

export async function _analyzeAllPages(btn, origHTML, signal) {
    const totalPages = state.totalPages;
    if (totalPages <= 0) return;

    const reviewMode = _usesReviewCandidatesV1();
    const hasAny = state.stamps.some(s => s.isAutoGenerated);
    if (!reviewMode && hasAny) {
        if (!confirm(`全部解析将重新分析全部 ${totalPages} 页；失败页保留旧数据，是否继续？`)) return;
    }

    const pages = Array.from({ length: totalPages }, (_, index) => index + 1);
    let stagedVectorDebug = new Map();
    const vectorDebugRequestBudget = createVectorDebugRequestBudget();
    const common = {
        state,
        pages,
        signal,
        analyzePage: async pageIndex => {
            const completed = pageIndex - 1;
            showLoading(`解析中 ${completed}/${totalPages} (第 ${pageIndex} 页)`);
            btn.innerHTML = `<span>取消 ${completed}/${totalPages}</span>`;
            const result = await runAnalysis(pageIndex, {
                signal,
                vectorDebugRequest: vectorDebugRequestBudget.take(),
            });
            stagedVectorDebug = captureVectorDebugResponse(
                stagedVectorDebug,
                pageIndex,
                result,
            );
            return result;
        },
        beforeCommit: replaceCurrentHistorySnapshot,
        onProgress: ({ completed }) => {
            btn.innerHTML = `<span>取消 ${completed}/${totalPages}</span>`;
        },
    };
    const outcome = reviewMode
        ? await executeReviewCandidatesTransaction({
            ...common,
            pushHistory: _commitReviewHistory,
        })
        : await executeReanalysisTransaction({
            ...common,
            stagePage: _stageAnalysisResult,
            pushHistory: _commitAnalysisHistory,
        });
    if (_reportStaleAnalysisOutcome(outcome)) return outcome;
    applyVectorDebugOutcome(stagedVectorDebug, outcome);
    if (outcome.status === 'success' || outcome.status === 'partial') {
        _refreshAfterAnalysis({ review: reviewMode });
    }
    const skippedCount = Array.isArray(outcome.skippedPages)
        ? outcome.skippedPages.length
        : 0;
    if (
        skippedCount > 0
        && (outcome.status === 'partial' || outcome.status === 'failed')
    ) {
        alert(`全部解析已停止：${outcome.succeededPages.length} 页成功，`
            + `${outcome.failedPages.length} 页失败，${skippedCount} 页未尝试；旧数据均保留。`);
    } else if (outcome.status === 'partial') {
        alert(`全部解析完成：${outcome.succeededPages.length} 页成功，`
            + `${outcome.failedPages.length} 页失败并保留旧数据。`);
    }
    return outcome;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 按钮绑定
// ═══════════════════════════════════════════════════════════════════════════════

document.getElementById('btn-analyze').addEventListener('click', async () => {
    if (state.isAnalysisRunning) {
        state.analysisAbortController?.abort();
        return;
    }
    if (state.isDocumentImportRunning) return alert(DOCUMENT_LOADING_MESSAGE);
    if (!state.currentPdfBuffer) return alert('请先打开图纸文档！');

    const btn = document.getElementById('btn-analyze');
    // 保存完整 innerHTML（含 sparkles SVG 图标），仅用 innerText 会丢 SVG，
    // finally 恢复时只剩纯文本"解析"，图标永久消失。
    const origHTML = btn.innerHTML;
    btn.innerHTML = '<span>取消解析</span>';
    btn.disabled  = false;
    const controller = new AbortController();
    state.analysisAbortController = controller;

    try {
        if (state.analyzeScope === 'all') await _analyzeAllPages(btn, origHTML, controller.signal);
        else                              await _analyzeCurrentPage(btn, origHTML, controller.signal);
    } finally {
        hideLoading();
        state.analysisAbortController = null;
        btn.innerHTML = origHTML;
        btn.disabled  = state.isAnalysisRunning;
    }
});

// 控制面板"解析范围" radio → 同步到 state.analyzeScope
document.querySelectorAll('input[name="analyze-scope"]').forEach(r => {
    r.addEventListener('change', e => { state.analyzeScope = e.target.value; });
});
