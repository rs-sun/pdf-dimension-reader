// app_interaction.js — orchestrator 拆分子模块
// 职责：画布 hit-testing + mousedown/mousemove/mouseup/dblclick/wheel 交互引擎
//       + stamp 路径重算 / viewBox re-route 工具
//
// 依赖注入：`commitStamp` 由主 app.js 通过 `registerInteractionCallbacks()` 晚绑定。
// 原因：commitStamp 需要访问 DOM 输入框 + pushHistory，是编排层职责；本模块只调用不实现。

import { state } from './state.js';
import {
    els, updateTransform, syncDrawLayerResolution,
    getCanvasCoords,
} from './canvas_renderer.js';
import {
    initCrop,
    generateRoutedStamps,
    ensureDimBbox,
    recalcArrowTipFromDimBbox,
} from './stamp.js';
import { pushHistory, cancelPanTo, stepZoom } from './app_utils.js';
import { updateCursor, syncToolbarFromSelection } from './app_cursor.js';
import { updateSidebar } from './app_sidebar.js';

// ── 晚绑定回调 ────────────────────────────────────────────────────────────────
let _commitStamp = null;
let _requestStampOCR = null;

/** 由主 app.js 注入 commitStamp + _requestStampOCR 实现。必须在 DOM 事件触发前调用。 */
export function registerInteractionCallbacks(commitStampFn, requestStampOCRFn) {
    _commitStamp = commitStampFn;
    _requestStampOCR = requestStampOCRFn || null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hit-testing 工具
// ═══════════════════════════════════════════════════════════════════════════════

/** 点到线段的最短距离 */
function pointToLineDist(px, py, x1, y1, x2, y2) {
    const A = px-x1, B = py-y1, C = x2-x1, D = y2-y1;
    const dot = A*C + B*D, len_sq = C*C + D*D;
    let param = len_sq !== 0 ? dot / len_sq : -1;
    let xx, yy;
    if      (param < 0) { xx = x1; yy = y1; }
    else if (param > 1) { xx = x2; yy = y2; }
    else                { xx = x1 + param*C; yy = y1 + param*D; }
    const dx = px-xx, dy = py-yy;
    return Math.sqrt(dx*dx + dy*dy);
}

/** 检测点是否命中视图框角/边句柄，返回句柄类型或 null
 *  R14.5 A3：圆形 vb 不支持 8 角矩形 resize 句柄，直接跳过。 */
function _hitViewBoxHandle(pt) {
    const threshold = 6 / state.zoom;
    for (let i = state.viewBoxes.length - 1; i >= 0; i--) {
        const vb = state.viewBoxes[i];
        if ((vb.shape || 'rect') === 'circle') continue;
        const l = vb.x, r = vb.x + vb.w;
        const t = vb.y, b = vb.y + vb.h;
        const cx = (l + r) / 2, cy = (t + b) / 2;
        const handles = [
            { t: 'vbox_nw', x: l,  y: t  },
            { t: 'vbox_n',  x: cx, y: t  },
            { t: 'vbox_ne', x: r,  y: t  },
            { t: 'vbox_w',  x: l,  y: cy },
            { t: 'vbox_e',  x: r,  y: cy },
            { t: 'vbox_sw', x: l,  y: b  },
            { t: 'vbox_s',  x: cx, y: b  },
            { t: 'vbox_se', x: r,  y: b  },
        ];
        for (const h of handles) {
            if (Math.hypot(pt.x - h.x, pt.y - h.y) <= threshold)
                return { viewBoxId: vb.id, type: h.t };
        }
    }
    return null;
}

/** 检测点是否落在视图框边界附近（R14.5 A2: 改"整框命中"为"边界 ±6px 命中"，
 *  让中间区域不响应视图框，章/文字/空地交互不再被视图框遮挡）。
 *  R14.5 A3：圆形 vb 用 |dist(pt, center) - r| <= 6/zoom 判圆边缘。 */
function _hitViewBoxBody(pt) {
    const edgeThreshold = 6 / state.zoom;
    for (let i = state.viewBoxes.length - 1; i >= 0; i--) {
        const vb = state.viewBoxes[i];
        const shape = vb.shape || 'rect';
        if (shape === 'circle' && typeof vb.cx === 'number'
                && typeof vb.cy === 'number' && typeof vb.r === 'number') {
            const d = Math.hypot(pt.x - vb.cx, pt.y - vb.cy);
            if (Math.abs(d - vb.r) <= edgeThreshold) {
                return { viewBoxId: vb.id, type: 'vbox_body' };
            }
            continue;
        }
        const l = vb.x, r = vb.x + vb.w;
        const t = vb.y, b = vb.y + vb.h;
        if (pt.x < l - edgeThreshold || pt.x > r + edgeThreshold ||
            pt.y < t - edgeThreshold || pt.y > b + edgeThreshold) continue;
        // 距最近边的距离 ≤ threshold 才算命中边界
        const distToEdge = Math.min(
            Math.abs(pt.x - l), Math.abs(pt.x - r),
            Math.abs(pt.y - t), Math.abs(pt.y - b)
        );
        if (distToEdge <= edgeThreshold) {
            return { viewBoxId: vb.id, type: 'vbox_body' };
        }
    }
    return null;
}

function detectHit(pt) {
    const threshold = 6 / state.zoom;

    // 新优先级：
    //   0. 已选中 Stamp 的 crop / body_xx / arrow 句柄   ← 最高
    //   1. 全量 Stamp 气泡 body / 引线 line              ← 压过 vbox 句柄
    //   2. 视图框角/边句柄
    //   3. 视图框体（兜底）

    for (let i = state.stamps.length - 1; i >= 0; i--) {
        const s = state.stamps[i];
        if (!state.selectedUUIDs.includes(s.uuid)) continue;

        // cropBox 句柄（R14.4 B 路线：读 s.dimBbox 绝对坐标，跟 arrowTip 解绑）
        if (!(s.source === 'legacy_stamp' && s.dimBboxPlaceholder === true)) {
            ensureDimBbox(s);
            const db = s.dimBbox;
            if (db && db.w > 0 && db.h > 0) {
                const l = db.x, r = db.x + db.w;
                const t = db.y, b = db.y + db.h;
                const cx = (l+r)/2, cy = (t+b)/2;
                const cropHandles = [
                    {t:'crop_nw',x:l,y:t},{t:'crop_n',x:cx,y:t},{t:'crop_ne',x:r,y:t},
                    {t:'crop_w', x:l,y:cy},                      {t:'crop_e', x:r,y:cy},
                    {t:'crop_sw',x:l,y:b},{t:'crop_s',x:cx,y:b},{t:'crop_se',x:r,y:b},
                ];
                for (const c of cropHandles) {
                    if (Math.hypot(pt.x-c.x, pt.y-c.y) <= threshold)
                        return { uuid: s.uuid, type: c.t };
                }
                if (pt.x >= l && pt.x <= r && pt.y >= t && pt.y <= b)
                    return { uuid: s.uuid, type: 'crop_body' };
            }
        }

        // 四角缩放句柄
        {
            const r2 = s.fs * (s.ratio || 0.60);
            const idLen = s.id ? s.id.toString().length : 1;
            const extraW = idLen <= 1 ? 0 : (idLen - 1) * s.fs * ((s.ratio || 0.60) * 0.5);
            const pad = 4;
            const bLeft = s.circleCenter.absX - extraW - r2 - pad;
            const bRight = s.circleCenter.absX + extraW + r2 + pad;
            const bTop = s.circleCenter.absY - r2 - pad;
            const bBottom = s.circleCenter.absY + r2 + pad;
            const corners = [
                { t: 'body_nw', x: bLeft, y: bTop },
                { t: 'body_ne', x: bRight, y: bTop },
                { t: 'body_sw', x: bLeft, y: bBottom },
                { t: 'body_se', x: bRight, y: bBottom },
            ];
            for (const c of corners) {
                if (Math.hypot(pt.x - c.x, pt.y - c.y) <= threshold)
                    return { uuid: s.uuid, type: c.t };
            }
        }

        // arrowTip 句柄
        if (Math.hypot(pt.x-s.arrowTip.absX, pt.y-s.arrowTip.absY) <= threshold)
            return { uuid: s.uuid, type: 'arrow' };
    }

    // 全量：气泡 AABB + routingPath 折线
    for (let i = state.stamps.length - 1; i >= 0; i--) {
        const s = state.stamps[i];
        const r2     = s.fs * (s.ratio || 0.60);
        const idLen  = s.id ? s.id.toString().length : 1;
        const extraW = idLen <= 1 ? 0 : (idLen-1) * s.fs * ((s.ratio||0.60)*0.5);
        const box = {
            left:   s.circleCenter.absX - extraW - r2,
            right:  s.circleCenter.absX + extraW + r2,
            top:    s.circleCenter.absY - r2,
            bottom: s.circleCenter.absY + r2,
        };
        if (pt.x >= box.left && pt.x <= box.right &&
            pt.y >= box.top  && pt.y <= box.bottom)
            return { uuid: s.uuid, type: 'body' };

        const rp = (s.routingPath && s.routingPath.length >= 2)
            ? s.routingPath
            : [s.circleCenter, s.arrowTip];
        for (let k = 0; k < rp.length - 1; k++) {
            if (pointToLineDist(pt.x, pt.y,
                    rp[k].absX, rp[k].absY, rp[k+1].absX, rp[k+1].absY) <= threshold)
                return { uuid: s.uuid, type: 'line' };
        }
    }

    const vbHandleHit = _hitViewBoxHandle(pt);
    if (vbHandleHit) return vbHandleHit;

    const vbBodyHit = _hitViewBoxBody(pt);
    if (vbBodyHit) return vbBodyHit;

    return null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stamp 路径重算 + viewBox re-route
// ═══════════════════════════════════════════════════════════════════════════════

function _recalcRoutingPath(s) {
    s.routingPath = [
        { absX: s.circleCenter.absX, absY: s.circleCenter.absY },
        { absX: s.arrowTip.absX, absY: s.arrowTip.absY },
    ];
}

/** R14.4：dragCache.original 兼容老 stamp（无 dimBbox）→ 从 arrowTip+crop 算
 *  正常情况 mousedown 时已 ensureDimBbox，但保留 fallback 防边界 */
function _legacyDimBboxFromOrig(orig) {
    if (orig.dimBbox && orig.dimBbox.w > 0 && orig.dimBbox.h > 0) return orig.dimBbox;
    if (!orig.crop || !orig.arrowTip) return null;
    return {
        x: orig.arrowTip.absX + orig.crop.l,
        y: orig.arrowTip.absY + orig.crop.t,
        w: orig.crop.r - orig.crop.l,
        h: orig.crop.b - orig.crop.t,
    };
}

/** 估算 stamp 的 AABB 占位（_reRouteViewBox 用）*/
function _stampOccupancyAABB(s) {
    const PAD = 8;
    const idDigits = String(s.id || '1').length;
    const W = idDigits <= 1 ? 44 : idDigits <= 2 ? 58 : 72;
    const H = 28;
    const cx = s.circleCenter.absX;
    const cy = s.circleCenter.absY;
    return { x: cx - W / 2 - PAD, y: cy - H / 2 - PAD, w: W + PAD * 2, h: H + PAD * 2 };
}

function _reRouteViewBox(viewBoxId) {
    const vb = state.viewBoxes.find(v => v.id === viewBoxId);
    if (!vb) return;

    const affected = state.stamps.filter(
        s => s.viewBoxId === viewBoxId && s.isAutoGenerated && s.sourceBbox
    );
    if (affected.length === 0) return;

    affected.sort((a, b) => (parseInt(a.id) || 0) - (parseInt(b.id) || 0));

    const sharedOccupancy = state.stamps
        .filter(s => s.viewBoxId !== viewBoxId && s.circleCenter)
        .map(_stampOccupancyAABB);

    const sourceBboxes = affected.map(s => ({ ...s.sourceBbox }));
    let repelCX;
    let repelCY;
    if ((vb.shape || 'rect') === 'circle' && typeof vb.cx === 'number'
            && typeof vb.cy === 'number' && typeof vb.r === 'number') {
        repelCX = vb.cx;
        repelCY = vb.cy;
    } else {
        repelCX = vb.x + vb.w / 2;
        repelCY = vb.y + vb.h / 2;
    }

    const routed = generateRoutedStamps(
        sourceBboxes, repelCX, repelCY,
        sharedOccupancy, []
    );

    for (let i = 0; i < affected.length && i < routed.length; i++) {
        const ns = routed[i];
        const sourceIndex = Number.isInteger(ns._sourceIndex) ? ns._sourceIndex : i;
        const s  = affected[sourceIndex];
        if (!s) continue;
        s.circleCenter = ns.circleCenter;
        delete s.perimeterArc;
        delete s.perimeterSide;
        // arrowTipManual=true 意味着用户已手动落点，re-route 不能覆盖
        if (!s.arrowTipManual) s.arrowTip = ns.arrowTip;
        _recalcRoutingPath(s);

        // R14.5 B2：crop 用 cropSourceBbox（带竖直 20pt 护栏）；
        // dimBbox（黄框）仍用 sourceBbox（贴合文字，避免被 20pt 护栏撑大）。
        // 缺 cropSourceBbox 的老 stamp（reroute 之前的存档）fallback 用 sourceBbox 维持原行为。
        const sb  = s.sourceBbox;
        const csb = s.cropSourceBbox || s.sourceBbox;
        if (sb && sb.w > 0 && sb.h > 0) {
            const pad = 4;
            if (csb && csb.w > 0 && csb.h > 0) {
                s.crop = {
                    l: csb.x - s.arrowTip.absX - pad,
                    r: csb.x + csb.w - s.arrowTip.absX + pad,
                    t: csb.y - s.arrowTip.absY - pad,
                    b: csb.y + csb.h - s.arrowTip.absY + pad,
                };
            }
            // R14.4 B 路线：dimBbox 绝对坐标，跟 arrowTip 解绑（用真实 sourceBbox）
            s.dimBbox = {
                x: sb.x - pad,
                y: sb.y - pad,
                w: sb.w + pad * 2,
                h: sb.h + pad * 2,
            };
            s.cropAdjusted = true;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 交互事件绑定（模块加载即注册）
// ═══════════════════════════════════════════════════════════════════════════════

window.isPanning = false;
const DRAG_THRESHOLD_PX = 4;

// ── mousedown ──────────────────────────────────────────────────────────────────
els.workspace.addEventListener('mousedown', e => {
    // 用户鼠标按下时立即终止 panTo 动画
    cancelPanTo();
    if (document.activeElement.tagName === 'INPUT') document.activeElement.blur();
    if (e.target.closest('#toolbar') || e.target.closest('#sidebar-container') ||
        e.target.closest('#minimap') || e.target.closest('#stamp-options')) return;

    // 平移：中键 / Space+左键 / 手型工具
    if (e.button === 1 ||
        (state.keys.space && e.button === 0) ||
        (!state.stampActive && state.baseTool === 'hand')) {
        e.preventDefault();
        window.isPanning = true;
        updateCursor();
        return;
    }
    if (e.button !== 0 && e.button !== 2) return;

    const pt = getCanvasCoords(e);

    // ── 标记绘制模式 ────────────────────────────────────────────────────────
    if (state.stampActive) {
        const mode = state.drawMode;
        // tempType 完全由 #global-color-toggle 单击控制（红=normal, 蓝=key），
        // 不在 mousedown 里覆盖。逻辑：单击切换器决定下一笔类型，连续画都是同色。
        if (mode === 'click') {
            if (state.drawPhase === 0) {
                state.tempP1 = pt;
                state.lastMousePt = pt;
                state.drawPhase = 1;
                updateCursor();
                syncDrawLayerResolution();
            } else if (state.drawPhase === 1) {
                const startPt = state.tempP1;
                state.drawPhase = 0; state.tempP1 = null;
                const order = state.drawOrder;
                if (order === 'arrow_first') _commitStamp(pt, startPt);
                else                         _commitStamp(startPt, pt);
                updateCursor();
            }
        } else if (mode === 'drag') {
            state.tempP1 = pt;
            state.lastMousePt = pt;
            state.drawPhase = 2;
            updateCursor();
        }
        return;
    }

    // ── 指针工具（选中 / 拖拽 / 框选）──────────────────────────────────────
    if (state.baseTool === 'pointer' && e.button === 0) {
        const hit = detectHit(pt);
        if (hit && hit.type && hit.type.startsWith('vbox_')) {
            const vb = state.viewBoxes.find(v => v.id === hit.viewBoxId);
            if (vb) {
                state.dragCache = {
                    type: hit.type, viewBoxId: hit.viewBoxId, startPt: pt,
                    screenStart: { x: e.clientX, y: e.clientY },
                    hasMoved: false,
                    original: JSON.parse(JSON.stringify(vb)),
                };
            }
        } else if (hit) {
            if (e.ctrlKey || e.metaKey) {
                const pos = state.selectedUUIDs.indexOf(hit.uuid);
                if (pos > -1) state.selectedUUIDs.splice(pos, 1);
                else state.selectedUUIDs.push(hit.uuid);
            } else {
                if (!state.selectedUUIDs.includes(hit.uuid))
                    state.selectedUUIDs = [hit.uuid];
            }
            syncToolbarFromSelection();
            const ts = state.stamps.find(s => s.uuid === hit.uuid);
            // R14.4：mousedown 时确保 dimBbox 已存在，dragCache.original 拷贝才完整
            ensureDimBbox(ts);
            state.dragCache = {
                type: hit.type, uuid: hit.uuid, startPt: pt,
                screenStart: { x: e.clientX, y: e.clientY },
                hasMoved: false,
                original: JSON.parse(JSON.stringify(ts)),
            };
        } else {
            state.selectedUUIDs = [];
            syncToolbarFromSelection();
            state.selectionBox = { startX: pt.x, startY: pt.y, endX: pt.x, endY: pt.y };
        }
        updateSidebar(); syncDrawLayerResolution();
    }
});

// ── mousemove ─────────────────────────────────────────────────────────────────
window.addEventListener('mousemove', e => {
    if (window.isPanning) {
        state.panX += e.movementX; state.panY += e.movementY;
        updateTransform(); return;
    }
    if (e.target.closest && (
        e.target.closest('#toolbar') || e.target.closest('#sidebar-container') ||
        e.target.closest('#stamp-options'))) return;

    const pt = getCanvasCoords(e);
    state.lastMousePt = pt;

    if (state.stampActive) {
        if ((state.drawPhase === 1 || state.drawPhase === 2) && state.tempP1)
            syncDrawLayerResolution();
        return;
    }

    if (state.baseTool === 'pointer') {
        if (state.dragCache && !state.dragCache.hasMoved) {
            const ss = state.dragCache.screenStart;
            if (ss) {
                const scrDx = e.clientX - ss.x;
                const scrDy = e.clientY - ss.y;
                if (Math.hypot(scrDx, scrDy) < DRAG_THRESHOLD_PX) return;
            }
            state.dragCache.hasMoved = true;
        }
        if (state.dragCache && state.dragCache.type && state.dragCache.type.startsWith('vbox_')) {
            const vb   = state.viewBoxes.find(v => v.id === state.dragCache.viewBoxId);
            const orig = state.dragCache.original;
            const dx   = pt.x - state.dragCache.startPt.x;
            const dy   = pt.y - state.dragCache.startPt.y;
            if (vb) {
                const MIN_SIZE = 20;
                const t = state.dragCache.type;
                if (t === 'vbox_body') {
                    vb.x = orig.x + dx;
                    vb.y = orig.y + dy;
                } else {
                    let L = orig.x, R = orig.x + orig.w;
                    let T = orig.y, B = orig.y + orig.h;
                    if (t.includes('w')) L = Math.min(orig.x + dx, R - MIN_SIZE);
                    if (t.includes('e')) R = Math.max(orig.x + orig.w + dx, L + MIN_SIZE);
                    if (t.includes('n')) T = Math.min(orig.y + dy, B - MIN_SIZE);
                    if (t.includes('s')) B = Math.max(orig.y + orig.h + dy, T + MIN_SIZE);
                    vb.x = L; vb.y = T;
                    vb.w = R - L; vb.h = B - T;
                }
                vb.isEdited = true;
            }
            syncDrawLayerResolution();
        } else if (state.dragCache) {
            const s    = state.stamps.find(st => st.uuid === state.dragCache.uuid);
            const orig = state.dragCache.original;
            const dx   = pt.x - state.dragCache.startPt.x;
            const dy   = pt.y - state.dragCache.startPt.y;

            // R14.4 B 路线：所有黄框相关操作改写 s.dimBbox（绝对坐标）；
            //   也保留 s.crop 同步以兼容老代码路径（surface_roughness 渲染等）
            if (state.dragCache.type.startsWith('crop_')) {
                const ct = state.dragCache.type;
                const odb = orig.dimBbox || _legacyDimBboxFromOrig(orig);
                if (!s.dimBbox) s.dimBbox = { ...odb };
                let nx = odb.x, ny = odb.y, nw = odb.w, nh = odb.h;
                if (ct === 'crop_body') {
                    nx = odb.x + dx; ny = odb.y + dy;
                } else {
                    // 边/角句柄：l = x；r = x+w；t = y；b = y+h
                    if (ct.includes('w')) { nx = odb.x + dx; nw = odb.w - dx; }
                    if (ct.includes('e')) {                  nw = odb.w + dx; }
                    if (ct.includes('n')) { ny = odb.y + dy; nh = odb.h - dy; }
                    if (ct.includes('s')) {                  nh = odb.h + dy; }
                    // 防止 w/h 翻负
                    if (nw < 4) nw = 4;
                    if (nh < 4) nh = 4;
                }
                s.dimBbox = { x: nx, y: ny, w: nw, h: nh };
                // 同步老 crop 字段（保持向后兼容；surface_roughness 等仍在用）
                if (s.arrowTip) {
                    s.crop = {
                        l: nx - s.arrowTip.absX,
                        r: nx + nw - s.arrowTip.absX,
                        t: ny - s.arrowTip.absY,
                        b: ny + nh - s.arrowTip.absY,
                    };
                }
            } else if (state.dragCache.type.startsWith('body_')) {
                const deltaY = pt.y - state.dragCache.startPt.y;
                const newFs = Math.max(8, Math.min(48, state.dragCache.original.fs - deltaY * 0.2));
                s.fs = Math.round(newFs);
                _recalcRoutingPath(s);
            } else if (state.dragCache.type === 'body') {
                // 拖章身：circleCenter 平移；dimBbox 不动（黄框钉在原位）；
                //   arrowTip 自动重算沿新视线 → dimBbox 边缘 + 8pt 外推（"环绕旋转"）
                s.circleCenter.absX = orig.circleCenter.absX + dx;
                s.circleCenter.absY = orig.circleCenter.absY + dy;
                ensureDimBbox(s);
                recalcArrowTipFromDimBbox(s);
                s.arrowTipManual = false;  // 章拖动 = 重算箭头，撤销 manual 锁
                _recalcRoutingPath(s);
            } else if (state.dragCache.type === 'arrow') {
                // 拖蓝色小圈：arrowTip 跟到鼠标；dimBbox 不动（修复"黄框跟着飞"bug）
                s.arrowTip.absX = orig.arrowTip.absX + dx;
                s.arrowTip.absY = orig.arrowTip.absY + dy;
                s.arrowTipManual = true;
                _recalcRoutingPath(s);
            } else if (state.dragCache.type === 'line') {
                // 拖引线：章 + 箭头 + 黄框整体平移
                s.circleCenter.absX = orig.circleCenter.absX + dx;
                s.circleCenter.absY = orig.circleCenter.absY + dy;
                s.arrowTip.absX    = orig.arrowTip.absX + dx;
                s.arrowTip.absY    = orig.arrowTip.absY + dy;
                s.arrowTipManual = true;
                const odb = orig.dimBbox || _legacyDimBboxFromOrig(orig);
                if (odb) {
                    s.dimBbox = { x: odb.x + dx, y: odb.y + dy, w: odb.w, h: odb.h };
                }
                _recalcRoutingPath(s);
            }
            syncDrawLayerResolution();
        } else if (state.selectionBox) {
            state.selectionBox.endX = pt.x; state.selectionBox.endY = pt.y;
            syncDrawLayerResolution();
        } else {
            state.hoverNode = detectHit(pt); updateCursor();
            // R14.5 A1：根据 hover 命中更新 hoverViewBoxId（仅矩形 vb 的边缘命中），
            // drawViewBoxesLayer 据此决定要不要画 8 句柄。圆形 vb 不支持矩形 resize 句柄，跳过。
            const hovered = (state.hoverNode &&
                (state.hoverNode.type === 'vbox_body'
                 || (state.hoverNode.type && state.hoverNode.type.startsWith('vbox_'))))
                ? state.hoverNode.viewBoxId : null;
            // 圆形 vb 不显示句柄
            let nextHoverVbId = null;
            if (hovered) {
                const vb = state.viewBoxes.find(v => v.id === hovered);
                if (vb && (vb.shape || 'rect') === 'rect') nextHoverVbId = hovered;
            }
            if (state.hoverViewBoxId !== nextHoverVbId) {
                state.hoverViewBoxId = nextHoverVbId;
                syncDrawLayerResolution();
            }
        }
    }
});

// ── mouseup ───────────────────────────────────────────────────────────────────
window.addEventListener('mouseup', e => {
    if (window.isPanning) { window.isPanning = false; updateCursor(); return; }

    const pt = getCanvasCoords(e);

    if (state.stampActive && state.drawPhase === 2 && state.tempP1) {
        const startPt = state.tempP1;
        state.drawPhase = 0; state.tempP1 = null;
        if (Math.hypot(pt.x-startPt.x, pt.y-startPt.y) > 5 / state.zoom) {
            const order = state.drawOrder;
            if (order === 'arrow_first') _commitStamp(pt, startPt);
            else                         _commitStamp(startPt, pt);
        } else {
            syncDrawLayerResolution();
        }
        updateCursor();
        return;
    }

    if (!state.stampActive && state.baseTool === 'pointer') {
        if (state.dragCache) {
            if (!state.dragCache.hasMoved) {
                state.dragCache = null; syncToolbarFromSelection();
                return;
            }
            if (state.dragCache.type.startsWith('crop_')) {
                const s = state.stamps.find(st => st.uuid === state.dragCache.uuid);
                if (s) {
                    s.cropAdjusted = true;
                    if (s.source === 'legacy_stamp') {
                        s.dimBboxPlaceholder = false;
                    }
                    // HITL #5: crop 调整结束后自动重 OCR（只对手绘章；fire-and-forget）
                    if (!s.isAutoGenerated && _requestStampOCR) {
                        _requestStampOCR(s);
                    }
                }
            }
            if (state.dragCache.type.startsWith('vbox_')) {
                _reRouteViewBox(state.dragCache.viewBoxId);
            }
            pushHistory(); state.dragCache = null; syncToolbarFromSelection();
        } else if (state.selectionBox) {
            const sx = Math.min(state.selectionBox.startX, state.selectionBox.endX);
            const ex = Math.max(state.selectionBox.startX, state.selectionBox.endX);
            const sy = Math.min(state.selectionBox.startY, state.selectionBox.endY);
            const ey = Math.max(state.selectionBox.startY, state.selectionBox.endY);
            state.selectedUUIDs = [];
            state.stamps.forEach(s => {
                if (s.circleCenter.absX >= sx && s.circleCenter.absX <= ex &&
                    s.circleCenter.absY >= sy && s.circleCenter.absY <= ey)
                    state.selectedUUIDs.push(s.uuid);
            });
            state.selectionBox = null;
            syncToolbarFromSelection(); updateSidebar(); syncDrawLayerResolution();
        }
    }
});

// ── dblclick: 画布内联编辑 Stamp ID ─────────────────────────────────────────
els.workspace.addEventListener('dblclick', e => {
    if (state.baseTool !== 'pointer' || state.stampActive) return;
    if (e.target.closest('#toolbar') || e.target.closest('#sidebar-container') ||
        e.target.closest('#minimap') || e.target.closest('#stamp-options')) return;

    const pt = getCanvasCoords(e);
    const hit = detectHit(pt);
    if (!hit || hit.type !== 'body') return;

    const stamp = state.stamps.find(s => s.uuid === hit.uuid);
    if (!stamp) return;

    const existing = document.querySelector('.inline-editor');
    if (existing) existing.remove();

    const screenX = stamp.circleCenter.absX * state.zoom + state.panX;
    const screenY = stamp.circleCenter.absY * state.zoom + state.panY;

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'inline-editor';
    input.value = stamp.id;
    input.style.left = screenX + 'px';
    input.style.top = screenY + 'px';
    input.style.fontSize = (stamp.fs * state.zoom) + 'px';

    els.workspace.appendChild(input);
    input.focus();
    input.select();

    const save = () => {
        const newId = input.value.trim();
        if (newId && newId !== stamp.id) {
            stamp.id = newId;
            pushHistory();
            syncDrawLayerResolution();
            syncToolbarFromSelection();
            updateSidebar();
        }
        if (input.parentNode) input.remove();
    };

    input.addEventListener('blur', save);
    input.addEventListener('keydown', ev => {
        if (ev.key === 'Enter') input.blur();
        if (ev.key === 'Escape') {
            input.value = stamp.id;
            input.blur();
        }
    });
});

// ── wheel ─────────────────────────────────────────────────────────────────────
els.workspace.addEventListener('wheel', e => {
    e.preventDefault();
    if (state.keys.ctrl || state.keys.alt) stepZoom(e.deltaY > 0 ? -1 : 1);
    else if (state.keys.space) { state.panX -= e.deltaY; updateTransform(); }
    else { state.panY -= e.deltaY; updateTransform(); }
}, { passive: false });

// 禁用 workspace 右键原生菜单
els.workspace.addEventListener('contextmenu', e => e.preventDefault());
