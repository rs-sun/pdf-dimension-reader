// js/canvas_renderer.js
// TDD 参考: 2.2.3, 6.4
import { state } from './state.js';
import { CONFIG } from './config.js';
import { initCrop, ensureDimBbox } from './stamp.js';
import {
    createPageGeometry,
    getPageGeometry,
    normalizePageRotation,
    pageLocalBboxToDocumentGlobal,
} from './page_coordinates.js';
import { listReviewCandidateEntries } from './review_candidates_v1.js';
import { prefersReducedMotion } from './motion_preferences.js';

// ── 闪烁高亮状态（由侧边栏行点击触发，render 循环读） ────────────────
const FLASH_DURATION = 800;  // ms 总时长
const FLASH_CYCLES   = 3;    // 振荡次数（每次 = 一次明暗）
let _flashUuid  = null;
let _flashStart = 0;
let _flashRafId = null;
let _inspectionPulseRafId = null;

// 启动指定 stamp 的闪烁高亮：800ms 内 sin 震荡 3 次，期间每帧 rAF 触发重绘
export function flashStamp(uuid) {
    if (!uuid) return;
    if (_flashRafId) cancelAnimationFrame(_flashRafId);
    if (prefersReducedMotion()) {
        _flashUuid = null;
        _flashRafId = null;
        updateTransform();
        return;
    }
    _flashUuid  = uuid;
    _flashStart = performance.now();
    const tick = (now) => {
        if (prefersReducedMotion()) {
            _flashUuid = null;
            _flashRafId = null;
            updateTransform();
            return;
        }
        const elapsed = now - _flashStart;
        updateTransform();  // 重绘触发 render 循环读 _flashUuid
        if (elapsed < FLASH_DURATION) {
            _flashRafId = requestAnimationFrame(tick);
        } else {
            _flashUuid  = null;
            _flashRafId = null;
            updateTransform();  // 最终重绘清除 halo
        }
    };
    _flashRafId = requestAnimationFrame(tick);
}

// ── Stamp 颜色工具 ──────────────────────────────────────────────────────────
export function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// 解析一枚 stamp 在 UI 分类里是"重点"(跑道框)还是"普通"
// isKeyOverride: true/false 用户手动覆盖（HITL）；null/undefined 走后端默认
// 后端 canonical：is_key（跑道框形状检测）= 重点；其他一律普通
export function resolveIsKey(stamp) {
    if (typeof stamp !== 'object' || !stamp) return false;
    if (stamp.isKeyOverride === true || stamp.isKeyOverride === false) return stamp.isKeyOverride;
    return !!stamp.is_key;
}

export function getStampColor(stamp) {
    const TYPE_SOURCE = { 'key': 'geometric_anchor', 'normal': 'ocr' };
    const colors = state.typeColors || {};
    // 字符串入参（光标 / 顶部按钮等场景）：按名字/key-normal 映射直接查
    if (typeof stamp === 'string') {
        return colors[stamp] || colors[TYPE_SOURCE[stamp]] || colors['_default'] || '#6c757d';
    }
    // 1. 手动颜色覆盖（预设色 swatch 点出来的）优先级最高
    if (stamp._colorOverride) return stamp._colorOverride;
    // 2. KEY 优先：override 或后端 is_key 都视为 KEY → 蓝色
    if (resolveIsKey(stamp)) return colors['geometric_anchor'] || colors['_default'] || '#6c757d';
    // 3. 非 KEY：按 source 找颜色（支持 surface_roughness 等多类色）
    const source = stamp.source || TYPE_SOURCE[stamp.type] || '_default';
    return colors[source] || colors['ocr'] || colors['_default'] || '#6c757d';
}

function isLegacyPositionOnlyStamp(stamp) {
    if (stamp?.source !== 'legacy_stamp') return false;
    if (stamp?.legacyArrowPrecise === true || stamp?.legacy?.arrow_geometry === 'appearance_stream') {
        return false;
    }
    return stamp?.legacyPositionOnly === true
        || stamp?.dimBboxPlaceholder === true
        || stamp?.status === 'legacy_position_only';
}

// ── DOM 元素缓存 ────────────────────────────────────────────────────────────
export const els = {
    wrapper:       document.getElementById('canvas-wrapper'),
    pagesContainer:document.getElementById('pdf-pages-container'),
    drawCanvas:    document.getElementById('draw-layer'),
    drawCtx:       document.getElementById('draw-layer').getContext('2d'),
    workspace:     document.getElementById('workspace'),
    minimap:       document.getElementById('minimap'),
    miniContainer: document.getElementById('minimap-canvas-container'),
    miniViewport:  document.getElementById('minimap-viewport'),
};

// ── 防竞态渲染令牌 ──────────────────────────────────────────────────────────
let renderToken = 0;

// =============================================================================
// 公开 API
// =============================================================================

/**
 * 加载 PDF 后渲染所有页面底图及缩略图
 * @param {boolean} isInitial  true = 首次加载，重置 zoom/pan
 */
export async function renderBaseCanvases(isInitial = false) {
    if (!window.pdfDoc) return;

    els.pagesContainer.innerHTML = '';
    els.miniContainer.innerHTML  = '';
    state.pageOffsets = [];
    state.totalPages  = window.pdfDoc.numPages;

    let totalHeight = 0, maxWidth = 0;
    for (let i = 1; i <= state.totalPages; i++) {
        const page = await window.pdfDoc.getPage(i);
        const rawViewport = page.getViewport({ scale: 1.0, rotation: 0 });
        const rotation = normalizePageRotation((page.rotate || 0) + state.rotation);
        const geometry = createPageGeometry({
            pageIndex: i,
            start: totalHeight,
            sourceWidth: rawViewport.width,
            sourceHeight: rawViewport.height,
            rotation,
        });
        state.pageOffsets.push(geometry);
        totalHeight += geometry.height;
        maxWidth = Math.max(maxWidth, geometry.width);
    }
    state.pdfWidth  = maxWidth;
    state.pdfHeight = totalHeight;

    const wsRect   = els.workspace.getBoundingClientRect();
    const fitScale = (wsRect.width - 60) / maxWidth;
    let baseScale  = fitScale * 1.5;
    if (maxWidth * baseScale > CONFIG.CANVAS_MAX_WIDTH) baseScale = CONFIG.CANVAS_MAX_WIDTH / maxWidth;
    state.miniScale = Math.min(240 / maxWidth, 350 / totalHeight);

    for (let i = 1; i <= state.totalPages; i++) {
        const page       = await window.pdfDoc.getPage(i);
        const rotation   = state.pageOffsets[i - 1].rotation;
        const viewport   = page.getViewport({ scale: baseScale,  rotation });
        const cssViewport= page.getViewport({ scale: 1.0,        rotation });

        const canvas = document.createElement('canvas');
        canvas.id        = `pdf-page-${i}`;
        canvas.className = 'base-layer';
        canvas.width     = viewport.width;
        canvas.height    = viewport.height;
        canvas.style.width  = `${cssViewport.width}px`;
        canvas.style.height = `${cssViewport.height}px`;
        els.pagesContainer.appendChild(canvas);
        await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;

        // 缩略图
        const miniVp = page.getViewport({ scale: state.miniScale, rotation });
        const mCan   = document.createElement('canvas');
        mCan.width   = miniVp.width;
        mCan.height  = miniVp.height;
        els.miniContainer.appendChild(mCan);
        await page.render({ canvasContext: mCan.getContext('2d'), viewport: miniVp }).promise;
    }

    if (isInitial) {
        state.zoom = Math.max(0.25, Math.min(fitScale, 32.0));
        state.panX = (wsRect.width - maxWidth * state.zoom) / 2;
        state.panY = 30;
    } else {
        state.panX = wsRect.width  / 2 - (maxWidth     * state.zoom) / 2;
        state.panY = wsRect.height / 2 - (totalHeight  * state.zoom) / 2;
    }
    updateTransform();
}

/**
 * 高缩放级别下按视口裁剪的高分辨率叠加渲染（防竞态）
 */
export async function triggerHighResRender() {
    if (!window.pdfDoc) return;
    const myToken = ++renderToken;
    const dpr = window.devicePixelRatio || 1;
    document.querySelectorAll('.hr-overlay').forEach(el => el.remove());
    // R14.5 修糊：判定基准从 state.zoom 改成 device pixel scale (zoom × dpr)。
    //   dpr=1: 阈值仍是 zoom>=1.2（保持原行为）
    //   dpr=2 retina: zoom>=0.6 即触发，让 100%/110% 在 retina 屏不糊
    if (state.zoom * dpr < 1.2) return;

    const wsRect         = els.workspace.getBoundingClientRect();
    const MAX_AREA_PIXELS = 16777216;

    for (let i = 1; i <= state.totalPages; i++) {
        const pageCanvas = document.getElementById(`pdf-page-${i}`);
        if (!pageCanvas) continue;

        const rect      = pageCanvas.getBoundingClientRect();
        const intersectL = Math.max(rect.left,   wsRect.left   - 200);
        const intersectT = Math.max(rect.top,    wsRect.top    - 200);
        const intersectR = Math.min(rect.right,  wsRect.right  + 200);
        const intersectB = Math.min(rect.bottom, wsRect.bottom + 200);
        if (intersectL >= intersectR || intersectT >= intersectB) continue;

        const pdfL = (intersectL - rect.left) / state.zoom;
        const pdfT = (intersectT - rect.top)  / state.zoom;
        const pdfR = (intersectR - rect.left) / state.zoom;
        const pdfB = (intersectB - rect.top)  / state.zoom;
        const rectW = pdfR - pdfL, rectH = pdfB - pdfT;

        let targetPixelScale = state.zoom * dpr;
        if (rectW * rectH * targetPixelScale * targetPixelScale > MAX_AREA_PIXELS) {
            targetPixelScale = Math.sqrt(MAX_AREA_PIXELS / (rectW * rectH));
        }
        const pixelW = Math.round(rectW * targetPixelScale);
        const pixelH = Math.round(rectH * targetPixelScale);
        if (pixelW <= 0 || pixelH <= 0) continue;

        const overlay           = document.createElement('canvas');
        overlay.className       = 'hr-overlay';
        overlay.width           = pixelW;
        overlay.height          = pixelH;
        overlay.style.position  = 'absolute';
        overlay.style.pointerEvents = 'none';
        overlay.style.left      = `${pdfL}px`;
        overlay.style.top       = `${pdfT + state.pageOffsets[i - 1].start}px`;
        overlay.style.width     = `${rectW}px`;
        overlay.style.height    = `${rectH}px`;
        overlay.style.zIndex    = '10';

        const page      = await window.pdfDoc.getPage(i);
        const transform = [targetPixelScale, 0, 0, targetPixelScale,
                           -pdfL * targetPixelScale, -pdfT * targetPixelScale];
        await page.render({
            canvasContext: overlay.getContext('2d'),
            transform,
            viewport: page.getViewport({
                scale: 1.0,
                rotation: state.pageOffsets[i - 1]?.rotation ?? state.rotation,
            }),
        }).promise;

        if (renderToken !== myToken) { overlay.remove(); return; }
        els.pagesContainer.appendChild(overlay);
    }
}

/**
 * 更新 CSS transform、小地图视口、页码指示器，并触发高分辨率渲染
 */
export function updateTransform() {
    state.zoom = Math.max(0.25, Math.min(state.zoom, 32.0));
    document.getElementById('zoom-val').innerText = Math.round(state.zoom * 100) + '%';

    const wsRect  = els.workspace.getBoundingClientRect();
    // 侧栏用 position:fixed 盖在 workspace 上，getBoundingClientRect 不会扣它。
    // 展开时需手动扣 380px，否则居中算法把画布中心放在 vw/2，被侧栏遮住半边。
    const sbEl = document.getElementById('sidebar-container');
    const sbOpen = sbEl && sbEl.classList.contains('active');
    const sbWidth = sbOpen ? 380 : 0;
    const visibleW = wsRect.width - sbWidth;
    const scaledW = state.pdfWidth  * state.zoom;
    const scaledH = state.pdfHeight * state.zoom;

    const overflowX = visibleW / 3;
    const overflowY = wsRect.height / 3;
    if (scaledW <= visibleW) {
        state.panX = (visibleW - scaledW) / 2;
    } else {
        state.panX = Math.max(visibleW - scaledW - overflowX, Math.min(state.panX, overflowX));
    }
    if (scaledH <= wsRect.height) {
        state.panY = (wsRect.height - scaledH) / 2;
    } else {
        state.panY = Math.max(wsRect.height - scaledH - overflowY, Math.min(state.panY, overflowY));
    }

    els.wrapper.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    document.querySelectorAll('.hr-overlay').forEach(el => el.remove());

    syncDrawLayerResolution();

    // 平移/缩放后通知监听者（如选中章悬浮窗）重定位，避免循环依赖直接 import
    document.dispatchEvent(new CustomEvent('pdf-reader:transform-update'));

    if (state.pdfWidth) {
        const s = state.miniScale;
        els.miniViewport.style.width  = `${(wsRect.width  / state.zoom) * s}px`;
        els.miniViewport.style.height = `${(wsRect.height / state.zoom) * s}px`;
        const pdfCenterX = (-state.panX + wsRect.width  / 2) / state.zoom;
        const pdfCenterY = (-state.panY + wsRect.height / 2) / state.zoom;
        els.miniViewport.style.left = `${pdfCenterX * s}px`;
        els.miniViewport.style.top  = `${pdfCenterY * s}px`;
    }

    if (state.pageOffsets.length > 0) {
        const viewCenterY = (-state.panY / state.zoom) + (wsRect.height / state.zoom) / 2;
        let curPage = 1;
        for (let i = 0; i < state.pageOffsets.length; i++) {
            if (viewCenterY >= state.pageOffsets[i].start &&
                viewCenterY <= state.pageOffsets[i].start + state.pageOffsets[i].height) {
                curPage = i + 1; break;
            }
        }
        document.getElementById('page-indicator').innerText = `${curPage}/${state.totalPages}`;
    }

    clearTimeout(state.renderTimer);
    state.renderTimer = setTimeout(() => { triggerHighResRender(); }, 120);
}

/**
 * 重设 drawCanvas 物理分辨率（DPR 感知），然后触发全量重绘
 */
export function syncDrawLayerResolution() {
    const rect = els.workspace.getBoundingClientRect();
    const dpr  = window.devicePixelRatio || 1;
    els.drawCanvas.width  = rect.width  * dpr;
    els.drawCanvas.height = rect.height * dpr;
    els.drawCanvas.style.width  = `${rect.width}px`;
    els.drawCanvas.style.height = `${rect.height}px`;
    els.drawCtx.resetTransform();
    els.drawCtx.scale(dpr, dpr);
    repaintDrawOverlay();
}

/**
 * Repaint only the dynamic draw layer. Unlike updateTransform(), this seam
 * deliberately keeps the PDF transform, canvas resolution, HR overlays and
 * delayed high-resolution render intact.
 */
function repaintDrawOverlay() {
    const inspectionPulseActive = renderStampsCore();
    reconcileInspectionPulseFrame(inspectionPulseActive);
}

function reconcileInspectionPulseFrame(active) {
    if (prefersReducedMotion()) active = false;
    if (active) {
        if (_inspectionPulseRafId !== null) return;
        _inspectionPulseRafId = requestAnimationFrame(() => {
            _inspectionPulseRafId = null;
            repaintDrawOverlay();
        });
        return;
    }
    if (_inspectionPulseRafId !== null) {
        cancelAnimationFrame(_inspectionPulseRafId);
        _inspectionPulseRafId = null;
    }
}

/**
 * 主绘制函数。Z 序：PDF 底图（CSS transform）→ Stamps → 选中句柄 → 框选框
 * @returns {boolean} 是否需要继续 inspection NG overlay 动画
 */
function renderStampsCore() {
    const ctx = els.drawCtx;
    ctx.clearRect(0, 0, els.drawCanvas.width, els.drawCanvas.height);
    let hasInspectionNg = false;

    // ── 层 0: 视图框（蓝色虚线方框，stamps 之下）──────────────────────────
    drawViewBoxesLayer(ctx);
    drawBasicDimensionsLayer(ctx);
    drawDatumReferencesLayer(ctx);
    drawReviewCandidatesLayer(ctx);

    // ── 层 1: 已保存的 Stamps ───────────────────────────────────────────────
    state.stamps.forEach(s => {
        // S44h: 隐藏低置信度 AI 标记（已 confirmed 的不隐藏）
        if (state.hideLowConfidence
                && s.isAutoGenerated
                && !s.confirmed
                && s.aiData?.confidence === 'low') {
            return;
        }
        const color     = getStampColor(s);
        const sfs       = s.fs * state.zoom;
        const sRatio    = s.ratio || 0.60;
        const isSelected = state.selectedUUIDs.includes(s.uuid);
        const inspectionResult = _inspectionResultForStamp(s);
        const isInspectionNg = inspectionResult === 'NOK' || inspectionResult === 'NG';
        if (isInspectionNg) hasInspectionNg = true;

        // 屏幕坐标 routingPath（兼容旧存档无 routingPath 字段）
        const screenPath = resolveScreenPath(s);

        // 裁剪框（R14.4 B 路线：读 s.dimBbox，跟 arrowTip 解绑）
        //   - 选中 → 醒目色 + 句柄（可拖动）
        //   - stampActive 时，未校准的手绘 stamp 常显淡色边框
        const legacyPositionOnly = isLegacyPositionOnlyStamp(s);
        const hasRealDimBbox = !(s.source === 'legacy_stamp' && s.dimBboxPlaceholder === true);

        if (isSelected && !legacyPositionOnly && hasRealDimBbox) {
            ensureDimBbox(s);
            drawDimBbox(ctx, s.dimBbox, false);
        } else if (state.stampActive && !s.isAutoGenerated && !s.cropAdjusted) {
            ensureDimBbox(s);
            drawDimBbox(ctx, s.dimBbox, true);
        }
        if (isInspectionNg && !legacyPositionOnly && hasRealDimBbox) {
            ensureDimBbox(s);
            drawInspectionDimAlert(ctx, s.dimBbox);
        }

        // 选中态 Apple 蓝发光 halo（在主 stamp 之前画，位于下层）
        if (isSelected) {
            drawSelectionHalo(ctx, screenPath, s.id, sfs, sRatio, s.source);
        }
        if (isInspectionNg) {
            drawInspectionNgHalo(ctx, screenPath, s.id, sfs, sRatio, s.source);
        }

        // 闪烁 halo（侧边栏点击行 → 定位后的 3 次脉冲高亮，金色，置于选中 halo 之上）
        if (s.uuid === _flashUuid && !prefersReducedMotion()) {
            const elapsed = performance.now() - _flashStart;
            if (elapsed < FLASH_DURATION) {
                // alpha 跟随 sin 波在 [0, 1] 间震荡 FLASH_CYCLES 次
                const phase = (elapsed / FLASH_DURATION) * Math.PI * FLASH_CYCLES;
                const alpha = Math.abs(Math.sin(phase));
                drawFlashHalo(ctx, screenPath, s.id, sfs, sRatio, s.source, alpha);
            }
        }

        if (s.type === 'note') {
            drawStampShape(ctx, screenPath[0].x, screenPath[0].y, s.id, color, sfs, sRatio, s.confirmed);
            return;
        }

        // 表面粗糙度：只画圆角矩形 + 文字，跳过序号/箭头
        if (s.source === 'surface_roughness' && s.crop) {
            const ax = s.arrowTip.absX * state.zoom + state.panX;
            const ay = s.arrowTip.absY * state.zoom + state.panY;
            const cl = s.crop.l * state.zoom, cr = s.crop.r * state.zoom;
            const ct = s.crop.t * state.zoom, cb = s.crop.b * state.zoom;
            const rx = ax + cl, ry = ay + ct;
            const rw = cr - cl, rh = cb - ct;
            const rad = 4 * state.zoom;
            ctx.save();
            ctx.globalAlpha = 0.25;
            ctx.fillStyle = '#C9A94A';
            ctx.beginPath();
            ctx.roundRect(rx, ry, rw, rh, rad);
            ctx.fill();
            ctx.globalAlpha = 1.0;
            ctx.strokeStyle = '#A68A35';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = '#666';
            ctx.font = `${10 * state.zoom}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(s.label || s.aiData?.nominal || '', rx + rw / 2, ry + rh / 2);
            ctx.restore();
        } else {
        // 折线 + 箭头 + 气泡
        if (legacyPositionOnly) {
            drawLegacyPositionStamp(ctx, screenPath, s.id, color, sfs, sRatio, isSelected);
        } else {
            drawSingleStamp(ctx, screenPath, s.id, color, sfs, sRatio, s.confirmed);
        }
        // HITL #4: 低置信度视觉警告（AI 章 + confidence=low + 未 confirmed）
        //             橙色 dashed 外圈，提示用户重点复核
        if (!s.confirmed && s.isAutoGenerated && s.aiData?.confidence === 'low') {
            drawLowConfRing(ctx, screenPath, s.id, sfs, sRatio);
        }
        // HITL #8: OCR 失败红点（手绘章 + 已请求但 nominal 仍空）
        //             右上角 3pt 红点标识"后端未能识别任何文本"
        if (!s.isAutoGenerated && s.ocrRequested && !s.aiData?.nominal) {
            drawOcrFailDot(ctx, screenPath, s.id, sfs, sRatio);
        }
        if (isInspectionNg) {
            drawNgBadge(ctx, screenPath, s.id, sfs, sRatio);
        }
        }

        // 选中句柄（指针工具下常驻，与 cropBox 开关无关）
        if (!legacyPositionOnly && !state.stampActive && state.baseTool === 'pointer' && isSelected) {
            const sx1 = screenPath[0].x, sy1 = screenPath[0].y;
            const sx2 = screenPath[screenPath.length - 1].x;
            const sy2 = screenPath[screenPath.length - 1].y;
            drawSelectionHandles(ctx, sx1, sy1, sx2, sy2, s.id, sfs, sRatio);
        }
    });

    // ── 层 3: 手动绘制预览（stampActive 模式中的临时图章）──────────────────
    if (state.stampActive && state.tempP1) {
        const tfs    = parseInt(document.getElementById('default-font').value) * state.zoom;
        const tRatio = parseFloat(document.getElementById('shape-ratio').value) || 0.60;
        const _tc = getStampColor(state.tempType || 'ocr');
        const tColor = hexToRgba(_tc, 0.75);
        const pt = state.lastMousePt || state.tempP1;
        // 判"鼠标是否真正离开了第一点"：PDF 坐标系距离 > 1pt 才显示完整橡皮筋，
        // 否则（刚落下、未移动）只画 tempP1 位置的气泡 + 十字定位标记
        const dMove = Math.hypot(pt.x - state.tempP1.x, pt.y - state.tempP1.y);
        const showFullPreview = state.drawPhase === 2 || (state.drawPhase === 1 && dMove > 1);

        if (showFullPreview) {
            const drawOrder = state.drawOrder || 'arrow_first';
            let circlePt, arrowPt;
            if (drawOrder === 'arrow_first') { circlePt = pt; arrowPt = state.tempP1; }
            else                              { circlePt = state.tempP1; arrowPt = pt; }

            // 橡皮筋预览：直线（2 点 screenPath）
            const screenPath = [
                { x: circlePt.x * state.zoom + state.panX, y: circlePt.y * state.zoom + state.panY },
                { x: arrowPt.x  * state.zoom + state.panX, y: arrowPt.y  * state.zoom + state.panY },
            ];
            drawSingleStamp(ctx, screenPath,
                document.getElementById('current-id').value, tColor, tfs, tRatio, false);
        } else {
            // drawPhase===1 刚落下、尚未移动鼠标：画 tempP1 位置气泡 + 同色十字定位
            const sx1 = state.tempP1.x * state.zoom + state.panX;
            const sy1 = state.tempP1.y * state.zoom + state.panY;
            drawStampShape(ctx, sx1, sy1,
                document.getElementById('current-id').value, tColor, tfs, tRatio);
            ctx.save();
            ctx.strokeStyle = _tc;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(sx1 - 10, sy1); ctx.lineTo(sx1 + 10, sy1);
            ctx.moveTo(sx1, sy1 - 10); ctx.lineTo(sx1, sy1 + 10);
            ctx.stroke();
            ctx.restore();
        }
    }

    // ── 层 4: 框选矩形 ───────────────────────────────────────────────────────
    if (state.selectionBox) {
        const sx = state.selectionBox.startX * state.zoom + state.panX;
        const sy = state.selectionBox.startY * state.zoom + state.panY;
        const ex = state.selectionBox.endX   * state.zoom + state.panX;
        const ey = state.selectionBox.endY   * state.zoom + state.panY;
        ctx.fillStyle   = 'rgba(10, 132, 255, 0.15)';
        ctx.strokeStyle = '#0A84FF';
        ctx.lineWidth   = 1;
        ctx.fillRect(sx, sy, ex - sx, ey - sy);
        ctx.strokeRect(sx, sy, ex - sx, ey - sy);
    }

    return hasInspectionNg
        && !!state.inspection?.active
        && !prefersReducedMotion();
}

function _linearReviewCandidateLabel(entry) {
    const suffix = entry.unit === 'degree' ? '°' : '';
    const multiplier = entry.quantity > 1 ? `${entry.quantity}X ` : '';
    const main = `${multiplier}${entry.prefix || ''}${entry.nominal}${suffix}`;
    if (entry.symmetric_tolerance !== null) {
        return `${main} ±${entry.symmetric_tolerance}${suffix}`;
    }
    if (
        entry.upper_tolerance !== null
        && entry.lower_tolerance !== null
    ) {
        return `${main} ${entry.upper_tolerance}${suffix}/${entry.lower_tolerance}${suffix}`;
    }
    return main;
}

function drawReviewCandidatesLayer(ctx) {
    const entries = listReviewCandidateEntries(state.reviewWorkspace).filter(
        entry => entry.status === 'pending' || entry.status === 'edited',
    );
    if (!entries.length) return;
    ctx.save();
    ctx.font = 'bold 11px sans-serif';
    ctx.textBaseline = 'bottom';
    for (const entry of entries) {
        let bbox;
        try {
            const page = getPageGeometry(
                state.pageOffsets,
                entry.envelope.page_index + 1,
            );
            bbox = pageLocalBboxToDocumentGlobal(entry.item.bbox, page);
        } catch (_error) {
            continue;
        }
        const selected = state.reviewWorkspace.selectedCandidateId
            === entry.item.candidate_id;
        const sx = bbox.x * state.zoom + state.panX;
        const sy = bbox.y * state.zoom + state.panY;
        const sw = bbox.w * state.zoom;
        const sh = bbox.h * state.zoom;
        if (![sx, sy, sw, sh].every(Number.isFinite) || sw <= 0 || sh <= 0) continue;
        ctx.strokeStyle = selected ? '#8a4208' : '#d97706';
        ctx.fillStyle = selected ? '#6f3506' : '#92400e';
        ctx.lineWidth = selected ? 2.5 : 1.5;
        ctx.setLineDash(selected ? [7, 3] : [4, 3]);
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.setLineDash([]);
        const label = entry.type === 'gdt'
            ? `${entry.symbol_unicode} ${entry.tolerance_value}`
            : _linearReviewCandidateLabel(entry);
        ctx.fillText(
            label,
            sx,
            Math.max(10, sy - 3),
        );
    }
    ctx.restore();
}

/**
 * 将 PDF 坐标系 routingPath 转换为屏幕坐标数组
 * 若 Stamp 无 routingPath（旧存档兼容），退化为 [circleCenter, arrowTip] 直线
 * @param {Stamp} s
 * @returns {Array<{x,y}>}  屏幕坐标点数组（2 或 3 点）
 */
export function resolveScreenPath(s) {
    if (s?.type === 'note' && s.circleCenter) {
        return [{
            x: s.circleCenter.absX * state.zoom + state.panX,
            y: s.circleCenter.absY * state.zoom + state.panY,
        }];
    }
    const path = (s.routingPath && s.routingPath.length >= 2)
        ? s.routingPath
        : [s.circleCenter, s.arrowTip];

    return path.map(p => ({
        x: p.absX * state.zoom + state.panX,
        y: p.absY * state.zoom + state.panY,
    }));
}

/**
 * 将鼠标事件坐标转换为 PDF 物理坐标（全局连续坐标系）
 */
export function getCanvasCoords(e) {
    const rect = els.workspace.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left - state.panX) / state.zoom,
        y: (e.clientY - rect.top  - state.panY) / state.zoom,
    };
}

// =============================================================================
// 私有绘制辅助
// =============================================================================

/**
 * 绘制所有视图框（蓝色虚线方框 + 8 个角/边句柄）
 * 轴对齐矩形，坐标系为 PDF 全局坐标（含 pageOffset）
 */
function drawViewBoxesLayer(ctx) {
    if (!state.viewBoxes || state.viewBoxes.length === 0) return;

    ctx.save();
    const FILL = 'rgba(10, 132, 255, 0.04)';
    const EDGE = 'rgba(10, 132, 255, 0.50)';
    const dragVbId = (state.dragCache && state.dragCache.viewBoxId) || null;

    state.viewBoxes.forEach((vb, idx) => {
        const sx = vb.x * state.zoom + state.panX;
        const sy = vb.y * state.zoom + state.panY;
        const sw = vb.w * state.zoom;
        const sh = vb.h * state.zoom;
        const shape = vb.shape || 'rect';

        // 标签 V{idx+1} 在矩形/圆形分支末尾各自定位
        const label = `V${idx + 1}`;
        ctx.font         = '11px -apple-system, Arial, sans-serif';
        ctx.textBaseline = 'top';
        ctx.textAlign    = 'left';
        const pad = 3;
        const tw = ctx.measureText(label).width;

        if (shape === 'circle' && typeof vb.cx === 'number'
                && typeof vb.cy === 'number' && typeof vb.r === 'number') {
            // R14.5 A3：圆形视图分支 — 蓝虚线圆边 + 淡蓝填充 + 标签放圆顶
            const ccx = vb.cx * state.zoom + state.panX;
            const ccy = vb.cy * state.zoom + state.panY;
            const cr  = Math.max(1, vb.r * state.zoom);

            ctx.fillStyle = FILL;
            ctx.beginPath();
            ctx.arc(ccx, ccy, cr, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = EDGE;
            ctx.lineWidth   = 1.0;
            ctx.setLineDash([8, 4]);
            ctx.beginPath();
            ctx.arc(ccx, ccy, cr, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // 标签居中放圆顶（cx, cy-r 上方）
            const lx = ccx - (tw + pad * 2) / 2;
            const ly = ccy - cr - 18;
            ctx.fillStyle = 'rgba(10, 132, 255, 0.55)';
            ctx.fillRect(lx, ly, tw + pad * 2, 15);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, lx + pad, ly + 2);
        } else {
            // 矩形分支（默认）
            ctx.fillStyle = FILL;
            ctx.fillRect(sx, sy, sw, sh);

            ctx.strokeStyle = EDGE;
            ctx.lineWidth   = 1.0;
            ctx.setLineDash([8, 4]);
            ctx.strokeRect(sx, sy, sw, sh);
            ctx.setLineDash([]);

            ctx.fillStyle = 'rgba(10, 132, 255, 0.55)';
            ctx.fillRect(sx, sy, tw + pad * 2, 15);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, sx + pad, sy + 2);
        }

        // R14.5 A1：8 句柄改 hover/拖拽时显示（避免永久 if (false) 死代码）。
        // 命中条件：当前 hoverViewBoxId 命中 vb，或 vb 正在被 vbox_ 拖拽时常亮。
        // 圆形 vb 不画 8 句柄（圆形 resize 不支持）；矩形 vb 才画。
        const showHandles = shape === 'rect'
            && state.baseTool === 'pointer'
            && !state.stampActive
            && (state.hoverViewBoxId === vb.id || dragVbId === vb.id);
        if (showHandles) {
            const cx = sx + sw / 2;
            const cy = sy + sh / 2;
            const handles = [
                { x: sx,      y: sy      },
                { x: cx,      y: sy      },
                { x: sx + sw, y: sy      },
                { x: sx,      y: cy      },
                { x: sx + sw, y: cy      },
                { x: sx,      y: sy + sh },
                { x: cx,      y: sy + sh },
                { x: sx + sw, y: sy + sh },
            ];
            ctx.fillStyle   = '#ffffff';
            ctx.strokeStyle = '#0A84FF';
            ctx.lineWidth   = 1.5;
            handles.forEach(p => {
                ctx.fillRect(p.x - 4, p.y - 4, 8, 8);
                ctx.strokeRect(p.x - 4, p.y - 4, 8, 8);
            });
        }
    });
    ctx.restore();
}

function _drawReferenceOverlayBox(ctx, item, style) {
    const b = item?.bbox;
    if (!b) return;
    const sx = b.x * state.zoom + state.panX;
    const sy = b.y * state.zoom + state.panY;
    const sw = b.w * state.zoom;
    const sh = b.h * state.zoom;
    if (![sx, sy, sw, sh].every(Number.isFinite) || sw <= 1 || sh <= 1) return;

    ctx.save();
    ctx.fillStyle = style.fill;
    ctx.strokeStyle = style.edge;
    ctx.lineWidth = 1.5;
    const radius = Math.min(4, Math.max(1, Math.min(sw, sh) * 0.18));
    ctx.beginPath();
    ctx.roundRect(sx, sy, sw, sh, radius);
    ctx.fill();
    ctx.stroke();

    if (style.label && state.zoom >= 0.45) {
        ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif';
        ctx.textBaseline = 'top';
        ctx.textAlign = 'left';
        const pad = 3;
        const tw = ctx.measureText(style.label).width;
        const th = 14;
        const lx = sx;
        const ly = Math.max(2, sy - th - 2);
        ctx.fillStyle = style.tag;
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw + pad * 2, th, 3);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.fillText(style.label, lx + pad, ly + 2);
    }
    ctx.restore();
}

function drawBasicDimensionsLayer(ctx) {
    if (!state.basicDimensions || state.basicDimensions.length === 0) return;
    for (const item of state.basicDimensions) {
        _drawReferenceOverlayBox(ctx, item, {
            label: '理论',
            fill: 'rgba(14, 165, 233, 0.06)',
            edge: 'rgba(2, 132, 199, 0.92)',
            tag: 'rgba(2, 132, 199, 0.86)',
        });
    }
}

function drawDatumReferencesLayer(ctx) {
    if (!state.datumReferences || state.datumReferences.length === 0) return;
    for (const item of state.datumReferences) {
        _drawReferenceOverlayBox(ctx, item, {
            label: '基准',
            fill: 'rgba(245, 158, 11, 0.06)',
            edge: 'rgba(217, 119, 6, 0.92)',
            tag: 'rgba(217, 119, 6, 0.86)',
        });
    }
}

/**
 * 绘制 dim 识别框（amber 色，圆角矩形）
 *   R14.4 B 路线：参数从 (sx2, sy2, crop) 改成 dimBbox 绝对坐标
 *   形状从直角矩形改成小圆角，更柔和
 * @param {{x,y,w,h}} dimBbox  PDF 坐标系绝对位置
 * @param {boolean}   faded    true = 淡色轮廓（stampActive 常显）；false = 实心 + 8 句柄（选中态）
 */
function drawDimBbox(ctx, dimBbox, faded = false) {
    if (!dimBbox || dimBbox.w <= 0 || dimBbox.h <= 0) return;
    const l  = dimBbox.x * state.zoom + state.panX;
    const t  = dimBbox.y * state.zoom + state.panY;
    const w  = dimBbox.w * state.zoom;
    const h  = dimBbox.h * state.zoom;
    const r  = l + w;
    const b  = t + h;
    const cx = l + w / 2;
    const cy = t + h / 2;
    // 小圆角：6pt 基准（R14.5 加大避免被句柄遮蔽），受框尺寸限制（极小框退化成直角）
    const rad = Math.min(6 * state.zoom, w / 4, h / 4, 12);

    ctx.save();
    if (faded) {
        ctx.fillStyle   = 'rgba(255, 193, 7, 0.05)';
        ctx.beginPath(); ctx.roundRect(l, t, w, h, rad); ctx.fill();
        ctx.strokeStyle = 'rgba(255, 193, 7, 0.55)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.roundRect(l, t, w, h, rad); ctx.stroke();
        ctx.setLineDash([]);
    } else {
        ctx.fillStyle   = 'rgba(255, 193, 7, 0.15)';
        ctx.beginPath(); ctx.roundRect(l, t, w, h, rad); ctx.fill();
        ctx.strokeStyle = '#ffc107';
        ctx.lineWidth   = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath(); ctx.roundRect(l, t, w, h, rad); ctx.stroke();
        ctx.setLineDash([]);

        // R14.5：4 corner 改 6×6 白底黄描边（让圆角顶部露出）；
        //        4 edge mid 保持 8×8 实心黄（拖拽热区清晰）
        const edges = [
            { x: cx, y: t  },
            { x: l,  y: cy }, { x: r,  y: cy },
            { x: cx, y: b  },
        ];
        const corners = [
            { x: l, y: t }, { x: r, y: t },
            { x: l, y: b }, { x: r, y: b },
        ];
        ctx.fillStyle   = '#ffc107';
        ctx.strokeStyle = '#ffc107';
        ctx.lineWidth   = 1;
        edges.forEach(p => {
            ctx.fillRect(p.x - 4, p.y - 4, 8, 8);
            ctx.strokeRect(p.x - 4, p.y - 4, 8, 8);
        });
        ctx.fillStyle = '#fff';
        ctx.lineWidth = 1.5;
        corners.forEach(p => {
            ctx.fillRect(p.x - 3, p.y - 3, 6, 6);
            ctx.strokeRect(p.x - 3, p.y - 3, 6, 6);
        });
    }
    ctx.restore();
}

/**
 * 绘制选中句柄：气泡 AABB 虚线框 + 四角点 + arrowTip 控制点
 */
function drawSelectionHandles(ctx, sx, sy, tipX, tipY, text, sfs, ratio) {
    const r      = sfs * ratio;
    const idLen  = text.toString().length;
    const extraW = idLen <= 1 ? 0 : (idLen - 1) * sfs * (ratio * 0.5);
    const pad    = 4;
    const left   = sx - extraW - r - pad;
    const right  = sx + extraW + r + pad;
    const top    = sy - r - pad;
    const bottom = sy + r + pad;

    ctx.strokeStyle = '#0A84FF';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(left, top, right - left, bottom - top);
    ctx.setLineDash([]);

    // 四角 + arrowTip 句柄
    ctx.fillStyle   = 'white';
    ctx.strokeStyle = '#0A84FF';
    ctx.lineWidth   = 2;
    [
        { x: left, y: top }, { x: right, y: top },
        { x: left, y: bottom }, { x: right, y: bottom },
        { x: tipX, y: tipY },
    ].forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });
}

/**
 * 绘制完整 Stamp：routingPath 折线 + 箭头（最后线段方向）+ 气泡序号
 * 箭头角度按 routingPath 最后线段计算（TDD 6.4）
 *
 * @param {ctx}              CanvasRenderingContext2D
 * @param {Array<{x,y}>}     screenPath  屏幕坐标路径（2 或 3 点）
 * @param {string}           text        序号文本
 * @param {string}           colorStr    CSS 颜色
 * @param {number}           fs          字号（已含 zoom）
 * @param {number}           ratio       气泡比例系数
 */
function drawSingleStamp(ctx, screenPath, text, colorStr, fs, ratio, confirmed) {
    // 最后线段：用于计算箭头方向
    const tip  = screenPath[screenPath.length - 1];
    const prev = screenPath[screenPath.length - 2];
    const angle  = Math.atan2(tip.y - prev.y, tip.x - prev.x);
    // 引线粗细收敛到大厂工业制图惯例：~0.6pt（fs/10），绝不超过 fs/7 的 1pt 以免粗糙感
    const lineW  = Math.max(0.8, Math.min(fs / 10, 2.2));
    const arrowL = Math.max(8, Math.min(fs * 0.55 + 3, 26));

    // 折线引线（最后线段缩短，为箭头留位）
    const retractX = tip.x - (arrowL * 0.85) * Math.cos(angle);
    const retractY = tip.y - (arrowL * 0.85) * Math.sin(angle);

    ctx.strokeStyle = colorStr;
    ctx.lineWidth   = lineW;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';
    ctx.beginPath();
    ctx.moveTo(screenPath[0].x, screenPath[0].y);
    for (let i = 1; i < screenPath.length - 1; i++) {
        ctx.lineTo(screenPath[i].x, screenPath[i].y);
    }
    ctx.lineTo(retractX, retractY);
    ctx.stroke();

    // 实心三角箭头
    ctx.fillStyle = colorStr;
    ctx.beginPath();
    ctx.moveTo(tip.x, tip.y);
    ctx.lineTo(tip.x - arrowL * Math.cos(angle - Math.PI / 6),
               tip.y - arrowL * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(tip.x - arrowL * Math.cos(angle + Math.PI / 6),
               tip.y - arrowL * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    // 气泡序号（转发 confirmed 状态）
    drawStampShape(ctx, screenPath[0].x, screenPath[0].y, text, colorStr, fs, ratio, confirmed);
}

function drawLegacyPositionStamp(ctx, screenPath, text, colorStr, fs, ratio, selected = false) {
    const origin = screenPath[0];
    const tip = screenPath[screenPath.length - 1];

    ctx.save();
    drawStampShape(ctx, origin.x, origin.y, text, colorStr, fs, ratio, selected);

    if (tip && selected) {
        ctx.fillStyle = 'rgba(255, 193, 7, 0.85)';
        ctx.strokeStyle = 'rgba(90, 70, 20, 0.65)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, Math.max(2.5, Math.min(5, fs * 0.18)), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    ctx.restore();
}

/**
 * 绘制序号气泡（圆形或胶囊形）+ 数字
 */
/**
 * 选中状态 halo：在气泡外圈画 Apple 蓝发光环（shadowBlur 产生光晕 + 略大的描边轮廓）
 * 参数与 drawStampShape 的气泡几何保持一致，视觉效果 = "气泡外围一圈蓝色发光"
 */
function drawSelectionHalo(ctx, screenPath, text, fs, ratio, source) {
    if (source === 'surface_roughness') return;  // 粗糙度走 crop 矩形的 cropBox 虚线，无气泡
    const x = screenPath[0].x;
    const y = screenPath[0].y;
    const len    = (text || '').toString().length;
    const r      = fs * ratio;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * (ratio * 0.5);

    ctx.save();
    ctx.shadowColor = 'rgba(10, 132, 255, 0.95)';
    ctx.shadowBlur  = Math.max(14, fs * 0.9);
    ctx.strokeStyle = 'rgba(10, 132, 255, 0.90)';
    ctx.lineWidth   = Math.max(2.5, fs / 6);

    ctx.beginPath();
    if (len <= 1) {
        ctx.arc(x, y, r + 3, 0, Math.PI * 2);
    } else {
        ctx.arc(x - extraW, y, r + 3, Math.PI / 2,  Math.PI * 1.5);
        ctx.arc(x + extraW, y, r + 3, Math.PI * 1.5, Math.PI / 2);
        ctx.closePath();
    }
    ctx.stroke();

    // 引线尾段的淡光效（从气泡边缘向外 8pt 长光晕）
    if (screenPath.length >= 2) {
        const p0 = screenPath[0];
        const p1 = screenPath[1];
        const dx = p1.x - p0.x, dy = p1.y - p0.y;
        const len2 = Math.hypot(dx, dy);
        if (len2 > 0.001) {
            const ux = dx / len2, uy = dy / len2;
            const startX = p0.x + ux * (r + 4);
            const startY = p0.y + uy * (r + 4);
            const endX   = startX + ux * Math.min(18, len2 - r - 4);
            const endY   = startY + uy * Math.min(18, len2 - r - 4);
            ctx.shadowBlur  = 10;
            ctx.lineWidth   = Math.max(1.5, fs / 9);
            ctx.strokeStyle = 'rgba(10, 132, 255, 0.55)';
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
            ctx.stroke();
        }
    }
    ctx.restore();
}

/**
 * 闪烁 halo：金色亮环叠在选中 halo 之上，alpha 由调用方传入的相位决定。
 * 几何参考 drawSelectionHalo 但半径再大 3pt + 无引线光晕（只脉冲章本体）
 */
function drawFlashHalo(ctx, screenPath, text, fs, ratio, source, alpha) {
    if (source === 'surface_roughness') return;
    if (alpha <= 0.01) return;
    const x = screenPath[0].x;
    const y = screenPath[0].y;
    const len    = (text || '').toString().length;
    const r      = fs * ratio;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * (ratio * 0.5);

    ctx.save();
    ctx.shadowColor = `rgba(255, 200, 64, ${alpha})`;
    ctx.shadowBlur  = Math.max(18, fs * 1.2) * alpha;
    ctx.strokeStyle = `rgba(255, 200, 64, ${alpha})`;
    ctx.lineWidth   = Math.max(3, fs / 5);

    ctx.beginPath();
    if (len <= 1) {
        ctx.arc(x, y, r + 6, 0, Math.PI * 2);
    } else {
        ctx.arc(x - extraW, y, r + 6, Math.PI / 2,  Math.PI * 1.5);
        ctx.arc(x + extraW, y, r + 6, Math.PI * 1.5, Math.PI / 2);
        ctx.closePath();
    }
    ctx.stroke();
    ctx.restore();
}

function _inspectionResultForStamp(stamp) {
    const info = state.inspection?.byStampUuid?.[stamp.uuid] || stamp.inspection || null;
    return String(info?.result || '').trim().toUpperCase();
}

function drawInspectionNgHalo(ctx, screenPath, text, fs, ratio, source) {
    if (source === 'surface_roughness') return;
    const x = screenPath[0].x;
    const y = screenPath[0].y;
    const len    = (text || '').toString().length;
    const r      = fs * ratio;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * (ratio * 0.5);
    const pulse = prefersReducedMotion()
        ? 0.65
        : 0.55 + Math.sin(performance.now() / 260) * 0.20;

    ctx.save();
    ctx.shadowColor = `rgba(217, 48, 37, ${pulse})`;
    ctx.shadowBlur  = Math.max(14, fs * 1.0);
    ctx.strokeStyle = `rgba(217, 48, 37, ${Math.min(0.95, pulse + 0.2)})`;
    ctx.lineWidth   = Math.max(2.5, fs / 6);
    ctx.setLineDash([Math.max(5, fs * 0.25), Math.max(3, fs * 0.16)]);
    ctx.beginPath();
    if (len <= 1) {
        ctx.arc(x, y, r + 10, 0, Math.PI * 2);
    } else {
        ctx.arc(x - extraW, y, r + 10, Math.PI / 2,  Math.PI * 1.5);
        ctx.arc(x + extraW, y, r + 10, Math.PI * 1.5, Math.PI / 2);
        ctx.closePath();
    }
    ctx.stroke();
    ctx.restore();
}

function drawNgBadge(ctx, screenPath, text, fs, ratio) {
    const x = screenPath[0].x;
    const y = screenPath[0].y;
    const len    = (text || '').toString().length;
    const r      = fs * ratio;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * (ratio * 0.5);
    const bw = Math.max(22, fs * 1.35);
    const bh = Math.max(13, fs * 0.72);
    const bx = x + extraW + r * 0.62;
    const by = y - r - bh * 0.55;
    const br = Math.min(6, bh / 2);

    ctx.save();
    ctx.fillStyle = '#D93025';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = Math.max(1.2, fs / 18);
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, br);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#FFFFFF';
    ctx.font = `800 ${Math.max(9, fs * 0.42)}px -apple-system, "Segoe UI", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('NG', bx + bw / 2, by + bh / 2 + 0.5);
    ctx.restore();
}

function drawInspectionDimAlert(ctx, dimBbox) {
    if (!dimBbox || dimBbox.w <= 0 || dimBbox.h <= 0) return;
    const l = dimBbox.x * state.zoom + state.panX;
    const t = dimBbox.y * state.zoom + state.panY;
    const w = dimBbox.w * state.zoom;
    const h = dimBbox.h * state.zoom;
    const rad = Math.min(7 * state.zoom, w / 4, h / 4, 14);
    const pulse = prefersReducedMotion()
        ? 0.55
        : 0.45 + Math.sin(performance.now() / 260) * 0.18;

    ctx.save();
    ctx.shadowColor = `rgba(217,48,37,${pulse})`;
    ctx.shadowBlur = Math.max(8, 8 * state.zoom);
    ctx.strokeStyle = `rgba(217,48,37,${Math.min(0.95, pulse + 0.25)})`;
    ctx.lineWidth = Math.max(2, 2 * state.zoom);
    ctx.setLineDash([Math.max(6, 5 * state.zoom), Math.max(4, 3 * state.zoom)]);
    ctx.beginPath();
    ctx.roundRect(l, t, w, h, rad);
    ctx.stroke();
    ctx.restore();
}

/**
 * HITL #4: 低置信度橙色 dashed 外圈
 * 半径 r+7：推到选中蓝 halo（r+3 + shadowBlur）外侧避免被蓝光覆盖；
 * 闪烁金环在 r+6 且只是瞬时脉冲，不会和常显橙环长期同位。
 */
function drawLowConfRing(ctx, screenPath, text, fs, ratio) {
    const x = screenPath[0].x;
    const y = screenPath[0].y;
    const len    = (text || '').toString().length;
    const r      = fs * ratio;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * (ratio * 0.5);
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 145, 0, 0.9)';
    ctx.lineWidth   = Math.max(1.8, fs / 9);
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    if (len <= 1) {
        ctx.arc(x, y, r + 7, 0, Math.PI * 2);
    } else {
        ctx.arc(x - extraW, y, r + 7, Math.PI / 2,  Math.PI * 1.5);
        ctx.arc(x + extraW, y, r + 7, Math.PI * 1.5, Math.PI / 2);
        ctx.closePath();
    }
    ctx.stroke();
    ctx.restore();
}

/**
 * HITL #8: OCR 失败红点（右上角小圆）
 * 手绘章 /analyze_stamp 回空 nominal 时点亮，告诉用户"后端没识别出文字需手填"
 */
function drawOcrFailDot(ctx, screenPath, text, fs, ratio) {
    const x = screenPath[0].x;
    const y = screenPath[0].y;
    const len    = (text || '').toString().length;
    const r      = fs * ratio;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * (ratio * 0.5);
    // 右上角锚点：气泡右缘 + 顶部
    const dotX = x + extraW + r * 0.7;
    const dotY = y - r * 0.9;
    const dotR = Math.max(2.5, fs * 0.18);
    ctx.save();
    ctx.fillStyle   = '#D93025';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth   = Math.max(1, fs / 22);
    ctx.beginPath();
    ctx.arc(dotX, dotY, dotR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}

function drawStampShape(ctx, x, y, text, colorStr, fs, ratio, confirmed) {
    ctx.save();
    ctx.strokeStyle = colorStr;
    // 气泡描边：未确认用 1.2pt 虚线（Figma 草稿感），已确认用 1.4pt 实线
    ctx.lineWidth   = Math.max(1, Math.min(fs / 9, 2.2));
    // 字体栈改 SF Pro + 系统回退 + 数字等宽（SF Pro 内建 tnum 默认启用）
    // Canvas 对 "font-variant-numeric" 不可配，但 SF Pro / Menlo 等等宽数字体默认对齐
    ctx.font        = `600 ${fs}px -apple-system, "SF Pro Display", "SF Pro Text", "Segoe UI", "PingFang SC", system-ui, sans-serif`;
    ctx.textAlign   = 'center';
    ctx.textBaseline= 'middle';

    const len    = text.toString().length;
    const r      = fs * ratio;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * (ratio * 0.5);

    // 已确认气泡：底部淡阴影（Apple 浮凸）
    if (confirmed) {
        ctx.shadowColor = 'rgba(15,17,22,0.18)';
        ctx.shadowBlur  = Math.max(4, fs * 0.25);
        ctx.shadowOffsetY = Math.max(1, fs * 0.08);
    }

    ctx.beginPath();
    if (len <= 1) {
        ctx.arc(x, y, r, 0, Math.PI * 2);
    } else {
        ctx.arc(x - extraW, y, r, Math.PI / 2,   Math.PI * 1.5);
        ctx.arc(x + extraW, y, r, Math.PI * 1.5, Math.PI / 2);
        ctx.closePath();
    }
    // 填充：已确认纯白、未确认半透明（微暗示"未定稿"）
    ctx.fillStyle = confirmed ? '#ffffff' : 'rgba(255,255,255,0.92)';
    ctx.fill();

    // 关闭阴影（只给填充用），描边单独画
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    // 未确认用 dashed 描边（Figma 草稿感），已确认用实线
    if (!confirmed) {
        ctx.setLineDash([Math.max(2, fs * 0.18), Math.max(1.5, fs * 0.12)]);
    } else {
        ctx.setLineDash([]);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // 气泡内数字：已确认的颜色更深、未确认的带透明度
    ctx.fillStyle = confirmed ? colorStr : hexToRgba(colorStr, 0.85);
    // 视觉中心修正：不用 textBaseline='middle' 的几何中心——它对数字/字母偏上，
    // 改用 alphabetic + 实测 actualBoundingBox 把字形外包框中心对到气泡圆心 y。
    // 好处是字体/字符组合切换（半角数字 vs "A-3" 带连字符等）都能自适应。
    const textStr = String(text);
    ctx.textBaseline = 'alphabetic';
    const tm = ctx.measureText(textStr);
    const ascent  = (tm.actualBoundingBoxAscent  != null) ? tm.actualBoundingBoxAscent  : fs * 0.72;
    const descent = (tm.actualBoundingBoxDescent != null) ? tm.actualBoundingBoxDescent : fs * 0.20;
    // alphabetic baseline 画在 y'，字形中心 = y' - (ascent - descent) / 2
    // 要让字形中心 = y（气泡圆心），解得 y' = y + (ascent - descent) / 2
    ctx.fillText(textStr, x, y + (ascent - descent) / 2);
    ctx.restore();
}
