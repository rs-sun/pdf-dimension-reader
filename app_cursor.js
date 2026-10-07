// app_cursor.js — 动态光标 + 工具栏同步

import { state } from './state.js';
import { els, getStampColor } from './canvas_renderer.js';

// 颜色切换按钮：单击在 普通(红) ↔ 重点(蓝) 之间切，颜色 = 下一笔 stamp 颜色
export function _syncColorToggleBtn() {
    const btn = document.getElementById('global-color-toggle');
    if (!btn) return;
    const isKey = state.tempType === 'key';
    const c = isKey
        ? ((state.typeColors && state.typeColors.geometric_anchor) || '#2E5AAE')
        : ((state.typeColors && state.typeColors.ocr)              || '#C23340');
    btn.style.background = c;
    btn.style.borderColor = c;
    btn.classList.toggle('key', isKey);
    btn.classList.toggle('normal', !isKey);
    btn.title = isKey ? '当前：重点(蓝) — 点击切换为普通(红)'
                       : '当前：普通(红) — 点击切换为重点(蓝)';
}

// 光标 SVG（URL-encoded data URI），颜色跟即将落下的章同色：
// - 按住 Shift 或已提交 tempType='key' → 用 geometric_anchor 色（默认蓝 #4C6FB3）
// - 其他（普通章）                      → 用 ocr 色（默认红 #B85C63）
// 红叉 = 下一笔落"箭头尖"；气泡 = 下一笔落"圆心"。热点都放在图形几何中心。
const _cursorCache = new Map();

export function _makeArrowCursor(hex) {
    const k = 'x|' + hex;
    let v = _cursorCache.get(k);
    if (v) return v;
    const c = hex.replace('#', '%23');
    v = "url(\"data:image/svg+xml;utf8," +
        "<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'>" +
        `<line x1='4' y1='4' x2='20' y2='20' stroke='${c}' stroke-width='3.2' stroke-linecap='round'/>` +
        `<line x1='20' y1='4' x2='4' y2='20' stroke='${c}' stroke-width='3.2' stroke-linecap='round'/>` +
        `<circle cx='12' cy='12' r='1.6' fill='${c}'/>` +
        "</svg>\") 12 12, crosshair";
    _cursorCache.set(k, v);
    return v;
}

// 气泡光标：完整复刻 canvas 里的 drawStampShape（字号、数字、颜色、未确认虚线、胶囊拉伸）
// 热点放在气泡几何中心，让用户看到的 = 松手后落下来的。
export function _makeBubbleCursor(hex, label, fs, ratio) {
    const k = `b|${hex}|${label}|${fs}|${ratio}`;
    let v = _cursorCache.get(k);
    if (v) return v;
    const c = hex.replace('#', '%23');
    // Canvas cursor 上限 128×128（Chrome），过大会被浏览器忽略回退默认。
    // 用 canvas 的视觉参数 * 屏幕 scale（估 1.0，章已经足够大）。
    const r      = Math.max(8, fs * (ratio || 0.6));
    const len    = (label || '').toString().length || 1;
    const extraW = len <= 1 ? 0 : (len - 1) * fs * ((ratio || 0.6) * 0.5);
    const halfW  = r + extraW + 3;   // +3 留虚线外边
    const halfH  = r + 3;
    let size = Math.ceil(Math.max(halfW, halfH) * 2);
    if (size > 120) size = 120;      // Chrome cursor 上限保护
    const cx = size / 2, cy = size / 2;
    const dashA = Math.max(2, fs * 0.18).toFixed(1);
    const dashB = Math.max(1.5, fs * 0.12).toFixed(1);
    const lineW = Math.max(1, Math.min(fs / 9, 2.2)).toFixed(1);
    const fillText = (label || '').toString();
    // 简版字体栈（SVG + cursor 没法精细 font-feature，退而求次用 -apple-system）
    // 文字颜色微透明（canvas 未确认态 0.85 alpha）
    let shape;
    if (len <= 1) {
        shape = `<circle cx='${cx}' cy='${cy}' r='${r}' fill='white' stroke='${c}' stroke-width='${lineW}' stroke-dasharray='${dashA},${dashB}'/>`;
    } else {
        // 胶囊：用 rect + 圆角模拟 canvas 两端半圆 + 中间矩形
        const rx = cx - extraW - r;
        const w  = (extraW + r) * 2;
        shape = `<rect x='${rx}' y='${cy - r}' width='${w}' height='${r * 2}' rx='${r}' ry='${r}' fill='white' stroke='${c}' stroke-width='${lineW}' stroke-dasharray='${dashA},${dashB}'/>`;
    }
    // 视觉中心修正：SVG 没有 Canvas 的 measureText 精确量上下包围盒。
    // 实测 Chrome 对 SF Pro 的 `dominant-baseline='central'` 本身已相当接近视觉中心，
    // 只需 y 再下移 fs*0.07 微调即可落到圆心。不用 alphabetic + 0.35fs 方案——
    // 那会让数字下沉过度（SVG 不同浏览器的 alphabetic baseline 语义差异大，不如 central 稳）。
    const textSVG = `<text x='${cx}' y='${(cy + fs * 0.07).toFixed(1)}' text-anchor='middle' dominant-baseline='central' font-family='-apple-system,sans-serif' font-weight='600' font-size='${fs}' fill='${c}' fill-opacity='0.88'>${fillText}</text>`;
    v = "url(\"data:image/svg+xml;utf8," +
        `<svg xmlns='http://www.w3.org/2000/svg' width='${size}' height='${size}' viewBox='0 0 ${size} ${size}'>` +
        shape + textSVG +
        `</svg>\") ${cx} ${cy}, crosshair`;
    _cursorCache.set(k, v);
    return v;
}

// 读下一枚章应显示的内容：序号 / 字号 / 圈比；都来自 stamp-options 输入框
export function _readNextStampPreview() {
    const idInput    = document.getElementById('current-id');
    const fsInput    = document.getElementById('default-font');
    const ratioInput = document.getElementById('shape-ratio');
    const label = idInput ? (idInput.value || '1') : '1';
    let fs = fsInput ? parseInt(fsInput.value, 10) : 12;
    if (!Number.isFinite(fs) || fs <= 0) fs = 12;
    // 光标里 fs 放大 1.1 倍，看得更清楚；上限 24 避免气泡超过 cursor 限制
    fs = Math.min(24, Math.max(10, Math.round(fs * 1.1)));
    let ratio = ratioInput ? parseFloat(ratioInput.value) : 0.6;
    if (!Number.isFinite(ratio) || ratio <= 0) ratio = 0.6;
    return { label, fs, ratio };
}

export function updateCursor() {
    const ws = els.workspace;
    if (window.isPanning) { ws.style.cursor = 'grabbing'; return; }
    if (state.stampActive) {
        // 下一笔是"圆心"还是"箭头尖"——判定口径与 commitStamp 的参数顺序对齐：
        //   arrow_first: 第一笔=箭头尖, 第二笔=圆心
        //   否则(number_first): 第一笔=圆心, 第二笔=箭头尖
        // drag 模式 drawPhase=2 期间(鼠标拖动中)与 click 模式 drawPhase=1 同义，
        // 都是"正在瞄准第二笔"，所以 phase>0 一律按"下一笔=第二笔"处理。
        const ao = state.drawOrder === 'arrow_first';
        const isFirstClick = state.drawPhase === 0;
        const needCircle = isFirstClick ? !ao : ao;
        // 颜色：按住 Shift 预告"关键章"(蓝)，或 tempType 已为 'key'
        // （toggle 按钮设的或第二笔已提交的）
        const isKey = state.keys.shift || state.tempType === 'key';
        const color = getStampColor(isKey ? 'key' : 'normal');
        if (needCircle) {
            const { label, fs, ratio } = _readNextStampPreview();
            ws.style.cursor = _makeBubbleCursor(color, label, fs, ratio);
        } else {
            ws.style.cursor = _makeArrowCursor(color);
        }
        return;
    }
    if (state.baseTool === 'hand') { ws.style.cursor = 'grab'; return; }
    if (state.hoverNode)           { ws.style.cursor = 'move'; return; }
    ws.style.cursor = 'default';
}

// NOTE: syncToolbarFromSelection calls updateSelectedPanel() which remains in app.js.
// ES modules don't allow external reassignment of imported bindings — use setter.
let _onUpdateSelectedPanel = null;
export function registerCursorCallbacks(updateSelectedPanel) {
    _onUpdateSelectedPanel = updateSelectedPanel;
}

export function syncToolbarFromSelection() {
    if (state.selectedUUIDs.length === 1) {
        const s = state.stamps.find(st => st.uuid === state.selectedUUIDs[0]);
        if (s) {
            document.getElementById('current-id').value    = s.id;
            document.getElementById('default-font').value  = s.fs;
            document.getElementById('shape-ratio').value   = s.ratio || 0.60;
        }
    } else if (state.selectedUUIDs.length > 1) {
        const s = state.stamps.find(st => st.uuid === state.selectedUUIDs[0]);
        if (s) {
            document.getElementById('default-font').value  = s.fs;
            document.getElementById('shape-ratio').value   = s.ratio || 0.60;
            document.getElementById('current-id').value   = '-';
        }
    }
    _syncColorToggleBtn();
    updateToolbarState();
    if (_onUpdateSelectedPanel) _onUpdateSelectedPanel();
    // 通知选中章悬浮窗刷新（避免循环依赖 app_sidebar）
    document.dispatchEvent(new CustomEvent('pdf-reader:selection-change'));
}

export function updateToolbarState() {
    const optBar = document.getElementById('stamp-options');
    if (state.stampActive || state.selectedUUIDs.length > 0) {
        optBar.classList.add('active-panel');
    } else {
        optBar.classList.remove('active-panel');
    }
}
