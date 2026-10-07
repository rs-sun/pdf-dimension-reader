// app_sidebar/symbol_palette.js — 全局单例符号面板浮层
// 被 list.js / popup.js 在字段聚焦时调用，独占 _sym* 模块状态。

// ── 上下文感知符号面板 ─────────────────────────────────────────────────────
// 按 Gemini 方案：根据焦点字段动态显示对应的符号子集
const _SYM_CONTEXT = {
    // 一般尺寸前缀符号（锁定属性，与标称值分开存储）
    prefix: [
        { sym: '⌀', tip: '直径 Diameter' },           // ⌀
        { sym: 'R',      tip: '半径 Radius' },
        { sym: 'C',      tip: '倒角 Chamfer' },
        { sym: '□', tip: '正方形截面 Square' },       // □
        { sym: 'S⌀',tip: '球直径 Spherical ⌀' }, // S⌀
        { sym: 'SR',     tip: '球半径 Spherical R' },
        { sym: '°', tip: '角度 Degree' },             // °
        { sym: '↧', tip: '深度 Depth' },              // ↧
    ],
    // 表面粗糙度前缀
    roughness: [
        { sym: 'Ra',     tip: '算术平均粗糙度' },
        { sym: 'Rz',     tip: '最大高度粗糙度' },
        { sym: 'Rmax',   tip: '最大粗糙度峰值' },
    ],
    // 上下偏差：不弹面板，直接打字
    // GD&T 几何特征符号
    symbol: [
        { sym: '⊕', tip: '位置度 Position' },         // ⊕
        { sym: '⊥', tip: '垂直度 Perpendicularity' }, // ⊥
        { sym: '∥', tip: '平行度 Parallelism' },      // ∥
        { sym: '◎', tip: '同轴度 Concentricity' },    // ◎
        { sym: '○', tip: '圆度 Roundness' },          // ○
        { sym: '⌭', tip: '圆柱度 Cylindricity' },     // ⌭
        { sym: '—', tip: '直线度 Straightness' },     // —
        { sym: '∠', tip: '角度公差 Angularity' },     // ∠
        { sym: '⌓', tip: '面轮廓度 Surface Profile' },// ⌓
        { sym: '⌒', tip: '线轮廓度 Line Profile' },   // ⌒
        { sym: '▱', tip: '平面度 Flatness' },         // ▱
        { sym: '≡', tip: '对称度 Symmetry' },         // ≡
        { sym: '↗', tip: '圆跳动 Circular Runout' },  // ↗
        { sym: '↗↗', tip: '全跳动 Total Runout' }, // ↗↗
    ],
    // 材料条件修饰符
    datum: [
        { sym: 'Ⓜ', tip: '最大实体 MMC' },            // Ⓜ
        { sym: 'Ⓛ', tip: '最小实体 LMC' },            // Ⓛ
        { sym: 'Ⓕ', tip: '自由状态 Free State' },     // Ⓕ
        { sym: 'Ⓟ', tip: '延伸公差带 Projected' },    // Ⓟ
    ],
    // 备注 / 偏差：纯文本，不弹符号面板
};

// 字段 CSS class → 上下文 key 的映射（决定聚焦时弹哪组符号面板）
export const _FIELD_TO_CTX = {
    'field-prefix':    'prefix',      // 一般尺寸 ⌀/R/C/□ 面板
    'field-roughness': 'roughness',   // 表面粗糙度 Ra/Rz 面板
    'field-symbol':    'symbol',      // GD&T 符号面板
    'field-modifier':  'datum',       // 修饰符 Ⓜ Ⓛ Ⓕ Ⓟ 面板
    // field-nominal / field-upper-tol / field-lower-tol / field-note: 不弹面板
};

let _symTargetInput = null;
let _symCloseHandler = null;
let _symScrollCloseHandler = null;  // Bug 3：滚动关闭面板，避免锚点漂走后悬空

function _buildPaletteHTML(ctxKey) {
    const items = _SYM_CONTEXT[ctxKey] || [];
    return items.map(i =>
        `<span class="sym-btn" data-sym="${i.sym}" title="${i.tip}">${i.sym}</span>`
    ).join('');
}

export function _showSymbolPalette(anchorEl, targetInput, ctxKey) {
    const palette = document.getElementById('symbol-palette');
    if (!palette) return;

    // 无对应符号的字段不弹面板
    const items = _SYM_CONTEXT[ctxKey];
    if (!items) return;
    if (Array.isArray(items) && items.length === 0) return;

    _symTargetInput = targetInput;

    // 每次重建内容（上下文不同）
    palette.innerHTML = _buildPaletteHTML(ctxKey);

    // 点击插入符号（事件委托，每次重建所以用 onclick）
    palette.onmousedown = e => e.preventDefault();
    palette.onclick = e => {
        const btn = e.target.closest('.sym-btn');
        if (!btn || !_symTargetInput) return;
        const sym = btn.dataset.sym;
        const pos = _symTargetInput.selectionStart ?? _symTargetInput.value.length;
        _symTargetInput.value =
            _symTargetInput.value.slice(0, pos) + sym + _symTargetInput.value.slice(pos);
        _symTargetInput.selectionStart = _symTargetInput.selectionEnd = pos + sym.length;
        _symTargetInput.focus();
        _symTargetInput.dispatchEvent(new Event('blur'));
    };

    // Bug 2.1：改用 viewport 坐标 + position:fixed，让 sidebar / popup 共用同一定位逻辑
    const rect = anchorEl.getBoundingClientRect();
    palette.style.position = 'fixed';
    palette.style.display = 'flex';
    const pw = palette.offsetWidth || 234;
    const ph = palette.offsetHeight || 160;

    let left = rect.left;
    if (left + pw > window.innerWidth - 6) left = window.innerWidth - pw - 6;
    if (left < 6) left = 6;

    let top = rect.bottom + 4;
    // 下方空间不足时，翻到锚点上方
    if (top + ph > window.innerHeight - 6) {
        top = rect.top - ph - 4;
        if (top < 6) top = 6;
    }
    palette.style.left = left + 'px';
    palette.style.top  = top + 'px';

    // 清理上一次的关闭监听
    if (_symCloseHandler) {
        document.removeEventListener('mousedown', _symCloseHandler, true);
        _symCloseHandler = null;
    }
    if (_symScrollCloseHandler) {
        window.removeEventListener('scroll', _symScrollCloseHandler, true);
        _symScrollCloseHandler = null;
    }

    // 点击面板外部 & 非当前输入框时关闭
    _symCloseHandler = ev => {
        if (!palette.contains(ev.target) && ev.target !== anchorEl && ev.target !== targetInput) {
            palette.style.display = 'none';
            document.removeEventListener('mousedown', _symCloseHandler, true);
            _symCloseHandler = null;
            if (_symScrollCloseHandler) {
                window.removeEventListener('scroll', _symScrollCloseHandler, true);
                _symScrollCloseHandler = null;
            }
        }
    };
    document.addEventListener('mousedown', _symCloseHandler, true);

    // Bug 3：任意层级滚动 → 关闭面板（capture 捕获嵌套滚动）
    _symScrollCloseHandler = () => {
        palette.style.display = 'none';
        window.removeEventListener('scroll', _symScrollCloseHandler, true);
        _symScrollCloseHandler = null;
        if (_symCloseHandler) {
            document.removeEventListener('mousedown', _symCloseHandler, true);
            _symCloseHandler = null;
        }
    };
    window.addEventListener('scroll', _symScrollCloseHandler, { capture: true, passive: true });
}
