// app_sidebar/expand_form.js — 展开态字段表单：HTML 构建 + commit 写回 + 启发推断
// isEditingField 在本模块声明（owner），通过 getter/setter 给 list.js / popup.js 使用。

import { state } from '../state.js';
import { syncDrawLayerResolution } from '../canvas_renderer.js';
import { commitStampTypeChange, escHtml, pushHistory } from '../app_utils.js';

// ═══════════════════════════════════════════════════════════════════════════════
// isEditingField — 模块级编辑标记，避免重渲染时清空用户正在输入的焦点
// ═══════════════════════════════════════════════════════════════════════════════

let isEditingField = false;

export function getIsEditingField() { return isEditingField; }
export function setIsEditingField(v) { isEditingField = v; }

/**
 * Commit a type selection, release the render guard, then rebuild type-driven fields.
 * The later blur event observes the committed value and therefore cannot add a
 * duplicate history snapshot.
 */
export function finishStampTypeChangeEditing(stamp, nextType, rebuild) {
    const changed = commitStampTypeChange(stamp, nextType);
    setIsEditingField(false);
    rebuild?.();
    return changed;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 展开态 HTML 构建（手风琴内容区）
// ═══════════════════════════════════════════════════════════════════════════════

/** 构建展开态 HTML（手风琴内容区）
 *
 *  布局策略：类型驱动字段显隐，极致紧凑
 *    一般尺寸   → [类型][前缀符号][标称值][+上偏][-下偏] 一行 + 附注
 *    形位公差   → [类型][符号][公差值] 一行 + 基准/修饰 + 附注
 *    表面粗糙度 → [类型][Ra/Rz][数值] 一行 + 附注
 */
export const _DIM_TYPES = ['一般尺寸','形位公差','表面粗糙度'];

// 已知前缀列表（长前缀在前，贪婪匹配）
export const _KNOWN_PREFIXES = ['S⌀','SR','S∅','SØ',
    '⌀','∅','Ø','R','C','□','°','↧'];
// 长前缀在前（贪婪匹配）：Rmax 必须排 Ra/Rz 之前，否则 'Rmax3.2' 会被 startsWith('R'+'a'?) 误吞
export const _KNOWN_ROUGHNESS = ['Rmax','Ra','Rz'];

/** 从 nominal 文本中自动拆分已知前缀，写回 aiData（仅首次拆分）
 *  P2.4：effectiveType 由调用方显式传入，避免依赖 s.aiData.type 中可能被污染的推断值。
 *  缺省回落到 s.aiData.type 以保持向后兼容。
 */
export function _autoSplitPrefix(s, effectiveType) {
    if (!s.aiData) return;
    // 已有 prefix → 不覆盖
    if (s.aiData.prefix) return;
    const nom = s.aiData.nominal || s.aiData.Nominal || '';
    if (!nom) return;
    const t = effectiveType !== undefined ? effectiveType : s.aiData.type;
    const list = (t === '表面粗糙度') ? _KNOWN_ROUGHNESS : _KNOWN_PREFIXES;
    for (const p of list) {
        if (nom.startsWith(p)) {
            s.aiData.prefix  = p;
            s.aiData.nominal = nom.slice(p.length).trim();
            return;
        }
    }
}

export function _buildExpandHtml(s) {
    const ai = s.aiData || {};
    const getAi = (capKey, lowerKey) => ai[capKey] || ai[lowerKey] || '';
    const uid = escHtml(s.uuid);

    // 类型推断：后端 source 自动映射，用户可改
    //   优先级：1) 用户显式选择 (aiData.Type/type) → 2) source=gdt_line_frame → 形位公差
    //           3) source=surface_roughness 或 OCR 文本以 Ra/Rz 开头 → 表面粗糙度
    //           4) 兜底 → 一般尺寸
    let curType = getAi('Type', 'type');
    // Fix 2 (P2)：把启发逻辑的 gate 从 !curType 改为 !curType || !_DIM_TYPES.includes(curType)。
    //   旧 gate 只在 type 完全空时跑 probe；当后端写了 legacy 值（如 '线性' / '粗糙度' / 'roughness'）
    //   curType 已 truthy → probe 跳过 → 下面"二次保险"把 curType 直接清成"一般尺寸"，
    //   Ra 永远不被识别。改成"非空但不在白名单"也跑 probe，让 Ra/GD&T 路径有机会命中。
    if (!curType || !_DIM_TYPES.includes(curType)) {
        const src = s.source || '';
        // E2：probe 纳入识别相关字段 + s.text + s.dimText + s.raw_text。
        //   粗糙度前缀紧接数值的文本可能位于 aiData.symbol，而非 aiData.prefix；
        //   原先只读 Nominal/Prefix 会漏判，仍归类为"一般尺寸"。
        // Fix 4 (P3 advisory)：白名单只扫识别字段（Nominal/Prefix/Symbol，大小写两版）。
        //   旧实现 Object.values(s.aiData) 会带上 remarks/note/datum/modifier 等用户附注字段，
        //   附注里写"Ra 3.2 typical"会误命中，把类型推断成表面粗糙度。
        // Fix 3 (P3 advisory follow-up)：再补 _raw_text / raw_text 这两个 legacy / 内部字段，
        //   后端老路径偶尔把原始 OCR 文本只写在这些字段；同样仍排除 remarks/note/datum/modifier。
        const aiVals = s.aiData ? [
            s.aiData.Nominal, s.aiData.nominal,
            s.aiData.Prefix,  s.aiData.prefix,
            s.aiData.Symbol,  s.aiData.symbol,
            s.aiData._raw_text, s.aiData.raw_text,
        ].filter(v => typeof v === 'string').join(' ') : '';
        // probe 末段同步加几个常见 stamp 顶层字段名（raw_dim_text / ocr_text / rawText），
        // 后端字段名漂移时不至于一变就漏识别 Ra。
        const probe = aiVals + ' ' +
                      (s.text || '') + ' ' +
                      (s.dimText || '') + ' ' +
                      (s.raw_text || '') + ' ' +
                      (s.raw_dim_text || '') + ' ' +
                      (s.ocr_text || '') + ' ' +
                      (s.rawText || '');
        // P2.4：lookahead 替代 \b：粗糙度前缀后紧接数字时，单词边界不能正确匹配；
        // 改成 (?=\d|\s|[.+\-]|$) 显式列出粗糙度数值后续可能字符
        // R14.1：扩 Rmax（与后端 assembler_utils.py 的 ^R(?:a|z|max) 对齐）
        const looksRoughness = /R(?:max|a|z)(?=\d|\s|[.+\-]|$)/.test(probe);
        if (src === 'gdt_line_frame') {
            curType = '形位公差';
        } else if (src === 'surface_roughness' || looksRoughness) {
            curType = '表面粗糙度';
        } else {
            curType = '一般尺寸';
        }
    }
    // 二次保险：probe 仍然没把 curType 抬进 _DIM_TYPES（理论上 probe 必落到三选一），兜底"一般尺寸"。
    if (!_DIM_TYPES.includes(curType)) curType = '一般尺寸';
    const isGDT       = curType === '形位公差';
    const isRoughness = curType === '表面粗糙度';

    // P2.4：不再写回 curType 到 s.aiData.type；推断值仅用于本次渲染。
    // 用户显式选择由 .field-type change 处理器写入；保持推断与持久化解耦。
    // 自动拆分前缀（一次性，写回 aiData.prefix/nominal） —— effectiveType 显式传入
    _autoSplitPrefix(s, curType);

    const typeOpts = _DIM_TYPES.map(t =>
        `<option value="${t}" ${curType === t ? 'selected' : ''}>${t}</option>`
    ).join('');

    const prefix   = escHtml(getAi('Prefix','prefix'));
    const nom      = escHtml(getAi('Nominal','nominal'));
    const upper    = escHtml(getAi('upper_tol','upper_tol'));
    const lower    = escHtml(getAi('lower_tol','lower_tol'));
    const sym      = escHtml(getAi('Symbol','symbol'));
    const modifier = escHtml(getAi('Modifier','modifier'));
    const note     = escHtml(getAi('Note','remarks'));
    const methodRaw = getAi('_symbol_method','symbol_method');
    const methodLabel = methodRaw === 'vector_inversion' ? 'vector' : (methodRaw || '');
    const confRaw = getAi('_gdt_symbol_confidence','gdt_symbol_confidence');
    const confNum = Number(confRaw);
    const methodText = escHtml([
        methodLabel,
        Number.isFinite(confNum) && confRaw !== '' ? confNum.toFixed(2) : ''
    ].filter(Boolean).join(' '));

    // 基准拆分：兼容旧 "A B C" 字符串 → datum_1/2/3
    let d1 = escHtml(getAi('datum_1','datum_1'));
    let d2 = escHtml(getAi('datum_2','datum_2'));
    let d3 = escHtml(getAi('datum_3','datum_3'));
    if (!d1 && !d2 && !d3) {
        const rawDatum = getAi('Datum','datum');
        if (rawDatum) {
            const parts = rawDatum.split(/[\s/]+/).filter(Boolean);
            d1 = escHtml(parts[0] || '');
            d2 = escHtml(parts[1] || '');
            d3 = escHtml(parts[2] || '');
        }
    }

    let mainRow = '';
    let extraRow = '';

    if (isGDT) {
        mainRow = `
            <select class="field-type ex-type" data-uuid="${uid}">${typeOpts}</select>
            <input class="field-symbol ex-sym" data-uuid="${uid}" value="${sym}" placeholder="符号" title="GD&T 几何特征符号">
            <input class="field-nominal ex-grow" data-uuid="${uid}" value="${nom}" placeholder="公差值" title="公差带 (如 ⌀0.6)">`;
        extraRow = `
            <div class="ex-row">
                <span class="ex-tag">基准</span>
                <input class="field-datum1 ex-datum-slot" data-uuid="${uid}" value="${d1}" placeholder="第一" title="第一基准">
                <input class="field-datum2 ex-datum-slot" data-uuid="${uid}" value="${d2}" placeholder="第二" title="第二基准">
                <input class="field-datum3 ex-datum-slot" data-uuid="${uid}" value="${d3}" placeholder="第三" title="第三基准">
                <span class="ex-tag">修饰</span>
                <input class="field-modifier ex-modifier" data-uuid="${uid}" value="${modifier}" placeholder="Ⓜ" title="材料条件修饰符">
                ${methodText ? `<span class="ex-tag" title="GD&T 符号识别方法 / 置信度">${methodText}</span>` : ''}
            </div>`;
    } else if (isRoughness) {
        mainRow = `
            <select class="field-type ex-type" data-uuid="${uid}">${typeOpts}</select>
            <input class="field-roughness ex-prefix" data-uuid="${uid}" value="${prefix}" placeholder="Ra" title="粗糙度类型 (Ra / Rz / Rmax)">
            <input class="field-nominal ex-grow" data-uuid="${uid}" value="${nom}" placeholder="数值" title="粗糙度值 (如 1.6 / 6.3)">`;
    } else {
        // 一般尺寸：前缀符号可选，留空 = 普通距离尺寸
        mainRow = `
            <select class="field-type ex-type" data-uuid="${uid}">${typeOpts}</select>
            <input class="field-prefix ex-prefix" data-uuid="${uid}" value="${prefix}" title="前缀符号，留空=普通距离尺寸">
            <input class="field-nominal ex-nom" data-uuid="${uid}" value="${nom}" placeholder="标称值" title="数值">
            <input class="field-upper-tol ex-tol" data-uuid="${uid}" value="${upper}" placeholder="+上" title="上偏差">
            <input class="field-lower-tol ex-tol" data-uuid="${uid}" value="${lower}" placeholder="-下" title="下偏差">`;
    }

    return `
        <div class="stamp-expand" data-dim-type="${curType}">
            <div class="ex-row">${mainRow}</div>
            ${extraRow}
            <div class="ex-row">
                <input class="field-note ex-grow" data-uuid="${uid}" value="${note}" placeholder="附注 (如 2X / MAX / REF)" title="数量、附加说明">
            </div>
        </div>`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 展开态确认提交
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * 提交展开态编辑，写回 stamp，设置 confirmed，跳转到下一条未确认
 *
 * Fix 1 (P2)：加第 4 参数 nextUUID。S 快捷键传"当前之后第一个未确认（找不到 wrap 回头找）"，
 * 维持原 A/D/S 流式确认的顺序语义；sidebar OK 按钮 / Enter 确认沿用旧调用（不传），
 * 自动回退到"sorted 中第一个未确认（排除自身）"行为，无回归。
 *
 * 注意：updateSidebar 由调用方延迟注入（避免 expand_form ↔ list 循环依赖）。
 */
let _updateSidebarRef = null;
export function _registerUpdateSidebarForCommit(fn) { _updateSidebarRef = fn; }

export function _commitExpandedEdit(uuid, sorted, rootEl, nextUUID) {
    const s    = state.stamps.find(st => st.uuid === uuid);
    if (!s) return;
    // P1.2：优先从传入 rootEl（popup 等）读字段；缺省时回落到 sidebar DOM
    // 避免 popup OK 时读到 sidebar 旧值覆盖用户在 popup 中的编辑
    const root = rootEl || document.querySelector(`#stamp-list li[data-uuid="${uuid}"]`);
    // Fix 2 (P2)：root 必须含有 .field-* 才写回；否则（如 S 快捷键在折叠态触发）
    // root 存在但内部无展开字段，老逻辑会把 aiData 全清空。这里改成只在
    // 检测到展开字段时才走写回路径，让 S 在折叠态也能安全确认。
    if (root && root.querySelector('.field-nominal, .field-prefix, .field-roughness')) {
        const get = cls => root.querySelector(cls)?.value ?? '';
        if (!s.aiData) s.aiData = { type:'',prefix:'',nominal:'',upper_tol:'',lower_tol:'',symbol:'',datum:'',modifier:'',remarks:'' };
        s.aiData.type      = get('.field-type');
        s.aiData.prefix    = get('.field-prefix') || get('.field-roughness');
        s.aiData.nominal   = get('.field-nominal');
        s.aiData.upper_tol = get('.field-upper-tol');
        s.aiData.lower_tol = get('.field-lower-tol');
        s.aiData.symbol    = get('.field-symbol');
        s.aiData.datum_1   = get('.field-datum1');
        s.aiData.datum_2   = get('.field-datum2');
        s.aiData.datum_3   = get('.field-datum3');
        s.aiData.datum     = [s.aiData.datum_1, s.aiData.datum_2, s.aiData.datum_3].filter(Boolean).join(' / ');
        s.aiData.modifier  = get('.field-modifier');
        s.aiData.remarks   = get('.field-note');
    }
    s.confirmed = true;
    if (s.status === 'error') s.status = 'success';

    // 跳到下一条未确认
    // Fix 1 (P2)：调用方显式传 nextUUID 时（S 快捷键的"curIdx+1 wrap"语义）直接用作下一项；
    // 不传时维持旧"sorted 中第一个未确认（排除自身）"行为，兼容 sidebar OK / Enter 路径。
    const nextUnconf = nextUUID
        ? sorted.find(st => st.uuid === nextUUID)
        : sorted.find(st => !st.confirmed && st.uuid !== uuid);
    state.activeEditUUID = nextUnconf ? nextUnconf.uuid : null;
    if (nextUnconf) state.selectedUUIDs = [nextUnconf.uuid];

    pushHistory();
    if (_updateSidebarRef) _updateSidebarRef();
    syncDrawLayerResolution();

    // 滚动到新展开项
    if (nextUnconf) {
        const nextLi = document.querySelector(`#stamp-list li[data-uuid="${nextUnconf.uuid}"]`);
        if (nextLi) nextLi.scrollIntoView({ block: 'nearest' });
    }
}
