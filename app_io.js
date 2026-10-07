// app_io.js — orchestrator 拆分子模块
// 职责：文件载入（PDF / JSON / ZIP） + 工程归档导出（JSON + PDF 固化 + ZIP）+ HITL feedback
//
// 依赖的外部全局：window.pdfjsLib（来自 libs/pdf.min.js）/ window.PDFLib / window.JSZip

import { CONFIG } from './config.js';
import {
    state,
    advanceDocumentGeneration,
    createEmptyInspectionState,
    createEmptyReviewWorkspace,
    createEmptyVectorDebugState,
    rejectDocumentImportWhileAnalysisRunning,
} from './state.js';
import {
    renderBaseCanvases, syncDrawLayerResolution, getStampColor, resolveIsKey,
} from './canvas_renderer.js';
import {
    createHistorySnapshot, pushHistory, showLoading, hideLoading, toHalfWidthAndClean,
} from './app_utils.js';
import {
    updateEpsSlider,
} from './app_analysis.js';
import { generateUUID } from './state.js';
import { updateSidebar } from './app_sidebar.js';
import { runExportXlsx, runImportLegacyStamps, runImportMeasurements } from './api_client.js';
import { ensureDimBbox } from './stamp.js';
import { buildInspectionImportState } from './inspection_matching.js';
import {
    getPageGeometry,
    pageLocalBboxToDocumentGlobal,
    pageLocalPointToDocumentGlobal,
} from './page_coordinates.js';
import {
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_SCHEMA,
    buildArchiveArtifacts,
    buildExcelDimensions,
    buildExportSnapshot,
    executeImportTransaction,
    parseArchivePayload,
    sha256Hex,
} from './import_export_transaction.js';
import { renderVectorDebugUI } from './vector_debug_drawer.js';
import {
    applyPendingReviewBeforeUnload,
    confirmPendingReviewLeave,
} from './review_workspace_guard.js';
import { renderSnapshotPdf } from './snapshot_pdf_renderer.js';

// R14.4 B 路线：老存档 stamps 没 dimBbox，载入时批量补齐（从 arrowTip + crop 算）
function _migrateStampsForDimBbox(stamps) {
    if (!Array.isArray(stamps)) return;
    stamps.forEach(s => {
        ensureDimBbox(s);
        if (s?.source === 'legacy_stamp') {
            const preciseArrow = s.legacyArrowPrecise === true
                || s.legacy?.arrow_geometry === 'appearance_stream';
            const knownRectOnly = s.legacy?.arrow_geometry === 'stamp_rect'
                || s.legacyPositionOnly === true
                || s.status === 'legacy_position_only';
            s.dimBboxPlaceholder = s.dimBboxPlaceholder !== false;
            if (preciseArrow) {
                s.legacyPositionOnly = false;
                s.routingPathInferred = false;
            } else if (knownRectOnly) {
                s.legacyPositionOnly = true;
                s.routingPathInferred = true;
            } else {
                s.legacyPositionOnly = s.legacyPositionOnly === true;
                s.routingPathInferred = s.routingPathInferred !== false;
            }
            if (preciseArrow && s.status === 'legacy_position_only') {
                s.status = 'legacy_imported';
            }
            s.legacy = {
                ...(s.legacy || {}),
                geometry_source: s.legacy?.geometry_source
                    || (preciseArrow ? 'stamp_appearance_stream' : 'stamp_rect_inferred'),
            };
        }
    });
}

function _resetInspectionState() {
    state.inspection = createEmptyInspectionState();
    state.stamps.forEach(s => { delete s.inspection; });
}

function _resetDocumentTransientState() {
    state.deletedStamps = [];
    state.reviewWorkspace = createEmptyReviewWorkspace();
    state.vectorDebugState = createEmptyVectorDebugState();
    state.selectedUUIDs = [];
    state.basicDimensions = [];
    state.datumReferences = [];
    state.activeEditUUID = null;
    state.hoverNode = null;
    state.dragCache = null;
    state.selectionBox = null;
    state.lastMousePt = null;
}

function _captureImportRollbackState() {
    return {
        pdfDoc: window.pdfDoc || null,
        project: {
            docInfo: structuredClone(state.docInfo),
            currentPdfBuffer: state.currentPdfBuffer?.slice(0) || null,
            documentGeneration: state.documentGeneration,
            rotation: state.rotation,
            stamps: structuredClone(state.stamps),
            reviewWorkspace: structuredClone(state.reviewWorkspace),
            vectorDebugState: structuredClone(state.vectorDebugState),
            deletedStamps: structuredClone(state.deletedStamps),
            clusters: structuredClone(state.clusters),
            inspection: structuredClone(state.inspection),
            sidebarTab: state.sidebarTab,
            viewBoxes: structuredClone(state.viewBoxes),
            basicDimensions: structuredClone(state.basicDimensions),
            datumReferences: structuredClone(state.datumReferences),
            pageOffsets: structuredClone(state.pageOffsets),
            totalPages: state.totalPages,
            pdfWidth: state.pdfWidth,
            pdfHeight: state.pdfHeight,
            zoom: state.zoom,
            panX: state.panX,
            panY: state.panY,
            miniScale: state.miniScale,
            clusteringLocked: state.clusteringLocked,
            clusteringBaseIndex: state.clusteringBaseIndex,
            currentEps: state.currentEps,
            l1Cache: new Map(structuredClone([...state.l1Cache.entries()])),
            currentAnalyzedPages: new Set(structuredClone([...state.currentAnalyzedPages])),
            selectedUUIDs: structuredClone(state.selectedUUIDs),
            historyStack: structuredClone(state.historyStack),
            historyIndex: state.historyIndex,
            activeEditUUID: state.activeEditUUID,
        },
    };
}

async function _executeDocumentImportTransaction(options) {
    if (state.isDocumentImportRunning) {
        throw new Error('文档加载任务已在运行');
    }
    state.isDocumentImportRunning = true;
    try {
        return await executeImportTransaction(options);
    } finally {
        state.isDocumentImportRunning = false;
    }
}

function _assignProjectFields(project) {
    for (const [key, value] of Object.entries(project)) {
        state[key] = value instanceof Map
            ? new Map(structuredClone([...value.entries()]))
            : value instanceof Set
                ? new Set(structuredClone([...value]))
                : (value instanceof ArrayBuffer ? value.slice(0) : structuredClone(value));
    }
}

async function _rollbackImport(before) {
    if (!before) return;
    window.pdfDoc = before.pdfDoc;
    _assignProjectFields(before.project);
    if (window.pdfDoc) {
        await renderBaseCanvases(false);
        // renderBaseCanvases recalculates geometry; restore the exact user viewport.
        state.zoom = before.project.zoom;
        state.panX = before.project.panX;
        state.panY = before.project.panY;
    }
    updateEpsSlider();
    updateSidebar();
    syncDrawLayerResolution();
    renderVectorDebugUI();
}

function _finishDocumentCommit() {
    state.clusteringLocked = false;
    state.clusteringBaseIndex = -1;
    state.currentEps = null;
    state.selectedUUIDs = [];
    state.historyStack = [createHistorySnapshot()];
    state.historyIndex = 0;
    updateEpsSlider();
    updateSidebar();
    syncDrawLayerResolution();
    renderVectorDebugUI();
    document.getElementById('btn-undo').disabled = true;
    document.getElementById('btn-redo').disabled = true;
}

function _applyImportedProject(project) {
    _resetDocumentTransientState();
    state.docInfo = structuredClone(project.docInfo);
    state.stamps = structuredClone(project.stamps);
    state.deletedStamps = structuredClone(project.deletedStamps);
    state.viewBoxes = structuredClone(project.viewBoxes).map(viewBox => ({
        ...viewBox,
        shape: viewBox.shape || 'rect',
    }));
    state.basicDimensions = structuredClone(project.basicDimensions);
    state.datumReferences = structuredClone(project.datumReferences);
    state.inspection = structuredClone(project.inspection);
    state.clusters = structuredClone(project.clusters);
    state.l1Cache = new Map(structuredClone([...project.l1Cache.entries()]));
    state.currentAnalyzedPages = new Set(structuredClone([...project.currentAnalyzedPages]));
    _migrateStampsForDimBbox(state.stamps);
    _attachInspectionSummaries();
    state.sidebarTab = 'dimensions';
}

function _applyInspectionImport(importResult) {
    state.inspection = buildInspectionImportState(importResult, state.stamps, {
        docFilename: state.docInfo?.filename || '',
        resolveIsKey,
    });
    if (!['dimensions', 'inspection'].includes(state.sidebarTab)) {
        state.sidebarTab = 'inspection';
    }
    pushHistory();
}

function _attachInspectionSummaries() {
    // Inspection import is an overlay. Keep legacy stamp.inspection values readable
    // as a fallback, but do not write fresh inspection summaries into stamp data.
}

function _legacyPointToGlobal(point, pageIndex) {
    const page = getPageGeometry(state.pageOffsets, pageIndex);
    const global = pageLocalPointToDocumentGlobal({
        x: Number(point?.x || 0),
        y: Number(point?.y || 0),
    }, page);
    return {
        absX: global.x,
        absY: global.y,
    };
}

function _legacyBboxToGlobal(bbox, pageIndex) {
    const page = getPageGeometry(state.pageOffsets, pageIndex);
    return pageLocalBboxToDocumentGlobal({
        x: Number(bbox?.x || 0),
        y: Number(bbox?.y || 0),
        w: Number(bbox?.w || 0),
        h: Number(bbox?.h || 0),
    }, page);
}

function _findOwningViewBoxForPoint(point, pageIndex) {
    const candidates = (state.viewBoxes || []).filter(vb => {
        if (vb.pageIndex !== pageIndex) return false;
        if ((vb.shape || 'rect') === 'circle') {
            const dx = point.absX - Number(vb.cx || 0);
            const dy = point.absY - Number(vb.cy || 0);
            return Math.hypot(dx, dy) <= Number(vb.r || 0);
        }
        return point.absX >= vb.x && point.absX <= vb.x + vb.w
            && point.absY >= vb.y && point.absY <= vb.y + vb.h;
    });
    if (candidates.length === 0) return null;
    candidates.sort((a, b) => (a.w * a.h) - (b.w * b.h));
    return candidates[0];
}

function _legacyMarksToStamps(marks) {
    const stamps = [];
    for (const mark of marks || []) {
        const pageIndex = parseInt(mark.page || 1);
        const circleCenter = _legacyPointToGlobal(mark.circle_center, pageIndex);
        const arrowTip = _legacyPointToGlobal(mark.arrow_tip, pageIndex);
        const dimBbox = _legacyBboxToGlobal(mark.dim_bbox, pageIndex);
        const ownerViewBox = _findOwningViewBoxForPoint(arrowTip, pageIndex);
        const fs = Math.max(12, Math.min(24, Math.round(Number(mark.font_size || 12))));
        const dimBboxPlaceholder = mark.dim_bbox_placeholder !== false;
        const preciseArrow = mark.arrow_geometry === 'appearance_stream'
            || mark.arrow_geometry_precise === true;
        const crop = {
            l: dimBbox.x - arrowTip.absX,
            t: dimBbox.y - arrowTip.absY,
            r: dimBbox.x + dimBbox.w - arrowTip.absX,
            b: dimBbox.y + dimBbox.h - arrowTip.absY,
        };
        stamps.push({
            uuid: generateUUID(),
            id: String(mark.label || stamps.length + 1),
            type: 'normal',
            fs,
            ratio: 0.60,
            isAutoGenerated: false,
            source: 'legacy_stamp',
            pageIndex,
            viewBoxId: ownerViewBox?.id || null,
            circleCenter,
            arrowTip,
            routingPath: Array.isArray(mark.routing_path) && mark.routing_path.length >= 2
                ? mark.routing_path.map(pt => _legacyPointToGlobal(pt, pageIndex))
                : [circleCenter, arrowTip],
            arrowTipManual: false,
            crop,
            dimBbox,
            dimBboxPlaceholder,
            legacyPositionOnly: !preciseArrow,
            routingPathInferred: !preciseArrow,
            legacyArrowPrecise: preciseArrow,
            sourceBbox: { ...dimBbox },
            cropSourceBbox: { ...dimBbox },
            cropAdjusted: true,
            status: preciseArrow ? 'legacy_imported' : 'legacy_position_only',
            confirmed: false,
            sortX: arrowTip.absX,
            sortY: arrowTip.absY,
            aiData: {
                type: '--',
                nominal: '',
                tolerance: '',
                upper_tol: '',
                lower_tol: '',
                symbol: '',
                datum: '',
                remarks: '',
                confidence: 'legacy',
                _raw_type: 'legacy_stamp',
                _raw_text: String(mark.label || ''),
            },
            legacy: {
                label_bbox: mark.label_bbox || null,
                stamp_bbox: mark.stamp_bbox || null,
                direction: mark.direction || '',
                direction_angle_degrees: mark.direction_angle_degrees ?? null,
                arrow_geometry: mark.arrow_geometry || '',
                geometry_source: mark.geometry_source || 'stamp_rect_inferred',
            },
        });
    }
    return stamps;
}

function _applyLegacyStampImport(importResult) {
    const marks = importResult?.marks || [];
    if (marks.length === 0) {
        alert('旧序号图纸里没有解析到可转换的序号。');
        return;
    }
    if (importResult.page_count && state.totalPages
            && Number(importResult.page_count) !== Number(state.totalPages)) {
        const ok = confirm(
            `旧序号 PDF 共 ${importResult.page_count} 页，当前底图共 ${state.totalPages} 页。\n` +
            '页数不一致，仍然导入可匹配页码的序号吗？'
        );
        if (!ok) return;
    }
    if (state.stamps.length > 0) {
        const ok = confirm(`导入旧序号会替换当前 ${state.stamps.length} 个标记，继续吗？`);
        if (!ok) return;
    }

    state.stamps = _legacyMarksToStamps(marks);
    _resetDocumentTransientState();
    _migrateStampsForDimBbox(state.stamps);
    _resetInspectionState();
    state.sidebarTab = 'dimensions';
    state.selectedUUIDs = [];
    state.historyStack = [];
    state.historyIndex = -1;
    pushHistory();
    updateSidebar();
    syncDrawLayerResolution();

    const warnings = (importResult.warnings || []).length
        ? `\n警告：${importResult.warnings.join('；')}`
        : '';
    alert(`旧序号导入完成：${state.stamps.length} 个标记。${warnings}`);
}

function _zipBaseName(path) {
    return String(path || '').split('/').filter(Boolean).pop() || String(path || '');
}

function _stripExt(name) {
    return String(name || '').replace(/\.[^.]+$/, '');
}

function _isPdfFile(file) {
    return String(file?.name || '').toLowerCase().endsWith('.pdf');
}

function _arrayBufferFromUint8(bytes) {
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function _mergePdfFiles(files) {
    const PDFDocument = window.PDFLib?.PDFDocument;
    if (!PDFDocument) throw new Error('pdf-lib 未加载，无法合并 PDF');
    const merged = await PDFDocument.create();
    for (const file of files) {
        const srcBytes = await file.arrayBuffer();
        const src = await PDFDocument.load(srcBytes);
        const pages = await merged.copyPages(src, src.getPageIndices());
        pages.forEach(page => merged.addPage(page));
    }
    return _arrayBufferFromUint8(await merged.save());
}

function _mergedPdfName(files) {
    const stems = files.map(file => _stripExt(file.name));
    if (stems.length <= 2) return stems.join('+') + '.pdf';
    return `${stems[0]}+${stems.length - 1}more.pdf`;
}

function _legacyPackKey(name) {
    const stem = _stripExt(_zipBaseName(name)).trim();
    const m = stem.match(/^([A-Za-z]+[0-9]+[A-Za-z0-9]*)/);
    if (m) return m[1].toUpperCase();
    return stem.split(/[-_\s]/)[0].toUpperCase();
}

function _isPlainOriginalForKey(path, key) {
    const stem = _stripExt(_zipBaseName(path)).trim().toUpperCase();
    return stem === key;
}

function _buildLegacyZipPairs(zip) {
    const groups = new Map();
    for (const path in zip.files) {
        const entry = zip.files[path];
        if (entry.dir) continue;
        const lower = path.toLowerCase();
        if (!lower.endsWith('.pdf') && !lower.endsWith('.xls') && !lower.endsWith('.xlsx') && !lower.endsWith('.xlsm')) {
            continue;
        }
        const key = _legacyPackKey(path);
        if (!key) continue;
        if (!groups.has(key)) groups.set(key, { key, pdfs: [], excels: [] });
        const group = groups.get(key);
        if (lower.endsWith('.pdf')) group.pdfs.push(path);
        else group.excels.push(path);
    }

    const pairs = [];
    for (const group of groups.values()) {
        if (group.pdfs.length < 2) continue;
        const originals = group.pdfs.filter(path => _isPlainOriginalForKey(path, group.key));
        const originalPath = originals[0] || [...group.pdfs].sort((a, b) => _zipBaseName(a).length - _zipBaseName(b).length)[0];
        const stampedCandidates = group.pdfs.filter(path => path !== originalPath);
        if (stampedCandidates.length === 0) continue;
        const stampedPath = stampedCandidates.sort((a, b) => _zipBaseName(a).length - _zipBaseName(b).length)[0];
        pairs.push({
            key: group.key,
            originalPath,
            stampedPath,
            excelPath: group.excels[0] || null,
        });
    }
    return pairs.sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));
}

function _chooseLegacyZipPair(pairs) {
    if (pairs.length === 0) return null;
    if (pairs.length === 1) return pairs[0];
    const lines = pairs.map((pair, idx) =>
        `${idx + 1}. ${pair.key}: ${_zipBaseName(pair.originalPath)} + ${_zipBaseName(pair.stampedPath)}`
    );
    const raw = prompt(`检测到 ${pairs.length} 组旧序号图纸，请输入要打开的序号：\n\n${lines.join('\n')}`, '1');
    if (raw === null) return null;
    const idx = parseInt(raw, 10) - 1;
    if (!Number.isInteger(idx) || idx < 0 || idx >= pairs.length) {
        alert('输入序号无效。');
        return null;
    }
    return pairs[idx];
}

async function _loadPdfBufferAsCurrentDocument(filename, buffer) {
    if (rejectDocumentImportWhileAnalysisRunning(state)) return false;
    try {
        await _executeDocumentImportTransaction({
            stage: async () => {
                const pdfBuffer = buffer.slice(0);
                const pdfDoc = await pdfjsLib.getDocument(pdfBuffer.slice(0)).promise;
                if (pdfDoc.numPages > CONFIG.MAX_PAGE_COUNT) {
                    throw new RangeError(`PDF 超过 ${CONFIG.MAX_PAGE_COUNT} 页限制。`);
                }
                return { pdfBuffer, pdfDoc };
            },
            capture: _captureImportRollbackState,
            commit: async staged => {
                state.docInfo = {
                    filename: filename.replace(/\.[^/.]+$/, ''),
                    version: 1,
                };
                state.currentPdfBuffer = staged.pdfBuffer;
                state.rotation = 0;
                window.pdfDoc = staged.pdfDoc;
                await renderBaseCanvases(true);
                state.stamps = [];
                state.deletedStamps = [];
                state.clusters = [];
                state.viewBoxes = [];
                state.basicDimensions = [];
                state.datumReferences = [];
                state.inspection = createEmptyInspectionState();
                state.sidebarTab = 'dimensions';
                state.activeEditUUID = null;
                state.l1Cache = new Map();
                state.currentAnalyzedPages = new Set();
                _resetDocumentTransientState();
                _finishDocumentCommit();
                advanceDocumentGeneration(state);
            },
            rollback: _rollbackImport,
        });
        return true;
    } catch (error) {
        alert(`PDF 加载失败：${error.message || error}`);
        return false;
    }
}

async function _fileFromZip(zip, path, mime) {
    const buffer = await zip.files[path].async('arraybuffer');
    return new File([buffer], _zipBaseName(path), { type: mime });
}

async function _stageManifestArchive(zip) {
    const manifestEntry = zip.file(ARCHIVE_MANIFEST_NAME);
    if (!manifestEntry) return null;
    const manifest = JSON.parse(await manifestEntry.async('string'));
    if (manifest?.schema !== ARCHIVE_SCHEMA || !Array.isArray(manifest.files)) {
        throw new TypeError('ZIP manifest schema 不兼容');
    }
    const filesByRole = new Map();
    for (const descriptor of manifest.files) {
        const entry = zip.file(descriptor?.name || '');
        if (!entry) throw new TypeError(`ZIP 缺少 manifest 文件：${descriptor?.name || ''}`);
        const bytes = await entry.async('uint8array');
        const actualSha = await sha256Hex(bytes);
        if (actualSha !== descriptor.sha256) {
            throw new TypeError(`ZIP 文件 SHA-256 不匹配：${descriptor.name}`);
        }
        filesByRole.set(descriptor.role, bytes);
    }
    const jsonBytes = filesByRole.get('project_json');
    const pdfBytes = filesByRole.get('original_pdf');
    if (!jsonBytes || !pdfBytes) throw new TypeError('ZIP 缺少 project_json 或 original_pdf');
    const pdfBuffer = pdfBytes.buffer.slice(
        pdfBytes.byteOffset,
        pdfBytes.byteOffset + pdfBytes.byteLength,
    );
    const pdfDoc = await pdfjsLib.getDocument(pdfBuffer.slice(0)).promise;
    if (pdfDoc.numPages > CONFIG.MAX_PAGE_COUNT) {
        throw new RangeError(`PDF 超过 ${CONFIG.MAX_PAGE_COUNT} 页限制。`);
    }
    const project = parseArchivePayload(new TextDecoder().decode(jsonBytes), {
        pdfPageCount: pdfDoc.numPages,
    });
    if (Number(manifest.page_count) !== pdfDoc.numPages
            || Number(manifest.item_count) !== project.stamps.length) {
        throw new TypeError('ZIP manifest 的页数或条目数与存档不一致');
    }
    return { pdfBuffer, pdfDoc, project };
}

async function _commitManifestArchive(staged) {
    state.currentPdfBuffer = staged.pdfBuffer;
    state.rotation = 0;
    window.pdfDoc = staged.pdfDoc;
    await renderBaseCanvases(true);
    _applyImportedProject(staged.project);
    _finishDocumentCommit();
    advanceDocumentGeneration(state);
}

async function _tryLoadLegacyStampZip(zip) {
    try {
        const pairs = _buildLegacyZipPairs(zip);
        const pair = _chooseLegacyZipPair(pairs);
        if (!pair) return false;
        const importExcel = Boolean(pair.excelPath
            && confirm(`是否同时导入 ${_zipBaseName(pair.excelPath)} 的测量结果？`));
        await _executeDocumentImportTransaction({
            stage: async () => {
                showLoading(`校验 ${pair.key} 原始底图...`);
                const originalBuf = await zip.files[pair.originalPath].async('arraybuffer');
                const pdfBuffer = originalBuf.slice(0);
                const pdfDoc = await pdfjsLib.getDocument(pdfBuffer.slice(0)).promise;
                if (pdfDoc.numPages > CONFIG.MAX_PAGE_COUNT) {
                    throw new RangeError(`PDF 超过 ${CONFIG.MAX_PAGE_COUNT} 页限制。`);
                }
                showLoading(`校验 ${pair.key} 旧序号...`);
                const stampedFile = await _fileFromZip(zip, pair.stampedPath, 'application/pdf');
                const legacy = await runImportLegacyStamps(stampedFile);
                if (!legacy || !Array.isArray(legacy.marks) || legacy.marks.length === 0) {
                    throw new TypeError('旧序号图纸没有可导入标记');
                }
                if (legacy.page_count && Number(legacy.page_count) !== pdfDoc.numPages) {
                    throw new TypeError('旧序号图纸与原始底图页数不一致');
                }
                let inspection = null;
                if (importExcel) {
                    showLoading(`校验 ${pair.key} 测量结果...`);
                    const excelFile = await _fileFromZip(zip, pair.excelPath, 'application/vnd.ms-excel');
                    inspection = await runImportMeasurements([excelFile]);
                    if (!inspection) throw new TypeError('测量结果导入失败');
                }
                return { pdfBuffer, pdfDoc, legacy, inspection };
            },
            capture: _captureImportRollbackState,
            commit: async staged => {
                state.docInfo = {
                    filename: _stripExt(_zipBaseName(pair.originalPath)),
                    version: 1,
                };
                state.currentPdfBuffer = staged.pdfBuffer;
                state.rotation = 0;
                window.pdfDoc = staged.pdfDoc;
                await renderBaseCanvases(true);
                _resetDocumentTransientState();
                state.viewBoxes = [];
                state.stamps = _legacyMarksToStamps(staged.legacy.marks);
                state.deletedStamps = [];
                state.basicDimensions = [];
                state.datumReferences = [];
                state.clusters = [];
                state.l1Cache = new Map();
                state.currentAnalyzedPages = new Set();
                _migrateStampsForDimBbox(state.stamps);
                state.inspection = staged.inspection
                    ? buildInspectionImportState(staged.inspection, state.stamps, {
                        docFilename: state.docInfo.filename,
                        resolveIsKey,
                    })
                    : createEmptyInspectionState();
                state.sidebarTab = staged.inspection ? 'inspection' : 'dimensions';
                _finishDocumentCommit();
                advanceDocumentGeneration(state);
            },
            rollback: _rollbackImport,
        });
        return true;
    } finally {
        hideLoading();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 文件 I/O：PDF / JSON / ZIP 载入
// ═══════════════════════════════════════════════════════════════════════════════

// 整页拖放支持：拖任意 .pdf / .json / .zip 到窗口任意位置 = 等价于点"打开"选文件
// 实现方式：把拖入的 file 塞进 file-upload input + dispatch change 事件，
// 复用下面已有的 change handler（不重复一份解析逻辑）
let _dragCounter = 0;
function _setDropOverlay(active) {
    document.body.classList.toggle('drag-over', active);
}
['dragenter', 'dragover'].forEach(evt => {
    window.addEventListener(evt, e => {
        if (!e.dataTransfer || !e.dataTransfer.types.includes('Files')) return;
        e.preventDefault();
        if (evt === 'dragenter') {
            _dragCounter += 1;
            if (_dragCounter === 1) _setDropOverlay(true);
        }
    });
});
window.addEventListener('dragleave', e => {
    if (!e.dataTransfer || !e.dataTransfer.types.includes('Files')) return;
    _dragCounter = Math.max(0, _dragCounter - 1);
    if (_dragCounter === 0) _setDropOverlay(false);
});
window.addEventListener('drop', e => {
    if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
    e.preventDefault();
    _dragCounter = 0;
    _setDropOverlay(false);
    if (rejectDocumentImportWhileAnalysisRunning(state)) return;
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 1 && !files.every(_isPdfFile)) {
        alert('多文件合并仅支持 PDF。');
        return;
    }
    if (files.length === 1) {
        const lower = files[0].name.toLowerCase();
        if (!lower.endsWith('.pdf') && !lower.endsWith('.json') && !lower.endsWith('.zip')) {
            alert('仅支持 .pdf / .json / .zip 文件');
            return;
        }
    }
    if (files.length === 0) {
        return;
    }
    const input = document.getElementById('file-upload');
    const dt = new DataTransfer();
    files.forEach(file => dt.items.add(file));
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
});

document.getElementById('file-upload').addEventListener('change', async e => {
    const files = Array.from(e.target.files || []);
    const file = files[0];
    if (!file) return;
    if (rejectDocumentImportWhileAnalysisRunning(state)) {
        e.target.value = '';
        return;
    }
    if (!confirmPendingReviewLeave(state.reviewWorkspace)) {
        e.target.value = '';
        return;
    }

    const maxBytes = CONFIG.MAX_FILE_SIZE_MB * 1024 * 1024;
    if (files.some(f => f.size > maxBytes)) {
        alert(`文件超过本地打开 ${CONFIG.MAX_FILE_SIZE_MB}MB 限制，请选择更小的文件。`);
        e.target.value = ''; return;
    }
    if (files.reduce((sum, f) => sum + f.size, 0) > maxBytes) {
        alert(`合并后文件预计超过本地打开 ${CONFIG.MAX_FILE_SIZE_MB}MB 限制，请减少 PDF 数量。`);
        e.target.value = ''; return;
    }

    if (files.length > 1) {
        if (!files.every(_isPdfFile)) {
            alert('多文件合并仅支持 PDF。');
            e.target.value = ''; return;
        }
        try {
            showLoading(`合并 ${files.length} 个 PDF...`);
            const mergedBuffer = await _mergePdfFiles(files);
            await _loadPdfBufferAsCurrentDocument(_mergedPdfName(files), mergedBuffer);
        } catch (err) {
            alert(`PDF 合并失败：${err.message || err}`);
        } finally {
            hideLoading();
            e.target.value = '';
        }
        return;
    }

    if (file.name.toLowerCase().endsWith('.json')) {
        try {
            if (!window.pdfDoc) throw new TypeError('请先打开与数据存档匹配的 PDF');
            await _executeDocumentImportTransaction({
                stage: async () => parseArchivePayload(await file.text(), {
                    pdfPageCount: window.pdfDoc.numPages,
                }),
                capture: _captureImportRollbackState,
                commit: async project => {
                    _applyImportedProject(project);
                    _finishDocumentCommit();
                    advanceDocumentGeneration(state);
                },
                rollback: _rollbackImport,
            });
            alert('数据载入完成。');
        } catch (error) {
            alert(`数据文件格式错误，当前项目未改变：${error.message || error}`);
        }

    } else if (file.name.toLowerCase().endsWith('.pdf')) {
        const buffer = await file.arrayBuffer();
        await _loadPdfBufferAsCurrentDocument(file.name, buffer);

    } else if (file.name.toLowerCase().endsWith('.zip')) {
        try {
            const buffer = await file.arrayBuffer();
            const zip    = await JSZip.loadAsync(buffer);
            if (rejectDocumentImportWhileAnalysisRunning(state)) {
                e.target.value = '';
                return;
            }
            const hasManifest = Boolean(zip.file(ARCHIVE_MANIFEST_NAME));
            if (hasManifest) {
                await _executeDocumentImportTransaction({
                    stage: () => _stageManifestArchive(zip),
                    capture: _captureImportRollbackState,
                    commit: _commitManifestArchive,
                    rollback: _rollbackImport,
                });
                alert('工程压缩包载入完成。');
            } else {
                const loadedLegacy = await _tryLoadLegacyStampZip(zip);
                if (!loadedLegacy) throw new TypeError('ZIP 缺少兼容 manifest');
            }
        } catch (error) {
            alert(`解析压缩包失败，当前项目未改变：${error.message || error}`);
        }
    }
    e.target.value = '';
});

document.getElementById('measurement-upload')?.addEventListener('change', async e => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    showLoading('导入测量结果...');
    try {
        const result = await runImportMeasurements(files);
        if (!result) return;
        _applyInspectionImport(result);
        updateSidebar();
        syncDrawLayerResolution();
        const s = state.inspection.stats;
        const skipped = s.skippedRecordCount ? `，忽略其他图纸 ${s.skippedRecordCount} 条` : '';
        alert(`测量结果导入完成：${s.fileCount} 个文件，${s.recordCount} 条记录，匹配 ${s.matchedStampCount} 个图章，NG ${s.nokCount} 条${skipped}。`);
    } finally {
        hideLoading();
        e.target.value = '';
    }
});

document.getElementById('legacy-stamp-upload')?.addEventListener('change', async e => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!state.currentPdfBuffer || !window.pdfDoc) {
        alert('请先打开原始底图 PDF，再导入旧序号 PDF。');
        e.target.value = '';
        return;
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('旧序号导入目前只支持 PDF 文件。');
        e.target.value = '';
        return;
    }
    showLoading('导入旧序号...');
    try {
        const result = await runImportLegacyStamps(file);
        if (result) _applyLegacyStampImport(result);
    } finally {
        hideLoading();
        e.target.value = '';
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// 导出（JSON + PDF 固化 + ZIP 归档）
// ═══════════════════════════════════════════════════════════════════════════════

async function _renderSnapshotPdf(snapshot) {
    return renderSnapshotPdf(snapshot, {
        pdfLib: window.PDFLib,
        sourcePdf: state.currentPdfBuffer,
        resolveStampColor: getStampColor,
        cleanText: toHalfWidthAndClean,
    });
}

async function _saveArchiveBlob(blob, zipName) {
    if (typeof window.showSaveFilePicker === 'function') {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: zipName,
                types: [{ description: 'ZIP 归档', accept: { 'application/zip': ['.zip'] } }],
            });
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            return true;
        } catch (error) {
            if (error.name === 'AbortError') return false;
        }
    }
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = zipName;
    anchor.click();
    URL.revokeObjectURL(url);
    return true;
}

async function _exportTransactionalArchive() {
    if (!confirmPendingReviewLeave(state.reviewWorkspace)) return;
    if (state.stamps.length === 0) return alert('无数据可供导出。');
    if (!state.currentPdfBuffer || !window.pdfDoc) return alert('请先打开图纸文档。');
    const confirmedOnly = document.getElementById('toggle-export-confirmed-only')?.checked ?? false;
    const selected = confirmedOnly ? state.stamps.filter(stamp => stamp.confirmed) : state.stamps;
    if (confirmedOnly && selected.length === 0) {
        return alert('当前没有已确认的标记，无法导出。');
    }
    if (!confirmedOnly) {
        const unconfirmed = state.stamps.filter(stamp => !stamp.confirmed).length;
        if (unconfirmed > 0 && !confirm(`当前仍有 ${unconfirmed} 条标记未确认，是否强制导出？`)) return;
        if (unconfirmed > 0 && !confirm('强制导出的归档将包含未确认数据，请再次确认。')) return;
    }

    const button = document.getElementById('btn-submit');
    const originalText = button.innerText;
    button.disabled = true;
    try {
        button.innerText = '生成统一快照...';
        const now = new Date();
        const timestamp = now.toISOString();
        const snapshot = buildExportSnapshot(state, { confirmedOnly, timestamp });
        const excelDimensions = buildExcelDimensions(snapshot);
        const safeTime = timestamp.replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
        const base = `${snapshot.doc_info.filename}_V${snapshot.doc_info.version}_${safeTime}`;
        const names = {
            json: `${base}_数据存档.json`,
            pdf: `${base}_固化输出.pdf`,
            originalPdf: `${base}_原始底图.pdf`,
            excel: `${base}_尺寸表.xlsx`,
        };

        button.innerText = '生成必需产物...';
        const artifacts = await buildArchiveArtifacts({
            snapshot,
            originalPdf: state.currentPdfBuffer.slice(0),
            renderPdf: _renderSnapshotPdf,
            excelDimensions,
            exportExcel: dims => runExportXlsx(1, {
                partNo: snapshot.doc_info.filename || 'drawing',
                allPages: true,
                silent: true,
                dims,
            }),
            names,
        });

        button.innerText = '校验并打包 ZIP...';
        const zip = new JSZip();
        for (const [name, bytes] of Object.entries(artifacts.files)) zip.file(name, bytes);
        zip.file(ARCHIVE_MANIFEST_NAME, JSON.stringify(artifacts.manifest, null, 2));
        const blob = await zip.generateAsync({ type: 'blob' });
        const zipName = `${base}_工程归档.zip`;
        if (!await _saveArchiveBlob(blob, zipName)) return;

        state.docInfo.version = snapshot.doc_info.version;
        submitFeedback().catch(() => {});
    } catch (error) {
        console.error('[archive export]', error);
        alert(`导出失败，未生成归档：${error.message || error}`);
    } finally {
        button.innerText = originalText;
        button.disabled = false;
    }
}

document.getElementById('btn-submit').addEventListener('click', _exportTransactionalArchive);
window.addEventListener('beforeunload', event => (
    applyPendingReviewBeforeUnload(event, state.reviewWorkspace)
));
// ═══════════════════════════════════════════════════════════════════════════════
// HITL 反馈提交（导出时自动调用，静默失败）
// ═══════════════════════════════════════════════════════════════════════════════

async function submitFeedback() {
    const actions = [];
    for (const stamp of state.stamps) {
        const action = stamp.isAutoGenerated ? 'keep' : 'add';
        actions.push({
            stamp_id: stamp.id || null,
            action: action,
            nominal: stamp.aiData?.nominal || '',
            source: stamp.source || '',
            confidence: stamp.aiData?.confidence || '',
            bbox: stamp.crop || null,
        });
    }
    for (const stamp of state.deletedStamps) {
        actions.push({
            stamp_id: stamp.id || null,
            action: 'delete',
            nominal: stamp.aiData?.nominal || '',
            source: stamp.source || '',
            confidence: stamp.aiData?.confidence || '',
            bbox: stamp.crop || null,
        });
    }
    if (actions.length === 0) return;
    try {
        await fetch(`${CONFIG.API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pdf_name: state.docInfo?.filename || 'unknown',
                page: parseInt(document.getElementById('page-indicator')?.innerText?.split('/')[0]) || 1,
                timestamp: new Date().toISOString(),
                actions,
            }),
        });
    } catch (e) {
        console.warn('[HITL] Feedback submission failed:', e);
    }
}
