// Transactional project import + one-source full-document export contract.

import {
    documentGlobalPointToPageLocal,
    getPageGeometry,
} from './page_coordinates.js';

export const ARCHIVE_SCHEMA = 'pdf_reader.archive/v2';
export const ARCHIVE_MANIFEST_NAME = 'manifest.json';

function clone(value) {
    return structuredClone(value);
}

function asArray(value, name) {
    if (!Array.isArray(value)) throw new TypeError(`${name} 必须是数组`);
    return value;
}

function pageNumber(value, name) {
    const number = Number(value);
    if (!Number.isInteger(number) || number < 1) {
        throw new TypeError(`${name} 页码无效`);
    }
    return number;
}

function projectPageCount(source) {
    const explicit = Number(source?.totalPages);
    if (Number.isInteger(explicit) && explicit > 0) return explicit;
    if (Array.isArray(source?.pages) && source.pages.length > 0) return source.pages.length;
    if (Array.isArray(source?.pageOffsets) && source.pageOffsets.length > 0) {
        return source.pageOffsets.length;
    }
    throw new TypeError('文档页数无效');
}

function projectPages(source) {
    return Array.from({ length: projectPageCount(source) }, (_, index) => index + 1);
}

function serializableMap(map) {
    return clone([...(map instanceof Map ? map : new Map()).entries()]);
}

function serializableSet(set) {
    return clone([...(set instanceof Set ? set : new Set()).values()]);
}

function exportDocInfo(source, stamps, confirmedOnly, timestamp) {
    const original = clone(source.docInfo || source.doc_info || {});
    const currentVersion = Number(original.version);
    return {
        ...original,
        version: Number.isFinite(currentVersion) ? currentVersion + 1 : 1,
        timestamp,
        totalPages: projectPageCount(source),
        totalStamps: stamps.length,
        confirmedCount: stamps.filter(stamp => stamp.confirmed === true).length,
        exportMode: confirmedOnly ? 'confirmed_only' : 'all',
    };
}

function filteredInspection(rawInspection, stamps) {
    const inspection = clone(rawInspection || {});
    const allowed = new Set(stamps.map(stamp => stamp.uuid).filter(Boolean));
    inspection.byStampUuid = Object.fromEntries(
        Object.entries(inspection.byStampUuid || {}).filter(([uuid]) => allowed.has(uuid)),
    );
    inspection.records = (inspection.records || []).filter(
        record => !record?.stampUuid || allowed.has(record.stampUuid),
    );
    inspection.unmatchedRecords = (inspection.unmatchedRecords || []).filter(
        record => !record?.stampUuid || allowed.has(record.stampUuid),
    );
    if (inspection.stats && typeof inspection.stats === 'object') {
        const summaries = Object.values(inspection.byStampUuid);
        inspection.stats = {
            ...inspection.stats,
            recordCount: inspection.records.length,
            matchedStampCount: summaries.filter(item => item?.matchStatus === 'matched_by_no').length,
            unmatchedRecordCount: inspection.unmatchedRecords.length,
            nokCount: summaries.filter(item => item?.result === 'NOK').length,
            unmeasuredCount: summaries.filter(item => item?.result === 'UNMEASURED').length,
            keyNokCount: summaries.filter(item => item?.isKey && item?.result === 'NOK').length,
        };
    }
    return inspection;
}

export function buildExportSnapshot(source, {
    confirmedOnly = false,
    timestamp = new Date().toISOString(),
} = {}) {
    const stamps = clone((source.stamps || []).filter(
        stamp => !confirmedOnly || stamp.confirmed === true,
    ));
    const pages = projectPages(source);
    return {
        schema: ARCHIVE_SCHEMA,
        pages,
        page_geometries: clone(source.pageOffsets || source.page_geometries || []),
        doc_info: exportDocInfo(source, stamps, confirmedOnly, timestamp),
        stamps,
        deletedStamps: clone(source.deletedStamps || []),
        viewBoxes: clone(source.viewBoxes || []),
        basicDimensions: clone(source.basicDimensions || []),
        datumReferences: clone(source.datumReferences || []),
        inspection: filteredInspection(source.inspection, stamps),
        project_metadata: {
            analyzed_pages: serializableSet(source.currentAnalyzedPages),
            // Raw analysis cache may contain excluded dimensions; never leak it
            // through a confirmed-only archive.
            l1_cache_entries: confirmedOnly ? [] : serializableMap(source.l1Cache),
            clusters: clone(source.clusters || []),
        },
    };
}

function backendType(stamp) {
    if (stamp.aiData?._raw_type) return stamp.aiData._raw_type;
    if (stamp.source === 'surface_roughness') return 'surface_rough';
    if (stamp.source === 'gdt_line_frame') return 'gdt';
    return '';
}

function localTip(snapshot, stamp) {
    const tip = stamp.arrowTip || { absX: 0, absY: 0 };
    const page = pageNumber(stamp.pageIndex || 1, `stamp ${stamp.uuid || stamp.id || ''}`);
    if (!Array.isArray(snapshot.page_geometries) || snapshot.page_geometries.length === 0) {
        return { x: Number(tip.absX || 0), y: Number(tip.absY || 0) };
    }
    return documentGlobalPointToPageLocal(
        { x: Number(tip.absX || 0), y: Number(tip.absY || 0) },
        getPageGeometry(snapshot.page_geometries, page),
    );
}

function excelQuantity(ai) {
    const quantity = ai?.quantity;
    if (quantity === undefined || quantity === null || quantity === 1) return 1;
    if (!Number.isInteger(quantity) || quantity < 2 || quantity > 999) {
        throw new TypeError('Excel quantity 必须是 1..999 的整数');
    }
    return quantity;
}

export function buildExcelDimensions(snapshot) {
    if (snapshot?.schema !== ARCHIVE_SCHEMA) throw new TypeError('导出快照 schema 不兼容');
    const inspectionByStamp = snapshot.inspection?.byStampUuid || {};
    return clone(snapshot.stamps).sort((left, right) => {
        const pageDiff = Number(left.pageIndex || 1) - Number(right.pageIndex || 1);
        if (pageDiff !== 0) return pageDiff;
        return String(left.id || '').localeCompare(String(right.id || ''), undefined, {
            numeric: true,
            sensitivity: 'base',
        });
    }).flatMap(stamp => {
        const ai = stamp.aiData || {};
        const inspection = inspectionByStamp[stamp.uuid] || stamp.inspection || {};
        const tip = localTip(snapshot, stamp);
        const row = {
            uuid: stamp.uuid || '',
            no: String(stamp.id || ''),
            stamp_id: String(stamp.id || ''),
            page: pageNumber(stamp.pageIndex || 1, `stamp ${stamp.uuid || stamp.id || ''}`),
            nominal: ai.nominal || '',
            upper_tol: ai.upper_tol ?? null,
            lower_tol: ai.lower_tol ?? null,
            prefix: ai.prefix || ai.symbol || '',
            unit: ai.unit || '',
            type: backendType(stamp),
            dim_type: ai.type || stamp.dimType || '',
            source: stamp.source || '',
            datum: ai.datum || [ai.datum_1, ai.datum_2, ai.datum_3].filter(Boolean).join(' / '),
            datum_1: ai.datum_1 || '',
            datum_2: ai.datum_2 || '',
            datum_3: ai.datum_3 || '',
            modifier: ai.modifier || '',
            symbol_name: ai._symbol_name || '',
            symbol_unicode: ai._symbol_unicode || '',
            text: ai._raw_text || ai.nominal || '',
            is_key: stamp.isKeyOverride === true || stamp.isKey === true,
            raw_item_name: inspection.rawItemName || ai._raw_text || ai.nominal || '',
            source_file: inspection.sourceFile || '',
            match_status: inspection.matchStatus || 'not_imported',
            actuals: clone(inspection.actuals || []),
            result: inspection.result || '',
            equipment: inspection.equipment || '',
            bbox: { x: tip.x, y: tip.y, w: 0, h: 0 },
        };
        return Array.from(
            { length: excelQuantity(ai) },
            () => clone(row),
        );
    });
}

function validatePageOwned(items, name, maxPage) {
    for (const item of items) {
        const page = pageNumber(item?.pageIndex || item?.page || 1, name);
        if (page > maxPage) throw new RangeError(`${name} 页码 ${page} 超出文档页数 ${maxPage}`);
    }
}

export function parseArchivePayload(input, { pdfPageCount } = {}) {
    const data = typeof input === 'string' ? JSON.parse(input) : clone(input);
    if (!data || typeof data !== 'object') throw new TypeError('数据存档必须是对象');
    if (data.schema !== ARCHIVE_SCHEMA) {
        throw new TypeError(`数据存档 schema 不兼容: ${String(data.schema || 'missing')}`);
    }
    const pages = asArray(data.pages, 'pages').map((page, index) => pageNumber(page, `pages[${index}]`));
    if (pages.length === 0 || pages.some((page, index) => page !== index + 1)) {
        throw new TypeError('数据存档 pages 必须是从 1 开始的连续页集合');
    }
    const expectedPages = Number(pdfPageCount || data.doc_info?.totalPages || pages.length);
    if (!Number.isInteger(expectedPages) || expectedPages !== pages.length) {
        throw new RangeError(`数据存档页数 ${pages.length} 与 PDF page count ${expectedPages} 不一致`);
    }

    const stamps = asArray(data.stamps, 'stamps');
    const deletedStamps = asArray(data.deletedStamps || [], 'deletedStamps');
    const viewBoxes = asArray(data.viewBoxes || [], 'viewBoxes');
    const basicDimensions = asArray(data.basicDimensions || [], 'basicDimensions');
    const datumReferences = asArray(data.datumReferences || [], 'datumReferences');
    validatePageOwned(stamps, 'stamp', expectedPages);
    validatePageOwned(deletedStamps, 'deleted stamp', expectedPages);
    validatePageOwned(viewBoxes, 'view box', expectedPages);
    validatePageOwned(basicDimensions, 'basic dimension', expectedPages);
    validatePageOwned(datumReferences, 'datum reference', expectedPages);

    return {
        docInfo: clone(data.doc_info || {}),
        totalPages: expectedPages,
        pageOffsets: clone(data.page_geometries || []),
        stamps: clone(stamps),
        deletedStamps: clone(deletedStamps),
        viewBoxes: clone(viewBoxes),
        basicDimensions: clone(basicDimensions),
        datumReferences: clone(datumReferences),
        inspection: clone(data.inspection || {}),
        currentAnalyzedPages: new Set(clone(data.project_metadata?.analyzed_pages || [])),
        l1Cache: new Map(clone(data.project_metadata?.l1_cache_entries || [])),
        clusters: clone(data.project_metadata?.clusters || []),
    };
}

function normalizedDocInfo(info) {
    const copy = clone(info || {});
    delete copy.timestamp;
    delete copy.version;
    delete copy.totalStamps;
    delete copy.confirmedCount;
    delete copy.exportMode;
    return copy;
}

export function canonicalArchiveState(snapshot) {
    return {
        pages: clone(snapshot.pages || []),
        page_geometries: clone(snapshot.page_geometries || []),
        doc_info: normalizedDocInfo(snapshot.doc_info),
        stamps: clone(snapshot.stamps || []),
        deletedStamps: clone(snapshot.deletedStamps || []),
        viewBoxes: clone(snapshot.viewBoxes || []),
        basicDimensions: clone(snapshot.basicDimensions || []),
        datumReferences: clone(snapshot.datumReferences || []),
        inspection: clone(snapshot.inspection || {}),
        project_metadata: clone(snapshot.project_metadata || {}),
    };
}

async function bytesOf(value, label) {
    let bytes;
    if (value instanceof Uint8Array) bytes = value;
    else if (value instanceof ArrayBuffer) bytes = new Uint8Array(value);
    else if (typeof Blob !== 'undefined' && value instanceof Blob) {
        bytes = new Uint8Array(await value.arrayBuffer());
    } else if (typeof value === 'string') bytes = new TextEncoder().encode(value);
    else throw new TypeError(`${label} 产物类型无效`);
    if (bytes.byteLength === 0) throw new TypeError(`${label} 产物为空`);
    return bytes;
}

export async function sha256Hex(value) {
    const bytes = await bytesOf(value, 'SHA-256');
    const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

export async function buildArchiveArtifacts({
    snapshot,
    originalPdf,
    renderPdf,
    exportExcel,
    excelDimensions,
    names = {},
}) {
    if (snapshot?.schema !== ARCHIVE_SCHEMA) throw new TypeError('导出快照 schema 不兼容');
    const fileNames = {
        json: names.json || 'project.json',
        pdf: names.pdf || 'stamped.pdf',
        originalPdf: names.originalPdf || 'original.pdf',
        excel: names.excel || 'dimensions.xlsx',
    };
    const jsonBytes = await bytesOf(JSON.stringify(snapshot, null, 2), 'JSON');
    const originalBytes = await bytesOf(originalPdf, '原始 PDF');
    const stampedBytes = await bytesOf(await renderPdf(snapshot), '固化 PDF');
    const excelBytes = await bytesOf(
        await exportExcel(excelDimensions || buildExcelDimensions(snapshot), snapshot),
        'Excel',
    );
    const files = {
        [fileNames.json]: jsonBytes,
        [fileNames.pdf]: stampedBytes,
        [fileNames.originalPdf]: originalBytes,
        [fileNames.excel]: excelBytes,
    };
    const roles = {
        [fileNames.json]: 'project_json',
        [fileNames.pdf]: 'stamped_pdf',
        [fileNames.originalPdf]: 'original_pdf',
        [fileNames.excel]: 'dimensions_xlsx',
    };
    const manifestFiles = [];
    for (const [name, bytes] of Object.entries(files)) {
        manifestFiles.push({
            name,
            role: roles[name],
            bytes: bytes.byteLength,
            sha256: await sha256Hex(bytes),
        });
    }
    return {
        files,
        manifest: {
            schema: ARCHIVE_SCHEMA,
            page_count: snapshot.pages.length,
            item_count: snapshot.stamps.length,
            export_mode: snapshot.doc_info?.exportMode || 'all',
            files: manifestFiles,
        },
    };
}

export async function executeImportTransaction({ stage, commit, capture, rollback }) {
    const staged = await stage();
    const before = capture ? await capture() : undefined;
    try {
        await commit(staged);
    } catch (error) {
        if (rollback) await rollback(before, error);
        throw error;
    }
    return staged;
}
