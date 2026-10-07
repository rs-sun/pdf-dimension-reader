// inspection_matching.js — pure helpers for M3 measurement import matching.

export function canonicalInspectionNo(value) {
    if (value === null || value === undefined || value === '') return '';
    const text = String(value).trim();
    const subNumber = text.match(/^(\d+)\s*-\s*(\d+)\s*$/);
    if (subNumber) {
        return `${parseInt(subNumber[1], 10)}-${parseInt(subNumber[2], 10)}`;
    }
    const n = Number(value);
    if (Number.isFinite(n)) return String(Math.trunc(n));
    const match = text.match(/^\d+/);
    return match ? String(parseInt(match[0], 10)) : '';
}

function _normalizedInspectionResult(result) {
    const s = String(result || '').trim().toUpperCase();
    if (s === 'NG') return 'NOK';
    if (['OK', 'NOK', 'CRITICAL', 'UNMEASURED', 'UNKNOWN'].includes(s)) return s;
    if (s === '未测量') return 'UNMEASURED';
    if (s === '临界') return 'CRITICAL';
    return s || 'UNKNOWN';
}

function _normalizeMatchText(value) {
    return String(value || '').normalize('NFKC').toUpperCase();
}

function _addKeyVariant(keys, rawToken) {
    const token = String(rawToken || '').replace(/[^A-Z0-9]/g, '');
    if (!token) return;

    if (/^F\d{2}[A-Z0-9]{5,}$/.test(token)) {
        keys.add(token);
        if (token.length >= 11 && /\d{2}$/.test(token) && /[A-Z]/.test(token[token.length - 3])) {
            keys.add(token.slice(0, -2));
        }
        const shortSuffix = token.match(/[A-Z]\d{2,4}$/);
        if (shortSuffix) keys.add(shortSuffix[0]);
        return;
    }

    if (/^[A-Z]{1,3}\d{2,4}[A-Z]?$/.test(token)) {
        keys.add(token);
    }
}

export function extractInspectionMatchKeys(...values) {
    const keys = new Set();
    const visit = (value) => {
        if (value === null || value === undefined || value === '') return;
        if (Array.isArray(value)) {
            value.forEach(visit);
            return;
        }
        const text = _normalizeMatchText(value);
        text.replace(/[^A-Z0-9]+/g, ' ')
            .split(/\s+/)
            .filter(Boolean)
            .forEach(token => _addKeyVariant(keys, token));
    };
    values.forEach(visit);
    return [...keys];
}

export function sourceKeysForWorkbook(workbook) {
    const meta = workbook?.meta || {};
    return extractInspectionMatchKeys(
        workbook?.source_file,
        meta.calypso_file,
        meta.measurement_plan,
        meta.part_no
    );
}

export function inspectionKeysCompatible(leftKeys, rightKeys) {
    const left = Array.from(leftKeys || []).filter(Boolean);
    const right = Array.from(rightKeys || []).filter(Boolean);
    if (!left.length || !right.length) return false;

    for (const a of left) {
        for (const b of right) {
            if (a === b) return true;
            const minLen = Math.min(a.length, b.length);
            const lenDelta = Math.abs(a.length - b.length);
            if (minLen >= 8 && lenDelta <= 2 && (a.startsWith(b) || b.startsWith(a))) {
                return true;
            }
        }
    }
    return false;
}

function _inspectionSummary(inspection) {
    return {
        result: inspection.result,
        matchStatus: inspection.matchStatus,
        rawItemName: inspection.rawItemName || '',
        sourceFile: inspection.sourceFile || '',
        isKey: !!inspection.isKey,
    };
}

function _recordSourceKeys(record, sourceKeysByFile) {
    const fromWorkbook = sourceKeysByFile.get(record.source_file) || [];
    const fromRecord = extractInspectionMatchKeys(record.source_file);
    return [...new Set([...fromWorkbook, ...fromRecord])];
}

function _aggregateActuals(records) {
    const firstWithActuals = records.find(r => Array.isArray(r.actuals) && r.actuals.length);
    return firstWithActuals ? firstWithActuals.actuals : [];
}

function _stampIsKey(stamp, resolveIsKey, records) {
    if (stamp?.isKeyOverride === true || stamp?.isKeyOverride === false) {
        return stamp.isKeyOverride;
    }
    const recordIsKey = records.some(r => r.is_key === true || r.is_key === 'Y');
    const currentIsKey = typeof resolveIsKey === 'function'
        ? resolveIsKey(stamp)
        : !!(stamp?.isKeyOverride ?? stamp?.is_key);
    return !!(currentIsKey || recordIsKey);
}

function _aggregateInspectionForStamp(stamp, records, resolveIsKey) {
    const normalized = records.map(r => ({ ...r, result: _normalizedInspectionResult(r.result) }));
    let result = 'UNMEASURED';
    if (normalized.some(r => r.result === 'NOK')) result = 'NOK';
    else if (normalized.some(r => r.result === 'CRITICAL')) result = 'CRITICAL';
    else if (normalized.some(r => r.result === 'OK')) result = 'OK';
    else if (normalized.some(r => r.result === 'UNKNOWN')) result = 'UNKNOWN';

    const first = normalized[0] || {};
    return {
        stampUuid: stamp.uuid,
        stampId: stamp.id,
        no: canonicalInspectionNo(stamp.id),
        result,
        records: normalized,
        actuals: _aggregateActuals(normalized),
        utl: first.utl ?? null,
        ltl: first.ltl ?? null,
        equipment: first.equipment || '',
        rawItemName: first.raw_item_name || first.item_text || '',
        sourceFile: first.source_file || '',
        matchStatus: normalized.length ? 'matched_by_no' : 'not_found',
        isKey: _stampIsKey(stamp, resolveIsKey, normalized),
    };
}

export function buildInspectionImportState(importResult, stamps, options = {}) {
    const sourceKeysByFile = new Map();
    (importResult.workbooks || []).forEach(workbook => {
        sourceKeysByFile.set(workbook.source_file, sourceKeysForWorkbook(workbook));
    });

    const docKeys = extractInspectionMatchKeys(options.docFilename || '');
    const records = (importResult.records || []).map(r => {
        const sourceMatchKeys = _recordSourceKeys(r, sourceKeysByFile);
        return {
            ...r,
            sourceMatchKeys,
            result: _normalizedInspectionResult(r.result),
            matchStatus: 'unmatched',
        };
    });

    const hasCompatibleDocumentRecords = !!docKeys.length
        && records.some(record => inspectionKeysCompatible(docKeys, record.sourceMatchKeys));
    const recordAllowedForDocument = (record) => {
        if (!hasCompatibleDocumentRecords) return true;
        if (!record.sourceMatchKeys.length) return false;
        return inspectionKeysCompatible(docKeys, record.sourceMatchKeys);
    };

    const byNo = new Map();
    records.forEach(record => {
        if (!recordAllowedForDocument(record)) {
            record.matchStatus = 'skipped_by_document';
            return;
        }
        const key = canonicalInspectionNo(record.no);
        if (!key) return;
        if (!byNo.has(key)) byNo.set(key, []);
        byNo.get(key).push(record);
    });

    const matchedRecordIds = new Set();
    const byStampUuid = {};
    stamps.forEach(stamp => {
        const matches = byNo.get(canonicalInspectionNo(stamp.id)) || [];
        matches.forEach(record => {
            matchedRecordIds.add(record.record_id);
            record.matchStatus = 'matched_by_no';
            record.stampUuid = stamp.uuid;
        });
        const inspection = _aggregateInspectionForStamp(stamp, matches, options.resolveIsKey);
        byStampUuid[stamp.uuid] = inspection;
    });

    const unmatchedRecords = records.filter(record => record.matchStatus === 'unmatched'
        && !matchedRecordIds.has(record.record_id));
    const skippedRecordCount = records.filter(record => record.matchStatus === 'skipped_by_document').length;
    const keyNokCount = stamps.filter(stamp => {
        const inspection = byStampUuid[stamp.uuid];
        return inspection?.isKey && inspection?.result === 'NOK';
    }).length;

    return {
        active: true,
        importedAt: new Date().toISOString(),
        sourceFiles: (importResult.workbooks || []).map(w => w.source_file),
        docMatchKeys: docKeys,
        records,
        byStampUuid,
        unmatchedRecords,
        warnings: importResult.warnings || [],
        stats: {
            fileCount: importResult.stats?.file_count || (importResult.workbooks || []).length,
            recordCount: records.length,
            matchedStampCount: Object.values(byStampUuid).filter(x => x.matchStatus === 'matched_by_no').length,
            unmatchedRecordCount: unmatchedRecords.length,
            skippedRecordCount,
            nokCount: Object.values(byStampUuid).filter(x => x.result === 'NOK').length,
            unmeasuredCount: Object.values(byStampUuid).filter(x => x.result === 'UNMEASURED').length,
            keyNokCount,
        },
        filter: 'all',
        selectedRecordId: null,
    };
}
