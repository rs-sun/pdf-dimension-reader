import test from 'node:test';
import assert from 'node:assert/strict';

import {
    ARCHIVE_SCHEMA,
    buildArchiveArtifacts,
    buildExportSnapshot,
    buildExcelDimensions,
    canonicalArchiveState,
    executeImportTransaction,
    parseArchivePayload,
} from '../import_export_transaction.js';

// All drawing metadata, dimensions and results below are synthetic test data.
function projectState() {
    return {
        docInfo: { filename: 'six-pages', version: 7, drawingNo: 'synthetic-drawing' },
        totalPages: 6,
        pageOffsets: Array.from({ length: 6 }, (_, index) => ({
            pageIndex: index + 1,
            start: index * 100,
            width: 200,
            height: 100,
            sourceWidth: 200,
            sourceHeight: 100,
            rotation: 0,
        })),
        stamps: [
            {
                uuid: 'page-1-confirmed', id: '1', pageIndex: 1, confirmed: true,
                arrowTip: { absX: 15, absY: 20 }, circleCenter: { absX: 8, absY: 10 },
                aiData: {
                    nominal: '10.25',
                    upper_tol: '+0.1',
                    lower_tol: '-0.1',
                    prefix: 'Ø',
                    unit: '',
                },
            },
            {
                uuid: 'page-2-edited', id: '48-1', pageIndex: 2, confirmed: true,
                arrowTip: { absX: 25, absY: 140 }, circleCenter: { absX: 18, absY: 130 },
                aiData: {
                    nominal: '12.50',
                    prefix: '',
                    unit: 'degree',
                    remarks: 'front-end edit',
                },
            },
            {
                uuid: 'page-3-unconfirmed', id: '3', pageIndex: 3, confirmed: false,
                arrowTip: { absX: 35, absY: 260 }, circleCenter: { absX: 28, absY: 250 },
                aiData: { nominal: '99' },
            },
        ],
        deletedStamps: [{ uuid: 'deleted-manual', id: 'D1', pageIndex: 4, confirmed: true }],
        viewBoxes: [{ id: 'view-2', pageIndex: 2, x: 5, y: 110, w: 80, h: 50, isEdited: true }],
        basicDimensions: [{ id: 'basic-5', pageIndex: 5, text: '25', bbox: { x: 1, y: 410, w: 5, h: 5 } }],
        datumReferences: [{ id: 'datum-6', pageIndex: 6, text: 'A', bbox: { x: 1, y: 510, w: 5, h: 5 } }],
        inspection: {
            active: true,
            records: [
                { record_id: 'p2-record', stampUuid: 'page-2-edited' },
                { record_id: 'p3-record', stampUuid: 'page-3-unconfirmed' },
            ],
            unmatchedRecords: [],
            stats: { recordCount: 2, matchedStampCount: 2 },
            byStampUuid: {
                'page-2-edited': { result: 'OK', actuals: [12.49], matchStatus: 'matched_by_no' },
                'page-3-unconfirmed': { result: 'NOK', actuals: [100], matchStatus: 'matched_by_no' },
            },
        },
        currentAnalyzedPages: new Set([1, 2, 3, 4, 5, 6]),
        l1Cache: new Map([[2, [{ text: 'cached' }]]]),
        clusters: [{ id: 'cluster-2', pageIndex: 2 }],
        selectedUUIDs: ['page-2-edited'],
        historyStack: [{ old: true }],
        historyIndex: 0,
    };
}

test('one confirmed-only full-document snapshot drives JSON and Excel item/page identity', () => {
    const snapshot = buildExportSnapshot(projectState(), {
        confirmedOnly: true,
        timestamp: '2026-07-13T08:09:10.000Z',
    });
    const excel = buildExcelDimensions(snapshot);

    assert.equal(snapshot.schema, ARCHIVE_SCHEMA);
    assert.deepEqual(snapshot.pages, [1, 2, 3, 4, 5, 6]);
    assert.deepEqual(snapshot.stamps.map(item => item.uuid), ['page-1-confirmed', 'page-2-edited']);
    assert.deepEqual(excel.map(item => item.uuid), ['page-1-confirmed', 'page-2-edited']);
    assert.deepEqual(excel.map(item => item.page), [1, 2]);
    assert.deepEqual(excel.map(item => item.stamp_id), ['1', '48-1']);
    assert.deepEqual(
        excel.map(item => ({ prefix: item.prefix, unit: item.unit })),
        [{ prefix: 'Ø', unit: '' }, { prefix: '', unit: 'degree' }],
    );
    assert.equal(excel[1].nominal, '12.50');
    assert.equal(excel[1].result, 'OK');
    assert.equal(excel[1].bbox.y, 40, 'page-2 coordinates are page-local in Excel');
    assert.deepEqual(Object.keys(snapshot.inspection.byStampUuid), ['page-2-edited']);
    assert.deepEqual(snapshot.inspection.records.map(record => record.record_id), ['p2-record']);
    assert.deepEqual(snapshot.project_metadata.l1_cache_entries, []);
    assert.equal(snapshot.doc_info.totalStamps, 2);
    assert.equal(snapshot.doc_info.exportMode, 'confirmed_only');
});

test('implicit and explicit quantity one both preserve one byte-identical Excel row', () => {
    const snapshot = buildExportSnapshot(projectState(), {
        confirmedOnly: true,
        timestamp: 'quantity-one-control',
    });
    snapshot.stamps = [snapshot.stamps[0]];
    const implicit = buildExcelDimensions(snapshot);

    const explicitSnapshot = structuredClone(snapshot);
    explicitSnapshot.stamps[0].aiData.quantity = 1;
    const explicit = buildExcelDimensions(explicitSnapshot);

    assert.equal(implicit.length, 1);
    assert.deepEqual(explicit, implicit);
    assert.equal(JSON.stringify(explicit), JSON.stringify(implicit));
});

test('quantity survives archive import while the project still owns one stamp', () => {
    const snapshot = buildExportSnapshot(projectState(), {
        confirmedOnly: true,
        timestamp: 'quantity-round-trip',
    });
    snapshot.stamps = [snapshot.stamps[0]];
    snapshot.stamps[0].aiData.quantity = 3;

    const imported = parseArchivePayload(JSON.stringify(snapshot), {
        pdfPageCount: 6,
    });

    assert.equal(imported.stamps.length, 1);
    assert.equal(imported.stamps[0].aiData.quantity, 3);
    assert.equal(buildExcelDimensions(snapshot).length, 3);
});

test('Excel fan-out rejects malformed serialized quantity', () => {
    const snapshot = buildExportSnapshot(projectState(), {
        confirmedOnly: true,
        timestamp: 'invalid-quantity',
    });
    snapshot.stamps = [snapshot.stamps[0]];

    for (const quantity of [0, -1, 1000, 2.5, '3', true]) {
        snapshot.stamps[0].aiData.quantity = quantity;
        assert.throws(
            () => buildExcelDimensions(snapshot),
            /quantity|倍数/i,
        );
    }
});

test('archive payload validates schema/page references and restores all project-owned state', () => {
    const snapshot = buildExportSnapshot(projectState(), { timestamp: 'fixed' });
    const parsed = parseArchivePayload(JSON.stringify(snapshot), { pdfPageCount: 6 });

    assert.equal(parsed.docInfo.drawingNo, 'synthetic-drawing');
    assert.equal(parsed.stamps[1].aiData.remarks, 'front-end edit');
    assert.equal(parsed.stamps[0].aiData.prefix, 'Ø');
    assert.equal(parsed.stamps[1].aiData.unit, 'degree');
    assert.equal(parsed.stamps[1].confirmed, true);
    assert.equal(parsed.deletedStamps[0].uuid, 'deleted-manual');
    assert.equal(parsed.viewBoxes[0].isEdited, true);
    assert.equal(parsed.basicDimensions[0].id, 'basic-5');
    assert.equal(parsed.datumReferences[0].id, 'datum-6');
    assert.deepEqual([...parsed.currentAnalyzedPages], [1, 2, 3, 4, 5, 6]);

    assert.throws(
        () => parseArchivePayload(JSON.stringify({ ...snapshot, schema: 'future/v99' }), { pdfPageCount: 6 }),
        /不兼容|incompatible/i,
    );
    assert.throws(
        () => parseArchivePayload(JSON.stringify(snapshot), { pdfPageCount: 5 }),
        /页数|page count/i,
    );
});

test('failed import staging is side-effect free; successful staging commits exactly once', async () => {
    const state = projectState();
    const before = structuredClone(state);
    let commits = 0;

    await assert.rejects(() => executeImportTransaction({
        stage: async () => { throw new Error('corrupt zip'); },
        commit: async () => { commits += 1; },
    }), /corrupt zip/);
    assert.deepEqual(state, before);
    assert.equal(commits, 0);

    const staged = { project: { stamps: [{ uuid: 'replacement' }] } };
    const result = await executeImportTransaction({
        stage: async () => staged,
        commit: async value => {
            commits += 1;
            state.stamps = value.project.stamps;
        },
    });
    assert.equal(result, staged);
    assert.equal(commits, 1);
    assert.deepEqual(state.stamps, [{ uuid: 'replacement' }]);
});

test('all required artifacts are fail-closed and manifest hashes every emitted file', async () => {
    const snapshot = buildExportSnapshot(projectState(), { timestamp: 'fixed' });
    const originalPdf = new Uint8Array([37, 80, 68, 70]);

    await assert.rejects(() => buildArchiveArtifacts({
        snapshot,
        originalPdf,
        renderPdf: async () => new Uint8Array([1, 2, 3]),
        exportExcel: async () => null,
    }), /Excel/);

    const archive = await buildArchiveArtifacts({
        snapshot,
        originalPdf,
        renderPdf: async () => new Uint8Array([1, 2, 3]),
        exportExcel: async dims => {
            assert.equal(dims.length, 3);
            return new Uint8Array([4, 5, 6]);
        },
        names: {
            json: 'archive.json',
            pdf: 'stamped.pdf',
            originalPdf: 'original.pdf',
            excel: 'dimensions.xlsx',
        },
    });

    assert.equal(archive.manifest.schema, ARCHIVE_SCHEMA);
    assert.equal(archive.manifest.page_count, 6);
    assert.equal(archive.manifest.item_count, 3);
    assert.deepEqual(Object.keys(archive.files).sort(), [
        'archive.json', 'dimensions.xlsx', 'original.pdf', 'stamped.pdf',
    ]);
    assert.deepEqual(
        archive.manifest.files.map(file => file.name).sort(),
        Object.keys(archive.files).sort(),
    );
    assert.ok(archive.manifest.files.every(file => /^[a-f0-9]{64}$/.test(file.sha256)));
});

test('export-import-export is canonically identical for editable project state', () => {
    const first = buildExportSnapshot(projectState(), { timestamp: 'first' });
    const imported = parseArchivePayload(JSON.stringify(first), { pdfPageCount: 6 });
    const second = buildExportSnapshot({ ...projectState(), ...imported }, { timestamp: 'second' });

    assert.deepEqual(canonicalArchiveState(second), canonicalArchiveState(first));
});
