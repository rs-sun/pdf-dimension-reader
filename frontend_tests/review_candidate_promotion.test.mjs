// Contract values and all measurement examples below are synthetic.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
    buildReviewConfirmationPayload,
    buildReviewProvenance,
    buildReviewedFormalDimension,
} from '../review_candidate_promotion.js';
import {
    buildStampDimensionProjection,
} from '../dimension_stamp_projection.js';
import {
    ARCHIVE_SCHEMA,
    buildExcelDimensions,
} from '../import_export_transaction.js';
import {
    confirmReviewCandidate,
    listReviewCandidateEntries,
} from '../review_candidates_v1.js';

function linearCandidate() {
    return {
        candidate_id: 'linear-1',
        bbox: { x: 10, y: 20, w: 30, h: 8 },
        type: 'linear',
        nominal: '234.567',
        symmetric_tolerance: '0.089',
        upper_tolerance: null,
        lower_tolerance: null,
        prefix: null,
        unit: null,
        review_status: 'pending',
        source_stage: 'L5_review_required',
        trace: { candidate_digest: 'linear-digest' },
    };
}

function gdtCandidate() {
    return {
        candidate_id: 'gdt-1',
        bbox: { x: 40, y: 50, w: 42, h: 10 },
        type: 'gdt',
        symbol_name: 'position',
        symbol_unicode: '⊕',
        tolerance_value: '0.05',
        datum_1: null,
        datum_2: null,
        datum_3: null,
        review_status: 'pending',
        source_stage: 'L5_review_required',
        trace: { candidate_digest: 'gdt-digest' },
    };
}

function entry(item) {
    return {
        item,
        envelope: {
            schema_version: 'review_candidates_v1',
            trace_id: 'trace-gdt-page-0',
            page_index: 0,
            coordinate_space: 'pdf_page_points_top_left',
            audit: {
                schema_version: 'review_candidates_audit_v1',
                semantic_digest: 'audit-digest',
            },
        },
    };
}

test('linear review promotion preserves the existing formal dimension transaction shape', () => {
    const candidate = linearCandidate();
    const values = {
        nominal: '234.568',
        symmetric_tolerance: '0.071',
        upper_tolerance: null,
        lower_tolerance: null,
    };

    assert.deepEqual(buildReviewedFormalDimension(candidate, values), {
        bbox: candidate.bbox,
        text: '234.568±0.071',
        nominal: '234.568',
        upper_tol: '+0.071',
        lower_tol: '-0.071',
        prefix: '',
        unit: '',
        type: 'linear',
        source: 'reviewed_vector',
        confidence: 'human_confirmed',
    });
    assert.deepEqual(
        buildReviewConfirmationPayload(candidate, values, 'formal-linear-1'),
        {
            formalStampUuid: 'formal-linear-1',
            nominal: '234.568',
            symmetric_tolerance: '0.071',
            upper_tolerance: null,
            lower_tolerance: null,
        },
    );
});

test('quantity keeps one candidate, confirmation, and stamp while Excel alone fans out rows', () => {
    const candidate = { ...linearCandidate(), quantity: 3 };
    const values = {
        nominal: '217.463',
        symmetric_tolerance: '0.083',
        upper_tolerance: null,
        lower_tolerance: null,
    };
    const dimension = buildReviewedFormalDimension(candidate, values);
    assert.deepEqual(dimension, {
        bbox: candidate.bbox,
        text: '3X 217.463±0.083',
        nominal: '217.463',
        upper_tol: '+0.083',
        lower_tol: '-0.083',
        prefix: '',
        unit: '',
        quantity: 3,
        type: 'linear',
        source: 'reviewed_vector',
        confidence: 'human_confirmed',
    });

    const payload = buildReviewConfirmationPayload(
        candidate,
        values,
        'formal-linear-quantity-1',
    );
    assert.deepEqual(payload, {
        formalStampUuid: 'formal-linear-quantity-1',
        nominal: '217.463',
        symmetric_tolerance: '0.083',
        upper_tolerance: null,
        lower_tolerance: null,
    });
    const workspace = {
        envelopes: [{
            ...entry(candidate).envelope,
            items: [candidate],
        }],
        decisions: {},
        selectedCandidateId: candidate.candidate_id,
    };
    const confirmed = confirmReviewCandidate(
        workspace,
        candidate.candidate_id,
        payload,
    );
    assert.equal(confirmed.envelopes[0].items.length, 1);
    assert.equal(Object.keys(confirmed.decisions).length, 1);

    const stamp = {
        uuid: 'formal-linear-quantity-1',
        id: '12',
        pageIndex: 1,
        confirmed: true,
        arrowTip: { absX: 12, absY: 18 },
        ...buildStampDimensionProjection(dimension),
    };
    const snapshot = {
        schema: ARCHIVE_SCHEMA,
        page_geometries: [],
        stamps: [stamp],
        inspection: { byStampUuid: {} },
    };
    assert.equal(snapshot.stamps.length, 1);
    assert.equal(stamp.aiData.quantity, 3);

    const rows = buildExcelDimensions(snapshot);
    assert.equal(rows.length, 3);
    assert.deepEqual(rows[0], rows[1]);
    assert.deepEqual(rows[1], rows[2]);
    assert.notEqual(rows[0], rows[1]);
    assert.deepEqual(
        rows.map(row => ({
            no: row.no,
            stamp_id: row.stamp_id,
            nominal: row.nominal,
            upper_tol: row.upper_tol,
            lower_tol: row.lower_tol,
            uuid: row.uuid,
        })),
        Array.from({ length: 3 }, () => ({
            no: '12',
            stamp_id: '12',
            nominal: '217.463',
            upper_tol: '+0.083',
            lower_tol: '-0.083',
            uuid: 'formal-linear-quantity-1',
        })),
    );
});

test('diameter bilateral promotion reaches stamp metadata and Excel projection', () => {
    const candidate = {
        ...linearCandidate(),
        nominal: '12',
        symmetric_tolerance: null,
        upper_tolerance: '+0.2',
        lower_tolerance: '-0.8',
        prefix: 'Ø',
    };
    const values = {
        nominal: '12.1',
        symmetric_tolerance: null,
        upper_tolerance: '+0.3',
        lower_tolerance: '-0.7',
    };

    const dimension = buildReviewedFormalDimension(candidate, values);
    assert.deepEqual(dimension, {
        bbox: candidate.bbox,
        text: 'Ø12.1 +0.3/-0.7',
        nominal: '12.1',
        upper_tol: '+0.3',
        lower_tol: '-0.7',
        prefix: 'Ø',
        unit: '',
        type: 'diameter',
        source: 'reviewed_vector',
        confidence: 'human_confirmed',
    });

    const stampProjection = buildStampDimensionProjection(dimension);
    assert.equal(stampProjection.aiData.prefix, 'Ø');
    assert.equal(stampProjection.aiData.unit, '');
    const [row] = buildExcelDimensions({
        schema: ARCHIVE_SCHEMA,
        page_geometries: [],
        stamps: [{
            uuid: 'diameter-bilateral-1',
            id: '9',
            pageIndex: 1,
            confirmed: true,
            arrowTip: { absX: 12, absY: 18 },
            ...stampProjection,
        }],
        inspection: { byStampUuid: {} },
    });
    assert.equal(row.prefix, 'Ø');
    assert.equal(row.unit, '');
    assert.equal(row.upper_tol, '+0.3');
    assert.equal(row.lower_tol, '-0.7');
    assert.equal(row.type, 'diameter');
});

test('radius, unilateral, and degree review values keep their formal semantics', () => {
    const cases = [
        {
            candidate: {
                ...linearCandidate(),
                nominal: '5',
                symmetric_tolerance: null,
                prefix: 'R',
            },
            values: {
                nominal: '5',
                symmetric_tolerance: null,
                upper_tolerance: null,
                lower_tolerance: null,
            },
            expected: {
                text: 'R5',
                upper_tol: '',
                lower_tol: '',
                prefix: 'R',
                unit: '',
                type: 'radius',
            },
        },
        {
            candidate: {
                ...linearCandidate(),
                nominal: '12',
                symmetric_tolerance: null,
                upper_tolerance: '+0.2',
                lower_tolerance: '0',
            },
            values: {
                nominal: '12',
                symmetric_tolerance: null,
                upper_tolerance: '+0.2',
                lower_tolerance: '0',
            },
            expected: {
                text: '12 +0.2/0',
                upper_tol: '+0.2',
                lower_tol: '0',
                prefix: '',
                unit: '',
                type: 'linear',
            },
        },
        {
            candidate: {
                ...linearCandidate(),
                nominal: '12',
                symmetric_tolerance: null,
                upper_tolerance: '0',
                lower_tolerance: '-1',
            },
            values: {
                nominal: '12',
                symmetric_tolerance: null,
                upper_tolerance: '0',
                lower_tolerance: '-1',
            },
            expected: {
                text: '12 0/-1',
                upper_tol: '0',
                lower_tol: '-1',
                prefix: '',
                unit: '',
                type: 'linear',
            },
        },
        {
            candidate: {
                ...linearCandidate(),
                nominal: '63.217',
                symmetric_tolerance: '0.089',
                unit: 'degree',
            },
            values: {
                nominal: '63.217',
                symmetric_tolerance: '0.089',
                upper_tolerance: null,
                lower_tolerance: null,
            },
            expected: {
                text: '63.217°±0.089°',
                upper_tol: '+0.089',
                lower_tol: '-0.089',
                prefix: '',
                unit: 'degree',
                type: 'angle',
            },
        },
        {
            candidate: {
                ...linearCandidate(),
                nominal: '63.217',
                symmetric_tolerance: null,
                upper_tolerance: '+0.5',
                lower_tolerance: '-1',
                unit: 'degree',
            },
            values: {
                nominal: '63.217',
                symmetric_tolerance: null,
                upper_tolerance: '+0.5',
                lower_tolerance: '-1',
            },
            expected: {
                text: '63.217° +0.5°/-1°',
                upper_tol: '+0.5',
                lower_tol: '-1',
                prefix: '',
                unit: 'degree',
                type: 'angle',
            },
        },
    ];

    for (const { candidate, values, expected } of cases) {
        const dimension = buildReviewedFormalDimension(candidate, values);
        assert.deepEqual(
            Object.fromEntries(
                Object.keys(expected).map(key => [key, dimension[key]]),
            ),
            expected,
        );
    }
});

test('linear promotion rejects non-string asymmetric tolerance values fail-closed', () => {
    const candidate = {
        ...linearCandidate(),
        symmetric_tolerance: null,
        upper_tolerance: '+0.2',
        lower_tolerance: '-0.8',
    };

    assert.throws(
        () => buildReviewedFormalDimension(candidate, {
            nominal: '12',
            symmetric_tolerance: null,
            upper_tolerance: 0.2,
            lower_tolerance: '-0.8',
        }),
        /asymmetric tolerance fields are invalid/,
    );
});

test('GD&T review promotion creates a reviewed_vector formal dimension with permanently blank datums', () => {
    const candidate = gdtCandidate();
    const values = { tolerance_value: '0.089' };
    const dimension = buildReviewedFormalDimension(candidate, values);

    assert.deepEqual(dimension, {
        bbox: candidate.bbox,
        text: '⊕0.089',
        nominal: '0.089',
        tolerance_value: '0.089',
        upper_tol: '',
        lower_tol: '',
        prefix: '⊕',
        symbol_name: 'position',
        symbol_unicode: '⊕',
        modifier: '',
        datum: '',
        datum_1: '',
        datum_2: '',
        datum_3: '',
        type: 'gdt',
        source: 'reviewed_vector',
        confidence: 'human_confirmed',
    });
    assert.deepEqual(
        buildReviewConfirmationPayload(candidate, values, 'formal-gdt-1'),
        {
            formalStampUuid: 'formal-gdt-1',
            tolerance_value: '0.089',
        },
    );
});

test('GD&T provenance retains envelope audit and read-only symbol identity without datum leakage', () => {
    const candidate = gdtCandidate();
    const reviewEntry = entry(candidate);
    const provenance = buildReviewProvenance(reviewEntry, {
        tolerance_value: '0.089',
    });

    assert.deepEqual(provenance, {
        schema_version: 'review_candidates_v1',
        candidate_id: 'gdt-1',
        trace_id: 'trace-gdt-page-0',
        page_index: 0,
        coordinate_space: 'pdf_page_points_top_left',
        original_candidate: candidate,
        reviewed_values: {
            symbol_name: 'position',
            symbol_unicode: '⊕',
            tolerance_value: '0.089',
            datum_1: '',
            datum_2: '',
            datum_3: '',
        },
        audit: reviewEntry.envelope.audit,
        action: 'human_confirmed',
    });

    candidate.symbol_name = 'mutated-after-confirm';
    reviewEntry.envelope.audit.semantic_digest = 'mutated-after-confirm';
    assert.equal(provenance.original_candidate.symbol_name, 'position');
    assert.equal(provenance.audit.semantic_digest, 'audit-digest');
});

test('GD&T confirmation payload preserves the workspace idempotency lock', () => {
    const candidate = gdtCandidate();
    const reviewEntry = entry(candidate);
    const workspace = {
        envelopes: [{
            ...reviewEntry.envelope,
            items: [candidate],
        }],
        decisions: {},
        selectedCandidateId: candidate.candidate_id,
    };
    const payload = buildReviewConfirmationPayload(
        candidate,
        { tolerance_value: '0.089' },
        'formal-gdt-1',
    );
    const confirmed = confirmReviewCandidate(
        workspace,
        candidate.candidate_id,
        payload,
    );
    const repeated = confirmReviewCandidate(
        confirmed,
        candidate.candidate_id,
        payload,
    );

    assert.deepEqual(repeated, confirmed);
    assert.deepEqual(listReviewCandidateEntries(confirmed)[0].decision, {
        status: 'confirmed',
        tolerance_value: '0.089',
        formal_stamp_uuid: 'formal-gdt-1',
    });
    assert.throws(
        () => confirmReviewCandidate(
            confirmed,
            candidate.candidate_id,
            { ...payload, formalStampUuid: 'formal-gdt-2' },
        ),
        /different formal stamp/,
    );
});

test('GD&T promotion rejects every datum write path and signed tolerance text', () => {
    const candidate = gdtCandidate();

    assert.throws(
        () => buildReviewedFormalDimension(candidate, {
            tolerance_value: '0.05',
            datum_1: 'A',
        }),
        /datum_1 is not editable/,
    );
    assert.throws(
        () => buildReviewedFormalDimension(
            { ...candidate, datum_2: 'B' },
            { tolerance_value: '0.05' },
        ),
        /datum_2 must remain null/,
    );
    assert.throws(
        () => buildReviewedFormalDimension(candidate, {
            tolerance_value: '+0.05',
        }),
        /unsigned decimal text/,
    );
});

test('GD&T confirmation passes through the dimensionsToStamps metadata projection and Excel rows', () => {
    const dimension = buildReviewedFormalDimension(gdtCandidate(), {
        tolerance_value: '0.089',
    });
    const stamp = {
        uuid: 'formal-gdt-1',
        id: '7',
        pageIndex: 1,
        confirmed: true,
        arrowTip: { absX: 12, absY: 18 },
        ...buildStampDimensionProjection(dimension),
    };
    const rows = buildExcelDimensions({
        schema: ARCHIVE_SCHEMA,
        page_geometries: [],
        stamps: [stamp],
        inspection: { byStampUuid: {} },
    });

    assert.deepEqual(rows.map(row => ({
        type: row.type,
        nominal: row.nominal,
        prefix: row.prefix,
        symbol_name: row.symbol_name,
        symbol_unicode: row.symbol_unicode,
        datum: row.datum,
        datum_1: row.datum_1,
        datum_2: row.datum_2,
        datum_3: row.datum_3,
        source: row.source,
    })), [{
        type: 'gdt',
        nominal: '0.089',
        prefix: '⊕',
        symbol_name: 'position',
        symbol_unicode: '⊕',
        datum: '',
        datum_1: '',
        datum_2: '',
        datum_3: '',
        source: 'reviewed_vector',
    }]);
});

test('dimensionsToStamps is wired to the shared executable metadata projection', async () => {
    const source = await readFile(new URL('../app_analysis.js', import.meta.url), 'utf8');
    assert.match(
        source,
        /import \{\s*buildStampDimensionProjection,\s*\} from '\.\/dimension_stamp_projection\.js';/,
    );
    assert.match(
        source,
        /Object\.assign\(s, buildStampDimensionProjection\(dim\)\)/,
    );
});
