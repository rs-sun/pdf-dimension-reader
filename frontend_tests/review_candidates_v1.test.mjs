import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import {
    REVIEW_CANDIDATES_ROOT_KEY,
    confirmReviewCandidate,
    dismissReviewCandidate,
    editReviewCandidate,
    executeReviewCandidatesTransaction,
    ingestReviewCandidatesV1,
    listReviewCandidateEntries,
    pendingReviewCandidateCount,
    validateReviewCandidatesV1,
} from '../review_candidates_v1.js';
import { buildExportSnapshot } from '../import_export_transaction.js';

const fixture = JSON.parse(await readFile(
    new URL('../contracts/review_candidates_v1.canonical.json', import.meta.url),
    'utf8',
));

function emptyWorkspace() {
    return { envelopes: [], decisions: {}, selectedCandidateId: null };
}

function pythonFloatLiteral(value) {
    if (Object.is(value, -0)) return '-0.0';
    const sign = value < 0 ? '-' : '';
    const match = Math.abs(value).toExponential().match(
        /^(\d)(?:\.(\d+))?e([+\-])(\d+)$/,
    );
    assert.ok(match);
    const [, leading, fraction = '', exponentSign, exponentDigits] = match;
    const exponent = Number(`${exponentSign}${exponentDigits}`);
    if (exponent < -4 || exponent >= 16) {
        const mantissa = fraction ? `${leading}.${fraction}` : leading;
        return `${sign}${mantissa}e${exponent < 0 ? '-' : '+'}${String(Math.abs(exponent)).padStart(2, '0')}`;
    }
    const digits = `${leading}${fraction}`;
    const decimalIndex = exponent + 1;
    if (decimalIndex <= 0) return `${sign}0.${'0'.repeat(-decimalIndex)}${digits}`;
    if (decimalIndex >= digits.length) {
        return `${sign}${digits}${'0'.repeat(decimalIndex - digits.length)}.0`;
    }
    return `${sign}${digits.slice(0, decimalIndex)}.${digits.slice(decimalIndex)}`;
}

function bboxJson(bbox) {
    return `{"h":${pythonFloatLiteral(bbox.h)},"w":${pythonFloatLiteral(bbox.w)},` +
        `"x":${pythonFloatLiteral(bbox.x)},"y":${pythonFloatLiteral(bbox.y)}}`;
}

function digestText(value) {
    return createHash('sha256').update(value, 'utf8').digest('hex');
}

function candidateDigest(value, item) {
    const quantityField = Object.hasOwn(item, 'quantity')
        ? `"quantity":${JSON.stringify(item.quantity)},`
        : '';
    const syntaxSuffix = Object.hasOwn(item, 'syntax')
        ? `,"syntax":${JSON.stringify(item.syntax)}`
        : '';
    const syntaxField = Object.hasOwn(item, 'syntax')
        ? `"syntax":${JSON.stringify(item.syntax)},`
        : '';
    const extended = [
        item.upper_tolerance,
        item.lower_tolerance,
        item.prefix,
        item.unit,
    ].some(field => field !== null);
    if (extended) {
        return digestText(
            `{"bbox":${bboxJson(item.bbox)},` +
            `"lower_tolerance":${JSON.stringify(item.lower_tolerance)},` +
            `"nominal":${JSON.stringify(item.nominal)},` +
            `"page_index":${value.page_index},` +
            `"physical_core_digest":${JSON.stringify(item.trace.physical_core_digest)},` +
            `"prefix":${JSON.stringify(item.prefix)},` +
            quantityField +
            `"schema_version":"review_candidates_v1",` +
            `"symmetric_tolerance":${JSON.stringify(item.symmetric_tolerance)},` +
            `${syntaxField}"type":"linear","unit":${JSON.stringify(item.unit)},` +
            `"upper_tolerance":${JSON.stringify(item.upper_tolerance)}}`,
        );
    }
    return digestText(
        `{"bbox":${bboxJson(item.bbox)},"nominal":${JSON.stringify(item.nominal)},` +
        `"page_index":${value.page_index},` +
        `"physical_core_digest":${JSON.stringify(item.trace.physical_core_digest)},` +
        quantityField +
        `"schema_version":"review_candidates_v1",` +
        `"symmetric_tolerance":${JSON.stringify(item.symmetric_tolerance)}` +
        `${syntaxSuffix}}`,
    );
}

function semanticDigest(value) {
    const items = value.items.map(item => (
        (() => {
            const quantityField = Object.hasOwn(item, 'quantity')
                ? `"quantity":${JSON.stringify(item.quantity)},`
                : '';
            const syntaxSuffix = Object.hasOwn(item, 'syntax')
                ? `,"syntax":${JSON.stringify(item.syntax)}`
                : '';
            const syntaxField = Object.hasOwn(item, 'syntax')
                ? `"syntax":${JSON.stringify(item.syntax)},`
                : '';
            return [
            item.upper_tolerance,
            item.lower_tolerance,
            item.prefix,
            item.unit,
        ].some(field => field !== null)
            ? (
                `{"bbox":${bboxJson(item.bbox)},` +
                `"candidate_id":${JSON.stringify(item.candidate_id)},` +
                `"lower_tolerance":${JSON.stringify(item.lower_tolerance)},` +
                `"nominal":${JSON.stringify(item.nominal)},` +
                `"physical_core_digest":${JSON.stringify(item.trace.physical_core_digest)},` +
                `"prefix":${JSON.stringify(item.prefix)},` +
                quantityField +
                `"review_status":${JSON.stringify(item.review_status)},` +
                `"source_stage":${JSON.stringify(item.source_stage)},` +
                `"symmetric_tolerance":${JSON.stringify(item.symmetric_tolerance)},` +
                `${syntaxField}"type":"linear","unit":${JSON.stringify(item.unit)},` +
                `"upper_tolerance":${JSON.stringify(item.upper_tolerance)}}`
            )
            : (
                `{"bbox":${bboxJson(item.bbox)},` +
                `"candidate_id":${JSON.stringify(item.candidate_id)},` +
                `"nominal":${JSON.stringify(item.nominal)},` +
                `"physical_core_digest":${JSON.stringify(item.trace.physical_core_digest)},` +
                quantityField +
                `"review_status":${JSON.stringify(item.review_status)},` +
                `"source_stage":${JSON.stringify(item.source_stage)},` +
                `"symmetric_tolerance":${JSON.stringify(item.symmetric_tolerance)}` +
                `${syntaxSuffix}}`
            )
        })()
    )).join(',');
    return digestText(
        `{"coordinate_space":"pdf_page_points_top_left","items":[${items}],` +
        `"page_index":${value.page_index},"schema_version":"review_candidates_v1"}`,
    );
}

function refreshFrozenDigests(value, physicalDigit = null) {
    for (const item of value.items) {
        if (physicalDigit !== null) {
            item.trace.physical_core_digest = physicalDigit.repeat(64);
        }
        item.trace.candidate_digest = candidateDigest(value, item);
        item.candidate_id = `rcv1_${item.trace.candidate_digest.slice(0, 24)}`;
    }
    value.audit.semantic_digest = semanticDigest(value);
    return value;
}

function pageFixture(pageIndex, digit = '5') {
    const value = structuredClone(fixture);
    value.page_index = pageIndex;
    value.trace_id = `trace-page-${pageIndex}`;
    value.audit.chain.page_index = pageIndex;
    value.audit.chain.trace_id = value.trace_id;
    return refreshFrozenDigests(value, digit);
}

function frontendCompatibleLinearFixture() {
    const value = structuredClone(fixture);
    assert.deepEqual(
        validateReviewCandidatesV1(value, { expectedPageIndex: 0 }),
        [],
        'the generation-gate fixture must match the current frontend contract',
    );
    return value;
}

function extendedLinearFixture(overrides = {}) {
    const value = structuredClone(fixture);
    Object.assign(value.items[0], overrides);
    return refreshFrozenDigests(value);
}

function gdtFixture() {
    const value = structuredClone(fixture);
    value.items = [{
        candidate_id: 'rcv1_6ac682237e5ad68f1a0d1b44',
        bbox: { x: 120, y: 30, w: 72, h: 18 },
        type: 'gdt',
        symbol_name: 'position',
        symbol_unicode: '⊕',
        tolerance_value: '0.05',
        datum_1: null,
        datum_2: null,
        datum_3: null,
        review_status: 'pending',
        source_stage: 'L5_review_required',
        trace: {
            frame_digest: 'c'.repeat(64),
            decimal_quad_id: 'decimal_quad_001',
            symbol_method: 'vector_inversion',
            symbol_confidence: 0.91,
            symbol_compartment_index: 0,
            tolerance_compartment_index: 1,
            reader_phrase_id: 'gdt_tolerance_phrase_001',
            template_content_sha256: 'a'.repeat(64),
            candidate_digest: '6ac682237e5ad68f1a0d1b44591a1cdbf7d0e57688ab9ac5ff8105922ff07105',
        },
    }];
    value.audit.template_identity.content_sha256 = 'a'.repeat(64);
    value.audit.semantic_digest = 'b211e1b245cdffb2cffe73fdc990b52873bf3d8da60a032c82bbe63f8832f5bc';
    return value;
}

test('canonical fixture is accepted as the sole exact review_candidates_v1 shape', () => {
    assert.equal(REVIEW_CANDIDATES_ROOT_KEY, 'review_candidates_v1');
    assert.equal(
        fixture.items[0].trace.candidate_digest,
        'de4112e365c27fad4f94a8f189087fb2a138a3705580cf92f073fbf54f80697a',
    );
    assert.equal(
        fixture.audit.semantic_digest,
        '283081a148ae19c1fc2b500f120f7b9846bcb2c7b4ec75a3b13307784a945461',
    );
    assert.deepEqual(validateReviewCandidatesV1(fixture, { expectedPageIndex: 0 }), []);

    const extraTopLevel = structuredClone(fixture);
    extraTopLevel.items_v0 = [];
    assert.ok(validateReviewCandidatesV1(extraTopLevel).includes('envelope_keys_invalid'));

    const extraItem = structuredClone(fixture);
    extraItem.items[0].legacy_nominal = extraItem.items[0].nominal;
    assert.ok(validateReviewCandidatesV1(extraItem).includes('item_keys_invalid'));

    const missingType = structuredClone(fixture);
    delete missingType.items[0].type;
    assert.ok(validateReviewCandidatesV1(missingType).includes('item_type_invalid'));
});

test('bilateral linear candidate is accepted with frozen widened digests', () => {
    const value = extendedLinearFixture({
        symmetric_tolerance: null,
        upper_tolerance: '+0.2',
        lower_tolerance: '-0.8',
    });

    assert.deepEqual(validateReviewCandidatesV1(value), []);
    const [entry] = listReviewCandidateEntries(
        ingestReviewCandidatesV1(emptyWorkspace(), value),
    );
    assert.equal(entry.upper_tolerance, '+0.2');
    assert.equal(entry.lower_tolerance, '-0.8');
    assert.equal(entry.symmetric_tolerance, null);
    assert.equal(entry.prefix, null);
    assert.equal(entry.unit, null);
});

test('supported prefix, degree, and unilateral candidate matrix is accepted', () => {
    const cases = [
        { prefix: 'Ø' },
        { prefix: 'R' },
        { unit: 'degree' },
        {
            symmetric_tolerance: null,
            upper_tolerance: '+0.2',
            lower_tolerance: '0',
        },
        {
            symmetric_tolerance: null,
            upper_tolerance: '0',
            lower_tolerance: '-0.8',
        },
    ];

    for (const fields of cases) {
        const value = extendedLinearFixture(fields);
        assert.deepEqual(
            validateReviewCandidatesV1(value),
            [],
            JSON.stringify(fields),
        );
    }
});

test('widened linear candidate rejects invalid values and remains closed', () => {
    const cases = [
        [item => { item.prefix = '⌀'; }, 'item_prefix_invalid'],
        [item => { item.unit = 'radian'; }, 'item_unit_invalid'],
        [
            item => {
                item.prefix = 'R';
                item.unit = 'degree';
            },
            'item_prefix_unit_invalid',
        ],
        [
            item => { item.symmetric_tolerance = '0.1'; },
            'item_tolerance_invalid',
        ],
        [item => { item.upper_tolerance = null; }, 'item_tolerance_invalid'],
        [item => { item.legacy_upper_tol = '+0.2'; }, 'item_keys_invalid'],
    ];

    for (const [mutate, reason] of cases) {
        const value = extendedLinearFixture({
            symmetric_tolerance: null,
            upper_tolerance: '+0.2',
            lower_tolerance: '-0.8',
            prefix: 'Ø',
        });
        mutate(value.items[0]);
        assert.ok(validateReviewCandidatesV1(value).includes(reason), reason);
    }
});

test('optional syntax labels are digest-bound while the legacy fixture stays byte-stable', () => {
    for (const syntax of [
        'ok',
        'incomplete_separator',
        'unread_glyph_present',
        'invalid_dimension_syntax',
    ]) {
        const value = extendedLinearFixture({ syntax });
        assert.deepEqual(
            validateReviewCandidatesV1(value),
            [],
            syntax,
        );
    }

    const stale = extendedLinearFixture({ syntax: 'incomplete_separator' });
    stale.items[0].syntax = 'unread_glyph_present';
    const staleReasons = validateReviewCandidatesV1(stale);
    assert.ok(staleReasons.includes('item_candidate_digest_invalid'));
    assert.ok(staleReasons.includes('audit_semantic_digest_invalid'));

    const unknown = extendedLinearFixture({ syntax: 'looks_plausible' });
    assert.ok(
        validateReviewCandidatesV1(unknown).includes('item_syntax_invalid'),
    );
    assert.equal(
        fixture.items[0].trace.candidate_digest,
        'de4112e365c27fad4f94a8f189087fb2a138a3705580cf92f073fbf54f80697a',
    );
});

test('linear quantity is optional, digest-bound, and kept out of GD&T', () => {
    const ordinary = frontendCompatibleLinearFixture();
    assert.equal(Object.hasOwn(ordinary.items[0], 'quantity'), false);

    const multiplied = extendedLinearFixture({ quantity: 3 });
    assert.deepEqual(validateReviewCandidatesV1(multiplied), []);
    const [entry] = listReviewCandidateEntries(
        ingestReviewCandidatesV1(emptyWorkspace(), multiplied),
    );
    assert.equal(entry.quantity, 3);
    assert.equal(entry.item.quantity, 3);

    multiplied.items[0].quantity = 4;
    const staleReasons = validateReviewCandidatesV1(multiplied);
    assert.ok(staleReasons.includes('item_candidate_digest_invalid'));
    assert.ok(staleReasons.includes('audit_semantic_digest_invalid'));

    for (const quantity of [0, -1, 1, 1000, 2.5, '3', true]) {
        const invalid = extendedLinearFixture({ quantity: 3 });
        invalid.items[0].quantity = quantity;
        assert.ok(
            validateReviewCandidatesV1(invalid).includes(
                'item_quantity_invalid',
            ),
            JSON.stringify(quantity),
        );
    }

    const gdt = gdtFixture();
    gdt.items[0].quantity = 2;
    assert.ok(validateReviewCandidatesV1(gdt).includes('item_keys_invalid'));
});

test('non-ok syntax admits a blank correction candidate but confirmation requires a real nominal', () => {
    const value = extendedLinearFixture({
        nominal: null,
        symmetric_tolerance: null,
        upper_tolerance: null,
        lower_tolerance: null,
        prefix: null,
        unit: null,
        syntax: 'invalid_dimension_syntax',
    });
    assert.deepEqual(validateReviewCandidatesV1(value), []);

    const id = value.items[0].candidate_id;
    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), value);
    let [entry] = listReviewCandidateEntries(workspace);
    assert.equal(entry.nominal, null);
    assert.equal(entry.item.syntax, 'invalid_dimension_syntax');
    assert.throws(
        () => confirmReviewCandidate(workspace, id, {
            formalStampUuid: 'correction-1',
            nominal: null,
            symmetric_tolerance: null,
            upper_tolerance: null,
            lower_tolerance: null,
        }),
        /nominal must be decimal text/,
    );

    workspace = editReviewCandidate(workspace, id, {
        nominal: '99.129',
        symmetric_tolerance: '0.3',
        upper_tolerance: null,
        lower_tolerance: null,
    });
    [entry] = listReviewCandidateEntries(workspace);
    assert.equal(entry.status, 'edited');
    assert.equal(entry.nominal, '99.129');
});

test('bilateral edit and confirmation preserve both signed tolerance sides', () => {
    const value = extendedLinearFixture({
        symmetric_tolerance: null,
        upper_tolerance: '+0.2',
        lower_tolerance: '-0.8',
        prefix: 'Ø',
    });
    const id = value.items[0].candidate_id;
    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), value);
    const reviewed = {
        nominal: '14.376',
        symmetric_tolerance: null,
        upper_tolerance: '+0.3',
        lower_tolerance: '-0.7',
    };

    workspace = editReviewCandidate(workspace, id, reviewed);
    assert.deepEqual(workspace.decisions[id], {
        status: 'edited',
        ...reviewed,
    });
    assert.deepEqual(
        (({
            nominal,
            symmetric_tolerance,
            upper_tolerance,
            lower_tolerance,
            prefix,
            unit,
        }) => ({
            nominal,
            symmetric_tolerance,
            upper_tolerance,
            lower_tolerance,
            prefix,
            unit,
        }))(listReviewCandidateEntries(workspace)[0]),
        {
            ...reviewed,
            prefix: 'Ø',
            unit: null,
        },
    );
    assert.throws(
        () => editReviewCandidate(workspace, id, {
            ...reviewed,
            prefix: 'R',
        }),
        /linear review values keys/,
    );

    workspace = confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'formal-bilateral-1',
        ...reviewed,
    });
    assert.deepEqual(workspace.decisions[id], {
        status: 'confirmed',
        ...reviewed,
        formal_stamp_uuid: 'formal-bilateral-1',
    });
});

test('GD&T exact 12-key item and backend-frozen digests are accepted', () => {
    const value = gdtFixture();
    assert.deepEqual(validateReviewCandidatesV1(value, { expectedPageIndex: 0 }), []);

    const [entry] = listReviewCandidateEntries(
        ingestReviewCandidatesV1(emptyWorkspace(), value),
    );
    assert.equal(entry.type, 'gdt');
    assert.equal(entry.symbol_name, 'position');
    assert.equal(entry.symbol_unicode, '⊕');
    assert.equal(entry.tolerance_value, '0.05');
    assert.deepEqual(
        [entry.datum_1, entry.datum_2, entry.datum_3],
        [null, null, null],
    );
});

test('GD&T item rejects open key sets, non-null datum, and invalid tolerance text', () => {
    const extraKey = gdtFixture();
    extraKey.items[0].modifier = 'MMC';
    assert.ok(validateReviewCandidatesV1(extraKey).includes('item_keys_invalid'));

    const missingKey = gdtFixture();
    delete missingKey.items[0].datum_3;
    assert.ok(validateReviewCandidatesV1(missingKey).includes('item_keys_invalid'));

    for (const key of ['datum_1', 'datum_2', 'datum_3']) {
        const nonNullDatum = gdtFixture();
        nonNullDatum.items[0][key] = 'A';
        assert.ok(
            validateReviewCandidatesV1(nonNullDatum).includes('item_gdt_datum_not_null'),
        );
    }

    for (const tolerance of ['+0.05', '-0.05', '0.', '0.0000', '1000', 'text', null]) {
        const invalidTolerance = gdtFixture();
        invalidTolerance.items[0].tolerance_value = tolerance;
        assert.ok(
            validateReviewCandidatesV1(invalidTolerance).includes(
                'item_gdt_tolerance_invalid',
            ),
            `expected ${JSON.stringify(tolerance)} to fail`,
        );
    }
});

test('labelled GD&T typo remains review-visible but cannot be formally confirmed unchanged', () => {
    const value = gdtFixture();
    const item = value.items[0];
    item.tolerance_value = '+0.05';
    item.syntax = 'invalid_dimension_syntax';
    item.candidate_id = 'rcv1_a598efd6f6ea219c73657724';
    item.trace.candidate_digest = (
        'a598efd6f6ea219c7365772498d1ce39d255bf9d47f9ebbb798ae98a2377dc56'
    );
    value.audit.semantic_digest = (
        '63c1de255cbe254f0a15d5cbbc3f5414e39195b9187c01cddf3a09277c78c8bd'
    );
    assert.deepEqual(validateReviewCandidatesV1(value), []);

    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), value);
    assert.throws(
        () => confirmReviewCandidate(workspace, item.candidate_id, {
            formalStampUuid: 'gdt-correction-1',
            tolerance_value: '+0.05',
        }),
        /unsigned decimal/,
    );
    workspace = editReviewCandidate(workspace, item.candidate_id, {
        tolerance_value: '0.05',
    });
    assert.equal(
        listReviewCandidateEntries(workspace)[0].tolerance_value,
        '0.05',
    );

    const stale = structuredClone(value);
    stale.items[0].syntax = 'incomplete_separator';
    const reasons = validateReviewCandidatesV1(stale);
    assert.ok(reasons.includes('item_candidate_digest_invalid'));
    assert.ok(reasons.includes('audit_semantic_digest_invalid'));

    stale.items[0].syntax = 'ok';
    assert.ok(
        validateReviewCandidatesV1(stale).includes(
            'item_gdt_tolerance_invalid',
        ),
    );
});

test('GD&T symbol pair and vector trace validation mirror the backend contract', () => {
    const wrongSymbol = gdtFixture();
    wrongSymbol.items[0].symbol_unicode = '⌖';
    assert.ok(validateReviewCandidatesV1(wrongSymbol).includes('item_gdt_symbol_invalid'));

    const wrongMethod = gdtFixture();
    wrongMethod.items[0].trace.symbol_method = 'ocr';
    assert.ok(
        validateReviewCandidatesV1(wrongMethod).includes(
            'item_gdt_symbol_method_invalid',
        ),
    );

    const lowConfidence = gdtFixture();
    lowConfidence.items[0].trace.symbol_confidence = 0.59;
    assert.ok(
        validateReviewCandidatesV1(lowConfidence).includes(
            'item_gdt_symbol_confidence_invalid',
        ),
    );

    const mismatchedTemplate = gdtFixture();
    mismatchedTemplate.audit.template_identity.content_sha256 = 'b'.repeat(64);
    assert.ok(
        validateReviewCandidatesV1(mismatchedTemplate).includes(
            'item_gdt_template_identity_mismatch',
        ),
    );
});

test('frozen candidate and semantic digests reject stale value or audit identity', () => {
    const staleCandidate = structuredClone(fixture);
    staleCandidate.items[0].nominal = '14.376';
    assert.ok(
        validateReviewCandidatesV1(staleCandidate).includes('item_candidate_digest_invalid'),
    );

    const staleAudit = structuredClone(fixture);
    staleAudit.audit.semantic_digest = 'f'.repeat(64);
    assert.ok(
        validateReviewCandidatesV1(staleAudit).includes('audit_semantic_digest_invalid'),
    );

    const staleWidened = extendedLinearFixture({ prefix: 'Ø' });
    staleWidened.items[0].prefix = 'R';
    const widenedReasons = validateReviewCandidatesV1(staleWidened);
    assert.ok(widenedReasons.includes('item_candidate_digest_invalid'));
    assert.ok(widenedReasons.includes('audit_semantic_digest_invalid'));
});

test('authority, zero-OCR runtime, page identity, and candidate identity fail closed', () => {
    const cases = [
        value => { value.audit.authority.formal_project_data_allowed = true; },
        value => { value.audit.runtime.paddleocr_call_count = 1; },
        value => { value.page_index = 1; },
        value => { value.items[0].candidate_id = 'other'; },
        value => { value.items[0].review_status = 'confirmed'; },
    ];
    for (const mutate of cases) {
        const value = structuredClone(fixture);
        mutate(value);
        assert.notDeepEqual(
            validateReviewCandidatesV1(value, { expectedPageIndex: 0 }),
            [],
        );
    }
});

test('GD&T edit and confirm expose only tolerance_value and preserve blank datum', () => {
    const value = gdtFixture();
    const id = value.items[0].candidate_id;
    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), value);

    assert.throws(() => editReviewCandidate(workspace, id, {
        tolerance_value: '0.06',
        symbol_name: 'flatness',
    }), /gdt review values keys/);
    assert.throws(() => editReviewCandidate(workspace, id, {
        tolerance_value: '0.06',
        datum_1: 'A',
    }), /gdt review values keys/);
    assert.throws(() => editReviewCandidate(workspace, id, {
        tolerance_value: '-0.06',
    }), /unsigned decimal/);

    workspace = editReviewCandidate(workspace, id, {
        tolerance_value: '0.06',
    });
    assert.deepEqual(workspace.decisions[id], {
        status: 'edited',
        tolerance_value: '0.06',
    });
    let [entry] = listReviewCandidateEntries(workspace);
    assert.equal(entry.status, 'edited');
    assert.equal(entry.tolerance_value, '0.06');
    assert.equal(entry.symbol_name, 'position');
    assert.deepEqual(
        [entry.datum_1, entry.datum_2, entry.datum_3],
        [null, null, null],
    );

    assert.throws(() => confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'gdt-formal-1',
        tolerance_value: '0.06',
        datum_1: 'A',
    }), /gdt review values keys/);

    workspace = confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'gdt-formal-1',
        tolerance_value: '0.06',
    });
    assert.deepEqual(workspace.decisions[id], {
        status: 'confirmed',
        tolerance_value: '0.06',
        formal_stamp_uuid: 'gdt-formal-1',
    });
    [entry] = listReviewCandidateEntries(workspace);
    assert.equal(entry.status, 'confirmed');
    assert.equal(entry.tolerance_value, '0.06');
    assert.deepEqual(
        [entry.datum_1, entry.datum_2, entry.datum_3],
        [null, null, null],
    );
    assert.equal(pendingReviewCandidateCount(workspace), 0);

    const repeated = confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'gdt-formal-1',
        tolerance_value: '0.06',
    });
    assert.deepEqual(repeated, workspace);
    assert.throws(() => confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'gdt-formal-2',
        tolerance_value: '0.06',
    }), /different formal stamp/);
    assert.throws(
        () => dismissReviewCandidate(workspace, id),
        /confirmed candidate cannot be dismissed/,
    );
});

test('GD&T dismiss is an idempotent tombstone and blocks edit or confirm', () => {
    const value = gdtFixture();
    const id = value.items[0].candidate_id;
    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), value);
    workspace = dismissReviewCandidate(workspace, id);
    assert.deepEqual(workspace.decisions[id], { status: 'dismissed' });
    assert.deepEqual(dismissReviewCandidate(workspace, id), workspace);
    assert.throws(() => editReviewCandidate(workspace, id, {
        tolerance_value: '0.06',
    }), /already dismissed/);
    assert.throws(() => confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'gdt-formal-1',
        tolerance_value: '0.06',
    }), /dismissed candidate cannot be confirmed/);
});

test('editing remains review-only and repeat2 replaces the envelope without duplication', () => {
    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), fixture, {
        expectedPageIndex: 0,
    });
    const id = fixture.items[0].candidate_id;
    workspace = editReviewCandidate(workspace, id, {
        nominal: '14.376',
        symmetric_tolerance: '0.050',
        upper_tolerance: null,
        lower_tolerance: null,
    });
    workspace = ingestReviewCandidatesV1(workspace, structuredClone(fixture), {
        expectedPageIndex: 0,
    });

    const entries = listReviewCandidateEntries(workspace);
    assert.equal(workspace.envelopes.length, 1);
    assert.equal(entries.length, 1);
    assert.equal(entries[0].status, 'edited');
    assert.equal(entries[0].nominal, '14.376');
    assert.equal(entries[0].symmetric_tolerance, '0.050');
    assert.equal(pendingReviewCandidateCount(workspace), 1);
});

test('confirmed candidate is idempotently locked across repeat2 and cannot create a second stamp decision', () => {
    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), fixture);
    const id = fixture.items[0].candidate_id;
    workspace = confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'formal-1',
        nominal: '234.567',
        symmetric_tolerance: '0.089',
        upper_tolerance: null,
        lower_tolerance: null,
    });
    workspace = ingestReviewCandidatesV1(workspace, structuredClone(fixture));

    assert.equal(pendingReviewCandidateCount(workspace), 0);
    assert.equal(listReviewCandidateEntries(workspace)[0].status, 'confirmed');
    const repeated = confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'formal-1',
        nominal: '234.567',
        symmetric_tolerance: '0.089',
        upper_tolerance: null,
        lower_tolerance: null,
    });
    assert.deepEqual(repeated, workspace);
    assert.throws(() => confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'formal-2',
        nominal: '234.567',
        symmetric_tolerance: '0.089',
        upper_tolerance: null,
        lower_tolerance: null,
    }), /different formal stamp/);
});

test('dismissed candidate remains a session tombstone across repeat2', () => {
    let workspace = ingestReviewCandidatesV1(emptyWorkspace(), fixture);
    const id = fixture.items[0].candidate_id;
    workspace = dismissReviewCandidate(workspace, id);
    workspace = ingestReviewCandidatesV1(workspace, structuredClone(fixture));

    assert.equal(pendingReviewCandidateCount(workspace), 0);
    assert.equal(listReviewCandidateEntries(workspace)[0].status, 'dismissed');
    assert.throws(() => confirmReviewCandidate(workspace, id, {
        formalStampUuid: 'formal-1',
        nominal: '234.567',
        symmetric_tolerance: '0.089',
        upper_tolerance: null,
        lower_tolerance: null,
    }), /dismissed/);
    assert.deepEqual(emptyWorkspace(), {
        envelopes: [], decisions: {}, selectedCandidateId: null,
    });
});

test('review transaction commits exact root envelopes once and never mutates formal stamps', async () => {
    const state = {
        stamps: [{ uuid: 'formal-existing', confirmed: true }],
        reviewWorkspace: emptyWorkspace(),
        isAnalysisRunning: false,
    };
    let historyWrites = 0;
    const response = {
        status: 'success',
        [REVIEW_CANDIDATES_ROOT_KEY]: structuredClone(fixture),
    };
    const result = await executeReviewCandidatesTransaction({
        state,
        pages: [1],
        analyzePage: async () => response,
        pushHistory: () => { historyWrites += 1; },
    });

    assert.equal(result.status, 'success');
    assert.deepEqual(result.succeededPages, [1]);
    assert.deepEqual(state.stamps, [{ uuid: 'formal-existing', confirmed: true }]);
    assert.equal(pendingReviewCandidateCount(state.reviewWorkspace), 1);
    assert.equal(historyWrites, 1);
});

test('multi-page partial review replaces only valid pages and reports contract failures', async () => {
    const prior = pageFixture(1, '6');
    const state = {
        stamps: [{ uuid: 'formal-existing' }],
        reviewWorkspace: ingestReviewCandidatesV1(emptyWorkspace(), prior),
        isAnalysisRunning: false,
    };
    const beforeStamps = structuredClone(state.stamps);
    const result = await executeReviewCandidatesTransaction({
        state,
        pages: [1, 2],
        analyzePage: async page => page === 1
            ? { status: 'success', [REVIEW_CANDIDATES_ROOT_KEY]: pageFixture(0, '7') }
            : { status: 'success' },
    });

    assert.equal(result.status, 'partial');
    assert.deepEqual(result.succeededPages, [1]);
    assert.deepEqual(result.failedPages, [2]);
    assert.equal(result.pageFailures[0].code, 'review_candidates_v1_missing');
    assert.deepEqual(state.stamps, beforeStamps);
    assert.deepEqual(state.reviewWorkspace.envelopes.map(value => value.page_index), [0, 1]);
});

test('review transaction stops after a non-retryable transport failure', async () => {
    const state = {
        stamps: [],
        reviewWorkspace: emptyWorkspace(),
        isAnalysisRunning: false,
    };
    const calls = [];
    const result = await executeReviewCandidatesTransaction({
        state,
        pages: [1, 2, 3],
        analyzePage: async page => {
            calls.push(page);
            throw Object.assign(new Error('request exceeds whole-body limit'), {
                code: 'request_body_too_large',
                status: 413,
                retryable: false,
            });
        },
    });

    assert.deepEqual(calls, [1]);
    assert.equal(result.status, 'failed');
    assert.deepEqual(result.failedPages, [1]);
    assert.deepEqual(result.skippedPages, [2, 3]);
    assert.equal(result.pageFailures[0].code, 'request_body_too_large');
    assert.equal(result.pageFailures[0].status, 413);
    assert.equal(result.pageFailures[0].retryable, false);
    assert.deepEqual(state.reviewWorkspace, emptyWorkspace());
});

test('cancellation discards already staged review pages', async () => {
    const state = {
        stamps: [],
        reviewWorkspace: emptyWorkspace(),
        isAnalysisRunning: false,
    };
    const controller = new AbortController();
    const result = await executeReviewCandidatesTransaction({
        state,
        pages: [1, 2],
        signal: controller.signal,
        analyzePage: async page => {
            if (page === 1) controller.abort();
            return { status: 'success', [REVIEW_CANDIDATES_ROOT_KEY]: pageFixture(page - 1) };
        },
    });

    assert.equal(result.status, 'cancelled');
    assert.deepEqual(state.reviewWorkspace, emptyWorkspace());
    assert.equal(state.isAnalysisRunning, false);
});

test('document switch while review analysis is deferred keeps the imported workspace and returns stale', async () => {
    const state = {
        stamps: [{ uuid: 'doc-a-stamp' }],
        reviewWorkspace: emptyWorkspace(),
        pageOffsets: [{ start: 0, width: 100, height: 100 }],
        documentGeneration: 8,
        isAnalysisRunning: false,
    };
    let resolveAnalysis;
    const deferred = new Promise(resolve => { resolveAnalysis = resolve; });
    let historyWrites = 0;
    const transaction = executeReviewCandidatesTransaction({
        state,
        pages: [1],
        analyzePage: async () => {
            await deferred;
            return {
                status: 'success',
                [REVIEW_CANDIDATES_ROOT_KEY]: structuredClone(fixture),
            };
        },
        pushHistory: () => { historyWrites += 1; },
    });

    assert.equal(state.isAnalysisRunning, true);
    state.stamps = [{ uuid: 'imported-doc-b-stamp' }];
    state.reviewWorkspace = {
        envelopes: [{ imported: 'doc-b-review' }],
        decisions: { imported: { status: 'pending' } },
        selectedCandidateId: 'imported',
    };
    state.pageOffsets = [{ start: 0, width: 800, height: 500 }];
    state.documentGeneration += 1;
    const importedState = structuredClone({
        stamps: state.stamps,
        reviewWorkspace: state.reviewWorkspace,
        pageOffsets: state.pageOffsets,
        documentGeneration: state.documentGeneration,
    });

    resolveAnalysis();
    const result = await transaction;

    assert.equal(result.status, 'stale');
    assert.equal(result.statePreserved, true);
    assert.equal(result.expectedDocumentGeneration, 8);
    assert.equal(result.currentDocumentGeneration, 9);
    assert.equal(historyWrites, 0);
    assert.deepEqual(state.stamps, importedState.stamps);
    assert.deepEqual(state.reviewWorkspace, importedState.reviewWorkspace);
    assert.deepEqual(state.pageOffsets, importedState.pageOffsets);
    assert.equal(state.documentGeneration, importedState.documentGeneration);
    assert.equal(state.isAnalysisRunning, false);
});

test('review commit gate rechecks generation immediately before workspace assignment', async () => {
    const importedWorkspace = {
        envelopes: [{ imported: 'last-moment-doc' }],
        decisions: { imported: { status: 'pending' } },
        selectedCandidateId: 'imported',
    };
    const state = {
        stamps: [],
        reviewWorkspace: emptyWorkspace(),
        documentGeneration: 12,
        isAnalysisRunning: false,
    };
    let historyWrites = 0;
    const result = await executeReviewCandidatesTransaction({
        state,
        pages: [1],
        analyzePage: async () => ({
            status: 'success',
            [REVIEW_CANDIDATES_ROOT_KEY]: frontendCompatibleLinearFixture(),
        }),
        beforeCommit: () => {
            state.reviewWorkspace = structuredClone(importedWorkspace);
            state.documentGeneration += 1;
        },
        pushHistory: () => { historyWrites += 1; },
    });

    assert.equal(result.status, 'stale');
    assert.equal(result.expectedDocumentGeneration, 12);
    assert.equal(result.currentDocumentGeneration, 13);
    assert.deepEqual(state.reviewWorkspace, importedWorkspace);
    assert.equal(historyWrites, 0);
});

test('pending, edited, and dismissed review state is absent from the formal project export', () => {
    let reviewWorkspace = ingestReviewCandidatesV1(emptyWorkspace(), fixture);
    const id = fixture.items[0].candidate_id;
    reviewWorkspace = editReviewCandidate(reviewWorkspace, id, {
        nominal: '14.376',
        symmetric_tolerance: '0.050',
        upper_tolerance: null,
        lower_tolerance: null,
    });
    const snapshot = buildExportSnapshot({
        totalPages: 1,
        pageOffsets: [],
        docInfo: { filename: 'controlled.pdf', version: 1 },
        stamps: [],
        deletedStamps: [],
        viewBoxes: [],
        basicDimensions: [],
        datumReferences: [],
        inspection: {},
        currentAnalyzedPages: new Set(),
        l1Cache: new Map(),
        clusters: [],
        reviewWorkspace,
    }, { timestamp: '2026-07-13T00:00:00.000Z' });

    assert.deepEqual(snapshot.stamps, []);
    assert.equal(Object.hasOwn(snapshot, 'reviewWorkspace'), false);
    assert.equal(JSON.stringify(snapshot).includes(id), false);
});
