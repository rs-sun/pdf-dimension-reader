import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
    VECTOR_DEBUG_MAX_WORKSPACE_OVERLAYS,
    VECTOR_DEBUG_MAX_WORKSPACE_PAGES,
    VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS,
    VECTOR_LAYER_DEBUG_COORDINATE_SPACE,
    VECTOR_LAYER_DEBUG_ROOT_KEY,
    VECTOR_LAYER_DEBUG_SCHEMA_VERSION,
    VectorLayerDebugContractError,
    createEmptyVectorDebugWorkspace,
    ingestVectorLayerDebugV1,
    selectVectorDebugWorkspacePage,
    validateVectorLayerDebugV1,
} from '../vector_layer_debug_v1.js';

const fixture = JSON.parse(await readFile(
    new URL('../contracts/vector_layer_debug_v1.canonical.json', import.meta.url),
    'utf8',
));

function unavailableFixture(pageIndex = 0) {
    const value = structuredClone(fixture);
    value.trace_id = `trace-unavailable-${pageIndex}`;
    value.page_index = pageIndex;
    value.status = 'unavailable';
    value.unavailable_reason = 'debug_payload_not_produced';
    value.stages = [];
    value.overlays = [];
    value.limits = {
        total: 0,
        returned: 0,
        truncated: false,
    };
    return value;
}

test('canonical vector_layer_debug_v1 fixture is accepted exactly', () => {
    assert.equal(VECTOR_LAYER_DEBUG_ROOT_KEY, 'vector_layer_debug_v1');
    assert.equal(VECTOR_LAYER_DEBUG_SCHEMA_VERSION, 'vector_layer_debug_v1');
    assert.equal(
        VECTOR_LAYER_DEBUG_COORDINATE_SPACE,
        'pdf_page_points_top_left',
    );
    assert.deepEqual(validateVectorLayerDebugV1(fixture, { expectedPageIndex: 0 }), []);
    assert.deepEqual(validateVectorLayerDebugV1(unavailableFixture()), []);
});

test('strict shape rejects every extra or missing envelope, nested, bbox, and authority key', () => {
    const cases = [
        [value => { value.legacy_layers = []; }, 'envelope_keys_invalid'],
        [value => { delete value.status; }, 'envelope_keys_invalid'],
        [value => { value.stages[0].legacy_count = 1; }, 'stage_keys_invalid'],
        [value => { delete value.stages[0].elapsed_ms; }, 'stage_keys_invalid'],
        [value => { value.stages[0].first_drop.legacy_reason = 'x'; }, 'first_drop_keys_invalid'],
        [value => { delete value.overlays[0].label; }, 'overlay_keys_invalid'],
        [value => { value.overlays[0].bbox.right = 15.5; }, 'bbox_keys_invalid'],
        [value => { delete value.limits.truncated; }, 'limits_keys_invalid'],
        [value => { value.authority.review_only = true; }, 'authority_keys_invalid'],
        [value => { delete value.authority.release_allowed; }, 'authority_keys_invalid'],
    ];
    for (const [mutate, expectedReason] of cases) {
        const value = structuredClone(fixture);
        mutate(value);
        assert.ok(
            validateVectorLayerDebugV1(value).includes(expectedReason),
            expectedReason,
        );
    }
});

test('non-finite, negative, fractional, and degenerate numeric evidence is rejected', () => {
    const cases = [
        [value => { value.stages[0].elapsed_ms = Number.NaN; }, 'stage_elapsed_ms_invalid'],
        [value => { value.stages[0].elapsed_ms = Number.POSITIVE_INFINITY; }, 'stage_elapsed_ms_invalid'],
        [value => { value.stages[0].input_count = -1; }, 'stage_count_invalid'],
        [value => { value.stages[0].output_count = 1.5; }, 'stage_count_invalid'],
        [value => { value.overlays[0].bbox.x = Number.NEGATIVE_INFINITY; }, 'bbox_coordinate_invalid'],
        [value => { value.overlays[0].bbox.w = 0; }, 'bbox_extent_invalid'],
        [value => { value.stages[0].first_drop.bbox.h = -2; }, 'bbox_extent_invalid'],
        [value => { value.page_index = -1; }, 'page_index_invalid'],
    ];
    for (const [mutate, expectedReason] of cases) {
        const value = structuredClone(fixture);
        mutate(value);
        assert.ok(
            validateVectorLayerDebugV1(value).includes(expectedReason),
            expectedReason,
        );
    }
});

test('diagnostic authority is fail-closed and cannot authorize any consumer or release', () => {
    const cases = [
        value => { value.authority.diagnostic_only = false; },
        value => { value.authority.consumer_allowed = true; },
        value => { value.authority.release_allowed = true; },
        value => { value.authority.consumer_allowed = 0; },
    ];
    for (const mutate of cases) {
        const value = structuredClone(fixture);
        mutate(value);
        assert.ok(validateVectorLayerDebugV1(value).includes('authority_fail_open'));
    }
});

test('stage identity, first-drop, overlay references, and truncation accounting stay coherent', () => {
    const cases = [
        [value => { value.stages[1].stage_id = value.stages[0].stage_id; }, 'stage_id_duplicate'],
        [value => { value.stages[0].drop_count = 0; }, 'stage_first_drop_inconsistent'],
        [value => { value.stages[0].first_drop = null; }, 'stage_first_drop_inconsistent'],
        [value => { value.overlays[1].overlay_id = value.overlays[0].overlay_id; }, 'overlay_id_duplicate'],
        [value => { value.overlays[0].stage_id = 'missing'; }, 'overlay_stage_unknown'],
        [value => { value.limits.returned = 2; }, 'limits_returned_mismatch'],
        [value => { value.limits.total = 2; }, 'limits_total_invalid'],
        [value => { value.limits.truncated = false; }, 'limits_truncated_mismatch'],
    ];
    for (const [mutate, expectedReason] of cases) {
        const value = structuredClone(fixture);
        mutate(value);
        assert.ok(
            validateVectorLayerDebugV1(value).includes(expectedReason),
            expectedReason,
        );
    }
});

test('one page cannot exceed the frontend overlay authority cap', () => {
    const oversized = structuredClone(fixture);
    oversized.overlays = Array.from(
        { length: VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS + 1 },
        (_, index) => ({
            overlay_id: `overlay_${index}`,
            stage_id: oversized.stages[0].stage_id,
            role: 'context',
            bbox: { x: index, y: 1, w: 1, h: 1 },
            label: null,
        }),
    );
    oversized.limits = {
        total: oversized.overlays.length,
        returned: oversized.overlays.length,
        truncated: false,
    };
    assert.ok(
        validateVectorLayerDebugV1(oversized).includes('overlay_page_budget_exceeded'),
    );
    assert.equal(VECTOR_DEBUG_MAX_WORKSPACE_PAGES, 50);
    assert.equal(VECTOR_DEBUG_MAX_WORKSPACE_OVERLAYS, 10_000);
});

test('status and expected page identity fail closed', () => {
    const badOk = structuredClone(fixture);
    badOk.unavailable_reason = 'must_be_null';
    assert.ok(validateVectorLayerDebugV1(badOk).includes('unavailable_reason_invalid'));

    const badUnavailable = unavailableFixture();
    badUnavailable.unavailable_reason = null;
    assert.ok(validateVectorLayerDebugV1(badUnavailable).includes('unavailable_reason_invalid'));

    const wrongPage = structuredClone(fixture);
    assert.ok(validateVectorLayerDebugV1(
        wrongPage,
        { expectedPageIndex: 1 },
    ).includes('page_index_mismatch'));
});

test('empty workspace and single-page ingest are pure, replacing a page without duplication', () => {
    const empty = createEmptyVectorDebugWorkspace();
    assert.deepEqual(empty, {
        pages: [],
        selectedPageIndex: null,
        selectedStageId: null,
    });

    const source = structuredClone(fixture);
    const first = ingestVectorLayerDebugV1(empty, source, { expectedPageIndex: 0 });
    assert.deepEqual(empty, createEmptyVectorDebugWorkspace());
    assert.equal(first.pages.length, 1);
    assert.equal(first.selectedPageIndex, 0);
    assert.equal(first.selectedStageId, 'L0');

    source.stages[0].input_count = 999;
    assert.equal(first.pages[0].stages[0].input_count, 128);

    const replacement = structuredClone(fixture);
    replacement.trace_id = 'trace-vector-layer-debug-repeat2';
    replacement.stages[0].elapsed_ms = 0.5;
    const second = ingestVectorLayerDebugV1(first, replacement);
    assert.equal(second.pages.length, 1);
    assert.equal(second.pages[0].trace_id, 'trace-vector-layer-debug-repeat2');
    assert.equal(second.selectedStageId, 'L0');
    assert.equal(first.pages[0].trace_id, fixture.trace_id);
});

test('ingest preserves a valid selection, orders pages, and rejects invalid payload/workspace', () => {
    const first = ingestVectorLayerDebugV1(
        createEmptyVectorDebugWorkspace(),
        fixture,
    );
    const selected = {
        ...first,
        selectedStageId: 'L1',
    };
    const pageOne = structuredClone(fixture);
    pageOne.page_index = 1;
    pageOne.trace_id = 'trace-page-1';
    const second = ingestVectorLayerDebugV1(selected, pageOne);
    assert.deepEqual(second.pages.map(page => page.page_index), [0, 1]);
    assert.equal(second.selectedPageIndex, 0);
    assert.equal(second.selectedStageId, 'L1');

    const invalidPayload = structuredClone(fixture);
    invalidPayload.authority.release_allowed = true;
    assert.throws(
        () => ingestVectorLayerDebugV1(second, invalidPayload),
        error => error instanceof VectorLayerDebugContractError
            && error.reasons.includes('authority_fail_open'),
    );
    assert.throws(
        () => ingestVectorLayerDebugV1({ pages: 'not-an-array' }, fixture),
        error => error instanceof VectorLayerDebugContractError
            && error.reasons.includes('workspace_invalid'),
    );
});

test('page selection is explicit and resets the selected stage to that page', () => {
    const first = ingestVectorLayerDebugV1(
        createEmptyVectorDebugWorkspace(),
        fixture,
    );
    const pageTwo = structuredClone(fixture);
    pageTwo.page_index = 1;
    pageTwo.trace_id = 'trace-page-two';
    pageTwo.stages[0].stage_id = 'P2-L0';
    pageTwo.stages[0].first_drop = {
        ...pageTwo.stages[0].first_drop,
        candidate_id: 'candidate-page-two',
    };
    for (const overlay of pageTwo.overlays) {
        if (overlay.stage_id === 'L0') overlay.stage_id = 'P2-L0';
    }
    const workspace = ingestVectorLayerDebugV1(first, pageTwo);
    const selected = selectVectorDebugWorkspacePage(workspace, 1);
    assert.equal(selected.selectedPageIndex, 1);
    assert.equal(selected.selectedStageId, 'P2-L0');
    assert.equal(workspace.selectedPageIndex, 0);
});

test('the real Python projector remains accepted by the frontend contract', () => {
    const repoRoot = fileURLToPath(new URL('..', import.meta.url));
    const script = String.raw`
import json
from vector_layer_debug_contract import build_vector_layer_debug_v1
pipe = {
    "timings": {},
    "vector_phrase_region_dump_v1": {
        "trace_id": "trace_cross_language",
        "page_index": 0,
        "status": "ok",
        "regions": [{"phrase_id": "phrase_a", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
        "stats": {"pre_coalesce_phrase_count": 1, "phrase_count": 1, "suppressed_reader_input_count": 0},
    },
    "vector_pitch_phrase_read_dump_v1": {
        "trace_id": "trace_cross_language",
        "page_index": 0,
        "status": "ok",
        "reads": [{"phrase_id": "phrase_a", "status": "read", "text": "7.5", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
    },
}
print(json.dumps(build_vector_layer_debug_v1(
    pipe,
    trace_id="trace_cross_language",
    page_index=0,
    include_overlays=True,
    overlay_limit=20,
), allow_nan=False))
`;
    const raw = execFileSync('python3', ['-c', script], {
        cwd: repoRoot,
        env: {
            ...process.env,
            PYTHONPATH: `${repoRoot}/backend`,
        },
        encoding: 'utf8',
    });
    const payload = JSON.parse(raw);
    assert.deepEqual(
        validateVectorLayerDebugV1(payload, { expectedPageIndex: 0 }),
        [],
    );
});
