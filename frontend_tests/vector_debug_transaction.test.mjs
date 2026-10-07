import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const storage = new Map();
globalThis.localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key),
};

const fixture = JSON.parse(await readFile(
    new URL('../contracts/vector_layer_debug_v1.canonical.json', import.meta.url),
    'utf8',
));
const reviewFixture = JSON.parse(await readFile(
    new URL('../contracts/review_candidates_v1.canonical.json', import.meta.url),
    'utf8',
));
const { state } = await import('../state.js');
const { executeReviewCandidatesTransaction } = await import('../review_candidates_v1.js');
const {
    applyVectorDebugOutcome,
    captureVectorDebugResponse,
    commitVectorDebugOutcome,
    initVectorDebugUI,
    isolateVectorDebugOutcome,
    projectVectorDebugBboxToViewport,
    renderVectorDebugUI,
} = await import('../vector_debug_drawer.js');

function debugState() {
    return {
        enabled: true,
        drawerOpen: false,
        workspace: { pages: [], selectedPageIndex: null, selectedStageId: null },
        error: null,
    };
}

test('successful page diagnostics commit independently from product state', () => {
    const product = {
        stamps: [{ uuid: 'formal-1' }],
        reviewWorkspace: { envelopes: [{ trace_id: 'review' }] },
        historyStack: [{ stamps: ['formal-1'] }],
    };
    const before = structuredClone(product);
    const staged = captureVectorDebugResponse(
        new Map(),
        1,
        { vector_layer_debug_v1: fixture },
    );
    const next = commitVectorDebugOutcome(debugState(), {
        stagedByPage: staged,
        outcome: { status: 'success', succeededPages: [1] },
    });

    assert.equal(next.workspace.pages.length, 1);
    assert.equal(next.workspace.pages[0].trace_id, fixture.trace_id);
    assert.equal(next.error, null);
    assert.deepEqual(product, before);
});

test('partial outcome commits only product-success pages', () => {
    const pageTwo = structuredClone(fixture);
    pageTwo.page_index = 1;
    pageTwo.trace_id = 'trace-page-two';
    let staged = captureVectorDebugResponse(
        new Map(), 1, { vector_layer_debug_v1: fixture },
    );
    staged = captureVectorDebugResponse(
        staged, 2, { vector_layer_debug_v1: pageTwo },
    );
    const next = commitVectorDebugOutcome(debugState(), {
        stagedByPage: staged,
        outcome: { status: 'partial', succeededPages: [1], failedPages: [2] },
    });
    assert.deepEqual(next.workspace.pages.map(page => page.page_index), [0]);
});

test('missing or invalid sidecar records an error without throwing or committing', () => {
    const missing = commitVectorDebugOutcome(debugState(), {
        stagedByPage: new Map([[1, null]]),
        outcome: { status: 'success', succeededPages: [1] },
    });
    assert.match(missing.error, /未返回分层诊断/);
    assert.deepEqual(missing.workspace.pages, []);

    const invalid = structuredClone(fixture);
    invalid.authority.consumer_allowed = true;
    const bad = commitVectorDebugOutcome(debugState(), {
        stagedByPage: new Map([[1, invalid]]),
        outcome: { status: 'success', succeededPages: [1] },
    });
    assert.match(bad.error, /authority_fail_open/);
    assert.deepEqual(bad.workspace.pages, []);
});

test('cancelled run and disabled diagnostics preserve sidecar state', () => {
    const existing = debugState();
    existing.workspace.pages = [structuredClone(fixture)];
    existing.workspace.selectedPageIndex = 0;
    existing.workspace.selectedStageId = fixture.stages[0].stage_id;
    const cancelled = commitVectorDebugOutcome(existing, {
        stagedByPage: new Map(),
        outcome: { status: 'cancelled', succeededPages: [] },
    });
    assert.deepEqual(cancelled.workspace, existing.workspace);

    existing.enabled = false;
    const disabled = commitVectorDebugOutcome(existing, {
        stagedByPage: new Map([[1, fixture]]),
        outcome: { status: 'success', succeededPages: [1] },
    });
    assert.deepEqual(disabled, existing);
});

test('corrupt diagnostic state is isolated instead of throwing into product flow', () => {
    const corrupted = {
        enabled: true,
        drawerOpen: true,
        workspace: { pages: 'corrupt' },
        error: null,
    };
    const next = isolateVectorDebugOutcome(corrupted, {
        stagedByPage: new Map([[1, fixture]]),
        outcome: { status: 'success', succeededPages: [1] },
    });
    assert.equal(next.enabled, true);
    assert.equal(next.drawerOpen, true);
    assert.deepEqual(next.workspace.pages, []);
    assert.match(next.error, /分层诊断旁路失败/);
});

test('capture and apply are no-throw even when diagnostic cloning or DOM access fails', () => {
    const proxyStages = new Proxy(fixture.stages, {});
    const unclonable = {
        ...fixture,
        stages: proxyStages,
    };
    let staged;
    assert.doesNotThrow(() => {
        staged = captureVectorDebugResponse(
            new Map(),
            1,
            { vector_layer_debug_v1: unclonable },
        );
    });
    const next = commitVectorDebugOutcome(debugState(), {
        stagedByPage: staged,
        outcome: { status: 'success', succeededPages: [1] },
    });
    assert.deepEqual(next.workspace.pages, []);
    assert.match(next.error, /capture|旁路|克隆/i);

    assert.doesNotThrow(() => {
        applyVectorDebugOutcome(staged, {
            status: 'success',
            succeededPages: [1],
        });
    });
    assert.doesNotThrow(() => renderVectorDebugUI());
    assert.doesNotThrow(() => initVectorDebugUI());
});

test('capture swallows a throwing debug sidecar getter', () => {
    const response = {};
    Object.defineProperty(response, 'vector_layer_debug_v1', {
        get() { throw new Error('diagnostic getter failed'); },
    });
    assert.doesNotThrow(() => captureVectorDebugResponse(new Map(), 1, response));
});

test('a diagnostic clone failure cannot turn a valid review page into product failure', async () => {
    state.isAnalysisRunning = false;
    state.reviewWorkspace = {
        envelopes: [],
        decisions: {},
        selectedCandidateId: null,
    };
    let staged = new Map();
    const unclonableDebug = {
        ...fixture,
        stages: new Proxy(fixture.stages, {}),
    };
    const outcome = await executeReviewCandidatesTransaction({
        state,
        pages: [1],
        analyzePage: async page => {
            const response = {
                status: 'success',
                review_candidates_v1: structuredClone(reviewFixture),
                vector_layer_debug_v1: unclonableDebug,
            };
            staged = captureVectorDebugResponse(staged, page, response);
            return response;
        },
    });

    assert.equal(outcome.status, 'success');
    assert.deepEqual(outcome.succeededPages, [1]);
    assert.equal(state.reviewWorkspace.envelopes.length, 1);
    const debug = commitVectorDebugOutcome(debugState(), {
        stagedByPage: staged,
        outcome,
    });
    assert.deepEqual(debug.workspace.pages, []);
    assert.match(debug.error, /capture|旁路|克隆/i);
});

test('page-local bbox projection respects page rotation and viewport transform', () => {
    const projected = projectVectorDebugBboxToViewport(
        { x: 10, y: 20, w: 5, h: 8 },
        {
            pageIndex: 2,
            start: 200,
            sourceWidth: 100,
            sourceHeight: 80,
            rotation: 90,
        },
        { zoom: 2, panX: 3, panY: -7 },
    );
    assert.deepEqual(projected, {
        x: 107,
        y: 413,
        w: 16,
        h: 10,
    });
});
