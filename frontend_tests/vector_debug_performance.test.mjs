import test from 'node:test';
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';

const storage = new Map();
globalThis.localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key),
};

const {
    captureVectorDebugResponse,
    commitVectorDebugOutcome,
} = await import('../vector_debug_drawer.js');

function densePage(pageIndex, overlayCount = 2000) {
    const stageId = `L0_page_${pageIndex}`;
    const overlays = Array.from({ length: overlayCount }, (_, index) => ({
        overlay_id: `p${pageIndex}_o${index}`,
        stage_id: stageId,
        role: 'context',
        bbox: { x: index, y: pageIndex, w: 1, h: 1 },
        label: null,
    }));
    return {
        schema_version: 'vector_layer_debug_v1',
        trace_id: `trace_page_${pageIndex}`,
        page_index: pageIndex,
        coordinate_space: 'pdf_page_points_top_left',
        status: 'ok',
        unavailable_reason: null,
        stages: [{
            stage_id: stageId,
            status: 'ok',
            input_count: overlayCount,
            output_count: overlayCount,
            drop_count: 0,
            elapsed_ms: 1,
            first_drop: null,
        }],
        overlays,
        limits: {
            total: overlayCount,
            returned: overlayCount,
            truncated: false,
        },
        authority: {
            diagnostic_only: true,
            consumer_allowed: false,
            release_allowed: false,
        },
    };
}

test('50 x 2000 diagnostics stay linear and retain at most the session budget', () => {
    const started = performance.now();
    let staged = new Map();
    for (let pageIndex = 0; pageIndex < 50; pageIndex += 1) {
        staged = captureVectorDebugResponse(staged, pageIndex + 1, {
            vector_layer_debug_v1: densePage(pageIndex),
        });
    }
    const next = commitVectorDebugOutcome({
        enabled: true,
        drawerOpen: false,
        workspace: { pages: [], selectedPageIndex: null, selectedStageId: null },
        error: null,
    }, {
        stagedByPage: staged,
        outcome: {
            status: 'success',
            succeededPages: Array.from({ length: 50 }, (_, index) => index + 1),
        },
    });
    const elapsedMs = performance.now() - started;

    assert.equal(next.workspace.pages.length, 50);
    assert.equal(
        next.workspace.pages.reduce((total, page) => total + page.overlays.length, 0),
        10_000,
    );
    assert.ok(elapsedMs < 1500, `bounded debug ingest took ${elapsedMs.toFixed(1)} ms`);
});

test('page-budget omissions are aggregated instead of reported as page failures', () => {
    let staged = new Map();
    for (let pageIndex = 0; pageIndex < 51; pageIndex += 1) {
        staged = captureVectorDebugResponse(staged, pageIndex + 1, {
            vector_layer_debug_v1: densePage(pageIndex, 0),
        });
    }
    const next = commitVectorDebugOutcome({
        enabled: true,
        drawerOpen: false,
        workspace: { pages: [], selectedPageIndex: null, selectedStageId: null },
        error: null,
    }, {
        stagedByPage: staged,
        outcome: {
            status: 'success',
            succeededPages: Array.from({ length: 51 }, (_, index) => index + 1),
        },
    });
    assert.equal(next.workspace.pages.length, 50);
    assert.match(next.error, /请求页数预算已满，省略 1 页/);
    assert.doesNotMatch(next.error, /第 51 页未返回/);
});
