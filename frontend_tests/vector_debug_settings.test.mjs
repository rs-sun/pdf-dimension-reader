import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const storage = new Map();
globalThis.localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key),
};

const { state } = await import('../state.js');
const {
    buildAnalyzeSettingsPayload,
    createVectorDebugRequestBudget,
} = await import('../api_client.js');
const analysisSource = await readFile(
    new URL('../app_analysis.js', import.meta.url),
    'utf8',
);

function resetStrictSettings() {
    state.recognitionSettings = {
        dimension_path: 'vector_only',
        ocr_engine: 'rapidocr',
        view_seg: 'yolo',
        gdt_backend: 'hybrid',
    };
}

test('vector layer diagnostics are absent from the default analyze request', () => {
    resetStrictSettings();
    state.vectorDebugState = {
        enabled: false,
        drawerOpen: false,
        workspace: { pages: [], selectedPageIndex: null, selectedStageId: null },
        error: null,
    };

    assert.deepEqual(buildAnalyzeSettingsPayload(), {
        dimension_path: 'vector_only',
        view_seg: 'yolo',
        gdt_backend: 'hybrid',
    });
});

test('explicit session opt-in adds the bounded versioned debug request only', () => {
    resetStrictSettings();
    state.vectorDebugState = {
        enabled: true,
        drawerOpen: false,
        workspace: { pages: [], selectedPageIndex: null, selectedStageId: null },
        error: null,
    };

    assert.deepEqual(buildAnalyzeSettingsPayload(), {
        dimension_path: 'vector_only',
        view_seg: 'yolo',
        gdt_backend: 'hybrid',
        vector_layer_debug: {
            schema_version: 'vector_layer_debug_request_v1',
            include_overlays: true,
            overlay_limit: 2000,
        },
    });
});

test('Analyze All shares one bounded page and overlay request budget', () => {
    resetStrictSettings();
    state.vectorDebugState.enabled = true;
    const budget = createVectorDebugRequestBudget();
    const allocations = Array.from({ length: 51 }, () => budget.take());

    assert.deepEqual(
        allocations.slice(0, 5).map(item => item.overlayLimit),
        [2000, 2000, 2000, 2000, 2000],
    );
    assert.equal(allocations[5].enabled, true);
    assert.equal(allocations[5].overlayLimit, 0);
    assert.equal(allocations[49].enabled, true);
    assert.equal(allocations[50].enabled, false);

    assert.deepEqual(
        buildAnalyzeSettingsPayload({ vectorDebugRequest: allocations[5] })
            .vector_layer_debug,
        {
            schema_version: 'vector_layer_debug_request_v1',
            include_overlays: false,
            overlay_limit: 0,
        },
    );
    assert.equal(
        Object.hasOwn(
            buildAnalyzeSettingsPayload({ vectorDebugRequest: allocations[50] }),
            'vector_layer_debug',
        ),
        false,
    );
    assert.match(analysisSource, /createVectorDebugRequestBudget\(\)/);
    assert.match(
        analysisSource,
        /vectorDebugRequest:\s*vectorDebugRequestBudget\.take\(\)/,
    );
});
