import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// 最小 localStorage 桩
const store = {};
globalThis.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: (k) => { delete store[k]; },
};

const {
  state,
  saveRecognitionSettings,
  resetRecognitionSettingsToRecommended,
  updateRecognitionCapabilitiesFromPayload,
  DIMENSION_PATHS,
  OCR_ENGINES,
  VIEW_SEG_BACKENDS,
  GDT_BACKENDS,
  RECOMMENDED_RECOGNITION_SETTINGS,
  computeDimensionPathRowStates,
  computeOcrEngineRowStates,
  computeViewSegRowStates,
  computeGdtBackendRowStates,
  computeGdtRowStates,
  STRICT_GDT_BACKEND_COMPATIBILITY_NOTE,
} = await import('../state.js');
const { buildAnalyzeSettingsPayload } = await import('../api_client.js');
const indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');

test('default dimension_path is vector_only', () => {
  assert.equal(state.recognitionSettings.dimension_path, 'vector_only');
});

test('analysis defaults to all pages while retaining the current-page escape hatch', () => {
  assert.equal(state.analyzeScope, 'all');
  assert.match(
    indexHtml,
    /<input[^>]+name="analyze-scope"[^>]+value="all"[^>]*checked/,
  );
  assert.match(
    indexHtml,
    /<input[^>]+name="analyze-scope"[^>]+value="current"/,
  );
  assert.doesNotMatch(
    indexHtml,
    /<input[^>]+name="analyze-scope"[^>]+value="current"[^>]*checked/,
  );
});

test('whitelist matches backend', () => {
  assert.deepEqual(DIMENSION_PATHS, ['vector_only', 'vector_plus_ocr', 'ocr_only']);
});

test('OCR_ENGINES whitelist matches backend', () => {
  assert.deepEqual(OCR_ENGINES, ['rapidocr', 'paddle', 'yolo_char']);
});

test('VIEW_SEG_BACKENDS whitelist matches backend', () => {
  assert.deepEqual(VIEW_SEG_BACKENDS, ['yolo', 'watershed', 'minesweeper']);
});

test('GDT_BACKENDS whitelist', () => {
  assert.deepEqual(GDT_BACKENDS, ['hybrid', 'vector', 'yolo']);
});

test('saveRecognitionSettings persists to localStorage', () => {
  state.recognitionSettings.dimension_path = 'ocr_only';
  saveRecognitionSettings();
  assert.equal(JSON.parse(store.recognitionSettings).dimension_path, 'ocr_only');
});

test('default ocr_engine is rapidocr and persists', () => {
  assert.equal(state.recognitionSettings.ocr_engine, 'rapidocr');
  state.recognitionSettings.ocr_engine = 'paddle';
  saveRecognitionSettings();
  assert.equal(JSON.parse(store.recognitionSettings).ocr_engine, 'paddle');
});

test('default view_seg is yolo and persists', () => {
  assert.equal(state.recognitionSettings.view_seg, 'yolo');
  state.recognitionSettings.view_seg = 'watershed';
  saveRecognitionSettings();
  assert.equal(JSON.parse(store.recognitionSettings).view_seg, 'watershed');
});

test('a legacy compatible-mode view preference survives module initialization', async () => {
  store.recognitionSettings = JSON.stringify({
    dimension_path: 'vector_only',
    ocr_engine: 'rapidocr',
    view_seg: 'watershed',
    gdt_backend: 'hybrid',
  });
  const isolated = await import(`../state.js?legacy-view-pref=${Date.now()}`);
  assert.equal(isolated.state.recognitionSettings.dimension_path, 'vector_only');
  assert.equal(isolated.state.recognitionSettings.view_seg, 'watershed');
  delete store.recognitionSettings;
});

test('default gdt_backend is hybrid and persists', () => {
  assert.equal(state.recognitionSettings.gdt_backend, 'hybrid');
  state.recognitionSettings.gdt_backend = 'vector';
  saveRecognitionSettings();
  assert.equal(JSON.parse(store.recognitionSettings).gdt_backend, 'vector');
});

test('buildAnalyzeSettingsPayload reflects current state', () => {
  state.recognitionSettings.dimension_path = 'vector_plus_ocr';
  state.recognitionSettings.ocr_engine = 'rapidocr';
  state.recognitionSettings.view_seg = 'yolo';
  state.recognitionSettings.gdt_backend = 'hybrid';
  assert.deepEqual(buildAnalyzeSettingsPayload(), {
    dimension_path: 'vector_plus_ocr',
    ocr_engine: 'rapidocr',
    view_seg: 'yolo',
    gdt_backend: 'hybrid',
  });
});

test('vector_only omits ocr_engine while OCR-enabled modes preserve it', () => {
  state.recognitionSettings.dimension_path = 'vector_only';
  state.recognitionSettings.ocr_engine = 'paddle';
  state.recognitionSettings.view_seg = 'watershed';
  state.recognitionSettings.gdt_backend = 'hybrid';
  const strictPayload = buildAnalyzeSettingsPayload();
  assert.equal(Object.hasOwn(strictPayload, 'ocr_engine'), false);
  assert.deepEqual(strictPayload, {
    dimension_path: 'vector_only',
    view_seg: 'yolo',
    gdt_backend: 'hybrid',
  });
  assert.equal(
    state.recognitionSettings.view_seg,
    'watershed',
    'strict request normalization must not erase the compatible-mode preference',
  );

  for (const dimensionPath of ['vector_plus_ocr', 'ocr_only']) {
    state.recognitionSettings.dimension_path = dimensionPath;
    const payload = buildAnalyzeSettingsPayload();
    assert.equal(payload.ocr_engine, 'paddle');
    assert.equal(payload.view_seg, 'watershed');
    assert.equal(Object.hasOwn(payload, 'ocr_engine'), true);
  }
});

test('payload includes gdt_backend', () => {
  state.recognitionSettings.gdt_backend = 'hybrid';
  assert.equal(buildAnalyzeSettingsPayload().gdt_backend, 'hybrid');
});

test('resetRecognitionSettingsToRecommended restores and persists defaults', () => {
  state.recognitionSettings = {
    dimension_path: 'ocr_only',
    ocr_engine: 'paddle',
    view_seg: 'watershed',
    gdt_backend: 'vector',
  };
  resetRecognitionSettingsToRecommended();
  assert.deepEqual(state.recognitionSettings, RECOMMENDED_RECOGNITION_SETTINGS);
  assert.deepEqual(JSON.parse(store.recognitionSettings), RECOMMENDED_RECOGNITION_SETTINGS);
});

test('capability cache accepts analyze and readyz payload shapes', () => {
  updateRecognitionCapabilitiesFromPayload({
    capabilities: {
      ocr_engines: {
        rapidocr: { available: true, reason: 'ok' },
        paddle: { available: false, reason: 'arm_mac_paddle_segfault' },
        yolo_char: { available: false, reason: 'not_built' },
      },
      view_seg: {
        yolo: { available: false, reason: 'model_file_missing' },
        watershed: { available: true, reason: 'ok' },
        minesweeper: { available: true, reason: 'experimental' },
      },
      gdt: {
        hybrid: { available: true, reason: 'ok' },
        vector: { available: true, reason: 'ok' },
        yolo: { available: false, reason: 'model_not_loaded' },
      },
    },
  });
  assert.equal(state.ocrEngineCapabilities.rapidocr.available, true);
  assert.equal(state.viewSegCapabilities.yolo.reason, 'model_file_missing');
  assert.equal(state.gdtCapabilities.hybrid.available, true);

  updateRecognitionCapabilitiesFromPayload({
    checks: [{
      name: 'strict_runtime_assets',
      ok: false,
      reason_code: 'strict_yolo_model_missing',
    }, {
      name: 'strict_forbidden_runtime',
      ok: true,
      reason_code: null,
    }, {
      name: 'r33_m1_template_runtime',
      ok: false,
      reason_code: 'template_library_path_unconfigured',
    }],
    components: {
      ocr_engines: {
        rapidocr: { available: false, reason: 'rapidocr_not_installed' },
        paddle: { available: false, reason: 'arm_mac_paddle_segfault' },
        yolo_char: { available: false, reason: 'not_built' },
      },
      view_seg: {
        yolo: { available: true, reason: 'ok' },
        watershed: { available: true, reason: 'ok' },
        minesweeper: { available: true, reason: 'experimental' },
      },
      gdt: {
        hybrid: { available: true, reason: 'ok' },
        vector: { available: true, reason: 'ok' },
        yolo: { available: false, reason: 'model_not_loaded' },
      },
    },
  });
  assert.equal(state.ocrEngineCapabilities.paddle.available, false);
  assert.equal(state.viewSegCapabilities.watershed.available, true);
  assert.equal(state.gdtCapabilities.yolo.reason, 'model_not_loaded');
  assert.equal(state.strictViewSegCapability.available, false);
  assert.equal(state.strictViewSegCapability.reason, 'strict_yolo_model_missing');
  assert.equal(state.strictTemplateCapability.available, false);
  assert.equal(state.strictTemplateCapability.reason, 'template_library_path_unconfigured');
});

test('vector_only path is unavailable when either strict authority is missing', () => {
  const templateMissing = Object.fromEntries(
    computeDimensionPathRowStates(
      { available: true, reason: 'ok' },
      { available: false, reason: 'template_library_path_unconfigured' },
    ).map((row) => [row.path, row]),
  );
  assert.equal(templateMissing.vector_only.disabled, true);
  assert.equal(templateMissing.vector_only.note, 'template_library_path_unconfigured');
  assert.equal(templateMissing.vector_plus_ocr.disabled, false);
  assert.equal(templateMissing.ocr_only.disabled, false);

  const yoloMissing = Object.fromEntries(
    computeDimensionPathRowStates(
      { available: false, reason: 'strict_yolo_model_missing' },
      { available: true, reason: 'ok' },
    ).map((row) => [row.path, row]),
  );
  assert.equal(yoloMissing.vector_only.disabled, true);
  assert.equal(yoloMissing.vector_only.note, 'strict_yolo_model_missing');

  const bothMissing = computeDimensionPathRowStates(
    { available: false, reason: 'strict_yolo_model_missing' },
    { available: false, reason: 'template_library_path_unconfigured' },
  ).find((row) => row.path === 'vector_only');
  assert.deepEqual(bothMissing.reasons, [
    'strict_yolo_model_missing',
    'template_library_path_unconfigured',
  ]);

  const ready = Object.fromEntries(
    computeDimensionPathRowStates(
      { available: true, reason: 'ok' },
      { available: true, reason: 'ok' },
    ).map((row) => [row.path, row]),
  );
  assert.equal(ready.vector_only.disabled, false);
  assert.equal(ready.vector_only.note, '');
});

test('row states grey unavailable engines and whole axis when vector_only', () => {
  const caps = {
    rapidocr: { available: true, reason: 'ok' },
    paddle: { available: false, reason: 'arm_mac_paddle_segfault' },
    yolo_char: { available: false, reason: 'not_built' },
  };
  let rows = computeOcrEngineRowStates('vector_only', caps);
  assert.ok(rows.every((r) => r.disabled));
  assert.match(rows[0].note || '', /不跑 OCR|不需要/);

  rows = computeOcrEngineRowStates('ocr_only', caps);
  const by = Object.fromEntries(rows.map((r) => [r.engine, r]));
  assert.equal(by.rapidocr.disabled, false);
  assert.equal(by.paddle.disabled, true);
  assert.equal(by.yolo_char.disabled, true);
});

test('view_seg greying: yolo disabled when model missing, watershed always on', () => {
  const caps = {
    yolo: { available: false, reason: 'model_file_missing' },
    watershed: { available: true, reason: 'ok' },
    minesweeper: { available: true, reason: 'experimental' },
  };
  const by = Object.fromEntries(computeViewSegRowStates(caps).map((r) => [r.engine, r]));
  assert.equal(by.yolo.disabled, true);
  assert.equal(by.watershed.disabled, false);
  assert.equal(by.minesweeper.note, 'experimental');
});

test('vector_only exposes only the strict yolo view path without losing compatible choices', () => {
  const caps = {
    yolo: { available: true, reason: 'ok' },
    watershed: { available: true, reason: 'ok' },
    minesweeper: { available: true, reason: 'experimental' },
  };
  const strict = Object.fromEntries(
    computeViewSegRowStates(
      caps,
      'vector_only',
      { available: false, reason: 'strict_yolo_model_missing' },
    ).map((row) => [row.engine, row]),
  );
  assert.equal(strict.yolo.disabled, true);
  assert.equal(strict.yolo.note, 'strict_yolo_model_missing');
  assert.equal(strict.watershed.disabled, true);
  assert.equal(strict.minesweeper.disabled, true);
  assert.equal(strict.watershed.note, 'vector_only_requires_yolo_view');
  assert.equal(strict.minesweeper.note, 'vector_only_requires_yolo_view');

  const compatible = Object.fromEntries(
    computeViewSegRowStates(caps, 'vector_plus_ocr').map((row) => [row.engine, row]),
  );
  assert.equal(compatible.yolo.disabled, false);
  assert.equal(compatible.watershed.disabled, false);
  assert.equal(compatible.minesweeper.disabled, false);
  assert.equal(compatible.minesweeper.note, 'experimental');

  const legacyUnavailable = {
    ...caps,
    yolo: { available: false, reason: 'model_file_missing' },
  };
  const strictReady = Object.fromEntries(
    computeViewSegRowStates(
      legacyUnavailable,
      'vector_only',
      { available: true, reason: 'ok' },
    ).map((row) => [row.engine, row]),
  );
  const compatibleUnavailable = Object.fromEntries(
    computeViewSegRowStates(legacyUnavailable, 'vector_plus_ocr')
      .map((row) => [row.engine, row]),
  );
  assert.equal(strictReady.yolo.disabled, false);
  assert.equal(compatibleUnavailable.yolo.disabled, true);
});

test('gdt greying: yolo disabled when model not loaded', () => {
  const caps = {
    hybrid: { available: true, reason: 'ok' },
    vector: { available: true, reason: 'ok' },
    yolo: { available: false, reason: 'model_not_loaded' },
  };
  const by = Object.fromEntries(computeGdtRowStates(caps).map((r) => [r.engine, r]));
  assert.equal(by.yolo.disabled, true);
  assert.equal(by.hybrid.disabled, false);
});

test('vector_only greys every legacy GD&T backend without erasing compatible capability state', () => {
  const caps = {
    hybrid: { available: true, reason: 'ok' },
    vector: { available: true, reason: 'ok' },
    yolo: { available: false, reason: 'model_not_loaded' },
  };
  const strict = computeGdtBackendRowStates(caps, 'vector_only');
  assert.ok(strict.every((row) => row.disabled));
  assert.ok(strict.every(
    (row) => row.note === STRICT_GDT_BACKEND_COMPATIBILITY_NOTE,
  ));

  const compatible = Object.fromEntries(
    computeGdtBackendRowStates(caps, 'vector_plus_ocr')
      .map((row) => [row.engine, row]),
  );
  assert.equal(compatible.hybrid.disabled, false);
  assert.equal(compatible.vector.disabled, false);
  assert.equal(compatible.yolo.disabled, true);
  assert.equal(compatible.yolo.note, 'model_not_loaded');
});
