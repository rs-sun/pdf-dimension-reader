import test from 'node:test';
import assert from 'node:assert/strict';

const {
  computeDimensionPathRowStates,
  createRecognitionCapabilitiesLoader,
} = await import('../state.js');

function fullCapabilitiesPayload(
  rootKey = 'components',
  {
    strictAvailable = false,
    strictRuntimeClean = true,
    strictTemplateAvailable = true,
    legacyYoloAvailable = false,
  } = {},
) {
  return {
    status: rootKey === 'components' ? 'not_ready' : 'success',
    checks: [
      {
        name: 'strict_runtime_assets',
        ok: strictAvailable,
        reason_code: strictAvailable ? null : 'strict_yolo_model_missing',
      },
      {
        name: 'strict_forbidden_runtime',
        ok: strictRuntimeClean,
        reason_code: strictRuntimeClean ? null : 'strict_forbidden_runtime_loaded',
      },
      {
        name: 'r33_m1_template_runtime',
        ok: strictTemplateAvailable,
        reason_code: strictTemplateAvailable ? null : 'template_library_path_unconfigured',
      },
    ],
    [rootKey]: {
      ocr_engines: {
        rapidocr: { available: true, reason: 'ok' },
        paddle: { available: false, reason: 'paddleocr_not_installed' },
        yolo_char: { available: false, reason: 'not_built' },
      },
      view_seg: {
        yolo: {
          available: legacyYoloAvailable,
          reason: legacyYoloAvailable ? 'ok' : 'model_file_missing',
        },
        watershed: { available: true, reason: 'ok' },
        minesweeper: { available: true, reason: 'experimental' },
      },
      gdt: {
        hybrid: { available: true, reason: 'ok' },
        vector: { available: true, reason: 'ok' },
        yolo: { available: false, reason: 'model_not_loaded' },
      },
    },
  };
}

function emptyCapabilityState() {
  return {
    ocrEngineCapabilities: null,
    viewSegCapabilities: null,
    gdtCapabilities: null,
    strictViewSegCapability: null,
    strictTemplateCapability: null,
    legacyJsonBase64MaxRequestBodyBytes: null,
    recognitionCapabilitiesStatus: 'idle',
    recognitionCapabilitiesError: null,
  };
}

test('readiness captures the legacy JSON whole-body limit as transport state', async () => {
  const targetState = emptyCapabilityState();
  const payload = fullCapabilitiesPayload();
  payload.transport_limits = {
    schema_version: 'transport_limits_v1',
    legacy_json_base64: {
      supported: true,
      max_request_body_bytes: 33_554_432,
      size_includes: 'entire_http_body',
      encoding: 'rfc4648_base64_in_utf8_json',
    },
  };
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => jsonResponse(payload, { status: 503, ok: false }),
  });

  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(targetState.legacyJsonBase64MaxRequestBodyBytes, 33_554_432);
});

test('malformed transport capability clears an old limit and falls back to HTTP handling', async () => {
  const targetState = emptyCapabilityState();
  targetState.legacyJsonBase64MaxRequestBodyBytes = 33_554_432;
  const payload = fullCapabilitiesPayload();
  payload.transport_limits = {
    schema_version: 'transport_limits_v0',
    legacy_json_base64: {
      supported: true,
      max_request_body_bytes: 33_554_432,
      size_includes: 'raw_pdf_only',
      encoding: 'not-the-contract',
    },
  };
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => jsonResponse(payload, { status: 503, ok: false }),
  });

  assert.equal((await loader.refresh({ force: true })).status, 'ready');
  assert.equal(targetState.legacyJsonBase64MaxRequestBodyBytes, null);
});

function jsonResponse(payload, { status = 200, ok = status < 400 } = {}) {
  return {
    status,
    ok,
    json: async () => structuredClone(payload),
  };
}

test('concurrent refresh callers share one readiness request', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  let resolveFetch;
  const fetchImpl = () => {
    fetchCount += 1;
    return new Promise((resolve) => { resolveFetch = resolve; });
  };
  const loader = createRecognitionCapabilitiesLoader({ targetState, fetchImpl });

  const first = loader.refresh();
  const second = loader.refresh();
  assert.strictEqual(second, first);
  await Promise.resolve();
  assert.equal(fetchCount, 1);
  assert.equal(targetState.recognitionCapabilitiesStatus, 'loading');

  resolveFetch(jsonResponse(fullCapabilitiesPayload()));
  const result = await first;
  assert.equal(result.status, 'ready');
  assert.equal(result.httpStatus, 200);
  assert.equal(fetchCount, 1);
  assert.equal(targetState.recognitionCapabilitiesStatus, 'ready');
  assert.equal(targetState.viewSegCapabilities.yolo.reason, 'model_file_missing');
  assert.equal(targetState.strictViewSegCapability.reason, 'strict_yolo_model_missing');
});

test('a transient failure is not cached and the next panel refresh can recover', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  const fetchImpl = async () => {
    fetchCount += 1;
    if (fetchCount === 1) throw new TypeError('backend is still starting');
    return jsonResponse(fullCapabilitiesPayload('capabilities'));
  };
  const settled = [];
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl,
    onSettled: result => settled.push(result.status),
  });

  const failed = await loader.refresh();
  assert.equal(failed.status, 'error');
  assert.equal(targetState.recognitionCapabilitiesStatus, 'error');
  assert.equal(targetState.recognitionCapabilitiesError, 'backend is still starting');
  assert.equal(targetState.ocrEngineCapabilities, null);

  const recovered = await loader.refresh();
  assert.equal(recovered.status, 'ready');
  assert.equal(fetchCount, 2);
  assert.equal(targetState.recognitionCapabilitiesStatus, 'ready');
  assert.equal(targetState.ocrEngineCapabilities.rapidocr.available, true);
  assert.deepEqual(settled, ['error', 'ready']);

  const cached = await loader.refresh();
  assert.equal(cached.status, 'cached');
  assert.equal(fetchCount, 2);
});

test('503 readiness JSON still supplies capabilities without being treated as a fetch failure', async () => {
  const targetState = emptyCapabilityState();
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => jsonResponse(fullCapabilitiesPayload(), {
      status: 503,
      ok: false,
    }),
  });

  const result = await loader.refresh();
  assert.equal(result.status, 'ready');
  assert.equal(result.httpStatus, 503);
  assert.equal(targetState.recognitionCapabilitiesStatus, 'ready');
  assert.equal(targetState.gdtCapabilities.yolo.reason, 'model_not_loaded');
  assert.equal(targetState.strictViewSegCapability.available, false);
});

test('complementary incomplete payloads cannot combine into a mixed capability snapshot', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  const complete = fullCapabilitiesPayload();
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => {
      fetchCount += 1;
      if (fetchCount === 1) {
        return jsonResponse({
          checks: complete.checks,
          components: { ocr_engines: complete.components.ocr_engines },
        });
      }
      if (fetchCount === 2) {
        return jsonResponse({
          checks: complete.checks,
          components: {
            view_seg: complete.components.view_seg,
            gdt: complete.components.gdt,
          },
        });
      }
      return jsonResponse(complete);
    },
  });

  assert.equal((await loader.refresh()).status, 'error');
  assert.equal(targetState.ocrEngineCapabilities, null);
  assert.equal((await loader.refresh()).status, 'error');
  assert.equal(targetState.ocrEngineCapabilities, null);
  assert.equal(targetState.viewSegCapabilities, null);
  assert.equal(
    targetState.recognitionCapabilitiesError,
    'recognition_capabilities_incomplete',
  );

  const recovered = await loader.refresh();
  assert.equal(recovered.status, 'ready');
  assert.equal(fetchCount, 3);
});

test('manual force refresh replaces a complete negative 503 snapshot after recovery', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => {
      fetchCount += 1;
      if (fetchCount === 1) {
        return jsonResponse(fullCapabilitiesPayload(), { status: 503, ok: false });
      }
      return jsonResponse(fullCapabilitiesPayload('components', {
        strictAvailable: true,
        legacyYoloAvailable: true,
      }));
    },
  });

  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(targetState.strictViewSegCapability.available, false);
  assert.equal((await loader.refresh()).status, 'cached');
  assert.equal(fetchCount, 1);

  assert.equal((await loader.refresh({ force: true })).status, 'ready');
  assert.equal(fetchCount, 2);
  assert.equal(targetState.strictViewSegCapability.available, true);
});

test('a forced freshness check invalidates old strict authority while loading', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  let resolveRefresh;
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => {
      fetchCount += 1;
      if (fetchCount === 1) {
        return jsonResponse(fullCapabilitiesPayload('components', {
          strictAvailable: true,
        }));
      }
      return new Promise((resolve) => { resolveRefresh = resolve; });
    },
  });

  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(targetState.strictViewSegCapability.available, true);
  assert.equal(targetState.strictTemplateCapability.available, true);
  targetState.legacyJsonBase64MaxRequestBodyBytes = 33_554_432;

  const refreshing = loader.refresh({ force: true });
  assert.equal(targetState.recognitionCapabilitiesStatus, 'loading');
  assert.equal(targetState.strictViewSegCapability, null);
  assert.equal(targetState.strictTemplateCapability, null);
  assert.equal(targetState.legacyJsonBase64MaxRequestBodyBytes, null);
  const loadingPath = computeDimensionPathRowStates(
    targetState.strictViewSegCapability,
    targetState.strictTemplateCapability,
  ).find((row) => row.path === 'vector_only');
  assert.equal(loadingPath.disabled, true);

  await Promise.resolve();
  resolveRefresh(jsonResponse(fullCapabilitiesPayload('components', {
    strictAvailable: true,
  })));
  assert.equal((await refreshing).status, 'ready');
  const readyPath = computeDimensionPathRowStates(
    targetState.strictViewSegCapability,
    targetState.strictTemplateCapability,
  ).find((row) => row.path === 'vector_only');
  assert.equal(readyPath.disabled, false);
});

test('strict assets cannot appear usable while a forbidden runtime is loaded', async () => {
  const targetState = emptyCapabilityState();
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => jsonResponse(fullCapabilitiesPayload('components', {
      strictAvailable: true,
      strictRuntimeClean: false,
      legacyYoloAvailable: true,
    }), { status: 503, ok: false }),
  });

  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(targetState.strictViewSegCapability.available, false);
  assert.equal(
    targetState.strictViewSegCapability.reason,
    'strict_forbidden_runtime_loaded',
  );
  assert.equal(targetState.viewSegCapabilities.yolo.available, true);
});

test('strict vector readiness keeps template authority separate from YOLO assets', async () => {
  const targetState = emptyCapabilityState();
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => jsonResponse(fullCapabilitiesPayload('components', {
      strictAvailable: true,
      strictTemplateAvailable: false,
      legacyYoloAvailable: true,
    }), { status: 503, ok: false }),
  });

  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(targetState.strictViewSegCapability.available, true);
  assert.equal(targetState.strictTemplateCapability.available, false);
  assert.equal(
    targetState.strictTemplateCapability.reason,
    'template_library_path_unconfigured',
  );
});

test('a readiness payload without template authority is incomplete and retryable', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => {
      fetchCount += 1;
      const payload = fullCapabilitiesPayload('components', {
        strictAvailable: true,
      });
      if (fetchCount === 1) {
        payload.checks = payload.checks.filter(
          check => check.name !== 'r33_m1_template_runtime',
        );
      }
      return jsonResponse(payload);
    },
  });

  assert.equal((await loader.refresh()).status, 'error');
  assert.equal(targetState.ocrEngineCapabilities, null);
  assert.equal(targetState.strictTemplateCapability, null);
  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(fetchCount, 2);
});

test('an HTTP error other than the readiness 503 stays retryable', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => {
      fetchCount += 1;
      return jsonResponse(fullCapabilitiesPayload(), {
        status: fetchCount === 1 ? 500 : 200,
        ok: fetchCount !== 1,
      });
    },
  });

  const failed = await loader.refresh();
  assert.equal(failed.status, 'error');
  assert.equal(failed.error, 'recognition_capabilities_http_500');
  assert.equal(targetState.ocrEngineCapabilities, null);
  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(fetchCount, 2);
});

test('malformed capability entries do not become a permanent cache', async () => {
  const targetState = emptyCapabilityState();
  const malformed = fullCapabilitiesPayload();
  malformed.components.view_seg.yolo.available = 'false';
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => jsonResponse(malformed),
  });

  assert.equal((await loader.refresh()).status, 'error');
  assert.equal(targetState.viewSegCapabilities, null);
  assert.equal(
    targetState.recognitionCapabilitiesError,
    'recognition_capabilities_incomplete',
  );
});

test('a failed forced refresh cannot hide behind the previous complete cache', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  const loader = createRecognitionCapabilitiesLoader({
    targetState,
    fetchImpl: async () => {
      fetchCount += 1;
      if (fetchCount === 2) throw new TypeError('backend restarted');
      return jsonResponse(fullCapabilitiesPayload());
    },
  });

  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal((await loader.refresh({ force: true })).status, 'error');
  assert.equal(targetState.recognitionCapabilitiesStatus, 'error');
  assert.equal(targetState.strictViewSegCapability, null);
  assert.equal((await loader.refresh()).status, 'ready');
  assert.equal(fetchCount, 3);
});

test('a synchronous fetch wrapper re-entry still shares the installed lock', async () => {
  const targetState = emptyCapabilityState();
  let fetchCount = 0;
  let nestedPromise = null;
  let loader;
  const fetchImpl = () => {
    fetchCount += 1;
    nestedPromise = loader.refresh();
    return jsonResponse(fullCapabilitiesPayload());
  };
  loader = createRecognitionCapabilitiesLoader({ targetState, fetchImpl });

  const outerPromise = loader.refresh();
  assert.equal((await outerPromise).status, 'ready');
  assert.strictEqual(nestedPromise, outerPromise);
  assert.equal(fetchCount, 1);
});
