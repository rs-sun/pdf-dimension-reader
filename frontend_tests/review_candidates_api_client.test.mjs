import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const storage = new Map();
globalThis.localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key),
    key: index => [...storage.keys()][index] ?? null,
    get length() { return storage.size; },
};
const alerts = [];
globalThis.alert = message => alerts.push(String(message));

const fixture = JSON.parse(await readFile(
    new URL('../contracts/review_candidates_v1.canonical.json', import.meta.url),
    'utf8',
));
const { state } = await import('../state.js');
const { AnalysisApiError, runAnalysis } = await import('../api_client.js');
const { TransportError } = await import('../legacy_pdf_transport.js');
const {
    executeReviewCandidatesTransaction,
    validateReviewCandidatesV1,
} = await import('../review_candidates_v1.js');

function unavailableEnvelope() {
    const value = structuredClone(fixture);
    value.trace_id = 'trace-unavailable';
    value.items = [];
    value.audit.status = 'unavailable';
    value.audit.unavailable_reason = 'template_library_path_unconfigured';
    value.audit.chain.trace_id = value.trace_id;
    value.audit.chain.status = 'unavailable';
    value.audit.chain.semantic_projection_sha256 = null;
    for (const key of Object.keys(value.audit.chain.stage_status)) {
        value.audit.chain.stage_status[key] = 'not_run';
    }
    for (const key of Object.keys(value.audit.chain.stage_digests)) {
        value.audit.chain.stage_digests[key] = null;
    }
    for (const key of Object.keys(value.audit.chain.chain_counts)) {
        value.audit.chain.chain_counts[key] = 0;
    }
    value.audit.counts = {
        source_review_count: 0,
        candidate_count: 0,
        dropped_count: 0,
        drop_reasons: [],
    };
    value.audit.template_identity.template_version = null;
    value.audit.template_identity.content_sha256 = null;
    value.audit.semantic_digest = '7751ab8b06f4a3d3604d4d862c7f0a7519e62e7fcbbe85985f370917e5a94d6e';
    assert.deepEqual(validateReviewCandidatesV1(value, { expectedPageIndex: 0 }), []);
    return value;
}

function unavailableChainAudit() {
    return {
        schema_version: 'r33_m1_phrase_chain_audit_v1',
        trace_id: 'trace-unavailable',
        page_index: 0,
        status: 'unavailable',
        reason: 'template_library_path_unconfigured',
        details: {},
        template_identity: null,
        stage_status: {
            phrase_region: 'ok', pitch_reader: 'not_run', quality_gate: 'not_run',
            hypothesis: 'not_run', spatial_dedup: 'not_run', l3: 'not_run',
            l4: 'not_run', l5: 'not_run',
        },
        counts: {
            page_context_build_count: 1, page_context_extraction_call_count: 0,
            phrase_region_count: 0, reader_call_count: 0, reader_success_count: 0,
            quality_pass_count: 0, ordinary_hypothesis_count: 0,
            dedup_winner_count: 0, l3_dimension_count: 0, l4_recovered_count: 0,
            l5_review_required_count: 0,
        },
        semantic_projection_sha256: 'c'.repeat(64),
        ocr_family_call_count: 0,
        paddleocr_call_count: 0,
        rapidocr_call_count: 0,
        yolo_char_call_count: 0,
        yolo_b_call_count: 0,
        dimensions_promoted_count: 0,
        release_allowed: false,
        consumer_allowed: false,
    };
}

function prepareAnalysisState() {
    state.currentPdfBuffer = Uint8Array.from([37, 80, 68, 70]).buffer;
    state.recognitionSettings.dimension_path = 'vector_only';
    state.recognitionSettings.ocr_engine = 'rapidocr';
    state.strictViewSegCapability = null;
    state.strictTemplateCapability = null;
    state.recognitionCapabilitiesStatus = 'idle';
    state.recognitionCapabilitiesError = null;
    state.legacyJsonBase64MaxRequestBodyBytes = null;
    state.analyzeRuns = [];
    state.lastDimensions = null;
    alerts.length = 0;
}

test('runAnalysis rejects a known oversize body before base64 allocation or fetch', async () => {
    prepareAnalysisState();
    state.legacyJsonBase64MaxRequestBodyBytes = 1;
    let btoaCalls = 0;
    let fetchCalls = 0;
    const originalBtoa = globalThis.btoa;
    globalThis.btoa = () => {
        btoaCalls += 1;
        throw new Error('oversize analysis must not allocate base64');
    };
    globalThis.fetch = async () => {
        fetchCalls += 1;
        throw new Error('oversize analysis must not fetch');
    };

    try {
        await assert.rejects(
            runAnalysis(1),
            error => {
                assert.ok(error instanceof TransportError);
                assert.equal(error.code, 'request_body_too_large');
                assert.equal(error.maxRequestBodyBytes, 1);
                assert.ok(error.estimatedRequestBodyBytes > 1);
                return true;
            },
        );
    } finally {
        globalThis.btoa = originalBtoa;
    }

    assert.equal(btoaCalls, 0);
    assert.equal(fetchCalls, 0);
    assert.equal(state.analyzeRuns.length, 1);
    assert.equal(state.analyzeRuns[0].code, 'request_body_too_large');
    assert.equal(state.analyzeRuns[0].http_status, null);
    assert.match(alerts[0], /超过服务端/);
});

test('runAnalysis preserves an HTML proxy 413 as a stable transport error', async () => {
    prepareAnalysisState();
    globalThis.fetch = async () => ({
        ok: false,
        status: 413,
        statusText: 'Request Entity Too Large',
        text: async () => '<html><h1>413 Request Entity Too Large</h1></html>',
    });

    await assert.rejects(
        runAnalysis(1),
        error => {
            assert.ok(error instanceof TransportError);
            assert.equal(error.code, 'request_body_too_large');
            assert.equal(error.status, 413);
            return true;
        },
    );

    assert.equal(state.analyzeRuns.length, 1);
    assert.equal(state.analyzeRuns[0].code, 'request_body_too_large');
    assert.equal(state.analyzeRuns[0].http_status, 413);
    assert.match(alerts[0], /HTTP 413/);
});

test('runAnalysis reports a stable contract error for an unparsable success body', async () => {
    prepareAnalysisState();
    globalThis.fetch = async () => ({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => '<html>not the analysis contract</html>',
    });

    await assert.rejects(
        runAnalysis(1),
        error => {
            assert.ok(error instanceof TransportError);
            assert.equal(error.code, 'invalid_response_payload');
            assert.equal(error.status, 200);
            assert.equal(error.retryable, false);
            return true;
        },
    );
    assert.equal(state.analyzeRuns[0].code, 'invalid_response_payload');
});

test('caller cancellation records a stable code without a failure alert', async () => {
    prepareAnalysisState();
    const controller = new AbortController();
    controller.abort();
    globalThis.fetch = async (_url, options) => {
        assert.equal(options.signal.aborted, true);
        throw Object.assign(new Error('aborted'), { name: 'AbortError' });
    };

    assert.equal(await runAnalysis(1, { signal: controller.signal }), null);

    assert.equal(state.analyzeRuns.length, 1);
    assert.equal(state.analyzeRuns[0].code, 'cancelled');
    assert.equal(state.analyzeRuns[0].error, 'cancelled');
    assert.deepEqual(alerts, []);
});

test('runAnalysis blocks strict vector_only before /analyze when template readiness is negative', async () => {
    prepareAnalysisState();
    state.recognitionCapabilitiesStatus = 'ready';
    state.strictViewSegCapability = { available: true, reason: 'ok' };
    state.strictTemplateCapability = {
        available: false,
        reason: 'template_library_path_unconfigured',
    };
    let fetchCalls = 0;
    globalThis.fetch = async () => {
        fetchCalls += 1;
        throw new Error('/analyze must not be called');
    };

    assert.equal(await runAnalysis(1), null);
    assert.equal(fetchCalls, 0);
    assert.equal(state.recognitionSettings.dimension_path, 'vector_only');
    assert.equal(state.analyzeRuns.length, 0);
    assert.equal(alerts.length, 1);
    assert.match(alerts[0], /template_library_path_unconfigured/);
    assert.match(alerts[0], /重试能力检测/);
    assert.match(alerts[0], /兼容识别模式/);
});

test('runAnalysis blocks strict vector_only while capability refresh is loading', async () => {
    prepareAnalysisState();
    state.recognitionCapabilitiesStatus = 'loading';
    let fetchCalls = 0;
    globalThis.fetch = async () => {
        fetchCalls += 1;
        throw new Error('/analyze must not be called');
    };

    assert.equal(await runAnalysis(1), null);
    assert.equal(fetchCalls, 0);
    assert.equal(state.recognitionSettings.dimension_path, 'vector_only');
    assert.equal(state.analyzeRuns.length, 0);
    assert.equal(alerts.length, 1);
    assert.match(alerts[0], /能力检测仍在进行/);
    assert.match(alerts[0], /兼容识别模式/);
});

test('runAnalysis accepts an exact candidate-only success and records the versioned envelope', async () => {
    prepareAnalysisState();
    globalThis.fetch = async (_url, options) => {
        const body = JSON.parse(options.body);
        assert.equal(body.page, 1);
        assert.equal(body.settings.dimension_path, 'vector_only');
        assert.equal(Object.hasOwn(body.settings, 'ocr_engine'), false);
        return {
            ok: true,
            status: 200,
            statusText: 'OK',
            json: async () => ({
                status: 'success',
                dimensions: [],
                review_candidates_v1: structuredClone(fixture),
            }),
        };
    };

    const result = await runAnalysis(1);

    assert.equal(result.review_candidates_v1.schema_version, 'review_candidates_v1');
    assert.equal(state.analyzeRuns.length, 1);
    assert.deepEqual(
        state.analyzeRuns[0].review_candidates_v1,
        fixture,
    );
    assert.equal(state.analyzeRuns[0].ok, true);
    assert.equal(state.analyzeRuns[0].http_status, 200);
    assert.deepEqual(alerts, []);
});

test('400 and 422 JSON errors propagate backend metadata after being recorded', async () => {
    const cases = [
        {
            status: 400,
            statusText: 'Bad Request',
            payload: {
                schema_version: 'vector_only_error_v1',
                status: 'fail_closed',
                code: 'vector_only_invalid_request',
                reasons: ['invalid_request'],
                message: 'page must be an integer',
                trace_id: null,
            },
        },
        {
            status: 422,
            statusText: 'Unprocessable Entity',
            payload: {
                schema_version: 'vector_only_error_v1',
                status: 'fail_closed',
                code: 'vector_only_request_unprocessable',
                reasons: ['page_out_of_range'],
                message: 'page is outside the document',
                trace_id: 'trace-422',
            },
        },
    ];

    for (const item of cases) {
        prepareAnalysisState();
        globalThis.fetch = async () => ({
            ok: false,
            status: item.status,
            statusText: item.statusText,
            json: async () => structuredClone(item.payload),
        });

        await assert.rejects(
            runAnalysis(1),
            error => {
                assert.ok(error instanceof AnalysisApiError);
                assert.equal(error.code, item.payload.code);
                assert.deepEqual(error.reasons, item.payload.reasons);
                assert.equal(error.message, item.payload.message);
                assert.equal(error.trace, item.payload.trace_id);
                assert.equal(error.status, item.status);
                assert.deepEqual(error.payload, item.payload);
                return true;
            },
        );
        assert.equal(state.analyzeRuns.length, 1);
        assert.equal(state.analyzeRuns[0].http_status, item.status);
        assert.equal(state.analyzeRuns[0].code, item.payload.code);
        assert.deepEqual(state.analyzeRuns[0].reasons, item.payload.reasons);
        assert.equal(state.analyzeRuns[0].trace_id, item.payload.trace_id);
        assert.equal(state.analyzeRuns[0].ok, false);
        const expectedAlert = [
            '智能解析失败：',
            `HTTP ${item.status}`,
            `code: ${item.payload.code}`,
            `reasons: ${item.payload.reasons.join(', ')}`,
            ...(item.payload.trace_id ? [`trace_id: ${item.payload.trace_id}`] : []),
            `message: ${item.payload.message}`,
        ].join('\n');
        assert.deepEqual(alerts, [expectedAlert]);
    }
});

test('backend retryable false stops ordinary API failures from fanning out', async () => {
    prepareAnalysisState();
    let fetchCalls = 0;
    const payload = {
        status: 'error',
        code: 'invalid_document_contract',
        message: 'the same document cannot succeed on another page',
        reasons: ['invalid_document_contract'],
        retryable: false,
    };
    globalThis.fetch = async () => {
        fetchCalls += 1;
        return {
            ok: false,
            status: 422,
            statusText: 'Unprocessable Entity',
            json: async () => structuredClone(payload),
        };
    };

    const outcome = await executeReviewCandidatesTransaction({
        state,
        pages: [1, 2, 3],
        analyzePage: page => runAnalysis(page),
    });

    assert.equal(fetchCalls, 1);
    assert.deepEqual(outcome.failedPages, [1]);
    assert.deepEqual(outcome.skippedPages, [2, 3]);
    assert.equal(outcome.pageFailures[0].retryable, false);
});

test('analysis deadline remains armed until its JSON body is consumed', async () => {
    prepareAnalysisState();
    const originalSetTimeout = globalThis.setTimeout;
    const originalClearTimeout = globalThis.clearTimeout;
    const cleared = [];
    globalThis.setTimeout = () => 75;
    globalThis.clearTimeout = id => cleared.push(id);
    globalThis.fetch = async () => ({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => {
            assert.deepEqual(cleared, []);
            return JSON.stringify({
                status: 'success',
                dimensions: [],
                review_candidates_v1: structuredClone(fixture),
            });
        },
    });

    try {
        const result = await runAnalysis(1);
        assert.equal(result.status, 'success');
    } finally {
        globalThis.setTimeout = originalSetTimeout;
        globalThis.clearTimeout = originalClearTimeout;
    }

    assert.deepEqual(cleared, [75]);
});

test('legal 503 unavailable keeps the exact empty envelope/audit and preserves review state', async () => {
    prepareAnalysisState();
    const envelope = unavailableEnvelope();
    const chainAudit = unavailableChainAudit();
    const payload = {
        schema_version: 'r33_m1_review_projection_error_v1',
        status: 'fail_closed',
        code: 'r33_m1_review_projection_unavailable',
        reasons: ['template_library_path_unconfigured'],
        trace_id: 'trace-unavailable',
        review_candidates_v1: envelope,
        r33_m1_phrase_chain_audit_v1: chainAudit,
    };
    globalThis.fetch = async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => structuredClone(payload),
    });
    state.reviewWorkspace = {
        envelopes: [structuredClone(fixture)],
        decisions: {},
        selectedCandidateId: null,
    };
    const before = structuredClone(state.reviewWorkspace);

    const outcome = await executeReviewCandidatesTransaction({
        state,
        pages: [1],
        analyzePage: page => runAnalysis(page),
        pushHistory: () => { throw new Error('must not commit unavailable review'); },
    });

    assert.equal(outcome.status, 'failed');
    assert.equal(outcome.statePreserved, true);
    assert.deepEqual(outcome.failedPages, [1]);
    assert.deepEqual(state.reviewWorkspace, before);
    assert.equal(outcome.pageFailures[0].code, payload.code);
    assert.deepEqual(outcome.pageFailures[0].reasons, payload.reasons);
    assert.equal(outcome.pageFailures[0].trace, payload.trace_id);
    assert.equal(outcome.pageFailures[0].status, 503);
    assert.deepEqual(outcome.pageFailures[0].review_candidates_v1, envelope);
    assert.deepEqual(outcome.pageFailures[0].r33_m1_phrase_chain_audit_v1, chainAudit);
    assert.equal(state.analyzeRuns.length, 1);
    assert.deepEqual(state.analyzeRuns[0].review_candidates_v1, envelope);
    assert.deepEqual(state.analyzeRuns[0].r33_m1_phrase_chain_audit_v1, chainAudit);
    assert.equal(state.analyzeRuns[0].http_status, 503);
    assert.deepEqual(alerts, [[
        '智能解析失败：',
        'HTTP 503',
        `code: ${payload.code}`,
        `reasons: ${payload.reasons.join(', ')}`,
        `trace_id: ${payload.trace_id}`,
        'message: HTTP 503 Service Unavailable',
    ].join('\n')]);
});

test('chain-stage 503 shows diagnostics from the top-level projection details', async () => {
    prepareAnalysisState();
    const envelope = unavailableEnvelope();
    envelope.audit.unavailable_reason = 'phrase_chain_stage_failed';
    const chainAudit = unavailableChainAudit();
    const details = {
        first_failed_stage: 'quality_gate',
        error: {
            schema_version: 'vector_phrase_quality_gate_error_v1',
            code: 'invalid_quality_gate_input',
            reasons: ['phrase_read_set_mismatch'],
            consumer_allowed: false,
        },
        reasons: ['phrase_read_set_mismatch'],
    };
    chainAudit.status = 'fail_closed';
    chainAudit.reason = 'phrase_chain_stage_failed';
    chainAudit.details = structuredClone(details);
    chainAudit.stage_status = {
        phrase_region: 'ok',
        pitch_reader: 'ok',
        quality_gate: 'fail_closed',
        hypothesis: 'fail_closed',
        spatial_dedup: 'fail_closed',
        l3: 'fail_closed',
        l4: 'fail_closed',
        l5: 'fail_closed',
    };
    const payload = {
        schema_version: 'r33_m1_review_projection_error_v1',
        status: 'fail_closed',
        code: 'r33_m1_review_projection_unavailable',
        reasons: ['phrase_chain_stage_failed'],
        trace_id: 'trace-unavailable',
        details,
        review_candidates_v1: envelope,
        r33_m1_phrase_chain_audit_v1: chainAudit,
    };
    globalThis.fetch = async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => structuredClone(payload),
    });

    await assert.rejects(
        runAnalysis(1),
        error => {
            assert.ok(error instanceof AnalysisApiError);
            assert.deepEqual(error.diagnostics, {
                firstFailedStage: 'quality_gate',
                errorCode: 'invalid_quality_gate_input',
                reasons: ['phrase_read_set_mismatch'],
            });
            return true;
        },
    );

    assert.deepEqual(alerts, [[
        '智能解析失败：',
        'HTTP 503',
        `code: ${payload.code}`,
        'reasons: phrase_chain_stage_failed',
        'failed_stage: quality_gate',
        'stage_error: invalid_quality_gate_input',
        'stage_reasons: phrase_read_set_mismatch',
        `trace_id: ${payload.trace_id}`,
        'message: HTTP 503 Service Unavailable',
    ].join('\n')]);
});

test('chain-stage diagnostics fall back to raw audit details', async () => {
    prepareAnalysisState();
    const chainAudit = unavailableChainAudit();
    chainAudit.details = {
        first_failed_stage: 'quality_gate',
        error: {
            code: 'invalid_quality_gate_input',
            reasons: ['phrase_read_set_mismatch'],
        },
    };
    const payload = {
        code: 'r33_m1_review_projection_unavailable',
        reasons: ['phrase_chain_stage_failed'],
        trace_id: 'trace-unavailable',
        details: {},
        r33_m1_phrase_chain_audit_v1: chainAudit,
    };
    globalThis.fetch = async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => structuredClone(payload),
    });

    await assert.rejects(
        runAnalysis(1),
        error => {
            assert.ok(error instanceof AnalysisApiError);
            assert.equal(error.diagnostics?.firstFailedStage, 'quality_gate');
            return true;
        },
    );
});

test('preloaded OCR 503 shows the structured backend restart recovery guidance', async () => {
    prepareAnalysisState();
    const payload = {
        schema_version: 'vector_only_error_v1',
        status: 'fail_closed',
        code: 'vector_only_preloaded_runtime_forbidden',
        reasons: ['strict_forbidden_runtime_loaded'],
        forbidden_components: ['ocr:rapidocr'],
        recovery: 'restart_backend_required',
    };
    globalThis.fetch = async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => structuredClone(payload),
    });

    await assert.rejects(
        runAnalysis(1),
        error => {
            assert.ok(error instanceof AnalysisApiError);
            assert.equal(error.code, payload.code);
            assert.equal(error.recovery, payload.recovery);
            return true;
        },
    );

    assert.deepEqual(alerts, [[
        '智能解析失败：',
        'HTTP 503',
        `code: ${payload.code}`,
        `reasons: ${payload.reasons.join(', ')}`,
        '恢复指引：本进程已加载 OCR，需重启后端后恢复严格模式。',
        'message: HTTP 503 Service Unavailable',
    ].join('\n')]);
});

test('ordinary analysis errors keep the existing generic alert behavior', async () => {
    prepareAnalysisState();
    globalThis.fetch = async () => {
        throw new TypeError('network disconnected');
    };

    assert.equal(await runAnalysis(1), null);
    assert.deepEqual(alerts, ['智能解析失败：\nnetwork disconnected']);
    assert.equal(state.analyzeRuns.length, 1);
    assert.equal(state.analyzeRuns[0].error, 'network disconnected');
});
