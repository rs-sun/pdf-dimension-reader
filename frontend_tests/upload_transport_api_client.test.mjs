import test from 'node:test';
import assert from 'node:assert/strict';

const store = new Map();
globalThis.localStorage = {
    getItem: key => store.get(key) ?? null,
    setItem: (key, value) => store.set(key, String(value)),
    removeItem: key => store.delete(key),
};
globalThis.alert = () => {};

const { state } = await import('../state.js');
const { runExportXlsx, runStampOCR } = await import('../api_client.js');
const { TransportError } = await import('../legacy_pdf_transport.js');

function prepareTransportState() {
    state.currentPdfBuffer = Uint8Array.from([37, 80, 68, 70]).buffer;
    state.legacyJsonBase64MaxRequestBodyBytes = null;
    state.stampOcrCalls = [];
    state.exportCalls = [];
}

test('stamp OCR stays silent but records a stable code for HTML proxy 413', async () => {
    prepareTransportState();
    globalThis.fetch = async () => ({
        ok: false,
        status: 413,
        statusText: 'Request Entity Too Large',
        text: async () => '<html><h1>413 Request Entity Too Large</h1></html>',
    });

    assert.equal(
        await runStampOCR(1, { x: 1, y: 2, w: 3, h: 4 }, 300, 'stamp-1'),
        null,
    );

    assert.equal(state.stampOcrCalls.length, 1);
    assert.equal(state.stampOcrCalls[0].code, 'request_body_too_large');
    assert.equal(state.stampOcrCalls[0].http_status, 413);
    assert.match(state.stampOcrCalls[0].error, /HTTP 413/);
});

test('silent archive export propagates the original HTML proxy 413 cause', async () => {
    prepareTransportState();
    globalThis.fetch = async () => ({
        ok: false,
        status: 413,
        statusText: 'Request Entity Too Large',
        text: async () => '<html><h1>413 Request Entity Too Large</h1></html>',
    });

    await assert.rejects(
        runExportXlsx(1, {
            partNo: 'drawing',
            allPages: true,
            silent: true,
            dims: [],
        }),
        error => {
            assert.ok(error instanceof TransportError);
            assert.equal(error.code, 'request_body_too_large');
            assert.equal(error.status, 413);
            return true;
        },
    );

    assert.equal(state.exportCalls.length, 1);
    assert.equal(state.exportCalls[0].code, 'request_body_too_large');
    assert.equal(state.exportCalls[0].http_status, 413);
});

test('stamp deadline remains armed until the response body has been consumed', async () => {
    prepareTransportState();
    const originalSetTimeout = globalThis.setTimeout;
    const originalClearTimeout = globalThis.clearTimeout;
    const cleared = [];
    globalThis.setTimeout = () => 73;
    globalThis.clearTimeout = id => cleared.push(id);
    globalThis.fetch = async () => ({
        ok: true,
        status: 200,
        text: async () => {
            assert.deepEqual(cleared, []);
            return JSON.stringify({ status: 'success', text: 'OK', candidates: [] });
        },
    });

    try {
        assert.deepEqual(
            await runStampOCR(1, { x: 1, y: 2, w: 3, h: 4 }),
            { text: 'OK', candidates: [] },
        );
    } finally {
        globalThis.setTimeout = originalSetTimeout;
        globalThis.clearTimeout = originalClearTimeout;
    }

    assert.deepEqual(cleared, [73]);
});

test('export deadline remains armed until the binary body has been consumed', async () => {
    prepareTransportState();
    const originalSetTimeout = globalThis.setTimeout;
    const originalClearTimeout = globalThis.clearTimeout;
    const cleared = [];
    const expected = new Blob(['xlsx']);
    globalThis.setTimeout = () => 74;
    globalThis.clearTimeout = id => cleared.push(id);
    globalThis.fetch = async () => ({
        ok: true,
        status: 200,
        blob: async () => {
            assert.deepEqual(cleared, []);
            return expected;
        },
    });

    try {
        assert.strictEqual(
            await runExportXlsx(1, { partNo: 'drawing', dims: [] }),
            expected,
        );
    } finally {
        globalThis.setTimeout = originalSetTimeout;
        globalThis.clearTimeout = originalClearTimeout;
    }

    assert.deepEqual(cleared, [74]);
});

test('a structured 413 teaches later calls the effective server limit', async () => {
    prepareTransportState();
    let fetchCalls = 0;
    globalThis.fetch = async () => {
        fetchCalls += 1;
        return {
            ok: false,
            status: 413,
            statusText: 'Request Entity Too Large',
            text: async () => JSON.stringify({
                schema_version: 'request_error_v1',
                status: 'error',
                code: 'request_body_too_large',
                message: 'too large',
                max_request_body_bytes: 1,
                retryable: false,
            }),
        };
    };

    assert.equal(await runStampOCR(1, { x: 1, y: 2, w: 3, h: 4 }), null);
    assert.equal(state.legacyJsonBase64MaxRequestBodyBytes, 1);

    assert.equal(await runStampOCR(1, { x: 1, y: 2, w: 3, h: 4 }), null);
    assert.equal(fetchCalls, 1);
    assert.equal(state.stampOcrCalls.at(-1).code, 'request_body_too_large');
    assert.equal(state.stampOcrCalls.at(-1).http_status, null);
});

test('silent archive export preserves network and cancellation causes', async () => {
    prepareTransportState();
    for (const error of [
        new TypeError('socket reset'),
        Object.assign(new Error('aborted'), { name: 'AbortError' }),
    ]) {
        globalThis.fetch = async () => { throw error; };
        await assert.rejects(
            runExportXlsx(1, {
                partNo: 'drawing',
                allPages: true,
                silent: true,
                dims: [],
            }),
            caught => caught === error,
        );
    }
});

test('known oversize stamp and export fail before base64 allocation or fetch', async () => {
    prepareTransportState();
    state.legacyJsonBase64MaxRequestBodyBytes = 1;
    const originalBtoa = globalThis.btoa;
    let btoaCalls = 0;
    let fetchCalls = 0;
    globalThis.btoa = () => {
        btoaCalls += 1;
        throw new Error('known oversize must not encode');
    };
    globalThis.fetch = async () => {
        fetchCalls += 1;
        throw new Error('known oversize must not fetch');
    };

    try {
        assert.equal(await runStampOCR(1, { x: 1, y: 2, w: 3, h: 4 }), null);
        await assert.rejects(
            runExportXlsx(1, {
                partNo: 'drawing',
                allPages: true,
                silent: true,
                dims: [],
            }),
            error => error instanceof TransportError
                && error.code === 'request_body_too_large',
        );
    } finally {
        globalThis.btoa = originalBtoa;
    }

    assert.equal(btoaCalls, 0);
    assert.equal(fetchCalls, 0);
});

test('stamp records a stable contract error for an unparsable success body', async () => {
    prepareTransportState();
    globalThis.fetch = async () => ({
        ok: true,
        status: 200,
        statusText: 'OK',
        text: async () => '<html>not stamp JSON</html>',
    });

    assert.equal(await runStampOCR(1, { x: 1, y: 2, w: 3, h: 4 }), null);
    assert.equal(state.stampOcrCalls[0].code, 'invalid_response_payload');
    assert.equal(state.stampOcrCalls[0].http_status, 200);
});
