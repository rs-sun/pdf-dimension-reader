import test from 'node:test';
import assert from 'node:assert/strict';

import {
    TransportError,
    estimateJsonBase64RequestBytes,
    legacyJsonBase64MaxRequestBodyBytes,
    normalizeTransportError,
    prepareJsonBase64Body,
    readResponsePayload,
} from '../legacy_pdf_transport.js';

test('legacy JSON estimator counts base64 quanta and UTF-8 payload bytes exactly', () => {
    const bodyWithoutPdf = {
        page: 1,
        part_no: '零件😀',
        dims: [{ nominal: '直径⌀10' }],
    };
    const fixedBytes = new TextEncoder().encode(JSON.stringify({
        pdf_base64: '',
        ...bodyWithoutPdf,
    })).byteLength;

    for (const [rawBytes, encodedBytes] of [[0, 0], [1, 4], [2, 4], [3, 4], [4, 8]]) {
        const buffer = new Uint8Array(rawBytes).buffer;
        assert.equal(
            estimateJsonBase64RequestBytes(buffer, bodyWithoutPdf),
            fixedBytes + encodedBytes,
        );
    }
});

test('transport capability accepts only the explicit whole-body legacy schema', () => {
    const valid = {
        transport_limits: {
            schema_version: 'transport_limits_v1',
            legacy_json_base64: {
                supported: true,
                max_request_body_bytes: 33_554_432,
                size_includes: 'entire_http_body',
                encoding: 'rfc4648_base64_in_utf8_json',
            },
        },
    };

    assert.equal(legacyJsonBase64MaxRequestBodyBytes(valid), 33_554_432);
    assert.equal(legacyJsonBase64MaxRequestBodyBytes({}), null);
    assert.equal(legacyJsonBase64MaxRequestBodyBytes({
        ...valid,
        transport_limits: {
            ...valid.transport_limits,
            legacy_json_base64: {
                ...valid.transport_limits.legacy_json_base64,
                max_request_body_bytes: 0,
            },
        },
    }), null);
    assert.equal(legacyJsonBase64MaxRequestBodyBytes({
        ...valid,
        transport_limits: {
            ...valid.transport_limits,
            legacy_json_base64: {
                ...valid.transport_limits.legacy_json_base64,
                size_includes: 'raw_pdf_only',
            },
        },
    }), null);
});

test('default 50 MiB analysis is measured as a 69,905,189-byte request', () => {
    const fiftyMiB = { byteLength: 50 * 1024 * 1024 };
    assert.equal(estimateJsonBase64RequestBytes(fiftyMiB, {
        page: 1,
        dpi: 200,
        settings: {
            dimension_path: 'vector_only',
            view_seg: 'yolo',
            gdt_backend: 'hybrid',
        },
    }), 69_905_189);
});

test('prepared request matches the estimate exactly at the configured boundary', () => {
    const buffer = Uint8Array.from([0, 1, 2, 3]).buffer;
    const fields = { page: 1, part_no: '零件😀' };
    const exactLimit = estimateJsonBase64RequestBytes(buffer, fields);

    const prepared = prepareJsonBase64Body(buffer, fields, exactLimit);

    assert.equal(prepared.requestBodyBytes, exactLimit);
    assert.equal(new TextEncoder().encode(prepared.body).byteLength, exactLimit);
    assert.deepEqual(JSON.parse(prepared.body), {
        pdf_base64: 'AAECAw==',
        ...fields,
    });
});

test('HTML proxy 413 normalizes to the stable request-body error', async () => {
    const response = {
        ok: false,
        status: 413,
        statusText: 'Request Entity Too Large',
        text: async () => '<html><h1>413 Request Entity Too Large</h1></html>',
    };

    const payload = await readResponsePayload(response);
    const error = normalizeTransportError(response, payload, {
        estimatedRequestBodyBytes: 70_000_000,
        maxRequestBodyBytes: 33_554_432,
    });

    assert.equal(payload, null);
    assert.ok(error instanceof TransportError);
    assert.equal(error.code, 'request_body_too_large');
    assert.equal(error.status, 413);
    assert.equal(error.estimatedRequestBodyBytes, 70_000_000);
    assert.equal(error.maxRequestBodyBytes, 33_554_432);
    assert.equal(error.retryable, false);
    assert.match(error.message, /HTTP 413/);
});

test('structured Flask 413 preserves the configured limit and server message', () => {
    const payload = {
        schema_version: 'request_error_v1',
        status: 'error',
        code: 'request_body_too_large',
        message: '请求体超过服务端上限',
        max_request_body_bytes: 33_554_432,
        retryable: false,
    };
    const error = normalizeTransportError({
        status: 413,
        statusText: 'Request Entity Too Large',
    }, payload, { estimatedRequestBodyBytes: 69_905_189 });

    assert.equal(error.code, payload.code);
    assert.equal(error.message, payload.message);
    assert.equal(error.maxRequestBodyBytes, payload.max_request_body_bytes);
    assert.equal(error.estimatedRequestBodyBytes, 69_905_189);
    assert.strictEqual(error.payload, payload);
});

test('HTTP 413 is non-retryable even when an untrusted payload says otherwise', () => {
    const error = normalizeTransportError({
        status: 413,
        statusText: 'Request Entity Too Large',
    }, {
        code: 'request_body_too_large',
        message: 'too large',
        retryable: true,
    });

    assert.equal(error.code, 'request_body_too_large');
    assert.equal(error.retryable, false);
});

test('response payload reader preserves cancellation from a streaming body', async () => {
    const aborted = Object.assign(new Error('aborted while reading body'), {
        name: 'AbortError',
    });

    await assert.rejects(
        readResponsePayload({ text: async () => { throw aborted; } }),
        error => error === aborted,
    );
});

test('response payload reader preserves non-abort streaming failures', async () => {
    const disconnected = new TypeError('socket reset while reading body');

    await assert.rejects(
        readResponsePayload({ text: async () => { throw disconnected; } }),
        error => error === disconnected,
    );
});

test('oversize legacy request fails before base64 allocation', () => {
    const originalBtoa = globalThis.btoa;
    let btoaCalls = 0;
    globalThis.btoa = () => {
        btoaCalls += 1;
        throw new Error('btoa must not run for a known oversize request');
    };

    try {
        const buffer = new Uint8Array(65).buffer;
        const fields = { page: 1, dpi: 200, settings: {} };
        const estimated = estimateJsonBase64RequestBytes(buffer, fields);

        assert.throws(
            () => prepareJsonBase64Body(buffer, fields, estimated - 1),
            error => {
                assert.ok(error instanceof TransportError);
                assert.equal(error.code, 'request_body_too_large');
                assert.equal(error.estimatedRequestBodyBytes, estimated);
                assert.equal(error.maxRequestBodyBytes, estimated - 1);
                assert.equal(error.retryable, false);
                return true;
            },
        );
        assert.equal(btoaCalls, 0);
    } finally {
        globalThis.btoa = originalBtoa;
    }
});
