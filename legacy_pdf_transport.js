function _rawByteLength(buffer) {
    const byteLength = buffer?.byteLength;
    if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
        throw new TypeError('PDF buffer must expose a non-negative byteLength');
    }
    return byteLength;
}

function _bytes(buffer) {
    if (ArrayBuffer.isView(buffer)) {
        return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }
    return new Uint8Array(buffer);
}

function _bodyWithEmptyPdf(bodyWithoutPdf) {
    if (bodyWithoutPdf === null || typeof bodyWithoutPdf !== 'object' || Array.isArray(bodyWithoutPdf)) {
        throw new TypeError('JSON request fields must be an object');
    }
    if (Object.hasOwn(bodyWithoutPdf, 'pdf_base64')) {
        throw new TypeError('pdf_base64 is owned by the transport');
    }
    return { pdf_base64: '', ...bodyWithoutPdf };
}

export function estimateJsonBase64RequestBytes(buffer, bodyWithoutPdf) {
    const fixedJsonBytes = new TextEncoder().encode(
        JSON.stringify(_bodyWithEmptyPdf(bodyWithoutPdf)),
    ).byteLength;
    return fixedJsonBytes + 4 * Math.ceil(_rawByteLength(buffer) / 3);
}

export class TransportError extends Error {
    constructor({
        code,
        message,
        status = null,
        estimatedRequestBodyBytes = null,
        maxRequestBodyBytes = null,
        retryable = false,
        payload = null,
        cause,
    }) {
        super(message, cause === undefined ? undefined : { cause });
        this.name = 'TransportError';
        this.code = code;
        this.status = status;
        this.estimatedRequestBodyBytes = estimatedRequestBodyBytes;
        this.maxRequestBodyBytes = maxRequestBodyBytes;
        this.retryable = retryable;
        this.payload = payload;
    }
}

function _arrayBufferToBase64(buffer) {
    const bytes = _bytes(buffer);
    let binary = '';
    const chunk = 8192;
    for (let index = 0; index < bytes.byteLength; index += chunk) {
        binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
    }
    return btoa(binary);
}

export function prepareJsonBase64Body(
    buffer,
    bodyWithoutPdf,
    maxRequestBodyBytes = null,
) {
    const estimatedRequestBodyBytes = estimateJsonBase64RequestBytes(buffer, bodyWithoutPdf);
    if (
        Number.isSafeInteger(maxRequestBodyBytes)
        && maxRequestBodyBytes >= 0
        && estimatedRequestBodyBytes > maxRequestBodyBytes
    ) {
        throw new TransportError({
            code: 'request_body_too_large',
            message: `当前请求预计 ${estimatedRequestBodyBytes} 字节，超过服务端 ${maxRequestBodyBytes} 字节上限`,
            estimatedRequestBodyBytes,
            maxRequestBodyBytes,
            retryable: false,
        });
    }

    const body = JSON.stringify({
        pdf_base64: _arrayBufferToBase64(buffer),
        ...bodyWithoutPdf,
    });
    return Object.freeze({
        body,
        requestBodyBytes: estimatedRequestBodyBytes,
    });
}

function _isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export async function readResponsePayload(response) {
    if (typeof response?.text === 'function') {
        try {
            const text = await response.text();
            if (!text || !text.trim()) return null;
            try {
                const parsed = JSON.parse(text);
                return _isRecord(parsed) ? parsed : null;
            } catch (_error) {
                return null;
            }
        } catch (error) {
            throw error;
        }
    }
    if (typeof response?.json === 'function') {
        try {
            const parsed = await response.json();
            return _isRecord(parsed) ? parsed : null;
        } catch (error) {
            throw error;
        }
    }
    return null;
}

function _safeByteCount(value, fallback = null) {
    return Number.isSafeInteger(value) && value >= 0 ? value : fallback;
}

export function legacyJsonBase64MaxRequestBodyBytes(payload) {
    const limits = payload?.transport_limits;
    const legacy = limits?.legacy_json_base64;
    if (
        limits?.schema_version !== 'transport_limits_v1'
        || legacy?.supported !== true
        || legacy?.size_includes !== 'entire_http_body'
        || legacy?.encoding !== 'rfc4648_base64_in_utf8_json'
    ) return null;
    const maxRequestBodyBytes = _safeByteCount(legacy.max_request_body_bytes);
    return maxRequestBodyBytes !== null && maxRequestBodyBytes > 0
        ? maxRequestBodyBytes
        : null;
}

export function normalizeTransportError(response, payload, context = {}) {
    if (payload instanceof TransportError) return payload;
    const record = _isRecord(payload) ? payload : null;
    const status = Number.isInteger(response?.status) ? response.status : null;
    const statusText = typeof response?.statusText === 'string'
        ? response.statusText.trim()
        : '';
    const defaultCodes = {
        413: 'request_body_too_large',
        502: 'upstream_unavailable',
        504: 'upstream_timeout',
    };
    const code = status === 413
        ? 'request_body_too_large'
        : (typeof record?.code === 'string' && record.code
        ? record.code
        : (defaultCodes[status] || (status === null ? 'transport_request_failed' : `http_${status}`)));
    const fallbackMessage = status === null
        ? 'Transport request failed'
        : `HTTP ${status}${statusText ? ` ${statusText}` : ''}`;
    const message = typeof record?.message === 'string' && record.message.trim()
        ? record.message.trim()
        : fallbackMessage;
    const maxRequestBodyBytes = _safeByteCount(
        record?.max_request_body_bytes,
        _safeByteCount(context.maxRequestBodyBytes),
    );
    const estimatedRequestBodyBytes = _safeByteCount(context.estimatedRequestBodyBytes);
    const retryable = status === 413
        ? false
        : (typeof record?.retryable === 'boolean'
        ? record.retryable
        : status === 502 || status === 504);
    return new TransportError({
        code,
        message,
        status,
        estimatedRequestBodyBytes,
        maxRequestBodyBytes,
        retryable,
        payload: record,
    });
}
