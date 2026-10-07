// R33-M1 sole frontend consumer for contracts/review_candidates_v1.canonical.json.
// There is intentionally no legacy alias or candidate-shape adapter in this module.

export const REVIEW_CANDIDATES_ROOT_KEY = 'review_candidates_v1';
export const REVIEW_CANDIDATES_SCHEMA_VERSION = 'review_candidates_v1';
export const REVIEW_CANDIDATES_AUDIT_VERSION = 'review_candidates_audit_v1';
export const REVIEW_CANDIDATES_COORDINATE_SPACE = 'pdf_page_points_top_left';
export const REVIEW_CANDIDATES_SOURCE_STAGE = 'L5_review_required';

const ENVELOPE_KEYS = [
    'schema_version', 'trace_id', 'page_index', 'coordinate_space', 'items', 'audit',
];
const LINEAR_ITEM_KEYS = [
    'candidate_id', 'bbox', 'type', 'nominal', 'symmetric_tolerance',
    'upper_tolerance', 'lower_tolerance', 'prefix', 'unit',
    'review_status', 'source_stage', 'trace',
];
const GDT_ITEM_KEYS = [
    'candidate_id', 'bbox', 'type', 'symbol_name', 'symbol_unicode',
    'tolerance_value', 'datum_1', 'datum_2', 'datum_3',
    'review_status', 'source_stage', 'trace',
];
const OPTIONAL_ITEM_KEYS = ['syntax'];
const LINEAR_OPTIONAL_ITEM_KEYS = [...OPTIONAL_ITEM_KEYS, 'quantity'];
const BBOX_KEYS = ['x', 'y', 'w', 'h'];
const LINEAR_ITEM_TRACE_KEYS = [
    'l5_review_id', 'phrase_id', 'physical_core_digest',
    'source_l3_semantic_digest', 'source_text', 'canonical_text',
    'candidate_digest',
];
const GDT_ITEM_TRACE_KEYS = [
    'frame_digest', 'decimal_quad_id', 'symbol_method',
    'symbol_confidence', 'symbol_compartment_index',
    'tolerance_compartment_index', 'reader_phrase_id',
    'template_content_sha256', 'candidate_digest',
];
const GDT_CANDIDATE_TRACE_KEYS = GDT_ITEM_TRACE_KEYS.filter(
    key => key !== 'candidate_digest',
);
const AUDIT_KEYS = [
    'schema_version', 'status', 'unavailable_reason', 'chain', 'counts',
    'template_identity', 'runtime', 'authority', 'semantic_digest',
];
const CHAIN_KEYS = [
    'schema_version', 'trace_id', 'page_index', 'status',
    'semantic_projection_sha256', 'stage_status', 'stage_digests', 'chain_counts',
];
const STAGE_STATUS_KEYS = [
    'phrase_region', 'pitch_reader', 'quality_gate', 'hypothesis',
    'spatial_dedup', 'l3', 'l4', 'l5',
];
const STAGE_DIGEST_KEYS = [
    'pitch_reader', 'quality_gate', 'hypothesis', 'spatial_dedup', 'l3', 'l4', 'l5',
];
const CHAIN_COUNT_KEYS = [
    'page_context_build_count', 'page_context_extraction_call_count',
    'phrase_region_count', 'reader_call_count', 'reader_success_count',
    'quality_pass_count', 'ordinary_hypothesis_count', 'dedup_winner_count',
    'l3_dimension_count', 'l4_recovered_count', 'l5_review_required_count',
];
const COUNTS_KEYS = [
    'source_review_count', 'candidate_count', 'dropped_count', 'drop_reasons',
];
const DROP_REASON_KEYS = ['reason', 'count'];
const TEMPLATE_KEYS = ['schema_version', 'template_version', 'content_sha256'];
const RUNTIME_KEYS = [
    'ocr_family_call_count', 'paddleocr_call_count', 'rapidocr_call_count',
    'yolo_char_call_count', 'yolo_b_call_count',
];
const AUTHORITY_KEYS = [
    'review_only', 'formal_project_data_allowed', 'consumer_allowed', 'release_allowed',
];
const DROP_REASON_ORDER = [
    'source_not_l3_dimension', 'source_trace_invalid', 'semantics_invalid',
    'kind_not_linear', 'quantity_not_one', 'prefix_not_null', 'unit_not_null',
    'qualifier_not_null', 'tolerance_kind_not_supported', 'decimal_text_invalid',
    'bbox_invalid', 'text_missing', 'final_position_duplicate',
];

const DECIMAL_TEXT = /^[+\-]?\d{1,3}(?:\.\d{1,3})?$/;
const SHA256 = /^[0-9a-f]{64}$/;
const DIMENSION_SYNTAX_LABELS = new Set([
    'ok',
    'incomplete_separator',
    'unread_glyph_present',
    'invalid_dimension_syntax',
]);
const GDT_CORE_SYMBOLS = Object.freeze({
    position: '⊕',
    perpendicularity: '⊥',
    parallelism: '∥',
    coaxiality: '◎',
    roundness: '○',
    cylindricity: '⌭',
    straightness: '—',
    angularity: '∠',
    surface_profile: '⌓',
    line_profile: '⌒',
    flatness: '▱',
    symmetry: '≡',
});

// Digest verification mirrors backend/review_candidates_contract.py.  The
// producer normalizes every bbox coordinate to a Python float before hashing,
// so integral coordinates must retain the `.0` that JSON.parse normally loses.
class PythonJsonFloat {
    constructor(value) {
        this.value = value;
    }
}

const SHA256_INITIAL = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const SHA256_ROUND = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

function rotateRight(value, amount) {
    return (value >>> amount) | (value << (32 - amount));
}

function sha256HexUtf8(text) {
    const input = new TextEncoder().encode(text);
    const byteLength = Math.ceil((input.length + 9) / 64) * 64;
    const padded = new Uint8Array(byteLength);
    padded.set(input);
    padded[input.length] = 0x80;
    const view = new DataView(padded.buffer);
    view.setUint32(byteLength - 8, Math.floor(input.length / 0x20000000), false);
    view.setUint32(byteLength - 4, (input.length * 8) >>> 0, false);

    const hash = [...SHA256_INITIAL];
    const words = new Uint32Array(64);
    for (let offset = 0; offset < byteLength; offset += 64) {
        for (let index = 0; index < 16; index += 1) {
            words[index] = view.getUint32(offset + index * 4, false);
        }
        for (let index = 16; index < 64; index += 1) {
            const left = words[index - 15];
            const right = words[index - 2];
            const sigma0 = rotateRight(left, 7) ^ rotateRight(left, 18) ^ (left >>> 3);
            const sigma1 = rotateRight(right, 17) ^ rotateRight(right, 19) ^ (right >>> 10);
            words[index] = (
                words[index - 16] + sigma0 + words[index - 7] + sigma1
            ) >>> 0;
        }

        let [a, b, c, d, e, f, g, h] = hash;
        for (let index = 0; index < 64; index += 1) {
            const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
            const choose = (e & f) ^ (~e & g);
            const temp1 = (h + sum1 + choose + SHA256_ROUND[index] + words[index]) >>> 0;
            const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
            const majority = (a & b) ^ (a & c) ^ (b & c);
            const temp2 = (sum0 + majority) >>> 0;
            h = g;
            g = f;
            f = e;
            e = (d + temp1) >>> 0;
            d = c;
            c = b;
            b = a;
            a = (temp1 + temp2) >>> 0;
        }
        hash[0] = (hash[0] + a) >>> 0;
        hash[1] = (hash[1] + b) >>> 0;
        hash[2] = (hash[2] + c) >>> 0;
        hash[3] = (hash[3] + d) >>> 0;
        hash[4] = (hash[4] + e) >>> 0;
        hash[5] = (hash[5] + f) >>> 0;
        hash[6] = (hash[6] + g) >>> 0;
        hash[7] = (hash[7] + h) >>> 0;
    }
    return hash.map(value => value.toString(16).padStart(8, '0')).join('');
}

function pythonFloatLiteral(value) {
    if (!Number.isFinite(value)) throw new TypeError('non-finite digest number');
    if (Object.is(value, -0)) return '-0.0';
    const sign = value < 0 ? '-' : '';
    const match = Math.abs(value).toExponential().match(
        /^(\d)(?:\.(\d+))?e([+\-])(\d+)$/,
    );
    if (!match) throw new TypeError('unsupported digest number');
    const [, leading, fraction = '', exponentSign, exponentDigits] = match;
    const exponent = Number(`${exponentSign}${exponentDigits}`);
    if (exponent < -4 || exponent >= 16) {
        const mantissa = fraction ? `${leading}.${fraction}` : leading;
        const formattedExponent = String(Math.abs(exponent)).padStart(2, '0');
        return `${sign}${mantissa}e${exponent < 0 ? '-' : '+'}${formattedExponent}`;
    }
    const digits = `${leading}${fraction}`;
    const decimalIndex = exponent + 1;
    let fixed;
    if (decimalIndex <= 0) {
        fixed = `0.${'0'.repeat(-decimalIndex)}${digits}`;
    } else if (decimalIndex >= digits.length) {
        fixed = `${digits}${'0'.repeat(decimalIndex - digits.length)}.0`;
    } else {
        fixed = `${digits.slice(0, decimalIndex)}.${digits.slice(decimalIndex)}`;
    }
    return `${sign}${fixed}`;
}

function pythonCanonicalJson(value) {
    if (value instanceof PythonJsonFloat) return pythonFloatLiteral(value.value);
    if (value === null) return 'null';
    if (typeof value === 'string') return JSON.stringify(value);
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'number') {
        if (!Number.isFinite(value)) throw new TypeError('non-finite digest number');
        if (!Number.isInteger(value) || Object.is(value, -0)) return pythonFloatLiteral(value);
        return String(value);
    }
    if (Array.isArray(value)) {
        return `[${value.map(pythonCanonicalJson).join(',')}]`;
    }
    if (value !== null && typeof value === 'object') {
        return `{${Object.keys(value).sort().map(key => (
            `${JSON.stringify(key)}:${pythonCanonicalJson(value[key])}`
        )).join(',')}}`;
    }
    throw new TypeError('unsupported digest value');
}

function frozenBboxForDigest(bbox) {
    return Object.fromEntries(BBOX_KEYS.map(key => [
        key,
        new PythonJsonFloat(bbox[key]),
    ]));
}

function hasWidenedLinearValue(item) {
    return [
        item.upper_tolerance,
        item.lower_tolerance,
        item.prefix,
        item.unit,
    ].some(value => value !== null);
}

function computeLinearCandidateDigest(item, pageIndex) {
    const payload = {
        schema_version: REVIEW_CANDIDATES_SCHEMA_VERSION,
        page_index: pageIndex,
        physical_core_digest: item.trace.physical_core_digest,
        bbox: frozenBboxForDigest(item.bbox),
        nominal: item.nominal,
        symmetric_tolerance: item.symmetric_tolerance,
    };
    if (hasWidenedLinearValue(item)) {
        Object.assign(payload, {
            type: 'linear',
            upper_tolerance: item.upper_tolerance,
            lower_tolerance: item.lower_tolerance,
            prefix: item.prefix,
            unit: item.unit,
        });
    }
    if (Object.hasOwn(item, 'quantity')) payload.quantity = item.quantity;
    if (Object.hasOwn(item, 'syntax')) payload.syntax = item.syntax;
    return sha256HexUtf8(pythonCanonicalJson(payload));
}

function frozenGdtTraceForDigest(trace, { includeCandidateDigest = false } = {}) {
    const keys = includeCandidateDigest
        ? GDT_ITEM_TRACE_KEYS
        : GDT_CANDIDATE_TRACE_KEYS;
    return Object.fromEntries(keys.map(key => [
        key,
        key === 'symbol_confidence'
            ? new PythonJsonFloat(trace[key])
            : trace[key],
    ]));
}

function computeGdtCandidateDigest(item, pageIndex) {
    const payload = {
        schema_version: REVIEW_CANDIDATES_SCHEMA_VERSION,
        page_index: pageIndex,
        type: 'gdt',
        bbox: frozenBboxForDigest(item.bbox),
        symbol_name: item.symbol_name,
        symbol_unicode: item.symbol_unicode,
        tolerance_value: item.tolerance_value,
        datum_1: null,
        datum_2: null,
        datum_3: null,
        trace: frozenGdtTraceForDigest(item.trace),
    };
    if (Object.hasOwn(item, 'syntax')) payload.syntax = item.syntax;
    return sha256HexUtf8(pythonCanonicalJson(payload));
}

function computeCandidateDigest(item, pageIndex) {
    return item.type === 'gdt'
        ? computeGdtCandidateDigest(item, pageIndex)
        : computeLinearCandidateDigest(item, pageIndex);
}

function computeSemanticDigest(items, pageIndex) {
    const semanticItems = items.map(item => {
        if (item.type === 'gdt') {
            const gdtItem = {
                candidate_id: item.candidate_id,
                bbox: frozenBboxForDigest(item.bbox),
                type: 'gdt',
                symbol_name: item.symbol_name,
                symbol_unicode: item.symbol_unicode,
                tolerance_value: item.tolerance_value,
                datum_1: item.datum_1,
                datum_2: item.datum_2,
                datum_3: item.datum_3,
                review_status: item.review_status,
                source_stage: item.source_stage,
                trace: frozenGdtTraceForDigest(item.trace, {
                    includeCandidateDigest: true,
                }),
            };
            if (Object.hasOwn(item, 'syntax')) gdtItem.syntax = item.syntax;
            return gdtItem;
        }
        const linearItem = {
            candidate_id: item.candidate_id,
            bbox: frozenBboxForDigest(item.bbox),
            nominal: item.nominal,
            symmetric_tolerance: item.symmetric_tolerance,
            review_status: item.review_status,
            source_stage: item.source_stage,
            physical_core_digest: item.trace?.physical_core_digest,
        };
        if (hasWidenedLinearValue(item)) {
            Object.assign(linearItem, {
                type: 'linear',
                upper_tolerance: item.upper_tolerance,
                lower_tolerance: item.lower_tolerance,
                prefix: item.prefix,
                unit: item.unit,
            });
        }
        if (Object.hasOwn(item, 'quantity')) {
            linearItem.quantity = item.quantity;
        }
        if (Object.hasOwn(item, 'syntax')) linearItem.syntax = item.syntax;
        return linearItem;
    });
    return sha256HexUtf8(pythonCanonicalJson({
        schema_version: REVIEW_CANDIDATES_SCHEMA_VERSION,
        page_index: pageIndex,
        coordinate_space: REVIEW_CANDIDATES_COORDINATE_SPACE,
        items: semanticItems,
    }));
}

function clone(value) {
    return structuredClone(value);
}

function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasExactKeys(value, expected) {
    if (!isRecord(value)) return false;
    const actual = Object.keys(value).sort();
    const wanted = [...expected].sort();
    return actual.length === wanted.length
        && actual.every((key, index) => key === wanted[index]);
}

function hasRequiredAndOptionalKeys(value, required, optional) {
    if (!isRecord(value)) return false;
    const actual = new Set(Object.keys(value));
    return required.every(key => actual.has(key))
        && [...actual].every(
            key => required.includes(key) || optional.includes(key),
        );
}

function isText(value) {
    return typeof value === 'string' && value.length > 0 && !value.includes('\0');
}

function isCount(value) {
    return Number.isInteger(value) && value >= 0;
}

function isPageIndex(value) {
    return Number.isInteger(value) && value >= 0;
}

function isDecimal(value) {
    return typeof value === 'string' && DECIMAL_TEXT.test(value);
}

function isUnsignedDecimal(value) {
    return isDecimal(value) && !value.startsWith('+') && !value.startsWith('-');
}

function isValidLinearTolerances(symmetric, upper, lower) {
    if (symmetric !== null) {
        return isUnsignedDecimal(symmetric)
            && upper === null
            && lower === null;
    }
    if (upper === null || lower === null) {
        return upper === null && lower === null;
    }
    const validUpper = upper === '0'
        || (
            typeof upper === 'string'
            && upper.startsWith('+')
            && isUnsignedDecimal(upper.slice(1))
        );
    const validLower = lower === '0'
        || (
            typeof lower === 'string'
            && lower.startsWith('-')
            && isUnsignedDecimal(lower.slice(1))
        );
    return validUpper && validLower && !(upper === '0' && lower === '0');
}

function isSha256(value) {
    return typeof value === 'string' && SHA256.test(value);
}

function isBbox(value) {
    return hasExactKeys(value, BBOX_KEYS)
        && BBOX_KEYS.every(key => Number.isFinite(value[key]))
        && value.w > 0
        && value.h > 0;
}

function addReason(reasons, condition, reason) {
    if (!condition) reasons.push(reason);
}

function validateCommonItemFields(item, pageIndex, expectedType) {
    const reasons = [];
    addReason(reasons, isText(item.candidate_id), 'item_candidate_id_invalid');
    addReason(reasons, isBbox(item.bbox), 'item_bbox_invalid');
    addReason(reasons, item.type === expectedType, 'item_type_invalid');
    addReason(reasons, item.review_status === 'pending', 'item_review_status_invalid');
    addReason(reasons, item.source_stage === REVIEW_CANDIDATES_SOURCE_STAGE, 'item_source_stage_invalid');
    addReason(reasons, isPageIndex(pageIndex), 'item_page_identity_invalid');
    return reasons;
}

function validateLinearItem(item, pageIndex) {
    const reasons = validateCommonItemFields(item, pageIndex, 'linear');
    addReason(
        reasons,
        hasRequiredAndOptionalKeys(
            item,
            LINEAR_ITEM_KEYS,
            LINEAR_OPTIONAL_ITEM_KEYS,
        ),
        'item_keys_invalid',
    );
    const syntaxValid = !Object.hasOwn(item, 'syntax')
        || DIMENSION_SYNTAX_LABELS.has(item.syntax);
    addReason(reasons, syntaxValid, 'item_syntax_invalid');
    const correctionCandidate = item.nominal === null
        && DIMENSION_SYNTAX_LABELS.has(item.syntax)
        && item.syntax !== 'ok';
    addReason(
        reasons,
        isDecimal(item.nominal) || correctionCandidate,
        'item_nominal_invalid',
    );
    addReason(
        reasons,
        isValidLinearTolerances(
            item.symmetric_tolerance,
            item.upper_tolerance,
            item.lower_tolerance,
        )
            && (
                !correctionCandidate
                || [
                    item.symmetric_tolerance,
                    item.upper_tolerance,
                    item.lower_tolerance,
                ].every(value => value === null)
            ),
        'item_tolerance_invalid',
    );
    addReason(
        reasons,
        item.prefix === null || item.prefix === 'Ø' || item.prefix === 'R',
        'item_prefix_invalid',
    );
    addReason(
        reasons,
        item.unit === null || item.unit === 'degree',
        'item_unit_invalid',
    );
    addReason(
        reasons,
        (
            (['Ø', 'R', null].includes(item.prefix) && item.unit === null)
            || (item.prefix === null && item.unit === 'degree')
        ),
        'item_prefix_unit_invalid',
    );
    const hasQuantity = Object.hasOwn(item, 'quantity');
    const quantityValid = !hasQuantity || (
        Number.isInteger(item.quantity)
        && item.quantity >= 2
        && item.quantity <= 999
    );
    addReason(reasons, quantityValid, 'item_quantity_invalid');
    if (!hasExactKeys(item.trace, LINEAR_ITEM_TRACE_KEYS)) {
        reasons.push('item_trace_keys_invalid');
        return reasons;
    }
    for (const key of ['l5_review_id', 'phrase_id', 'source_text', 'canonical_text']) {
        addReason(reasons, isText(item.trace[key]), `item_trace_${key}_invalid`);
    }
    addReason(reasons, isSha256(item.trace.physical_core_digest), 'item_physical_core_digest_invalid');
    addReason(reasons, isSha256(item.trace.source_l3_semantic_digest), 'item_l3_digest_invalid');
    addReason(reasons, isSha256(item.trace.candidate_digest), 'item_candidate_digest_invalid');
    const toleranceValid = isValidLinearTolerances(
        item.symmetric_tolerance,
        item.upper_tolerance,
        item.lower_tolerance,
    );
    const prefixUnitValid = (
        (['Ø', 'R', null].includes(item.prefix) && item.unit === null)
        || (item.prefix === null && item.unit === 'degree')
    );
    if (
        isPageIndex(pageIndex)
        && isBbox(item.bbox)
        && (isDecimal(item.nominal) || correctionCandidate)
        && toleranceValid
        && prefixUnitValid
        && quantityValid
        && isSha256(item.trace.physical_core_digest)
    ) {
        const expectedDigest = computeCandidateDigest(item, pageIndex);
        addReason(
            reasons,
            item.trace.candidate_digest === expectedDigest,
            'item_candidate_digest_invalid',
        );
        addReason(
            reasons,
            item.candidate_id === `rcv1_${expectedDigest.slice(0, 24)}`,
            'item_candidate_id_invalid',
        );
    }
    return reasons;
}

function validateGdtTraceFields(trace) {
    const reasons = [];
    addReason(reasons, isSha256(trace.frame_digest), 'item_gdt_frame_digest_invalid');
    addReason(reasons, isText(trace.decimal_quad_id), 'item_gdt_decimal_quad_id_invalid');
    addReason(reasons, trace.symbol_method === 'vector_inversion', 'item_gdt_symbol_method_invalid');
    addReason(
        reasons,
        Number.isFinite(trace.symbol_confidence)
            && trace.symbol_confidence >= 0.60
            && trace.symbol_confidence <= 1.0,
        'item_gdt_symbol_confidence_invalid',
    );
    addReason(
        reasons,
        Number.isInteger(trace.symbol_compartment_index)
            && trace.symbol_compartment_index === 0,
        'item_gdt_symbol_compartment_invalid',
    );
    addReason(
        reasons,
        Number.isInteger(trace.tolerance_compartment_index)
            && trace.tolerance_compartment_index >= 1,
        'item_gdt_tolerance_compartment_invalid',
    );
    addReason(reasons, isText(trace.reader_phrase_id), 'item_gdt_reader_phrase_id_invalid');
    addReason(
        reasons,
        isSha256(trace.template_content_sha256),
        'item_gdt_template_sha_invalid',
    );
    return reasons;
}

function isGdtSymbolPair(symbolName, symbolUnicode) {
    return typeof symbolName === 'string'
        && typeof symbolUnicode === 'string'
        && GDT_CORE_SYMBOLS[symbolName] === symbolUnicode;
}

function validateGdtItem(item, pageIndex) {
    const reasons = validateCommonItemFields(item, pageIndex, 'gdt');
    addReason(
        reasons,
        hasRequiredAndOptionalKeys(
            item,
            GDT_ITEM_KEYS,
            OPTIONAL_ITEM_KEYS,
        ),
        'item_keys_invalid',
    );
    const syntaxValid = !Object.hasOwn(item, 'syntax')
        || DIMENSION_SYNTAX_LABELS.has(item.syntax);
    addReason(reasons, syntaxValid, 'item_syntax_invalid');
    addReason(
        reasons,
        isGdtSymbolPair(item.symbol_name, item.symbol_unicode),
        'item_gdt_symbol_invalid',
    );
    const correctionCandidate = isText(item.tolerance_value)
        && DIMENSION_SYNTAX_LABELS.has(item.syntax)
        && item.syntax !== 'ok';
    const toleranceValid = isUnsignedDecimal(item.tolerance_value)
        || correctionCandidate;
    addReason(reasons, toleranceValid, 'item_gdt_tolerance_invalid');
    const datumBlank = ['datum_1', 'datum_2', 'datum_3'].every(
        key => item[key] === null,
    );
    addReason(reasons, datumBlank, 'item_gdt_datum_not_null');
    if (!hasExactKeys(item.trace, GDT_ITEM_TRACE_KEYS)) {
        reasons.push('item_trace_keys_invalid');
        return reasons;
    }
    const traceReasons = validateGdtTraceFields(item.trace);
    reasons.push(...traceReasons);
    addReason(
        reasons,
        isSha256(item.trace.candidate_digest),
        'item_candidate_digest_invalid',
    );
    if (
        isPageIndex(pageIndex)
        && isBbox(item.bbox)
        && isGdtSymbolPair(item.symbol_name, item.symbol_unicode)
        && toleranceValid
        && datumBlank
        && traceReasons.length === 0
    ) {
        const expectedDigest = computeGdtCandidateDigest(item, pageIndex);
        addReason(
            reasons,
            item.trace.candidate_digest === expectedDigest,
            'item_candidate_digest_invalid',
        );
        addReason(
            reasons,
            item.candidate_id === `rcv1_${expectedDigest.slice(0, 24)}`,
            'item_candidate_id_invalid',
        );
    }
    return reasons;
}

function validateItem(item, pageIndex) {
    if (!isRecord(item)) return ['item_not_object'];
    if (item.type === 'linear') return validateLinearItem(item, pageIndex);
    if (item.type === 'gdt') return validateGdtItem(item, pageIndex);
    return ['item_keys_invalid', 'item_type_invalid'];
}

function validateChain(chain, { traceId, pageIndex, status }) {
    const reasons = [];
    if (!hasExactKeys(chain, CHAIN_KEYS)) return ['audit_chain_keys_invalid'];
    addReason(reasons, chain.schema_version === 'r33_m1_phrase_chain_audit_v1', 'audit_chain_schema_invalid');
    addReason(reasons, chain.trace_id === traceId, 'audit_chain_trace_invalid');
    addReason(reasons, chain.page_index === pageIndex, 'audit_chain_page_invalid');
    addReason(reasons, hasExactKeys(chain.stage_status, STAGE_STATUS_KEYS), 'audit_stage_status_keys_invalid');
    addReason(reasons, hasExactKeys(chain.stage_digests, STAGE_DIGEST_KEYS), 'audit_stage_digest_keys_invalid');
    addReason(reasons, hasExactKeys(chain.chain_counts, CHAIN_COUNT_KEYS), 'audit_chain_count_keys_invalid');
    if (status === 'ok') {
        addReason(reasons, chain.status === 'ok', 'audit_chain_status_invalid');
        addReason(reasons, isSha256(chain.semantic_projection_sha256), 'audit_chain_semantic_digest_invalid');
        if (hasExactKeys(chain.stage_status, STAGE_STATUS_KEYS)) {
            addReason(reasons, STAGE_STATUS_KEYS.every(key => chain.stage_status[key] === 'ok'), 'audit_stage_status_invalid');
        }
        if (hasExactKeys(chain.stage_digests, STAGE_DIGEST_KEYS)) {
            addReason(reasons, STAGE_DIGEST_KEYS.every(key => isSha256(chain.stage_digests[key])), 'audit_stage_digest_invalid');
        }
        if (hasExactKeys(chain.chain_counts, CHAIN_COUNT_KEYS)) {
            addReason(reasons, CHAIN_COUNT_KEYS.every(key => isCount(chain.chain_counts[key])), 'audit_chain_count_invalid');
        }
    } else {
        addReason(reasons, chain.status === 'unavailable', 'unavailable_chain_status_invalid');
        addReason(reasons, chain.semantic_projection_sha256 === null, 'unavailable_chain_digest_invalid');
        if (hasExactKeys(chain.stage_digests, STAGE_DIGEST_KEYS)) {
            addReason(reasons, STAGE_DIGEST_KEYS.every(key => chain.stage_digests[key] === null), 'unavailable_stage_digest_invalid');
        }
        if (hasExactKeys(chain.chain_counts, CHAIN_COUNT_KEYS)) {
            addReason(reasons, CHAIN_COUNT_KEYS.every(key => chain.chain_counts[key] === 0), 'unavailable_chain_count_invalid');
        }
    }
    return reasons;
}

function validateCounts(counts, { itemCount, status }) {
    const reasons = [];
    if (!hasExactKeys(counts, COUNTS_KEYS)) return ['audit_counts_keys_invalid'];
    for (const key of ['source_review_count', 'candidate_count', 'dropped_count']) {
        addReason(reasons, isCount(counts[key]), `audit_${key}_invalid`);
    }
    if (!Array.isArray(counts.drop_reasons)) {
        reasons.push('audit_drop_reasons_invalid');
        return reasons;
    }
    const names = [];
    let dropSum = 0;
    for (const drop of counts.drop_reasons) {
        if (!hasExactKeys(drop, DROP_REASON_KEYS)) {
            reasons.push('audit_drop_reason_keys_invalid');
            continue;
        }
        names.push(drop.reason);
        addReason(reasons, DROP_REASON_ORDER.includes(drop.reason), 'audit_drop_reason_invalid');
        addReason(reasons, isCount(drop.count) && drop.count > 0, 'audit_drop_count_invalid');
        if (isCount(drop.count)) dropSum += drop.count;
    }
    const ordered = DROP_REASON_ORDER.filter(reason => names.includes(reason));
    addReason(reasons, names.length === new Set(names).size && names.every((name, index) => name === ordered[index]), 'audit_drop_order_invalid');
    addReason(reasons, dropSum === counts.dropped_count, 'audit_dropped_count_mismatch');
    addReason(reasons, counts.candidate_count === itemCount, 'audit_candidate_count_mismatch');
    if (status === 'ok') {
        addReason(
            reasons,
            counts.source_review_count === counts.candidate_count + counts.dropped_count,
            'audit_source_count_mismatch',
        );
    } else {
        addReason(
            reasons,
            counts.source_review_count === 0 && counts.candidate_count === 0 && counts.dropped_count === 0,
            'unavailable_counts_invalid',
        );
    }
    return reasons;
}

function validateAudit(audit, { traceId, pageIndex, items }) {
    const reasons = [];
    if (!hasExactKeys(audit, AUDIT_KEYS)) return ['audit_keys_invalid'];
    addReason(reasons, audit.schema_version === REVIEW_CANDIDATES_AUDIT_VERSION, 'audit_schema_invalid');
    addReason(reasons, audit.status === 'ok' || audit.status === 'unavailable', 'audit_status_invalid');
    const status = audit.status;
    if (status === 'ok') {
        addReason(reasons, audit.unavailable_reason === null, 'audit_unavailable_reason_invalid');
    } else if (status === 'unavailable') {
        addReason(reasons, isText(audit.unavailable_reason), 'audit_unavailable_reason_invalid');
        addReason(reasons, items.length === 0, 'unavailable_items_invalid');
    }
    reasons.push(...validateChain(audit.chain, { traceId, pageIndex, status }));
    reasons.push(...validateCounts(audit.counts, { itemCount: items.length, status }));
    if (!hasExactKeys(audit.template_identity, TEMPLATE_KEYS)) {
        reasons.push('audit_template_keys_invalid');
    } else if (status === 'ok') {
        addReason(reasons, audit.template_identity.schema_version === 'r33_m1_template_identity_v1', 'audit_template_schema_invalid');
        addReason(reasons, isText(audit.template_identity.template_version), 'audit_template_version_invalid');
        addReason(reasons, isSha256(audit.template_identity.content_sha256), 'audit_template_sha_invalid');
    } else {
        addReason(reasons, audit.template_identity.schema_version === 'r33_m1_template_identity_v1', 'audit_template_schema_invalid');
        addReason(reasons, audit.template_identity.template_version === null, 'unavailable_template_version_invalid');
        addReason(reasons, audit.template_identity.content_sha256 === null, 'unavailable_template_sha_invalid');
    }
    if (
        status === 'ok'
        && isRecord(audit.template_identity)
        && isSha256(audit.template_identity.content_sha256)
        && items.some(item => (
            isRecord(item)
            && item.type === 'gdt'
            && (
                !isRecord(item.trace)
                || item.trace.template_content_sha256
                    !== audit.template_identity.content_sha256
            )
        ))
    ) {
        reasons.push('item_gdt_template_identity_mismatch');
    }
    if (!hasExactKeys(audit.runtime, RUNTIME_KEYS)) {
        reasons.push('audit_runtime_keys_invalid');
    } else {
        addReason(reasons, RUNTIME_KEYS.every(key => audit.runtime[key] === 0), 'audit_runtime_nonzero');
    }
    if (!hasExactKeys(audit.authority, AUTHORITY_KEYS)) {
        reasons.push('audit_authority_keys_invalid');
    } else {
        addReason(reasons, audit.authority.review_only === true, 'audit_review_only_invalid');
        addReason(reasons, audit.authority.formal_project_data_allowed === false, 'audit_formal_project_authority_invalid');
        addReason(reasons, audit.authority.consumer_allowed === false, 'audit_consumer_authority_invalid');
        addReason(reasons, audit.authority.release_allowed === false, 'audit_release_authority_invalid');
    }
    addReason(reasons, isSha256(audit.semantic_digest), 'audit_semantic_digest_invalid');
    try {
        addReason(
            reasons,
            audit.semantic_digest === computeSemanticDigest(items, pageIndex),
            'audit_semantic_digest_invalid',
        );
    } catch (_error) {
        reasons.push('audit_semantic_payload_invalid');
    }
    return reasons;
}

export function validateReviewCandidatesV1(value, { expectedPageIndex } = {}) {
    const reasons = [];
    if (!hasExactKeys(value, ENVELOPE_KEYS)) return ['envelope_keys_invalid'];
    addReason(reasons, value.schema_version === REVIEW_CANDIDATES_SCHEMA_VERSION, 'schema_version_invalid');
    addReason(reasons, isText(value.trace_id), 'trace_id_invalid');
    addReason(reasons, isPageIndex(value.page_index), 'page_index_invalid');
    if (expectedPageIndex !== undefined) {
        addReason(reasons, value.page_index === expectedPageIndex, 'page_index_mismatch');
    }
    addReason(reasons, value.coordinate_space === REVIEW_CANDIDATES_COORDINATE_SPACE, 'coordinate_space_invalid');
    if (!Array.isArray(value.items)) {
        reasons.push('items_invalid');
        return [...new Set(reasons)].sort();
    }
    const ids = [];
    for (const item of value.items) {
        reasons.push(...validateItem(item, value.page_index));
        if (isRecord(item) && typeof item.candidate_id === 'string') ids.push(item.candidate_id);
    }
    addReason(
        reasons,
        ids.length === value.items.length
            && ids.length === new Set(ids).size
            && ids.every((id, index) => index === 0 || ids[index - 1] <= id),
        'candidate_order_or_uniqueness_invalid',
    );
    reasons.push(...validateAudit(value.audit, {
        traceId: value.trace_id,
        pageIndex: value.page_index,
        items: value.items,
    }));
    return [...new Set(reasons)].sort();
}

export class ReviewCandidatesContractError extends TypeError {
    constructor(code, reasons = []) {
        super(`${code}${reasons.length ? `: ${reasons.join(', ')}` : ''}`);
        this.name = 'ReviewCandidatesContractError';
        this.code = code;
        this.reasons = [...reasons];
    }
}

function workspaceClone(workspace) {
    const value = isRecord(workspace) ? workspace : {};
    return {
        envelopes: clone(Array.isArray(value.envelopes) ? value.envelopes : []),
        decisions: clone(isRecord(value.decisions) ? value.decisions : {}),
        selectedCandidateId: typeof value.selectedCandidateId === 'string'
            ? value.selectedCandidateId
            : null,
    };
}

export function ingestReviewCandidatesV1(workspace, envelope, { expectedPageIndex } = {}) {
    const reasons = validateReviewCandidatesV1(envelope, { expectedPageIndex });
    if (reasons.length) {
        throw new ReviewCandidatesContractError('review_candidates_v1_invalid', reasons);
    }
    if (envelope.audit.status !== 'ok') {
        throw new ReviewCandidatesContractError(
            'review_candidates_v1_unavailable',
            [envelope.audit.unavailable_reason],
        );
    }
    const next = workspaceClone(workspace);
    next.envelopes = next.envelopes
        .filter(existing => existing.page_index !== envelope.page_index)
        .concat([clone(envelope)])
        .sort((left, right) => left.page_index - right.page_index);
    const liveIds = new Set(next.envelopes.flatMap(item => item.items.map(candidate => candidate.candidate_id)));
    if (next.selectedCandidateId && !liveIds.has(next.selectedCandidateId)) {
        next.selectedCandidateId = null;
    }
    return next;
}

export function listReviewCandidateEntries(workspace, { pageIndex } = {}) {
    const current = workspaceClone(workspace);
    const entries = [];
    for (const envelope of current.envelopes) {
        if (pageIndex !== undefined && envelope.page_index !== pageIndex) continue;
        for (const item of envelope.items) {
            const decision = current.decisions[item.candidate_id] || null;
            const base = {
                envelope,
                item,
                decision,
                status: decision?.status || 'pending',
                type: item.type,
            };
            if (item.type === 'gdt') {
                entries.push({
                    ...base,
                    symbol_name: item.symbol_name,
                    symbol_unicode: item.symbol_unicode,
                    tolerance_value: decision?.tolerance_value ?? item.tolerance_value,
                    datum_1: null,
                    datum_2: null,
                    datum_3: null,
                });
            } else {
                entries.push({
                    ...base,
                    nominal: decision?.nominal ?? item.nominal,
                    symmetric_tolerance: Object.hasOwn(decision || {}, 'symmetric_tolerance')
                        ? decision.symmetric_tolerance
                        : item.symmetric_tolerance,
                    upper_tolerance: Object.hasOwn(decision || {}, 'upper_tolerance')
                        ? decision.upper_tolerance
                        : item.upper_tolerance,
                    lower_tolerance: Object.hasOwn(decision || {}, 'lower_tolerance')
                        ? decision.lower_tolerance
                        : item.lower_tolerance,
                    prefix: item.prefix,
                    unit: item.unit,
                    ...(Object.hasOwn(item, 'quantity')
                        ? { quantity: item.quantity }
                        : {}),
                });
            }
        }
    }
    return entries;
}

function requireEntry(workspace, candidateId) {
    const entry = listReviewCandidateEntries(workspace).find(
        candidate => candidate.item.candidate_id === candidateId,
    );
    if (!entry) throw new RangeError(`unknown review candidate: ${candidateId}`);
    return entry;
}

function validateLinearReviewValues(values, { includeFormalStampUuid = false } = {}) {
    const keys = includeFormalStampUuid
        ? [
            'formalStampUuid', 'nominal', 'symmetric_tolerance',
            'upper_tolerance', 'lower_tolerance',
        ]
        : [
            'nominal', 'symmetric_tolerance',
            'upper_tolerance', 'lower_tolerance',
        ];
    if (!hasExactKeys(values, keys)) {
        throw new TypeError('linear review values keys are invalid');
    }
    const {
        nominal,
        symmetric_tolerance: symmetricTolerance,
        upper_tolerance: upperTolerance,
        lower_tolerance: lowerTolerance,
    } = values;
    if (!isDecimal(nominal)) throw new TypeError('nominal must be decimal text');
    if (!isValidLinearTolerances(
        symmetricTolerance,
        upperTolerance,
        lowerTolerance,
    )) {
        throw new TypeError('linear tolerance fields are invalid');
    }
    return {
        nominal,
        symmetricTolerance,
        upperTolerance,
        lowerTolerance,
    };
}

function validateGdtReviewValues(values, { includeFormalStampUuid = false } = {}) {
    const keys = includeFormalStampUuid
        ? ['formalStampUuid', 'tolerance_value']
        : ['tolerance_value'];
    if (!hasExactKeys(values, keys)) {
        throw new TypeError('gdt review values keys are invalid');
    }
    if (!isUnsignedDecimal(values.tolerance_value)) {
        throw new TypeError('tolerance_value must be unsigned decimal text');
    }
    return { toleranceValue: values.tolerance_value };
}

export function editReviewCandidate(workspace, candidateId, values) {
    const entry = requireEntry(workspace, candidateId);
    if (entry.status === 'confirmed' || entry.status === 'dismissed') {
        throw new TypeError(`candidate is already ${entry.status}`);
    }
    const next = workspaceClone(workspace);
    if (entry.item.type === 'gdt') {
        const { toleranceValue } = validateGdtReviewValues(values);
        if (toleranceValue === entry.item.tolerance_value) {
            delete next.decisions[candidateId];
        } else {
            next.decisions[candidateId] = {
                status: 'edited',
                tolerance_value: toleranceValue,
            };
        }
        return next;
    }
    const {
        nominal,
        symmetricTolerance,
        upperTolerance,
        lowerTolerance,
    } = validateLinearReviewValues(values);
    if (
        nominal === entry.item.nominal
        && symmetricTolerance === entry.item.symmetric_tolerance
        && upperTolerance === entry.item.upper_tolerance
        && lowerTolerance === entry.item.lower_tolerance
    ) {
        delete next.decisions[candidateId];
    } else {
        next.decisions[candidateId] = {
            status: 'edited',
            nominal,
            symmetric_tolerance: symmetricTolerance,
            upper_tolerance: upperTolerance,
            lower_tolerance: lowerTolerance,
        };
    }
    return next;
}

export function dismissReviewCandidate(workspace, candidateId) {
    const entry = requireEntry(workspace, candidateId);
    if (entry.status === 'confirmed') throw new TypeError('confirmed candidate cannot be dismissed');
    const next = workspaceClone(workspace);
    next.decisions[candidateId] = { status: 'dismissed' };
    if (next.selectedCandidateId === candidateId) next.selectedCandidateId = null;
    return next;
}

export function confirmReviewCandidate(workspace, candidateId, values) {
    const entry = requireEntry(workspace, candidateId);
    if (entry.status === 'dismissed') throw new TypeError('dismissed candidate cannot be confirmed');
    if (entry.status === 'confirmed') {
        if (entry.decision.formal_stamp_uuid !== values.formalStampUuid) {
            throw new TypeError('candidate already confirmed with a different formal stamp');
        }
        return workspaceClone(workspace);
    }
    const formalStampUuid = values.formalStampUuid;
    if (!isText(formalStampUuid)) throw new TypeError('formalStampUuid is required');
    const next = workspaceClone(workspace);
    if (entry.item.type === 'gdt') {
        const { toleranceValue } = validateGdtReviewValues(values, {
            includeFormalStampUuid: true,
        });
        next.decisions[candidateId] = {
            status: 'confirmed',
            tolerance_value: toleranceValue,
            formal_stamp_uuid: formalStampUuid,
        };
        if (next.selectedCandidateId === candidateId) next.selectedCandidateId = null;
        return next;
    }
    const {
        nominal,
        symmetricTolerance,
        upperTolerance,
        lowerTolerance,
    } = validateLinearReviewValues(values, {
        includeFormalStampUuid: true,
    });
    next.decisions[candidateId] = {
        status: 'confirmed',
        nominal,
        symmetric_tolerance: symmetricTolerance,
        upper_tolerance: upperTolerance,
        lower_tolerance: lowerTolerance,
        formal_stamp_uuid: formalStampUuid,
    };
    if (next.selectedCandidateId === candidateId) next.selectedCandidateId = null;
    return next;
}

export function selectReviewCandidate(workspace, candidateId) {
    requireEntry(workspace, candidateId);
    const next = workspaceClone(workspace);
    next.selectedCandidateId = candidateId;
    return next;
}

export function pendingReviewCandidateCount(workspace) {
    return listReviewCandidateEntries(workspace).filter(
        entry => entry.status === 'pending' || entry.status === 'edited',
    ).length;
}

function normalizePages(pages) {
    const values = [...new Set((pages || []).map(Number))];
    if (values.some(page => !Number.isInteger(page) || page < 1)) {
        throw new TypeError('pages must contain positive integers');
    }
    return values;
}

function documentGeneration(state) {
    return Number.isSafeInteger(state?.documentGeneration)
        && state.documentGeneration >= 0
        ? state.documentGeneration
        : 0;
}

function staleReviewOutcome({
    state,
    expectedDocumentGeneration,
    staged = [],
    failedPages = [],
    pageFailures = [],
    skippedPages = [],
}) {
    return {
        status: 'stale',
        code: 'document_generation_changed',
        kind: REVIEW_CANDIDATES_SCHEMA_VERSION,
        succeededPages: [],
        discardedPages: staged.map(envelope => envelope.page_index + 1),
        failedPages,
        ...(skippedPages.length ? { skippedPages } : {}),
        statePreserved: true,
        pageFailures,
        expectedDocumentGeneration,
        currentDocumentGeneration: documentGeneration(state),
    };
}

export async function executeReviewCandidatesTransaction({
    state,
    pages,
    analyzePage,
    beforeCommit,
    pushHistory,
    signal,
    onProgress,
}) {
    if (state.isAnalysisRunning) {
        return {
            status: 'busy', kind: REVIEW_CANDIDATES_SCHEMA_VERSION,
            succeededPages: [], failedPages: [], statePreserved: true, pageFailures: [],
        };
    }
    const requestedPages = normalizePages(pages);
    if (!requestedPages.length) {
        return {
            status: 'failed', kind: REVIEW_CANDIDATES_SCHEMA_VERSION,
            succeededPages: [], failedPages: [], statePreserved: true, pageFailures: [],
        };
    }
    state.isAnalysisRunning = true;
    const expectedDocumentGeneration = documentGeneration(state);
    const before = workspaceClone(state.reviewWorkspace);
    const staged = [];
    const failedPages = [];
    const pageFailures = [];
    const skippedPages = [];
    try {
        for (let index = 0; index < requestedPages.length; index += 1) {
            if (documentGeneration(state) !== expectedDocumentGeneration) {
                return staleReviewOutcome({
                    state,
                    expectedDocumentGeneration,
                    staged,
                    failedPages,
                    pageFailures,
                    skippedPages,
                });
            }
            if (signal?.aborted) {
                return {
                    status: 'cancelled', kind: REVIEW_CANDIDATES_SCHEMA_VERSION,
                    succeededPages: [], failedPages, statePreserved: true, pageFailures,
                };
            }
            const page = requestedPages[index];
            let stopAfterCurrentPage = false;
            try {
                const response = await analyzePage(page, { signal });
                if (signal?.aborted) {
                    return {
                        status: 'cancelled', kind: REVIEW_CANDIDATES_SCHEMA_VERSION,
                        succeededPages: [], failedPages, statePreserved: true, pageFailures,
                    };
                }
                if (documentGeneration(state) !== expectedDocumentGeneration) {
                    return staleReviewOutcome({
                        state,
                        expectedDocumentGeneration,
                        staged,
                        failedPages,
                        pageFailures,
                        skippedPages,
                    });
                }
                if (!isRecord(response) || response.status !== 'success') {
                    throw new ReviewCandidatesContractError('analysis_status_not_success');
                }
                if (!Object.hasOwn(response, REVIEW_CANDIDATES_ROOT_KEY)) {
                    throw new ReviewCandidatesContractError('review_candidates_v1_missing');
                }
                const envelope = response[REVIEW_CANDIDATES_ROOT_KEY];
                const reasons = validateReviewCandidatesV1(envelope, {
                    expectedPageIndex: page - 1,
                });
                if (reasons.length) {
                    throw new ReviewCandidatesContractError('review_candidates_v1_invalid', reasons);
                }
                if (envelope.audit.status !== 'ok') {
                    throw new ReviewCandidatesContractError(
                        'review_candidates_v1_unavailable',
                        [envelope.audit.unavailable_reason],
                    );
                }
                staged.push(envelope);
            } catch (error) {
                failedPages.push(page);
                const failure = {
                    pageIndex: page,
                    code: error?.code || 'review_candidates_v1_request_failed',
                    reasons: Array.isArray(error?.reasons) ? [...error.reasons] : [],
                    message: error?.message || String(error),
                    trace: typeof error?.trace === 'string' ? error.trace : null,
                    status: Number.isInteger(error?.status) ? error.status : null,
                    retryable: error?.retryable !== false,
                };
                const payload = isRecord(error?.payload) ? error.payload : null;
                if (payload && Object.hasOwn(payload, REVIEW_CANDIDATES_ROOT_KEY)) {
                    failure[REVIEW_CANDIDATES_ROOT_KEY] = payload[REVIEW_CANDIDATES_ROOT_KEY];
                }
                if (payload && Object.hasOwn(payload, 'r33_m1_phrase_chain_audit_v1')) {
                    failure.r33_m1_phrase_chain_audit_v1 = payload.r33_m1_phrase_chain_audit_v1;
                }
                pageFailures.push(failure);
                stopAfterCurrentPage = error?.retryable === false;
            }
            onProgress?.({
                completed: index + 1,
                total: requestedPages.length,
                pageIndex: page,
            });
            if (stopAfterCurrentPage) {
                skippedPages.push(...requestedPages.slice(index + 1));
                break;
            }
        }
        if (documentGeneration(state) !== expectedDocumentGeneration) {
            return staleReviewOutcome({
                state,
                expectedDocumentGeneration,
                staged,
                failedPages,
                pageFailures,
                skippedPages,
            });
        }
        if (!staged.length) {
            return {
                status: 'failed', kind: REVIEW_CANDIDATES_SCHEMA_VERSION,
                succeededPages: [], failedPages, statePreserved: true, pageFailures,
                ...(skippedPages.length ? { skippedPages } : {}),
            };
        }
        let next = before;
        for (const envelope of staged) {
            next = ingestReviewCandidatesV1(next, envelope, {
                expectedPageIndex: envelope.page_index,
            });
        }
        beforeCommit?.();
        if (documentGeneration(state) !== expectedDocumentGeneration) {
            return staleReviewOutcome({
                state,
                expectedDocumentGeneration,
                staged,
                failedPages,
                pageFailures,
                skippedPages,
            });
        }
        state.reviewWorkspace = next;
        pushHistory?.();
        const succeededPages = staged.map(envelope => envelope.page_index + 1);
        return {
            status: failedPages.length ? 'partial' : 'success',
            kind: REVIEW_CANDIDATES_SCHEMA_VERSION,
            succeededPages,
            failedPages,
            ...(skippedPages.length ? { skippedPages } : {}),
            statePreserved: false,
            pageFailures,
        };
    } catch (error) {
        if (documentGeneration(state) === expectedDocumentGeneration) {
            state.reviewWorkspace = before;
        }
        throw error;
    } finally {
        state.isAnalysisRunning = false;
    }
}
