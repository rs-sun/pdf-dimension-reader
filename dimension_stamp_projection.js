// Pure dimension -> formal stamp metadata projection shared by
// dimensionsToStamps and contract/export tests.

const DIMENSION_TYPE_LABELS = Object.freeze({
    linear: '线性',
    angle: '角度',
    angular: '角度',
    diameter: '线性',
    radius: '线性',
    gdt: '形位公差',
    gd_t: '形位公差',
    reference: '线性',
    thread: '线性',
    chamfer: '线性',
    repeat: '线性',
    limit_max: '线性',
    limit_min: '线性',
    surface_roughness: '粗糙度',
});

function hasValue(value) {
    return value !== null && value !== undefined && String(value).trim() !== '';
}

function firstPresent(...values) {
    for (const value of values) {
        if (value !== null && value !== undefined) return value;
    }
    return undefined;
}

function firstNonBlank(...values) {
    for (const value of values) {
        if (hasValue(value)) return value;
    }
    return '';
}

function toText(value) {
    if (value === null || value === undefined) return '';
    return String(value);
}

export function dimensionTypeLabel(type) {
    return DIMENSION_TYPE_LABELS[type] || '--';
}

export function buildStampAiData(dimension, dimType = dimensionTypeLabel(dimension?.type)) {
    if (!dimension || typeof dimension !== 'object' || Array.isArray(dimension)) {
        throw new TypeError('dimension must be an object');
    }
    const upperTol = hasValue(dimension.upper_tol) ? dimension.upper_tol : '';
    const lowerTol = hasValue(dimension.lower_tol) ? dimension.lower_tol : '';
    const tolerance = hasValue(upperTol) && hasValue(lowerTol)
        ? `${upperTol}/${lowerTol}`
        : (upperTol || lowerTol || '');

    const result = {
        type: dimType,
        nominal: toText(firstPresent(
            dimension.nominal,
            dimension.nominal_text,
            dimension.text,
        )),
        tolerance,
        upper_tol: toText(upperTol),
        lower_tol: toText(lowerTol),
        prefix: dimension.prefix || '',
        unit: dimension.unit || '',
        symbol: dimension.prefix || '',
        datum: dimension.datum || '',
        datum_1: dimension.datum_1 || '',
        datum_2: dimension.datum_2 || '',
        datum_3: dimension.datum_3 || '',
        modifier: dimension.modifier || dimension.gdt_modifiers || '',
        remarks: '',
        confidence: dimension.confidence || 'medium',
        _raw_type: dimension.type || '',
        _symbol_name: dimension.symbol_name || '',
        _symbol_unicode: dimension.symbol_unicode || '',
        _symbol_method: dimension.symbol_method || dimension.method || '',
        _gdt_symbol_confidence: toText(firstPresent(
            dimension.gdt_symbol_confidence,
            dimension.vector_confidence,
            dimension.yolo_confidence,
            '',
        )),
        _gdt_frame_symbols: dimension.gdt_frame_symbols || [],
        _limit_value: dimension.limit_value || '',
        _raw_text: toText(firstNonBlank(
            dimension.text,
            dimension.raw_text,
            dimension._raw_text,
            dimension.nominal_text,
            dimension.nominal,
        )),
    };
    if (
        Number.isInteger(dimension.quantity)
        && dimension.quantity >= 2
        && dimension.quantity <= 999
    ) {
        result.quantity = dimension.quantity;
    }
    return result;
}

export function buildStampDimensionProjection(dimension) {
    if (!dimension || typeof dimension !== 'object' || Array.isArray(dimension)) {
        throw new TypeError('dimension must be an object');
    }
    const isKey = Boolean(dimension.is_key || dimension.in_capsule_blue);
    const dimType = dimensionTypeLabel(dimension.type);
    return {
        source: dimension.source || 'ocr',
        is_key: isKey,
        type: isKey ? 'key' : 'normal',
        dimType,
        aiData: buildStampAiData(dimension, dimType),
    };
}
