// Pure projection helpers for promoting a human-reviewed candidate into the
// existing formal dimension/stamp pipeline. Keeping these helpers free of DOM
// and project state makes the confirmation boundary directly testable.

const DECIMAL_TEXT = /^[+\-]?\d{1,3}(?:\.\d{1,3})?$/;

function clone(value) {
    return structuredClone(value);
}

function requireRecord(value, name) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new TypeError(`${name} must be an object`);
    }
    return value;
}

function hasExactKeys(value, keys) {
    const actual = Object.keys(requireRecord(value, 'reviewed values')).sort();
    const expected = [...keys].sort();
    return actual.length === expected.length
        && actual.every((key, index) => key === expected[index]);
}

function requireDecimal(value, name, { unsigned = false } = {}) {
    if (
        typeof value !== 'string'
        || !DECIMAL_TEXT.test(value)
        || (unsigned && (value.startsWith('+') || value.startsWith('-')))
    ) {
        throw new TypeError(`${name} must be ${unsigned ? 'unsigned ' : ''}decimal text`);
    }
    return value;
}

function requireLinearCandidate(candidate) {
    const prefix = candidate.prefix;
    const unit = candidate.unit;
    if (
        !(
            (['Ø', 'R', null].includes(prefix) && unit === null)
            || (prefix === null && unit === 'degree')
        )
    ) {
        throw new TypeError('linear candidate prefix/unit combination is invalid');
    }
    const hasQuantity = Object.hasOwn(candidate, 'quantity');
    const quantity = hasQuantity ? candidate.quantity : 1;
    if (
        !Number.isInteger(quantity)
        || (
            hasQuantity
                ? quantity < 2 || quantity > 999
                : quantity !== 1
        )
    ) {
        throw new TypeError('linear candidate quantity is invalid');
    }
    return { prefix, unit, quantity };
}

function requireLinearValues(values) {
    const source = requireRecord(values, 'reviewed values');
    if (!hasExactKeys(source, [
        'nominal', 'symmetric_tolerance',
        'upper_tolerance', 'lower_tolerance',
    ])) {
        throw new TypeError('linear reviewed values keys are invalid');
    }
    const nominal = requireDecimal(source.nominal, 'nominal');
    const symmetricTolerance = source.symmetric_tolerance;
    const upperTolerance = source.upper_tolerance;
    const lowerTolerance = source.lower_tolerance;
    if (symmetricTolerance !== null) {
        requireDecimal(symmetricTolerance, 'symmetric_tolerance', { unsigned: true });
        if (upperTolerance !== null || lowerTolerance !== null) {
            throw new TypeError('symmetric and asymmetric tolerances are mutually exclusive');
        }
    } else if (upperTolerance === null || lowerTolerance === null) {
        if (upperTolerance !== null || lowerTolerance !== null) {
            throw new TypeError('upper_tolerance and lower_tolerance must be paired');
        }
    } else {
        const upperValid = upperTolerance === '0'
            || (
                typeof upperTolerance === 'string'
                && upperTolerance.startsWith('+')
                && requireDecimal(
                    upperTolerance.slice(1),
                    'upper_tolerance',
                    { unsigned: true },
                )
            );
        const lowerValid = lowerTolerance === '0'
            || (
                typeof lowerTolerance === 'string'
                && lowerTolerance.startsWith('-')
                && requireDecimal(
                    lowerTolerance.slice(1),
                    'lower_tolerance',
                    { unsigned: true },
                )
            );
        if (!upperValid || !lowerValid || (
            upperTolerance === '0' && lowerTolerance === '0'
        )) {
            throw new TypeError('asymmetric tolerance fields are invalid');
        }
    }
    return {
        nominal,
        symmetricTolerance,
        upperTolerance,
        lowerTolerance,
    };
}

function requireGdtCandidate(candidate) {
    for (const key of ['symbol_name', 'symbol_unicode']) {
        if (typeof candidate[key] !== 'string' || candidate[key].length === 0) {
            throw new TypeError(`${key} must be non-empty text`);
        }
    }
    for (const key of ['datum_1', 'datum_2', 'datum_3']) {
        if (candidate[key] !== null) {
            throw new TypeError(`${key} must remain null on a review candidate`);
        }
    }
}

function requireGdtValues(values) {
    const source = requireRecord(values, 'reviewed values');
    for (const key of ['datum', 'datum_1', 'datum_2', 'datum_3']) {
        if (Object.hasOwn(source, key)) {
            throw new TypeError(`${key} is not editable in first-batch GD&T review`);
        }
    }
    return requireDecimal(source.tolerance_value, 'tolerance_value', { unsigned: true });
}

function reviewedLinearText(
    { prefix, unit, quantity },
    { nominal, symmetricTolerance, upperTolerance, lowerTolerance },
) {
    const suffix = unit === 'degree' ? '°' : '';
    const multiplier = quantity > 1 ? `${quantity}X ` : '';
    const main = `${multiplier}${prefix || ''}${nominal}${suffix}`;
    if (symmetricTolerance !== null) {
        return `${main}±${symmetricTolerance}${suffix}`;
    }
    if (upperTolerance !== null && lowerTolerance !== null) {
        return `${main} ${upperTolerance}${suffix}/${lowerTolerance}${suffix}`;
    }
    return main;
}

function reviewedLinearType(prefix, unit) {
    if (unit === 'degree') return 'angle';
    if (prefix === 'Ø') return 'diameter';
    if (prefix === 'R') return 'radius';
    return 'linear';
}

function gdtReviewedValues(candidate, toleranceValue) {
    return {
        symbol_name: candidate.symbol_name,
        symbol_unicode: candidate.symbol_unicode,
        tolerance_value: toleranceValue,
        datum_1: '',
        datum_2: '',
        datum_3: '',
    };
}

export function buildReviewedFormalDimension(candidateValue, values) {
    const candidate = requireRecord(candidateValue, 'candidate');
    if (candidate.type === 'linear') {
        const { prefix, unit, quantity } = requireLinearCandidate(candidate);
        const reviewed = requireLinearValues(values);
        const {
            nominal,
            symmetricTolerance,
            upperTolerance,
            lowerTolerance,
        } = reviewed;
        return {
            bbox: clone(candidate.bbox),
            text: reviewedLinearText({ prefix, unit, quantity }, reviewed),
            nominal,
            upper_tol: symmetricTolerance === null
                ? (upperTolerance ?? '')
                : `+${symmetricTolerance}`,
            lower_tol: symmetricTolerance === null
                ? (lowerTolerance ?? '')
                : `-${symmetricTolerance}`,
            prefix: prefix || '',
            unit: unit || '',
            ...(quantity > 1 ? { quantity } : {}),
            type: reviewedLinearType(prefix, unit),
            source: 'reviewed_vector',
            confidence: 'human_confirmed',
        };
    }
    if (candidate.type === 'gdt') {
        requireGdtCandidate(candidate);
        const toleranceValue = requireGdtValues(values);
        return {
            bbox: clone(candidate.bbox),
            text: `${candidate.symbol_unicode}${toleranceValue}`,
            nominal: toleranceValue,
            tolerance_value: toleranceValue,
            upper_tol: '',
            lower_tol: '',
            prefix: candidate.symbol_unicode,
            symbol_name: candidate.symbol_name,
            symbol_unicode: candidate.symbol_unicode,
            modifier: '',
            datum: '',
            datum_1: '',
            datum_2: '',
            datum_3: '',
            type: 'gdt',
            source: 'reviewed_vector',
            confidence: 'human_confirmed',
        };
    }
    throw new TypeError(`unsupported review candidate type: ${String(candidate.type)}`);
}

export function buildReviewConfirmationPayload(candidateValue, values, formalStampUuid) {
    const candidate = requireRecord(candidateValue, 'candidate');
    if (typeof formalStampUuid !== 'string' || formalStampUuid.length === 0) {
        throw new TypeError('formalStampUuid is required');
    }
    if (candidate.type === 'linear') {
        requireLinearCandidate(candidate);
        const {
            nominal,
            symmetricTolerance,
            upperTolerance,
            lowerTolerance,
        } = requireLinearValues(values);
        return {
            formalStampUuid,
            nominal,
            symmetric_tolerance: symmetricTolerance,
            upper_tolerance: upperTolerance,
            lower_tolerance: lowerTolerance,
        };
    }
    if (candidate.type === 'gdt') {
        requireGdtCandidate(candidate);
        return {
            formalStampUuid,
            tolerance_value: requireGdtValues(values),
        };
    }
    throw new TypeError(`unsupported review candidate type: ${String(candidate.type)}`);
}

export function buildReviewProvenance(entryValue, values) {
    const entry = requireRecord(entryValue, 'review entry');
    const envelope = requireRecord(entry.envelope, 'review envelope');
    const candidate = requireRecord(entry.item, 'candidate');
    let reviewedValues;
    if (candidate.type === 'linear') {
        const { prefix, unit, quantity } = requireLinearCandidate(candidate);
        const {
            nominal,
            symmetricTolerance,
            upperTolerance,
            lowerTolerance,
        } = requireLinearValues(values);
        reviewedValues = {
            nominal,
            symmetric_tolerance: symmetricTolerance,
            upper_tolerance: upperTolerance,
            lower_tolerance: lowerTolerance,
            prefix,
            unit,
            ...(quantity > 1 ? { quantity } : {}),
        };
    } else if (candidate.type === 'gdt') {
        requireGdtCandidate(candidate);
        reviewedValues = gdtReviewedValues(
            candidate,
            requireGdtValues(values),
        );
    } else {
        throw new TypeError(`unsupported review candidate type: ${String(candidate.type)}`);
    }
    return {
        schema_version: envelope.schema_version,
        candidate_id: candidate.candidate_id,
        trace_id: envelope.trace_id,
        page_index: envelope.page_index,
        coordinate_space: envelope.coordinate_space,
        original_candidate: clone(candidate),
        reviewed_values: reviewedValues,
        audit: clone(envelope.audit),
        action: 'human_confirmed',
    };
}
