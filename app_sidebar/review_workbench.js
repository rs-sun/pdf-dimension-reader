// Pure helpers for the review workbench.  Keep selection and value-shape
// decisions out of DOM handlers so Node tests can verify the strict review UX.

export const REVIEW_GDT_DATUM_NOTICE = '基准：首批不识别（留空）';

const GDT_SYMBOL_CHINESE_NAMES = Object.freeze({
    position: '位置度',
    perpendicularity: '垂直度',
    parallelism: '平行度',
    coaxiality: '同轴度',
    roundness: '圆度',
    cylindricity: '圆柱度',
    straightness: '直线度',
    angularity: '倾斜度',
    surface_profile: '面轮廓度',
    line_profile: '线轮廓度',
    flatness: '平面度',
    symmetry: '对称度',
});

export function reviewCandidateType(entry) {
    const type = entry?.item?.type;
    if (type === 'linear' || type === 'gdt') return type;
    throw new TypeError(`unsupported review candidate type: ${type}`);
}

export function reviewGdtSymbolChineseName(symbolName) {
    return GDT_SYMBOL_CHINESE_NAMES[symbolName] || '未知形位符号';
}

export function isActionableReviewEntry(entry) {
    return entry?.status === 'pending' || entry?.status === 'edited';
}

/**
 * Return the explicitly checked, still-actionable ids in visible entry order.
 * There is intentionally no "all entries" fallback: an empty selection stays
 * empty, preserving the strict human-confirmation boundary.
 */
export function selectedReviewCandidateIds(entries, selectedCandidateIds) {
    const selected = new Set(selectedCandidateIds || []);
    return (entries || [])
        .filter(entry => (
            isActionableReviewEntry(entry)
            && selected.has(entry.item.candidate_id)
        ))
        .map(entry => entry.item.candidate_id);
}

export async function runSelectedReviewCandidateActions(
    entries,
    selectedCandidateIds,
    action,
) {
    if (typeof action !== 'function') {
        throw new TypeError('review batch action must be a function');
    }
    const attempted = selectedReviewCandidateIds(entries, selectedCandidateIds);
    const succeeded = [];
    const failed = [];
    for (const candidateId of attempted) {
        try {
            await action(candidateId);
            succeeded.push(candidateId);
        } catch (error) {
            failed.push({ candidateId, error });
        }
    }
    return { attempted, succeeded, failed };
}

export function syncReviewSelectionDom(listElement, selectedCandidateIds) {
    const selected = new Set(selectedCandidateIds || []);
    const rows = Array.from(
        listElement?.querySelectorAll?.('.review-candidate-row') || [],
    );
    for (const row of rows) {
        const checked = selected.has(row.dataset?.candidateId);
        row.classList?.toggle?.('batch-selected', checked);
        const checkbox = row.querySelector?.('.review-select-checkbox');
        if (checkbox && !checkbox.disabled) checkbox.checked = checked;
    }
    return rows.length;
}

export function toggleReviewCandidateSelection(
    selectedCandidateIds,
    candidateId,
    checked,
) {
    const next = new Set(selectedCandidateIds || []);
    if (checked) next.add(candidateId);
    else next.delete(candidateId);
    return next;
}

export function reviewCandidateCurrentValues(entry) {
    if (reviewCandidateType(entry) === 'gdt') {
        return {
            tolerance_value: Object.hasOwn(entry || {}, 'tolerance_value')
                ? entry.tolerance_value
                : (entry?.item?.tolerance_value ?? ''),
        };
    }
    return {
        nominal: entry?.nominal ?? entry?.item?.nominal ?? '',
        symmetric_tolerance: Object.hasOwn(entry || {}, 'symmetric_tolerance')
            ? entry.symmetric_tolerance
            : (entry?.item?.symmetric_tolerance ?? null),
        upper_tolerance: Object.hasOwn(entry || {}, 'upper_tolerance')
            ? entry.upper_tolerance
            : (entry?.item?.upper_tolerance ?? null),
        lower_tolerance: Object.hasOwn(entry || {}, 'lower_tolerance')
            ? entry.lower_tolerance
            : (entry?.item?.lower_tolerance ?? null),
    };
}

/**
 * Build the exact type-specific edit/confirm payload.  In particular, the GD&T
 * branch exposes only tolerance_value and cannot manufacture datum writes.
 */
export function buildReviewCandidateValues(entry, rawValues = {}) {
    if (reviewCandidateType(entry) === 'gdt') {
        return {
            tolerance_value: String(rawValues.tolerance_value ?? '').trim(),
        };
    }
    const symmetric = String(rawValues.symmetric_tolerance ?? '').trim();
    const upper = String(rawValues.upper_tolerance ?? '').trim();
    const lower = String(rawValues.lower_tolerance ?? '').trim();
    return {
        nominal: String(rawValues.nominal ?? '').trim(),
        symmetric_tolerance: symmetric || null,
        upper_tolerance: upper || null,
        lower_tolerance: lower || null,
    };
}
