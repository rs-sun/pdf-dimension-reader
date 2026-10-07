import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
    REVIEW_GDT_DATUM_NOTICE,
    buildReviewCandidateValues,
    reviewCandidateCurrentValues,
    reviewCandidateType,
    reviewGdtSymbolChineseName,
    runSelectedReviewCandidateActions,
    selectedReviewCandidateIds,
    syncReviewSelectionDom,
    toggleReviewCandidateSelection,
} from '../app_sidebar/review_workbench.js';

function entry(candidateId, status = 'pending', item = {}) {
    return {
        status,
        item: {
            candidate_id: candidateId,
            type: 'linear',
            nominal: '12.5',
            symmetric_tolerance: null,
            upper_tolerance: null,
            lower_tolerance: null,
            prefix: null,
            unit: null,
            ...item,
        },
    };
}

test('GD&T review values expose tolerance only and never expose datum writes', () => {
    const gdt = entry('gdt-1', 'edited', {
        type: 'gdt',
        symbol_name: 'position',
        symbol_unicode: '⊕',
        tolerance_value: '0.05',
        datum_1: null,
        datum_2: null,
        datum_3: null,
    });
    gdt.tolerance_value = '0.075';

    assert.equal(reviewCandidateType(gdt), 'gdt');
    assert.deepEqual(reviewCandidateCurrentValues(gdt), {
        tolerance_value: '0.075',
    });
    assert.deepEqual(buildReviewCandidateValues(gdt, {
        tolerance_value: ' 0.100 ',
        datum_1: 'A',
        datum_2: 'B',
        datum_3: 'C',
    }), {
        tolerance_value: '0.100',
    });
    assert.equal(REVIEW_GDT_DATUM_NOTICE, '基准：首批不识别（留空）');
});

test('linear review values preserve an edited null tolerance', () => {
    const linear = entry('linear-1', 'edited', {
        symmetric_tolerance: '0.05',
    });
    linear.nominal = '12.6';
    linear.symmetric_tolerance = null;

    assert.deepEqual(reviewCandidateCurrentValues(linear), {
        nominal: '12.6',
        symmetric_tolerance: null,
        upper_tolerance: null,
        lower_tolerance: null,
    });
    assert.deepEqual(buildReviewCandidateValues(linear, {
        nominal: ' 12.7 ',
        symmetric_tolerance: ' ',
        upper_tolerance: ' ',
        lower_tolerance: ' ',
    }), {
        nominal: '12.7',
        symmetric_tolerance: null,
        upper_tolerance: null,
        lower_tolerance: null,
    });
});

test('asymmetric review values preserve signed and explicit-zero sides', () => {
    const linear = entry('linear-asymmetric', 'edited', {
        nominal: '12',
        symmetric_tolerance: null,
        upper_tolerance: '0',
        lower_tolerance: '-1',
        prefix: 'R',
    });
    linear.upper_tolerance = '+0.2';
    linear.lower_tolerance = '-0.8';

    assert.deepEqual(reviewCandidateCurrentValues(linear), {
        nominal: '12',
        symmetric_tolerance: null,
        upper_tolerance: '+0.2',
        lower_tolerance: '-0.8',
    });
    assert.deepEqual(buildReviewCandidateValues(linear, {
        nominal: ' 12 ',
        symmetric_tolerance: '',
        upper_tolerance: ' 0 ',
        lower_tolerance: ' -1 ',
        prefix: 'Ø',
        unit: 'degree',
    }), {
        nominal: '12',
        symmetric_tolerance: null,
        upper_tolerance: '0',
        lower_tolerance: '-1',
    });
});

test('workbench helpers fail closed on an unknown candidate discriminator', () => {
    assert.throws(
        () => reviewCandidateType({ item: { type: 'legacy' } }),
        /unsupported review candidate type/,
    );
});

test('GD&T symbol names have reader-facing Chinese labels', () => {
    const expected = {
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
    };
    for (const [name, label] of Object.entries(expected)) {
        assert.equal(reviewGdtSymbolChineseName(name), label);
    }
    assert.equal(reviewGdtSymbolChineseName('not-a-symbol'), '未知形位符号');
});

test('batch actions resolve only the explicitly checked actionable candidates', () => {
    const entries = [
        entry('pending-1'),
        entry('confirmed-1', 'confirmed'),
        entry('edited-1', 'edited'),
        entry('dismissed-1', 'dismissed'),
        entry('pending-2'),
    ];
    const checked = new Set([
        'pending-2',
        'confirmed-1',
        'edited-1',
        'stale-from-another-document',
    ]);

    assert.deepEqual(
        selectedReviewCandidateIds(entries, checked),
        ['edited-1', 'pending-2'],
    );
    assert.deepEqual(
        selectedReviewCandidateIds(entries, new Set()),
        [],
        'an empty explicit selection must not fall back to every candidate',
    );
});

test('batch executor calls only checked candidates and reports per-row failures', async () => {
    const entries = [
        entry('pending-1'),
        entry('unchecked-1'),
        entry('edited-1', 'edited'),
        entry('resolved-1', 'confirmed'),
    ];
    const calls = [];
    const result = await runSelectedReviewCandidateActions(
        entries,
        new Set(['pending-1', 'edited-1', 'resolved-1']),
        async candidateId => {
            calls.push(candidateId);
            if (candidateId === 'edited-1') throw new Error('invalid edit');
        },
    );

    assert.deepEqual(calls, ['pending-1', 'edited-1']);
    assert.deepEqual(result.attempted, ['pending-1', 'edited-1']);
    assert.deepEqual(result.succeeded, ['pending-1']);
    assert.equal(result.failed.length, 1);
    assert.equal(result.failed[0].candidateId, 'edited-1');
    assert.match(result.failed[0].error.message, /invalid edit/);
});

test('checkbox selection toggles one candidate without implicitly selecting peers', () => {
    const first = toggleReviewCandidateSelection(new Set(), 'candidate-2', true);
    assert.deepEqual([...first], ['candidate-2']);

    const second = toggleReviewCandidateSelection(first, 'candidate-3', true);
    assert.deepEqual([...second], ['candidate-2', 'candidate-3']);

    const third = toggleReviewCandidateSelection(second, 'candidate-2', false);
    assert.deepEqual([...third], ['candidate-3']);
});

test('checkbox DOM sync is in-place and preserves focus plus unsaved value drafts', () => {
    const makeRow = (candidateId, draftValue, focused = false) => {
        const classes = new Set();
        const checkbox = { checked: false, disabled: false, focused };
        const draft = { value: draftValue };
        return {
            dataset: { candidateId },
            classes,
            checkbox,
            draft,
            classList: {
                toggle(name, enabled) {
                    if (enabled) classes.add(name);
                    else classes.delete(name);
                },
            },
            querySelector(selector) {
                if (selector === '.review-select-checkbox') return checkbox;
                if (selector === '.review-primary-input') return draft;
                return null;
            },
        };
    };
    const first = makeRow('candidate-1', '0.075', true);
    const second = makeRow('candidate-2', '0.125');
    const listElement = {
        querySelectorAll: () => [first, second],
    };

    assert.equal(
        syncReviewSelectionDom(listElement, new Set(['candidate-1'])),
        2,
    );
    assert.equal(first.checkbox.checked, true);
    assert.equal(second.checkbox.checked, false);
    assert.equal(first.classes.has('batch-selected'), true);
    assert.equal(second.classes.has('batch-selected'), false);
    assert.equal(first.checkbox.focused, true);
    assert.equal(first.draft.value, '0.075');
    assert.equal(second.draft.value, '0.125');
});

test('review markup provides selected-only controls and read-only semantic fields', async () => {
    const [indexSource, listSource, canvasSource] = await Promise.all([
        readFile(new URL('../index.html', import.meta.url), 'utf8'),
        readFile(new URL('../app_sidebar/list.js', import.meta.url), 'utf8'),
        readFile(new URL('../canvas_renderer.js', import.meta.url), 'utf8'),
    ]);

    assert.match(indexSource, /id="btn-review-confirm-selected"[^>]*>确认选中</);
    assert.match(indexSource, /id="btn-review-dismiss-selected"[^>]*>拒绝选中</);
    assert.doesNotMatch(indexSource, /id="btn-review-(?:select|confirm)-all"/);
    assert.match(listSource, /class="review-gdt-symbol"/);
    assert.match(listSource, /class="review-gdt-tolerance review-primary-input"/);
    assert.match(listSource, /class="review-datum-note" role="note"/);
    assert.doesNotMatch(listSource, /class="review-datum-[123]"/);
    assert.match(listSource, /class="review-upper-tolerance"/);
    assert.match(listSource, /class="review-lower-tolerance"/);
    assert.match(listSource, /class="review-semantic-chip review-prefix"/);
    assert.match(listSource, /class="review-semantic-chip review-unit"/);
    assert.match(listSource, /class="review-semantic-chip review-quantity"/);
    assert.match(listSource, /倍数 \$\{entry\.item\.quantity\}X/);
    assert.match(canvasSource, /`\$\{entry\.quantity\}X `/);
    assert.doesNotMatch(listSource, /class="review-(?:prefix|unit)-input"/);
    assert.doesNotMatch(listSource, /class="review-quantity-input"/);
});
