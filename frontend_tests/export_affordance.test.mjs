import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { getExportAffordance } from '../app_sidebar/export_affordance.js';

test('export affordance distinguishes empty, warning, and ready states', () => {
    assert.deepEqual(getExportAffordance([]), {
        state: 'empty',
        unconfirmedCount: 0,
        title: '无数据可供导出',
    });

    assert.deepEqual(getExportAffordance([
        { confirmed: true },
        { confirmed: false },
        { confirmed: false },
    ]), {
        state: 'warning',
        unconfirmedCount: 2,
        title: '含 2 条未确认，导出需二次确认',
    });

    assert.deepEqual(getExportAffordance([
        { confirmed: true },
        { confirmed: true },
    ]), {
        state: 'ready',
        unconfirmedCount: 0,
        title: '导出工程与记录',
    });
});

test('forced export still requires the same two confirmations in order', async () => {
    const source = await readFile(new URL('../app_io.js', import.meta.url), 'utf8');
    const first = '当前仍有 ${unconfirmed} 条标记未确认，是否强制导出？';
    const second = '强制导出的归档将包含未确认数据，请再次确认。';

    const firstIndex = source.indexOf(first);
    const secondIndex = source.indexOf(second);
    assert.notEqual(firstIndex, -1);
    assert.notEqual(secondIndex, -1);
    assert.ok(firstIndex < secondIndex);
});

test('legacy blocked affordance class is absent from runtime sources', async () => {
    const [listSource, cssSource] = await Promise.all([
        readFile(new URL('../app_sidebar/list.js', import.meta.url), 'utf8'),
        readFile(new URL('../styles.css', import.meta.url), 'utf8'),
    ]);
    assert.doesNotMatch(listSource, /export-blocked/);
    assert.doesNotMatch(cssSource, /export-blocked/);
    assert.match(listSource, /export-warning/);
    assert.match(cssSource, /#btn-submit\.export-warning/);
});
