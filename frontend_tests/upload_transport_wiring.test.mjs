import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('application startup primes readiness before the first user analysis', async () => {
    const source = await readFile(new URL('../app.js', import.meta.url), 'utf8');

    assert.match(source, /refreshRecognitionCapabilities/);
    assert.match(
        source,
        /window\.addEventListener\('load',[\s\S]*refreshRecognitionCapabilities\(\)/,
    );
});

test('multi-page completion copy distinguishes unattempted pages', async () => {
    const source = await readFile(new URL('../app_analysis.js', import.meta.url), 'utf8');

    assert.match(source, /outcome\.skippedPages/);
    assert.match(source, /outcome\.status === 'failed'/);
    assert.match(source, /页未尝试/);
});
