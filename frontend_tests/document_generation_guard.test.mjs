import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const storage = new Map();
globalThis.localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key),
};

const stateModule = await import('../state.js');

test('document generation starts at zero and the import guard uses the live analysis flag', () => {
    assert.equal(stateModule.state.documentGeneration, 0);
    assert.equal(stateModule.state.isDocumentImportRunning, false);
    assert.equal(
        typeof stateModule.rejectDocumentImportWhileAnalysisRunning,
        'function',
    );

    const notices = [];
    assert.equal(
        stateModule.rejectDocumentImportWhileAnalysisRunning(
            { isAnalysisRunning: true },
            message => notices.push(message),
        ),
        true,
    );
    assert.deepEqual(notices, ['识别任务运行中，请等待完成后再加载新文件。']);
    assert.equal(
        stateModule.rejectDocumentImportWhileAnalysisRunning(
            { isAnalysisRunning: false },
            message => notices.push(message),
        ),
        false,
    );
    assert.equal(notices.length, 1);
});

test('PDF, JSON, ZIP, and window drop entries share the real analysis guard and no dead flag remains', async () => {
    const [stateSource, ioSource, appSource, analysisSource] = await Promise.all([
        readFile(new URL('../state.js', import.meta.url), 'utf8'),
        readFile(new URL('../app_io.js', import.meta.url), 'utf8'),
        readFile(new URL('../app.js', import.meta.url), 'utf8'),
        readFile(new URL('../app_analysis.js', import.meta.url), 'utf8'),
    ]);
    const allSources = [stateSource, ioSource, appSource, analysisSource].join('\n');

    assert.doesNotMatch(allSources, /\bisL3Running\b/);
    assert.match(
        ioSource,
        /function _loadPdfBufferAsCurrentDocument[\s\S]*?rejectDocumentImportWhileAnalysisRunning/,
    );
    assert.match(
        ioSource,
        /window\.addEventListener\('drop'[\s\S]*?rejectDocumentImportWhileAnalysisRunning/,
    );
    assert.match(
        ioSource,
        /getElementById\('file-upload'\)\.addEventListener\('change'[\s\S]*?rejectDocumentImportWhileAnalysisRunning/,
    );

    const generationCommits = ioSource.match(/advanceDocumentGeneration\(state\)/g) || [];
    assert.equal(generationCommits.length, 4);
    const importTransactions = ioSource.match(
        /_executeDocumentImportTransaction\(\{/g,
    ) || [];
    assert.equal(importTransactions.length, 4);
    assert.match(
        ioSource,
        /state\.isDocumentImportRunning = true;[\s\S]*?finally \{[\s\S]*?state\.isDocumentImportRunning = false;/,
    );
    const completedCommitBoundaries = ioSource.match(
        /_finishDocumentCommit\(\);\s*advanceDocumentGeneration\(state\);/g,
    ) || [];
    assert.equal(completedCommitBoundaries.length, 4);
    assert.match(
        analysisSource,
        /state\.isDocumentImportRunning\) return alert\(DOCUMENT_LOADING_MESSAGE\)/,
    );
    assert.match(
        analysisSource,
        /解析结果属于已切换的文档，已放弃/,
    );
    assert.match(
        analysisSource,
        /_reportStaleAnalysisOutcome\(outcome\)\) return outcome;\s*applyVectorDebugOutcome/,
    );
});
