import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const noop = () => {};
const context = new Proxy({}, {
    get: () => noop,
    set: () => true,
});
function fakeElement() {
    return new Proxy({
        style: {},
        dataset: {},
        classList: {
            add: noop,
            remove: noop,
            toggle: noop,
            contains: () => false,
        },
        getContext: () => context,
        addEventListener: noop,
        querySelectorAll: () => [],
        querySelector: () => null,
        getBoundingClientRect: () => ({
            width: 100,
            height: 100,
            left: 0,
            top: 0,
        }),
    }, {
        get: (target, property) => property in target ? target[property] : noop,
        set: (target, property, value) => {
            target[property] = value;
            return true;
        },
    });
}

const elements = new Map();
globalThis.document = {
    getElementById: id => {
        if (!elements.has(id)) elements.set(id, fakeElement());
        return elements.get(id);
    },
    addEventListener: noop,
    createElement: () => fakeElement(),
};
globalThis.window = {
    addEventListener: noop,
    innerWidth: 1000,
    innerHeight: 800,
};
globalThis.localStorage = {
    getItem: () => null,
    setItem: noop,
};

const { state } = await import('../state.js');
const {
    commitStampTypeChange,
    createHistorySnapshot,
} = await import('../app_utils.js');
const {
    finishStampTypeChangeEditing,
    getIsEditingField,
    setIsEditingField,
} = await import('../app_sidebar/expand_form.js');

function resetHistory(stamp) {
    state.stamps = [stamp];
    state.deletedStamps = [];
    state.inspection = { active: false, byStampUuid: {} };
    state.historyStack = [createHistorySnapshot()];
    state.historyIndex = 0;
    state.clusteringBaseIndex = -1;
    state.docInfo = { filename: 'Untitled', version: 1 };
}

test('stamp type selection records one undoable snapshot after the value changes', () => {
    const stamp = { uuid: 'stamp-1', aiData: { type: 'linear' } };
    resetHistory(stamp);

    const changed = commitStampTypeChange(stamp, 'diameter');

    assert.equal(changed, true);
    assert.equal(stamp.aiData.type, 'diameter');
    assert.equal(state.historyIndex, 1);
    assert.equal(state.historyStack.length, 2);
    assert.equal(state.historyStack[1].stamps[0].aiData.type, 'diameter');
});

test('stamp type selection does not create a duplicate snapshot for the same value', () => {
    const stamp = { uuid: 'stamp-1', aiData: { type: 'diameter' } };
    resetHistory(stamp);

    const changed = commitStampTypeChange(stamp, 'diameter');

    assert.equal(changed, false);
    assert.equal(state.historyIndex, 0);
    assert.equal(state.historyStack.length, 1);
});

test('type selection ends editing before rebuilding with one after-state snapshot', () => {
    const stamp = { uuid: 'stamp-1', aiData: { type: '一般尺寸' } };
    resetHistory(stamp);
    setIsEditingField(true);
    const rebuildObservations = [];

    const changed = finishStampTypeChangeEditing(stamp, '形位公差', () => {
        rebuildObservations.push({
            isEditing: getIsEditingField(),
            type: stamp.aiData.type,
            historyIndex: state.historyIndex,
        });
    });

    assert.equal(changed, true);
    assert.equal(getIsEditingField(), false);
    assert.deepEqual(rebuildObservations, [{
        isEditing: false,
        type: '形位公差',
        historyIndex: 1,
    }]);
    assert.equal(state.historyStack.length, 2);
    assert.equal(state.historyStack[1].stamps[0].aiData.type, '形位公差');
});

test('same type ends editing and rebuilds without adding history', () => {
    const stamp = { uuid: 'stamp-1', aiData: { type: '形位公差' } };
    resetHistory(stamp);
    setIsEditingField(true);
    let rebuilds = 0;

    const changed = finishStampTypeChangeEditing(stamp, '形位公差', () => {
        rebuilds += 1;
        assert.equal(getIsEditingField(), false);
    });

    assert.equal(changed, false);
    assert.equal(getIsEditingField(), false);
    assert.equal(rebuilds, 1);
    assert.equal(state.historyIndex, 0);
    assert.equal(state.historyStack.length, 1);
});

test('sidebar and popup type selectors finish editing through the rebuild seam', async () => {
    const [listSource, popupSource] = await Promise.all([
        readFile(new URL('../app_sidebar/list.js', import.meta.url), 'utf8'),
        readFile(new URL('../app_sidebar/popup.js', import.meta.url), 'utf8'),
    ]);

    for (const [name, source] of [
        ['sidebar', listSource],
        ['popup', popupSource],
    ]) {
        assert.match(source, /finishStampTypeChangeEditing/, `${name} imports the rebuild seam`);
        assert.match(
            source,
            /finishStampTypeChangeEditing\(s,\s*e\.target\.value,/,
            `${name} commits type changes before rebuilding`,
        );
    }
});
