import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const noop = () => {};
const canvasContext = new Proxy({}, {
    get: () => noop,
    set: () => true,
});

function fakeElement(id = '') {
    const classes = new Set();
    return new Proxy({
        id,
        style: {},
        dataset: {},
        clientWidth: 200,
        clientHeight: 100,
        width: 200,
        height: 100,
        classList: {
            add: value => classes.add(value),
            remove: value => classes.delete(value),
            toggle: (value, force) => {
                if (force === true) classes.add(value);
                else if (force === false) classes.delete(value);
                else if (classes.has(value)) classes.delete(value);
                else classes.add(value);
            },
            contains: value => classes.has(value),
        },
        getContext: () => canvasContext,
        getBoundingClientRect: () => ({
            width: 200,
            height: 100,
            left: 0,
            top: 0,
        }),
        addEventListener: noop,
        removeEventListener: noop,
        appendChild: noop,
        querySelector: () => null,
        querySelectorAll: () => [],
        remove: noop,
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
    getElementById(id) {
        if (!elements.has(id)) elements.set(id, fakeElement(id));
        return elements.get(id);
    },
    createElement: tag => fakeElement(tag),
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: noop,
    removeEventListener: noop,
    dispatchEvent: noop,
};
globalThis.window = globalThis;
globalThis.addEventListener = noop;
globalThis.removeEventListener = noop;
globalThis.devicePixelRatio = 1;
globalThis.localStorage = {
    getItem: () => null,
    setItem: noop,
    removeItem: noop,
};
globalThis.CustomEvent = class CustomEvent {
    constructor(type) {
        this.type = type;
    }
};

let mediaChangeListener = null;
const mediaQueryList = {
    matches: true,
    addEventListener(type, listener) {
        if (type === 'change') mediaChangeListener = listener;
    },
};
globalThis.matchMedia = query => {
    assert.equal(query, '(prefers-reduced-motion: reduce)');
    return mediaQueryList;
};

let rafCalls = 0;
let cancelCalls = 0;
let pendingFrame = null;
globalThis.requestAnimationFrame = callback => {
    rafCalls += 1;
    pendingFrame = callback;
    return rafCalls;
};
globalThis.cancelAnimationFrame = () => {
    cancelCalls += 1;
    pendingFrame = null;
};

const { state } = await import('../state.js');
const {
    cancelPanTo,
    panTo,
} = await import('../app_utils.js');
const { flashStamp } = await import('../canvas_renderer.js');
const { prefersReducedMotion } = await import('../motion_preferences.js');

function setReducedMotion(matches) {
    mediaQueryList.matches = matches;
    mediaChangeListener?.({ matches });
}

function resetViewport() {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    state.pdfWidth = 2000;
    state.pdfHeight = 2000;
    state.pageOffsets = [];
    state.stamps = [];
    state.inspection = { active: false, byStampUuid: {} };
    rafCalls = 0;
    cancelCalls = 0;
    pendingFrame = null;
}

test('matchMedia change events update the shared reduced-motion preference', () => {
    assert.equal(prefersReducedMotion(), true);
    assert.equal(typeof mediaChangeListener, 'function');

    setReducedMotion(false);
    assert.equal(prefersReducedMotion(), false);

    setReducedMotion(true);
    assert.equal(prefersReducedMotion(), true);
});

test('panTo lands immediately without scheduling animation when motion is reduced', () => {
    resetViewport();
    setReducedMotion(true);

    panTo({ arrowTip: { absX: 50, absY: 25 } });

    assert.equal(state.panX, 50);
    assert.equal(state.panY, 25);
    assert.equal(rafCalls, 0);
});

test('panTo keeps its normal animation and finishes immediately if preference changes mid-flight', () => {
    resetViewport();
    setReducedMotion(false);

    panTo({ arrowTip: { absX: 50, absY: 25 } });

    assert.equal(rafCalls, 1);
    assert.equal(typeof pendingFrame, 'function');
    assert.equal(state.panX, 0);
    assert.equal(state.panY, 0);

    setReducedMotion(true);
    pendingFrame(performance.now());

    assert.equal(state.panX, 50);
    assert.equal(state.panY, 25);
    cancelPanTo();
    assert.equal(cancelCalls, 0);
});

test('canvas location feedback skips its long flash when motion is reduced', () => {
    resetViewport();
    setReducedMotion(true);

    flashStamp('stamp-1');

    assert.equal(rafCalls, 0);

    setReducedMotion(false);
    flashStamp('stamp-1');
    assert.equal(rafCalls, 1);
    const normalFrame = pendingFrame;

    setReducedMotion(true);
    normalFrame(performance.now());
    assert.equal(rafCalls, 1);
});

test('CSS has a global reduced-motion fallback', async () => {
    const css = await readFile(new URL('../styles.css', import.meta.url), 'utf8');
    assert.match(css, /@media\s*\(prefers-reduced-motion:\s*reduce\)/);
    assert.match(css, /transition-duration:\s*0\.01ms\s*!important/);
    assert.match(css, /animation-iteration-count:\s*1\s*!important/);
});
