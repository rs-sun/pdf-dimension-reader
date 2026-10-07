// All measurement labels below are synthetic rendering fixtures.
import test from 'node:test';
import assert from 'node:assert/strict';

const noop = () => {};
const drawCalls = {
    clearRect: 0,
    fillText: [],
};
const drawContext = new Proxy({
    clearRect() {
        drawCalls.clearRect += 1;
    },
    fillText(text) {
        drawCalls.fillText.push(String(text));
    },
    measureText(text) {
        return { width: String(text).length * 8 };
    },
}, {
    get(target, property) {
        if (property in target) return target[property];
        return noop;
    },
    set(target, property, value) {
        target[property] = value;
        return true;
    },
});

const canvasResolutionWrites = [];
const drawCanvas = {
    style: {},
    getContext: () => drawContext,
};
let canvasWidth = 0;
let canvasHeight = 0;
Object.defineProperties(drawCanvas, {
    width: {
        get: () => canvasWidth,
        set(value) {
            canvasWidth = value;
            canvasResolutionWrites.push(['width', value]);
        },
    },
    height: {
        get: () => canvasHeight,
        set(value) {
            canvasHeight = value;
            canvasResolutionWrites.push(['height', value]);
        },
    },
});

function fakeElement(overrides = {}) {
    return {
        style: {},
        classList: {
            contains: () => false,
            add: noop,
            remove: noop,
        },
        innerText: '',
        value: '',
        getBoundingClientRect: () => ({
            width: 800,
            height: 600,
            left: 0,
            top: 0,
            right: 800,
            bottom: 600,
        }),
        appendChild: noop,
        getContext: () => drawContext,
        ...overrides,
    };
}

const elements = new Map([
    ['draw-layer', drawCanvas],
    ['canvas-wrapper', fakeElement()],
    ['pdf-pages-container', fakeElement()],
    ['workspace', fakeElement()],
    ['minimap', fakeElement()],
    ['minimap-canvas-container', fakeElement()],
    ['minimap-viewport', fakeElement()],
    ['zoom-val', fakeElement()],
    ['sidebar-container', fakeElement()],
    ['page-indicator', fakeElement()],
]);
let hrOverlayRemovals = 0;
let transformEvents = 0;

globalThis.document = {
    getElementById(id) {
        if (!elements.has(id)) elements.set(id, fakeElement());
        return elements.get(id);
    },
    querySelectorAll(selector) {
        if (selector !== '.hr-overlay') return [];
        return [{ remove: () => { hrOverlayRemovals += 1; } }];
    },
    dispatchEvent(event) {
        if (event.type === 'pdf-reader:transform-update') transformEvents += 1;
        return true;
    },
    createElement: () => fakeElement(),
};
globalThis.CustomEvent = class CustomEvent {
    constructor(type) {
        this.type = type;
    }
};
globalThis.localStorage = {
    getItem: () => null,
    setItem: noop,
};
globalThis.window = globalThis;
globalThis.devicePixelRatio = 2;

let nextRafId = 1;
const rafCallbacks = new Map();
globalThis.requestAnimationFrame = callback => {
    const id = nextRafId++;
    rafCallbacks.set(id, callback);
    return id;
};
globalThis.cancelAnimationFrame = id => {
    rafCallbacks.delete(id);
};

let nextTimerId = 100;
const timerCallbacks = new Map();
const timerSets = [];
const timerClears = [];
globalThis.setTimeout = (callback, delay) => {
    const id = nextTimerId++;
    timerCallbacks.set(id, callback);
    timerSets.push([id, delay]);
    return id;
};
globalThis.clearTimeout = id => {
    timerCallbacks.delete(id);
    timerClears.push(id);
};

const { state } = await import('../state.js');
const { updateTransform } = await import('../canvas_renderer.js');
const {
    _buildCollapseHeaderHtml,
    _buildSummaryText,
} = await import('../app_sidebar/shared_templates.js');

test('inspection NG pulse repaints only the draw overlay on its next animation frame', () => {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    state.pdfWidth = 500;
    state.pdfHeight = 400;
    state.pageOffsets = [];
    state.viewBoxes = [];
    state.basicDimensions = [];
    state.datumReferences = [];
    state.selectedUUIDs = [];
    state.reviewWorkspace = { envelopes: [], decisions: {}, selectedCandidateId: null };
    state.stampActive = false;
    state.selectionBox = null;
    state.inspection = {
        active: true,
        byStampUuid: {
            'ng-stamp': { result: 'NG' },
        },
    };
    state.stamps = [{
        uuid: 'ng-stamp',
        id: '7',
        type: 'note',
        source: 'legacy_stamp',
        legacyPositionOnly: true,
        circleCenter: { absX: 25, absY: 30 },
        fs: 12,
        ratio: 0.6,
    }];

    updateTransform();
    assert.equal(rafCallbacks.size, 1, 'initial overlay draw schedules one NG pulse frame');
    const highResTimer = state.renderTimer;
    const [pulseRafId, pulseFrame] = [...rafCallbacks.entries()][0];
    rafCallbacks.delete(pulseRafId);

    canvasResolutionWrites.length = 0;
    drawCalls.clearRect = 0;
    hrOverlayRemovals = 0;
    transformEvents = 0;
    timerSets.length = 0;
    timerClears.length = 0;

    pulseFrame(16);

    assert.equal(drawCalls.clearRect, 1, 'the dynamic overlay is repainted');
    assert.equal(rafCallbacks.size, 1, 'the active NG pulse keeps scheduling overlay frames');
    assert.deepEqual({
        canvasResolutionWrites,
        hrOverlayRemovals,
        transformEvents,
        timerClears,
        timerSets,
        highResTimerPreserved: state.renderTimer === highResTimer,
    }, {
        canvasResolutionWrites: [],
        hrOverlayRemovals: 0,
        transformEvents: 0,
        timerClears: [],
        timerSets: [],
        highResTimerPreserved: true,
    }, 'overlay-only repaint has no transform or high-resolution scheduling side effects');

    state.inspection.active = false;
    const [nextPulseRafId, nextPulseFrame] = [...rafCallbacks.entries()][0];
    rafCallbacks.delete(nextPulseRafId);
    nextPulseFrame(32);

    assert.equal(drawCalls.clearRect, 2, 'the last queued frame observes the new overlay state');
    assert.equal(rafCallbacks.size, 0, 'the scheduler stops once the NG pulse is inactive');
    assert.deepEqual(canvasResolutionWrites, []);
    assert.equal(state.renderTimer, highResTimer);
});

test('review candidate canvas labels branch by candidate type', () => {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    state.pdfWidth = 500;
    state.pdfHeight = 400;
    state.totalPages = 1;
    state.pageOffsets = [{ start: 0, width: 500, height: 400 }];
    state.viewBoxes = [];
    state.basicDimensions = [];
    state.datumReferences = [];
    state.selectedUUIDs = [];
    state.stamps = [];
    state.stampActive = false;
    state.selectionBox = null;
    state.inspection = { active: false, byStampUuid: {} };
    state.reviewWorkspace = {
        envelopes: [{
            page_index: 0,
            items: [{
                candidate_id: 'gdt-candidate',
                bbox: { x: 10, y: 20, w: 80, h: 18 },
                type: 'gdt',
                symbol_name: 'position',
                symbol_unicode: '⊕',
                tolerance_value: '0.083',
                datum_1: null,
                datum_2: null,
                datum_3: null,
            }, {
                candidate_id: 'linear-candidate',
                bbox: { x: 10, y: 60, w: 80, h: 18 },
                type: 'linear',
                nominal: '12.5',
                symmetric_tolerance: '0.083',
                upper_tolerance: null,
                lower_tolerance: null,
                prefix: null,
                unit: null,
            }, {
                candidate_id: 'diameter-bilateral',
                bbox: { x: 10, y: 100, w: 80, h: 18 },
                type: 'linear',
                nominal: '12',
                symmetric_tolerance: null,
                upper_tolerance: '+0.2',
                lower_tolerance: '-0.8',
                prefix: 'Ø',
                unit: null,
            }, {
                candidate_id: 'radius',
                bbox: { x: 10, y: 140, w: 80, h: 18 },
                type: 'linear',
                nominal: '5',
                symmetric_tolerance: null,
                upper_tolerance: null,
                lower_tolerance: null,
                prefix: 'R',
                unit: null,
            }, {
                candidate_id: 'degree-unilateral',
                bbox: { x: 10, y: 180, w: 80, h: 18 },
                type: 'linear',
                nominal: '45',
                symmetric_tolerance: null,
                upper_tolerance: '0',
                lower_tolerance: '-1',
                prefix: null,
                unit: 'degree',
            }, {
                candidate_id: 'quantity-symmetric',
                bbox: { x: 10, y: 220, w: 80, h: 18 },
                type: 'linear',
                nominal: '217.463',
                symmetric_tolerance: '0.083',
                upper_tolerance: null,
                lower_tolerance: null,
                prefix: null,
                unit: null,
                quantity: 3,
            }],
        }],
        decisions: {},
        selectedCandidateId: null,
    };

    drawCalls.fillText.length = 0;
    updateTransform();

    assert.deepEqual(drawCalls.fillText, [
        '⊕ 0.083',
        '12.5 ±0.083',
        'Ø12 +0.2/-0.8',
        'R5',
        '45° 0°/-1°',
        '3X 217.463 ±0.083',
    ]);
    assert.equal(drawCalls.fillText.some(text => text.includes('undefined')), false);
    assert.equal(drawCalls.fillText.some(text => /(?:^|\s)[ABC](?:\s|$)/.test(text)), false);
});

test('confirmed stamp summaries render widened linear semantics', () => {
    const summary = aiData => _buildSummaryText({ aiData });

    assert.equal(summary({
        type: '线性',
        prefix: 'Ø',
        nominal: '12',
        upper_tol: '+0.2',
        lower_tol: '-0.8',
        unit: '',
    }), 'Ø12 +0.2/-0.8');
    assert.equal(summary({
        type: '线性',
        prefix: 'R',
        nominal: '5',
        upper_tol: '',
        lower_tol: '',
        unit: '',
    }), 'R5');
    assert.equal(summary({
        type: '角度',
        prefix: '',
        nominal: '45',
        upper_tol: '+1',
        lower_tol: '-1',
        unit: 'degree',
    }), '45° ±1°');
    assert.equal(summary({
        type: '角度',
        prefix: '',
        nominal: '45°',
        upper_tol: '0°',
        lower_tol: '-1°',
        unit: 'degree',
    }), '45° 0°/-1°');
    assert.equal(summary({
        type: '线性',
        prefix: '',
        nominal: '217.463',
        upper_tol: '+0.083',
        lower_tol: '-0.083',
        unit: '',
        quantity: 3,
    }), '3X 217.463 ±0.083');

    const inline = _buildCollapseHeaderHtml({
        uuid: 'quantity-stamp',
        id: '12',
        type: 'normal',
        status: 'done',
        confirmed: true,
        aiData: {
            type: '线性',
            nominal: '217.463',
            upper_tol: '+0.083',
            lower_tol: '-0.083',
            quantity: 3,
        },
    });
    assert.match(inline, /class="ai-quantity"[^>]*>3X<\/span>/);
});
