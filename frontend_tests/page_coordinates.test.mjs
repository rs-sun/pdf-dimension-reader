import test from 'node:test';
import assert from 'node:assert/strict';

import {
    PAGE_COORDINATE_TOLERANCE,
    createPageGeometry,
    documentGlobalBboxToPageLocal,
    documentGlobalPointToPageLocal,
    pageLocalBboxToDocumentGlobal,
    pageLocalPointToDocumentGlobal,
} from '../page_coordinates.js';

function close(actual, expected, tolerance = PAGE_COORDINATE_TOLERANCE) {
    assert.ok(
        Math.abs(actual - expected) <= tolerance,
        `expected ${actual} to be within ${tolerance} of ${expected}`,
    );
}

test('page-local points and bboxes use the page document offset', () => {
    const page = createPageGeometry({
        pageIndex: 2,
        start: 842,
        sourceWidth: 595,
        sourceHeight: 842,
        rotation: 0,
    });

    assert.deepEqual(
        pageLocalPointToDocumentGlobal({ x: 20, y: 30 }, page),
        { x: 20, y: 872 },
    );
    assert.deepEqual(
        pageLocalBboxToDocumentGlobal({ x: 20, y: 30, w: 40, h: 10 }, page),
        { x: 20, y: 872, w: 40, h: 10 },
    );
});

test('different page sizes and orthogonal rotations round-trip within the declared tolerance', () => {
    const sourceBbox = { x: 17.25, y: 31.5, w: 42.75, h: 13.125 };
    const sourcePoint = { x: 58.75, y: 91.125 };
    const cases = [
        { sourceWidth: 595, sourceHeight: 842, rotation: 0, start: 0 },
        { sourceWidth: 842, sourceHeight: 595, rotation: 90, start: 842 },
        { sourceWidth: 420, sourceHeight: 297, rotation: 180, start: 1684 },
        { sourceWidth: 297, sourceHeight: 420, rotation: 270, start: 1981 },
    ];

    for (const [index, spec] of cases.entries()) {
        const page = createPageGeometry({ pageIndex: index + 1, ...spec });
        const pointRoundTrip = documentGlobalPointToPageLocal(
            pageLocalPointToDocumentGlobal(sourcePoint, page),
            page,
        );
        const bboxRoundTrip = documentGlobalBboxToPageLocal(
            pageLocalBboxToDocumentGlobal(sourceBbox, page),
            page,
        );

        close(pointRoundTrip.x, sourcePoint.x);
        close(pointRoundTrip.y, sourcePoint.y);
        close(bboxRoundTrip.x, sourceBbox.x);
        close(bboxRoundTrip.y, sourceBbox.y);
        close(bboxRoundTrip.w, sourceBbox.w);
        close(bboxRoundTrip.h, sourceBbox.h);
    }
});

test('90-degree rotation maps a local bbox into the rendered page coordinate system', () => {
    const page = createPageGeometry({
        pageIndex: 3,
        start: 1200,
        sourceWidth: 200,
        sourceHeight: 100,
        rotation: 90,
    });

    assert.deepEqual(page, {
        pageIndex: 3,
        start: 1200,
        sourceWidth: 200,
        sourceHeight: 100,
        rotation: 90,
        width: 100,
        height: 200,
    });
    assert.deepEqual(
        pageLocalBboxToDocumentGlobal({ x: 10, y: 20, w: 30, h: 40 }, page),
        { x: 40, y: 1210, w: 40, h: 30 },
    );
});

test('page-two note, view, dimension, arrow, and crop geometry share one offset and rotation', () => {
    const page = createPageGeometry({
        pageIndex: 2,
        start: 500,
        sourceWidth: 200,
        sourceHeight: 100,
        rotation: 90,
    });
    const localBboxes = {
        note: { x: 10, y: 12, w: 70, h: 18 },
        rectangularView: { x: 5, y: 8, w: 180, h: 84 },
        dimension: { x: 90, y: 44, w: 32, h: 9 },
        crop: { x: 86, y: 40, w: 40, h: 17 },
    };
    const localPoints = {
        circularViewCenter: { x: 150, y: 55 },
        arrow: { x: 122, y: 48.5 },
    };

    for (const local of Object.values(localBboxes)) {
        const global = pageLocalBboxToDocumentGlobal(local, page);
        assert.ok(global.y >= page.start);
        assert.deepEqual(documentGlobalBboxToPageLocal(global, page), local);
    }
    for (const local of Object.values(localPoints)) {
        const global = pageLocalPointToDocumentGlobal(local, page);
        assert.ok(global.y >= page.start);
        assert.deepEqual(documentGlobalPointToPageLocal(global, page), local);
    }
});
