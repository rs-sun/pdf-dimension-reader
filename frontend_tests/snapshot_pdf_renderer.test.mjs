import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { createPageGeometry } from '../page_coordinates.js';
import { renderSnapshotPdf } from '../snapshot_pdf_renderer.js';

const require = createRequire(import.meta.url);
const PDFLib = require('../libs/pdf-lib.min.js');
const pdfjsLib = require('../libs/pdf.min.js');
pdfjsLib.GlobalWorkerOptions.workerSrc = fileURLToPath(
    new URL('../libs/pdf.worker.min.js', import.meta.url),
);
const NUMBER = '-?(?:\\d+(?:\\.\\d+)?|\\.\\d+)(?:e[+-]?\\d+)?';

function close(actual, expected, tolerance = 1e-6) {
    assert.ok(
        Math.abs(actual - expected) <= tolerance,
        `expected ${actual} to be within ${tolerance} of ${expected}`,
    );
}

function closePoint(actual, expected, tolerance = 1e-6) {
    close(actual.x, expected.x, tolerance);
    close(actual.y, expected.y, tolerance);
}

function decodePageContent(pdfDoc, pageIndex = 0) {
    const contents = pdfDoc.getPage(pageIndex).node.Contents();
    const streams = typeof contents?.size === 'function'
        ? Array.from({ length: contents.size() }, (_, index) => contents.get(index))
        : [contents];
    return streams.map(stream => {
        const rawStream = pdfDoc.context.lookup(stream);
        return Buffer.from(PDFLib.decodePDFRawStream(rawStream).decode()).toString('latin1');
    }).join('\n');
}

function lineSegments(content) {
    const pattern = new RegExp(
        `(${NUMBER}) (${NUMBER}) m\\s+(${NUMBER}) (${NUMBER}) m\\s+`
        + `(${NUMBER}) (${NUMBER}) l\\s+S`,
        'gi',
    );
    return [...content.matchAll(pattern)].map(match => ({
        start: { x: Number(match[3]), y: Number(match[4]) },
        end: { x: Number(match[5]), y: Number(match[6]) },
    }));
}

function circleGeometry(content) {
    const pattern = new RegExp(
        `(${NUMBER}) (${NUMBER}) (${NUMBER}) (${NUMBER}) (${NUMBER}) (${NUMBER}) c`,
        'gi',
    );
    const coordinates = [...content.matchAll(pattern)].flatMap(match => [
        { x: Number(match[1]), y: Number(match[2]) },
        { x: Number(match[3]), y: Number(match[4]) },
        { x: Number(match[5]), y: Number(match[6]) },
    ]);
    assert.equal(coordinates.length, 12, 'one exported stamp must contain one four-curve circle');
    const xs = coordinates.map(point => point.x);
    const ys = coordinates.map(point => point.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return {
        center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2 },
        radiusX: (maxX - minX) / 2,
        radiusY: (maxY - minY) / 2,
    };
}

function textMatrix(content) {
    const pattern = new RegExp(
        `(${NUMBER}) (${NUMBER}) (${NUMBER}) (${NUMBER}) (${NUMBER}) (${NUMBER}) Tm`,
        'gi',
    );
    const matrices = [...content.matchAll(pattern)].map(match => match.slice(1).map(Number));
    assert.equal(matrices.length, 1, 'one exported stamp must contain one text matrix');
    return matrices[0];
}

async function createSourcePdf({ rotation, cropBox, userUnit = 1 }) {
    const pdfDoc = await PDFLib.PDFDocument.create();
    const page = pdfDoc.addPage([200, 100]);
    page.setCropBox(cropBox.x, cropBox.y, cropBox.width, cropBox.height);
    page.setRotation(PDFLib.degrees(rotation));
    if (userUnit !== 1) {
        page.node.set(
            PDFLib.PDFName.of('UserUnit'),
            PDFLib.PDFNumber.of(userUnit),
        );
    }
    return pdfDoc.save({ useObjectStreams: false });
}

async function helveticaBoldWidth(text, size) {
    const pdfDoc = await PDFLib.PDFDocument.create();
    const font = await pdfDoc.embedFont(PDFLib.StandardFonts.HelveticaBold);
    return font.widthOfTextAtSize(text, size);
}

async function geometryFromPdfJs(sourcePdf, { pageIndex = 1, start = 0 } = {}) {
    const pdfDoc = await pdfjsLib.getDocument({
        data: sourcePdf.slice(0),
        verbosity: 0,
    }).promise;
    try {
        const page = await pdfDoc.getPage(pageIndex);
        const viewport = page.getViewport({ scale: 1, rotation: 0 });
        return {
            geometry: createPageGeometry({
                pageIndex,
                start,
                sourceWidth: viewport.width,
                sourceHeight: viewport.height,
                rotation: page.rotate,
            }),
            userUnit: page.userUnit,
        };
    } finally {
        await pdfDoc.destroy();
    }
}

function expectedArrowWingEnds(previous, tip, fontSize) {
    const angle = Math.atan2(tip.y - previous.y, tip.x - previous.x);
    const arrowLength = Math.max(8, Math.min(fontSize * 0.6 + 4, 30));
    return [-Math.PI / 6, Math.PI / 6].map(offset => ({
        x: tip.x - arrowLength * Math.cos(angle + offset),
        y: tip.y - arrowLength * Math.sin(angle + offset),
    })).sort((left, right) => left.x - right.x || left.y - right.y);
}

test('90-degree page export preserves the visible stamp geometry in PDF user space', async () => {
    const sourcePdf = await createSourcePdf({
        rotation: 90,
        cropBox: { x: 0, y: 0, width: 200, height: 100 },
    });
    const snapshot = {
        pages: [1],
        page_geometries: [{
            pageIndex: 1,
            start: 0,
            sourceWidth: 200,
            sourceHeight: 100,
            rotation: 90,
            width: 100,
            height: 200,
        }],
        stamps: [{
            uuid: 'rotated-stamp',
            id: '7',
            pageIndex: 1,
            fs: 10,
            ratio: 0.6,
            routingPath: [
                { absX: 20, absY: 30 },
                { absX: 60, absY: 30 },
                { absX: 60, absY: 150 },
            ],
        }],
    };

    const exported = await renderSnapshotPdf(snapshot, {
        pdfLib: PDFLib,
        sourcePdf,
        resolveStampColor: () => '#112233',
        cleanText: value => value,
    });
    const pdfDoc = await PDFLib.PDFDocument.load(exported);
    const content = decodePageContent(pdfDoc);
    const segments = lineSegments(content);

    assert.equal(segments.length, 4, 'two route segments and two arrow wings are exported');
    closePoint(segments[0].start, { x: 30, y: 20 });
    closePoint(segments[0].end, { x: 30, y: 60 });
    closePoint(segments[1].start, { x: 30, y: 60 });
    closePoint(segments[1].end, { x: 150, y: 60 });
    closePoint(segments[2].start, { x: 150, y: 60 });
    closePoint(segments[3].start, { x: 150, y: 60 });
    const wingEnds = segments.slice(2).map(segment => segment.end).sort((left, right) => left.y - right.y);
    closePoint(wingEnds[0], { x: 141.3397459621556, y: 55 });
    closePoint(wingEnds[1], { x: 141.3397459621556, y: 65 });

    closePoint(circleGeometry(content).center, { x: 30, y: 20 });

    const textWidth = await helveticaBoldWidth('7', 10);
    const matrix = textMatrix(content);
    close(matrix[0], 0);
    close(matrix[1], 1);
    close(matrix[2], -1);
    close(matrix[3], 0);
    close(matrix[4], 33.6);
    close(matrix[5], 20 - textWidth / 2);
});

test('0-degree page export keeps the established geometry unchanged', async () => {
    const sourcePdf = await createSourcePdf({
        rotation: 0,
        cropBox: { x: 0, y: 0, width: 200, height: 100 },
    });
    const snapshot = {
        pages: [1],
        page_geometries: [{
            pageIndex: 1,
            start: 0,
            sourceWidth: 200,
            sourceHeight: 100,
            rotation: 0,
            width: 200,
            height: 100,
        }],
        stamps: [{
            uuid: 'unrotated-stamp',
            id: '7',
            pageIndex: 1,
            fs: 14,
            ratio: 0.6,
            routingPath: [
                { absX: 30, absY: 20 },
                { absX: 60, absY: 40 },
                { absX: 80, absY: 70 },
            ],
        }],
    };

    const exported = await renderSnapshotPdf(snapshot, {
        pdfLib: PDFLib,
        sourcePdf,
        resolveStampColor: () => '#112233',
        cleanText: value => value,
    });
    const pdfDoc = await PDFLib.PDFDocument.load(exported);
    const content = decodePageContent(pdfDoc);
    const segments = lineSegments(content);

    assert.equal(segments.length, 4);
    closePoint(segments[0].start, { x: 30, y: 80 });
    closePoint(segments[0].end, { x: 60, y: 60 });
    closePoint(segments[1].start, { x: 60, y: 60 });
    closePoint(segments[1].end, { x: 80, y: 30 });
    closePoint(segments[2].start, { x: 80, y: 30 });
    closePoint(segments[3].start, { x: 80, y: 30 });
    const wingEnds = segments.slice(2).map(segment => segment.end).sort((left, right) => left.x - right.x);
    closePoint(wingEnds[0], { x: 68.88452085355613, y: 35.49600976572744 });
    closePoint(wingEnds[1], { x: 79.20194450334539, y: 42.37429219892029 });

    const circle = circleGeometry(content);
    closePoint(circle.center, { x: 30, y: 80 });
    close(circle.radiusX, 8.4);
    close(circle.radiusY, 8.4);

    const matrix = textMatrix(content);
    assert.deepEqual(matrix.slice(0, 4), [1, 0, 0, 1]);
    close(matrix[4], 26.108);
    close(matrix[5], 74.96);
});

test('mixed 0/90/180/270-degree pages preserve every stamp primitive with an offset CropBox', async () => {
    const rotations = [0, 90, 180, 270];
    const starts = [0, 100, 300, 400];
    const sourceDoc = await PDFLib.PDFDocument.create();
    for (const rotation of rotations) {
        const page = sourceDoc.addPage([300, 200]);
        page.setCropBox(20, 30, 200, 100);
        page.setRotation(PDFLib.degrees(rotation));
    }
    const sourcePdf = await sourceDoc.save({ useObjectStreams: false });
    const snapshot = {
        pages: [1, 2, 3, 4],
        page_geometries: rotations.map((rotation, index) => ({
            pageIndex: index + 1,
            start: starts[index],
            sourceWidth: 200,
            sourceHeight: 100,
            rotation,
            width: rotation === 90 || rotation === 270 ? 100 : 200,
            height: rotation === 90 || rotation === 270 ? 200 : 100,
        })),
        stamps: rotations.map((_rotation, index) => ({
            uuid: `crop-rotation-${index}`,
            id: '7',
            pageIndex: index + 1,
            fs: 10,
            ratio: 0.6,
            routingPath: [
                { absX: 30, absY: starts[index] + 20 },
                { absX: 60, absY: starts[index] + 40 },
                { absX: 80, absY: starts[index] + 70 },
            ],
        })),
    };
    const expectedRoutes = [
        [{ x: 50, y: 110 }, { x: 80, y: 90 }, { x: 100, y: 60 }],
        [{ x: 40, y: 60 }, { x: 60, y: 90 }, { x: 90, y: 110 }],
        [{ x: 190, y: 50 }, { x: 160, y: 70 }, { x: 140, y: 100 }],
        [{ x: 200, y: 100 }, { x: 180, y: 70 }, { x: 150, y: 50 }],
    ];

    const exported = await renderSnapshotPdf(snapshot, {
        pdfLib: PDFLib,
        sourcePdf,
        resolveStampColor: () => '#112233',
        cleanText: value => value,
    });
    const pdfDoc = await PDFLib.PDFDocument.load(exported);
    const textWidth = await helveticaBoldWidth('7', 10);
    const expectedTextMatrices = [
        [1, 0, 0, 1, 50 - textWidth / 2, 106.4],
        [0, 1, -1, 0, 43.6, 60 - textWidth / 2],
        [-1, 0, 0, -1, 190 + textWidth / 2, 53.6],
        [0, -1, 1, 0, 196.4, 100 + textWidth / 2],
    ];

    for (const [pageIndex, route] of expectedRoutes.entries()) {
        const content = decodePageContent(pdfDoc, pageIndex);
        const segments = lineSegments(content);
        assert.equal(segments.length, 4, `page ${pageIndex + 1} exports all line primitives`);
        closePoint(segments[0].start, route[0]);
        closePoint(segments[0].end, route[1]);
        closePoint(segments[1].start, route[1]);
        closePoint(segments[1].end, route[2]);
        closePoint(segments[2].start, route[2]);
        closePoint(segments[3].start, route[2]);

        const expectedWings = expectedArrowWingEnds(route[1], route[2], 10);
        const actualWings = segments.slice(2).map(segment => segment.end)
            .sort((left, right) => left.x - right.x || left.y - right.y);
        closePoint(actualWings[0], expectedWings[0]);
        closePoint(actualWings[1], expectedWings[1]);

        const circle = circleGeometry(content);
        closePoint(circle.center, route[0]);
        close(circle.radiusX, 6);
        close(circle.radiusY, 6);

        const matrix = textMatrix(content);
        expectedTextMatrices[pageIndex].forEach((expected, index) => close(matrix[index], expected));
    }
});

test('PDF.js UserUnit geometry survives rotated offset-CropBox export without rescaling', async () => {
    const sourcePdf = await createSourcePdf({
        rotation: 90,
        cropBox: { x: 20, y: 10, width: 160, height: 80 },
        userUnit: 2,
    });
    const { geometry, userUnit: pdfJsUserUnit } = await geometryFromPdfJs(sourcePdf, {
        start: 50,
    });
    assert.equal(pdfJsUserUnit, 2);
    assert.deepEqual(geometry, {
        pageIndex: 1,
        start: 50,
        sourceWidth: 160,
        sourceHeight: 80,
        rotation: 90,
        width: 80,
        height: 160,
    });
    const snapshot = {
        pages: [1],
        page_geometries: [geometry],
        stamps: [{
            uuid: 'user-unit-stamp',
            id: '7',
            pageIndex: 1,
            fs: 10,
            ratio: 0.6,
            routingPath: [
                { absX: 20, absY: 80 },
                { absX: 60, absY: 80 },
                { absX: 60, absY: 170 },
            ],
        }],
    };

    const exported = await renderSnapshotPdf(snapshot, {
        pdfLib: PDFLib,
        sourcePdf,
        resolveStampColor: () => '#112233',
        cleanText: value => value,
    });
    const pdfDoc = await PDFLib.PDFDocument.load(exported);
    const page = pdfDoc.getPage(0);
    const userUnit = page.node.get(PDFLib.PDFName.of('UserUnit'));
    assert.equal(userUnit.asNumber(), 2);

    const content = decodePageContent(pdfDoc);
    const segments = lineSegments(content);
    closePoint(segments[0].start, { x: 50, y: 30 });
    closePoint(segments[0].end, { x: 50, y: 70 });
    closePoint(segments[1].end, { x: 140, y: 70 });
    closePoint(circleGeometry(content).center, { x: 50, y: 30 });
    const textWidth = await helveticaBoldWidth('7', 10);
    const matrix = textMatrix(content);
    [0, 1, -1, 0].forEach((expected, index) => close(matrix[index], expected));
    close(matrix[4], 53.6);
    close(matrix[5], 30 - textWidth / 2);
});
