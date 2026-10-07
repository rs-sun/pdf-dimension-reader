// page_coordinates.js — authoritative page-local <-> document-global geometry

export const PAGE_COORDINATE_TOLERANCE = 1e-6;

export function normalizePageRotation(rotation) {
    const normalized = ((Number(rotation) % 360) + 360) % 360;
    if (![0, 90, 180, 270].includes(normalized)) {
        throw new RangeError(`page rotation must be 0, 90, 180, or 270; received ${rotation}`);
    }
    return normalized;
}

function _finiteNonNegative(value, name) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0) {
        throw new TypeError(`${name} must be a finite non-negative number`);
    }
    return number;
}

export function createPageGeometry({
    pageIndex,
    start,
    sourceWidth,
    sourceHeight,
    rotation = 0,
}) {
    const index = Number(pageIndex);
    if (!Number.isInteger(index) || index < 1) {
        throw new TypeError('pageIndex must be a positive integer');
    }
    const normalizedRotation = normalizePageRotation(rotation);
    const rawWidth = _finiteNonNegative(sourceWidth, 'sourceWidth');
    const rawHeight = _finiteNonNegative(sourceHeight, 'sourceHeight');
    const documentStart = _finiteNonNegative(start, 'start');
    const swapsAxes = normalizedRotation === 90 || normalizedRotation === 270;
    return {
        pageIndex: index,
        start: documentStart,
        sourceWidth: rawWidth,
        sourceHeight: rawHeight,
        rotation: normalizedRotation,
        width: swapsAxes ? rawHeight : rawWidth,
        height: swapsAxes ? rawWidth : rawHeight,
    };
}

function _point(point) {
    const x = Number(point?.x);
    const y = Number(point?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw new TypeError('point x/y must be finite numbers');
    }
    return { x, y };
}

function _bbox(bbox) {
    const x = Number(bbox?.x);
    const y = Number(bbox?.y);
    const w = Number(bbox?.w);
    const h = Number(bbox?.h);
    if (![x, y, w, h].every(Number.isFinite) || w < 0 || h < 0) {
        throw new TypeError('bbox x/y/w/h must be finite and w/h non-negative');
    }
    return { x, y, w, h };
}

function _pdfBox(box, name) {
    const x = Number(box?.x);
    const y = Number(box?.y);
    const width = Number(box?.width);
    const height = Number(box?.height);
    if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) {
        throw new TypeError(`${name} x/y/width/height must be finite and width/height positive`);
    }
    return { x, y, width, height };
}

export function resolvePdfVisibleBox({ mediaBox, cropBox }) {
    const media = _pdfBox(mediaBox, 'mediaBox');
    const crop = cropBox ? _pdfBox(cropBox, 'cropBox') : media;
    const x = Math.max(media.x, crop.x);
    const y = Math.max(media.y, crop.y);
    const right = Math.min(media.x + media.width, crop.x + crop.width);
    const top = Math.min(media.y + media.height, crop.y + crop.height);
    if (right <= x || top <= y) return media;
    return { x, y, width: right - x, height: top - y };
}

function _pageGeometry(page) {
    if (!page || typeof page !== 'object') throw new TypeError('page geometry is required');
    return createPageGeometry({
        pageIndex: page.pageIndex ?? 1,
        start: page.start ?? 0,
        sourceWidth: page.sourceWidth ?? page.width ?? 0,
        sourceHeight: page.sourceHeight ?? page.height ?? 0,
        rotation: page.rotation ?? 0,
    });
}

export function pageLocalPointToDocumentGlobal(point, pageGeometry) {
    const { x, y } = _point(point);
    const page = _pageGeometry(pageGeometry);
    let rendered;
    switch (page.rotation) {
        case 90:
            rendered = { x: page.sourceHeight - y, y: x };
            break;
        case 180:
            rendered = { x: page.sourceWidth - x, y: page.sourceHeight - y };
            break;
        case 270:
            rendered = { x: y, y: page.sourceWidth - x };
            break;
        default:
            rendered = { x, y };
    }
    return { x: rendered.x, y: rendered.y + page.start };
}

export function documentGlobalPointToPageLocal(point, pageGeometry) {
    const global = _point(point);
    const page = _pageGeometry(pageGeometry);
    const x = global.x;
    const y = global.y - page.start;
    switch (page.rotation) {
        case 90:
            return { x: y, y: page.sourceHeight - x };
        case 180:
            return { x: page.sourceWidth - x, y: page.sourceHeight - y };
        case 270:
            return { x: page.sourceWidth - y, y: x };
        default:
            return { x, y };
    }
}

export function documentGlobalPointToPdfUserSpace(point, pageGeometry, pdfVisibleBox) {
    const page = _pageGeometry(pageGeometry);
    const visibleBox = _pdfBox(pdfVisibleBox, 'pdfVisibleBox');
    const local = documentGlobalPointToPageLocal(point, page);
    return {
        x: visibleBox.x + local.x,
        y: visibleBox.y + visibleBox.height - local.y,
    };
}

export function pageLocalBboxToDocumentGlobal(bbox, pageGeometry) {
    const box = _bbox(bbox);
    const page = _pageGeometry(pageGeometry);
    const corners = [
        pageLocalPointToDocumentGlobal({ x: box.x, y: box.y }, page),
        pageLocalPointToDocumentGlobal({ x: box.x + box.w, y: box.y }, page),
        pageLocalPointToDocumentGlobal({ x: box.x, y: box.y + box.h }, page),
        pageLocalPointToDocumentGlobal({ x: box.x + box.w, y: box.y + box.h }, page),
    ];
    const xs = corners.map(point => point.x);
    const ys = corners.map(point => point.y);
    const x = Math.min(...xs);
    const y = Math.min(...ys);
    return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

export function documentGlobalBboxToPageLocal(bbox, pageGeometry) {
    const box = _bbox(bbox);
    const page = _pageGeometry(pageGeometry);
    const corners = [
        documentGlobalPointToPageLocal({ x: box.x, y: box.y }, page),
        documentGlobalPointToPageLocal({ x: box.x + box.w, y: box.y }, page),
        documentGlobalPointToPageLocal({ x: box.x, y: box.y + box.h }, page),
        documentGlobalPointToPageLocal({ x: box.x + box.w, y: box.y + box.h }, page),
    ];
    const xs = corners.map(point => point.x);
    const ys = corners.map(point => point.y);
    const x = Math.min(...xs);
    const y = Math.min(...ys);
    return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

export function getPageGeometry(pageGeometries, pageIndex) {
    const page = pageGeometries?.[Number(pageIndex) - 1];
    if (!page) throw new RangeError(`missing geometry for page ${pageIndex}`);
    return _pageGeometry({ ...page, pageIndex: Number(pageIndex) });
}
