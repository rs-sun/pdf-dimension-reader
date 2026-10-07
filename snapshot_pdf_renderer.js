// snapshot_pdf_renderer.js — burn formal stamps into the source PDF snapshot

import {
    documentGlobalPointToPdfUserSpace,
    getPageGeometry,
    normalizePageRotation,
    resolvePdfVisibleBox,
} from './page_coordinates.js';

export async function renderSnapshotPdf(snapshot, {
    pdfLib,
    sourcePdf,
    resolveStampColor,
    cleanText,
}) {
    const { PDFDocument, degrees, rgb, StandardFonts } = pdfLib || {};
    if (!PDFDocument || !sourcePdf) throw new Error('pdf-lib 或原始 PDF 不可用');
    const pdfDoc = await PDFDocument.load(sourcePdf.slice(0));
    const pages = pdfDoc.getPages();
    if (pages.length !== snapshot.pages.length) throw new Error('PDF 页数与导出快照不一致');
    const font = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

    for (const stamp of snapshot.stamps) {
        const pageIndex = Number(stamp.pageIndex || 1);
        const page = pages[pageIndex - 1];
        const rawGeometry = snapshot.page_geometries[pageIndex - 1];
        if (!page || !rawGeometry) throw new Error(`标记 ${stamp.uuid || stamp.id || ''} 的页码无效`);
        const geometry = getPageGeometry(snapshot.page_geometries, pageIndex);
        const visibleBox = resolvePdfVisibleBox({
            mediaBox: page.getMediaBox(),
            cropBox: page.getCropBox(),
        });
        const toPdfUserSpace = point => documentGlobalPointToPdfUserSpace(
            point,
            geometry,
            visibleBox,
        );
        const route = stamp.routingPath?.length >= 2
            ? stamp.routingPath
            : [stamp.circleCenter, stamp.arrowTip];
        if (!route?.[0] || !route?.[1]) throw new Error(`标记 ${stamp.uuid || stamp.id || ''} 缺少坐标`);
        const globalRoute = route.map(point => ({
            x: Number(point?.absX ?? point?.x),
            y: Number(point?.absY ?? point?.y),
        }));
        if (globalRoute.some(point => !Number.isFinite(point.x) || !Number.isFinite(point.y))) {
            throw new Error(`标记 ${stamp.uuid || stamp.id || ''} 坐标无效`);
        }
        const points = globalRoute.map(toPdfUserSpace);
        const colorHex = resolveStampColor(stamp);
        const color = rgb(
            parseInt(colorHex.slice(1, 3), 16) / 255,
            parseInt(colorHex.slice(3, 5), 16) / 255,
            parseInt(colorHex.slice(5, 7), 16) / 255,
        );
        const fontSize = Number(stamp.fs || 14);
        const lineWidth = Math.max(1, Math.min(fontSize / 7, 4));
        for (let index = 0; index < points.length - 1; index += 1) {
            page.drawLine({
                start: points[index],
                end: points[index + 1],
                thickness: lineWidth,
                color,
            });
        }
        const tip = points.at(-1);
        const previous = points.at(-2);
        const angle = Math.atan2(tip.y - previous.y, tip.x - previous.x);
        const arrowLength = Math.max(8, Math.min(fontSize * 0.6 + 4, 30));
        const wing1 = {
            x: tip.x - arrowLength * Math.cos(angle - Math.PI / 6),
            y: tip.y - arrowLength * Math.sin(angle - Math.PI / 6),
        };
        const wing2 = {
            x: tip.x - arrowLength * Math.cos(angle + Math.PI / 6),
            y: tip.y - arrowLength * Math.sin(angle + Math.PI / 6),
        };
        page.drawLine({ start: tip, end: wing1, thickness: lineWidth, color });
        page.drawLine({ start: tip, end: wing2, thickness: lineWidth, color });

        const center = points[0];
        const text = cleanText(String(stamp.id || ''));
        const radius = fontSize * Number(stamp.ratio || 0.60);
        const textWidth = font.widthOfTextAtSize(text, fontSize);
        page.drawCircle({
            x: center.x,
            y: center.y,
            size: Math.max(radius, textWidth / 2 + 3),
            color: rgb(1, 1, 1),
            borderColor: color,
            borderWidth: lineWidth,
        });
        const baseline = toPdfUserSpace({
            x: globalRoute[0].x - textWidth / 2,
            y: globalRoute[0].y + fontSize * 0.36,
        });
        const textRotation = normalizePageRotation(page.getRotation().angle);
        page.drawText(text, {
            x: baseline.x,
            y: baseline.y,
            size: fontSize,
            font,
            color,
            rotate: degrees(textRotation),
        });
    }
    return pdfDoc.save();
}
