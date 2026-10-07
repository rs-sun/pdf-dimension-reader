// R4 frontend stamp layout regression dump.
//
// Pure helper for deterministic Node replay tests. It does not read or mutate
// app state and is not part of the interactive rendering path.

export const STAMP_LAYOUT_SCHEMA_VERSION = 'stamp_layout_v1';

function _round(value) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.round(n * 1000) / 1000 : null;
}

function _point(point, xKey = 'x', yKey = 'y') {
    if (!point) return null;
    const x = point.absX ?? point[xKey];
    const y = point.absY ?? point[yKey];
    if (x === undefined || y === undefined) return null;
    return { x: _round(x), y: _round(y) };
}

function _bbox(box) {
    if (!box) return null;
    const x = Number(box.x);
    const y = Number(box.y);
    const w = Number(box.w);
    const h = Number(box.h);
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
    return { x: _round(x), y: _round(y), w: _round(w), h: _round(h) };
}

function _stampAabb(stamp) {
    const center = _point(stamp.circleCenter);
    if (!center) return null;
    const idDigits = String(stamp.id ?? '').length || 1;
    const fs = Number(stamp.fs || 12);
    const pad = 8;
    const scale = fs / 16;
    const w = (idDigits <= 1 ? 44 : idDigits <= 2 ? 58 : 72) * scale;
    const h = 28 * scale;
    return {
        x: center.x - w / 2 - pad,
        y: center.y - h / 2 - pad,
        w: w + pad * 2,
        h: h + pad * 2,
    };
}

function _overlap(a, b) {
    if (!a || !b) return false;
    return a.x < b.x + b.w && a.x + a.w > b.x
        && a.y < b.y + b.h && a.y + a.h > b.y;
}

function _orientation(a, b, c) {
    return (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
}

function _segmentsIntersect(a, b, c, d) {
    if (!a || !b || !c || !d) return false;
    const o1 = _orientation(a, b, c);
    const o2 = _orientation(a, b, d);
    const o3 = _orientation(c, d, a);
    const o4 = _orientation(c, d, b);
    return o1 * o2 < 0 && o3 * o4 < 0;
}

function _segmentIntersectsBox(a, b, box) {
    if (!a || !b || !box) return false;
    const insideA = a.x >= box.x && a.x <= box.x + box.w
        && a.y >= box.y && a.y <= box.y + box.h;
    const insideB = b.x >= box.x && b.x <= box.x + box.w
        && b.y >= box.y && b.y <= box.y + box.h;
    if (insideA || insideB) return true;
    const p1 = { x: box.x, y: box.y };
    const p2 = { x: box.x + box.w, y: box.y };
    const p3 = { x: box.x + box.w, y: box.y + box.h };
    const p4 = { x: box.x, y: box.y + box.h };
    return _segmentsIntersect(a, b, p1, p2)
        || _segmentsIntersect(a, b, p2, p3)
        || _segmentsIntersect(a, b, p3, p4)
        || _segmentsIntersect(a, b, p4, p1);
}

function _center(box) {
    if (!box) return null;
    return { x: box.x + box.w / 2, y: box.y + box.h / 2 };
}

function _viewGroupForStamp(stamp, viewBoxes) {
    if (stamp.viewBoxId) return String(stamp.viewBoxId);
    const dim = _bbox(stamp.dimBbox || stamp.sourceBbox);
    const center = _center(dim);
    if (!center) return null;
    const containing = (viewBoxes || []).filter(vb => {
        const box = _bbox(vb);
        return box
            && center.x >= box.x
            && center.x <= box.x + box.w
            && center.y >= box.y
            && center.y <= box.y + box.h;
    });
    if (containing.length === 0) return null;
    containing.sort((a, b) => (a.w * a.h) - (b.w * b.h));
    return String(containing[0].id ?? '');
}

function _numericId(id) {
    const match = String(id ?? '').match(/^\d+/);
    return match ? Number(match[0]) : Number.POSITIVE_INFINITY;
}

function _layoutOrderKey(row) {
    return [
        row.page_index ?? 0,
        row.view_group ?? '',
        row.sort_y ?? row.dim_bbox?.y ?? 0,
        row.sort_x ?? row.dim_bbox?.x ?? 0,
        _numericId(row.stamp_id),
        row.stamp_id ?? '',
    ];
}

function _compareOrder(a, b) {
    const ak = _layoutOrderKey(a);
    const bk = _layoutOrderKey(b);
    for (let i = 0; i < ak.length; i++) {
        if (ak[i] < bk[i]) return -1;
        if (ak[i] > bk[i]) return 1;
    }
    return 0;
}

export function buildStampLayoutV1(stamps, options = {}) {
    const source = Array.isArray(stamps) ? stamps : [];
    const viewBoxes = Array.isArray(options.viewBoxes) ? options.viewBoxes : [];
    const aabbs = source.map(_stampAabb);
    const leaders = source.map(stamp => ({
        from: _point(stamp.circleCenter),
        to: _point(stamp.arrowTip),
    }));

    return source.map((stamp, index) => {
        const dimBox = _bbox(stamp.dimBbox || stamp.sourceBbox);
        const circle = _point(stamp.circleCenter);
        const arrowTip = _point(stamp.arrowTip);
        const overlapsStamp = aabbs.some((box, otherIndex) => (
            otherIndex !== index && _overlap(aabbs[index], box)
        ));
        const leaderCrossesLeader = leaders.some((leader, otherIndex) => (
            otherIndex !== index
            && _segmentsIntersect(
                leaders[index].from,
                leaders[index].to,
                leader.from,
                leader.to,
            )
        ));
        const leaderIntersectsOtherDim = source.some((otherStamp, otherIndex) => (
            otherIndex !== index
            && _segmentIntersectsBox(
                leaders[index].from,
                leaders[index].to,
                _bbox(otherStamp.dimBbox || otherStamp.sourceBbox),
            )
        ));

        return {
            schema_version: STAMP_LAYOUT_SCHEMA_VERSION,
            stamp_id: String(stamp.id ?? ''),
            uuid: stamp.uuid ? String(stamp.uuid) : '',
            page_index: Number.isInteger(stamp.pageIndex) ? stamp.pageIndex : null,
            view_group: _viewGroupForStamp(stamp, viewBoxes),
            dim_bbox: dimBox,
            circle,
            arrow_tip: arrowTip,
            fs: Number(stamp.fs || 12),
            sort_x: _round(stamp.sortX ?? dimBox?.x ?? circle?.x),
            sort_y: _round(stamp.sortY ?? dimBox?.y ?? circle?.y),
            collision_flags: {
                missing_dim_bbox: !dimBox,
                missing_circle: !circle,
                missing_arrow_tip: !arrowTip,
                circle_overlaps_stamp: overlapsStamp,
                leader_crosses_leader: leaderCrossesLeader,
                leader_intersects_other_dim: leaderIntersectsOtherDim,
            },
        };
    });
}

export function summarizeStampLayoutV1(layout) {
    const rows = Array.isArray(layout) ? layout : [];
    const sorted = [...rows].sort(_compareOrder);
    let orderMismatchCount = 0;
    for (let i = 0; i < sorted.length; i++) {
        if (sorted[i].stamp_id !== String(i + 1)) orderMismatchCount++;
    }
    const collisionCounts = {};
    for (const row of rows) {
        for (const [key, value] of Object.entries(row.collision_flags || {})) {
            if (value) collisionCounts[key] = (collisionCounts[key] || 0) + 1;
        }
    }
    return {
        schema_version: STAMP_LAYOUT_SCHEMA_VERSION,
        stamp_count: rows.length,
        stamp_order_mismatch_count: orderMismatchCount,
        collision_counts: collisionCounts,
    };
}
