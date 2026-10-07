import test from 'node:test';
import assert from 'node:assert/strict';

import { generateRoutedStamps } from '../stamp.js';
import {
    STAMP_LAYOUT_SCHEMA_VERSION,
    buildStampLayoutV1,
    summarizeStampLayoutV1,
} from '../stamp_layout.js';

function stamp(id, x, y, extra = {}) {
    return {
        uuid: `stamp-${id}`,
        id: String(id),
        pageIndex: 1,
        viewBoxId: 'view-a',
        fs: 12,
        circleCenter: { absX: x, absY: y },
        arrowTip: { absX: x + 20, absY: y },
        dimBbox: { x: x + 18, y: y - 4, w: 12, h: 8 },
        sortX: x,
        sortY: y,
        ...extra,
    };
}

function roundedPoint(point) {
    return {
        x: Number(point.absX.toFixed(4)),
        y: Number(point.absY.toFixed(4)),
    };
}

test('stamp_layout_v1 records routed stamp order and geometry without collisions', () => {
    const boxes = [
        { x: 20, y: 20, w: 12, h: 8 },
        { x: 60, y: 20, w: 12, h: 8 },
        { x: 20, y: 60, w: 12, h: 8 },
    ];
    const stamps = generateRoutedStamps(boxes, 50, 50, [], []);
    stamps.forEach((s, idx) => {
        s.pageIndex = 1;
        s.viewBoxId = 'view-a';
        s.sortX = boxes[idx].x + boxes[idx].w / 2;
        s.sortY = boxes[idx].y + boxes[idx].h / 2;
    });

    const layout = buildStampLayoutV1(stamps);
    const summary = summarizeStampLayoutV1(layout);

    assert.equal(layout.length, 3);
    assert.equal(layout[0].schema_version, STAMP_LAYOUT_SCHEMA_VERSION);
    assert.deepEqual(layout.map(row => row.stamp_id), ['1', '2', '3']);
    assert.ok(layout.every(row => row.dim_bbox && row.circle && row.arrow_tip));
    assert.ok(layout.every(row => row.view_group === 'view-a'));
    assert.equal(summary.stamp_order_mismatch_count, 0);
    assert.deepEqual(summary.collision_counts, {});
});

test('generateRoutedStamps keeps the active radial-routing geometry deterministic', () => {
    const boxes = [
        { x: 20, y: 20, w: 12, h: 8 },
        { x: 60, y: 20, w: 12, h: 8 },
        { x: 20, y: 60, w: 12, h: 8 },
    ];

    const route = () => generateRoutedStamps(boxes, 50, 50, [], []).map(s => ({
        id: s.id,
        fs: s.fs,
        circle: roundedPoint(s.circleCenter),
        arrow: roundedPoint(s.arrowTip),
        dimBbox: s.dimBbox,
    }));
    const expected = [
        {
            id: '1', fs: 12,
            circle: { x: 5.6081, y: 14.4911 },
            arrow: { x: 12.7495, y: 17.8212 },
            dimBbox: { x: 16, y: 16, w: 20, h: 16 },
        },
        {
            id: '2', fs: 12,
            circle: { x: 86.3919, y: 14.4911 },
            arrow: { x: 79.2505, y: 17.8212 },
            dimBbox: { x: 56, y: 16, w: 20, h: 16 },
        },
        {
            id: '3', fs: 12,
            circle: { x: 5.6081, y: 73.5089 },
            arrow: { x: 12.7495, y: 70.1788 },
            dimBbox: { x: 16, y: 56, w: 20, h: 16 },
        },
    ];

    assert.deepEqual(route(), expected);
    assert.deepEqual(route(), expected);
});

test('generateRoutedStamps honors and extends caller-owned shared routing state', () => {
    const box = { x: 100, y: 100, w: 20, h: 10 };
    const occupancy = [{ x: 120, y: 80, w: 50, h: 50 }];
    const leaderLines = [];

    const [routed] = generateRoutedStamps(
        [box], 100, 100, occupancy, leaderLines
    );

    assert.ok(routed.circleCenter.absX < box.x, 'blocker should force the stamp left');
    assert.equal(occupancy.length, 2, 'new stamp AABB should be shared with later groups');
    assert.equal(leaderLines.length, 1, 'new leader should be shared with later groups');
    assert.deepEqual(leaderLines[0], {
        x1: routed.circleCenter.absX,
        y1: routed.circleCenter.absY,
        x2: 110,
        y2: 105,
    });
});

test('stamp_layout_v1 flags stamp collisions for deterministic replay', () => {
    const layout = buildStampLayoutV1([
        stamp(1, 100, 100),
        stamp(2, 104, 100),
    ]);
    const summary = summarizeStampLayoutV1(layout);

    assert.equal(layout[0].collision_flags.circle_overlaps_stamp, true);
    assert.equal(layout[1].collision_flags.circle_overlaps_stamp, true);
    assert.equal(summary.collision_counts.circle_overlaps_stamp, 2);
});

test('stamp_layout_v1 reports numbering order mismatches by layout order', () => {
    const layout = buildStampLayoutV1([
        stamp(2, 20, 20, { sortX: 20, sortY: 20 }),
        stamp(1, 60, 20, { sortX: 60, sortY: 20 }),
    ]);
    const summary = summarizeStampLayoutV1(layout);

    assert.equal(summary.stamp_order_mismatch_count, 2);
});
