// js/dbscan.js
// 纯函数模块，无副作用，不依赖任何全局状态
// TDD 参考: 2.2.2, 2.2.5, 4.3

// ─────────────────────────────────────────────────────────────────────────────
// 1. DBSCAN 聚类
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 标准 DBSCAN 聚类算法（欧氏距离）
 * @param {Array<{x: number, y: number}>} points  bbox 中心点数组
 * @param {number} eps      邻域半径
 * @param {number} minPts   最小邻域点数（固定传 1，单点孤立标注自成 Cluster）
 * @returns {number[]} labels  与 points 等长，值为簇编号(0-indexed)；minPts=1 时无噪声点
 */
export function dbscan(points, eps, minPts) {
    const n = points.length;
    const UNVISITED = -2;
    const NOISE     = -1;

    const labels = new Array(n).fill(UNVISITED);
    let clusterId = 0;

    // 预计算邻域（O(n²)，200 点以内 < 1ms，满足性能预算）
    function neighbors(idx) {
        const res = [];
        const px = points[idx].x, py = points[idx].y;
        for (let j = 0; j < n; j++) {
            const dx = px - points[j].x, dy = py - points[j].y;
            if (dx * dx + dy * dy <= eps * eps) res.push(j);
        }
        return res;
    }

    for (let i = 0; i < n; i++) {
        if (labels[i] !== UNVISITED) continue;

        const nbrs = neighbors(i);
        if (nbrs.length < minPts) {
            // minPts=1 时此分支永不触发，保留以支持未来参数调整
            labels[i] = NOISE;
            continue;
        }

        labels[i] = clusterId;
        const seeds = nbrs.filter(j => j !== i);

        while (seeds.length > 0) {
            const q = seeds.pop();
            if (labels[q] === NOISE) { labels[q] = clusterId; continue; }
            if (labels[q] !== UNVISITED) continue;

            labels[q] = clusterId;
            const qNbrs = neighbors(q);
            if (qNbrs.length >= minPts) {
                for (const r of qNbrs) {
                    if (labels[r] === UNVISITED || labels[r] === NOISE) {
                        seeds.push(r);
                    }
                }
            }
        }
        clusterId++;
    }

    // minPts=1：将所有噪声点升格为孤立单点 Cluster
    for (let i = 0; i < n; i++) {
        if (labels[i] === NOISE) {
            labels[i] = clusterId++;
        }
    }

    return labels;
}
