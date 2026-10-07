// js/config.js
// 全局配置模块 —— 所有硬编码值的唯一来源
// 修改此文件即可全局生效，禁止在任何其他 .js 文件中硬编码后端地址

export const CONFIG = {
    // ── 后端地址（唯一出处，A-04 D-02 修复）──────────────────────────────
    API_BASE_URL: '',

    // ── 本地打开/合并限制；服务端请求体上限由 /readyz transport_limits 发布 ──
    MAX_FILE_SIZE_MB: 50,
    MAX_PAGE_COUNT:   50,

    // ── Canvas 分辨率上限（A0 图纸按此上限缩放渲染）────────────────────────
    CANVAS_MAX_WIDTH: 4096,

    // ── DBSCAN 聚类参数────────────────────────────────────────────────────
    DBSCAN_EPS_FACTOR:     0.06,  // 默认邻域半径 = pdfWidth × 0.06（原 0.10，过大）
    DBSCAN_EPS_MIN_FACTOR: 0.02,  // Slider 下限（原 0.03）
    DBSCAN_EPS_MAX_FACTOR: 0.20,  // Slider 上限（原 0.30）
    DBSCAN_MIN_PTS:        1,     // 单点孤立标注也自成一个 Cluster

};
