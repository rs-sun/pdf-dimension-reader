// export_affordance.js — 导出按钮三态的纯展示模型；不决定导出数据或行为。

export function getExportAffordance(stamps) {
    const rows = Array.isArray(stamps) ? stamps : [];
    if (rows.length === 0) {
        return {
            state: 'empty',
            unconfirmedCount: 0,
            title: '无数据可供导出',
        };
    }

    const unconfirmedCount = rows.filter(stamp => !stamp.confirmed).length;
    if (unconfirmedCount > 0) {
        return {
            state: 'warning',
            unconfirmedCount,
            title: `含 ${unconfirmedCount} 条未确认，导出需二次确认`,
        };
    }

    return {
        state: 'ready',
        unconfirmedCount: 0,
        title: '导出工程与记录',
    };
}
