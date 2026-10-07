// app_sidebar.js — public entry, 只 re-export 子模块
// 重构 R14：拆 app_sidebar.js 为 6 个子模块，外部 import 路径不变
//   app_sidebar/shared_templates.js  — 折叠头部 / 确认行 HTML 拼装
//   app_sidebar/symbol_palette.js    — 全局单例符号面板
//   app_sidebar/expand_form.js       — 展开态字段表单 + commit + isEditingField owner
//   app_sidebar/popup.js             — 选中章悬浮编辑窗 + tether 拖拽 + CustomEvent 桥接
//   app_sidebar/list.js              — 侧边栏主列表渲染 + 排序
//   app_sidebar/control_panel.js     — 控制面板（颜色配置 + 选中章面板）

export {
    updateSidebar,
    naturalSortLogic,
    idSortLogic,
    sortOrder,
    setSortOrder,
    registerReviewCandidateCallbacks,
    getReviewNavigationCandidateIds,
    selectReviewCandidateInWorkbench,
    focusReviewCandidateEditor,
    cancelReviewCandidateEdit,
    confirmReviewCandidateFromWorkbench,
    dismissReviewCandidateFromWorkbench,
} from './app_sidebar/list.js';

export { _commitExpandedEdit } from './app_sidebar/expand_form.js';

export {
    updateStampPopup,
    repositionStampPopup,
} from './app_sidebar/popup.js';

export { _showSymbolPalette } from './app_sidebar/symbol_palette.js';

export {
    initColorRows,
    updateSelectedPanel,
    openControlPanel,
    closeControlPanel,
    refreshRecognitionCapabilities,
    registerSidebarCallbacks,
} from './app_sidebar/control_panel.js';
