import { test } from 'node:test';
import assert from 'node:assert/strict';

const storage = new Map();
globalThis.localStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: key => storage.delete(key),
};

function fakeElement(id = '') {
  return {
    id,
    style: { setProperty() {} },
    dataset: {},
    classList: { add() {}, remove() {}, toggle() {} },
    innerHTML: '',
    innerText: '',
    textContent: '',
    disabled: false,
    appendChild() {},
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    getContext() { return {}; },
    getBoundingClientRect() {
      return { width: 100, height: 100, left: 0, top: 0 };
    },
  };
}

const elements = new Map();
globalThis.document = {
  body: fakeElement('body'),
  addEventListener() {},
  removeEventListener() {},
  createElement: tag => fakeElement(tag),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, fakeElement(id));
    return elements.get(id);
  },
  querySelector() { return null; },
  querySelectorAll() { return []; },
};
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.removeEventListener = () => {};
globalThis.requestAnimationFrame = () => 1;
globalThis.cancelAnimationFrame = () => {};

const { state } = await import('../state.js');
const {
  formatTransportLimitText,
  formatViewSegCapabilityReason,
  syncAnalyzeScopeHintVisibility,
  updateSelectedPanel,
} = await import('../app_sidebar/control_panel.js');
const {
  ARCHIVE_SCHEMA,
  parseArchivePayload,
} = await import('../import_export_transaction.js');

test('selected stamp source fallback cannot inject archived markup', () => {
  const payload = '<img src=x onerror="globalThis.__archivedXss = true">';
  const archived = parseArchivePayload({
    schema: ARCHIVE_SCHEMA,
    pages: [1],
    doc_info: { totalPages: 1 },
    stamps: [{
      uuid: 'malicious-stamp',
      id: '7',
      pageIndex: 1,
      source: payload,
    }],
  }, { pdfPageCount: 1 });
  state.stamps = archived.stamps;
  state.selectedUUIDs = ['malicious-stamp'];

  updateSelectedPanel();

  const html = document.getElementById('selected-stamp-info').innerHTML;
  assert.doesNotMatch(html, /<img\b/i);
  assert.match(html, /&lt;img src=x onerror=&quot;/);
});

test('transport limit copy distinguishes local file access from whole request body', () => {
  assert.equal(
    formatTransportLimitText(33_554_432),
    '服务端旧版传输上限：32.00 MiB（包含 JSON 与 base64；不是原始 PDF 大小）',
  );
  assert.equal(
    formatTransportLimitText(null),
    '服务端旧版传输上限：后端未报告；提交时仍会按 HTTP 状态处理',
  );
});

test('strict template readiness reasons are rendered as actionable text', () => {
  assert.equal(
    formatViewSegCapabilityReason('template_library_path_unconfigured'),
    '严格模式数字模板未配置',
  );
  assert.equal(
    formatViewSegCapabilityReason('template_library_invalid'),
    '严格模式数字模板无效',
  );
  assert.equal(
    formatViewSegCapabilityReason('strict_forbidden_runtime_loaded'),
    '本进程已加载 OCR 或严格模式禁用组件，需重启后端后恢复严格模式',
  );
  assert.equal(
    formatViewSegCapabilityReason('vector_only_preloaded_runtime_forbidden'),
    '本进程已加载 OCR，需重启后端后恢复严格模式',
  );
});

test('analyze scope hint is visible only for OCR-enabled dimension paths', () => {
  const hint = document.getElementById('analyze-scope-ocr-hint');

  state.recognitionSettings.dimension_path = 'vector_only';
  syncAnalyzeScopeHintVisibility();
  assert.equal(hint.style.display, 'none');

  for (const dimensionPath of ['vector_plus_ocr', 'ocr_only']) {
    state.recognitionSettings.dimension_path = dimensionPath;
    syncAnalyzeScopeHintVisibility();
    assert.equal(hint.style.display, '');
  }

  state.recognitionSettings.dimension_path = 'vector_only';
  syncAnalyzeScopeHintVisibility();
});
