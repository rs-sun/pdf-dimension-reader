import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const css = await readFile(new URL('../styles.css', import.meta.url), 'utf8');
const drawerModule = await readFile(
    new URL('../vector_debug_drawer.js', import.meta.url),
    'utf8',
);

test('vector debug controls and drawer are explicit and default hidden/off', () => {
    assert.match(html, /id="toggle-vector-layer-debug"/);
    assert.doesNotMatch(
        html.match(/<input[^>]+id="toggle-vector-layer-debug"[^>]*>/)?.[0] || '',
        /\bchecked\b/,
    );
    assert.match(html, /id="btn-open-vector-debug" disabled/);
    assert.match(html, /<aside id="vector-debug-drawer" hidden/);
    assert.match(html, /<canvas id="vector-debug-layer" hidden/);
    assert.match(html, /<select id="vector-debug-page-select"/);
    assert.match(drawerModule, /vector-debug-page-select/);
});

test('vector debug overlay remains isolated below the formal draw layer', () => {
    const debugRule = css.match(/#vector-debug-layer\s*\{[^}]+\}/s)?.[0] || '';
    const productRule = css.match(/#draw-layer\s*\{[^}]+\}/s)?.[0] || '';
    assert.match(debugRule, /pointer-events:\s*none/);
    assert.match(debugRule, /z-index:\s*999/);
    assert.match(productRule, /z-index:\s*1000/);
});

test('debug drawer is not embedded in the product sidebar', () => {
    const sidebar = html.match(/<div id="sidebar-container">[\s\S]*?<\/div>\s*<\/div>\s*<!-- 控制面板 -->/)?.[0] || '';
    assert.ok(sidebar.length > 0);
    assert.doesNotMatch(sidebar, /vector-debug-drawer/);
});

test('opening the debug drawer cannot remain behind the control-panel backdrop', () => {
    const drawerRule = css.match(/#vector-debug-drawer\s*\{[^}]+\}/s)?.[0] || '';
    assert.match(drawerRule, /z-index:\s*5100/);
    assert.match(drawerModule, /controlPanel\.style\.display = 'none'/);
    assert.match(drawerModule, /controlBackdrop\.style\.display = 'none'/);
});
