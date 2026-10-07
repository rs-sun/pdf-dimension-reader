import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
    resolveReviewShortcut,
    resolveReviewShortcutTarget,
    shouldPreserveNativeKeyboardTarget,
} from '../review_keyboard.js';

const candidateIds = ['a', 'b', 'c'];

test('review arrow keys mirror clamped dimensions-list selection', () => {
    assert.deepEqual(resolveReviewShortcut({
        key: 'ArrowDown',
        candidateIds,
    }), { action: 'select', candidateId: 'a' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'ArrowDown',
        candidateIds,
        selectedCandidateId: 'b',
    }), { action: 'select', candidateId: 'c' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'ArrowDown',
        candidateIds,
        selectedCandidateId: 'c',
    }), { action: 'select', candidateId: 'c' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'ArrowUp',
        candidateIds,
        selectedCandidateId: 'b',
    }), { action: 'select', candidateId: 'a' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'ArrowUp',
        candidateIds,
    }), { action: 'select', candidateId: 'a' });
});

test('review Enter enters editing and confirms only from the active editor', () => {
    assert.deepEqual(resolveReviewShortcut({
        key: 'Enter',
        candidateIds,
        selectedCandidateId: 'b',
        selectedStatus: 'pending',
        isEditing: false,
    }), { action: 'edit', candidateId: 'b' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'Enter',
        candidateIds,
        selectedCandidateId: 'b',
        selectedStatus: 'edited',
        isEditing: true,
    }), { action: 'confirm', candidateId: 'b' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'Enter',
        candidateIds,
        selectedCandidateId: 'b',
        selectedStatus: 'confirmed',
        isEditing: false,
    }), { action: 'none' });
});

test('review Escape cancels editing and delete keys dismiss only outside inputs', () => {
    assert.deepEqual(resolveReviewShortcut({
        key: 'Escape',
        candidateIds,
        selectedCandidateId: 'a',
        selectedStatus: 'pending',
        isEditing: true,
    }), { action: 'cancel-edit', candidateId: 'a' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'Delete',
        candidateIds,
        selectedCandidateId: 'a',
        selectedStatus: 'pending',
        isEditing: false,
    }), { action: 'dismiss', candidateId: 'a' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'Backspace',
        candidateIds,
        selectedCandidateId: 'a',
        selectedStatus: 'pending',
        isEditing: true,
    }), { action: 'none' });
    assert.deepEqual(resolveReviewShortcut({
        key: 'Delete',
        candidateIds,
        selectedCandidateId: 'a',
        selectedStatus: 'dismissed',
        isEditing: false,
    }), { action: 'none' });
});

test('review shortcut ownership preserves native descendant controls and editor text keys', () => {
    assert.equal(resolveReviewShortcutTarget({
        key: 'Enter',
        isListTarget: true,
    }), 'list');
    assert.equal(resolveReviewShortcutTarget({
        key: 'Enter',
        isEditor: true,
    }), 'editor');
    assert.equal(resolveReviewShortcutTarget({
        key: 'Escape',
        isEditor: true,
    }), 'editor');
    assert.equal(resolveReviewShortcutTarget({
        key: 'Backspace',
        isEditor: true,
    }), 'native');
    assert.equal(resolveReviewShortcutTarget({
        key: 'Delete',
        isEditor: true,
    }), 'native');
    assert.equal(resolveReviewShortcutTarget({
        key: 'ArrowDown',
        isEditor: true,
    }), 'native');
    assert.equal(resolveReviewShortcutTarget({
        key: 'Enter',
        isListTarget: false,
        isEditor: false,
    }), 'native');
});

test('page-level shortcuts preserve native keyboard activation for review controls', () => {
    assert.equal(shouldPreserveNativeKeyboardTarget({
        tagName: 'BUTTON',
        isNativeInteractive: true,
    }), true);
    assert.equal(shouldPreserveNativeKeyboardTarget({
        tagName: 'SUMMARY',
        isNativeInteractive: true,
    }), true);
    assert.equal(shouldPreserveNativeKeyboardTarget({
        tagName: 'INPUT',
    }), true);
    assert.equal(shouldPreserveNativeKeyboardTarget({
        tagName: 'UL',
    }), false);
});

test('the review tab keydown path dispatches every review intent through workbench APIs', async () => {
    const source = await readFile(new URL('../app.js', import.meta.url), 'utf8');
    assert.doesNotMatch(source, /if \(state\.sidebarTab === 'review'\) return;/);
    assert.match(source, /shouldPreserveNativeKeyboardTarget\(\{/);
    assert.match(source, /button, summary, a\[href\], \[role="button"\]/);
    assert.match(source, /resolveReviewShortcutTarget\(\{/);
    assert.match(source, /if \(shortcutTarget === 'native'\) return;/);
    assert.match(source, /resolveReviewShortcut\(\{/);
    assert.match(source, /selectReviewCandidateInWorkbench\(intent\.candidateId\)/);
    assert.match(source, /focusReviewCandidateEditor\(intent\.candidateId\)/);
    assert.match(source, /cancelReviewCandidateEdit\(intent\.candidateId\)/);
    assert.match(source, /await confirmReviewCandidateFromWorkbench\(intent\.candidateId\)/);
    assert.match(source, /dismissReviewCandidateFromWorkbench\(intent\.candidateId\)/);
});
