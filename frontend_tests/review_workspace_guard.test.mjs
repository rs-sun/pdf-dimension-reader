import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
    applyPendingReviewBeforeUnload,
    confirmPendingReviewLeave,
    pendingReviewLeaveMessage,
} from '../review_workspace_guard.js';

function workspace(statuses) {
    const items = statuses.map((status, index) => ({
        candidate_id: `candidate-${index}`,
    }));
    return {
        envelopes: [{ page_index: 0, items }],
        decisions: Object.fromEntries(statuses.flatMap((status, index) => (
            status === 'pending'
                ? []
                : [[`candidate-${index}`, { status }]]
        ))),
        selectedCandidateId: null,
    };
}

test('pending review reminder counts only pending and edited candidates', () => {
    assert.equal(
        pendingReviewLeaveMessage(workspace([
            'pending', 'edited', 'confirmed', 'dismissed',
        ])),
        '还有 2 条候选未确认，离开将丢失复核进度',
    );
    assert.equal(
        pendingReviewLeaveMessage(workspace(['confirmed', 'dismissed'])),
        '',
    );
});

test('export and document-load guards prompt once and honor cancel or continue', () => {
    const prompts = [];
    const source = workspace(['pending', 'edited']);
    assert.equal(confirmPendingReviewLeave(source, message => {
        prompts.push(message);
        return false;
    }), false);
    assert.equal(prompts.length, 1);

    assert.equal(confirmPendingReviewLeave(source, message => {
        prompts.push(message);
        return true;
    }), true);
    assert.equal(prompts.length, 2);

    assert.equal(confirmPendingReviewLeave(
        workspace(['confirmed']),
        () => {
            throw new Error('resolved workspaces must not prompt');
        },
    ), true);
});

test('beforeunload is blocked only while unresolved review progress exists', () => {
    let prevented = 0;
    const event = {
        returnValue: undefined,
        preventDefault() { prevented += 1; },
    };
    const message = applyPendingReviewBeforeUnload(
        event,
        workspace(['pending', 'edited']),
    );
    assert.equal(message, '还有 2 条候选未确认，离开将丢失复核进度');
    assert.equal(event.returnValue, message);
    assert.equal(prevented, 1);

    const resolvedEvent = {
        returnValue: undefined,
        preventDefault() {
            throw new Error('resolved workspaces must not block unload');
        },
    };
    assert.equal(
        applyPendingReviewBeforeUnload(resolvedEvent, workspace(['dismissed'])),
        undefined,
    );
    assert.equal(resolvedEvent.returnValue, undefined);
});

test('archive export, new-document load, and beforeunload use the shared guard', () => {
    const source = fs.readFileSync(
        new URL('../app_io.js', import.meta.url),
        'utf8',
    );
    assert.match(
        source,
        /getElementById\('file-upload'\)\.addEventListener\('change'[\s\S]*?confirmPendingReviewLeave\(state\.reviewWorkspace\)/,
    );
    assert.match(
        source,
        /async function _exportTransactionalArchive\(\) \{\s*if \(!confirmPendingReviewLeave\(state\.reviewWorkspace\)\) return;/,
    );
    assert.match(
        source,
        /addEventListener\('beforeunload'[\s\S]*?applyPendingReviewBeforeUnload\(event, state\.reviewWorkspace\)/,
    );
});
