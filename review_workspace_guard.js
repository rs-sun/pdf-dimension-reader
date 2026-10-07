// Session-only review candidates are intentionally excluded from project
// archives. These guards make that loss boundary explicit without changing
// the archive schema or persisting the review workspace.

import { pendingReviewCandidateCount } from './review_candidates_v1.js';

export function pendingReviewLeaveMessage(workspace) {
    const count = pendingReviewCandidateCount(workspace);
    return count > 0
        ? `还有 ${count} 条候选未确认，离开将丢失复核进度`
        : '';
}

export function confirmPendingReviewLeave(
    workspace,
    confirmImpl = globalThis.confirm,
) {
    const message = pendingReviewLeaveMessage(workspace);
    if (!message) return true;
    return typeof confirmImpl === 'function' && confirmImpl(message) === true;
}

export function applyPendingReviewBeforeUnload(event, workspace) {
    const message = pendingReviewLeaveMessage(workspace);
    if (!message) return undefined;
    event?.preventDefault?.();
    if (event) event.returnValue = message;
    return message;
}
