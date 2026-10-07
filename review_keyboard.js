// Pure intent resolver for the review workbench. DOM/state mutations remain in
// app.js so the key map can be tested without booting the browser application.

const RESOLVED_REVIEW_STATUSES = new Set(['confirmed', 'dismissed']);

export function shouldPreserveNativeKeyboardTarget({
    tagName = '',
    isContentEditable = false,
    isNativeInteractive = false,
}) {
    return ['input', 'textarea', 'select'].includes(String(tagName).toLowerCase())
        || isContentEditable
        || isNativeInteractive;
}

export function resolveReviewShortcutTarget({
    key,
    isListTarget = false,
    isEditor = false,
}) {
    if (isEditor) {
        return key === 'Enter' || key === 'Escape' ? 'editor' : 'native';
    }
    return isListTarget ? 'list' : 'native';
}

function navigationCandidate(candidateIds, selectedCandidateId, direction) {
    const ids = Array.isArray(candidateIds) ? candidateIds : [];
    if (!ids.length) return null;
    const current = ids.indexOf(selectedCandidateId);
    const nextIndex = direction === 'down'
        ? Math.min(current + 1, ids.length - 1)
        : Math.max(current - 1, 0);
    return ids[nextIndex] || null;
}

export function resolveReviewShortcut({
    key,
    candidateIds,
    selectedCandidateId = null,
    selectedStatus = null,
    isEditing = false,
}) {
    if (key === 'ArrowDown' || key === 'ArrowUp') {
        const candidateId = navigationCandidate(
            candidateIds,
            selectedCandidateId,
            key === 'ArrowDown' ? 'down' : 'up',
        );
        return candidateId
            ? { action: 'select', candidateId }
            : { action: 'none' };
    }

    if (!selectedCandidateId) return { action: 'none' };
    const resolved = RESOLVED_REVIEW_STATUSES.has(selectedStatus);

    if (key === 'Enter' && !resolved) {
        return {
            action: isEditing ? 'confirm' : 'edit',
            candidateId: selectedCandidateId,
        };
    }
    if (key === 'Escape' && isEditing) {
        return { action: 'cancel-edit', candidateId: selectedCandidateId };
    }
    if (
        (key === 'Delete' || key === 'Backspace')
        && !isEditing
        && !resolved
    ) {
        return { action: 'dismiss', candidateId: selectedCandidateId };
    }
    return { action: 'none' };
}
