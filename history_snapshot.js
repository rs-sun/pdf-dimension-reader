// history_snapshot.js — serializable undo snapshots for frontend document state

function _clone(value) {
    return structuredClone(value);
}

export function createHistorySnapshot(source) {
    return {
        stamps: _clone(source.stamps || []),
        deletedStamps: _clone(source.deletedStamps || []),
        inspection: _clone(source.inspection || {}),
        reviewWorkspace: _clone(source.reviewWorkspace || {
            envelopes: [], decisions: {}, selectedCandidateId: null,
        }),
        viewBoxes: _clone(source.viewBoxes || []),
        basicDimensions: _clone(source.basicDimensions || []),
        datumReferences: _clone(source.datumReferences || []),
        clusters: _clone(source.clusters || []),
        selectedUUIDs: _clone(source.selectedUUIDs || []),
        l1CacheEntries: _clone([...(source.l1Cache || new Map()).entries()]),
        analyzedPages: _clone([...(source.currentAnalyzedPages || new Set()).values()]),
    };
}

export function restoreHistorySnapshot(target, snapshot) {
    // Old archives used the bare stamp array as their history representation.
    if (Array.isArray(snapshot)) {
        target.stamps = _clone(snapshot);
        target.reviewWorkspace = {
            envelopes: [], decisions: {}, selectedCandidateId: null,
        };
        target.viewBoxes = [];
        target.basicDimensions = [];
        target.datumReferences = [];
        return;
    }
    target.stamps = _clone(snapshot?.stamps || []);
    if (snapshot && Object.prototype.hasOwnProperty.call(snapshot, 'deletedStamps')) {
        target.deletedStamps = _clone(snapshot.deletedStamps || []);
    }
    if (snapshot && Object.prototype.hasOwnProperty.call(snapshot, 'inspection')) {
        target.inspection = _clone(snapshot.inspection || {});
    }
    target.reviewWorkspace = _clone(snapshot?.reviewWorkspace || {
        envelopes: [], decisions: {}, selectedCandidateId: null,
    });
    target.viewBoxes = _clone(snapshot?.viewBoxes || []);
    target.basicDimensions = _clone(snapshot?.basicDimensions || []);
    target.datumReferences = _clone(snapshot?.datumReferences || []);
    if (snapshot?.clusters) target.clusters = _clone(snapshot.clusters);
    if (snapshot?.selectedUUIDs) target.selectedUUIDs = _clone(snapshot.selectedUUIDs);
    if (snapshot?.l1CacheEntries) target.l1Cache = new Map(_clone(snapshot.l1CacheEntries));
    if (snapshot?.analyzedPages) {
        target.currentAnalyzedPages = new Set(_clone(snapshot.analyzedPages));
    }
}
