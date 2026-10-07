// reanalysis_transaction.js — collect per-page results, then commit once

function _clone(value) {
    return structuredClone(value);
}

function _documentGeneration(state) {
    return Number.isSafeInteger(state?.documentGeneration)
        && state.documentGeneration >= 0
        ? state.documentGeneration
        : 0;
}

function _staleOutcome({
    state,
    expectedDocumentGeneration,
    stagedPages = [],
    failedPages = [],
    pageFailures = [],
    skippedPages = [],
}) {
    return {
        status: 'stale',
        code: 'document_generation_changed',
        succeededPages: [],
        discardedPages: stagedPages.map(page => page.pageIndex),
        failedPages,
        ...(skippedPages.length > 0 ? { skippedPages } : {}),
        statePreserved: true,
        pageFailures,
        expectedDocumentGeneration,
        currentDocumentGeneration: _documentGeneration(state),
    };
}

const CLOSED_MODES = new Set(['shadow', 'audit', 'audit_only', 'unavailable', 'closed', 'fail_closed']);
const CLOSED_RESULT_KINDS = new Set(['shadow', 'audit', 'audit_only', 'unavailable', 'fail_closed']);

function _responseAuthorityReasons(result) {
    const reasons = [];
    if (result.consumer_allowed === false) reasons.push('response_consumer_closed');
    if (result.display_allowed === false) reasons.push('response_display_closed');
    if (result.release_allowed === false) reasons.push('response_release_closed');
    if (result.audit_only === true) reasons.push('response_audit_only');
    if (CLOSED_RESULT_KINDS.has(result.result_kind)) reasons.push('response_result_kind_closed');
    if (CLOSED_MODES.has(result.effective_mode)) reasons.push('response_mode_not_product');

    const isStrictVector = result.effective_settings?.dimension_path === 'vector_only'
        || result.strict_yolo_view_status_v1?.schema_version === 'strict_yolo_view_status_v1'
        || result.r32_consumer_runtime_audit_v1?.schema_version === 'r32_consumer_runtime_audit_v1';
    if (isStrictVector && result.feeds_display_l3 === false) {
        reasons.push('strict_vector_display_feed_closed');
    }

    const hasModeAuthority = Object.prototype.hasOwnProperty.call(result, 'analysis_mode_v1');
    const mode = result.analysis_mode_v1;
    if (hasModeAuthority) {
        if (!mode || typeof mode !== 'object'
            || mode.schema_version !== 'analysis_mode_v1') {
            reasons.push('analysis_mode_schema_invalid');
        } else if (mode.effective_mode === 'product') {
            if (mode.consumer_allowed !== true) reasons.push('analysis_mode_consumer_not_open');
            if (mode.display_allowed !== true) reasons.push('analysis_mode_display_not_open');
            if (mode.release_allowed !== true) reasons.push('analysis_mode_release_not_open');
            if (mode.audit_only === true) reasons.push('analysis_mode_audit_only');
            if (CLOSED_RESULT_KINDS.has(mode.result_kind)) reasons.push('analysis_mode_result_kind_closed');
        } else {
            reasons.push('analysis_mode_not_product');
            if (mode.consumer_allowed === false) reasons.push('analysis_mode_consumer_closed');
            if (mode.display_allowed === false) reasons.push('analysis_mode_display_closed');
            if (mode.release_allowed === false) reasons.push('analysis_mode_release_closed');
            if (mode.audit_only === true) reasons.push('analysis_mode_audit_only');
            if (CLOSED_RESULT_KINDS.has(mode.result_kind)) reasons.push('analysis_mode_result_kind_closed');
        }
    } else {
        // R32 strict success responses predate analysis_mode_v1. Their product-level
        // closure is conveyed by the vector_only setting plus strict runtime audits.
        // Once a versioned product mode exists above, detector-local audit flags no
        // longer decide product eligibility.
        const strictConsumerClosed = isStrictVector && (
            result.r32_consumer_runtime_audit_v1?.consumer_allowed === false
            || result.strict_yolo_view_status_v1?.consumer_allowed === false
            || result.view_seg_status_v1?.consumer_allowed === false
        );
        if (strictConsumerClosed) reasons.push('strict_vector_shadow_consumer_closed');
        if (isStrictVector && reasons.length === 0) {
            reasons.push('strict_vector_product_authority_missing');
        }
    }
    return reasons;
}

function _analysisResultEligibility(result) {
    if (!result || typeof result !== 'object') {
        return { usable: false, reasons: ['analysis_result_missing'] };
    }
    if (result.status !== 'success') {
        return { usable: false, reasons: ['analysis_status_not_success'] };
    }
    const authorityReasons = _responseAuthorityReasons(result);
    if (authorityReasons.length > 0) {
        return { usable: false, reasons: authorityReasons };
    }
    const hasVisibleArray = [
        result.dimensions,
        result.basic_dimensions,
        result.view_boxes,
        result.notes,
    ].some(items => Array.isArray(items) && items.length > 0);
    const hasDatumReference = Array.isArray(result.references)
        && result.references.some(reference => reference?.type === 'datum_reference');
    if (!hasVisibleArray && !hasDatumReference) {
        return { usable: false, reasons: ['analysis_result_not_usable'] };
    }
    return { usable: true, reasons: [] };
}

export function isUsableAnalysisResult(result) {
    return _analysisResultEligibility(result).usable;
}

export function snapshotReanalysisState(state) {
    return {
        documentGeneration: _documentGeneration(state),
        stamps: _clone(state.stamps || []),
        viewBoxes: _clone(state.viewBoxes || []),
        basicDimensions: _clone(state.basicDimensions || []),
        datumReferences: _clone(state.datumReferences || []),
        clusters: _clone(state.clusters || []),
        selectedUUIDs: _clone(state.selectedUUIDs || []),
        l1Cache: new Map(_clone([...(state.l1Cache || new Map()).entries()])),
        currentAnalyzedPages: new Set(_clone([...(state.currentAnalyzedPages || new Set()).values()])),
    };
}

function _normalizeStagedPage(staged, pageIndex) {
    if (!staged || typeof staged !== 'object') {
        throw new TypeError(`page ${pageIndex} did not produce staged data`);
    }
    const normalized = {
        pageIndex,
        stamps: _clone(staged.stamps || []),
        viewBoxes: _clone(staged.viewBoxes || []),
        basicDimensions: _clone(staged.basicDimensions || []),
        datumReferences: _clone(staged.datumReferences || []),
        clusters: _clone(staged.clusters || []),
        cacheValue: _clone(staged.cacheValue ?? []),
    };
    const hasVisibleData = [
        normalized.stamps,
        normalized.viewBoxes,
        normalized.basicDimensions,
        normalized.datumReferences,
    ].some(items => items.length > 0);
    if (!hasVisibleData) {
        throw new TypeError(`page ${pageIndex} produced no valid staged data`);
    }
    return normalized;
}

export function buildReanalysisCommit(snapshot, stagedPages) {
    const successfulPages = new Set(stagedPages.map(page => page.pageIndex));
    const stagedStamps = stagedPages.flatMap(page => page.stamps);
    const stagedViewBoxes = stagedPages.flatMap(page => page.viewBoxes);
    const stagedBasic = stagedPages.flatMap(page => page.basicDimensions);
    const stagedReferences = stagedPages.flatMap(page => page.datumReferences);
    const stagedClusters = stagedPages.flatMap(page => page.clusters);

    const stamps = snapshot.stamps
        .filter(stamp => !(stamp.isAutoGenerated && successfulPages.has(stamp.pageIndex)))
        .concat(stagedStamps);
    const stampIds = new Set(stamps.map(stamp => stamp.uuid));
    const l1Cache = new Map(snapshot.l1Cache);
    const currentAnalyzedPages = new Set(snapshot.currentAnalyzedPages);
    for (const page of stagedPages) {
        l1Cache.set(page.pageIndex, page.cacheValue);
        currentAnalyzedPages.add(page.pageIndex);
    }

    return {
        stamps,
        viewBoxes: snapshot.viewBoxes
            .filter(box => !successfulPages.has(box.pageIndex))
            .concat(stagedViewBoxes),
        basicDimensions: snapshot.basicDimensions
            .filter(item => !successfulPages.has(item.pageIndex))
            .concat(stagedBasic),
        datumReferences: snapshot.datumReferences
            .filter(item => !successfulPages.has(item.pageIndex))
            .concat(stagedReferences),
        clusters: snapshot.clusters
            .filter(cluster => !successfulPages.has(cluster.pageIndex))
            .concat(stagedClusters),
        selectedUUIDs: snapshot.selectedUUIDs.filter(uuid => stampIds.has(uuid)),
        l1Cache,
        currentAnalyzedPages,
    };
}

export function commitReanalysisState(
    state,
    next,
    expectedDocumentGeneration,
) {
    if (
        !Number.isSafeInteger(expectedDocumentGeneration)
        || expectedDocumentGeneration < 0
        || _documentGeneration(state) !== expectedDocumentGeneration
    ) return false;
    state.stamps = next.stamps;
    state.viewBoxes = next.viewBoxes;
    state.basicDimensions = next.basicDimensions;
    state.datumReferences = next.datumReferences;
    state.clusters = next.clusters;
    state.selectedUUIDs = next.selectedUUIDs;
    state.l1Cache = next.l1Cache;
    state.currentAnalyzedPages = next.currentAnalyzedPages;
    return true;
}

function _uniquePages(pages) {
    const normalized = [...new Set((pages || []).map(Number))];
    if (normalized.some(page => !Number.isInteger(page) || page < 1)) {
        throw new TypeError('pages must contain positive integers');
    }
    return normalized;
}

export async function executeReanalysisTransaction({
    state,
    pages,
    analyzePage,
    stagePage,
    pushHistory,
    beforeCommit,
    signal,
    onProgress,
}) {
    if (state.isAnalysisRunning) {
        return {
            status: 'busy',
            succeededPages: [],
            failedPages: [],
            statePreserved: true,
            pageFailures: [],
        };
    }
    const requestedPages = _uniquePages(pages);
    if (requestedPages.length === 0) {
        return {
            status: 'failed',
            succeededPages: [],
            failedPages: [],
            statePreserved: true,
            pageFailures: [],
        };
    }

    state.isAnalysisRunning = true;
    const snapshot = snapshotReanalysisState(state);
    const expectedDocumentGeneration = snapshot.documentGeneration;
    const stagedPages = [];
    const failedPages = [];
    const pageFailures = [];
    const skippedPages = [];
    try {
        for (let index = 0; index < requestedPages.length; index += 1) {
            if (_documentGeneration(state) !== expectedDocumentGeneration) {
                return _staleOutcome({
                    state,
                    expectedDocumentGeneration,
                    stagedPages,
                    failedPages,
                    pageFailures,
                    skippedPages,
                });
            }
            if (signal?.aborted) {
                return {
                    status: 'cancelled',
                    succeededPages: [],
                    failedPages,
                    statePreserved: true,
                    pageFailures,
                };
            }
            const pageIndex = requestedPages[index];
            let rawResult = null;
            let failurePhase = 'analysis';
            let stopAfterCurrentPage = false;
            try {
                rawResult = await analyzePage(pageIndex, { signal });
                if (signal?.aborted) {
                    return {
                        status: 'cancelled',
                        succeededPages: [],
                        failedPages,
                        statePreserved: true,
                        pageFailures,
                    };
                }
                if (_documentGeneration(state) !== expectedDocumentGeneration) {
                    return _staleOutcome({
                        state,
                        expectedDocumentGeneration,
                        stagedPages,
                        failedPages,
                        pageFailures,
                        skippedPages,
                    });
                }
                const eligibility = _analysisResultEligibility(rawResult);
                if (!eligibility.usable) {
                    failedPages.push(pageIndex);
                    pageFailures.push({
                        pageIndex,
                        code: 'analysis_result_not_product_eligible',
                        reasons: eligibility.reasons,
                    });
                } else {
                    failurePhase = 'staging';
                    const stagedPage = _normalizeStagedPage(
                        await stagePage(rawResult, pageIndex),
                        pageIndex,
                    );
                    if (_documentGeneration(state) !== expectedDocumentGeneration) {
                        return _staleOutcome({
                            state,
                            expectedDocumentGeneration,
                            stagedPages,
                            failedPages,
                            pageFailures,
                            skippedPages,
                        });
                    }
                    stagedPages.push(stagedPage);
                }
            } catch (error) {
                if (signal?.aborted) {
                    return {
                        status: 'cancelled',
                        succeededPages: [],
                        failedPages,
                        statePreserved: true,
                        pageFailures,
                    };
                }
                failedPages.push(pageIndex);
                if (
                    failurePhase === 'analysis'
                    && typeof error?.code === 'string'
                    && error.code
                ) {
                    pageFailures.push({
                        pageIndex,
                        code: error.code,
                        reasons: Array.isArray(error.reasons) ? [...error.reasons] : [],
                        message: error?.message || String(error),
                        status: Number.isInteger(error?.status) ? error.status : null,
                        retryable: error?.retryable !== false,
                    });
                    stopAfterCurrentPage = error?.retryable === false;
                } else {
                    pageFailures.push({
                        pageIndex,
                        code: failurePhase === 'staging'
                            ? 'analysis_result_staging_failed'
                            : 'analysis_request_failed',
                        reasons: [failurePhase === 'staging'
                            ? 'analysis_result_staging_failed'
                            : 'analysis_request_failed'],
                    });
                }
            }
            onProgress?.({
                completed: index + 1,
                total: requestedPages.length,
                pageIndex,
                ok: stagedPages.some(page => page.pageIndex === pageIndex),
            });
            if (stopAfterCurrentPage) {
                skippedPages.push(...requestedPages.slice(index + 1));
                break;
            }
        }

        if (_documentGeneration(state) !== expectedDocumentGeneration) {
            return _staleOutcome({
                state,
                expectedDocumentGeneration,
                stagedPages,
                failedPages,
                pageFailures,
                skippedPages,
            });
        }
        if (stagedPages.length === 0) {
            return {
                status: 'failed',
                succeededPages: [],
                failedPages,
                statePreserved: true,
                pageFailures,
                ...(skippedPages.length > 0 ? { skippedPages } : {}),
            };
        }
        const next = buildReanalysisCommit(snapshot, stagedPages);
        beforeCommit?.(snapshot);
        if (!commitReanalysisState(state, next, expectedDocumentGeneration)) {
            return _staleOutcome({
                state,
                expectedDocumentGeneration,
                stagedPages,
                failedPages,
                pageFailures,
                skippedPages,
            });
        }
        pushHistory?.();
        return {
            status: failedPages.length > 0 ? 'partial' : 'success',
            succeededPages: stagedPages.map(page => page.pageIndex),
            failedPages,
            ...(skippedPages.length > 0 ? { skippedPages } : {}),
            ...(pageFailures.length > 0 ? { pageFailures } : {}),
        };
    } finally {
        state.isAnalysisRunning = false;
    }
}
