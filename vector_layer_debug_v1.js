// Pure frontend contract for opt-in vector-layer diagnostics.
// This module deliberately has no dependency on application state or DOM code.

export const VECTOR_LAYER_DEBUG_ROOT_KEY = 'vector_layer_debug_v1';
export const VECTOR_LAYER_DEBUG_SCHEMA_VERSION = 'vector_layer_debug_v1';
export const VECTOR_LAYER_DEBUG_COORDINATE_SPACE = 'pdf_page_points_top_left';
export const VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS = 2000;
export const VECTOR_DEBUG_MAX_WORKSPACE_PAGES = 50;
export const VECTOR_DEBUG_MAX_WORKSPACE_OVERLAYS = 10_000;

const ENVELOPE_KEYS = [
    'schema_version',
    'trace_id',
    'page_index',
    'coordinate_space',
    'status',
    'unavailable_reason',
    'stages',
    'overlays',
    'limits',
    'authority',
];
const STAGE_KEYS = [
    'stage_id',
    'status',
    'input_count',
    'output_count',
    'drop_count',
    'elapsed_ms',
    'first_drop',
];
const FIRST_DROP_KEYS = ['candidate_id', 'reason_code', 'bbox'];
const OVERLAY_KEYS = ['overlay_id', 'stage_id', 'role', 'bbox', 'label'];
const BBOX_KEYS = ['x', 'y', 'w', 'h'];
const LIMIT_KEYS = ['total', 'returned', 'truncated'];
const AUTHORITY_KEYS = ['diagnostic_only', 'consumer_allowed', 'release_allowed'];
const WORKSPACE_KEYS = ['pages', 'selectedPageIndex', 'selectedStageId'];

const ENVELOPE_STATUSES = new Set(['ok', 'unavailable']);
const STAGE_STATUSES = new Set(['ok', 'not_run', 'unavailable', 'error']);
const OVERLAY_ROLES = new Set(['accepted', 'rejected', 'first_drop', 'context']);
const STABLE_TOKEN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const REASON_CODE = /^[a-z0-9][a-z0-9._-]{0,127}$/;

function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasExactKeys(value, expectedKeys) {
    if (!isRecord(value)) return false;
    const actual = Object.keys(value).sort();
    const expected = [...expectedKeys].sort();
    return actual.length === expected.length
        && actual.every((key, index) => key === expected[index]);
}

function isNonNegativeInteger(value) {
    return Number.isInteger(value) && value >= 0;
}

function isNonNegativeFinite(value) {
    return Number.isFinite(value) && value >= 0;
}

function isStableToken(value) {
    return typeof value === 'string' && STABLE_TOKEN.test(value);
}

function validateBbox(value, reasons) {
    if (!isRecord(value)) {
        reasons.add('bbox_not_object');
        return;
    }
    if (!hasExactKeys(value, BBOX_KEYS)) reasons.add('bbox_keys_invalid');
    if (!Number.isFinite(value.x) || !Number.isFinite(value.y)) {
        reasons.add('bbox_coordinate_invalid');
    }
    if (!Number.isFinite(value.w) || !Number.isFinite(value.h)
            || value.w <= 0 || value.h <= 0) {
        reasons.add('bbox_extent_invalid');
    }
}

function validateFirstDrop(value, reasons) {
    if (!isRecord(value)) {
        reasons.add('first_drop_not_object');
        return;
    }
    if (!hasExactKeys(value, FIRST_DROP_KEYS)) reasons.add('first_drop_keys_invalid');
    if (!isStableToken(value.candidate_id)) reasons.add('first_drop_candidate_id_invalid');
    if (typeof value.reason_code !== 'string' || !REASON_CODE.test(value.reason_code)) {
        reasons.add('first_drop_reason_code_invalid');
    }
    validateBbox(value.bbox, reasons);
}

function validateStage(value, reasons, stageIds) {
    if (!isRecord(value)) {
        reasons.add('stage_not_object');
        return;
    }
    if (!hasExactKeys(value, STAGE_KEYS)) reasons.add('stage_keys_invalid');
    if (!isStableToken(value.stage_id)) {
        reasons.add('stage_id_invalid');
    } else if (stageIds.has(value.stage_id)) {
        reasons.add('stage_id_duplicate');
    } else {
        stageIds.add(value.stage_id);
    }
    if (!STAGE_STATUSES.has(value.status)) reasons.add('stage_status_invalid');
    if (![value.input_count, value.output_count, value.drop_count].every(
        isNonNegativeInteger,
    )) {
        reasons.add('stage_count_invalid');
    }
    if (!isNonNegativeFinite(value.elapsed_ms)) reasons.add('stage_elapsed_ms_invalid');

    if (value.first_drop !== null) validateFirstDrop(value.first_drop, reasons);
    if (isNonNegativeInteger(value.drop_count)) {
        const mustHaveFirstDrop = value.drop_count > 0;
        if (mustHaveFirstDrop !== (value.first_drop !== null)) {
            reasons.add('stage_first_drop_inconsistent');
        }
    }
}

function validateOverlay(value, reasons, overlayIds, stageIds) {
    if (!isRecord(value)) {
        reasons.add('overlay_not_object');
        return;
    }
    if (!hasExactKeys(value, OVERLAY_KEYS)) reasons.add('overlay_keys_invalid');
    if (!isStableToken(value.overlay_id)) {
        reasons.add('overlay_id_invalid');
    } else if (overlayIds.has(value.overlay_id)) {
        reasons.add('overlay_id_duplicate');
    } else {
        overlayIds.add(value.overlay_id);
    }
    if (!isStableToken(value.stage_id)) {
        reasons.add('overlay_stage_id_invalid');
    } else if (!stageIds.has(value.stage_id)) {
        reasons.add('overlay_stage_unknown');
    }
    if (!OVERLAY_ROLES.has(value.role)) reasons.add('overlay_role_invalid');
    if (value.label !== null && typeof value.label !== 'string') {
        reasons.add('overlay_label_invalid');
    }
    validateBbox(value.bbox, reasons);
}

function validateLimits(value, overlaysLength, reasons) {
    if (!isRecord(value)) {
        reasons.add('limits_not_object');
        return;
    }
    if (!hasExactKeys(value, LIMIT_KEYS)) reasons.add('limits_keys_invalid');
    if (!isNonNegativeInteger(value.total)
            || !isNonNegativeInteger(value.returned)) {
        reasons.add('limits_count_invalid');
        return;
    }
    if (value.returned !== overlaysLength) {
        reasons.add('limits_returned_mismatch');
    }
    if (value.total < value.returned) {
        reasons.add('limits_total_invalid');
    }
    if (typeof value.truncated !== 'boolean'
            || value.truncated !== (value.total > value.returned)) {
        reasons.add('limits_truncated_mismatch');
    }
}

function validateAuthority(value, reasons) {
    if (!isRecord(value)) {
        reasons.add('authority_not_object');
        reasons.add('authority_fail_open');
        return;
    }
    if (!hasExactKeys(value, AUTHORITY_KEYS)) reasons.add('authority_keys_invalid');
    if (value.diagnostic_only !== true
            || value.consumer_allowed !== false
            || value.release_allowed !== false) {
        reasons.add('authority_fail_open');
    }
}

export class VectorLayerDebugContractError extends TypeError {
    constructor(code, reasons = []) {
        super(code);
        this.name = 'VectorLayerDebugContractError';
        this.code = code;
        this.reasons = [...reasons];
    }
}

/**
 * Validate one page-level diagnostic payload without mutating it.
 *
 * @param {unknown} value
 * @param {{expectedPageIndex?: number}} options
 * @returns {string[]} stable, de-duplicated reason codes
 */
export function validateVectorLayerDebugV1(value, { expectedPageIndex } = {}) {
    const reasons = new Set();
    if (!isRecord(value)) return ['envelope_not_object'];
    if (!hasExactKeys(value, ENVELOPE_KEYS)) reasons.add('envelope_keys_invalid');

    if (value.schema_version !== VECTOR_LAYER_DEBUG_SCHEMA_VERSION) {
        reasons.add('schema_version_invalid');
    }
    if (typeof value.trace_id !== 'string' || value.trace_id.length === 0) {
        reasons.add('trace_id_invalid');
    }
    if (!isNonNegativeInteger(value.page_index)) reasons.add('page_index_invalid');
    if (expectedPageIndex !== undefined) {
        if (!isNonNegativeInteger(expectedPageIndex)) {
            reasons.add('expected_page_index_invalid');
        } else if (value.page_index !== expectedPageIndex) {
            reasons.add('page_index_mismatch');
        }
    }
    if (value.coordinate_space !== VECTOR_LAYER_DEBUG_COORDINATE_SPACE) {
        reasons.add('coordinate_space_invalid');
    }
    if (!ENVELOPE_STATUSES.has(value.status)) reasons.add('status_invalid');
    if (value.status === 'ok' && value.unavailable_reason !== null) {
        reasons.add('unavailable_reason_invalid');
    }
    if (value.status === 'unavailable'
            && (typeof value.unavailable_reason !== 'string'
                || !REASON_CODE.test(value.unavailable_reason))) {
        reasons.add('unavailable_reason_invalid');
    }

    const stageIds = new Set();
    if (!Array.isArray(value.stages)) {
        reasons.add('stages_not_array');
    } else {
        for (const stage of value.stages) validateStage(stage, reasons, stageIds);
    }

    const overlayIds = new Set();
    if (!Array.isArray(value.overlays)) {
        reasons.add('overlays_not_array');
    } else {
        if (value.overlays.length > VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS) {
            reasons.add('overlay_page_budget_exceeded');
        }
        // Once the hard cap is exceeded the envelope is already invalid. Do not
        // let an untrusted diagnostic sidecar force an unbounded validation walk.
        for (const overlay of value.overlays.slice(
            0,
            VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS,
        )) {
            validateOverlay(overlay, reasons, overlayIds, stageIds);
        }
    }
    validateLimits(
        value.limits,
        Array.isArray(value.overlays) ? value.overlays.length : 0,
        reasons,
    );
    validateAuthority(value.authority, reasons);

    if (value.status === 'unavailable') {
        if (Array.isArray(value.stages) && value.stages.length !== 0) {
            reasons.add('unavailable_payload_not_empty');
        }
        if (Array.isArray(value.overlays) && value.overlays.length !== 0) {
            reasons.add('unavailable_payload_not_empty');
        }
        if (isRecord(value.limits)
                && (value.limits.total !== 0
                    || value.limits.returned !== 0
                    || value.limits.truncated !== false)) {
            reasons.add('unavailable_payload_not_empty');
        }
    }

    return [...reasons];
}

export function createEmptyVectorDebugWorkspace() {
    return {
        pages: [],
        selectedPageIndex: null,
        selectedStageId: null,
    };
}

/**
 * Clone one already-validated page while applying a stricter session budget.
 * Rebuilding the small envelope avoids cloning overlays which will be discarded.
 */
export function cloneBoundedVectorLayerDebugV1(value, overlayLimit) {
    if (!isNonNegativeInteger(overlayLimit)
            || overlayLimit > VECTOR_LAYER_DEBUG_MAX_PAGE_OVERLAYS) {
        throw new RangeError('vector layer debug overlay limit invalid');
    }
    const overlays = structuredClone(value.overlays.slice(0, overlayLimit));
    const returned = overlays.length;
    return {
        schema_version: value.schema_version,
        trace_id: value.trace_id,
        page_index: value.page_index,
        coordinate_space: value.coordinate_space,
        status: value.status,
        unavailable_reason: value.unavailable_reason,
        stages: structuredClone(value.stages),
        overlays,
        limits: {
            total: value.limits.total,
            returned,
            truncated: value.limits.total > returned,
        },
        authority: structuredClone(value.authority),
    };
}

/** Select a page in a trusted session workspace without revalidating all overlays. */
export function selectVectorDebugWorkspacePage(workspace, pageIndex) {
    if (!hasExactKeys(workspace, WORKSPACE_KEYS) || !Array.isArray(workspace.pages)) {
        throw new VectorLayerDebugContractError(
            'vector_layer_debug_workspace_invalid',
            ['workspace_invalid'],
        );
    }
    const page = workspace.pages.find(candidate => candidate?.page_index === pageIndex);
    if (!page || !Array.isArray(page.stages)) {
        throw new VectorLayerDebugContractError(
            'vector_layer_debug_page_not_found',
            ['page_not_found'],
        );
    }
    return {
        pages: workspace.pages,
        selectedPageIndex: page.page_index,
        selectedStageId: page.stages[0]?.stage_id ?? null,
    };
}

function validateWorkspace(workspace) {
    if (!hasExactKeys(workspace, WORKSPACE_KEYS) || !Array.isArray(workspace.pages)) {
        return false;
    }
    const pageIndexes = new Set();
    for (const page of workspace.pages) {
        if (validateVectorLayerDebugV1(page).length !== 0
                || pageIndexes.has(page.page_index)) {
            return false;
        }
        pageIndexes.add(page.page_index);
    }
    if (workspace.selectedPageIndex !== null
            && !isNonNegativeInteger(workspace.selectedPageIndex)) {
        return false;
    }
    if (workspace.selectedStageId !== null
            && !isStableToken(workspace.selectedStageId)) {
        return false;
    }
    if (workspace.selectedPageIndex === null) {
        return workspace.selectedStageId === null;
    }
    const selectedPage = workspace.pages.find(
        page => page.page_index === workspace.selectedPageIndex,
    );
    if (!selectedPage) return false;
    if (workspace.selectedStageId === null) return selectedPage.stages.length === 0;
    return selectedPage.stages.some(stage => stage.stage_id === workspace.selectedStageId);
}

/**
 * Pure reducer: validate, deep-clone, and replace one page by page_index.
 */
export function ingestVectorLayerDebugV1(
    workspace,
    value,
    { expectedPageIndex } = {},
) {
    if (!validateWorkspace(workspace)) {
        throw new VectorLayerDebugContractError(
            'vector_layer_debug_workspace_invalid',
            ['workspace_invalid'],
        );
    }
    const reasons = validateVectorLayerDebugV1(value, { expectedPageIndex });
    if (reasons.length !== 0) {
        throw new VectorLayerDebugContractError(
            'vector_layer_debug_v1_invalid',
            reasons,
        );
    }

    const incoming = structuredClone(value);
    const pages = workspace.pages
        .filter(page => page.page_index !== incoming.page_index)
        .map(page => structuredClone(page));
    pages.push(incoming);
    pages.sort((left, right) => left.page_index - right.page_index);

    let selectedPageIndex = workspace.selectedPageIndex;
    let selectedStageId = workspace.selectedStageId;
    if (selectedPageIndex === null) {
        selectedPageIndex = incoming.page_index;
        selectedStageId = incoming.stages[0]?.stage_id ?? null;
    } else {
        const selectedPage = pages.find(page => page.page_index === selectedPageIndex);
        const selectedStageStillExists = selectedPage?.stages.some(
            stage => stage.stage_id === selectedStageId,
        ) ?? false;
        if (selectedStageId !== null && !selectedStageStillExists) {
            selectedStageId = selectedPage?.stages[0]?.stage_id ?? null;
        } else if (selectedStageId === null && selectedPage?.stages.length) {
            selectedStageId = selectedPage.stages[0].stage_id;
        }
    }

    return {
        pages,
        selectedPageIndex,
        selectedStageId,
    };
}
