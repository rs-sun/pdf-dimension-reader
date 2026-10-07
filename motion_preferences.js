// motion_preferences.js — 单一系统动态效果偏好源

export const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

let _mediaQueryList = null;
let _prefersReducedMotion = false;
let _initialized = false;

function _initializeReducedMotionPreference() {
    if (_initialized) return;
    _initialized = true;

    if (typeof globalThis.matchMedia !== 'function') return;

    _mediaQueryList = globalThis.matchMedia(REDUCED_MOTION_QUERY);
    _prefersReducedMotion = Boolean(_mediaQueryList.matches);

    const handleChange = event => {
        _prefersReducedMotion = Boolean(event.matches);
    };
    if (typeof _mediaQueryList.addEventListener === 'function') {
        _mediaQueryList.addEventListener('change', handleChange);
    } else if (typeof _mediaQueryList.addListener === 'function') {
        // Safari 13 及更早版本。
        _mediaQueryList.addListener(handleChange);
    }
}

export function prefersReducedMotion() {
    _initializeReducedMotionPreference();
    return _prefersReducedMotion;
}
