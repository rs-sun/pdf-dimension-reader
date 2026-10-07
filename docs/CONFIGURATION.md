# Configuration and external assets

[README](../README.md) · [Architecture](ARCHITECTURE.md)

## Development server and readiness

Use the development command in the README with `FLASK_HOST=127.0.0.1` and `PDF_READER_EAGER_LOAD_MODELS=0`. The same service serves the frontend and API. The development server is not a production deployment recipe.

`GET /health` reports process health. `GET /readyz` checks strict runtime readiness and returns HTTP `200` for `ready`, or HTTP `503` for `not_ready` with structured checks and reason codes. A healthy process or accessible review page does not mean recognition is ready.

The source release intentionally lacks model weights and controlled outline templates. It also lacks a completed external-asset binding. A non-ready result is therefore expected.

## Required recognition configuration

These environment variables are read when the backend starts. Restart after changing them.

| Variable | Requirement |
| --- | --- |
| `YOLO_VIEW_MODEL` | Explicit path to an authorized ONNX view-grouping model |
| `YOLO_VIEW_MODEL_SHA256` | SHA-256 of the exact authorized model; the model must match it |
| `R33_M1_TEMPLATE_LIBRARY_PATH` | Explicit path to a valid authorized controlled glyph-template library |
| `PDF_READER_EAGER_LOAD_MODELS` | Keep `0` for the documented strict development entry point; eager compatibility loading can make strict readiness fail |
| `FLASK_HOST` | Use `127.0.0.1` for local development |

No model, font, or template download is supplied by this repository. A view model provides spatial grouping and clipping only. PT fallback and arbitrary unpinned models are forbidden in strict mode.

The controlled template loader requires a nonempty valid library with a version and content digest. It does not search test, scratch, or research-output directories for a substitute. The library can contain full outlines at runtime, but those outlines are not part of this source release and require their own rights and data review.

## The intentionally empty template binding

Setting the variables above is necessary but insufficient. The published source retains finite character and step tables while removing their former external controlled-template identity.

The checked-in manifest is `assets/directional_walk/manifest_v1.json`. Its controlled-template identity is empty, and `backend/directional_walk_pipeline.py` has an empty `_CONTROLLED_TEMPLATE_SHA256`. Table and manifest contents are separately digest-checked by the loader.

`backend/app.py::_directional_walk_assets_check` compares the loaded template content digest with that controlled identity. Consequently, a valid external library still produces `directional_walk_template_identity_mismatch` until a compatible asset set has been rebuilt and its binding reviewed. This repository provides no automated asset-builder or acceptance command that completes that step.

Completing asset integration is development work:

1. Obtain or create assets whose use is authorized, with traceable provenance. The source license does not authorize copying vendor fonts or private drawings.
2. Establish and validate compatibility between the controlled template library and the finite character/step tables. Changing a digest does not make incompatible tables valid.
3. Review the external library identity, the manifest schema and identities, and all corresponding source pins together. If a manifest or table changes, its checked digest must change coherently. Keep outlines, model weights, input drawings, and generated evidence outside the repository.
4. Configure explicit local paths, restart, and examine every `/readyz` check. Do not disable a failed check or substitute a test loader to obtain a ready response.
5. Validate strict zero-OCR/zero-arrow behavior, candidate and human-confirmation contracts, and independent authorized multipage results. Readiness is only a runtime prerequisite, not an accuracy certificate.

The full binding/build/acceptance workflow is unfinished. This guide explains the constraint; it does not claim that an external font or model can be made compatible merely by editing a hash.

## Other readiness checks

Readiness also checks upload limits, feedback-log access, final-consumption approval, forbidden runtime components, and eager-load failures. Inspect the returned reason codes rather than relying on one asset check. Legacy engine configuration must not enable OCR or raster recognition inside strict mode.

Health, readiness, startup logs, feedback, and diagnostics can describe local runtime assets or processed input. Keep generated files and logs outside the source tree and review them before sharing. The default app entry point is for local development; production authentication, transport, dependency locking, and deployment acceptance are not supplied by these instructions.

## Testing without recognition assets

Install `backend/requirements-test.txt` and run the README test commands. These tests use synthetic values, in-memory workbooks/blank PDFs, and mocked loader responses. They do not supply production assets, enable recognition, or establish real-drawing accuracy.
