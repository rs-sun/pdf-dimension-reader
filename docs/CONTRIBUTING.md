# Contributing

[README](../README.md) · [Architecture](ARCHITECTURE.md) · [Configuration](CONFIGURATION.md)

Use current code and canonical contracts as the technical authority. Read `AGENTS.md` and the architecture guide before changing recognition behavior.

## Data boundary

Contribute source code, explanations, and synthetic scalar examples. Do not commit engineering drawings, screenshots or crops, extracted drawing geometry, measurement reports, customer/supplier identifiers, private experiment records, original Git history, credentials, runtime logs, caches, model weights, font binaries, or full glyph-outline templates.

Tests may generate blank PDFs or invented workbooks in memory or disposable test directories. Do not store those binaries as repository fixtures. Do not copy a numeric example, comment, path, or identifier from a real drawing simply because it looks generic. Use independently invented values and inspect the complete diff.

Limited character-structure statistics and advance rules are already part of the source release. New asset tables still need provenance and content review; their use here does not authorize distributing full fonts, outlines, training data, or drawing-derived evidence.

`.gitignore` helps prevent accidental staging. It does not remove already tracked files or review file contents. Inspect every staged change, including newly generated files, before committing. Keep local runtime assets outside the checkout.

## Recognition changes

Preserve these boundaries:

- Strict mode makes zero OCR and source-drawing arrow-detector calls and has no OCR/PT/test-asset fallback.
- View boxes remain spatial grouping/clipping only; they do not establish character content or part-feature ownership.
- Unknown or ambiguous readings stay blank or review-only. Syntax must not manufacture unread characters.
- Missing assets, inconsistent identities, and invalid contracts stop recognition.
- Candidates require human confirmation before formal annotation or export. GD&T datum/reference-letter fields remain blank.
- Diagnostic settings must not alter recognition output.

The current ordinary producer is directional walking. Do not restore retired corridor/phrase/pitch readers as an import repair. A new mechanism or asset binding needs explicit design and validation; compatibility code must remain separate from the strict path.

## Tests and change descriptions

Run backend and frontend regressions using the commands in the README. Add focused tests for meaningful behavior or safety properties when changing a contract or recognition rule. Use mocked identities for asset-error paths, never real outline fixtures or local model files.

In a change description, state the behavior, relevant test results, and remaining limitations. Distinguish synthetic regression coverage, conditional glyph reading, runtime readiness, and real multipage delivery. Do not infer overall recognition accuracy from a successful local character match or a passing contract test.

## Licensing

Project source contributions use AGPL-3.0-only. Retain third-party license texts and notices. External drawing, font, model, and template rights are separate from the source license. If a contribution includes a third-party component, provide its source, version, and applicable license.
