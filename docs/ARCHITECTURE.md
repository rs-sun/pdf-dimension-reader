# Current architecture

[README](../README.md) · [Configuration](CONFIGURATION.md) · [Contributing](CONTRIBUTING.md)

This document describes the shipped code. It does not assert recognition accuracy, completed GD&T delivery, or deployment acceptance.

## Strict ordinary-dimension path

The main product setting is `dimension_path=vector_only`:

```text
PDF vector primitives in a shared immutable page context
  -> decimal / degree / diameter shape anchors
  -> directional walk using finite structure and character-step tables
  -> controlled glyph reading, direction/fragment/claim resolution
  -> validated phrase projection, hypotheses, deduplication and gates
  -> review_candidates_v1
  -> human confirmation
  -> reviewed_vector annotations, save and Excel export
```

An anchor identifies real source primitives and provides local position, axis, and scale evidence. It does not prove a whole dimension or associate it with a part feature. Walking queries nearby unclaimed primitives at controlled character advances. Glyph structure and the authorized controlled template library supply character evidence. Termination and ambiguity remain explicit; syntax does not invent unread characters.

The producer reconciles direction, fragments, and competing primitive claims. Retained phrase, hypothesis, deduplication, and contract stages project results into review candidates. The browser validates that public contract before displaying candidates. Only valid human-confirmed values enter formal annotations and export.

The current ordinary branch uses the directional-walk producer. Older corridor/phrase/pitch readers were retired; adding their historical source back is not necessary to complete this runtime import chain.

## Code map

| Area | Entry points and responsibilities |
| --- | --- |
| HTTP and readiness | `backend/app.py`: requests, guarded static serving, runtime checks, workbook import/export |
| Orchestration | `backend/pipeline.py`: separates strict-vector and legacy compatibility paths |
| Page evidence | `backend/vector_page_context.py`: shared page primitives and integrity checks |
| Discovery | `backend/decimal_point_quad.py`, `backend/degree_polyline_strict.py`, `backend/pdf_analyzer_extract.py`: shape-anchor evidence |
| Current ordinary producer | `backend/directional_walk_pipeline.py`, `backend/directional_walk/`: structure keys, step queries, controlled reading, resolution and claims |
| Controlled runtime | `backend/r33_m1_phrase_runtime.py`, `backend/vector_digit_fixed_template.py`: explicit external template loading and retained projection interfaces |
| Candidate contract | `backend/review_candidates_contract.py`, `contracts/`, `review_candidates_v1.js`: shared schema and validation |
| Review UI | `app.js`, `app_analysis.js`, `app_interaction.js`, `app_sidebar/`: page interactions, editing and confirmation |
| Spreadsheet adapter | `backend/_inspection_workbook_decoder.py`: openpyxl/calamine decoding |

## Invariants

- Strict OCR and source-drawing arrow-detector call counts stay zero. Source arrows, dimension lines, and leaders do not establish recognition content.
- View boxes provide spatial grouping and clipping only. They do not determine characters, view meaning, or part-feature ownership.
- Unknown and ambiguous content stays blank or review-only. Automatic output does not authorize final consumption.
- Missing assets, digest mismatches, and invalid contracts fail closed. Test assets and compatibility engines cannot supply a fallback.
- GD&T datum/reference letters remain blank. Threads and fits are outside the current first-batch scope.
- Diagnostics must not alter recognition decisions. Runtime evidence belongs outside a source release.

## GD&T and legacy paths

Vector GD&T frame, symbol, and tolerance building blocks are present. The default retained GD&T reader is fail-closed because its former reader was retired. Complete strict GD&T candidate delivery and acceptance remain unfinished; the existence of those modules is not a completion claim.

OCR, raster, and hybrid helpers remain for separate compatibility or diagnostic paths. They are not part of strict-vector recognition and must not be used to bypass its readiness checks.

## Assets and validation

Checked-in directional-walk tables contain limited structural statistics and advances. Full glyph-outline templates and model weights are external. The public release intentionally has an empty controlled-template binding; see [configuration](CONFIGURATION.md) before attempting recognition.

Synthetic contract and security tests establish particular software properties. Independent authorized inputs are still needed for multipage accuracy, misses, false positives, duplicates, performance, editing/export consistency, installation, and cold-start acceptance.
