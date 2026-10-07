# pdf-dimension-reader

English | [简体中文](README.zh-CN.md)

A research tool for dimension recognition and human review of **vector engineering-drawing PDFs exported from Siemens NX (formerly UG)**. It uses engineering-font glyph structure, stroke topology, and character layout to discover dimension candidates. Human-confirmed candidates can become annotations and Excel rows. The input is PDF; native NX/UG CAD files are not parsed.

Read the [research summary](RESEARCH_SUMMARY.en.md) for the methods and lessons, [architecture](docs/ARCHITECTURE.md) for the current code path, and [configuration guide](docs/CONFIGURATION.md) for the runtime asset requirements.

## Current status

- Ordinary dimensions have source code connecting shape-anchor discovery, directional walking, controlled glyph reading, candidate validation, and the human-review workbench.
- The strict vector path makes no OCR calls and does not use source-drawing arrows, dimension lines, or leaders to prove text content.
- The release retains recognition algorithms, limited glyph-structure statistics, and character step rules. NX/UG font binaries and complete glyph-outline templates are excluded.
- Strict GD&T integration and multipage accuracy, misses, false positives, and performance still require acceptance testing. Datum/reference-letter fields remain blank. Threads and fits are deferred.
- **This is a source release with incomplete runtime asset integration.** Recognition models and controlled outline templates are absent, and the controlled-template identity binding is intentionally empty. Strict readiness is expected to return HTTP `503` with `not_ready`. Supplying environment variables alone does not complete that binding.

The code and synthetic tests do not establish production readiness or accuracy on arbitrary NX/UG exports.

## Release contents

The repository contains recognition and review source code, limited character-structure and step tables, frontend libraries and their licenses, synthetic contract examples, and regression tests. It contains no engineering drawings, drawing crops, measurement reports, NX/UG font binaries, complete outline templates, training datasets, model weights, or original development history.

The structure tables contain finite features such as quantized width and height, primitive categories, endpoint counts, and character advances. They do not contain complete stroke-coordinate sequences.

The official PDF.js distribution embeds a generic font-loading test file with a period glyph. pdf-lib embeds public standard-font metrics and encoding tables. These upstream components are documented in [third-party notices](THIRD_PARTY_NOTICES.md); they are not NX/UG fonts or engineering-drawing data.

## Local development

Use Python 3.10 or newer. Frontend regressions have been run with Node.js 22. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
PDF_READER_EAGER_LOAD_MODELS=0 FLASK_HOST=127.0.0.1 python backend/app.py
```

Open `http://127.0.0.1:5000/`. The backend serves the frontend and API from the same origin. These commands describe the development entry point; installation of the entire recognition dependency set has not been verified across all platforms.

Recognition requires lawfully obtained local runtime assets and explicit `YOLO_VIEW_MODEL`, `YOLO_VIEW_MODEL_SHA256`, and `R33_M1_TEMPLATE_LIBRARY_PATH` configuration. Rebuilding controlled templates also requires reviewing the structure-table/template binding, updating the controlled manifest and corresponding source digests, and validating the strict contracts. An otherwise valid external template will still fail readiness with `directional_walk_template_identity_mismatch` while the release binding is empty. There is no supplied one-command asset build and acceptance workflow; see [configuration](docs/CONFIGURATION.md).

Missing or inconsistent assets must stop recognition. Do not substitute OCR, test assets, PT fallback, or unpinned assets. Legacy OCR/hybrid modules remain for compatibility and diagnostics; their presence does not enable OCR in the strict vector path. A working page or spreadsheet endpoint is not recognition acceptance evidence.

## Spreadsheet compatibility

`.xlsx` and `.xlsm` use openpyxl to read cached values. `.xls` uses python-calamine 0.8.2. Ordinary text, numbers, empty row/column positions, and numeric boolean representations are supported. Date, time, or duration cells in `.xls` cause an explicit rejection; convert or clean those cells before importing. Excel error cells are not treated as numerical error codes; empty decoded values are treated as missing.

Recalculate and save workbooks containing formulas in spreadsheet software before import. The decoder does not calculate formulas.

## Tests

The backend test dependency set omits recognition models and OCR runtimes:

```bash
python -m pip install -r backend/requirements-test.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider backend/tests
node --test frontend_tests/*.test.mjs
```

Backend tests cover dimension syntax, settings, controlled-template readiness, diagnostic data boundaries, final-consumption approval, HTTP static-file boundaries, and workbook decoding. Frontend tests cover review contracts and interaction behavior. Examples use invented scalar values, mocked loaders, in-memory blank PDFs, or generated workbooks; no real drawing, screenshot, or measurement fixture is included. Passing these tests does not measure real-drawing recognition accuracy.

## Contributing

See [the contribution guide](docs/CONTRIBUTING.md). Preserve fail-closed behavior, explicit asset identity, human confirmation, and the private-data boundary. Add synthetic tests rather than drawing-derived fixtures. Changes to recognition assets or their bindings require their own review and validation.

## License

Project source is licensed under **AGPL-3.0-only**; see [LICENSE](LICENSE). Third-party components retain their own licenses, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The source license does not grant rights to drawings, fonts, models, or template assets obtained or generated separately. This is an independent research tool with no official affiliation with or endorsement by Siemens.
