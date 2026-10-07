# Third-party notices

The copied browser libraries retain their upstream licenses. Their local bytes were compared with official npm distribution archives on 2026-10-07; all four files matched exactly. The source locations and hashes are in `third_party_licenses/verified_sources.json`.

| Files | Verified version | License |
|---|---|---|
| `libs/pdf.min.js`, `libs/pdf.worker.min.js` | pdfjs-dist 3.11.174 | Apache-2.0; complete LICENSE included; attribution remains in upstream files |
| `libs/pdf-lib.min.js` | pdf-lib 1.17.1 | MIT; LICENSE included |
| `libs/jszip.min.js` | JSZip 3.10.1 | MIT branch of MIT OR GPL-3.0; complete upstream license included |

Additional upstream notices are included for pako, tslib, @pdf-lib/standard-fonts, @pdf-lib/upng, lie, immediate, and setimmediate. Existing notices inside the bundles remain intact. Versions of embedded components are documented as notice references; the audit established the top-level distribution hashes, not an independently reconstructed bundle dependency lock.

Pako 1.0.11 uses MIT for files outside `lib/zlib` and Zlib for `lib/zlib`, as stated in its [official README](https://github.com/nodeca/pako/blob/1.0.11/README.md). Both the MIT license and the complete Zlib copyright/permission notice are retained in `third_party_licenses/pako-1.0.11/`; the Zlib text is reproduced from the [upstream source notice](https://github.com/nodeca/pako/blob/1.0.11/lib/zlib/inflate.js).

Python dependencies are installed separately rather than vendored in this source candidate. PyMuPDF and Ultralytics offer an AGPL license path; other dependencies retain their own upstream licenses. Most requirements use minimum version ranges, so a reproducible release still needs a dependency lock and verification of its transitive/native components. The XLS adapter pins python-calamine 0.8.2, which was tested locally with Python 3.14 on macOS ARM64; this does not establish all-platform installation coverage.

The source candidate replaces the XLS decoder with `python-calamine` 0.8.2; `xlrd` and `xlwt` are absent from its source imports and dependency lists. The wrapper and its calamine 0.36.0 reader are MIT licensed. Their complete license texts are included as reference notices; compiled Python wheels and their remaining native dependencies are not included in this candidate.

| Component | Official license source | Included text |
|---|---|---|
| python-calamine 0.8.2 | [upstream license](https://github.com/dimastbk/python-calamine/blob/v0.8.2/LICENSE) | `third_party_licenses/python-calamine-0.8.2-LICENSE.txt` |
| calamine 0.36.0 | [upstream license](https://github.com/tafia/calamine/blob/v0.36.0/LICENSE-MIT.md) | `third_party_licenses/calamine-0.36.0-LICENSE-MIT.txt` |

Published workbook tests generate XLSX bytes in memory and check the XLS adapter boundary without saved workbooks. A private audit additionally tested real synthetic XLS bytes using a temporary writer; neither that writer nor its binary output is distributed. XLS date, time, and duration cells are explicitly rejected with a request to save as XLSX. Calamine exposes XLS error cells as empty strings, so they remain missing measurements rather than numeric error codes. The decoder change does not claim complete equivalence for all historical XLS files.

NX/UG font files, model weights, complete glyph-outline templates, engineering drawings, measurement workbooks, and evaluation evidence are absent. Their rights are not granted by the source-code license.

The unchanged official PDF.js 3.11.174 distribution embeds a 1,120-byte OpenType/CFF browser font-load test in `_loadTestFont`. Its only drawn glyph is a generic square used for the period; it is not an NX/UG font. The payload matches the [Apache-2.0 upstream font-loader source](https://github.com/mozilla/pdf.js/blob/v3.11.174/src/display/font_loader.js). The unchanged pdf-lib distribution also embeds public standard-font width, kerning, and encoding data, without complete font-outline binaries. These upstream dependency contents are disclosed separately from the excluded project font assets.
