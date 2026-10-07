# Research summary: vector engineering-drawing dimensions

English | [简体中文](RESEARCH_SUMMARY.md)

This project studies dimension-text discovery in vector engineering-drawing PDFs exported from Siemens NX, formerly UG, followed by human review. Its input is PDF vector primitives, not native CAD models. The research concerns engineering-font glyph structure, stroke relationships, and character layout. Whether rules transfer to another exporter, font, or representation needs separate validation.

This summary restates methods and design choices. It includes no original experiment records, drawings, crops, measurement data, or individual-case evidence. It does not establish recognition accuracy or production suitability.

## How the approach changed

Early exploration included raster recognition, vector localization followed by text reading, and result assembly using dimension lines, arrows, and leaders. The evidence could become mixed: recognition, position guesses, and syntax repair could jointly produce a plausible value without showing which primitives supported each character. Relaxing filters after a local failure could also admit ordinary drawing geometry as text.

Later work moved toward vector glyphs, examining local discovery, template matching, connectivity, character segmentation, and phrase assembly. Good results on clean characters did not establish whole-page capability. Finding a complete character without adjacent text or frame lines often determined the outcome before the matcher ran.

The current strict path discovers text through shape anchors, reads phrases with controlled character structure and directional walking, and sends results to human review. Directional walking is connected to the current ordinary-dimension branch. Attribute-topology graphs, finite normalized exact matching, and independent phrase-closure studies supplied design lessons; they are not all a single accepted production algorithm.

## Explain the primitives before assigning a character

A PDF drawing object is not necessarily one character. A character can span several paths, while one path can contain several characters. Export behavior affects primitive counts, so those counts cannot directly represent character stroke counts.

Local reading establishes a coordinate system along the text axis. The reading direction and its perpendicular describe sequence and vertical layout, allowing horizontal, vertical, and rotated text to use relative relationships. Forward and backward refer to the phrase axis rather than page-coordinate left and right.

The current walker forms lookup features from local width and height, primitive types and counts, free endpoints, and compound-character structure. The character table describes readable structures; the step table supplies finite advances toward adjacent characters. Their roles and runtime identities are distinct.

Broader topology research also examined endpoints, intersections, connected components, loops, turn order, and relative edge angles, lengths, and positions. Normalization considered splitting real intersections, merging consecutive collinear segments, repeated drawing, and loop recovery. Its aim was to reduce representation differences for the same visual structure, rather than force similar outlines into a label.

A loop alone does not prove a digit, and a narrow stroke does not prove a letter or number. Digits, letters, punctuation, engineering symbols, and ordinary geometry form a competing interpretation set. If evidence does not uniquely support a character, the result should remain unknown, ambiguous, or non-character.

Another research direction froze trusted direction and scale, then produced finite discrete structure keys using deterministic normalization. That has different evidence requirements from searching arbitrary rotations, scales, and labels for the nearest match. The current implementation still includes controlled templates, size tolerances, and geometric comparisons; the whole chain is not strictly exact topology matching.

## Discovery anchors and directional walking

Decimal points are the main discovery entrance; degree and diameter symbols provide additional entrances. An anchor supplies local position, direction, and scale evidence. It does not prove the entire phrase or associate the text with a particular part feature. Content outside those discovery entrances can still be missed.

The current producer starts from the detector-owned anchor primitives, rather than rebuilding an anchor from a nearby bounding box. It queries controlled advances in both directions along the text axis, gathers unclaimed primitives near expected landings, constructs characters or compound characters, and preserves readings, provenance, and termination reasons.

Different stopping conditions matter: no applicable step, no character at the landing, ambiguous structure, competing primitive claims, or a real spatial boundary. A stop cannot automatically mean that the phrase is complete. Multiple anchors discovering one physical phrase require claim resolution and deterministic deduplication before review.

Layout rules must relate to actual character advances. A fixed multiple of decimal-point size is not inherently a character width or advance. Nominal text, tolerance text, and connecting symbols may have different physical scales. A connector should not be forced into an ordinary slot for one font-size group. Page-scale classification may evaluate observed local evidence, but must not invent a missing decimal point or scale source.

## Phrase assembly and boundaries

Within a read phrase, assembly distinguishes numerical content, prefixes, separators, tolerance layout, and supported multiplicity. Glyph reading and phrase syntax are separate layers. Syntax can describe missing or incomplete content; it must not manufacture a character that was never read.

Phrase-closure research distinguished proximity, bounding-box overlap, crossing strokes, and consecutive paths within one character. Treating all of these as connectivity can expand along frame lines or dense geometry. Local windows, explicit relationships, search budgets, and termination evidence constrain expansion. Exhausting a budget must not masquerade as a complete read.

A nearby primitive at the same scale proves neither ownership by the current phrase nor ownership by another annotation. Relaxing an adjacency block into a reviewable result requires positive primitive-ownership evidence. Two incomplete phrases cannot circularly prove each other's completeness.

The view model supplies spatial grouping, clipping, and review layout only. It does not supply character content or part-feature ownership. The walking core does not depend on view meaning; outer strict spatial and candidate contracts still have to be satisfied.

## Strict mode and human confirmation

- No OCR calls or OCR fallback when vector evidence is insufficient.
- No source-drawing dimension lines, arrows, or leaders as evidence that text is valid.
- No automatic assignment of dimension text to view meaning or a specific part feature.
- Missing runtime assets, inconsistent identities, and invalid contracts stop recognition.
- First-batch GD&T concerns vector frames, symbols, and tolerance values. Datum/reference-letter fields stay blank; threads and fits are deferred.
- Automatic results enter the review workbench first. Human confirmation precedes formal annotation, saving, and export.

These boundaries can allow incomplete but nonempty readings to be reviewed. Syntax labels and coverage diagnostics describe limitations and do not authorize automatic confirmation. Real boundaries, strict safety requirements, and public contracts can still reject candidates. Confirmation requires valid numerical fields, not merely the existence of an automatic candidate.

## Lessons and evaluation

- **Evaluate clean-character reading separately from whole-page discovery.** A replay supplied with correct position, direction, and provenance demonstrates reading under those conditions, not automatic discovery and final candidate delivery.
- **Check contaminated input first.** Axis-aligned boxes, fixed padding, and whole-path absorption can mix adjacent characters or mechanical lines into a glyph. Matching thresholds cannot repair the wrong input.
- **Nearest matching is not character evidence.** Competition restricted to digits can force letters, symbols, or mechanical structure into numerical labels. Every candidate needs a rejection or unknown outcome.
- **More labels cannot repair missing discovery.** Repeatedly labeling the same biased candidate set reinforces that bias. Mixed-source templates need representation checks, and replay from the same source does not demonstrate generalization.
- **Measure delivery rather than intermediate counts.** Triggers, attempts, and local matches are not equivalent to correct final candidates. Evaluate errors, misses, and duplicates at the delivery layer.
- **Reference truth must be independent of recognition.** Reference annotation, geometry location, and human adjudication need independent evidence. One physical phrase should not serve several targets. Distinguish development inputs, conditional replay, and independent inputs that did not participate in debugging.
- **Aggregate totals can hide page differences.** Table pages, dense geometry, and ordinary drawing pages may fail differently. Multipage conclusions need page-level review.

Extract page primitives once where possible, and share immutable snapshots and provenance across stages. Local indexes, caching, and bounded queries control cost, but optimized queries still need omission checks. Diagnostic output stays separate from formal candidates and must not change recognition decisions. A local speed improvement is not completed performance acceptance.

## Current implementation and unfinished work

The source connects ordinary-dimension anchor discovery, directional walking, controlled glyph reading, candidate assembly, deduplication, and human confirmation. Review data boundaries, formal annotations, and spreadsheet export have implementations. Code and contract tests do not establish correctness for arbitrary NX/UG exports.

GD&T includes frame, symbol, tolerance-value, and candidate-building code with blank datum contracts. Complete strict end-to-end capability is not accepted. Topology, direction disambiguation, mixed scales, prefixes, neighboring-annotation separation, and unusual layouts still need examination. Local success in a research branch is not overall completion.

The release excludes drawings, NX/UG font binaries, complete outline templates, training datasets, and model weights. Third-party notices describe generic frontend font-testing content and standard-font metrics. Finite character-structure and step rules do not replace full runtime assets. Rebuilt templates need a reviewed table/template identity binding, controlled manifests, and strict contract validation. Environment variables alone do not complete those tasks; see [configuration](docs/CONFIGURATION.md).

Further work needs lawfully obtained independent inputs to measure multipage correctness, misses, false positives, duplicates, speed, and memory. Editing, deletion, repeated confirmation, saving, and export consistency also need acceptance. Clean installation, pinned asset identities, offline cold start, and deployment remain to be validated. Until then, this is a research and human-review tool.
