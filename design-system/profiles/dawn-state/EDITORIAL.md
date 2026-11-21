# Editorial web context

**One reading family, with personal and publication titles kept distinct.**

Select `webContexts.editorial` in Dawn State's `tokens.json` for Matt's homepage
and publication readers. This is an explicit web context, not a replacement for
Dawn State's default Ivory/slate/orange identity. Data graphics retain their
original palette and geometry. Literal color-word fills are measured data, never
semantic UI accents.

## Type roles

| Role | Eligible content | Not eligible |
| --- | --- | --- |
| Personal | Greeting, personal monogram, personal margin headings | Publications, report prose, controls |
| Publication | Study titles on the homepage and in readers | Ordinary section headings, prose, kickers |
| Heading and body | Inter section headings, prose, UI and labels | Personal greeting or study title |
| Code | Code, receipts and literal technical strings | Prose and ordinary controls |

Publication titles share one rule: Newsreader 600, optical sizing enabled, 32 px
(2 rem), 1.25 line height and −0.04 em tracking. Keep the same rule at phone width;
wrap the title rather than introducing a smaller role. Fraunces is reserved for
personal headings. Reading prose can remain larger than homepage summaries.

## Surfaces and controls

Neutral canvas, warm-white paper, quiet decorative borders, Field green links,
focus and active controls. Active controls use paper text and retain an explicit
state label. Surfaces are flat; use the profile's 8 px control and 12 px container
radii. The report retains Onyx prose and slate analytical structure. The homepage
retains field-toned prose and its personal illustrations.

## Signal loom

The compact mark identifies Inside LFM3. It has one marked node per released
backbone layer: 13 convolution, five sliding-attention and six global-attention
layers. `signal-loom.json` owns the order and schematic geometry; the generator
resolves its colors from the editorial context. Its two strands are a metaphor,
not two discovered pathways or a causal circuit map. Use a small unboxed mark
beside an architecture note, not a tall artwork panel or filter controls.

## Generated consumers

Run `python3 design-system/scripts/build.py --embed-fonts --write-brand`.
The build emits `editorial/tokens.css`, `report-tokens.css`, `signal-loom.svg`
and a source/contrast receipt. The website's `sync-warm-precision.mjs` copies and
verifies those artifacts; its report exporter calls the same sync before rendering.
The website builds from committed snapshots without a sibling repository.
Do not edit the generated CSS or SVG independently.

Frozen report evidence keeps the design checksum from its original figure build.
The editorial receipt describes the new page treatment separately. No evidence
regeneration, new inference or experiment is required to change the surrounding UI.
