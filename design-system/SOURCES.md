# Influences and adaptation decisions

Reviewed 23 October 2026; updated 3 November 2026. These are influences, not runtime dependencies. Repository defaults may change; record an installed version when integrating a vendor tool.

## Dawn State

Source: `BRAND_KIT.md` and its evidence-led editorial voice. The current profile uses Newsreader for display and Inter for headings and prose, with flat surfaces, scarce accents and a warm canvas.

Version 2 (1 November 2026) adopts OpenCode Mirror as the reference implementation: Newsreader 600 display lines and wordmark, Inter kickers in Slate Deep, 22% decorative rules, 12% tracks and the 1200 px share-card ramp. Mirror renders its display lines by synthesizing bold from Newsreader 500; the profile uses the real 600 weight, which is the closest true face and renders the same in every browser and export. Version 2 also adds Slate Mist as a context color, the voice guide, document types and the data standard.

Version 1 of Warm Precision made distribution and margins compositional defaults, made typography and strokes destination-specific, and added background-aware contrast rules. It separates personal doctrine from brand profile and project context. The original brand kit is refined in place; this package records the adaptation rationale.

Version 2.2 adds Matt's approved editorial web context: the Sunday homepage's
neutral/paper surfaces and green UI, Newsreader 600 publication titles, Fraunces
personal headings, and Inter prose/UI. The default Dawn State palette remains
unchanged. The compact signal loom preserves the Inside LFM3 report's released
layer order, with explicitly symbolic geometry. See [Editorial web](profiles/dawn-state/EDITORIAL.md)
for eligibility, generated consumer artifacts and the retained research treatment.

## Voice evidence

`VOICE.md` was derived from 2,303 human prompts in the author's local OpenCode archive (11 June to 1 November 2026, 51 projects), read with OpenCode Mirror's filter, plus approved copy on the Dawn State site, the personal homepage and Mirror. The prompts stay local; only the rules and short quoted edits are published.

## Impeccable

- https://github.com/pbakaus/impeccable
- https://github.com/pbakaus/impeccable/blob/main/skill/SKILL.src.md
- https://github.com/pbakaus/impeccable/blob/main/skill/reference/init.md
- https://github.com/pbakaus/impeccable/blob/main/skill/reference/document.md
- https://github.com/pbakaus/impeccable/blob/main/skill/reference/operate.md
- https://github.com/pbakaus/impeccable/blob/main/skill/reference/critique.md

Adopt durable product context separate from visual direction, surface-specific modes, pinned-brief precedence, appropriate familiarity in tools, and rendered review. Use critique/audit/distill/adapt/polish as a reusable working vocabulary. Warm Precision adds Explain for analytical artifacts; this is a local extension, not an Impeccable command or mode claim.

## Diagram Design

- https://github.com/cathrynlavery/diagram-design
- https://github.com/cathrynlavery/diagram-design/blob/main/skills/diagram-design/SKILL.md
- https://github.com/cathrynlavery/diagram-design/blob/main/skills/diagram-design/references/style-guide.md
- https://github.com/cathrynlavery/diagram-design/blob/main/skills/diagram-design/references/profiles.md

Adopt semantic selection, deliberate subtraction, constrained focal emphasis, structural grid discipline, portable profile selection, accessible SVG, and fidelity ledgers. Adapt serif typography, small label defaults, hairline geometry, and tinted fills to Dawn State and each destination. Split accent fill/text/stroke roles because raw Dawn State orange does not meet the vendor's all-role contrast requirement on Ivory.

## Contrast and accessibility

- https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html
- https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html

Measure WCAG 2.x relative luminance against the actual background; composite translucent colors first. Normal text needs at least 4.5:1, large text 3:1, and essential marks 3:1. Supporting labels/shape cues also preserve meaning without color. Automated results do not certify the whole artifact's accessibility.

## Fonts

Fonts are distributed under the SIL Open Font License:

- [Inter license](https://github.com/rsms/inter/blob/master/LICENSE.txt)
- [Newsreader license](https://github.com/google/fonts/blob/main/ofl/newsreader/OFL.txt)
- [Fraunces license](https://github.com/google/fonts/blob/main/ofl/fraunces/OFL.txt)
