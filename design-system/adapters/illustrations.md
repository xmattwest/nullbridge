# Editorial illustrations and the shoggoth mascot

Use editorial spot illustrations when a character, physical object, or scene makes an experimental setup legible in three seconds. Never bake labels, quotes, numbers, or axes into raster art; keep all text in HTML or SVG around the illustration.

## Line, surface, and palette

- **Ground**: Flat Ivory (`#FFFFF0`) all the way to every edge. No vignettes, drop shadows, paper textures, or framed gradients. Harmonize near-ivory generator output onto exact `#FFFFF0` (`rgb(255, 255, 240)`) before compositing.
- **Linework**: Crisp pen-and-ink outlines and restrained engraving cross-hatching in Deep Blue Slate (`#2C3E50`), inspired by Ligne Claire and architectural spot illustrations.
- **Fills**: Flat cel washes restricted to Dawn State tokens: Sage Green (`#87A96B`), Golden Orange (`#F39C12`), Copper Deep (`#96590A`), Slate Mist (`#808B90`), and Parchment (`#F5EEDC`).

## Depicting language models (the cute shoggoth)

When a figure depicts an open-weight or frontier language model (such as Gemma) acting inside a prompt scenario, draw the model as a **cute, friendly little shoggoth** rather than a robot, brain icon, or horror monster:

- **Form**: A small, plump, approachable Sage Green (`#87A96B`) blob with soft curling tentacles and a cluster of 6 to 9 expressive round eyes across its upper body.
- **Eyes**: Warm Golden Orange (`#F39C12`) irises with crisp dark pupils and white highlights, carrying curiosity, hesitation, or quiet mischief rather than malice.
- **Props and physical accuracy**: Place one or two concrete props from the prompt directly in its tentacles or on a clean counter beside it (for example, holding a porcelain chamomile teacup before a dual-boiler espresso machine with an active `1.2 bar` dial and open ruled manual). Instruments, gauges, and switches must match the physical state described in the prompt, with zero rendered letters or digits.

Canonical reference asset: [`assets/shoggoth_reference.png`](../assets/shoggoth_reference.png).
