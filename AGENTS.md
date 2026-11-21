# Nullbridge Agent & Design Instructions

## 1. Warm Precision Design & Editorial Rules
For design, document, presentation, diagram, or chart work, read `design-system/PHILOSOPHY.md`, `design-system/profiles/dawn-state/tokens.json`, and the relevant file in `design-system/adapters/`. Choose the document type and shape from `design-system/DOCUMENTS.md`, write copy to `design-system/VOICE.md`, and present numbers, figures, and tables with `design-system/adapters/data.md` (use `design-system/scripts/figures.py` or the generated `warm-precision.mplstyle`).

Key invariants across all published copy, figures, and reports:
- **Zero em dashes** in prose, captions, headings, or UI copy. Use periods, commas, colons, or parentheses.
- Lead with what it is and let exact numbers, units, and sample sizes carry confidence.
- Palette: Ivory (`#FFFFF0`) canvas, Parchment (`#F5EEDC`) detail wells, Onyx (`#0F0F0F`) primary ink, Blue Slate (`#2C3E50`) structure and primary data series, Slate Mist (`#808B90`) context series, Sage Green (`#87A96B` / `#5F7F4A` text) secondary/positive, and Golden Orange (`#F39C12` with `2 px` Blue Slate outline; `#96590A` text) for the single focal element.

## 2. Repository Hygiene & Reproducibility Contract
- Run `uv run nullbridge guard` before every commit or push.
- Never commit API keys (`NDIF_API_KEY`, `HF_TOKEN`, `RUNPOD_API_KEY`, Azure SAS tokens) or local machine paths (`/` + `Users/...`).
- Store locked empirical JSON summaries in `results/<phase>/`, raw weights/traces in gitignored `runs/`, local cloud launch configs in gitignored `scripts/local/`, and claim-verified manuscript specs in `folio/`.
