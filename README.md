# Nullbridge

**Nullbridge** investigates algebraic, zero-runtime-cost alignment inside looped and recurrent transformers by projecting unwanted latent subspaces directly out of cross-pass communication bridges:

$$R_{\text{safe}} = R_{\text{proc}}\left(I - Q Q^\top\right)$$

Whereas standard post-training alignment (SFT, DPO, safety LoRAs) primarily modifies late verbalization layers while leaving intermediate latent biases intact to compound across silent recurrent passes, **Nullbridge** places undesirable directions (sycophancy, covert bias, sandbagging, multi-hop jailbreak indirection, reward-hacking heuristics) in the exact null space of the recurrent loop bridge or cross-pass message gate.

## Repository Layout

- `src/nullbridge/`: Core subspace extraction, Orthogonal Procrustes + null-space bridge surgery, LoRA/surface-filter baselines, and NDIF/`nnsight` probes for native looped transformers (`Inkling 2 Mini`, `TML-d24-varloop-research`).
- `results/`: Immutable locked JSON result artifacts and reproducibility manifests.
- `folio/`: Claim-verified manuscript specifications compiled via [Folio](https://github.com/xmattwest/folio).
- `runs/` *(gitignored)*: Raw activation traces, adapter checkpoints, and compiled publication builds.

## Quickstart

```bash
uv sync --extra dev
uv run nullbridge guard
```

## License

MIT
