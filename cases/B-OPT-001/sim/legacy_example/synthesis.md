# Synthesis: LLM pretraining optimizers from Adam to 2026 (example submission)

## 1. Lineage

- **Element-wise adaptive**: Adam → AdamW (decoupled weight decay) → Adafactor (factored second moment to save memory) → LAMB (layer-wise trust ratio, large batch).
- **Sign and program search**: Lion was found by program search and uses a single momentum state; the paper notes that its gains depend on relatively large batch sizes.
- **Second-order / preconditioning**: Shampoo → SOAP (Adam in the Shampoo eigenbasis); Sophia (diagonal Hessian estimate + clipping).
- **Orthogonalization**: Muon (Newton–Schulz orthogonalized momentum); 2502.16982 adds weight decay and RMS matching and scales it up; MuonClip further handles exploding attention logits.
- **Momentum structure**: AdEMAMix (fast and slow EMAs); MARS (variance reduction); Cautious (same-sign mask); Schedule-Free (no schedule, parameter averaging).

## 2. Key conclusions and evidence strength

| Conclusion | Evidence | Strength |
|---|---|---|
| Once baselines are tuned, most "2× speedup" claims shrink | 2509.02046: the speedup of matrix optimizers over AdamW drops from about 1.4× at 0.1B to about 1.1× at 1.2B | High (multiple scales and data ratios) |
| Rankings at intermediate checkpoints are unreliable | 2509.02046: rankings flip during learning-rate decay | High |
| The best method depends on the setting (scale, batch size, training length) | Multi-setting benchmark of 2509.01440 | Medium |
| The effect of adaptivity is concentrated in a few layers | Layer-wise ablations of 2407.07972 | Medium |

## 3. The gap I see

Orthogonalization is all-or-nothing: Muon whitens every 2D parameter at every training stage with the same strength. No existing work decides the degree of whitening from the spectral structure of the momentum itself. That is the entry point of RankGate.
