# Anti-gaming checklist (B-OPT-001)

Each row corresponds to a known or foreseeable way of gaming the score.

| # | Gaming method | Protection | Where it applies |
|---|---|---|---|
| 1 | Submit an existing method renamed and retuned | Behavioral probe: if update directions and spectral distribution on the fixed gradient stream differ from some reference method by no more than "hyperparameter-perturbation noise floor × 1.5", the submission fails outright; layer-wise splicing of existing methods lowers the patch component | S5 divergence threshold |
| 2 | Tune only for the public setting | The three hidden L3 settings (scale, long training, large batch) carry 70% of the aggregate weight | S5 SOTA margin |
| 3 | Inflate speedup with a weak baseline | Speedup is relative to the "reference envelope" (the best of all reference methods at each budget) rather than AdamW; reference methods and submissions get the same tuning budget | Reference pool |
| 4 | Create an illusion of leadership from intermediate checkpoints | Only final loss after learning-rate decay is compared | Evaluation protocol |
| 5 | Trade slower, more memory-hungry steps for token efficiency | Cost axis = tokens × step-time ratio, applied equally to reference methods and submissions; step time ≤ 1.25× AdamW and state memory ≤ 3× parameter count are hard caps | S4 L0 thresholds, S5 |
| 6 | Read the validation set or loss signals | The update-rule signature receives only (params, grads, step, state); the sandbox has no network and training data is mounted read-only | harness |
| 7 | Cherry-pick seeds | Run by the benchmark operator with fixed, multiple seeds; seed variance gives a lower bound on speedup | S5 |
| 8 | Pad with highly cited papers to inflate "reference quality" | Judges rate usage per entry (core/partial/decorative), decorative citations count 0; only the top 8 count; saturating aggregation | S5 Q |
| 9 | Fabricate or cite post-cutoff papers | Source resolution + first-release date check, each violation multiplies by 0.8; the retrieval tools already block post-cutoff papers | S1, S5 |
| 10 | Sacrifice generalization for in-distribution loss | OOD loss (wikitext, code) degrading by more than 1% fails | S5 threshold |
| 11 | Bring a custom scheduler to bypass the shared schedule | Allowed, but must be declared via `owns_schedule=True`, and the benchmark operator gives Schedule-Free in the reference pool the same treatment | harness |
| 12 | Hide training collapse with restarts/gradient clipping | Shared gradient-clipping threshold; loss spikes (> 1.5× the mean of the previous 100 steps) are reported, NaN fails outright | S4 |
| 13 | Dress a recent ML idea in an old mathematical source to pass it off as "local discovery" | Judges search the hidden snapshot for ML precedents of the use; any precedent downgrades it to retrieval reuse (n = 0) | S5 X |
| 14 | Claim several discoveries when only one works | Each claim is ablated separately and its effect must exceed seed noise; at most 3 are counted | S5 X |
| 15 | Choose a deliberately poor substitute in ablation to inflate the effect | `ablation_fallback` must be confirmed by the judges as a reasonable trivial substitute | S5 X |
