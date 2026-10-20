# ABench Case X-ROBUST-009: Calibration, out-of-distribution detection and selective abstention for small models

On a fixed small encoder and offline intent/safety data, design training and inference schemes that jointly optimize ID task quality, probability calibration, OOD detection, risk–coverage, safety abstention and inference cost. The goal is a reviewable Pareto front; a single accuracy cannot stand in for reliability metrics.

- Corpus cutoff: 2026-07-01.
- Difficulty: **high**; resource rating: **medium**; budget tier: **medium**.
- Proxy: FROZEN, SURG, SUBSET; development about 3 GPU-h, per verification about 1.5 GPU-h, reference precompute about 15 GPU-h.
- Resources: case/resources.yaml; data, checkpoints and the shift manifest are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Task boundary

Entrants may change the calibration head, abstention head, loss, temperature/vector scaling, OOD exposure, threshold selection and lightweight training recipe, but may not read hidden splits, labels or evaluator state. The public validation set is used to choose thresholds, and the server replays on unseen topics, compositions, near-distribution OOD samples and safety-policy samples. Safety-abstention labels are offline, verifiable category labels; no online moderation or external model is called.

## Metrics and scoring

Metric definitions are in case/eval_protocol.yaml and case/metric_registry.yaml. objective_mode: pareto: quality and coverage are maximized; calibration error, risk, over-refusal rate, latency, memory and training cost are minimized; safety and ID-quality thresholds come before non-dominance. Results must report per-shift and per-seed details and the full non-dominated set.

## Blockers before launch

- frozen_base_checkpoint_tokenizer_and_label_taxonomy
- offline_public_calibration_and_hidden_shift_bundle
- server_hidden_safety_labels_and_latency_probe

Protocol references: spec/pareto_protocol.md; spec/sandbox_profiles.md.
Resource profile: posttrain_24gb; objective mode: objective_mode: pareto.
