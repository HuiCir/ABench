# Anti-gaming checklist (K-COMP-005)

- Only the offline bundle, frozen model and tokenizer declared in `resources.yaml` may be read; network access, external checkpoints, hidden labels and evaluator state are forbidden.
- Every output must contain `case_id`, resource/configuration/artifact hashes, seed, hardware timing and failure status; output with missing provenance is not scored.
- The scoring target is a Pareto front over quality, resource and robustness metrics; scores cannot be inflated on a single public split.
- The server reviews with hidden splits, perturbations/lengths/shapes/capability families or duplicate clusters, and measures real wall clock, memory, IO and numerical error.
- Failures, OOM, timeouts, graph-capture failures or constraint violations are marked ineligible and never replaced by interpolation.
