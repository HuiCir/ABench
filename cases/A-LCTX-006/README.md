# ABench Case A-LCTX-006: Under fixed training and inference budgets, design a position encoding, state cache or context-compression scheme so that long-context recall, short-context loss, memory and throughput form a better Pareto front

This case is currently marked `deferred_out_of_scope_v1` and is kept only for direction auditing; while resources are tight in this round it is not in the reference plan. When enabled in future, it will evaluate only reproducible long-context architecture/training changes without requiring a single SOTA point; final ranking is by Pareto dominance over quality, memory and throughput.

- Corpus cutoff: 2026-07-01.
- Difficulty: **high**; resource rating: **medium**; budget tier: **medium**.
- Proxy: SYN, MICRO, INFER; development about 4 GPU-h, per verification about 2 GPU-h, reference precompute about 20 GPU-h.
- Resources: `case/resources.yaml`; all resources are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Proxy setting

Fixed offline data and a small decoder-only model; maximum training context 4k, hidden inference lengths 8k/16k/32k; all submissions use the same tokenizer, data and base checkpoint.

## Metrics and scoring

Metric definitions are in `case/eval_protocol.yaml`. This case uses `pareto_frontier`: quality metrics are maximized, cost metrics minimized, and constraint metrics must pass first; the output is the full non-dominated set and the hypervolume relative to the reference envelope.

## Blockers before launch

- `frozen_tokenizer_and_base_model_hash`
- `offline_long_context_probe_and_hidden_lengths`
- `fixed_gpu_memory_and_throughput_timer`


Protocol references: [`spec/pareto_protocol.md`](../../spec/pareto_protocol.md); [`spec/sandbox_profiles.md`](../../spec/sandbox_profiles.md).
Resource profile: `single_gpu_24gb`; objective mode: `objective_mode: pareto`.
