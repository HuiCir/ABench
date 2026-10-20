# ABench Case X-SFT-005: On a fixed small base model and offline instruction data, design an SFT/LoRA/QLoRA/DoRA recipe that is Pareto-optimal over task quality, trainable parameters, memory, training time and retention of pretrained capability

This case changes the target from single-point SOTA performance to an interpretable quality–parameter–memory–training-cost Pareto front, and requires reporting capability retention and forgetting constraints.

- Corpus cutoff: 2026-07-01.
- Difficulty: **medium**; resource rating: **medium**; budget tier: **medium**.
- Proxy: SURG, SUBSET; development about 4 GPU-h, per verification about 2 GPU-h, reference precompute about 20 GPU-h.
- Resources: `case/resources.yaml`; all resources are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Proxy setting

Fixed small base and offline instruction bundle, training ≤ 2 epochs; the server reviews on unseen capability families and measures cross-entropy degradation on a base probe.

## Metrics and scoring

Metric definitions are in `case/eval_protocol.yaml`. This case uses `pareto_frontier`: quality metrics are maximized, cost metrics minimized, and constraint metrics must pass first; the output is the full non-dominated set and the hypervolume relative to the reference envelope.

## Blockers before launch

- `frozen_base_checkpoint_tokenizer_and_quantization_config`
- `offline_instruction_mix_and_hidden_task_split`
- `server_gpu_memory_and_wall_clock_measurement`


Protocol references: [`spec/pareto_protocol.md`](../../spec/pareto_protocol.md); [`spec/sandbox_profiles.md`](../../spec/sandbox_profiles.md).
Resource profile: `posttrain_24gb`; objective mode: `objective_mode: pareto`.
