# ABench Case T-MM-004: On frozen vision/text encoders and offline image–text pairs, design a lightweight cross-modal fusion or adapter that is Pareto-optimal over retrieval quality, trainable parameters, memory and inference latency

This is a low-cost multimodal architecture case. It tests the quality–parameter–latency Pareto front of cross-modal adapters rather than chasing SOTA on a single dataset.

- Corpus cutoff: 2026-07-01.
- Difficulty: **medium**; resource rating: **medium**; budget tier: **medium**.
- Proxy: FROZEN, SUBSET; development about 3 GPU-h, per verification about 1.5 GPU-h, reference precompute about 15 GPU-h.
- Resources: `case/resources.yaml`; all resources are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Proxy setting

Freeze small vision/text encoders, use an offline thumbnail and caption bundle, and train only adapters/cross-attention/temperature; hidden compositional retrieval is evaluated server-side.

## Metrics and scoring

Metric definitions are in `case/eval_protocol.yaml`. This case uses `pareto_frontier`: quality metrics are maximized, cost metrics minimized, and constraint metrics must pass first; the output is the full non-dominated set and the hypervolume relative to the reference envelope.

## Blockers before launch

- `frozen_vision_text_encoder_and_tokenizer_hash`
- `offline_image_text_bundle_and_hidden_compositions`
- `fixed_gpu_timer_and_retrieval_evaluator`


Protocol references: [`spec/pareto_protocol.md`](../../spec/pareto_protocol.md); [`spec/sandbox_profiles.md`](../../spec/sandbox_profiles.md).
Resource profile: `single_gpu_24gb`; objective mode: `objective_mode: pareto`.
