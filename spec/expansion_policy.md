# ABench executable expansion scope (v2026.10)

This round of expansion accepts only model-architecture, training-optimization, post-training and inference-efficiency problems that can be verified in an offline sandbox. Tasks may depend on fixed snapshots of documents, images, audio or small code data, but not on browsers, desktop control, online APIs, special robot simulators or irreproducible external services.

Resources are currently tight, so the following directions are not in this round's executable batch: long context/long-term memory, video understanding or generation, large repository-level software engineering, OSWorld/WebArena/Computer Use, and tasks that need multiple machines or special simulators. Existing skeletons may be kept for direction auditing, but they should be marked deferred or out of budget and must not consume this round's reference budget.

## Priority additions

- **Classic post-training**: SFT and PEFT, the DPO family, knowledge distillation; RM/RLHF/RLAIF, IA3 or rejection sampling are added only when shared data splits, stability metrics and a single-GPU budget are clear.
- **Architectures and sequence layers**: position encodings, SSMs, MoE, hybrid attention, KV/cache structures and test-time compute; long-context extrapolation is not an acceptance target this round.
- **Inference and systems**: AWQ/GPTQ/SmoothQuant, FP8/FP4, pruning/distillation, torch.compile/Inductor, Triton/FlashAttention, speculative decoding, KV paging and single-GPU serving schedulers.
- **Minimal audio**: CTC/classification/lightweight adapters on a frozen audio encoder, with fixed short audio clips and offline features; no real-time streaming, TTS or large-scale speech pretraining.
- **Low-cost code-model tasks**: limited to fixed small Python repositories, offline compilation, unit tests and hidden defect splits; no large repositories or computer operation.
- **Data engineering and reliability**: deduplication/contamination detection, data-mixing ratios, calibration, OOD, selective abstention; each direction keeps at most one case that produces a clear training or inference improvement signal.

## Minimum metadata for each new case

Each case must give, in `anchor_card.yaml` or `bench_contract.yaml`:

```yaml
readiness: contract_only
scope: active_v1              # or deferred_out_of_scope_v1
task_family: architecture|training|post_training|inference|data|multimodal|code
budget_tier: low|medium|high
difficulty: low|medium|high
sandbox_profile: low_cpu|single_gpu_24gb|frozen_inference_24gb|posttrain_24gb|pro_4xh100
objective_mode: pareto|gate|judge
resource_profile:
  sandbox_profile: single_gpu_24gb
  hardware_class: single_gpu_24gb
  cost_unit: GPU-h
  limits:
    agent_dev: {value: 2, unit: GPU-h}
    per_submission_verify: {value: 0.5, unit: GPU-h}
    reference_precompute: {value: 10, unit: GPU-h}
resource_budget:
  agent_dev_gpu_h: 2
  per_submission_gpu_h: 0.5
  reference_precompute_gpu_h: 10
```

A budget of 0 can only mean "not yet estimated", never zero resource cost. Dependencies on shared reference artifacts must be listed explicitly in `dependencies`; `ref: 0` must not imply a dependency. New cases must state `scope: active_v1` or `scope: deferred_out_of_scope_v1`; deferred cases must not enter this round's reference plan.

## Pareto migration principle

Architecture, optimizer, quantization, compilation, KV cache, sampler, data recipe, audio adaptation and distillation cases prefer `objective_mode: pareto`. Quality still has a minimum threshold, but the final comparison considers quality, throughput, wall clock, peak memory, GPU-h, parameter count and stability together. Pure judgment tasks, hard conservation thresholds and tasks with a unique correct answer keep gate/judge.

## Launch order

First freeze offline resources and the reference pool, then complete a real smoke run, server-side timing, hidden splits, proxy-fidelity calibration and the anti-gaming audit; only after passing can a case move from `contract_only` to `proxy_ready` or `reference_ready`. New cases generate no simulated results to fill directories. The current batch prioritizes the audio, classic post-training and system quantization/compilation cases, then wires real runners into older skeletons.
