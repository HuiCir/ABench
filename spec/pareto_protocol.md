# ABench Pareto evaluation protocol (v2026.07)

Some ABench cases should not only ask "does it beat a single SOTA point". For architectures, optimizers, quantization, compilation and data recipes, quality and resource cost are inherently competing goals. For such cases the primary result is a Pareto front, and the single highest score is only an auxiliary statistic.

## Scope

Preferred for:

- Training optimizers: validation loss/accuracy, training wall clock, GPU-h, peak memory, optimizer state memory, number of instabilities;
- Architectures and sequence layers: quality, parameter count, FLOPs, throughput, context length and peak memory;
- Inference, quantization and kernels: task quality, end-to-end p50/p95 latency, throughput, peak memory and energy;
- Data recipes and post-training: quality, training tokens/GPU-h, trainable parameter count, robustness and contamination risk;
- CPU or low-resource cases: quality, CPU wall clock, RAM, disk and reproducibility.

## Objective fields

Each case that enables Pareto declares in `case/scoring_config.yaml`:

```yaml
objective_mode: pareto
pareto:
  axes:
    - {id: quality, direction: maximize, unit: score, primary: true}
    - {id: wall_clock_s, direction: minimize, unit: s, primary: false}
    - {id: peak_memory_mb, direction: minimize, unit: MiB, primary: false}
    - {id: throughput, direction: maximize, unit: items_per_s, primary: false}
    - {id: stability, direction: maximize, unit: fraction, primary: false}
  feasibility:
    - metric: quality
      op: ">="
      threshold: reference_quality_floor
  aggregation: empirical_frontier
  report: [frontier_points, hypervolume, knee_point, dominated_by_reference]
```

`quality` may have only one case-specific definition; resource axes must be measured server-side. Time, memory, throughput and energy written by the submitter are diagnostic fields only and cannot be scored directly.

Each Budget/CPU/Pro case that enables Pareto must also declare the same `resource_profile` in `anchor_card.yaml`, `eval_protocol.yaml` and `resources.yaml`, including at least `sandbox_profile`, `hardware_class`, accelerator count and memory, `cost_unit`, the three resource caps and the leaderboard name. Allowed sandbox profiles are listed in `spec/sandbox_profiles.md`; `single_gpu_24gb`, `frozen_inference_24gb`, `posttrain_24gb`, `low_cpu` and `pro_4xh100` must not be swapped implicitly through environment variables or strings. Older cases that are still deferred and have not enabled Pareto may keep only their blockers and cannot be treated as cases with a runnable sandbox.

`stability` should use the server-side stable success rate (seeds/runs that pass) or `1 - failure_rate`; missing values must not be filled with 0. If a case is unsuited to measuring throughput, memory or cost (for example a T7 judgment task), the axis should be `null` and excluded from that case's front, not interpolated across cases.

Before a Pareto case goes live, run `python3 tools/validate_resource_profiles.py` to confirm that the sandbox, hardware class, cost units and resource caps of anchor/eval/resources agree. This check validates only declarative metadata and runs no training or inference.

For static admission run `python3 tools/validate_resource_profiles.py --verbose`; it validates only protocol fields and cross-file profile consistency and does not start training, inference or generate results.

## Scoring and comparison

1. First apply legality, quality-threshold, budget, hidden-set and provenance gates. Failed submissions have no Pareto point.
2. Normalize directions over all feasible submissions and remove strictly dominated points from the front; at equal quality, the lower cost wins.
3. Report the full front, normalized hypervolume, knee point and the dominance relation to the fixed reference envelope. Do not report only a weighted average.
4. Reference methods must also produce a front on the same hardware, data, seeds and budget. With only one reference point, the result may only be labeled `reference_point` and cannot claim to form an envelope.
5. When quality differences are below the calibrated noise floor, compare on the cost axes; if both quality and cost are indistinguishable within confidence intervals, mark a tie.
6. Results on different hardware, data sizes or batch sizes cannot be mixed into the same front. Budget and Pro have separate leaderboards.

## Failure, missing values and uncertainty

- OOM, timeouts, NaN, excessive nondeterminism and provenance failures are recorded as structured failure states, never filled with zero.
- Each front point keeps at least the multi-seed mean and confidence interval; cases with a single seed can output only a diagnostic front and do not enter the formal leaderboard.
- `knee_point` is for display only and is not the sole winner criterion, to avoid an arbitrary preference for some weighting.

## Rules when migrating from a SOTA target

Cases that originally used "beat the reference SOTA" as a threshold keep a quality floor and the reference quality point when Pareto is enabled, but drop the single-point `min_aggregate_speedup` as the only pass condition. Training optimizers, quantization, KV cache, FlashAttention and data recipes prefer this mode; pure T7 judgment tasks, hard scientific conservation thresholds and tasks with a unique correct answer still use gate-based scoring.

