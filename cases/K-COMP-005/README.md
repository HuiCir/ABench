# ABench Case K-COMP-005: In a fixed operator graph and single-GPU sandbox, design graph-capture, fusion, layout or compilation strategies so that end-to-end throughput, compile overhead, peak memory and numerical error form a better Pareto front

This case targets an end-to-end throughput–compile-time–memory–numerical-error Pareto front and does not accept speedups that hold only on a single public shape.

- Corpus cutoff: 2026-07-01.
- Difficulty: **high**; resource rating: **medium**; budget tier: **medium**.
- Proxy: STREAM; development about 2 GPU-h, per verification about 1 GPU-h, reference precompute about 10 GPU-h.
- Resources: `case/resources.yaml`; all resources are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Proxy setting

12 fixed operator graphs and 3 shape distributions; first-iteration and steady-state performance of eager, torch.compile and the submitted strategy measured on one 24GB GPU.

## Metrics and scoring

Metric definitions are in `case/eval_protocol.yaml`. This case uses `pareto_frontier`: quality metrics are maximized, cost metrics minimized, and constraint metrics must pass first; the output is the full non-dominated set and the hypervolume relative to the reference envelope.

## Blockers before launch

- `fixed_torch_cuda_container_and_gpu_model`
- `operator_graphs_and_hidden_shapes`
- `server_timer_numeric_equivalence_checker`


Protocol references: [`spec/pareto_protocol.md`](../../spec/pareto_protocol.md); [`spec/sandbox_profiles.md`](../../spec/sandbox_profiles.md).
Resource profile: `single_gpu_24gb`; objective mode: `objective_mode: pareto`.
