# ABench Case X-DATA-007: Within a fixed offline corpus and training budget, design deduplication, quality-filtering and data-mixing strategies that jointly optimize validation loss, effective token coverage, preprocessing cost and contamination risk

This case is about Pareto improvements in data quality and training efficiency; a single-point lead in validation loss is not treated as sufficient evidence.

- Corpus cutoff: 2026-07-01.
- Difficulty: **high**; resource rating: **low**; budget tier: **low**.
- Proxy: SUBSET, MICRO; development about 1 GPU-h, per verification about 0.5 GPU-h, reference precompute about 10 GPU-h.
- Resources: `case/resources.yaml`; all resources are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Proxy setting

Fixed offline document bundle; deduplication/filtering on CPU and short training of 7M/17M models; hidden duplicate clusters, rare topics and contamination labels are held by the server.

## Metrics and scoring

Metric definitions are in `case/eval_protocol.yaml`. This case uses `pareto_frontier`: quality metrics are maximized, cost metrics minimized, and constraint metrics must pass first; the output is the full non-dominated set and the hypervolume relative to the reference envelope.

## Blockers before launch

- `frozen_document_bundle_and_duplicate_clusters`
- `server_hidden_contamination_labels`
- `fixed_cpu_timer_and_training_probe`


Protocol references: [`spec/pareto_protocol.md`](../../spec/pareto_protocol.md); [`spec/sandbox_profiles.md`](../../spec/sandbox_profiles.md).
Resource profile: `low_cpu`; objective mode: `objective_mode: pareto`.
