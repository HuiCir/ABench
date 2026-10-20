# ABench generic bench contract

This file is the minimum structural convention for all cases. It describes an auditable bench skeleton and does not mean that real data, weights, reference runs or leaderboard results already exist.

## Directory and status

Every case contains at least:

```text
cases/<CASE_ID>/
├── README.md
├── case/
│   ├── anchor_card.yaml
│   ├── task_prompt.md
│   ├── eval_protocol.yaml
│   ├── resources.yaml
│   ├── scoring_config.yaml
│   ├── bench_contract.yaml
│   ├── metric_registry.yaml
│   ├── run_schema.json
│   └── anti_gaming.md
├── harness/
│   └── README.md
├── scoring/
│   └── run_case.py
└── example_submission/
```

`harness/` only loads submissions, runs minimal checks and collects reproducible artifacts; `scoring/` only reads results provided by the benchmark operator and computes metrics; `example_submission/` is an interface example and must not be treated as a real entry.

Case status is one of:

- `planned`: scheduled in the index, no contract yet.
- `spec`: task protocol finalized, directory skeleton not yet complete.
- `scaffold`: directory, entry points, resources and scoring contract exist, but real resources, reference runs, calibration or hidden evaluation are still missing.
- `built`: real resources, reference runs, hidden evaluation and reproduction checks are all complete.

Without real training, inference or judge artifacts, a case cannot be marked `built`. The skeleton stage produces no simulated leaderboards, simulated training curves or fabricated reference results.

In the index, `status` is the build state of the contract/directory in the repository; `readiness` in the anchor card is evaluation readiness, and the two must not be mixed. `readiness` takes the values:

- `deferred`: explicitly kept, but key resources are missing or the problem still needs rework;
- `contract_only`: directory and sandbox contract complete, but no usable proxy or real reference run yet;
- `proxy_ready`: proxy task and calibration negatives frozen; local preflight possible;
- `reference_ready`: real reference runs, resource hashes and metric calibration frozen;
- `hidden_ready`: hidden splits, server-side timing and reproduction audit also passed.

## Minimum protocol fields

`anchor_card.yaml` records case id, task title, anchor, cutoff date, task family, deliverables, budget, primary metric and risks; `task_prompt.md` states what information the submitter may use, the deliverables and what is forbidden; `eval_protocol.yaml` states the proxy, public/hidden layering, budget, seeds, hardware, failure states and result-reuse rules; `resources.yaml` records resource sources, licenses, versions and hashes still to be frozen; `metric_registry.yaml` lists in machine-readable form each metric's direction, unit, aggregation and Pareto role; `scoring_config.yaml` records thresholds and statistical methods. Fields marked `benchmark_defined`/`heuristic` in the registry must not be used for leaderboard scoring before real calibration.

Report delivery is no longer a per-case invention. In v2 every case declares the same `report_profile` in `case/oj.yaml` and `bench_contract.yaml`, and collects a minimal report and references according to [spec/report_profiles.yaml](report_profiles.yaml). Method cases use an offline, self-contained `report/index.html`; T7 and other `report_only` judgment tasks use a judgment report. Reports serve auditing and reproduction and cannot replace server-side metrics, resource hashes or hidden evaluation.

## Admission for real launch

Before formal evaluation, the benchmark operator must provide:

1. versions and hashes of data, models, dependency images, code and tokenizers;
2. at least one reproducible real reference run and calibration negatives;
3. a server-side sandbox, fixed-hardware timing, peak memory and OOM/timeout/nondeterminism records;
4. hidden splits, hidden random seeds and a leak audit;
5. multi-seed confidence intervals, noise floor and proxy-fidelity records;
6. failure states and artifact signature verification.

`case/run_schema.json` must be a Draft 2020-12 JSON Schema. Every real run carries at least `schema_version`, `case_id`, `provenance`, `runner_version`, `config_hash`, `resource_hash`, `artifact_hash` and an object-valued `metrics`; results missing resource/artifact hashes or of unclear provenance are never scored.

Cases that enable Pareto must also pass `python3 tools/validate_resource_profiles.py`, confirming that the sandbox, hardware class, cost units and resource caps of anchor/eval agree. `python3 tools/validate_scaffolds.py --strict` and this profile preflight check only contracts and generate no simulated results.

Cases that need real data licenses, special simulators, domain experts or large-scale hardware may stay at `scaffold` and list their blocking inputs in the README; do not write a simulator unrelated to the real task just to fill the directory.

