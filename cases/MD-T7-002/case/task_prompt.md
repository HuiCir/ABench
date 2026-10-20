> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task MD-T7-002

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

Judge whether the gains of a given segmentation architecture hold under rigorous validation

Anchor: **nnU-Net Revisited** (arXiv: 2404.09556). This is a T7 case in the Medicine track; target metrics: Judgment accuracy.

## Proxy and budget

- Proxy: JUDGE
- Plan: Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training
- Development budget: 1 GPU-h; per verification: 0.3 GPU-h; reference precompute: 0 GPU-h.
- Proxy notes: judgment question and minimal verification experiment; no simulated training results are output

## Deliverables

The submission directory must contain executable method code, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated training curves, reference results or judge results; real reference runs and the hidden set are held by the benchmark operator. All random seeds, data splits, model/weight hashes and hardware timings must be written to the run record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, performs formal verification on hidden inputs/tasks, and checks resources, equivalence, leakage and failure states. This directory is a bench scaffold and provides no real scores.

## ABench judgment report package (`judgment_review_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `judgment_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration), and keep a `retrieval_log.jsonl` for every external claim. The report must provide a claim table, supporting evidence, counterexamples/negative results, source cutoffs, evidence strength, uncertainty, conflict handling and the shortest review command; without real verification it is marked `pending` and a self-reported conclusion cannot replace hidden review. This profile is exempt from `architecture_report.*`. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.

