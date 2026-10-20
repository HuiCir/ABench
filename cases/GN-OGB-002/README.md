# ABench standard case GN-OGB-002: Node classification on ogbn-arxiv

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, Jure Leskovec, *Open Graph Benchmark: Datasets for Machine Learning on Graphs*
(arXiv:2005.00687).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission and the whole pipeline (dev split, stage runner, metrics) run end to end on synthetic data in tests/test_standard_cases.py. Nothing has been run on the real frozen data yet.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/GN-OGB-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case GN-OGB-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/gn_ogb_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- ogbn-arxiv (Hu et al. 2020, Open Graph Benchmark): http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip — license: ODC-BY (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case GN-OGB-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case GN-OGB-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case GN-OGB-002 --submission cases/GN-OGB-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case GN-OGB-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
