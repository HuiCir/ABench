# ABench case X-ROBUST-009: An intent classifier that knows when not to answer: selective risk on CLINC150 with out-of-scope, shifted and noisy queries

Case built to task statement standard v2.1 from the template of X-LORA-004. Anchor: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, *On Calibration of Modern Neural Networks*
(arXiv:1706.04599).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference (scikit-learn, CPU) was trained and run in the build container on the real public data (training 36 s). On the self-check split its area under the risk-coverage curve is 0.022; run once against the frozen held-out set as an operator calibration check it is 0.128 (in-scope accuracy 0.924, ECE 0.017, out-of-scope AUROC 0.879, decision accuracy 0.663: it answers 82% of out-of-scope and shifted queries). These are calibration checks of the case, not results.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/X-ROBUST-009.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case X-ROBUST-009`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/x_robust_009.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- CLINC150 (Larson et al. 2019), data_full.json: https://github.com/clinc/oos-eval — license: CC BY 3.0 (verified 2026-10-22)
- Banking77 test set (Casanueva et al. 2020), banking_data/test.csv: https://github.com/PolyAI-LDN/task-specific-datasets — license: CC BY 4.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case X-ROBUST-009 --list-sources                 # what to download
python3 tools/prepare/run.py --case X-ROBUST-009 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case X-ROBUST-009 --submission cases/X-ROBUST-009/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case X-ROBUST-009 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
