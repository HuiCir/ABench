# ABench case FN-LTSF-005: A long-horizon forecasting architecture that beats a tuned linear model: five multivariate series, horizons 96 to 720

Case built to task statement standard v2.1 from the template of DS-PIPE-002. Anchor: Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu, *Are Transformers Effective for Time Series Forecasting?*
(arXiv:2205.13504).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The whole case ran on the real data in the build container (NumPy reference, CPU). Freezing fits the linear reference: its test MSE at horizon 96 is 0.354 (ETTh1), 0.239 (ETTh2), 0.294 (ETTm1), 0.156 (ETTm2) and 0.137 (electricity), in line with the published linear baselines. The reference submission (a lookback and penalty search over the same family, refitted on training and validation rows; training 4 minutes) was then run once against the frozen test windows as an operator calibration check: MSE ratio 0.999 (95% interval 0.993–1.007), better than the reference on 13 of 20 pairs, worst ratio 1.11 (ETTh2, 336); on the self-check split it scores 0.965. These are calibration checks of the case, not results.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/FN-LTSF-005.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case FN-LTSF-005`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/fn_ltsf_005.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- ETT-small (ETTh1, ETTh2, ETTm1, ETTm2), Zhou et al. 2021: https://github.com/zhouhaoyi/ETDataset — license: CC BY-ND 4.0 (verified 2026-10-22)
- Electricity, 321 clients hourly (Lai et al. 2018) from UCI ElectricityLoadDiagrams20112014: https://github.com/laiguokun/multivariate-time-series-data — license: CC BY 4.0 (UCI ElectricityLoadDiagrams20112014); the repository states no licence of its own (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case FN-LTSF-005 --list-sources                 # what to download
python3 tools/prepare/run.py --case FN-LTSF-005 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case FN-LTSF-005 --submission cases/FN-LTSF-005/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case FN-LTSF-005 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
