# ABench case PH-MLIP-003: A machine-learned interatomic potential for small molecules from 1,000 DFT configurations: force accuracy and 10 ps of stable dynamics (rMD17)

Case built to task statement standard v2.1 from the template of GN-OGB-002. Anchor: Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, Gábor Csányi, *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*
(arXiv:2206.07697).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not run on the real data: rMD17 could not be downloaded where the case was built (the operator does this at activation). The reference's analytic forces were checked against finite differences of its energy (agreement to 1e-9), and the reference and the whole pipeline (bond detection, Langevin dynamics, stability check, metrics) run end to end on a synthetic harmonic molecule in tests/test_standard_cases.py; there the reference reaches a force error of about 1% of the mean force magnitude and stays stable for 10 ps (without its quadratic pair terms its trajectories broke within 40 fs). The integrator and stability check were also checked on a harmonic triatomic (stable for 1 ps) and its sign-flipped version (breaks within 10 fs).
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/PH-MLIP-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PH-MLIP-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/ph_mlip_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Revised MD17 dataset (rMD17), figshare article 12672038 version 4: rmd17_<molecule>.npz and the split files index_train_01.csv / index_test_01.csv: https://doi.org/10.6084/m9.figshare.12672038.v4 — license: CC0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PH-MLIP-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case PH-MLIP-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PH-MLIP-003 --submission cases/PH-MLIP-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PH-MLIP-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
