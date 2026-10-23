# ABench case IC-AUG-003: Searching a data-augmentation policy from image-transformation primitives for a fixed CIFAR-10 recipe

Case built to task statement standard v2.1 from the template of IC-SPD-001. Anchor: Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, *AutoAugment: Learning Augmentation Policies from Data*
(arXiv:1805.09501).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the training recipe needs PyTorch and a GPU, which were not available where the case was built; the operator runs it at activation. The recipe is the IC-SPD-001 reference network and optimiser; the policy check (AST scan) was run on the baseline, the reference policy and a violating example.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/flipcrop`.

## Files

- `spec/cases/IC-AUG-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case IC-AUG-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/ic_aug_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- CIFAR-10 (Krizhevsky 2009): https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz — license: no explicit license on the dataset page (to_verify), as in IC-SPD-001
- CIFAR-10.1 v6 (Recht et al. 2018, arXiv:1806.00451): https://github.com/modestyachts/CIFAR-10.1/tree/d9982abb0bfc4846b8d13a11e66b887d946205d0/datasets — license: code MIT; images follow Tiny Images terms (to_verify)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case IC-AUG-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case IC-AUG-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case IC-AUG-003 --submission cases/IC-AUG-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case IC-AUG-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
