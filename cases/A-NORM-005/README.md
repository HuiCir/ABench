# ABench case A-NORM-005: A normalisation and residual scheme for small language models that stays stable at a high learning rate (two scales, two learning rates)

Case built to task statement standard v2.1 from the template of B-OPT-001. Anchor: Biao Zhang, Rico Sennrich, *Root Mean Square Layer Normalization*
(arXiv:1910.07467).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The training recipe is B-OPT-001's AdamW setting moved into the shared lm_train.py; its spike counter was unit-checked in NumPy.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/prenorm`.

## Files

- `spec/cases/A-NORM-005.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case A-NORM-005`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/a_norm_005.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE, Hugging Face kjj0/fineweb10B-gpt2 (data of the modded-nanogpt speedrun): https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case A-NORM-005 --list-sources                 # what to download
python3 tools/prepare/run.py --case A-NORM-005 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case A-NORM-005 --submission cases/A-NORM-005/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case A-NORM-005 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
