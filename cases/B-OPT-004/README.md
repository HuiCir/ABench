# ABench case B-OPT-004: An update rule that keeps its efficiency as the batch grows (fixed token budget, three batch sizes)

Case built to task statement standard v2.1 from the template of B-OPT-001. Anchor: Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh, *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes*
(arXiv:1904.00962).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The training loop is B-OPT-001's with gradient accumulation added; the reference LAMB rule was run on NumPy arrays through optim_api.Ops.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/adamw`.

## Files

- `spec/cases/B-OPT-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case B-OPT-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/b_opt_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE, Hugging Face kjj0/fineweb10B-gpt2 (data of the modded-nanogpt speedrun): https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case B-OPT-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case B-OPT-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case B-OPT-004 --submission cases/B-OPT-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case B-OPT-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
