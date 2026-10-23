# ABench standard case PRO-SPD-004: Pretraining speedrun: GPT-2 Medium scale to a FineWeb loss target on 4 × H100

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford and others, *Training Compute-Optimal Large Language Models*
(arXiv:2203.15556).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch >= 2.7 and four H100s, which were not available where the case was built; the operator runs them at activation. The reference is a port of a public record whose own log reports val_loss 2.918512 after 1,496 s of training on 8 x H100 (PyTorch 2.7.0); the 4-GPU port, its 6,100-step budget and the evaluation interface have not run.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/PRO-SPD-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PRO-SPD-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/pro_spd_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE, Hugging Face kjj0/fineweb10B-gpt2 (data of the modded-nanogpt speedrun): https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PRO-SPD-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case PRO-SPD-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PRO-SPD-004 --submission cases/PRO-SPD-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PRO-SPD-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
