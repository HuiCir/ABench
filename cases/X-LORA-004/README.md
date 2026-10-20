# ABench standard case X-LORA-004: Parameter-efficient fine-tuning of roberta-base on GLUE under a parameter cap

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, *LoRA: Low-Rank Adaptation of Large Language Models*
(arXiv:2106.09685).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/X-LORA-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case X-LORA-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/x_lora_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- GLUE (Wang et al. 2019), Hugging Face nyu-mll/glue: rte, mrpc, cola, sst2: https://huggingface.co/datasets/nyu-mll/glue — license: per-task terms; the dataset card lists 'other' (research use) (verified 2026-10-19)
- roberta-base, Hugging Face FacebookAI/roberta-base: https://huggingface.co/FacebookAI/roberta-base — license: MIT (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case X-LORA-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case X-LORA-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case X-LORA-004 --submission cases/X-LORA-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case X-LORA-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
