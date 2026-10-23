# ABench case PRO-MOE-003: Pretraining a mixture-of-experts language model from scratch on 4 × H100 in 24 hours: lowest FineWeb loss at a fixed compute per token

Case built to task statement standard v2.1 from the template of PRO-SPD-004. Anchor: Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Deli Chen and others, *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*
(arXiv:2401.06066).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the audit kit (hash chain, traceability, sampled resume) was run end to end on a toy program, including a run from random initialisation; the model, the FLOPs count and the training need PyTorch and four H100s, which were not available here: the operator runs both reference recipes at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/PRO-MOE-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PRO-MOE-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/pro_moe_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE (kjj0/fineweb10B-gpt2), as PRO-SPD-004: https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19 (PRO-SPD-004))

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PRO-MOE-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case PRO-MOE-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PRO-MOE-003 --submission cases/PRO-MOE-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PRO-MOE-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
