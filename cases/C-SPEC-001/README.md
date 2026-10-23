# ABench case C-SPEC-001: Lossless speculative decoding for a 1.5B target with a 0.5B draft: end-to-end speedup on five workloads

Case built to task statement standard v2.1 from the template of C-QNT-003. Anchor: Yaniv Leviathan, Matan Kalman, Yossi Matias, *Fast Inference from Transformers via Speculative Decoding*
(arXiv:2211.17192).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference needs PyTorch, transformers and the Qwen2.5 weights, which were not available where the case was built; the operator runs it at activation. Its decoding loop (proposal, verification, cache cropping, adaptive draft length) was checked against plain greedy decoding in a mock without PyTorch: 200 random prompts with drafts agreeing 5–100% of the time all gave exactly the greedy output.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/C-SPEC-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case C-SPEC-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/c_spec_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- GSM8K (openai/gsm8k main), HumanEval (openai/openai_humaneval), CNN/DailyMail 3.0.0 (abisee/cnn_dailymail), Dolly 15k (databricks/databricks-dolly-15k): https://huggingface.co/datasets/openai/gsm8k — license: MIT, MIT, Apache-2.0, CC BY-SA 3.0 (verified 2026-10-22)
- Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct (Qwen Team 2024): https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case C-SPEC-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case C-SPEC-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case C-SPEC-001 --submission cases/C-SPEC-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case C-SPEC-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
