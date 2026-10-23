# ABench case X-DPO-006: Preference optimisation of SmolLM2-360M-Instruct from offline pairs: win rate under a hidden reward model, with drift and length costs

Case built to task statement standard v2.1 from the template of X-LORA-004. Anchor: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
(arXiv:2305.18290).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference and the evaluation need PyTorch, transformers, the SmolLM2 weights and the reward model, which were not available where the case was built; the operator runs them at activation. The length-controlled win rate (logistic fit at equal length) was checked in the container on simulated wins with a known length effect (recovered 0.552 for a true 0.550), and the data preparation runs on synthetic pairs.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/X-DPO-006.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case X-DPO-006`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/x_dpo_006.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- UltraFeedback, binarised (HuggingFaceH4/ultrafeedback_binarized): https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized — license: MIT (verified 2026-10-22)
- SmolLM2-360M-Instruct and the evaluator-only reward model Skywork-Reward-V2-Qwen3-0.6B: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case X-DPO-006 --list-sources                 # what to download
python3 tools/prepare/run.py --case X-DPO-006 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case X-DPO-006 --submission cases/X-DPO-006/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case X-DPO-006 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
