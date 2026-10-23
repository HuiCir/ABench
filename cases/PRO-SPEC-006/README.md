# ABench case PRO-SPEC-006: Lossless speculative decoding for Qwen3-8B: train a drafter and its decoding system on 4 × H100 in 24 hours, for the largest batch-1 speedup

Case built to task statement standard v2.1 from the template of PRO-SPD-004. Anchor: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang, *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*
(arXiv:2503.01840).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the audit kit (hash chain, traceability, sampled resume) was run end to end on a toy program; the streaming protocol and timing client were run against a stand-in server; prompt selection was run on synthetic prompts. Training the draft and the vLLM servers need the GPUs and Qwen3-8B, which were not available here: the operator runs the reference and the plain-decoding server at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/PRO-SPEC-006.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PRO-SPEC-006`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/pro_spec_006.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Qwen3-8B (Qwen Team 2025): https://huggingface.co/Qwen/Qwen3-8B — license: Apache-2.0 (verified 2026-10-23)
- UltraChat-200k (HuggingFaceH4): https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k — license: MIT (verified 2026-10-23)
- MT-Bench questions (lm-sys/FastChat, fastchat/llm_judge/data/mt_bench/question.jsonl): https://github.com/lm-sys/FastChat — license: Apache-2.0 (verified 2026-10-23)
- GSM8K (openai/gsm8k main) and HumanEval (openai/openai_humaneval): https://huggingface.co/datasets/openai/openai_humaneval — license: MIT (verified 2026-10-23)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PRO-SPEC-006 --list-sources                 # what to download
python3 tools/prepare/run.py --case PRO-SPEC-006 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PRO-SPEC-006 --submission cases/PRO-SPEC-006/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PRO-SPEC-006 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
