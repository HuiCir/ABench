# ABench case PRO-SRV-005: Serving Qwen3-8B on 4 × H100: the highest request rate under a latency SLO on replayed production traffic, with exact greedy outputs

Case built to task statement standard v2.1 from the template of PRO-SPD-004. Anchor: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica, *Efficient Memory Management for Large Language Model Serving with PagedAttention*
(arXiv:2309.06180).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the streaming protocol, the trace replay, the latency statistics and the rate ladder were run in the build container against a stand-in server; trace windows and prompts were built from synthetic traces. vLLM, Qwen3-8B and the four GPUs were not available here: the operator runs the vLLM and SGLang references at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/PRO-SRV-005.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PRO-SRV-005`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/pro_srv_005.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Azure LLM inference traces 2023 (conversation and code): https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md — license: CC-BY-4.0 (verified 2026-10-23)
- Qwen3-8B (Qwen Team 2025): https://huggingface.co/Qwen/Qwen3-8B — license: Apache-2.0 (verified 2026-10-23)
- FineWeb documents for prompt contents (kjj0/fineweb10B-gpt2) and the GPT-2 tokenizer: https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; FineWeb text ODC-BY; GPT-2 tokenizer MIT (verified 2026-10-19 (B-OPT-001))

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PRO-SRV-005 --list-sources                 # what to download
python3 tools/prepare/run.py --case PRO-SRV-005 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PRO-SRV-005 --submission cases/PRO-SRV-005/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PRO-SRV-005 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
