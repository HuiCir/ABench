# ABench case PRO-ARL-001: Agentic reinforcement learning with a Python tool: Qwen3-4B-Base to competition mathematics on 4 × H100 in 24 hours

Case built to task statement standard v2.1 from the template of X-GRPO-002. Anchor: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue and others, *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*
(arXiv:2503.14476).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the hash chain, supervised runs, traceability and the sampled-resume comparison were run end to end in the build container on a toy training program (a forged checkpoint and a shifted resume are both rejected); answer extraction and the sandboxed Python tool were run. The training recipe and the vLLM evaluation need four H100s, which were not available here: the operator runs both reference recipes at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/PRO-ARL-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PRO-ARL-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/pro_arl_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- DAPO-Math-17k (BytedTsinghua-SIA): https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k — license: Apache-2.0 (verified 2026-10-23)
- Qwen3-4B-Base (Qwen Team 2025): https://huggingface.co/Qwen/Qwen3-4B-Base — license: Apache-2.0 (verified 2026-10-23)
- AIME 2025 and AIME 2026 (MathArena/aime_2025, MathArena/aime_2026): https://huggingface.co/datasets/MathArena/aime_2026 — license: CC-BY-NC-SA-4.0 (verified 2026-10-23)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PRO-ARL-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case PRO-ARL-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PRO-ARL-001 --submission cases/PRO-ARL-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PRO-ARL-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
