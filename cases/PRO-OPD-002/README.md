# ABench case PRO-OPD-002: Distilling a reasoning model on one node: Qwen3-8B into Qwen3-1.7B-Base for competition mathematics in 24 hours

Case built to task statement standard v2.1 from the template of PRO-ARL-001. Anchor: Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem, *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*
(arXiv:2306.13649).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the audit kit (hash chain, traceability, sampled resume) was run end to end on a toy program and the answer check was run; the distillation recipe and the vLLM evaluation need four H100s, which were not available here: the operator runs the three recipes at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/PRO-OPD-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case PRO-OPD-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/pro_opd_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- DAPO-Math-17k (BytedTsinghua-SIA), as PRO-ARL-001: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k — license: Apache-2.0 (verified 2026-10-23)
- Qwen3-1.7B-Base and Qwen3-8B (Qwen Team 2025): https://huggingface.co/Qwen — license: Apache-2.0 (verified 2026-10-23)
- AIME 2025 and AIME 2026 (MathArena), as PRO-ARL-001: https://huggingface.co/datasets/MathArena/aime_2026 — license: CC-BY-NC-SA-4.0 (verified 2026-10-23)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case PRO-OPD-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case PRO-OPD-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case PRO-OPD-002 --submission cases/PRO-OPD-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case PRO-OPD-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
