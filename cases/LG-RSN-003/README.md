# ABench case LG-RSN-003: Legal reasoning with a frozen 1.5B model: a pipeline for LegalBench tasks seen and unseen during development

Case built to task statement standard v2.1 from the template of FN-QA-004. Anchor: Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton et al., *LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models*
(arXiv:2308.11462).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the task selection, prompt slots, label sets and the freezing code were run on the 35 tasks' prompts and training rows from the LegalBench GitHub repository (test rows come from the Hugging Face copy, which only the operator can download); the pipeline and metrics run on synthetic tasks in tests/test_standard_cases.py with a stand-in pipeline. The frozen model has not run here (no PyTorch); the operator runs the reference at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/LG-RSN-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case LG-RSN-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/lg_rsn_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- LegalBench task data (nguha/legalbench), data/<task>/train.tsv and test.tsv for 35 tasks: https://huggingface.co/datasets/nguha/legalbench — license: CC BY 4.0 (dataset card); each selected task CC BY 4.0 or MIT per its README (verified 2026-10-23)
- LegalBench base prompts (HazyResearch/legalbench), tasks/<task>/base_prompt.txt: https://github.com/HazyResearch/legalbench — license: per task (CC BY 4.0 or MIT for the selected tasks) (verified 2026-10-23)
- Qwen2.5-1.5B-Instruct (Qwen Team 2024): https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case LG-RSN-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case LG-RSN-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case LG-RSN-003 --submission cases/LG-RSN-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case LG-RSN-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
