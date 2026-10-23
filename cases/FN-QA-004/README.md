# ABench case FN-QA-004: Numerical reasoning over financial reports with a frozen 1.5B model: a retrieval and program pipeline for FinQA

Case built to task statement standard v2.1 from the template of C-SPEC-001. Anchor: Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, William Yang Wang, *FinQA: A Dataset of Numerical Reasoning over Financial Data*
(arXiv:2109.00122).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the case's program executor reproduces the gold answer of all 8,281 FinQA train, dev and test questions; the reference's retriever was run on the real dev split (the gold evidence is fully inside its top 8 facts for 81.7% of questions, 89.2% of gold facts recalled) and its preparation program on the real training split (6,243 exemplars, 1 s); the pipeline, metrics and bootstrap interval were run end to end on the real dev split with a stand-in that returns the gold program for three questions in four (execution accuracy 0.751). The frozen model has not run here (no PyTorch); the operator runs the reference at activation.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/FN-QA-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case FN-QA-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/fn_qa_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FinQA (Chen et al. 2021), dataset/train.json, dev.json, test.json: https://github.com/czyssrs/FinQA — license: MIT (verified 2026-10-23)
- Qwen2.5-1.5B-Instruct (Qwen Team 2024): https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case FN-QA-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case FN-QA-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case FN-QA-004 --submission cases/FN-QA-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case FN-QA-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
