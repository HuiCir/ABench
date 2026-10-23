# ABench case LG-LONG-001: Classifying long court decisions: violated articles in European Court of Human Rights cases

Case built to task statement standard v2.1 from the template of TC-FT-001. Anchor: Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, Nikolaos Aletras, *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*
(arXiv:2110.00976).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission and the whole pipeline (dev split, stage runner, metrics) run end to end on synthetic data in tests/test_standard_cases.py, with the checkpoint built by the reference train.py. Nothing has been run on the real frozen data yet.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/LG-LONG-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case LG-LONG-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/lg_long_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- LexGLUE ECtHR task A (Chalkidis et al. 2022), Hugging Face coastalcph/lex_glue, config ecthr_a: https://huggingface.co/datasets/coastalcph/lex_glue — license: CC BY 4.0 (verified 2026-10-22)
- LEGAL-BERT base uncased (Chalkidis et al. 2020), Hugging Face nlpaueb/legal-bert-base-uncased: https://huggingface.co/nlpaueb/legal-bert-base-uncased — license: CC BY-SA 4.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case LG-LONG-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case LG-LONG-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case LG-LONG-001 --submission cases/LG-LONG-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case LG-LONG-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
