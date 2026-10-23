# ABench case X-DATA-007: A data pipeline for a raw web pool: deduplication, filtering and mixing that minimise a fixed small model's loss on held-out educational text

Case built to task statement standard v2.1 from the template of A-NORM-005. Anchor: Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini, *Deduplicating Training Data Makes Language Models Better*
(arXiv:2107.06499).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet on the real pool: the pool and the training stage need the FineWeb files and PyTorch, which were not available where the case was built; the operator runs them at activation. The reference pipeline (CPU) was run in the container on a synthetic pool with planted near-copies, and the duplicate metrics were checked on its selection.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/X-DATA-007.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case X-DATA-007`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/x_data_007.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb (HuggingFaceFW/fineweb), first shard of six dumps: https://huggingface.co/datasets/HuggingFaceFW/fineweb — license: ODC-BY (verified 2026-10-22)
- FineWeb-Edu sample/10BT (HuggingFaceFW/fineweb-edu): https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — license: ODC-BY
- GPT-2 tokenizer (openai-community/gpt2): https://huggingface.co/openai-community/gpt2 — license: MIT (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case X-DATA-007 --list-sources                 # what to download
python3 tools/prepare/run.py --case X-DATA-007 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case X-DATA-007 --submission cases/X-DATA-007/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case X-DATA-007 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
