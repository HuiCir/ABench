# ABench case A-KV-004: An attention and key-value cache structure for SmolLM2-135M that keeps quality at a third of the cache

Case built to task statement standard v2.1 from the template of X-LORA-004. Anchor: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
(arXiv:2305.13245).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code, baseline and reference submission need PyTorch, safetensors and the SmolLM2 weights, which were not available where the case was built; the operator runs them at activation. The SmolLM2 configuration (30 layers, width 576, 9 query and 3 key-value heads of 64, rope theta 100,000, tied embeddings) was read from the model's config.json at the pinned revision, giving 23,040 cache bytes per token in bf16; the reference and the baseline both reach 7,680 by construction.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/mqa`.

## Files

- `spec/cases/A-KV-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case A-KV-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/a_kv_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb-Edu sample/10BT, Hugging Face HuggingFaceFW/fineweb-edu: 000_00000.parquet (training) and 013_00000.parquet (held out): https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — license: ODC-BY (verified 2026-10-22)
- SmolLM2-135M, Hugging Face HuggingFaceTB/SmolLM2-135M: https://huggingface.co/HuggingFaceTB/SmolLM2-135M — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case A-KV-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case A-KV-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case A-KV-004 --submission cases/A-KV-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case A-KV-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
