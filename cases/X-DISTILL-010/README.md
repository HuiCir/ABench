# ABench case X-DISTILL-010: Knowledge distillation from SmolLM2-1.7B into SmolLM2-135M on a fixed token budget: held-out language-modelling loss

Case built to task statement standard v2.1 from the template of A-KV-004. Anchor: Geoffrey Hinton, Oriol Vinyals, Jeff Dean, *Distilling the Knowledge in a Neural Network*
(arXiv:1503.02531).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference needs PyTorch, transformers and the SmolLM2 weights, which were not available where the case was built; the operator runs it at activation. The token data is that of A-KV-004 and U-TTT-008 (prepare/smol_data.py); the stage and metrics run on synthetic tokens.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/X-DISTILL-010.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case X-DISTILL-010`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/x_distill_010.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb-Edu sample/10BT (prepare/smol_data.py): https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — license: ODC-BY
- SmolLM2-135M and SmolLM2-1.7B (HuggingFaceTB): https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case X-DISTILL-010 --list-sources                 # what to download
python3 tools/prepare/run.py --case X-DISTILL-010 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case X-DISTILL-010 --submission cases/X-DISTILL-010/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case X-DISTILL-010 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
