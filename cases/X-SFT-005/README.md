# ABench case X-SFT-005: Supervised fine-tuning of SmolLM2-360M for verifiable instruction following, with the cost in trained parameters and forgetting

Case built to task statement standard v2.1 from the template of X-LORA-004. Anchor: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer, *QLoRA: Efficient Finetuning of Quantized LLMs*
(arXiv:2305.14314).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference needs PyTorch, transformers and the SmolLM2 weights, which were not available where the case was built; the operator runs it at activation. The constraint checkers were tested in the container on hand-written positive and negative responses for every type, and the constraint generator (2,000 draws, all fourteen types, no conflicting pair) and the data preparation run on synthetic conversations.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/X-SFT-005.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case X-SFT-005`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/x_sft_005.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- smol-smoltalk (HuggingFaceTB): https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk — license: Apache-2.0 (verified 2026-10-22)
- SmolLM2-360M (HuggingFaceTB): https://huggingface.co/HuggingFaceTB/SmolLM2-360M — license: Apache-2.0 (verified 2026-10-22)
- FineWeb-Edu sample/10BT (HuggingFaceFW/fineweb-edu, as in prepare/smol_data.py): https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — license: ODC-BY

## Activation (operator)

```bash
python3 tools/prepare/run.py --case X-SFT-005 --list-sources                 # what to download
python3 tools/prepare/run.py --case X-SFT-005 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case X-SFT-005 --submission cases/X-SFT-005/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case X-SFT-005 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
