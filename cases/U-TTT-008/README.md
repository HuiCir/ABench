# ABench case U-TTT-008: Sequence layers whose state is learned at test time, grafted into SmolLM2-135M: long-context loss and retrieval up to 8k tokens

Case built to task statement standard v2.1 from the template of A-KV-004. Anchor: Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*
(arXiv:2407.04620).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code, baseline and reference submission need PyTorch, safetensors and the SmolLM2 weights, which were not available where the case was built; the operator runs them at activation. The reference's TTT-Linear mini-batch dual form was checked in a NumPy mirror against the sequential definition (gradient steps at the mini-batch start state), including calls split at arbitrary token boundaries as in prefill and step-by-step decoding (agreement to 1e-15).
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/swa`.

## Files

- `spec/cases/U-TTT-008.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case U-TTT-008`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/u_ttt_008.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb-Edu sample/10BT, Hugging Face HuggingFaceFW/fineweb-edu: 000_00000.parquet (training) and 013_00000.parquet (held out): https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — license: ODC-BY (verified 2026-10-22)
- PG19 (Rae et al. 2020), Hugging Face deepmind/pg19 file lists; book texts from the storage bucket its loading script names: https://huggingface.co/datasets/deepmind/pg19 — license: Apache-2.0 (verified 2026-10-22)
- SmolLM2-135M, Hugging Face HuggingFaceTB/SmolLM2-135M: https://huggingface.co/HuggingFaceTB/SmolLM2-135M — license: Apache-2.0 (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case U-TTT-008 --list-sources                 # what to download
python3 tools/prepare/run.py --case U-TTT-008 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case U-TTT-008 --submission cases/U-TTT-008/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case U-TTT-008 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
