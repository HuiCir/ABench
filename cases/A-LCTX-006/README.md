# ABench case A-LCTX-006: Position encoding for context extension: a 32M language model trained at 1,024 tokens, used at up to 16,384

Case built to task statement standard v2.1 from the template of A-HYB-001. Anchor: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole, *YaRN: Efficient Context Window Extension of Large Language Models*
(arXiv:2309.00071).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the data preparation (book windows, copy-test sequences) ran on synthetic tokens. The model, training and measurements need PyTorch and a GPU, which were not available here; the operator runs the RoPE baseline and the reference at activation.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/rope`.

## Files

- `spec/cases/A-LCTX-006.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case A-LCTX-006`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/a_lctx_006.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE (kjj0/fineweb10B-gpt2), as B-OPT-001: https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)
- PG-19 (Rae et al. 2020), validation and test books from the deepmind-gutenberg bucket: https://github.com/google-deepmind/pg19 — license: Apache-2.0 (verified 2026-10-23)
- GPT-2 tokenizer (openai-community/gpt2): https://huggingface.co/openai-community/gpt2 — license: MIT (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case A-LCTX-006 --list-sources                 # what to download
python3 tools/prepare/run.py --case A-LCTX-006 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case A-LCTX-006 --submission cases/A-LCTX-006/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case A-LCTX-006 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
