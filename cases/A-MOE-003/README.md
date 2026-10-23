# ABench case A-MOE-003: Expert capacity and routing for a small mixture-of-experts language model at fixed active and total parameters

Case built to task statement standard v2.1 from the template of B-OPT-001. Anchor: William Fedus, Barret Zoph, Noam Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*
(arXiv:2101.03961).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code, baseline and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The parameter and compute budgets of the reference (16 experts of width 512, top-4) equal those of the baseline by construction.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/top2`.

## Files

- `spec/cases/A-MOE-003.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case A-MOE-003`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/a_moe_003.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE, Hugging Face kjj0/fineweb10B-gpt2 (data of the modded-nanogpt speedrun): https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case A-MOE-003 --list-sources                 # what to download
python3 tools/prepare/run.py --case A-MOE-003 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case A-MOE-003 --submission cases/A-MOE-003/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case A-MOE-003 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
