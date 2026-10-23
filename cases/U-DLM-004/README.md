# ABench case U-DLM-004: A noise process and objective that narrow the likelihood gap between masked diffusion and autoregressive language models (two scales)

Case built to task statement standard v2.1 from the template of B-OPT-001. Anchor: Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, Volodymyr Kuleshov, *Simple and Effective Masked Diffusion Language Models*
(arXiv:2406.07524).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code, baseline and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation, together with the autoregressive reference (lm_model.LM, same sizes, data and recipe) that fills AR_REFERENCE in metrics.py. The evaluator's bound estimator was checked in a NumPy mirror under the reference's position-dependent schedule: for a context-free denoiser it recovers the exact unigram negative log-likelihood within its Monte-Carlo error (bias 0.002, standard deviation 0.012 nats per 100 windows), and the schedule passes the window check (width 0.4 ≥ 0.2).
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/mdlm`.

## Files

- `spec/cases/U-DLM-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case U-DLM-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/u_dlm_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE, Hugging Face kjj0/fineweb10B-gpt2 (data of the modded-nanogpt speedrun): https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case U-DLM-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case U-DLM-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case U-DLM-004 --submission cases/U-DLM-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case U-DLM-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
