# ABench case U-DLM-005: A faster sampler for a frozen masked diffusion language model: text quality at 16, 32 and 64 network evaluations

Case built to task statement standard v2.1 from the template of V-SMP-002. Anchor: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li, *Large Language Diffusion Models*
(arXiv:2502.09992).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Partly executed: the window selection of the data preparation ran on synthetic token streams. The sampler, the denoiser wrapper and the scoring need PyTorch, transformers and the model weights, which were not available here; the operator runs the reference and the baseline at activation, and checks that the reference's perplexity at NFE 64 is close to the published MDLM figures.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/ancestral`.

## Files

- `spec/cases/U-DLM-005.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case U-DLM-005`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/u_dlm_005.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- MDLM trained on OpenWebText (kuleshov-group/mdlm-owt): https://huggingface.co/kuleshov-group/mdlm-owt — license: Apache-2.0 (verified 2026-10-23)
- GPT-2 and GPT-2 Large (openai-community): https://huggingface.co/openai-community/gpt2-large — license: MIT (verified 2026-10-23)
- FineWeb 10B pre-tokenized with GPT-2 BPE (kjj0/fineweb10B-gpt2), train shard 1 and the validation shard: https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case U-DLM-005 --list-sources                 # what to download
python3 tools/prepare/run.py --case U-DLM-005 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case U-DLM-005 --submission cases/U-DLM-005/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case U-DLM-005 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
