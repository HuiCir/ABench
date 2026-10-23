# ABench case U-BYTE-007: A tokenizer-free byte-level language model with dynamic patching, scored in bits per byte at equal compute and under character noise

Case built to task statement standard v2.1 from the template of B-OPT-001. Anchor: Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer, *Byte Latent Transformer: Patches Scale Better Than Tokens*
(arXiv:2412.09871).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the stage code, baseline and reference submission need PyTorch and a GPU, which were not available where the case was built; the operator runs them at activation. The token-to-byte decoding of the prepare module (GPT-2 byte table, separator byte 0) and the noise function were checked in the container on test strings.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/static4`.

## Files

- `spec/cases/U-BYTE-007.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case U-BYTE-007`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/u_byte_007.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- FineWeb 10B pre-tokenized with GPT-2 BPE, Hugging Face kjj0/fineweb10B-gpt2 (data of the modded-nanogpt speedrun): https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 — license: MIT on the token repository; the underlying FineWeb text is ODC-BY (verified 2026-10-19)
- GPT-2 byte-level vocabulary, Hugging Face openai-community/gpt2 vocab.json (used only to decode tokens to bytes): https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/vocab.json — license: MIT (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case U-BYTE-007 --list-sources                 # what to download
python3 tools/prepare/run.py --case U-BYTE-007 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case U-BYTE-007 --submission cases/U-BYTE-007/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case U-BYTE-007 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
