# ABench standard case BI-DNA-002: Genomic sequence classification on three Genomic Benchmarks datasets

Standard (reference) case of its track, built to task statement standard v2.1. Anchor: Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Callum Birch-Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Stephen A. Baccus, Chris Ré, *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution*
(arXiv:2306.15794).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference submission and the whole pipeline (dev split, stage runner, metrics) run end to end on synthetic data in tests/test_standard_cases.py. Nothing has been run on the real frozen data yet.
- No number in the standard-case files is a result: there are no training curves or reference scores.

## Files

- `spec/standard_cases/BI-DNA-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case BI-DNA-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/bi_dna_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Genomic Benchmarks (Grešová et al. 2023, BMC Genomic Data, doi:10.1186/s12863-023-01123-8): https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks — license: Apache-2.0 on the upstream repository; the Hugging Face mirrors state no license (verified 2026-10-19)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case BI-DNA-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case BI-DNA-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case BI-DNA-002 --submission cases/BI-DNA-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case BI-DNA-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
