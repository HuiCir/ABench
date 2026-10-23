# ABench case BI-PROT-001: Protein mutation-effect prediction with a frozen ESM-2 on 30 held-out deep mutational scans (ProteinGym, 64 labels per assay)

Case built to task statement standard v2.1 from the template of X-LORA-004. Anchor: Pascal Notin, Aaron W. Kollasch, Daniel Ritter, Lood van Niekerk, Steffanie Paul, Hansen Spinner, Nathan Rollins, Ada Shaw, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Rose Orenbuch, Yarin Gal, Debora S. Marks, *ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction*
(arXiv:2312.05451).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference needs PyTorch, transformers and the ESM-2 weights, which were not available where the case was built; the operator runs it at activation. The assay selection was run in the container on the ProteinGym reference file (commit 144fe22b): 158 eligible assays, 30 held out for three test seeds with the stratified quota and no UniProt protein shared between held-out and public assays. The pipeline runs end to end on synthetic assays in tests/test_standard_cases.py with a NumPy stand-in model.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/BI-PROT-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case BI-PROT-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/bi_prot_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- ProteinGym v1.3 DMS substitutions (DMS_ProteinGym_substitutions.zip) and reference_files/DMS_substitutions.csv: https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip — license: MIT (ProteinGym); the assays come from their original publications (verified 2026-10-22)
- ESM-2 (Lin et al. 2023), Hugging Face facebook/esm2_t30_150M_UR50D and facebook/esm2_t33_650M_UR50D: https://huggingface.co/facebook/esm2_t33_650M_UR50D — license: MIT (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case BI-PROT-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case BI-PROT-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case BI-PROT-001 --submission cases/BI-PROT-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case BI-PROT-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
