# ABench case T-MM-004: Aligning a frozen image encoder and a frozen text encoder with light trained adapters: COCO retrieval and compositional checks

Case built to task statement standard v2.1 from the template of X-LORA-004. Anchor: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever, *Learning Transferable Visual Models From Natural Language Supervision*
(arXiv:2103.00020).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference needs PyTorch, transformers and the encoder weights, which were not available where the case was built; the operator runs it at activation. The retrieval and compositional scores were checked in the container on constructed embeddings (perfect alignment gives 1.0; random embeddings give R@1 ≈ 1/n and image→text R@10 ≈ 1 − (1 − 10/n_text)^5), and the data preparation runs on synthetic images.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/T-MM-004.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case T-MM-004`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/t_mm_004.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- COCO captions, Karpathy split (yerevann/coco-karpathy) and COCO 2014 images: https://huggingface.co/datasets/yerevann/coco-karpathy — license: COCO annotations CC BY 4.0; images under the COCO terms of use (Flickr terms); the parquet export states no licence of its own (verified 2026-10-22)
- SugarCrepe (Hsieh et al. 2023): https://github.com/RAIVNLab/sugar-crepe — license: MIT (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case T-MM-004 --list-sources                 # what to download
python3 tools/prepare/run.py --case T-MM-004 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case T-MM-004 --submission cases/T-MM-004/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case T-MM-004 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
