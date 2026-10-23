# ABench case RS-GRAPH-002: Collaborative filtering on Gowalla with Recall@20 over all items

Case built to task statement standard v2.1 from the template of RS-SEQ-001. Anchor: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang, *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*
(arXiv:2002.02126).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): The reference (implicit ALS, 128 factors, 12 sweeps, NumPy) was trained and run in the build container (CPU, 2 cores; training 204 s, recommendations for all users 9 s). On the self-check split it reaches Recall@20 0.1726 (NDCG@20 0.1160; α0 and λ chosen among five settings on this split); trained on all public interactions and scored against the frozen held-out interactions as an operator calibration check, Recall@20 0.1591 and NDCG@20 0.1287, below LightGCN's published 0.1830 / 0.1554 on the same split. These are calibration checks of the case, not results; the pipeline runs end to end on synthetic data in tests/test_standard_cases.py.
- No number in the case files is a result: there are no training curves or reference scores.

## Files

- `spec/cases/RS-GRAPH-002.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case RS-GRAPH-002`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/rs_graph_002.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Gowalla split of NGCF/LightGCN, github.com/kuandeng/LightGCN Data/gowalla/{train,test}.txt: https://github.com/kuandeng/LightGCN/tree/b067ec05f16a24e0e1efe63cd82385f0c469af20/Data/gowalla — license: no licence file in the repository; the check-ins come from SNAP loc-gowalla (Cho, Myers & Leskovec 2011), published without an explicit licence — to_verify before release (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case RS-GRAPH-002 --list-sources                 # what to download
python3 tools/prepare/run.py --case RS-GRAPH-002 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case RS-GRAPH-002 --submission cases/RS-GRAPH-002/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case RS-GRAPH-002 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
