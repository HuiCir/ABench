# ABench B-OPT-001 end-to-end run (simulated data)

> Training losses and judge outputs are simulated; behavioral probes, code diff, paper weights and all scoring logic are real computations.

| Submission | Role | L0 | SOTA | Differentiated | Aggregate speedup | M | D_beh (nearest reference) | D_mech | D | Q | X | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| example_submission | Example submission (log-gated RankGate) | ✓ | ✓ | ✓ | 1.0344× | 0.2366 | 0.1174 (cautious_adamw) | 0.6667 | 0.3988 | 0.6393 | 0.9371 | **41.67** |
| clone_muon | Negative: renamed Muon clone | ✓ | ✓ | ✗ | 1.0105× | 0.1421 | 0.0745 (muon) | 0.0 | 0.1573 | 0.9937 | 0.0 | **0.0** |
| rankgate_linear_gate | Negative: early linear-gate version (degenerates to Adafactor) | ✓ | ✗ | ✗ | 0.7761× | 0.1 | 0.0283 (adafactor) | 0.6667 | 0.3534 | 0.7999 | 0.0 | **0.0** |

Local-discovery verdicts:

- example_submission: stable_rank_gate: local discovery; power_iteration_sigma: retrieval reuse (the same use already exists in ML)
- clone_muon: —
- rankgate_linear_gate: stable_rank_gate: decorative import (no significant difference in ablation); power_iteration_sigma: retrieval reuse (the same use already exists in ML)

Detailed reports: outputs/<submission>/score_report.md
