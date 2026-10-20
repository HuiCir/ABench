# ABench

ABench asks an agent to start from a classic paper, retrieve and synthesise prior work from before a knowledge cutoff
(2026-07-01), and design a method that improves on it. Evaluation works like an online judge: the agent sees only the
task statement, the interface, report templates and public training data; the test set never enters the sandbox.

- **98 cases in 20 tracks** (`index.yaml`); every track has at least one **standard case** built to task statement
  v2.1 (`spec/task_statement_standard.md`, `spec/standard_cases/`).
- **The agent trains, the evaluator only evaluates.** Checkpoint cases submit model code, `checkpoint/` and the
  `train.py` that produced it; the evaluator only loads, infers and measures. Where the training program itself is
  measured (time to target, optimizers) the evaluator trains from scratch; kernels and samplers are only measured.
- **Workflow** required by every standard task: retrieve (offline arXiv search, `tools/paper_search/`), synthesise
  (report Section 2, Related Work), design (changes relative to the anchor with code locations), develop and verify
  with `selfcheck.py` (the evaluator's own pipeline on a dev split), submit.

## Layout

| Path | Contents |
|---|---|
| `index.yaml` | tracks and cases |
| `cases/<ID>/` | `case/` (task statement, `oj.yaml`), `harness/` (interface, stage pipeline, metrics, self-check), `example_submission/` |
| `spec/` | task statement standard, OJ contract, report profiles, standard-case specs and reference reports |
| `tools/` | judge (`oj_eval.py`), sandbox runner, data preparation (`prepare/`), builders and validators, paper search |
| `sandbox/` | report templates, the local launch script for TC-FT-001 and the plumbing agent |
| `site/` | static demo site (`python3 tools/build_site_data.py` regenerates its data) |
| `tests/` | unit and end-to-end tests (synthetic data only) |

## Quick start

```bash
python3 -m pip install --user numpy pyyaml scipy scikit-learn
python3 -m unittest discover -s tests                 # ~1 minute, no data needed
python3 tools/validate_task_statement.py              # conformance of every task statement
bash sandbox/run_local_claude_tc_ft_001.sh preflight  # local agent run of a CPU case (see the script header)
```

Data is not in the repository. Each standard case lists its verified sources (`python3 tools/prepare/run.py --case ID
--list-sources`); the operator downloads and freezes them, which writes the public files, the encrypted held-out split
and `case/data_lock.yaml`. Hidden-set keys stay outside the repository.

No standard case is activated yet: all wait for the operator's frozen data and a calibration run, so no result is
scorable. Cases needing PyTorch or a GPU have not been executed (see `spec/standard_cases/README.md`).
