# ABench case X-PPO-001: A policy-optimisation algorithm for small continuous-control tasks at a fixed sample budget, trained by the evaluator over five seeds

Case built to task statement standard v2.1 from the template of E-ACT-001. Anchor: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, *Proximal Policy Optimization Algorithms*
(arXiv:1707.06347).

## Status

- Judge: `harness` (shared stage protocol `harness/abench_stage.py`), activation **awaiting_data**: the operator still has to
  freeze the data (below); until then `tools/oj_eval.py` returns `judge_pending` for this case.
- Reference submission (`example_submission/`): Not executed yet: the reference submission and the baseline need PyTorch, which was not available where the case was built; the operator runs them at activation. The four environments were checked in the build container against the gymnasium 1.0.0 source (constants, update equations, resets, episode limits; commit 99389909) and with fixed controllers: uniformly random actions give mean returns of −1,232 (Pendulum, gymnasium reports about −1,200) and −33 (MountainCarContinuous); a PD controller started upright keeps the pendulum at a return of −0.3 to −0.7 and the cart-pole at 499.8 of 500; pumping with the velocity sign reaches the acrobot goal in about 87 steps. The pipeline (budgeted environments, stage runner, metrics) runs end to end with a NumPy agent in tests/test_standard_cases.py.
- No number in the case files is a result: there are no training curves or reference scores.
- Baseline run by the judge alongside every submission: `harness/baselines/ppo`.

## Files

- `spec/cases/X-PPO-001.yaml` — source of the agent-facing task statement, interface, sandbox guide and `case/oj.yaml`
  (regenerate with `python3 tools/build_standard_cases.py --case X-PPO-001`).
- `case/agent/task.md` — the task statement the agent sees; `harness/` — every agent-visible interface file
  (`interface.md`, `sandbox_guide.md`, `stages.py`, `metrics.py`, the shared kit and any case helpers).
- `tools/prepare/x_ppo_001.py` — data preparation (sources, checks, splits).
- `example_submission/` — reference submission with its design report (`report/index.html`, `references.yaml`).

## Data sources (links only; nothing is downloaded in the repository)

- Classic-control dynamics of gymnasium 1.0.0 (Pendulum-v1, MountainCarContinuous-v0, CartPole-v1, Acrobot-v1), re-implemented for batches in cases/X-PPO-001/harness/control_envs.py; held-out initial states drawn with a secret seed (ABENCH_X_PPO_SEED): https://github.com/Farama-Foundation/Gymnasium/tree/9938990977eb5e85ea4523ecf5cb2678c1708ba4/gymnasium/envs/classic_control — license: MIT (gymnasium) (verified 2026-10-22)

## Activation (operator)

```bash
python3 tools/prepare/run.py --case X-PPO-001 --list-sources                 # what to download
python3 tools/prepare/run.py --case X-PPO-001 --raw-dir RAW --key-file KEY    # freeze public/hidden data, write case/data_lock.yaml
python3 tools/oj_eval.py --case X-PPO-001 --submission cases/X-PPO-001/example_submission --hidden-key-file KEY --out runs/...
python3 tools/prepare/run.py --case X-PPO-001 --activate                     # after reviewing the calibration run
```

Thresholds in `harness/metrics.py` are provisional until this calibration run on the frozen data and reference hardware.
