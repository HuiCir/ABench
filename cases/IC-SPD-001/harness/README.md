# IC-SPD-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same timing parent on the dev split.

## Timing parent (`speedrun_runner.py`)

```
IC-SPD-001 timing parent process: starts speedrun_harness.py and times with the parent's own clock.

The judge (outside the sandbox) and selfcheck.py (inside the sandbox) both call run_once() here, with exactly the same timing rules:

  - every run is a fresh process; HOME, TMPDIR and the torch.compile / Triton / CUDA cache directories point to
    empty directories owned by this run alone, so compiled results or any caches cannot be reused across runs;
  - training time = interval between receiving handshake byte R and receiving T (parent clock);
  - inference time = interval from T to P, which must not exceed predict_cap seconds;
  - the interval from T to process exit must not exceed predict_cap + tail_slack seconds, so training cannot be moved after T;
  - the handshake must be exactly the three bytes R, T, P in the right order.
```

## Files

- `gpu_sandbox.md`
- `interface.md`
- `selfcheck.py`
- `speedrun_harness.py`
- `speedrun_runner.py`

The held-out data, keys and judge configuration are never in this directory.
