# ABench Case B-T7-005: Judging the verifiability of pretraining-optimizer speedup claims

This is a **bench scaffold** built in the unified ABench pattern, not a finished real evaluation. The directory contains only the protocol, contracts and the real-run entry point; it provides no simulated training, simulated review, fabricated reference curves or placeholder scores.

- Track: B; anchor: Fantastic Pretraining Optimizers (arXiv: 2509.02046)
- Subset: main; difficulty: medium; resource tier: low
- Current status: `scaffold`; external resources, reference runs, hidden set and hardware calibration to be frozen by the benchmark operator

## Usage

1. Prepare real resources and the server-side run environment according to `case/bench_contract.yaml`.
2. The benchmark operator produces a real `runs.json` and a blind-review `judge.json`, keeping provenance, configuration hashes and artifact hashes.
3. Run:

```bash
python3 scoring/run_case.py \
  --submission /path/to/submission \
  --runs /path/to/real_runs.json \
  --judge /path/to/real_judge.json \
  --out /path/to/report
```

The entry point rejects results marked mock/simulated/demo/synthetic; it currently only validates contracts and computes no uncalibrated formal score.

## Directory

- `case/`: task card, evaluation protocol, resources/reference pool, scoring and anti-gaming contracts.
- `harness/`: notes on the server-side adapter to be wired in; no runtime is faked in this scaffold.
- `example_submission/`: notes on the submission format, with no simulated implementation that could pass as a baseline.
- `scoring/`: an entry point that accepts only real run artifacts.
- `outputs/reference_runs/`: destination for real reference artifacts, currently empty.
