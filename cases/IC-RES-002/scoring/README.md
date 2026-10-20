# IC-RES-002 scoring contract

Scoring consumes only benchmark-generated real artifacts. No simulation, mock run, fabricated reference curve or local leaderboard is included in this scaffold.

Required inputs: candidate metrics.json and run_metadata.json; server-side reference runs and hidden holdout outputs; frozen scoring configuration and evaluator version.

The scorer must report raw metrics, aggregation by dataset/task, resource usage, confidence intervals and gate failures. Missing or unverifiable artifacts are failed or unknown runs, not imputed scores.
