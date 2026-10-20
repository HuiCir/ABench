# DS-T7-003 harness contract

This scaffold defines the server-side contract only; it does not run training or create synthetic results.

## Required interface

- Read the immutable manifest supplied by the benchmark.
- Write deterministic metrics.json and run_metadata.json.
- Record seed, input/resource hashes, wall-clock time, peak memory and exit status.
- Keep network and filesystem access within case/eval_protocol.yaml sandbox policy.

## Pending

The benchmark maintainer must freeze the executable runner, container image, resource hashes and failure semantics before accepting submissions.
