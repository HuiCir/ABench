# X-ROBUST-009 harness contract

Only the interface is defined at present; no simulated implementation is provided. The formal harness reads frozen offline resources from the server manifest, limits file, network and process access, runs the public smoke and hidden replay, and outputs a deterministic run.jsonl, metrics.json and artifacts.sha256.

The harness must save the calibration split, thresholds, seeds, model/resource hashes, and per-shift and per-policy-family metrics. The server measures latency, peak memory and GPU-h; without real resources or hidden results it returns only pending_reference_runs.
