> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task C-QNT-003

On a fixed 0.5B–1.5B decoder, tokenizer and single-GPU runtime, design AWQ/GPTQ/SmoothQuant-style weight/activation quantization configurations. Public calibration text is for development; hidden models, tasks, batch/sequence shapes and low-batch decode are for final evaluation. Under numerical-equivalence and quality-floor constraints, the submission should push quantized quality, model size, throughput, p95 latency, memory and calibration cost to a better Pareto front.

Do not download external checkpoints/kernels, read hidden inputs or report only self-measured throughput; all timing, memory, model size, failures and hashes are recorded server-side.

## ABench unified systems report package (`system_efficiency_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `system_report.md`, `report_manifest.json`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the system boundary, shapes/dtypes/layouts, correctness and equivalence checks, the end-to-end timing boundary, throughput/latency/memory measurement, baselines, runnable ablations, real results or `pending`, failure modes and the reproduction command. If the model topology changes, also submit `architecture_report.html` and `architecture_report.md`; self-reported numbers in the report are not scored, and simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.

