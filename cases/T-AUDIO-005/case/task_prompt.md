> **v1 legacy, for operators only.** The agent never sees this file; the agent view contains only `case/agent/task.md`, the interface files listed in `case/oj.yaml`, report templates and public data (see `spec/oj_contract.md`).

# ABench task T-AUDIO-005

You are a research engineer taking part in ABench. The corpus cutoff is **2026-07-01**; use only material that was public and verifiable before the cutoff.

## Task

On a frozen small wav2vec 2.0/HuBERT audio encoder, design a lightweight CTC, low-rank/bottleneck adapter or pooling head and build a small ASR system on a fixed offline speech bundle. The goal is a Pareto front over WER, out-of-speaker generalization, trainable parameters, memory, real-time factor and GPU-h.

Anchor: **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**. This is a low-cost case of audio architecture/training transfer that tests classic CTC/adapter designs on frozen representations, without chasing SOTA on a single public dataset.

## Proxy and budget

- Proxy: FROZEN, SUBSET; fixed 16 kHz mono, 2–8 second audio and speaker-disjoint splits.
- Only a CTC linear layer, low-rank/bottleneck adapters or a lightweight pooling head may be trained; the encoder, processor and vocabulary must stay frozen.
- Development budget: 3 GPU-h; per verification: 1.5 GPU-h; reference precompute: 15 GPU-h; difficulty: medium; resource rating: medium; resource profile: `single_gpu_24gb`.

## Deliverables

The submission directory must contain an executable method, `manifest.json`, source declarations, a deterministic run configuration and machine-readable result records. Do not submit or rely on fabricated curves, reference results, judge results or simulated scores. All random seeds, audio preprocessing, resource/model hashes, speaker splits, hardware timings and failure states must be written to the record.

## Evaluation boundary

`case/eval_protocol.yaml` and `case/scoring_config.yaml` are authoritative. The server replays the visible smoke run, reviews on hidden speaker/accent/noise resources, and forms a Pareto front from the non-dominance of quality, robustness, resource and latency metrics.

## ABench audio method report package (`audio_adapter_v1`)

Once this case enters `active_v1` or `proxy_ready`, `submission/` must contain `method_report.md`, `report_manifest.json`, `data_provenance.yaml`, `references.yaml` (`sources.yaml` during migration) and `evidence.yaml` (`claim_evidence.jsonl` during migration); keep `retrieval_log.jsonl` when external sources are used. The report must describe the frozen encoder, sample rate/features, adapter interface, speaker-disjoint split, preprocessing, leak prevention, measurement of WER/CER and real-time factor/memory, ablations, real results or `pending`, limitations and the reproduction command. This profile is exempt from `architecture_report.*`, but the provenance of the encoder, data and adapter must be traceable; simulated results are forbidden. Fields and checks are in `spec/report_contract.md` and `spec/report_profiles.yaml`. In the `contract_only`/`deferred` stage only the interfaces are frozen and no citation budget is consumed.

