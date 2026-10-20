# ABench Case T-AUDIO-005: Lightweight CTC/adapter ASR on a frozen audio encoder

This is a low-cost, fully offline audio-model optimization case. It tests classic CTC/adapter designs on frozen wav2vec 2.0/HuBERT representations, aiming for Pareto optimality over WER, out-of-speaker generalization, trainable parameters, memory, real-time factor and compute cost.

- Corpus cutoff: 2026-07-01.
- Difficulty: **medium**; resource rating: **medium**; budget tier: **medium**.
- Proxy: FROZEN, SUBSET; development about 3 GPU-h, per verification about 1.5 GPU-h, reference precompute about 15 GPU-h.
- Resources: `case/resources.yaml`; all resources are mounted offline by the server, and network access at run time is forbidden.
- Currently only a scaffold, with no real reference runs, hidden evaluation or simulated scores.

## Proxy setting

Freeze a wav2vec 2.0/HuBERT audio encoder of ≤ 100M with fixed 16 kHz mono speech and CTC vocabulary, and train only a linear CTC layer, low-rank/bottleneck adapters or a lightweight pooling head. A public speaker-disjoint split is used for selection; hidden speakers, accents and noise conditions are used for review.

## Metrics and scoring

Metric definitions are in `case/eval_protocol.yaml`. This case uses `pareto_frontier`: WER, CER and speaker-held-out WER are minimized, and parameter count, peak memory, real-time factor and GPU-h are minimized too; constraints must pass first, and the output is the full non-dominated set and the hypervolume relative to the reference envelope.

## Blockers before launch

- `frozen_audio_encoder_processor_and_vocab_hash`
- `offline_16khz_speech_bundle_and_speaker_holdout`
- `fixed_audio_preprocess_and_server_asr_evaluator`
- `fixed_gpu_timer_and_peak_memory_probe`

Protocol references: [`spec/pareto_protocol.md`](../../spec/pareto_protocol.md); [`spec/sandbox_profiles.md`](../../spec/sandbox_profiles.md).
Resource profile: `single_gpu_24gb`; objective mode: `objective_mode: pareto`.
