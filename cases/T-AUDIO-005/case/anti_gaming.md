# Anti-gaming checklist (T-AUDIO-005)

- Only the offline 16 kHz speech, processor, vocabulary and frozen encoder declared in `resources.yaml` may be read; network access, external checkpoints, external transcripts and augmented audio data are forbidden.
- The evaluation server verifies the weights of the frozen encoder and the processor/vocabulary hashes; any encoder gradient, parameter change or replacement invalidates the run.
- Trainable parameters may belong only to the CTC linear layer, low-rank/bottleneck adapters or the declared pooling head; unfreezing the full encoder and passing it off as an adapter is forbidden.
- Every output must contain `case_id`, resource/configuration/artifact hashes, seed, preprocessing configuration, hardware timing and failure status; output with missing provenance is not scored.
- The public split is isolated by speaker; hidden speaker, accent, noise and duplicate-cluster checks prevent transcript memorization and speaker leakage.
- The scoring target is a Pareto front over WER/robustness, parameter count, peak memory, real-time factor and compute cost; scores cannot be inflated on a single public split.
- OOM, timeouts, audio decoding failures, real-time-factor/quality constraint violations or hidden-input access are marked ineligible and never replaced by interpolation.
