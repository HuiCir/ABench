# T-AUDIO-005 harness

The server-side harness mounts the fixed offline 16 kHz audio bundle, verifies the frozen encoder/processor/vocabulary hashes, isolates training output, measures WER/CER, speaker-held-out WER, real-time factor, peak memory and GPU-h, and produces the real run records the contract requires. In the scaffold stage no audio is downloaded, no model is trained and no simulated score is generated.
