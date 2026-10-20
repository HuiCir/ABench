# ABench Case X-DISTILL-010: Knowledge-distillation recipe for a small student model

This is an offline, single-GPU post-training scaffold that tests the Pareto front of classic knowledge distillation over capability retention, student inference cost and training cost. No training, reference runs or simulated scores are currently provided.

- Corpus cutoff: 2026-07-01; anchor: Hinton et al., "Distilling the Knowledge in a Neural Network" (arXiv: 1503.02531).
- Difficulty: **medium**; resource tier: **medium**; sandbox: `posttrain_24gb`.
- Development about 3 GPU-h, per verification about 1.5 GPU-h, reference precompute about 15 GPU-h.
- The teacher (≤ 350M) is frozen and the student (≤ 125M) is trained; teacher logits/cache may be precomputed by the server.
- Only hard-label CE, logit KD, feature/sequence-level KD and temperature/loss weights are compared; PEFT variables such as LoRA/QLoRA/DoRA are not introduced.

The server freezes hidden splits by capability family, template and length, and measures teacher gap, pretraining forgetting, student throughput, memory and training GPU-h. Formal scoring requires real resources, reference runs and hidden evaluation.
