# Anti-gaming checklist (X-DISTILL-010)

- The teacher, the student's initial weights, the tokenizer, the layer mapping and the teacher-logit cache are all frozen by hash on the server; the submission must not replace the teacher or read hidden labels.
- The hidden split is held out by capability family, template and length, to avoid memorizing public templates or teacher outputs.
- Only the declared CE/KD/feature/sequence losses, temperatures and weights are allowed; scores must not be inflated through extra PEFT, external data or a hidden post-distillation retraining stage.
- The server measures real training GPU-h, student tokens/s, memory, OOM, timeouts and multi-seed stability; failures are recorded as ineligible.
- Student capability, teacher gap and pretraining CE regression are reported together; a single accuracy cannot replace the full Pareto evaluation.
