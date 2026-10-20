# ABench Pro Set (v2026.07)

## 1. Why it is needed

The Budget edition squeezes every case onto one 24GB card, which allows large-scale, low-cost comparison of agents, but cannot answer a reasonable objection: **can the agent do the tasks production settings truly depend on, with industrial-scale compute**? The Pro Set picks a few problems that are widely used in industry and cannot be done without large compute, provides ample resources, and ranks them separately.

| Item | Rule |
|---|---|
| Hardware | One node with 4 × H100 80GB SXM (NVLink), local NVMe ≥ 4TB, CPU memory ≥ 1TB |
| Wall clock | 24 hours, covering development, debugging and final training; allocated by the agent |
| Network | Isolated; only allowlisted mirrors of the HF Hub and pip are open |
| Starting weights | Only the allowlisted HF models specified by the case; hashes recorded by the benchmark operator |
| Leaderboard | A separate Pro leaderboard, not mixed with the Budget edition |

## 2. Verification: audit instead of retraining

In the Budget edition the benchmark operator retrains the agent's method. In the Pro Set one training run is 96 H100-h, and rerunning it for every submission is unaffordable. So the deliverable changes from "method" to "method + training artifacts + an auditable process":

1. **Artifact evaluation**: the checkpoint or serving system delivered by the agent is evaluated on a hidden evaluation set, which costs only inference.
2. **Process integrity**: logs, the checkpoint hash chain, data shard numbers and GPU telemetry are all written by the harness, and the agent cannot modify them. Checkpoint hashes are registered every 30 minutes, and the delivered checkpoint must be traceable to the starting weights.
3. **Sampled reproduction**: the benchmark operator randomly picks one registered point from the hash chain, resumes for 1 hour with the agent's training script, and checks that the loss or reward trajectory matches the log (within noise tolerance). A mismatch invalidates the submission.
4. **Full reruns of short tasks**: tasks that are short by nature (pretraining speedrun, serving optimization) are rerun in full twice by the benchmark operator, taking the median.

## 3. Scoring

SOTA margin M, divergence D, reference quality Q and local discovery X are kept, with these changes:

- **Reference methods**: each case holds only 1–2 strong industry recipes, run by the benchmark operator under the same 4 × H100 × 24 h. Compute only allows this many, so there is no 10-method envelope.
- **Noise floor**: reference recipes run with 2 seeds, or evaluation questions are bootstrapped. A lead counts only above 2σ.
- **Divergence**: most Pro cases have no behavioral probe like the fixed gradient stream, so D relies mainly on mechanism divergence from the reference recipes and on code divergence.
- **Engineering metrics**: MFU, throughput, goodput, training interruptions and loss spikes are reported separately. Under the hard 24-hour constraint hardware utilization translates directly into results, so it is not scored again.

## 4. Relation to the Budget edition

Some Pro cases have micro-scale counterparts in the Budget edition (A-MOE-003 to PRO-MOE-003, X-GRPO-002 to PRO-ARL-001). The agreement of the same agent's rankings across the two editions serves as external evidence of the Budget edition's proxy fidelity, and can replace or supplement the outer review of Section 6 of `compute_budget.md`.

## 5. Anti-gaming

| Gaming method | Protection |
|---|---|
| Swapping in stronger external weights or data | Allowlisted starting weights and data mirrors, traced through the hash chain |
| Training on evaluation data | Hidden evaluation set, preferably with questions created after the cutoff; n-gram overlap detection between training data and the evaluation set |
| Fabricating or cherry-picking logs | Logs and telemetry written by the harness; sampled reproduction |
| RL reward hacking | Training reward separated from the evaluation grader; hidden reward variants used at evaluation |
| Serving optimization that sacrifices output quality | Token-by-token agreement under greedy decoding, or a distribution-equivalence test for sampling |
| Training several times within 24 h and delivering only the best | Allowed as normal R&D; but every run is registered in the hash chain, and the report states "on which attempt the final method was obtained" |

## 6. Cost

| Item | H100-h |
|---|---|
| Per agent per case | 96 (agent) + about 8–16 (benchmark verification) |
| Per agent across all 6 cases | about 640 |
| Reference recipes (one-off, 6 cases) | about 720 |
| 10 agents through the full set | about 7100 |
