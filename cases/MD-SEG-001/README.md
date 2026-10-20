# ABench Case MD-SEG-001: Replace the network architecture inside the nnU-Net pipeline to design a better 3D segmentation model

- Track: MD · Medicine · domain
- Anchor: nnU-Net (arXiv: 1809.10486)
- Subset: main; difficulty: medium; resource tier: high; status: **deferred**
- Task: Replace the network architecture inside the nnU-Net pipeline to design a better 3D segmentation model
- Metrics: Dice / NSD (MSD subtasks); memory; inference time
- Proxy: SUBSET; plan: 3 small MSD tasks (Hippocampus, Prostate, Heart), 100 epochs, 1 fold, plus 1 hidden task; no 5-fold 1000-epoch runs

## Current deliverables

This directory only sets up the bench contract scaffold: task card, evaluation protocol, resource registry, scoring contract and directory-level notes. There is no simulated training, no simulated reference run, no simulated score and no fabricated leaderboard result. Formal runs require the benchmark operator to inject real resources, reference runs, the hidden set and server-generated run records.

## Blockers and reservations

This case is held back from the formal leaderboard for now; only the protocol and interfaces are kept.

Blocking inputs:

- `public_deidentified_dataset_and_licence`
- `patient_level_split_and_preprocessing_lock`
- `medical_metric_review`
- `specific_anchor_or_citation_verification`

Until these inputs are frozen, only interfaces/contracts are added; no excessive environment setup or simulated substitutes.

## Files

- `case/anchor_card.yaml`: task anchor and acceptance boundary.
- `case/task_prompt.md`: the task given to the agent under test.
- `case/eval_protocol.yaml`: proxy ladder, budget, hidden evaluation and output constraints.
- `case/resources.yaml`: registry of data, models, software, hardware and private resources.
- `case/scoring_config.yaml`: metrics and thresholds pending calibration.
- `harness/README.md`: execution interface conventions.
- `scoring/README.md`: scoring input/output conventions.
- `example_submission/README.md`: example submission directory conventions.
