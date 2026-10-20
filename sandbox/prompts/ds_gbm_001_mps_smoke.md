You are running the ABench DS-GBM-001 local smoke inside an isolated workspace.

Work only inside `$ABENCH_WORKSPACE`. Use the fixed local data interface at
`$ABENCH_DATA_DIR` and the device exposed through `$ABENCH_DEVICE`; do not use
the network, hidden labels, or paths outside the workspace and declared model
store. You may train a tiny tabular model and save temporary files under
`$ABENCH_CHECKPOINT_DIR` or `$ABENCH_MODEL_DIR`.

Before finishing, write only the declared submission files under
`$ABENCH_SUBMISSION_DIR`: `model.py`, `checkpoint/`, `report/index.html`,
`references.yaml`, and an optional `README.md`. Do not write self-reported
metrics or result files. Smoke output is marked `smoke_unscored`; all metrics
are computed later by the evaluator from the submitted model and checkpoint.
