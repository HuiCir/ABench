# GN-OGB-002 reference submission

Closed-form one-vs-rest ridge regression to one-hot labels on SGC features (model.features), with the ridge strength
chosen on a validation set. By default the public valid nodes are left out entirely (they are the self-check's dev
set) and a random 10% of the train nodes is used for validation; --include-dev trains on train nodes and selects on
the valid nodes. Saves checkpoint/ridge.npz.

    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint                 # dev rows left out, for selfcheck.py
    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint --include-dev   # the checkpoint that is submitted

`model.py` is the entry point the evaluator loads with `checkpoint/`; `train.py` is kept for the audit and never run
by the evaluator. Design report: `report/index.html`; references: `references.yaml`.
