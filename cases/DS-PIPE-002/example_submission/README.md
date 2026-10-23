# DS-PIPE-002 reference submission

The same procedure for every dataset: histogram gradient-boosted trees (missing values handled natively) stopped
early on an internal 10% validation split, in the spirit of the anchor's robust defaults. By default the dev rows
(dev_idx) are left out so that selfcheck.py gives an unbiased dev estimate.

    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint                 # dev rows left out, for selfcheck.py
    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint --include-dev   # the checkpoint that is submitted

`model.py` is the entry point the evaluator loads with `checkpoint/`; `train.py` is kept for the audit and never run
by the evaluator. Design report: `report/index.html`; references: `references.yaml`.
