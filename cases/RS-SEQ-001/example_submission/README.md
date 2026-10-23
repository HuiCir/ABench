# RS-SEQ-001 reference submission

Counts consecutive item transitions i -> j within every user's sequence (log-damped) and a 90-day time-decayed
popularity, and saves both to checkpoint/markov.npz (float16 transition matrix, about 28 MB). By default each user's
final public item is left out, matching the dev split of selfcheck.py; --include-dev uses the full histories.

    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint                 # dev rows left out, for selfcheck.py
    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint --include-dev   # the checkpoint that is submitted

`model.py` is the entry point the evaluator loads with `checkpoint/`; `train.py` is kept for the audit and never run
by the evaluator. Design report: `report/index.html`; references: `references.yaml`.
