# FN-LTSF-005 reference submission

A channel-shared ridge forecaster with last-value normalisation (`src/linear.py`), with lookback (96–720) and penalty
chosen per dataset and horizon on the validation rows. NumPy only.

    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint            # training rows only, for selfcheck.py
    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint --final    # refit on training + validation rows

Design report: `report/index.html`; references: `references.yaml`.
