# TC-FT-001 reference submission

Hashed unigram + bigram counts → sublinear tf-idf → linear classifier trained by SGD on the modified Huber loss with an
averaged iterate (scikit-learn). The learned idf vector, weights and intercepts are saved as one .npz; model.py
re-creates the same features with a stateless hashing function and needs only NumPy and scikit-learn's vectorizer.
By default the dev documents (dev_idx) are left out so that selfcheck.py gives an unbiased dev estimate.

    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint                 # dev rows left out, for selfcheck.py
    python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint --include-dev   # the checkpoint that is submitted

`model.py` is the entry point the evaluator loads with `checkpoint/`; `train.py` is kept for the audit and never run
by the evaluator. Design report: `report/index.html`; references: `references.yaml`.
