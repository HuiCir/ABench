# Sandbox guide (TC-FT-001)

## 1. Where you are

| | |
|---|---|
| Hardware | 8 CPU cores, no GPU; the evaluation uses the same CPU allotment |
| Network | None. You cannot download data or models or install packages |
| Software | Python 3, NumPy, SciPy, scikit-learn; trust the versions reported in the sandbox |
| Session length | At most 3 hours wall clock; the deadline is in `$ABENCH_DEADLINE` (Unix seconds), and all processes are killed at that time |
| Compute | At most 8 CPU-core-hours (one hour of the 8-core allotment) during development; the sandbox records usage, and exceeding it is a violation |

Check the environment first:

```bash
nproc
python3 -c "import numpy, sklearn; print(numpy.__version__, sklearn.__version__)"
python3 -c "import os,time; print(round((int(os.environ['ABENCH_DEADLINE'])-time.time())/60), 'min left')"
```

## 2. Directories and environment variables

| Variable | Path and purpose | Access |
|---|---|---|
| `ABENCH_VIEW_DIR` | Root of the task statement, interface, templates | read-only |
| `ABENCH_DATA_DIR` | Public data of this case (see the Data section of the task) | read-only |
| `ABENCH_WORKSPACE` | Your workspace: code drafts, logs, temporary files | writable, deleted after the session |
| `ABENCH_CHECKPOINT_DIR` | Intermediate training results (never submitted) | writable, deleted after the session |
| `ABENCH_SUBMISSION_DIR` | Final submission directory, collected when the session ends | writable |
| `ABENCH_DEVICE` | Device name; `cpu` in this sandbox | — |
| `ABENCH_DEADLINE` | Session deadline (Unix seconds) | — |

`HOME` and `TMPDIR` both point to temporary directories inside the workspace. Paths outside the workspace are not writable and should not be read.

## 3. Recommended workflow

```bash
cd "$ABENCH_WORKSPACE"
ls -la "$ABENCH_DATA_DIR"
# ... develop, train and test in the workspace ...
python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<topic of the anchor>"   # related work before the cutoff
python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" show <arXiv id>                  # title, authors, date, abstract
mkdir -p "$ABENCH_SUBMISSION_DIR"      # copy the deliverables listed in the task statement here
python3 "$ABENCH_SUBMISSION_DIR/train.py" --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"   # your training program (the evaluator only loads the checkpoint)
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick      # interface check on a small slice, 1 seed
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --seeds 2    # full dev split, 2 seeds
```

`selfcheck.py` runs your submission through the evaluator's own pipeline (`abench_stage.py`, `stages.py`) and metric
code (`metrics.py`) on a dev split carved out of the public data: every stage is a fresh process with empty caches,
timed by the parent process, and held-out inputs appear only after the timed phase. Dev metrics are an estimate of
the held-out metrics, not a promise.

## 4. Case notes

- No GPU is available in this case; the evaluation runs on CPU only. The submission may depend only on NumPy, SciPy, scikit-learn and the Python standard library.
- The self-check scores your checkpoint on the dev documents (`dev_idx`). Leave them out of training if you want an unbiased estimate; you may include them in the final checkpoint.
