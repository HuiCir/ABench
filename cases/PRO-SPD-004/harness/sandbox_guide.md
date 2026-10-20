# Sandbox guide (PRO-SPD-004)

## 1. Where you are

| | |
|---|---|
| Hardware | 4 × H100 80GB in one node (NVLink); the evaluation uses the same node type |
| Network | None. You cannot download data or models or install packages |
| Software | Python 3, PyTorch (CUDA, Triton, torch.distributed / torchrun, torch.compile), NumPy; trust the versions reported in the sandbox |
| Session length | At most 24 hours wall clock; the deadline is in `$ABENCH_DEADLINE` (Unix seconds), and all processes are killed at that time |
| Compute | At most 96 H100-hours (24 hours of the 4-GPU node) during development; the sandbox records usage, and exceeding it is a violation |

Check the environment first:

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
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
| `ABENCH_DEVICE` | Device name; `cuda` in this sandbox | — |
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
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick      # interface check on a small slice, 1 seed
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --seeds 2    # full dev split, 2 seeds
```

`selfcheck.py` runs your submission through the evaluator's own pipeline (`abench_stage.py`, `stages.py`) and metric
code (`metrics.py`) on a dev split carved out of the public data: every stage is a fresh process with empty caches,
timed by the parent process, and held-out inputs appear only after the timed phase. Dev metrics are an estimate of
the held-out metrics, not a promise.

## 4. Case notes

- `selfcheck.py --quick` passes `--quick` to `train.py` (your script decides what a short run is) and evaluates on 1,048,576 tokens of the last training shard; `--seeds 1` runs your full recipe once and evaluates on the last 10,485,760 tokens of that shard, which are also training data, so the dev loss is optimistic.
- Budget your development: a full run of the example recipe takes hours, and the evaluation runs your recipe twice.
