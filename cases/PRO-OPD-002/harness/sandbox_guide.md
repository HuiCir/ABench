# Sandbox guide (PRO-OPD-002)

## 1. Where you are

| | |
|---|---|
| Hardware | 4 × H100 80GB in one node (NVLink); the evaluation uses the same node type |
| Network | Online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run. Data and pretrained weights come only from the directories below |
| Software | Python 3, PyTorch (CUDA), vLLM, transformers, safetensors, NumPy, pandas; verl, OpenRLHF and TRL from the allowlisted pip mirror; trust the versions reported in the sandbox |
| Session length | At most 24 hours wall clock; the deadline is in `$ABENCH_DEADLINE` (Unix seconds), and all processes are killed at that time |
| Compute | At most 96 H100-hours (24 hours of the 4-GPU node), teacher inference, development and final training included during development; the sandbox records usage, and exceeding it is a violation |

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
| `ABENCH_MODELS_DIR` | Frozen pretrained files allowed in this case | read-only |
| `ABENCH_WORKSPACE` | Your workspace: code drafts, logs, temporary files | writable, deleted after the session |
| `ABENCH_CHECKPOINT_DIR` | Intermediate training results (never submitted) | writable, deleted after the session |
| `ABENCH_SUBMISSION_DIR` | Final submission directory, collected when the session ends | writable |
| `ABENCH_DEVICE` | Device name; `cuda` in this sandbox | — |
| `ABENCH_DEADLINE` | Session deadline (Unix seconds) | — |

`HOME` and `TMPDIR` both point to temporary directories inside the workspace. Paths outside the workspace are not writable and should not be read.

## 3. Recommended workflow

```bash
cd "$ABENCH_WORKSPACE"
ls -la "$ABENCH_DATA_DIR" "$ABENCH_MODELS_DIR"
# ... develop, train and test in the workspace ...
python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<topic or problem>"     # related work, at any point
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

- Run training as `python3 interface/pro_audit.py run --name NAME --code DIR -- <command>`; write checkpoints to `$ABENCH_CKPT_DIR/step_<n>/` with `model/`, your resume state and a final `DONE` file, at least every 30 minutes, and keep them all.
- Teacher answers can be generated once and cached in the session's work directory; they count as derived data and stay inside the session.
- `selfcheck.py` runs the audit on your session, the evaluation on the 60 dev problems and a short resume (`--quick`: 8 problems).
