# GPU sandbox guide (IC-SPD-001)

## 1. Where you are

| | |
|---|---|
| GPU | One 24GB reference card (RTX 4090 class). Evaluation runs on the same model of card |
| Network | None. You cannot download data or models or install packages |
| Software | Python 3, PyTorch (CUDA, Triton), NumPy; trust the versions reported in the sandbox |
| Session length | At most 3 hours wall clock; the deadline is in `$ABENCH_DEADLINE` (Unix seconds), and all processes are killed at that time |
| GPU usage | At most 1 GPU-hour during development; the sandbox records GPU time, and exceeding it is a violation |

Check the environment first:

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
python3 -c "import os,time; print(round((int(os.environ['ABENCH_DEADLINE'])-time.time())/60), 'min left')"
```

## 2. Directories and environment variables

| Variable | Path and purpose | Access |
|---|---|---|
| `ABENCH_VIEW_DIR` | Root of the task statement, interface, templates and data | read-only |
| `ABENCH_DATA_DIR` | `train.npz`: `x` (50000,32,32,3) uint8, `y` int64, `dev_idx` (5000,) | read-only |
| `ABENCH_WORKSPACE` | Your workspace: code drafts, logs, temporary files | writable, deleted after the session |
| `ABENCH_CHECKPOINT_DIR` | Intermediate training results (never submitted) | writable, deleted after the session |
| `ABENCH_SUBMISSION_DIR` | Final submission directory, collected when the session ends | writable |
| `ABENCH_DEVICE` | Device name; `cuda` in the GPU sandbox | — |
| `ABENCH_DEADLINE` | Session deadline (Unix seconds) | — |

`HOME` and `TMPDIR` both point to temporary directories inside the workspace. Paths outside the workspace are not writable and should not be read.

## 3. Recommended workflow

```bash
cd "$ABENCH_WORKSPACE"
python3 - <<'PY'                                   # take a look at the data
import numpy as np, os
z = np.load(os.path.join(os.environ["ABENCH_DATA_DIR"], "train.npz"))
print(z["x"].shape, z["x"].dtype, np.bincount(z["y"]), z["dev_idx"].shape)
PY
mkdir -p "$ABENCH_SUBMISSION_DIR" && cp my_method.py "$ABENCH_SUBMISSION_DIR/method.py"
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick      # interface check: about 2k images, 1 seed
python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --seeds 2    # timing + dev accuracy
```

`selfcheck.py` times exactly as the evaluation does: every run is a new process with empty caches, and `torch.compile` compile time counts as training time. dev has only 5,000 images, so near 94% accuracy a single run has a standard error of about 0.34 percentage points; dev accuracy is only an approximation of test accuracy, and the self-check also trains on 5,000 fewer images.

## 4. Timing rules (same as evaluation)

- **Counted as training time**: importing `method.py` (including module-level code) and everything inside `train()`: moving data to the GPU, preprocessing, model construction, `torch.compile` compilation, warmup and training itself, until `train()` returns and CUDA synchronization completes.
- **Not counted**: reading `train.npz` into memory and creating the CUDA context. The timing script does both, identically for every submission.
- **Inference**: `predict()` must process all test images within 5 seconds (including synchronization), otherwise the run is invalid.
- **Caches**: `HOME`, `TMPDIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` and `CUDA_CACHE_PATH` are new empty directories for every run, so compiled results cannot be reused across runs.

## 5. Common ways to keep the GPU busy

- Move the training data to GPU memory once and do normalization and augmentation on the GPU, avoiding host-to-device copies every step.
- Use `channels_last` and half precision (fp16/bf16 via `torch.autocast`), and enable `torch.backends.cudnn.benchmark`.
- `torch.compile` can speed things up, but its compile time counts as training time; weigh the trade-off.
- Avoid `.item()`, printing tensors or other operations that force synchronization inside the training loop.
- Check peak memory with `torch.cuda.max_memory_allocated()`; exceeding 24GB fails outright.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is False | No device allocated; check `nvidia-smi`; if still unavailable, the only option is to end the session |
| selfcheck reports `protocol_violation` | The submitted code wrote to the timing pipe, or kept running after `predict`; do not touch the file descriptor given by `--fd` |
| selfcheck reports `predict_time_exceeded` | Inference took more than 5 seconds, or the process did not exit promptly after `predict` returned (for example a background thread still training) |
| selfcheck reports `runtime_error` | Look at `stderr_tail` in the output; common causes are OOM or a return shape other than (M, 10) |
| Not finished before the deadline | The session is killed at the deadline and only files already in `ABENCH_SUBMISSION_DIR` are collected; put a working version there early |
