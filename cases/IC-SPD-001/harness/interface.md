# IC-SPD-001 interface

## Entry point: `method.py`

`method.py` at the root of the submission directory must define two functions:

```python
def train(data: dict, seed: int, device: str):
    """Train a CIFAR-10 classifier from scratch and return it.

    data["x"]  numpy.ndarray, uint8, shape (N, 32, 32, 3), RGB, channels last, pixels 0–255, not normalized
    data["y"]  numpy.ndarray, int64, shape (N,), values 0–9
    seed       integer; all randomness (initialization, data order, augmentation) must be determined by it
    device     string, "cuda" during evaluation
    returns    any Python object, passed unchanged to predict() by the judge
    """


def predict(model, x, device: str):
    """Output scores for 10 classes for x (logits or probabilities both work).

    x          numpy.ndarray, uint8, shape (M, 32, 32, 3), same format as data["x"]
    returns    numpy.ndarray or torch.Tensor of shape (M, 10), finite floats; the argmax is taken as the predicted class
    """
```

- `train` receives **all 50,000** training images (during evaluation) or the 45,000 left after removing the dev split (during the self-check). Do not assume a fixed N.
- Class indices: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck.
- `predict` may use test-time augmentation (such as averaging over horizontal flips), but must not update model parameters and must finish within the inference time limit.
- `method.py` may `import` modules under `src/` in the same directory; `speedrun_harness.py` adds the submission directory to `sys.path`.

## Timing code

- `speedrun_harness.py`: the timed child process. It imports your `method.py`, calls `train` and `predict`, and sends the timing handshake.
- `speedrun_runner.py`: the timing parent process. It times with its own clock and prepares empty cache directories for every run.
- `selfcheck.py`: uses the two files above to self-check on the public dev split.

The judge calls the same `speedrun_runner.py` and `speedrun_harness.py`; the timing rules are in the "Metrics" section of the task statement.

## Available software

Trust the output of `python3 -c "import torch; print(torch.__version__, torch.version.cuda)"` in the sandbox: Python 3, PyTorch (with CUDA and Triton; `torch.compile` is available) and NumPy. No other packages can be installed; the submission may depend only on these libraries and the Python standard library.
