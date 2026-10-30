"""IC-SPD-001 on the shared R/T/G/P protocol, with the speedrun timing contract of speedrun_runner.py: the unlabeled
test images are in the stage directory from the start (pre/held_pre.npz, like --x-test), training is timed from R to
T, and P must come within predict_cap (5 s) of G, which the parent sends right after T since no inputs are delivered
then; writing the predictions may use tail_slack (3 s) more."""
from pathlib import Path
import numpy as np
from abench_stage import Stage

PARAMS = {"entry": "method.py"}
HELD_OUT_KEYS = []
HELD_OUT_PRE_KEYS = ["x"]
PIPELINE = [{"stage": "train", "timeout": 600, "finish_cap": 5, "tail_slack": 3, "slack_after_p": True,
             "threads": 8, "held_out": False, "held_out_pre": True, "models": False}]


def setup(ctx):
    with np.load(ctx.public / "train.npz", allow_pickle=False) as z:
        return {"x": np.ascontiguousarray(z["x"]), "y": np.asarray(z["y"], dtype=np.int64)}


def timed(ctx, data):
    mod = ctx.import_submission(ctx.params.get("entry", "method.py"), required=("train", "predict"))
    return mod, mod.train(data, seed=ctx.seed, device=ctx.device)


def finish(ctx, state):
    mod, model = state
    with np.load(ctx.pre_dir / "held_pre.npz", allow_pickle=False) as z:
        pred = mod.predict(model, z["x"], device=ctx.device)
    if hasattr(pred, "detach"):
        pred = pred.detach().float().cpu().numpy()
    return {"pred": np.asarray(pred, dtype=np.float32)}


STAGES = {"train": Stage(setup=setup, timed=timed, finish=finish)}
