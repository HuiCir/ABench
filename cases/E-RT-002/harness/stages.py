"""E-RT-002 pipeline: your trained policy controls a LIBERO-Spatial robot asynchronously while its plans arrive late;
the evaluator runs every episode (evaluator and self-check use this same file).

One stage, "rollout" (no pretrained models):
  timed   start two processes from policy.py: load_planner(checkpoint_dir, device) on the GPU and
          make_controller(checkpoint_dir) on the CPU; both must be ready within 600 s
  finish  for each injected latency of DELAYS control steps and each held-out initial state, one episode of
          libero_eval.run_async (the planner sees observations, the controller executes actions; they communicate only
          through requests and chunks); per episode: success, steps, plans, planning time, effective delay; and the
          numbers stored in checkpoint/                                                                (≤ 5,400 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import libero_eval as LE
from abench_stage import Context, Stage

SUITE = "libero_spatial"
DELAYS = (0, 3, 6)
MAX_PARAMS = 100_000_000
PARAMS = {"suite": SUITE, "delays_steps": list(DELAYS), "control_hz": LE.CONTROL_HZ, "max_steps": LE.MAX_STEPS[SUITE],
          "settle_steps": LE.SETTLE, "max_checkpoint_numbers": MAX_PARAMS}
HELD_OUT_KEYS = ["init_states", "init_len", "task_id"]
PIPELINE = [{"stage": "rollout", "timeout": 600 + 5400, "finish_cap": 5400, "tail_slack": 30, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    entry, ck = ctx.submission / "policy.py", ctx.submission / "checkpoint"
    planner = LE.AgentProcess(entry, "load_planner", {"checkpoint_dir": str(ck), "device": ctx.device},
                              device=ctx.device, threads=4)
    controller = LE.AgentProcess(entry, "make_controller", {"checkpoint_dir": str(ck)}, threads=1)
    return {"planner": planner, "controller": controller}


def rollouts(planner, controller, held: dict, seed: int, delays=DELAYS, make_env=None, suite: str = SUITE) -> dict:
    """Run every (initial state, delay) episode; arrays of shape (n_states, n_delays)."""
    make_env = make_env or LE.make_env
    states, lens, tasks = held["init_states"], held["init_len"].astype(int), held["task_id"].astype(int)
    shape = (len(states), len(delays))
    out = {k: np.zeros(shape, np.float64) for k in ("success", "steps", "plans", "plan_ms", "mean_delay")}
    envs = {}
    try:
        for i in range(len(states)):
            if tasks[i] not in envs:
                envs[tasks[i]] = make_env(suite, int(tasks[i]), seed)
            env, lang, _ = envs[tasks[i]]
            for j, d in enumerate(delays):
                r = LE.run_async(planner, controller, env, lang, states[i, :lens[i]], LE.MAX_STEPS[suite], int(d))
                for k in out:
                    out[k][i, j] = r[k]
    finally:
        for env, _, _ in envs.values():
            if hasattr(env, "close"):
                env.close()
    return out


def _finish(ctx: Context, st):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        held = {k: z[k] for k in z.files}
    try:
        out = rollouts(st["planner"], st["controller"], held, ctx.seed)
    finally:
        st["planner"].close()
        st["controller"].close()
    out["checkpoint_numbers"] = np.asarray([LE.checkpoint_numbers(ctx.submission / "checkpoint")], np.int64)
    return out


STAGES = {"rollout": Stage(timed=_timed, finish=_finish)}


def held_out_from(states: list, tasks: list) -> dict:
    a, lens = LE.pack_states(states)
    return {"init_states": a, "init_len": lens, "task_id": np.asarray(tasks, np.int64)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: LIBERO's published initial states of the ten LIBERO-Spatial tasks (the held-out states are new
    placements from a secret seed): the first 5 of each task (--quick: the first one)."""
    k = 1 if quick else 5
    states, tasks = [], []
    for tid in range(10):
        _, _, inits = LE.task_info(SUITE, tid)
        states += [np.asarray(s) for s in inits[:k]]
        tasks += [tid] * k
    held = held_out_from(states, tasks)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"task_id": held["task_id"]},
            "time_scale": 1.0}
