"""E-VLA-003 pipeline: your action head on the frozen SmolVLM2 backbone controls LIBERO-Spatial and LIBERO-Object
robots; the evaluator runs every episode (evaluator and self-check use this same file).

One stage, "rollout" (models: SmolVLM2-500M-Video-Instruct, smolvla_base):
  timed   start the policy's process: the evaluator loads and freezes the VLM there (vla_frozen.before) and calls
          policy.py's load_policy(checkpoint_dir, models_dir, device, vlm, processor); ready within 900 s
  finish  one synchronous episode (libero_eval.run_sync) per held-out initial state of both suites; per episode:
          success, steps, mean policy time per control step; whether the VLM's weights are unchanged afterwards
          (vla_frozen.after); the numbers stored in checkpoint/                                         (≤ 7,200 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import libero_eval as LE
from abench_stage import Context, Stage

SUITES = LE.SUITES                                   # ("libero_spatial", "libero_object")
MAX_PARAMS, MAX_STEP_MS = 100_000_000, 50.0
PARAMS = {"suites": list(SUITES), "max_steps": {s: LE.MAX_STEPS[s] for s in SUITES}, "settle_steps": LE.SETTLE,
          "max_checkpoint_numbers": MAX_PARAMS, "max_mean_ms_per_step": MAX_STEP_MS, "vlm": "SmolVLM2-500M-Video-Instruct"}
HELD_OUT_KEYS = ["init_states", "init_len", "task_id", "suite_id"]
PIPELINE = [{"stage": "rollout", "timeout": 900 + 7200, "finish_cap": 7200, "tail_slack": 30, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    hook = Path(__file__).resolve().parent / "vla_frozen.py"
    return LE.AgentProcess(ctx.submission / "policy.py", "load_policy",
                           {"checkpoint_dir": str(ctx.submission / "checkpoint"), "models_dir": str(ctx.models),
                            "device": ctx.device},
                           device=ctx.device, threads=4, hook=hook,
                           hook_kwargs={"models_dir": str(ctx.models), "device": ctx.device}, load_timeout=900)


def rollouts(policy, held: dict, seed: int, make_env=None) -> dict:
    make_env = make_env or LE.make_env
    states, lens = held["init_states"], held["init_len"].astype(int)
    tasks, suites = held["task_id"].astype(int), held["suite_id"].astype(int)
    out = {k: np.zeros(len(states)) for k in ("success", "steps", "policy_ms")}
    envs = {}
    try:
        for i in range(len(states)):
            key = (suites[i], tasks[i])
            if key not in envs:
                envs[key] = make_env(SUITES[suites[i]], int(tasks[i]), seed)
            env, lang, _ = envs[key]
            r = LE.run_sync(policy, env, lang, states[i, :lens[i]], LE.MAX_STEPS[SUITES[suites[i]]])
            for k in out:
                out[k][i] = r[k]
    finally:
        for env, _, _ in envs.values():
            if hasattr(env, "close"):
                env.close()
    return out


def _finish(ctx: Context, policy):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        held = {k: z[k] for k in z.files}
    try:
        out = rollouts(policy, held, ctx.seed)
        fin = policy.final() or {}
    finally:
        policy.close()
    out["vlm_unchanged"] = np.asarray([int(fin.get("vlm_unchanged", 0))], np.int64)
    out["checkpoint_numbers"] = np.asarray([LE.checkpoint_numbers(ctx.submission / "checkpoint")], np.int64)
    return out


STAGES = {"rollout": Stage(timed=_timed, finish=_finish)}


def held_out_from(states: list, tasks: list, suites: list) -> dict:
    a, lens = LE.pack_states(states)
    return {"init_states": a, "init_len": lens, "task_id": np.asarray(tasks, np.int64),
            "suite_id": np.asarray(suites, np.int64)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: LIBERO's published initial states of the 20 tasks (the held-out states are new placements from a
    secret seed): the first 3 of each task (--quick: the first one)."""
    k = 1 if quick else 3
    states, tasks, suites = [], [], []
    for si, suite in enumerate(SUITES):
        for tid in range(10):
            _, _, inits = LE.task_info(suite, tid)
            states += [np.asarray(s) for s in inits[:k]]
            tasks += [tid] * k
            suites += [si] * k
    held = held_out_from(states, tasks, suites)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz",
            "reference": {"task_id": held["task_id"], "suite_id": held["suite_id"]}, "time_scale": 1.0}
