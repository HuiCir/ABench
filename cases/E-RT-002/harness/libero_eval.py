"""LIBERO evaluation shared by E-RT-002 and E-VLA-003 (evaluator code, agent-visible; identical copy in each case).

Environment: LIBERO (Liu et al. 2023; github.com/Lifelong-Robot-Learning/LIBERO, MIT) on robosuite/MuJoCo, a Franka
arm controlled at 20 Hz with actions of 7 numbers (end-effector delta position 3, delta rotation 3, gripper -1 open /
+1 close; clipped to [-1, 1]), 128 x 128 RGB cameras.

Observation given to a policy (obs_dict), in the orientation and units of the LIBERO demonstration files
(scripts/create_dataset.py of LIBERO: no image flipping):
  "agentview_rgb"    uint8 (128, 128, 3)   third-person camera        (hdf5: obs/agentview_rgb)
  "eye_in_hand_rgb"  uint8 (128, 128, 3)   wrist camera               (hdf5: obs/eye_in_hand_rgb)
  "state"            float32 (8,)          end-effector position (3), orientation as axis-angle (3), gripper finger
                                           positions (2)               (hdf5: obs/ee_pos, obs/ee_ori, obs/gripper_states)
  "instruction"      str                   the task's language instruction

Episode: the task's environment is set to an initial state (object placements), 10 no-op steps with the gripper open
let objects settle, then the policy acts for at most MAX_STEPS[suite] steps; success is LIBERO's own goal predicate,
checked after every step. Initial states for the self-check are LIBERO's published ones (.pruned_init files); held-out
initial states are new placements drawn from each task's own placement distribution with a secret seed and kept only
if no object moves more than 1 cm while settling (sample_init_states).

The submission's objects (policy, planner, controller) each run in their own process (AgentProcess) and see only
what the episode loop sends them; the simulator stays in the evaluator's process.
Synchronous episodes (run_sync, E-VLA-003): reset(instruction); then act(obs) -> action every step.
Asynchronous episodes (run_async, E-RT-002): a planner and a controller in separate processes; see run_async.
"""
from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SETTLE, RES, CONTROL_HZ, STABLE_M = 10, 128, 20, 0.01
NOOP = np.array([0, 0, 0, 0, 0, 0, -1.0])


def quat2axisangle(q) -> np.ndarray:
    """robosuite convention: quaternion (x, y, z, w) -> axis * angle (as in LIBERO's dataset script)."""
    q = np.asarray(q, np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-9:
        return np.zeros(3)
    return q[:3] * 2.0 * math.acos(w) / den


def obs_dict(obs: dict, instruction: str) -> dict:
    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    return {"agentview_rgb": np.array(obs["agentview_image"], np.uint8),
            "eye_in_hand_rgb": np.array(obs["robot0_eye_in_hand_image"], np.uint8),
            "state": state.astype(np.float32), "instruction": instruction}


# ----------------------------------------------------------------------------------------------- environments
def task_info(suite: str, task_id: int) -> tuple:
    """(bddl file path, language instruction, the published initial states) of one task."""
    from libero.libero import benchmark, get_libero_path
    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return bddl, task.language, np.asarray(s.get_task_init_states(task_id))


def make_env(suite: str, task_id: int, seed: int):
    """(env, instruction, published initial states) for one task of a LIBERO suite."""
    from libero.libero.envs import OffScreenRenderEnv
    bddl, lang, inits = task_info(suite, task_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
    env.seed(seed)
    return env, lang, inits


def _object_qpos(env) -> np.ndarray:
    """Positions in qpos that do not belong to the robot (object joints)."""
    inner = env.env
    r = inner.robots[0]
    robot = set(r._ref_joint_pos_indexes) | set(getattr(r, "_ref_gripper_joint_pos_indexes", []) or [])
    q = np.asarray(inner.sim.data.qpos)
    return q[[i for i in range(len(q)) if i not in robot]]


def sample_init_states(suite: str, task_id: int, n: int, seed: int, max_tries: int = 20) -> np.ndarray:
    """n new initial states of a task: env.reset() draws object placements from the task's own placement
    distribution (seeded); a state is kept if no object coordinate moves by more than STABLE_M while settling."""
    env, _, _ = make_env(suite, task_id, seed)
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > n * max_tries:
            raise RuntimeError(f"{suite} task {task_id}: too many unstable placements")
        s = int(rng.integers(0, 2**31 - 1))
        np.random.seed(s)
        env.seed(s)
        env.reset()
        state = np.asarray(env.get_sim_state()).copy()
        env.set_init_state(state)
        before = _object_qpos(env)
        for _ in range(SETTLE):
            env.step(NOOP)
        if np.max(np.abs(_object_qpos(env) - before)) <= STABLE_M:
            out.append(state)
    env.close()
    return np.stack(out)


def pack_states(states: list) -> tuple:
    """Pad variable-length flattened states to one array; returns (array, lengths)."""
    L = max(len(s) for s in states)
    a = np.zeros((len(states), L), np.float64)
    for i, s in enumerate(states):
        a[i, :len(s)] = s
    return a, np.asarray([len(s) for s in states], np.int64)


def _start(env, init_state):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(NOOP)
    return obs


def _action(a) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1)
    if a.shape != (7,) or not np.all(np.isfinite(a)):
        raise ValueError(f"an action must be 7 finite numbers, got shape {a.shape}")
    return np.clip(a, -1.0, 1.0)


# ----------------------------------------------------------------------------------------------- agent processes
def _import(path: Path, name: str):
    for d in (path.parent, path.parent / "src"):
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _agent_main(conn, entry: str, factory: str, kwargs: dict, hook: str | None, hook_kwargs: dict, device: str,
                threads: int):
    """Child process holding one object made by the submission (a policy, planner or controller).

    Messages: ("call", method, args) -> (result, milliseconds); ("final",) -> the evaluator hook's after() (for
    example a digest of frozen weights); ("stop",)."""
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:                                   # noqa: BLE001 (numpy-only submissions)
        torch = None
    try:
        hk = _import(Path(hook), "abench_agent_hook") if hook else None
        extra = hk.before(**hook_kwargs) if hk else {}
        obj = getattr(_import(Path(entry), "abench_agent"), factory)(**kwargs, **extra)
        conn.send(("ok", None))
    except BaseException as e:                          # noqa: BLE001
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        try:
            if msg[0] == "final":
                conn.send(("ok", hk.after(**extra) if hk and hasattr(hk, "after") else None))
                continue
            _, method, args = msg
            t0 = time.perf_counter()
            res = getattr(obj, method)(*args)
            if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            if res is not None and not isinstance(res, dict):
                res = np.asarray(res, np.float64)
            conn.send(("ok", (res, ms)))
        except BaseException as e:                      # noqa: BLE001
            conn.send(("error", f"{type(e).__name__}: {e}"))


class AgentProcess:
    """An object of the submission in its own process: entry.py's factory(**kwargs) builds it. The episode loop and
    the simulator stay in the evaluator's process, so the object sees exactly what it is sent and nothing else."""

    def __init__(self, entry: Path, factory: str, kwargs: dict, device: str = "cpu", threads: int = 4,
                 hook: Path | None = None, hook_kwargs: dict | None = None, load_timeout: float = 600.0,
                 call_timeout: float = 120.0):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_agent_main, daemon=True,
                                args=(child, str(entry), factory, kwargs, str(hook) if hook else None,
                                      hook_kwargs or {}, device, threads))
        self.proc.start()
        self.call_timeout = call_timeout
        self._get(load_timeout)

    def _get(self, timeout: float):
        end = time.monotonic() + timeout
        while not self.conn.poll(min(1.0, max(end - time.monotonic(), 0.0))):
            if not self.proc.is_alive():
                raise RuntimeError(f"the submission's process exited (code {self.proc.exitcode})")
            if time.monotonic() >= end:
                self.close()
                raise TimeoutError("the submission's process did not answer in time")
        kind, val = self.conn.recv()
        if kind == "error":
            raise RuntimeError(f"submission error: {val}")
        return val

    def call(self, method: str, *args):
        """(result, milliseconds spent inside the call)."""
        self.conn.send(("call", method, args))
        return self._get(self.call_timeout)

    def final(self):
        self.conn.send(("final",))
        return self._get(self.call_timeout)

    def close(self) -> None:
        try:
            self.conn.send(("stop",))
        except (OSError, BrokenPipeError, ValueError):
            pass
        self.proc.join(10)
        if self.proc.is_alive():
            self.proc.kill()


class InProcess:
    """Same interface as AgentProcess for an object already in this process (self-check plumbing tests only)."""

    def __init__(self, obj):
        self.obj = obj

    def call(self, method: str, *args):
        t0 = time.perf_counter()
        res = getattr(self.obj, method)(*args)
        if res is not None and not isinstance(res, dict):
            res = np.asarray(res, np.float64)
        return res, (time.perf_counter() - t0) * 1e3

    def final(self):
        return None

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------------------------- episodes
def run_sync(policy, env, instruction: str, init_state, max_steps: int) -> dict:
    """policy: AgentProcess/InProcess of an object with reset(instruction) and act(obs) -> 7 numbers."""
    obs = _start(env, init_state)
    policy.call("reset", instruction)
    spent, actions = 0.0, []
    done = False
    for t in range(max_steps):
        a, ms = policy.call("act", obs_dict(obs, instruction))
        spent += ms
        a = _action(a)
        actions.append(a)
        obs, _, done, _ = env.step(a)
        if done:
            break
    n = len(actions)
    return {"success": int(bool(done)), "steps": n, "policy_ms": spent / max(n, 1)}


def delay_steps(injected: int, ms: float) -> int:
    """Control steps between issuing a plan and its delivery: the injected latency plus the measured planning time
    at 20 Hz (a plan that takes 0 < ms <= 50 adds one step)."""
    return int(injected + math.ceil(ms / (1000.0 / CONTROL_HZ)))


def _chunk(c) -> np.ndarray:
    c = np.asarray(c, np.float64)
    if c.ndim != 2 or c.shape[1] != 7 or len(c) == 0 or not np.all(np.isfinite(c)):
        raise ValueError("plan() must return a finite (H, 7) array with H >= 1")
    return c


def run_async(planner, controller, env, instruction: str, init_state, max_steps: int, delay: int) -> dict:
    """Asynchronous execution under inference latency (E-RT-002). planner and controller are separate processes.

    The robot never waits. Before the first step, controller.request(0) gives a request and the planner plans from
    the initial observation; that first chunk is delivered at once (controller.receive(chunk, 0, 0)). Then at every
    step t:
      1. a plan in flight whose delivery step has come is delivered: controller.receive(chunk, t_issue, t);
      2. if no plan is in flight, controller.request(t) returns None (no new plan now) or a request (a dict of numbers
         or arrays, for example the actions it will execute meanwhile); with a request, planner.plan(obs_t, t,
         request) is timed in the planner's process and its chunk is delivered at step t + delay_steps(delay, ms);
      3. the executed action is controller.act(t) -> 7 numbers.
    """
    obs = _start(env, init_state)
    planner.call("reset", instruction)
    controller.call("reset", instruction)
    req, _ = controller.call("request", 0)
    chunk, _ = planner.call("plan", obs_dict(obs, instruction), 0, req)
    controller.call("receive", _chunk(chunk), 0, 0)
    inflight, delays, plan_ms = None, [], []
    done, steps = False, 0
    for t in range(max_steps):
        if inflight is not None and t >= inflight[0]:
            controller.call("receive", inflight[1], inflight[2], t)
            inflight = None
        if inflight is None:
            req, _ = controller.call("request", t)
            if req is not None:
                chunk, ms = planner.call("plan", obs_dict(obs, instruction), t, req)
                d = delay_steps(delay, ms)
                plan_ms.append(ms)
                delays.append(d)
                inflight = (t + d, _chunk(chunk), t)
        a, _ = controller.call("act", t)
        obs, _, done, _ = env.step(_action(a))
        steps = t + 1
        if done:
            break
    return {"success": int(bool(done)), "steps": steps, "plans": len(plan_ms) + 1,
            "plan_ms": float(np.mean(plan_ms)) if plan_ms else 0.0,
            "mean_delay": float(np.mean(delays)) if delays else 0.0}


def checkpoint_numbers(ck: Path) -> int:
    """Numbers stored in the checkpoint's tensor files (.safetensors, .pt, .pth, .bin, .npz, .npy)."""
    n = 0
    for p in sorted(Path(ck).rglob("*")):
        if p.suffix == ".safetensors":
            import json as _json
            with open(p, "rb") as f:
                head = _json.loads(f.read(int.from_bytes(f.read(8), "little")))
            n += sum(int(np.prod(v["shape"])) for k, v in head.items() if k != "__metadata__")
        elif p.suffix in (".npz", ".npy"):
            z = np.load(p, allow_pickle=False)
            n += sum(int(z[k].size) for k in z.files) if p.suffix == ".npz" else int(z.size)
        elif p.suffix in (".pt", ".pth", ".bin"):
            import torch
            stack = [torch.load(p, map_location="cpu", weights_only=True)]
            while stack:
                o = stack.pop()
                if torch.is_tensor(o):
                    n += o.numel()
                elif isinstance(o, dict):
                    stack += list(o.values())
                elif isinstance(o, (list, tuple)):
                    stack += list(o)
    return n


def demo_files(data_dir, suite: str) -> list:
    """The suite's LIBERO demonstration files (robomimic hdf5: data/demo_i/{actions, obs/...})."""
    return sorted((Path(data_dir) / suite).glob("*_demo.hdf5"))
