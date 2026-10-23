"""X-PPO-001 environments: four small continuous-control tasks, vectorised in NumPy (evaluator and agent use this same
file; it is the only way the agent interacts with them).

  pendulum        gymnasium Pendulum-v1 (dynamics, reward, 200-step limit)                         obs 3, action 1
  mountain_car    gymnasium MountainCarContinuous-v0 (dynamics, reward, 999-step limit)              obs 2, action 1
  cartpole_swing  gymnasium CartPole-v1 physics with a continuous force of 10·a and the pole starting
                  down; reward (1 + cos θ) / 2 per step while |x| ≤ 2.4, episode ends when the cart
                  leaves the track or after 500 steps                                                obs 5, action 1
  acrobot         gymnasium Acrobot-v1 dynamics (RK4, dt 0.2) with a continuous torque in [-1, 1];
                  reward -1 per step until the tip rises above the bar, 500-step limit             obs 6, action 1

Actions are clipped to [-1, 1] (Pendulum's torque is 2·a). The dynamics follow gymnasium 1.0 (MIT licence), rewritten
for batches of environments.

`VecEnv(name, n, seed)`: `reset() -> obs (n, obs_dim)`; `step(action (n, 1)) -> obs, reward, terminated, truncated`.
Finished environments reset automatically: the returned `obs` is then the first observation of the next episode and
`env.final_obs` holds the last observation of the finished one. Each call to `step` uses `n` samples of the budget;
the training budget is enforced by `BudgetedEnv` in stages.py.
"""
from __future__ import annotations


import numpy as np

ENVS = ("pendulum", "mountain_car", "cartpole_swing", "acrobot")
OBS_DIM = {"pendulum": 3, "mountain_car": 2, "cartpole_swing": 5, "acrobot": 6}
ACT_DIM = 1
MAX_STEPS = {"pendulum": 200, "mountain_car": 999, "cartpole_swing": 500, "acrobot": 500}
RETURN_UPPER = {"pendulum": 0.0, "mountain_car": 100.0, "cartpole_swing": 500.0, "acrobot": 0.0}   # bounds for normalising


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class _Dyn:
    state_dim = 0

    def init(self, rng: np.random.Generator, n: int) -> np.ndarray: ...
    def obs(self, s: np.ndarray) -> np.ndarray: ...
    def step(self, s: np.ndarray, a: np.ndarray): ...           # -> new state, reward, terminated


class Pendulum(_Dyn):
    state_dim = 2
    g, m, l, dt, max_speed, max_torque = 10.0, 1.0, 1.0, 0.05, 8.0, 2.0

    def init(self, rng, n):
        return np.stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-1.0, 1.0, n)], axis=1)

    def obs(self, s):
        return np.stack([np.cos(s[:, 0]), np.sin(s[:, 0]), s[:, 1]], axis=1)

    def step(self, s, a):
        th, thdot = s[:, 0], s[:, 1]
        u = np.clip(a[:, 0], -1.0, 1.0) * self.max_torque
        cost = _wrap(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2
        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        return np.stack([newth, newthdot], axis=1), -cost, np.zeros(len(s), bool)


class MountainCar(_Dyn):
    state_dim = 2
    min_pos, max_pos, max_speed, goal, power = -1.2, 0.6, 0.07, 0.45, 0.0015

    def init(self, rng, n):
        return np.stack([rng.uniform(-0.6, -0.4, n), np.zeros(n)], axis=1)

    def obs(self, s):
        return s.copy()

    def step(self, s, a):
        pos, vel = s[:, 0].copy(), s[:, 1].copy()
        force = np.clip(a[:, 0], -1.0, 1.0)
        vel = np.clip(vel + force * self.power - 0.0025 * np.cos(3 * pos), -self.max_speed, self.max_speed)
        pos = np.clip(pos + vel, self.min_pos, self.max_pos)
        vel = np.where((pos == self.min_pos) & (vel < 0), 0.0, vel)
        term = (pos >= self.goal) & (vel >= 0.0)
        reward = np.where(term, 100.0, 0.0) - 0.1 * force ** 2
        return np.stack([pos, vel], axis=1), reward, term


class CartPoleSwing(_Dyn):
    state_dim = 4
    gravity, masscart, masspole, length, force_mag, tau, x_limit = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02, 2.4

    def init(self, rng, n):
        s = rng.uniform(-0.05, 0.05, (n, 4))
        s[:, 2] += np.pi                                        # pole hanging down
        return s

    def obs(self, s):
        return np.stack([s[:, 0], s[:, 1], np.cos(s[:, 2]), np.sin(s[:, 2]), s[:, 3]], axis=1)

    def step(self, s, a):
        x, x_dot, th, th_dot = s.T
        force = np.clip(a[:, 0], -1.0, 1.0) * self.force_mag
        total = self.masspole + self.masscart
        pml = self.masspole * self.length
        ct, st = np.cos(th), np.sin(th)
        temp = (force + pml * th_dot ** 2 * st) / total
        thacc = (self.gravity * st - ct * temp) / (self.length * (4.0 / 3.0 - self.masspole * ct ** 2 / total))
        xacc = temp - pml * thacc * ct / total
        x, x_dot = x + self.tau * x_dot, x_dot + self.tau * xacc
        th, th_dot = th + self.tau * th_dot, th_dot + self.tau * thacc
        ns = np.stack([x, x_dot, th, th_dot], axis=1)
        out = np.abs(x) > self.x_limit
        reward = np.where(out, 0.0, (1.0 + np.cos(th)) / 2.0)
        return ns, reward, out


class Acrobot(_Dyn):
    state_dim = 4
    dt, l1, m1, m2, lc1, lc2, i1, i2, g = 0.2, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 9.8
    max_v1, max_v2 = 4 * np.pi, 9 * np.pi

    def init(self, rng, n):
        return rng.uniform(-0.1, 0.1, (n, 4))

    def obs(self, s):
        return np.stack([np.cos(s[:, 0]), np.sin(s[:, 0]), np.cos(s[:, 1]), np.sin(s[:, 1]), s[:, 2], s[:, 3]], axis=1)

    def _dsdt(self, s, a):
        m1, m2, l1, lc1, lc2, i1, i2, g = self.m1, self.m2, self.l1, self.lc1, self.lc2, self.i1, self.i2, self.g
        t1, t2, dt1, dt2 = s.T
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(t2)) + i1 + i2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(t2)) + i2
        phi2 = m2 * lc2 * g * np.cos(t1 + t2 - np.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * dt2 ** 2 * np.sin(t2) - 2 * m2 * l1 * lc2 * dt2 * dt1 * np.sin(t2)
                + (m1 * lc1 + m2 * l1) * g * np.cos(t1 - np.pi / 2) + phi2)
        ddt2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dt1 ** 2 * np.sin(t2) - phi2) / (m2 * lc2 ** 2 + i2 - d2 ** 2 / d1)
        ddt1 = -(d2 * ddt2 + phi1) / d1
        return np.stack([dt1, dt2, ddt1, ddt2], axis=1)

    def step(self, s, a):
        u = np.clip(a[:, 0], -1.0, 1.0)
        h = self.dt
        k1 = self._dsdt(s, u)
        k2 = self._dsdt(s + h / 2 * k1, u)
        k3 = self._dsdt(s + h / 2 * k2, u)
        k4 = self._dsdt(s + h * k3, u)
        ns = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        ns[:, 0], ns[:, 1] = _wrap(ns[:, 0]), _wrap(ns[:, 1])
        ns[:, 2] = np.clip(ns[:, 2], -self.max_v1, self.max_v1)
        ns[:, 3] = np.clip(ns[:, 3], -self.max_v2, self.max_v2)
        term = (-np.cos(ns[:, 0]) - np.cos(ns[:, 1] + ns[:, 0])) > 1.0
        return ns, np.where(term, 0.0, -1.0), term


DYNAMICS = {"pendulum": Pendulum, "mountain_car": MountainCar, "cartpole_swing": CartPoleSwing, "acrobot": Acrobot}


class VecEnv:
    def __init__(self, name: str, n: int, seed: int, init_states: np.ndarray | None = None):
        if name not in DYNAMICS:
            raise KeyError(name)
        self.name, self.n = name, int(n)
        self.dyn = DYNAMICS[name]()
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.act_dim, self.max_steps = OBS_DIM[name], ACT_DIM, MAX_STEPS[name]
        self._init = None if init_states is None else np.asarray(init_states, np.float64)
        self.s = None
        self.t = np.zeros(self.n, int)
        self.final_obs = None

    def reset(self) -> np.ndarray:
        self.s = self._init.copy() if self._init is not None else self.dyn.init(self.rng, self.n)
        self.t[:] = 0
        return self.dyn.obs(self.s).astype(np.float32)

    def step(self, action):
        a = np.asarray(action, np.float64).reshape(self.n, ACT_DIM)
        if not np.isfinite(a).all():
            raise ValueError("non-finite action")
        ns, r, term = self.dyn.step(self.s, a)
        self.t += 1
        trunc = (self.t >= self.max_steps) & ~term
        done = term | trunc
        obs = self.dyn.obs(ns)
        self.final_obs = obs.astype(np.float32).copy()
        if done.any():
            fresh = self.dyn.init(self.rng, int(done.sum()))
            ns[done] = fresh
            self.t[done] = 0
            obs[done] = self.dyn.obs(fresh)
        self.s = ns
        return obs.astype(np.float32), r.astype(np.float64), term, trunc


def rollout_returns(act, name: str, init_states: np.ndarray) -> np.ndarray:
    """Undiscounted return of one episode from each initial state, with the deterministic policy act(obs) -> action."""
    n = len(init_states)
    env = VecEnv(name, n, 0, init_states=init_states)
    obs = env.reset()
    ret, alive = np.zeros(n), np.ones(n, bool)
    for _ in range(env.max_steps):
        a = np.asarray(act(obs), np.float64).reshape(n, ACT_DIM)
        if not np.isfinite(a).all():
            raise ValueError("non-finite action")
        obs, r, term, trunc = env.step(a)
        ret += np.where(alive, r, 0.0)
        alive &= ~(term | trunc)
        if not alive.any():
            break
    return ret


def random_policy(seed: int):
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, (len(obs), ACT_DIM))


