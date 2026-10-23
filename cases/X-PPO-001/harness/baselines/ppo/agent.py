"""Baseline: PPO with clipped surrogate objective (Schulman et al. 2017) in the configuration of CleanRL's
ppo_continuous_action.py: one environment, rollouts of 2,048 steps, 10 epochs of 32 minibatches, Adam 3e-4 annealed
linearly, gamma 0.99, GAE lambda 0.95, clip 0.2, value coefficient 0.5, gradient norm 0.5, tanh MLPs 64x64,
state-independent log standard deviation, running observation normalisation (clipped at 10) and reward scaling by the
running standard deviation of the discounted return. Episode ends (termination or time limit) cut the return."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape)
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot


def layer(i, o, std=np.sqrt(2)):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.constant_(lin.bias, 0.0)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, 1, 1.0))
        self.mu = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, act_dim, 0.01))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp().expand_as(self.mu(x)))


class Policy:
    def __init__(self, net, rms):
        self.net, self.mean, self.std = net, rms.mean.copy(), np.sqrt(rms.var + 1e-8)

    def act(self, obs):
        x = np.clip((np.asarray(obs, np.float64) - self.mean) / self.std, -10, 10)
        with torch.no_grad():
            return np.clip(self.net.mu(torch.as_tensor(x, dtype=torch.float32)).numpy(), -1, 1)


def train(make_env, spec, seed, n_envs=1, n_steps=2048, epochs=10, minibatches=32, lr=3e-4, gamma=0.99, lam=0.95,
          clip=0.2, vf_coef=0.5, max_grad=0.5, bootstrap_truncation=False, target_kl=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    env = make_env(n_envs)
    net = ActorCritic(spec["obs_dim"], spec["act_dim"])
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    obs_rms, ret_rms = RunningMeanStd((spec["obs_dim"],)), RunningMeanStd(())
    ret_acc = np.zeros(n_envs)
    norm = lambda o: np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)   # noqa: E731
    o = env.reset()
    obs_rms.update(o)
    iters = spec["budget"] // (n_envs * n_steps)
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = lr * (1.0 - it / iters)
        B_o, B_a, B_lp, B_r, B_d, B_v, B_bv = [], [], [], [], [], [], []
        for _ in range(n_steps):
            x = torch.as_tensor(norm(o), dtype=torch.float32)
            with torch.no_grad():
                d = net.dist(x)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
                v = net.critic(x).squeeze(-1)
            o2, r, term, trunc = env.step(a.numpy())
            ret_acc = ret_acc * gamma + r
            ret_rms.update(ret_acc)
            done = term | trunc
            ret_acc[done] = 0.0
            boot = np.zeros(n_envs)
            if bootstrap_truncation and trunc.any():
                with torch.no_grad():
                    fo = torch.as_tensor(norm(env.final_obs), dtype=torch.float32)
                    boot = np.where(trunc, net.critic(fo).squeeze(-1).numpy(), 0.0)
            B_o.append(x); B_a.append(a); B_lp.append(lp); B_v.append(v)
            B_r.append(r / np.sqrt(ret_rms.var + 1e-8)); B_d.append(done); B_bv.append(boot)
            obs_rms.update(o2)
            o = o2
        with torch.no_grad():
            last_v = net.critic(torch.as_tensor(norm(o), dtype=torch.float32)).squeeze(-1).numpy()
        V = torch.stack(B_v).numpy()
        adv = np.zeros_like(V)
        gae = np.zeros(n_envs)
        for t in reversed(range(n_steps)):
            nv = last_v if t == n_steps - 1 else V[t + 1]
            nonterm = 1.0 - B_d[t]
            delta = B_r[t] + gamma * (nv * nonterm + B_bv[t] * B_d[t]) - V[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + V
        bo, ba = torch.cat(B_o), torch.cat(B_a)
        blp = torch.cat(B_lp)
        badv, bret = torch.as_tensor(adv.reshape(-1), dtype=torch.float32), torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        n = len(bo)
        mb = n // minibatches
        for _ in range(epochs):
            perm = torch.randperm(n)
            kls = []
            for k in range(minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                d = net.dist(bo[idx])
                lp = d.log_prob(ba[idx]).sum(-1)
                ratio = (lp - blp[idx]).exp()
                a_ = badv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - clip, 1 + clip)).mean()
                vl = 0.5 * ((net.critic(bo[idx]).squeeze(-1) - bret[idx]) ** 2).mean()
                loss = pg + vf_coef * vl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad)
                opt.step()
                with torch.no_grad():
                    kls.append(float(((ratio - 1) - (lp - blp[idx])).mean()))
            if target_kl is not None and np.mean(kls) > target_kl:
                break
    return Policy(net, obs_rms)
