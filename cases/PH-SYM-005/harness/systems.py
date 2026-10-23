"""PH-SYM-005 system generator and term library (evaluator and agent use this same file).

Every system is an autonomous ODE  dx/dt = f(x)  in d = 2..4 variables whose right-hand side is a sparse linear
combination of the terms of LIBRARY(d): the constant, all monomials of degree 1..3, and sin(x_j), cos(x_j).
A system is fully described by its coefficient matrix C of shape (n_terms(d), d): dx_i/dt = sum_k C[k, i] * term_k(x).

sample_system(rng) draws one system: a family (a parametric family of a known model, or a random sparse polynomial
system with cubic damping), random parameters, a random permutation of the variables and (for variables that do not
appear inside sin/cos) a random rescaling. The family is not part of the output. The evaluator's held-out systems are
drawn with a secret seed; you can draw as many development systems as you like with your own seeds.
"""
from __future__ import annotations

import itertools

import numpy as np

MAX_D = 4
FAMILIES = ("damped_oscillator", "duffing", "van_der_pol", "lotka_volterra", "pendulum", "lorenz", "rossler",
            "brusselator", "fitzhugh_nagumo", "thomas", "hopf", "random_sparse")
N_TRAJ, N_SAMPLES, N_TEST = 8, 1000, 4
NOISE_LEVELS = (0.0, 0.002, 0.005, 0.01)          # Gaussian measurement noise, fraction of each variable's std


def library(d: int) -> list[tuple]:
    """Term descriptors in a fixed order: ("poly", exponents) or ("sin", j) / ("cos", j)."""
    terms = [("poly", (0,) * d)]
    for deg in (1, 2, 3):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            e = [0] * d
            for j in combo:
                e[j] += 1
            terms.append(("poly", tuple(e)))
    terms += [("sin", j) for j in range(d)] + [("cos", j) for j in range(d)]
    return terms


def term_names(d: int) -> list[str]:
    out = []
    for kind, a in library(d):
        if kind == "poly":
            parts = [f"x{j}" + (f"^{e}" if e > 1 else "") for j, e in enumerate(a) if e]
            out.append("*".join(parts) if parts else "1")
        else:
            out.append(f"{kind}(x{a})")
    return out


def n_terms(d: int) -> int:
    return len(library(d))


def eval_library(x: np.ndarray, d: int | None = None) -> np.ndarray:
    """Theta(x): (n, n_terms(d)) matrix of all library terms evaluated at the rows of x (n, d)."""
    x = np.atleast_2d(np.asarray(x, np.float64))
    d = x.shape[1] if d is None else d
    cols = []
    for kind, a in library(d):
        if kind == "poly":
            cols.append(np.prod(x ** np.asarray(a), axis=1))
        elif kind == "sin":
            cols.append(np.sin(x[:, a]))
        else:
            cols.append(np.cos(x[:, a]))
    return np.stack(cols, axis=1)


def rhs(coef: np.ndarray):
    coef = np.asarray(coef, np.float64)
    d = coef.shape[1]

    def f(_t, x):
        return (eval_library(x[None, :], d) @ coef)[0]
    return f


def _idx(d: int) -> dict:
    return {t: k for k, t in enumerate(library(d))}


def _mono(d: int, *vars_: int) -> tuple:
    e = [0] * d
    for j in vars_:
        e[j] += 1
    return ("poly", tuple(e))


def _family(name: str, rng: np.random.Generator):
    """Returns (d, list of (equation, term, coefficient), initial-condition sampler, duration, trig variables)."""
    u = lambda lo, hi: float(rng.uniform(lo, hi))       # noqa: E731
    M = _mono
    if name == "damped_oscillator":
        a, b = u(0.5, 4.0), u(0.05, 0.5)
        terms = [(0, M(2, 1), 1.0), (1, M(2, 0), -a), (1, M(2, 1), -b)]
        return 2, terms, lambda: rng.uniform(-2, 2, 2), 20.0, ()
    if name == "duffing":
        a, b, c = u(0.5, 2.0), u(0.5, 2.0), u(0.05, 0.3)
        terms = [(0, M(2, 1), 1.0), (1, M(2, 0), -a), (1, M(2, 0, 0, 0), -b), (1, M(2, 1), -c)]
        return 2, terms, lambda: rng.uniform(-2, 2, 2), 20.0, ()
    if name == "van_der_pol":
        mu = u(0.5, 2.5)
        terms = [(0, M(2, 1), 1.0), (1, M(2, 1), mu), (1, M(2, 0, 0, 1), -mu), (1, M(2, 0), -1.0)]
        return 2, terms, lambda: rng.uniform(-3, 3, 2), 20.0, ()
    if name == "lotka_volterra":
        a, b, c, e = u(0.5, 1.5), u(0.3, 1.0), u(0.5, 1.5), u(0.3, 1.0)
        terms = [(0, M(2, 0), a), (0, M(2, 0, 1), -b), (1, M(2, 1), -c), (1, M(2, 0, 1), e)]
        return 2, terms, lambda: rng.uniform(0.5, 3.0, 2), 20.0, ()
    if name == "pendulum":
        g, b = u(1.0, 4.0), u(0.05, 0.4)
        terms = [(0, M(2, 1), 1.0), (1, ("sin", 0), -g), (1, M(2, 1), -b)]
        return 2, terms, lambda: np.array([rng.uniform(-2.8, 2.8), rng.uniform(-1.0, 1.0)]), 20.0, (0,)
    if name == "lorenz":
        s, r, bb = u(8.0, 12.0), u(20.0, 32.0), u(2.0, 3.0)
        terms = [(0, M(3, 0), -s), (0, M(3, 1), s), (1, M(3, 0), r), (1, M(3, 1), -1.0), (1, M(3, 0, 2), -1.0),
                 (2, M(3, 0, 1), 1.0), (2, M(3, 2), -bb)]
        return 3, terms, lambda: rng.uniform([-15, -15, 10], [15, 15, 35]), 5.0, ()
    if name == "rossler":
        a, b, c = u(0.1, 0.25), u(0.1, 0.3), u(4.0, 7.0)
        terms = [(0, M(3, 1), -1.0), (0, M(3, 2), -1.0), (1, M(3, 0), 1.0), (1, M(3, 1), a), (2, M(3), b),
                 (2, M(3, 0, 2), 1.0), (2, M(3, 2), -c)]
        return 3, terms, lambda: rng.uniform([-6, -6, 0], [6, 6, 2]), 40.0, ()
    if name == "brusselator":
        A, B = u(0.8, 1.5), u(2.0, 3.5)
        terms = [(0, M(2), A), (0, M(2, 0), -(B + 1)), (0, M(2, 0, 0, 1), 1.0), (1, M(2, 0), B), (1, M(2, 0, 0, 1), -1.0)]
        return 2, terms, lambda: rng.uniform(0.2, 3.0, 2), 20.0, ()
    if name == "fitzhugh_nagumo":
        a, b, eps, I = u(0.6, 0.8), u(0.7, 0.9), u(0.06, 0.12), u(0.3, 0.6)
        terms = [(0, M(2, 0), 1.0), (0, M(2, 0, 0, 0), -1.0 / 3.0), (0, M(2, 1), -1.0), (0, M(2), I),
                 (1, M(2, 0), eps), (1, M(2), eps * a), (1, M(2, 1), -eps * b)]
        return 2, terms, lambda: rng.uniform(-2.5, 2.5, 2), 60.0, ()
    if name == "thomas":
        bb = u(0.15, 0.25)
        terms = [(0, ("sin", 1), 1.0), (0, M(3, 0), -bb), (1, ("sin", 2), 1.0), (1, M(3, 1), -bb),
                 (2, ("sin", 0), 1.0), (2, M(3, 2), -bb)]
        return 3, terms, lambda: rng.uniform(-3, 3, 3), 60.0, (0, 1, 2)
    if name == "hopf":
        mu, om = u(0.2, 1.0), u(0.5, 2.0)
        terms = [(0, M(2, 0), mu), (0, M(2, 1), -om), (0, M(2, 0, 0, 0), -1.0), (0, M(2, 0, 1, 1), -1.0),
                 (1, M(2, 0), om), (1, M(2, 1), mu), (1, M(2, 0, 0, 1), -1.0), (1, M(2, 1, 1, 1), -1.0)]
        return 2, terms, lambda: rng.uniform(-2, 2, 2) * rng.choice([0.2, 1.0]), 20.0, ()
    if name == "random_sparse":
        d = int(rng.integers(3, MAX_D + 1))
        terms = []
        # sparse linear part with a rotation so that trajectories keep moving, cubic damping on every variable
        for i in range(d):
            terms.append((i, M(d, i), float(rng.choice([-1, 1]) * u(0.1, 0.3))))
            j = int((i + 1) % d)
            terms.append((i, M(d, j), float(rng.choice([-1, 1]) * u(0.5, 2.0))))
            terms.append((i, M(d, i, i, i), -u(0.1, 0.5)))
        quad = [M(d, *c) for c in itertools.combinations_with_replacement(range(d), 2)]
        slots = [(i, q) for i in range(d) for q in range(len(quad))]
        for s in rng.choice(len(slots), size=int(rng.integers(2, 5)), replace=False):
            i, q = slots[int(s)]
            terms.append((i, quad[q], float(rng.choice([-1, 1]) * u(0.3, 1.0))))
        return d, terms, lambda: rng.uniform(-2, 2, d), 20.0, ()
    raise KeyError(name)


def _coef(d: int, terms) -> np.ndarray:
    ix = _idx(d)
    C = np.zeros((n_terms(d), d))
    for i, t, c in terms:
        C[ix[t], i] += c
    return C


def transform(C: np.ndarray, perm: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Coefficients in new variables z with x[perm[j]] = scale[j] * z[j] (trig variables must have scale 1)."""
    d = C.shape[1]
    lib, ix = library(d), _idx(d)
    out = np.zeros_like(C)
    inv = np.argsort(perm)                          # z index of each x variable
    for k, (kind, a) in enumerate(lib):
        for i in range(d):
            c = C[k, i]
            if c == 0.0:
                continue
            zi = inv[i]
            if kind == "poly":
                e = np.zeros(d, int)
                fac = 1.0
                for j, p in enumerate(a):
                    if p:
                        e[inv[j]] += p
                        fac *= scale[inv[j]] ** p
                t = ("poly", tuple(int(v) for v in e))
            else:
                t = (kind, int(inv[a]))
                fac = 1.0
            out[ix[t], zi] += c * fac / scale[zi]
    return out


def simulate(C: np.ndarray, x0: np.ndarray, duration: float, n: int = N_SAMPLES, bound: float = 1e4):
    """Integrates dx/dt = Theta(x) C from x0; returns (t, X) with n samples, or None if the solution leaves |x| < bound
    or the integrator fails."""
    from scipy.integrate import solve_ivp
    t = np.linspace(0.0, duration, n)

    def blow(_t, x):
        return bound - float(np.max(np.abs(x)))
    blow.terminal = True
    sol = solve_ivp(rhs(C), (0.0, duration), np.asarray(x0, np.float64), t_eval=t, method="LSODA", rtol=1e-8,
                    atol=1e-10, events=blow)
    if sol.status != 0 or sol.y.shape[1] != n or not np.isfinite(sol.y).all():
        return None
    return t, sol.y.T


def identifiable(C: np.ndarray, X: np.ndarray, rel: float = 0.01, min_cond: float = 1e-6) -> bool:
    """The library is well conditioned on the clean trajectories X (n, d): the smallest singular value of the
    column-normalised term matrix is at least `min_cond` times the largest; and every true term contributes at least
    `rel` of the spread of its equation's right-hand side, so that its presence is visible in the data."""
    th = eval_library(X, C.shape[1])
    thn = th / (np.sqrt((th ** 2).mean(axis=0)) + 1e-12)
    sv = np.linalg.svd(thn, compute_uv=False)
    if sv[-1] < min_cond * sv[0]:
        return False
    for i in range(C.shape[1]):
        f = th @ C[:, i]
        sf = float(f.std()) + 1e-12
        for k in np.flatnonzero(C[:, i]):
            if float(np.abs(C[k, i]) * th[:, k].std()) < rel * sf and k != 0:
                return False
    return True


def sample_system(rng: np.random.Generator, family: str | None = None) -> dict:
    """One system: coefficient matrix, N_TRAJ noisy training trajectories and N_TEST held-out initial conditions."""
    for _ in range(200):
        fam = family or FAMILIES[int(rng.integers(len(FAMILIES)))]
        d, terms, ic, duration, trig = _family(fam, rng)
        C0 = _coef(d, terms)
        perm = rng.permutation(d)
        scale = np.where(np.isin(perm, trig), 1.0, np.exp(rng.uniform(np.log(0.5), np.log(2.0), d)))
        C = transform(C0, perm, scale)
        trajs, test_x0 = [], []
        ok = True
        for _k in range(N_TRAJ + N_TEST):
            x0 = ic()
            z0 = np.asarray(x0)[perm] / scale
            r = simulate(C, z0, duration)
            if r is None or np.ptp(r[1], axis=0).min() < 1e-3:
                ok = False
                break
            (trajs if len(trajs) < N_TRAJ else test_x0).append(r if len(trajs) < N_TRAJ else z0)
        if not ok:
            continue
        if not identifiable(C, np.concatenate([x for _, x in trajs])):
            continue
        noise = float(NOISE_LEVELS[int(rng.integers(len(NOISE_LEVELS)))])
        t = trajs[0][0]
        X = np.stack([x for _, x in trajs])                       # (N_TRAJ, N_SAMPLES, d)
        sd = X.reshape(-1, d).std(axis=0)
        Xn = X + rng.normal(size=X.shape) * noise * sd
        return {"family": fam, "d": d, "coef": C, "t": t, "x": Xn, "x_clean": X, "noise": noise,
                "test_x0": np.stack(test_x0), "duration": duration}
    raise RuntimeError("could not draw a bounded system")


def sample_set(seed: int, n: int, families: list | None = None) -> list[dict]:
    """n systems from one seed. With families=None, no known family repeats and the rest (at least n // 3) are
    random_sparse systems."""
    rng = np.random.default_rng(seed)
    if families is None:
        known = [f for f in FAMILIES if f != "random_sparse"]
        n_known = min(len(known), n - n // 3)
        pick = list(rng.choice(known, size=n_known, replace=False)) + ["random_sparse"] * (n - n_known)
        families = [str(f) for f in rng.permutation(pick)]
    return [sample_system(rng, f) for f in families]


def support(C: np.ndarray, tol: float = 0.0) -> np.ndarray:
    return np.abs(np.asarray(C)) > tol


def describe(C: np.ndarray, digits: int = 3) -> list[str]:
    d = C.shape[1]
    names = term_names(d)
    eqs = []
    for i in range(d):
        parts = [f"{C[k, i]:+.{digits}g}·{names[k]}" for k in range(len(names)) if C[k, i] != 0]
        eqs.append(f"dx{i}/dt = " + (" ".join(parts) if parts else "0"))
    return eqs


assert n_terms(4) == 43 and n_terms(2) == 14 and n_terms(3) == 26, (n_terms(2), n_terms(3), n_terms(4))
