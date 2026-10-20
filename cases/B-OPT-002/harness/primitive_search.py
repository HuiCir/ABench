"""Restricted, replayable Lion-primitive searcher.

Search candidates are not arbitrary Python: each candidate is represented only by a frozen program dict, and the interpreter supports only
EMA, add/sub/mul, abs/sign/safe_div/normalize_rms and constants. The search script uses it to validate candidates and
compute proxy scores on the fixed gradient stream; the same module is also used to replay the search from search_log.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

GRAMMAR_VERSION = "lion-primitive-v1"
ALLOWED_OPS = {"identity", "sign", "abs", "sqrt", "normalize_rms", "add", "sub", "mul", "safe_div", "min", "max"}
ALLOWED_TERMS = {"g", "m_fast", "m_slow", "delta", "zero", "half"}


def _stable_id(program: dict) -> str:
    raw = json.dumps(program, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-30))


def _eval(expr: Any, env: Dict[str, np.ndarray], hp: Dict[str, float]) -> np.ndarray:
    if isinstance(expr, str):
        if expr in env:
            return env[expr]
        if expr == "zero":
            return np.zeros_like(env["g"])
        if expr == "half":
            return np.asarray(0.5, dtype=env["g"].dtype)
        if expr in hp:
            return np.asarray(hp[expr], dtype=env["g"].dtype)
        raise ValueError(f"unknown term: {expr}")
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError("expression must be a one-key AST node")
    op, args = next(iter(expr.items()))
    if op not in ALLOWED_OPS:
        raise ValueError(f"op {op!r} not in frozen grammar")
    if op in {"identity", "sign", "abs", "sqrt", "normalize_rms"}:
        x = _eval(args, env, hp)
        if op == "identity": return x
        if op == "sign": return np.sign(x)
        if op == "abs": return np.abs(x)
        if op == "sqrt": return np.sqrt(np.maximum(x, 0.0) + 1e-30)
        return x / (_rms(x) + 1e-12)
    if not isinstance(args, list) or len(args) != 2:
        raise ValueError(f"{op} expects two arguments")
    a, b = (_eval(args[0], env, hp), _eval(args[1], env, hp))
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "safe_div": return a / (np.abs(b) + 1e-12)
    if op == "min": return np.minimum(a, b)
    return np.maximum(a, b)


def validate_program(program: dict) -> dict:
    if not isinstance(program, dict):
        raise ValueError("program must be a mapping")
    required = {"beta_fast", "beta_slow", "alpha", "tau", "update"}
    missing = required - set(program)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    for k in ("beta_fast", "beta_slow"):
        if not 0.0 <= float(program[k]) < 1.0:
            raise ValueError(f"{k} outside [0,1)")
    for k in ("alpha", "tau"):
        if not math.isfinite(float(program[k])) or float(program[k]) < 0:
            raise ValueError(f"{k} must be finite and nonnegative")
    # Walk the AST and reject unknown operations/terms.
    def walk(x):
        if isinstance(x, str):
            if x not in ALLOWED_TERMS and x not in {"beta_fast", "beta_slow", "alpha", "tau"}:
                raise ValueError(f"unknown term {x!r}")
            return 1
        if not isinstance(x, dict) or len(x) != 1:
            raise ValueError("invalid AST node")
        op, args = next(iter(x.items()))
        if op not in ALLOWED_OPS:
            raise ValueError(f"unknown op {op!r}")
        if op in {"identity", "sign", "abs", "sqrt", "normalize_rms"}:
            return 1 + walk(args)
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("binary node needs two args")
        return 1 + walk(args[0]) + walk(args[1])
    nodes = walk(program["update"])
    if nodes > 40:
        raise ValueError("program exceeds 40 AST nodes")
    canonical = dict(program)
    canonical.pop("program_id", None)
    return {**canonical, "program_id": _stable_id(canonical)}


def apply_program(program: dict, m_fast: np.ndarray, m_slow: np.ndarray, g: np.ndarray) -> np.ndarray:
    p = validate_program(program)
    hp = {k: float(p[k]) for k in ("beta_fast", "beta_slow", "alpha", "tau")}
    env = {"g": g, "m_fast": m_fast, "m_slow": m_slow, "delta": m_fast - m_slow}
    return _eval(p["update"], env, hp)


def replay(program: dict, streams: Iterable[np.ndarray]) -> List[np.ndarray]:
    p = validate_program(program)
    seq = list(streams)
    if not seq:
        return []
    mf = np.zeros_like(seq[0]); ms = np.zeros_like(seq[0]); out = []
    for g in seq:
        mf = p["beta_fast"] * mf + (1 - p["beta_fast"]) * g
        ms = p["beta_slow"] * ms + (1 - p["beta_slow"]) * g
        out.append(apply_program(p, mf, ms, g))
    return out


def proxy_score(program: dict, streams: List[np.ndarray], reference: List[np.ndarray] | None = None) -> float:
    """Fixed-stream proxy: direction, spectral-distribution stability and magnitude stability; higher is better."""
    updates = replay(program, streams)
    if not updates or any(not np.all(np.isfinite(x)) for x in updates):
        return -1e9
    # A good optimizer follows the negative gradient without becoming a raw sign clone.
    direction, amp, profiles = [], [], []
    for u, g in zip(updates, streams):
        c = float(np.sum(u * (-g)) / (np.linalg.norm(u) * np.linalg.norm(g) + 1e-12))
        direction.append((c + 1.0) / 2.0)
        amp.append(float(np.log1p(np.linalg.norm(u)) / (1 + np.log1p(np.linalg.norm(g)))))
        sv = np.linalg.svd(u, compute_uv=False)
        profiles.append(sv / (sv.sum() + 1e-12))
    spectral_d = [float(np.abs(a - b).sum()) for a, b in zip(profiles[1:], profiles[:-1])]
    spectral = 1.0 - min(1.0, float(np.mean(spectral_d)) if spectral_d else 0.0)
    amplitude = max(0.0, 1.0 - float(np.std(amp)))
    score = 0.45 * float(np.mean(direction)) + 0.35 * spectral + 0.20 * amplitude
    return float(score)


def fixed_stream(seed: int = 17, steps: int = 64) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    seq = []
    for t in range(steps):
        signal = rng.standard_normal((32, 16)) * (1.0 + 0.2 * math.sin(t / 8))
        low_rank = np.outer(rng.standard_normal(32), rng.standard_normal(16)) * 0.08
        seq.append((signal + low_rank).astype(np.float64))
    return seq
