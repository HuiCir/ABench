"""The primitive language of B-OPT-002 (evaluator code, agent-visible): an optimizer is a short program over
element-wise and tensor primitives, compiled into a B-OPT-001 FunctionalRule that runs on NumPy and PyTorch.

program.yaml:
  state: [m, v]                 # at most 2 state tensors, shaped like the parameter, initialised to zero
  constants: {b1: 0.9, b2: 0.99, wd: 0.1}   # at most 6 named real constants (your hyperparameters)
  lr: 3.0e-4                    # peak learning rate of 2D parameters (1D parameters get 0.15 x, as in B-OPT-001)
  program:                      # at most 20 assignments, run in order at every step for every parameter
    - c = interp(m, g, b1)
    - m = interp(m, g, b2)
    - update = add(sign(c), mul(wd, p))

Each assignment is `name = expression`. An expression is a call of a primitive, a name or a number. Names: the inputs
g (gradient) and p (parameter), the state tensors (their value from the previous step until reassigned), constants,
and earlier assignments. The program must assign `update`; the parameter moves by -lr * schedule * update. State
tensors keep the last value assigned to them. Expressions may nest; the whole program may contain at most 40
primitive calls.

Primitives (x, y tensors or scalars; b a constant or number):
  add(x, y)  sub(x, y)  mul(x, y)  div(x, y) = x / (|y| + 1e-8) * sign*(y)   max(x, y)  min(x, y)
  neg(x)  abs(x)  sign(x)  square(x)  sqrt(x) = sqrt(|x|)  rsqrt(x) = 1 / (sqrt(|x|) + 1e-8)
  clip(x, b) = clamp to [-b, b]   interp(x, y, b) = b x + (1 - b) y   bias(b) = 1 - b^step (scalar)
  rms(x) (scalar)   normalize(x) = x / (rms(x) + 1e-8)   orth(x) = Newton-Schulz orthogonalisation of a matrix
  (identity for 1D parameters)   rownorm(x) = each row divided by its RMS (identity for 1D)
sign*(y) is the sign of y with sign(0) = 1. Every primitive is defined on both backends by optim_api.Ops.
"""
from __future__ import annotations

import re
from typing import Any

from optim_api import FunctionalRule, Ops, newton_schulz5

MAX_STATE, MAX_CONST, MAX_LINES, MAX_CALLS = 2, 6, 20, 40
INPUTS = ("g", "p")
UNARY = {"neg", "abs", "sign", "square", "sqrt", "rsqrt", "rms", "normalize", "orth", "rownorm"}
BINARY = {"add", "sub", "mul", "div", "max", "min", "clip"}
TERNARY = {"interp"}
SCALAR_ONLY = {"bias"}
NAME = re.compile(r"^[a-z_][a-z0-9_]*$")
TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+(?:\.\d*)?(?:e-?\d+)?)|(?P<name>[a-z_][a-z0-9_]*)|(?P<p>[(),]))", re.I)


def parse_expr(src: str):
    """'add(sign(c), mul(wd, p))' -> ('add', [('sign', ['c']), ('mul', ['wd', 'p'])]); numbers become floats."""
    toks, pos = [], 0
    while pos < len(src):
        m = TOKEN.match(src, pos)
        if not m or m.end() == pos:
            if src[pos:].strip() == "":
                break
            raise ValueError(f"cannot parse {src[pos:]!r}")
        pos = m.end()
        toks.append(("num", float(m.group("num"))) if m.group("num") else ("name", m.group("name")) if m.group("name")
                    else ("p", m.group("p")))
    i = 0

    def node():
        nonlocal i
        if i >= len(toks):
            raise ValueError("unexpected end of expression")
        kind, v = toks[i]
        i += 1
        if kind == "num":
            return v
        if kind != "name":
            raise ValueError(f"unexpected {v!r}")
        if i < len(toks) and toks[i] == ("p", "("):
            i += 1
            args = []
            while True:
                args.append(node())
                if i >= len(toks):
                    raise ValueError("missing ')'")
                if toks[i] == ("p", ")"):
                    i += 1
                    return (v, args)
                if toks[i] != ("p", ","):
                    raise ValueError("expected ',' or ')'")
                i += 1
        return v
    out = node()
    if i != len(toks):
        raise ValueError(f"trailing tokens in {src!r}")
    return out


def validate(prog: dict) -> dict:
    """Check a program.yaml dict; returns {"state", "constants", "lr", "lines": [(name, ast)], "calls"}."""
    state = list(prog.get("state") or [])
    consts = {str(k): float(v) for k, v in (prog.get("constants") or {}).items()}
    if len(state) > MAX_STATE:
        raise ValueError(f"at most {MAX_STATE} state tensors")
    if len(consts) > MAX_CONST:
        raise ValueError(f"at most {MAX_CONST} constants")
    lr = float(prog["lr"])
    if not 0 < lr < 1:
        raise ValueError("lr must be in (0, 1)")
    names = set(INPUTS) | set(state) | set(consts)
    for n in list(state) + list(consts):
        if not NAME.match(n) or n in INPUTS or n == "update" or n in UNARY | BINARY | TERNARY | SCALAR_ONLY:
            raise ValueError(f"bad name {n!r}")
    lines, calls = [], 0
    raw = prog.get("program") or []
    if not 1 <= len(raw) <= MAX_LINES:
        raise ValueError(f"the program needs 1 to {MAX_LINES} assignments")
    for s in raw:
        if "=" not in str(s):
            raise ValueError(f"not an assignment: {s!r}")
        lhs, rhs = (x.strip() for x in str(s).split("=", 1))
        if not NAME.match(lhs) or lhs in INPUTS or lhs in consts:
            raise ValueError(f"cannot assign to {lhs!r}")
        ast = parse_expr(rhs)
        calls += _check(ast, names)
        names.add(lhs)
        lines.append((lhs, ast))
    if "update" not in {l for l, _ in lines}:
        raise ValueError("the program must assign update")
    if calls > MAX_CALLS:
        raise ValueError(f"at most {MAX_CALLS} primitive calls")
    return {"state": state, "constants": consts, "lr": lr, "lines": lines, "calls": calls}


def _check(ast, names: set) -> int:
    if isinstance(ast, float):
        return 0
    if isinstance(ast, str):
        if ast not in names:
            raise ValueError(f"unknown name {ast!r} (inputs, state, constants and earlier assignments only)")
        return 0
    op, args = ast
    arity = 1 if op in UNARY | SCALAR_ONLY else 2 if op in BINARY else 3 if op in TERNARY else None
    if arity is None:
        raise ValueError(f"unknown primitive {op!r}")
    if len(args) != arity:
        raise ValueError(f"{op} takes {arity} argument(s)")
    if op == "bias" and not isinstance(args[0], (float, str)):
        raise ValueError("bias takes a constant")
    return 1 + sum(_check(a, names) for a in args)


def _sign1(x, ops: Ops):
    return ops.where(x >= 0, 1.0 + 0 * x, -1.0 + 0 * x)


def _eval(ast, env: dict, ops: Ops, step: int):
    if isinstance(ast, float):
        return ast
    if isinstance(ast, str):
        return env[ast]
    op, args = ast
    if op == "bias":
        b = env[args[0]] if isinstance(args[0], str) else args[0]
        return 1.0 - float(b) ** step
    a = [_eval(x, env, ops, step) for x in args]
    x = a[0]
    if op == "neg":
        return -x
    if op == "abs":
        return ops.abs(x) if hasattr(x, "shape") else abs(x)
    if op == "sign":
        return ops.sign(x)
    if op == "square":
        return x * x
    if op == "sqrt":
        return ops.sqrt(ops.abs(x)) if hasattr(x, "shape") else abs(x) ** 0.5
    if op == "rsqrt":
        return 1.0 / ((ops.sqrt(ops.abs(x)) if hasattr(x, "shape") else abs(x) ** 0.5) + 1e-8)
    if op == "rms":
        return ops.rms(x) if hasattr(x, "shape") else abs(x)
    if op == "normalize":
        return x / (ops.rms(x) + 1e-8)
    if op == "orth":
        return newton_schulz5(x, ops) if hasattr(x, "shape") and ops.ndim(x) == 2 else x
    if op == "rownorm":
        if hasattr(x, "shape") and ops.ndim(x) == 2:
            return x / (ops.sqrt(ops.mean(x * x, axis=1, keepdims=True)) + 1e-8)
        return x
    y = a[1]
    if op == "add":
        return x + y
    if op == "sub":
        return x - y
    if op == "mul":
        return x * y
    if op == "div":
        if hasattr(y, "shape"):
            return x / (ops.abs(y) + 1e-8) * _sign1(y, ops)
        return x / ((abs(y) + 1e-8) * (1 if y >= 0 else -1))
    if op == "max":
        return ops.maximum(x, y) if hasattr(x, "shape") or hasattr(y, "shape") else max(x, y)
    if op == "min":
        return -ops.maximum(-x, -y) if hasattr(x, "shape") or hasattr(y, "shape") else min(x, y)
    if op == "clip":
        c = abs(float(y))
        return ops.where(x > c, c + 0 * x, ops.where(x < -c, -c + 0 * x, x))
    if op == "interp":
        b = float(a[2])
        return b * x + (1.0 - b) * y
    raise ValueError(op)


class ProgramRule(FunctionalRule):
    """A validated program as a FunctionalRule (B-OPT-001 interface)."""

    def __init__(self, prog: dict):
        v = validate(prog)
        super().__init__(name="program", hp=dict(v["constants"]))
        self.v = v

    def init_state(self, p, ops: Ops):
        return {s: ops.zeros_like(p) for s in self.v["state"]}

    def update(self, p, g, s, step, lr, ops: Ops):
        env = {"g": g, "p": p, **self.v["constants"], **s}
        for name, ast in self.v["lines"]:
            env[name] = _eval(ast, env, ops, step)
        for k in self.v["state"]:
            val = env[k]
            s[k] = val if hasattr(val, "shape") else val + ops.zeros_like(p)
        u = env["update"]
        return -lr * (u if hasattr(u, "shape") else u + ops.zeros_like(p))


def load_program(path) -> tuple[ProgramRule, float]:
    import yaml
    from pathlib import Path
    prog = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rule = ProgramRule(prog)
    return rule, rule.v["lr"]
