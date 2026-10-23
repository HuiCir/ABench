"""FinQA programs and answers (evaluator code, agent-visible; FN-QA-004).

A program is FinQA's arithmetic language (Chen et al. 2021): steps separated by ", ", each `op(arg1, arg2)`, where an
argument is a number ("5829", "14%", "$1,200"), a constant ("const_100", "const_1000", "const_m1"), a reference to an
earlier step ("#0") or, for the table operations, the exact header of a table row. Operations: add, subtract,
multiply, divide, exp (arg1 ** arg2), greater ("yes" if arg1 > arg2), table_max, table_min, table_sum, table_average
(over the numbers of the named row; arg2 is "none"). The value of a program is the value of its last step, rounded
to 5 decimals, as in FinQA's own evaluation script. Percentages are fractions: "14%" is 0.14.

An answer is correct (execution accuracy) when it equals the gold executed value: "yes"/"no" exactly, numbers within
max(5e-5, 1e-4 * |gold|) after rounding to 5 decimals.
"""
from __future__ import annotations

import math
import re

OPS = ("add", "subtract", "multiply", "divide", "exp", "greater", "table_max", "table_min", "table_sum", "table_average")
STEP = re.compile(r"^\s*([a-z_]+)\(\s*(.*?)\s*,\s*(.*?)\s*\)\s*$")


def to_num(text: str):
    """FinQA number syntax -> float, or None."""
    t = str(text).strip().replace(",", "").replace("$", "").strip()
    t = t.split("(")[0].strip() if not t.startswith("(") else t
    if t.startswith("const_"):
        t = t[6:]
        return -1.0 if t == "m1" else _float(t)
    if t.endswith("%"):
        v = _float(t[:-1])
        return None if v is None else v / 100.0
    return _float(t)


def _float(t: str):
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def split_steps(program: str) -> list[str]:
    """'a(1, 2), b(#0, 3)' -> ['a(1, 2)', 'b(#0, 3)']"""
    out, depth, cur = [], 0, ""
    for ch in program.strip():
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def execute(program: str, table: list[list[str]]):
    """Value of a program ("yes"/"no" or a float rounded to 5 decimals), or None if it is invalid."""
    rows = {str(r[0]).strip(): r[1:] for r in table or [] if r}
    res: list = []
    try:
        steps = split_steps(program)
        if not steps:
            return None
        for s in steps:
            m = STEP.match(s)
            if not m or m.group(1) not in OPS:
                return None
            op, a1, a2 = m.groups()
            if op.startswith("table_"):
                row = rows.get(a1.strip())
                if row is None:
                    return None
                nums = [to_num(x) for x in row]
                if not nums or any(v is None for v in nums):
                    return None
                v = {"table_max": max, "table_min": min, "table_sum": sum}.get(op, lambda z: sum(z) / len(z))(nums)
            else:
                x, y = (_arg(a, res) for a in (a1, a2))
                if x is None or y is None or isinstance(x, str) or isinstance(y, str):
                    return None
                if op == "add":
                    v = x + y
                elif op == "subtract":
                    v = x - y
                elif op == "multiply":
                    v = x * y
                elif op == "divide":
                    if y == 0:
                        return None
                    v = x / y
                elif op == "exp":
                    v = x ** y
                    if isinstance(v, complex) or not math.isfinite(v):
                        return None
                else:
                    v = "yes" if x > y else "no"
            res.append(v)
    except (ValueError, OverflowError, ZeroDivisionError, IndexError):
        return None
    last = res[-1]
    return last if isinstance(last, str) else round(float(last), 5)


def _arg(a: str, res: list):
    a = a.strip()
    if a.startswith("#"):
        try:
            return res[int(a[1:])]
        except (ValueError, IndexError):
            return None
    return to_num(a)


def parse_answer(x):
    """A submitted answer (number, numeric string, 'yes'/'no') -> float, 'yes'/'no' or None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(float(x)) else None
    s = str(x).strip().lower()
    if s in ("yes", "no"):
        return s
    return to_num(s)


def correct(pred, gold) -> bool:
    p = parse_answer(pred)
    g = gold if isinstance(gold, str) else float(gold)
    if isinstance(g, str):
        return isinstance(p, str) and p == g.strip().lower()
    if not isinstance(p, float):
        return False
    return abs(round(p, 5) - g) <= max(5e-5, 1e-4 * abs(g))


def render_table(table: list[list[str]]) -> str:
    """One line per row: 'header | v1 | v2 ...' (the first row is the column header)."""
    return "\n".join(" | ".join(str(c).strip() for c in r) for r in table or [])
