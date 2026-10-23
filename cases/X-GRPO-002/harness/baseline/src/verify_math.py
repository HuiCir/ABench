"""Verifiable tasks of X-GRPO-002 (evaluator code, agent-visible): the Countdown generator, the two prompt formats,
answer extraction and the verifiers used both for scoring and, if you like, as your training reward.

Countdown: given 3 or 4 numbers (1-99) and a target (1-999), write an arithmetic expression that uses each given
number at most once and at least two of them (some may be left out), only + - * / and parentheses, and equals the target exactly. Every
generated instance has at least one solution (checked by exhaustive search).
GSM8K: grade-school word problems (Cobbe et al. 2021); the answer is a number.

A response must contain exactly one <answer>...</answer> block; anything else is a format violation and scores 0.
"""
from __future__ import annotations

import ast
import itertools
import math
import operator
import re
from fractions import Fraction

COUNTDOWN_PROMPT = ("Using the numbers {nums}, create an equation that equals {target}. You may use +, -, *, / and "
                    "parentheses, and each number at most once. Think step by step, then write only the equation "
                    "(without '= {target}') inside <answer> </answer> tags.")
GSM8K_PROMPT = ("{question}\nThink step by step, then write only the final number inside <answer> </answer> tags.")
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def extract(response: str):
    """The single <answer> block's content, or None if there is not exactly one."""
    m = ANSWER.findall(response or "")
    return m[0].strip() if len(m) == 1 else None


def _eval(node, used: list):
    if isinstance(node, ast.Expression):
        return _eval(node.body, used)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = _eval(node.left, used), _eval(node.right, used)
        if isinstance(node.op, ast.Div) and b == 0:
            raise ZeroDivisionError
        return OPS[type(node.op)](a, b)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        used.append(node.value)
        return Fraction(node.value)
    raise ValueError("only integers, + - * / and parentheses are allowed")


def check_countdown(expr: str, nums: list, target: int) -> bool:
    if expr is None or len(expr) > 200:
        return False
    expr = expr.replace("×", "*").replace("÷", "/").split("=")[0].strip()
    try:
        used = []
        val = _eval(ast.parse(expr, mode="eval"), used)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, TypeError):
        return False
    if len(used) < 2:                                   # a single number is not an equation
        return False
    pool = list(nums)
    for u in used:
        if u not in pool:
            return False
        pool.remove(u)
    return val == target


def to_number(s: str):
    if s is None:
        return None
    t = s.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", t)
    return float(t) if m else None


def check_gsm8k(ans: str, gold: str) -> bool:
    v, g = to_number(ans), to_number(gold)
    return v is not None and g is not None and abs(v - g) < 1e-6


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K's answer field ends with '#### <number>'."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def solvable(nums: list, target: int) -> bool:
    """Exhaustive search over subsets, orders and operations (exact rational arithmetic)."""
    def vals(xs):
        if len(xs) == 1:
            yield xs[0]
            return
        for i in range(1, len(xs)):
            for a in vals(xs[:i]):
                for b in vals(xs[i:]):
                    yield a + b
                    yield a - b
                    yield a * b
                    if b != 0:
                        yield a / b
    fr = [Fraction(n) for n in nums]
    for k in range(2, len(fr) + 1):
        for sub in itertools.permutations(fr, k):
            if any(v == target for v in vals(list(sub))):
                return True
    return False


def make_countdown(rng, n: int) -> list[dict]:
    """n solvable instances: 3 or 4 numbers in 1-99 and a target in 10-999, not itself one of the numbers, reachable from them."""
    out = []
    while len(out) < n:
        k = int(rng.integers(3, 5))
        nums = [int(x) for x in rng.integers(1, 100, size=k)]
        a = [Fraction(x) for x in nums]
        rng.shuffle(a)
        v = a[0]
        for x in a[1:]:
            op = int(rng.integers(0, 4))
            v = v + x if op == 0 else v - x if op == 1 else v * x if op == 2 else (v / x if x else v)
        if v.denominator != 1 or not 10 <= v <= 999:
            continue
        target = int(v)
        if target not in nums and solvable(nums, target):
            out.append({"nums": nums, "target": target})
    return out


def countdown_prompt(inst: dict) -> str:
    return COUNTDOWN_PROMPT.format(nums=inst["nums"], target=inst["target"])


def gsm8k_prompt(question: str) -> str:
    return GSM8K_PROMPT.format(question=question.strip())
