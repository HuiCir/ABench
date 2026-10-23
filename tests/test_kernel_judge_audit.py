"""The kernel judge (tools/harness_kit/kernels/kb_bench.py) against fifteen audited submissions: it must accept the six
legitimate kernels and reject the nine that break a rule, each through the check named in
tests/fixtures/kernel_audit/expected.yaml. The static scan runs everywhere; the rest needs PyTorch (CPU is enough,
except for the two Triton kernels, which need CUDA)."""
from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "kernel_audit"
sys.path.insert(0, str(ROOT / "tools" / "harness_kit" / "kernels"))
import kb_bench as KB  # noqa: E402

EXP = yaml.safe_load((FIX / "expected.yaml").read_text(encoding="utf-8"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _cuda() -> bool:
    if not HAS_TORCH:
        return False
    import torch
    return torch.cuda.is_available()


class StaticScan(unittest.TestCase):
    def test_forbidden_api(self):
        pat = re.compile(KB.FORBIDDEN)
        for sid, e in EXP["submissions"].items():
            hit = bool(pat.findall((FIX / f"{sid}.py").read_text(encoding="utf-8")))
            self.assertEqual(hit, e.get("caught_by") == "forbidden_api", sid)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class JudgeVerdicts(unittest.TestCase):
    def _verdict(self, sid: str, e: dict, device: str) -> tuple[bool, str]:
        import torch
        g = KB.load_problem(FIX / e["problem"], EXP["problems"][e["problem"]])
        sd = [11, 12, 13, 14, 15, 16, 17][:len(KB.SEED_COLS)]
        snap = KB.snapshot()
        saved = {k: getattr(torch, k) for k in ("sum", "matmul", "relu", "softmax")}
        try:
            mod = KB.load_solution(FIX / f"{sid}.py")
            if KB.tampered(snap):
                return False, "tampered"
            worst, new_m = KB.check_before_timing(g, mod.ModelNew, sd, device)
            if worst > 1.0:
                return False, "before_timing"
            x = KB.make_inputs(g, "native", sd[1], device)
            with torch.no_grad():                         # the calls of time_call, without the clock
                for _ in range(KB.WARMUP + KB.REPS):
                    new_m(*x)
            if KB.check_after_timing(g, new_m, sd, device) > 1.0:
                return False, "after_timing"
            return (False, "tampered") if KB.tampered(snap) else (True, "")
        except Exception as ex:                           # a crash fails the problem
            return False, f"error: {ex}"
        finally:
            for k, v in saved.items():
                setattr(torch, k, v)

    def test_verdicts(self):
        device = "cuda" if _cuda() else "cpu"
        for sid, e in EXP["submissions"].items():
            if e.get("caught_by") == "forbidden_api" or (e.get("needs_cuda") and device != "cuda") or \
                    (e.get("cpu_only") and device != "cpu"):
                continue
            with self.subTest(sid=sid):
                ok, why = self._verdict(sid, e, device)
                self.assertEqual(ok, e["legitimate"], f"{sid}: {why}")
                if not e["legitimate"]:
                    self.assertEqual(why, e["caught_by"], sid)


if __name__ == "__main__":
    unittest.main()
