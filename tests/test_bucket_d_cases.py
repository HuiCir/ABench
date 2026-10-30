"""Units of the cases designed from bucket D and the Pro Set (2026-10-23) that can run without PyTorch: verifiers,
executors, generators, metric arithmetic on synthetic stage outputs, and the self-check data path."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "harness_kit"))
import abench_stage  # noqa: E402


def harness(case: str, name: str):
    h = ROOT / "cases" / case / "harness"
    sys.path.insert(0, str(h))
    if name == "libero_eval":                    # imported by name, as in the stage runner (spawned processes need it)
        import importlib
        return importlib.import_module("libero_eval")
    return abench_stage.load_module(h / f"{name}.py", f"t_{case.replace('-', '_')}_{name}")


def prepare(case: str):
    import importlib
    return importlib.import_module("prepare." + case.lower().replace("-", "_"))


def materialise(fz, public: Path) -> None:
    import shutil
    public.mkdir(parents=True, exist_ok=True)
    for name, arrays in fz.public.items():
        np.savez(public / name, **arrays)
    for name, src in fz.raw_files.items():
        shutil.copyfile(src, public / name)


class XGrpo(unittest.TestCase):
    def test_verifiers_and_metrics(self):
        V = harness("X-GRPO-002", "verify_math")
        self.assertTrue(V.check_countdown("(44 + 19) * 2", [44, 19, 2, 5], 126))
        self.assertFalse(V.check_countdown("44 * 44", [44, 19], 1936))           # a number used twice
        self.assertFalse(V.check_countdown("2 ** 10", [2, 10], 1024))            # operator not allowed
        self.assertFalse(V.check_countdown("45", [45, 3], 45))                   # a single number
        self.assertFalse(V.check_countdown("__import__('os')", [1], 1))
        self.assertIsNone(V.extract("<answer>1</answer> <answer>2</answer>"))
        inst = V.make_countdown(np.random.default_rng(3), 50)
        self.assertTrue(all(V.solvable(i["nums"], i["target"]) and i["target"] not in i["nums"] for i in inst))
        M, S = harness("X-GRPO-002", "metrics"), harness("X-GRPO-002", "stages")
        cd = [{"nums": [44, 19, 2], "target": 126}, {"nums": [3, 5, 7], "target": 22}]
        gs = [{"question": "q", "gold": "18"}, {"question": "q", "gold": "7"}]
        held, ref = S.held_out_from(cd, gs)
        resp = ["... <answer>(44 + 19) * 2</answer>", "<answer>3*5</answer>", "so <answer>18</answer>", "18"]
        m = M.score({"eval": {"response": np.asarray(resp, dtype=np.str_)}}, ref, {}, {"eval": 1.0})
        self.assertEqual((m["accuracy_countdown"], m["accuracy_gsm8k"], m["mean_accuracy"]), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(m["format_violation_rate"], 0.25)

    def test_synthetic_freeze_and_dev(self):
        fz = prepare("X-GRPO-002").synthetic(40)
        with tempfile.TemporaryDirectory() as d:
            materialise(fz, Path(d) / "pub")
            dev = harness("X-GRPO-002", "stages").make_dev(Path(d) / "pub", Path(d) / "dev", quick=True)
            self.assertEqual(len(dev["reference"]["task"]), 80)
        self.assertEqual(int((fz.hidden["task"] == 0).sum()), 1000)


class FinQA(unittest.TestCase):
    def test_executor(self):
        F = harness("FN-QA-004", "finqa")
        t = [["", "2019", "2018"], ["revenue", "$ 1,200", "1,000"], ["margin", "14%", "12%"]]
        self.assertEqual(F.execute("subtract(1200, 1000), divide(#0, 1000)", t), 0.2)
        self.assertEqual(F.execute("table_average(margin, none)", t), 0.13)
        self.assertEqual(F.execute("greater(14%, 12%)", t), "yes")
        self.assertIsNone(F.execute("divide(1, 0)", t))
        self.assertIsNone(F.execute("table_sum(missing, none)", t))
        self.assertTrue(F.correct("0.20001", 0.2) and F.correct("20%", 0.2) and not F.correct("20", 0.2))


class LegalTasks(unittest.TestCase):
    def test_labels_and_score(self):
        L = harness("LG-RSN-003", "legaltasks")
        self.assertEqual(L.fill("Q: {{text}} A:", {"text": "x"}), "Q: x A:")
        self.assertEqual(L.match_label(" yes ", ["No", "Yes"]), "Yes")
        self.assertIsNone(L.match_label("maybe", ["No", "Yes"]))
        self.assertAlmostEqual(L.balanced_accuracy(["a", "a", "a", "b"], ["a", "a", "a", "a"]), 0.5)


class PrimitiveLanguage(unittest.TestCase):
    def test_compiles_known_rules_and_screens(self):
        D, B = harness("B-OPT-002", "primitive_dsl"), harness("B-OPT-002", "behaviour")
        R = harness("B-OPT-002", "reference_rules")
        from optim_api import Ops
        ops, rng = Ops("numpy"), np.random.default_rng(0)
        lion = {"state": ["m"], "constants": {"b1": 0.9, "b2": 0.99, "wd": 0.1}, "lr": 3e-4,
                "program": ["c = interp(m, g, b1)", "m = interp(m, g, b2)", "update = add(sign(c), mul(wd, p))"]}
        r, ref = D.ProgramRule(lion), R.Lion()
        p = rng.standard_normal((6, 4))
        s1, s2 = r.init_state(p, ops), ref.init_state(p, ops)
        for step in range(1, 20):
            g = rng.standard_normal(p.shape)
            np.testing.assert_allclose(r.update(p, g, s1, step, 1e-3, ops), ref.update(p, g, s2, step, 1e-3, ops))
        self.assertGreater(B.screen(r, 5)["max_similarity"], 0.999)            # a rediscovery of Lion
        prog, _ = D.load_program(ROOT / "cases" / "B-OPT-002" / "example_submission" / "program.yaml")
        res = B.screen(prog, 5)
        self.assertLess(res["max_similarity"], 0.98)
        self.assertEqual(res["stable_extreme_scales"], 1.0)
        for bad in ({"lr": 1e-3, "program": ["update = foo(g)"]}, {"lr": 1e-3, "program": ["x = g"]},
                    {"state": ["a", "b", "c"], "lr": 1e-3, "program": ["update = g"]}):
            with self.assertRaises(ValueError):
                D.validate(bad)


class DiffusionSampler(unittest.TestCase):
    def test_metrics_and_windows(self):
        M = harness("U-DLM-005", "metrics")
        rng = np.random.default_rng(0)
        sc = {"nll_reference": rng.normal(3.0, 0.1, 50), "entropy_reference_cond": np.full(50, 4.0),
              "entropy_reference_uncond": np.full(50, 5.0)}
        for n in (16, 32, 64):
            for mode, e in (("uncond", 5.0), ("cond", 4.0)):
                sc[f"nll_{mode}_{n}"] = np.full(50, 4.0 - n / 64)
                sc[f"entropy_{mode}_{n}"] = np.full(50, e * (0.85 if (n, mode) == (16, "cond") else 1.0))
        m = M.score({"score": sc}, {}, {}, {})
        self.assertAlmostEqual(m["mean_log_gen_ppl"], np.mean([4.0 - n / 64 for n in (16, 32, 64)]))
        self.assertAlmostEqual(m["min_entropy_ratio"], 0.85)
        P = prepare("U-DLM-005")
        fz = P.synthetic(300)
        self.assertEqual(fz.hidden["references"].shape, (256, 1024))
        self.assertFalse((fz.hidden["references"] == P.EOT).any())
        self.assertTrue(np.array_equal(fz.hidden["prefixes"], fz.hidden["references"][:, :256]))


class LongContext(unittest.TestCase):
    def test_copy_sequences_and_metrics(self):
        LC = harness("A-LCTX-006", "longctx")
        stream = np.random.default_rng(1).integers(0, 50257, 200_000)
        x, start = LC.make_copy(stream, 8192, 5, np.random.default_rng(2))
        for row, s in zip(x, start):
            seg = row[s:]
            self.assertEqual(len(seg), LC.SEG)
            self.assertTrue(any(np.array_equal(row[p:p + LC.SEG], seg) for p in range(16, 8192 // 4 + 1)))
        M = harness("A-LCTX-006", "metrics")
        out = {"train": {"diverged": np.array([0.0]), "position_params": np.array([0]), "short_loss": np.array([3.5]),
                         **{f"long_loss_{L}": np.full(10, 3.0 + L / 16384) for L in (4096, 8192, 16384)},
                         "copy_loss_8192": np.full(5, 2.0)}}
        m = M.score(out, {}, {}, {"train": 100.0})
        self.assertAlmostEqual(m["long_loss_mean"], 3.0 + (0.25 + 0.5 + 1.0) / 3)
        r = M.relative(m, {**m, "short_loss": 3.49, "long_loss_mean": 4.0})
        self.assertAlmostEqual(r["short_loss_delta_vs_rope"], 0.01)


if __name__ == "__main__":
    unittest.main()


TOY_POLICY = '''
import numpy as np
def _toward(state):
    return np.r_[-np.clip(10 * state[:3], -1, 1), 0, 0, 0, -1.0]
class Planner:
    def reset(self, instruction): pass
    def plan(self, obs, t, request): return np.tile(_toward(obs["state"]), (8, 1))
class Controller:
    def reset(self, instruction): self.chunk, self.start, self.last = None, 0, -4
    def request(self, t):
        if t - self.last >= 4:
            self.last = t
            return {}
    def receive(self, chunk, t_issue, t_now): self.chunk, self.start = chunk, t_issue
    def act(self, t): return self.chunk[min(t - self.start, len(self.chunk) - 1)]
class Policy:
    def __init__(self, scale): self.scale = scale
    def reset(self, instruction): pass
    def act(self, obs): return _toward(obs["state"])
def load_planner(checkpoint_dir, device): return Planner()
def make_controller(checkpoint_dir): return Controller()
def load_policy(checkpoint_dir, models_dir, device, vlm, processor):
    if checkpoint_dir.endswith("tamper"):
        vlm["w"] = vlm["w"] + 1
    return Policy(vlm["w"])
'''
TOY_HOOK = '''
_D = {}
def before(models_dir, device):
    vlm = {"w": 1.0}
    _D["w"] = vlm["w"]
    return {"vlm": vlm, "processor": None}
def after(vlm, processor):
    return {"vlm_unchanged": int(vlm["w"] == _D["w"])}
'''


class ToyEnv:
    """A point that must be brought to the origin; observation keys as in robosuite."""

    def __init__(self, *_):
        self.p = np.zeros(3)

    def reset(self):
        pass

    def set_init_state(self, s):
        self.p = np.asarray(s[:3], float).copy()
        return self._obs()

    def _obs(self):
        z = np.zeros((128, 128, 3), np.uint8)
        return {"robot0_eef_pos": self.p.copy(), "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2), "agentview_image": z, "robot0_eye_in_hand_image": z}

    def step(self, a):
        self.p = self.p + 0.02 * np.asarray(a[:3])
        return self._obs(), 0.0, bool(np.all(np.abs(self.p) < 0.03)), {}

    def close(self):
        pass


def toy_env(suite, task_id, seed):
    return ToyEnv(), f"task {task_id}", None


class LiberoCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "policy.py").write_text(TOY_POLICY)
        (cls.tmp / "hook.py").write_text(TOY_HOOK)
        rng = np.random.default_rng(0)
        cls.states = [rng.uniform(-0.3, 0.3, 3) for _ in range(6)]

    def test_quat_and_delay(self):
        LE = harness("E-RT-002", "libero_eval")
        q = np.array([0, 0, np.sin(0.25), np.cos(0.25)])          # 0.5 rad about z
        np.testing.assert_allclose(LE.quat2axisangle(q), [0, 0, 0.5], atol=1e-12)
        self.assertEqual(LE.delay_steps(3, 0.0), 3)
        self.assertEqual(LE.delay_steps(3, 50.0), 4)
        self.assertEqual(LE.delay_steps(0, 50.1), 2)
        a, n = LE.pack_states([np.ones(3), np.ones(5)])
        self.assertEqual(a.shape, (2, 5))
        self.assertEqual(n.tolist(), [3, 5])

    def test_async_rollouts_in_processes_and_metrics(self):
        S, M = harness("E-RT-002", "stages"), harness("E-RT-002", "metrics")
        LE = harness("E-RT-002", "libero_eval")
        held = S.held_out_from(self.states, [0, 0, 1, 1, 2, 2])
        pl = LE.AgentProcess(self.tmp / "policy.py", "load_planner", {"checkpoint_dir": str(self.tmp), "device": "cpu"})
        co = LE.AgentProcess(self.tmp / "policy.py", "make_controller", {"checkpoint_dir": str(self.tmp)})
        try:
            out = S.rollouts(pl, co, held, 0, make_env=toy_env)
        finally:
            pl.close()
            co.close()
        self.assertEqual(out["success"].shape, (6, 3))
        self.assertTrue((out["mean_delay"][:, 0] >= 1).all())            # measured time adds at least one step
        self.assertTrue((out["mean_delay"][:, 2] >= 7).all())
        out["checkpoint_numbers"] = np.asarray([10])
        m = M.score({"rollout": out}, {"task_id": held["task_id"]}, {}, {"rollout": 1.0})
        self.assertAlmostEqual(m["mean_success"], float(out["success"].mean()))
        self.assertGreaterEqual(m["success_d0"], m["success_d6"])
        ci = abench_stage_ci(M, {"rollout": out}, {"task_id": held["task_id"]})
        self.assertIsNotNone(ci)

    def test_sync_rollouts_and_frozen_check(self):
        S, M = harness("E-VLA-003", "stages"), harness("E-VLA-003", "metrics")
        LE = harness("E-VLA-003", "libero_eval")
        held = S.held_out_from(self.states, [0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1])
        res = {}
        for tag in ("ok", "tamper"):
            pol = LE.AgentProcess(self.tmp / "policy.py", "load_policy",
                                  {"checkpoint_dir": str(self.tmp / tag), "models_dir": "", "device": "cpu"},
                                  hook=self.tmp / "hook.py", hook_kwargs={"models_dir": "", "device": "cpu"})
            try:
                out = S.rollouts(pol, held, 0, make_env=toy_env)
                fin = pol.final()
            finally:
                pol.close()
            out["vlm_unchanged"] = np.asarray([fin["vlm_unchanged"]])
            out["checkpoint_numbers"] = np.asarray([5])
            res[tag] = M.score({"rollout": out}, {"task_id": held["task_id"], "suite_id": held["suite_id"]}, {}, {})
        self.assertEqual(res["ok"]["vlm_unchanged"], 1.0)
        self.assertEqual(res["tamper"]["vlm_unchanged"], 0.0)
        self.assertEqual(res["ok"]["mean_success"], 1.0)                 # synchronous toy control always succeeds
        self.assertLess(res["ok"]["ms_per_step"], 50.0)

    def test_bad_actions_rejected(self):
        LE = harness("E-RT-002", "libero_eval")
        with self.assertRaises(ValueError):
            LE._action([0, 0, 0])
        with self.assertRaises(ValueError):
            LE._chunk(np.zeros((0, 7)))
        with self.assertRaises(ValueError):
            LE._action([np.nan] * 7)

    def test_prepare_synthetic(self):
        for c, keys in (("E-RT-002", {"init_states", "init_len", "task_id"}),
                        ("E-VLA-003", {"init_states", "init_len", "task_id", "suite_id"})):
            fz = prepare(c).synthetic(40)
            self.assertEqual(set(fz.hidden), keys)
            S = harness(c, "stages")
            self.assertTrue(set(S.HELD_OUT_KEYS) <= set(fz.hidden))


def abench_stage_ci(M, outputs, reference):
    import abench_metrics
    return abench_metrics.bootstrap_ci(M.score, outputs, reference, {}, {}, M.RESAMPLE(reference), M.PRIMARY["metric"], n=50)


class Segmentation3D(unittest.TestCase):
    def test_measures(self):
        S = harness("MD-SEG-001", "seg3d")
        a = np.zeros((20, 20, 20), np.uint8)
        a[5:15, 5:15, 5:15] = 1
        self.assertEqual(S.dice(a, a, 1), 1.0)
        self.assertEqual(S.nsd(a, a, 1, (1, 1, 1), 1.0), 1.0)
        b = np.roll(a, 1, axis=0)
        self.assertAlmostEqual(S.dice(a, b, 1), 0.9)
        self.assertEqual(S.nsd(a, b, 1, (1, 1, 1), 1.0), 1.0)              # one voxel shift is within 1 mm
        self.assertLess(S.nsd(a, b, 1, (3, 1, 1), 1.0), 1.0)              # but not when that axis is 3 mm
        self.assertEqual(S.dice(np.zeros_like(a), np.zeros_like(a), 1), 1.0)
        self.assertEqual(S.nsd(np.zeros_like(a), a, 1, (1, 1, 1), 1.0), 0.0)

    def test_sampling_and_sliding_window(self):
        S = harness("MD-SEG-001", "seg3d")
        rng = np.random.default_rng(0)
        seg = np.zeros((30, 50, 37), np.uint8)
        seg[10:14, 20:30, 5:9] = 2
        img = rng.normal(size=(2, *seg.shape)).astype(np.float32)
        for fg in (True, False):
            x, y = S.random_patch(img, seg, (20, 32, 48), rng, fg)
            self.assertEqual(x.shape, (2, 20, 32, 48))
            if fg:
                self.assertTrue((y > 0).any())
        xb, yb = S.augment(*S.batch([(img, seg)] * 3, S.TASKS["prostate"] | {"patch": (16, 32, 32)}, rng), rng)
        self.assertEqual(xb.shape, (2, 2, 16, 32, 32))
        # an oracle predictor that reads the label under each window reproduces the label exactly
        lab = seg.copy()

        class Oracle:
            def __init__(self):
                self.calls = []

            def __call__(self, x):
                return x

        tag = np.stack([lab == k for k in range(3)]).astype(np.float32)
        pred = S.sliding_window(lambda x: x, tag, (16, 32, 32), 3)
        np.testing.assert_array_equal(pred, lab)
        small = S.sliding_window(lambda x: x, tag[:, :8, :8, :8], (16, 32, 32), 3)
        np.testing.assert_array_equal(small, lab[:8, :8, :8])
        pk = S.pack([seg, seg[:3]])
        self.assertEqual([u.shape for u in S.unpack(*pk)], [seg.shape, (3, 50, 37)])

    def test_dev_path_and_metrics(self):
        S, St, M = harness("MD-SEG-001", "seg3d"), harness("MD-SEG-001", "stages"), harness("MD-SEG-001", "metrics")
        fz = prepare("MD-SEG-001").synthetic(24)
        self.assertTrue(set(St.HELD_OUT_KEYS) <= set(fz.hidden) and set(St.HELD_OUT_PRE_KEYS) <= set(fz.hidden))
        tmp = Path(tempfile.mkdtemp())
        materialise(fz, tmp / "pub")
        dev = St.make_dev(tmp / "pub", tmp / "dev")
        ref = dev["reference"]
        perfect, half = {}, {}
        for t in M.TASKS:
            gts = S.unpack(ref[f"{t}_test_seg"], ref[f"{t}_test_seg_shape"])
            flat, shp = S.pack(gts)
            base = {"pred_shape": shp, "params": np.asarray([10]), "diverged": np.asarray([0.0]), "peak_mem_gb": np.asarray([1.0])}
            perfect[t] = {**base, "pred": flat.astype(np.uint8)}
            half[t] = {**base, "pred": np.zeros_like(flat, np.uint8)}
        p = M.score(perfect, ref, {}, {})
        z = M.score(half, ref, {}, {})
        self.assertEqual(p["mean_dice"], 1.0)
        self.assertEqual(p["mean_nsd"], 1.0)
        self.assertLess(z["mean_dice"], 0.5)
        self.assertAlmostEqual(M.relative(p, z)["mean_dice_delta_vs_unet"], p["mean_dice"] - z["mean_dice"])


class LatentDiffusion(unittest.TestCase):
    def test_schedule_and_ddim(self):
        L = harness("V-DIT-003", "ldm")
        ab = L.alphas_cumprod()
        self.assertEqual(len(ab), 1000)
        self.assertTrue(np.all(np.diff(ab) < 0))
        ts = L.ddim_timesteps()
        self.assertEqual((len(ts), ts[0], ts[-1]), (50, 980, 0))
        rng = np.random.default_rng(0)
        x0, eps = rng.normal(size=8), rng.normal(size=8)
        x = np.sqrt(ab[980]) * x0 + np.sqrt(1 - ab[980]) * eps
        for k, t in enumerate(ts):                       # with the true noise, DDIM walks back to x0 exactly
            a_prev = ab[ts[k + 1]] if k + 1 < len(ts) else 1.0
            x = L.ddim_update(x, eps, ab[t], a_prev)
        np.testing.assert_allclose(x, x0, atol=1e-9)
        self.assertEqual(np.bincount(L.labels_for(5000)).tolist(), [500] * 10)

    def test_metrics_and_dev(self):
        S, M = harness("V-DIT-003", "stages"), harness("V-DIT-003", "metrics")
        fz = prepare("V-DIT-003").synthetic(40)
        tmp = Path(tempfile.mkdtemp())
        materialise(fz, tmp / "pub")
        dev = S.make_dev(tmp / "pub", tmp / "dev", quick=True)
        ref = dev["reference"]
        mu, sg = ref["ref_mu"], ref["ref_sigma"]
        outs = {"train": {"flops": np.asarray([100]), "params": np.asarray([5]), "diverged": np.asarray([0])},
                "fid": {"mu_cfg15": mu, "sigma_cfg15": sg, "mu_cfg10": mu + 1.0, "sigma_cfg10": sg}}
        m = M.score(outs, ref, {}, {})
        self.assertLess(abs(m["fid_cfg15"]), 0.1)                  # sqrtm of a rank-64 covariance is not exact
        self.assertLess(abs(m["fid_cfg10"] - 2048.0), 0.1)
        base = dict(m, flops=95.0, fid_cfg15=3.0)
        r = M.relative(m, base)
        self.assertAlmostEqual(r["flops_ratio_vs_dit"], 100 / 95)
        self.assertLess(r["fid_cfg15_delta_vs_dit"], 0)


class SparseViews(unittest.TestCase):
    def test_split_measures_projection(self):
        N = harness("V-3DG-005", "nvs")
        tr, te = N.split(194)
        self.assertEqual((len(tr), len(te)), (24, 25))
        self.assertFalse(set(tr) & set(te))
        a = np.random.default_rng(0).integers(0, 256, (30, 40, 3)).astype(np.uint8)
        self.assertEqual(N.ssim(a, a), 1.0)
        self.assertGreater(N.psnr(a, np.clip(a.astype(int) + 1, 0, 255).astype(np.uint8)), 45)
        K, w2c = np.array([[50.0, 0, 20], [0, 50.0, 15], [0, 0, 1]]), np.eye(4)
        uv, z = N.project(np.array([[0, 0, 2.0], [0.2, -0.1, 1.0]]), K, w2c)
        np.testing.assert_allclose(uv, [[20, 15], [30, 10]])

    def test_dev_path_and_metrics(self):
        S, M = harness("V-3DG-005", "stages"), harness("V-3DG-005", "metrics")
        fz = prepare("V-3DG-005").synthetic(40)
        self.assertTrue(set(S.HELD_OUT_KEYS) <= set(fz.hidden) and set(S.HELD_OUT_PRE_KEYS) <= set(fz.hidden))
        tmp = Path(tempfile.mkdtemp())
        materialise(fz, tmp / "pub")
        dev = S.make_dev(tmp / "pub", tmp / "dev")
        ref = dev["reference"]
        with np.load(dev["held_out"]) as z:
            self.assertEqual(z["scene0_test_K"].shape[0], len(ref["scene0_test_images"]))
        perfect = {s: {"images": ref[f"{s}_test_images"]} for s in S.SCENES}
        sec = {s: 500.0 for s in S.SCENES}
        m = M.score(perfect, ref, {}, sec)
        self.assertEqual(m["mean_psnr"], 120.0)
        grey = {s: {"images": np.full_like(ref[f"{s}_test_images"], 128)} for s in S.SCENES}
        g = M.score(grey, ref, {}, dict(sec, scene3=700.0))
        self.assertLess(g["mean_psnr"], 12)                      # uniform noise vs mid-grey: about 10.8 dB
        self.assertEqual(g["max_fit_seconds"], 700.0)
        self.assertGreater(M.relative(m, g)["mean_psnr_delta_vs_3dgs"], 100)


class WellSurrogate(unittest.TestCase):
    def test_vrmse_rollout_and_splits(self):
        W = harness("PH-WELL-002", "wellsurr")
        rng = np.random.default_rng(0)
        t = rng.normal(size=(3, 2, 64, 64))
        np.testing.assert_allclose(W.vrmse(t, t), 0.0)
        np.testing.assert_allclose(W.vrmse(np.broadcast_to(t.mean((-1, -2), keepdims=True), t.shape), t), 1.0, atol=1e-6)
        self.assertEqual(float(W.vrmse(np.full_like(t, np.nan), t)[0]), W.BAD)
        traj = rng.normal(size=(20, 2, 64, 64)).astype(np.float32)
        x, y = W.windows(traj, [0, 6])
        self.assertEqual((x.shape, y.shape), ((2, 4, 2, 64, 64), (2, 10, 2, 64, 64)))
        oracle_calls = []

        def nan_after_two(system, h, p):
            oracle_calls.append(1)
            return h[:, -1] if len(oracle_calls) <= 2 else np.full(h.shape[:1] + h.shape[2:], np.nan, np.float32)
        out = W.rollout(nan_after_two, "gray_scott", x, np.zeros((2, 2), np.float32))
        self.assertTrue(np.isfinite(out[:, :2].astype(np.float32)).all() and np.isnan(out[:, 2:].astype(np.float32)).all())
        P = prepare("PH-WELL-002")
        self.assertTrue(P.unseen("active_matter", P.file_params("active_matter", "active_matter_L_10.0_zeta_9.0_alpha_-5.0.hdf5")))
        self.assertFalse(P.unseen("turbulent_radiative", P.file_params("turbulent_radiative", "turbulent_radiative_layer_tcool_0.10.hdf5")))

    def test_dev_path_and_metrics(self):
        S, M = harness("PH-WELL-002", "stages"), harness("PH-WELL-002", "metrics")
        fz = prepare("PH-WELL-002").synthetic(10)
        self.assertTrue(set(S.HELD_OUT_KEYS) <= set(fz.hidden))
        for s in ("turbulent_radiative", "active_matter", "gray_scott"):     # no unseen parameter set is public
            pub = fz.public[f"{s}.npz"]["params"]
            self.assertFalse(any(prepare("PH-WELL-002").unseen(s, list(p)) for p in pub))
        tmp = Path(tempfile.mkdtemp())
        materialise(fz, tmp / "pub")
        dev = S.make_dev(tmp / "pub", tmp / "dev")
        ref = dev["reference"]
        with np.load(dev["held_out"]) as z:
            held = {k: z[k] for k in z.files}
        W = harness("PH-WELL-002", "wellsurr")
        persist = {f"pred_{s}_{t}": W.rollout(lambda sy, h, p: h[:, -1], s, held[f"{s}_{t}_x"], held[f"{s}_{t}_params"])
                   for s, t in M.PAIRS}
        perfect = {f"pred_{s}_{t}": ref[f"{s}_{t}_y"] for s, t in M.PAIRS}
        mp = M.score({"rollout": persist}, ref, {}, {})
        mo = M.score({"rollout": perfect}, ref, {}, {})
        self.assertEqual(mo["mean_rollout_vrmse"], 0.0)
        self.assertGreater(mp["mean_rollout_vrmse"], 0.1)
        import abench_metrics
        ci = abench_metrics.bootstrap_ci(M.score, {"rollout": persist}, ref, {}, {}, M.RESAMPLE(ref), M.PRIMARY["metric"], n=30)
        self.assertIsNotNone(ci)


TOY_TRAIN = r'''
import os, pathlib, random
ck = pathlib.Path(os.environ["ABENCH_CKPT_DIR"]); r = os.environ.get("ABENCH_RESUME_FROM")
s0 = int(pathlib.Path(r).name.split("_")[1]) if r else 0
random.seed(s0)
for s in range(s0, s0 + 60):
    print(f"ABENCH step={s} reward={0.1 + 0.005 * s + 0.001 * random.random():.5f} loss={3.0 - 0.01 * s:.5f}", flush=True)
    if s % 20 == 0:
        p = ck / f"step_{s:06d}"; (p / "model").mkdir(parents=True, exist_ok=True)
        (p / "model" / "model.safetensors").write_bytes(f"w{s}".encode()); (p / "model" / "config.json").write_text("{}")
        (p / "state.pt").write_bytes(b"o"); (p / "DONE").write_text("")
'''


def toy_session(tmp: Path):
    """A supervised toy run, its session directory, the start weights and a submission delivering step 40."""
    import os
    A = harness("PRO-ARL-001", "pro_audit")
    start = tmp / "start"
    start.mkdir()
    (start / "model.safetensors").write_bytes(b"w0")
    (start / "config.json").write_text("{}")
    sub = tmp / "sub"
    sub.mkdir()
    (sub / "train.py").write_text(TOY_TRAIN)
    sess = tmp / "session"
    A.supervise("r1", [sys.executable, "train.py"], sess / "audit", sess / "checkpoints" / "r1", sub, start)
    (sub / "checkpoint").mkdir()
    for f in ("model.safetensors", "config.json"):
        (sub / "checkpoint" / f).write_bytes((sess / "checkpoints" / "r1" / "step_000040" / "model" / f).read_bytes())
    os.environ["ABENCH_PRO_SESSION"] = str(sess)
    return A, start, sub, sess


class ProAudit(unittest.TestCase):
    def test_audit_resume_and_forgery(self):
        tmp = Path(tempfile.mkdtemp())
        A, start, sub, sess = toy_session(tmp)
        P = harness("PRO-ARL-001", "pro_stages")
        a = P.audit(sub, start)
        self.assertEqual((int(a["chain_ok"][0]), int(a["trace_ok"][0]), int(a["code_ok"][0])), (1, 1, 1))
        r = P.resume(sub, tmp / "work", 0.4, "reward", 0.10, max_seconds=60)
        self.assertEqual(int(r["resume_ok"][0]), 1, r["resume_json"][0])
        (sub / "train.py").write_text(TOY_TRAIN.replace("0.005 * s", "0.008 * s"))    # different program
        self.assertEqual(int(P.audit(sub, start)["code_ok"][0]), 0)
        r2 = P.resume(sub, tmp / "work2", 0.4, "reward", 0.10, max_seconds=60)
        self.assertEqual(int(r2["resume_ok"][0]), 0)
        (sub / "checkpoint" / "model.safetensors").write_bytes(b"forged")
        self.assertEqual(int(P.audit(sub, start)["trace_ok"][0]), 0)
        self.assertEqual(P.scores({"audit": a, "resume": r}), {"chain_ok": 1.0, "trace_ok": 1.0, "code_ok": 1.0, "resume_ok": 1.0})


class AgenticRL(unittest.TestCase):
    def test_dev_path_and_metrics(self):
        S, M = harness("PRO-ARL-001", "stages"), harness("PRO-ARL-001", "metrics")
        fz = prepare("PRO-ARL-001").synthetic(10)
        self.assertTrue(set(S.HELD_OUT_KEYS) <= set(fz.hidden))
        tmp = Path(tempfile.mkdtemp())
        materialise(fz, tmp / "pub")
        dev = S.make_dev(tmp / "pub", tmp / "dev", quick=True)
        ref = dev["reference"]
        gold = ref["answer"]
        resp = [rf"<python>print(1)</python><output>1</output> so \boxed{{{g if k % 4 else g + 1}}}" for g in gold for k in range(16)]
        ev = {"same_arch": np.asarray([1]), "response": np.asarray(resp), "tool_calls": np.ones(len(resp)),
              "response_tokens": np.full(len(resp), 100)}
        ok = {"chain_ok": np.asarray([1]), "trace_ok": np.asarray([1]), "code_ok": np.asarray([1])}
        outs = {"audit": ok, "eval": ev, "resume": {"resume_ok": np.asarray([1])}}
        m = M.score(outs, ref, {}, {})
        self.assertAlmostEqual(m["avg16_accuracy"], 0.75)
        self.assertEqual(m["tool_calls_mean"], 1.0)
        import abench_metrics
        self.assertIsNotNone(abench_metrics.bootstrap_ci(M.score, outs, ref, {}, {}, M.RESAMPLE(ref), "avg16_accuracy", n=30))
        bad = M.score({"audit": ok, "eval": {"same_arch": np.asarray([0])}}, ref, {}, {})
        self.assertEqual((bad["avg16_accuracy"], bad["resume_ok"]), (0.0, 0.0))


class Distillation(unittest.TestCase):
    def test_dev_path_and_metrics(self):
        S, M = harness("PRO-OPD-002", "stages"), harness("PRO-OPD-002", "metrics")
        self.assertEqual(S.START, "Qwen3-1.7B-Base")
        fz = prepare("PRO-OPD-002").synthetic(10)
        tmp = Path(tempfile.mkdtemp())
        materialise(fz, tmp / "pub")
        ref = S.make_dev(tmp / "pub", tmp / "dev", quick=True)["reference"]
        resp = [rf"\boxed{{{g}}}" if k < 8 else "no" for g in ref["answer"] for k in range(16)]
        ok = {"chain_ok": np.asarray([1]), "trace_ok": np.asarray([1]), "code_ok": np.asarray([1])}
        m = M.score({"audit": ok, "resume": {"resume_ok": np.asarray([1])},
                     "eval": {"same_arch": np.asarray([1]), "response": np.asarray(resp), "response_tokens": np.ones(len(resp))}}, ref, {}, {})
        self.assertAlmostEqual(m["avg16_accuracy"], 0.5)
        self.assertAlmostEqual(m["no_answer_rate"], 0.5)


class MoEPretraining(unittest.TestCase):
    def test_random_init_audit_and_metrics(self):
        import os
        A = harness("PRO-MOE-003", "pro_audit")
        P = harness("PRO-MOE-003", "pro_stages")
        tmp = Path(tempfile.mkdtemp())
        sub = tmp / "sub"
        sub.mkdir()
        (sub / "train.py").write_text(TOY_TRAIN)
        sess = tmp / "session"
        A.supervise("moe", [sys.executable, "train.py"], sess / "audit", sess / "checkpoints" / "moe", sub, None)
        (sub / "checkpoint").mkdir()
        for f in ("model.safetensors", "config.json"):
            (sub / "checkpoint" / f).write_bytes((sess / "checkpoints" / "moe" / "step_000020" / "model" / f).read_bytes())
        os.environ["ABENCH_PRO_SESSION"] = str(sess)
        a = P.audit(sub, None)
        self.assertEqual((int(a["chain_ok"][0]), int(a["trace_ok"][0]), int(a["code_ok"][0])), (1, 1, 1))
        r = P.resume(sub, tmp / "w", 0.0, "loss", 0.01, max_seconds=60)
        self.assertEqual(int(r["resume_ok"][0]), 1, r["resume_json"][0])
        M = harness("PRO-MOE-003", "metrics")
        ev = {"val_loss": np.asarray([3.1]), "flops_per_token": np.asarray([3e8]), "stored_elements": np.asarray([10**9])}
        m = M.score({"audit": a, "resume": r, "eval": ev}, {}, {}, {})
        self.assertEqual((m["val_loss"], m["resume_ok"]), (3.1, 1.0))
        fz = prepare("PRO-MOE-003").synthetic(4)
        self.assertTrue(set(harness("PRO-MOE-003", "stages").HELD_OUT_KEYS) <= set(fz.hidden))


class Serving(unittest.TestCase):
    def test_ladder_with_stand_in_server_and_metrics(self):
        S, M = harness("PRO-SRV-005", "stages"), harness("PRO-SRV-005", "metrics")
        SB = harness("PRO-SRV-005", "serve_bench")
        fz = prepare("PRO-SRV-005").synthetic(10)
        self.assertTrue(set(S.HELD_OUT_KEYS) <= set(fz.hidden))
        srv = SB.FakeServer(0, delay=0.0005)
        try:
            prompts = [[i + 1] for i in range(40)]
            r = S.ladder(srv.srv.server_port, prompts, [4] * 40, np.linspace(0, 2.0, 40), factors=(1.0, 2.0))
        finally:
            srv.close()
            srv.srv.server_close()
        self.assertTrue((r["attainment"] >= 0.9).all())
        self.assertAlmostEqual(M.max_factor(np.asarray([1.0, 0.95, 0.5])), 2.0 + 2.0 * 0.05 / 0.45)
        self.assertEqual(M.max_factor(np.asarray([0.5])), 0.0)
        outs = {"serve": {f"{t}_attainment": np.asarray([1.0, 1.0, 0.8]) for t in M.TRACES}, "check": {"consistent": np.ones(10)}}
        for t in M.TRACES:
            outs["serve"][f"{t}_native_rate"] = np.asarray([2.0])
            outs["serve"][f"{t}_tokens_per_s"] = np.asarray([10.0, 20.0, 25.0])
            outs["serve"][f"{t}_check_complete"] = np.asarray([1.0])
        m = M.score(outs, {}, {}, {"serve": 100.0})
        self.assertAlmostEqual(m["sustained_rate"], 2.0 * (2.0 + 2.0 * 0.1 / 0.2))
        self.assertEqual(m["consistency"], 1.0)


class SpeculativeDecoding(unittest.TestCase):
    def test_dev_path_and_metrics(self):
        S, M = harness("PRO-SPEC-006", "stages"), harness("PRO-SPEC-006", "metrics")
        fz = prepare("PRO-SPEC-006").synthetic(1)
        self.assertTrue(set(S.HELD_OUT_KEYS) <= set(fz.hidden))
        tmp = Path(tempfile.mkdtemp())
        materialise(fz, tmp / "pub")
        dev = S.make_dev(tmp / "pub", tmp / "dev", quick=True)
        ref = dev["reference"]
        n = len(ref["set_id"])
        self.assertEqual(n, 15)
        prompts, check, u = S.prompts_of(tmp / "dev")
        self.assertEqual((len(prompts), u), (15, 0.5))
        ok = {"chain_ok": np.asarray([1]), "trace_ok": np.asarray([1]), "code_ok": np.asarray([1])}
        outs = {"audit": ok, "resume": {"resume_ok": np.asarray([1])}, "check": {"consistent": np.ones(5)},
                "vanilla": {"seconds": np.full(n, 6.0), "complete": np.ones(n)},
                "spec": {"seconds": np.where(ref["set_id"] == 1, 2.0, 3.0), "complete": np.ones(n)}}
        m = M.score(outs, ref, {}, {})
        self.assertAlmostEqual(m["speedup_gsm8k"], 3.0)
        self.assertAlmostEqual(m["speedup"], (2 + 3 + 2) / 3)
        import abench_metrics
        self.assertIsNotNone(abench_metrics.bootstrap_ci(M.score, outs, ref, {}, {}, M.RESAMPLE(ref), "speedup", n=30))
