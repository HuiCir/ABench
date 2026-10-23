"""tools/prepare/run.py: source verification against published hashes, and escrow of secret generator seeds."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare import run  # noqa: E402

REV = "c1f9ee939b7d05667af864ee1cb066393154bf85"


def module(files):
    return types.SimpleNamespace(SOURCES=[{"name": n, "url": f"https://huggingface.co/datasets/o/r/resolve/{REV}/d/{n}"}
                                          for n in files])


class SourceVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self.tmp.name)
        (self.raw / "a.parquet").write_bytes(b"alpha")
        self.sha = hashlib.sha256(b"alpha").hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_verified(self):
        with mock.patch.object(run, "_hf_expected", return_value={"sha256": self.sha}):
            rows = run.verify_sources(module(["a.parquet"]), self.raw, allow_unverified=False)
        self.assertEqual(rows[0]["status"], "verified")

    def test_mismatch_stops(self):
        with mock.patch.object(run, "_hf_expected", return_value={"sha256": "0" * 64}), self.assertRaises(SystemExit):
            run.verify_sources(module(["a.parquet"]), self.raw, allow_unverified=True)

    def test_missing_file_stops(self):
        with mock.patch.object(run, "_hf_expected", return_value={"sha256": self.sha}), self.assertRaises(SystemExit):
            run.verify_sources(module(["b.parquet"]), self.raw, allow_unverified=True)

    def test_offline_needs_explicit_opt_in(self):
        with mock.patch.object(run, "_hf_expected", side_effect=OSError("offline")):
            with self.assertRaises(SystemExit):
                run.verify_sources(module(["a.parquet"]), self.raw, allow_unverified=False)
            rows = run.verify_sources(module(["a.parquet"]), self.raw, allow_unverified=True)
        self.assertTrue(rows[0]["status"].startswith("unverified"))

    def test_git_blob_fallback(self):
        blob = hashlib.sha1(b"blob 5\0alpha").hexdigest()
        with mock.patch.object(run, "_hf_expected", return_value={"git_sha1": blob}):
            rows = run.verify_sources(module(["a.parquet"]), self.raw, allow_unverified=False)
        self.assertEqual(rows[0]["status"], "verified")


class SeedEscrow(unittest.TestCase):
    def test_escrow_is_private(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"ABENCH_TEST_SEED": "424242"}):
            key = Path(d) / "keys" / "hidden.key"
            key.parent.mkdir()
            key.write_bytes(b"k")
            p = Path(run.escrow_seed("X-TEST-001", "ABENCH_TEST_SEED", key, "ab" * 32))
            self.assertEqual(p.parent, key.parent / "seeds")
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
            rec = json.loads(p.read_text())
            self.assertEqual((rec["seed"], rec["case"], rec["hidden_sha256_plain"]), ("424242", "X-TEST-001", "ab" * 32))


class GeneratedCasesDeclareSeeds(unittest.TestCase):
    def test_secret_seed_modules(self):
        for case in ("A-LIN-002", "K-FA-003", "U-LOOP-003", "V-SMP-002", "E-ACT-001", "C-FA-004", "T-SPEC-003",
                     "V-GEN-001", "V-DST-004", "T-FM-001", "PH-SYM-005", "X-PPO-001", "U-LOOP-002", "A-HYB-001", "BI-PROT-001", "U-TTT-008", "C-SPEC-001", "C-KV-002",
                     "K-KB-001", "K-FUS-002", "X-ROBUST-009", "T-AUDIO-005", "X-SFT-005", "X-DPO-006",
                     "X-DATA-007", "U-KAN-009"):
            src = (ROOT / "tools" / "prepare" / (case.lower().replace("-", "_") + ".py")).read_text()
            self.assertIn("SECRET_SEED_ENV", src, case)


if __name__ == "__main__":
    unittest.main()
