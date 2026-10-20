#!/usr/bin/env python3
"""Regression tests for OJ submission result-file filtering."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_submission import SELF_REPORT  # noqa: E402
from sandbox_runner import collect_submission  # noqa: E402


class SelfReportFilterTest(unittest.TestCase):
    def test_result_data_is_reserved_but_source_modules_are_not(self) -> None:
        reserved = (
            "metrics.json",
            "metrics.jsonl",
            "results.csv",
            "scores.npy",
            "eval_results.parquet",
            "evaluation.pkl",
            "metrics",
        )
        source = (
            "src/metrics.py",
            "src/scorer.py",
            "src/result_types.py",
            "evaluation.py",
            "src/evaluation_metrics.py",
        )
        for path in reserved:
            with self.subTest(path=path):
                self.assertIsNotNone(SELF_REPORT.search(path))
        for path in source:
            with self.subTest(path=path):
                self.assertIsNone(SELF_REPORT.search(path))

    def test_collection_keeps_code_named_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dst = root / "agent", root / "submission"
            (src / "src").mkdir(parents=True)
            (src / "model.py").write_text("# entry\n", encoding="utf-8")
            (src / "src/metrics.py").write_text("def score(x): return x\n", encoding="utf-8")
            (src / "src/result_types.py").write_text("class Result: pass\n", encoding="utf-8")
            (src / "metrics.json").write_text("{}\n", encoding="utf-8")
            kept, dropped = collect_submission(
                src,
                dst,
                {"submission": {"allowed_top_level": ["model.py", "src"]}},
            )
            self.assertIn("src/metrics.py", kept)
            self.assertIn("src/result_types.py", kept)
            self.assertIn("metrics.json", dropped)
            self.assertTrue((dst / "src/metrics.py").is_file())
            self.assertFalse((dst / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
