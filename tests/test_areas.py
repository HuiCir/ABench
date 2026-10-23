"""spec/areas.yaml: the testing and ownership split of the cases."""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Areas(unittest.TestCase):
    def setUp(self):
        self.areas = yaml.safe_load((ROOT / "spec" / "areas.yaml").read_text(encoding="utf-8"))["areas"]
        idx = yaml.safe_load((ROOT / "index.yaml").read_text(encoding="utf-8"))
        self.ids = [c["id"] for t in idx["tracks"] for c in t["cases"]]

    def test_every_case_in_exactly_one_subarea(self):
        listed = [cid for a in self.areas for sa in a["subareas"] for cid in sa["cases"]]
        self.assertEqual(sorted(listed), sorted(self.ids))
        self.assertEqual(len(listed), len(set(listed)))

    def test_sizes_and_pro_apart(self):
        for a in self.areas:
            cases = [cid for sa in a["subareas"] for cid in sa["cases"]]
            if a["id"] == "pro":
                self.assertTrue(all(c.startswith("PRO-") for c in cases))
                self.assertEqual(sorted(cases), sorted(c for c in self.ids if c.startswith("PRO-")))
            else:
                self.assertTrue(15 <= len(cases) <= 29, (a["id"], len(cases)))
                self.assertGreaterEqual(len(a["subareas"]), 2, a["id"])
                self.assertFalse(any(c.startswith("PRO-") for c in cases))


if __name__ == "__main__":
    unittest.main()
