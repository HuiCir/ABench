#!/usr/bin/env python3
"""Freeze the data of one ABench standard case (operator only, on a networked machine with the raw files).

  python3 tools/prepare/run.py --case TC-FT-001 --list-sources              # what to download, from where, licenses
  python3 tools/prepare/run.py --case TC-FT-001 --raw-dir RAW --key-file ~/.abench/hidden.key
  ABENCH_ALLOW_SYNTHETIC=1 python3 tools/prepare/run.py --case TC-FT-001 --synthetic 2000 --key-file K
      synthetic data is for plumbing tests only; the lock file is marked synthetic and the case stays awaiting_data
  python3 tools/prepare/run.py --case TC-FT-001 --activate                  # after the baseline calibration run

Writes sandbox/data/<ID>/*.npz (public), sandbox/hidden/<ID>/hidden.npz.enc (encrypted), sandbox/models/<ID>/...
(frozen pretrained files, if any) and cases/<ID>/case/data_lock.yaml, then regenerates case/oj.yaml.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE.parent))
import hidden_store  # noqa: E402
from prepare.common import Frozen, npz_bytes, sha256_file, tree_hash  # noqa: E402
import hashlib  # noqa: E402


def module_for(case: str):
    return importlib.import_module("prepare." + case.lower().replace("-", "_"))


def freeze(case: str, fz: Frozen, key_file: Path, synthetic: bool) -> dict:
    pub_dir = ROOT / "sandbox" / "data" / case
    hid_path = ROOT / "sandbox" / "hidden" / case / "hidden.npz.enc"
    mod_dir = ROOT / "sandbox" / "models" / case
    for d in (pub_dir, hid_path.parent):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    lock: dict = {"case_id": case, "synthetic": bool(synthetic), "activation": "awaiting_data",
                  "public_dir": str(pub_dir.relative_to(ROOT)), "public": []}
    for name, arrays in fz.public.items():
        b = npz_bytes(**arrays)
        p = pub_dir / name
        p.write_bytes(b)
        first = next(iter(arrays.values()))
        lock["public"].append({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(b).hexdigest(),
                               "keys": sorted(arrays), "rows": int(len(first))})
    for name, src in fz.raw_files.items():
        p = pub_dir / name
        shutil.copyfile(src, p)
        lock["public"].append({"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)})
    plain = npz_bytes(**fz.hidden)
    hid_path.write_bytes(hidden_store.encrypt_bytes(hidden_store.read_key(key_file), plain))
    hid_path.chmod(0o600)
    first = next(iter(fz.hidden.values()))
    lock["hidden"] = {"path": str(hid_path.relative_to(ROOT)), "sha256_plain": hashlib.sha256(plain).hexdigest(),
                      "keys": sorted(fz.hidden), "rows": int(len(first))}
    if fz.models:
        if mod_dir.exists():
            shutil.rmtree(mod_dir)
        lock["models_dir"] = str(mod_dir.relative_to(ROOT))
        lock["models"] = []
        for name, src in fz.models.items():
            dst = mod_dir / name
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", ".cache", "*.lock", "*.metadata"))
            lock["models"].append({"name": name, "path": str(dst.relative_to(ROOT)), "tree_sha256": tree_hash(dst)})
    lock["sources_sha256"] = fz.sources_sha256
    if fz.notes:
        lock["notes"] = fz.notes
    return lock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--raw-dir")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--key-file")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--activate", action="store_true")
    a = ap.parse_args()
    mod = module_for(a.case)
    lock_path = ROOT / "cases" / a.case / "case" / "data_lock.yaml"
    if a.list_sources:
        print(yaml.safe_dump(mod.SOURCES, allow_unicode=True, sort_keys=False, width=110))
        return 0
    if a.activate:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        if lock.get("synthetic"):
            raise SystemExit("refusing to activate a case whose data is synthetic")
        lock["activation"] = "active"
        lock_path.write_text(yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        if not a.key_file:
            raise SystemExit("--key-file is required")
        if a.synthetic:
            if os.environ.get("ABENCH_ALLOW_SYNTHETIC") != "1":
                raise SystemExit("synthetic data requires ABENCH_ALLOW_SYNTHETIC=1 (plumbing tests only)")
            fz = mod.synthetic(a.synthetic)
        elif a.raw_dir:
            fz = mod.build(Path(a.raw_dir))
        else:
            raise SystemExit("pass --raw-dir (real data) or --synthetic N (plumbing test)")
        lock = freeze(a.case, fz, Path(a.key_file), bool(a.synthetic))
        lock_path.write_text("# written by tools/prepare/run.py; do not edit by hand\n"
                             + yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_standard_cases.py"), "--case", a.case], check=True)
    print(f"{a.case}: lock {lock_path.relative_to(ROOT)} activation={lock['activation']}"
          + ("  [SYNTHETIC]" if lock.get("synthetic") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
