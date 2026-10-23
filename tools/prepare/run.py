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
import re
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


HF_RESOLVE = re.compile(r"^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)$")
HF_TREE = re.compile(r"^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/tree/([0-9a-f]{40})/?$")


def _get_json(url: str):
    import json
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def _git_blob_sha1(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _hf_expected(dataset: bool, repo: str, rev: str, path: str) -> dict:
    """Hashes Hugging Face records for one file at a pinned revision: LFS sha256, or the git blob sha1."""
    import posixpath
    d = posixpath.dirname(path)
    api = f"https://huggingface.co/api/{'datasets/' if dataset else 'models/'}{repo}/tree/{rev}" + (f"/{d}" if d else "")
    for e in _get_json(api):
        if e.get("path") == path:
            return {"sha256": (e.get("lfs") or {}).get("oid"), "git_sha1": e.get("oid"), "size": e.get("size")}
    raise SystemExit(f"{path} not found at {repo}@{rev}")


def _locate(raw: Path, name: str) -> Path | None:
    p = raw / name
    if p.is_file():
        return p
    hits = [x for x in raw.rglob(Path(name).name) if x.is_file()]
    return hits[0] if len(hits) == 1 else None


def verify_sources(mod, raw: Path, allow_unverified: bool) -> list[dict]:
    """Check every raw file against what its publisher records: Hugging Face hashes at the pinned revision, or a
    published checksum file (checksum_url). md5/sha256 given in SOURCES are checked by the case module itself."""
    import urllib.request
    out = []
    for s in getattr(mod, "SOURCES", []):
        url, name = s.get("url") or "", s.get("name") or ""
        targets = []
        m = HF_RESOLVE.match(url)
        if m:
            targets.append((name, bool(m.group(1)), m.group(2), m.group(3), m.group(4)))
        m = HF_TREE.match(url)
        if m and s.get("files"):
            targets += [(f"{name.rstrip('/')}/{f}", bool(m.group(1)), m.group(2), m.group(3), f) for f in s["files"]]
        for local_name, ds, repo, rev, path in targets:
            p = _locate(raw, local_name)
            row = {"file": local_name, "source": f"{repo}@{rev[:12]}/{path}"}
            if p is None:
                row["status"] = "raw file not found"
            else:
                try:
                    exp = _hf_expected(ds, repo, rev, path)
                    if exp.get("sha256"):
                        ok = sha256_file(p) == exp["sha256"]
                        row.update(check="lfs sha256", ok=ok)
                    else:
                        ok = _git_blob_sha1(p) == exp["git_sha1"]
                        row.update(check="git blob sha1", ok=ok)
                    row["status"] = "verified" if ok else "MISMATCH"
                except (OSError, ValueError) as e:
                    row["status"] = f"unverified ({type(e).__name__}: {e})"
            out.append(row)
        if s.get("checksum_url"):
            p = _locate(raw, name)
            row = {"file": name, "source": s["checksum_url"]}
            try:
                with urllib.request.urlopen(s["checksum_url"], timeout=60) as r:
                    want = re.findall(r"\b[0-9a-f]{32}\b", r.read().decode())[0]
                got = hashlib.md5(p.read_bytes()).hexdigest() if p else None
                row.update(check="published md5", ok=got == want, status="verified" if got == want else "MISMATCH")
            except (OSError, IndexError, ValueError) as e:
                row["status"] = f"unverified ({type(e).__name__}: {e})"
            out.append(row)
    for r in out:
        print(f"  {r['status']:>10}  {r['file']}  <- {r['source']}")
    bad = [r for r in out if r["status"] == "MISMATCH" or r["status"] == "raw file not found"]
    if bad:
        raise SystemExit(f"source verification failed for {[r['file'] for r in bad]}")
    if not allow_unverified and any(r["status"].startswith("unverified") for r in out):
        raise SystemExit("some sources could not be verified (offline?); rerun with network, or pass --allow-unverified "
                         "to record them as unverified in the lock file")
    return out


def escrow_seed(case: str, env: str, key_file: Path, hidden_sha: str) -> str:
    """Store the secret generator seed next to the hidden-set key (0600), so the held-out set can be regenerated."""
    import json
    import time
    d = key_file.expanduser().resolve().parent / "seeds"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    p = d / f"{case}.json"
    body = json.dumps({"case": case, "env": env, "seed": os.environ[env], "hidden_sha256_plain": hidden_sha,
                       "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # never readable by others, even briefly
    with os.fdopen(fd, "w") as f:
        f.write(body)
    p.chmod(0o600)
    return str(p)


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
        p.parent.mkdir(parents=True, exist_ok=True)
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
    if fz.judge_models:                                  # evaluator-only models, e.g. a reward model
        jdir = ROOT / "sandbox" / "judge_models" / case
        if jdir.exists():
            shutil.rmtree(jdir)
        lock["judge_models_dir"] = str(jdir.relative_to(ROOT))
        lock["judge_models"] = []
        for name, src in fz.judge_models.items():
            dst = jdir / name
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", ".cache", "*.lock", "*.metadata"))
            lock["judge_models"].append({"name": name, "path": str(dst.relative_to(ROOT)), "tree_sha256": tree_hash(dst)})
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
    ap.add_argument("--allow-unverified", action="store_true",
                    help="record sources that could not be checked against their publisher (no network) instead of failing")
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
            seed_env = getattr(mod, "SECRET_SEED_ENV", None)
            if seed_env and not os.environ.get(seed_env):
                raise SystemExit(f"set {seed_env} to a secret integer before freezing {a.case}")
            print(f"{a.case}: verifying raw files against their publishers")
            verified = verify_sources(mod, Path(a.raw_dir), a.allow_unverified)
            fz = mod.build(Path(a.raw_dir))
        else:
            raise SystemExit("pass --raw-dir (real data) or --synthetic N (plumbing test)")
        lock = freeze(a.case, fz, Path(a.key_file), bool(a.synthetic))
        if a.raw_dir and not a.synthetic:
            lock["source_verification"] = verified
            if seed_env:
                where = escrow_seed(a.case, seed_env, Path(a.key_file), lock["hidden"]["sha256_plain"])
                lock["secret_seed"] = {"env": seed_env, "escrowed": True,
                                       "note": "stored next to the hidden-set key (seeds/<case>.json, 0600), not in the repository"}
                print(f"{a.case}: secret seed escrowed at {where}")
        lock_path.write_text("# written by tools/prepare/run.py; do not edit by hand\n"
                             + yaml.safe_dump(lock, allow_unicode=True, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_standard_cases.py"), "--case", a.case], check=True)
    print(f"{a.case}: lock {lock_path.relative_to(ROOT)} activation={lock['activation']}"
          + ("  [SYNTHETIC]" if lock.get("synthetic") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
