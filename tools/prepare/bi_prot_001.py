"""BI-PROT-001: ProteinGym v1.3 deep mutational scanning substitutions (Notin et al. 2023, MIT) with the frozen ESM-2
protein language models (facebook/esm2_t30_150M_UR50D and esm2_t33_650M_UR50D, MIT).

Eligible assays (from reference_files/DMS_substitutions.csv at commit 144fe22b): wild type at most 600 residues and
500-10,000 single-substitution mutants (158 of 217). Single substitutions only. All assays of one UniProt protein stay
together. With a secret seed, proteins are drawn until 30 held-out assays are reached with the selection types
stratified as in ProteinGym (12 Stability, 8 OrganismalFitness, 6 Activity, 3 Expression, 1 Binding; a type is
skipped once full); every other eligible assay is public with its labels. Per held-out assay, 64 random mutants with
labels are given to the submission (few-shot) and the rest are scored.
"""
from __future__ import annotations

import csv
import io
import os
import zipfile
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

PG_COMMIT = "144fe22b07dfaeec2b366f2346203a9838a55b4c"
ZIP_URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
REF_URL = f"https://github.com/OATML-Markslab/ProteinGym/raw/{PG_COMMIT}/reference_files/DMS_substitutions.csv"
ESM = {"esm2_t30_150M_UR50D": "a695f6045e2e32885fa60af20c13cb35398ce30c",
       "esm2_t33_650M_UR50D": "08e4846e537177426273712802403f7ba8261b6c"}
ESM_FILES = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]
SOURCES = [{"name": "DMS_ProteinGym_substitutions.zip", "url": ZIP_URL, "license": "MIT (ProteinGym); assays from their publications"},
           {"name": "DMS_substitutions.csv", "url": REF_URL, "license": "MIT"}] + [
          {"name": f"{m}/", "url": f"https://huggingface.co/facebook/{m}/tree/{r}", "files": ESM_FILES, "license": "MIT"}
          for m, r in ESM.items()]
SECRET_SEED_ENV = "ABENCH_BI_PROT_SEED"          # escrowed by tools/prepare/run.py next to the hidden-set key
QUOTA = {"Stability": 12, "OrganismalFitness": 8, "Activity": 6, "Expression": 3, "Binding": 1}
N_SHOT = 64


def eligible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r["seq_len"]) <= 600 and 500 <= int(r["DMS_number_single_mutants"]) <= 10000]


def choose_hidden(rows: list[dict], seed: int) -> set:
    rng = np.random.default_rng(seed)
    by_prot: dict = {}
    for r in rows:
        by_prot.setdefault(r["UniProt_ID"], []).append(r)
    left, hidden = dict(QUOTA), set()
    for p in rng.permutation(sorted(by_prot)):
        group = by_prot[p]
        need = {}
        for r in group:
            need[r["coarse_selection_type"]] = need.get(r["coarse_selection_type"], 0) + 1
        if all(left.get(t, 0) >= c for t, c in need.items()):
            for t, c in need.items():
                left[t] -= c
            hidden |= {r["DMS_id"] for r in group}
        if not any(left.values()):
            break
    assert not any(left.values()), left
    return hidden


def _assay(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    rows = [r for r in rows if ":" not in r["mutant"]]
    return np.asarray([r["mutant"] for r in rows]), np.asarray([float(r["DMS_score"]) for r in rows], np.float64)


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (selection of the held-out assays)")
    zp, rp = raw / "DMS_ProteinGym_substitutions.zip", raw / "DMS_substitutions.csv"
    sums = {zp.name: check(zp), rp.name: check(rp)}
    rows = eligible(list(csv.DictReader(rp.open(encoding="utf-8"))))
    assert len(rows) == 158, len(rows)
    hidden_ids = choose_hidden(rows, seed)
    rng = np.random.default_rng(seed + 1)
    public, hidden = {}, {}
    with zipfile.ZipFile(zp) as zf:
        names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".csv")}
        j = 0
        for r in sorted(rows, key=lambda r: r["DMS_id"]):
            mut, y = _assay(zf, names[r["DMS_id"]])
            assert len(mut) == int(r["DMS_number_single_mutants"]), r["DMS_id"]
            meta = {"wild_type": np.asarray(r["target_seq"]), "selection_type": np.asarray(r["coarse_selection_type"])}
            if r["DMS_id"] not in hidden_ids:
                public[f"dev_{r['DMS_id']}.npz"] = dict(meta, mutants=mut, scores=y, uniprot=np.asarray(r["UniProt_ID"]))
                continue
            p = rng.permutation(len(mut))
            k = f"a{j:02d}"
            hidden.update({f"{k}_wt": meta["wild_type"], f"{k}_type": meta["selection_type"],
                           f"{k}_fs_mut": mut[p[:N_SHOT]], f"{k}_fs_y": y[p[:N_SHOT]],
                           f"{k}_mut": mut[p[N_SHOT:]], f"{k}_y": y[p[N_SHOT:]], f"{k}_dms_id": np.asarray(r["DMS_id"])})
            j += 1
    assert j == 30, j
    models = {}
    for m in ESM:
        for f in ESM_FILES:
            sums[f"{m}/{f}"] = check(raw / m / f)
        models[m] = raw / m
    return Frozen(public=public, hidden=hidden, sources_sha256=sums, models=models)


def synthetic(n: int) -> Frozen:
    """Six toy assays whose fitness depends on the substituted amino acid's hydrophobicity."""
    rng = np.random.default_rng(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    hyd = rng.normal(size=20)
    public, hidden = {}, {}
    for a in range(6):
        wt = "".join(rng.choice(list(aa), 60))
        muts, ys = [], []
        for _ in range(max(n, N_SHOT + 20)):
            pos = int(rng.integers(60))
            to = aa[int(rng.integers(20))]
            muts.append(f"{wt[pos]}{pos + 1}{to}")
            ys.append(hyd[aa.index(to)] + 0.3 * rng.normal())
        mut, y = np.asarray(muts), np.asarray(ys)
        if a < 4:
            public[f"dev_toy{a}.npz"] = {"wild_type": np.asarray(wt), "selection_type": np.asarray("Stability"),
                                         "mutants": mut, "scores": y, "uniprot": np.asarray(f"TOY{a}")}
        else:
            k = f"a{a - 4:02d}"
            hidden.update({f"{k}_wt": np.asarray(wt), f"{k}_type": np.asarray("Stability"), f"{k}_fs_mut": mut[:N_SHOT],
                           f"{k}_fs_y": y[:N_SHOT], f"{k}_mut": mut[N_SHOT:], f"{k}_y": y[N_SHOT:],
                           f"{k}_dms_id": np.asarray(f"toy{a}")})
    return Frozen(public=public, hidden=hidden, sources_sha256={"synthetic": f"n{n}"})
