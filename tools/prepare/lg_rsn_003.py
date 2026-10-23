"""LG-RSN-003: LegalBench (Guha et al. 2023) classification tasks under CC BY 4.0 or MIT, with the frozen
Qwen2.5-1.5B-Instruct (Apache-2.0).

Sources: the task data (train.tsv, test.tsv) from the Hugging Face dataset nguha/legalbench at REVISION, and each
task's base prompt (instructions and few-shot examples with {{column}} slots) from the LegalBench GitHub repository
at COMMIT. Only the columns used by the base prompt are given to pipelines (others, such as hearsay's `slice`, would
reveal the label).

Seen tasks (20): public = the task description, labels, training rows and up to 100 dev rows drawn from its test
split with a public seed; hidden = up to 300 other test rows drawn with the secret seed.
Unseen tasks (15): nothing public; hidden = the task description, labels, training rows and up to 300 test rows.
Three unseen tasks come from families with no seen task (unfair_tos, privacy_policy_qa,
international_citizenship_questions); the others share a family (contract_nli, cuad, maud, supply_chain) with a
seen task but ask about a different clause or standard.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

REVISION = "daec8237410aa23e3faf4bc41ad8b3a7e1696826"
COMMIT = "b46bf4ffae90524b2b72aaa30e7745fe9db64481"
MODEL, MODEL_REV = "Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SECRET_SEED_ENV = "ABENCH_LG_RSN_SEED"
SEEN = ["abercrombie", "hearsay", "ucc_v_common_law", "proa", "insurance_policy_interpretation",
        "function_of_decision_section", "oral_argument_question_purpose", "overruling", "corporate_lobbying",
        "nys_judicial_ethics", "citation_prediction_classification", "sara_entailment",
        "contract_nli_confidentiality_of_agreement", "contract_nli_explicit_identification", "contract_nli_limited_use",
        "cuad_anti-assignment", "cuad_audit_rights", "cuad_affiliate_license-licensee",
        "maud_accuracy_of_target_general_rw_bringdown_timing_answer",
        "supply_chain_disclosure_best_practice_accountability"]
NEW_FAMILY = ["unfair_tos", "privacy_policy_qa", "international_citizenship_questions"]
UNSEEN = NEW_FAMILY + ["contract_nli_inclusion_of_verbally_conveyed_information", "contract_nli_no_licensing",
                       "contract_nli_notice_on_compelled_disclosure", "cuad_affiliate_license-licensor",
                       "cuad_cap_on_liability", "cuad_change_of_control", "cuad_competitive_restriction_exception",
                       "maud_accuracy_of_fundamental_target_rws_bringdown_standard",
                       "maud_accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer",
                       "maud_additional_matching_rights_period_for_modifications_(cor)",
                       "supply_chain_disclosure_best_practice_audits",
                       "supply_chain_disclosure_best_practice_certification"]
N_DEV, N_TEST, DEV_SEED = 100, 300, 20230811
MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt"]
SOURCES = [{"name": "nguha/legalbench data/<task>/{train,test}.tsv for the 35 tasks",
            "url": f"https://huggingface.co/datasets/nguha/legalbench/tree/{REVISION}/data", "license": "CC BY 4.0 / MIT (per task)",
            "note": "RAW/legalbench/data/<task>/"},
           {"name": "HazyResearch/legalbench tasks/<task>/base_prompt.txt",
            "url": f"https://github.com/HazyResearch/legalbench/tree/{COMMIT}/tasks", "license": "per task (CC BY 4.0 / MIT)",
            "note": "RAW/legalbench-prompts/tasks/<task>/base_prompt.txt"},
           {"name": f"{MODEL}/", "url": f"https://huggingface.co/Qwen/{MODEL}/tree/{MODEL_REV}", "files": MODEL_FILES,
            "license": "Apache-2.0", "note": f"RAW/{MODEL}/"}]
SLOT = re.compile(r"\{\{(.*?)\}\}")


def read_tsv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def task_spec(name: str, template: str, train: list, test: list) -> dict:
    cols = sorted(set(SLOT.findall(template)))
    labels = sorted({r["answer"].strip() for r in train + test})
    return {"name": name, "prompt_template": template, "labels": labels, "columns": cols,
            "train": [{**{c: r.get(c, "") for c in cols}, "answer": r["answer"].strip()} for r in train]}


def _rows(rows: list, cols: list) -> list:
    return [{**{c: r.get(c, "") for c in cols}, "answer": r["answer"].strip()} for r in rows]


def _freeze(data: dict, seed: int, sums: dict) -> Frozen:
    rng_dev, rng = np.random.default_rng(DEV_SEED), np.random.default_rng(seed)
    pub, tasks, rows, rt, gold, group = [], [], [], [], [], []
    for name in SEEN + UNSEEN:
        template, train, test = data[name]
        spec = task_spec(name, template, train, test)
        test = _rows(test, spec["columns"])
        if name in SEEN:
            perm = rng_dev.permutation(len(test))
            n_dev = min(N_DEV, len(test) // 3)
            pub.append({**spec, "dev": [test[i] for i in perm[:n_dev]]})
            rest = [test[i] for i in perm[n_dev:]]
            grp = "seen"
        else:
            rest = test
            grp = "unseen_new_family" if name in NEW_FAMILY else "unseen"
        pick = rng.permutation(len(rest))[:N_TEST]
        tasks.append(spec)
        group.append(grp)
        for i in sorted(pick):
            r = rest[i]
            rows.append({c: r[c] for c in spec["columns"]})
            rt.append(len(tasks) - 1)
            gold.append(r["answer"])
    order = rng.permutation(len(tasks))                       # task order in the hidden file reveals nothing
    remap = {int(o): j for j, o in enumerate(order)}
    tasks, group = [tasks[o] for o in order], [group[o] for o in order]
    rt = [remap[k] for k in rt]
    hidden = {"tasks_json": np.asarray([json.dumps(t) for t in tasks], dtype=np.str_),
              "rows_json": np.asarray([json.dumps(r) for r in rows], dtype=np.str_),
              "row_task": np.asarray(rt, np.int64), "gold": np.asarray(gold, dtype=np.str_),
              "task_labels_json": np.asarray([json.dumps(t["labels"]) for t in tasks], dtype=np.str_),
              "task_group": np.asarray(group, dtype=np.str_), "task_name": np.asarray([t["name"] for t in tasks], dtype=np.str_)}
    return Frozen(public={}, hidden=hidden, sources_sha256=sums, raw_files={"legalbench_public.json": _write_public(pub)},
                  notes={"rows_per_task": {t["name"]: int(sum(1 for k in rt if k == j)) for j, t in enumerate(tasks)}})


def _write_public(pub: list) -> Path:
    import tempfile
    p = Path(tempfile.mkdtemp(prefix="lgrsn-")) / "legalbench_public.json"
    p.write_text(json.dumps({"tasks": pub}, ensure_ascii=False), encoding="utf-8")
    return p


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (test-row sampling)")
    data, sums = {}, {}
    for name in SEEN + UNSEEN:
        d, pr = raw / "legalbench" / "data" / name, raw / "legalbench-prompts" / "tasks" / name / "base_prompt.txt"
        for f in ("train.tsv", "test.tsv"):
            sums[f"{name}/{f}"] = check(d / f)
        sums[f"{name}/base_prompt.txt"] = check(pr)
        data[name] = (pr.read_text(encoding="utf-8"), read_tsv(d / "train.tsv"), read_tsv(d / "test.tsv"))
        assert data[name][2], name
    return Frozen(**{**_freeze(data, seed, sums).__dict__, "models": {MODEL: raw / MODEL}})


def synthetic(n: int) -> Frozen:
    """Plumbing only: 35 tasks whose label is a keyword in the text (n test rows each)."""
    rng = np.random.default_rng(0)
    data = {}
    for name in SEEN + UNSEEN:
        labels = ["Yes", "No"] if rng.random() < 0.7 else ["A", "B", "C"]
        def row(i):
            lab = labels[i % len(labels)]
            return {"index": str(i), "text": f"clause {i} marked {lab.lower()} by the parties", "answer": lab}
        template = "Decide the label.\n\nText: {{text}}\nLabel:"
        data[name] = (template, [row(i) for i in range(len(labels))], [row(i) for i in range(n)])
    return _freeze(data, 12345, {"synthetic": f"n{n}"})
