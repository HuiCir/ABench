#!/usr/bin/env python3
"""Print the essentials of an evaluation record (evaluation.json written by tools/oj_eval.py, or a run directory of
tools/sandbox_runner.py): verdict, isolation, primary metric with its interval, spread over seeds, gates."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if p.is_dir():
    p = p / "evaluation.json"
e = json.loads(p.read_text())
print(f"record        {p}")
print(f"judge         verdict={e.get('verdict')} scorable={e.get('scorable')} isolation={e.get('judge_isolation')}")
print(f"              reason: {e.get('scoreability_reason')}")
judge_gpus = [g["name"] for g in e.get("judge_gpu_inventory") or [] if g]
agent = e.get("agent_hardware") or {}
if judge_gpus or agent.get("gpus"):
    reference = e.get("judge_gpu_class_matches") is not False and agent.get("gpu_class_matches") is not False
    print(f"gpus          agent {', '.join(g['name'] for g in agent.get('gpus') or []) or '-'}; "
          f"judge {', '.join(judge_gpus) or '-'}" + ("" if reference else "  (non-reference)"))
d = e.get("judge_isolation_detail") or {}
if d:
    pf = d.get("preflight") or {}
    print(f"preflight     user={d.get('user')} netns={d.get('netns')} network={d.get('network_probe')} "
          f"unreachable={sum(not c['reachable'] for c in pf.get('forbidden_paths', []))}/{len(pf.get('forbidden_paths', []))}")
m = e.get("metrics") or {}
pm = (e.get("primary") or {}).get("metric")
if pm:
    print(f"primary       {pm} = {m.get(pm)}  95% CI {e.get('primary_ci95')}")
    if f"{pm}_sd_over_runs" in m:
        print(f"              sd over seeds = {m[f'{pm}_sd_over_runs']} ({e.get('n_seeds')} seeds)")
if e.get("gates"):
    print("gates         " + ", ".join(f"{k}:{'pass' if v['pass'] else 'FAIL'}" for k, v in e["gates"].items()))
for r in e.get("runs") or []:
    print(f"seed {r.get('seed_index')}        status={r.get('status')} stages=" +
          ", ".join(f"{s['stage']}:{s['status']}:{s.get('timed_seconds')}s" for s in r.get("stages", [])))
if e.get("baseline"):
    bm = e["baseline"].get("metrics") or {}
    print(f"baseline      {e['baseline'].get('path')} (run by the judge next to the submission): "
          + (f"{pm} = {bm.get(pm)}" if pm in bm else f"{len(bm)} metrics"))
if e.get("error"):
    print(f"judge error   {e['error']}")
