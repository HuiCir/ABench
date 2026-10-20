"""Run the whole case end to end: the example submission + two control groups, writing a comparison summary.

  python sim/make_mock_runs.py && python scoring/run_demo.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = {
    "example_submission": (ROOT / "example_submission", ROOT / "sim/mock_judge.json"),
    "clone_muon": (ROOT / "sim/controls/clone_muon", ROOT / "sim/controls/clone_muon/judge.json"),
    "rankgate_linear_gate": (ROOT / "sim/controls/rankgate_linear_gate",
                             ROOT / "sim/controls/rankgate_linear_gate/judge.json"),
}
ROLE = {"example_submission": "Example submission (log-gated RankGate)",
        "clone_muon": "Negative: renamed Muon clone",
        "rankgate_linear_gate": "Negative: early linear-gate version (degenerates to Adafactor)"}


def main():
    rows = []
    for name, (sub, judge) in CANDIDATES.items():
        out = ROOT / "outputs" / name
        subprocess.run([sys.executable, str(ROOT / "scoring/run_case.py"), "--submission", str(sub),
                        "--runs", str(ROOT / f"outputs/mock/runs_{name}.json"), "--judge", str(judge),
                        "--out", str(out), "--allow-simulated"], check=True, capture_output=True)
        r = json.loads((out / "score_report.json").read_text(encoding="utf-8"))
        c, b = r["composite"], r["D"]["behavioral"]
        rows.append((name, c, r["M"]["aggregate_speedup"], b, r["D"]["mechanism"]["D_mech"],
                     "; ".join(f"{d['module']}: {d['status']}" for d in r["X"]["declarations"]) or "—"))

    yes = lambda x: "✓" if x else "✗"
    L = ["# ABench B-OPT-001 end-to-end run (simulated data)", "",
         "> Training losses and judge outputs are simulated; behavioral probes, code diff, paper weights and all scoring logic are real computations.", "",
         "| Submission | Role | L0 | SOTA | Differentiated | Aggregate speedup | M | D_beh (nearest reference) | D_mech | D | Q | X | Overall |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c, S, b, dm, xs in rows:
        g = c["gates"]
        L.append(f"| {name} | {ROLE[name]} | {yes(g['L0'])} | {yes(g['SOTA'])} | {yes(g['differentiated'])} | {S}× | {c['M']} | "
                 f"{b['D_beh']} ({b['nearest_reference']}) | {dm} | {c['D']} | {c['Q']} | {c['X']} | **{c['score']}** |")
    L += ["", "Local-discovery verdicts:", ""] + [f"- {name}: {xs}" for name, *_, xs in rows]
    L += ["", "Detailed reports: outputs/<submission>/score_report.md"]
    (ROOT / "outputs/summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
