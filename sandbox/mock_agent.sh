#!/usr/bin/env bash
# ABench plumbing agent: walks the Workflow of a checkpoint-mode standard case using ONLY the agent view, without an
# LLM. It proves that the launch path works before real agent time is spent:
#   view readable -> paper tool callable -> reference code runs -> selfcheck passes -> submission collected -> judged.
#   python3 tools/sandbox_runner.py --case TC-FT-001 --harness command --smoke --device cpu \
#       --command "bash $PWD/sandbox/mock_agent.sh"
# Everything it does is logged to stdout (kept in the run record as logs/agent_stdout.log).
set -euo pipefail
V="$ABENCH_VIEW_DIR"; SUB="$ABENCH_SUBMISSION_DIR"; PY="${PYTHON:-python3}"
step() { printf '\n[mock-agent] %s\n' "$*"; }

step "0 environment"
for v in ABENCH_CASE_ID ABENCH_VIEW_DIR ABENCH_DATA_DIR ABENCH_WORKSPACE ABENCH_SUBMISSION_DIR ABENCH_DEVICE \
         ABENCH_DEADLINE ABENCH_CUTOFF ABENCH_RETRIEVAL_LOG; do
  printf '  %-22s %s\n' "$v" "${!v:-<unset>}"
done
echo "  python: $($PY -c 'import sys; print(sys.version.split()[0])')  cwd: $PWD"
$PY -c "import numpy, sklearn; print('  numpy', numpy.__version__, 'sklearn', sklearn.__version__)"

step "1 read the view"
(cd "$V" && find . -type f | sort | sed 's/^/  /')
for f in task.md interface/interface.md interface/sandbox_guide.md; do
  test -s "$V/$f" || { echo "missing $f"; exit 1; }; echo "  $f: $(wc -l < "$V/$f") lines"
done
grep -q '### Workflow' "$V/task.md" && echo "  task.md has the Workflow"
if touch "$V/.write_probe" 2>/dev/null; then echo "  WARNING: the view is writable"; rm -f "$V/.write_probe"; else echo "  view is read-only"; fi
ls "$ABENCH_DATA_DIR"

step "2 retrieve (offline paper search)"
$PY "$V/tools/abench_papers.py" info || echo "  (no paper index installed: search unavailable, continuing)"
$PY "$V/tools/abench_papers.py" search "text classification n-gram linear" -n 3 || true

step "3 develop: start from interface/baseline"
cp "$V"/interface/baseline/*.py "$SUB"/
(cd "$SUB" && $PY train.py --data "$ABENCH_DATA_DIR" --out "$SUB/checkpoint")
du -sh "$SUB/checkpoint"

step "4 verify with the self-check"
$PY "$V/interface/selfcheck.py" --quick
$PY "$V/interface/selfcheck.py" --seeds 1

step "5 report and references (templates, with the case id filled in)"
mkdir -p "$SUB/report"
sed "s/⟨CASE_ID⟩/$ABENCH_CASE_ID/g" "$V/templates/report_template.html" > "$SUB/report/index.html"
cp "$V/templates/references_template.yaml" "$SUB/references.yaml"
(cd "$SUB" && find . -type f | sort | sed 's/^/  /')
step "done"
