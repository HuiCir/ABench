#!/usr/bin/env bash
# ABench OJ calibration (code only, no agent): freeze a case's data, train its reference submission with the
# reference train.py, and judge it with tools/oj_eval.py on the held-out split. The record gives the base score of the
# case (the reference every submission is compared with), its 95% interval, and shows that freezing, decryption, the
# stage protocol and the metrics work on this host.
#
#   sudo bash sandbox/calibration/calibrate.sh DS-PIPE-002                     # synthetic data (plumbing)
#   sudo bash sandbox/calibration/calibrate.sh TC-FT-001
#   sudo DATA=/path/to/raw bash sandbox/calibration/calibrate.sh FN-LTSF-005    # real data
#
# Steps
#   1. host checks; the key (KEY, default ~/.abench/hidden.key, created 0600 if missing)
#   2. a copy of the repository in a directory only root can read ($ABENCH_CAL_HOME/work/<case>, default
#      ~/.abench/calibration), so freezing never touches your checkout
#   3. freeze: DATA=synthetic (default; never scorable) or DATA=<raw dir> (python3 tools/prepare/run.py --case <ID>
#      --list-sources); ALLOW_UNVERIFIED=1 records sources that could not be checked online
#   4. train the reference checkpoint: example_submission/train.py on the public files, with the final flag of
#      sandbox/calibration/cases.yaml
#   5. judge: tools/oj_eval.py; with root and the user abench-judge the submission's code runs as that user in a new
#      network namespace (ABENCH_JUDGE_CHILD_USER), otherwise as the caller (recorded as host_soft)
#   6. the record goes to runs/<case>/<run id>/evaluation.json of this checkout; a summary is printed
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CASE="${1:?usage: calibrate.sh CASE_ID}"
PY="${PYTHON:-python3}"
DATA="${DATA:-synthetic}"
KEY="${KEY:-$HOME/.abench/hidden.key}"
CAL="${ABENCH_CAL_HOME:-$HOME/.abench/calibration}"
RUN_ID="${RUN_ID:-ojcal-$(date +%Y%m%d-%H%M%S)-$(echo "$CASE" | tr 'A-Z' 'a-z')}"
say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

say "host checks"
"$PY" -c "import numpy, scipy, sklearn, yaml" 2>/dev/null || fail "$PY needs numpy, scipy, scikit-learn and pyyaml"
"$PY" -c "import sys, yaml; c = yaml.safe_load(open('$ROOT/sandbox/calibration/cases.yaml')); sys.exit(0 if '$CASE' in c else 1)" \
  || fail "$CASE is not in sandbox/calibration/cases.yaml"
cfg() { "$PY" -c "import yaml; c = yaml.safe_load(open('$ROOT/sandbox/calibration/cases.yaml')); v = c['$CASE'].get('$1'); print('' if v is None else v)"; }
ISO=0
if [ "$(uname -s)" = Linux ] && [ "$(id -u)" = 0 ] && id abench-judge >/dev/null 2>&1 \
   && command -v setpriv >/dev/null && command -v unshare >/dev/null; then
  ISO=1; echo "  judge children: user abench-judge, new network namespace"
else
  echo "  judge children: run as $(id -un) (needs Linux, root and the user abench-judge for isolation; recorded as host_soft)"
fi
mkdir -p "$(dirname "$KEY")"; chmod 700 "$(dirname "$KEY")"
[ -f "$KEY" ] || "$PY" "$ROOT/tools/hidden_store.py" keygen "$KEY"
chmod 600 "$KEY"
echo "  key: $KEY"

say "work tree"
WORK="$CAL/work/$CASE"
mkdir -p "$CAL"; chmod 700 "$CAL"
rm -rf "$WORK"; mkdir -p "$WORK"
(cd "$ROOT" && tar -cf - --exclude=./runs --exclude=./.git --exclude=__pycache__ --exclude=./sandbox/data \
    --exclude=./sandbox/hidden --exclude=./sandbox/models .) | tar -xf - -C "$WORK"
echo "  $WORK"
cd "$WORK"

say "freeze data ($DATA)"
if [ "$DATA" = synthetic ]; then
  export ABENCH_ALLOW_SYNTHETIC=1
  "$PY" tools/prepare/run.py --case "$CASE" --synthetic "$(cfg synthetic_rows)" --key-file "$KEY" | tail -1
else
  [ -d "$DATA" ] || fail "DATA must be 'synthetic' or a directory with the raw files"
  "$PY" tools/prepare/run.py --case "$CASE" --raw-dir "$DATA" --key-file "$KEY" ${ALLOW_UNVERIFIED:+--allow-unverified} | tail -1
fi

say "train the reference checkpoint"
SUB="$WORK/reference_submission"
rm -rf "$SUB"; cp -R "cases/$CASE/example_submission" "$SUB"; rm -rf "$SUB/checkpoint" "$SUB/__pycache__"
PUB="$("$PY" -c "import yaml; print(yaml.safe_load(open('cases/$CASE/case/oj.yaml'))['data']['public_dir'])")"
MOD="$("$PY" -c "import yaml; print(yaml.safe_load(open('cases/$CASE/case/oj.yaml'))['data'].get('models_dir') or '')")"
# the environment train.py documents: the interface files (as the sandbox view lays them out) and the data directories
VIEW="$WORK/reference_view"; rm -rf "$VIEW"; mkdir -p "$VIEW"; cp -R "cases/$CASE/harness" "$VIEW/interface"
export ABENCH_VIEW_DIR="$VIEW" ABENCH_DATA_DIR="$WORK/$PUB"
[ -n "$MOD" ] && export ABENCH_MODELS_DIR="$WORK/$MOD"
t0=$(date +%s)
(cd "$SUB" && PYTHONDONTWRITEBYTECODE=1 "$PY" train.py --data "$WORK/$PUB" --out checkpoint $(cfg final_flag)) > "$CAL/$RUN_ID.train.log" 2>&1 \
  || { tail -20 "$CAL/$RUN_ID.train.log"; fail "reference train.py failed"; }
echo "  trained in $(( $(date +%s) - t0 )) s ($(du -sh "$SUB/checkpoint" | cut -f1)); log $CAL/$RUN_ID.train.log"

say "judge (tools/oj_eval.py)"
# the judge copies what its children need into its own temporary directory; the work tree stays unreadable to them
[ "$ISO" = 1 ] && export ABENCH_JUDGE_CHILD_USER=abench-judge
export ABENCH_JUDGE_PYTHONPATH="${ABENCH_JUDGE_PYTHONPATH:-$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}"
export ABENCH_JUDGE_TOOLS_SHA256="$("$PY" -c "import sys; sys.path.insert(0, 'tools'); import oj_eval; print(oj_eval.tree_hash(oj_eval.TOOLS))")"
DEV="$(cfg device)"; [ -n "$DEV" ] && export ABENCH_JUDGE_DEVICE="$DEV"
OUT="$ROOT/runs/$CASE/$RUN_ID/evaluation.json"
mkdir -p "$(dirname "$OUT")"
set +e
"$PY" tools/oj_eval.py --case "$CASE" --submission "$SUB" --hidden-key-file "$KEY" --out "$OUT" > "$CAL/$RUN_ID.judge.log" 2>&1
rc=$?
set -e
[ -f "$OUT" ] || { tail -30 "$CAL/$RUN_ID.judge.log"; fail "the judge did not write $OUT"; }
cp "$CAL/$RUN_ID.train.log" "$ROOT/runs/$CASE/$RUN_ID/reference_train.log"

say "result (judge exit code $rc)"
"$PY" "$ROOT/sandbox/calibration/summarize.py" "$OUT"
