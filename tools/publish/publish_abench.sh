#!/usr/bin/env bash
# Publish ABench with the GitHub CLI logged in on this machine:
#   1. a PRIVATE repository with the clean code (no run records, frozen data, encrypted splits, keys or caches);
#   2. the static site (site/) to the existing GitHub Pages endpoint that serves ABench.
#
#   bash ~/Documents/ABench/tools/publish/publish_abench.sh discover   # read-only: account, repos, Pages endpoints
#   bash ~/Documents/ABench/tools/publish/publish_abench.sh repo       # create the private repo and push (first time)
#   bash ~/Documents/ABench/tools/publish/publish_abench.sh sync       # update the existing code repo to this tree
#   bash ~/Documents/ABench/tools/publish/publish_abench.sh web        # update the Pages endpoint and wait for it
#   bash ~/Documents/ABench/tools/publish/publish_abench.sh            # discover, repo, web
#
# Options (environment):
#   CODE_REPO   name of the private code repo (default ABench). GitHub names are case-insensitive, so if the Pages site
#               is served from a repo called "abench", choose another name, e.g. CODE_REPO=ABench-code.
#   PAGES_REPO  owner/name of the repo that serves the endpoint (default: discovered automatically)
#   PAGES_DIR   folder inside that repo's Pages source that holds the site (default: discovered: "" when the repo
#               itself is the abench site, "abench" or "ABench" when it is a folder of <owner>.github.io)
#   DRY_RUN=1   show what the sync and web steps would change, without committing or pushing
#   SYNC_MSG_FILE  file with the commit message of the sync step (default: a dated one-line message)
#   WEB_MSG     commit message of the web step
#   SKIP_TESTS=1   skip the test run of the sync step
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${ABENCH_PUBLISH_WORK:-$HOME/Documents/_abench_xfer/publish}"
CODE_REPO="${CODE_REPO:-ABench}"
TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_017i9Dv2ssM4n8pCN1j9YmVt'
say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

command -v gh >/dev/null || fail "gh not found"
command -v git >/dev/null || fail "git not found"
gh auth status >/dev/null 2>&1 || fail "gh is not logged in (gh auth login)"
LOGIN="$(gh api user -q .login)"
POLL="${PUBLISH_POLL_SECONDS:-6}"
mkdir -p "$WORK"

# Pages sites of the account whose URL or folder mentions abench: lines "repo<TAB>url<TAB>branch<TAB>path<TAB>build_type<TAB>dir"
pages_candidates() {
  gh api "users/$LOGIN/repos?per_page=100&type=owner" --paginate -q '.[] | select(.has_pages) | .full_name' |
  while read -r repo; do
    info="$(gh api "repos/$repo/pages" -q '[.html_url, (.source.branch // ""), (.source.path // "/"), (.build_type // "legacy")] | @tsv' 2>/dev/null || true)"
    [ -n "$info" ] || continue
    url="$(printf '%s' "$info" | cut -f1)"; branch="$(printf '%s' "$info" | cut -f2)"; path="$(printf '%s' "$info" | cut -f3)"
    if printf '%s' "$url" | grep -qi 'abench'; then
      printf '%s\t%s\t\n' "$repo" "$info"
    elif [ -n "$branch" ]; then
      sub="${path#/}"; [ -n "$sub" ] && sub="$sub/"
      for d in abench ABench Abench; do
        if gh api "repos/$repo/contents/${sub}$d?ref=$branch" -q '.[0].name' >/dev/null 2>&1; then
          printf '%s\t%s\t%s\n' "$repo" "$info" "$d"
        fi
      done
    fi
  done
}

discover() {
  say "discover (read-only)"
  echo "  account: $LOGIN"
  if gh repo view "$LOGIN/$CODE_REPO" >/dev/null 2>&1; then
    echo "  code repo $LOGIN/$CODE_REPO: EXISTS ($(gh repo view "$LOGIN/$CODE_REPO" --json visibility,isEmpty -q '.visibility + (if .isEmpty then ", empty" else ", not empty" end)'))"
  else
    echo "  code repo $LOGIN/$CODE_REPO: does not exist yet (will be created private)"
  fi
  echo "  Pages endpoints that look like ABench:"
  pages_candidates > "$WORK.candidates" || true
  if [ -s "$WORK.candidates" ]; then
    while IFS=$'\t' read -r repo url branch path bt dir; do
      title="$(curl -fsSL "${url%/}/${dir:+$dir/}" 2>/dev/null | grep -o '<title>[^<]*' | head -1 | sed 's/<title>//' || true)"
      echo "    repo=$repo url=${url%/}/${dir:+$dir/} branch=$branch path=$path build=$bt folder=${dir:-<repo root>} title=\"$title\""
    done < "$WORK.candidates"
  else
    echo "    none found automatically; set PAGES_REPO=owner/name and PAGES_DIR=folder"
  fi
}

export_clean() {   # clean copy of the repository into $WORK/ABench
  rm -rf "$WORK/ABench"; mkdir -p "$WORK/ABench"
  (cd "$ROOT" && tar -cf - --exclude=.DS_Store --exclude=.ruff_cache --exclude=__pycache__ --exclude='*.pyc' \
      --exclude=./runs --exclude=./sandbox/data --exclude=./sandbox/hidden --exclude=./sandbox/models \
      --exclude=./_to_delete --exclude=./.git --exclude='*.key' --exclude='*.enc' .) | tar -xf - -C "$WORK/ABench"
  # nothing secret or machine-specific may leave
  if grep -rIlE --exclude=publish_abench.sh 'sk-ant-[A-Za-z0-9]|ghp_[A-Za-z0-9]{20}|github_pat_|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|hf_[A-Za-z0-9]{30}' "$WORK/ABench" ; then
    fail "possible secret in the export (files listed above)"
  fi
  echo "  clean export: $(find "$WORK/ABench" -type f | wc -l | tr -d ' ') files, $(du -sh "$WORK/ABench" | cut -f1)"
}

repo() {
  say "private code repo $LOGIN/$CODE_REPO"
  mkdir -p "$WORK"
  if gh repo view "$LOGIN/$CODE_REPO" >/dev/null 2>&1; then
    vis="$(gh repo view "$LOGIN/$CODE_REPO" --json visibility -q .visibility)"
    empty="$(gh repo view "$LOGIN/$CODE_REPO" --json isEmpty -q .isEmpty)"
    [ "$vis" = PRIVATE ] || fail "$LOGIN/$CODE_REPO exists and is $vis (maybe the Pages site). Rerun with CODE_REPO=ABench-code"
    [ "$empty" = true ] || fail "$LOGIN/$CODE_REPO exists and is not empty; refusing to overwrite. Choose CODE_REPO=... or push manually"
  fi
  git config user.name >/dev/null && git config user.email >/dev/null \
    || fail "git user.name / user.email are not set (git config --global user.name ...; git config --global user.email ...)"
  export_clean
  cd "$WORK/ABench"
  (cd "$WORK/ABench" && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests >/dev/null 2>&1) \
    && echo "  tests pass on the clean export" || echo "  WARNING: tests did not pass on the clean export (missing packages?); continuing"
  find . -name __pycache__ -prune -exec rm -rf {} +
  git init -q -b main
  git add -A
  git commit -q -m "ABench: clean code snapshot (task statement v2.1, 20 standard cases, OJ judge, paper search, site)$TRAILER"
  if gh repo view "$LOGIN/$CODE_REPO" >/dev/null 2>&1; then
    git remote add origin "https://github.com/$LOGIN/$CODE_REPO.git"
    git push -u origin main
  else
    gh repo create "$CODE_REPO" --private --source . --remote origin --push \
      --description "ABench: architecture-design benchmark for research agents (task statements, OJ judge, sandbox)"
  fi
  echo "  https://github.com/$LOGIN/$CODE_REPO  (visibility: $(gh repo view "$LOGIN/$CODE_REPO" --json visibility -q .visibility))"
}

sync() {   # the code repo exists: replace its tree with a clean export of this one (deletions included)
  say "sync the code repo $LOGIN/$CODE_REPO"
  gh repo view "$LOGIN/$CODE_REPO" >/dev/null 2>&1 || fail "$LOGIN/$CODE_REPO does not exist; run the repo step first"
  git config user.name >/dev/null && git config user.email >/dev/null \
    || fail "git user.name / user.email are not set (git config --global user.name ...; git config --global user.email ...)"
  rm -rf "$WORK/ABench-clone"
  gh repo clone "$LOGIN/$CODE_REPO" "$WORK/ABench-clone" -- -q
  export_clean
  if [ "${SKIP_TESTS:-0}" != 1 ]; then          # about 15 minutes; SKIP_TESTS=1 when they were just run
    (cd "$WORK/ABench" && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests >/dev/null 2>&1) \
      && echo "  tests pass on the clean export" || echo "  WARNING: tests did not pass on the clean export (missing packages?); continuing"
  fi
  find "$WORK/ABench" -name __pycache__ -prune -exec rm -rf {} +
  mv "$WORK/ABench-clone/.git" "$WORK/ABench/.git"; rm -rf "$WORK/ABench-clone"
  cd "$WORK/ABench"
  git add -A
  if git diff --cached --quiet; then echo "  the repo already holds this tree"; return 0; fi
  git diff --cached --shortstat
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "  DRY_RUN=1: not committed"; return 0; fi
  if [ -n "${SYNC_MSG_FILE:-}" ]; then
    git commit -q -F "$SYNC_MSG_FILE"
  else
    git commit -q -m "ABench: sync $(date +%Y-%m-%d)$TRAILER"
  fi
  git push -q origin HEAD
  echo "  pushed $(git log -1 --format='%h %s') to https://github.com/$LOGIN/$CODE_REPO"
}

web() {
  say "Pages endpoint"
  mkdir -p "$WORK"
  local repo url branch path bt dir
  if [ -n "${PAGES_REPO:-}" ]; then
    repo="$PAGES_REPO"
    IFS=$'\t' read -r url branch path bt < <(gh api "repos/$repo/pages" -q '[.html_url, (.source.branch // ""), (.source.path // "/"), (.build_type // "legacy")] | @tsv')
    dir="${PAGES_DIR:-}"
  else
    pages_candidates > "$WORK.candidates" || true
    n="$(grep -c . "$WORK.candidates" || true)"
    [ "$n" = 1 ] || { cat "$WORK.candidates"; fail "found $n ABench Pages endpoints; set PAGES_REPO=owner/name (and PAGES_DIR) explicitly"; }
    IFS=$'\t' read -r repo url branch path bt dir < "$WORK.candidates"
    dir="${PAGES_DIR:-$dir}"
  fi
  [ "$bt" = legacy ] || fail "$repo publishes Pages with a GitHub Actions workflow ($bt); update it through that workflow instead"
  [ -n "$branch" ] || fail "$repo has no Pages source branch"
  local live="${url%/}/${dir:+$dir/}"
  echo "  repo=$repo branch=$branch path=$path folder=${dir:-<root>}  live=$live"
  old_title="$(curl -fsSL "$live" 2>/dev/null | grep -o '<title>[^<]*' | head -1 | sed 's/<title>//' || true)"
  echo "  currently served: \"$old_title\""

  rm -rf "$WORK/pages"; gh repo clone "$repo" "$WORK/pages" -- --branch "$branch" --depth 20 -q
  local target="$WORK/pages"; local sub="${path#/}"; [ -n "$sub" ] && target="$target/$sub"; [ -n "$dir" ] && target="$target/$dir"
  mkdir -p "$target"
  # keep repository plumbing that belongs to the Pages repo itself
  local keep=(CNAME .nojekyll .git .github README.md LICENSE _config.yml)
  if [ -z "$dir" ]; then
    find "$target" -mindepth 1 -maxdepth 1 | while read -r f; do
      b="$(basename "$f")"; k=0; for x in "${keep[@]}"; do [ "$b" = "$x" ] && k=1; done
      [ "$k" = 1 ] || rm -rf "$f"
    done
  else
    rm -rf "$target"; mkdir -p "$target"
  fi
  (cd "$ROOT/site" && tar -cf - --exclude=.DS_Store --exclude=server.mjs --exclude=start.command --exclude=README.md .) | tar -xf - -C "$target"
  touch "$WORK/pages/${sub:+$sub/}.nojekyll"
  cd "$WORK/pages"
  git add -A
  if git diff --cached --quiet; then echo "  the endpoint already serves this version"; return 0; fi
  git diff --cached --stat | tail -15
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "  DRY_RUN=1: not committed"; return 0; fi
  git commit -q -m "${WEB_MSG:-Update the ABench site}$TRAILER"
  git push -q origin "$branch"
  echo "  pushed; waiting for the Pages build"
  for _ in $(seq 1 40); do
    st="$(gh api "repos/$repo/pages/builds/latest" -q .status 2>/dev/null || echo unknown)"
    [ "$st" = built ] && break
    [ "$st" = errored ] && fail "Pages build errored: gh api repos/$repo/pages/builds/latest"
    sleep "$POLL"
  done
  for _ in $(seq 1 20); do
    if curl -fsSL "$live?nocache=$(date +%s)" 2>/dev/null | grep -q 'v2.1 · OJ'; then
      echo "  LIVE: $live serves the v2.1 site"; return 0
    fi
    sleep "$POLL"
  done
  echo "  pushed, but $live does not show v2.1 yet (CDN cache can take a few minutes); check it in a browser"
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(discover repo web)
for s in "${steps[@]}"; do
  case "$s" in discover|repo|sync|web) "$s" ;; *) fail "unknown step $s (discover repo sync web)" ;; esac
done
