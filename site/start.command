#!/bin/bash
# Double-click to start the ABench demo site (macOS). Tries Node (including nvm and Homebrew installs), Bun, then Python 3.
cd "$(dirname "$0")" || exit 1
PORT=8765
URL="http://localhost:$PORT"
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use; opening $URL directly"
  open "$URL"
  exit 0
fi
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/.bun/bin"
( sleep 1; open "$URL" ) &
if command -v node >/dev/null 2>&1; then
  exec node server.mjs "$PORT"
elif command -v bun >/dev/null 2>&1; then
  exec bun server.mjs "$PORT"
elif xcode-select -p >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  echo "ABench demo site running: $URL (press Ctrl+C to stop)"
  exec python3 -m http.server "$PORT" --bind 127.0.0.1
else
  echo "node, bun or python3 not found. You can double-click index.html to open it directly; no server is needed."
  open index.html
  read -r -p "Press Enter to close"
fi
