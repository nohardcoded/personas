#!/usr/bin/env bash
# persona-gen launcher.  Usage: bash scripts/run.sh [port]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PORT="${1:-7860}"
# Localhost only by default - the app has no auth. To expose it on your LAN (at your own
# risk), run:  HOST=0.0.0.0 bash scripts/run.sh
HOST="${HOST:-127.0.0.1}"
[ -x .venv/bin/python ] || { echo "Not installed. Run: bash scripts/install.sh"; exit 1; }
[ -f config.yaml ] || { echo "No config.yaml. Run install.sh, or: cp config.example.yaml config.yaml"; exit 1; }
echo "Starting persona-gen -> http://localhost:$PORT"
PYTHONPATH=src .venv/bin/python -m persona_gen.server --port "$PORT" --host "$HOST" &
SRV=$!; trap 'kill $SRV 2>/dev/null' EXIT INT TERM
# Probe the address the server actually bound to (a wildcard bind maps to loopback), so a
# LAN bind like HOST=192.168.x.x is not wrongly declared dead.
PROBE="$HOST"; [ "$HOST" = "0.0.0.0" ] && PROBE=127.0.0.1; [ "$HOST" = "::" ] && PROBE=::1
case "$PROBE" in *:*) PURL="[$PROBE]";; *) PURL="$PROBE";; esac
# Wait for the server to answer, but bail out if it died (e.g. missing model / bad config)
# instead of opening the browser and claiming it is running.
UP=0
for _ in $(seq 1 30); do
  kill -0 "$SRV" 2>/dev/null || { echo "Server exited during startup - see the error above (often: model not downloaded, or bad config.yaml)."; wait "$SRV"; exit 1; }
  curl -s "http://$PURL:$PORT/api/status" >/dev/null 2>&1 && { UP=1; break; }
  sleep 1
done
[ "$UP" = 1 ] || { echo "Server did not become ready in time on port $PORT."; kill "$SRV" 2>/dev/null; exit 1; }
open "http://localhost:$PORT" 2>/dev/null || true
echo "Running. Ctrl-C or close the terminal to stop."
wait $SRV
