#!/usr/bin/env bash
# Local LiveKit server control.
#
# Built from source into .local/livekit-server. The project ships no macOS
# binaries and its install script routes through Homebrew, so it is built here
# with CGO_ENABLED=1 (go-osstat needs cgo for darwin CPU counters).
#
# --dev uses the well-known devkey/secret credentials. Local development only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/.local/livekit-server"
PIDFILE="$ROOT/.local/livekit.pid"
PORT=7880
LOG="$ROOT/.local/livekit.log"

case "${1:-status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    # BIND=0.0.0.0 to reach the SFU from a phone on the same network.
    # Localhost by default: --dev uses the well-known devkey/secret pair, so
    # binding wider puts an unauthenticated media server on the LAN.
    # Refuse to start while the port is still held. LiveKit shuts down
    # gracefully and keeps answering HTTP while it drains, so a start issued
    # straight after a stop fails to bind and dies silently, leaving a health
    # check that passes against the process being killed.
    for _ in $(seq 1 40); do
      lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break
      sleep 0.5
    done
    if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
      echo "port $PORT is still in use; something else is listening" >&2
      exit 1
    fi

    "$BIN" --dev --bind "${BIND:-127.0.0.1}" >"$LOG" 2>&1 &
    PID=$!
    echo $PID > "$PIDFILE"

    # Confirm it survived. Writing a pidfile for a process that died on bind
    # is how the failure above stayed invisible.
    sleep 1
    if ! kill -0 "$PID" 2>/dev/null; then
      rm -f "$PIDFILE"
      echo "livekit-server exited immediately; see $LOG" >&2
      tail -3 "$LOG" >&2
      exit 1
    fi
    echo "livekit-server started (pid $PID) on ${BIND:-127.0.0.1}:$PORT, log: $LOG"
    ;;
  stop)
    if [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null; then
      rm -f "$PIDFILE"
      # Wait for the port, not the process: shutdown is graceful and the
      # socket outlives the SIGTERM by a few seconds.
      for _ in $(seq 1 40); do
        lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break
        sleep 0.5
      done
      echo "stopped"
    else
      rm -f "$PIDFILE"
      echo "not running"
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running (pid $(cat "$PIDFILE"))"
    else
      echo "not running"
    fi
    ;;
  log) tail -n "${2:-40}" "$LOG" ;;
  *) echo "usage: $0 {start|stop|status|log}" >&2; exit 2 ;;
esac
