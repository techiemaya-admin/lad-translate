#!/usr/bin/env bash
# Local Postgres control. Binaries live under .local/, no admin needed.
# Run tools/bootstrap.sh once to create the cluster this drives.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/.local/pgsql/bin"
DATA="$ROOT/.local/pgdata"
PORT=55432

# Postgres refuses to start as root, so on a box where this runs as root every
# command drops to the postgres account, which is the owner bootstrap.sh gives
# the data directory. On a normal user account this is a no-op.
run() {
  local bin="$1"; shift
  if [ "$(id -u)" = 0 ] && id postgres >/dev/null 2>&1; then
    local quoted="" a
    for a in "$@"; do quoted+=" $(printf '%q' "$a")"; done
    su postgres -c "$(printf '%q' "$BIN/$bin")$quoted"
  else
    "$BIN/$bin" "$@"
  fi
}

case "${1:-status}" in
  start) run pg_ctl -D "$DATA" -o "-p $PORT -k $ROOT/.local" -l "$ROOT/.local/pg.log" start ;;
  stop)  run pg_ctl -D "$DATA" stop ;;
  status) run pg_ctl -D "$DATA" status ;;
  psql)  shift; "$BIN/psql" -h 127.0.0.1 -p $PORT -U lad -d salesmaya_agent "$@" ;;
  *) echo "usage: $0 {start|stop|status|psql}" >&2; exit 2 ;;
esac
