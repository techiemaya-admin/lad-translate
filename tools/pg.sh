#!/usr/bin/env bash
# Local Postgres control. Binaries live under .local/, no admin needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/.local/pgsql/bin"
DATA="$ROOT/.local/pgdata"
PORT=55432

case "${1:-status}" in
  start) "$BIN/pg_ctl" -D "$DATA" -o "-p $PORT -k $ROOT/.local" -l "$ROOT/.local/pg.log" start ;;
  stop)  "$BIN/pg_ctl" -D "$DATA" stop ;;
  status) "$BIN/pg_ctl" -D "$DATA" status ;;
  psql)  shift; "$BIN/psql" -h 127.0.0.1 -p $PORT -U lad -d salesmaya_agent "$@" ;;
  *) echo "usage: $0 {start|stop|status|psql}" >&2; exit 2 ;;
esac
