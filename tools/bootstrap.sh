#!/usr/bin/env bash
# Provision everything the demo needs, from a clean clone.
#
# tools/pg.sh and tools/livekit.sh drive a Postgres cluster and a LiveKit
# server under .local/, and until now nothing created them: the README described
# extracting EnterpriseDB binaries and building LiveKit by hand, so a fresh
# clone could run the tests and nothing else. This is that description, executed.
#
#   ./tools/bootstrap.sh              venv, Postgres, LiveKit, models
#   ./tools/bootstrap.sh --no-models  skip the model downloads
#   ./tools/bootstrap.sh --recreate-db  throw the cluster away and rebuild it
#
# Idempotent: every step checks for its own output first, so it is safe to run
# again after fixing whatever it complained about. Nothing is installed system
# wide and nothing needs admin rights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIVEKIT_VERSION="${LIVEKIT_VERSION:-v1.13.6}"
PG_PORT=55432
PG_DATA="$ROOT/.local/pgdata"
PG_HOME="$ROOT/.local/pgsql"
DB_NAME=salesmaya_agent
DB_USER=lad

WANT_MODELS=1
RECREATE_DB=0
for arg in "$@"; do
  case "$arg" in
    --no-models) WANT_MODELS=0 ;;
    --recreate-db) RECREATE_DB=1 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
note() { printf '    %s\n' "$1"; }
die() { printf '\n\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

# Postgres refuses to run as root. On a Linux box or in a container where this
# runs as root, every postgres command drops to the postgres account instead,
# which is why the data directory is created owned by it.
PG_RUNAS=""
if [ "$(id -u)" = 0 ]; then
  id postgres >/dev/null 2>&1 || die "running as root and there is no postgres user to drop to"
  PG_RUNAS=postgres
fi
pg() {
  local bin="$1"; shift
  if [ -n "$PG_RUNAS" ]; then
    local quoted="" a
    for a in "$@"; do quoted+=" $(printf '%q' "$a")"; done
    su "$PG_RUNAS" -c "$(printf '%q' "$PG_HOME/bin/$bin")$quoted"
  else
    "$PG_HOME/bin/$bin" "$@"
  fi
}

# ---------------------------------------------------------------- python -----
say "Python environment"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.11
    uv venv --python 3.11
  else
    command -v python3.11 >/dev/null 2>&1 || die "need python3.11 or uv (https://astral.sh/uv)"
    python3.11 -m venv .venv
  fi
fi
# The CPU backend extras are part of the demo, not an optional afterthought:
# without them serve_session.py cannot import its own adapters.
EXTRAS='.[dev,api,livekit,db,stt-cpu,mt-cpu,tts-cpu]'
if command -v uv >/dev/null 2>&1; then
  uv pip install -e "$EXTRAS"
else
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install -e "$EXTRAS"
fi
note "$(.venv/bin/python --version)"

# -------------------------------------------------------------- postgres -----
say "Postgres"
if [ ! -d "$PG_HOME/bin" ]; then
  # Find a server installation. pg_config alone is not enough: the libpq-dev
  # style packages ship it without initdb, so the bindir is checked for the
  # binaries actually needed rather than trusted because pg_config answered.
  found=""
  for candidate in \
      "$(command -v pg_config >/dev/null 2>&1 && pg_config --bindir || true)" \
      /usr/lib/postgresql/1[6-9]/bin /usr/lib/postgresql/2*/bin \
      /opt/homebrew/opt/postgresql@1[6-9]/bin /usr/local/opt/postgresql@1[6-9]/bin \
      /Library/PostgreSQL/1[6-9]/bin /Applications/Postgres.app/Contents/Versions/*/bin; do
    [ -n "$candidate" ] && [ -x "$candidate/initdb" ] && [ -x "$candidate/pg_ctl" ] && found="$candidate" && break
  done
  [ -n "$found" ] || die "no Postgres server binaries found. Install Postgres 16+ (macOS: brew install postgresql@16, Debian/Ubuntu: apt install postgresql-16), or unpack the EnterpriseDB tarball into .local/pgsql, then run this again."
  mkdir -p "$ROOT/.local"
  ln -sfn "$(dirname "$found")" "$PG_HOME"
  note "using $(dirname "$found")"
fi
note "$("$PG_HOME/bin/postgres" --version)"

if [ "$RECREATE_DB" = 1 ] && [ -d "$PG_DATA" ]; then
  pg pg_ctl -D "$PG_DATA" stop >/dev/null 2>&1 || true
  rm -rf "$PG_DATA"
  note "old cluster removed"
fi

if [ ! -d "$PG_DATA" ]; then
  mkdir -p "$ROOT/.local"
  [ -n "$PG_RUNAS" ] && chown "$PG_RUNAS" "$ROOT/.local"
  # Bootstrap superuser is lad, so LAD_DATABASE_URL connects as itself with no
  # role-mapping step. trust auth: the cluster listens on loopback only.
  pg initdb -D "$PG_DATA" -U "$DB_USER" --auth=trust --encoding=UTF8 >/dev/null
  note "cluster created at .local/pgdata"
fi

if ! "$PG_HOME/bin/pg_isready" -h 127.0.0.1 -p "$PG_PORT" -q 2>/dev/null; then
  "$ROOT/tools/pg.sh" start >/dev/null
fi
for _ in $(seq 1 40); do
  "$PG_HOME/bin/pg_isready" -h 127.0.0.1 -p "$PG_PORT" -q 2>/dev/null && break
  sleep 0.5
done
"$PG_HOME/bin/pg_isready" -h 127.0.0.1 -p "$PG_PORT" -q || die "Postgres did not come up; see .local/pg.log"

if ! "$PG_HOME/bin/psql" -h 127.0.0.1 -p "$PG_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; then
  "$PG_HOME/bin/createdb" -h 127.0.0.1 -p "$PG_PORT" -U "$DB_USER" -O "$DB_USER" "$DB_NAME"
  note "database $DB_NAME created"
fi

export LAD_DATABASE_URL="postgresql://$DB_USER@127.0.0.1:$PG_PORT/$DB_NAME"
export LAD_CONTROL_SCHEMA="lad_dev"
.venv/bin/python tools/seed_tenant.py --slug techiemaya | sed 's/^/    /'

# --------------------------------------------------------------- livekit -----
say "LiveKit server"
if [ ! -x "$ROOT/.local/livekit-server" ]; then
  command -v go >/dev/null 2>&1 || die "need Go to build livekit-server (https://go.dev/dl/), or drop a release binary at .local/livekit-server"
  command -v git >/dev/null 2>&1 || die "need git to fetch the livekit source"
  # Cloned and built, not `go install`. From v1.10 the module's go.mod carries
  # replace directives, and `go install module@version` refuses those with
  # "It must not contain directives that would cause it to be interpreted
  # differently than if it were the main module". Building inside a checkout,
  # where it IS the main module, is the supported route and the one the README
  # describes.
  #
  # CGO_ENABLED=1 because go-osstat reads darwin CPU counters through cgo; a
  # CGO_ENABLED=0 build fails with "undefined: cpu.Get", which looks like
  # broken darwin support and is not.
  note "building $LIVEKIT_VERSION from source (a few minutes, once)"
  SRC="$ROOT/.local/src/livekit"
  if [ ! -d "$SRC/.git" ]; then
    rm -rf "$SRC"
    mkdir -p "$ROOT/.local/src"
    git clone --depth 1 --branch "$LIVEKIT_VERSION" https://github.com/livekit/livekit.git "$SRC"
  fi
  ( cd "$SRC" && CGO_ENABLED=1 GOFLAGS=-buildvcs=false go build -o "$ROOT/.local/livekit-server" ./cmd/server )
fi
note "$("$ROOT/.local/livekit-server" --version)"

# ---------------------------------------------------------------- models -----
if [ "$WANT_MODELS" = 1 ]; then
  say "Models"
  note "translation and voice models, from the Hub (a few hundred MB)"
  .venv/bin/python tools/fetch_mt_models.py --pair en-fr --pair en-ar --pair en-hi \
    || note "MT model fetch failed; rerun tools/fetch_mt_models.py once the Hub is reachable"
  .venv/bin/python tools/fetch_tts_voices.py --defaults \
    || note "voice fetch failed; rerun tools/fetch_tts_voices.py once the Hub is reachable"
else
  say "Models skipped (--no-models)"
  note "translated audio needs: tools/fetch_mt_models.py and tools/fetch_tts_voices.py"
fi

say "Ready"
cat <<EOF

    ./tools/demo.sh up        start everything and print the join URL
    ./tools/demo.sh status    what is running
    ./tools/demo.sh down      stop everything

EOF
