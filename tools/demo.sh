#!/usr/bin/env bash
# Bring the whole stack up and print the join URL.
#
# Starts Postgres, LiveKit, the join service, and a looping translation
# session, then tells you where to listen. Everything runs from .local/ with
# no admin rights and nothing installed system-wide.
#
#   ./tools/demo.sh up          start everything, print the URL
#   ./tools/demo.sh down        stop everything
#   ./tools/demo.sh status      what is running
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LAD_DATABASE_URL="postgresql://lad@127.0.0.1:55432/salesmaya_agent"
export LAD_CONTROL_SCHEMA="lad_dev"
# LAN=1 exposes the demo to other devices on the network.
#
# The advertised LIVEKIT_URL must change with the bind address, not just the
# bind. That URL is handed to the browser inside the listener token, so a
# phone given "ws://127.0.0.1:7880" tries to connect to ITSELF and sits on
# "Connecting..." with nothing in any log to explain it.
if [ "${LAN:-0}" = "1" ]; then
  HOST_IP="${HOST_IP:-$("$ROOT/.venv/bin/python" -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('192.0.2.1', 1))
    print(s.getsockname()[0])
finally:
    s.close()")}"
  export BIND=0.0.0.0
  export SERVE_HOST=0.0.0.0
  export LIVEKIT_URL="ws://${HOST_IP}:7880"
else
  HOST_IP=127.0.0.1
  export BIND=127.0.0.1
  export SERVE_HOST=127.0.0.1
  export LIVEKIT_URL="ws://127.0.0.1:7880"
fi
export LIVEKIT_API_KEY="devkey"
export LIVEKIT_API_SECRET="secret"

PY="$ROOT/.venv/bin/python"
SCRATCH="${TMPDIR:-/tmp}/lad-demo"
mkdir -p "$SCRATCH"

# Five languages exceeds what two cores can serve without shedding audio.
# Override with: TARGETS=fr,ar ./tools/demo.sh up
TARGETS="${TARGETS:-fr,ar,hi}"
AUDIO="${AUDIO:-fixtures/jfk.wav}"

# STT settings. The constraint is emit_interval > window_s * RTF, where RTF is
# measured UNDER LOAD rather than in isolation: the STT shares two cores with
# every TTS voice and translation chain.
#
#   tiny   RTF ~0.06 isolated. Keeps up comfortably, mistranscribes.
#   small  RTF ~0.39 isolated, far worse under contention. Transcribes well
#          and needs a long interval and a short window to avoid shedding.
MODEL="${MODEL:-tiny}"
EMIT="${EMIT:-3.0}"
WINDOW="${WINDOW:-6.0}"

case "${1:-up}" in
up)
  ./tools/pg.sh start >/dev/null 2>&1 || echo "postgres already running"
  ./tools/livekit.sh start >/dev/null 2>&1 || true

  # Wait for readiness rather than sleeping a fixed interval. A fixed sleep is
  # a race: LiveKit took longer than 2s once and the session crashed on
  # connect with "Connection refused", leaving a demo that looked started and
  # published nothing.
  for _ in $(seq 1 40); do
    curl -sf -o /dev/null http://127.0.0.1:7880/ && break
    sleep 0.5
  done
  if ! curl -sf -o /dev/null http://127.0.0.1:7880/; then
    echo "livekit did not come up; see .local/livekit.log" >&2
    exit 1
  fi

  for _ in $(seq 1 40); do
    "$ROOT/.local/pgsql/bin/pg_isready" -h 127.0.0.1 -p 55432 -q && break
    sleep 0.5
  done

  pkill -f serve_join.py 2>/dev/null || true
  nohup "$PY" tools/serve_join.py --host "$SERVE_HOST" --port 8080 > "$SCRATCH/join.log" 2>&1 &
  for _ in $(seq 1 40); do
    curl -sf -o /dev/null http://127.0.0.1:8080/healthz && break
    sleep 0.5
  done

  SESSION=$("$PY" - <<'PYEOF'
import asyncio, os, sys, uuid
sys.path.insert(0, "src")
from lad_translate.config import LanguageTarget, SessionConfig, TenantContext
from lad_translate.db.sessions import SessionStore
from lad_translate.adapters.tts_piper import DEFAULT_VOICES
from lad_translate.obs.log import configure
configure("ERROR")
URL = os.environ["LAD_DATABASE_URL"]
TARGETS = [t.strip() for t in os.environ.get("TARGETS", "fr,ar,hi").split(",") if t.strip()]

async def main():
    import asyncpg
    pool = await asyncpg.create_pool(URL, min_size=1, max_size=2)
    row = await pool.fetchrow("SELECT id::text, schema_name FROM lad_dev.tenants WHERE slug='techiemaya'")
    tenant = TenantContext(tenant_id=row[0], database_url=URL, schema=row[1])
    store = SessionStore(pool, tenant)
    for r in await store.live_sessions():
        try: await store.end_session(r["session_id"])
        except LookupError: pass
    cfg = SessionConfig(
        session_id=str(uuid.uuid4()), tenant=tenant, room_name="demo-room",
        event_name="Sharjah Innovation Summit", source_language="en",
        targets=[LanguageTarget(c, DEFAULT_VOICES[c]) for c in TARGETS])
    await store.create_session(cfg, latency_credible=False)
    await store.mark_live(cfg.session_id)
    print(cfg.session_id)
    await pool.close()
asyncio.run(main())
PYEOF
)
  export TARGETS
  pkill -f session_live.py 2>/dev/null || true
  nohup "$PY" tools/session_live.py --audio "$AUDIO" --targets "$TARGETS" \
    --model "$MODEL" --emit-interval "$EMIT" --window "$WINDOW" \
    --room demo-room --loop > "$SCRATCH/session.log" 2>&1 &

  echo
  echo "  stt       : $MODEL (emit ${EMIT}s, window ${WINDOW}s)"
  echo "  languages : $TARGETS"
  echo "  source    : $AUDIO (looping)"
  echo "  logs      : $SCRATCH"
  echo
  echo "  Wait ~30s for the tracks to appear, then open:"
  echo "      http://${HOST_IP}:8080/s/$SESSION"
  if [ "${LAN:-0}" = "1" ]; then
    echo
    echo "  Reachable from any device on this network. Scan the QR with:"
    echo "      .venv/bin/python tools/make_qr.py --session $SESSION --base http://${HOST_IP}:8080"
  fi
  echo
  echo "  Tap a language. The tap is required: browsers refuse to start"
  echo "  audio without one."
  echo
  ;;
down)
  pkill -f session_live.py 2>/dev/null || true
  pkill -f serve_join.py 2>/dev/null || true
  ./tools/livekit.sh stop 2>/dev/null || true
  ./tools/pg.sh stop 2>/dev/null || true
  echo "stopped"
  ;;
status)
  ./tools/livekit.sh status | sed 's/^/  livekit:  /'
  ./tools/pg.sh status >/dev/null 2>&1 && echo "  postgres: running" || echo "  postgres: not running"
  curl -s -o /dev/null -w "  join:     %{http_code}\n" http://127.0.0.1:8080/healthz || true
  echo "  session:  $(pgrep -fc session_live.py 2>/dev/null || echo 0) running"
  ;;
*) echo "usage: $0 {up|down|status}" >&2; exit 2 ;;
esac
