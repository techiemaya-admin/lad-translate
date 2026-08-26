#!/usr/bin/env bash
# TLS in front of the join service and LiveKit signalling.
#
# Needed for one reason: browsers only expose a microphone in a secure context.
# navigator.mediaDevices is undefined on plain http from anything but
# localhost, so a phone on http://192.168.x.x cannot be the speaker at all.
#
# Only the SIGNALLING WebSocket is proxied. WebRTC media is UDP with DTLS-SRTP
# and is already encrypted, so it goes direct and untouched.
#
#   ./tools/tls.sh up      generate a config for this machine's LAN IP and run
#   ./tools/tls.sh down    stop
#   ./tools/tls.sh url     print the addresses to use
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CADDY="$ROOT/.local/bin/caddy"
CONF="$ROOT/.local/caddy/Caddyfile"
PIDFILE="$ROOT/.local/caddy.pid"
LOG="$ROOT/.local/caddy.log"
VERSION="2.11.4"
PORT=8443

lan_ip() {
  "$ROOT/.venv/bin/python" - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("192.0.2.1", 1))   # TEST-NET-1, never routed; sends nothing
    print(s.getsockname()[0])
finally:
    s.close()
PY
}

install_caddy() {
  [ -x "$CADDY" ] && return
  case "$(uname -s)-$(uname -m)" in
    Darwin-x86_64) ASSET="caddy_${VERSION}_mac_amd64.tar.gz" ;;
    Darwin-arm64)  ASSET="caddy_${VERSION}_mac_arm64.tar.gz" ;;
    Linux-x86_64)  ASSET="caddy_${VERSION}_linux_amd64.tar.gz" ;;
    *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
  esac
  mkdir -p "$ROOT/.local/bin"
  echo "fetching caddy $VERSION"
  curl -sSfL "https://github.com/caddyserver/caddy/releases/download/v${VERSION}/${ASSET}" \
    | tar xz -C "$ROOT/.local/bin" caddy
}

write_config() {
  local ip="$1"
  mkdir -p "$(dirname "$CONF")"
  cat > "$CONF" <<CADDY
{
	# Caddy's own CA. The phone warns once; after you proceed the origin counts
	# as secure, which is what getUserMedia requires.
	local_certs
	admin off
	storage file_system {
		root ${ROOT}/.local/caddy/data
	}
}

https://${ip}:${PORT} {
	@livekit path /rtc /rtc/* /twirp/*
	reverse_proxy @livekit 127.0.0.1:7880
	reverse_proxy 127.0.0.1:8080
}
CADDY
}

case "${1:-up}" in
up)
  install_caddy
  IP="${HOST_IP:-$(lan_ip)}"
  write_config "$IP"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PIDFILE"))"
  else
    "$CADDY" run --config "$CONF" >"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "caddy exited; see $LOG" >&2; exit 1; }
  fi
  echo
  echo "  Run the services with:"
  echo "    export LIVEKIT_URL=wss://${IP}:${PORT}          # what browsers dial"
  echo "    export LIVEKIT_INTERNAL_URL=ws://127.0.0.1:7880 # what this host dials"
  echo
  echo "  Those must differ. The Python SDK does not trust Caddy's CA and fails"
  echo "  with 'invalid peer certificate: UnknownIssuer' if pointed at the proxy."
  ;;
down)
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE" && echo stopped || echo "not running"
  ;;
url)
  echo "https://$(lan_ip):${PORT}"
  ;;
*) echo "usage: $0 {up|down|url}" >&2; exit 2 ;;
esac
