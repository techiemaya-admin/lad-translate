# Bring the whole stack up on Windows and print the join URL.
#
# The PowerShell counterpart of tools/demo.sh. Starts Postgres, LiveKit, the
# join service, and a looping translation session, then tells you where to
# listen. Everything runs from .local\ with no admin rights and nothing
# installed system-wide.
#
#   .\tools\demo.ps1 up        start everything, print the URL
#   .\tools\demo.ps1 down      stop everything
#   .\tools\demo.ps1 status    what is running
#
# The shell version finds its background processes again with `pkill -f
# session_live.py`, matching on the command line. There is no such thing here,
# so each child's id is written to a pidfile under .local\ and checked against
# the process's actual command line before anything is killed. A pidfile on its
# own is not proof: Windows reuses process ids, and killing whatever inherited
# one is worse than leaving a stale file behind.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:LAD_DATABASE_URL = 'postgresql://lad@127.0.0.1:55432/salesmaya_agent'
$env:LAD_CONTROL_SCHEMA = 'lad_dev'

# LAN=1 exposes the demo to other devices on the network.
#
# The advertised LIVEKIT_URL must change with the bind address, not just the
# bind. That URL is handed to the browser inside the listener token, so a phone
# given "ws://127.0.0.1:7880" tries to connect to ITSELF and sits on
# "Connecting..." with nothing in any log to explain it.
$Py = Join-Path $Root '.venv\Scripts\python.exe'
if ($env:LAN -eq '1') {
  $HostIp = $env:HOST_IP
  if (-not $HostIp) {
    $HostIp = & $Py -c @"
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('192.0.2.1', 1))
    print(s.getsockname()[0])
finally:
    s.close()
"@
  }
  $env:BIND = '0.0.0.0'
  $ServeHost = '0.0.0.0'
  $env:LIVEKIT_URL = "ws://${HostIp}:7880"
} else {
  $HostIp = '127.0.0.1'
  $env:BIND = '127.0.0.1'
  $ServeHost = '127.0.0.1'
  $env:LIVEKIT_URL = 'ws://127.0.0.1:7880'
}
$env:LIVEKIT_API_KEY = 'devkey'
$env:LIVEKIT_API_SECRET = 'secret'

# Logs and the generated session script go to the temp directory, not into the
# checkout. demo.sh uses ${TMPDIR:-/tmp}/lad-demo for the same reason: a
# generated Python file sitting under .local\ is one ruff walks and reports on,
# and lint findings against a scratch file are noise in a gate people have to
# keep trusting.
$Scratch = Join-Path $env:TEMP 'lad-demo'
New-Item -ItemType Directory -Force -Path $Scratch | Out-Null

# Override with: $env:TARGETS='fr,ar'; .\tools\demo.ps1 up
$Targets = if ($env:TARGETS) { $env:TARGETS } else { 'fr,ar,hi' }
$Audio   = if ($env:AUDIO)   { $env:AUDIO }   else { 'fixtures/jfk.wav' }

# STT settings. The constraint is emit_interval > window_s * RTF, where RTF is
# measured UNDER LOAD rather than in isolation: the STT shares the machine with
# every TTS voice and translation chain.
#
#   tiny   RTF ~0.06 isolated. Keeps up comfortably, mistranscribes.
#   small  RTF ~0.39 isolated, far worse under contention. Transcribes well
#          and needs a long interval and a short window to avoid shedding.
$Model  = if ($env:MODEL)  { $env:MODEL }  else { 'tiny' }
$Emit   = if ($env:EMIT)   { $env:EMIT }   else { '3.0' }
$Window = if ($env:WINDOW) { $env:WINDOW } else { '6.0' }

# Energy gate. Below this a buffer is never handed to the model, which is the
# only defence that costs nothing and the only one an invented phrase cannot get
# past. Measured on a phone in a room: background noise reached 0.037 and speech
# started at 0.067, and the old 0.006 passed all of the former.
#
# fixtures/holmes.wav is quiet narration at 0.029-0.050 and falls below this, so
# scoring against it needs SPEECH_RMS=0.006.
$SpeechRms = if ($env:SPEECH_RMS) { $env:SPEECH_RMS } else { '0.05' }

function Get-TrackedProcess {
  # A pidfile names a process that may be gone, or an id Windows has since
  # handed to something else. The command line is checked for the marker before
  # the id is trusted.
  param($PidFile, $Marker)
  if (-not (Test-Path $PidFile)) { return $null }
  $id = (Get-Content $PidFile -Raw).Trim()
  if (-not $id) { return $null }
  $p = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
  if (-not $p) { return $null }
  if ($p.CommandLine -notlike "*$Marker*") { return $null }
  return $p
}

function Stop-Tracked {
  param($PidFile, $Marker)
  $p = Get-TrackedProcess -PidFile $PidFile -Marker $Marker
  if ($p) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Start-Tracked {
  param($PidFile, $ArgList, $LogName)
  $proc = Start-Process -FilePath $Py -ArgumentList $ArgList `
    -RedirectStandardOutput (Join-Path $Scratch "$LogName.log") `
    -RedirectStandardError  (Join-Path $Scratch "$LogName.err") `
    -WindowStyle Hidden -PassThru
  Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
  return $proc
}

function Test-Http {
  param($Url)
  try {
    Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 | Out-Null
    return $true
  } catch { return $false }
}

$JoinPid    = Join-Path $Root '.local\join.pid'
$SessionPid = Join-Path $Root '.local\session.pid'

$command = $args[0]
if (-not $command) { $command = 'up' }

switch ($command) {
'up' {
  & (Join-Path $Root 'tools\pg.ps1') start 2>$null | Out-Null
  & (Join-Path $Root 'tools\livekit.ps1') start 2>$null | Out-Null

  # Wait for readiness rather than sleeping a fixed interval. A fixed sleep is
  # a race: LiveKit took longer than 2s once and the session crashed on connect
  # with "Connection refused", leaving a demo that looked started and published
  # nothing.
  for ($i = 0; $i -lt 40; $i++) {
    if (Test-Http 'http://127.0.0.1:7880/') { break }
    Start-Sleep -Milliseconds 500
  }
  if (-not (Test-Http 'http://127.0.0.1:7880/')) {
    Write-Error "livekit did not come up; see .local\livekit.log"
  }

  & $Py (Join-Path $Root 'tools\pg_admin.py') ready --wait 20
  if ($LASTEXITCODE -ne 0) { Write-Error "postgres did not come up; see .local\pg.log" }

  Stop-Tracked -PidFile $JoinPid -Marker 'serve_join.py'
  Start-Tracked -PidFile $JoinPid -LogName 'join' -ArgList @(
    (Join-Path $Root 'tools\serve_join.py'), '--host', $ServeHost, '--port', '8080') | Out-Null
  for ($i = 0; $i -lt 40; $i++) {
    if (Test-Http 'http://127.0.0.1:8080/healthz') { break }
    Start-Sleep -Milliseconds 500
  }

  $env:TARGETS = $Targets
  $mkSession = Join-Path $Scratch 'make_session.py'
  Set-Content -Path $mkSession -Encoding utf8 -Value @'
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
'@
  $Session = (& $Py $mkSession).Trim()
  if (-not $Session) { Write-Error "could not register a session; is Postgres seeded? Run tools\bootstrap.ps1" }

  Stop-Tracked -PidFile $SessionPid -Marker 'session_live.py'
  Start-Tracked -PidFile $SessionPid -LogName 'session' -ArgList @(
    (Join-Path $Root 'tools\session_live.py'),
    '--audio', $Audio, '--targets', $Targets,
    '--model', $Model, '--emit-interval', $Emit, '--window', $Window,
    '--speech-rms', $SpeechRms,
    '--room', 'demo-room', '--loop') | Out-Null

  Write-Host ""
  Write-Host "  stt       : $Model (emit ${Emit}s, window ${Window}s, gate ${SpeechRms})"
  Write-Host "  languages : $Targets"
  Write-Host "  source    : $Audio (looping)"
  Write-Host "  logs      : $Scratch"
  Write-Host ""
  Write-Host "  Wait ~30s for the tracks to appear, then open:"
  Write-Host "      http://${HostIp}:8080/s/$Session"
  if ($env:LAN -eq '1') {
    Write-Host ""
    Write-Host "  Reachable from any device on this network. Scan the QR with:"
    Write-Host "      .venv\Scripts\python.exe tools\make_qr.py --session $Session --base http://${HostIp}:8080"
  }
  Write-Host ""
  Write-Host "  Tap a language. The tap is required: browsers refuse to start"
  Write-Host "  audio without one."
  Write-Host ""
}
'down' {
  Stop-Tracked -PidFile $SessionPid -Marker 'session_live.py'
  Stop-Tracked -PidFile $JoinPid -Marker 'serve_join.py'
  & (Join-Path $Root 'tools\livekit.ps1') stop 2>$null | Out-Null
  & (Join-Path $Root 'tools\pg.ps1') stop 2>$null | Out-Null
  "stopped"
}
'status' {
  "  livekit:  " + (& (Join-Path $Root 'tools\livekit.ps1') status)
  "  postgres: " + (& (Join-Path $Root 'tools\pg.ps1') status)
  if (Test-Http 'http://127.0.0.1:8080/healthz') { "  join:     200" } else { "  join:     down" }
  $s = Get-TrackedProcess -PidFile $SessionPid -Marker 'session_live.py'
  if ($s) { "  session:  running (pid $($s.ProcessId))" } else { "  session:  not running" }
}
default { Write-Error "usage: demo.ps1 {up|down|status}" }
}
