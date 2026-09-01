# Local LiveKit server control on Windows.
#
# The PowerShell counterpart of tools/livekit.sh. The shell version builds the
# server from source because the project ships no macOS binaries; it ships
# Windows ones, so bootstrap.ps1 downloads the release and nothing is compiled
# here. No Go toolchain, no cgo, none of the darwin CPU-counter problem.
#
# --dev uses the well-known devkey/secret credentials. Local development only.
$ErrorActionPreference = 'Stop'
$Root    = Split-Path -Parent $PSScriptRoot
$Bin     = Join-Path $Root '.local\livekit-server.exe'
$PidFile = Join-Path $Root '.local\livekit.pid'
$Log     = Join-Path $Root '.local\livekit.log'
$Port    = 7880

function Get-RunningPid {
  # A pidfile alone proves nothing: the process it names may be gone, or the id
  # may have been reused by something unrelated. The name is checked too.
  if (-not (Test-Path $PidFile)) { return $null }
  $id = (Get-Content $PidFile -Raw).Trim()
  if (-not $id) { return $null }
  try { $p = Get-Process -Id ([int]$id) -ErrorAction Stop } catch { return $null }
  if ($p.ProcessName -ne 'livekit-server') { return $null }
  return $p
}

function Test-PortHeld {
  $c = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
  return ($null -ne $c)
}

$command = $args[0]
if (-not $command) { $command = 'status' }

switch ($command) {
  'start' {
    $running = Get-RunningPid
    if ($running) { "already running (pid $($running.Id))"; exit 0 }
    if (-not (Test-Path $Bin)) { Write-Error "no livekit-server.exe under .local\. Run tools\bootstrap.ps1 first." }

    # Refuse to start while the port is still held. LiveKit shuts down
    # gracefully and keeps answering HTTP while it drains, so a start issued
    # straight after a stop fails to bind and dies silently, leaving a health
    # check that passes against the process being killed.
    for ($i = 0; $i -lt 40; $i++) {
      if (-not (Test-PortHeld)) { break }
      Start-Sleep -Milliseconds 500
    }
    if (Test-PortHeld) { Write-Error "port $Port is still in use; something else is listening" }

    # BIND=0.0.0.0 to reach the SFU from a phone on the same network.
    # Localhost by default: --dev uses the well-known devkey/secret pair, so
    # binding wider puts an unauthenticated media server on the LAN.
    $bind = $env:BIND
    if (-not $bind) { $bind = '127.0.0.1' }

    $proc = Start-Process -FilePath $Bin -ArgumentList '--dev', '--bind', $bind `
      -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
      -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii

    # Confirm it survived. Writing a pidfile for a process that died on bind is
    # how the failure above stayed invisible.
    Start-Sleep -Seconds 1
    if ($proc.HasExited) {
      Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
      if (Test-Path "$Log.err") { Get-Content "$Log.err" -Tail 5 | Write-Host }
      Write-Error "livekit-server exited immediately; see $Log"
    }
    "livekit-server started (pid $($proc.Id)) on ${bind}:$Port, log: $Log"
  }
  'stop' {
    $running = Get-RunningPid
    if ($running) {
      Stop-Process -Id $running.Id -Force
      Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
      # Wait for the port, not the process: shutdown releases the socket a
      # moment after the process is gone.
      for ($i = 0; $i -lt 40; $i++) {
        if (-not (Test-PortHeld)) { break }
        Start-Sleep -Milliseconds 500
      }
      "stopped"
    } else {
      Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
      "not running"
    }
  }
  'status' {
    $running = Get-RunningPid
    if ($running) { "running (pid $($running.Id))" } else { "not running" }
  }
  'log' {
    $n = 40
    if ($args.Count -gt 1) { $n = [int]$args[1] }
    if (Test-Path $Log) { Get-Content $Log -Tail $n } else { "no log yet" }
  }
  default { Write-Error "usage: livekit.ps1 {start|stop|status|log [n]}" }
}
