# TLS in front of the join service and LiveKit signalling, on Windows.
#
# The PowerShell counterpart of tools/tls.sh, and it exists for one reason:
# browsers only expose a microphone in a secure context. navigator.mediaDevices
# is undefined on plain http from anything but localhost, so a phone on
# http://192.168.x.x cannot be the speaker at all.
#
# Only the SIGNALLING WebSocket is proxied. WebRTC media is UDP with DTLS-SRTP
# and is already encrypted, so it goes direct and untouched -- which is why the
# firewall notes below matter: proxying the signalling is not enough on its own.
#
#   .\tools\tls.ps1 up      generate a config for this machine's LAN IP and run
#   .\tools\tls.ps1 down    stop
#   .\tools\tls.ps1 url     print the addresses to use
#   .\tools\tls.ps1 firewall  print the elevated command that opens the ports
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root    = Split-Path -Parent $PSScriptRoot
$Caddy   = Join-Path $Root '.local\bin\caddy.exe'
$Conf    = Join-Path $Root '.local\caddy\Caddyfile'
$PidFile = Join-Path $Root '.local\caddy.pid'
$Log     = Join-Path $Root '.local\caddy.log'
$Version = '2.11.4'
$Port    = 8443
$Py      = Join-Path $Root '.venv\Scripts\python.exe'

function Get-LanIp {
  & $Py -c @"
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('192.0.2.1', 1))   # TEST-NET-1, never routed; sends nothing
    print(s.getsockname()[0])
finally:
    s.close()
"@
}

function Install-Caddy {
  if (Test-Path $Caddy) { return }
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  $dl = Join-Path $Root '.local\dl'
  New-Item -ItemType Directory -Force -Path $dl, (Split-Path -Parent $Caddy) | Out-Null
  $zip = Join-Path $dl "caddy_$Version.zip"
  Write-Host "fetching caddy $Version"
  Invoke-WebRequest "https://github.com/caddyserver/caddy/releases/download/v$Version/caddy_${Version}_windows_amd64.zip" `
    -OutFile $zip -UseBasicParsing -TimeoutSec 600
  $stage = Join-Path $dl 'caddy'
  Expand-Archive $zip -DestinationPath $stage -Force
  Move-Item (Join-Path $stage 'caddy.exe') $Caddy -Force
}

function Get-RunningPid {
  if (-not (Test-Path $PidFile)) { return $null }
  $id = (Get-Content $PidFile -Raw).Trim()
  if (-not $id) { return $null }
  try { $p = Get-Process -Id ([int]$id) -ErrorAction Stop } catch { return $null }
  if ($p.ProcessName -ne 'caddy') { return $null }
  return $p
}

function Write-CaddyConfig {
  param($Ip)
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Conf) | Out-Null
  # Caddyfile paths use forward slashes. A Windows path goes in verbatim
  # otherwise, and Caddy reads the backslashes as escapes.
  $rootFwd = $Root -replace '\\', '/'
  $template = @'
{
	# Caddy's own CA. The phone warns once; after you proceed the origin counts
	# as secure, which is what getUserMedia requires.
	local_certs
	admin off
	storage file_system {
		root __ROOT__/.local/caddy/data
	}
}

https://__IP__:__PORT__ {
	# Access log. Without it there is no way to tell "the phone never reached
	# us" from "the phone reached us and something failed", which is exactly
	# the question that matters when someone says the page will not load.
	log {
		output file __ROOT__/.local/caddy-access.log
		format json
	}

	# The root CA, over plain http so a phone that does not yet trust us can
	# still fetch it. Installing it once removes the warning entirely, and on
	# iOS the warning is not merely cosmetic: Safari may render the page and
	# still refuse the WebSocket, which fails silently.
	handle /ca.crt {
		root * __ROOT__/.local/caddy/data/pki/authorities/local
		rewrite * /root.crt
		file_server
	}

	@livekit path /rtc /rtc/* /twirp/*
	reverse_proxy @livekit 127.0.0.1:7880
	reverse_proxy 127.0.0.1:8080
}

# Plain http, for one job only: handing out the CA to a device that cannot yet
# trust the https listener. Chicken and egg otherwise.
http://__IP__:8081 {
	handle /ca.crt {
		root * __ROOT__/.local/caddy/data/pki/authorities/local
		rewrite * /root.crt
		file_server
	}
	# Must be a handle block, not a bare redir. Caddy orders directives by its
	# own table and runs redir BEFORE handle, so a bare redirect swallows the
	# CA download and returns 302 for the one request that cannot follow it.
	handle {
		redir https://__IP__:__PORT__{uri}
	}
}
'@
  $text = $template.Replace('__ROOT__', $rootFwd).Replace('__IP__', $Ip).Replace('__PORT__', "$Port")
  Set-Content -Path $Conf -Value $text -Encoding utf8
}

function Show-FirewallCommand {
  # Windows blocks inbound connections by default, and unlike the Unix boxes
  # this project was written on there is no way around that from an
  # unprivileged shell. Proxying the signalling is not enough on its own:
  # WebRTC media does not go through Caddy, so the phone has to reach LiveKit's
  # own media ports directly or it connects and stays silent.
  Write-Host ""
  Write-Host "  Windows blocks inbound connections. Run ONCE, in an elevated PowerShell:"
  Write-Host ""
  Write-Host "      New-NetFirewallRule -DisplayName 'LAD translate (TCP)' -Direction Inbound ``"
  Write-Host "        -Action Allow -Protocol TCP -LocalPort 8443,8081,7881"
  Write-Host "      New-NetFirewallRule -DisplayName 'LAD translate (UDP)' -Direction Inbound ``"
  Write-Host "        -Action Allow -Protocol UDP -LocalPort 7882"
  Write-Host ""
  Write-Host "  8443/8081 are Caddy, 7881/7882 are LiveKit's own media ports, which"
  Write-Host "  are NOT proxied: WebRTC media is UDP and goes phone-to-host direct."
  Write-Host ""
  $profile = Get-NetConnectionProfile -ErrorAction SilentlyContinue | Where-Object { $_.IPv4Connectivity -ne 'Disconnected' } | Select-Object -First 1
  if ($profile -and $profile.NetworkCategory -eq 'Public') {
    Write-Host "  This network ($($profile.InterfaceAlias)) is classified Public, which is the"
    Write-Host "  stricter profile. On a network you trust, also run:"
    Write-Host ""
    Write-Host "      Set-NetConnectionProfile -InterfaceAlias '$($profile.InterfaceAlias)' -NetworkCategory Private"
    Write-Host ""
  }
}

$command = $args[0]
if (-not $command) { $command = 'up' }

switch ($command) {
  'up' {
    Install-Caddy
    $ip = if ($env:HOST_IP) { $env:HOST_IP } else { (Get-LanIp).Trim() }
    Write-CaddyConfig -Ip $ip
    $running = Get-RunningPid
    if ($running) {
      "already running (pid $($running.Id))"
    } else {
      $proc = Start-Process -FilePath $Caddy -ArgumentList 'run', '--config', $Conf `
        -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
        -WindowStyle Hidden -PassThru
      Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
      Start-Sleep -Seconds 2
      if ($proc.HasExited) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        if (Test-Path "$Log.err") { Get-Content "$Log.err" -Tail 10 | Write-Host }
        Write-Error "caddy exited; see $Log"
      }
      "caddy started (pid $($proc.Id)) on ${ip}:$Port"
    }
    Write-Host ""
    Write-Host "  Run the services with:"
    Write-Host "    `$env:LIVEKIT_URL='wss://${ip}:${Port}'          # what browsers dial"
    Write-Host "    `$env:LIVEKIT_INTERNAL_URL='ws://127.0.0.1:7880' # what this host dials"
    Write-Host ""
    Write-Host "  Those must differ. The Python SDK does not trust Caddy's CA and fails"
    Write-Host "  with 'invalid peer certificate: UnknownIssuer' if pointed at the proxy."
    Write-Host ""
    Write-Host "  If a phone will not load the page, install the CA once:"
    Write-Host "    http://${ip}:8081/ca.crt"
    Write-Host "  iOS: open that, allow the profile, then Settings > General > About >"
    Write-Host "       Certificate Trust Settings and switch it on. Android: Settings >"
    Write-Host "       Security > Install a certificate > CA certificate."
    Show-FirewallCommand
  }
  'down' {
    $running = Get-RunningPid
    if ($running) {
      Stop-Process -Id $running.Id -Force
      Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
      "stopped"
    } else {
      Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
      "not running"
    }
  }
  'url' { "https://$((Get-LanIp).Trim()):$Port" }
  'firewall' { Show-FirewallCommand }
  default { Write-Error "usage: tls.ps1 {up|down|url|firewall}" }
}
