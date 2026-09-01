# Local Postgres control on Windows. Binaries live under .local\, no admin needed.
# Run tools\bootstrap.ps1 once to create the cluster this drives.
#
# The PowerShell counterpart of tools/pg.sh. Three differences from the shell
# version, all forced by the platform rather than chosen.
#
# `psql` became `sql`. The Windows Postgres build under .local\pgsql ships the
# server -- initdb, pg_ctl, postgres -- and none of the client programs, so
# there is no psql.exe to hand arguments to. tools\pg_admin.py runs a statement
# over asyncpg instead, which is a dependency this project already has.
#
# There is no `-k` socket directory. Unix domain sockets do not exist here, so
# the cluster is reached over loopback TCP and nothing else.
#
# `start` is idempotent. pg_ctl writes "another server might be running" to
# stderr and starts anyway, and with $ErrorActionPreference = 'Stop' PowerShell
# turns any native stderr into a terminating error -- so the shell version's
# `pg.sh start || echo already running` becomes a caller that dies. The state is
# checked here instead, and a second start is a no-op that says so.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Bin  = Join-Path $Root '.local\pgsql\bin'
$Data = Join-Path $Root '.local\pgdata'
$Log  = Join-Path $Root '.local\pg.log'
$Port = 55432
$Py   = Join-Path $Root '.venv\Scripts\python.exe'
$Url  = "postgresql://lad@127.0.0.1:$Port/salesmaya_agent"

if (-not (Test-Path (Join-Path $Bin 'pg_ctl.exe'))) {
  Write-Error "no Postgres under .local\pgsql. Run tools\bootstrap.ps1 first."
}

function Invoke-Native {
  # Run a native program without letting its stderr become a terminating error,
  # and hand back the exit code the program actually returned.
  param([string]$Exe, [string[]]$Arguments, [switch]$Quiet)
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    if ($Quiet) { & $Exe @Arguments *> $null } else { & $Exe @Arguments }
    return $LASTEXITCODE
  } finally { $ErrorActionPreference = $previous }
}

function Test-Running {
  # pg_ctl status exits 0 when the server is up, 3 when it is not, 4 when there
  # is no data directory at all.
  return (Invoke-Native -Exe "$Bin\pg_ctl.exe" -Arguments @('-D', $Data, 'status') -Quiet) -eq 0
}

$command = $args[0]
if (-not $command) { $command = 'status' }
$rest = @()
if ($args.Count -gt 1) { $rest = $args[1..($args.Count - 1)] }

switch ($command) {
  'start' {
    if (-not (Test-Path (Join-Path $Data 'PG_VERSION'))) {
      Write-Error "no cluster at .local\pgdata. Run tools\bootstrap.ps1 first."
    }
    if (Test-Running) { "already running"; exit 0 }
    $code = Invoke-Native -Exe "$Bin\pg_ctl.exe" -Arguments @('-D', $Data, '-o', "-p $Port -h 127.0.0.1", '-l', $Log, 'start')
    if ($code -ne 0) { Write-Error "postgres did not start; see $Log" }
  }
  'stop' {
    if (-not (Test-Running)) { "not running"; exit 0 }
    Invoke-Native -Exe "$Bin\pg_ctl.exe" -Arguments @('-D', $Data, 'stop') | Out-Null
    "stopped"
  }
  'status' {
    if (Test-Running) { "running" } else { "not running"; exit 3 }
  }
  'ready' {
    $code = Invoke-Native -Exe $Py -Arguments @((Join-Path $Root 'tools\pg_admin.py'), '--url', $Url, 'ready', '--wait', '20')
    exit $code
  }
  'sql' {
    if ($rest.Count -lt 1) { Write-Error "usage: pg.ps1 sql ""SELECT 1""" }
    Invoke-Native -Exe $Py -Arguments @((Join-Path $Root 'tools\pg_admin.py'), '--url', $Url, 'sql', $rest[0])
  }
  'log' {
    $n = 40
    if ($rest.Count -ge 1) { $n = [int]$rest[0] }
    if (Test-Path $Log) { Get-Content $Log -Tail $n } else { "no log yet" }
  }
  default {
    Write-Error "usage: pg.ps1 {start|stop|status|ready|sql <statement>|log [n]}"
  }
}
