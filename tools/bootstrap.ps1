# Provision everything the demo needs on Windows, from a clean checkout.
#
# The PowerShell counterpart of tools/bootstrap.sh. tools\pg.ps1 and
# tools\livekit.ps1 drive a Postgres cluster and a LiveKit server under .local\,
# and nothing creates them; this is the thing that does.
#
#   .\tools\bootstrap.ps1               venv, Postgres, LiveKit, models
#   .\tools\bootstrap.ps1 -NoModels     skip the model downloads
#   .\tools\bootstrap.ps1 -RecreateDb   throw the cluster away and rebuild it
#
# Idempotent: every step checks for its own output first, so it is safe to run
# again after fixing whatever it complained about. Nothing is installed system
# wide and nothing needs admin rights.
#
# Three things this does differently from the shell version, and each is forced
# by the platform rather than chosen.
#
# LiveKit is downloaded, not built. The project publishes Windows release
# binaries, so there is no Go toolchain, no cgo and none of the go-osstat darwin
# problem the shell script has to work around. The pin still tracks the vendored
# client for the reason bootstrap.sh gives: livekit-client 2.22.0 signals on
# /rtc/v1 and falls back to /rtc on a 404, so an older server costs every join a
# failed WebSocket handshake first.
#
# Postgres is downloaded too, rather than found. The shell script hunts for an
# existing installation because Homebrew and apt put one there; on Windows the
# installer wants admin rights and the EnterpriseDB binary zip was not reachable
# from here at all. The binaries come from the embedded-postgres build on Maven
# Central instead, which is a plain server tree that unpacks and runs in place.
#
# That build ships the server and none of the client programs -- no psql, no
# createdb, no pg_isready -- so tools\pg_admin.py does those three jobs over
# asyncpg, which is already a dependency.
[CmdletBinding()]
param(
  [switch]$NoModels,
  [switch]$RecreateDb
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LiveKitVersion = if ($env:LIVEKIT_VERSION) { $env:LIVEKIT_VERSION } else { 'v1.13.6' }
$PgVersion = if ($env:PG_VERSION) { $env:PG_VERSION } else { '16.9.0' }
$PgPort   = 55432
$PgData   = Join-Path $Root '.local\pgdata'
$PgHome   = Join-Path $Root '.local\pgsql'
$Downloads = Join-Path $Root '.local\dl'
$DbName   = 'salesmaya_agent'
$DbUser   = 'lad'
$Py       = Join-Path $Root '.venv\Scripts\python.exe'

function Say  { param($m) Write-Host "`n==> $m" -ForegroundColor White }
function Note { param($m) Write-Host "    $m" }
function Die  { param($m) Write-Host "`nerror: $m" -ForegroundColor Red; exit 1 }

function Get-Remote {
  param($Url, $Path)
  if (Test-Path $Path) { return }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
  $tmp = "$Path.partial"
  Invoke-WebRequest -Uri $Url -OutFile $tmp -UseBasicParsing -TimeoutSec 900
  # Download to a partial name and rename on success, so an interrupted fetch
  # cannot leave a truncated file that the next run treats as already present.
  Move-Item $tmp $Path -Force
}

# ---------------------------------------------------------------- python -----
Say 'Python environment'
if (-not (Test-Path $Py)) {
  if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv python install 3.11
    uv venv --python 3.11
  } else {
    $p311 = Get-Command py -ErrorAction SilentlyContinue
    if (-not $p311) { Die 'need uv (https://astral.sh/uv) or the Python launcher with 3.11 installed' }
    py -3.11 -m venv .venv
  }
}
# The CPU backend extras are part of the demo, not an optional afterthought:
# without them serve_session.py cannot import its own adapters.
$extras = '.[dev,api,livekit,db,stt-cpu,mt-cpu,tts-cpu]'
if (Get-Command uv -ErrorAction SilentlyContinue) {
  uv pip install --python $Py -e $extras
} else {
  & $Py -m pip install --quiet --upgrade pip
  & $Py -m pip install -e $extras
}
Note (& $Py --version)

# -------------------------------------------------------------- postgres -----
Say 'Postgres'
if (-not (Test-Path (Join-Path $PgHome 'bin\initdb.exe'))) {
  $jar = Join-Path $Downloads "embedded-postgres-$PgVersion.jar"
  $url = "https://repo1.maven.org/maven2/io/zonky/test/postgres/embedded-postgres-binaries-windows-amd64/$PgVersion/embedded-postgres-binaries-windows-amd64-$PgVersion.jar"
  Note "fetching Postgres $PgVersion binaries"
  Get-Remote -Url $url -Path $jar
  # The jar holds one .txz holding the server tree. Python does the unpacking
  # because Expand-Archive reads neither a .jar extension nor xz.
  & $Py -c @"
import io, lzma, os, sys, tarfile, zipfile
jar, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(jar) as z:
    member = next(n for n in z.namelist() if n.endswith('.txz'))
    blob = z.read(member)
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(blob))) as t:
    t.extractall(out)
"@ $jar $PgHome
  if ($LASTEXITCODE -ne 0) { Die 'could not unpack the Postgres binaries' }
}
Note (& (Join-Path $PgHome 'bin\postgres.exe') --version)

if ($RecreateDb -and (Test-Path $PgData)) {
  & (Join-Path $PgHome 'bin\pg_ctl.exe') -D $PgData stop 2>$null | Out-Null
  Remove-Item $PgData -Recurse -Force
  Note 'old cluster removed'
}

if (-not (Test-Path (Join-Path $PgData 'PG_VERSION'))) {
  # Bootstrap superuser is lad, so LAD_DATABASE_URL connects as itself with no
  # role-mapping step. trust auth: the cluster listens on loopback only.
  & (Join-Path $PgHome 'bin\initdb.exe') -D $PgData -U $DbUser --auth=trust --encoding=UTF8 | Out-Null
  if ($LASTEXITCODE -ne 0) { Die 'initdb failed' }
  Note 'cluster created at .local\pgdata'
}

$env:LAD_DATABASE_URL = "postgresql://$DbUser@127.0.0.1:$PgPort/$DbName"
$env:LAD_CONTROL_SCHEMA = 'lad_dev'

& $Py (Join-Path $Root 'tools\pg_admin.py') ready --wait 2 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  & (Join-Path $Root 'tools\pg.ps1') start | Out-Null
}
& $Py (Join-Path $Root 'tools\pg_admin.py') ready --wait 20
if ($LASTEXITCODE -ne 0) { Die "Postgres did not come up; see .local\pg.log" }

& $Py (Join-Path $Root 'tools\pg_admin.py') createdb | ForEach-Object { Note $_ }
& $Py (Join-Path $Root 'tools\seed_tenant.py') --slug techiemaya | ForEach-Object { Note $_ }

# --------------------------------------------------------------- livekit -----
Say 'LiveKit server'
$lkBin = Join-Path $Root '.local\livekit-server.exe'
if (-not (Test-Path $lkBin)) {
  $bare = $LiveKitVersion.TrimStart('v')
  $zip = Join-Path $Downloads "livekit_$bare`_windows_amd64.zip"
  Note "fetching livekit-server $LiveKitVersion"
  Get-Remote -Url "https://github.com/livekit/livekit/releases/download/$LiveKitVersion/livekit_${bare}_windows_amd64.zip" -Path $zip
  $stage = Join-Path $Downloads 'livekit'
  Expand-Archive -Path $zip -DestinationPath $stage -Force
  Move-Item (Join-Path $stage 'livekit-server.exe') $lkBin -Force
}
Note (& $lkBin --version)

# ---------------------------------------------------------------- models -----
if (-not $NoModels) {
  Say 'Models'
  Note 'translation and voice models, from the Hub (a few hundred MB)'
  & $Py (Join-Path $Root 'tools\fetch_mt_models.py') --pair en-fr --pair en-ar --pair en-hi
  if ($LASTEXITCODE -ne 0) { Note 'MT model fetch failed; rerun tools\fetch_mt_models.py once the Hub is reachable' }
  # hi routes to NLLB, and the demo asks for hi by default, so the 600M model is
  # part of the default language set rather than an extra.
  & $Py (Join-Path $Root 'tools\fetch_mt_models.py') --nllb
  if ($LASTEXITCODE -ne 0) { Note 'NLLB fetch failed; rerun tools\fetch_mt_models.py --nllb' }
  & $Py (Join-Path $Root 'tools\fetch_tts_voices.py') --defaults
  if ($LASTEXITCODE -ne 0) { Note 'voice fetch failed; rerun tools\fetch_tts_voices.py once the Hub is reachable' }
} else {
  Say 'Models skipped (-NoModels)'
  Note 'translated audio needs: tools\fetch_mt_models.py and tools\fetch_tts_voices.py'
}

Say 'Ready'
Write-Host @"

    .\tools\demo.ps1 up        start everything and print the join URL
    .\tools\demo.ps1 status    what is running
    .\tools\demo.ps1 down      stop everything

"@
