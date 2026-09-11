#!/usr/bin/env bash
#
# Provision the LAD Live Translation VM (develop): SFU + session worker.
#
# Runs on stock Debian 12, which ships Python 3.11 - the version this project
# requires. No GPU and no CUDA: the SFU is pure media routing, and the worker's
# STT keeps up on CPU at the window the chunker uses (see deploy/README.md).
#
#   sudo LAD_TRANSLATE_SFU_HOST=translate-sfu-dev.mrlads.com \
#        GCP_PROJECT=lad-develop \
#        bash deploy/vm/bootstrap.sh
#
# Idempotent: safe to re-run after a change to the units or the config.

set -euo pipefail

: "${LAD_TRANSLATE_SFU_HOST:?set LAD_TRANSLATE_SFU_HOST, e.g. translate-sfu-dev.mrlads.com}"
: "${GCP_PROJECT:=lad-develop}"

REPO_DIR="/opt/lad-translate"
REPO_URL="https://github.com/techiemaya-admin/lad-translate.git"
# develop, not main. This is the develop-environment VM, the Cloud Build
# trigger fires on ^develop$, and main carries none of this - it is still at
# the commit before any deployment work. Defaulting to main here would
# `git reset --hard origin/main` over the checkout and install the wrong tree,
# quietly, on a box that then looks provisioned.
BRANCH="${LAD_TRANSLATE_BRANCH:-develop}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "==> $*"; }

# -----------------------------------------------------------------------------
log "System packages"
# -----------------------------------------------------------------------------
apt-get update -qq
apt-get install -y --no-install-recommends \
    git curl ca-certificates gnupg debian-keyring debian-archive-keyring \
    apt-transport-https ffmpeg build-essential

# -----------------------------------------------------------------------------
log "Users"
# -----------------------------------------------------------------------------
id -u livekit      &>/dev/null || useradd --system --no-create-home --shell /usr/sbin/nologin livekit
id -u ladtranslate &>/dev/null || useradd --system --create-home --home-dir /var/lib/ladtranslate --shell /bin/bash ladtranslate

# The console reads the session unit's journal to report chunks, drops and
# latency. journalctl shows a system user its own messages and nothing else, so
# without this it returned one line and every field on the status panel came
# back empty while the log itself was full - a console that looked broken and
# was only blind.
#
# Read-only, and deliberately a group rather than a sudoers entry: reading logs
# should not go through the same door as restarting units.
usermod -aG systemd-journal ladtranslate
install -d -o livekit -g livekit /var/lib/livekit
install -d -m 0755 /etc/livekit /etc/lad-translate

# -----------------------------------------------------------------------------
log "LiveKit server"
# -----------------------------------------------------------------------------
# The official binary works on Linux. The build-from-source dance with
# CGO_ENABLED=1 in the README is a macOS-only problem and does not apply here.
if ! command -v livekit-server &>/dev/null; then
    curl -sSfL https://get.livekit.io | bash
fi
livekit-server --version

install -m 0644 "${HERE}/livekit.yaml" /etc/livekit/livekit.yaml

# -----------------------------------------------------------------------------
log "Secrets from Secret Manager"
# -----------------------------------------------------------------------------
# The VM's service account needs roles/secretmanager.secretAccessor. Nothing is
# baked into an image and nothing is committed: .gitleaks.toml gates the repo,
# and it only helps if the real values never go near it.
sm() { gcloud secrets versions access latest --secret="$1" --project="${GCP_PROJECT}"; }

LIVEKIT_KEY="$(sm lad-translate-livekit-api-key)"
LIVEKIT_SECRET="$(sm lad-translate-livekit-api-secret)"
DATABASE_URL="$(sm lad-translate-database-url)"

# LiveKit's own key file: one "key: secret" mapping. This pair is validated
# against tokens minted by the Cloud Run service, so rotating one side alone
# rejects every listener.
umask 077
printf '%s: %s\n' "${LIVEKIT_KEY}" "${LIVEKIT_SECRET}" > /etc/livekit/keys.yaml
chown livekit:livekit /etc/livekit/keys.yaml
chmod 0400 /etc/livekit/keys.yaml

# Google sign-in for the console. Three secrets, and the console refuses to
# serve without them rather than serving unprotected - see console/auth.py.
CONSOLE_CLIENT_ID="$(sm lad-translate-console-oauth-client-id 2>/dev/null || true)"
CONSOLE_CLIENT_SECRET="$(sm lad-translate-console-oauth-client-secret 2>/dev/null || true)"
CONSOLE_SESSION_SECRET="$(sm lad-translate-console-session-secret 2>/dev/null || true)"
if [[ -z "${CONSOLE_CLIENT_ID}" || -z "${CONSOLE_CLIENT_SECRET}" ]]; then
    log "  WARNING: no console OAuth secrets; the console will answer 503"
fi
CONSOLE_HASH=""   # retained only so the route block below stays conditional

cat > /etc/lad-translate/secrets.env <<EOF
LAD_DATABASE_URL=${DATABASE_URL}
LIVEKIT_API_KEY=${LIVEKIT_KEY}
LIVEKIT_API_SECRET=${LIVEKIT_SECRET}
EOF
chown ladtranslate:ladtranslate /etc/lad-translate/secrets.env
chmod 0400 /etc/lad-translate/secrets.env
umask 022

# -----------------------------------------------------------------------------
log "Application"
# -----------------------------------------------------------------------------
# safe.directory, or the second run of this "idempotent" script dies.
#
# The first run clones as root and then chowns the tree to ladtranslate. Every
# run after that has root operating on a repo owned by someone else, which git
# refuses as "dubious ownership" - and under set -e that aborts bootstrap right
# here, after the packages and the venv and before the units. The box is then
# half-provisioned while the run looks like it merely stopped early.
#
# Set per-invocation rather than in root's global config: this grants an
# exception for exactly this path, for exactly these commands.
GIT="git -c safe.directory=${REPO_DIR}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
else
    ${GIT} -C "${REPO_DIR}" fetch --prune origin
    ${GIT} -C "${REPO_DIR}" checkout "${BRANCH}"
    ${GIT} -C "${REPO_DIR}" reset --hard "origin/${BRANCH}"
fi
chown -R ladtranslate:ladtranslate "${REPO_DIR}"
log "  at $(${GIT} -C "${REPO_DIR}" log --oneline -1)"

command -v uv &>/dev/null || {
    curl -LsSf https://astral.sh/uv/0.12.5/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
}

# Backends installed here, unlike the Cloud Run image which has none. This is
# the machine that actually runs inference.
sudo -u ladtranslate env PATH="/usr/local/bin:${PATH}" bash -c "
    set -e
    cd '${REPO_DIR}'
    # --allow-existing, because uv venv is a hard error when .venv is already
    # there and this script is meant to be re-runnable. NOT --clear: that
    # deletes and rebuilds the environment, so a re-provision would reinstall
    # every wheel to reach the state it was already in.
    # 3.12, not Debian 12's stock 3.11.2. NeMo's safe_extract passes filter=
    # to TarFile.extract, which landed in 3.11.4; on 3.11.2 no .nemo file loads
    # at all and the TypeError names neither Python nor the version, so the
    # streaming STT backend is simply unavailable on the system interpreter.
    # uv fetches a standalone 3.12 rather than touching the OS one.
    # --allow-existing REUSES the interpreter a venv was built with, so on a
    # box provisioned before this change it would quietly stay on 3.11 and the
    # streaming backend would remain unavailable with no error anywhere. Check
    # the version and rebuild when it is wrong.
    want=3.12
    have=\$(.venv/bin/python -c 'import sys;print(\"%d.%d\" % sys.version_info[:2])' 2>/dev/null || echo none)
    if [ \"\$have\" != \"\$want\" ]; then
        echo \"    venv is python \$have, rebuilding on \$want\"
        rm -rf .venv
    fi
    uv venv --python 3.12 --allow-existing
    uv pip install -e '.[stt-cpu,mt-cpu,tts-cpu,stt-streaming,livekit,db,api,console]'
"

# -----------------------------------------------------------------------------
log "Models"
# -----------------------------------------------------------------------------
# ~2GB, fetched once onto local disk. This is a large part of why the worker is
# not a Cloud Run service: paying this on a cold start would blow the entire
# two second latency budget before a word is transcribed.
#
# The STT model is pulled on first use by faster-whisper, not here. Warm it
# during provisioning rather than during a keynote - the benchmark below does
# exactly that as a side effect.
#
# The MT weights come from third-party CTranslate2 conversions on the Hub,
# whose coverage is patchy and which can vanish. Fine for develop. Before a
# venue, convert the official Helsinki-NLP models and serve them from our own
# bucket - README, "Getting the models".
sudo -u ladtranslate bash -c "
    set -e
    cd '${REPO_DIR}'
    .venv/bin/python tools/fetch_mt_models.py --pair en-fr --pair en-de --pair en-es --pair en-ar
    .venv/bin/python tools/fetch_tts_voices.py --defaults
"

# -----------------------------------------------------------------------------
log "Caddy (TLS for the signalling connection)"
# -----------------------------------------------------------------------------
if ! command -v caddy &>/dev/null; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq
    apt-get install -y caddy
fi

install -m 0644 "${HERE}/Caddyfile" /etc/caddy/Caddyfile
install -d -m 0755 /etc/caddy/conf.d

# The console route exists only when it has a password to sit behind. Emitting
# it with an empty hash is what took Caddy - and with it every listener's TLS -
# down on the first deploy of this feature.
if [[ -n "${CONSOLE_CLIENT_ID}" ]]; then
    cat > /etc/caddy/conf.d/console.conf <<EOF
handle /console* {
	# No basic_auth. The console verifies Google identity itself, so there is
	# no password to mistype, no username field to leave blank, and no browser
	# dialog re-challenging uncached subresources. All of those happened.
	#
	# No strip_prefix either: the app owns /console and serves its own assets
	# from /console/static, so the prefix has to survive the proxy.
	reverse_proxy localhost:8090
}
EOF
    log "  console exposed at /console"
else
    rm -f /etc/caddy/conf.d/console.conf
    log "  console NOT exposed: no lad-translate-console-oauth-client-id secret"
fi

# Validate before restarting. A bad Caddyfile takes the SFU offline, and
# finding that out from `systemctl restart` means it is already down.
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 || {
    log "ERROR: Caddyfile does not validate; leaving the running config alone"
    exit 1
}
install -d /etc/systemd/system/caddy.service.d
cat > /etc/systemd/system/caddy.service.d/override.conf <<EOF
[Service]
Environment=LAD_TRANSLATE_SFU_HOST=${LAD_TRANSLATE_SFU_HOST}
EOF

# -----------------------------------------------------------------------------
log "Units"
# -----------------------------------------------------------------------------
install -m 0644 "${HERE}/livekit-server.service"        /etc/systemd/system/
install -m 0644 "${HERE}/lad-translate-session@.service" /etc/systemd/system/
install -m 0644 "${HERE}/lad-translate-console.service"  /etc/systemd/system/

# The console manages units through three verbs on one unit pattern rather than
# running as root. It serves a web page; the blast radius of a bug in it should
# be a restarted translation session and nothing else.
install -m 0440 -o root -g root "${HERE}/lad-translate-console.sudoers" \
    /etc/sudoers.d/lad-translate-console
visudo -cf /etc/sudoers.d/lad-translate-console >/dev/null

# The console writes session.env, so it has to own it. Everything in there is
# operational config; the credentials live in secrets.env, which stays 0400.
# Who may sign in is operational state, not a secret, and it is edited on the
# box - so carry the existing values forward. This file is REGENERATED every
# run, so without this an allowlist entry added after a deploy silently
# disappears at the next one and locks its owner out.
EXISTING_EMAILS="$(grep -oP '^CONSOLE_ALLOWED_EMAILS=\K.*' /etc/lad-translate/console.env 2>/dev/null || true)"
EXISTING_DOMAINS="$(grep -oP '^CONSOLE_ALLOWED_DOMAINS=\K.*' /etc/lad-translate/console.env 2>/dev/null || true)"

cat > /etc/lad-translate/console.env <<EOF
LAD_TRANSLATE_PUBLIC_BASE=${LAD_TRANSLATE_PUBLIC_BASE:-https://lad-translate-dev-kunfx3bnvq-ww.a.run.app}
CONSOLE_OAUTH_CLIENT_ID=${CONSOLE_CLIENT_ID}
CONSOLE_OAUTH_CLIENT_SECRET=${CONSOLE_CLIENT_SECRET}
CONSOLE_OAUTH_REDIRECT_URI=https://${LAD_TRANSLATE_SFU_HOST}/console/auth/callback
CONSOLE_SESSION_SECRET=${CONSOLE_SESSION_SECRET}
CONSOLE_ALLOWED_DOMAINS=${CONSOLE_ALLOWED_DOMAINS:-${EXISTING_DOMAINS:-techiemaya.com}}
CONSOLE_ALLOWED_EMAILS=${CONSOLE_ALLOWED_EMAILS:-${EXISTING_EMAILS}}
EOF
# Carries the OAuth client secret, so it is not world-readable like the rest of
# the operational config.
chmod 0400 /etc/lad-translate/console.env
chown ladtranslate:ladtranslate /etc/lad-translate/console.env

# Never clobber an edited session.env on a re-run: it carries the per-event
# language list and the chunker pair, and losing those mid-setup is silent.
#
# But "leave it alone" is not enough either. A release that adds a setting - as
# LAD_TRANSLATE_STT_THREADS did - installs a unit that references a variable
# the live file has never heard of, and systemd expands it to nothing. The
# service then starts with an empty flag value and dies in argparse, on a box
# that provisioned without a single error.
#
# So: keep every existing value, and append only the keys that are missing.
if [[ ! -f /etc/lad-translate/session.env ]]; then
    install -m 0644 "${HERE}/session.env.example" /etc/lad-translate/session.env
    sed -i "s|translate-sfu-dev.mrlads.com|${LAD_TRANSLATE_SFU_HOST}|" /etc/lad-translate/session.env
else
    added=0
    while IFS= read -r key; do
        if ! grep -q "^${key}=" /etc/lad-translate/session.env; then
            line="$(grep "^${key}=" "${HERE}/session.env.example")"
            line="${line//translate-sfu-dev.mrlads.com/${LAD_TRANSLATE_SFU_HOST}}"
            printf '\n# added by bootstrap.sh: new setting in this release\n%s\n' "$line" \
                >> /etc/lad-translate/session.env
            log "  session.env: added ${key}"
            added=$((added + 1))
        fi
    done < <(grep -oE '^[A-Z_]+=' "${HERE}/session.env.example" | tr -d '=')
    log "  /etc/lad-translate/session.env kept, ${added} new key(s) appended"
fi
chown ladtranslate:ladtranslate /etc/lad-translate/session.env

# -----------------------------------------------------------------------------
log "Database"
# -----------------------------------------------------------------------------
# Apply pending tenant migrations before anything that reads the new tables
# starts. Nothing at runtime migrates: the session and the console both assume
# the schema is there, and 002_audio_outputs.sql taught us what "assume" costs
# - a console that answered "no devices" for a table that did not exist.
#
# --migrate-only, so a deploy can never invent a tenant. Seeding one is a
# decision about identity that a person makes once, with --id, so this box and
# the platform agree on who the tenant is.
if [[ -n "${DATABASE_URL}" ]]; then
    TENANT_SLUG="$(grep -oP '^LAD_TRANSLATE_TENANT=\K.*' /etc/lad-translate/session.env || true)"
    CONTROL_SCHEMA="$(grep -oP '^LAD_CONTROL_SCHEMA=\K.*' /etc/lad-translate/session.env || true)"
    if [[ -n "${TENANT_SLUG}" ]]; then
        sudo -u ladtranslate env \
            LAD_DATABASE_URL="${DATABASE_URL}" LAD_CONTROL_SCHEMA="${CONTROL_SCHEMA}" \
            "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/tools/seed_tenant.py" \
            --slug "${TENANT_SLUG}" --migrate-only \
            || log "WARNING: tenant migrations did not apply; the console's hardware output panel will say so"
    else
        log "  no LAD_TRANSLATE_TENANT in session.env; skipping tenant migrations"
    fi
else
    log "  no database URL; skipping tenant migrations"
fi

systemctl daemon-reload
systemctl enable --now livekit-server caddy lad-translate-console
systemctl restart caddy lad-translate-console

# -----------------------------------------------------------------------------
log "Verify"
# -----------------------------------------------------------------------------
# A unit being "active" says the process started, not that it works. Ask it.
sleep 3
systemctl is-active --quiet livekit-server || { journalctl -u livekit-server -n 40 --no-pager; exit 1; }
curl -fsS "http://127.0.0.1:7880/" >/dev/null && log "SFU answering on 7880"

# The one number that decides whether this box can run an event. STT must
# finish a 6s window inside the 3.0s emit interval or the backlog guard starts
# shedding audio, which is heard as gaps rather than as an error.
log "STT benchmark (this warms the model cache too)"
sudo -u ladtranslate "${REPO_DIR}/.venv/bin/python" \
    "${REPO_DIR}/deploy/vm/benchmark_stt.py" --model "${LAD_TRANSLATE_STT_MODEL:-small}" \
    --threads "${LAD_TRANSLATE_STT_THREADS:-8}" \
    || log "WARNING: benchmark failed. Do not run an event until this passes."

cat <<EOF

Provisioned.

  SFU        wss://${LAD_TRANSLATE_SFU_HOST}   (signalling, via Caddy)
  media      UDP 50000-60000 / TCP 7881 direct to this VM's external IP
  app        ${REPO_DIR}
  console    https://${LAD_TRANSLATE_SFU_HOST}/console   (Google sign-in)
  STT        CPU. Re-run deploy/vm/benchmark_stt.py after any resize.

Start a talk:

  systemctl start lad-translate-session@keynote-hall-a
  journalctl -u lad-translate-session@keynote-hall-a -f

The join and speak URLs are printed in that log, and they are served by the
Cloud Run service, not by this box.
EOF
