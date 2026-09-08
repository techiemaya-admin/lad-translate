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
    uv venv --python 3.11
    uv pip install -e '.[stt-cpu,mt-cpu,tts-cpu,livekit,db,api]'
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

systemctl daemon-reload
systemctl enable --now livekit-server caddy
systemctl restart caddy

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
  STT        CPU. Re-run deploy/vm/benchmark_stt.py after any resize.

Start a talk:

  systemctl start lad-translate-session@keynote-hall-a
  journalctl -u lad-translate-session@keynote-hall-a -f

The join and speak URLs are printed in that log, and they are served by the
Cloud Run service, not by this box.
EOF
