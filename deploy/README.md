# Deploying LAD Live Translation — develop

## Why this is two targets, not one

The repo is one codebase and three runtime processes, and they do not want the
same platform.

| Process | Entry point | Runs on | Why |
|---|---|---|---|
| Join API | `tools/serve_join.py` | **Cloud Run** `lad-translate-dev` | Plain HTTP, no models, bursty load when a QR code goes up |
| LiveKit SFU | `livekit-server` | **GPU VM** | Needs UDP 50000–60000. Cloud Run has no UDP ingress at all |
| Session worker | `tools/serve_session.py` | **GPU VM** | Holds one WebRTC connection for a whole talk, ~2GB of weights resident, 2s latency budget |

Putting the SFU on Cloud Run is not a tuning problem, it is impossible: without
UDP every listener falls back to TCP relay, and transport eats the budget the
product is built around.

The worker is a softer no, and still a no. Cloud Run instances are reclaimed on
the platform's schedule and there is no way to say "this session belongs to this
instance". An instance recycled 40 minutes into a keynote takes the audience's
audio with it.

The code already anticipates the split. `api/tokens.py` carries `LIVEKIT_URL`
(what browsers dial) separately from `LIVEKIT_INTERNAL_URL` (what server-side
components dial), because on the dev Mac they must differ.

```
              phones                     venue laptop / phone
                 |                                |
          https  |  wss (signalling)              | wss (publish)
                 v                                v
   +---------------------------+        +------------------------+
   |  Cloud Run                |        |   GPU VM (GCE, L4)     |
   |  lad-translate-dev        | -----> |                        |
   |  join page, tokens        |  wss   |   Caddy :443           |
   +---------------------------+        |     -> LiveKit :7880   |
                 |                      |   UDP 50000-60000      |
                 |  asyncpg             |   session worker       |
                 v                      +------------------------+
        Postgres salesmaya_agent                  |
        control schema lad_dev  <-----------------+
```

## One-time setup in `lad-develop`

Region **`me-central2`** (Dammam), which deviates from VOAG's `asia-south1` on
purpose: this is a latency product and the venues are in Dubai. Dammam is about
430km away against Mumbai's ~1,900km, and media crosses the SFU twice — speaker
in, listener out — so the saving is paid twice per phrase.

`me-central1` (Doha) is nearer still and cannot be used. It has **no GPUs in any
zone** — `gcloud compute accelerator-types list --filter="zone~'me-central1'"`
returns nothing at all, and there are no `g2` machine types there either. Cloud
Run and Artifact Registry both exist in me-central1, so it is possible to put
the API there and the VM in me-central2; don't. It buys nothing and adds a
cross-region hop to `RoomInspector`, which the join page calls on every load.

### 1. Secrets

Three, shared by both targets. The LiveKit pair is one pair: Cloud Run mints the
tokens, the VM validates them. Rotating one side alone rejects every listener.

```bash
gcloud config set project lad-develop

printf 'postgresql://USER:PASS@165.22.221.77:5432/salesmaya_agent' \
  | gcloud secrets create lad-translate-database-url --data-file=-

openssl rand -hex 16 | tr -d '\n' \
  | gcloud secrets create lad-translate-livekit-api-key --data-file=-
openssl rand -base64 32 | tr -d '\n' \
  | gcloud secrets create lad-translate-livekit-api-secret --data-file=-
```

`tr -d '\n'` is not cosmetic. A trailing newline inside a secret is how the
Stripe 500 on the billing work happened: the value looks right everywhere you
read it and fails only where it is compared.

### 2. Artifact Registry

```bash
gcloud artifacts repositories create lad-translate-dev \
  --repository-format=docker --location=me-central2 \
  --description="LAD Live Translation, develop"
```

### 3. Cloud Build trigger

1st-gen GitHub App trigger, same shape as `voag-develop`:

| Field | Value |
|---|---|
| Repository | `techiemaya-admin/lad-translate` |
| Branch | `^main$` |
| Config | `cloudbuild-develop.yaml` |

The build service account needs `roles/run.admin`,
`roles/iam.serviceAccountUser`, `roles/artifactregistry.writer` and
`roles/secretmanager.secretAccessor`.

## The GPU VM

### Create it

`g2-standard-8` is one NVIDIA L4 (24GB), which is the "24GB box" the GPU
backends were written against.

```bash
gcloud compute instances create lad-translate-sfu-dev \
  --project=lad-develop \
  --zone=me-central2-a \
  --machine-type=g2-standard-8 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu124-debian-11 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=lad-translate-sfu
```

Verified in `lad-develop` on 7 Sep 2026, rather than assumed:

- L4 is in `me-central2-a` and `me-central2-c`. **Not `-b`** — putting the
  instance there fails with no capacity, which reads like a stock problem and
  is not one.
- `g2-standard-8` exists in both.
- `NVIDIA_L4_GPUS` quota is `limit=1, usage=0`, so one VM needs no quota
  request. A second one does.

Re-check before building, since stock moves:

```bash
gcloud compute accelerator-types list --filter="zone~'me-central2'"
gcloud compute regions describe me-central2 --format=json \
  | python3 -c "import json,sys; print([q for q in json.load(sys.stdin)['quotas'] if q['metric']=='NVIDIA_L4_GPUS'])"
```

Give the VM's service account `roles/secretmanager.secretAccessor` —
`bootstrap.sh` reads all three secrets at provision time.

### Firewall

Four rules, and the UDP one is the one people forget. Without it every listener
silently falls back to TCP and the latency numbers stop meaning anything.

```bash
gcloud compute firewall-rules create lad-translate-sfu-web \
  --allow=tcp:80,tcp:443 --target-tags=lad-translate-sfu \
  --description="ACME challenge and wss signalling via Caddy"

gcloud compute firewall-rules create lad-translate-sfu-media \
  --allow=udp:50000-60000,tcp:7881 --target-tags=lad-translate-sfu \
  --description="WebRTC media, and the TCP fallback for UDP-blocked wifi"
```

Port 7880 stays closed to the internet. Caddy fronts it on 443; the worker
reaches it on localhost.

### DNS

Point `translate-sfu-dev.mrlads.com` at the VM's external IP **before**
bootstrapping. Caddy asks Let's Encrypt for a certificate on first start, and
that fails against a name that does not resolve yet.

Reserve the IP as static, or the name breaks the next time the VM restarts:

```bash
gcloud compute addresses create lad-translate-sfu-dev --region=me-central2
```

### Provision

```bash
gcloud compute ssh lad-translate-sfu-dev --zone=me-central2-a
sudo git clone https://github.com/techiemaya-admin/lad-translate.git /opt/lad-translate
sudo LAD_TRANSLATE_SFU_HOST=translate-sfu-dev.mrlads.com GCP_PROJECT=lad-develop \
     bash /opt/lad-translate/deploy/vm/bootstrap.sh
```

Idempotent. Re-run it after a change to the units or the SFU config; it leaves
an edited `/etc/lad-translate/session.env` alone.

## Running a talk

```bash
systemctl start lad-translate-session@keynote-hall-a
journalctl -u lad-translate-session@keynote-hall-a -f
```

The log prints the session id and the `/speak/<id>` and `/s/<id>` paths. Both
are served by the **Cloud Run** service, not by the VM.

The unit is templated on room name so two concurrent talks cannot share a
session row or a set of language tracks. It is deliberately `Restart=on-failure`
and not `Restart=always`: the worker exiting when the talk ends is success, and
restarting would open a fresh session against an empty room every time.

Per-event settings — languages, Whisper size, chunker pair — live in
`/etc/lad-translate/session.env`.

## Verifying, rather than trusting a green deploy

A Cloud Run deploy reports SUCCESS for a container that starts and then fails
every request, so `cloudbuild-develop.yaml` curls `/healthz` on the new
revision as its last step.

The SFU has no equivalent, and the check that matters is not that the parts are
up. `tools/e2e.py` drives one real session through every layer and asserts that
three real WebRTC listeners receive audio above the silence floor:

```bash
cd /opt/lad-translate
sudo -u ladtranslate .venv/bin/python tools/e2e.py --targets fr,ar
```

Everything upstream can look healthy while the hall hears nothing. Only a
subscriber that actually receives audio proves otherwise.

## Known gaps in this deployment

Real, and not fixed by the scaffolding here.

- **The GPU backends have never run.** FastConformer STT, Qwen3-ASR and
  Chatterbox are marked "written, unrun" in the README. What this VM deploys is
  faster-whisper on CUDA — a real speedup over the dev Mac, but not the
  streaming transducer the latency argument depends on.
- **`serve_session.py` hardcodes the tenant.** Line 66 reads
  `FROM lad_dev.tenants WHERE slug='techiemaya'`, ignoring
  `LAD_CONTROL_SCHEMA`. That is exactly the pattern the repo's own Constraints
  section forbids, and it means this VM serves one tenant. Fine for develop,
  blocking for anything else.
- **The core Postgres is on DigitalOcean, and the worker now sits further from
  it.** `165.22.221.77` is a DigitalOcean address (`DIGITALOCEAN-165-22-0-0`),
  not a GCP one, and `session/pipeline.py:323` awaits `record_transcript` inside
  the per-language worker loop — so DB round-trip time comes out of the worker's
  real-time headroom on every phrase, per language. Moving from `asia-south1` to
  `me-central2` buys media latency and may cost DB latency. Measure it on the
  box before an event (`psql ... -c '\timing on' -c 'select 1'`); if it hurts,
  the fix is to make `persist` fire-and-forget rather than to move the region
  back, because the listener is what the budget is about.
- **One VM is one venue.** Sessions are pinned to the box running the SFU.
  Scaling past a single hall needs a room-to-node routing decision that does not
  exist yet.
- **No autoscaling and no off-hours schedule.** An L4 VM left running bills
  continuously. VOAG's dev VM is stopped off-hours; this one should be too.
