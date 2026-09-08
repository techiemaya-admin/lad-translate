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
   |  Cloud Run                |        |   VM (GCE, n2, CPU)    |
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

Region **`me-central1`** (Doha) — ~10ms from Dubai, which is where this sells.

An earlier revision put this in `asia-south1` because the GPU backends want an
L4 and Doha has no GPUs. That reasoning was wrong: **the GPU was for code that
has never run.** FastConformer, Qwen3-ASR and Chatterbox are all "written,
unrun" in the README status table. What executes is faster-whisper + Opus-MT +
Piper, and measured at the window the chunker actually uses — 6s window, 3.0s
emit, so STT must finish 6s of audio in under 3s — CPU int8 keeps up:

| Model | Threads | p50 | p95 | Budget |
|---|---|---|---|---|
| `tiny` | 8 | 0.26s | 0.45s | 3.0s |
| `small` | 8 | 1.39s | 1.63s | 3.0s |

Measured on an Apple M4, so read it as evidence that CPU is the right shape,
not as the number for an `n2`. An n2 vCPU is slower per thread; the answer is
more of them, which is why the VM is 32 vCPU. **Re-run
`deploy/vm/benchmark_stt.py` on the VM itself before an event.**

Note what the SFU needs, which is nothing: LiveKit is pure media routing, no
inference. Only the session worker loads a model. Conflating the two is what
sent this to Mumbai.

The Gulf alternatives, for the record:

| Region | Cloud Run | Compute | GPU | Verdict |
|---|---|---|---|---|
| `me-central1` (Doha) | yes | yes | none | **chosen** — CPU is enough |
| `me-central2` (Dammam) | blocked | blocked | L4 | region not enabled for this project |
| `me-west1` (Tel Aviv) | yes | yes | T4, A100 | ~2,000km from Dubai, no better than Mumbai |

Dammam returns `PERMISSION_DENIED ... Access to the region is unavailable.
Please contact our sales team` — an allowlisted-region entitlement, not IAM and
not `constraints/gcp.resourceLocations` (`allValues: ALLOW` here). Worth
requesting in the background: if the streaming FastConformer work lands, an L4
there is its natural home. Not worth waiting for now.

**The catalog is not the entitlement.** `machine-types list` and
`accelerator-types list` read a global catalog and will list hardware in a
region the project cannot touch — exactly how Dammam looked viable. Probe it:

```bash
gcloud compute addresses create probe --region=<REGION> && \
  gcloud compute addresses delete probe --region=<REGION> --quiet
```

Verified in `lad-develop` on 8 Sep 2026: Artifact Registry create/delete,
Cloud Run list and a Compute address probe all succeed in `me-central1`, and
`CPUS` quota is `limit=100 usage=0`.

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

### 1b. Make the service public (once, as a project owner)

Only after the first deploy has created the service, and only once — a Cloud
Run service's IAM policy belongs to the service rather than to a revision, so
it survives every later deploy.

```bash
gcloud run services add-iam-policy-binding lad-translate-dev \
  --region=me-central1 --member=allUsers --role=roles/run.invoker
```

This is deliberately not `--allow-unauthenticated` in the build. That flag calls
`run.services.setIamPolicy`, which **`roles/editor` does not include** — an
editor can create and update a service but not grant access to it, which is
Google preventing privilege escalation rather than an oversight. The build
service account is an editor, so the flag can only warn and continue, leaving a
service nobody can reach while the deploy reports success.

The alternative is granting the build SA `roles/run.admin`. It is a shared
account used by every build in this project, so a one-time binding is the
smaller change. `cloudbuild-develop.yaml` verifies the binding on every build
and fails if it is missing, which is read-only and works with the roles the SA
already has.

Being unreachable presents badly: an unauthorised request to a private Cloud
Run service returns a Google HTML 404, not a 403, so it reads like a missing
route on a service that is running perfectly.

### 2. Artifact Registry

Already created in `lad-develop` on 7 Sep 2026. To recreate:

```bash
gcloud artifacts repositories create lad-translate-dev \
  --repository-format=docker --location=me-central1 \
  --description="LAD Live Translation, develop"
```

### 3. Cloud Build trigger

1st-gen GitHub App trigger, same shape as `voag-develop`:

| Field | Value |
|---|---|
| Repository | `techiemaya-admin/lad-translate` |
| Branch | `^develop$` |
| Config | `cloudbuild-develop.yaml` |

Connecting the repository is a console step and must happen first — the GitHub
App is authorised per repository, and until it is, trigger creation fails with
`FAILED_PRECONDITION: Repository mapping does not exist`:

<https://console.cloud.google.com/cloud-build/triggers;region=global/connect?project=160078175457>

```bash
gcloud builds triggers create github \
  --name=lad-translate-develop --region=global \
  --repo-owner=techiemaya-admin --repo-name=lad-translate \
  --branch-pattern='^develop$' \
  --build-config=cloudbuild-develop.yaml \
  --service-account=projects/lad-develop/serviceAccounts/160078175457-compute@developer.gserviceaccount.com \
  --description='LAD Live Translation - develop'
```

The build service account needs `roles/run.admin`,
`roles/iam.serviceAccountUser`, `roles/artifactregistry.writer` and
`roles/secretmanager.secretAccessor`. Already satisfied in `lad-develop`:
`160078175457-compute@` holds `roles/editor` and
`roles/secretmanager.secretAccessor`, and editor covers the first three.

## Branches

Three long-lived branches, matching every other LAD repo:

| Branch | Environment | Trigger |
|---|---|---|
| `develop` | develop | `lad-translate-develop` → Cloud Run `lad-translate-dev` |
| `stage` | stage | none yet — needs `cloudbuild-stage.yaml` and its own secrets |
| `main` | production | none yet |

Work lands on `develop` and is promoted forward. Note the trap carried over from
the rest of the platform: **merging to `stage` is not deploying.** Verify the
running revision actually carries your commit before believing it shipped.

## The GPU VM

### Create it

`n2-standard-32` — 32 vCPU, 128GB, no GPU. Sized off the measurement above:
STT is the only heavy stage and it is single-stream, so cores buy latency
headroom rather than throughput, and MT and TTS fan out per language on the
rest. Start here, then cut it down once a real event has produced numbers — an
idle n2-standard-32 is the most expensive thing in this design.

```bash
gcloud compute instances create lad-translate-sfu-dev \
  --project=lad-develop \
  --zone=me-central1-a \
  --machine-type=n2-standard-32 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=lad-translate-sfu
```

No GPU quota to request and no accelerator stock to chase, which is most of why
Doha is available today and Dammam is not. `CPUS` quota in `me-central1` is
`limit=100 usage=0`, so this fits with room for a second instance.

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
gcloud compute addresses create lad-translate-sfu-dev --region=me-central1
```

### Provision

```bash
gcloud compute ssh lad-translate-sfu-dev --zone=me-central1-a

# git first. A stock Debian image does not ship it, and bootstrap.sh only
# installs it AFTER this clone - so leaving this out fails with a bare
# "sudo: git: command not found" before the script has run a line.
sudo apt-get update && sudo apt-get install -y git

sudo git clone --branch develop https://github.com/techiemaya-admin/lad-translate.git /opt/lad-translate
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
every request, so `cloudbuild-develop.yaml` curls `/health` on the new
revision as its last step.

**Not `/healthz`.** Google's frontend reserves that exact string on
`*.run.app` and answers it itself with an HTML 404 that never reaches the
container — only that one path, while `/health`, `/livez`, `/readyz` and even
`/healthz2` pass through normally. It is an unpleasant failure to read: the
service is healthy, the logs show a clean startup, every other route including
the static bundle serves fine, and only the health check is dead. The app
answers on both paths; anything probing from outside Cloud Run must use
`/health`.

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
