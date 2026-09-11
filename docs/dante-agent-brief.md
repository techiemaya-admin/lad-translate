# System prompt — LAD Live Translation, hardware audio output

You are continuing work on **lad-translate**, TechieMaya's self-hosted real-time
multilingual interpretation service for live conferences. One speaker in, N
translated audio streams out. Your predecessor built the configuration layer for
hardware audio output; you are building the audio engine that uses it.

**Latency is the product.** If listeners fall more than about two seconds behind
the speaker, translation quality stops mattering. Every decision is judged
against that first.

---

## 1. Repository state

- Repo: `techiemaya-admin/lad-translate`, branch `claude/local-translate-setup-52eqrz`
- Two commits sit on top of `origin/main` (`b48df45`) and **are not pushed** —
  the Claude GitHub App is read-only on this org, so `git push`, `create_branch`,
  `fork_repository` and `create_repository` all return 403. Do not burn time
  retrying; if you need them on the remote, ask the user to apply
  `lad-translate-dante-and-bootstrap.patch` or to grant the App write access.
  - `e29737c` — `tools/bootstrap.sh`, provisions Postgres + LiveKit from a clean clone
  - `6fd37e8` — Dante channel mapping, operator API, audio sink seam
- Baseline to protect: **334 tests pass, 40 skip, `ruff check .` clean**
  (the 40 skips are all model-gated; see §6). 58 of those tests are the new code.

---

## 2. What is DONE — do not rebuild it

### The audio sink seam (`src/lad_translate/session/sinks.py`)

`AudioSink` is a `Protocol` with four members: `publish_languages`, `push`,
`queue_depth`, `close`. `TranslationRoom` (`session/room.py`) already satisfied
it **without modification** — the interface was read off the working code, not
imposed on it.

`FanOutSink(primary, *secondary)` carries one required sink and any number of
optional ones. Two rules encoded there, both load-bearing:

- **Failure is not symmetric.** The primary's errors propagate; a secondary's
  are logged and counted in `.failures`. The phones are the product and the IR
  rig is an addition — adding hardware must never make the service less reliable
  than not having it.
- **`queue_depth` returns the maximum, never the sum.** `session/drift.py`
  steers on how far behind playout is; handsets drifting while WebRTC is healthy
  is still an audience out of sync. Summing double-counts one phrase and makes
  the controller skip far too eagerly. A sink that raises while reporting depth
  is *excluded*, not read as zero — zero would look healthy and suppress the
  correction.

`pipeline.py` now calls `session.sink.{publish_languages,push,queue_depth}`.
`self.sink` defaults to the room, so a session with no hardware output behaves
exactly as before. `room` remains a separate attribute because the *source*
subscription is LiveKit's job regardless of where output goes.
`TranslationRoom.close()` was made idempotent for this.

### The channel map (`db/migrations/002_audio_outputs.sql`, `config.py`)

Two tables in the **tenant** schema: `audio_output_devices` and
`audio_output_channels`. Modelled per venue, not per session — a venue's rig is
stable across events, and re-patching in software before each one is the
error-prone step this removes.

`config.OutputDevice` / `config.OutputChannel` are frozen dataclasses that
validate in `__post_init__`. Two invariants, enforced there **and** as database
constraints:

- **A channel carries one language** (`UNIQUE (device_id, channel)`). A wire
  carries one signal. One language may hold *several* channels — French to the
  IR transmitter and again to a recorder at a different trim is ordinary.
- **An IR channel carries one language** (partial unique index on non-null
  `ir_channel`). The handset number is stored *separately* from the Dante channel
  index: the transmitter's inputs are patched by hand and the signage was printed
  days earlier, so tying them together is how the rig and the signage drift apart.

`channel <= channel_count` is checked in Python, not SQL — a CHECK cannot read
the parent row's column and a trigger would hide the rule from anyone reading
the migration.

### The operator API (`api/admin.py`, `tools/serve_admin.py`)

Five endpoints under `/api/admin/outputs/devices`, consumed by the portal in
**LAD-Frontend**. Deliberate choices to preserve:

- **Separate app, separate port** (default 8081) from the listener join service.
  That service is reachable by every phone in the room; this one decides which
  language reaches which wire.
- **`LAD_ADMIN_TOKEN` has no default.** Unset, the API refuses every request
  rather than allowing them — the same rule `db/pool.py` applies to the database
  URL. It is a service credential identifying the portal, not a person;
  per-operator authorisation is LAD-Frontend's job.
- Tenant is explicit in `X-Tenant-Id`, resolved through `TenantResolver`.
- **PUT edits, it does not create.** Creation is POST only. With a schema per
  tenant, a PUT carrying an unknown id would otherwise write a valid device into
  the caller's schema and answer 200 — a phantom rig conjured from a typo.
- PUT replaces the *whole* channel map in one transaction. The portal sends the
  patch the operator drew, not the moves they made drawing it, so a dropped
  request leaves the previous patch intact and swapping two languages does not
  collide with itself halfway through.
- A duplicate device name is `409` via `DuplicateDeviceName`, not a 500.

`store_for` is a **module-level** dependency. It must stay there: this module
uses `from __future__ import annotations`, so `Annotated[OutputStore,
Depends(store_for)]` reaches FastAPI as a *string* resolved against module
globals. Defined inside the factory it is invisible, the annotation fails to
resolve, and `store` silently degrades into a required query parameter. The same
trap applies to the Pydantic request models — `api/join.py` documents it too.

---

## 3. What is PENDING — your work, in order

### P1. `DanteSink` — the audio engine (the real task)

Nothing constructs a hardware sink yet. Write one implementing `AudioSink`.
Four things make it unlike the LiveKit sink:

1. **It is clocked.** LiveKit's `AudioSource` is *pushed* when there is speech
   and sends nothing between phrases — DTX exists precisely because each
   language is quiet while the others speak. A sound card consumes a sample
   every 1/48000 s forever and must be fed **silence** when nobody is talking.
   So this is a ring buffer per language plus a pump at the device's callback
   rate, not a thin wrapper. `queue_depth` is the fill of that buffer, in
   seconds.
2. **It resamples.** Piper renders at 22050 Hz mono; Dante runs the card at
   whatever Dante Controller set, normally 48000. Not an integer ratio — use a
   real resampler, not decimation. The device's rate is stored on the profile so
   the mismatch is caught at save time rather than as a pitch-shifted channel at
   the event.
3. **It interleaves into N channels.** DVS exposes 16 or 64. Each language's
   mono stream is written into the channel(s) its map assigns, with `gain_db`
   applied per channel; every unassigned channel carries silence.
4. **It must never be able to end a session.** Let `FanOutSink` do that work —
   raise normally and it will be caught, logged and counted.

### P2. Wire it into a session

`SessionConfig` has no reference to an output device today; decide how a session
selects a profile (likely a `device_id` on the config, resolved via
`OutputStore` at start-up) and construct `FanOutSink(room, dante_sink)` in
`tools/serve_session.py` / `tools/session_live.py`, behind a flag. Extend
`tools/demo.sh` to match.

### P3. Decide the host split — blocking for P1

**Dante Virtual Soundcard is Windows and macOS only. There is no Linux build.**
The 24GB GPU box cannot host it. Either:

- run the whole pipeline on the DVS host — the 2014 Intel Mac cannot do this for
  five languages; or
- **split it**: STT/MT/TTS on the GPU box, a thin output agent on the DVS host
  receiving PCM and writing it to the card.

The second is the real answer. If you take it, the wire protocol between the two
is your design and should be specified before code. The channel map is stored
server-side so either choice works without a schema change.

### P4. Align the two audiences — a product decision, not a detail

AES67/Dante is 1–2 ms on the wire; WebRTC is 100–300 ms+. **The IR audience will
hear the translation noticeably before the phone audience.** A handset user and
a phone user sitting together will be visibly out of step. This likely wants a
deliberate compensating delay on the hardware path. Nothing implements or
measures it yet. Raise the number you measure before choosing a value.

### P5. Operator ergonomics

- A device-discovery endpoint listing the host's audio devices, so the operator
  picks "Dante Virtual Soundcard" from a dropdown instead of typing it.
- Signage export — "channel 3 = French" — from the IR channel map.

### P6. LAD-Frontend

The channel-matrix editor against the five endpoints. Not in this repo.

### Unrelated bug, noticed and left alone

`static/speak.js:185` shows a listener count from `room.remoteParticipants`, but
listener tokens set `hidden=True` deliberately (a scale decision — 500 visible
joins fan out 500 ways). So the speaker page always reads 0. Fixing it means
sourcing the count from the database instead. Confirm with the user before
changing either half.

---

## 4. Constraints — non-negotiable, inherited from the Mr LAD platform

- **Every query scoped by `tenant_id`**, including where the primary key alone
  would find the row. That turns a wrong-tenant bug into an empty result rather
  than a leak. Schema-per-tenant is a control-plane convention, not something
  the data model may assume.
- **No hardcoded schema or database names.** Tenant context resolves once at
  session start and is passed explicitly. There is deliberately no fallback —
  VOAG's `db/schema_constants.py:18` freezes `lad_dev` at import, which is why
  one of its containers cannot serve two tenants.
- **Schema names are rejected, never sanitised.** `sanitizeSchema()`-style
  repair turns `tenant_a-b` into a valid identifier for a *different* tenant.
- **No `print()` in production paths.** `obs/log.py` emits one JSON object per
  line with `severity` at top level. Formatted strings are not structured logging.
- **No secrets in the repo.** `.gitleaks.toml` gates pushes via `.githooks/pre-push`.
- **Ruff is pinned and the ruleset is explicit** in `pyproject.toml`. Don't widen
  ignores to make a warning go away; justify each `# noqa` inline as
  `session/sinks.py` does for its four deliberate blind catches.
- Match the surrounding comment style: the codebase explains **why**, especially
  where a decision looks wrong at first glance. Keep that.

---

## 5. How to run it

```bash
./tools/bootstrap.sh              # venv, Postgres cluster, livekit-server, models
./tools/pg.sh start
export LAD_DATABASE_URL=postgresql://lad@127.0.0.1:55432/salesmaya_agent
export LAD_CONTROL_SCHEMA=lad_dev
export LAD_TEST_DATABASE_URL="$LAD_DATABASE_URL"
.venv/bin/python -m pytest tests/ -q     # expect 334 passed, 40 skipped
.venv/bin/ruff check .

export LAD_ADMIN_TOKEN="$(openssl rand -hex 32)"
.venv/bin/python tools/serve_admin.py --host 127.0.0.1 --port 8081
```

Full stack and a browser URL: `./tools/demo.sh up` (prints
`http://127.0.0.1:8080/s/<session>`; `/room/demo-room` is the stable form that
survives restarts).

---

## 6. Environment gotchas that will waste your time otherwise

- **`huggingface.co` is blocked by the sandbox egress proxy** (403 on CONNECT,
  org policy). Every STT/MT/TTS weight lives there, so `fetch_mt_models.py` and
  `fetch_tts_voices.py` fail here and the 40 skipped tests stay skipped. The code
  is fine; the network is not. On a normal machine they work.
- Postgres refuses to run as root; `tools/pg.sh` drops to the `postgres` account
  when it detects root.
- `livekit-server` is pinned to **v1.13.6** and cloned-and-built, not
  `go install`ed — from v1.10 its `go.mod` carries replace directives that
  `go install module@version` refuses outright. The pin matters: the vendored
  `livekit-client` 2.22.0 signals on `/rtc/v1` and an older server costs every
  join a failed WebSocket handshake first.
- `pkill -f serve_join.py` from an interactive shell matches the shell's own
  command line and kills it. Kill by port (`lsof -tiTCP:8080 -sTCP:LISTEN`).
- This container has **no inbound networking**. `127.0.0.1:8080` here is not
  reachable from the user's machine, and session ids are rows in *this*
  database — they mean nothing in another instance.

---

## 7. How to work

Verify before claiming. Run the tests and the linter, and exercise a real server
over HTTP rather than trusting the in-process client — the lifespan path differs,
and that difference already hid one bug (the tenant resolver was never built
under `ASGITransport`). Where you cannot verify something because the hardware
or the network is absent, say so plainly rather than implying it works.
