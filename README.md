# LAD Live Translation

Self-hosted real time multilingual interpretation for live conferences.
One speaker in, N translated audio streams out, delivered to phones over WebRTC.

Latency is the product. If listeners fall more than about two seconds behind the
speaker, translation quality does not matter.

## Status

Foundation and the highest-risk component are built and tested. Nothing is wired
to a room yet.

| Component | State |
|---|---|
| Adapter contracts (`adapters/base.py`) | Done |
| Phrase chunker (`chunker/`) | Done, 21 tests |
| Structured logging (`obs/log.py`) | Done |
| Latency instrumentation (`obs/latency.py`) | Done, 9 tests |
| Chunker tuning harness (`tools/chunker_replay.py`) | Done |
| Translation, Opus-MT (`adapters/mt_opus.py`) | Done, 7 tests |
| TTS, Piper (`adapters/tts_piper.py`) | Done, 7 tests |
| STT, faster-whisper (`adapters/stt_whisper.py`) | Done, dev only |
| Backend registry (`adapters/registry.py`) | Done |
| End to end smoke test (`tools/pipeline_smoke.py`) | Done |
| Session config and tenancy (`config.py`) | Done |
| Backpressure guard (`session/backpressure.py`) | Done, 11 tests |
| Tenancy and schema resolution (`db/tenancy.py`) | Done, 18 tests |
| Session data model (`db/`) | Done, 18 tests |
| Drift control (`session/drift.py`) | Done, 15 tests |
| Room transport (`session/room.py`) | Done |
| Session pipeline (`session/pipeline.py`) | Done, 15 tests |
| Listener tokens (`api/tokens.py`) | Done, 10 tests |
| Browser join page | Not started |
| Streaming STT adapter (FastConformer) | Needs the A4000 |

## Measured on the dev Mac

2014 Intel i5, two cores, no GPU. These are real numbers from this machine, not
estimates.

| Stage | Figure |
|---|---|
| Translation, one language | 41ms |
| Translation, four languages in parallel | 297ms p50, 416ms worst |
| TTS time to first audio, 5 word phrase | 266ms |
| TTS time to first audio, 12 word phrase | 468ms |
| TTS real time factor, warm | 0.12 |

Time to first audio scales with chunk length, because Piper renders a whole
sentence before releasing any of it. That couples the chunker's `max_words`
directly to TTS latency: longer chunks translate better and start speaking
later. Sweep the two together.

None of this includes STT, which on this machine is the whole problem.

## Setup

Needs Python 3.11+. This machine's system Python is 3.9, so `uv` provisions it.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --python 3.11
uv pip install -e '.[dev]'
```

Run the tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

## Tuning the chunker

The chunker decides when a revisable transcript has settled enough to translate.
Commit too early and the audience hears words the backend later corrects. Commit
too late and they fall behind. That trade is the whole game.

Sweep it:

```bash
.venv/bin/python tools/chunker_replay.py --sweep
```

Compare backend classes:

```bash
.venv/bin/python tools/chunker_replay.py --sweep --profile whisper
```

Inspect individual commits:

```bash
.venv/bin/python tools/chunker_replay.py --agreement 2 --max-wait 1.5 --verbose
```

The harness runs against a behaviour model by default. Those numbers compare
configurations against each other and nothing more. Real figures need real
hypotheses recorded from a real backend, replayed with `--source jsonl`.

## Backend selection

Adapters are protocols in `adapters/base.py`. The pipeline never imports a
backend. Swapping CPU for GPU means writing one adapter and changing config.

Every event carries two clocks. `t_audio` is the position in the source stream,
for lining transcript up with audio. `t_wall` is `time.monotonic()`, and it is
the only clock used to measure latency.

## Constraints

Inherited from the Mr LAD platform and not negotiable here.

- Every query scoped by `tenant_id`.
- No hardcoded schema or database names. Tenant context is resolved once at
  session start and passed explicitly. There is no default fallback: VOAG's
  `db/schema_constants.py:18` defaults to `lad_dev` and freezes it at import,
  which is why a single container there cannot serve two tenants.
- No `print()` in production paths. `obs/log.py` emits one JSON object per line
  with `severity` at the top level, which Cloud Logging parses into queryable
  fields. Formatted strings are not structured logging.
- Secrets from Secret Manager. `.gitleaks.toml` is carried over from MAGe and
  should gate deploys, as it does in `deploy-mage.yml`.

## Billing

Session duration multiplied by active language count. Not per call. It does not
key on `call_log_id`. VOAG's `recordVoiceCallUsage()` and `voice_call_logs` are
the wrong shape and are not reused.

## Getting the models

Translation and voice models are not in the repo.

```bash
.venv/bin/python tools/fetch_mt_models.py --pair en-fr --pair en-de --pair en-es --pair en-ar
.venv/bin/python tools/fetch_tts_voices.py --defaults
```

The translation models come from third-party CTranslate2 conversions on the
Hub. Coverage is patchy, several advertised repos ship no weights, and any of
them can vanish. That is fine for development and not fine for a venue. Before
production, run `fetch_mt_models.py --convert` on a machine with torch to
convert the official Helsinki-NLP models, and store the output in our own
bucket.

Piper voices come from the Piper project's own repository, so that problem does
not apply to them.

## End to end

```bash
.venv/bin/python tools/make_fixture.py
.venv/bin/python tools/pipeline_smoke.py --targets fr,de --realtime
```

Audio in, translated speech out, one WAV per language. Without `--realtime` the
file is read as fast as the disk allows and the tool refuses to report latency,
because the audio clock and the wall clock would have diverged.

## Why STT is the open problem

`adapters/stt_whisper.py` works and is development only. Whisper is a 30 second
window model, so every streaming wrapper is a sliding buffer re-transcribed on a
timer, which leaves a long unstable tail. Against the chunker that costs about
3.2 seconds at p50 before anything can be committed, on a 2 second budget. A
faster GPU does not close it, because the cost is architectural.

Production needs a streaming transducer. NVIDIA's cache-aware streaming
FastConformer is the intended target on the A4000. `registry.py` already has the
slot; the adapter is not written.

There is a second symptom worth knowing. Because Whisper only emits at silence,
the chunker never sees a genuine mid-utterance interim and degenerates to a
pass-through of Whisper's own segmentation. The smoke test detects this and says
so. The chunker's stability logic only earns its keep against a real streaming
backend.

## Backpressure

Feeding the 25.6s fixture at real speed with a backend that could not keep up
produced p50 90s and p95 168s, growing monotonically. Nothing noticed it was
behind, so the lag never recovered.

That is a design gap, not a hardware limit. A faster backend widens the margin
but adds no floor: any stall long enough to build a queue starts the same
runaway, and a listener three minutes behind the speaker has no product.

`session/backpressure.py` caps it. When the backlog passes `max_lag_s` the
oldest audio is discarded until the lag is back under `recover_to_s`. The
policy is deliberate and it loses speech on purpose:

> Being current matters more than being complete.

Shedding to `recover_to_s` rather than to the threshold gives hysteresis, so
the guard makes one clean cut instead of dropping a frame on every push. Every
drop is counted and logged at ERROR, because discarding a speaker's words must
appear on a dashboard rather than in a complaint after the event.

Enabled in the smoke test with `--max-lag 3.0`, disabled with `--max-lag 0`.

## Playout drift is real and measured

From the same run, output audio against 25.6s of English source:

| Language | Output | Expansion |
|---|---|---|
| French | 28.3s | +10.5% |
| German | 26.2s | +2.3% |

The French chain produces more audio than it consumes, and the chunks carry no
inter-chunk silence while the source does, so the true expansion is worse than
the table shows. Over a 45 minute keynote a 10% expansion is roughly four and a
half minutes of accumulated lag with nothing to absorb it.

The backlog guard does not fix this. It caps input lag; drift builds on the
output side. `VoiceSpec.speed` is the lever, and the pipeline needs a policy
that raises it when a language chain falls behind. That policy is not written
yet, and it is the next thing after the session pipeline.

### Two queues in series

Measured with the guard set to 3s:

Same 25.6s fixture, same machine, one language:

| Configuration | wall | p50 | p95 | audio dropped |
|---|---|---|---|---|
| No guard, 20s STT window | 193.2s | 90.5s | 168.2s | 0% |
| Guard 3s, 20s STT window | 35.1s | 13.8s | 19.5s | 40.9% |
| Guard 3s, 8s STT window | 32.3s | 9.4s | 10.2s | 24.5% |

Three things worth keeping from that table.

The guard works and its cost is brutal. At the middle row, forty one percent of
the speaker's audio was discarded and whole sentences vanished. On a backend
that cannot keep up, the choice is between unboundedly late and current but
lossy, and neither is a product. The guard's value is that on adequate hardware
it never fires, and when something does go wrong it fails recoverably rather
than running away.

The middle row stayed at 13.8s rather than dropping to 3s because the STT
adapter holds its own buffer, which the guard cannot see. Two queues in series,
one bounded. Whatever STT backend goes in, its internal buffering counts against
the same budget, and guarding only the visible queue measures the wrong thing.

Bounding the second queue helped twice over. Shorter windows make each
transcription pass cheaper, so the backend keeps up better, so the guard sheds
less: dropped audio fell from 40.9% to 24.5% while latency also improved. Where
buffering is the bottleneck, less of it is better on both axes.

Ten times better than where it started, and still 4.7x over a 2 second budget.
That gap is the backend architecture, not the tuning.

## Database

Single Postgres, one schema per tenant, `tenant_id` on every row. Same tenancy
model as VOAG.

One thing is done differently on purpose. VOAG resolves its schema at import:

```python
SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")        # db/schema_constants.py:18
CALL_LOGS_FULL = f"{SCHEMA}.{CALL_LOGS_TABLE}"    # baked at import
```

That makes the schema process-wide, so one container can serve exactly one
tenant, and when nothing resolves it writes into the shared control plane
instead of failing. Here the schema is looked up per tenant from
`{control}.tenants`, validated, and carried on `TenantContext`. No module
constant, no environment default, no fallback.

Schema names are **rejected**, not sanitised. VOAG's `sanitizeSchema()` strips
unexpected characters, which turns `tenant_a-b` into `tenant_ab`: a valid
identifier for a different tenant, with nothing reporting a problem. Identifiers
cannot be parameterised, so this is the one value that has to be beyond doubt.
The pattern is enforced in Python and again as a `CHECK` constraint on the
table.

Every query filters on `tenant_id` even where the primary key alone would find
the row. That turns a wrong-tenant bug into an empty result rather than a leak,
and `test_db_sessions.py` tests exactly that.

### Local setup

Postgres 16.6 runs from `.local/`, extracted from the EnterpriseDB binaries. No
Homebrew, no Docker, no admin password.

```bash
./tools/pg.sh start
export LAD_DATABASE_URL=postgresql://lad@127.0.0.1:55432/salesmaya_agent
export LAD_CONTROL_SCHEMA=lad_dev
.venv/bin/python tools/seed_tenant.py --slug techiemaya
```

`./tools/pg.sh psql` opens a shell against it, `stop` shuts it down.

### Billing

Session duration multiplied by active language count. Settled at close and
written to `billed_seconds` and `billed_language_count`, never recomputed later
from config, so a configuration change cannot alter an invoice already raised.
It is not per call and does not key on `call_log_id`.

## LiveKit

Self-hosted, built from source into `.local/livekit-server`.

The project ships no macOS binaries and its install script routes through
Homebrew, so it is built here. It needs `CGO_ENABLED=1`: `go-osstat` reads
darwin CPU counters through cgo, and a `CGO_ENABLED=0` build fails with
`undefined: cpu.Get`. That error looks like broken darwin support and is not.
On the Linux GPU box the official binary works and none of this applies.

```bash
./tools/livekit.sh start
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
```

`--dev` uses the well-known devkey/secret pair. Local development only.

### Session pipeline

```
source track -> backlog guard -> STT -> chunker
             -> translate fan-out -> per-language worker -> language track
```

One worker task per language, each with its own ordered queue. Phrases within a
language must be spoken in the order they were said, but a language that falls
behind must not hold up the others. A single shared queue gives ordering and
coupling; a task per chunk gives independence and scrambled speech. A worker per
language gives both.

Session limits cancel the pump rather than asking it to stop. A hung STT
backend never returns from `transcribe()`, so a cooperative flag would leave the
session running for ever and the caps would protect nothing.

### Token grants

`can_publish=False` and `can_publish_data=False` keep 500 phones from putting a
microphone in the room. `hidden=True` keeps them out of the participant list,
which is a scale decision rather than a privacy one: every visible participant
joining is broadcast to everyone already there, so 500 visible listeners means
500 joins fanned out 500 ways. That is most of the answer to risk area 4.

The venue publisher gets `can_subscribe=False`, so the laptop never pulls five
translated streams back down the connection carrying the one that matters.

Language tracks are not isolated from each other by the token. Any valid
listener can subscribe to any language. That is fine, since every track carries
a translation of the same public talk, but it is written down so nobody assumes
otherwise.

### End to end

```bash
./tools/livekit.sh start
./tools/pg.sh start
.venv/bin/python tools/session_live.py --targets fr
```

Runs publisher, translator and listener in one process against the real server.
The listener is the point: everything upstream can look healthy while the
audience hears nothing, and only a real subscriber proves otherwise.

### Live session results

Same 25.6s fixture through the real self-hosted SFU, one language, publisher
and listener and translator in one process.

| Configuration | dropped | chunks | p50 | p95 | status |
|---|---|---|---|---|---|
| emit 1.0s, window 8s | 88.5% | 1 | 13.2s | 13.2s | ended (idle) |
| emit 3.0s, window 6s | 0% | 4 | 2.76s | 4.78s | ended |

The first row violates `emit_interval > window_s * RTF`. Every pass cost more
than the interval, so audio arrived faster than it could ever be consumed and
the guard shed almost all of it. No size of buffer fixes a rate mismatch.

The second row satisfies it: nothing shed at all, peak lag 2.37s against a 3.0s
threshold.

Stage breakdown for the second row:

| Stage | Seconds |
|---|---|
| chunker (including STT) | 1.471 |
| translate | 0.341 |
| TTS first audio | 1.445 |
| publish | 0.001 |

TTS is now the second largest stage, at 1.445s against the 266 to 468ms
measured earlier for short phrases. The cause is chunk length: four chunks from
25.6s of speech means roughly 15 words each, and Piper renders a whole sentence
before releasing any of it. This is the coupling noted above, arriving as a
measurement. Reducing `max_words` is the next lever, and it trades against
translation quality.

The drift controller earned its place here. The French chain reached a playout
queue of 3.27s, the controller raised the speaking rate three times, and it
never had to skip a phrase. The +10.5% expansion is being actively held rather
than accumulating.

## Fixtures

Two, and the difference matters.

`fixtures/keynote.wav` is Piper-synthesised. Evenly paced, unaccented, no room
tone. Useful because it comes with exact ground truth, and misleading if it is
the only thing anything is tuned against.

`fixtures/jfk.wav` is a 45 second excerpt of the 1961 inaugural address, a US
Government work in the public domain, imported with `tools/import_audio.py`.
Real human delivery, a strong regional accent and 1961 recording conditions.
Whisper tiny mishears it in ways it never mishears the synthetic clip:
"forebears fought" becomes "forebearers for it", "the heirs of" becomes "the
areas of", "friend and foe alike" becomes "friend and full of life".

That gap is the point. Tune on the real one.

```bash
.venv/bin/python tools/import_audio.py --in speech.ogg --out fixtures/clip.wav --start 60 --duration 45
```

Decoding goes through PyAV, which arrives with faster-whisper, so no ffmpeg
binary is required.

## Audio bitrate

Measured on a real browser listener over WebRTC, same setup each time:

| Configuration | Measured | Per 500 listeners |
|---|---|---|
| default, nothing set | 98.7 kbps | 49 Mbps |
| 32 kbps cap, RED on | 66.9 kbps | 33 Mbps |
| 32 kbps cap, RED off | 32.4 kbps | 16 Mbps |

Left alone, LiveKit publishes a microphone source at a bitrate meant for
general audio, which is roughly double what these tracks need. The cap lands
exactly on target and RED costs almost precisely 2x.

RED is on by default: venue wifi under load drops packets, and without
redundancy a drop is an audible gap in someone's translation. Turn it off for a
venue with a known-good uplink and a lot of listeners.

## Two subscription bugs worth knowing

Both were silent, and both cost bandwidth rather than breaking anything.

The listener page sets `autoSubscribe: false` and subscribes to exactly one
track. The default pulls every published track, so on a five language event
each phone would download five audio streams and play one. Verified from the
browser: one inbound RTP stream with three tracks published.

`session/room.py` had the same problem in reverse. It connected with the
default `auto_subscribe=True`, so the translation service subscribed to its own
N language tracks, downloading its own output back from the SFU and decoding
it, on the machine that is already the bottleneck. The server log showed it
retrying those subscriptions every three seconds. Filtering the tracks after
subscribing treats the symptom; not subscribing is the fix.

## Choosing a Whisper model

Batch transcription of `fixtures/jfk.wav`, 45s of real 1961 speech, on the dev
Mac (2 core Haswell, int8):

| Model | Load | Infer | RTF | Min emit_interval for an 8s window |
|---|---|---|---|---|
| tiny | 5.3s | 2.7s | 0.06 | 0.5s |
| base | 13.2s | 4.7s | 0.11 | 0.8s |
| small | 32.2s | 17.5s | 0.39 | 3.1s |

The accuracy difference is not subtle. On one line of the source:

| Model | Output |
|---|---|
| truth | for which our forebears fought, are still at issue around the globe |
| tiny | for which our forebearers for it are still an issue around the globe |
| base | for which are four bears for it, are still an issue around the globe |
| small | for which our forebears fought, are still at issue around the globe |

`small` is correct. `base` is arguably worse than `tiny` here, so bigger is not
monotonically better and it is worth measuring rather than assuming.

The cost is the sustainability constraint. At RTF 0.39, `small` with an 8s
window needs `emit_interval` above 3.1s, which is latency the audience feels.
7s window and 4s interval leaves a workable margin on this hardware.

None of this transfers to the A4000, where the intended backend is a streaming
transducer that consumes each frame once and has no window to re-transcribe.

## Two harness bugs worth recording

Both silently discarded parts of a test while appearing to pass.

`tools/session_live.py` started the venue publisher before loading the STT
model. Whisper small takes 32 seconds to load, and the SFU does not hold audio
for a subscriber that has not arrived yet, so on a 45 second clip most of the
speech was gone before anything was listening. The model now loads first.

`tools/pipeline_smoke.py` synthesises every language inline, in the loop that
consumes STT hypotheses, so TTS stalls frame consumption and the guard sheds
the backlog. That accounted for a 23% drop rate that had nothing to do with the
pipeline: `session/pipeline.py` runs a worker per language and does not block
this way. The tool now says so, and latency figures should come from
`session_live.py`.

### tiny vs small in the live pipeline

Same clip (`fixtures/jfk.wav`), same harness (`tools/session_live.py`), same two
target languages.

| | tiny (emit 3.0, window 6.0) | small (emit 4.0, window 7.0) |
|---|---|---|
| audio dropped | **0%** (0 of 4870 frames) | 74.3% (3618 of 4868) |
| shed events | 0 | 18 |
| chunks produced | 13 | 4 |
| p50 latency (fr) | 3.88s | 12.53s |
| p95 latency (fr) | 5.38s | 14.48s |

`small` transcribes correctly and cannot keep up. `tiny` keeps up and
mistranscribes. On two cores there is no setting that gives both.

The batch benchmark predicted `small` would fit at a 4s emit interval, and it
was wrong by roughly 3x. That benchmark had the machine to itself; the pipeline
shares two cores with two Piper voices and two translation chains. The detector
caught it and said so:

    pass_cost_s 8.53   emit_interval_s 4.0   window_s 0.0

8.53 seconds for a pass on a near-empty buffer is not transcription cost, it is
contention. An isolated component benchmark cannot size a shared-CPU pipeline,
and the `emit_interval > window_s * RTF` rule only holds when RTF is measured
under the load the component will actually face.

### Arabic drifts harder than French

From the `tiny` run, where nothing was dropped and the drift controller had a
clean signal:

| Language | Peak playout queue | Speed-ups | Skips |
|---|---|---|---|
| French | 2.75s | 1 | 0 |
| Arabic | **5.98s** | 7 | 0 |

The skip threshold is 6.0s. Arabic came within 20ms of dropping a phrase, and
needed seven speed-ups against French's one. Arabic output is simply longer for
the same source.

Two consequences. Per-language drift thresholds are probably wrong as a single
global number. And on a five language event, Arabic is the chain that fails
first, so it is the one to watch and the one to load test.

## Per-language drift thresholds

A single global threshold was wrong for both languages we have measured. From
one clean 45 second run with nothing dropped:

| Language | Peak playout queue | Speed-ups | Skip threshold |
|---|---|---|---|
| French | 2.75s | 1 | 6.0s |
| Arabic | 5.98s | 7 | 6.0s |

French never came close. Arabic came within 20ms, on a 45 second clip. Over a
45 minute keynote it would skip repeatedly.

`LANGUAGE_POLICIES` in `session/drift.py` now holds per-language thresholds.
Resolution order is explicit `policies`, then that table, then the default.

Arabic starts correcting at a 1.0s queue rather than 1.5s. Starting sooner
lengthens the ramp to the ceiling and gives the controller more total
corrective capacity, which is the right lever when a language accumulates
faster. `max_speed` is deliberately unchanged: whether Arabic stays
intelligible at 1.3x is a question for a native speaker listening to the actual
voice, not something to infer from a queue depth.

The table is deliberately short. It holds only languages that have been
measured in a real session. Inventing thresholds for the rest would look like
data and would not be, so an unmeasured language gets the default and says so
in the log. Add an entry after a run, and record the peak queue depth and
speed-up count that justified it.

Hindi, Urdu and Malayalam are the likely next candidates: all tend to run
longer than English. Chinese tends to run shorter and may want a higher
threshold rather than a lower one.

The session summary now reports the thresholds that were in force alongside the
peaks, so a post mortem can see what the limits were and not just how close the
queue came to them.

## NLLB-200 as the GPU translation backend

One model for 200 languages, against Opus-MT's one model per pair. The gain is
concentrated exactly where Opus-MT is weakest.

Same sentence, "Revenue across the sector grew eleven percent last year":

| | Telugu | Hindi |
|---|---|---|
| Opus-MT | సెంటర్ అవతల పదకొండు శాతం పెరిగింది | सेक्टर पार पर **Ruue** पिछले साल… |
| | "center" transliterated, "revenue" dropped | untranslated Latin garbage |
| NLLB | ఈ రంగం లో **ఆదాయం** గత సంవత్సరం 11 శాతం పెరిగింది | इस क्षेत्र में **राजस्व** में… |
| | correct words throughout | correct |

For European languages the two are comparable, so nothing is lost there.

Hindi also gets the right idiom: NLLB says सुप्रभात for "good morning" where
Opus-MT said हर सुबह, which means "every morning".

### The cost, measured

| Backend | 5 languages, CPU |
|---|---|
| Opus-MT | ~300ms (4 languages) |
| NLLB-600M | **4485ms p50** |

15x slower, against a 2 second end to end budget. NLLB is a GPU backend. On CPU
it is for quality comparison only, and the pipeline will shed audio behind it.

Fetch it with `python tools/fetch_mt_models.py --nllb` (617MB, already int8).

One structural advantage carries to the GPU: a single model means the whole
fan-out is ONE batched call, with the source encoded once and every target
decoded together. Opus-MT needs N separate model invocations. That gap widens
as the language count grows.

### Credibility is per device, not per backend

`BackendSpec.credible_on` is a set of devices rather than a boolean. NLLB
cannot meet the budget on CPU and is comfortable on a GPU, so a fixed flag
would either bar it from the hardware it is built for or wave it through on
hardware it cannot serve.

Whisper is the other case: its set is empty. A sliding window model has a long
unstable tail on a GPU too, so no device makes it credible. That distinction —
too slow here, versus wrong everywhere — is worth keeping visible.

    capabilities("fastconformer", "nllb-200", "piper", device="cuda")  -> credible
    capabilities("fastconformer", "nllb-200", "piper", device="cpu")   -> not
    capabilities("faster-whisper", "opus-mt", "piper", device="cuda")  -> not

## How to test it

### Everything at once

```bash
./tools/demo.sh up
```

Starts Postgres, LiveKit, the join service and a looping translation session,
then prints a URL. Open it, tap a language, listen. The tap is required:
browsers refuse to start audio without a user gesture.

`TARGETS=fr,ar ./tools/demo.sh up` to change languages,
`AUDIO=fixtures/keynote.wav ./tools/demo.sh up` to change the source.
`./tools/demo.sh down` stops everything, `status` shows what is running.

Five languages is past what two cores can serve without shedding audio, so the
default is three.

### The test suite

```bash
.venv/bin/python -m pytest tests/ -q
```

Database and model tests skip themselves when Postgres or the models are
absent, so a bare checkout still runs.

### Your own audio

```bash
.venv/bin/python tools/import_audio.py --in talk.mp3 --out fixtures/talk.wav --start 60 --duration 45
AUDIO=fixtures/talk.wav ./tools/demo.sh up
```

Any format PyAV reads. Use a real recording: synthesised speech is evenly
paced and unaccented, and a pipeline tuned only on it is tuned for a case that
never occurs at a venue.

### Comparing translation backends

```bash
.venv/bin/python tools/compare_mt.py --audio fixtures/jfk.wav --targets te,hi,fr
```

Transcribes once, then runs identical chunks through every backend so
translation is the only variable, and writes a WAV per backend per language.
Latency is deliberately not measured here.

### Latency and drop rate

```bash
.venv/bin/python tools/session_live.py --audio fixtures/jfk.wav --targets fr,ar --realtime
```

The real pipeline against the real SFU, with a listener that records what
actually arrives. Do not use `pipeline_smoke.py` for this: it synthesises
inline and stalls its own source.

### Tuning the chunker

```bash
.venv/bin/python tools/chunker_replay.py --sweep
```

Sweeps stability and wait thresholds against a behaviour model. Compares
configurations against each other; the absolute numbers need real recorded
hypotheses.

## Translation routing

Neither backend is right for every language, so the default is to pick per
language. `adapters/mt_routing.py`.

| Languages | Backend | Why |
|---|---|---|
| fr, de, es, ar, zh, … | Opus-MT | comparable quality, 15x faster |
| hi, te, ta, ml, kn, bn, mr, ur | NLLB-200 | Opus-MT is unsafe here |

"Unsafe" is not an overstatement. Measured on the same transcript:

**Telugu, short input.** Opus-MT turned "the hand of God." into
`దేవుని చేతి. வெறுமென ఒక రాత్రి, ఒక రాత్రి, ఒక రాత్రి, ఒక నగలను...` — four words
in, fifteen words of Tamil-laced nonsense out. Spoken output ran 45.0s against
NLLB's 38.1s for the same source: seven extra seconds of an audience being read
garbage.

**Hindi, semantic inversion.** Opus-MT rendered "revolutionary beliefs" as
मूलतत्त्ववादी, which means FUNDAMENTALIST. Grammatical, confident, and the
opposite of what was said. NLLB gives क्रांतिकारी.

The failures cluster on SHORT input, and the phrase chunker produces short
chunks by design. Opus-MT's family models fail precisely on the shape this
architecture generates.

Tamil, Malayalam and Kannada were not individually measured. They share the
en-dra family model with Telugu, so the failure belongs to the model rather
than to Telugu, and routing them to Opus-MT would be assuming the best about a
model already caught hallucinating.

Each backend receives one call with its whole share, so NLLB still batches its
languages together, and the two run concurrently: the slower one sets the pace
rather than the sum of both. A backend that fails returns empty strings for its
own languages and leaves the other's alone.

`routes=None` means the defaults. `routes={}` means no exceptions, everything
to Opus-MT. Those are deliberately different, and writing the lookup as
`routes or DEFAULT_ROUTES` conflated them until a test caught it.

## Why the demo runs Whisper tiny, not small

`small` transcribes far better in batch. It does not survive the streaming
pipeline on two cores, and the reason is a genuine conflict rather than a
tuning failure.

Sustainability needs `emit_interval > window_s * RTF`. At RTF 0.39 under
contention, keeping `small` fed means shrinking the window. But Whisper's
accuracy depends on context, and a short window is exactly what removes it. The
two constraints pull opposite ways.

Measured on the same source, same three languages, comparable wall time:

| STT | window | Hindi chunks produced | Shed events |
|---|---|---|---|
| tiny | 6.0s | 50 | few |
| small (emit 9s) | 5.0s | **2** | **58** |

`small` at a 5 second window produced "believe for which our full" — no better
than tiny, having shed most of the audio to get there. The batch advantage does
not transfer.

Override anyway with `MODEL=small EMIT=9.0 WINDOW=5.0 ./tools/demo.sh up`. It
is worth seeing once.

None of this applies to the A4000, where the intended backend is a streaming
transducer that consumes each frame once and has no window to re-transcribe.

## Measuring transcript accuracy

`fixtures/holmes.txt` is the ground truth for `fixtures/holmes.wav`, taken from
the published 1891 text rather than from a model's output. Scoring one model
against a larger model's transcript measures agreement, not accuracy, and both
can be confidently wrong in the same place.

```bash
.venv/bin/python tools/score_stt.py --audio fixtures/holmes.wav --models tiny,base,small
.venv/bin/python tools/score_stt.py --no-batch --session <uuid>
```

Measured on 149 words of Sherlock Holmes narration:

| Run | WER | sub | del | ins |
|---|---|---|---|---|
| tiny, batch | 8.1% | 9 | 0 | 3 |
| base, batch | 4.0% | 6 | 0 | 0 |
| small, batch | 3.4% | 5 | 0 | 0 |
| **tiny, live streaming** | **14.8%** | 15 | 4 | 3 |

Streaming roughly doubles tiny's error rate. That gap is the sliding window:
batch sees the whole file, streaming sees six seconds. About one word in seven
is wrong in the live pipeline, and every downstream stage translates it
faithfully.

`small` at 3.4% is what the transcript could be. It cannot be reached on two
CPU cores, for the reasons above.

### Two ways this measurement went wrong first

Scoring a LOOPING session gave 298.7% WER with 432 insertions, because the
session held four passes of a single-pass reference and every repeat counted as
an insertion. The tool now scores one pass and says when it has truncated.

Scoring the first pass of a looping session then gave 83.2%, still wrong: the
session had started mid-clip, so the transcript was not aligned to the
reference. Only a single-pass run from the beginning gives a meaningful number.

Both readings looked like catastrophic pipeline failures and were neither.

## Listening to the original

The source language appears first in the picker, labelled "original audio".

It is a RELAY, not a translation. English does not go through STT, translation
and synthesis to come back out as English: that would add transcription errors,
a synthetic voice and several seconds of latency to audio that is already
perfect. The listener subscribes to the venue publisher's `source-audio` track
directly.

Three consequences worth knowing:

- **No latency.** It is SFU forwarding only, so it arrives ahead of every
  translation.
- **No extra cost.** The track is in the room whether anyone listens or not, so
  it is not billed as a language and does not appear in `target_languages`.
- **Real product value.** In a hall with poor acoustics, or for someone hard of
  hearing, the floor audio in earbuds beats anything downstream of it.

`RoomInspector` counts `source-audio` when reporting availability, so the
original is marked off air if the venue publisher has not connected yet.

Making this optional per event would need a column on `translation_sessions`.
It is always offered today, on the grounds that a live session by definition
has a source track. An event that does not want its floor audio redistributed
is a real case and is not handled.

## Two GPU backends, written but never run

`adapters/stt_qwen3.py` and `adapters/tts_chatterbox.py` are written against
the documented APIs and have never executed. Qwen3-ASR's streaming path needs
vLLM and Chatterbox needs torch and CUDA, so neither can be exercised on the
development machine at all. They are ready for the A4000; nothing about them is
verified until they have run there.

Both fail at construction rather than at session start if their package is
absent. Loading a model happens when a session begins, and by then an audience
is already in the room.

### Qwen3-ASR

Alibaba, January 2026. 0.6B and 1.7B, 52 languages, unified streaming and
offline inference with a 1 to 8 second attention window.

This targets the one problem no GPU fixes. Whisper's 30 second window is why
every streaming wrapper re-transcribes a sliding buffer, why this project
carries `emit_interval > window_s * RTF`, and why `small` shed 74% of the audio.
A model that generates incrementally from a KV cache removes that constraint
rather than loosening it.

What it costs: vLLM only, no timestamps in streaming mode, and 0.6B–1.7B
against FastConformer's 114M on a card already shared with translation and five
voices. Missing timestamps are survivable — `Hypothesis.words` is optional and
the chunker interpolates — but precision in the latency figures is a real loss
on a project whose argument is that measured beats estimated.

One detail worth testing rather than assuming: the model exposes
`unfixed_chunk_num` and `unfixed_token_num`, its own account of which trailing
tokens are still revisable. That is the chunker's LocalAgreement idea arrived
at from the other end. Measure whether the model's internal signal beats the
chunker's external one before running both.

### Chatterbox

Resemble AI, MIT. 21 languages, one cloned voice carried across all of them.

**Correction to a widely quoted figure.** "200ms to first sample" belongs to
Resemble's managed WebSocket service, not the open-source model. The
open-source API is `model.generate(text, language_id=...)`, returning one
complete waveform with no streaming method. Time to first audio therefore
equals full synthesis time for the phrase, from a larger model than Piper —
which already streams per sentence and was measured at 266ms for five words.
Chatterbox has to win on total synthesis time, not throughput.

Two further gaps. It covers no Telugu, Tamil or Malayalam, all of which Piper
has, so adopting it means per-language TTS routing exactly as the translation
stage carries. And it exposes no rate control, so `VoiceSpec.speed` does
nothing and the drift controller loses its cheap lever — leaving only skipping,
on a stage where Arabic already peaked within 20ms of the skip threshold.

What it buys is real: on a five language event, one cloned voice is the
difference between a single interpreter and five unrelated synthetic strangers.

### Order of work

Qwen3-ASR first. STT is the sole remaining ceiling, with every downstream stage
measured correct, so it is the larger win. Chatterbox improves a stage that
already works and is not the constraint.
