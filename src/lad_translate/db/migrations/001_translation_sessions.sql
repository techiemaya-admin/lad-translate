-- Translation session tables.
--
-- Single database, one schema per tenant, every row carrying tenant_id. Same
-- shape as VOAG, with one difference: the schema name is supplied by the
-- caller at run time, never frozen into a module constant. VOAG's
-- db/schema_constants.py resolves DB_SCHEMA once at import and bakes every
-- table name from it, which is why one of its containers cannot serve two
-- tenants.
--
-- Applied with the target schema substituted for {schema}. See db/migrate.py.

CREATE SCHEMA IF NOT EXISTS {schema};

-- ---------------------------------------------------------------------------
-- translation_sessions
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS {schema}.translation_sessions (
    session_id        uuid PRIMARY KEY,
    tenant_id         uuid        NOT NULL,
    event_name        text        NOT NULL,
    room_name         text        NOT NULL,
    source_language   text        NOT NULL,
    target_languages  text[]      NOT NULL,
    status            text        NOT NULL DEFAULT 'starting',
    started_at        timestamptz NOT NULL DEFAULT now(),
    ended_at          timestamptz,
    failure_reason    text,

    -- Billing is duration multiplied by active language count. It is settled
    -- here at close rather than recomputed later from target_languages,
    -- because a config change after the fact must never alter an invoice that
    -- has already been raised.
    billed_seconds          integer,
    billed_language_count   smallint,

    -- Recorded so a session run on a backend that cannot meet the SLO is never
    -- mistaken for one that can. See adapters/registry.py.
    latency_credible  boolean     NOT NULL DEFAULT false,
    backend_stt       text,
    backend_mt        text,
    backend_tts       text,

    CONSTRAINT translation_sessions_status_check
        CHECK (status IN ('starting', 'live', 'ended', 'failed')),
    CONSTRAINT translation_sessions_targets_not_empty
        CHECK (cardinality(target_languages) > 0),
    CONSTRAINT translation_sessions_ended_after_started
        CHECK (ended_at IS NULL OR ended_at >= started_at),
    -- A failed session must say why. Silent failures are how a venue post
    -- mortem turns into guesswork.
    CONSTRAINT translation_sessions_failure_has_reason
        CHECK (status <> 'failed' OR failure_reason IS NOT NULL)
);

-- Every query is scoped by tenant_id, so it leads every index.
CREATE INDEX IF NOT EXISTS translation_sessions_tenant_started_idx
    ON {schema}.translation_sessions (tenant_id, started_at DESC);

CREATE INDEX IF NOT EXISTS translation_sessions_tenant_live_idx
    ON {schema}.translation_sessions (tenant_id)
    WHERE status IN ('starting', 'live');

-- ---------------------------------------------------------------------------
-- session_listeners
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS {schema}.session_listeners (
    listener_id  uuid PRIMARY KEY,
    session_id   uuid        NOT NULL,
    tenant_id    uuid        NOT NULL,
    language     text        NOT NULL,
    joined_at    timestamptz NOT NULL DEFAULT now(),
    left_at      timestamptz,

    CONSTRAINT session_listeners_session_fk
        FOREIGN KEY (session_id)
        REFERENCES {schema}.translation_sessions (session_id)
        ON DELETE CASCADE,
    CONSTRAINT session_listeners_left_after_joined
        CHECK (left_at IS NULL OR left_at >= joined_at)
);

CREATE INDEX IF NOT EXISTS session_listeners_session_idx
    ON {schema}.session_listeners (tenant_id, session_id);

-- Concurrent listener count per language, for the room load picture.
CREATE INDEX IF NOT EXISTS session_listeners_active_idx
    ON {schema}.session_listeners (tenant_id, session_id, language)
    WHERE left_at IS NULL;

-- ---------------------------------------------------------------------------
-- session_transcripts
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS {schema}.session_transcripts (
    transcript_id    bigserial PRIMARY KEY,
    session_id       uuid        NOT NULL,
    tenant_id        uuid        NOT NULL,
    chunk_id         integer     NOT NULL,
    language         text        NOT NULL,
    source_text      text        NOT NULL,
    translated_text  text,

    -- Position in the source audio, not wall clock. Survives restarts and
    -- lines a transcript up with a recording.
    t_audio_start    double precision NOT NULL,
    t_audio_end      double precision NOT NULL,

    -- Glass to glass for this chunk in this language, seconds. Null when the
    -- chunk never reached the room.
    latency_s        double precision,

    commit_reason    text,
    revised          boolean NOT NULL DEFAULT false,
    created_at       timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT session_transcripts_session_fk
        FOREIGN KEY (session_id)
        REFERENCES {schema}.translation_sessions (session_id)
        ON DELETE CASCADE,
    CONSTRAINT session_transcripts_audio_span
        CHECK (t_audio_end >= t_audio_start),
    -- One row per chunk per language.
    CONSTRAINT session_transcripts_chunk_language_unique
        UNIQUE (session_id, chunk_id, language)
);

CREATE INDEX IF NOT EXISTS session_transcripts_session_chunk_idx
    ON {schema}.session_transcripts (tenant_id, session_id, chunk_id);
