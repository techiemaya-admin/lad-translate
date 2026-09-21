-- Feedback on a transcript line: a thumbs, and what it should have been.
--
-- The same shape as the WhatsApp agent's AI Learnings, because that is what an
-- operator has already learned to use: a thumbs on each line, a "should have
-- been", a switch to turn a learning off without deleting it.
--
-- What it teaches is different and the difference is structural. There is no
-- prompt here - the recognisers and translators are fixed models - so a
-- thumbs-down can only teach the one thing this pipeline can act on: a word
-- rule in the corrections table, derived from the diff and applied to the next
-- phrase. correction_id links the two. It is NULL when the feedback could not
-- become a rule (a rephrase, an added word, a function word), and the row is
-- kept anyway: as an example for later, and as a per-language quality signal
-- that says which translator a venue can trust.
--
-- Applied with the target schema substituted for {schema}. See db/migrate.py.

CREATE SCHEMA IF NOT EXISTS {schema};

-- A learning can be switched off without being deleted, and the session's
-- reload path honours it. Added here rather than in 004 because this is the
-- release that needed it; every existing rule stays on.
ALTER TABLE {schema}.corrections
    ADD COLUMN IF NOT EXISTS is_active boolean NOT NULL DEFAULT true;

CREATE TABLE IF NOT EXISTS {schema}.transcript_feedback (
    feedback_id     bigserial PRIMARY KEY,
    tenant_id       uuid        NOT NULL,
    session_id      uuid        NOT NULL,
    chunk_id        integer     NOT NULL,

    -- The line the thumbs was given on: the source code for the speaker's own
    -- words, or one target code.
    language        text        NOT NULL,
    rating          text        NOT NULL,

    -- What the pipeline produced, captured at the moment of the thumbs, so a
    -- later replay of corrections over the transcript cannot make the
    -- feedback look like it was about something else.
    produced_text   text        NOT NULL,
    expected_text   text,
    note            text,

    -- The rule this feedback became, if it became one. ON DELETE SET NULL:
    -- deleting a rule from the corrections panel must not delete the
    -- operator's feedback, only unlink it.
    correction_id   bigint      REFERENCES {schema}.corrections (correction_id)
                                ON DELETE SET NULL,
    -- Why it did or did not become a rule, in a sentence, for the panel.
    learned_reason  text        NOT NULL DEFAULT '',

    is_active       boolean     NOT NULL DEFAULT true,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    created_by      text,

    CONSTRAINT transcript_feedback_session_fk
        FOREIGN KEY (session_id)
        REFERENCES {schema}.translation_sessions (session_id)
        ON DELETE CASCADE,
    CONSTRAINT transcript_feedback_rating_chk
        CHECK (rating IN ('like', 'dislike')),
    -- One thumbs per line per language. Changing your mind updates the row.
    CONSTRAINT transcript_feedback_one_per_line
        UNIQUE (session_id, chunk_id, language)
);

-- The panel's list and the quality stats both read by session.
CREATE INDEX IF NOT EXISTS transcript_feedback_by_session
    ON {schema}.transcript_feedback (tenant_id, session_id, created_at DESC);
