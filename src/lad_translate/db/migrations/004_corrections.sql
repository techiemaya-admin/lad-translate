-- Operator corrections: "that word is wrong, here is the right one".
--
-- A recogniser gets a name wrong the same way every time, so a correction is
-- worth far more as a rule that changes the next phrase than as an edit to the
-- last one. The session reloads this table while it runs, which is the whole
-- point: a name discovered wrong in the first minute of a talk is fixed for
-- the remaining fifty-nine without restarting anything. Restarting a live
-- session to pick up a glossary would cost the audience the audio in flight.
--
-- Rules live per TENANT, not per session. The names a venue gets wrong are its
-- own - its products, its speakers, its city - and re-typing them before every
-- event is the error-prone step this exists to remove. `room_name` narrows a
-- rule to one room when a term is genuinely event-specific.
--
-- `language` is either the session's SOURCE code, in which case the phrase is
-- corrected before translation and every target benefits at once, or a single
-- TARGET code, for when the English was right and one language got it wrong.
--
-- Applied with the target schema substituted for {schema}. See db/migrate.py.

CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.corrections (
    correction_id  bigserial PRIMARY KEY,
    tenant_id      uuid        NOT NULL,

    -- Source code (fixes every language at once) or one target code.
    language       text        NOT NULL,

    -- Named wrong_text/right_text rather than wrong/right: RIGHT is a
    -- reserved word in SQL and an unquoted column of that name is a syntax
    -- error in the first query someone writes by hand.
    wrong_text     text        NOT NULL,
    right_text     text        NOT NULL,

    -- NULL means every room in the tenant, which is what a product name wants.
    room_name      text,

    created_at     timestamptz NOT NULL DEFAULT now(),
    created_by     text,

    CONSTRAINT corrections_wrong_not_blank CHECK (btrim(wrong_text) <> ''),
    CONSTRAINT corrections_language_not_blank CHECK (btrim(language) <> '')
);

-- One rule per phrase per language per scope. Case-insensitive because the
-- matcher is: without this, "Adler" and "adler" are two rows that quietly
-- disagree about the answer. coalesce() because NULL never equals NULL, so a
-- plain unique constraint would let a tenant-wide rule be added twice.
CREATE UNIQUE INDEX IF NOT EXISTS corrections_unique_rule
    ON {schema}.corrections (tenant_id, language, lower(btrim(wrong_text)), coalesce(room_name, ''));

-- The session's reload path: every rule for a tenant that applies to a room.
CREATE INDEX IF NOT EXISTS corrections_by_tenant_room
    ON {schema}.corrections (tenant_id, room_name);
