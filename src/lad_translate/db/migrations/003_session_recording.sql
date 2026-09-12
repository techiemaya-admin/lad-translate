-- Whether a session is being recorded, for the pages the speaker and the
-- audience see. The session process flips it when a recording starts or
-- stops; the join API reads it. It lives on the row rather than in the
-- process because the speaker's page has no other way to find out, and a
-- recording nobody was told about is the kind of thing a venue is sued over.
ALTER TABLE {schema}.translation_sessions
    ADD COLUMN IF NOT EXISTS recording boolean NOT NULL DEFAULT false;
