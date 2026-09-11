-- Hardware audio output: Dante channel mapping and IR channel assignment.
--
-- WebRTC is not the only way out of a session. A venue that already owns an
-- infrared interpretation system wants the same translated audio on physical
-- handsets, and the path to those handsets is almost always analog or Dante
-- into the IR transmitter's per-channel inputs. This is where the operator
-- says which language lands on which wire.
--
-- Two tables rather than one. A venue's Dante rig is stable across events, so
-- the device and its channel map are a profile that many sessions reuse; the
-- alternative -- channel mapping stored per session -- means re-patching in
-- software before every event, which is exactly the error-prone step this is
-- supposed to remove.
--
-- Applied with the target schema substituted for {schema}. See db/migrate.py.

CREATE SCHEMA IF NOT EXISTS {schema};

-- ---------------------------------------------------------------------------
-- audio_output_devices
-- ---------------------------------------------------------------------------
-- One row per physical output the venue has. Today that is a Dante Virtual
-- Soundcard instance; the kind column is here so an AES67 sender or a plain
-- multichannel ASIO/Core Audio interface is a new value rather than a new
-- table.

CREATE TABLE IF NOT EXISTS {schema}.audio_output_devices (
    device_id      uuid PRIMARY KEY,
    tenant_id      uuid        NOT NULL,
    name           text        NOT NULL,
    kind           text        NOT NULL DEFAULT 'dante-vsc',

    -- The host audio device to open, exactly as the operating system names it
    -- ("Dante Virtual Soundcard"). Opaque here: this schema does not know what
    -- a Core Audio device is, and the sink that opens it is responsible for
    -- reporting a name it cannot find.
    device_name    text        NOT NULL,

    -- Channels the device exposes. DVS licences are 16x16 or 64x64, so this is
    -- a property of the licence and not something to guess.
    channel_count  smallint    NOT NULL,

    -- Dante runs the card at a fixed rate, 48k or 44.1k, set in Dante
    -- Controller. Piper renders at 22050, so the sink resamples to whatever is
    -- recorded here. Storing it means the mismatch is caught when the profile
    -- is saved rather than as a pitch-shifted channel at the event.
    sample_rate    integer     NOT NULL DEFAULT 48000,

    enabled        boolean     NOT NULL DEFAULT true,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT audio_output_devices_kind_check
        CHECK (kind IN ('dante-vsc', 'aes67', 'coreaudio', 'asio', 'alsa')),
    CONSTRAINT audio_output_devices_channels_sane
        CHECK (channel_count BETWEEN 1 AND 64),
    -- The rates Dante and the pro-audio world actually run at. A typo here
    -- would otherwise surface as a resampler that quietly does the wrong thing.
    CONSTRAINT audio_output_devices_rate_check
        CHECK (sample_rate IN (44100, 48000, 88200, 96000)),
    CONSTRAINT audio_output_devices_name_not_blank
        CHECK (length(btrim(name)) > 0),
    -- Operators pick a device by name in the portal, so two devices sharing one
    -- is an ambiguity the audience pays for.
    CONSTRAINT audio_output_devices_tenant_name_unique
        UNIQUE (tenant_id, name)
);

CREATE INDEX IF NOT EXISTS audio_output_devices_tenant_idx
    ON {schema}.audio_output_devices (tenant_id, name);

-- ---------------------------------------------------------------------------
-- audio_output_channels
-- ---------------------------------------------------------------------------
-- The patch itself: one row per occupied channel on a device.

CREATE TABLE IF NOT EXISTS {schema}.audio_output_channels (
    channel_id   uuid PRIMARY KEY,
    device_id    uuid        NOT NULL,
    tenant_id    uuid        NOT NULL,

    -- BCP-47, matching config.LanguageTarget.code. The source language is a
    -- legitimate value: venues routinely put the floor feed on channel 1 so
    -- the booth and the recorder get the original alongside the translations.
    language     text        NOT NULL,

    -- 1-based position on the device. Bounded against the parent's
    -- channel_count in db/outputs.py, not here: a CHECK cannot read another
    -- table's column, and a trigger to do it would hide the rule from anyone
    -- reading this file.
    channel      smallint    NOT NULL,

    -- What the number on the handset means. The IR transmitter's inputs are
    -- patched from Dante outputs by hand, and the audience is told "channel 3
    -- is French" on signage printed days earlier -- so the mapping the audience
    -- sees is not necessarily the Dante channel index, and has to be recorded
    -- separately or the signage and the rig drift apart. Null when this channel
    -- does not feed the IR system.
    ir_channel   smallint,

    -- Printed on signage and shown in the portal. Falls back to the language's
    -- own name when blank.
    label        text        NOT NULL DEFAULT '',

    -- Per-channel trim. IR transmitters expect a hotter input than a line-level
    -- recorder, so the same language often needs two levels on two channels.
    gain_db      real        NOT NULL DEFAULT 0.0,

    enabled      boolean     NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT audio_output_channels_device_fk
        FOREIGN KEY (device_id)
        REFERENCES {schema}.audio_output_devices (device_id)
        ON DELETE CASCADE,
    CONSTRAINT audio_output_channels_channel_positive
        CHECK (channel BETWEEN 1 AND 64),
    CONSTRAINT audio_output_channels_ir_positive
        CHECK (ir_channel IS NULL OR ir_channel BETWEEN 1 AND 99),
    -- Trim, not a mix desk. Beyond this the gain structure upstream is wrong
    -- and boosting here only raises the noise floor with it.
    CONSTRAINT audio_output_channels_gain_range
        CHECK (gain_db BETWEEN -60.0 AND 12.0),
    CONSTRAINT audio_output_channels_language_not_blank
        CHECK (length(btrim(language)) > 0),

    -- A wire carries one signal. One language may occupy several channels --
    -- French to the IR transmitter and again to the recorder is normal -- but
    -- two languages on one channel is two people talking at once.
    CONSTRAINT audio_output_channels_device_channel_unique
        UNIQUE (device_id, channel)
);

CREATE INDEX IF NOT EXISTS audio_output_channels_device_idx
    ON {schema}.audio_output_channels (tenant_id, device_id, channel);

CREATE INDEX IF NOT EXISTS audio_output_channels_language_idx
    ON {schema}.audio_output_channels (tenant_id, device_id, language)
    WHERE enabled;

-- Two handsets showing channel 3 for different languages is the same collision
-- as two languages on one wire, one step further down the chain. Partial, so
-- the channels that do not feed IR are exempt rather than forced to collide on
-- null.
CREATE UNIQUE INDEX IF NOT EXISTS audio_output_channels_device_ir_unique
    ON {schema}.audio_output_channels (device_id, ir_channel)
    WHERE ir_channel IS NOT NULL;
