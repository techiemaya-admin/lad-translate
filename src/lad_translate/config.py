"""
Session configuration.

Deliberately explicit. There is no module-level constant that freezes a schema
or a database at import time, because that is what stops a single VOAG
container serving two tenants (db/schema_constants.py:18). Tenant context is
resolved once at session start and carried on the config object.

Backend choice is a string resolved through adapters/registry.py. The pipeline
never sees a backend name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .chunker.types import ChunkerConfig


@dataclass(frozen=True, slots=True)
class TenantContext:
    """
    Resolved once, at session start, then passed explicitly everywhere.

    There is no fallback and no default. A missing tenant is an error, not a
    reason to write into a shared schema.
    """

    tenant_id: str
    database_url: str
    schema: str

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id is required; there is no default tenant")
        if not self.database_url:
            raise ValueError(f"no database_url resolved for tenant {self.tenant_id}")
        if not self.schema:
            raise ValueError(f"no schema resolved for tenant {self.tenant_id}")


@dataclass(frozen=True, slots=True)
class LanguageTarget:
    """One output language: what to translate to, and which voice speaks it."""

    code: str
    """BCP-47, for example 'ar', 'fr', 'hi'."""

    voice_id: str
    """Backend-specific voice identifier. Opaque to the pipeline."""

    label: str = ""
    """What the audience sees in the language picker."""

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("language target needs a code")


@dataclass(slots=True)
class BackendSelection:
    """Which adapters to build. Resolved through adapters/registry.py."""

    stt: str = field(default_factory=lambda: os.getenv("STT_BACKEND", "faster-whisper"))
    mt: str = field(default_factory=lambda: os.getenv("MT_BACKEND", "opus-mt"))
    tts: str = field(default_factory=lambda: os.getenv("TTS_BACKEND", "piper"))

    stt_options: dict[str, str] = field(default_factory=dict)
    mt_options: dict[str, str] = field(default_factory=dict)
    tts_options: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SessionLimits:
    """
    Hard caps on a session.

    VOAG has none of these, and event sessions run long. A runaway session with
    five language chains burns GPU and storage with nothing to stop it. These
    are enforced by the pipeline, not advisory.
    """

    max_duration_s: float = 4 * 60 * 60
    """Wall clock ceiling. Sessions end at this point regardless of state."""

    max_languages: int = 8
    """
    Fan-out ceiling. Cost and GPU load scale with this, not with audience size.
    Eight is the point where a 16GB A4000 starts losing headroom.
    """

    max_idle_s: float = 15 * 60
    """End the session after this long with no source audio."""

    warn_at_fraction: float = 0.9
    """Log a warning once this fraction of any limit is reached."""


DEVICE_KINDS = ("dante-vsc", "aes67", "coreaudio", "asio", "alsa")
"""Output device kinds, matching the CHECK in migrations/002_audio_outputs.sql."""

DEVICE_SAMPLE_RATES = (44100, 48000, 88200, 96000)
"""Rates Dante and the pro-audio world run at. Mirrors the same CHECK."""


@dataclass(frozen=True, slots=True)
class OutputChannel:
    """
    One language on one physical channel of an output device.

    A language may hold several channels -- the same French feeding the IR
    transmitter and a recorder is ordinary -- but a channel carries exactly one
    language, because a wire carries one signal.
    """

    language: str
    """BCP-47, matching LanguageTarget.code. The source language is allowed:
    venues put the floor feed on a channel for the booth and the recorder."""

    channel: int
    """1-based position on the device."""

    ir_channel: int | None = None
    """
    The number the audience's handset shows, when this channel feeds the IR
    transmitter. Recorded separately from `channel` because the transmitter's
    inputs are patched by hand and the signage was printed days earlier; tying
    the two together is how the rig and the signage drift apart.
    """

    label: str = ""
    """Shown in the portal and printed on signage. Blank falls back to the
    language's own name."""

    gain_db: float = 0.0
    """Per-channel trim. An IR transmitter wants a hotter feed than a
    recorder, so one language often needs two levels on two channels."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("an output channel needs a language")
        if self.channel < 1:
            raise ValueError(
                f"channel numbers are 1-based; got {self.channel} for {self.language!r}"
            )
        if self.ir_channel is not None and not 1 <= self.ir_channel <= 99:
            raise ValueError(
                f"IR channel {self.ir_channel} for {self.language!r} is outside 1-99"
            )
        if not -60.0 <= self.gain_db <= 12.0:
            raise ValueError(
                f"gain {self.gain_db}dB for {self.language!r} is outside -60..+12; "
                "beyond that the gain structure upstream is wrong and this only "
                "raises the noise floor with it"
            )


@dataclass(frozen=True, slots=True)
class OutputDevice:
    """
    A venue's hardware output and its channel map.

    Held as a profile rather than per session because a venue's rig is stable
    across events. Re-patching in software before every event is exactly the
    error-prone step this removes.
    """

    device_id: str
    name: str
    device_name: str
    """The host audio device to open, as the OS names it ("Dante Virtual
    Soundcard"). Opaque here: the sink that opens it reports a name it cannot
    find."""

    channel_count: int
    kind: str = "dante-vsc"
    sample_rate: int = 48000
    """Dante fixes the card's rate in Dante Controller. Piper renders at 22050,
    so the sink resamples to this; storing it catches the mismatch when the
    profile is saved rather than as a pitch-shifted channel at the event."""

    enabled: bool = True
    channels: tuple[OutputChannel, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an output device needs a name")
        if not self.device_name.strip():
            raise ValueError(f"device {self.name!r} needs a host device name to open")
        if self.kind not in DEVICE_KINDS:
            raise ValueError(f"unknown device kind {self.kind!r}; one of {DEVICE_KINDS}")
        if self.sample_rate not in DEVICE_SAMPLE_RATES:
            raise ValueError(
                f"sample rate {self.sample_rate} is not one of {DEVICE_SAMPLE_RATES}"
            )
        if not 1 <= self.channel_count <= 64:
            raise ValueError(
                f"device {self.name!r} claims {self.channel_count} channels; "
                "DVS licences are 16x16 or 64x64"
            )

        # The bound a CHECK constraint cannot express, because channel_count
        # lives on this row and the channel lives on another table's.
        for channel in self.channels:
            if channel.channel > self.channel_count:
                raise ValueError(
                    f"channel {channel.channel} ({channel.language}) is beyond the "
                    f"{self.channel_count} channels {self.name!r} exposes"
                )

        taken = [c.channel for c in self.channels]
        if len(taken) != len(set(taken)):
            raise ValueError(f"two languages share a channel on {self.name!r}: {sorted(taken)}")

        ir = [c.ir_channel for c in self.channels if c.ir_channel is not None]
        if len(ir) != len(set(ir)):
            raise ValueError(
                f"two languages share an IR channel on {self.name!r}: {sorted(ir)}; "
                "the audience would tune to one number and hear either"
            )

    @property
    def languages(self) -> list[str]:
        """Distinct languages this device carries, in channel order."""
        seen: list[str] = []
        for channel in sorted(self.channels, key=lambda c: c.channel):
            if channel.enabled and channel.language not in seen:
                seen.append(channel.language)
        return seen

    def channels_for(self, language: str) -> list[OutputChannel]:
        """Every enabled channel carrying one language, in channel order."""
        return sorted(
            (c for c in self.channels if c.enabled and c.language == language),
            key=lambda c: c.channel,
        )


@dataclass(slots=True)
class SessionConfig:
    """Everything one translation session needs."""

    session_id: str
    tenant: TenantContext
    room_name: str
    event_name: str
    source_language: str
    targets: list[LanguageTarget]

    backends: BackendSelection = field(default_factory=BackendSelection)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    limits: SessionLimits = field(default_factory=SessionLimits)

    slo_seconds: float = 2.0
    """
    Glass to glass target, measured from the END of a source phrase to the
    START of its translated audio.

    Measured from the start of a phrase this target is unreachable, because a
    four second sentence cannot be translated before it has been spoken. The
    definition is part of the config so nobody has to guess which one the
    dashboard means.
    """

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("a session needs at least one target language")
        if len(self.targets) > self.limits.max_languages:
            raise ValueError(
                f"{len(self.targets)} target languages exceeds the cap of "
                f"{self.limits.max_languages}; raise SessionLimits.max_languages "
                "deliberately, after checking GPU headroom"
            )
        codes = [t.code for t in self.targets]
        if len(codes) != len(set(codes)):
            raise ValueError(f"duplicate target languages: {codes}")
        if self.source_language in codes:
            raise ValueError(
                f"source language {self.source_language!r} is also a target; "
                "that would publish a track that just echoes the speaker"
            )

    @property
    def target_codes(self) -> list[str]:
        return [t.code for t in self.targets]

    def voice_for(self, code: str) -> str:
        for target in self.targets:
            if target.code == code:
                return target.voice_id
        raise KeyError(f"no target language configured for {code!r}")
