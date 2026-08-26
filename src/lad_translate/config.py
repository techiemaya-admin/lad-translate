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
