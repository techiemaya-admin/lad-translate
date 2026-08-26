import pytest

from lad_translate.config import (
    BackendSelection,
    LanguageTarget,
    SessionConfig,
    SessionLimits,
    TenantContext,
)


def tenant(**over) -> TenantContext:
    return TenantContext(
        **{"tenant_id": "t-1", "database_url": "postgres://x/y", "schema": "tenant_1", **over}
    )


def session(**over) -> SessionConfig:
    base = dict(
        session_id="s-1",
        tenant=tenant(),
        room_name="room-1",
        event_name="Summit",
        source_language="en",
        targets=[LanguageTarget("fr", "fr_FR-siwis-medium"), LanguageTarget("ar", "ar_JO-kareem-medium")],
    )
    return SessionConfig(**{**base, **over})


# --- tenancy ----------------------------------------------------------------


@pytest.mark.parametrize("field", ["tenant_id", "database_url", "schema"])
def test_tenant_context_refuses_to_be_partially_resolved(field):
    """
    There is no default tenant and no fallback schema.

    VOAG's db/schema_constants.py:18 defaults to lad_dev at import time, so an
    unresolved tenant writes into the shared control plane instead of failing.
    """
    with pytest.raises(ValueError):
        tenant(**{field: ""})


def test_tenant_context_is_immutable():
    with pytest.raises(Exception):
        tenant().tenant_id = "t-2"  # type: ignore[misc]


# --- session validation -----------------------------------------------------


def test_session_needs_at_least_one_target():
    with pytest.raises(ValueError, match="at least one target"):
        session(targets=[])


def test_duplicate_targets_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        session(targets=[LanguageTarget("fr", "a"), LanguageTarget("fr", "b")])


def test_source_language_cannot_also_be_a_target():
    """Otherwise we publish a track that just echoes the speaker."""
    with pytest.raises(ValueError, match="also a target"):
        session(targets=[LanguageTarget("en", "en_GB-alba-medium")])


def test_language_cap_is_enforced_not_advisory():
    """
    Fan-out cost scales with language count, not audience size. Eight is where
    a 16GB A4000 starts losing headroom.
    """
    many = [LanguageTarget(c, f"voice-{c}") for c in ("fr", "de", "es", "ar", "zh")]
    with pytest.raises(ValueError, match="exceeds the cap"):
        session(targets=many, limits=SessionLimits(max_languages=4))


def test_language_cap_can_be_raised_deliberately():
    many = [LanguageTarget(c, f"voice-{c}") for c in ("fr", "de", "es", "ar", "zh")]
    assert len(session(targets=many, limits=SessionLimits(max_languages=8)).targets) == 5


# --- accessors --------------------------------------------------------------


def test_voice_lookup():
    cfg = session()
    assert cfg.voice_for("fr") == "fr_FR-siwis-medium"
    with pytest.raises(KeyError):
        cfg.voice_for("de")


def test_target_codes_preserve_order():
    assert session().target_codes == ["fr", "ar"]


def test_limits_have_a_duration_cap_by_default():
    """VOAG has none, and event sessions run long."""
    limits = SessionLimits()
    assert limits.max_duration_s > 0
    assert limits.max_idle_s > 0


def test_backends_default_to_the_local_cpu_set():
    b = BackendSelection()
    assert (b.stt, b.mt, b.tts) == ("faster-whisper", "opus-mt", "piper")
