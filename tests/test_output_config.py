"""
Output channel and device validation.

Pure config, no database and no hardware. These are the invariants that stop a
venue discovering its patch is wrong during a talk, so they are tested where
they are cheapest to run.
"""

from __future__ import annotations

import pytest

from lad_translate.config import OutputChannel, OutputDevice


def device(**over) -> OutputDevice:
    base = dict(
        device_id="d1",
        name="Main hall DVS",
        device_name="Dante Virtual Soundcard",
        channel_count=16,
    )
    base.update(over)
    return OutputDevice(**base)


# --- channels ---------------------------------------------------------------


def test_a_channel_needs_a_language():
    with pytest.raises(ValueError, match="needs a language"):
        OutputChannel(language="  ", channel=1)


def test_channels_are_one_based():
    """Zero is the array index, not the channel. Dante counts from one."""
    with pytest.raises(ValueError, match="1-based"):
        OutputChannel(language="fr", channel=0)


@pytest.mark.parametrize("ir", [0, 100])
def test_ir_channel_stays_in_handset_range(ir):
    with pytest.raises(ValueError, match="outside 1-99"):
        OutputChannel(language="fr", channel=1, ir_channel=ir)


def test_ir_channel_is_optional():
    """A channel feeding a recorder rather than the IR rig has no handset number."""
    assert OutputChannel(language="fr", channel=1).ir_channel is None


@pytest.mark.parametrize("gain", [-61.0, 12.5])
def test_gain_is_trim_not_a_mix_desk(gain):
    with pytest.raises(ValueError, match="outside -60"):
        OutputChannel(language="fr", channel=1, gain_db=gain)


# --- devices ----------------------------------------------------------------


def test_device_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown device kind"):
        device(kind="firewire")


def test_device_rejects_a_rate_dante_does_not_run():
    with pytest.raises(ValueError, match="sample rate"):
        device(sample_rate=22050)


def test_device_needs_a_host_device_to_open():
    with pytest.raises(ValueError, match="host device name"):
        device(device_name="")


def test_channel_beyond_the_device_is_refused():
    """The bound a CHECK constraint cannot express: it spans two tables."""
    with pytest.raises(ValueError, match="beyond the 16 channels"):
        device(channels=(OutputChannel(language="fr", channel=17),))


def test_two_languages_cannot_share_a_wire():
    with pytest.raises(ValueError, match="share a channel"):
        device(
            channels=(
                OutputChannel(language="fr", channel=3),
                OutputChannel(language="ar", channel=3),
            )
        )


def test_two_languages_cannot_share_an_ir_channel():
    """The audience would tune to one number and hear either."""
    with pytest.raises(ValueError, match="share an IR channel"):
        device(
            channels=(
                OutputChannel(language="fr", channel=1, ir_channel=2),
                OutputChannel(language="ar", channel=2, ir_channel=2),
            )
        )


def test_one_language_may_hold_several_channels():
    """French to the IR transmitter and again to the recorder is ordinary."""
    d = device(
        channels=(
            OutputChannel(language="fr", channel=1, ir_channel=1),
            OutputChannel(language="fr", channel=9, gain_db=-6.0),
        )
    )
    assert d.languages == ["fr"]
    assert [c.channel for c in d.channels_for("fr")] == [1, 9]


def test_source_language_is_a_legitimate_channel():
    """Venues put the floor feed on a channel for the booth and the recorder."""
    d = device(
        channels=(
            OutputChannel(language="en", channel=1, label="Floor"),
            OutputChannel(language="fr", channel=2, ir_channel=2),
        )
    )
    assert d.languages == ["en", "fr"]


def test_languages_are_listed_in_channel_order_not_insertion_order():
    d = device(
        channels=(
            OutputChannel(language="hi", channel=5),
            OutputChannel(language="fr", channel=2),
        )
    )
    assert d.languages == ["fr", "hi"]


def test_disabled_channels_are_not_carried():
    """Muting a channel in the portal must actually take it out of the map."""
    d = device(
        channels=(
            OutputChannel(language="fr", channel=1),
            OutputChannel(language="ar", channel=2, enabled=False),
        )
    )
    assert d.languages == ["fr"]
    assert d.channels_for("ar") == []
