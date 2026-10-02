"""The palette's two helpers: every tab derives its tints through them."""
import pytest

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import Color

A, B = Color(10, 200, 30), Color(110, 0, 130)


@pytest.mark.unit
def test_mix_walks_from_one_colour_to_the_other_and_never_past_either():
    """A meter hands in its fill ratio, which runs over 1 on an overloaded gauge: the result
    must stay a colour a terminal can show."""
    assert Theme.mix(A, B, 0) == A
    assert Theme.mix(A, B, 1) == B
    assert Theme.mix(A, B, 0.5) == Color(60, 100, 80)
    assert Theme.mix(A, B, 1.7) == B
    assert Theme.mix(A, B, -0.2) == A


@pytest.mark.unit
def test_every_accent_has_a_deep_companion_and_any_other_colour_fades_up_from_the_panel():
    for accent in Theme.DEEP:
        assert Theme.deep(accent) != accent
    stranger = Color(200, 100, 50)
    faded = Theme.deep(stranger)
    for channel in "rgb":
        low, high = sorted((getattr(Theme.COLOR_PANEL_BG, channel), getattr(stranger, channel)))
        assert low <= getattr(faded, channel) <= high
    assert faded not in (stranger, Theme.COLOR_PANEL_BG)
