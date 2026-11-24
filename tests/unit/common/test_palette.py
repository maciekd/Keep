"""common/Palette.py: the colours as plain numbers, and what a terminal of 256 gets instead."""
import pytest

from common import Palette


@pytest.mark.unit
@pytest.mark.parametrize('rgb, index', [((0, 0, 0), 16), ((255, 255, 255), 231), ((255, 0, 0), 196), ((0, 0, 255), 21),
                                        ((95, 135, 175), 67), ((100, 130, 180), 67), ((40, 40, 40), 16), ((50, 50, 50), 59)],
                         ids=['black', 'white', 'red', 'blue', 'a colour of the cube', 'one beside it', 'nearer black', 'nearer the first step'])
def test_the_nearest_of_the_256_is_the_cube_colour_closest_on_each_channel(rgb, index):
    """Contract with xterm's 256-colour table: 16 + 36·r + 6·g + b over the levels 0, 95, 135 … 255."""
    assert Palette.nearest_256(rgb) == index


@pytest.mark.unit
def test_every_accent_lands_on_a_colour_of_its_own():
    accents = [Palette.GREEN, Palette.RED, Palette.YELLOW, Palette.CYAN]
    assert len({Palette.nearest_256(rgb) for rgb in accents}) == len(accents)


@pytest.mark.unit
def test_a_mix_runs_from_the_one_colour_to_the_other_and_no_further():
    deep, accent = Palette.DEEP[Palette.CYAN], Palette.CYAN
    assert Palette.mix(deep, accent, 0) == deep and Palette.mix(deep, accent, 1) == accent
    assert Palette.mix(deep, accent, -3) == deep and Palette.mix(deep, accent, 7) == accent
    assert Palette.mix((0, 100, 200), (100, 200, 0), 0.5) == (50, 150, 100)


@pytest.mark.unit
def test_the_consoles_theme_shades_every_accent_from_the_palettes_companion():
    """interfaces/console/Theme.py is built from the palette: an accent added there without its
    deep companion here would fade up from the panel's ground."""
    from interfaces.console import Theme
    for accent, companion in Palette.DEEP.items():
        shade = Theme.deep(Theme.Color(*accent))
        assert (shade.r, shade.g, shade.b) == companion
