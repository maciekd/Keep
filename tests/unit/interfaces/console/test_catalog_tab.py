import pytest

from assistant.integration import bootstrap
from assistant.integration.Integration import IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor, UnavailableIntegration
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs import CatalogTab as module
from assistant.tools.WebAccess import WebAccess
from interfaces.console.tabs.CatalogTab import CatalogTab
from interfaces.telegram.TelegramTransport import TelegramTransport
from interfaces.voice.tts.VoiceFactory import ENGINES

from .integration_fakes import (FakeIntegration, KettleIntegration, LampIntegration, RadioIntegration, disable, enable,
                                named)

ENABLED = module.STATUS[IntegrationState.OK].label
DISABLED = module.STATUS[IntegrationState.DISABLED].label
DESCRIPTION_MAX = 48    # "a few words": what fits the table's "What it is" column on a narrow screen
DETAILS_MAX = 300       # one to three sentences


class ToasterIntegration(FakeIntegration):
    """Toasts bread."""
    name = named("Toaster")
    config_keys = ["toaster.token"]


@pytest.fixture
def monitor(monkeypatch):
    monitor = IntegrationMonitor()
    monkeypatch.setattr(IntegrationMonitor, "_instance", monitor)
    return monitor


def working(monitor, cls):
    enable(monitor, cls)
    monitor.find(cls.config_name()).state = IntegrationState.OK


def rows(tab, width=140, height=24):
    screen = TUIScreen(width, height)
    tab.draw(screen, 0)
    return ["".join(c.char for c in line).rstrip() for line in screen.next_frame]


def row_of(lines, name):
    return next(line for line in lines if f"● {name}" in line)


def test_it_lists_every_integration_the_enabled_ones_first(monitor):
    disable(monitor, KettleIntegration)
    working(monitor, RadioIntegration)
    disable(monitor, LampIntegration)

    lines = rows(CatalogTab())

    listed = [line.split("●")[1].split()[0] for line in lines if "●" in line]
    assert listed == ["Radio", "Kettle", "Lamp"]


def test_a_row_says_whether_the_integration_is_enabled(monitor):
    working(monitor, LampIntegration)
    disable(monitor, KettleIntegration)

    lines = rows(CatalogTab())

    lamp, kettle = row_of(lines, "Lamp"), row_of(lines, "Kettle")
    assert ENABLED in lamp and DISABLED not in lamp
    assert DISABLED in kettle and ENABLED not in kettle


def test_a_row_says_what_the_integration_is(monitor):
    disable(monitor, KettleIntegration)
    working(monitor, LampIntegration)

    lines = rows(CatalogTab())

    kettle, lamp = row_of(lines, "Kettle"), row_of(lines, "Lamp")
    assert kettle.endswith("Boils water on request")
    assert lamp.endswith("Switches the reading lamp")


def test_the_selected_integration_says_what_it_does_but_not_its_code_notes(monitor):
    working(monitor, LampIntegration)

    text = "\n".join(rows(CatalogTab(), width=180, height=40))

    assert "on at dusk and off when the room is empty" in text and "how long" not in text


@pytest.mark.parametrize("cls", [*bootstrap.integration_classes(), WebAccess, TelegramTransport, *ENGINES],
                         ids=lambda cls: cls.__name__)
def test_every_integration_of_this_repository_describes_itself_for_the_catalog(cls):
    """The first two paragraphs of the class docstring: a few words, then one to three sentences."""
    assert 0 < len(cls.description()) <= DESCRIPTION_MAX
    assert 0 < len(cls.details()) <= DETAILS_MAX


def test_the_selected_integration_says_what_it_brings(monitor):
    working(monitor, LampIntegration)
    disable(monitor, KettleIntegration)
    disable(monitor, RadioIntegration)
    tab = CatalogTab(frozenset({LampIntegration}))

    def brings():
        return next(line.split("Brings")[1] for line in rows(tab, width=180, height=40) if "Brings" in line)

    lamp = brings()
    tab.handle_input('down')
    kettle = brings()
    tab.handle_input('down')
    radio = brings()

    assert "own tab" in lamp and "feed" not in lamp and "private data" not in lamp
    assert "private data" in kettle and "feed" not in kettle and "own tab" not in kettle
    assert "feed" in radio and "private data" not in radio


def test_a_row_follows_its_integration_being_enabled(monitor):
    disable(monitor, LampIntegration)
    tab = CatalogTab()
    assert DISABLED in row_of(rows(tab), "Lamp")

    working(monitor, LampIntegration)

    assert DISABLED not in row_of(rows(tab), "Lamp")


def test_nothing_registered_reads_as_still_starting(monitor):
    assert module.STARTING in "\n".join(rows(CatalogTab()))


def test_an_enabled_integration_that_is_not_working_is_named_as_needing_attention(monitor):
    working(monitor, KettleIntegration)
    enable(monitor, LampIntegration)
    monitor.find("Lamp").state = IntegrationState.NEEDS_2FA

    tiles = dict((label, (value, segments)) for label, value, _, segments in CatalogTab._tiles(CatalogTab()._entries()))

    value, segments = tiles["Needs attention"]
    assert value == "1"
    assert ["Lamp" in text for text, _ in segments] == [True]


def test_the_arrows_pick_the_integration_shown_in_full(monitor):
    working(monitor, LampIntegration)
    disable(monitor, KettleIntegration)
    tab = CatalogTab()
    text = "\n".join(rows(tab, width=180, height=40))
    assert "disable Lamp" in text and "enable Kettle" not in text

    assert tab.handle_input('down')

    text = "\n".join(rows(tab, width=180, height=40))
    assert "enable Kettle" in text and "disable Lamp" not in text


def test_the_selected_integration_says_why_it_could_not_be_built(monitor):
    monitor.register(UnavailableIntegration(KettleIntegration, RuntimeError("no kettle section")))

    text = "\n".join(rows(CatalogTab(), width=180, height=40))

    assert "no kettle section" in text


def test_the_selected_integration_names_the_config_keys_it_misses(monitor):
    enable(monitor, ToasterIntegration)

    text = "\n".join(rows(CatalogTab(), width=180, height=40))

    assert monitor.find("Toaster").state is IntegrationState.NOT_CONFIGURED
    assert "toaster.token" in text


def test_the_selection_scrolls_a_list_longer_than_the_screen(monitor):
    disable(monitor, KettleIntegration)
    disable(monitor, LampIntegration)
    disable(monitor, RadioIntegration)
    tab = CatalogTab()
    height = next(h for h in range(30, 8, -1) if not any("● Radio" in line for line in rows(tab, height=h)))

    tab.handle_input('down')
    tab.handle_input('down')

    lines = rows(tab, height=height)
    assert any("● Radio" in line for line in lines) and not any("● Kettle" in line for line in lines)


def test_a_blank_row_parts_the_enabled_from_the_disabled_and_the_selection_crosses_it(monitor):
    """The gap is a row of the table but not an integration: stepping down from the last
    enabled one must land on the first disabled one, tinted, not on the blank row."""
    working(monitor, LampIntegration)
    working(monitor, RadioIntegration)
    disable(monitor, KettleIntegration)
    tab = CatalogTab()
    rows(tab, width=180, height=40)             # the order is known once drawn
    tab.handle_input('down')
    tab.handle_input('down')
    screen = TUIScreen(180, 40)
    tab.draw(screen, 0)
    lines = ["".join(c.char for c in line).rstrip() for line in screen.next_frame]

    radio = next(i for i, line in enumerate(lines) if "● Radio" in line)
    assert "●" not in lines[radio + 1], "no blank row between the enabled and the disabled"
    assert "● Kettle" in lines[radio + 2]
    dot = lines[radio + 2].index("●") + 1
    assert screen.next_frame[radio + 2][dot].style.bg == Theme.COLOR_SELECTED_BG
    assert screen.next_frame[radio + 1][dot].style.bg != Theme.COLOR_SELECTED_BG, "the blank row took the tint"
    assert "enable Kettle" in "\n".join(lines)


def test_no_blank_row_when_every_integration_is_on_the_same_side(monitor):
    disable(monitor, LampIntegration)
    disable(monitor, KettleIntegration)

    lines = rows(CatalogTab())

    kettle = next(i for i, line in enumerate(lines) if "● Kettle" in line)
    assert "● Lamp" in lines[kettle + 1]
