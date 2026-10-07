import pytest

from assistant.integration.Integration import IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor, UnavailableIntegration
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.IntegrationsGroup import IntegrationsGroup
from interfaces.console.tabs.TabNavigator import TabNavigator

from .integration_fakes import (FakeTab, KettleIntegration, KettleTab, LampIntegration, LampTab, RadioIntegration,
                                RadioTab, disable, enable)
from .test_tab_navigator import FakeTab as PlainTab


@pytest.fixture
def monitor(monkeypatch):
    monitor = IntegrationMonitor()
    monkeypatch.setattr(IntegrationMonitor, "_instance", monitor)
    monkeypatch.setattr(FakeTab, "built", [])
    return monitor


def make_group():
    return IntegrationsGroup([LampTab, KettleTab, RadioTab])


def names(group):
    group.refresh()
    return [tab.name for tab in group.tabs]


def test_a_console_built_before_the_integrations_register_has_the_catalog_alone(monitor):
    group = make_group()

    assert names(group) == ["Catalog"]
    assert group.content_tab().name == "Catalog"
    assert FakeTab.built == []


def test_the_row_follows_the_monitor_in_alphabetical_order_after_the_catalog(monitor):
    group = make_group()
    enable(monitor, RadioIntegration)
    enable(monitor, LampIntegration)
    disable(monitor, KettleIntegration)
    assert names(group) == ["Catalog", "Lamp", "Radio"]

    enable(monitor, KettleIntegration)
    assert names(group) == ["Catalog", "Kettle", "Lamp", "Radio"]

    disable(monitor, LampIntegration)
    assert names(group) == ["Catalog", "Kettle", "Radio"]


def test_the_tab_of_a_disabled_integration_is_never_built(monitor):
    group = make_group()
    enable(monitor, LampIntegration)
    disable(monitor, KettleIntegration)

    group.draw(TUIScreen(100, 30), 0)
    group.draw(TUIScreen(100, 30), 1)

    assert [type(tab) for tab in FakeTab.built] == [LampTab]


@pytest.mark.parametrize("state", [IntegrationState.NOT_CONFIGURED, IntegrationState.NEEDS_2FA, IntegrationState.ERROR])
def test_an_enabled_integration_keeps_its_tab_whatever_its_health(monitor, state):
    group = make_group()
    lamp = LampIntegration()
    monitor.register(lamp)
    lamp.state = state

    assert names(group) == ["Catalog", "Lamp"]


def test_an_integration_whose_constructor_failed_still_has_its_tab(monitor):
    group = make_group()
    monitor.register(UnavailableIntegration(KettleIntegration, RuntimeError("no kettle section")))

    assert names(group) == ["Catalog", "Kettle"]


def test_the_catalog_stays_selected_while_tabs_join_the_row(monitor):
    group = make_group()
    assert names(group) == ["Catalog"]
    enable(monitor, LampIntegration)

    group.refresh()

    assert group.content_tab().name == "Catalog"


def test_the_selected_tab_stays_selected_when_the_row_changes_around_it(monitor):
    group = make_group()
    enable(monitor, LampIntegration)
    enable(monitor, RadioIntegration)
    group.on_activate()
    group.select(2)
    radio = group.content_tab()
    assert radio.name == "Radio"
    seen = list(radio.events)

    enable(monitor, KettleIntegration)
    disable(monitor, LampIntegration)
    group.refresh()

    assert group.content_tab() is radio
    assert radio.events == seen


def test_the_tab_on_screen_gives_way_to_its_neighbour_when_its_integration_is_disabled(monitor):
    group = make_group()
    enable(monitor, LampIntegration)
    enable(monitor, RadioIntegration)
    group.on_activate()
    group.select(1)
    lamp = group.content_tab()

    disable(monitor, LampIntegration)
    group.draw(TUIScreen(100, 30), 0)

    radio = group.content_tab()
    assert radio.name == "Radio"
    assert lamp.events[-1] == "deactivate"
    assert radio.events[:2] == ["activate", "draw"]


def test_a_tab_dropped_off_screen_is_not_told_twice_that_it_left(monitor):
    group = make_group()
    enable(monitor, LampIntegration)
    group.on_activate()
    group.select(1)
    lamp = group.content_tab()
    group.on_deactivate()

    disable(monitor, LampIntegration)
    group.refresh()

    assert lamp.events.count("deactivate") == 1


def test_an_integration_enabled_again_gets_a_fresh_tab(monitor):
    group = make_group()
    enable(monitor, LampIntegration)
    group.refresh()
    first = group.tabs[1]

    disable(monitor, LampIntegration)
    group.refresh()
    enable(monitor, LampIntegration)
    group.refresh()

    assert group.tabs[1].name == "Lamp" and group.tabs[1] is not first


def test_the_tab_walk_enters_the_row_as_it_is_now_not_as_last_drawn(monitor):
    home, group = PlainTab("Home"), make_group()
    nav = TabNavigator([home, group])
    enable(monitor, LampIntegration)
    enable(monitor, KettleIntegration)

    nav.handle_key('shift_tab')
    assert nav.path == "Integrations/Lamp"
    nav.handle_key('shift_tab')
    assert nav.path == "Integrations/Kettle"

    nav.handle_key('tab')
    nav.handle_key('tab')
    nav.handle_key('tab')
    assert nav.path == "Integrations/Catalog"


class BrokenTab(FakeTab):
    integration = KettleIntegration

    def __init__(self):
        raise RuntimeError("no kettle on the network")


def test_a_tab_that_fails_to_build_is_left_out_while_the_rest_of_the_row_builds(monitor):
    group = IntegrationsGroup([LampTab, BrokenTab, RadioTab])
    enable(monitor, LampIntegration)
    enable(monitor, KettleIntegration)
    enable(monitor, RadioIntegration)

    assert names(group) == ["Catalog", "Lamp", "Radio"]
    group.draw(TUIScreen(100, 30), 0)
    assert {type(tab) for tab in FakeTab.built} == {LampTab, RadioTab}


def test_the_selection_clamps_to_the_new_last_tab_when_the_last_one_is_disabled(monitor):
    group = make_group()
    enable(monitor, LampIntegration)
    enable(monitor, RadioIntegration)
    group.on_activate()
    group.select(2)
    assert group.content_tab().name == "Radio"

    disable(monitor, RadioIntegration)
    group.draw(TUIScreen(100, 30), 0)

    assert names(group) == ["Catalog", "Lamp"]
    assert group.content_tab().name == "Lamp"
