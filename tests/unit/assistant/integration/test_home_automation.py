"""The HomeAutomation contract: the data model, the display round trip, the pruning and the
renderer, `active()` resolving the backend, and the not-configured stand-in."""
from unittest.mock import patch

import pytest

from assistant.integration.Integration import Integration
from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration, UnavailableIntegration
from assistant.integration import HomeAutomation as module
from assistant.integration.CompositeHome import CompositeHome
from assistant.integration.HomeAutomation import (
    Entity, HomeAutomation, HomeSnapshot, NotConfiguredHome, STARTING_HINT, STARTUP_POLL_INTERVAL, active, prune, render)


class FakeHome(Integration, HomeAutomation):
    """A backend with two config keys, remembering what it was asked."""
    calls: list = []

    def __init__(self, missing=()):
        super().__init__()
        self._missing = list(missing)

    @property
    def name(self): return "Fake Home"
    @property
    def config_keys(self): return ["fake.url", "fake.token"]
    @property
    def config_section(self): return "fake"
    def check_health(self): return True
    def missing_config_keys(self): return self._missing
    def snapshot(self): return HomeSnapshot(variables=[Entity("Night", "true")])
    def turn_on(self, name): self.calls.append(("on", name)); return ""
    def turn_off(self, name): self.calls.append(("off", name)); return ""
    def dim(self, name, level): self.calls.append(("dim", name, level)); return ""
    def execute(self, name): self.calls.append(("execute", name)); return ""


class OtherHome(FakeHome):
    @property
    def name(self): return "Other Home"


@pytest.fixture
def monitor():
    IntegrationMonitor.reset_instance()
    module._last_state = None
    module._composite = None
    with patch.object(IntegrationMonitor, "_config_manager", create=True):
        m = IntegrationMonitor.get_instance()
        yield m
    IntegrationMonitor.reset_instance()


def _register(monitor, integration):
    with monitor._lock:
        monitor._integrations[integration.name] = integration


@pytest.mark.unit
class TestContract:
    def test_display_round_trip(self):
        assert HomeAutomation.to_display('"Tak (Kök)": on') == '"Tak [Kök]": on'
        assert HomeAutomation.from_display("Tak [Kök]") == "Tak (Kök)"
        assert HomeAutomation.from_display(HomeAutomation.to_display("Tak (Kök) (2)")) == "Tak (Kök) (2)"

    def test_stop_is_not_a_verb_a_backend_has_to_bring(self):
        """Covers came with Plejd; Indigo and Home Assistant predate them. A backend that
        implements the abstract verbs alone still constructs, and asked to stop something it
        refuses in its own name instead of raising or claiming success (an empty answer)."""
        answer = FakeHome().stop("Markis (Uterum)")
        assert answer and "Fake Home" in answer

    def test_the_not_configured_stand_in_answers_stop_with_the_same_hint_as_every_verb(self):
        home = NotConfiguredHome("set home.url first")
        assert home.stop("Markis") == home.turn_on("Markis") == home.execute("Markis")


@pytest.mark.unit
class TestSnapshot:
    def test_flag_reads_truthy_and_falsy_spellings(self):
        snap = HomeSnapshot(variables=[Entity("Night", "True"), Entity("Away", "off"),
                                       Entity("Mode", "Home"), Entity("Count", "1")])
        assert snap.flag("Night") is True
        assert snap.flag("Away") is False
        assert snap.flag("Count") is True
        assert snap.flag("Mode") is None, "a value that is neither is unknown"
        assert snap.flag("Missing") is None

    def test_is_empty(self):
        assert HomeSnapshot().is_empty()
        assert not HomeSnapshot(actions=[Entity("Go", "")]).is_empty()


@pytest.mark.unit
class TestRender:
    def test_renders_the_four_blocks_in_order_with_areas_and_display_names(self):
        snap = HomeSnapshot(lights=[Entity("Tak (Kök)", "on"), Entity("Lamp", "40", area="Office")],
                            sensors=[Entity("Temp", "22.123456 °C")],
                            variables=[Entity("Night", "false")],
                            actions=[Entity("Go home", "")])
        assert render(snap) == (
            '## LIGHTS\n"Tak [Kök]": on\n"Lamp [Office]": 40\n\n'
            '## SENSORS\n"Temp": 22.12 °C\n\n'
            '## VARIABLES\n"Night": false\n\n'
            '## ACTIONS\n"Go home":\n\n')

    def test_an_empty_list_has_no_block(self):
        assert render(HomeSnapshot(actions=[Entity("Go", "")])) == '## ACTIONS\n"Go":\n\n'
        assert render(HomeSnapshot()) == ""

    def test_filters_drop_matching_lines_case_insensitively(self):
        snap = HomeSnapshot(sensors=[Entity("Humidity (Bad)", "50"), Entity("Temp", "20")],
                            variables=[Entity("Noisy", "1"), Entity("Night", "true")])
        out = render(prune(snap, ["noisy"], {"sensors": ("humidity",)}))
        assert out == '## SENSORS\n"Temp": 20\n\n## VARIABLES\n"Night": true\n\n'

    def test_a_backend_filter_applies_to_its_section_only(self):
        snap = HomeSnapshot(lights=[Entity("Humidity light", "on")], sensors=[Entity("Humidity", "50")])
        out = render(prune(snap, [], {"sensors": ("Humidity",)}))
        assert '"Humidity light": on' in out and '"Humidity": 50' not in out

    def test_a_filter_matches_the_rendered_line_so_an_area_or_a_value_can_be_the_criterion(self):
        snap = HomeSnapshot(lights=[Entity("Lamp", "on", area="Garage"), Entity("Lamp", "unavailable", area="Hall")])
        assert prune(snap, ["[garage]", "unavailable"], {}).lights == []
        assert prune(snap, [], {}).lights == snap.lights

    def test_the_backend_prunes_with_its_own_config_section(self):
        snap = HomeSnapshot(lights=[Entity("Noisy", "on"), Entity("Lamp", "on")])
        with patch.object(module.config, "get", side_effect=lambda key, default=None: ["noisy"] if key == "fake.prompt_filters" else default):
            assert [e.name for e in FakeHome().prune(snap).lights] == ["Lamp"]
        with patch.object(module.config, "get", side_effect=lambda key, default=None: None):
            assert len(FakeHome().prune(snap).lights) == 2, "an unset key means no filters"


@pytest.mark.unit
class TestActive:
    def test_no_backend_registered_in_any_form_is_still_starting(self, monitor):
        home = active()
        assert isinstance(home, NotConfiguredHome)
        assert home.starting is True
        assert home.state_poll_interval == STARTUP_POLL_INTERVAL
        assert home.hint == STARTING_HINT
        assert home.turn_on("x") == home.dim("x", 1) == home.execute("x") == home.error_logs() == STARTING_HINT
        assert home.snapshot().is_empty()

    def test_a_placeholder_ends_the_starting_state(self, monitor):
        _register(monitor, DisabledIntegration(FakeHome))
        home = active()
        assert home.starting is False
        assert home.state_poll_interval == HomeAutomation.state_poll_interval

    def test_an_unavailable_backend_gives_its_error(self, monitor):
        _register(monitor, UnavailableIntegration(FakeHome, KeyError("fake")))
        home = active()
        assert isinstance(home, NotConfiguredHome)
        assert home.starting is False
        assert home.hint == "FakeHome could not be started: KeyError: 'fake' (restart_integration once fixed)."

    def test_the_configured_backend_wins(self, monitor):
        fake = FakeHome()
        _register(monitor, fake)
        assert active() is fake

    def test_an_unconfigured_backend_gives_a_hint_naming_its_keys(self, monitor):
        _register(monitor, FakeHome(missing=["fake.token"]))
        home = active()
        assert isinstance(home, NotConfiguredHome)
        assert "Fake Home is enabled but not configured: set fake.token" in home.hint

    def test_a_disabled_backend_gives_the_enable_line(self, monitor):
        _register(monitor, DisabledIntegration(FakeHome))
        home = active()
        assert isinstance(home, NotConfiguredHome)
        assert home.hint == 'Fake Home is disabled: enable_integration("Fake Home").'

    def test_a_disabled_non_home_integration_is_ignored(self, monitor):
        class Weather(Integration):
            @property
            def name(self): return "Weather"
            @property
            def config_keys(self): return []
            def check_health(self): return True
        _register(monitor, DisabledIntegration(Weather))
        assert active().hint == STARTING_HINT


    def test_a_broken_non_home_integration_is_ignored(self, monitor):
        """Tesla failing its constructor must not read as a broken house."""
        class Weather(Integration):
            @property
            def name(self): return "Weather"
            @property
            def config_keys(self): return []
            def check_health(self): return True
        _register(monitor, UnavailableIntegration(Weather, KeyError("weather")))
        assert active().hint == STARTING_HINT

    def test_a_disabled_and_a_broken_backend_are_both_named(self, monitor):
        _register(monitor, DisabledIntegration(FakeHome))
        _register(monitor, UnavailableIntegration(OtherHome, KeyError("other")))
        home = active()
        assert home.starting is False
        assert 'Fake Home is disabled: enable_integration("Fake Home").' in home.hint
        assert "OtherHome could not be started: KeyError: 'other' (restart_integration once fixed)." in home.hint

    def test_two_configured_backends_make_one_composite_house(self, monitor):
        first, second = FakeHome(), OtherHome()
        _register(monitor, first)
        _register(monitor, second)
        with patch.object(module, "info") as info:
            home = active()
        assert isinstance(home, CompositeHome)
        assert home.backends == [first, second]
        assert "2 backends enabled" in info.call_args.args[0]

    def test_the_composite_is_kept_while_the_set_of_backends_holds(self, monitor):
        """It carries the name index, so a new one per call would route nothing."""
        first, second = FakeHome(), OtherHome()
        _register(monitor, first)
        _register(monitor, second)
        home = active()
        assert active() is home
        third = type("ThirdHome", (FakeHome,), {"name": property(lambda s: "Third Home")})()
        _register(monitor, third)
        assert active() is not home
        assert active().backends == [first, second, third]

    def test_a_ready_backend_beside_an_unconfigured_one_is_used_alone(self, monitor):
        fake = FakeHome()
        _register(monitor, fake)
        _register(monitor, OtherHome(missing=["fake.token"]))
        assert active() is fake

    def test_an_unchanged_resolution_is_logged_once(self, monitor):
        with patch.object(module, "trace") as trace:
            assert active().starting is True
            assert active().starting is True
        assert trace.call_count == 1
        _register(monitor, FakeHome())
        with patch.object(module, "trace") as trace:
            active()
            active()
        assert trace.call_count == 1

    def test_of_type_lists_instances_in_registration_order(self, monitor):
        fake, other = FakeHome(), OtherHome()
        _register(monitor, fake)
        _register(monitor, DisabledIntegration(Weather := type("W", (Integration,), {
            "name": property(lambda s: "W"), "config_keys": property(lambda s: []), "check_health": lambda s: True})))
        _register(monitor, other)
        assert monitor.of_type(HomeAutomation) == [fake, other]
        assert [d.cls for d in monitor.of_type(DisabledIntegration)] == [Weather]
