"""CompositeHome: several backends as one house — the merged snapshot, the suffix a name two
backends share gets, the routing of a command to the owner, per-backend pruning, a failing
backend's last good snapshot, and the joined error logs."""
from unittest.mock import patch

import pytest

from assistant.integration import CompositeHome as composite_module
from assistant.integration import HomeAutomation as module
from assistant.integration.CompositeHome import CompositeHome
from assistant.integration.HomeAutomation import Entity, HomeAutomation, HomeSnapshot, render


class Backend(HomeAutomation):
    def __init__(self, name, snapshot=None, section="fake", poll=30):
        self._name, self._snapshot, self._section = name, snapshot or HomeSnapshot(), section
        self.state_poll_interval = poll
        self.section_filters = {}
        self.calls = []
        self.fail = None

    @property
    def name(self): return self._name
    @property
    def config_section(self): return self._section
    def snapshot(self):
        if self.fail:
            raise self.fail
        return self._snapshot
    def turn_on(self, name): self.calls.append(("on", name)); return ""
    def turn_off(self, name): self.calls.append(("off", name)); return ""
    def dim(self, name, level): self.calls.append(("dim", name, level)); return ""
    def execute(self, name): self.calls.append(("execute", name)); return ""
    def stop(self, name): self.calls.append(("stop", name)); return ""
    def error_logs(self): return f"{self._name} log"


@pytest.fixture
def house():
    indigo = Backend("Indigo", HomeSnapshot(
        lights=[Entity("Tak (Kök)", "on"), Entity("Golvlampa (Vardagsrum)", "40")],
        variables=[Entity("Night", "true")],
        actions=[Entity("Godnatt", "all off")]), section="indigo", poll=5)
    ha = Backend("Home Assistant", HomeSnapshot(
        lights=[Entity("Tak", "off", area="Kök"), Entity("Spots", "on", area="Hall")],
        sensors=[Entity("Temp", "21.5 °C", area="Hall")],
        variables=[Entity("Night", "off")]), section="home_assistant", poll=30)
    return indigo, ha, CompositeHome([indigo, ha])


@pytest.fixture(autouse=True)
def no_filters():
    with patch.object(module.config, "get", side_effect=lambda key, default=None: default):
        yield


@pytest.mark.unit
class TestSnapshot:
    def test_merges_in_backend_order_and_suffixes_the_later_copy_of_a_shared_name(self, house):
        indigo, ha, home = house
        snap = home.snapshot()
        assert [e.name for e in snap.lights] == ["Tak (Kök)", "Golvlampa (Vardagsrum)", "Tak (Kök) (Home Assistant)", "Spots"]
        assert snap.lights[2].area is None, "the area is folded into the name so the system comes last"
        assert [e.name for e in snap.variables] == ["Night", "Night (Home Assistant)"]
        assert snap.flag("Night") is True, "the first backend's Night is the house's"
        assert '"Tak [Kök] [Home Assistant]": off' in render(snap)

    def test_a_shared_name_is_announced_once(self, house):
        _, _, home = house
        with patch.object(composite_module, "info") as info:
            home.snapshot()
            home.snapshot()
        assert sum("Tak (Kök)" in c.args[0] for c in info.call_args_list) == 1

    def test_the_poll_interval_is_the_fastest_backends(self, house):
        assert house[2].state_poll_interval == 5

    def test_the_name_names_the_backends(self, house):
        assert "Indigo" in house[2].name and "Home Assistant" in house[2].name


@pytest.mark.unit
class TestRouting:
    def test_a_command_goes_to_the_owner_with_the_backends_own_name(self, house):
        indigo, ha, home = house
        home.snapshot()
        assert home.turn_on("Golvlampa (Vardagsrum)") == ""
        assert home.dim("spots (hall)", 20) == ""
        assert home.turn_off("Tak (Kök) (Home Assistant)") == ""
        assert home.execute("Godnatt") == ""
        assert home.stop("Spots (Hall)") == ""
        assert indigo.calls == [("on", "Golvlampa (Vardagsrum)"), ("execute", "Godnatt")]
        assert ha.calls == [("dim", "Spots (Hall)", 20), ("off", "Tak (Kök)"), ("stop", "Spots (Hall)")]

    def test_a_shared_full_name_stays_with_the_first_backend_and_its_bare_form_is_answered_with_both(self, house):
        indigo, ha, home = house
        home.snapshot()
        assert home.turn_on("Tak (Kök)") == ""
        answer = home.turn_on("Tak")
        assert "Tak [Kök]" in answer and "Tak [Kök] [Home Assistant]" in answer
        assert indigo.calls == [("on", "Tak (Kök)")] and ha.calls == [], "an ambiguous bare name is not guessed"

    def test_a_free_bare_name_reaches_the_backend_with_the_area_entity(self, house):
        indigo, ha, home = house
        home.snapshot()
        assert home.turn_on("spots") == ""
        assert ha.calls == [("on", "Spots")], "the backend gets the bare form it resolves itself"

    def test_an_unknown_name_refreshes_the_index_once_then_names_the_backends(self, house):
        indigo, ha, home = house
        assert home.turn_on("Spots (Hall)") == "", "the first command before any poll fills the index"
        answer = home.turn_on("Fönster")
        assert "Fönster" in answer and "Indigo" in answer and "Home Assistant" in answer
        assert indigo.calls == [] and ha.calls == [("on", "Spots (Hall)")], "no backend was asked to guess"

    def test_a_part_of_a_name_is_answered_with_the_names_it_is_part_of(self, house):
        indigo, ha, home = house
        home.snapshot()
        answer = home.dim("golvlampa", 10)
        assert "Golvlampa [Vardagsrum]" in answer and indigo.calls == []

    def test_the_backends_answer_is_passed_through(self, house):
        indigo, ha, home = house
        home.snapshot()
        indigo.turn_on = lambda name: "Request timed out"
        assert home.turn_on("Godnatt") == "Request timed out"

    def test_an_unknown_name_is_answered_even_when_the_refresh_fails_everywhere(self, house):
        indigo, ha, home = house
        indigo.fail = RuntimeError("Indigo down")
        ha.fail = RuntimeError("HA down")
        answer = home.turn_on("Fönster")
        assert isinstance(answer, str) and "Fönster" in answer
        assert indigo.calls == [] and ha.calls == []

    @pytest.mark.parametrize("name", ["", "   "])
    def test_an_empty_name_is_not_answered_with_every_entity_as_a_lookalike(self, house, name):
        indigo, ha, home = house
        home.snapshot()
        answer = home.turn_on(name)
        assert "Tak" not in answer and "Spots" not in answer and "Godnatt" not in answer
        assert indigo.calls == [] and ha.calls == []

    def test_a_bare_name_another_backend_claimed_through_its_area_is_suffixed_and_routed_apart(self):
        a = Backend("A", HomeSnapshot(lights=[Entity("Lamp", "on", area="Hall")]))
        b = Backend("B", HomeSnapshot(lights=[Entity("Lamp", "off")]))
        home = CompositeHome([a, b])
        snap = home.snapshot()
        assert [e.name for e in snap.lights] == ["Lamp", "Lamp (B)"]
        assert home.turn_on("Lamp") == "" and home.turn_on("Lamp (B)") == ""
        assert a.calls == [("on", "Lamp")] and b.calls == [("on", "Lamp")]


@pytest.mark.unit
class TestPrune:
    def test_each_backends_filters_apply_to_its_own_entities_only(self, house):
        indigo, ha, home = house
        ha.section_filters = {"lights": ("Tak",)}
        filters = {"indigo.prompt_filters": ["Golvlampa"]}
        with patch.object(module.config, "get", side_effect=lambda key, default=None: filters.get(key, default)):
            pruned = home.prune(home.snapshot())
        assert [e.name for e in pruned.lights] == ["Tak (Kök)", "Spots"]
        assert [e.name for e in pruned.sensors] == ["Temp"]

    def test_an_entity_the_index_does_not_know_stays(self, house):
        _, _, home = house
        pruned = home.prune(HomeSnapshot(lights=[Entity("Stray", "on")]))
        assert [e.name for e in pruned.lights] == ["Stray"]


@pytest.mark.unit
class TestFailures:
    def test_a_failing_backend_keeps_its_last_good_snapshot_in_the_merge(self, house):
        indigo, ha, home = house
        home.snapshot()
        ha.fail = RuntimeError("socket closed")
        snap = home.snapshot()
        assert "Spots" in [e.name for e in snap.lights]
        assert home.turn_on("Spots (Hall)") == "" and ha.calls == [("on", "Spots (Hall)")]

    def test_a_backend_that_never_answered_contributes_nothing(self, house):
        indigo, ha, home = house
        ha.fail = RuntimeError("socket closed")
        snap = home.snapshot()
        assert [e.name for e in snap.lights] == ["Tak (Kök)", "Golvlampa (Vardagsrum)"]

    def test_every_backend_failing_raises_so_the_section_keeps_its_content(self, house):
        indigo, ha, home = house
        indigo.fail = ha.fail = RuntimeError("down")
        with pytest.raises(RuntimeError):
            home.snapshot()

    def test_error_logs_are_joined_per_backend(self, house):
        logs = house[2].error_logs()
        assert "--- Indigo ---\nIndigo log" in logs and "--- Home Assistant ---\nHome Assistant log" in logs
