"""The sections tied to an integration follow the IntegrationMonitor's verdict: built and
started when it says on, stopped and dropped when it says off, and rendered under their own
headings in the prompt's one {INTEGRATION_DATA} block while they run."""
from unittest.mock import patch

import pytest

from assistant.prompt.PromptManager import PromptManager
from assistant.prompt.PromptSectionGenerator import PromptSectionGenerator


class Brewing:
    """A contract a core section may need."""


class KettleSectionGenerator(PromptSectionGenerator):
    __module__ = "integrations.kettle.KettleSectionGenerator"
    key = "kettle"
    heading = "# THE KETTLE"
    intro = "What the kettle is up to:"
    data = "Kettle: 40°C, heating"

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "KettleState")
        self.content = type(self).data

    def _fetch_content(self) -> str:
        return type(self).data


class CarSectionGenerator(KettleSectionGenerator):
    """Takes a key SECTION_ORDER places, to stand ahead of the kettle in the block."""
    __module__ = "integrations.garage.CarSectionGenerator"
    key = "car_info"
    heading = "# THE CAR"
    intro = ""
    data = "Car: parked"

    def __init__(self, prompt_manager):
        PromptSectionGenerator.__init__(self, prompt_manager, "CarState")
        self.content = type(self).data


class BrewSectionGenerator(KettleSectionGenerator):
    """A core section tied to a contract."""
    __module__ = "tests.brew"
    key = "brew"
    needs = Brewing
    heading = "## Brewing"
    intro = ""
    data = "Brewing: nothing"

    def __init__(self, prompt_manager):
        PromptSectionGenerator.__init__(self, prompt_manager, "BrewState")
        self.content = type(self).data


class FakeMonitor:
    def __init__(self):
        self.folders, self.contracts, self.listeners = {}, {}, []

    def folder_enabled(self, folder):
        return self.folders.get(folder)

    def contract_enabled(self, contract):
        return self.contracts.get(contract)

    def add_listener(self, callback):
        self.listeners.append(callback)

    def remove_listener(self, callback):
        self.listeners.remove(callback)

    def notify(self):
        for callback in list(self.listeners):
            callback()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "prompt.md").write_text("Header\n{MEMORY}\nMiddle\n{INTEGRATION_DATA}\nFooter\n")
    monkeypatch.chdir(tmp_path)
    KettleSectionGenerator.data = "Kettle: 40°C, heating"
    CarSectionGenerator.data = "Car: parked"
    BrewSectionGenerator.data = "Brewing: nothing"


@pytest.fixture
def manager(workspace):
    """A manager whose generators record starts and stops instead of running threads, on a
    scripted monitor."""
    monitor = FakeMonitor()
    lifecycle = []
    with patch.object(PromptManager, "_monitor", lambda self: monitor), \
            patch.object(PromptSectionGenerator, "start", lambda self: lifecycle.append(("start", self.section_name))), \
            patch.object(PromptSectionGenerator, "stop", lambda self: lifecycle.append(("stop", self.section_name))):
        pm = PromptManager(section_generators=[KettleSectionGenerator, CarSectionGenerator, BrewSectionGenerator])
        pm.monitor, pm.lifecycle = monitor, lifecycle
        yield pm
        pm.stop()


@pytest.mark.unit
def test_a_gated_section_waits_for_its_integration_and_the_manager_listens_for_it(manager):
    assert "kettle" not in manager.sections and "brew" not in manager.sections
    assert {"kettle", "car_info", "brew"} <= set(manager.section_classes())
    assert manager._on_integrations_changed in manager.monitor.listeners
    assert ("start", "KettleState") not in manager.lifecycle
    assert manager.DATA_BLOCK not in manager.prompt and "THE KETTLE" not in manager.prompt


@pytest.mark.unit
def test_the_section_starts_when_its_folder_is_on_and_stands_under_its_heading(manager):
    manager.monitor.folders["kettle"] = True
    manager.monitor.notify()
    assert "kettle" in manager.sections and ("start", "KettleState") in manager.lifecycle
    assert "# THE KETTLE\nWhat the kettle is up to:\nKettle: 40°C, heating\n" in manager.prompt
    assert manager.prompt.index("Middle") < manager.prompt.index("# THE KETTLE") < manager.prompt.index("Footer")


@pytest.mark.unit
def test_the_section_stops_and_leaves_the_prompt_when_its_folder_goes_off(manager):
    manager.monitor.folders["kettle"] = True
    manager.monitor.notify()
    manager.monitor.folders["kettle"] = False
    manager.monitor.notify()
    assert "kettle" not in manager.sections and ("stop", "KettleState") in manager.lifecycle
    assert "THE KETTLE" not in manager.prompt
    assert "kettle" in manager.section_classes(), "it is still known, for the day the kettle is back"


@pytest.mark.unit
def test_a_core_section_follows_the_contract_it_needs(manager):
    manager.monitor.contracts[Brewing] = True
    manager.monitor.notify()
    assert "brew" in manager.sections and "## Brewing\nBrewing: nothing\n" in manager.prompt
    manager.monitor.contracts[Brewing] = None
    manager.monitor.notify()
    assert "brew" not in manager.sections


@pytest.mark.unit
def test_the_block_keeps_section_order_and_leaves_out_a_section_with_nothing_to_say(manager):
    manager.monitor.folders.update(kettle=True, garage=True)
    manager.monitor.notify()
    assert manager.prompt.index("# THE CAR") < manager.prompt.index("# THE KETTLE"), \
        "car_info has a place in SECTION_ORDER, a new folder's key comes after"
    assert manager._section_positions["kettle"] > manager._section_positions["car_info"] > manager._section_positions["memory"]
    CarSectionGenerator.data = "   "
    manager.sections["car_info"].content = "   "
    manager._assemble_prompt()
    assert "# THE CAR" not in manager.prompt and "# THE KETTLE" in manager.prompt


@pytest.mark.unit
def test_a_change_to_a_block_section_reaches_the_llm_as_a_delta_note(manager):
    manager.monitor.folders["kettle"] = True
    manager.monitor.notify()
    manager.sections["kettle"].content = "Kettle: 100°C, boiled"
    manager.on_section_updated("KettleState")
    notes = [e["content"] for e in manager.message_history.transient_entries if e.get("section") == "kettle"]
    assert len(notes) == 1 and "+Kettle: 100°C, boiled" in notes[0]
    assert "Kettle: 40°C, heating" in manager.prompt, "the prompt bytes stay as they were"


@pytest.mark.unit
def test_a_notify_that_changes_nothing_still_rebuilds_for_the_toolkit_index(manager):
    before = manager._last_assembly_time
    manager.monitor.notify()
    assert manager._last_assembly_time >= before and "kettle" not in manager.sections


class BrokenSectionGenerator(KettleSectionGenerator):
    """A folder's generator whose constructor fails while `broken` is set."""
    __module__ = "integrations.toaster.BrokenSectionGenerator"
    key = "toaster"
    heading = "# THE TOASTER"
    intro = ""
    data = "Toaster: idle"
    broken = True

    def __init__(self, prompt_manager):
        if type(self).broken:
            raise RuntimeError("no toaster on the network")
        PromptSectionGenerator.__init__(self, prompt_manager, "ToasterState")
        self.content = type(self).data


@pytest.mark.unit
def test_a_gated_generator_that_fails_to_build_costs_only_itself_until_it_builds(workspace):
    BrokenSectionGenerator.broken = True
    monitor, lifecycle = FakeMonitor(), []
    with patch.object(PromptManager, "_monitor", lambda self: monitor), \
            patch.object(PromptSectionGenerator, "start", lambda self: lifecycle.append(("start", self.section_name))), \
            patch.object(PromptSectionGenerator, "stop", lambda self: lifecycle.append(("stop", self.section_name))):
        pm = PromptManager(section_generators=[KettleSectionGenerator, BrokenSectionGenerator])
        try:
            monitor.folders.update(kettle=True, toaster=True)
            monitor.notify()
            assert "kettle" in pm.sections and "toaster" not in pm.sections
            assert ("start", "KettleState") in lifecycle and ("start", "ToasterState") not in lifecycle
            assert "toaster" in pm.section_classes(), "still gated, for the next verdict"

            BrokenSectionGenerator.broken = False
            monitor.notify()
            assert "toaster" in pm.sections and ("start", "ToasterState") in lifecycle
            assert "# THE TOASTER\nToaster: idle\n" in pm.prompt
        finally:
            pm.stop()
