"""An integration that needs macOS (`Integration.mac_only`) on a server that is not a Mac: never
built, listed with the reason, and off to every gate."""
from typing import List

import pytest

from assistant.integration import HomeAutomation as home
from assistant.integration.HomeAutomation import HomeAutomation
from assistant.integration.Integration import Integration, IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor, MacOnlyIntegration
from assistant.integration.bootstrap import integration_classes
from common import Platform


class GardenLightsIntegration(Integration):
    """Fictional lights only a Mac can reach.

    Switches them.

    Counts its constructions, which is what the tests ask about."""
    mac_only = True
    enabled_by_default = True
    built = 0

    def __init__(self):
        super().__init__()
        type(self).built += 1

    @property
    def name(self) -> str:
        return "Garden Lights"

    @property
    def config_keys(self) -> List[str]:
        return []

    def check_health(self) -> bool:
        return True


class GardenCoversIntegration(GardenLightsIntegration, HomeAutomation):
    """Fictional covers behind the house contract, Mac-only too.

    Opens them.

    Never constructed in these tests: the contract's methods are left abstract."""

    @property
    def name(self) -> str:
        return "Garden Covers"


@pytest.fixture
def monitor(monkeypatch):
    GardenLightsIntegration.built = 0
    monkeypatch.setattr(GardenLightsIntegration, "_instance", None, raising=False)
    monkeypatch.setattr(IntegrationMonitor, "_instance", None)
    return IntegrationMonitor.get_instance()


@pytest.mark.unit
class TestOnLinux:
    @pytest.fixture(autouse=True)
    def linux(self, monkeypatch):
        monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)

    def test_it_is_never_constructed_though_the_config_enables_it(self, monitor):
        started = []
        registered = monitor.register_class(GardenLightsIntegration, start=started.append)
        assert GardenLightsIntegration.built == 0 and started == []
        assert isinstance(registered, MacOnlyIntegration)

    def test_the_status_says_it_needs_a_mac_and_every_gate_reads_it_as_off(self, monitor):
        monitor.register_class(GardenLightsIntegration)
        status, = monitor.get_all_statuses()
        assert status["needs_mac"] and status["state"] == IntegrationState.DISABLED.value
        assert "Garden Lights" in status["last_error"] and "Linux" in status["last_error"]

    def test_the_contract_it_implements_reads_as_off(self, monitor):
        monitor.register_class(GardenCoversIntegration)
        assert monitor.contract_enabled(HomeAutomation) is False

    def test_a_restart_does_not_build_it_either(self, monitor):
        monitor.register_class(GardenLightsIntegration)
        result = monitor.restart("Garden Lights")
        assert GardenLightsIntegration.built == 0
        assert result["status"]["needs_mac"]

    def test_the_house_does_not_tell_the_llm_to_enable_it(self, monitor):
        monitor.register_class(GardenCoversIntegration)
        assert "enable_integration" not in home.active().hint


@pytest.mark.unit
def test_on_a_mac_it_is_built_like_any_other(monitor, monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
    registered = monitor.register_class(GardenLightsIntegration)
    assert GardenLightsIntegration.built == 1
    assert not monitor.get_all_statuses()[0]["needs_mac"]
    assert registered.state is not IntegrationState.DISABLED


@pytest.mark.unit
def test_the_folders_that_load_a_mac_framework_are_the_mac_only_ones():
    """Photos reads the Mac's Photos library, Reminders EventKit, HomeClaw an app that exists
    for macOS alone: a Linux server must not build them from a config copied from a Mac."""
    mac_only = {cls.config_name() for cls in integration_classes() if cls.mac_only}
    assert mac_only == {"Photos", "Reminders", "HomeClaw"}
