"""The sections that override _start_monitoring each run their own poll loop. Those loops wait
on _wait (not time.sleep) so Assistant.stop() ends them at once; stop() joins with a 5 s
timeout, so a loop that slid back to time.sleep would leak its thread silently — iCloud's
and the music library's both sit in a 60 s wait. The per-section tests stub _wait and cannot
catch that; this one runs the real threads and parks each loop in its wait until stop()."""
from unittest.mock import Mock

import pytest

from common.vault.Vault import Vault
from tests.unit.assistant.prompt.parked_wait import ParkedEvent
from tests.unit.waiting import FAILURE_BOUND


def memory(tmp_path, monkeypatch):
    from assistant.prompt.sections.MemorySectionGenerator import MemorySectionGenerator
    section = MemorySectionGenerator(None)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text("remember")
    section.file_path = str(memory_file)
    return section


def sonos(tmp_path, monkeypatch):
    from assistant.integration.music import MusicSystem as music
    from assistant.prompt.sections.MusicLibrarySectionGenerator import MusicLibrarySectionGenerator
    monkeypatch.setattr(music, "active", lambda: [Mock(has_library=Mock(return_value=True))])
    return MusicLibrarySectionGenerator(None)


def cars(tmp_path, monkeypatch):
    from assistant.prompt.sections import CarSectionGenerator as module
    monkeypatch.setattr(module, "config", Mock(get_home_latitude=Mock(return_value=57.0),
                                               get_home_longitude=Mock(return_value=12.0)))
    monkeypatch.setattr(module, "active", lambda: [])
    return module.CarSectionGenerator(None)


def icloud(tmp_path, monkeypatch):
    from integrations.icloud.iCloudSectionGenerator import iCloudSectionGenerator
    return iCloudSectionGenerator(None)


def vault(tmp_path, monkeypatch):
    from assistant.prompt.sections.VaultSectionGenerator import VaultSectionGenerator
    return VaultSectionGenerator(None, vault=Vault(tmp_path / "vault.bin", passphrase="test"))


@pytest.mark.unit
@pytest.mark.parametrize("build", [memory, sonos, cars, icloud, vault], ids=lambda f: f.__name__)
def test_stop_ends_the_sections_own_loop_at_once(build, tmp_path, monkeypatch):
    section = build(tmp_path, monkeypatch)
    section._fetch_content = lambda: "x"
    # The event the section's _wait waits on parks the loop until stop(): a loop that waited with
    # time.sleep never reaches it, and one that leaves it was ended by stop().
    waits_on = "_wake" if hasattr(section, "_wake") else "_stop_event"
    setattr(section, waits_on, parked := ParkedEvent())
    section.start()
    assert parked.waiting.wait(FAILURE_BOUND), f"{section.section_name}: the loop never reached its wait"
    assert section.get_content() == "x", f"{section.section_name}: the loop never reached its first fetch"

    section.stop()
    section._thread.join(FAILURE_BOUND)
    assert not section._thread.is_alive(), f"{section.section_name}: stop() left the loop waiting out its interval"
