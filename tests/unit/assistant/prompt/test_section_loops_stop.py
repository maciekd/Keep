"""The sections that override _start_monitoring each run their own poll loop. Those loops wait
on _wait (not time.sleep) so Assistant.stop() ends them at once; stop() joins with a 5 s
timeout, so a loop that slid back to time.sleep would leak its thread silently — iCloud's
sits in a 60 s wait, Sonos's in a 30 min one. The per-section tests stub _wait and cannot
catch that; this one runs the real threads and times stop()."""
import time
from unittest.mock import Mock

import pytest

from common.vault.Vault import Vault


def memory(tmp_path, monkeypatch):
    from assistant.prompt.sections.MemorySectionGenerator import MemorySectionGenerator
    section = MemorySectionGenerator(None)
    memory_file = tmp_path / "memory.md"
    memory_file.write_text("remember")
    section.file_path = str(memory_file)
    return section


def sonos(tmp_path, monkeypatch):
    from assistant.prompt.sections import SonosFavoritesSectionGenerator as module
    loaded = Mock(get_playlists=Mock(return_value=["p"]), get_favorites=Mock(return_value=[]))
    monkeypatch.setattr(module.MusicManager, "peek_instance", lambda: loaded)
    return module.SonosFavoritesSectionGenerator(None)


def tesla(tmp_path, monkeypatch):
    from assistant.prompt.sections import TeslaSectionGenerator as module
    monkeypatch.setattr(module, "config", Mock(get_home_latitude=Mock(return_value=57.0),
                                               get_home_longitude=Mock(return_value=12.0)))
    monkeypatch.setattr(module, "TeslaIntegration", Mock())
    return module.TeslaSectionGenerator(None)


def icloud(tmp_path, monkeypatch):
    from assistant.prompt.sections.iCloudSectionGenerator import iCloudSectionGenerator
    return iCloudSectionGenerator(None)


def vault(tmp_path, monkeypatch):
    from assistant.prompt.sections.VaultSectionGenerator import VaultSectionGenerator
    return VaultSectionGenerator(None, vault=Vault(tmp_path / "vault.bin", passphrase="test"))


@pytest.mark.unit
@pytest.mark.parametrize("build", [memory, sonos, tesla, icloud, vault], ids=lambda f: f.__name__)
def test_stop_ends_the_sections_own_loop_at_once(build, tmp_path, monkeypatch):
    section = build(tmp_path, monkeypatch)
    section._fetch_content = lambda: "x"
    section.start()
    deadline = time.time() + 2
    while section.get_content() != "x" and time.time() < deadline:
        time.sleep(0.01)
    assert section.get_content() == "x", f"{section.section_name}: the loop never reached its first fetch"

    started = time.time()
    section.stop()
    assert not section._thread.is_alive(), f"{section.section_name}: stop() left the loop waiting out its interval"
    assert time.time() - started < 0.5
