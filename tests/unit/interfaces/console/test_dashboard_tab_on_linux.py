"""The Dashboard on a server that is not a Mac: no GPU or temperature monitor is built, their
tiles and the ones of the server's own audio are not drawn, and the rest of the board still is."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistant.integration.IntegrationMonitor import IntegrationMonitor
from common import Platform
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs import DashboardTab as module

MAC_TILES = ("GPU Usage", "GPU Memory", "CPU Temp", "Mic noise floor")
MAC_ROWS = ("Processed photos", "Volume", "Background noise")
WIDTH, HEIGHT = 200, 70


def _console():
    numbers = dict.fromkeys(
        ("current_noise_db", "db_size_gb", "fd_count", "photos_indexed", "photos_total", "process_memory_gb",
         "system_available_memory_gb", "system_used_memory_gb", "target_server_volume", "voice_cache_size_gb"), 0)
    timings = dict.fromkeys(
        ("time_to_first_sound_chunk", "time_to_first_token", "time_to_first_word", "time_to_silence_detect",
         "time_to_speaker", "time_to_user_reply"), -1)
    prompt_manager = Mock()
    prompt_manager.get_section_stats.return_value = []
    return SimpleNamespace(**numbers, **timings, system_cpu_percent=37.0, process_cpu_percent=12.0,
                           system_total_memory_gb=32.0, voice_cache_max_gb=1.0, outpost_latencies={},
                           llm_from_cache=False, voice_from_cache=False, stt_trigger="", prompt_manager=prompt_manager)


@pytest.fixture
def board(monkeypatch):
    """`build(system)` → (the tab, the monitors it built by name); `text(tab)` → the drawn screen."""
    monkeypatch.setattr(IntegrationMonitor, "_instance", IntegrationMonitor())
    built = []
    for name in ("GpuMonitor", "TempMonitor", "AnthropicCostMonitor", "MemoryLogMonitor", "CacheHitLogMonitor"):
        monkeypatch.setattr(module, name, lambda *a, _name=name, **k: built.append(_name) or Mock())

    def build(system):
        monkeypatch.setattr(Platform, "current", lambda: system)
        del built[:]
        return module.DashboardTab(_console()), list(built)

    def text(tab):
        screen = TUIScreen(WIDTH, HEIGHT)
        tab.on_activate()
        tab.draw(screen, 0)
        tab.on_deactivate()
        return "\n".join("".join(c.char for c in line) for line in screen.next_frame)

    return SimpleNamespace(build=build, text=text)


@pytest.mark.unit
def test_on_linux_neither_monitor_is_built_and_their_tiles_are_gone_while_the_cpu_stays(board):
    tab, built = board.build(Platform.LINUX)
    assert "GpuMonitor" not in built and "TempMonitor" not in built
    drawn = board.text(tab)
    for title in MAC_TILES + MAC_ROWS:
        assert title not in drawn
    for title in ("CPU Usage", "Utilization", "Feeds", "Total CPU"):
        assert title in drawn
    assert "Loading GPU data" not in drawn and "37%" in drawn


@pytest.mark.unit
def test_on_linux_the_tiles_that_are_left_fill_the_row(board):
    """Two summary tiles share the width three had: the last one ends at the right margin."""
    tab, _ = board.build(Platform.LINUX)
    lines = board.text(tab).splitlines()
    title_row = next(line for line in lines if "Utilization" in line)
    assert title_row.index("Feeds") > WIDTH // 3, "the Feeds tile still sits in a third of its own"


@pytest.mark.unit
def test_on_a_mac_the_board_is_the_full_one(board):
    tab, built = board.build(Platform.MAC)
    assert {"GpuMonitor", "TempMonitor"} <= set(built)
    drawn = board.text(tab)
    for title in MAC_TILES + MAC_ROWS:
        assert title in drawn
