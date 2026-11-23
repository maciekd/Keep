"""The `<!-- mac -->` / `<!-- linux -->` passages of the prompt texts: each system reads its own,
and a Mac reads exactly what the text said before the markers were put in."""
import pytest

from assistant import FirstRun
from assistant.commands import Toolkits
from assistant.commands.formats import Rewriter
from common import Platform

TEXT = """Both.
<!-- mac -->
Open System Settings.
<!-- /mac -->
<!-- linux -->
Run systemctl.
<!-- /linux -->
Both again.
"""
SETUP = Toolkits.by_name("setup")


def _on(monkeypatch, system):
    monkeypatch.setattr(Platform, "current", lambda: system)


def _without_markers(text: str, dropped: str) -> str:
    """`text` as a system reads it, worked out line by line: the *dropped* system's passages
    gone, every marker line gone."""
    kept, inside = [], None
    for line in text.splitlines(keepends=True):
        word = line.strip()
        if word.startswith("<!-- /") and word[6:-4] in Rewriter.PLATFORM_MARKERS:
            inside = None
        elif word.startswith("<!-- ") and word[5:-4] in Rewriter.PLATFORM_MARKERS:
            inside = word[5:-4]
        elif inside != dropped:
            kept.append(line)
    return "".join(kept)


@pytest.mark.unit
class TestGate:
    @pytest.mark.parametrize("system, kept, dropped", [(Platform.MAC, "System Settings", "systemctl"),
                                                       (Platform.LINUX, "systemctl", "System Settings")])
    def test_each_system_reads_its_own_passage_and_no_marker(self, monkeypatch, system, kept, dropped):
        _on(monkeypatch, system)
        text = Rewriter.gate_platform(TEXT)
        assert kept in text and dropped not in text and "<!--" not in text
        assert text.startswith("Both.\n") and text.endswith("Both again.\n")

    def test_another_system_reads_neither(self, monkeypatch):
        _on(monkeypatch, "win32")
        assert Rewriter.gate_platform(TEXT) == "Both.\nBoth again.\n"

    def test_a_marker_without_its_pair_is_left_as_written(self, monkeypatch):
        _on(monkeypatch, Platform.LINUX)
        text = "Before.\n<!-- mac -->\nNo closer.\n"
        assert Rewriter.gate_platform(text) == text

    def test_a_marker_inside_a_line_is_prose_and_not_a_gate(self, monkeypatch):
        """The texts document the markers themselves (CLAUDE.md does): only a marker that is a
        line of its own opens or closes a passage."""
        _on(monkeypatch, Platform.LINUX)
        text = "Write `<!-- mac -->` over it\nand `<!-- /mac -->` under it.\n"
        assert Rewriter.gate_platform(text) == text

    @pytest.mark.parametrize("system, kept", [(Platform.MAC, ["one", "three"]), (Platform.LINUX, ["two"])])
    def test_each_passage_ends_at_its_own_closer_whatever_stands_between_two_of_a_kind(self, monkeypatch, system, kept):
        _on(monkeypatch, system)
        text = ("<!-- mac -->\none\n<!-- /mac -->\n<!-- linux -->\ntwo\n<!-- /linux -->\nshared\n"
                "<!-- mac -->\nthree\n<!-- /mac -->\n")
        lines = Rewriter.gate_platform(text).splitlines()
        assert [line for line in lines if line != "shared"] == kept and "shared" in lines

    @pytest.mark.parametrize("system, fmt_name, expected", [
        (Platform.LINUX, "keep", "Both.\nRun systemctl.\nTyped the Keep way.\nEnd.\n"),
        (Platform.LINUX, "qwen", "Both.\nRun systemctl.\nEnd.\n"),
        (Platform.MAC, "keep", "Both.\nEnd.\n"),
    ])
    def test_the_rewriter_resolves_the_system_and_the_format_in_one_text(self, monkeypatch, system, fmt_name, expected):
        """A prompt text that passes `rewrite` (res/prompt.md) is gated there: a Keep-only
        passage inside another system's passage goes with it, markers and all."""
        from assistant.commands.formats.KeepFormat import KeepFormat
        from assistant.commands.formats.QwenFormat import QwenFormat
        _on(monkeypatch, system)
        fmt = {"keep": KeepFormat, "qwen": QwenFormat}[fmt_name]([])
        text = ("Both.\n<!-- linux -->\nRun systemctl.\n<!-- keep -->\nTyped the Keep way.\n<!-- /keep -->\n"
                "<!-- /linux -->\nEnd.\n")
        assert Rewriter.rewrite(text, fmt, {}) == expected


@pytest.mark.unit
class TestTheSetupToolkit:
    def test_a_mac_reads_the_file_with_the_linux_passages_and_the_markers_taken_out(self, monkeypatch):
        _on(monkeypatch, Platform.MAC)
        raw = SETUP.path.read_text(encoding="utf-8")
        assert SETUP.text() == _without_markers(raw, dropped=Platform.LINUX)
        assert "show_full_disk_access_setting()" in SETUP.text()

    def test_linux_reads_no_command_that_acts_on_the_mac_and_no_way_to_system_settings(self, monkeypatch):
        _on(monkeypatch, Platform.LINUX)
        text = SETUP.text()
        for command in Toolkits.MAC_ONLY_COMMANDS:
            assert command not in text
        assert "System Settings" not in text and "<!--" not in text
        assert len(text) <= Toolkits.MAX_FILE_CHARS

    def test_both_systems_are_told_the_same_stop_and_wait_commands(self, monkeypatch):
        _on(monkeypatch, Platform.MAC)
        on_mac = SETUP.stop_and_wait()
        _on(monkeypatch, Platform.LINUX)
        assert on_mac == SETUP.stop_and_wait()
        assert not on_mac & Toolkits.MAC_ONLY_COMMANDS

    def test_every_marker_of_every_toolkit_has_its_pair(self):
        for toolkit in Toolkits.TOOLKITS:
            raw = toolkit.path.read_text(encoding="utf-8")
            for name in Rewriter.PLATFORM_MARKERS:
                assert raw.count(f"<!-- {name} -->\n") == raw.count(f"<!-- /{name} -->\n"), f"{toolkit.name}: {name}"


@pytest.mark.unit
class TestTheFirstRunBrief:
    def test_a_mac_reads_the_brief_with_the_linux_passages_and_the_markers_taken_out(self, monkeypatch):
        _on(monkeypatch, Platform.MAC)
        owner = FirstRun.brief().split("\n", 1)[0].removeprefix(FirstRun.BRIEF.split("{owner}")[0])
        assert FirstRun.brief() == _without_markers(FirstRun.BRIEF, dropped=Platform.LINUX).format(owner=owner)
        assert "show_full_disk_access_setting" in FirstRun.brief()

    def test_on_linux_the_brief_sends_nobody_to_the_macs_settings_or_to_photos(self, monkeypatch):
        _on(monkeypatch, Platform.LINUX)
        brief = FirstRun.brief()
        assert "<!--" not in brief and "{owner}" not in brief
        for mac_step in ("show_full_disk_access_setting", "Full Disk Access", 'enable_integration("Photos")', "shortcut"):
            assert mac_step not in brief
        assert "Telegram" in brief and "satellite" in brief
