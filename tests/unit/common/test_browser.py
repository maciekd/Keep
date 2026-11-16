import subprocess
from types import SimpleNamespace

import pytest

from common import Browser, Platform


def test_a_link_is_handed_to_the_macs_open(monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.MAC)
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: commands.append(cmd) or SimpleNamespace(returncode=0))
    assert Browser.open_url("https://keep.example/login")
    assert commands == [["open", "https://keep.example/login"]]


def test_off_the_mac_nothing_is_run_and_the_caller_is_told(monkeypatch):
    monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)
    for display in Browser.DISPLAY_VARIABLES:      # a server: no desktop session
        monkeypatch.delenv(display, raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no tool is run off macOS"))
    assert Browser.open_url("https://keep.example/login") is False
