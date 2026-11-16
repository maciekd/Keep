"""What WebAccess does differently per OS: where it looks for Chrome and the user agent it launches with."""
import pytest

from assistant.tools.WebAccess import WebAccess
from common import Platform


def _on(monkeypatch, system):
    monkeypatch.setattr(Platform, "current", lambda: system)


def test_the_linux_chrome_paths_are_probed_on_linux_only(monkeypatch):
    _on(monkeypatch, Platform.MAC)
    on_the_mac = WebAccess._chrome_search_paths()
    _on(monkeypatch, Platform.LINUX)
    on_linux = WebAccess._chrome_search_paths()
    assert not any(path.startswith("/usr/bin/") for path in on_the_mac)
    assert set(on_the_mac) < set(on_linux) and "/usr/bin/chromium" in on_linux


@pytest.mark.parametrize("system, token", [(Platform.MAC, "Macintosh"), (Platform.LINUX, "Linux"),
                                           ("win32", "Windows"), ("freebsd14", "Linux")])
def test_the_user_agent_names_the_system_chrome_runs_on(monkeypatch, system, token):
    _on(monkeypatch, system)
    assert token in WebAccess._ua_platform_token()
