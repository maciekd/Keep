"""The sockets' bind address follows `interfaces.lan_access`: loopback until the key is on."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.Notifier import NullNotifier, set_notifier
from common.ConfigManager import config
from interfaces.Interface import bind_host, LAN_ACCESS_KEY, LOOPBACK, ALL_INTERFACES
from interfaces.chat.ChatHub import ChatHub
from interfaces.console.ConsoleInterface import ConsoleInterface
from interfaces.http.HttpInterface import HttpInterface
from common import ConfigStore


def _config_with(value):
    real_get = config.get
    return patch.object(config, "get", side_effect=lambda key, *a, **k: value if key == LAN_ACCESS_KEY
                        else real_get(key, *a, **k))


def test_an_install_binds_loopback_until_the_key_is_set():
    with patch.object(config, "get", side_effect=lambda key, default=None, *a, **k: default):
        assert bind_host(config, "The HTTP server") == LOOPBACK


def test_lan_access_true_binds_every_interface_of_the_machine():
    with _config_with(True):
        assert bind_host(config, "The TUI console") == ALL_INTERFACES


def test_the_assistant_may_set_the_key_contract_with_config_yaml():
    """`interfaces.http` is denied to config_set, the lan key beside it is the one the assistant
    flips for the phone app — so it must stay outside the denied prefixes and in the example."""
    ConfigStore.check_key(LAN_ACCESS_KEY)
    assert ConfigStore.coerce(LAN_ACCESS_KEY, "true") is True


# --- the two sockets follow the key, not only the helper -----------------------------------


@pytest.fixture
def chat_hub_reset():
    ChatHub.reset_instance()
    yield
    ChatHub.reset_instance()
    set_notifier(NullNotifier())


@pytest.mark.parametrize("lan_access, host", [(False, LOOPBACK), (True, ALL_INTERFACES)])
def test_the_tui_console_binds_the_address_the_key_picks(chat_hub_reset, lan_access, host):
    """The TUI socket has no authentication either: it must bind through the same key as the
    HTTP server, not keep TUIServer's own default of every interface."""
    assistant = MagicMock()
    assistant.prompt_manager.message_history.get_merged_entries.return_value = []
    interface = ConsoleInterface()
    with _config_with(lan_access), \
            patch("interfaces.console.framework.TUIServer.TUIServer") as server_cls, \
            patch("interfaces.console.TUIFloxyConsole.TUIFloxyConsole.get_instance"):
        interface.start(assistant, app=None)
    assert server_cls.call_args.kwargs["host"] == host


@pytest.mark.parametrize("lan_access, host", [(False, LOOPBACK), (True, ALL_INTERFACES)])
def test_the_http_server_binds_the_address_the_key_picks(lan_access, host):
    """What hypercorn is told to bind is what the phones and any web page on the Wi-Fi can
    reach: loopback until the owner opens the install to the network."""
    interface = HttpInterface(port=8808)
    interface.app = object()
    serve = AsyncMock()
    with _config_with(lan_access), \
            patch("hypercorn.asyncio.serve", serve), \
            patch("interfaces.http.HttpInterface.threading.Timer"), \
            patch.object(interface, "_install_signal_handlers"):
        asyncio.run(interface.serve())
    assert serve.call_args.args[1].bind == [f"{host}:8808"]
