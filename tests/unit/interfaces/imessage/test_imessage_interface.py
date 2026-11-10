"""IMessageInterface wires the iMessage transport onto the ChatHub and registers it with the
IntegrationMonitor by class, so the chat can be brought up at runtime once its keys are set
(restart_integration("iMessage Chat")) — and stays out of the process while its flag is off."""
from unittest.mock import MagicMock, Mock, patch

import pytest

from assistant import Destinations
from assistant.integration.Integration import IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor
from assistant.Notifier import NullNotifier, notifier, set_notifier
from common.ConfigManager import config
from interfaces.chat.ChatHub import ChatHub
from interfaces.imessage import IMessageTransport as module
from interfaces.imessage.IMessageInterface import IMessageInterface
from interfaces.imessage.IMessageTransport import IMessageTransport
from tests.unit.integrations.imessage.host_fakes import FakeHost

from .chat_fakes import configured


@pytest.fixture(autouse=True)
def clean_singletons():
    def reset():
        ChatHub.reset_instance()
        IntegrationMonitor.reset_instance()
        IMessageTransport.reset_instance()
    reset()
    before = set(Destinations.names())
    yield
    reset()
    for name in set(Destinations.names()) - before:
        Destinations.unregister(name)
    set_notifier(NullNotifier())


@pytest.fixture
def started(tmp_path):
    """The interface started with the cast on the lists, over a fake host; the poll thread
    runs and reads nothing until it is stopped."""
    host = FakeHost()
    with configured(interfaces__imessage__enabled=True) as keys, \
            patch.object(module.Hosts, "build", return_value=host), \
            patch.object(module.ChatState, "__init__", _state_in(tmp_path)), \
            patch.object(IMessageTransport, "_run", lambda self: self._stop.wait(5)):
        IntegrationMonitor.get_instance()._config_manager = config
        interface = IMessageInterface()
        yield interface, keys, host


def _state_in(tmp_path):
    from interfaces.imessage.ChatState import ChatState
    real = ChatState.__init__
    return lambda self, account, path=None: real(self, account, tmp_path / "imessage_chat.json")


@pytest.mark.unit
class TestTheSwitch:
    def test_the_interface_is_off_unless_its_flag_is_on(self):
        on = MagicMock(get=lambda key, default=None: {"interfaces.imessage.enabled": True}.get(key, default))
        off = MagicMock(get=lambda key, default=None: default)
        assert IMessageInterface.enabled(on) is True and IMessageInterface.enabled(off) is False

    def test_the_launcher_starts_it_last_and_never_in_the_prompt_suites_servers(self):
        import floxy_ai_server
        assert floxy_ai_server.INTERFACE_ORDER[-1] == "imessage", "its approvals are asked in the chats started before it"
        assert floxy_ai_server._interface_class("imessage") is IMessageInterface
        with patch.object(config, "get", side_effect=lambda key, default=None: True):
            assert floxy_ai_server._enabled("imessage", running_as_testsuite=True) is False
            assert floxy_ai_server._enabled("imessage", running_as_testsuite=False) is True

    def test_an_install_that_says_nothing_does_not_start_it(self):
        import floxy_ai_server
        with patch.object(config, "get", side_effect=lambda key, default=None: default):
            assert floxy_ai_server._enabled("imessage", running_as_testsuite=False) is False


@pytest.mark.unit
class TestStart:
    def test_a_configured_chat_is_on_the_hub_polling_and_listed_by_the_monitor(self, started):
        interface, _, _ = started
        assistant = MagicMock()
        interface.start(assistant, app=None)
        hub, transport = ChatHub.get_instance(), interface.transport
        assert isinstance(transport, IMessageTransport)
        assert hub.assistant is assistant and transport in hub.transports and notifier() is hub
        assert transport._thread is not None, "the poll was not started"
        assert IntegrationMonitor.get_instance().find("iMessage Chat") is transport
        assert "Charlotte" in Destinations.names()

    def test_with_nobody_to_serve_it_is_listed_as_not_configured_and_does_not_poll(self, started):
        interface, keys, _ = started
        keys.update({"imessage_chat.owners": [], "imessage_chat.guests": [], "imessage_chat.bots": []})
        interface.start(MagicMock(), app=None)
        transport = interface.transport
        assert transport not in ChatHub.get_instance().transports and transport._thread is None
        assert transport.state == IntegrationState.NOT_CONFIGURED

    def test_a_restart_once_the_lists_are_filled_brings_it_up_without_a_server_restart(self, started):
        interface, keys, _ = started
        cast = {key: keys[key] for key in ("imessage_chat.owners", "imessage_chat.guests", "imessage_chat.bots")}
        keys.update({key: [] for key in cast})
        interface.start(MagicMock(), app=None)
        first = interface.transport
        keys.update(cast)
        IntegrationMonitor.get_instance().restart("iMessage Chat")
        rebuilt = interface.transport
        assert rebuilt is not first and rebuilt in ChatHub.get_instance().transports and rebuilt._thread is not None

    def test_stop_takes_it_off_the_hub_and_drops_its_destinations(self, started):
        interface, _, _ = started
        interface.start(MagicMock(), app=None)
        interface.stop()
        assert interface.transport not in ChatHub.get_instance().transports
        assert "Charlotte" not in Destinations.names()


@pytest.mark.unit
class TestTheKeysTheAssistantSets:
    """A contract with config.example.yaml, the schema config_set and config_request check a key
    against: the section is there, its lists take one entry at a time, and what makes somebody
    an owner — or says where the account's messages go — is typed by the user alone."""

    @pytest.mark.parametrize("key", ["imessage_chat.owners[0]", "imessage_chat.guests[0]", "imessage_chat.bots[0]",
                                     "imessage_chat.host", "imessage_chat.host_token"])
    def test_the_user_types_who_is_served_and_where_the_account_is(self, key):
        from common import ConfigStore
        ConfigStore.check_key(key)      # config_request may ask for it
        with pytest.raises(ConfigStore.ConfigStoreError):
            ConfigStore.check_stated(key)   # config_set may not state it

    @pytest.mark.parametrize("key", ["imessage_chat.daily_limit", "imessage_chat.typing_dots", "interfaces.imessage.enabled"])
    def test_the_assistant_may_state_the_rest(self, key):
        from common import ConfigStore
        ConfigStore.check_key(key)
        ConfigStore.check_stated(key)

    @pytest.mark.parametrize("spelling", [
        "imessage_chat.guests[0]", "imessage_chat.bots[0]", "imessage_chat.guests", "imessage_chat.bots[7]",
        " imessage_chat.guests[0] ", "imessage_chat.guests[00]", "imessage_chat.guests(0)", "imessage_chat.guests.0",
        "imessage_chat.guests[0].x", "imessage_chat[0].guests", "imessage_chat. guests[0]", "imessage_chat.guests [0]",
        "imessage_chat.Guests[0]", "IMESSAGE_CHAT.guests[0]", "imessage_chat.guests[-1]", "imessage-chat.guests[0]",
        '"imessage_chat.guests[0]"',
    ])
    def test_no_spelling_of_a_lists_key_gets_a_stated_value_past_the_store(self, spelling):
        """What config_set does with its first argument, then the two checks it writes behind."""
        from common import ConfigSchema, ConfigStore
        key = ConfigSchema.normalize_key(spelling.strip().strip('"\''))
        with pytest.raises(ConfigStore.ConfigStoreError):
            ConfigStore.check_key(key)
            ConfigStore.check_stated(key)

    @pytest.mark.parametrize("key", ["imessage_chat.guests[0]", "imessage_chat.bots[0]", "imessage_chat.guests(1)"])
    def test_an_instruction_the_assistant_read_cannot_make_it_add_somebody(self, key):
        """The injected sequence starts with config_set("imessage_chat.guests[0]", "+<attacker>"):
        the command refuses and starts nothing, while config_request — the user typing the
        entry into a prompt — is still on offer."""
        from assistant.commands.Commands import Commands
        commands = Commands(Mock(command_reply=None, command_reply_contains_error=False))
        with patch.object(commands, "background_command") as started:
            answer = commands.cmd_config_set(key, "+46701740699")
            assert answer.startswith("config_set failed") and "config_request" in answer
            started.assert_not_called()
            commands._current_context = Mock(source="Console", conversation="tui")
            assert commands.cmd_config_request(key) is None
            started.assert_called_once()

    def test_the_example_leaves_the_chat_off_and_on_this_macs_own_messages(self):
        from common import ConfigSchema
        assert ConfigSchema.example_value("interfaces.imessage.enabled") is False
        assert ConfigSchema.example_value("imessage_chat.host") == "local"
        assert ConfigSchema.example_value("imessage_chat.owners") == []
