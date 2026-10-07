"""Proactive toolkit loading: a voice request finds the home toolkit (with the home data)
and the music toolkit in the window — put there when missing, left alone when there — no
other channel preloads anything of its own, and the FTUE adds the setup toolkit on every
channel. The mechanics are exercised on a home-only profile so each test sees one row."""
import threading
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from assistant.AssistantRequest import RequestContext
from assistant.ChannelProfile import SOURCE_VOICE, TELEGRAM, VOICE_TO_VOICE
from assistant.commands import ToolkitPreloader, Toolkits
from assistant.RequestProcessor import RequestProcessor

DATA = '## LIGHTS\n"Lamp [Kök]": off'
HOME_ONLY = replace(VOICE_TO_VOICE, preload_toolkits=("home",), fixed_toolkits=())
CONFIGURED = {}


@pytest.fixture(autouse=True)
def _configured_preloads():
    CONFIGURED.clear()
    with patch.object(ToolkitPreloader, "config") as config:
        config.get.side_effect = lambda key, default=None: CONFIGURED.get(key, default)
        yield


def _prompt_manager(loaded: dict, untracked: list[str] = ()):
    pm = Mock()
    pm.render_template = lambda text: text.replace("{PHOTO_COUNT}", "1234")
    pm.message_history.loaded_toolkits.return_value = loaded
    pm.untracked_sections.return_value = list(untracked)
    pm.carried_data = Mock(side_effect=lambda toolkit, record=True: Toolkits.render_sections(toolkit, {"home": DATA}))
    return pm


@pytest.mark.unit
def test_home_missing_from_the_window_is_loaded_with_its_data():
    pm = _prompt_manager({})
    assert ToolkitPreloader.preload(pm, HOME_ONLY) == ["home"]
    pm.add_command_result.assert_called_once()
    text, kwargs = pm.add_command_result.call_args.args[0], pm.add_command_result.call_args.kwargs
    assert kwargs == {"cap": False}
    assert text.startswith("The home toolkit is loaded ahead of this VoiceToVoice request, no load_toolkit needed")
    assert "\n# Toolkit: home\n" in text
    assert "\n## HOME AUTOMATION DATA\n" in text and text.rstrip().endswith('"Lamp [Kök]": off')
    pm.carried_data.assert_called_once_with(Toolkits.by_name("home"), record=True)


@pytest.mark.unit
def test_the_configured_list_replaces_the_channels_own():
    CONFIGURED["preload_toolkits"] = {SOURCE_VOICE: ["music", "tesla"], "Telegram": ["web"]}
    assert ToolkitPreloader.preload(_prompt_manager({}), HOME_ONLY) == ["music", "tesla"]
    CONFIGURED["preload_toolkits"] = {SOURCE_VOICE: []}
    assert ToolkitPreloader.preload(_prompt_manager({}), HOME_ONLY) == []


@pytest.mark.unit
def test_the_configured_list_cannot_drop_a_fixed_toolkit():
    """The voice dialog style is in the voice toolkit and nowhere in the prompt: a household
    trimming its preload list must not lose it."""
    profile = replace(HOME_ONLY, fixed_toolkits=("voice",))
    assert ToolkitPreloader.preload(_prompt_manager({}), profile) == ["voice", "home"]
    CONFIGURED["preload_toolkits"] = {SOURCE_VOICE: []}
    assert ToolkitPreloader.preload(_prompt_manager({}), profile) == ["voice"]
    assert ToolkitPreloader.preload(_prompt_manager({"voice": 1000.0}), profile) == []


@pytest.mark.unit
def test_voice_keeps_timers_whatever_the_configured_list_says():
    """"Set a timer for ten minutes" is spoken daily: a household that trims the voice
    preloads to its lamps still finds the timers toolkit in the window."""
    CONFIGURED["preload_toolkits"] = {SOURCE_VOICE: ["home"]}
    assert ToolkitPreloader.preload(_prompt_manager({}), VOICE_TO_VOICE) == ["voice", "timers", "home"]
    assert ToolkitPreloader.preload(_prompt_manager({"voice": 1000.0, "home": 1000.0}), VOICE_TO_VOICE) == ["timers"]


@pytest.mark.unit
def test_home_already_in_the_window_is_left_alone():
    pm = _prompt_manager({"home": 1000.0})
    assert ToolkitPreloader.preload(pm, HOME_ONLY) == []
    pm.add_command_result.assert_not_called()
    pm.carried_data.assert_not_called()


@pytest.mark.unit
def test_home_in_the_window_without_a_baseline_gets_its_data_again():
    """After a restart the reference is in the window but nothing was sent since, and after a
    pointer note the tracking has stopped: the LLM's copy is stale either way, so the data
    goes again — without the reference, which is still above."""
    pm = _prompt_manager({"home": 1000.0}, untracked=["home"])
    assert ToolkitPreloader.preload(pm, HOME_ONLY) == ["home"]
    text = pm.add_command_result.call_args.args[0]
    assert pm.add_command_result.call_args.kwargs == {"cap": False}
    assert text.startswith("The home toolkit's reference is above in this conversation")
    assert "# Toolkit: home" not in text and Toolkits.loaded_name(text) is None
    assert "\n## HOME AUTOMATION DATA\n" in text and text.rstrip().endswith('"Lamp [Kök]": off')
    pm.carried_data.assert_called_once_with(Toolkits.by_name("home"))


@pytest.mark.unit
def test_the_window_is_trimmed_before_it_is_looked_at():
    """The trim get_messages is about to run must not take the toolkit out of the window
    right after the preloader saw it there."""
    pm = _prompt_manager({"home": 1000.0})
    ToolkitPreloader.preload(pm, HOME_ONLY)
    calls = [c[0] for c in pm.message_history.method_calls]
    assert calls.index("trim_window") < calls.index("loaded_toolkits")


@pytest.mark.unit
def test_a_channel_without_preloads_never_looks_at_the_window():
    pm = _prompt_manager({})
    assert ToolkitPreloader.preload(pm, TELEGRAM) == []
    pm.message_history.loaded_toolkits.assert_not_called()
    pm.add_command_result.assert_not_called()


@pytest.mark.unit
def test_extra_toolkits_load_after_the_channels_own_without_duplicates():
    pm = _prompt_manager({})
    assert ToolkitPreloader.preload(pm, HOME_ONLY, ("setup", "home")) == ["home", "setup"]
    texts = [c.args[0] for c in pm.add_command_result.call_args_list]
    assert texts[1].startswith("The setup toolkit is loaded ahead of this VoiceToVoice request, no load_toolkit needed")
    assert "\n# Toolkit: setup\n" in texts[1]
    assert ToolkitPreloader.preload(_prompt_manager({"setup": 1000.0}), TELEGRAM, ("setup",)) == []


@pytest.mark.unit
def test_an_unknown_name_in_the_profile_is_skipped_and_the_rest_still_load():
    pm = _prompt_manager({})
    assert ToolkitPreloader.preload(pm, replace(HOME_ONLY, preload_toolkits=("nope", "home"))) == ["home"]
    pm.add_command_result.assert_called_once()


@pytest.mark.unit
def test_the_preloaded_row_counts_as_loaded_for_load_toolkit():
    """MessageHistory recognises the row by its `# Toolkit:` line wherever it sits, so a
    later load_toolkit("home") points back instead of sending the reference again."""
    pm = _prompt_manager({})
    ToolkitPreloader.preload(pm, HOME_ONLY)
    text = pm.add_command_result.call_args.args[0]
    assert [Toolkits.loaded_name(line) for line in text.split("\n") if Toolkits.loaded_name(line)] == ["home"]


def _racing_prompt_manager(persist_started: threading.Event, release_persist: threading.Event):
    """A prompt manager whose window only shows a row once its persist has finished — the
    real one reloads the history after the write — and whose persist blocks on
    *release_persist* so a second caller can be lined up behind it."""
    window: dict[str, float] = {}
    pm = _prompt_manager(window)
    pm.message_history.loaded_toolkits.side_effect = lambda: dict(window)

    def persist(text, cap=True):
        persist_started.set()
        release_persist.wait(timeout=5)
        window[Toolkits.loaded_name(text.split("\n")[2])] = 1000.0

    pm.add_command_result.side_effect = persist
    return pm


@pytest.mark.unit
def test_two_requests_preloading_at_once_persist_one_row():
    """The 2026-09-14 race: "Turn" and "Turn off" — speculative partial-transcript requests
    on two handler threads — both saw no home in the window and both persisted it. The
    check and the persist are one critical section now, so the second waits and finds
    the first's row."""
    persist_started, release_persist = threading.Event(), threading.Event()
    pm = _racing_prompt_manager(persist_started, release_persist)
    results = {}
    first = threading.Thread(target=lambda: results.update(first=ToolkitPreloader.preload(pm, HOME_ONLY)))
    first.start()
    assert persist_started.wait(timeout=5)
    second = threading.Thread(target=lambda: results.update(second=ToolkitPreloader.preload(pm, HOME_ONLY)))
    second.start()
    second.join(timeout=0.05)
    assert second.is_alive(), "the second preload must wait for the first's persist"
    release_persist.set()
    first.join(timeout=5), second.join(timeout=5)
    assert results == {"first": ["home"], "second": []}
    pm.add_command_result.assert_called_once()


@pytest.mark.unit
def test_a_request_killed_while_waiting_for_its_turn_loads_nothing():
    """"Turn" is killed by "Turn off" a few ms in: whatever it would have preloaded, the
    request that killed it preloads for itself, so it must not write a row of its own."""
    persist_started, release = threading.Event(), threading.Event()
    pm = _racing_prompt_manager(persist_started, release)
    pm.message_history.loaded_toolkits.side_effect = None
    pm.message_history.loaded_toolkits.return_value = {}     # the row never lands in the window
    killed = threading.Event()
    first = threading.Thread(target=lambda: ToolkitPreloader.preload(pm, HOME_ONLY))
    first.start()
    assert persist_started.wait(timeout=5)
    results = {}
    second = threading.Thread(
        target=lambda: results.update(second=ToolkitPreloader.preload(pm, HOME_ONLY, is_killed=killed.is_set)))
    second.start()
    killed.set()
    release.set()
    first.join(timeout=5), second.join(timeout=5)
    assert results == {"second": []}
    pm.add_command_result.assert_called_once()


@pytest.mark.unit
def test_a_live_request_ignores_the_kill_probe_until_the_lock_is_held():
    pm = _prompt_manager({})
    assert ToolkitPreloader.preload(pm, HOME_ONLY, is_killed=lambda: False) == ["home"]
    assert ToolkitPreloader.preload(_prompt_manager({}), HOME_ONLY, is_killed=lambda: True) == []


@pytest.mark.unit
def test_render_load_matches_load_toolkit():
    from assistant.commands.Commands import Commands
    pm = _prompt_manager({})
    commands = Commands(Mock(command_reply=None, command_reply_contains_error=False))
    with patch.object(Commands, "_prompt_manager", staticmethod(lambda: pm)):
        assert commands._run_load_toolkit("home") == ToolkitPreloader.render_load(Toolkits.by_name("home"), pm)
    assert ToolkitPreloader.render_load(Toolkits.by_name("web"), None).startswith("# Toolkit: web\n")


class TestRequestProcessorHook:
    @pytest.fixture(autouse=True)
    def _past_the_ftue(self):
        with patch("assistant.RequestProcessor.FirstRun.in_ftue", return_value=False) as in_ftue:
            yield in_ftue

    def _handler(self, ctx: RequestContext) -> RequestProcessor:
        h = RequestProcessor.__new__(RequestProcessor)
        h.request_context = ctx
        h.is_killed = False
        return h

    @pytest.mark.unit
    def test_voice_request_preloads_before_the_messages_are_built(self):
        pm = _prompt_manager({})
        self._handler(RequestContext(source=SOURCE_VOICE, user_input="hi"))._preload_toolkits(pm)
        texts = [c.args[0] for c in pm.add_command_result.call_args_list]
        assert [Toolkits.loaded_name(line) for t in texts for line in t.split("\n") if Toolkits.loaded_name(line)] \
            == ["voice", "timers", "home", "music", "tesla", "weather"]

    @pytest.mark.unit
    def test_voice_request_with_home_and_tesla_in_the_window_still_preloads_music(self):
        """The 2026-09-14 case: home was preloaded on an earlier turn, and "turn off the music"
        still spent a round on load_toolkit(music) — now music rides along with home."""
        pm = _prompt_manager({"voice": 1000.0, "timers": 1000.0, "home": 1000.0, "tesla": 1000.0, "weather": 1000.0})
        self._handler(RequestContext(source=SOURCE_VOICE, user_input="turn off the music"))._preload_toolkits(pm)
        pm.add_command_result.assert_called_once()
        text = pm.add_command_result.call_args.args[0]
        assert text.startswith("The music toolkit is loaded ahead of this VoiceToVoice request")
        assert "\n# Toolkit: music\n" in text
        assert "search_music" in text and "pause" in text

    @pytest.mark.unit
    def test_a_killed_voice_request_preloads_nothing(self):
        pm = _prompt_manager({})
        h = self._handler(RequestContext(source=SOURCE_VOICE, user_input="Turn"))
        h.is_killed = True
        h._preload_toolkits(pm)
        pm.add_command_result.assert_not_called()

    @pytest.mark.unit
    def test_test_suite_request_skips_the_preload(self):
        pm = _prompt_manager({})
        self._handler(RequestContext(source=SOURCE_VOICE, user_input="hi", test=True))._preload_toolkits(pm)
        pm.add_command_result.assert_not_called()

    @pytest.mark.unit
    def test_chat_request_in_the_ftue_preloads_setup(self, _past_the_ftue):
        _past_the_ftue.return_value = True
        pm = _prompt_manager({})
        self._handler(RequestContext(source="Telegram", user_input="hi"))._preload_toolkits(pm)
        text = pm.add_command_result.call_args.args[0]
        assert text.startswith("The setup toolkit is loaded ahead of this Telegram request")
        assert "\n# Toolkit: setup\n" in text
