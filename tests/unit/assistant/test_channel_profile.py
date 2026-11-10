"""The ChannelProfile table and every call site that used to compare `source ==`."""
from integrations.photos.PhotosCommands import MediaPayload
from unittest.mock import Mock, patch

import pytest

from assistant.ChannelProfile import AUTONOMOUS, PHOTO_TARGET_CHAT, PHOTO_TARGET_SCREEN, PROFILES, TELEGRAM, TEXT_CHAT, VOICE_TO_VOICE, ChannelProfile, profile_for


class TestTable:
    def test_none_is_autonomous(self):
        assert profile_for(None) is AUTONOMOUS

    def test_unknown_source_is_text_chat_and_logged_once(self):
        with patch("assistant.ChannelProfile.info") as info, patch("assistant.ChannelProfile.trace") as trace:
            assert profile_for("NeverSeenBefore") is TEXT_CHAT
            assert profile_for("NeverSeenBefore") is TEXT_CHAT
        assert info.call_count == 1
        assert trace.call_count == 1

    def test_every_row_is_complete(self):
        for name, profile in PROFILES.items():
            assert isinstance(profile, ChannelProfile)
            assert profile.name == name
            assert profile.llm_tier in ("fast", "smart")
            assert profile.photo_target in (PHOTO_TARGET_CHAT, PHOTO_TARGET_SCREEN)

# --- call sites ------------------------------------------------------------------------

class TestDialogManagerReadsProfile:
    """DialogManager.select_ai / _brevity_directive key off llm_tier / brevity_directive."""

    @pytest.fixture
    def dm(self):
        from assistant.DialogManager import DialogManager
        dm = DialogManager.__new__(DialogManager)
        dm.ai_fast, dm.ai_smart, dm.ai_vision = Mock(name="fast"), Mock(name="smart"), Mock(name="vision")
        dm.web_request_handler = Mock()
        return dm

    @pytest.mark.parametrize("source,tier_attr", [
        ("VoiceToVoice", "ai_fast"), ("TextChat", "ai_smart"), ("Telegram", "ai_smart"), (None, "ai_smart"),
    ])
    def test_select_ai_uses_profile_tier(self, dm, source, tier_attr):
        ctx = Mock(images=None, ai_override=None, source=source)
        dm.select_ai(ctx)
        assert dm.ai is getattr(dm, tier_attr)
        assert profile_for(source).llm_tier == ("fast" if tier_attr == "ai_fast" else "smart")

    def test_select_ai_custom_profile_tier_is_honoured(self, dm):
        custom = ChannelProfile(**{**VOICE_TO_VOICE.__dict__, "name": "Custom", "llm_tier": "smart"})
        with patch("assistant.DialogManager.profile_for", return_value=custom):
            dm.select_ai(Mock(images=None, ai_override=None, source="Custom"))
        assert dm.ai is dm.ai_smart

    @pytest.mark.parametrize("source", ["VoiceToVoice", "TextChat", "Telegram", None])
    def test_brevity_directive_follows_profile(self, dm, source):
        from assistant.DialogManager import DialogManager
        dm.web_request_handler.ctx = Mock(source=source)
        expected = DialogManager._BREVITY_DIRECTIVE if profile_for(source).brevity_directive else None
        assert dm._brevity_directive() == expected


class TestTheChannelsOwnStyle:
    """ChannelProfile.style_directive rides on every request of the channel, in front of the language."""

    @pytest.fixture
    def dm(self):
        from assistant.DialogManager import DialogManager
        dm = DialogManager.__new__(DialogManager)
        dm.web_request_handler = Mock()
        dm.byte_prefix_tail = False
        dm.reply_from_role = "user"
        return dm

    def test_an_imessage_request_is_told_to_write_plain_texts(self, dm):
        dm.web_request_handler.ctx = Mock(source="iMessage", lang="en")
        (last,) = dm._with_trailing_directives([{"role": "user", "content": "what's for dinner?"}])
        assert last["content"].startswith("what's for dinner?")
        assert "iMessage" in last["content"] and "plain text" in last["content"]

    @pytest.mark.parametrize("source", ["Telegram", "Console", "TextChat", "VoiceToVoice", None])
    def test_no_other_channel_gets_a_word_of_it(self, dm, source):
        dm.web_request_handler.ctx = Mock(source=source, lang="en")
        assert dm._style_directive() is None

    def test_the_language_stays_the_last_thing_the_model_reads(self, dm):
        dm.web_request_handler.ctx = Mock(source="iMessage", lang="sv")
        directives = dm._trailing_directives()
        assert len(directives) == 2 and directives[0] == profile_for("iMessage").style_directive


class TestCommandProcessorReadsProfile:
    """Thinking forwarding and command-notification routing read the profile."""

    @pytest.fixture
    def processor(self):
        from assistant.commands.CommandProcessor import CommandProcessor
        p = CommandProcessor(test_suite_mode=True)
        yield p
        p.kill()

    @staticmethod
    def _dm(source):
        dm = Mock()
        dm.web_request_handler.ctx = Mock(source=source)
        return dm

    @pytest.mark.parametrize("source", ["VoiceToVoice", "TextChat", "Telegram"])
    def test_thinking_forwarded_only_when_profile_says_so(self, processor, source):
        from assistant.AssistantEvents import observers
        fake = Mock()
        observers().subscribe(fake)
        processor._emit_thinking_chunk("hmm", self._dm(source))
        processor._finalize_thinking(0.5, self._dm(source))
        forwarded = profile_for(source).forward_thinking
        assert fake.on_thinking_chunk.called is forwarded
        assert fake.on_thinking_done.called is forwarded

    def test_profile_for_reads_ctx_source(self, processor):
        assert processor._profile_for(self._dm("Telegram")) is TELEGRAM
        assert processor._profile_for(None) is AUTONOMOUS

    def test_run_sets_the_commands_current_profile(self, processor):
        from assistant.commands.CommandProcessor import CommandProcessor
        import threading
        p = CommandProcessor(test_suite_mode=False)
        seen = {}
        done = threading.Event()

        def run(command, params):
            seen["profile"] = p.commands._current_profile
            done.set()
            return None

        p.commands.run = run
        dm = self._dm("VoiceToVoice")
        dm.is_killed = False
        dm.web_request_handler.wait_so_that_response_timing_is_natural.return_value = True
        with patch("assistant.commands.CommandProcessor.LOCAL_DEVELOPMENT_MODE", False):
            p.invoke_command("turn_on", ["kitchen"], dm)
            assert done.wait(timeout=2)
        p.kill()
        assert seen["profile"] is VOICE_TO_VOICE


class TestCommandsReadProfile:
    """send_screenshot / show_photo key off screenshot_allowed / photo_target."""

    @pytest.fixture
    def commands(self):
        from assistant.commands.Commands import Commands
        from assistant.tools.WebAccess import WebAccess
        cp = Mock(command_reply=None, command_reply_contains_error=False)
        with patch.object(WebAccess, "get_instance", return_value=Mock()):
            yield Commands(cp)

    @pytest.mark.parametrize("source", ["VoiceToVoice", "TextChat", "Telegram", "autonomous"])
    def test_send_screenshot_refused_iff_not_allowed(self, commands, source):
        commands._current_profile = profile_for(source)
        commands.web_access.get_last_screenshot = Mock(return_value=(None, None))
        result = commands._run_send_screenshot("look")
        if profile_for(source).screenshot_allowed:
            assert "voice mode" not in result
        else:
            assert "voice mode" in result

    @pytest.mark.parametrize("source", ["VoiceToVoice", "TextChat", "Telegram"])
    def test_show_photo_target_follows_profile(self, commands, source):
        commands._current_profile = profile_for(source)
        with patch.object(commands._owner("show_photo"), "_prepare_photo_send_payload", return_value=(MediaPayload(b"jpg", "cap", "uuid", "photo"), None)), \
             patch.object(commands._owner("show_photo"), "_show_photo_on_screen", return_value="tv") as tv, \
             patch("assistant.Notifier._notifier", Mock()) as bot, \
             patch("integrations.photos.PhotosCommands.config") as cfg:
            cfg.get.side_effect = lambda key, default=None: {"photos.show_on_screen": True}.get(key, default)
            result = commands._owner("show_photo")._run_show_single_photo(1)
        on_tv = profile_for(source).photo_target == PHOTO_TARGET_SCREEN
        assert tv.called is on_tv
        assert bot.send_photo.called is not on_tv
        assert result == ("tv" if on_tv else "")

    def test_album_collapses_to_first_photo_on_the_tv(self, commands):
        commands._current_profile = VOICE_TO_VOICE
        with patch.object(commands._owner("show_photo"), "_run_show_single_photo", return_value="") as single, \
             patch("integrations.photos.PhotosCommands.config") as cfg:
            cfg.get.side_effect = lambda key, default=None: {"photos.show_on_screen": True}.get(key, default)
            assert commands._owner("show_photo")._run_show_photo_album([7, 8, 9]) == ""
        single.assert_called_once_with(7, route=None)
